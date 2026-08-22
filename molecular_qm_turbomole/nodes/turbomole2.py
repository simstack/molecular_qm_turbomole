import logging
import math
import re
from pathlib import Path

from molecular_qm_models.molecule import MoleculeList
from molecular_qm_models.qm_result import QMResult
from molecular_qm_turbomole.lib.control_utils import append_control_groups
from molecular_qm_turbomole.lib.env import (
    build_define_script,
    build_frequency_script,
    build_ground_state_script,
    build_hyperpolarizability_script,
    prepend_tm_env,
)
from molecular_qm_turbomole.lib.opt_artifacts import (
    OPT_CHART_INTERVAL,
    OptimizationChartTracker,
    inspect_geometry_optimization,
)
from molecular_qm_turbomole.lib.hyperpol import (
    apply_hyperpolarizability_control,
    hyperpolarizability_requested,
    hyperpolarizability_wavelength_nm,
    parse_hyperpolarizability_table,
    validate_hyperpolarizability_request,
    verify_dynamic_hyperpols_output,
)
from molecular_qm_turbomole.lib.input_writer import (
    TurbomoleInputWriter,
    should_use_ri,
)
from molecular_qm_turbomole.lib.output_parser import (
    TurbomoleOutputParser,
    write_final_geometry_xyz,
)
from molecular_qm_turbomole.models.turbomole_input import TurbomoleQMInput2
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models import Parameters
from simstack.models.files import FileStack
from simstack.models.parameters import SlurmParameters

logger = logging.getLogger("Turbomole2Node")

_OUTPUT_TAIL = 2000


def _runner_output_tail(node_runner, limit: int = _OUTPUT_TAIL) -> str:
    for attr in ("last_stderr", "last_stdout"):
        text = str(getattr(node_runner, attr, "") or "").strip()
        if text:
            return text[-limit:]
    return ""


def _with_runner_output(node_runner, message: str) -> str:
    tail = _runner_output_tail(node_runner)
    if tail and tail not in message:
        return f"{message}\n{tail}"
    return message


def _fail(node_runner, message: str) -> None:
    """Mark the node failed and raise so Simstack stores the message on the registry."""
    node_runner.fail(message)
    raise RuntimeError(message)

slurm_parameters = SlurmParameters(
    nodes=1,
    tasks=1,
    tasks_per_node=8,
    cpus_per_task=1,
    mem="2G",
    time="0:10:00",
)

parameters = Parameters(
    resource="int-nano",
    queue="slurm-queue",
    slurm_parameters=slurm_parameters,
    force_rerun=True,
)

OUTPUT_FILES = (
    "control",
    "coord",
    "energy",
    "define.inp",
    "define.out",
    "ridft.out",
    "dscf.out",
    "jobex.out",
    "job.last",
    "final_geometry.xyz",
    "gradient",
    "aoforce.out",
    "vibspectrum",
    "escf.out",
    "hyperpols",
)


def _validate_request(qm_input: TurbomoleQMInput2) -> None:
    atoms = getattr(qm_input.molecule, "atoms", None)
    if not atoms:
        raise ValueError(
            "TURBOMOLE requires a non-empty molecule with 3D coordinates."
        )
    for index, atom in enumerate(atoms, start=1):
        element = str(getattr(atom, "element", "") or "").strip()
        if not re.fullmatch(r"[A-Za-z]{1,3}", element):
            raise ValueError(f"Atom {index} has an invalid element symbol {element!r}.")
        coordinates = (atom.x, atom.y, atom.z)
        try:
            finite = all(math.isfinite(float(value)) for value in coordinates)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(
                f"Atom {index} ({element}) has non-finite coordinates: {coordinates!r}."
            )
    validate_hyperpolarizability_request(qm_input)


def _collect_output_files() -> list[str]:
    return [name for name in OUTPUT_FILES if Path(name).exists()]


async def _run_optimization_chunks(qm_input: TurbomoleQMInput2, node_runner, kwargs: dict) -> None:
    tracker = OptimizationChartTracker(kwargs)
    last_cycles = 0
    last_energy_step = 0
    converged = False
    max_opt_cycles = int(qm_input.max_opt_cycles)
    try:
        while last_cycles < max_opt_cycles:
            chunk = min(OPT_CHART_INTERVAL, max_opt_cycles - last_cycles)
            run_script = prepend_tm_env(
                build_ground_state_script(
                    optimization=True,
                    use_ri=should_use_ri(qm_input),
                    gradients=False,
                    frequencies=False,
                    max_cycles=chunk,
                )
            )
            chunk_end = last_cycles + chunk
            subprocess_name = f"turbomole_exe_c{chunk_end:03d}"
            node_runner.info(
                f"Running jobex chunk cycles {last_cycles + 1}-{chunk_end} "
                f"(max {max_opt_cycles})."
            )
            ok = node_runner.subprocess(subprocess_name, run_script)
            tracker.update_from_directory(".")
            status, error = inspect_geometry_optimization(".")
            # Persist as soon as the chunk subprocess returns. Do not use
            # energy-file length as the cycle counter: jobex `-c 10` often
            # leaves 11 energy records (initial SCF + 10 opt cycles).
            await tracker.maybe_flush(force=True)
            if tracker.latest_step:
                node_runner.info(
                    f"Wrote optimization chart artifacts after jobex cycles "
                    f"{last_cycles + 1}-{chunk_end} "
                    f"({tracker.latest_step} energy record(s))."
                )

            if status == "converged":
                node_runner.info("Geometry optimization converged.")
                converged = True
                break
            if status == "failed":
                raise RuntimeError(
                    _with_runner_output(
                        node_runner, f"Geometry optimization failed: {error}"
                    )
                )
            if not ok and status != "continue":
                raise RuntimeError(
                    _with_runner_output(
                        node_runner,
                        f"Turbomole ground-state calculation failed. Check {subprocess_name}.log.",
                    )
                )

            new_energy_step = tracker.latest_step
            if new_energy_step <= last_energy_step:
                raise RuntimeError(
                    _with_runner_output(
                        node_runner,
                        "jobex made no progress; energy file did not gain a new cycle.",
                    )
                )
            last_energy_step = new_energy_step
            last_cycles += chunk
        if not converged:
            raise RuntimeError(
                f"Structure optimization did not converge in {max_opt_cycles} cycles."
            )
    finally:
        try:
            tracker.update_from_directory(".")
            await tracker.maybe_flush(force=True)
        except Exception as exc:
            node_runner.warning(f"Failed to store optimization charts: {exc}")


async def _run_ground_state(qm_input: TurbomoleQMInput2, node_runner, kwargs: dict) -> None:
    if qm_input.optimization:
        await _run_optimization_chunks(qm_input, node_runner, kwargs)
        if qm_input.frequencies:
            freq_script = prepend_tm_env(build_frequency_script())
            if not node_runner.subprocess("turbomole_aoforce", freq_script):
                raise RuntimeError(
                    _with_runner_output(
                        node_runner,
                        "Turbomole frequency calculation failed. Check turbomole_aoforce.log.",
                    )
                )
        return

    run_script = prepend_tm_env(
        build_ground_state_script(
            optimization=False,
            use_ri=should_use_ri(qm_input),
            gradients=bool(qm_input.gradients),
            frequencies=bool(qm_input.frequencies),
        )
    )
    if not node_runner.subprocess("turbomole_exe", run_script):
        raise RuntimeError(
            _with_runner_output(
                node_runner,
                "Turbomole ground-state calculation failed. Check turbomole_exe.log.",
            )
        )


@node(parameters=parameters)
async def turbomole2(qm_input: TurbomoleQMInput2, **kwargs) -> SimstackResult:
    """
    TURBOMOLE node for single-point, geometry optimization, and first hyperpolarizability (beta).

    Parameters:
        qm_input (TurbomoleQMInput2): Calculation settings and molecule.

    SimstackResult:
        result (QMResult): Final energy, structure, and convergence status.
        hyperpolarizability (SimpleTable): Frequency-pair beta_zzz in 10^-30 esu (z-dipole frame).
    """
    node_runner = kwargs["node_runner"]
    node_runner.info("Starting turbomole2 calculation")
    node_runner.info(
        "Request summary: "
        f"optimization={qm_input.optimization}, "
        f"gradients={qm_input.gradients}, "
        f"basis={qm_input.basis_set.basis_set}, "
        f"functional={qm_input.functional.keyword()}, "
        f"dispersion={qm_input.dispersion_enum().value}, "
        f"gridsize={qm_input.gridsize}, "
        f"scfconv={qm_input.scfconv}, "
        f"scfiterlimit={qm_input.scfiterlimit}, "
        f"max_opt_cycles={qm_input.max_opt_cycles}, "
        f"frequencies={qm_input.frequencies}, "
        f"hyperpolarizability={qm_input.hyperpolarizability.value}, "
        f"hyperpol_frequency_nm={hyperpolarizability_wavelength_nm(qm_input):.10g}"
    )

    try:
        _validate_request(qm_input)
    except Exception as exc:
        _fail(node_runner, f"Invalid TURBOMOLE input settings: {exc}")

    try:
        TurbomoleInputWriter(qm_input).write_files()
        node_runner.info("Input files generated")
    except Exception as exc:
        _fail(node_runner, f"Error creating Turbomole input files: {exc}")

    try:
        define_script = prepend_tm_env(build_define_script())
        if not node_runner.subprocess("turbomole_define", define_script):
            raise RuntimeError("Execution of Turbomole define failed")
        if not Path("./control").exists():
            raise RuntimeError(
                "Turbomole define produced no 'control' file. Check turbomole_define.log."
            )

        if qm_input.control_groups:
            appended = append_control_groups("control", qm_input.control_groups)
            node_runner.info(
                "Appended control_groups: "
                + ", ".join(group[0].split()[0] for group in appended)
            )

        await _run_ground_state(qm_input, node_runner, kwargs)

        if hyperpolarizability_requested(qm_input):
            wavelength_nm = hyperpolarizability_wavelength_nm(qm_input)
            apply_hyperpolarizability_control(qm_input)
            node_runner.info(
                "Launching TURBOMOLE escf hyperpolarizability response step "
                f"(mode={qm_input.hyperpolarizability.value}, lambda_nm={wavelength_nm:.10g})."
            )
            response_script = prepend_tm_env(build_hyperpolarizability_script())
            if not node_runner.subprocess("turbomole_response", response_script):
                raise RuntimeError(
                    "Turbomole hyperpolarizability step failed. Check turbomole_response.log and escf.out."
                )
            verify_dynamic_hyperpols_output(qm_input)
            hyperpol_table = parse_hyperpolarizability_table()
            if hyperpol_table is not None:
                node_runner.hyperpolarizability = hyperpol_table
                node_runner.info(
                    f"Parsed hyperpolarizability SimpleTable with {len(hyperpol_table.row)} frequency pair(s)."
                )
            else:
                node_runner.warning(
                    "escf produced hyperpols but no beta_zzz frequency pairs could be parsed."
                )

        if not Path("./control").exists():
            raise RuntimeError("Turbomole run produced no 'control' file.")
        if not (
            Path("./energy").exists()
            or Path("./ridft.out").exists()
            or Path("./dscf.out").exists()
            or Path("./job.last").exists()
            or Path("./hyperpols").exists()
            or Path("./vibspectrum").exists()
            or Path("./aoforce.out").exists()
        ):
            raise RuntimeError(
                "Turbomole run finished but key outputs are missing "
                "(energy/ridft.out/dscf.out/job.last/hyperpols/vibspectrum)."
            )

        node_runner.info("Parsing output files")
        tout = TurbomoleOutputParser(directory=".", node_runner=node_runner)
        tout.parse()

        if tout.final_energy is None:
            _fail(node_runner, "Failed to parse energy from Turbomole output")

        molecule_list = MoleculeList()
        if qm_input.optimization and tout.final_structure:
            molecule_list.add_molecule(tout.final_structure)

        written_xyz = write_final_geometry_xyz(tout.final_structure)
        if written_xyz is not None:
            node_runner.info(f"Wrote final geometry XYZ file: {written_xyz}")

        qm_result = QMResult(
            scf_converged=tout.properly_terminated,
            final_energy=tout.final_energy,
            energies=tout.energies,
            final_structure=tout.final_structure if tout.final_structure else qm_input.molecule,
            structures=molecule_list,
            vibrational_frequencies=tout.vibrational_frequencies,
            task_status=TaskStatus.COMPLETED,
        )
        node_runner.result = qm_result

        for out_file in _collect_output_files():
            file_stack = FileStack.from_local_file(
                out_file, in_memory=True, is_hashable=True, secure_source=True
            )
            await context.db.save(file_stack)
            qm_result.files.append(file_stack)

        node_runner.info(
            f"turbomole2 completed successfully with energy: {tout.final_energy}"
        )
        return node_runner.succeed()
    except Exception as exc:
        _fail(node_runner, _with_runner_output(node_runner, f"Turbomole calculation failed: {exc}"))
