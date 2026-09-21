from pathlib import Path

from molecular_qm_models.molecule import MoleculeList
from molecular_qm_models.qm_result import QMResult
from molecular_qm_turbomole.lib.control_utils import append_control_groups
from molecular_qm_turbomole.lib.input_writer import TurbomoleInputWriter
from molecular_qm_turbomole.lib.output_parser import (
    TurbomoleOutputParser,
    parse_coord_file,
    parse_ricc2_file,
    require_turbomole_normal_termination,
    write_final_geometry_xyz,
)
from molecular_qm_turbomole.lib.request_validation import validate_molecule_geometry
from molecular_qm_turbomole.models.turbomole_input import (
    TurbomoleMethodEnum,
    TurbomoleQMInput2,
    validate_turbomole_ricc2_request,
)
from molecular_qm_turbomole.nodes.turbomole2 import (
    _collect_turbomole_info_files,
    _collect_turbomole_restart_files,
    _fail,
    _run_monitored_subprocess,
    _with_runner_output,
    parameters,
)
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult





@node(parameters=parameters)
async def turbomole_ricc2(qm_input: TurbomoleQMInput2, **kwargs) -> SimstackResult:
    """
    TURBOMOLE node for single-reference wavefunction methods (HF, MP2, CC2, ADC(2)).

    Runs ``define``, ``dscf``, and ``ricc2`` (except HF, which stops after ``dscf``)
    via ``ResourceConfig.run("turbomole")`` using ``define_command``,
    ``dscf_command``, and ``ricc2_command``. Geometry optimization, gradients,
    frequencies, and hyperpolarizability are not supported.

    Parameters:
        qm_input (TurbomoleQMInput2): Calculation settings and molecule. ``method``
            must be HF, MP2, CC2, or ADC(2).

    SimstackResult:
        result (QMResult): Final energy, structure, and ADC(2) excited states.
    """
    node_runner = kwargs["node_runner"]
    method = qm_input.method_enum()
    node_runner.info("Starting turbomole_ricc2 calculation")
    node_runner.info(
        "Request summary: "
        f"method={method.value}, "
        f"basis={qm_input.basis_set.basis_set}, "
        f"states={qm_input.states}, "
        f"scfconv={qm_input.scfconv}, "
        f"scfiterlimit={qm_input.scfiterlimit}"
    )

    enter_scratch = getattr(node_runner, "enter_scratch", None)
    if callable(enter_scratch):
        enter_scratch("turbomole_ricc2")
    try:
        try:
            validate_turbomole_ricc2_request(qm_input)
            validate_molecule_geometry(qm_input)
        except Exception as exc:
            _fail(node_runner, f"Invalid TURBOMOLE ricc2 input settings: {exc}")

        try:
            writer = TurbomoleInputWriter(qm_input)
            writer.write_files()
            node_runner.info("Input files generated")
        except Exception as exc:
            _fail(node_runner, f"Error creating Turbomole input files: {exc}")

        ok = context.resource_config.run(
            "turbomole",
            node_runner=node_runner,
            command="define_command",
            name="turbomole_define",
        )
        if not ok:
            raise RuntimeError("Execution of Turbomole define failed")
        if not Path("./control").exists():
            raise RuntimeError(
                "Turbomole define produced no 'control' file. Check turbomole_define.log."
            )

        applied = writer.apply_wavefunction_control("control")
        if applied:
            node_runner.info(
                "Applied wavefunction control groups: "
                + ", ".join(group[0].split()[0] for group in applied)
            )

        if qm_input.control_groups:
            appended = append_control_groups("control", qm_input.control_groups)
            node_runner.info(
                "Appended control_groups: "
                + ", ".join(group[0].split()[0] for group in appended)
            )

        ok, _, _ = _run_monitored_subprocess(
            node_runner,
            "turbomole_dscf",
            "TURBOMOLE HF reference (dscf)",
            kwargs,
            command="dscf_command",
        )
        if not ok:
            raise RuntimeError(
                _with_runner_output(
                    node_runner,
                    "Turbomole dscf calculation failed. Check turbomole_dscf.log.",
                )
            )
        require_turbomole_normal_termination("dscf.out", "dscf")

        excited_states = None
        if method == TurbomoleMethodEnum.HF:
            tout = TurbomoleOutputParser(directory=".", node_runner=node_runner)
            tout.parse()
            final_energy = tout.final_energy
            properly_terminated = tout.properly_terminated
            final_structure = tout.final_structure
            energies = tout.energies
        else:
            ok, _, _ = _run_monitored_subprocess(
                node_runner,
                "turbomole_ricc2",
                f"TURBOMOLE {method.value} (ricc2)",
                kwargs,
                command="ricc2_command",
            )
            if not ok:
                raise RuntimeError(
                    _with_runner_output(
                        node_runner,
                        "Turbomole ricc2 calculation failed. Check turbomole_ricc2.log and ricc2.out.",
                    )
                )
            final_energy, excited_states = parse_ricc2_file("ricc2.out")
            if method == TurbomoleMethodEnum.ADC2 and (
                excited_states is None or not excited_states.row
            ):
                raise ValueError(
                    "ADC(2) finished but no excitation energies could be parsed from ricc2.out."
                )
            properly_terminated = True
            final_structure = parse_coord_file(Path("coord"))
            energies = [final_energy]

        if not Path("./control").exists():
            raise RuntimeError("Turbomole run produced no 'control' file.")
        if method == TurbomoleMethodEnum.HF:
            if not (Path("./energy").exists() or Path("./dscf.out").exists()):
                raise RuntimeError(
                    "Turbomole HF run finished but energy/dscf.out are missing."
                )
        elif not Path("./ricc2.out").exists():
            raise RuntimeError("Turbomole ricc2 run finished but ricc2.out is missing.")

        if final_energy is None:
            _fail(node_runner, "Failed to parse energy from Turbomole output")

        written_xyz = write_final_geometry_xyz(final_structure)
        if written_xyz is not None:
            node_runner.info(f"Wrote final geometry XYZ file: {written_xyz}")

        qm_result = QMResult(
            scf_converged=properly_terminated,
            final_energy=final_energy,
            energies=energies,
            final_structure=final_structure if final_structure else qm_input.molecule,
            structures=MoleculeList(),
            excited_states=excited_states,
            task_status=TaskStatus.COMPLETED,
        )
        node_runner.result = qm_result

        await _collect_turbomole_restart_files(node_runner, qm_result)
        _collect_turbomole_info_files(node_runner)
        node_runner.info(
            f"turbomole_ricc2 completed successfully with energy: {final_energy}"
        )
        return node_runner.succeed()
    except Exception as exc:
        await _collect_turbomole_restart_files(node_runner)
        _collect_turbomole_info_files(node_runner)
        _fail(
            node_runner,
            _with_runner_output(node_runner, f"Turbomole ricc2 calculation failed: {exc}"),
        )
    finally:
        leave_scratch = getattr(node_runner, "leave_scratch", None)
        if callable(leave_scratch):
            leave_scratch()
