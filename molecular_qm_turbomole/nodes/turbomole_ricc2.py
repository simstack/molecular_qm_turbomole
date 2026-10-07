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
from molecular_qm_turbomole.lib.ricc2_progress import Ricc2CrashProgress
from molecular_qm_turbomole.nodes.turbomole2 import (
    _collect_turbomole_info_files,
    _collect_turbomole_restart_files,
    _fail,
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
        result (QMResult): Final energy, structure, ADC(2) excited states, and
            occ→vir transition amplitudes.
    """
    node_runner = kwargs["node_runner"]
    method = qm_input.method_enum()
    progress = Ricc2CrashProgress(node_runner, kwargs, interval_s=30.0)

    enter_scratch = getattr(node_runner, "enter_scratch", None)
    if callable(enter_scratch):
        enter_scratch("turbomole_ricc2")
    try:
        progress.note("Starting turbomole_ricc2 calculation")
        progress.note(
            "Request summary: "
            f"method={method.value}, "
            f"basis={qm_input.basis_set.basis_set}, "
            f"states={qm_input.states}, "
            f"multiplicity={qm_input.multiplicity}, "
            f"scfconv={qm_input.scfconv}, "
            f"scfiterlimit={qm_input.scfiterlimit}, "
            f"solvent_mode={qm_input.solvent_mode.value}, "
            f"solvent={qm_input.solvent}"
        )
        await progress.publish_safely()
        try:
            validate_turbomole_ricc2_request(qm_input)
            validate_molecule_geometry(qm_input)
        except Exception as exc:
            _fail(node_runner, f"Invalid TURBOMOLE ricc2 input settings: {exc}")

        try:
            writer = TurbomoleInputWriter(qm_input)
            writer.write_files()
            progress.note("Input files generated")
            await progress.publish_safely()
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
            progress.note(
                "Applied wavefunction control groups: "
                + ", ".join(group[0].split()[0] for group in applied)
            )

        if qm_input.control_groups:
            appended = append_control_groups("control", qm_input.control_groups)
            progress.note(
                "Appended control_groups: "
                + ", ".join(group[0].split()[0] for group in appended)
            )

        cosmo_group = writer.apply_cosmo_control("control")
        if cosmo_group:
            progress.note("Configured TURBOMOLE COSMO: " + " ".join(cosmo_group))
        await progress.publish_safely()

        ok, _, _ = await progress.run_monitored(
            "turbomole_dscf",
            "TURBOMOLE HF reference (dscf)",
            command="dscf_command",
            output_name="dscf.out",
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
        excited_state_transitions = None
        if method == TurbomoleMethodEnum.HF:
            tout = TurbomoleOutputParser(directory=".", node_runner=node_runner)
            tout.parse()
            final_energy = tout.final_energy
            properly_terminated = tout.properly_terminated
            final_structure = tout.final_structure
            energies = tout.energies
        else:
            ok, _, _ = await progress.run_monitored(
                "turbomole_ricc2",
                f"TURBOMOLE {method.value} (ricc2)",
                command="ricc2_command",
                output_name="ricc2.out",
            )
            if not ok:
                raise RuntimeError(
                    _with_runner_output(
                        node_runner,
                        "Turbomole ricc2 calculation failed. Check turbomole_ricc2.log and ricc2.out.",
                    )
                )
            final_energy, excited_states, excited_state_transitions = parse_ricc2_file(
                "ricc2.out"
            )
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
            progress.note(f"Wrote final geometry XYZ file: {written_xyz}")

        qm_result = QMResult(
            scf_converged=properly_terminated,
            final_energy=final_energy,
            energies=energies,
            final_structure=final_structure if final_structure else qm_input.molecule,
            structures=MoleculeList(),
            excited_states=excited_states,
            excited_state_transitions=excited_state_transitions,
            task_status=TaskStatus.COMPLETED,
        )
        node_runner.result = qm_result

        await _collect_turbomole_restart_files(node_runner, qm_result)
        _collect_turbomole_info_files(node_runner)
        progress.note(
            f"turbomole_ricc2 completed successfully with energy: {final_energy}"
        )
        await progress.publish_safely()
        return node_runner.succeed()
    except Exception as exc:
        await progress.publish_safely()
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
