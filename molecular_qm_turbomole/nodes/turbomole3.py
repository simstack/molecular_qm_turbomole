"""TURBOMOLE node that preoptimizes a geometry, then runs turbomole2."""

from typing import Any, Optional, Tuple

from odmantic import ObjectId

from molecular_qm_models import Molecule, QMResult
from molecular_qm_turbomole.models.multistep_optimizer import PreOptimizerInput
from molecular_qm_turbomole.models.semiempirical import (
    TurbomoleSemiempiricalInput,
    TurbomoleSemiempiricalMethod,
)
from molecular_qm_turbomole.models.turbomole_input import (
    HyperpolarizabilityModeEnum,
    TurbomoleQMInput2,
)
from molecular_qm_turbomole.nodes.multistep_optimizer import (
    _append_step_row,
    _child_qm_result,
    _molecule_from_qm_result,
    _persist_semiempirical_input,
    _persist_step_molecule,
    _persist_turbomole_input,
    _qm_input_for_step,
    _setting_label,
    _timings_from_child_result,
)
from molecular_qm_turbomole.nodes.turbomole2 import parameters, turbomole2
from molecular_qm_turbomole.nodes.turbomole_semiempirical import turbomole_semiempirical
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models.simple_table import SimpleTable


def _required_positive_int(value: Any, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be >= 1, got {number}")
    return number


async def _call_step(
    awaitable, step_name: str
) -> Tuple[Any, Optional[QMResult], Optional[str]]:
    """Run one child node. A raised error or a failed result fails this step."""
    try:
        calc_result = await awaitable
    except Exception as exc:
        return None, None, f"{step_name} failed: {exc}"
    qm_result, error = _child_qm_result(calc_result)
    if error:
        return calc_result, None, f"{step_name} failed: {error}"
    return calc_result, qm_result, None


def _new_step_table() -> SimpleTable:
    step_table = SimpleTable(name="turbomole3 preoptimizer")
    step_table.add_column("step", "string")
    step_table.add_column("basis_set", "string")
    step_table.add_column("functional", "string")
    step_table.add_column("dispersion_correction", "string")
    step_table.add_column("scf_accuracy", "string")
    step_table.add_column("optimization_accuracy", "string")
    step_table.add_column("grid_type", "string")
    step_table.add_column("energy", "number")
    step_table.add_column("n_iterations", "number")
    step_table.add_column("wall_time_s", "number")
    step_table.add_column("cpu_time_s", "number")
    step_table.add_column("freq_wall_time_s", "number")
    step_table.add_column("freq_cpu_time_s", "number")
    return step_table


def _remember_timing(bucket: list, value) -> None:
    if value is not None:
        bucket.append(value)


def _adopt_final_result(node_runner, calc_result: Any, qm_result: QMResult) -> None:
    """Expose the final turbomole2 outputs on this node."""
    node_runner.result = qm_result
    node_runner.qm_result = qm_result
    for name in ("hyperpolarizability", "optimization_timing"):
        value = getattr(calc_result, name, None)
        if value is not None:
            setattr(node_runner, name, value)
    files = getattr(calc_result, "files", None) or []
    if files:
        node_runner.files = list(files)
    info_files = getattr(calc_result, "info_files", None) or []
    if info_files:
        node_runner.info_files = list(info_files)


@node(parameters=parameters)
async def turbomole3(
    qm_input: TurbomoleQMInput2,
    preopt: PreOptimizerInput,
    **kwargs,
) -> SimstackResult:
    """
    Run preoptimizer steps, then a turbomole2 calculation with TurbomoleQMInput2 settings.

    Semiempirical pre-optimization and each DFT preoptimizer step run in order.
    Each step receives the molecule produced by the previous step. A failed step
    fails this call and later steps are not started. After the preoptimizer,
    the original basis set, functional, dispersion, grid, solvent, optimization,
    frequencies, and hyperpolarizability settings are used for one turbomole2 run
    on the resulting molecule.

    An empty ``PreOptimizerInput`` skips the preoptimizer and runs turbomole2
    on the input molecule.

    Parameters:
        qm_input (TurbomoleQMInput2): Final calculation settings and starting molecule.
        preopt (PreOptimizerInput): Semiempirical pre-opt and DFT preoptimizer steps.

    Called Nodes:
        turbomole_semiempirical
        turbomole2

    SimstackResult:
        result (QMResult): Final turbomole2 energy, structure, and convergence status.
        qm_result (QMResult): Same final turbomole2 result.
        files (List[FileStack]): Restart and output files from the final turbomole2 run.
        info_files (List[FileStack]): Logs and diagnostics from the final turbomole2 run.
        hyperpolarizability (SimpleTable): Frequency-pair beta_zzz from the final run.
        optimization_timing (SimpleTable): Wall/CPU times from the final turbomole2 run.
        step_table (SimpleTable): Per-step settings, energy, iterations, and timings.
            Present when the preoptimizer ran.
    """
    node_runner = kwargs["node_runner"]
    steps = list(preopt.steps or [])
    has_semiempirical = bool(
        preopt.semiempirical_opt or preopt.semiempirical_method is not None
    )
    max_opt_cycles = _required_positive_int(qm_input.max_opt_cycles, "max_opt_cycles")
    scfiterlimit = _required_positive_int(qm_input.scfiterlimit, "scfiterlimit")
    molecule = qm_input.molecule

    step_table = None
    step_walls = []
    step_cpus = []
    step_iters = []
    step_freq_walls = []
    step_freq_cpus = []

    if has_semiempirical or steps:
        step_table = _new_step_table()
        node_runner.info(
            f"Starting turbomole3 preoptimizer: semiempirical={has_semiempirical}, "
            f"dft_steps={len(steps)}"
        )

    if has_semiempirical:
        if preopt.semiempirical_method is not None:
            semi_method = preopt.semiempirical_method
        elif (
            preopt.semiempirical_input is not None
            and preopt.semiempirical_input.method is not None
        ):
            semi_method = preopt.semiempirical_input.method
        else:
            raise ValueError(
                "semiempirical preoptimizer is enabled but semiempirical_method is not set"
            )
        if isinstance(semi_method, str):
            semi_method = TurbomoleSemiempiricalMethod.coerce(semi_method)

        if preopt.semiempirical_input is not None:
            semi_opts = preopt.semiempirical_input.model_copy(deep=True)
            semi_opts.method = semi_method
            semi_opts.charge = qm_input.charge
            semi_opts.multiplicity = qm_input.multiplicity
            semi_opts.optimization = True
            semi_opts.max_opt_cycles = max_opt_cycles
            semi_opts.scfiterlimit = scfiterlimit
        else:
            semi_opts = TurbomoleSemiempiricalInput(
                method=semi_method,
                charge=qm_input.charge,
                multiplicity=qm_input.multiplicity,
                optimization=True,
                max_opt_cycles=max_opt_cycles,
                scfiterlimit=scfiterlimit,
            )

        semi_opts = await _persist_semiempirical_input(semi_opts, node_runner)
        method_label = (
            semi_opts.method.value
            if hasattr(semi_opts.method, "value")
            else str(semi_opts.method)
        )
        node_runner.info(
            f"Starting semiempirical preoptimizer step "
            f"(method={method_label}, max_opt_cycles={max_opt_cycles}, "
            f"scfiterlimit={scfiterlimit})"
        )
        semi_kwargs = dict(kwargs)
        semi_kwargs["custom_name"] = f"semiempirical/{method_label}"
        calc_result, qm_result, error = await _call_step(
            turbomole_semiempirical(molecule, semi_opts, **semi_kwargs),
            "semiempirical",
        )
        if error:
            node_runner.step_table = step_table
            return node_runner.fail(error)

        next_mol = _molecule_from_qm_result(
            qm_result, molecule, node_runner, "semiempirical"
        )
        if next_mol is not molecule:
            molecule = await _persist_step_molecule(
                next_mol, node_runner, "semiempirical"
            )
        wall_s, cpu_s, n_iterations, freq_wall_s, freq_cpu_s = _timings_from_child_result(
            calc_result
        )
        _remember_timing(step_walls, wall_s)
        _remember_timing(step_cpus, cpu_s)
        _remember_timing(step_iters, n_iterations)
        _remember_timing(step_freq_walls, freq_wall_s)
        _remember_timing(step_freq_cpus, freq_cpu_s)
        _append_step_row(
            step_table,
            "semiempirical",
            method_label,
            method_label,
            qm_result,
            node_runner,
            n_iterations=n_iterations,
            wall_time_s=wall_s,
            cpu_time_s=cpu_s,
            freq_wall_time_s=freq_wall_s,
            freq_cpu_time_s=freq_cpu_s,
        )

    for index, step in enumerate(steps, start=1):
        step_name = f"preopt-{index}"
        basis_name = step.basis_set.basis_set
        functional_name = step.functional.keyword()
        dispersion_name = _setting_label(step.dispersion_correction.value)
        grid_name = step.grid
        node_runner.info(
            f"Starting {step_name}: basis={basis_name}, functional={functional_name}, "
            f"dispersion={dispersion_name}, grid={grid_name}, "
            f"scf_accuracy={_setting_label(step.scf_accuracy)}, "
            f"optimization_accuracy={_setting_label(step.optimization_accuracy)}, "
            f"max_opt_cycles={max_opt_cycles}, scfiterlimit={scfiterlimit}"
        )
        current_input = _qm_input_for_step(
            qm_input,
            step,
            molecule,
            frequencies=False,
            max_opt_cycles=max_opt_cycles,
            scfiterlimit=scfiterlimit,
        ).model_copy(
            update={
                "hyperpolarizability": HyperpolarizabilityModeEnum.NONE,
                "gradients": False,
                "frequencies": False,
            }
        )
        current_input = await _persist_turbomole_input(current_input, node_runner)
        step_kwargs = dict(kwargs)
        step_kwargs["custom_name"] = f"{step_name}/{basis_name}/{functional_name}"
        calc_result, qm_result, error = await _call_step(
            turbomole2(current_input, **step_kwargs),
            step_name,
        )
        if error:
            node_runner.step_table = step_table
            return node_runner.fail(error)

        next_mol = _molecule_from_qm_result(qm_result, molecule, node_runner, step_name)
        if next_mol is not molecule:
            molecule = await _persist_step_molecule(next_mol, node_runner, step_name)
        wall_s, cpu_s, n_iterations, freq_wall_s, freq_cpu_s = _timings_from_child_result(
            calc_result
        )
        _remember_timing(step_walls, wall_s)
        _remember_timing(step_cpus, cpu_s)
        _remember_timing(step_iters, n_iterations)
        _remember_timing(step_freq_walls, freq_wall_s)
        _remember_timing(step_freq_cpus, freq_cpu_s)
        _append_step_row(
            step_table,
            step_name,
            basis_name,
            functional_name,
            qm_result,
            node_runner,
            dispersion_correction=dispersion_name,
            scf_accuracy=_setting_label(step.scf_accuracy),
            optimization_accuracy=_setting_label(step.optimization_accuracy),
            grid_type=grid_name,
            n_iterations=n_iterations,
            wall_time_s=wall_s,
            cpu_time_s=cpu_s,
            freq_wall_time_s=freq_wall_s,
            freq_cpu_time_s=freq_cpu_s,
        )

    final_input = qm_input
    if molecule is not qm_input.molecule:
        final_input = await _persist_turbomole_input(
            qm_input.model_copy(
                deep=True,
                update={"id": ObjectId(), "molecule": molecule},
            ),
            node_runner,
        )
    basis_name = final_input.basis_set.basis_set
    functional_name = final_input.functional.keyword()
    node_runner.info(
        "Starting turbomole3 final calculation with TurbomoleQMInput2 settings: "
        f"basis={basis_name}, functional={functional_name}, "
        f"optimization={final_input.optimization}, frequencies={final_input.frequencies}, "
        f"hyperpolarizability={final_input.hyperpolarizability.value}, "
        f"gridsize={final_input.gridsize}"
    )
    final_kwargs = dict(kwargs)
    final_kwargs["custom_name"] = f"{basis_name}/{functional_name}"
    calc_result, qm_result, error = await _call_step(
        turbomole2(final_input, **final_kwargs),
        "turbomole3",
    )
    if error:
        if step_table is not None:
            node_runner.step_table = step_table
        return node_runner.fail(error)

    if step_table is not None:
        wall_s, cpu_s, n_iterations, freq_wall_s, freq_cpu_s = _timings_from_child_result(
            calc_result
        )
        _remember_timing(step_walls, wall_s)
        _remember_timing(step_cpus, cpu_s)
        _remember_timing(step_iters, n_iterations)
        _remember_timing(step_freq_walls, freq_wall_s)
        _remember_timing(step_freq_cpus, freq_cpu_s)
        _append_step_row(
            step_table,
            "turbomole3",
            basis_name,
            functional_name,
            qm_result,
            node_runner,
            dispersion_correction=_setting_label(final_input.dispersion_enum().value),
            scf_accuracy=str(final_input.scfconv),
            grid_type=final_input.gridsize,
            n_iterations=n_iterations,
            wall_time_s=wall_s,
            cpu_time_s=cpu_s,
            freq_wall_time_s=freq_wall_s,
            freq_cpu_time_s=freq_cpu_s,
        )
        _append_step_row(
            step_table,
            "total",
            "",
            "",
            None,
            node_runner,
            n_iterations=sum(step_iters) if step_iters else None,
            wall_time_s=sum(step_walls) if step_walls else None,
            cpu_time_s=sum(step_cpus) if step_cpus else None,
            freq_wall_time_s=sum(step_freq_walls) if step_freq_walls else None,
            freq_cpu_time_s=sum(step_freq_cpus) if step_freq_cpus else None,
        )
        node_runner.step_table = step_table

    _adopt_final_result(node_runner, calc_result, qm_result)
    node_runner.info(
        f"turbomole3 completed successfully with energy: {qm_result.final_energy}"
    )
    return node_runner.succeed()
