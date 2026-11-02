import logging
from typing import Any, List, Optional, Tuple, Union

from odmantic import ObjectId

from molecular_qm_models import (
    Molecule,
    OptimizationAccuracy,
    QMResult,
    SCFAccuracy,
)
from molecular_qm_turbomole.models.multistep_optimizer import (
    OptimizationStepInput,
    PreOptimizerInput,
)
from molecular_qm_turbomole.models.semiempirical import (
    TurbomoleSemiempiricalInput,
    TurbomoleSemiempiricalMethod,
)
from molecular_qm_turbomole.models.turbomole_functional import (
    TurbomoleFunctional,
    TurbomoleFunctionalEnum,
)
from molecular_qm_turbomole.models.turbomole_input import (
    TurbomoleBasisSet2,
    TurbomoleDispersionCorrection,
    TurbomoleMethodEnum,
    TurbomoleQMInput2,
)
from molecular_qm_turbomole.nodes.turbomole2 import parameters, turbomole2
from molecular_qm_turbomole.nodes.turbomole_semiempirical import turbomole_semiempirical
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models.simple_table import SimpleTable

logger = logging.getLogger("TurbomoleMultistepOptimizer")


def _scfconv_from_accuracy(
    scf_accuracy: Optional[Union[SCFAccuracy, str]] = None,
    optimization_accuracy: Optional[Union[OptimizationAccuracy, str]] = None,
    default_conv: int = 8,
) -> int:
    """Map SCF / Optimization accuracy levels to TURBOMOLE scfconv exponent."""
    mapping = {
        "sloppy": 5,
        "coarse": 6,
        "loose": 6,
        "medium": 7,
        "strong": 8,
        "tight": 8,
        "fine": 8,
        "verytight": 9,
        "veryfine": 9,
        "extreme": 10,
    }
    if scf_accuracy is not None:
        scf_acc_val = getattr(scf_accuracy, "value", scf_accuracy)
        key = str(scf_acc_val).strip().lower()
        if key in mapping:
            return mapping[key]

    if optimization_accuracy is not None:
        opt_acc_val = getattr(optimization_accuracy, "value", optimization_accuracy)
        key = str(opt_acc_val).strip().lower()
        if key in mapping:
            return mapping[key]

    return default_conv


def _setting_label(value: Any) -> str:
    if value is None:
        return ""
    while True:
        inner = getattr(value, "value", value)
        if inner is value:
            break
        value = inner
        if value is None:
            return ""
    return str(value)


def _child_qm_result(calc_result: Any) -> Tuple[Optional[QMResult], Optional[str]]:
    """Return (qm_result, error_message). error_message is None on success."""
    if calc_result is None:
        return None, "child node returned no result"
    if isinstance(calc_result, SimstackResult):
        if calc_result.status != TaskStatus.COMPLETED:
            return None, calc_result.error_message or "child node failed"
        for name in ("result", "qm_result", "turbomole_result"):
            value = getattr(calc_result, name, None)
            if isinstance(value, QMResult):
                return value, None
        return None, "child node returned no QMResult"
    if isinstance(calc_result, QMResult):
        return calc_result, None
    for name in ("result", "qm_result", "turbomole_result"):
        value = getattr(calc_result, name, None)
        if isinstance(value, QMResult):
            return value, None
    return None, f"unexpected child result type: {type(calc_result)}"


def _timings_from_child_result(calc_result: Any) -> Tuple[
    Optional[float], Optional[float], Optional[int], Optional[float], Optional[float]
]:
    """Extract (opt_wall_s, opt_cpu_s, n_iterations, freq_wall_s, freq_cpu_s) from child node result."""
    table = getattr(calc_result, "optimization_timing", None)
    if table is None and isinstance(calc_result, SimstackResult):
        table = getattr(calc_result, "optimization_timing", None)
    if table is None:
        return None, None, None, None, None

    rows = getattr(table, "row", []) or []
    opt_wall_s = None
    opt_cpu_s = None
    freq_wall_s = None
    freq_cpu_s = None
    iterations = []

    for r in rows:
        metric = r.get("metric")
        if metric == "optimize":
            opt_wall_s = r.get("wall_time_s")
            opt_cpu_s = r.get("cpu_time_s")
        elif metric == "total" and opt_wall_s is None:
            opt_wall_s = r.get("wall_time_s")
            opt_cpu_s = r.get("cpu_time_s")
        elif metric == "iteration":
            iterations.append(r.get("step"))
        elif metric == "frequencies":
            freq_wall_s = r.get("wall_time_s")
            freq_cpu_s = r.get("cpu_time_s")

    n_iterations = len(iterations) if iterations else None
    return opt_wall_s, opt_cpu_s, n_iterations, freq_wall_s, freq_cpu_s


def _molecule_from_qm_result(
    qm_result: Optional[QMResult],
    fallback: Molecule,
    node_runner: Any,
    step_name: str,
) -> Molecule:
    """Return the last step geometry, or fallback if none was stored."""
    structure = None if qm_result is None else getattr(qm_result, "final_structure", None)
    atoms = getattr(structure, "atoms", None) if structure is not None else None
    if not atoms:
        node_runner.warning(
            f"{step_name} returned no final_structure; reusing previous geometry"
        )
        return fallback
    next_mol = Molecule.from_molecule(structure)
    first = next_mol.atoms[0]
    node_runner.info(
        f"{step_name} propagating final_structure: {len(next_mol.atoms)} atoms, "
        f"{first.element}={first.x:.6f},{first.y:.6f},{first.z:.6f}"
    )
    return next_mol


async def _persist_step_molecule(
    molecule: Molecule, node_runner: Any, step_name: str, db: Any = None
) -> Molecule:
    """Save geometry so child nodes / references can reload it."""
    if db is None:
        try:
            db = context.db
        except Exception as exc:
            node_runner.warning(f"Could not persist {step_name} geometry: {exc}")
            return molecule
    if db is None:
        return molecule
    try:
        saved = await db.save(molecule)
    except Exception as exc:
        node_runner.warning(f"Could not persist {step_name} geometry: {exc}")
        return molecule
    return saved if saved is not None else molecule


async def _persist_turbomole_input(
    opts: TurbomoleQMInput2, node_runner: Any, db: Any = None
) -> TurbomoleQMInput2:
    """Save step TurbomoleQMInput2 to DB."""
    if db is None:
        try:
            db = context.db
        except Exception as exc:
            node_runner.warning(f"Could not persist TurbomoleQMInput2: {exc}")
            return opts
    if db is None:
        return opts
    try:
        saved = await db.save(opts)
    except Exception as exc:
        node_runner.warning(f"Could not persist TurbomoleQMInput2: {exc}")
        return opts
    return saved if saved is not None else opts


async def _persist_semiempirical_input(
    opts: TurbomoleSemiempiricalInput, node_runner: Any, db: Any = None
) -> TurbomoleSemiempiricalInput:
    """Save TurbomoleSemiempiricalInput to DB."""
    if db is None:
        try:
            db = context.db
        except Exception as exc:
            node_runner.warning(f"Could not persist TurbomoleSemiempiricalInput: {exc}")
            return opts
    if db is None:
        return opts
    try:
        saved = await db.save(opts)
    except Exception as exc:
        node_runner.warning(f"Could not persist TurbomoleSemiempiricalInput: {exc}")
        return opts
    return saved if saved is not None else opts


def _qm_input_for_step(
    base_input: TurbomoleQMInput2,
    step: OptimizationStepInput,
    molecule: Molecule,
    frequencies: bool,
    max_opt_cycles: int,
    scfiterlimit: int,
) -> TurbomoleQMInput2:
    """Build TurbomoleQMInput2 for one optimization step, inheriting max opt cycles and scf steps."""
    if not isinstance(base_input, TurbomoleQMInput2):
        raise ValueError(
            "multistep_optimizer qm_input must be TurbomoleQMInput2, "
            f"got {type(base_input).__name__}"
        )
    return base_input.model_copy(
        deep=True,
        update={
            "id": ObjectId(),
            "molecule": molecule,
            "optimization": True,
            "frequencies": frequencies,
            "basis_set": step.basis_set,
            "functional": step.functional,
            "dispersion_correction": step.dispersion_correction,
            "gridsize": step.grid,
            "max_opt_cycles": max_opt_cycles,
            "scfiterlimit": scfiterlimit,
            "scfconv": _scfconv_from_accuracy(
                step.scf_accuracy,
                step.optimization_accuracy,
                default_conv=base_input.scfconv,
            ),
            "field_name": "TurbomoleQMInput2",
        },
    )


def _append_step_row(
    table: SimpleTable,
    step_name: str,
    basis_set: str,
    functional: str,
    qm_result: Optional[QMResult],
    node_runner: Any,
    dispersion_correction: str = "",
    scf_accuracy: str = "",
    optimization_accuracy: str = "",
    grid_type: str = "",
    n_iterations=None,
    wall_time_s=None,
    cpu_time_s=None,
    freq_wall_time_s=None,
    freq_cpu_time_s=None,
) -> None:
    energy = None if qm_result is None else qm_result.final_energy
    table.add_row(
        {
            "step": step_name,
            "basis_set": basis_set,
            "functional": functional,
            "dispersion_correction": dispersion_correction,
            "scf_accuracy": scf_accuracy,
            "optimization_accuracy": optimization_accuracy,
            "grid_type": grid_type,
            "energy": energy,
            "n_iterations": n_iterations,
            "wall_time_s": wall_time_s,
            "cpu_time_s": cpu_time_s,
            "freq_wall_time_s": freq_wall_time_s,
            "freq_cpu_time_s": freq_cpu_time_s,
        }
    )
    node_runner.info(
        f"step={step_name} basis_set={basis_set} functional={functional} "
        f"dispersion_correction={dispersion_correction} "
        f"scf_accuracy={scf_accuracy} optimization_accuracy={optimization_accuracy} "
        f"grid_type={grid_type} energy={energy} n_iterations={n_iterations} "
        f"wall_time_s={wall_time_s} cpu_time_s={cpu_time_s} "
        f"freq_wall_time_s={freq_wall_time_s} freq_cpu_time_s={freq_cpu_time_s}"
    )


@node(parameters=parameters)
async def multistep_optimizer(
    qm_input: TurbomoleQMInput2,
    preopt: PreOptimizerInput,
    **kwargs,
) -> SimstackResult:
    """
    Run an optional TURBOMOLE semiempirical pre-optimization followed by sequential DFT optimizations.

    Each DFT step uses the previous step's final geometry. Charge, solvent, and
    other QM settings are copied from ``qm_input``; each step overrides
    basis set, functional, dispersion, grid, and accuracy.
    The maximum number of optimization steps and SCF steps are taken from ``qm_input`` for all steps.
    Frequencies from ``qm_input`` run only on the last step.

    Parameters:
        qm_input (TurbomoleQMInput2): Molecule and shared QM settings used as copy template.
        preopt (PreOptimizerInput): Semiempirical pre-opt settings and sequence of DFT optimization steps.

    Called Nodes:
        turbomole_semiempirical
        turbomole2

    SimstackResult:
        result (QMResult): Result of the last successful step.
        qm_result (QMResult): Result of the last successful step.
        step_table (SimpleTable): Per-step settings, energy, iterations, and wall/CPU timings.
    """
    node_runner = kwargs["node_runner"]
    steps = list(preopt.steps or [])
    has_semiempirical = bool(preopt.semiempirical_opt or preopt.semiempirical_method is not None)

    if not has_semiempirical and not steps:
        return node_runner.fail(
            "PreOptimizerInput has no steps and semiempirical_opt is False"
        )

    molecule = qm_input.molecule
    max_opt_cycles = int(
        getattr(qm_input, "max_opt_cycles", None)
        or getattr(qm_input, "max_optimization_iterations", None)
        or getattr(qm_input, "max_optimization_steps", None)
        or 100
    )
    scfiterlimit = int(
        getattr(qm_input, "scfiterlimit", None)
        or getattr(qm_input, "max_scf_iterations", None)
        or 100
    )

    last_result: Optional[QMResult] = None
    step_table = SimpleTable(name="Multistep optimizer")
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

    tolerate_failure = bool(getattr(qm_input, "tolerate_failure", False))
    step_walls = []
    step_cpus = []
    step_iters = []
    step_freq_walls = []
    step_freq_cpus = []

    # 1. Semiempirical pre-optimization step
    if has_semiempirical:
        semi_method = (
            preopt.semiempirical_method
            or (preopt.semiempirical_input.method if preopt.semiempirical_input else TurbomoleSemiempiricalMethod.AM1)
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
        method_label = semi_opts.method.value if hasattr(semi_opts.method, "value") else str(semi_opts.method)
        node_runner.info(
            f"Starting TURBOMOLE semiempirical pre-optimization "
            f"(method={method_label}, max_opt_cycles={max_opt_cycles}, scfiterlimit={scfiterlimit})"
        )

        try:
            calc_result = await turbomole_semiempirical(molecule, semi_opts, **kwargs)
        except Exception as exc:
            error = str(exc)
            qm_result = None
            calc_result = None
        else:
            qm_result, error = _child_qm_result(calc_result)

        if error:
            if tolerate_failure:
                node_runner.warning(f"Semiempirical pre-optimization failed: {error}")
            else:
                return node_runner.fail(f"Semiempirical pre-optimization failed: {error}")
        else:
            last_result = qm_result
            next_mol = _molecule_from_qm_result(qm_result, molecule, node_runner, "semiempirical")
            if next_mol is not molecule:
                molecule = await _persist_step_molecule(next_mol, node_runner, "semiempirical")

            wall_s, cpu_s, n_iterations, freq_wall_s, freq_cpu_s = _timings_from_child_result(calc_result)
            if wall_s is not None:
                step_walls.append(wall_s)
            if cpu_s is not None:
                step_cpus.append(cpu_s)
            if n_iterations is not None:
                step_iters.append(n_iterations)
            if freq_wall_s is not None:
                step_freq_walls.append(freq_wall_s)
            if freq_cpu_s is not None:
                step_freq_cpus.append(freq_cpu_s)

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
            node_runner.info("TURBOMOLE semiempirical pre-optimization finished")

    # 2. Sequential DFT optimization steps
    for index, step in enumerate(steps, start=1):
        step_name = f"turbomole-{index}"
        basis_name = step.basis_set.basis_set
        functional_name = step.functional.functional.value
        dispersion_name = _setting_label(step.dispersion_correction.value)
        grid_name = step.grid
        run_frequencies = bool(getattr(qm_input, "frequencies", False)) and index == len(steps)

        node_runner.info(
            f"Starting {step_name}: basis={basis_name}, functional={functional_name}, "
            f"dispersion={dispersion_name}, grid={grid_name}, "
            f"scf_accuracy={_setting_label(step.scf_accuracy)}, "
            f"optimization_accuracy={_setting_label(step.optimization_accuracy)}, "
            f"frequencies={run_frequencies}, "
            f"max_opt_cycles={max_opt_cycles}, scfiterlimit={scfiterlimit}"
        )

        current_input = _qm_input_for_step(
            qm_input,
            step,
            molecule,
            frequencies=run_frequencies,
            max_opt_cycles=max_opt_cycles,
            scfiterlimit=scfiterlimit,
        )
        current_input = await _persist_turbomole_input(current_input, node_runner)

        step_kwargs = dict(kwargs)
        step_kwargs["custom_name"] = f"{basis_name}/{functional_name}"

        try:
            calc_result = await turbomole2(current_input, **step_kwargs)
        except Exception as exc:
            error = str(exc)
            qm_result = None
            calc_result = None
        else:
            qm_result, error = _child_qm_result(calc_result)

        if error:
            if tolerate_failure:
                node_runner.warning(f"{step_name} failed: {error}")
                break
            return node_runner.fail(f"{step_name} failed: {error}")

        last_result = qm_result
        next_mol = _molecule_from_qm_result(qm_result, molecule, node_runner, step_name)
        if next_mol is not molecule:
            molecule = await _persist_step_molecule(next_mol, node_runner, step_name)

        wall_s, cpu_s, n_iterations, freq_wall_s, freq_cpu_s = _timings_from_child_result(calc_result)
        if wall_s is not None:
            step_walls.append(wall_s)
        if cpu_s is not None:
            step_cpus.append(cpu_s)
        if n_iterations is not None:
            step_iters.append(n_iterations)
        if freq_wall_s is not None:
            step_freq_walls.append(freq_wall_s)
        if freq_cpu_s is not None:
            step_freq_cpus.append(freq_cpu_s)

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

    if last_result is None:
        return node_runner.fail("multistep_optimizer produced no QMResult")

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

    node_runner.result = last_result
    node_runner.qm_result = last_result
    node_runner.step_table = step_table
    return node_runner.succeed()
