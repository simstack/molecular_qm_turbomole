import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from odmantic import Field, Model, Reference
from pydantic import model_validator

from molecular_qm_models.density_functional import Functional
from molecular_qm_models.dispersion_correction import DispersionCorrection
from molecular_qm_models.molecule import Molecule
from molecular_qm_models.qm_result import QMResult
from molecular_qm_turbomole.lib.output_parser import parse_coord_file, write_final_geometry_xyz
from molecular_qm_turbomole.models.turbomole_input import (
    HyperpolarizabilityModeEnum,
    TurbomoleBasisSet2,
    TurbomoleQMInput2,
)
from molecular_qm_turbomole.nodes.turbomole2 import turbomole2
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import Parameters, simstack_model
from simstack.models.parameters import SlurmParameters
from simstack.models.simple_table import SimpleTable
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema

logger = logging.getLogger("HyperpolarizabilityWorkflow")

WORKFLOW_FINAL_STRUCTURE_XYZ = "final_structure.xyz"
FIRST_SIX_FREQUENCY_TOLERANCE = 1e-6
RETRY_GRIDSIZE = "m5"
HYPERPOL_STEP_MODES = (
    HyperpolarizabilityModeEnum.STATIC,
    HyperpolarizabilityModeEnum.DYNAMIC,
)


def _make_workflow_parameters() -> Parameters:
    slurm_fields = getattr(SlurmParameters, "model_fields", {})
    slurm_kwargs: dict[str, object] = {
        "nodes": 1,
        "tasks": 1,
        "tasks_per_node": 8,
        "cpus_per_task": 1,
        "mem": "2G",
        "time": "2:00:00",
        "job_name": "turbomole_workflow",
    }
    slurm_kwargs = {key: value for key, value in slurm_kwargs.items() if key in slurm_fields}
    slurm_parameters = SlurmParameters(**slurm_kwargs)
    params_kwargs: dict[str, object] = {
        "resource": "int-nano",
        "queue": "slurm-queue",
        "recompute_artifacts": True,
        "force_rerun": True,
    }
    params_fields = getattr(Parameters, "model_fields", {})
    if "slurm_parameters" in params_fields:
        params_kwargs["slurm_parameters"] = slurm_parameters
    return Parameters(**params_kwargs)


workflow_parameters = _make_workflow_parameters()


@simstack_model
class HyperpolarizabilitySettings(Model):
    """Mode and optical wavelength for the post-optimization escf hyperpolarizability step."""

    field_name: str = "HyperpolarizabilitySettings"
    hyperpolarizability: HyperpolarizabilityModeEnum = Field(
        HyperpolarizabilityModeEnum.STATIC,
        json_schema_extra={
            "enum": [item.value for item in HYPERPOL_STEP_MODES],
            "title": "Hyperpolarizability",
            "description": (
                "static: $scfinstab hyperpol with no frequency lines. "
                "dynamic: same keyword plus hyperpol_frequency_nm."
            ),
        },
    )
    hyperpol_frequency_nm: float = Field(
        0.0,
        json_schema_extra={
            "description": "Optical wavelength in nm for dynamic beta. Ignored for static.",
            "title": "Hyperpol Frequency Nm",
        },
    )
    frequency_tolerance: float = Field(
        1e-6,
        json_schema_extra={
            "description": (
                "Imaginary-frequency cutoff in cm-1. After optimization, modes 7+ must be "
                "at least -|frequency_tolerance|. The first six modes must be near zero. "
                "If the check fails, optimization is retried with grid m5."
            ),
            "title": "Frequency Tolerance",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict) and "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @model_validator(mode="after")
    def validate_settings(self):
        mode = self.hyperpolarizability
        if mode not in HYPERPOL_STEP_MODES:
            raise ValueError("hyperpolarizability must be static or dynamic.")
        if mode == HyperpolarizabilityModeEnum.DYNAMIC and float(self.hyperpol_frequency_nm or 0.0) <= 0.0:
            raise ValueError("Dynamic hyperpolarizability requires a positive hyperpol_frequency_nm.")
        return self

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        properties = schema.setdefault("properties", {})
        properties["hyperpolarizability"] = {
            "type": "string",
            "enum": [item.value for item in HYPERPOL_STEP_MODES],
            "default": HyperpolarizabilityModeEnum.STATIC.value,
            "title": "Hyperpolarizability",
            "description": (
                "static: $scfinstab hyperpol with no frequency lines. "
                "dynamic: same keyword plus hyperpol_frequency_nm."
            ),
        }
        frequency_schema = properties.get("hyperpol_frequency_nm")
        if isinstance(frequency_schema, dict) and "anyOf" in frequency_schema:
            frequency_schema["type"] = "number"
            frequency_schema.pop("anyOf", None)
        tolerance_schema = properties.get("frequency_tolerance")
        if isinstance(tolerance_schema, dict) and "anyOf" in tolerance_schema:
            tolerance_schema["type"] = "number"
            tolerance_schema.pop("anyOf", None)
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["ui:order"] = [
            "hyperpolarizability",
            "hyperpol_frequency_nm",
            "frequency_tolerance",
            "id",
        ]
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        ui_schema.setdefault("hyperpolarizability", {})["ui:widget"] = "select"
        ui_schema.setdefault("hyperpol_frequency_nm", {})["ui:condition"] = {
            "hyperpolarizability": HyperpolarizabilityModeEnum.DYNAMIC.value
        }
        return ui_schema


def _copy_qm_input(qm_input: TurbomoleQMInput2) -> TurbomoleQMInput2:
    payload = qm_input.model_dump(exclude={"id", "molecule"})
    payload["molecule"] = qm_input.molecule
    return TurbomoleQMInput2.model_validate(payload)


def _disable_hyperpolarizability(qm_input: TurbomoleQMInput2) -> None:
    qm_input.hyperpolarizability = HyperpolarizabilityModeEnum.NONE
    qm_input.hyperpol_frequency_nm = 0.0


def _apply_hyperpolarizability_settings(
    qm_input: TurbomoleQMInput2,
    settings: HyperpolarizabilitySettings,
) -> None:
    qm_input.hyperpolarizability = settings.hyperpolarizability
    if settings.hyperpolarizability == HyperpolarizabilityModeEnum.DYNAMIC:
        qm_input.hyperpol_frequency_nm = float(settings.hyperpol_frequency_nm or 0.0)
    else:
        qm_input.hyperpol_frequency_nm = 0.0


def _build_optimization_input(base_input: TurbomoleQMInput2) -> TurbomoleQMInput2:
    opt_input = _copy_qm_input(base_input)
    opt_input.name = f"{base_input.name}_opt"
    opt_input.optimization = True
    opt_input.states = 0
    opt_input.frequencies = True
    _disable_hyperpolarizability(opt_input)
    return opt_input


def _build_hyperpol_input(
    base_input: TurbomoleQMInput2,
    optimized_structure: Molecule,
    settings: HyperpolarizabilitySettings,
) -> TurbomoleQMInput2:
    hyper_input = _copy_qm_input(base_input)
    hyper_input.name = f"{base_input.name}_hyperpol"
    hyper_input.molecule = optimized_structure
    hyper_input.optimization = False
    hyper_input.states = 0
    hyper_input.gradients = False
    hyper_input.frequencies = False
    _apply_hyperpolarizability_settings(hyper_input, settings)
    return hyper_input


def _optimization_grid_sizes(original_grid_size: str) -> list[str]:
    grids = [original_grid_size]
    if original_grid_size != RETRY_GRIDSIZE:
        grids.append(RETRY_GRIDSIZE)
    return grids


def _frequency_value(row: dict[str, Any]) -> float:
    for key in ("frequency_cm_1", "Frequency (cm-1)"):
        if row.get(key) is not None:
            return float(row[key])
    for key, value in row.items():
        if "frequency" in str(key).lower() and value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _check_vibrational_frequencies(
    result: Any,
    frequency_tolerance: float,
) -> tuple[bool, dict[str, str]]:
    table = None
    for candidate in _iter_result_candidates(result):
        candidate_table = getattr(candidate, "vibrational_frequencies", None)
        if candidate_table is not None and getattr(candidate_table, "row", None):
            table = candidate_table
            break
    if table is None or not table.row:
        return False, {"missing": "No vibrational frequencies found"}
    if len(table.row) < 6:
        return False, {
            "count": f"Only {len(table.row)} vibrational frequencies found; expected at least 6."
        }

    threshold = -abs(float(frequency_tolerance))
    failed: dict[str, str] = {}
    frequencies_ok = True
    for index, row in enumerate(table.row):
        freq_val = _frequency_value(row)
        if index < 6:
            if abs(freq_val) > FIRST_SIX_FREQUENCY_TOLERANCE:
                failed[str(index + 1)] = f"{freq_val:.2f} cm-1 (expected zero)"
                frequencies_ok = False
        elif freq_val < threshold:
            failed[str(index + 1)] = f"{freq_val:.2f} cm-1"
            frequencies_ok = False
    return frequencies_ok, failed


def _child_kwargs(parent_kwargs: dict) -> dict:
    child = dict(parent_kwargs)
    child.pop("node_runner", None)
    child["parameters"] = Parameters(
        resource="self",
        queue="default",
        recompute_artifacts=True,
        force_rerun=True,
    )
    return child


async def _run_turbomole_inline(qm_input: TurbomoleQMInput2, child_kwargs: dict) -> Any:
    return await turbomole2(qm_input, **child_kwargs)


def _iter_result_candidates(result: Any) -> list[Any]:
    candidates = [result]
    inner = getattr(result, "result", None)
    if inner is not None:
        candidates.append(inner)
    return candidates


def _extract_qm_result(result: Any) -> Optional[QMResult]:
    for candidate in _iter_result_candidates(result):
        if isinstance(candidate, QMResult):
            return candidate
        if getattr(candidate, "field_name", None) == "QMResult":
            return candidate
    return None


def _extract_hyperpolarizability_table(result: Any) -> Optional[SimpleTable]:
    for candidate in _iter_result_candidates(result):
        table = getattr(candidate, "hyperpolarizability", None)
        if isinstance(table, SimpleTable):
            return table
    return None


def beta_zzz_by_pair(table: Optional[SimpleTable]) -> dict[int, float]:
    values: dict[int, float] = {}
    if table is None:
        return values
    for row in table.row or []:
        pair = row.get("pair")
        beta = row.get("beta_zzz_1e30_esu")
        if pair is None or beta is None:
            continue
        try:
            values[int(pair)] = float(beta)
        except (TypeError, ValueError):
            continue
    return values


def _has_hyperpol_output(result: Any) -> bool:
    table = _extract_hyperpolarizability_table(result)
    if table is not None and table.row:
        return True
    for candidate in _iter_result_candidates(result):
        files = getattr(candidate, "files", None)
        if files is not None and hasattr(files, "find"):
            if files.find("hyperpols") is not None:
                return True
    return False


def _is_completed(result) -> bool:
    status = getattr(result, "task_status", None)
    if isinstance(status, TaskStatus):
        return status == TaskStatus.COMPLETED
    status_s = str(getattr(result, "status", status) or "").strip().lower()
    return status_s in {"completed", "taskstatus.completed"} or status_s.endswith(".completed")


def _result_debug(result) -> str:
    primary = _extract_qm_result(result) or result
    status = getattr(primary, "task_status", None) or getattr(primary, "status", None)
    error = getattr(primary, "error", None)
    return f"task_status={status}, error={error}"


def _to_odmantic_molecule(molecule: Any) -> Molecule:
    if isinstance(molecule, Molecule):
        return molecule
    if molecule is None:
        raise ValueError("Molecule is None.")
    if hasattr(molecule, "model_dump"):
        return Molecule(**molecule.model_dump(exclude={"id"}))
    raise TypeError(f"Unsupported molecule type: {type(molecule)}")


async def _ensure_db_molecule(molecule: Any) -> Molecule:
    normalized = _to_odmantic_molecule(molecule)
    return await context.db.save(normalized)


def _extract_structure_from_result(result: Any, node_runner: NodeRunner) -> Optional[Molecule]:
    for local_name in (WORKFLOW_FINAL_STRUCTURE_XYZ, "final_geometry.xyz"):
        path = Path(local_name)
        if path.is_file():
            try:
                parsed = Molecule.from_xyz(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                parsed = None
            if parsed is not None:
                node_runner.info(f"Recovered optimized structure from '{local_name}'.")
                return parsed

    for candidate in _iter_result_candidates(result):
        direct = getattr(candidate, "final_structure", None)
        if direct is not None:
            try:
                return _to_odmantic_molecule(direct)
            except Exception:
                pass
        structures = getattr(candidate, "structures", None)
        molecules = getattr(structures, "molecules", None) if structures is not None else None
        if molecules:
            try:
                return _to_odmantic_molecule(molecules[-1])
            except Exception:
                pass

    parsed = parse_coord_file(Path("coord"))
    if parsed is not None:
        node_runner.info("Recovered optimized structure from local 'coord'.")
        return parsed
    return None


@simstack_model
class HyperPolarizationRecord(Model):
    field_name: str = "HyperPolarizationRecord"
    molecule: Molecule = Reference()
    functional: Functional = Field(default_factory=Functional)
    dispersion_correction: Optional[DispersionCorrection] = Field(default=None)
    basis_set: TurbomoleBasisSet2 = Field(default_factory=TurbomoleBasisSet2)
    grids_used: List[str] = Field(default_factory=list)
    started_at: datetime
    hyperpol: Optional[SimpleTable] = None
    success: bool = False
    error: Optional[str] = None


@node(parameters=workflow_parameters)
async def hyperpolarizibility(
    optimization_qm_input: TurbomoleQMInput2,
    hyperpolarizability_settings: HyperpolarizabilitySettings,
    **kwargs,
) -> SimstackResult:
    """
    Optimize a molecule with turbomole2, then compute first hyperpolarizability on
    the optimized geometry.

    Parameters:
        optimization_qm_input (TurbomoleQMInput2): Ground-state optimization settings and molecule.
        hyperpolarizability_settings (HyperpolarizabilitySettings): Static or dynamic beta, plus frequency_tolerance.

    Called Nodes:
        turbomole2

    SimstackResult:
        result (QMResult): Hyperpolarizability turbomole2 result.
        hyperpolarizability (SimpleTable): Frequency-pair beta_zzz in 10^-30 esu.
        record (HyperPolarizationRecord): Persisted workflow summary.
    """
    node_runner: NodeRunner = kwargs["node_runner"]
    child_kwargs = _child_kwargs(kwargs)

    try:
        if optimization_qm_input.states > 0:
            raise ValueError("Workflow expects ground-state path for optimization. Please set states=0.")
        molecule = optimization_qm_input.molecule
        if hasattr(molecule, "make_smiles"):
            molecule.smiles = molecule.make_smiles()
        if hasattr(molecule, "make_formula"):
            molecule.formula = molecule.make_formula()
        node_runner.info(f"Running workflow '{optimization_qm_input.name}'.")

        hyperpolarization_record = HyperPolarizationRecord(
            molecule=await _ensure_db_molecule(molecule),
            functional=optimization_qm_input.functional,
            dispersion_correction=optimization_qm_input.functional.dispersion_correction,
            basis_set=optimization_qm_input.basis_set,
            started_at=datetime.now(timezone.utc),
        )
        node_runner.record = hyperpolarization_record

        optimization_qm_input = _copy_qm_input(optimization_qm_input)
        optimization_qm_input.molecule = hyperpolarization_record.molecule

        frequencies_ok = False
        failed_frequencies: dict[str, str] = {}
        optimization_call_result = None
        optimization_result = None
        grid_sizes = _optimization_grid_sizes(optimization_qm_input.gridsize)
        for grid_size in grid_sizes:
            optimization_qm_input.gridsize = grid_size
            optimization_input = _build_optimization_input(optimization_qm_input)
            hyperpolarization_record.grids_used.append(grid_size)
            node_runner.info(f"Running turbomole2 optimization with grid size {grid_size}.")
            optimization_call_result = await _run_turbomole_inline(optimization_input, child_kwargs)
            optimization_result = _extract_qm_result(optimization_call_result) or optimization_call_result
            if not (_is_completed(optimization_call_result) or _is_completed(optimization_result)):
                hyperpolarization_record.success = False
                hyperpolarization_record.error = "OPT"
                return node_runner.fail(
                    f"Optimization step did not complete successfully ({_result_debug(optimization_result)})."
                )
            if getattr(optimization_result, "scf_converged", None) is False:
                hyperpolarization_record.success = False
                hyperpolarization_record.error = "SCF"
                return node_runner.fail("SCF did not converge")

            frequencies_ok, failed_frequencies = _check_vibrational_frequencies(
                optimization_call_result,
                hyperpolarizability_settings.frequency_tolerance,
            )
            if not frequencies_ok and "missing" in failed_frequencies:
                frequencies_ok, failed_frequencies = _check_vibrational_frequencies(
                    optimization_result,
                    hyperpolarizability_settings.frequency_tolerance,
                )
            if "missing" in failed_frequencies:
                hyperpolarization_record.success = False
                hyperpolarization_record.error = "NOFREQ"
                return node_runner.fail("No vibrational frequencies found in optimization result.")
            if frequencies_ok:
                node_runner.info(f"Vibrational frequencies within threshold for grid {grid_size}.")
                break
            if grid_size != grid_sizes[-1]:
                node_runner.warning(
                    "Vibrational frequencies exceed threshold. retrying with gridsize m5"
                )

        if not frequencies_ok:
            hyperpolarization_record.success = False
            error_details = ", ".join(f"{index}: {value}" for index, value in failed_frequencies.items())
            hyperpolarization_record.error = f"BADFREQ: {error_details}"
            return node_runner.fail(f"Vibrational frequencies exceed threshold: {error_details}")

        optimized_structure = _extract_structure_from_result(optimization_call_result, node_runner)
        if optimized_structure is None:
            hyperpolarization_record.success = False
            hyperpolarization_record.error = "NOSTRUCT"
            return node_runner.fail(
                "Optimization step produced no usable structure for the hyperpolarizability step "
                f"({_result_debug(optimization_result)})."
            )
        optimized_structure = await _ensure_db_molecule(optimized_structure)
        write_final_geometry_xyz(optimized_structure, WORKFLOW_FINAL_STRUCTURE_XYZ)

        node_runner.info("Running turbomole2 hyperpolarizability step.")
        hyperpol_input = _build_hyperpol_input(
            optimization_qm_input,
            optimized_structure,
            hyperpolarizability_settings,
        )
        node_runner.info(
            "Prepared hyperpolarizability settings: "
            f"mode={hyperpol_input.hyperpolarizability.value}, "
            f"lambda_nm={float(hyperpol_input.hyperpol_frequency_nm or 0.0):.10g}"
        )
        hyperpol_call_result = await _run_turbomole_inline(hyperpol_input, child_kwargs)
        hyperpol_result = _extract_qm_result(hyperpol_call_result) or hyperpol_call_result
        hyperpol_table = _extract_hyperpolarizability_table(hyperpol_call_result)
        hyperpolarization_record.hyperpol = hyperpol_table
        if hyperpol_table is not None:
            node_runner.hyperpolarizability = hyperpol_table

        if not (_is_completed(hyperpol_call_result) or _is_completed(hyperpol_result)):
            hyperpolarization_record.success = False
            hyperpolarization_record.error = "HYPERPOL"
            return node_runner.fail(
                f"Hyperpolarizability step did not complete successfully ({_result_debug(hyperpol_result)})."
            )
        if not _has_hyperpol_output(hyperpol_call_result) and not _has_hyperpol_output(hyperpol_result):
            hyperpolarization_record.success = False
            hyperpolarization_record.error = "NOHYPERPOL"
            return node_runner.fail("Hyperpolarizability step completed but no beta output was detected.")

        hyperpolarization_record.success = True
        node_runner.result = hyperpol_result
        node_runner.info("Workflow completed successfully.")
        return node_runner.succeed()
    except Exception as exc:
        return node_runner.fail(f"Turbomole workflow failed: {exc}")
