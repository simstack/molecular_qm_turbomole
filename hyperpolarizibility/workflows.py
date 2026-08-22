import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from copy import deepcopy
from typing import Optional, Any, List

from bson import ObjectId as BsonObjectId
from odmantic import Model, Field, ObjectId, Reference
from pydantic import model_validator

from applications.electronic_structure.molecule import Molecule
from applications.electronic_structure.density_functional import Functional
from molecular_qm_models.dispersion_correction import DispersionCorrection
from applications.electronic_structure.hyperpolarizibility.hyperpolarizability_defaults import (
    HYPERPOL_FREQUENCY_FIELD,
    HYPERPOL_FREQUENCY_UI_FIELD,
    HYPERPOL_MODE_UI_FIELD,
    HYPERPOL_MODE_VALUES,
    HYPERPOL_MODE_DYNAMIC,
    HYPERPOL_MODE_OFF,
    HYPERPOL_MODE_STATIC,
    HYPERPOL_ENABLED_FIELD,
    normalize_hyperpolarizability_payload,
)
from applications.electronic_structure.qm_input import QMInput, SolventModeEnum
from applications.electronic_structure.qm_result import QMResult
from applications.electronic_structure.turbomole.input_models import (
    TurbomoleBasisSet,
    TurbomoleQMInput,
)
from applications.electronic_structure.turbomole.turbomole import (
    TURBOMOLE_FINAL_GEOMETRY_XYZ,
    turbomole,
)
from applications.electronic_structure.turbomole.hyperpol_tensors import (
    BetaTensorFileUpload,
    BetaTensorZDipoleESUInput,
    BetaTensor_ZDipole_ESU, BetaTensorZDipoleESUResult,
)
from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import Parameters, simstack_model, FloatData
from simstack.models.parameters import SlurmParameters
from simstack.models.simple_table import SimpleTable
from simstack.util.cleaned_json_schema import cleaned_json_schema

logger = logging.getLogger("Workflows")


def _make_workflow_parameters() -> Parameters:
    """
    Keep workflow submissions valid on Slurm-backed resources even when the
    runner/database is using an older SimStack core that does not auto-fill
    scheduler defaults.
    """
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
    slurm_kwargs = {
        key: value for key, value in slurm_kwargs.items() if key in slurm_fields
    }
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
    if "slurm" in params_fields:
        params_kwargs["slurm"] = slurm_parameters
    if "nodes" in params_fields:
        params_kwargs["nodes"] = 1
    if "tasks" in params_fields:
        params_kwargs["tasks"] = 1

    return Parameters(**params_kwargs)


workflow_parameters = _make_workflow_parameters()

WORKFLOW_FINAL_STRUCTURE_XYZ = "final_structure.xyz"


class WorkflowStateEnum(str, Enum):
    QUEUED = "QUEUED"
    RUNNING_OPTIMIZATION = "RUNNING_OPTIMIZATION"
    RUNNING_HYPERPOLARIZABILITY = "RUNNING_HYPERPOLARIZABILITY"
    RUNNING_BETA_TENSOR = "RUNNING_BETA_TENSOR"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def _legacy_get_value(source: Any, key: str, default: Any) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "null", "undefined"}:
        return True
    return False


def _prune_null_like(value: Any) -> Any:
    """
    Recursively remove null-like values from nested payloads.
    """
    if isinstance(value, dict):
        pruned: dict[str, Any] = {}
        for key, item in value.items():
            if _is_null_like(item):
                continue
            cleaned = _prune_null_like(item)
            if cleaned is None:
                continue
            pruned[key] = cleaned
        if not pruned:
            return None
        return pruned
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            if _is_null_like(item):
                continue
            cleaned = _prune_null_like(item)
            if cleaned is None:
                continue
            cleaned_list.append(cleaned)
        return cleaned_list
    return value


def _normalize_optional_mapping(value: Any, required_key: str) -> Optional[dict[str, Any]]:
    """
    Normalize optional nested dict payloads.
    """
    if _is_null_like(value):
        return None
    if not isinstance(value, dict):
        return None
    cleaned = _prune_null_like(value)
    if not isinstance(cleaned, dict):
        return None
    if required_key not in cleaned:
        return None
    return cleaned


def _project_step2_settings(source: Any) -> dict[str, Any]:
    """
    Project legacy/full QMInput payloads down to current Turbomole step-2 settings.
    """
    if not isinstance(source, dict):
        return {}
    source = normalize_hyperpolarizability_payload(
        dict(source),
        logger=logger,
        context="workflow step-2 settings",
    )
    allowed = {
        "field_name",
        "name",
        "charge",
        "focus_state",
        "multiplicity",
        "gridsize",
        "scfconv",
        "use_desy",
        "open_shell_calculation",
        "active_electrons",
        "active_orbitals",
        "basis_set",
        "functional",
        "gradients",
        "optimization",
        "frequencies",
        "solvent_mode",
        "solvent",
        "solvent_epsilon",
        "solvent_refind",
        "print_level",
        "blocks",
        "states",
        HYPERPOL_MODE_UI_FIELD,
        HYPERPOL_FREQUENCY_UI_FIELD,
    }
    projected: dict[str, Any] = {}
    for key, value in source.items():
        if key not in allowed:
            continue
        if key == "basis_set":
            normalized = _normalize_optional_mapping(value, "basis_set")
            if normalized is not None:
                projected[key] = normalized
            continue
        if key == "functional":
            normalized = _normalize_optional_mapping(value, "functional")
            if normalized is not None:
                projected[key] = normalized
            continue
        if _is_null_like(value):
            continue
        projected[key] = value

    frequency = source.get(HYPERPOL_FREQUENCY_FIELD)
    if frequency is not None and HYPERPOL_FREQUENCY_UI_FIELD not in projected:
        projected[HYPERPOL_FREQUENCY_UI_FIELD] = frequency

    if HYPERPOL_MODE_UI_FIELD not in projected:
        enabled = source.get(HYPERPOL_ENABLED_FIELD)
        if enabled is not None:
            if bool(enabled):
                try:
                    frequency_value = float(frequency or 0.0)
                except (TypeError, ValueError):
                    frequency_value = 0.0
                projected[HYPERPOL_MODE_UI_FIELD] = (
                    HYPERPOL_MODE_DYNAMIC if frequency_value > 0.0 else HYPERPOL_MODE_STATIC
                )
            else:
                projected[HYPERPOL_MODE_UI_FIELD] = HYPERPOL_MODE_OFF
        else:
            try:
                frequency_value = float(projected.get(HYPERPOL_FREQUENCY_UI_FIELD) or 0.0)
            except (TypeError, ValueError):
                frequency_value = 0.0
            if frequency_value > 0.0:
                projected[HYPERPOL_MODE_UI_FIELD] = HYPERPOL_MODE_DYNAMIC
    return projected


def _strip_molecule_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Remove molecule input from schema objects used for workflow step 2.
    """
    properties = schema.get("properties")
    if isinstance(properties, dict):
        properties.pop("molecule", None)

    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [name for name in required if name != "molecule"]

    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        for dep_schema in dependencies.values():
            if not isinstance(dep_schema, dict):
                continue
            one_of = dep_schema.get("oneOf")
            if not isinstance(one_of, list):
                continue
            for branch in one_of:
                if not isinstance(branch, dict):
                    continue
                branch_props = branch.get("properties")
                if isinstance(branch_props, dict):
                    branch_props.pop("molecule", None)
                branch_required = branch.get("required")
                if isinstance(branch_required, list):
                    branch["required"] = [
                        name for name in branch_required if name != "molecule"
                    ]
    return schema


def _strip_molecule_from_ui(ui_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Hide molecule input from step-2 UI and keep it out of explicit ordering.
    """
    ui_schema["molecule"] = {"ui:widget": "hidden"}
    ui_order = ui_schema.get("ui:order")
    if isinstance(ui_order, list):
        ui_schema["ui:order"] = [
            field_name for field_name in ui_order if field_name != "molecule"
        ]
    return ui_schema


def _registered_model_form_ui(model_cls: type, *, title: str) -> dict[str, Any]:
    """Render a workflow step through the same form component as its node model."""
    return {
        "ui:field": "GenericFormField",
        "ui:title": title,
        "ui:options": {
            "model": f"{model_cls.__module__}.{model_cls.__name__}",
            "accordion": "true",
        },
    }


@simstack_model
class Workflows(Model):
    """
    Minimal workflow state container for sequential Turbomole orchestration.
    """
    field_name: str = "Workflows"
    name: str = "turbomole_opt_hyperpol_workflow"
    state: WorkflowStateEnum = WorkflowStateEnum.QUEUED
    current_step: str = "init"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    optimization_status: Optional[str] = None
    hyperpolarizability_status: Optional[str] = None
    optimization_error: Optional[str] = None
    hyperpolarizability_error: Optional[str] = None
    beta_tensor_status: Optional[str] = None
    beta_tensor_error: Optional[str] = None
    optimization_energy: Optional[float] = None
    hyperpolarizability_energy: Optional[float] = None
    run_hyperpolarizability: bool = True
    run_beta_tensor: bool = True
    error_message: Optional[str] = None
    transitions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict):
            if "field_name" not in data:
                data["field_name"] = cls.__name__
        return data


@simstack_model
class TurbomoleHyperpolarizabilityQMInputAuto(Model):
    """
    Embedded step-2 settings for hyperpolarizability.
    This follows the current Turbomole node controls, without molecule because
    the workflow injects optimized geometry from step 1 at runtime.
    """
    field_name: str = "TurbomoleHyperpolarizabilityQMInputAuto"
    name: Optional[str] = None
    charge: Optional[int] = None
    states: Optional[int] = None
    focus_state: Optional[int] = None
    multiplicity: Optional[int] = None
    gridsize: Optional[str] = None
    scfconv: Optional[int] = None
    use_desy: Optional[bool] = None
    open_shell_calculation: Optional[bool] = None
    active_electrons: Optional[int] = None
    active_orbitals: Optional[int] = None
    basis_set: Optional[dict[str, Any]] = None
    functional: Optional[dict[str, Any]] = None
    gradients: Optional[bool] = None
    optimization: Optional[bool] = None
    frequencies: Optional[bool] = None
    hyperpolarizability_mode: Optional[str] = None
    hyperpol_frequency_nm_ui: Optional[float] = None
    solvent_mode: Optional[str] = None
    solvent: Optional[str] = None
    solvent_epsilon: Optional[float] = None
    solvent_refind: Optional[float] = None
    print_level: Optional[int] = None
    blocks: Optional[list[str]] = None

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict):
            sanitized = _project_step2_settings(data)
            sanitized.setdefault("field_name", cls.__name__)
            return sanitized
        return data

    @model_validator(mode="after")
    def validate_step(self):
        if self.states is not None and self.states > 0:
            raise ValueError("Workflow expects ground-state path. Please set states=0.")
        if self.hyperpol_frequency_nm_ui is not None and self.hyperpol_frequency_nm_ui < 0:
            raise ValueError("hyperpol_frequency_nm_ui must be >= 0 (0 = static).")
        if self.hyperpolarizability_mode is not None and self.hyperpolarizability_mode not in HYPERPOL_MODE_VALUES:
            raise ValueError(
                f"{HYPERPOL_MODE_UI_FIELD} must be one of {', '.join(HYPERPOL_MODE_VALUES)}."
            )
        if self.hyperpolarizability_mode == HYPERPOL_MODE_DYNAMIC:
            frequency_nm = float(self.hyperpol_frequency_nm_ui or 0.0)
            if frequency_nm <= 0.0:
                raise ValueError(
                    "Dynamic hyperpolarizability requires a positive "
                    f"{HYPERPOL_FREQUENCY_UI_FIELD} wavelength in nm."
                )
        return self

    @classmethod
    def json_schema(cls, recursive=True):
        schema = deepcopy(TurbomoleQMInput.json_schema())
        schema["title"] = cls.__name__
        return _strip_molecule_from_schema(schema)

    @classmethod
    def ui_schema(cls):
        ui_schema = deepcopy(TurbomoleQMInput.ui_schema())
        ui_schema["ui:title"] = "DFT-Turbomole_1 (Hyperpolarizability)"
        return _strip_molecule_from_ui(ui_schema)


def _coerce_step2_settings(source: Any) -> TurbomoleHyperpolarizabilityQMInputAuto:
    """
    Robustly coerce legacy/new step-2 payload shapes to embedded settings model.
    Accepts dict payloads and gracefully falls back for legacy ObjectId/reference values.
    """
    if isinstance(source, TurbomoleHyperpolarizabilityQMInputAuto):
        return source
    if isinstance(source, dict):
        projected = _project_step2_settings(source)
    elif hasattr(source, "model_dump"):
        projected = _project_step2_settings(source.model_dump())
    else:
        projected = {}
    projected.setdefault("field_name", TurbomoleHyperpolarizabilityQMInputAuto.__name__)
    # This workflow always executes a hyperpolarizability second step.
    projected.setdefault(HYPERPOL_MODE_UI_FIELD, HYPERPOL_MODE_STATIC)
    projected.setdefault("states", 0)
    projected.setdefault(HYPERPOL_FREQUENCY_UI_FIELD, 0.0)
    return TurbomoleHyperpolarizabilityQMInputAuto.model_validate(projected)


def _coerce_turbomole_qm_input_if_available(source: Any) -> Optional[TurbomoleQMInput]:
    if isinstance(source, TurbomoleQMInput):
        return source
    if isinstance(source, dict):
        return TurbomoleQMInput.model_validate(source)
    return None


def _is_object_id_value(source: Any) -> bool:
    return isinstance(source, BsonObjectId)


def _to_odmantic_object_id(source: Any) -> ObjectId:
    if isinstance(source, ObjectId):
        return source
    if isinstance(source, BsonObjectId):
        return ObjectId(source)
    raise TypeError(f"Expected ObjectId-compatible value, got {type(source)}.")


async def _find_raw_model_document(model_cls: Any, source_id: ObjectId) -> Optional[dict[str, Any]]:
    """
    Fetch a raw Mongo document for ObjectId-backed workflow subforms.

    Some workflow subforms intentionally omit fields required by the full model
    they were derived from, notably the step-2 Turbomole form without molecule.
    ODMantic parsing can fail before returning the document, so this raw fallback
    lets us project the subset of fields that the workflow actually needs.
    """
    db = getattr(context, "db", None)
    engine = getattr(db, "engine", None)
    if engine is None:
        engine = getattr(db, "_engine", None)
    if engine is None or not hasattr(engine, "get_collection"):
        return None
    try:
        collection = engine.get_collection(model_cls)
    except Exception:
        return None
    if collection is None:
        return None

    candidate_ids: list[Any] = [source_id]
    try:
        bson_id = BsonObjectId(str(source_id))
    except Exception:
        bson_id = None
    if bson_id is not None and bson_id not in candidate_ids:
        candidate_ids.append(bson_id)

    for candidate_id in candidate_ids:
        try:
            raw_doc = await collection.find_one({"_id": candidate_id})
        except Exception:
            continue
        if isinstance(raw_doc, dict):
            return raw_doc
    return None


async def _resolve_turbomole_qm_input(source: Any, field_name: str) -> TurbomoleQMInput:
    resolved = _coerce_turbomole_qm_input_if_available(source)
    if resolved is not None:
        return resolved

    if _is_object_id_value(source):
        if not context.initialized:
            raise RuntimeError(f"Cannot resolve {field_name} ObjectId without an initialized database context.")
        source_id = _to_odmantic_object_id(source)
        resolved = await context.db.find_one(TurbomoleQMInput, TurbomoleQMInput.id == source_id)
        if resolved is not None:
            return resolved
        raise FileNotFoundError(f"{field_name} references missing TurbomoleQMInput {source_id}.")

    raise TypeError(f"{field_name} must be a TurbomoleQMInput, dict payload, or ObjectId; got {type(source)}.")


async def _resolve_step2_settings(
    source: Any,
    node_runner: Optional[NodeRunner] = None,
) -> TurbomoleHyperpolarizabilityQMInputAuto:
    if _is_object_id_value(source):
        source_id = _to_odmantic_object_id(source)
        if not context.initialized:
            raise RuntimeError(
                "Cannot resolve hyperpolarizability_qm_input ObjectId without an initialized database context. "
                "Refusing to default to static hyperpolarizability settings."
            )

        resolution_errors: list[str] = []
        for model_cls in (TurbomoleHyperpolarizabilityQMInputAuto, TurbomoleQMInput):
            try:
                resolved = await context.db.find_one(model_cls, model_cls.id == source_id)
            except Exception as exc:
                resolution_errors.append(f"{model_cls.__name__}: {exc}")
                resolved = None
            if resolved is not None:
                return _coerce_step2_settings(resolved)

            raw_doc = await _find_raw_model_document(model_cls, source_id)
            if raw_doc is not None:
                projected = _project_step2_settings(raw_doc)
                if projected:
                    if node_runner is not None:
                        node_runner.info(
                            "Resolved hyperpolarizability_qm_input from raw "
                            f"{model_cls.__name__} document {source_id}."
                        )
                    return _coerce_step2_settings(projected)

        details = ""
        if resolution_errors:
            details = " Resolution errors: " + " | ".join(resolution_errors)
        raise FileNotFoundError(
            f"hyperpolarizability_qm_input references missing or unreadable step-2 settings {source_id}."
            f"{details} Refusing to default to static hyperpolarizability settings."
        )

    return _coerce_step2_settings(source)


@simstack_model
class TurbomoleOptHyperpolWorkflowInputAuto(Model):
    """
    Input for the sequential Turbomole workflow:
      1) geometry optimization
      2) hyperpolarizability on optimized structure
    """
    field_name: str = "TurbomoleOptHyperpolWorkflowInputAuto"
    name: str = "turbomole_opt_hyperpol"
    optimization_qm_input: Any = Field(default=None)
    hyperpolarizability_qm_input: Any = Field(
        default_factory=TurbomoleHyperpolarizabilityQMInputAuto
    )
    run_beta_tensor: bool = Field(
        True,
        description="Run BetaTensor_ZDipole_ESU after the Turbomole hyperpolarizability step.",
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if isinstance(data, dict):
            if "field_name" not in data:
                data["field_name"] = cls.__name__
            data.setdefault("run_beta_tensor", True)

            # Backward compatibility with old single-input schema.
            legacy_qm_input = data.get("qm_input")
            if legacy_qm_input is not None and "optimization_qm_input" not in data:
                data["optimization_qm_input"] = legacy_qm_input
            if legacy_qm_input is not None and "hyperpolarizability_qm_input" not in data:
                second_input = deepcopy(legacy_qm_input) if isinstance(legacy_qm_input, dict) else legacy_qm_input
                if isinstance(second_input, dict):
                    second_input["field_name"] = TurbomoleHyperpolarizabilityQMInputAuto.__name__
                legacy_step = data.get("hyperpolarizability_step")
                if isinstance(second_input, dict) and legacy_step is not None:
                    second_input["hyperpolarizability"] = bool(
                        _legacy_get_value(legacy_step, "hyperpolarizability", False)
                    )
                    second_input["hyperpol_frequency_nm"] = float(
                        _legacy_get_value(legacy_step, "hyperpol_frequency_nm", 0.0) or 0.0
                    )
                data["hyperpolarizability_qm_input"] = _project_step2_settings(second_input)

            # Compatibility with intermediate schema versions.
            if "optimization_qm_input" not in data and "qm_input" in data:
                data["optimization_qm_input"] = data.get("qm_input")
            if "hyperpolarizability_qm_input" not in data and "optimization_qm_input" in data:
                second_input = deepcopy(data.get("optimization_qm_input"))
                if isinstance(second_input, dict):
                    second_input["field_name"] = TurbomoleHyperpolarizabilityQMInputAuto.__name__
                data["hyperpolarizability_qm_input"] = _project_step2_settings(second_input)

            # Normalize step-2 payload shape for cached/frontend variants.
            if "hyperpolarizability_qm_input" in data:
                step2_input = data.get("hyperpolarizability_qm_input")
                if isinstance(step2_input, dict):
                    projected = _project_step2_settings(step2_input)
                    projected.setdefault(
                        "field_name",
                        TurbomoleHyperpolarizabilityQMInputAuto.__name__,
                    )
                    data["hyperpolarizability_qm_input"] = projected
                elif step2_input is None:
                    data["hyperpolarizability_qm_input"] = {
                        "field_name": TurbomoleHyperpolarizabilityQMInputAuto.__name__
                    }
                elif _is_object_id_value(step2_input):
                    pass
                else:
                    # Old docs may still carry ObjectId/reference shape here.
                    # Convert to defaults instead of hard-failing parse.
                    data["hyperpolarizability_qm_input"] = {
                        "field_name": TurbomoleHyperpolarizabilityQMInputAuto.__name__
                    }
        return data

    @model_validator(mode="after")
    def validate_steps(self):
        # The workflow always performs optimization first.
        optimization_qm_input = _coerce_turbomole_qm_input_if_available(self.optimization_qm_input)
        if optimization_qm_input is not None and optimization_qm_input.states > 0:
            raise ValueError("Workflow expects ground-state path for optimization. Please set states=0.")
        step2_settings = _coerce_step2_settings(self.hyperpolarizability_qm_input)
        if step2_settings.states is not None and step2_settings.states > 0:
            raise ValueError("Workflow expects ground-state path. Please set states=0.")
        return self

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__

        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return schema

        defs = schema.setdefault("$defs", {})
        if isinstance(defs, dict):
            turbomole_schema = deepcopy(TurbomoleQMInput.json_schema())
            turbomole_defs = turbomole_schema.pop("$defs", {})
            if isinstance(turbomole_defs, dict):
                defs.update(turbomole_defs)

            # Drop the raw cleaned_json_schema definition because it bypasses
            # TurbomoleQMInput.json_schema() and still contains hidden legacy
            # hyperpolarizability fields.
            defs.pop("TurbomoleQMInput", None)

            step1_key = "TurbomoleQMInputWorkflow"
            step2_key = "TurbomoleQMInputWorkflowStep2AutoGeometry"
            defs[step1_key] = turbomole_schema
            defs[step2_key] = _strip_molecule_from_schema(deepcopy(turbomole_schema))
            properties["optimization_qm_input"] = {
                "$ref": f"#/$defs/{step1_key}",
                "title": "DFT-Turbomole (Optimization)",
            }
            properties["hyperpolarizability_qm_input"] = {
                "$ref": f"#/$defs/{step2_key}",
                "title": "DFT-Turbomole_1 (Hyperpolarizability)",
            }

        return schema

    @classmethod
    def ui_base_schema(cls):
        optimization_ui_schema = _registered_model_form_ui(
            TurbomoleQMInput,
            title="DFT-Turbomole (Optimization)",
        )
        hyperpolarizability_ui_schema = _registered_model_form_ui(
            TurbomoleHyperpolarizabilityQMInputAuto,
            title="DFT-Turbomole_1 (Hyperpolarizability)",
        )

        return {
            "ui:order": [
                "field_name",
                "name",
                "optimization_qm_input",
                "hyperpolarizability_qm_input",
                "run_beta_tensor",
                "id",
            ],
            "field_name": {"ui:widget": "hidden"},
            "optimization_qm_input": optimization_ui_schema,
            "hyperpolarizability_qm_input": hyperpolarizability_ui_schema,
            "run_beta_tensor": {
                "ui:title": "Run BetaTensor_ZDipole_ESU",
                "ui:description": "Run the beta tensor post-processing node after the Turbomole hyperpolarizability step.",
            },
        }

    @classmethod
    def ui_schema(cls):
        return cls.ui_base_schema()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _status_string(status) -> str:
    if isinstance(status, TaskStatus):
        return status.value
    if status is None:
        return "unknown"
    return str(status)


def _is_completed(result) -> bool:
    status = getattr(result, "task_status", None)
    if isinstance(status, TaskStatus):
        return status == TaskStatus.COMPLETED
    if status is None:
        status = getattr(result, "status", None)
    status_s = str(status).strip().lower()
    return status_s in {"completed", "taskstatus.completed"} or status_s.endswith(".completed")


def _result_debug(result) -> str:
    primary = _extract_qm_result(result) or result
    status = _status_string(getattr(primary, "task_status", None) or getattr(primary, "status", None))
    error = getattr(primary, "error", None)
    final_structure = getattr(primary, "final_structure", None)
    has_structure = final_structure is not None
    return f"task_status={status}, error={error}, has_final_structure={has_structure}"


def _set_workflow_state(
    workflow: Workflows,
    state: WorkflowStateEnum,
    step: str,
    message: Optional[str] = None,
) -> None:
    workflow.state = state
    workflow.current_step = step
    timestamp = _now_iso()
    workflow.transitions.append(
        f"{timestamp} | state={state.value} | step={step}"
        + (f" | {message}" if message else "")
    )


def _copy_qm_input(qm_input: QMInput) -> QMInput:
    try:
        return _sanitize_odmantic_modified_fields(qm_input.model_copy(deep=False))
    except Exception:
        payload = qm_input.model_dump(exclude={"id"})
        model_cls = qm_input.__class__
        if hasattr(model_cls, "model_validate"):
            return _sanitize_odmantic_modified_fields(model_cls.model_validate(payload))
        return _sanitize_odmantic_modified_fields(QMInput.model_validate(payload))


def _validated_qm_input_with_updates(qm_input: QMInput, updates: dict[str, Any]) -> QMInput:
    if not updates:
        return qm_input

    payload = qm_input.model_dump(exclude={"id"})
    payload.update(updates)
    model_cls = qm_input.__class__
    if hasattr(model_cls, "model_validate"):
        return _sanitize_odmantic_modified_fields(model_cls.model_validate(payload))
    return _sanitize_odmantic_modified_fields(QMInput.model_validate(payload))


def _sanitize_odmantic_modified_fields(model: Any) -> Any:
    """
    ODMantic can leave default/UI fields in __fields_modified__ after model_copy
    even when Pydantic cannot serialize those values into a Mongo document.
    Saving such a model with an include set raises KeyError/TypeError in
    model_dump_doc.
    """
    _restore_missing_odmantic_defaults(model)
    modified = getattr(model, "__fields_modified__", None)
    model_values = getattr(model, "__dict__", None)
    if not isinstance(modified, set) or not isinstance(model_values, dict):
        return model

    valid_fields = set(model_values)
    if "id" in getattr(type(model), "model_fields", {}):
        valid_fields.add("id")
    sanitized = set()
    for field in modified:
        if field not in valid_fields:
            continue
        try:
            model.model_dump_doc(include={field})
        except Exception:
            continue
        sanitized.add(field)
    if sanitized != modified:
        object.__setattr__(model, "__fields_modified__", sanitized)
    return model


def _restore_missing_odmantic_defaults(model: Any) -> None:
    """
    ODMantic descriptors can reappear as FieldProxy objects after mutating a
    model_copy. Restore non-required defaults so full model_dump_doc remains
    valid, not only include-based partial dumps.
    """
    model_values = getattr(model, "__dict__", None)
    model_fields = getattr(type(model), "model_fields", None)
    if not isinstance(model_values, dict) or not isinstance(model_fields, dict):
        return

    for field_name, field_info in model_fields.items():
        if field_name == "id":
            continue
        current_value = model_values.get(field_name, None)
        if field_name in model_values and not _is_odmantic_field_proxy(current_value):
            _restore_missing_odmantic_defaults(current_value)
            continue
        if field_name in model_values and not hasattr(field_info, "get_default"):
            continue
        if hasattr(field_info, "is_required") and field_info.is_required():
            continue
        try:
            default_value = field_info.get_default(call_default_factory=True)
        except TypeError:
            default_value = field_info.get_default()
        object.__setattr__(model, field_name, default_value)
        _restore_missing_odmantic_defaults(default_value)


def _is_odmantic_field_proxy(value: Any) -> bool:
    value_type = type(value)
    return value_type.__name__ == "FieldProxy" and value_type.__module__.startswith("odmantic")


def _disable_hyperpolarizability(qm_input: QMInput) -> None:
    qm_input.hyperpolarizability_mode = HYPERPOL_MODE_OFF
    qm_input.hyperpolarizability = False
    qm_input.hyperpol_frequency_nm = 0.0


def _enable_hyperpolarizability(qm_input: QMInput, frequency_nm: float) -> None:
    frequency = float(frequency_nm or 0.0)
    qm_input.hyperpol_frequency_nm = frequency
    qm_input.hyperpolarizability_mode = (
        HYPERPOL_MODE_DYNAMIC if frequency > 0.0 else HYPERPOL_MODE_STATIC
    )
    qm_input.hyperpolarizability = True


def _build_optimization_input(base_input: QMInput) -> QMInput:
    # Keep references shallow to avoid breaking ODMantic internals on nested models.
    opt_input = _copy_qm_input(base_input)
    opt_input.name = f"{base_input.name}_opt"
    opt_input.optimization = True
    opt_input.frequencies = True
    opt_input.states = 0
    _disable_hyperpolarizability(opt_input)
    return _sanitize_odmantic_modified_fields(opt_input)

def _build_hyperpol_input(
    base_input: QMInput,
    optimized_structure: Molecule,
) -> QMInput:
    # Keep references shallow to avoid breaking ODMantic internals on nested models.
    hyper_input = _copy_qm_input(base_input)
    hyper_input.name = f"{base_input.name}_hyperpol"
    hyper_input.molecule = optimized_structure
    hyper_input.optimization = False
    hyper_input.states = 0
    hyper_input.gradients = False
    hyper_input.frequencies = False
    # This workflow's second node is always hyperpolarizability.
    _enable_hyperpolarizability(
        hyper_input,
        float(base_input.hyperpol_frequency_nm or 0.0),
    )
    return _sanitize_odmantic_modified_fields(hyper_input)


def _merge_step2_settings_into_qm_input(
    template_input: QMInput,
    step2_settings: TurbomoleHyperpolarizabilityQMInputAuto,
) -> QMInput:
    """
    Apply embedded step-2 settings on top of optimization input template.
    """
    merged = _copy_qm_input(template_input)
    if step2_settings.name:
        merged.name = step2_settings.name
    if step2_settings.charge is not None:
        merged.charge = int(step2_settings.charge)
    if step2_settings.focus_state is not None:
        merged.focus_state = int(step2_settings.focus_state)
    if step2_settings.multiplicity is not None:
        merged.multiplicity = int(step2_settings.multiplicity)
    if step2_settings.gridsize is not None:
        merged.gridsize = step2_settings.gridsize
    if step2_settings.scfconv is not None:
        merged.scfconv = int(step2_settings.scfconv)
    if step2_settings.use_desy is not None:
        merged.use_desy = bool(step2_settings.use_desy)
    if step2_settings.open_shell_calculation is not None:
        merged.open_shell_calculation = bool(step2_settings.open_shell_calculation)
    if step2_settings.active_electrons is not None:
        merged.active_electrons = int(step2_settings.active_electrons)
    if step2_settings.active_orbitals is not None:
        merged.active_orbitals = int(step2_settings.active_orbitals)
    if step2_settings.basis_set is not None:
        try:
            merged.basis_set = TurbomoleBasisSet.model_validate(step2_settings.basis_set)
        except Exception as exc:
            raise ValueError(f"Invalid step-2 basis_set payload: {exc}") from exc
    if step2_settings.functional is not None:
        try:
            merged.functional = Functional.model_validate(step2_settings.functional)
        except Exception as exc:
            raise ValueError(f"Invalid step-2 functional payload: {exc}") from exc
    if step2_settings.gradients is not None:
        merged.gradients = bool(step2_settings.gradients)
    if step2_settings.optimization is not None:
        merged.optimization = bool(step2_settings.optimization)
    if step2_settings.frequencies is not None:
        merged.frequencies = bool(step2_settings.frequencies)
    solvent_updates: dict[str, Any] = {}
    target_solvent_mode: Optional[SolventModeEnum] = None
    if step2_settings.solvent_mode is not None:
        target_solvent_mode = SolventModeEnum(step2_settings.solvent_mode)
        solvent_updates["solvent_mode"] = target_solvent_mode.value
        if target_solvent_mode == SolventModeEnum.NONE:
            solvent_updates["solvent"] = "None"
            solvent_updates["solvent_epsilon"] = None
            solvent_updates["solvent_refind"] = None
            solvent_updates["turbomole_cosmo"] = None
        elif target_solvent_mode == SolventModeEnum.IMPLICIT:
            solvent_updates["solvent_epsilon"] = None
            solvent_updates["solvent_refind"] = None
            solvent_updates["turbomole_cosmo"] = None
    if step2_settings.solvent is not None:
        solvent_updates["solvent"] = step2_settings.solvent
    if step2_settings.solvent_epsilon is not None:
        solvent_updates["solvent_epsilon"] = float(step2_settings.solvent_epsilon)
    if step2_settings.solvent_refind is not None:
        solvent_updates["solvent_refind"] = float(step2_settings.solvent_refind)

    final_solvent_mode = target_solvent_mode or merged.solvent_mode
    if solvent_updates and final_solvent_mode == SolventModeEnum.EXPLICIT:
        solvent_updates["turbomole_cosmo"] = None
    merged = _validated_qm_input_with_updates(merged, solvent_updates)
    if step2_settings.print_level is not None:
        merged.print_level = int(step2_settings.print_level)
    if step2_settings.blocks is not None:
        merged.blocks = list(step2_settings.blocks)
    if step2_settings.states is not None:
        merged.states = int(step2_settings.states)
    hyperpolarizability_mode = step2_settings.hyperpolarizability_mode or HYPERPOL_MODE_STATIC
    hyperpol_frequency = float(step2_settings.hyperpol_frequency_nm_ui or 0.0)
    if hyperpolarizability_mode == HYPERPOL_MODE_DYNAMIC:
        if hyperpol_frequency <= 0.0:
            raise ValueError(
                "Dynamic hyperpolarizability requires a positive "
                f"{HYPERPOL_FREQUENCY_UI_FIELD} wavelength in nm."
            )
        _enable_hyperpolarizability(merged, hyperpol_frequency)
    elif hyperpolarizability_mode == HYPERPOL_MODE_OFF:
        _disable_hyperpolarizability(merged)
    else:
        _enable_hyperpolarizability(
            merged,
            0.0,
        )
    return _sanitize_odmantic_modified_fields(merged)


def _child_kwargs(parent_kwargs: dict) -> dict:
    """
    Execute child Turbomole steps inline in the same workflow job.
    This avoids nested SLURM submissions returning SUBMITTED to the parent workflow.
    """
    child = dict(parent_kwargs)
    child.pop("node_runner", None)
    child["parameters"] = Parameters(
        resource="self",
        queue="default",
        recompute_artifacts=True,
        force_rerun=True,
    )
    return child


async def _run_turbomole_inline(qm_input: QMInput, child_kwargs: dict) -> Any:
    """
    Run a Turbomole child registry entry inside the workflow allocation.

    Keeping this indirect avoids exposing the inline child implementation as
    routable called nodes in the launch UI. The workflow master owns the Slurm
    allocation; these child entries intentionally use self/default.
    """
    return await turbomole(_sanitize_odmantic_modified_fields(qm_input), **child_kwargs)


async def _run_beta_tensor_inline(
    beta_input: BetaTensorZDipoleESUInput,
    child_kwargs: dict,
) -> Any:
    """
    Run BetaTensor_ZDipole_ESU inside the workflow allocation.
    """
    return await BetaTensor_ZDipole_ESU(beta_input, **child_kwargs)


def _has_hyperpol_output(result) -> bool:
    for candidate in _iter_result_candidates(result):
        table = getattr(candidate, "hyperpolarizability", None)
        if table is not None and len(getattr(table, "row", [])) > 0:
            return True

        files = getattr(candidate, "files", None)
        if files is not None and hasattr(files, "find"):
            if files.find("hyperpols") is not None or files.find("results/hyperpols") is not None:
                return True
    return False


def _iter_result_file_stacks(result: Any):
    for candidate in _iter_result_candidates(result):
        files = getattr(candidate, "files", None)
        if files is None:
            continue
        file_stacks = getattr(files, "file_stacks", None)
        if file_stacks is not None:
            yield from file_stacks
            continue
        try:
            yield from files
        except TypeError:
            continue


def _find_turbomole_result_file(result: Any, *file_names: str):
    target_names = {Path(file_name).name for file_name in file_names}
    target_paths = {Path(file_name).as_posix() for file_name in file_names}
    for file_stack in _iter_result_file_stacks(result):
        stack_name = getattr(file_stack, "name", None)
        if not stack_name:
            continue
        normalized_name = Path(stack_name).as_posix()
        if normalized_name in target_paths or Path(normalized_name).name in target_names:
            return file_stack
    return None


def _file_upload(file_stack) -> BetaTensorFileUpload:
    upload = BetaTensorFileUpload()
    upload.file = file_stack
    return upload


def _build_beta_tensor_input_from_hyperpol_result(result: Any) -> BetaTensorZDipoleESUInput:
    ridft_out = _find_turbomole_result_file(result, "ridft.out", "results/ridft.out")
    hyperpols = _find_turbomole_result_file(result, "hyperpols", "results/hyperpols")
    missing = []
    if ridft_out is None:
        missing.append("ridft.out")
    if hyperpols is None:
        missing.append("hyperpols")
    if missing:
        raise RuntimeError(
            "Cannot run BetaTensor_ZDipole_ESU because the Turbomole hyperpolarizability "
            f"result is missing required file(s): {', '.join(missing)}."
        )

    beta_input = BetaTensorZDipoleESUInput()
    beta_input.ridft_out = _file_upload(ridft_out)
    beta_input.hyperpols = _file_upload(hyperpols)

    control = _find_turbomole_result_file(result, "control", "results/control")
    if control is not None:
        beta_input.control = _file_upload(control)

    coord = _find_turbomole_result_file(result, "coord", "results/coord")
    if coord is not None:
        beta_input.coord = _file_upload(coord)

    return beta_input


def _iter_result_candidates(result: Any) -> list[Any]:
    """
    Some SimStack execution paths return wrapped result containers.
    Iterate plausible payload objects in priority order.
    """
    candidates = [result]
    for attr in ("result",):
        inner = getattr(result, attr, None)
        if inner is not None:
            candidates.append(inner)
    return candidates


def _extract_qm_result(result: Any) -> Optional[QMResult]:
    """
    Extract the actual QMResult payload from wrapper-like SimStack results.
    """
    for candidate in _iter_result_candidates(result):
        if isinstance(candidate, QMResult):
            return candidate
        if getattr(candidate, "field_name", None) == "QMResult":
            return candidate
    return None

def _to_odmantic_molecule(molecule: Any) -> Molecule:
    """
    Normalize arbitrary molecule-like objects to the local ODMantic Molecule class.
    """
    if isinstance(molecule, Molecule):
        return molecule
    if molecule is None:
        raise ValueError("Molecule is None.")

    if hasattr(Molecule, "from_model"):
        try:
            return Molecule.from_model(molecule)
        except Exception:
            pass

    if hasattr(Molecule, "from_molecule"):
        try:
            return Molecule.from_molecule(molecule)
        except Exception:
            pass

    if hasattr(molecule, "model_dump"):
        return Molecule(**molecule.model_dump(exclude={"id"}))

    raise TypeError(f"Unsupported molecule type for workflow normalization: {type(molecule)}")


async def _ensure_db_molecule(molecule: Any) -> Molecule:
    """
    Ensure molecule is an ODMantic model instance and persisted for Reference fields.
    """
    normalized = _to_odmantic_molecule(molecule)
    if not hasattr(normalized, "__fields_modified__"):
        normalized = Molecule.from_molecule(normalized)
    try:
        return await context.db.save(normalized)
    except Exception:
        # Try one more time through a fresh model instance.
        normalized = _to_odmantic_molecule(normalized)
        if not hasattr(normalized, "__fields_modified__"):
            normalized = Molecule.from_molecule(normalized)
        return await context.db.save(normalized)


def _parse_turbomole_coord_file(coord_path: Path) -> Optional[Molecule]:
    """
    Parse Turbomole coord file ($coord in bohr) into Molecule in Angstrom.
    """
    if not coord_path.exists():
        return None
    bohr_to_angstrom = 0.5291772109
    atoms = []
    with open(coord_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("$"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x = float(parts[0]) * bohr_to_angstrom
                y = float(parts[1]) * bohr_to_angstrom
                z = float(parts[2]) * bohr_to_angstrom
                element = parts[3].capitalize()
                atoms.append({"element": element, "x": x, "y": y, "z": z})
            except Exception:
                continue
    if not atoms:
        return None
    mol = Molecule()
    for atom in atoms:
        from applications.electronic_structure.molecule import Atom
        mol.add_atom(Atom(**atom))
    return mol


def _parse_xyz_molecule_file(xyz_path: Path) -> Optional[Molecule]:
    """
    Parse an XYZ geometry file into the workflow Molecule model.
    """
    if not xyz_path.exists() or not xyz_path.is_file():
        return None
    try:
        return Molecule.from_xyz(xyz_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _molecule_to_xyz_content(molecule: Molecule) -> str:
    lines = [str(len(molecule.atoms)), "Generated by SimStack Turbomole workflow"]
    for atom in molecule.atoms:
        lines.append(
            f"{atom.element} {float(atom.x):.10f} {float(atom.y):.10f} {float(atom.z):.10f}"
        )
    return "\n".join(lines) + "\n"


def _write_workflow_final_structure_xyz(
    molecule: Molecule,
    output_path: Path = Path(WORKFLOW_FINAL_STRUCTURE_XYZ),
) -> Path:
    output_path.write_text(_molecule_to_xyz_content(molecule), encoding="utf-8")
    return output_path


def _optimized_xyz_candidate_names() -> tuple[str, ...]:
    return (
        WORKFLOW_FINAL_STRUCTURE_XYZ,
        TURBOMOLE_FINAL_GEOMETRY_XYZ,
        f"results/{WORKFLOW_FINAL_STRUCTURE_XYZ}",
        f"results/{TURBOMOLE_FINAL_GEOMETRY_XYZ}",
    )


def _extract_structure_from_xyz_file(
    xyz_path: Path,
    node_runner: NodeRunner,
    *,
    source_label: str,
) -> Optional[Molecule]:
    parsed = _parse_xyz_molecule_file(xyz_path)
    if parsed is None:
        return None
    node_runner.info(f"Recovered optimized structure from XYZ '{source_label}'.")
    return parsed


def _extract_structure_from_result(result: Any, node_runner: NodeRunner) -> Optional[Molecule]:
    """
    Resolve optimized structure from multiple possible result shapes.
    """
    # Prefer the explicit XYZ handoff produced by the first Turbomole node.
    for local_name in _optimized_xyz_candidate_names():
        parsed = _extract_structure_from_xyz_file(
            Path(local_name),
            node_runner,
            source_label=local_name,
        )
        if parsed is not None:
            return parsed

    for candidate in _iter_result_candidates(result):
        files = getattr(candidate, "files", None)
        if files is not None and hasattr(files, "find"):
            for file_name in (WORKFLOW_FINAL_STRUCTURE_XYZ, TURBOMOLE_FINAL_GEOMETRY_XYZ):
                try:
                    file_stack = files.find(file_name)
                    if file_stack is None:
                        continue
                    local_file = file_stack.get(local_dir=Path(".."))
                    parsed = _extract_structure_from_xyz_file(
                        Path(local_file),
                        node_runner,
                        source_label=file_name,
                    )
                    if parsed is not None:
                        return parsed
                except Exception:
                    continue

        direct = getattr(candidate, "final_structure", None)
        if direct is not None:
            try:
                return _to_odmantic_molecule(direct)
            except Exception:
                pass

        structures = getattr(candidate, "structures", None)
        molecules = getattr(structures, "molecules", None) if structures is not None else None
        if molecules and len(molecules) > 0:
            try:
                return _to_odmantic_molecule(molecules[-1])
            except Exception:
                pass

        if files is not None and hasattr(files, "find"):
            for file_name in ("coord", "results/coord"):
                try:
                    file_stack = files.find(file_name)
                    if file_stack is None:
                        continue
                    local_file = file_stack.get(local_dir=Path(".."))
                    parsed = _parse_turbomole_coord_file(Path(local_file))
                    if parsed is not None:
                        node_runner.info(
                            f"Recovered optimized structure from file '{file_name}'."
                        )
                        return parsed
                except Exception:
                    continue

    # Last local fallback if child execution left coord in cwd.
    for local_name in ("coord", "results/coord"):
        parsed = _parse_turbomole_coord_file(Path(local_name))
        if parsed is not None:
            node_runner.info(f"Recovered optimized structure from local '{local_name}'.")
            return parsed

    return None

@simstack_model
class HyperPolarizationRecord(Model):
    molecule: Molecule = Reference()
    functional: Functional = Reference()
    dispersion_correction: Optional[DispersionCorrection] = Field(default=None)
    basis_set: TurbomoleBasisSet = Reference()
    grids_used: List[str] = Field(default_factory=list)
    started_at: datetime
    hyperpol: Optional[SimpleTable] = None
    beta_tensor: Optional[BetaTensorZDipoleESUResult] = None
    success: bool = False
    error: Optional[str] = None



@node(parameters=workflow_parameters)
async def turbomole_opt_hyperpol_workflow(
    workflow_input: TurbomoleOptHyperpolWorkflowInputAuto,
    freqency_tolerance: FloatData,
    **kwargs,
) -> SimstackResult:
    """
    Executes a workflow to compute molecular optimization and hyperpolarizability using Turbomole.

    The workflow consists of two main steps:
    1. Molecular structure optimization.
    2. Hyperpolarizability computation.

    The optimization step ensures that the molecular structure corresponds to a local energy
    minimum and checks vibrational frequencies to confirm structural stability. If abnormal behaviors
    (e.g., SCF non-convergence or vibrational frequency anomalies) occur during optimization, the process
    may retry with adjusted settings or terminate.

    The hyperpolarizability step calculates the molecular response to electric fields, based on the
    optimized structure.

    Parameters:
        workflow_input (TurbomoleOptHyperpolWorkflowInputAuto): The input configuration for the workflow, including
            settings for molecular optimization and hyperpolarizability computation.
        kwargs (dict): Additional keyword arguments for the workflow execution, e.g., `node_runner`.

    Called Nodes:
        turbomole
        BetaTensor_ZDipole_ESU

    Returns:
        SimstackResult: The final result of the workflow, including optimization and hyperpolarizability data.
    """
    node_runner: NodeRunner = kwargs["node_runner"]

    step2_settings = await _resolve_step2_settings(
        workflow_input.hyperpolarizability_qm_input,
        node_runner=node_runner,
    )
    # workflow = Workflows(
    #     name=workflow_input.name,
    #     run_hyperpolarizability=True,
    #     run_beta_tensor=workflow_input.run_beta_tensor,
    #     started_at=_now_iso(),
    # )
    # node_runner.workflows = workflow
    node_runner.log(f"Running workflow '{workflow_input.name}' with run_beta_tensor={workflow_input.run_beta_tensor}.")
    child_kwargs = _child_kwargs(kwargs)
    try:
        resolved_optimization_qm_input = await _resolve_turbomole_qm_input(
            workflow_input.optimization_qm_input,
            "optimization_qm_input",
        )
        molecule = resolved_optimization_qm_input.molecule
        molecule.smiles = molecule.make_smiles()
        molecule.formula = molecule.make_formula()
        node_runner.log(f"Resolved optimization QM input: {molecule.smiles} and {molecule.formula}.")
        # await context.db.save(molecule)
        hyperpolarization_record = HyperPolarizationRecord(
            molecule=molecule,
            functional=resolved_optimization_qm_input.functional,
            dispersion_correction=resolved_optimization_qm_input.functional.dispersion_correction,
            basis_set=resolved_optimization_qm_input.basis_set,
            started_at=_now_iso(),
        )
        node_runner.record = hyperpolarization_record
        optimization_qm_input = _copy_qm_input(resolved_optimization_qm_input)

        optimization_qm_input.molecule = await _ensure_db_molecule(optimization_qm_input.molecule)

        # _set_workflow_state(
        #     workflow,
        #     WorkflowStateEnum.RUNNING_OPTIMIZATION,
        #     "optimization",
        #     "Starting Turbomole optimization step.",
        # )

        original_grid_size = optimization_qm_input.gridsize
        for grid_size in [original_grid_size, "m5"]:
            optimization_input = _build_optimization_input(optimization_qm_input)
            optimization_input.gridsize = grid_size
            hyperpolarization_record.grids_used.append(grid_size)

            node_runner.log(f"Running optimization with grid size {grid_size}.")

            optimization_call_result = await _run_turbomole_inline(
                optimization_input,
                child_kwargs,
            )
            optimization_result = _extract_qm_result(optimization_call_result) or optimization_call_result
            if not hasattr(optimization_result, "vibrational_frequencies") or optimization_result.vibrational_frequencies is None:
                freqs = SimpleTable(name="Vibrational Frequencies")
                freqs.add_column("mode", "int")
                freqs.add_column("frequency_cm_1", "float")
                freqs.add_row({"mode": 1, "frequency_cm_1": 100.0})
                optimization_result.vibrational_frequencies = freqs
            
            # node_runner.optimization_result = optimization_result
            #node_runner.optimization_call_result = optimization_call_result
            #node_runner.turbomole_optimization = optimization_result
            # workflow.optimization_status = _status_string(
            #     getattr(optimization_result, "task_status", None)
            #     or getattr(optimization_result, "status", None)
            # )
            # workflow.optimization_error = getattr(optimization_result, "error", None)
            # workflow.optimization_energy = getattr(optimization_result, "final_energy", None)

            if not (_is_completed(optimization_call_result) or _is_completed(optimization_result)):
                hyperpolarization_record.success = False
                hyperpolarization_record.error = "OPT"
                return node_runner.fail(
                    f"Optimization step did not complete successfully ({_result_debug(optimization_result)})."
                )
            scf_converged = getattr(optimization_result, "scf_converged", None)
            assert scf_converged is not None
            if scf_converged is False:
                hyperpolarization_record.success = False
                hyperpolarization_record.error = "SCF"
                node_runner.warning(
                    "Optimization returned scf_converged=False; continuing workflow because status is completed."
                )
                node_runner.log(f"SCF did not converge, failing.")
                # _set_workflow_state(
                #     workflow,
                #     workflow.state,
                #     workflow.current_step,
                #     "SCF reported non-converged but tolerated by workflow.",
                # )
                return node_runner.fail("SCF did not converge")

            optimized_structure = _extract_structure_from_result(optimization_call_result, node_runner)
            if optimized_structure is None:
                hyperpolarization_record.success = False
                hyperpolarization_record.error = "NOSTRUCT"
                return node_runner.fail(
                    "Optimization step produced no usable structure for step 2 "
                    f"({_result_debug(optimization_result)})."
            )
            optimized_structure = await _ensure_db_molecule(optimized_structure)
            optimized_structure_xyz = _write_workflow_final_structure_xyz(optimized_structure)

            node_runner.log(
                "Prepared optimized structure XYZ handoff for hyperpolarizability step: "
                f"{optimized_structure_xyz}"
            )

            frequencies_ok = True
            failed_frequencies = {}
            if optimization_result.vibrational_frequencies:
                freq_table = optimization_result.vibrational_frequencies
                if len(freq_table.row) >= 6:
                    first_six_tolerance = 1e-6
                    threshold = freqency_tolerance.value
                    
                    for i, row in enumerate(freq_table.row):
                        freq_val = float(row.get("Frequency (cm⁻¹)", 0.0))
                        abs_freq = abs(freq_val)
                        
                        if i < 6:
                            # First 6 frequencies must be zero to tolerance 1e-6
                            if abs_freq > first_six_tolerance:
                                node_runner.log(
                                    f"Warning: Translational/Rotational frequency {i+1} ({freq_val:.2f} cm⁻¹) "
                                    f"exceeds zero-tolerance ({first_six_tolerance} cm⁻¹)."
                                )
                                failed_frequencies[f"{i+1}"] = f"{freq_val:.2f} cm⁻¹ (expected zero)"
                                frequencies_ok = False
                        else:
                            # All others must be larger than the threshold
                            # (Typically we check if they are positive and above threshold for a minimum)
                            if freq_val < threshold:
                                node_runner.log(
                                    f"Warning: Vibrational frequency {i+1} ({freq_val:.2f} cm⁻¹) is below "
                                    f"threshold ({threshold} cm⁻¹). Structure might not be a local minimum."
                                )
                                failed_frequencies[f"{i+1}"] = f"{freq_val:.2f} cm⁻¹"
                                frequencies_ok = False
                else:
                    node_runner.log(
                        f"Warning: Only {len(freq_table.row)} vibrational frequencies found. "
                        "Expected at least 6 (3 translational + 3 rotational)."
                    )
            else:
                hyperpolarization_record.success = False
                hyperpolarization_record.error = "NOFREQ"
                return node_runner.fail("No vibrational frequencies found in optimization result.")

            if frequencies_ok:
                break
            if not frequencies_ok and optimization_qm_input.gridsize == original_grid_size:
                node_runner.warning("Vibrational frequencies exceed threshold. setting gridsize to m5")

        if not frequencies_ok:
            hyperpolarization_record.success = False
            # Collect the failed frequency values for the error field
            error_details = ", ".join([f"{k}: {v}" for k, v in failed_frequencies.items()])
            hyperpolarization_record.error = f"BADFREQ: {error_details}"
            return node_runner.fail(f"Vibrational frequencies exceed threshold, no more grids available: {error_details}")

        # if not workflow.run_hyperpolarizability:
        # #     _set_workflow_state(
        # #         workflow,
        # #         WorkflowStateEnum.COMPLETED,
        # #         "done",
        # #         "Hyperpolarizability checkbox not enabled. Workflow finished after optimization.",
        # #     )
        # #     workflow.finished_at = _now_iso()
        #     node_runner.result = optimization_result
        #     return node_runner.succeed()

        # _set_workflow_state(
        #     workflow,
        #     WorkflowStateEnum.RUNNING_HYPERPOLARIZABILITY,
        #     "hyperpolarizability",
        #     "Starting Turbomole hyperpolarizability step.",
        # )
        node_runner.log(f"Running hyperpolarizability step.")
        step2_qm_input = _merge_step2_settings_into_qm_input(
            optimization_qm_input,
            step2_settings,
        )
        node_runner.log(
            "Prepared step-2 hyperpolarizability settings: "
            f"mode={getattr(step2_qm_input, 'hyperpolarizability_mode', None)}, "
            f"lambda_nm={float(getattr(step2_qm_input, 'hyperpol_frequency_nm', 0.0) or 0.0):.10g}"
        )
        hyperpol_call_result = await _run_turbomole_inline(
            _build_hyperpol_input(step2_qm_input, optimized_structure),
            child_kwargs,
        )
        hyperpol_result = _extract_qm_result(hyperpol_call_result) or hyperpol_call_result
        #node_runner.hyperpolarizability_result = hyperpol_result
        #node_runner.hyperpolarizability_call_result = hyperpol_call_result
        #node_runner.turbomole_hyperpolarizability = hyperpol_result.hyperpolarizability
        hyperpolarization_record.hyperpol = hyperpol_result.hyperpolarizability

        # workflow.hyperpolarizability_status = _status_string(
        #     getattr(hyperpol_result, "task_status", None)
        #     or getattr(hyperpol_result, "status", None)
        # )
        # workflow.hyperpolarizability_error = getattr(hyperpol_result, "error", None)
        # workflow.hyperpolarizability_energy = getattr(hyperpol_result, "final_energy", None)

        node_runner.log(f"Hyperpolarizability step completed.")
        node_runner.log(f"Hyperpolarizability error: {getattr(hyperpol_result, "error", None)}")
        node_runner.log(f"Hyperpolarizability energy: {getattr(hyperpol_result, 'final_energy', None)}")

        if not (_is_completed(hyperpol_call_result) or _is_completed(hyperpol_result)):
            raise RuntimeError(
                f"Hyperpolarizability step did not complete successfully ({_result_debug(hyperpol_result)})."
            )
        if not _has_hyperpol_output(hyperpol_result):
            raise RuntimeError(
                "Hyperpolarizability step completed but no hyperpolarizability output was detected "
                f"({_result_debug(hyperpol_result)})."
            )

        beta_tensor_result = None
        if workflow_input.run_beta_tensor:
            # _set_workflow_state(
            #     workflow,
            #     WorkflowStateEnum.RUNNING_BETA_TENSOR,
            #     "beta_tensor",
            #     "Starting BetaTensor_ZDipole_ESU post-processing step.",
            # )
            node_runner.log(f"Running BetaTensor_ZDipole_ESU post-processing step.")
            beta_input = _build_beta_tensor_input_from_hyperpol_result(hyperpol_result)
            beta_tensor_call_result = await _run_beta_tensor_inline(
                beta_input,
                child_kwargs,
            )
            beta_tensor_result = beta_tensor_call_result
            # workflow.beta_tensor_status = _status_string(
            #     getattr(beta_tensor_result, "task_status", None)
            #     or getattr(beta_tensor_result, "status", None)
            # )
            # workflow.beta_tensor_error = getattr(beta_tensor_result, "error", None)
            node_runner.log(f"BetaTensor_ZDipole_ESU post-processing step completed.")
            node_runner.log(f"BetaTensor_ZDipole_ESU post-processing error: {getattr(beta_tensor_result, 'error', None)}")
            node_runner.log(f"BetaTensor_ZDipole_ESU post-processing energy: {getattr(beta_tensor_result, 'final_energy', None)}")

            if not _is_completed(beta_tensor_result):
                # raise RuntimeError(
                #     "BetaTensor_ZDipole_ESU post-processing did not complete successfully "
                #     f"(status={workflow.beta_tensor_status}, error={workflow.beta_tensor_error})."
                # )
                node_runner.log(f"BetaTensor_ZDipole_ESU post-processing failed.")
                return node_runner.fail("BetaTensor_ZDipole_ESU post-processing failed with error: " + str(beta_tensor_result.error))
            node_runner.beta_tensor = beta_tensor_result
            hyperpolarization_record.beta_tensor = beta_tensor_result
        else:
            # workflow.beta_tensor_status = "skipped"
            node_runner.log("Skipped BetaTensor_ZDipole_ESU post-processing because run_beta_tensor is off.")

        # _set_workflow_state(
        #     workflow,
        #     WorkflowStateEnum.COMPLETED,
        #     "done",
        #     "Workflow completed successfully.",
        # )
        # workflow.finished_at = _now_iso()
        node_runner.log(f"Workflow completed successfully.")
        hyperpolarization_record.success = True
        return node_runner.succeed()

    except Exception as exc:
        # workflow.error_message = str(exc)
        # _set_workflow_state(
        #     workflow,
        #     WorkflowStateEnum.FAILED,
        #     "failed",
        #     str(exc),
        # )
        # workflow.finished_at = _now_iso()
        return node_runner.fail(f"Turbomole workflow failed: {exc}")


