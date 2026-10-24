from typing import Any, Dict, List, Optional, Union

from odmantic import EmbeddedModel, Field, Model
from pydantic import model_validator

from molecular_qm_models import GridType, OptimizationAccuracy, SCFAccuracy
from molecular_qm_models.basis_set import BasisSet
from molecular_qm_models.density_functional import Functional
from molecular_qm_turbomole.models.semiempirical import (
    TurbomoleSemiempiricalInput,
    TurbomoleSemiempiricalMethod,
)
from molecular_qm_turbomole.models.turbomole_functional import (
    TurbomoleFunctional,
    TurbomoleFunctionalEnum,
    as_turbomole_functional_doc,
)
from molecular_qm_turbomole.models.turbomole_input import (
    TURBOMOLE_BASIS_SET_VALUES,
    TURBOMOLE_DEFAULT_BASIS_SET,
    TURBOMOLE_DEFAULT_GRID_SIZE,
    TURBOMOLE_GRID_SIZE_VALUES,
    TurbomoleBasisSet2,
    TurbomoleDispersionCorrection,
    normalize_functional_and_dispersion,
)
from simstack.models import simstack_model
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema


def _coerce_grid_string(value: Any) -> str:
    if value is None:
        return TURBOMOLE_DEFAULT_GRID_SIZE
    if isinstance(value, GridType):
        mapping = {
            GridType.Grid1: "1",
            GridType.Grid2: "m3",
            GridType.Grid3: "m4",
            GridType.Grid4: "m5",
            GridType.Grid5: "5",
        }
        return mapping.get(value, "m3")
    raw = getattr(value, "value", value)
    text = str(raw).strip().lower()
    if text in TURBOMOLE_GRID_SIZE_VALUES:
        return text
    grid_type_aliases = {
        "grid1": "1",
        "grid2": "m3",
        "grid3": "m4",
        "grid4": "m5",
        "grid5": "5",
    }
    if text in grid_type_aliases:
        return grid_type_aliases[text]
    return text or TURBOMOLE_DEFAULT_GRID_SIZE


@simstack_model
class OptimizationStepInput(EmbeddedModel):
    """Settings for a single TURBOMOLE optimization step."""

    field_name: str = "OptimizationStepInput"

    basis_set: TurbomoleBasisSet2 = Field(default_factory=TurbomoleBasisSet2)
    functional: TurbomoleFunctional = Field(default_factory=TurbomoleFunctional)
    dispersion_correction: TurbomoleDispersionCorrection = Field(
        default_factory=TurbomoleDispersionCorrection,
        json_schema_extra={
            "description": "Dispersion correction for this optimization step",
            "title": "Dispersion Correction",
        },
    )
    grid: str = Field(
        TURBOMOLE_DEFAULT_GRID_SIZE,
        json_schema_extra={
            "enum": TURBOMOLE_GRID_SIZE_VALUES,
            "description": "TURBOMOLE DFT integration grid size (e.g. 1-7, m3-m5)",
            "title": "Grid size",
        },
    )
    scf_accuracy: SCFAccuracy = Field(
        SCFAccuracy.Medium,
        json_schema_extra={"description": "SCF convergence accuracy"},
    )
    optimization_accuracy: OptimizationAccuracy = Field(
        OptimizationAccuracy.Medium,
        json_schema_extra={"description": "Geometry optimization accuracy"},
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            return data
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        if "grid" not in data and "gridsize" in data:
            data["grid"] = data.pop("gridsize")
        elif "grid" not in data and "grid_type" in data:
            data["grid"] = data.pop("grid_type")
        if "grid" in data:
            data["grid"] = _coerce_grid_string(data["grid"])
        if "basis_set" in data and isinstance(data["basis_set"], str):
            data["basis_set"] = {"field_name": "TurbomoleBasisSet2", "basis_set": data["basis_set"]}
        data = normalize_functional_and_dispersion(data)
        return data

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        return schema

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        return ui


@simstack_model
class PreOptimizerInput(Model):
    """Parameters controlling the multistep optimization sequence."""

    field_name: str = "PreOptimizerInput"

    semiempirical_opt: bool = Field(
        False,
        json_schema_extra={"title": "Semiempirical pre-optimization"},
    )
    semiempirical_method: Optional[TurbomoleSemiempiricalMethod] = Field(
        None,
        json_schema_extra={
            "title": "Semiempirical Method",
            "description": "Select one of TURBOMOLE's semiempirical optimization methods (AM1, PM3, MNDO, MNDOD) or None",
            "enum": [None] + [m.value for m in TurbomoleSemiempiricalMethod],
        },
    )
    semiempirical_input: Optional[TurbomoleSemiempiricalInput] = Field(
        None,
        json_schema_extra={
            "title": "Semiempirical input",
            "description": "Full TURBOMOLE semiempirical settings used for the optional pre-optimization",
        },
    )
    steps: List[OptimizationStepInput] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            return data
        if "field_name" not in data:
            data["field_name"] = cls.__name__

        method_val = data.get("semiempirical_method")
        if method_val is not None:
            if isinstance(method_val, str) and method_val.strip().lower() in ("none", "null", ""):
                data["semiempirical_method"] = None
                method_val = None
            else:
                coerced = TurbomoleSemiempiricalMethod.coerce(method_val)
                data["semiempirical_method"] = coerced.value

        if "semiempirical_opt" not in data:
            data["semiempirical_opt"] = (
                data.get("semiempirical_method") is not None
                or data.get("semiempirical_input") is not None
            )

        if not data.get("semiempirical_opt"):
            if data.get("semiempirical_method") is None:
                data["semiempirical_input"] = None
                return data
            else:
                data["semiempirical_opt"] = True

        semi_inp = data.get("semiempirical_input")
        chosen_method = data.get("semiempirical_method") or TurbomoleSemiempiricalMethod.AM1.value
        if semi_inp is None:
            data["semiempirical_input"] = {
                "field_name": "TurbomoleSemiempiricalInput",
                "method": chosen_method,
                "optimization": True,
            }
        elif isinstance(semi_inp, dict):
            semi_inp.setdefault("field_name", "TurbomoleSemiempiricalInput")
            if data.get("semiempirical_method") is not None:
                semi_inp["method"] = chosen_method
            semi_inp["optimization"] = True
        return data

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        schema["properties"].pop("semiempirical_input", None)
        semi_schema = TurbomoleSemiempiricalInput.json_schema()
        for name, definition in (semi_schema.pop("$defs", None) or {}).items():
            schema.setdefault("$defs", {}).setdefault(name, definition)
        semi_schema["title"] = "Semiempirical input"
        semi_schema["description"] = (
            "Full TURBOMOLE semiempirical settings used for the optional pre-optimization"
        )
        semi_schema["default"] = {
            "field_name": "TurbomoleSemiempiricalInput",
            "method": TurbomoleSemiempiricalMethod.AM1.value,
            "optimization": True,
        }
        schema.setdefault("dependencies", {})["semiempirical_opt"] = {
            "oneOf": [
                {"properties": {"semiempirical_opt": {"const": False}}},
                {
                    "properties": {
                        "semiempirical_opt": {"const": True},
                        "semiempirical_input": semi_schema,
                    }
                },
            ]
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui = generate_ui_schema(cls)
        ui["field_name"] = {"ui:widget": "hidden"}
        ui["semiempirical_opt"] = {
            "ui:widget": "checkbox",
            "ui:title": "Semiempirical pre-optimization",
        }
        ui["semiempirical_method"] = {
            "ui:widget": "select",
            "ui:title": "Semiempirical method",
        }
        ui.setdefault("semiempirical_input", {})["ui:condition"] = {"semiempirical_opt": True}
        return ui
