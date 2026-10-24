from enum import Enum
from typing import Any, Optional

from odmantic import Field, Model
from pydantic import model_validator

from simstack.models import simstack_model
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema


class TurbomoleSemiempiricalMethod(str, Enum):
    """Semiempirical methods supported by TURBOMOLE."""

    AM1 = "am1"
    PM3 = "pm3"
    MNDO = "mndo"
    MNDOD = "mndod"

    @classmethod
    def coerce(cls, value: Any) -> "TurbomoleSemiempiricalMethod":
        if isinstance(value, cls):
            return value
        if value is None:
            raise ValueError("Semiempirical method cannot be None")
        raw = getattr(value, "value", value)
        text = str(raw).strip().lower().replace("/", "").replace("-", "").replace("_", "")
        mapping = {
            "am1": cls.AM1,
            "pm3": cls.PM3,
            "mndo": cls.MNDO,
            "mndod": cls.MNDOD,
        }
        if text in mapping:
            return mapping[text]
        raise ValueError(
            f"Unsupported TURBOMOLE semiempirical method: {value!r}. "
            f"Supported methods: {[m.value for m in cls]}"
        )


@simstack_model
class TurbomoleSemiempiricalInput(Model):
    """TURBOMOLE semiempirical calculation settings."""

    field_name: str = "TurbomoleSemiempiricalInput"

    method: TurbomoleSemiempiricalMethod = Field(
        default=TurbomoleSemiempiricalMethod.AM1,
        json_schema_extra={
            "enum": [m.value for m in TurbomoleSemiempiricalMethod],
            "title": "Method",
            "description": "TURBOMOLE semiempirical Hamiltonian",
        },
    )
    charge: int = Field(0, json_schema_extra={"description": "Net charge of the molecule"})
    multiplicity: int = Field(1, json_schema_extra={"description": "Spin multiplicity"})
    optimization: bool = Field(True, json_schema_extra={"description": "Perform geometry optimization"})
    max_opt_cycles: int = Field(
        100,
        ge=1,
        json_schema_extra={
            "description": "Maximum number of geometry optimization steps",
            "title": "max_opt_cycles",
        },
    )
    scfiterlimit: int = Field(
        100,
        ge=1,
        json_schema_extra={
            "description": "Maximum number of SCF iterations",
            "title": "scfiterlimit",
        },
    )
    scfconv: int = Field(
        8,
        json_schema_extra={
            "description": "TURBOMOLE SCF convergence threshold exponent ($scfconv)",
            "title": "scfconv",
        },
    )
    gradients: bool = Field(
        False, json_schema_extra={"description": "Calculate gradients (forces) for the molecule"}
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            return data
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        if "method" in data and data["method"] is not None:
            data["method"] = TurbomoleSemiempiricalMethod.coerce(data["method"]).value
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
        ui["method"] = {"ui:widget": "select"}
        return ui
