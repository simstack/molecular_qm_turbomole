from enum import Enum
from typing import Any, List, Optional

from odmantic import EmbeddedModel, Field, Model, Reference
from pydantic import field_validator, model_validator

from molecular_qm_models.dispersion_correction import DispersionCorrectionEnum
from molecular_qm_models.molecule import Molecule
from molecular_qm_turbomole.lib.control_utils import parse_control_groups
from molecular_qm_turbomole.models.turbomole_functional import (
    TurbomoleFunctional,
    TurbomoleFunctionalEnum,
    as_turbomole_functional_doc,
)
from simstack.models import simstack_model
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema

TURBOMOLE_DEFAULT_BASIS_SET = "def2-SVP"
TURBOMOLE_DEFAULT_GRID_SIZE = "m3"
TURBOMOLE_DEFAULT_SCFCONV = 8
TURBOMOLE_DEFAULT_SCFITERLIMIT = 100
TURBOMOLE_DEFAULT_MAX_OPT_CYCLES = 100

TURBOMOLE_BASIS_SET_VALUES: List[str] = [
    "SV",
    "SV(P)",
    "SVP",
    "TZVP",
    "TZVPP",
    "QZVP",
    "QZVPP",
    "def2-SV(P)",
    "def2-SVP",
    "def2-TZVP",
    "def2-TZVPP",
    "def2-SVPD",
    "def2-TZVPD",
    "def2-QZVPD",
    "def2-TZVPPD",
    "def2-QZVPPD",
    "cc-pVDZ",
    "cc-pVTZ",
    "cc-pVQZ",
    "aug-cc-pVDZ",
    "aug-cc-pVTZ",
    "aug-cc-pVQZ",
]

TURBOMOLE_GRID_SIZE_VALUES: List[str] = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "m3",
    "m4",
    "m5",
    "1a",
    "2a",
    "3a",
    "4a",
    "5a",
    "6a",
    "7a",
]


class SolventModeEnum(str, Enum):
    NONE = "none"
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"


class HyperpolarizabilityModeEnum(str, Enum):
    NONE = "none"
    STATIC = "static"
    DYNAMIC = "dynamic"


class TurbomoleMethodEnum(str, Enum):
    DFT = "DFT"
    HF = "HF"
    MP2 = "MP2"
    CC2 = "CC2"
    ADC2 = "ADC(2)"


RICC2_METHODS = (
    TurbomoleMethodEnum.HF,
    TurbomoleMethodEnum.MP2,
    TurbomoleMethodEnum.CC2,
    TurbomoleMethodEnum.ADC2,
)


def _nested_dispersion_payload(functional: Any) -> Any:
    if isinstance(functional, dict):
        nested = functional.get("dispersion_correction")
    else:
        nested = getattr(functional, "dispersion_correction", None)
    if nested in (None, {}):
        return None
    if hasattr(nested, "model_dump"):
        return nested.model_dump(exclude={"id"})
    return nested


def normalize_functional_and_dispersion(data: dict) -> dict:
    functional = data.get("functional")
    if data.get("dispersion_correction") in (None, {}):
        nested = _nested_dispersion_payload(functional)
        if nested is not None:
            data["dispersion_correction"] = nested
    if functional is not None:
        if isinstance(functional, TurbomoleFunctional):
            data["functional"] = functional
        else:
            data["functional"] = as_turbomole_functional_doc(functional)
    return data


@simstack_model
class TurbomoleDispersionCorrection(EmbeddedModel):
    """Dispersion correction for TURBOMOLE jobs, including non-DFT methods."""

    field_name: str = "DispersionCorrection"
    value: DispersionCorrectionEnum = Field(
        default=DispersionCorrectionEnum.NONE,
        json_schema_extra={
            "enum": [item.value for item in DispersionCorrectionEnum],
            "description": "Version of the dispersion correction to use",
            "title": "Dispersion Correction",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            if data in (None, ""):
                return {"field_name": cls.__name__, "value": DispersionCorrectionEnum.NONE}
            return {"field_name": cls.__name__, "value": data}
        data.pop("id", None)
        data.pop("_id", None)
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        schema["description"] = "Parameters for dispersion corrections"
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        return ui_schema


@simstack_model
class TurbomoleBasisSet2(EmbeddedModel):
    """Turbomole basis-set payload without legacy aux-basis UI surface."""

    field_name: str = "TurbomoleBasisSet2"
    basis_set: str = Field(
        TURBOMOLE_DEFAULT_BASIS_SET,
        json_schema_extra={"enum": TURBOMOLE_BASIS_SET_VALUES},
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            if data in (None, ""):
                return {"field_name": cls.__name__, "basis_set": TURBOMOLE_DEFAULT_BASIS_SET}
            return {"field_name": cls.__name__, "basis_set": str(data)}
        data.pop("id", None)
        data.pop("_id", None)
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        return ui_schema


@simstack_model
class TurbomoleQMInput2(Model):
    """
    TURBOMOLE input for turbomole2.

    Functional is a TurbomoleFunctional; DispersionCorrection is a required
    sibling field (use NONE to turn it off). Solvent and hyperpolarizability
    extra fields are hidden in the UI until the matching mode is selected;
    stored values are not rewritten.
    """

    field_name: str = "TurbomoleQMInput2"
    model_config = {"extra": "forbid"}

    molecule: Molecule = Reference()
    name: str = Field("Title", json_schema_extra={"description": "name of the calculation"})
    method: TurbomoleMethodEnum = Field(
        TurbomoleMethodEnum.DFT,
        json_schema_extra={
            "enum": [e.value for e in TurbomoleMethodEnum],
            "title": "Method",
            "description": (
                "Electronic-structure method. DFT uses turbomole2 (ridft/jobex). "
                "HF, MP2, CC2, and ADC(2) use turbomole_ricc2 (dscf/ricc2)."
            ),
        },
    )
    charge: int = Field(0, json_schema_extra={"description": "net charge of the molecule"})
    states: int = Field(
        0, json_schema_extra={"description": "number of states to calculate, zero for ground state only"}
    )
    focus_state: int = Field(1, json_schema_extra={"description": "state of focus"})
    multiplicity: int = Field(1, json_schema_extra={"description": "singlet,triplet,....."})
    gridsize: str = Field(
        TURBOMOLE_DEFAULT_GRID_SIZE,
        json_schema_extra={
            "enum": TURBOMOLE_GRID_SIZE_VALUES,
            "description": f"TURBOMOLE DFT integration grid size. Default: {TURBOMOLE_DEFAULT_GRID_SIZE}.",
            "title": "gridsize",
        },
    )
    scfconv: int = Field(
        TURBOMOLE_DEFAULT_SCFCONV,
        json_schema_extra={
            "description": (
                f"TURBOMOLE SCF convergence threshold exponent ($scfconv). "
                f"Default: {TURBOMOLE_DEFAULT_SCFCONV}."
            ),
            "title": "scfconv",
        },
    )
    scfiterlimit: int = Field(
        TURBOMOLE_DEFAULT_SCFITERLIMIT,
        ge=1,
        json_schema_extra={
            "description": (
                f"Maximum number of SCF iterations ($scfiterlimit). "
                f"Default: {TURBOMOLE_DEFAULT_SCFITERLIMIT}."
            ),
            "title": "scfiterlimit",
        },
    )
    open_shell_calculation: bool = Field(
        False, json_schema_extra={"description": "Open shell calculation"}
    )
    basis_set: TurbomoleBasisSet2 = Field(default_factory=TurbomoleBasisSet2)
    functional: TurbomoleFunctional = Field(default_factory=TurbomoleFunctional)
    dispersion_correction: TurbomoleDispersionCorrection = Field(
        default_factory=TurbomoleDispersionCorrection,
        json_schema_extra={
            "description": (
                "Dispersion correction for the calculation. Independent of the "
                "density functional so HF, MP2, and other non-DFT jobs can set it. "
                "Use NONE to omit a correction."
            ),
            "title": "Dispersion Correction",
        },
    )
    gradients: bool = Field(
        False, json_schema_extra={"description": "Calculate gradients (forces) for the molecule"}
    )
    optimization: bool = Field(
        False, json_schema_extra={"description": "Perform geometry optimization"}
    )
    max_opt_cycles: int = Field(
        TURBOMOLE_DEFAULT_MAX_OPT_CYCLES,
        ge=1,
        json_schema_extra={
            "description": (
                f"Maximum number of geometry optimization steps (jobex -c). "
                f"Default: {TURBOMOLE_DEFAULT_MAX_OPT_CYCLES}."
            ),
            "title": "max_opt_cycles",
        },
    )
    use_desy: bool = Field(
        False,
        json_schema_extra={
            "description": (
                "Run the define 'desy' step to detect symmetry automatically and "
                "symmetrize the imported geometry."
            )
        },
    )
    frequencies: bool = Field(False, json_schema_extra={"description": "Calculate frequencies"})
    hyperpolarizability: HyperpolarizabilityModeEnum = Field(
        HyperpolarizabilityModeEnum.NONE,
        json_schema_extra={
            "enum": [e.value for e in HyperpolarizabilityModeEnum],
            "title": "Hyperpolarizability",
            "description": (
                "none: skip β. static: $scfinstab hyperpol with no frequency lines. "
                "dynamic: same keyword plus hyperpol_frequency_nm."
            ),
        },
    )
    hyperpol_frequency_nm: float = Field(
        0.0,
        json_schema_extra={
            "description": "Optical wavelength in nm for dynamic β. Ignored for none and static.",
            "title": "Hyperpol Frequency Nm",
        },
    )
    solvent_mode: SolventModeEnum = Field(
        SolventModeEnum.NONE,
        json_schema_extra={
            "enum": [e.value for e in SolventModeEnum],
            "description": (
                "Solvent configuration mode. Use implicit for a named solvent, "
                "explicit for manual continuum parameters, or none for gas phase."
            ),
        },
    )
    solvent: str = "None"
    solvent_epsilon: Optional[float] = Field(
        default=None,
        json_schema_extra={
            "description": "Explicit dielectric constant for continuum solvent calculations."
        },
    )
    solvent_refind: Optional[float] = Field(
        default=None,
        json_schema_extra={"description": "Explicit refractive index for continuum solvent."},
    )
    print_level: int = Field(1, json_schema_extra={"description": "Print level for the calculation, 0-4"})
    control_groups: List[str] = Field(
        default_factory=list,
        json_schema_extra={
            "description": (
                "Extra TURBOMOLE control data groups appended after define. "
                "Each entry must be one '$name …' group (e.g. '$freeze\\n atoms 1-3')."
            )
        },
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            return data
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        return normalize_functional_and_dispersion(data)

    def functional_enum(self) -> TurbomoleFunctionalEnum:
        value = self.functional.functional
        if isinstance(value, TurbomoleFunctionalEnum):
            return value
        return TurbomoleFunctionalEnum.coerce(value)

    def dispersion_enum(self) -> DispersionCorrectionEnum:
        value = self.dispersion_correction.value
        if isinstance(value, DispersionCorrectionEnum):
            return value
        return DispersionCorrectionEnum(value)

    @field_validator("method", mode="before")
    @classmethod
    def validate_method(cls, value: Any) -> TurbomoleMethodEnum:
        if isinstance(value, TurbomoleMethodEnum):
            return value
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError("TURBOMOLE method must not be empty.")
        text = str(value).strip()
        for item in TurbomoleMethodEnum:
            if item.value == text or item.name == text:
                return item
        raise ValueError(f"Unsupported TURBOMOLE method: {value!r}.")

    @field_validator("control_groups")
    @classmethod
    def validate_control_groups(cls, value: List[str]) -> List[str]:
        parse_control_groups(value)
        return value

    def method_enum(self) -> TurbomoleMethodEnum:
        value = self.method
        if isinstance(value, TurbomoleMethodEnum):
            return value
        return TurbomoleMethodEnum(value)

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        properties = schema.setdefault("properties", {})
        properties["functional"] = TurbomoleFunctional.json_schema()
        properties["method"] = {
            "type": "string",
            "enum": [e.value for e in TurbomoleMethodEnum],
            "default": TurbomoleMethodEnum.DFT.value,
            "title": "Method",
            "description": (
                "Electronic-structure method. DFT uses turbomole2 (ridft/jobex). "
                "HF, MP2, CC2, and ADC(2) use turbomole_ricc2 (dscf/ricc2)."
            ),
        }
        properties["hyperpolarizability"] = {
            "type": "string",
            "enum": [e.value for e in HyperpolarizabilityModeEnum],
            "default": HyperpolarizabilityModeEnum.NONE.value,
            "title": "Hyperpolarizability",
            "description": (
                "none: skip β. static: $scfinstab hyperpol with no frequency lines. "
                "dynamic: same keyword plus hyperpol_frequency_nm."
            ),
        }
        for field_name in ("solvent_epsilon", "solvent_refind", "hyperpol_frequency_nm"):
            field_schema = properties.get(field_name)
            if not isinstance(field_schema, dict):
                continue
            if "anyOf" in field_schema:
                field_schema["type"] = "number"
                field_schema.pop("anyOf", None)
        required = schema.setdefault("required", [])
        for name in ("basis_set", "functional", "dispersion_correction"):
            if name not in required:
                required.append(name)
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["ui:order"] = [
            "molecule",
            "states",
            "focus_state",
            "charge",
            "multiplicity",
            "gridsize",
            "scfconv",
            "scfiterlimit",
            "basis_set",
            "method",
            "functional",
            "dispersion_correction",
            "solvent_mode",
            "solvent",
            "solvent_epsilon",
            "solvent_refind",
            "gradients",
            "optimization",
            "max_opt_cycles",
            "use_desy",
            "open_shell_calculation",
            "frequencies",
            "hyperpolarizability",
            "hyperpol_frequency_nm",
            "id",
            "print_level",
            "control_groups",
        ]
        ui_schema.setdefault("ui:options", {})["ui:foldable"] = True
        ui_schema.setdefault("method", {})["ui:widget"] = "select"
        ui_schema.setdefault("functional", {})["ui:condition"] = {
            "method": TurbomoleMethodEnum.DFT.value
        }
        ui_schema.setdefault("gridsize", {})["ui:condition"] = {
            "method": TurbomoleMethodEnum.DFT.value
        }
        ui_schema.setdefault("hyperpolarizability", {})["ui:widget"] = "select"
        ui_schema.setdefault("solvent", {})["ui:condition"] = {
            "solvent_mode": SolventModeEnum.IMPLICIT.value
        }
        ui_schema.setdefault("solvent_epsilon", {})["ui:condition"] = {
            "solvent_mode": SolventModeEnum.EXPLICIT.value
        }
        ui_schema.setdefault("solvent_refind", {})["ui:condition"] = {
            "solvent_mode": SolventModeEnum.EXPLICIT.value
        }
        ui_schema.setdefault("hyperpol_frequency_nm", {})["ui:condition"] = {
            "hyperpolarizability": HyperpolarizabilityModeEnum.DYNAMIC.value
        }
        ui_schema.setdefault("max_opt_cycles", {})["ui:condition"] = {
            "optimization": True
        }
        return ui_schema


def validate_turbomole2_method(qm_input: TurbomoleQMInput2) -> None:
    method = qm_input.method_enum()
    if method != TurbomoleMethodEnum.DFT:
        raise ValueError(
            f"turbomole2 only supports DFT, got {method.value!r}. "
            "Use turbomole_ricc2 for HF, MP2, CC2, and ADC(2)."
        )


def validate_turbomole_ricc2_request(qm_input: TurbomoleQMInput2) -> None:
    method = qm_input.method_enum()
    if method not in RICC2_METHODS:
        raise ValueError(
            f"turbomole_ricc2 does not support method {method.value!r}. "
            "Use turbomole2 for DFT."
        )
    if qm_input.optimization:
        raise ValueError(
            "Geometry optimization is not supported for wavefunction methods in turbomole_ricc2."
        )
    if qm_input.gradients:
        raise ValueError(
            "Gradients are not supported for wavefunction methods in turbomole_ricc2."
        )
    if qm_input.frequencies:
        raise ValueError(
            "Frequency calculations are not supported for wavefunction methods in turbomole_ricc2."
        )
    if qm_input.hyperpolarizability != HyperpolarizabilityModeEnum.NONE:
        raise ValueError(
            "Hyperpolarizability is not supported for wavefunction methods in turbomole_ricc2."
        )
    if qm_input.open_shell_calculation:
        raise ValueError("Open-shell calculations are not supported in turbomole_ricc2.")
    if int(qm_input.multiplicity) != 1:
        raise ValueError("turbomole_ricc2 requires multiplicity=1.")
    states = int(qm_input.states)
    if method == TurbomoleMethodEnum.ADC2:
        if states <= 0:
            raise ValueError("ADC(2) requires states > 0.")
    elif states != 0:
        raise ValueError(f"{method.value} requires states=0 (ground state only).")
