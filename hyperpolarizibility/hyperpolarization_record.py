from datetime import datetime
from typing import List, Optional

from odmantic import Field, Model, Reference
from pydantic import model_validator

from molecular_qm_models.molecule import Molecule
from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctional
from molecular_qm_turbomole.models.turbomole_input import (
    TurbomoleDispersionCorrection,
    TurbomoleBasisSet2,
    normalize_functional_and_dispersion,
)
from simstack.models import simstack_model
from simstack.models.simple_table import SimpleTable
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema


@simstack_model
class HyperPolarizationRecord(Model):
    field_name: str = "HyperPolarizationRecord"
    molecule: Molecule = Reference()
    functional: TurbomoleFunctional = Field(default_factory=TurbomoleFunctional)
    dispersion_correction: TurbomoleDispersionCorrection = Field(default_factory=TurbomoleDispersionCorrection)
    basis_set: TurbomoleBasisSet2 = Field(default_factory=TurbomoleBasisSet2)
    grids_used: List[str] = Field(default_factory=list)
    started_at: datetime
    wavelength: Optional[float] = Field(
        None,
        json_schema_extra={
            "title": "Wavelength",
            "description": "Optical wavelength in nm for dynamic beta. Empty for static.",
        },
    )
    hyperpol: Optional[SimpleTable] = None
    success: bool = False
    error: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            return data
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        return normalize_functional_and_dispersion(data)

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        properties = schema.setdefault("properties", {})
        properties["functional"] = TurbomoleFunctional.json_schema()
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        return ui_schema


def _normalize_prefixed_method(data: dict, prefix: str) -> None:
    nested = {
        "functional": data.get(f"{prefix}_functional"),
        "dispersion_correction": data.get(f"{prefix}_dispersion_correction"),
    }
    nested = normalize_functional_and_dispersion(nested)
    if nested.get("functional") is not None:
        data[f"{prefix}_functional"] = nested["functional"]
    if nested.get("dispersion_correction") is not None:
        data[f"{prefix}_dispersion_correction"] = nested["dispersion_correction"]


@simstack_model
class HyperPolarizationRecord2(Model):
    """Hyperpolarizability result with separate optimization and response methods."""

    field_name: str = "HyperPolarizationRecord2"
    molecule: Molecule = Reference()
    optimization_functional: TurbomoleFunctional
    optimization_dispersion_correction: TurbomoleDispersionCorrection
    optimization_basis_set: TurbomoleBasisSet2
    hyperpolarizability_functional: TurbomoleFunctional
    hyperpolarizability_dispersion_correction: TurbomoleDispersionCorrection
    hyperpolarizability_basis_set: TurbomoleBasisSet2
    grids_used: List[str] = Field(default_factory=list)
    started_at: datetime
    wavelength: Optional[float] = Field(
        None,
        json_schema_extra={
            "title": "Wavelength",
            "description": "Optical wavelength in nm for dynamic beta. Empty for static.",
        },
    )
    hyperpol: Optional[SimpleTable] = None
    success: bool = False
    error: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            return data
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        _normalize_prefixed_method(data, "optimization")
        _normalize_prefixed_method(data, "hyperpolarizability")
        return data

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        properties = schema.setdefault("properties", {})
        properties["optimization_functional"] = TurbomoleFunctional.json_schema()
        properties["hyperpolarizability_functional"] = TurbomoleFunctional.json_schema()
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        return ui_schema
