from datetime import datetime
from typing import List, Optional

from odmantic import Field, Model, Reference
from pydantic import model_validator

from molecular_qm_models.molecule import Molecule
from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctional
from molecular_qm_turbomole.models.turbomole_input import (
    DispersionCorrection,
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
    dispersion_correction: DispersionCorrection = Field(default_factory=DispersionCorrection)
    basis_set: TurbomoleBasisSet2 = Field(default_factory=TurbomoleBasisSet2)
    grids_used: List[str] = Field(default_factory=list)
    started_at: datetime
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
