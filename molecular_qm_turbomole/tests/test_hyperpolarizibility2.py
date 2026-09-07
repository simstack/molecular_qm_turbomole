from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from hyperpolarizibility.hyperpolarizibility2 import (
    _build_hyperpol_method_input,
    dataset_row_from_record2,
)
from hyperpolarizibility.hyperpolarization_record import HyperPolarizationRecord2
from hyperpolarizibility.hyperpolarization_records import _basis_label, _functional_label
from hyperpolarizibility.workflows import HyperpolarizabilitySettings
from molecular_qm_models.dispersion_correction import DispersionCorrectionEnum
from molecular_qm_models.molecule import Atom, Molecule
from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctionalEnum
from molecular_qm_turbomole.models.turbomole_input import (
    TurbomoleDispersionCorrection,
    HyperpolarizabilityModeEnum,
    SolventModeEnum,
    TurbomoleBasisSet2,
    TurbomoleQMInput2,
)
from simstack.models import FloatData, StringData
from simstack.models.simple_table import SimpleTable


def _water() -> Molecule:
    molecule = Molecule()
    molecule.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.1173]))
    molecule.add_atom(Atom.from_coords("H", [0.0, 0.7572, -0.4692]))
    molecule.add_atom(Atom.from_coords("H", [0.0, -0.7572, -0.4692]))
    molecule.formula = "H2O"
    return molecule


def _qm_input(**overrides) -> TurbomoleQMInput2:
    payload = {
        "molecule": _water(),
        "name": "water",
        "functional": TurbomoleFunctionalEnum.B3_LYP,
        "dispersion_correction": TurbomoleDispersionCorrection(value=DispersionCorrectionEnum.D3BJ),
        "basis_set": TurbomoleBasisSet2(basis_set="def2-SVP"),
        "solvent_mode": SolventModeEnum.IMPLICIT,
        "solvent": "chloroform",
        "gridsize": "m3",
    }
    payload.update(overrides)
    return TurbomoleQMInput2(**payload)


def _record2(**overrides) -> HyperPolarizationRecord2:
    payload = {
        "molecule": _water(),
        "optimization_functional": TurbomoleFunctionalEnum.B3_LYP,
        "optimization_dispersion_correction": TurbomoleDispersionCorrection(
            value=DispersionCorrectionEnum.D3BJ
        ),
        "optimization_basis_set": TurbomoleBasisSet2(basis_set="def2-SVP"),
        "hyperpolarizability_functional": TurbomoleFunctionalEnum.CAM_B3LYP,
        "hyperpolarizability_dispersion_correction": TurbomoleDispersionCorrection(
            value=DispersionCorrectionEnum.D3BJ
        ),
        "hyperpolarizability_basis_set": TurbomoleBasisSet2(basis_set="aug-cc-pVDZ"),
        "grids_used": ["m3", "m5"],
        "started_at": datetime.now(timezone.utc),
        "wavelength": 1064.0,
        "success": True,
    }
    payload.update(overrides)
    return HyperPolarizationRecord2(**payload)


def test_record2_requires_optimization_and_hyperpolarizability_methods():
    with pytest.raises(ValidationError):
        HyperPolarizationRecord2(
            molecule=_water(),
            started_at=datetime.now(timezone.utc),
        )


def test_record2_dump_doc_keeps_both_methods():
    record = _record2()
    doc = record.model_dump_doc()
    assert doc["optimization_functional"]["functional"] == "b3-lyp"
    assert doc["optimization_basis_set"]["basis_set"] == "def2-SVP"
    assert doc["optimization_dispersion_correction"]["value"] == DispersionCorrectionEnum.D3BJ.value
    assert doc["hyperpolarizability_functional"]["functional"] == "cam-b3lyp"
    assert doc["hyperpolarizability_basis_set"]["basis_set"] == "aug-cc-pVDZ"
    assert doc["grids_used"] == ["m3", "m5"]
    assert doc["wavelength"] == pytest.approx(1064.0)


def test_record2_schema_nests_both_functionals():
    schema = HyperPolarizationRecord2.json_schema()
    for key in ("optimization_functional", "hyperpolarizability_functional"):
        nested = schema["properties"][key].get("properties", {}).get("functional", schema["properties"][key])
        assert "b3-lyp" in nested["enum"]
        assert "cam-b3lyp" in nested["enum"]


def test_build_hyperpol_method_input_overrides_response_method_only():
    base = _qm_input(
        hyperpolarizability=HyperpolarizabilityModeEnum.NONE,
        optimization=True,
        frequencies=True,
    )
    hyper_input = _build_hyperpol_method_input(
        base,
        _water(),
        HyperpolarizabilitySettings(
            hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
            hyperpol_frequency_nm=1064.0,
        ),
        basis_set="aug-cc-pVDZ",
        functional_enum=TurbomoleFunctionalEnum.CAM_B3LYP,
    )
    assert hyper_input.optimization is False
    assert hyper_input.frequencies is False
    assert hyper_input.gradients is False
    assert hyper_input.functional.functional == TurbomoleFunctionalEnum.CAM_B3LYP
    assert hyper_input.basis_set.basis_set == "aug-cc-pVDZ"
    assert hyper_input.dispersion_correction.value == DispersionCorrectionEnum.D3BJ
    assert hyper_input.solvent == "chloroform"
    assert hyper_input.gridsize == "m3"
    assert hyper_input.hyperpolarizability == HyperpolarizabilityModeEnum.DYNAMIC
    assert hyper_input.hyperpol_frequency_nm == 1064.0
    assert hyper_input.name == "water_hyperpol_cam-b3lyp_aug-cc-pVDZ"
    assert base.functional.functional == TurbomoleFunctionalEnum.B3_LYP
    assert base.basis_set.basis_set == "def2-SVP"


def test_dataset_row_from_record2_includes_optimization_and_hyperpol_methods():
    table = SimpleTable(name="Hyperpolarizability")
    table.add_column("pair", "int")
    table.add_column("beta_zzz_1e30_esu", "float")
    table.add_row({"pair": 1, "beta_zzz_1e30_esu": 0.12})
    row = dataset_row_from_record2(_record2(hyperpol=table))
    assert isinstance(row["optimization_basis_set"], StringData)
    assert row["optimization_basis_set"].value == "def2-SVP"
    assert isinstance(row["optimization_functional"], StringData)
    assert row["optimization_functional"].value == "b3-lyp"
    assert row["basis_set"].value == "aug-cc-pVDZ"
    assert row["functional"].value == "cam-b3lyp"
    assert isinstance(row["frequency"], FloatData)
    assert row["frequency"].value == pytest.approx(1064.0)
    assert row["beta_pair_1_zzz_1e30_esu"].value == pytest.approx(0.12)
    assert row["error"].value == ""


def test_record2_table_labels_use_hyperpolarizability_method():
    record = _record2()
    assert _functional_label(record) == "cam-b3lyp"
    assert _basis_label(record) == "aug-cc-pVDZ"
