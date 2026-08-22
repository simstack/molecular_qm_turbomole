from bson import ObjectId
import pytest

from molecular_qm_models.molecule import Atom, Molecule
from molecular_qm_turbomole.lib.migrate_qm_input_docs import (
    as_basis_set_model_doc,
    as_dispersion_model_doc,
    build_hyperpol_dataset_from_records,
    delete_hyperpol_runner_datasets,
    is_hyperpol_runner_dataset,
    is_hyperpol_runner_template,
    record_row_name,
    upgrade_hyperpolarization_record_doc,
    upgrade_stored_document,
    upgrade_turbomole_qm_input_doc,
)
from molecular_qm_turbomole.models.turbomole_input import TurbomoleQMInput2


def test_embedded_basis_set_drops_aux_basis():
    upgraded = as_basis_set_model_doc(
        {"field_name": "TurbomoleBasisSet", "basis_set": "def2-TZVP", "aux_basis": {"aux_basis": "NONE"}}
    )
    assert upgraded["field_name"] == "TurbomoleBasisSet2"
    assert upgraded["basis_set"] == "def2-TZVP"
    assert "id" not in upgraded
    assert "aux_basis" not in upgraded


def test_dispersion_doc_strips_model_id():
    existing = ObjectId()
    upgraded = as_dispersion_model_doc(
        {"field_name": "DispersionCorrection", "value": "D3BJ", "id": existing}
    )
    assert "id" not in upgraded
    assert upgraded["value"] == "D3BJ"


def test_legacy_qm_input_is_converted_to_input2_shape():
    molecule_id = ObjectId()
    original = {
        "_id": ObjectId(),
        "field_name": "TurbomoleQMInput",
        "molecule": molecule_id,
        "name": "water",
        "charge": 0,
        "states": 0,
        "focus_state": 1,
        "multiplicity": 1,
        "gridsize": "m3",
        "scfconv": 8,
        "basis_set": {"field_name": "TurbomoleBasisSet", "basis_set": "aug-cc-pVDZ"},
        "functional": {
            "field_name": "Functional",
            "functional": "PBE",
            "dispersion_correction": {"field_name": "DispersionCorrection", "value": "D3"},
        },
        "hyperpolarizability": True,
        "hyperpolarizability_mode": "dynamic",
        "hyperpol_frequency_nm_ui": 1064.0,
        "hyperpol_frequency_nm": 0.0,
        "gw_enabled": False,
        "edelt": 0.005,
        "optimization": True,
    }
    upgraded, changed = upgrade_turbomole_qm_input_doc(original)
    assert changed
    assert upgraded["field_name"] == "TurbomoleQMInput2"
    assert upgraded["molecule"] == molecule_id
    assert upgraded["basis_set"]["basis_set"] == "aug-cc-pVDZ"
    assert "id" not in upgraded["basis_set"]
    assert upgraded["dispersion_correction"]["value"] == "D3"
    assert "id" not in upgraded["dispersion_correction"]
    assert upgraded["functional"] == {
        "field_name": "TurbomoleFunctional",
        "functional": "pbe",
    }
    assert upgraded["hyperpolarizability"] == "dynamic"
    assert upgraded["hyperpol_frequency_nm"] == 1064.0
    assert "gw_enabled" not in upgraded
    assert "edelt" not in upgraded
    assert "hyperpolarizability_mode" not in upgraded
    payload = dict(upgraded)
    payload.pop("_id", None)
    molecule = Molecule()
    molecule.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.1173]))
    payload["molecule"] = molecule
    parsed = TurbomoleQMInput2.model_validate(payload)
    assert parsed.basis_set.basis_set == "aug-cc-pVDZ"
    assert parsed.functional.keyword() == "pbe"
    assert parsed.dispersion_correction.value.value == "D3"
    assert parsed.hyperpolarizability.value == "dynamic"


def test_already_migrated_qm_input2_is_unchanged():
    original = {
        "_id": ObjectId(),
        "field_name": "TurbomoleQMInput2",
        "molecule": ObjectId(),
        "name": "Title",
        "charge": 0,
        "states": 0,
        "focus_state": 1,
        "multiplicity": 1,
        "gridsize": "m3",
        "scfconv": 8,
        "scfiterlimit": 100,
        "open_shell_calculation": False,
        "basis_set": {"field_name": "TurbomoleBasisSet2", "basis_set": "def2-SVP"},
        "functional": {"field_name": "TurbomoleFunctional", "functional": "b3-lyp"},
        "dispersion_correction": {"field_name": "DispersionCorrection", "value": "NONE"},
        "gradients": False,
        "optimization": False,
        "max_opt_cycles": 100,
        "use_desy": False,
        "frequencies": False,
        "hyperpolarizability": "none",
        "hyperpol_frequency_nm": 0.0,
        "solvent_mode": "none",
        "solvent": "None",
        "print_level": 1,
        "control_groups": [],
    }
    upgraded, changed = upgrade_turbomole_qm_input_doc(original)
    assert not changed
    assert upgraded["functional"] == {
        "field_name": "TurbomoleFunctional",
        "functional": "b3-lyp",
    }
    assert "id" not in upgraded["basis_set"]
    assert "id" not in upgraded["dispersion_correction"]


def test_static_bool_hyperpolarizability_without_wavelength():
    original = {
        "_id": ObjectId(),
        "field_name": "TurbomoleQMInput2",
        "molecule": ObjectId(),
        "name": "Title",
        "basis_set": {"field_name": "TurbomoleBasisSet2", "basis_set": "def2-SVP"},
        "functional": {"field_name": "Functional", "functional": "PBE"},
        "active_electrons": 8,
        "active_orbitals": 6,
        "hyperpolarizability": True,
        "hyperpol_frequency_nm": 0.0,
    }
    upgraded, changed = upgrade_turbomole_qm_input_doc(original)
    assert changed
    assert upgraded["hyperpolarizability"] == "static"
    assert upgraded["dispersion_correction"]["value"] == "NONE"
    assert upgraded["functional"] == {
        "field_name": "TurbomoleFunctional",
        "functional": "pbe",
    }
    assert "active_electrons" not in upgraded
    assert "active_orbitals" not in upgraded


def test_hyperpolarization_record_lifts_dispersion_and_nests_functional():
    original = {
        "_id": ObjectId(),
        "field_name": "HyperPolarizationRecord",
        "molecule": ObjectId(),
        "functional": {
            "field_name": "Functional",
            "functional": "CAM-B3LYP",
            "dispersion_correction": {"field_name": "DispersionCorrection", "value": "D4"},
        },
        "basis_set": {"field_name": "TurbomoleBasisSet2", "basis_set": "def2-TZVPP"},
        "grids_used": ["m3"],
        "success": True,
        "hyperpol": {
            "name": "Hyperpolarizability",
            "heading": ["pair", "beta_zzz_1e30_esu"],
            "row": [{"pair": 1, "beta_zzz_1e30_esu": 1.2}],
            "type": ["int", "float"],
        },
    }
    upgraded, changed = upgrade_hyperpolarization_record_doc(original)
    assert changed
    assert upgraded["functional"] == {
        "field_name": "TurbomoleFunctional",
        "functional": "cam-b3lyp",
    }
    assert "id" not in upgraded["basis_set"]
    assert upgraded["dispersion_correction"]["value"] == "D4"
    assert "id" not in upgraded["dispersion_correction"]
    assert isinstance(upgraded["hyperpol"]["id"], ObjectId)
    assert upgraded["hyperpol"]["row"][0]["beta_zzz_1e30_esu"] == 1.2


def test_upgrade_converts_string_embedded_fields():
    original = {
        "_id": ObjectId(),
        "field_name": "HyperPolarizationRecord",
        "molecule": ObjectId(),
        "functional": "B3LYP",
        "basis_set": "def2-TZVP",
        "dispersion_correction": "D3BJ",
        "grids_used": ["m3"],
        "success": True,
    }
    upgraded, changed = upgrade_hyperpolarization_record_doc(original)
    assert changed
    assert upgraded["functional"] == {
        "field_name": "TurbomoleFunctional",
        "functional": "b3-lyp",
    }
    assert upgraded["basis_set"] == {
        "field_name": "TurbomoleBasisSet2",
        "basis_set": "def2-TZVP",
    }
    assert upgraded["dispersion_correction"]["value"] == "D3BJ"


def test_upgrade_qm_input_converts_string_embedded_fields():
    original = {
        "_id": ObjectId(),
        "field_name": "TurbomoleQMInput",
        "molecule": ObjectId(),
        "name": "water",
        "basis_set": "def2-TZVP",
        "functional": "CAM-B3LYP",
        "dispersion_correction": "D3",
        "hyperpolarizability": "none",
        "hyperpol_frequency_nm": 0.0,
    }
    upgraded, changed = upgrade_turbomole_qm_input_doc(original)
    assert changed
    assert upgraded["functional"] == {
        "field_name": "TurbomoleFunctional",
        "functional": "cam-b3lyp",
    }
    assert upgraded["basis_set"] == {
        "field_name": "TurbomoleBasisSet2",
        "basis_set": "def2-TZVP",
    }
    assert upgraded["dispersion_correction"]["value"] == "D3"


def test_upgrade_stored_document_dispatches_by_field_name():
    qm_doc, qm_changed = upgrade_stored_document(
        {
            "_id": ObjectId(),
            "field_name": "TurbomoleQMInput",
            "molecule": ObjectId(),
            "basis_set": {"basis_set": "SVP"},
            "functional": {"functional": "PBE"},
        }
    )
    assert qm_changed
    assert qm_doc["field_name"] == "TurbomoleQMInput2"

    other, other_changed = upgrade_stored_document({"_id": ObjectId(), "field_name": "Molecule"})
    assert other is None
    assert not other_changed


def test_is_hyperpol_runner_dataset_and_template():
    from types import SimpleNamespace

    from hyperpolarizibility.hyperpol_runner import HYPERPOL_DATASET_TYPE

    assert is_hyperpol_runner_dataset(
        SimpleNamespace(metadata=SimpleNamespace(field_name=HYPERPOL_DATASET_TYPE))
    )
    assert not is_hyperpol_runner_dataset(SimpleNamespace(metadata=SimpleNamespace(field_name="other")))
    assert not is_hyperpol_runner_dataset(SimpleNamespace(metadata=None))
    assert is_hyperpol_runner_template(SimpleNamespace(dataset_type=HYPERPOL_DATASET_TYPE))
    assert not is_hyperpol_runner_template(SimpleNamespace(dataset_type="other"))


class _FakeDb:
    def __init__(self, *, datasets=None, templates=None, records=None):
        self.datasets = list(datasets or [])
        self.templates = list(templates or [])
        self.records = list(records or [])
        self.deleted = []
        self.saved = []

    async def find(self, model, *args):
        from hyperpolarizibility.hyperpolarization_record import HyperPolarizationRecord
        from simstack.models.dataset import DataSet
        from simstack.models.dataset_metadata import DataSetMetadataTemplate

        if model is DataSet:
            return list(self.datasets)
        if model is DataSetMetadataTemplate:
            return list(self.templates)
        if model is HyperPolarizationRecord:
            return list(self.records)
        return []

    async def delete(self, obj):
        self.deleted.append(obj)
        if obj in self.datasets:
            self.datasets.remove(obj)
        if obj in self.templates:
            self.templates.remove(obj)

    async def save(self, obj):
        self.saved.append(obj)
        return obj


@pytest.mark.asyncio
async def test_delete_hyperpol_runner_datasets_drops_matching_docs_and_template():
    from types import SimpleNamespace

    from hyperpolarizibility.hyperpol_runner import HYPERPOL_DATASET_TYPE

    keep = SimpleNamespace(metadata=SimpleNamespace(field_name="other"))
    drop_ds = SimpleNamespace(metadata=SimpleNamespace(field_name=HYPERPOL_DATASET_TYPE))
    drop_template = SimpleNamespace(dataset_type=HYPERPOL_DATASET_TYPE)
    keep_template = SimpleNamespace(dataset_type="other")
    db = _FakeDb(
        datasets=[keep, drop_ds],
        templates=[drop_template, keep_template],
    )
    deleted_datasets, deleted_templates = await delete_hyperpol_runner_datasets(db)
    assert deleted_datasets == 1
    assert deleted_templates == 1
    assert drop_ds in db.deleted
    assert drop_template in db.deleted
    assert keep not in db.deleted
    assert keep_template not in db.deleted
    assert db.datasets == [keep]
    assert db.templates == [keep_template]


@pytest.mark.asyncio
async def test_delete_hyperpol_runner_datasets_dry_run_does_not_write():
    from types import SimpleNamespace

    from hyperpolarizibility.hyperpol_runner import HYPERPOL_DATASET_TYPE

    drop_ds = SimpleNamespace(metadata=SimpleNamespace(field_name=HYPERPOL_DATASET_TYPE))
    drop_template = SimpleNamespace(dataset_type=HYPERPOL_DATASET_TYPE)
    db = _FakeDb(datasets=[drop_ds], templates=[drop_template])
    deleted_datasets, deleted_templates = await delete_hyperpol_runner_datasets(db, dry_run=True)
    assert deleted_datasets == 1
    assert deleted_templates == 1
    assert db.deleted == []
    assert db.datasets == [drop_ds]


def test_build_hyperpol_dataset_from_records_uses_formula_section_and_wavelength():
    from datetime import datetime, timezone

    from hyperpolarizibility.hyperpol_runner import HYPERPOL_DATASET_TYPE, HYPERPOL_RECORDS_DATASET_NAME
    from hyperpolarizibility.hyperpolarization_record import HyperPolarizationRecord
    from molecular_qm_models.molecule import Atom, Molecule
    from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctionalEnum
    from molecular_qm_turbomole.models.turbomole_input import TurbomoleBasisSet2
    from simstack.models import FloatData, StringData
    from simstack.models.simple_table import SimpleTable

    molecule = Molecule()
    molecule.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.1173]))
    molecule.add_atom(Atom.from_coords("H", [0.0, 0.7572, -0.4692]))
    molecule.add_atom(Atom.from_coords("H", [0.0, -0.7572, -0.4692]))
    molecule.formula = "H2O"

    table = SimpleTable(name="Hyperpolarizability")
    table.add_column("pair", "int")
    table.add_column("beta_zzz_1e30_esu", "float")
    table.add_row({"pair": 1, "beta_zzz_1e30_esu": 0.5})

    record = HyperPolarizationRecord(
        molecule=molecule,
        functional=TurbomoleFunctionalEnum.CAM_B3LYP,
        basis_set=TurbomoleBasisSet2(basis_set="def2-TZVP"),
        started_at=datetime.now(timezone.utc),
        wavelength=800.0,
        hyperpol=table,
        error=None,
    )

    dataset, added = build_hyperpol_dataset_from_records([record])
    assert added == 1
    assert dataset.field_name == HYPERPOL_RECORDS_DATASET_NAME
    assert dataset.metadata.field_name == HYPERPOL_DATASET_TYPE
    assert dataset.metadata.data["formula"] == "records"
    assert "H2O" in dataset.sections
    row = dataset["H2O"][record_row_name(record.id)]
    assert isinstance(row["functional"], StringData)
    assert row["functional"].value == "cam-b3lyp"
    assert row["basis_set"].value == "def2-TZVP"
    assert isinstance(row["frequency"], FloatData)
    assert row["frequency"].value == pytest.approx(800.0)
    assert row["beta_pair_1_zzz_1e30_esu"].value == pytest.approx(0.5)
    assert row["beta_pair_2_zzz_1e30_esu"].value == pytest.approx(0.0)
    assert row["error"].value == ""
    assert "hyperpolarizability" not in row
    assert set(row) == {
        "basis_set",
        "functional",
        "frequency",
        "beta_pair_1_zzz_1e30_esu",
        "beta_pair_2_zzz_1e30_esu",
        "beta_pair_3_zzz_1e30_esu",
        "error",
    }


def test_build_hyperpol_dataset_from_records_skips_util_missing_section_name():
    from datetime import datetime, timezone

    from hyperpolarizibility.hyperpolarization_record import HyperPolarizationRecord
    from molecular_qm_models.molecule import Atom, Molecule
    from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctionalEnum

    molecule = Molecule()
    molecule.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.1173]))
    molecule.formula = "Error: molecular_qm_util missing"
    molecule.smiles = "Error: molecular_qm_util missing"
    record = HyperPolarizationRecord(
        molecule=molecule,
        functional=TurbomoleFunctionalEnum.PBE,
        started_at=datetime.now(timezone.utc),
    )
    dataset, added = build_hyperpol_dataset_from_records([record])
    assert added == 1
    assert "Error: molecular_qm_util missing" not in dataset.sections
    assert any(not name.lower().startswith("error") for name in dataset.sections)
