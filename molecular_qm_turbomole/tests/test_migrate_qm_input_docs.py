from bson import ObjectId

from molecular_qm_models.molecule import Atom, Molecule
from molecular_qm_turbomole.lib.migrate_qm_input_docs import (
    as_basis_set_model_doc,
    as_dispersion_model_doc,
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
        "active_electrons": 0,
        "active_orbitals": 0,
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
