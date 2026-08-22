import pytest
from pydantic import ValidationError

from molecular_qm_models.density_functional import Functional, FunctionalEnum
from molecular_qm_models.dispersion_correction import DispersionCorrection, DispersionCorrectionEnum
from molecular_qm_models.molecule import Atom, Molecule
from molecular_qm_turbomole.lib.control_utils import replace_control_data_groups
from molecular_qm_turbomole.lib.input_writer import TurbomoleInputWriter
from molecular_qm_turbomole.models.turbomole_input import (
    HyperpolarizabilityModeEnum,
    SolventModeEnum,
    TurbomoleBasisSet2,
    TurbomoleQMInput2,
)


def _water() -> Molecule:
    molecule = Molecule()
    molecule.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.1173]))
    molecule.add_atom(Atom.from_coords("H", [0.0, 0.7572, -0.4692]))
    molecule.add_atom(Atom.from_coords("H", [0.0, -0.7572, -0.4692]))
    return molecule


def _qm_input(**overrides) -> TurbomoleQMInput2:
    payload = {
        "molecule": _water(),
        "functional": Functional(
            functional=FunctionalEnum.B3LYP,
            dispersion_correction=DispersionCorrection(value=DispersionCorrectionEnum.NONE),
        ),
        "basis_set": TurbomoleBasisSet2(basis_set="def2-SVP"),
    }
    payload.update(overrides)
    return TurbomoleQMInput2(**payload)


def test_turbomole_qm_input2_keeps_explicit_values_without_massaging():
    model = _qm_input(
        solvent_mode="none",
        solvent="water",
        solvent_epsilon=None,
        hyperpolarizability=HyperpolarizabilityModeEnum.NONE,
        hyperpol_frequency_nm=1064.0,
    )
    assert model.solvent == "water"
    assert model.solvent_mode.value == "none"
    assert model.hyperpolarizability == HyperpolarizabilityModeEnum.NONE
    assert model.hyperpol_frequency_nm == 1064.0
    assert "edelt" not in TurbomoleQMInput2.model_fields
    assert "gw_enabled" not in TurbomoleQMInput2.model_fields
    assert "gw_settings" not in TurbomoleQMInput2.model_fields
    assert "hyperpolarizability_settings" not in TurbomoleQMInput2.model_fields
    assert "hyperpol_frequency_nm_ui" not in TurbomoleQMInput2.model_fields
    assert "hyperpolarizability_mode" not in TurbomoleQMInput2.model_fields
    assert "turbomole_cosmo" not in TurbomoleQMInput2.model_fields
    assert "blocks" not in TurbomoleQMInput2.model_fields
    assert "control_groups" in TurbomoleQMInput2.model_fields


def test_turbomole_qm_input2_schema_has_no_gw_fields():
    schema = TurbomoleQMInput2.json_schema()
    ui = TurbomoleQMInput2.ui_schema()
    assert "gw_enabled" not in schema["properties"]
    assert "gw_settings" not in schema["properties"]
    assert "hyperpolarizability_settings" not in schema["properties"]
    assert "hyperpol_frequency_nm_ui" not in schema["properties"]
    assert "hyperpolarizability_mode" not in schema["properties"]
    assert "turbomole_cosmo" not in schema["properties"]
    assert "blocks" not in schema["properties"]
    assert "control_groups" in schema["properties"]
    assert "hyperpolarizability" in schema["properties"]
    assert schema["properties"]["hyperpolarizability"]["enum"] == [
        HyperpolarizabilityModeEnum.NONE.value,
        HyperpolarizabilityModeEnum.STATIC.value,
        HyperpolarizabilityModeEnum.DYNAMIC.value,
    ]
    assert "hyperpol_frequency_nm" in schema["properties"]
    assert "edelt" not in schema["properties"]
    assert "edelt" not in ui["ui:order"]
    assert "solvent_epsilon" in schema["properties"]
    assert "solvent_refind" in schema["properties"]
    assert "gw_enabled" not in ui["ui:order"]
    assert "turbomole_cosmo" not in ui["ui:order"]
    assert "control_groups" in ui["ui:order"]
    assert ui["ui:order"].index("optimization") < ui["ui:order"].index("use_desy")
    assert ui["ui:order"].index("hyperpolarizability") < ui["ui:order"].index(
        "hyperpol_frequency_nm"
    )
    assert ui["solvent"]["ui:condition"] == {
        "solvent_mode": SolventModeEnum.IMPLICIT.value
    }
    assert ui["solvent_epsilon"]["ui:condition"] == {
        "solvent_mode": SolventModeEnum.EXPLICIT.value
    }
    assert ui["solvent_refind"]["ui:condition"] == {
        "solvent_mode": SolventModeEnum.EXPLICIT.value
    }
    assert ui["hyperpol_frequency_nm"]["ui:condition"] == {
        "hyperpolarizability": HyperpolarizabilityModeEnum.DYNAMIC.value
    }
    assert "scfiterlimit" in schema["properties"]
    assert "max_opt_cycles" in schema["properties"]
    assert ui["ui:order"].index("scfconv") < ui["ui:order"].index("scfiterlimit")
    assert ui["ui:order"].index("optimization") < ui["ui:order"].index("max_opt_cycles")
    assert ui["ui:order"].index("max_opt_cycles") < ui["ui:order"].index("use_desy")
    assert ui["max_opt_cycles"]["ui:condition"] == {"optimization": True}


def test_boolean_hyperpolarizability_is_rejected():
    with pytest.raises(ValidationError):
        _qm_input(hyperpolarizability=True, hyperpol_frequency_nm=1064.0)


def test_leftover_edelt_payload_is_rejected():
    with pytest.raises(ValidationError):
        _qm_input(edelt=0.005)


def test_input_writer_emits_define_and_coord(tmp_path):
    qm_input = _qm_input(optimization=False, name="water-sp")
    writer = TurbomoleInputWriter(qm_input)
    coord = tmp_path / "coord"
    define = tmp_path / "define.inp"
    writer.write_coord(str(coord))
    writer.write_define_input(str(define))
    coord_text = coord.read_text(encoding="utf-8")
    define_text = define.read_text(encoding="utf-8")
    assert "$coord" in coord_text
    assert "o" in coord_text
    assert "b all def2-SVP" in define_text
    assert "func b3-lyp" in define_text
    assert "grid m3" in define_text
    assert "conv" in define_text
    assert "8" in define_text.split("conv", 1)[1].splitlines()[1]
    assert "100" in define_text.split("iter", 1)[1].splitlines()[1]
    assert "ri" in define_text


def test_input_writer_honors_custom_grid_scfconv_and_scfiterlimit(tmp_path):
    qm_input = _qm_input(gridsize="m4", scfconv=7, scfiterlimit=250)
    define = tmp_path / "define.inp"
    TurbomoleInputWriter(qm_input).write_define_input(str(define))
    define_text = define.read_text(encoding="utf-8")
    assert "grid m4" in define_text
    assert "7" in define_text.split("conv", 1)[1].splitlines()[1]
    assert "250" in define_text.split("iter", 1)[1].splitlines()[1]


def test_scfiterlimit_and_max_opt_cycles_reject_non_positive_values():
    with pytest.raises(ValidationError):
        _qm_input(scfiterlimit=0)
    with pytest.raises(ValidationError):
        _qm_input(max_opt_cycles=0)


def test_replace_control_data_groups_is_idempotent():
    original = "$title\nwater\n$dft\n functional pbe\n gridsize m3\n$end\n"
    group = ["$scfconv 8"]
    once = replace_control_data_groups(original, [group])
    twice = replace_control_data_groups(once, [group])
    assert once == twice
    assert once.count("$scfconv") == 1
