import pytest

from molecular_qm_turbomole.lib.control_utils import (
    append_control_groups,
    parse_control_group,
    parse_control_groups,
)
from molecular_qm_models.density_functional import Functional, FunctionalEnum
from molecular_qm_models.dispersion_correction import DispersionCorrection, DispersionCorrectionEnum
from molecular_qm_models.molecule import Atom, Molecule
from molecular_qm_turbomole.models.turbomole_input import (
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


def test_parse_control_group_accepts_multiline_freeze():
    group = parse_control_group("$freeze\n atoms 1-3\n")
    assert group[0].startswith("$freeze")
    assert "atoms 1-3" in group[1]


def test_parse_control_group_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        parse_control_group("   \n  ")


def test_parse_control_group_rejects_missing_dollar():
    with pytest.raises(ValueError, match="must start with"):
        parse_control_group("freeze\n atoms 1-3")


def test_parse_control_group_rejects_orca_percent_block():
    with pytest.raises(ValueError, match="ORCA"):
        parse_control_group("%tddft\nnroots 5\nend")


def test_parse_control_group_rejects_protected_scfiterlimit():
    with pytest.raises(ValueError, match="protected"):
        parse_control_group("$scfiterlimit 30")


def test_parse_control_group_rejects_multiple_groups_in_one_entry():
    with pytest.raises(ValueError, match="more than one data group"):
        parse_control_group("$freeze\n atoms 1-3\n$disp\n bj")


def test_parse_control_groups_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_control_groups(["$freeze\n atoms 1", "$freeze\n atoms 2"])


def test_model_rejects_invalid_control_groups():
    with pytest.raises(ValueError, match="must start with"):
        _qm_input(control_groups=["not a control group"])


def test_model_accepts_valid_control_groups():
    model = _qm_input(control_groups=["$freeze\n atoms 1-2", "$disp\n bj"])
    assert len(model.control_groups) == 2
    assert "blocks" not in TurbomoleQMInput2.model_fields
    assert "control_groups" in TurbomoleQMInput2.model_fields


def test_append_control_groups_writes_before_end(tmp_path):
    control = tmp_path / "control"
    control.write_text(
        "$title\nwater\n$dft\n functional b3-lyp\n gridsize m3\n$end\n",
        encoding="utf-8",
    )
    append_control_groups(control, ["$freeze\n atoms 1-3", "$disp\n bj"])
    text = control.read_text(encoding="utf-8")
    assert "$freeze" in text
    assert "atoms 1-3" in text
    assert "$disp" in text
    assert text.strip().endswith("$end")
    assert text.index("$freeze") < text.index("$end")
    assert text.index("$disp") < text.index("$end")
