import pytest

from molecular_qm_models.molecule import Atom, Molecule
from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctionalEnum
from molecular_qm_turbomole.lib.control_utils import (
    TURBOMOLE_STATIC_WAVELENGTH_NM,
    render_hyperpolarizability_data_group,
    replace_control_data_groups,
)
from molecular_qm_turbomole.lib.env import build_hyperpolarizability_script
from molecular_qm_turbomole.lib.hyperpol import (
    AU_TO_1E30_ESU,
    apply_hyperpolarizability_control,
    hyperpolarizability_requested,
    hyperpolarizability_wavelength_nm,
    parse_hyperpolarizability_table,
    validate_hyperpolarizability_request,
    verify_dynamic_hyperpols_output,
)
from molecular_qm_turbomole.models.turbomole_input import (
    HyperpolarizabilityModeEnum,
    TurbomoleBasisSet2,
    TurbomoleQMInput2,
)

SAMPLE_HYPERPOLS = """\
#
# Electronic dipole hyperpolarizability (length representation):
                           1st pair of frequencies
 Frequencies:            0.1000073587796313E-08     0.1000073587796313E-08
 Frequencies / eV:       0.2721339859667874E-07     0.2721339859667874E-07
 Frequencies / nm:        45560000000.00000          45560000000.00000
 Frequencies / cm^(-1):  0.2194907813544198E-03     0.2194907813544198E-03
 xxx   1.0     yxx  0.0     zxx  0.0
 xyx   0.0     yyx  0.0     zyx  0.0
 xzx   0.0     yzx  0.0     zzx  0.0
 xxy   0.0     yxy  0.0     zxy  0.0
 xyy   0.0     yyy  0.0     zyy  0.0
 xzy   0.0     yzy  0.0     zzy  0.0
 xxz   0.0     yxz  0.0     zxz  0.0
 xyz   0.0     yyz  0.0     zyz  0.0
 xzz   0.0     yzz  0.0     zzz   12.5
                           2nd pair of frequencies
 Frequencies:            0.1000073587796313E-08     0.3504873281538461E-01
 Frequencies / eV:       0.2721339859667874E-07     0.9537249538959102
 Frequencies / nm:        45560000000.00000          1064.000000000000
 Frequencies / cm^(-1):  0.2194907813544198E-03      9398.496240601504
 xxx   2.0     yxx  0.0     zxx  0.0
 xyx   0.0     yyx  0.0     zyx  0.0
 xzx   0.0     yzx  0.0     zzx  0.0
 xxy   0.0     yxy  0.0     zxy  0.0
 xyy   0.0     yyy  0.0     zyy  0.0
 xzy   0.0     yzy  0.0     zzy  0.0
 xxz   0.0     yxz  0.0     zxz  0.0
 xyz   0.0     yyz  0.0     zyz  0.0
 xzz   0.0     yzz  0.0     zzz   34.0
"""


def _water() -> Molecule:
    molecule = Molecule()
    molecule.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.1173]))
    molecule.add_atom(Atom.from_coords("H", [0.0, 0.7572, -0.4692]))
    molecule.add_atom(Atom.from_coords("H", [0.0, -0.7572, -0.4692]))
    return molecule


def _qm_input(**overrides) -> TurbomoleQMInput2:
    payload = {
        "molecule": _water(),
        "functional": TurbomoleFunctionalEnum.B3_LYP,
        "basis_set": TurbomoleBasisSet2(basis_set="def2-SVP"),
    }
    payload.update(overrides)
    return TurbomoleQMInput2(**payload)


def test_none_enum_is_truthy_but_not_requested():
    model = _qm_input(hyperpolarizability=HyperpolarizabilityModeEnum.NONE)
    assert bool(model.hyperpolarizability)
    assert not hyperpolarizability_requested(model)
    assert hyperpolarizability_wavelength_nm(model) == 0.0


def test_static_mode_ignores_leftover_frequency():
    model = _qm_input(
        hyperpolarizability=HyperpolarizabilityModeEnum.STATIC,
        hyperpol_frequency_nm=1064.0,
    )
    assert hyperpolarizability_requested(model)
    assert hyperpolarizability_wavelength_nm(model) == 0.0
    group = render_hyperpolarizability_data_group(hyperpolarizability_wavelength_nm(model))
    assert group == ["$scfinstab hyperpol nm"]


def test_dynamic_mode_writes_optical_and_dc_wavelengths():
    model = _qm_input(
        hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
        hyperpol_frequency_nm=1064.0,
    )
    assert hyperpolarizability_requested(model)
    assert hyperpolarizability_wavelength_nm(model) == pytest.approx(1064.0)
    group = render_hyperpolarizability_data_group(hyperpolarizability_wavelength_nm(model))
    assert group[0] == "$scfinstab hyperpol nm"
    assert f"{TURBOMOLE_STATIC_WAVELENGTH_NM:.1f}" in group
    assert "1064" in group


def test_dynamic_mode_requires_positive_frequency():
    model = _qm_input(
        hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
        hyperpol_frequency_nm=0.0,
    )
    with pytest.raises(ValueError, match="positive hyperpol_frequency_nm"):
        validate_hyperpolarizability_request(model)


def test_hyperpolarizability_rejects_excited_states():
    model = _qm_input(
        hyperpolarizability=HyperpolarizabilityModeEnum.STATIC,
        states=3,
    )
    with pytest.raises(ValueError, match="excited states"):
        validate_hyperpolarizability_request(model)


def test_apply_hyperpolarizability_control_patches_after_define(tmp_path):
    control = tmp_path / "control"
    control.write_text("$title\nwater\n$dft\n functional b3-lyp\n$end\n", encoding="utf-8")
    model = _qm_input(
        hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
        hyperpol_frequency_nm=532.0,
    )
    apply_hyperpolarizability_control(model, control)
    text = control.read_text(encoding="utf-8")
    assert text.count("$scfinstab hyperpol nm") == 1
    assert "532" in text
    assert "45560000000.0" in text
    assert text.strip().endswith("$end")


def test_replace_control_data_groups_is_idempotent_for_hyperpol():
    original = "$title\nwater\n$end\n"
    group = render_hyperpolarizability_data_group(1064.0)
    once = replace_control_data_groups(original, [group])
    twice = replace_control_data_groups(once, [group])
    assert once == twice
    assert once.count("$scfinstab") == 1


def test_hyperpolarizability_script_runs_escf_and_requires_hyperpols():
    script = build_hyperpolarizability_script()
    assert "escf > escf.out" in script
    assert "test -f hyperpols" in script


def test_parse_and_verify_dynamic_hyperpols(tmp_path):
    hyperpols = tmp_path / "hyperpols"
    hyperpols.write_text(SAMPLE_HYPERPOLS, encoding="utf-8")
    model = _qm_input(
        hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
        hyperpol_frequency_nm=1064.0,
    )
    pairs = verify_dynamic_hyperpols_output(model, hyperpols)
    assert any(abs(right - 1064.0) < 1e-6 for _left, right in pairs)
    control = tmp_path / "control"
    control.write_text(
        "$dipole from ridft\n  x    0.0    y    0.0    z    1.0    a.u.\n$end\n",
        encoding="utf-8",
    )
    table = parse_hyperpolarizability_table(hyperpols, workdir=tmp_path)
    assert table is not None
    assert table.heading == ["pair", "beta_zzz_1e30_esu"]
    assert [row["pair"] for row in table.row] == [1, 2]
    assert table.row[0]["beta_zzz_1e30_esu"] == pytest.approx(12.5 * AU_TO_1E30_ESU)
    assert table.row[1]["beta_zzz_1e30_esu"] == pytest.approx(34.0 * AU_TO_1E30_ESU)
    assert not (tmp_path / "hyperpol_processed.txt").exists()
    assert not (tmp_path / "hyperpol_processed.json").exists()


def test_verify_dynamic_hyperpols_rejects_static_only(tmp_path):
    hyperpols = tmp_path / "hyperpols"
    hyperpols.write_text(
        "1st pair of frequencies\n Frequencies / nm:  0.0  0.0\n zzz  1.0\n",
        encoding="utf-8",
    )
    model = _qm_input(
        hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
        hyperpol_frequency_nm=1064.0,
    )
    with pytest.raises(RuntimeError, match="only zero-frequency"):
        verify_dynamic_hyperpols_output(model, hyperpols)
