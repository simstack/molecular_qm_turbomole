import pytest

from hyperpolarizibility.hyperpol_runner import (
    HyperpolRunnerModel,
    _hyperpol_dataset_row,
    _molecule_section_name,
    _qm_input_for_combo,
    _settings_from_qm_input,
    _sweep_combos,
)
from molecular_qm_models.density_functional import Functional, FunctionalEnum, FunctionalModel
from molecular_qm_models.dispersion_correction import (
    DispersionCorrection as FunctionalDispersionCorrection,
    DispersionCorrectionEnum,
)
from molecular_qm_models.molecule import Atom, Molecule
from molecular_qm_turbomole.models.turbomole_input import (
    DispersionCorrection,
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
        "functional": FunctionalEnum.B3LYP,
        "dispersion_correction": DispersionCorrection(value=DispersionCorrectionEnum.D3BJ),
        "basis_set": TurbomoleBasisSet2(basis_set="def2-SVP"),
        "solvent_mode": SolventModeEnum.IMPLICIT,
        "solvent": "chloroform",
    }
    payload.update(overrides)
    return TurbomoleQMInput2(**payload)


def test_runner_model_schema_lists_basis_sets_and_functionals():
    schema = HyperpolRunnerModel.json_schema()
    assert "basis_sets" in schema["properties"]
    assert "functionals" in schema["properties"]
    assert schema["properties"]["basis_sets"]["items"]["enum"]
    assert "B3LYP" in schema["properties"]["functionals"]["items"]["enum"]
    assert "dispersion_correction" not in schema["properties"]


def test_runner_model_rejects_unknown_basis_set():
    with pytest.raises(ValueError, match="Unsupported TURBOMOLE basis"):
        HyperpolRunnerModel(functionals=[FunctionalEnum.PBE], basis_sets=["not-a-basis"])


def test_sweep_combos_are_cartesian_and_unique():
    model = HyperpolRunnerModel(
        functionals=[FunctionalEnum.PBE, FunctionalEnum.B3LYP, FunctionalEnum.PBE],
        basis_sets=["def2-SVP", "def2-TZVP", "def2-SVP"],
    )
    combos = _sweep_combos(model)
    assert combos == [
        (FunctionalEnum.PBE, "def2-SVP"),
        (FunctionalEnum.PBE, "def2-TZVP"),
        (FunctionalEnum.B3LYP, "def2-SVP"),
        (FunctionalEnum.B3LYP, "def2-TZVP"),
    ]


def test_sweep_combos_require_nonempty_lists():
    with pytest.raises(ValueError, match="functionals"):
        _sweep_combos(HyperpolRunnerModel(basis_sets=["def2-SVP"]))
    with pytest.raises(ValueError, match="basis_sets"):
        _sweep_combos(HyperpolRunnerModel(functionals=[FunctionalEnum.PBE]))


def test_settings_from_qm_input_use_static_when_none():
    settings = _settings_from_qm_input(_qm_input(hyperpolarizability=HyperpolarizabilityModeEnum.NONE))
    assert settings.hyperpolarizability == HyperpolarizabilityModeEnum.STATIC
    assert settings.hyperpol_frequency_nm == 0.0


def test_settings_from_qm_input_keep_dynamic_wavelength():
    settings = _settings_from_qm_input(
        _qm_input(
            hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
            hyperpol_frequency_nm=1064.0,
        )
    )
    assert settings.hyperpolarizability == HyperpolarizabilityModeEnum.DYNAMIC
    assert settings.hyperpol_frequency_nm == 1064.0


def test_qm_input_for_combo_overrides_basis_and_functional_only():
    base = _qm_input(
        gridsize="m4",
        charge=-1,
        hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
        hyperpol_frequency_nm=800.0,
    )
    combo = _qm_input_for_combo(
        base,
        basis_set="def2-TZVPP",
        functional_enum=FunctionalEnum.PBE0,
    )
    assert combo.basis_set.basis_set == "def2-TZVPP"
    assert combo.functional == FunctionalEnum.PBE0
    assert combo.dispersion_correction.value == DispersionCorrectionEnum.D3BJ
    assert combo.solvent == "chloroform"
    assert combo.solvent_mode == SolventModeEnum.IMPLICIT
    assert combo.gridsize == "m4"
    assert combo.charge == -1
    assert combo.hyperpolarizability == HyperpolarizabilityModeEnum.DYNAMIC
    assert combo.hyperpol_frequency_nm == 800.0
    assert combo.molecule is base.molecule
    assert combo.name == "water_PBE0_def2-TZVPP"


def test_molecule_section_name_uses_formula():
    molecule = _water()
    assert _molecule_section_name(molecule) == "H2O"
    molecule.formula = None
    molecule.smiles = "O"
    assert _molecule_section_name(molecule) == "O"


def test_hyperpol_dataset_row_has_basis_functional_frequency_and_beta_entries():
    table = SimpleTable(name="Hyperpolarizability")
    table.add_column("pair", "int")
    table.add_column("beta_zzz_1e30_esu", "float")
    table.add_row({"pair": 1, "beta_zzz_1e30_esu": 0.12})
    table.add_row({"pair": 2, "beta_zzz_1e30_esu": 0.34})
    functional = Functional(
        functional=FunctionalEnum.CAM_B3LYP,
        dispersion_correction=FunctionalDispersionCorrection(value=DispersionCorrectionEnum.NONE),
    )
    row = _hyperpol_dataset_row(
        basis_set="aug-cc-pVDZ",
        functional=functional,
        frequency_nm=1064.0,
        hyperpol_table=table,
    )
    assert isinstance(row["basis_set"], StringData)
    assert row["basis_set"].value == "aug-cc-pVDZ"
    assert isinstance(row["functional"], FunctionalModel)
    assert row["functional"].functional.functional == FunctionalEnum.CAM_B3LYP
    assert isinstance(row["frequency"], FloatData)
    assert row["frequency"].value == 1064.0
    assert row["hyperpolarizability"] is table
    assert row["beta_pair_1_zzz_1e30_esu"].value == pytest.approx(0.12)
    assert row["beta_pair_2_zzz_1e30_esu"].value == pytest.approx(0.34)
