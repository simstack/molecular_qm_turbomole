import pytest

from hyperpolarizibility.hyperpol_runner import (
    HYPERPOL_BETA_PAIRS,
    HyperpolRunnerModel,
    _hyperpol_dataset_row,
    _molecule_section_name,
    _qm_input_for_combo,
    _settings_from_qm_input,
    _sweep_combos,
    dataset_row_from_record,
)
from molecular_qm_models.dispersion_correction import DispersionCorrectionEnum
from molecular_qm_models.molecule import Atom, Molecule
from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctionalEnum
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
        "functional": TurbomoleFunctionalEnum.B3_LYP,
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
    assert "b3-lyp" in schema["properties"]["functionals"]["items"]["enum"]
    assert "dispersion_correction" not in schema["properties"]


def test_runner_model_rejects_unknown_basis_set():
    with pytest.raises(ValueError, match="Unsupported TURBOMOLE basis"):
        HyperpolRunnerModel(functionals=[TurbomoleFunctionalEnum.PBE], basis_sets=["not-a-basis"])


def test_sweep_combos_are_cartesian_and_unique():
    model = HyperpolRunnerModel(
        functionals=[TurbomoleFunctionalEnum.PBE, TurbomoleFunctionalEnum.B3_LYP, TurbomoleFunctionalEnum.PBE],
        basis_sets=["def2-SVP", "def2-TZVP", "def2-SVP"],
    )
    combos = _sweep_combos(model)
    assert combos == [
        (TurbomoleFunctionalEnum.PBE, "def2-SVP"),
        (TurbomoleFunctionalEnum.PBE, "def2-TZVP"),
        (TurbomoleFunctionalEnum.B3_LYP, "def2-SVP"),
        (TurbomoleFunctionalEnum.B3_LYP, "def2-TZVP"),
    ]


def test_sweep_combos_require_nonempty_lists():
    with pytest.raises(ValueError, match="functionals"):
        _sweep_combos(HyperpolRunnerModel(basis_sets=["def2-SVP"]))
    with pytest.raises(ValueError, match="basis_sets"):
        _sweep_combos(HyperpolRunnerModel(functionals=[TurbomoleFunctionalEnum.PBE]))


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
        functional_enum=TurbomoleFunctionalEnum.PBE0,
    )
    assert combo.basis_set.basis_set == "def2-TZVPP"
    assert combo.functional.functional == TurbomoleFunctionalEnum.PBE0
    assert combo.dispersion_correction.value == DispersionCorrectionEnum.D3BJ
    assert combo.solvent == "chloroform"
    assert combo.solvent_mode == SolventModeEnum.IMPLICIT
    assert combo.gridsize == "m4"
    assert combo.charge == -1
    assert combo.hyperpolarizability == HyperpolarizabilityModeEnum.DYNAMIC
    assert combo.hyperpol_frequency_nm == 800.0
    assert combo.molecule is base.molecule
    assert combo.name == "water_pbe0_def2-TZVPP"


def test_molecule_section_name_uses_formula():
    molecule = _water()
    assert _molecule_section_name(molecule) == "H2O"


def test_molecule_section_name_retries_util_missing_error():
    molecule = Molecule()
    molecule.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.1173]))
    molecule.add_atom(Atom.from_coords("H", [0.0, 0.7572, -0.4692]))
    molecule.add_atom(Atom.from_coords("H", [0.0, -0.7572, -0.4692]))
    molecule.formula = "Error: molecular_qm_util missing"
    molecule.smiles = "Error: molecular_qm_util missing"
    name = _molecule_section_name(molecule)
    assert not name.lower().startswith("error")
    assert molecule.smiles
    assert molecule.formula
    assert not str(molecule.smiles).lower().startswith("error")
    assert not str(molecule.formula).lower().startswith("error")


def test_fill_molecule_labels_does_not_persist_util_missing():
    from molecular_qm_turbomole.lib.molecule_labels import fill_molecule_labels, molecule_section_name

    class StubMolecule:
        smiles = None
        formula = "Error: molecular_qm_util missing"

        def make_smiles(self):
            return "Error: molecular_qm_util missing"

        def make_formula(self):
            return "Error: molecular_qm_util missing"

    molecule = StubMolecule()
    fill_molecule_labels(molecule)
    assert molecule.smiles is None
    assert molecule.formula == "Error: molecular_qm_util missing"
    assert molecule_section_name(molecule) == "molecule"


def test_hyperpol_dataset_row_has_basis_functional_frequency_and_beta_entries():
    table = SimpleTable(name="Hyperpolarizability")
    table.add_column("pair", "int")
    table.add_column("beta_zzz_1e30_esu", "float")
    table.add_row({"pair": 1, "beta_zzz_1e30_esu": 0.12})
    table.add_row({"pair": 2, "beta_zzz_1e30_esu": 0.34})
    row = _hyperpol_dataset_row(
        basis_set="aug-cc-pVDZ",
        functional=TurbomoleFunctionalEnum.CAM_B3LYP,
        frequency_nm=1064.0,
        hyperpol_table=table,
    )
    assert isinstance(row["basis_set"], StringData)
    assert row["basis_set"].value == "aug-cc-pVDZ"
    assert isinstance(row["functional"], StringData)
    assert row["functional"].value == "cam-b3lyp"
    assert isinstance(row["frequency"], FloatData)
    assert row["frequency"].value == 1064.0
    assert row["hyperpolarizability"] is table
    assert row["beta_pair_1_zzz_1e30_esu"].value == pytest.approx(0.12)
    assert row["beta_pair_2_zzz_1e30_esu"].value == pytest.approx(0.34)
    assert row["beta_pair_3_zzz_1e30_esu"].value == pytest.approx(0.0)
    assert isinstance(row["error"], StringData)
    assert row["error"].value == ""
    for pair in HYPERPOL_BETA_PAIRS:
        field = f"beta_pair_{pair}_zzz_1e30_esu"
        assert isinstance(row[field], FloatData)


def test_hyperpol_dataset_row_always_includes_error_and_beta_pairs():
    row = _hyperpol_dataset_row(
        basis_set="def2-SVP",
        functional=TurbomoleFunctionalEnum.PBE,
        frequency_nm=0.0,
        hyperpol_table=None,
        error="SCF did not converge",
    )
    assert row["error"].value == "SCF did not converge"
    assert row["beta_pair_1_zzz_1e30_esu"].value == pytest.approx(0.0)
    assert row["beta_pair_2_zzz_1e30_esu"].value == pytest.approx(0.0)
    assert row["beta_pair_3_zzz_1e30_esu"].value == pytest.approx(0.0)
    assert row["frequency"].value == pytest.approx(0.0)


def test_dataset_row_from_record_uses_wavelength_and_string_labels():
    from datetime import datetime, timezone

    from hyperpolarizibility.hyperpolarization_record import HyperPolarizationRecord

    table = SimpleTable(name="Hyperpolarizability")
    table.add_column("pair", "int")
    table.add_column("beta_zzz_1e30_esu", "float")
    table.add_row({"pair": 1, "beta_zzz_1e30_esu": 0.12})
    record = HyperPolarizationRecord(
        molecule=_water(),
        functional=TurbomoleFunctionalEnum.CAM_B3LYP,
        basis_set=TurbomoleBasisSet2(basis_set="aug-cc-pVDZ"),
        started_at=datetime.now(timezone.utc),
        wavelength=1064.0,
        hyperpol=table,
        error="NOFREQ",
    )
    row = dataset_row_from_record(record)
    assert isinstance(row["functional"], StringData)
    assert row["functional"].value == "cam-b3lyp"
    assert isinstance(row["basis_set"], StringData)
    assert row["basis_set"].value == "aug-cc-pVDZ"
    assert row["frequency"].value == pytest.approx(1064.0)
    assert row["beta_pair_1_zzz_1e30_esu"].value == pytest.approx(0.12)
    assert row["beta_pair_2_zzz_1e30_esu"].value == pytest.approx(0.0)
    assert row["error"].value == "NOFREQ"


def test_dataset_row_from_record_static_wavelength_is_zero():
    from datetime import datetime, timezone

    from hyperpolarizibility.hyperpolarization_record import HyperPolarizationRecord

    record = HyperPolarizationRecord(
        molecule=_water(),
        functional=TurbomoleFunctionalEnum.PBE,
        started_at=datetime.now(timezone.utc),
        wavelength=None,
    )
    row = dataset_row_from_record(record)
    assert row["frequency"].value == pytest.approx(0.0)
    assert row["error"].value == ""
    assert row["basis_set"].value == "def2-SVP"
