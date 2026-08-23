from datetime import datetime, timezone

import pytest

from hyperpolarizibility.hyperpolarization_record import HyperPolarizationRecord
from hyperpolarizibility.hyperpolarization_records import (
    _basis_label,
    _display_label,
    _fill_smiles_and_formula,
    _functional_label,
    _wavelength_label,
)
from hyperpolarizibility.workflows import (
    HyperpolarizabilitySettings,
    WorkflowFailure,
    _apply_hyperpolarizability_settings,
    _build_hyperpol_input,
    _build_optimization_input,
    _check_vibrational_frequencies,
    _disable_hyperpolarizability,
    _fail_workflow,
    _optimization_grid_sizes,
    beta_zzz_by_pair,
    child_exception_text,
)
from molecular_qm_models.density_functional import Functional, FunctionalEnum
from molecular_qm_models.dispersion_correction import DispersionCorrectionEnum
from molecular_qm_models.molecule import Atom, Molecule
from molecular_qm_turbomole.lib.env import build_ground_state_script
from molecular_qm_turbomole.lib.output_parser import parse_vibspectrum_file
from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctionalEnum
from molecular_qm_turbomole.models.turbomole_input import (
    TurbomoleDispersionCorrection,
    HyperpolarizabilityModeEnum,
    TurbomoleBasisSet2,
    TurbomoleQMInput2,
)
from simstack.models.simple_table import SimpleTable


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


def test_settings_schema_is_static_or_dynamic_only():
    schema = HyperpolarizabilitySettings.json_schema()
    ui = HyperpolarizabilitySettings.ui_schema()
    assert schema["properties"]["hyperpolarizability"]["enum"] == [
        HyperpolarizabilityModeEnum.STATIC.value,
        HyperpolarizabilityModeEnum.DYNAMIC.value,
    ]
    assert "molecule" not in schema["properties"]
    assert "frequency_tolerance" in schema["properties"]
    assert ui["hyperpol_frequency_nm"]["ui:condition"] == {
        "hyperpolarizability": HyperpolarizabilityModeEnum.DYNAMIC.value
    }


def test_settings_reject_none_and_dynamic_without_wavelength():
    with pytest.raises(ValueError, match="static or dynamic"):
        HyperpolarizabilitySettings(hyperpolarizability=HyperpolarizabilityModeEnum.NONE)
    with pytest.raises(ValueError, match="positive hyperpol_frequency_nm"):
        HyperpolarizabilitySettings(
            hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
            hyperpol_frequency_nm=0.0,
        )


def test_disable_and_apply_hyperpolarizability_settings():
    qm_input = _qm_input(
        hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
        hyperpol_frequency_nm=1064.0,
    )
    _disable_hyperpolarizability(qm_input)
    assert qm_input.hyperpolarizability == HyperpolarizabilityModeEnum.NONE
    assert qm_input.hyperpol_frequency_nm == 0.0
    _apply_hyperpolarizability_settings(
        qm_input,
        HyperpolarizabilitySettings(
            hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
            hyperpol_frequency_nm=800.0,
        ),
    )
    assert qm_input.hyperpolarizability == HyperpolarizabilityModeEnum.DYNAMIC
    assert qm_input.hyperpol_frequency_nm == 800.0
    _apply_hyperpolarizability_settings(
        qm_input,
        HyperpolarizabilitySettings(hyperpolarizability=HyperpolarizabilityModeEnum.STATIC),
    )
    assert qm_input.hyperpolarizability == HyperpolarizabilityModeEnum.STATIC
    assert qm_input.hyperpol_frequency_nm == 0.0


def test_build_optimization_and_hyperpol_inputs():
    base = _qm_input(hyperpolarizability=HyperpolarizabilityModeEnum.NONE, optimization=False)
    opt_input = _build_optimization_input(base)
    assert opt_input.optimization is True
    assert opt_input.frequencies is True
    assert opt_input.hyperpolarizability == HyperpolarizabilityModeEnum.NONE
    assert opt_input.molecule is base.molecule

    optimized = _water()
    hyper_input = _build_hyperpol_input(
        base,
        optimized,
        HyperpolarizabilitySettings(
            hyperpolarizability=HyperpolarizabilityModeEnum.DYNAMIC,
            hyperpol_frequency_nm=1064.0,
        ),
    )
    assert hyper_input.optimization is False
    assert hyper_input.gradients is False
    assert hyper_input.frequencies is False
    assert hyper_input.hyperpolarizability == HyperpolarizabilityModeEnum.DYNAMIC
    assert hyper_input.hyperpol_frequency_nm == 1064.0
    assert hyper_input.molecule is optimized
    assert hyper_input.functional == base.functional
    assert hyper_input.basis_set.basis_set == base.basis_set.basis_set
    assert hyper_input.dispersion_correction.value == base.dispersion_correction.value


def test_optimization_grid_sizes_tries_m5_after_original():
    assert _optimization_grid_sizes("m3") == ["m3", "m5"]
    assert _optimization_grid_sizes("m4") == ["m4", "m5"]
    assert _optimization_grid_sizes("m5") == ["m5"]


def _freq_table(values: list[float]) -> SimpleTable:
    table = SimpleTable(name="Vibrational Frequencies")
    table.add_column("frequency_cm_1", "float")
    for value in values:
        table.add_row({"frequency_cm_1": value})
    return table


def test_frequency_check_accepts_near_zero_translations_and_real_modes():
    result = type("Result", (), {"vibrational_frequencies": _freq_table([0.0] * 6 + [400.0, 1600.0, 3700.0])})()
    ok, failed = _check_vibrational_frequencies(result, 1e-6)
    assert ok
    assert failed == {}


def test_frequency_check_rejects_nonzero_rigid_modes_and_imaginary_modes():
    result = type(
        "Result",
        (),
        {"vibrational_frequencies": _freq_table([12.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 400.0])},
    )()
    ok, failed = _check_vibrational_frequencies(result, 1e-6)
    assert not ok
    assert "1" in failed
    assert "7" in failed


def test_frequency_check_allows_small_imaginary_mode_within_tolerance():
    result = type(
        "Result",
        (),
        {"vibrational_frequencies": _freq_table([0.0] * 6 + [-5.0, 400.0])},
    )()
    ok, _failed = _check_vibrational_frequencies(result, 10.0)
    assert ok


def test_ground_state_script_runs_aoforce_when_frequencies_requested():
    script = build_ground_state_script(
        optimization=True,
        use_ri=True,
        gradients=False,
        frequencies=True,
    )
    assert "jobex -ri" in script
    assert "aoforce > aoforce.out" in script
    assert "test -f vibspectrum" in script
    without = build_ground_state_script(
        optimization=True,
        use_ri=True,
        gradients=False,
        frequencies=False,
    )
    assert "aoforce" not in without


def test_parse_vibspectrum_file(tmp_path):
    path = tmp_path / "vibspectrum"
    path.write_text(
        """$vibrational spectrum
#  mode     symmetry     wave number   IR intensity    selection rules
#                         cm**(-1)        km/mol         IR     RAMAN
     1                       -0.00         0.00000        -       -
     2                       -0.00         0.00000        -       -
     7        a             315.76         0.00003       YES     YES
$end
""",
        encoding="utf-8",
    )
    table = parse_vibspectrum_file(path)
    assert table is not None
    assert [row["frequency_cm_1"] for row in table.row] == [-0.00, -0.00, 315.76]


def test_hyperpolarization_record_dump_doc_accepts_dispersion_correction():
    """Optional[Model] is persistable; Optional[EmbeddedModel] is not."""
    molecule = _water()
    record = HyperPolarizationRecord(
        molecule=molecule,
        functional=TurbomoleFunctionalEnum.B3_LYP,
        dispersion_correction=TurbomoleDispersionCorrection(value=DispersionCorrectionEnum.D3BJ),
        started_at=datetime.now(timezone.utc),
    )
    doc = record.model_dump_doc()
    assert doc["dispersion_correction"]["value"] == DispersionCorrectionEnum.D3BJ.value

    defaulted = HyperPolarizationRecord(
        molecule=molecule,
        started_at=datetime.now(timezone.utc),
    )
    default_doc = defaulted.model_dump_doc()
    assert default_doc["dispersion_correction"]["value"] == DispersionCorrectionEnum.NONE.value
    assert default_doc.get("wavelength") is None

    with_wavelength = HyperPolarizationRecord(
        molecule=molecule,
        started_at=datetime.now(timezone.utc),
        wavelength=1064.0,
    )
    assert with_wavelength.model_dump_doc()["wavelength"] == pytest.approx(1064.0)


def test_hyperpolarization_record_schema_nests_turbomole_functional():
    schema = HyperPolarizationRecord.json_schema()
    ui = HyperPolarizationRecord.ui_schema()
    functional = schema["properties"]["functional"]
    nested = functional.get("properties", {}).get("functional", functional)
    assert nested.get("enum")
    assert "b3-lyp" in nested["enum"]
    assert "cam-b3lyp" in nested["enum"]
    assert "r2scan" in nested["enum"]
    assert ui.get("functional", {}).get("ui:widget") != "select"


def test_functional_label_reads_enum_and_legacy_nested_functional():
    molecule = _water()
    record = HyperPolarizationRecord(
        molecule=molecule,
        functional=TurbomoleFunctionalEnum.CAM_B3LYP,
        basis_set=TurbomoleBasisSet2(basis_set="def2-TZVP"),
        started_at=datetime.now(timezone.utc),
    )
    assert _functional_label(record) == "cam-b3lyp"
    assert _basis_label(record) == "def2-TZVP"
    assert _wavelength_label(record) == "N/A"
    assert _wavelength_label(
        HyperPolarizationRecord(
            molecule=molecule,
            started_at=datetime.now(timezone.utc),
            wavelength=1064.0,
        )
    ) == "1064.0"

    nested = type("LegacyRecord", (), {"functional": Functional(functional=FunctionalEnum.PBE)})()
    assert _functional_label(nested) == "PBE"


def test_display_label_and_fill_missing_smiles_and_formula():
    assert _display_label(None) == "N/A"
    assert _display_label("Error: molecular_qm_util missing") == "N/A"

    molecule = _water()
    molecule.smiles = None
    molecule.formula = None
    assert _fill_smiles_and_formula(molecule)
    assert _display_label(molecule.smiles) != "N/A"
    assert _display_label(molecule.formula) != "N/A"
    assert not _fill_smiles_and_formula(molecule)


def test_child_exception_text_prefers_real_message_over_generic_failed_status():
    generic = RuntimeError(
        "Task task_id: 6a8995311bfa44458b455fe9 node: turbomole2 terminated with status TaskStatus.FAILED"
    )
    generic.__cause__ = RuntimeError("Geometry optimization failed: jobex did not end properly")
    text = child_exception_text(generic, node_name="turbomole2")
    assert "jobex did not end properly" in text
    assert "terminated with status" not in text


def test_fail_workflow_records_error_and_raises():
    record = HyperPolarizationRecord(
        molecule=_water(),
        started_at=datetime.now(timezone.utc),
    )
    runner = type(
        "Runner",
        (),
        {"fail": lambda self, msg: setattr(self, "error_message", msg) or self},
    )()
    with pytest.raises(WorkflowFailure, match="SCF did not converge"):
        _fail_workflow(runner, record, "SCF", "SCF did not converge")
    assert record.success is False
    assert record.error == "SCF"
    assert runner.error_message == "SCF did not converge"


def test_beta_zzz_by_pair_reads_simple_table():
    table = SimpleTable(name="Hyperpolarizability")
    table.add_column("pair", "int")
    table.add_column("beta_zzz_1e30_esu", "float")
    table.add_row({"pair": 1, "beta_zzz_1e30_esu": 0.12})
    table.add_row({"pair": 2, "beta_zzz_1e30_esu": 0.34})
    assert beta_zzz_by_pair(table) == {1: 0.12, 2: 0.34}
