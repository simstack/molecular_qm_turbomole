from pathlib import Path

import pytest

from molecular_qm_models.qm_result import QMResult
from molecular_qm_turbomole.lib.output_parser import (
    TurbomoleOutputParser,
    parse_turbomole_orbital_energies,
)


def test_energies_keep_only_the_final_scf_energy(tmp_path: Path):
    (tmp_path / "energy").write_text(
        "$energy      SCF               SCFKIN            SCFPOT\n"
        "    1   -76.001   75.0   -151.001\n"
        "    2   -76.002   75.1   -151.102\n"
        "    3   -76.003   75.2   -151.203\n"
        "$end\n",
        encoding="utf-8",
    )
    parser = TurbomoleOutputParser(str(tmp_path))
    parser.parse()
    assert parser.final_energy == pytest.approx(-76.003)
    assert parser.energies == pytest.approx([-76.003])


def test_excited_state_total_is_added_when_the_header_says_so(tmp_path: Path):
    (tmp_path / "energy").write_text(
        "$energy SCF excitation\n"
        "1 -76.0 75.0 0.20\n"
        "2 -76.1 75.1 0.15\n"
        "$end\n",
        encoding="utf-8",
    )
    parser = TurbomoleOutputParser(str(tmp_path))
    parser.parse()
    assert parser.energies == pytest.approx([-76.1, -75.95])


def test_orbital_energies_set_homo_lumo_in_ev(tmp_path: Path):
    (tmp_path / "control").write_text(
        "$closed shells\n"
        " a1 1-2 (2)\n"
        "$end\n",
        encoding="utf-8",
    )
    (tmp_path / "mos").write_text(
        "$scfmo\n"
        "     1  a1     eigenvalue=-.20000000000000D+02   nsaos=3\n"
        " 0.1D+00\n"
        "     2  a1     eigenvalue=-.10000000000000D+01   nsaos=3\n"
        " 0.2D+00\n"
        "     3  a1     eigenvalue=-.50000000000000D+00   nsaos=3\n"
        " 0.3D+00\n"
        "$end\n",
        encoding="utf-8",
    )
    frame = parse_turbomole_orbital_energies(tmp_path)
    result = QMResult(final_energy=-76.0, energies=[-76.0])
    result.set_values_from_orbital_energies_dataframe(frame)

    assert result.HOMO_value_Hartree == pytest.approx(-1.0)
    assert result.LUMO_value_Hartree == pytest.approx(-0.5)
    assert result.HOMO_value_eV == pytest.approx(-1.0 * 27.211386245988)
    assert result.LUMO_value_eV == pytest.approx(-0.5 * 27.211386245988)
    assert result.HOMO_LUMO_gap_eV == pytest.approx(0.5 * 27.211386245988)
    assert result.orbital_energies_table_eV is not None
    assert [row["orbital_type"] for row in result.orbital_energies_table_eV.row] == [
        "occupied",
        "occupied",
        "virtual",
    ]
    assert result.orbital_energies_hartree is not None
    assert result.orbital_energies_hartree.row[1]["energy"] == pytest.approx(-1.0)


def test_missing_shells_are_not_treated_as_empty_occupation(tmp_path: Path):
    (tmp_path / "control").write_text("$end\n", encoding="utf-8")
    (tmp_path / "mos").write_text(
        "$scfmo\n     1  a     eigenvalue=-.5   nsaos=1\n$end\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="closed shells"):
        parse_turbomole_orbital_energies(tmp_path)
