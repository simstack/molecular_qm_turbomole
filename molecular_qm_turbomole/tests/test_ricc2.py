import pytest
from pydantic import ValidationError

from molecular_qm_models.dispersion_correction import DispersionCorrectionEnum
from molecular_qm_models.molecule import Atom, Molecule
from molecular_qm_turbomole.lib.input_writer import TurbomoleInputWriter
from molecular_qm_turbomole.lib.output_parser import (
    parse_ricc2_file,
    require_turbomole_normal_termination,
)
from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctionalEnum
from molecular_qm_turbomole.models.turbomole_input import (
    HyperpolarizabilityModeEnum,
    TurbomoleBasisSet2,
    TurbomoleDispersionCorrection,
    TurbomoleMethodEnum,
    TurbomoleQMInput2,
    validate_turbomole2_method,
    validate_turbomole_ricc2_request,
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
        "functional": TurbomoleFunctionalEnum.B3_LYP,
        "basis_set": TurbomoleBasisSet2(basis_set="def2-SVP"),
        "dispersion_correction": TurbomoleDispersionCorrection(
            value=DispersionCorrectionEnum.NONE
        ),
    }
    payload.update(overrides)
    return TurbomoleQMInput2(**payload)


def test_empty_method_raises_value_error():
    with pytest.raises(ValidationError, match="must not be empty"):
        _qm_input(method="")


def test_unknown_method_raises_value_error():
    with pytest.raises(ValidationError, match="Unsupported TURBOMOLE method"):
        _qm_input(method="CASSCF")


def test_turbomole2_rejects_wavefunction_method():
    with pytest.raises(ValueError, match="turbomole_ricc2"):
        validate_turbomole2_method(_qm_input(method=TurbomoleMethodEnum.ADC2, states=3))


def test_ricc2_rejects_dft_method():
    with pytest.raises(ValueError, match="turbomole2"):
        validate_turbomole_ricc2_request(_qm_input(method=TurbomoleMethodEnum.DFT))


def test_adc2_requires_states():
    with pytest.raises(ValueError, match="states > 0"):
        validate_turbomole_ricc2_request(
            _qm_input(method=TurbomoleMethodEnum.ADC2, states=0)
        )


def test_mp2_rejects_excited_states():
    with pytest.raises(ValueError, match="states=0"):
        validate_turbomole_ricc2_request(
            _qm_input(method=TurbomoleMethodEnum.MP2, states=3)
        )


def test_adc2_rejects_optimization():
    with pytest.raises(ValueError, match="optimization"):
        validate_turbomole_ricc2_request(
            _qm_input(method=TurbomoleMethodEnum.ADC2, states=3, optimization=True)
        )


def test_adc2_rejects_hyperpolarizability():
    with pytest.raises(ValueError, match="Hyperpolarizability"):
        validate_turbomole_ricc2_request(
            _qm_input(
                method=TurbomoleMethodEnum.ADC2,
                states=3,
                hyperpolarizability=HyperpolarizabilityModeEnum.STATIC,
            )
        )


def test_ricc2_rejects_open_shell():
    with pytest.raises(ValueError, match="Open-shell"):
        validate_turbomole_ricc2_request(
            _qm_input(method=TurbomoleMethodEnum.HF, open_shell_calculation=True)
        )


def test_valid_adc2_request_passes():
    validate_turbomole_ricc2_request(
        _qm_input(method=TurbomoleMethodEnum.ADC2, states=2)
    )


def test_adc2_define_turns_dft_off_and_assigns_cbas(tmp_path):
    qm_input = _qm_input(method=TurbomoleMethodEnum.ADC2, states=2, name="water-adc2")
    define = tmp_path / "define.inp"
    TurbomoleInputWriter(qm_input).write_define_input(str(define))
    text = define.read_text(encoding="utf-8")
    assert "dft" in text
    assert "off" in text.split("dft", 1)[1]
    assert "func" not in text
    assert "grid" not in text
    assert "cbas" in text
    assert "ex a" not in text
    assert "soghf" in text
    scf_block = text.split("scf", 1)[1].split("dft", 1)[0]
    assert "off" in scf_block


def test_hf_define_has_no_cbas(tmp_path):
    define = tmp_path / "define.inp"
    TurbomoleInputWriter(_qm_input(method=TurbomoleMethodEnum.HF)).write_define_input(
        str(define)
    )
    text = define.read_text(encoding="utf-8")
    assert "dft" in text
    assert "off" in text.split("dft", 1)[1]
    assert "cbas" not in text
    assert "func" not in text
    assert "soghf" in text.split("scf", 1)[1]


def test_wavefunction_control_groups(tmp_path):
    control = tmp_path / "control"
    control.write_text("$title\nwater\n$end\n", encoding="utf-8")
    mp2_groups = TurbomoleInputWriter(
        _qm_input(method=TurbomoleMethodEnum.MP2)
    ).apply_wavefunction_control(str(control))
    mp2_text = control.read_text(encoding="utf-8")
    assert mp2_groups[0][0] == "$ricc2"
    assert "mp2" in mp2_text
    assert "$excitations" not in mp2_text

    control.write_text("$title\nwater\n$end\n", encoding="utf-8")
    cc2_groups = TurbomoleInputWriter(
        _qm_input(method=TurbomoleMethodEnum.CC2)
    ).apply_wavefunction_control(str(control))
    assert "cc2" in "\n".join(cc2_groups[0])

    control.write_text("$title\nwater\n$end\n", encoding="utf-8")
    adc_groups = TurbomoleInputWriter(
        _qm_input(method=TurbomoleMethodEnum.ADC2, states=4)
    ).apply_wavefunction_control(str(control))
    adc_text = control.read_text(encoding="utf-8")
    assert "adc(2)" in adc_text
    assert "$excitations" in adc_text
    assert "nexc=4" in adc_text
    assert "spectrum states=all operators=diplen" in adc_text
    assert adc_groups[1][0] == "$excitations"

    assert (
        TurbomoleInputWriter(
            _qm_input(method=TurbomoleMethodEnum.HF)
        ).apply_wavefunction_control(str(control))
        == []
    )


def test_parse_ricc2_energy_and_two_excited_states(tmp_path):
    ricc2_out = tmp_path / "ricc2.out"
    ricc2_out.write_text(
        """
      SCF energy                              :    -76.0000000000
      MP2 energy                              :    -76.2280123456
      Final MP2 energy                        :    -76.2280123456

      ADC(2) excitation energies
      number  irrep              excitation energy      oscillator strength
                                    (eV)          (nm)      (length)

           1    a                 7.8451234      158.04        0.054321
           2    a                 9.1234567      135.90        0.012345
""",
        encoding="utf-8",
    )
    energy, table, transitions = parse_ricc2_file(ricc2_out)
    assert energy == pytest.approx(-76.2280123456)
    assert table is not None
    assert transitions is None
    assert len(table.row) == 2
    assert table.row[0]["state"] == 1
    assert table.row[0]["energy_ev"] == pytest.approx(7.8451234)
    assert table.row[0]["oscillator_strength"] == pytest.approx(0.054321)
    assert table.row[1]["state"] == 2
    assert table.row[1]["energy_ev"] == pytest.approx(9.1234567)


def test_parse_ricc2_tm8_excitation_table_and_orbital_occupations(tmp_path):
    ricc2_out = tmp_path / "ricc2.out"
    ricc2_out.write_text(
        """
     *   Final MP2 energy                        :   -383.4404169927      *

   +================================================================================+
   | sym | multi | state |          ADC(2) excitation energies    |  %t1   |  %t2   |
   |     |       |       +----------------------------------------+--------+--------+
   |     |       |       |   Hartree    |    eV      |    cm-1    |    %   |    %   |
   +================================================================================+
   | a   |   1   |   1   |    0.1376098 |    3.74455 |  30201.869 |  91.74 |   8.26 |
   | a   |   1   |   2   |    0.1839143 |    5.00456 |  40364.521 |  90.13 |   9.87 |
   | a   |   1   |   3   |    0.2172040 |    5.91042 |  47670.772 |  91.90 |   8.10 |
   | a   |   1   |   4   |    0.2284356 |    6.21605 |  50135.818 |  85.30 |  14.70 |
   +================================================================================+


       Energy:     0.1376098 H      3.74455 eV    30201.869 cm-1

     +=======================================================================+
     | type: RE0                    symmetry: a               state:    1    |
     +-----------------------+-----------------------+-----------------------+
     | occ. orb.  index spin | vir. orb.  index spin |  coeff/|amp|     %    |
     +=======================+=======================+=======================+
     |   30 a       30       |   33 a       33       |   0.89110      79.4   |
     |   30 a       30       |   37 a       37       |  -0.34333      11.8   |
     |   30 a       30       |   41 a       41       |   0.14924       2.2   |
     |   27 a       27       |   33 a       33       |   0.12298       1.5   |
     |   19 a       19       |   33 a       33       |   0.08070       0.7   |
     +=======================+=======================+=======================+
     norm of printed elements:  0.95585


       Energy:     0.1839143 H      5.00456 eV    40364.521 cm-1

     +=======================================================================+
     | type: RE0                    symmetry: a               state:    2    |
     +-----------------------+-----------------------+-----------------------+
     | occ. orb.  index spin | vir. orb.  index spin |  coeff/|amp|     %    |
     +=======================+=======================+=======================+
     |   32 a       32       |   33 a       33       |  -0.67684      45.8   |
     |   31 a       31       |   34 a       34       |   0.45623      20.8   |
     |   31 a       31       |   33 a       33       |   0.44798      20.1   |
     |   32 a       32       |   34 a       34       |   0.32570      10.6   |
     +=======================+=======================+=======================+
     norm of printed elements:  0.97303


       Energy:     0.2172040 H      5.91042 eV    47670.772 cm-1

     +=======================================================================+
     | type: RE0                    symmetry: a               state:    3    |
     +-----------------------+-----------------------+-----------------------+
     | occ. orb.  index spin | vir. orb.  index spin |  coeff/|amp|     %    |
     +=======================+=======================+=======================+
     |   31 a       31       |   33 a       33       |   0.74428      55.4   |
     |   32 a       32       |   33 a       33       |   0.58230      33.9   |
     |   32 a       32       |   34 a       34       |   0.25469       6.5   |
     +=======================+=======================+=======================+
     norm of printed elements:  0.95789


       Energy:     0.2284356 H      6.21605 eV    50135.818 cm-1

     +=======================================================================+
     | type: RE0                    symmetry: a               state:    4    |
     +-----------------------+-----------------------+-----------------------+
     | occ. orb.  index spin | vir. orb.  index spin |  coeff/|amp|     %    |
     +=======================+=======================+=======================+
     |   30 a       30       |   34 a       34       |   0.98745      97.5   |
     +=======================+=======================+=======================+
     norm of printed elements:  0.97505

   ****  ricc2 : all done  ****
""",
        encoding="utf-8",
    )
    energy, table, transitions = parse_ricc2_file(ricc2_out)
    assert energy == pytest.approx(-383.4404169927)
    assert table is not None
    assert len(table.row) == 4
    assert table.row[0]["state"] == 1
    assert table.row[0]["symmetry"] == "a"
    assert table.row[0]["multiplicity"] == 1
    assert table.row[0]["energy_hartree"] == pytest.approx(0.1376098)
    assert table.row[0]["energy_ev"] == pytest.approx(3.74455)
    assert table.row[0]["energy_cm_1"] == pytest.approx(30201.869)
    assert table.row[0]["percent_t1"] == pytest.approx(91.74)
    assert table.row[3]["energy_ev"] == pytest.approx(6.21605)
    assert transitions is not None
    assert len(transitions.row) == 13
    first = transitions.row[0]
    assert first["state"] == 1
    assert first["symmetry"] == "a"
    assert first["occ_orbital"] == "30 a"
    assert first["occ_index"] == 30
    assert first["vir_orbital"] == "33 a"
    assert first["vir_index"] == 33
    assert first["coefficient"] == pytest.approx(0.89110)
    assert first["percent"] == pytest.approx(79.4)
    assert transitions.row[1]["coefficient"] == pytest.approx(-0.34333)
    assert transitions.row[5]["state"] == 2
    assert transitions.row[5]["occ_orbital"] == "32 a"
    assert transitions.row[5]["percent"] == pytest.approx(45.8)
    assert transitions.row[-1]["state"] == 4
    assert transitions.row[-1]["occ_orbital"] == "30 a"
    assert transitions.row[-1]["vir_orbital"] == "34 a"
    assert transitions.row[-1]["percent"] == pytest.approx(97.5)


def test_wavefunction_control_strips_soghf(tmp_path):
    control = tmp_path / "control"
    control.write_text(
        "$title\nwater\n$soghf\n$coulex\n"
        "$magnetic field\n"
        "Bx = 0.000000000 By = 0.000000000 Bz = 0.000000000\n"
        "$end\n",
        encoding="utf-8",
    )
    groups = TurbomoleInputWriter(
        _qm_input(method=TurbomoleMethodEnum.ADC2, states=2)
    ).apply_wavefunction_control(str(control))
    text = control.read_text(encoding="utf-8")
    assert "$soghf" not in text
    assert "$coulex" not in text
    assert "$magnetic" not in text
    assert "Bx =" not in text
    assert "$ricc2" in text
    assert "adc(2)" in text
    assert groups[0][0] == "$ricc2"

    control.write_text("$title\nwater\n$soghf\n$end\n", encoding="utf-8")
    assert (
        TurbomoleInputWriter(
            _qm_input(method=TurbomoleMethodEnum.HF)
        ).apply_wavefunction_control(str(control))
        == []
    )
    assert "$soghf" not in control.read_text(encoding="utf-8")


def test_dscf_abnormal_termination_is_reported(tmp_path):
    dscf_out = tmp_path / "dscf.out"
    dscf_out.write_text(
        "Program dscf only supports one-component approaches. Use ridft and $coulex.\n"
        "  Option $soghf found!\n"
        " dscf ended abnormally\n"
        " dscf ended abnormally\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"soghf"):
        require_turbomole_normal_termination(dscf_out, "dscf")


def test_parse_ricc2_reports_dscf_abend_instead_of_missing_energy(tmp_path):
    ricc2_out = tmp_path / "ricc2.out"
    ricc2_out.write_text(
        "data group $actual step is not empty\n"
        " due to the abend of dscf\n"
        " ricc2 ended abnormally\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ended abnormally"):
        parse_ricc2_file(ricc2_out)


def test_parse_ricc2_missing_energy_raises(tmp_path):
    ricc2_out = tmp_path / "ricc2.out"
    ricc2_out.write_text("no correlated energy here\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse correlated energy"):
        parse_ricc2_file(ricc2_out)
