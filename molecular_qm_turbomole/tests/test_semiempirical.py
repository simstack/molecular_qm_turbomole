from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from molecular_qm_models import Atom, Molecule, QMResult
from molecular_qm_turbomole.models.semiempirical import (
    TurbomoleSemiempiricalInput,
    TurbomoleSemiempiricalMethod,
)
from molecular_qm_turbomole.nodes.turbomole_semiempirical import (
    _write_semiempirical_coord,
    _write_semiempirical_define_input,
    turbomole_semiempirical,
)
from simstack.core.definitions import TaskStatus
from simstack.core.simstack_result import SimstackResult


def _water() -> Molecule:
    molecule = Molecule()
    molecule.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.1173]))
    molecule.add_atom(Atom.from_coords("H", [0.0, 0.7572, -0.4692]))
    molecule.add_atom(Atom.from_coords("H", [0.0, -0.7572, -0.4692]))
    return molecule


def test_turbomole_semiempirical_method_coerce():
    assert TurbomoleSemiempiricalMethod.coerce("am1") == TurbomoleSemiempiricalMethod.AM1
    assert TurbomoleSemiempiricalMethod.coerce("AM1") == TurbomoleSemiempiricalMethod.AM1
    assert TurbomoleSemiempiricalMethod.coerce("pm3") == TurbomoleSemiempiricalMethod.PM3
    assert TurbomoleSemiempiricalMethod.coerce("PM3") == TurbomoleSemiempiricalMethod.PM3
    assert TurbomoleSemiempiricalMethod.coerce("mndo") == TurbomoleSemiempiricalMethod.MNDO
    assert TurbomoleSemiempiricalMethod.coerce("MNDO") == TurbomoleSemiempiricalMethod.MNDO
    assert TurbomoleSemiempiricalMethod.coerce("mndod") == TurbomoleSemiempiricalMethod.MNDOD
    assert TurbomoleSemiempiricalMethod.coerce("MNDOD") == TurbomoleSemiempiricalMethod.MNDOD
    assert TurbomoleSemiempiricalMethod.coerce("mndo/d") == TurbomoleSemiempiricalMethod.MNDOD
    assert TurbomoleSemiempiricalMethod.coerce("MNDO-D") == TurbomoleSemiempiricalMethod.MNDOD

    with pytest.raises(ValueError):
        TurbomoleSemiempiricalMethod.coerce("invalid_method")
    with pytest.raises(ValueError):
        TurbomoleSemiempiricalMethod.coerce(None)


def test_turbomole_semiempirical_input_defaults():
    inp = TurbomoleSemiempiricalInput()
    assert inp.method == TurbomoleSemiempiricalMethod.AM1
    assert inp.charge == 0
    assert inp.multiplicity == 1
    assert inp.optimization is True
    assert inp.max_opt_cycles == 100
    assert inp.scfiterlimit == 100
    assert inp.scfconv == 8
    assert inp.gradients is False


def test_turbomole_semiempirical_input_schema():
    schema = TurbomoleSemiempiricalInput.json_schema()
    ui = TurbomoleSemiempiricalInput.ui_schema()
    assert schema["title"] == "TurbomoleSemiempiricalInput"
    assert "method" in schema["properties"]
    assert ui["field_name"]["ui:widget"] == "hidden"


def test_write_semiempirical_coord(tmp_path: Path):
    mol = _water()
    coord_file = tmp_path / "coord"
    _write_semiempirical_coord(mol, str(coord_file))
    content = coord_file.read_text(encoding="utf-8")
    assert "$coord" in content
    assert "o" in content
    assert "h" in content
    assert "$end" in content


def test_write_semiempirical_define_input(tmp_path: Path):
    mol = _water()
    inp = TurbomoleSemiempiricalInput(
        method=TurbomoleSemiempiricalMethod.PM3,
        charge=1,
        multiplicity=2,
        scfconv=7,
        scfiterlimit=150,
    )
    define_file = tmp_path / "define.inp"
    _write_semiempirical_define_input(mol, inp, path=str(define_file), name="test_pm3")
    content = define_file.read_text(encoding="utf-8")
    assert "a coord" in content
    assert "b all pm3" in content
    assert "1" in content  # charge
    assert "conv\n7" in content
    assert "iter\n150" in content
    assert "dft\noff" in content


@pytest.mark.asyncio
async def test_turbomole_semiempirical_node_success(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mol = _water()
    inp = TurbomoleSemiempiricalInput(method=TurbomoleSemiempiricalMethod.AM1)

    node_runner = MagicMock()
    node_runner.enter_scratch = MagicMock()
    node_runner.succeed = MagicMock(return_value=SimstackResult(status=TaskStatus.COMPLETED))

    with patch(
        "molecular_qm_turbomole.nodes.turbomole_semiempirical._run_monitored_subprocess",
        return_value=(True, 1.0, 1.0),
    ), patch(
        "molecular_qm_turbomole.nodes.turbomole_semiempirical._run_optimization_chunks",
        new=AsyncMock(),
    ), patch(
        "molecular_qm_turbomole.nodes.turbomole_semiempirical.TurbomoleOutputParser"
    ) as MockParser, patch(
        "molecular_qm_turbomole.nodes.turbomole_semiempirical._collect_turbomole_restart_files",
        new=AsyncMock(),
    ), patch(
        "molecular_qm_turbomole.nodes.turbomole_semiempirical._collect_turbomole_info_files"
    ):
        parser_inst = MockParser.return_value
        parser_inst.properly_terminated = True
        parser_inst.final_energy = -12.3456
        parser_inst.final_structure = mol
        parser_inst.energies = [-12.3, -12.3456]

        fn = getattr(turbomole_semiempirical, "__wrapped__", turbomole_semiempirical)
        res = await fn(mol, inp, node_runner=node_runner)
        assert res.status == TaskStatus.COMPLETED
        assert node_runner.qm_result is not None
        assert node_runner.qm_result.final_energy == -12.3456
