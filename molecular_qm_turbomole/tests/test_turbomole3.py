from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from molecular_qm_models import Atom, Molecule, QMResult
from molecular_qm_turbomole.models.multistep_optimizer import (
    OptimizationStepInput,
    PreOptimizerInput,
)
from molecular_qm_turbomole.models.semiempirical import TurbomoleSemiempiricalMethod
from molecular_qm_turbomole.models.turbomole_functional import (
    TurbomoleFunctional,
    TurbomoleFunctionalEnum,
)
from molecular_qm_turbomole.models.turbomole_input import (
    HyperpolarizabilityModeEnum,
    TurbomoleBasisSet2,
    TurbomoleQMInput2,
)
from molecular_qm_turbomole.nodes.turbomole3 import turbomole3
from simstack.core.definitions import TaskStatus
from simstack.core.simstack_result import SimstackResult
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType


def _water(z: float = 0.1173) -> Molecule:
    molecule = Molecule()
    molecule.add_atom(Atom.from_coords("O", [0.0, 0.0, z]))
    molecule.add_atom(Atom.from_coords("H", [0.0, 0.7572, -0.4692]))
    molecule.add_atom(Atom.from_coords("H", [0.0, -0.7572, -0.4692]))
    return molecule


def _qm_input(**overrides) -> TurbomoleQMInput2:
    payload = {
        "molecule": _water(),
        "functional": TurbomoleFunctional(functional=TurbomoleFunctionalEnum.B3_LYP),
        "basis_set": TurbomoleBasisSet2(basis_set="def2-QZVP"),
        "charge": 1,
        "multiplicity": 1,
        "gridsize": "m5",
        "max_opt_cycles": 40,
        "scfiterlimit": 60,
        "scfconv": 8,
        "optimization": False,
        "frequencies": True,
        "hyperpolarizability": HyperpolarizabilityModeEnum.DYNAMIC,
        "hyperpol_frequency_nm": 1064.0,
    }
    payload.update(overrides)
    return TurbomoleQMInput2(**payload)


def _completed(molecule: Molecule, energy: float, wall_s: float = 1.0) -> SimstackResult:
    result = SimstackResult(
        status=TaskStatus.COMPLETED,
        result=QMResult(
            final_energy=energy,
            final_structure=molecule,
            properly_terminated=True,
        ),
    )
    timing = SimpleTable(name="Optimization timing")
    timing.add_column("metric", SimpleTableColumnType.STRING)
    timing.add_column("step", SimpleTableColumnType.NUMBER)
    timing.add_column("wall_time_s", SimpleTableColumnType.NUMBER)
    timing.add_column("cpu_time_s", SimpleTableColumnType.NUMBER)
    timing.add_row(
        {"metric": "optimize", "step": None, "wall_time_s": wall_s, "cpu_time_s": wall_s * 2}
    )
    result.optimization_timing = timing
    return result


class _Runner:
    def __init__(self):
        self.messages = []
        self.fail = MagicMock(
            return_value=SimstackResult(status=TaskStatus.FAILED, error_message="failed")
        )
        self.succeed = MagicMock(
            return_value=SimstackResult(status=TaskStatus.COMPLETED)
        )

    def info(self, message):
        self.messages.append(str(message))

    def warning(self, message):
        self.messages.append(str(message))


def _runner():
    return _Runner()


@pytest.mark.asyncio
async def test_turbomole3_without_preoptimizer_runs_turbomole2_unchanged():
    parent = _qm_input()
    final_molecule = _water(0.2)
    final = _completed(final_molecule, -76.0, wall_s=4.0)
    energy_file = MagicMock()
    energy_file.name = "energy"
    final.files = [energy_file]
    node_runner = _runner()

    with patch(
        "molecular_qm_turbomole.nodes.turbomole3.turbomole2",
        new=AsyncMock(return_value=final),
    ) as mock_turbomole2, patch(
        "molecular_qm_turbomole.nodes.turbomole3.turbomole_semiempirical",
        new=AsyncMock(),
    ) as mock_semi:
        fn = getattr(turbomole3, "__wrapped__", turbomole3)
        res = await fn(parent, PreOptimizerInput(), node_runner=node_runner)

    assert res.status == TaskStatus.COMPLETED
    mock_semi.assert_not_awaited()
    mock_turbomole2.assert_awaited_once()
    assert mock_turbomole2.await_args.args[0] is parent
    assert node_runner.result.final_energy == -76.0
    assert node_runner.qm_result is node_runner.result
    assert node_runner.files == [energy_file]
    assert not hasattr(node_runner, "step_table")
    node_runner.succeed.assert_called_once()


@pytest.mark.asyncio
async def test_turbomole3_hands_molecule_through_preoptimizer_then_uses_qm_input():
    initial = _water(0.1173)
    parent = _qm_input(molecule=initial)
    mol_semi = _water(0.12)
    mol_dft1 = _water(0.119)
    mol_final_pre = _water(0.118)
    mol_calc = _water(0.1185)
    preopt = PreOptimizerInput(
        semiempirical_method=TurbomoleSemiempiricalMethod.PM3,
        steps=[
            OptimizationStepInput(
                basis_set="def2-SVP",
                functional=TurbomoleFunctionalEnum.B_P,
                grid="m3",
            ),
            OptimizationStepInput(
                basis_set="def2-TZVP",
                functional=TurbomoleFunctionalEnum.PBE,
                grid="m4",
            ),
        ],
    )
    node_runner = _runner()
    res_semi = _completed(mol_semi, -12.0, wall_s=1.0)
    res_dft1 = _completed(mol_dft1, -70.0, wall_s=2.0)
    res_dft2 = _completed(mol_final_pre, -75.0, wall_s=3.0)
    res_final = _completed(mol_calc, -76.5, wall_s=5.0)

    with patch(
        "molecular_qm_turbomole.nodes.turbomole3.turbomole_semiempirical",
        new=AsyncMock(return_value=res_semi),
    ) as mock_semi, patch(
        "molecular_qm_turbomole.nodes.turbomole3.turbomole2",
        new=AsyncMock(side_effect=[res_dft1, res_dft2, res_final]),
    ) as mock_turbomole2:
        fn = getattr(turbomole3, "__wrapped__", turbomole3)
        res = await fn(parent, preopt, node_runner=node_runner)

    assert res.status == TaskStatus.COMPLETED
    assert mock_semi.await_count == 1
    assert mock_turbomole2.await_count == 3

    semi_molecule, semi_opts = mock_semi.await_args.args[:2]
    assert semi_molecule is initial
    assert semi_opts.method == TurbomoleSemiempiricalMethod.PM3
    assert semi_opts.charge == 1
    assert semi_opts.optimization is True
    assert semi_opts.max_opt_cycles == 40
    assert semi_opts.scfiterlimit == 60

    first_step = mock_turbomole2.await_args_list[0].args[0]
    assert first_step.optimization is True
    assert first_step.frequencies is False
    assert first_step.gradients is False
    assert first_step.hyperpolarizability == HyperpolarizabilityModeEnum.NONE
    assert first_step.basis_set.basis_set == "def2-SVP"
    assert first_step.functional.functional == TurbomoleFunctionalEnum.B_P
    assert first_step.gridsize == "m3"
    assert first_step.charge == 1
    assert first_step.max_opt_cycles == 40
    assert first_step.molecule.atoms[0].z == pytest.approx(0.12)

    second_step = mock_turbomole2.await_args_list[1].args[0]
    assert second_step.basis_set.basis_set == "def2-TZVP"
    assert second_step.functional.functional == TurbomoleFunctionalEnum.PBE
    assert second_step.frequencies is False
    assert second_step.hyperpolarizability == HyperpolarizabilityModeEnum.NONE
    assert second_step.molecule.atoms[0].z == pytest.approx(0.119)

    final_step = mock_turbomole2.await_args_list[2].args[0]
    assert final_step is not parent
    assert final_step.molecule.atoms[0].z == pytest.approx(0.118)
    assert final_step.basis_set.basis_set == "def2-QZVP"
    assert final_step.functional.functional == TurbomoleFunctionalEnum.B3_LYP
    assert final_step.gridsize == "m5"
    assert final_step.scfconv == 8
    assert final_step.optimization is False
    assert final_step.frequencies is True
    assert final_step.hyperpolarizability == HyperpolarizabilityModeEnum.DYNAMIC
    assert final_step.hyperpol_frequency_nm == pytest.approx(1064.0)
    assert parent.molecule.atoms[0].z == pytest.approx(0.1173)

    assert node_runner.result.final_energy == -76.5
    assert node_runner.result.final_structure is mol_calc
    rows = node_runner.step_table.row
    assert [row["step"] for row in rows] == [
        "semiempirical",
        "preopt-1",
        "preopt-2",
        "turbomole3",
        "total",
    ]
    assert rows[3]["basis_set"] == "def2-QZVP"
    assert rows[3]["functional"] == TurbomoleFunctionalEnum.B3_LYP.value
    assert rows[3]["energy"] == -76.5
    assert rows[4]["wall_time_s"] == pytest.approx(1.0 + 2.0 + 3.0 + 5.0)


@pytest.mark.asyncio
async def test_turbomole3_failed_preoptimizer_step_stops_the_call():
    parent = _qm_input()
    preopt = PreOptimizerInput(
        steps=[
            OptimizationStepInput(
                basis_set="def2-SVP",
                functional=TurbomoleFunctionalEnum.B_P,
            ),
            OptimizationStepInput(
                basis_set="def2-TZVP",
                functional=TurbomoleFunctionalEnum.PBE,
            ),
        ]
    )
    node_runner = _runner()
    first = _completed(_water(0.12), -70.0)

    with patch(
        "molecular_qm_turbomole.nodes.turbomole3.turbomole2",
        new=AsyncMock(side_effect=[first, RuntimeError("scf did not converge")]),
    ) as mock_turbomole2:
        fn = getattr(turbomole3, "__wrapped__", turbomole3)
        res = await fn(parent, preopt, node_runner=node_runner)

    assert res.status == TaskStatus.FAILED
    assert mock_turbomole2.await_count == 2
    node_runner.fail.assert_called_once()
    message = node_runner.fail.call_args.args[0]
    assert "preopt-2 failed" in message
    assert "scf did not converge" in message
    assert [row["step"] for row in node_runner.step_table.row] == ["preopt-1"]
    node_runner.succeed.assert_not_called()


@pytest.mark.asyncio
async def test_turbomole3_failed_final_calculation_fails_the_call():
    parent = _qm_input()
    preopt = PreOptimizerInput(
        steps=[
            OptimizationStepInput(
                basis_set="def2-SVP",
                functional=TurbomoleFunctionalEnum.B_P,
            )
        ]
    )
    node_runner = _runner()
    first = _completed(_water(0.12), -70.0)

    with patch(
        "molecular_qm_turbomole.nodes.turbomole3.turbomole2",
        new=AsyncMock(side_effect=[first, RuntimeError("jobex failed")]),
    ) as mock_turbomole2:
        fn = getattr(turbomole3, "__wrapped__", turbomole3)
        res = await fn(parent, preopt, node_runner=node_runner)

    assert res.status == TaskStatus.FAILED
    assert mock_turbomole2.await_count == 2
    message = node_runner.fail.call_args.args[0]
    assert message.startswith("turbomole3 failed:")
    assert "jobex failed" in message
    assert not hasattr(node_runner, "result")
    node_runner.succeed.assert_not_called()
