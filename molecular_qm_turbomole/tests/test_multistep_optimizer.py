from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from molecular_qm_models import (
    Atom,
    BasisSet,
    Functional,
    GridType,
    Molecule,
    OptimizationAccuracy,
    QMInput,
    QMResult,
    SCFAccuracy,
)
from molecular_qm_turbomole.models.multistep_optimizer import (
    OptimizationStepInput,
    PreOptimizerInput,
    _coerce_grid_string,
)
from molecular_qm_turbomole.models.semiempirical import (
    TurbomoleSemiempiricalInput,
    TurbomoleSemiempiricalMethod,
)
from molecular_qm_turbomole.models.turbomole_functional import (
    TurbomoleFunctional,
    TurbomoleFunctionalEnum,
)
from molecular_qm_turbomole.models.turbomole_input import (
    TurbomoleBasisSet2,
    TurbomoleDispersionCorrection,
    TurbomoleQMInput2,
)
from molecular_qm_turbomole.nodes.multistep_optimizer import (
    _append_step_row,
    _child_qm_result,
    _molecule_from_qm_result,
    _qm_input_for_step,
    _scfconv_from_accuracy,
    _timings_from_child_result,
    multistep_optimizer,
)
from simstack.core.definitions import TaskStatus
from simstack.core.simstack_result import SimstackResult
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType


def _water() -> Molecule:
    molecule = Molecule()
    molecule.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.1173]))
    molecule.add_atom(Atom.from_coords("H", [0.0, 0.7572, -0.4692]))
    molecule.add_atom(Atom.from_coords("H", [0.0, -0.7572, -0.4692]))
    return molecule


def _qm_input(**overrides) -> TurbomoleQMInput2:
    payload = {
        "molecule": _water(),
        "functional": TurbomoleFunctional(functional=TurbomoleFunctionalEnum.B3_LYP),
        "basis_set": TurbomoleBasisSet2(basis_set="def2-SVP"),
        "max_opt_cycles": 42,
        "scfiterlimit": 75,
        "scfconv": 8,
    }
    payload.update(overrides)
    return TurbomoleQMInput2(**payload)


def test_coerce_grid_string():
    assert _coerce_grid_string("m3") == "m3"
    assert _coerce_grid_string("M4") == "m4"
    assert _coerce_grid_string("grid2") == "m3"
    assert _coerce_grid_string(GridType.Grid3) == "m4"
    assert _coerce_grid_string(None) == "m3"


def test_scfconv_from_accuracy():
    assert _scfconv_from_accuracy(SCFAccuracy.Loose) == 6
    assert _scfconv_from_accuracy(SCFAccuracy.Medium) == 7
    assert _scfconv_from_accuracy(SCFAccuracy.Tight) == 8
    assert _scfconv_from_accuracy(SCFAccuracy.VeryTight) == 9
    assert _scfconv_from_accuracy(optimization_accuracy=OptimizationAccuracy.Loose) == 6
    assert _scfconv_from_accuracy(optimization_accuracy=OptimizationAccuracy.VeryTight) == 9
    assert _scfconv_from_accuracy(default_conv=8) == 8


def test_preoptimizer_input_defaults_and_toggle():
    preopt = PreOptimizerInput()
    assert preopt.semiempirical_opt is False
    assert preopt.semiempirical_method is None
    assert preopt.semiempirical_input is None
    assert preopt.steps == []

    # method selection enables semiempirical_opt
    preopt_method = PreOptimizerInput(semiempirical_method=TurbomoleSemiempiricalMethod.AM1)
    assert preopt_method.semiempirical_opt is True
    assert preopt_method.semiempirical_method == TurbomoleSemiempiricalMethod.AM1
    assert preopt_method.semiempirical_input is not None
    assert preopt_method.semiempirical_input.method == TurbomoleSemiempiricalMethod.AM1

    # setting method to None or "none"
    preopt_none = PreOptimizerInput(semiempirical_method="none")
    assert preopt_none.semiempirical_opt is False
    assert preopt_none.semiempirical_method is None


def test_optimization_step_input_normalization():
    step = OptimizationStepInput(
        basis_set="def2-TZVP",
        functional=TurbomoleFunctionalEnum.PBE,
        grid="m4",
        scf_accuracy=SCFAccuracy.Tight,
    )
    assert step.basis_set.basis_set == "def2-TZVP"
    assert step.functional.functional == TurbomoleFunctionalEnum.PBE
    assert step.grid == "m4"


def test_qm_input_for_step_inherits_max_opt_cycles_and_scfiterlimit():
    parent_qm = _qm_input(max_opt_cycles=55, scfiterlimit=120)
    step = OptimizationStepInput(
        basis_set="def2-TZVP",
        functional=TurbomoleFunctionalEnum.PBE0,
        grid="m4",
        scf_accuracy=SCFAccuracy.VeryTight,
    )
    mol = _water()

    step_input = _qm_input_for_step(
        parent_qm,
        step,
        mol,
        frequencies=False,
        max_opt_cycles=55,
        scfiterlimit=120,
    )

    assert step_input.max_opt_cycles == 55
    assert step_input.scfiterlimit == 120
    assert step_input.basis_set.basis_set == "def2-TZVP"
    assert step_input.functional.functional == TurbomoleFunctionalEnum.PBE0
    assert step_input.gridsize == "m4"
    assert step_input.scfconv == 9
    assert step_input.optimization is True
    assert step_input.frequencies is False


def test_qm_input_for_step_sets_frequencies_on_last_step():
    parent_qm = _qm_input(frequencies=True, max_opt_cycles=40, scfiterlimit=80)
    step = OptimizationStepInput(
        basis_set="def2-SVP",
        functional=TurbomoleFunctionalEnum.B3_LYP,
    )
    mol = _water()

    step_input = _qm_input_for_step(
        parent_qm,
        step,
        mol,
        frequencies=True,
        max_opt_cycles=40,
        scfiterlimit=80,
    )
    assert step_input.frequencies is True


@pytest.mark.asyncio
async def test_multistep_optimizer_fails_when_no_steps_and_no_semiempirical():
    parent_qm = _qm_input()
    preopt = PreOptimizerInput(semiempirical_opt=False, steps=[])

    node_runner = MagicMock()
    node_runner.fail = MagicMock(return_value=SimstackResult(status=TaskStatus.FAILED))

    fn = getattr(multistep_optimizer, "__wrapped__", multistep_optimizer)
    res = await fn(parent_qm, preopt, node_runner=node_runner)
    assert res.status == TaskStatus.FAILED
    node_runner.fail.assert_called_once()


@pytest.mark.asyncio
async def test_multistep_optimizer_full_sequence(tmp_path):
    mol_init = _water()
    parent_qm = _qm_input(
        molecule=mol_init,
        max_opt_cycles=88,
        scfiterlimit=150,
        frequencies=True,
    )

    mol_semi = Molecule()
    mol_semi.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.12]))
    mol_semi.add_atom(Atom.from_coords("H", [0.0, 0.75, -0.46]))
    mol_semi.add_atom(Atom.from_coords("H", [0.0, -0.75, -0.46]))

    mol_dft1 = Molecule()
    mol_dft1.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.119]))
    mol_dft1.add_atom(Atom.from_coords("H", [0.0, 0.755, -0.465]))
    mol_dft1.add_atom(Atom.from_coords("H", [0.0, -0.755, -0.465]))

    mol_final = Molecule()
    mol_final.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.118]))
    mol_final.add_atom(Atom.from_coords("H", [0.0, 0.756, -0.468]))
    mol_final.add_atom(Atom.from_coords("H", [0.0, -0.756, -0.468]))

    preopt = PreOptimizerInput(
        semiempirical_method=TurbomoleSemiempiricalMethod.AM1,
        steps=[
            OptimizationStepInput(
                basis_set="def2-SVP",
                functional=TurbomoleFunctionalEnum.B_P,
                grid="m3",
            ),
            OptimizationStepInput(
                basis_set="def2-TZVP",
                functional=TurbomoleFunctionalEnum.B3_LYP,
                grid="m4",
            ),
        ],
    )

    node_runner = MagicMock()
    node_runner.succeed = MagicMock(return_value=SimstackResult(status=TaskStatus.COMPLETED))

    # Mock child node results
    res_semi = SimstackResult(
        status=TaskStatus.COMPLETED,
        result=QMResult(
            molecule=mol_init,
            final_energy=-12.5,
            final_structure=mol_semi,
            properly_terminated=True,
        ),
    )
    timing_table_semi = SimpleTable(name="Optimization timing")
    timing_table_semi.add_column("metric", SimpleTableColumnType.STRING)
    timing_table_semi.add_column("step", SimpleTableColumnType.NUMBER)
    timing_table_semi.add_column("wall_time_s", SimpleTableColumnType.NUMBER)
    timing_table_semi.add_column("cpu_time_s", SimpleTableColumnType.NUMBER)
    timing_table_semi.add_row({"metric": "optimize", "step": None, "wall_time_s": 2.0, "cpu_time_s": 4.0})
    timing_table_semi.add_row({"metric": "iteration", "step": 1, "wall_time_s": 1.0, "cpu_time_s": 2.0})
    timing_table_semi.add_row({"metric": "iteration", "step": 2, "wall_time_s": 1.0, "cpu_time_s": 2.0})
    res_semi.optimization_timing = timing_table_semi

    res_dft1 = SimstackResult(
        status=TaskStatus.COMPLETED,
        result=QMResult(
            molecule=mol_semi,
            final_energy=-76.2,
            final_structure=mol_dft1,
            properly_terminated=True,
        ),
    )
    timing_table_dft1 = SimpleTable(name="Optimization timing")
    timing_table_dft1.add_column("metric", SimpleTableColumnType.STRING)
    timing_table_dft1.add_column("step", SimpleTableColumnType.NUMBER)
    timing_table_dft1.add_column("wall_time_s", SimpleTableColumnType.NUMBER)
    timing_table_dft1.add_column("cpu_time_s", SimpleTableColumnType.NUMBER)
    timing_table_dft1.add_row({"metric": "optimize", "step": None, "wall_time_s": 5.0, "cpu_time_s": 10.0})
    timing_table_dft1.add_row({"metric": "iteration", "step": 1, "wall_time_s": 5.0, "cpu_time_s": 10.0})
    res_dft1.optimization_timing = timing_table_dft1

    res_dft2 = SimstackResult(
        status=TaskStatus.COMPLETED,
        result=QMResult(
            molecule=mol_dft1,
            final_energy=-76.4,
            final_structure=mol_final,
            properly_terminated=True,
        ),
    )
    timing_table_dft2 = SimpleTable(name="Optimization timing")
    timing_table_dft2.add_column("metric", SimpleTableColumnType.STRING)
    timing_table_dft2.add_column("step", SimpleTableColumnType.NUMBER)
    timing_table_dft2.add_column("wall_time_s", SimpleTableColumnType.NUMBER)
    timing_table_dft2.add_column("cpu_time_s", SimpleTableColumnType.NUMBER)
    timing_table_dft2.add_row({"metric": "optimize", "step": None, "wall_time_s": 10.0, "cpu_time_s": 20.0})
    timing_table_dft2.add_row({"metric": "frequencies", "step": None, "wall_time_s": 3.0, "cpu_time_s": 6.0})
    timing_table_dft2.add_row({"metric": "iteration", "step": 1, "wall_time_s": 10.0, "cpu_time_s": 20.0})
    res_dft2.optimization_timing = timing_table_dft2

    with patch(
        "molecular_qm_turbomole.nodes.multistep_optimizer.turbomole_semiempirical",
        new=AsyncMock(return_value=res_semi),
    ) as mock_semi, patch(
        "molecular_qm_turbomole.nodes.multistep_optimizer.turbomole2",
        side_effect=[res_dft1, res_dft2],
    ) as mock_turbomole2:
        fn = getattr(multistep_optimizer, "__wrapped__", multistep_optimizer)
        res = await fn(parent_qm, preopt, node_runner=node_runner)

        assert res.status == TaskStatus.COMPLETED
        assert mock_semi.await_count == 1
        assert mock_turbomole2.await_count == 2

        # Verify max_opt_cycles and scfiterlimit were passed to semiempirical
        semi_call_args = mock_semi.call_args[0]
        semi_opts = semi_call_args[1]
        assert semi_opts.max_opt_cycles == 88
        assert semi_opts.scfiterlimit == 150
        assert semi_opts.method == TurbomoleSemiempiricalMethod.AM1

        # Verify max_opt_cycles and scfiterlimit were passed to DFT steps
        dft_call1 = mock_turbomole2.call_args_list[0][0][0]
        assert dft_call1.max_opt_cycles == 88
        assert dft_call1.scfiterlimit == 150
        assert dft_call1.basis_set.basis_set == "def2-SVP"
        assert dft_call1.functional.functional == TurbomoleFunctionalEnum.B_P
        assert dft_call1.frequencies is False

        dft_call2 = mock_turbomole2.call_args_list[1][0][0]
        assert dft_call2.max_opt_cycles == 88
        assert dft_call2.scfiterlimit == 150
        assert dft_call2.basis_set.basis_set == "def2-TZVP"
        assert dft_call2.functional.functional == TurbomoleFunctionalEnum.B3_LYP
        assert dft_call2.frequencies is True  # frequencies enabled on last step

        # Verify final result attached
        assert node_runner.qm_result.final_energy == -76.4
        assert node_runner.qm_result.final_structure == mol_final

        # Verify step table
        table = node_runner.step_table
        assert table is not None
        rows = table.row
        assert len(rows) == 4  # semiempirical, turbomole-1, turbomole-2, total
        assert rows[0]["step"] == "semiempirical"
        assert rows[0]["energy"] == -12.5
        assert rows[1]["step"] == "turbomole-1"
        assert rows[1]["energy"] == -76.2
        assert rows[2]["step"] == "turbomole-2"
        assert rows[2]["energy"] == -76.4
        assert rows[3]["step"] == "total"
        assert rows[3]["wall_time_s"] == 2.0 + 5.0 + 10.0
        assert rows[3]["cpu_time_s"] == 4.0 + 10.0 + 20.0
        assert rows[3]["freq_wall_time_s"] == 3.0
        assert rows[3]["freq_cpu_time_s"] == 6.0
        assert rows[3]["n_iterations"] == 2 + 1 + 1
