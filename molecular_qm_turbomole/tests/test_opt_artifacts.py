import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from odmantic import ObjectId

from molecular_qm_turbomole.lib.env import build_ground_state_script
from molecular_qm_turbomole.lib.opt_artifacts import (
    OptimizationChartTracker,
    inspect_geometry_optimization,
    persist_opt_charts,
)
from molecular_qm_turbomole.lib.output_parser import parse_energy_history, parse_gradient_history
from molecular_qm_turbomole.nodes.turbomole2 import _run_optimization_chunks, _with_runner_output


def _write_energy(directory, n_cycles: int) -> None:
    lines = ["$energy      SCF               SCFKIN            SCFPOT"]
    for step in range(1, n_cycles + 1):
        lines.append(f"     {step}   {-76.0 - 0.001 * step}    75.0  -151.0")
    lines.append("$end")
    (directory / "energy").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gradient(directory, n_cycles: int, *, with_norm: bool = True) -> None:
    lines = ["$grad        cartesian gradients"]
    for step in range(1, n_cycles + 1):
        if with_norm:
            lines.append(
                f"  cycle =      {step}    SCF energy =   -76.0   |dE/dxyz| =  0.01{step:02d}"
            )
        else:
            lines.append(f"  cycle =      {step}    SCF energy =   -76.0")
        lines.append("    0.00000000000000      0.00000000000000      0.00000000000000      o")
        lines.append("    3.00000000000000      4.00000000000000      0.00000000000000")
    lines.append("$end")
    (directory / "gradient").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_energy_history(tmp_path):
    _write_energy(tmp_path, 3)
    history = parse_energy_history(tmp_path / "energy")
    assert [row["step"] for row in history] == [1, 2, 3]
    assert history[0]["energy"] == pytest.approx(-76.001)
    assert history[-1]["energy"] == pytest.approx(-76.003)


def test_parse_energy_history_missing_file(tmp_path):
    assert parse_energy_history(tmp_path / "energy") == []


def test_parse_gradient_history_from_header_norm(tmp_path):
    _write_gradient(tmp_path, 2, with_norm=True)
    history = parse_gradient_history(tmp_path / "gradient")
    assert [row["step"] for row in history] == [1, 2]
    assert history[0]["grad_norm"] == pytest.approx(0.0101)
    assert history[1]["grad_norm"] == pytest.approx(0.0102)


def test_parse_gradient_history_rms_fallback(tmp_path):
    _write_gradient(tmp_path, 1, with_norm=False)
    history = parse_gradient_history(tmp_path / "gradient")
    assert len(history) == 1
    assert history[0]["step"] == 1
    assert history[0]["grad_norm"] == pytest.approx(math.sqrt((9.0 + 16.0 + 0.0) / 3.0))


def test_parse_gradient_history_fortran_d_exponent(tmp_path):
    (tmp_path / "gradient").write_text(
        "$grad\n"
        "  cycle =      4    SCF energy =   -76.0   |dE/dxyz| =  0.1234D-01\n"
        "$end\n",
        encoding="utf-8",
    )
    history = parse_gradient_history(tmp_path / "gradient")
    assert len(history) == 1
    assert history[0]["step"] == 4
    assert history[0]["grad_norm"] == pytest.approx(0.01234)


def test_inspect_geometry_optimization_converged(tmp_path):
    (tmp_path / "GEO_OPT_CONVERGED").write_text("CONVERGED\n", encoding="utf-8")
    assert inspect_geometry_optimization(tmp_path) == ("converged", None)


def test_inspect_geometry_optimization_continue_on_cycle_limit(tmp_path):
    (tmp_path / "GEO_OPT_FAILED").write_text(
        "OPTIMIZATION DID NOT CONVERGE WITHIN MAXCYCLES\n",
        encoding="utf-8",
    )
    status, error = inspect_geometry_optimization(tmp_path)
    assert status == "continue"
    assert error == "opt not converged"


def test_inspect_geometry_optimization_scf_failure(tmp_path):
    (tmp_path / "GEO_OPT_FAILED").write_text(
        "your energy calculation did not converge\n",
        encoding="utf-8",
    )
    status, error = inspect_geometry_optimization(tmp_path)
    assert status == "failed"
    assert "SCF" in error


def test_inspect_geometry_optimization_running(tmp_path):
    (tmp_path / "GEO_OPT_RUNNING").write_text("running\n", encoding="utf-8")
    status, error = inspect_geometry_optimization(tmp_path)
    assert status == "failed"
    assert "did not end properly" in error


def test_ground_state_script_adds_jobex_cycle_limit():
    script = build_ground_state_script(
        optimization=True,
        use_ri=True,
        gradients=False,
        max_cycles=10,
    )
    assert "jobex -ri -c 10" in script
    assert "aoforce" not in script


@pytest.mark.asyncio
async def test_tracker_flushes_every_ten_steps(tmp_path, monkeypatch):
    flush_lengths = []

    async def fake_persist(energy_data, grad_data, kwargs, existing=(None, None)):
        flush_lengths.append(len(energy_data))
        return ("energy_chart", "grad_chart")

    monkeypatch.setattr(
        "molecular_qm_turbomole.lib.opt_artifacts.persist_opt_charts",
        fake_persist,
    )
    tracker = OptimizationChartTracker({})
    _write_energy(tmp_path, 9)
    _write_gradient(tmp_path, 9)
    tracker.update_from_directory(tmp_path)
    await tracker.maybe_flush()
    assert flush_lengths == []

    _write_energy(tmp_path, 10)
    _write_gradient(tmp_path, 10)
    tracker.update_from_directory(tmp_path)
    await tracker.maybe_flush()
    assert flush_lengths == [10]

    _write_energy(tmp_path, 20)
    _write_gradient(tmp_path, 20)
    tracker.update_from_directory(tmp_path)
    await tracker.maybe_flush()
    assert flush_lengths == [10, 20]


@pytest.mark.asyncio
async def test_tracker_flushes_on_seven_cycle_converged_exit(tmp_path, monkeypatch):
    flush_lengths = []

    async def fake_persist(energy_data, grad_data, kwargs, existing=(None, None)):
        flush_lengths.append(len(energy_data))
        return ("energy_chart", "grad_chart")

    monkeypatch.setattr(
        "molecular_qm_turbomole.lib.opt_artifacts.persist_opt_charts",
        fake_persist,
    )
    tracker = OptimizationChartTracker({})
    _write_energy(tmp_path, 7)
    _write_gradient(tmp_path, 7)
    tracker.update_from_directory(tmp_path)
    await tracker.maybe_flush()
    assert flush_lengths == []
    await tracker.maybe_flush(force=True)
    assert flush_lengths == [7]


@pytest.mark.asyncio
async def test_persist_opt_charts_saves_energy_and_gradient(monkeypatch):
    saved = []

    class FakeDB:
        async def save(self, obj):
            saved.append(obj)
            return obj

    monkeypatch.setattr(
        "molecular_qm_turbomole.lib.opt_artifacts._get_db",
        lambda: FakeDB(),
    )
    task_id = str(ObjectId())
    kwargs = {"task_id": task_id, "node_runner": MagicMock()}
    energy = [{"step": 10, "energy": -76.0}]
    grad = [{"step": 10, "grad_norm": 0.01}]
    charts = await persist_opt_charts(energy, grad, kwargs)
    assert len(saved) == 2
    assert saved[0].title.text == "TURBOMOLE optimization energy"
    assert saved[1].title.text == "TURBOMOLE optimization gradient norm"
    assert saved[0].data == energy
    assert saved[1].data == grad
    assert charts == (saved[0], saved[1])


@pytest.mark.asyncio
async def test_run_optimization_chunks_flushes_at_ten_and_twenty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    flush_steps = []

    async def fake_persist(energy_data, grad_data, kwargs, existing=(None, None)):
        flush_steps.append([row["step"] for row in energy_data])
        return (MagicMock(name="energy_chart"), MagicMock(name="grad_chart"))

    monkeypatch.setattr(
        "molecular_qm_turbomole.lib.opt_artifacts.persist_opt_charts",
        fake_persist,
    )

    calls = {"n": 0}

    def fake_subprocess(name, command, cwd=""):
        calls["n"] += 1
        n_cycles = 10 * calls["n"]
        _write_energy(tmp_path, n_cycles)
        _write_gradient(tmp_path, n_cycles)
        failed = tmp_path / "GEO_OPT_FAILED"
        converged = tmp_path / "GEO_OPT_CONVERGED"
        if calls["n"] == 1:
            failed.write_text("OPTIMIZATION DID NOT CONVERGE\n", encoding="utf-8")
            converged.unlink(missing_ok=True)
        else:
            failed.unlink(missing_ok=True)
            converged.write_text("CONVERGED\n", encoding="utf-8")
        assert f"-c {10}" in command or "jobex" in command
        return True

    node_runner = MagicMock()
    node_runner.subprocess.side_effect = fake_subprocess
    qm_input = SimpleNamespace(
        basis_set=SimpleNamespace(basis_set="def2-SVP"),
        max_opt_cycles=100,
    )
    await _run_optimization_chunks(qm_input, node_runner, {"node_runner": node_runner})
    assert calls["n"] == 2
    assert [steps[-1] for steps in flush_steps] == [10, 20, 20]


@pytest.mark.asyncio
async def test_run_optimization_chunks_flushes_when_energy_has_initial_scf(tmp_path, monkeypatch):
    """jobex -c 10 typically leaves 11 energy records (initial SCF + 10 cycles)."""
    monkeypatch.chdir(tmp_path)
    flush_steps = []

    async def fake_persist(energy_data, grad_data, kwargs, existing=(None, None)):
        flush_steps.append([row["step"] for row in energy_data])
        return (MagicMock(name="energy_chart"), MagicMock(name="grad_chart"))

    monkeypatch.setattr(
        "molecular_qm_turbomole.lib.opt_artifacts.persist_opt_charts",
        fake_persist,
    )

    calls = {"n": 0}

    def fake_subprocess(name, command, cwd=""):
        calls["n"] += 1
        n_energy = 11 if calls["n"] == 1 else 21
        _write_energy(tmp_path, n_energy)
        _write_gradient(tmp_path, n_energy - 1)
        failed = tmp_path / "GEO_OPT_FAILED"
        converged = tmp_path / "GEO_OPT_CONVERGED"
        if calls["n"] == 1:
            failed.write_text("OPTIMIZATION DID NOT CONVERGE\n", encoding="utf-8")
            converged.unlink(missing_ok=True)
        else:
            failed.unlink(missing_ok=True)
            converged.write_text("CONVERGED\n", encoding="utf-8")
        return True

    node_runner = MagicMock()
    node_runner.subprocess.side_effect = fake_subprocess
    qm_input = SimpleNamespace(
        basis_set=SimpleNamespace(basis_set="def2-SVP"),
        max_opt_cycles=100,
    )
    await _run_optimization_chunks(qm_input, node_runner, {"node_runner": node_runner})
    assert calls["n"] == 2
    assert flush_steps[0][-1] == 11
    assert all(steps[-1] == 21 for steps in flush_steps[1:])
    infos = [call.args[0] for call in node_runner.info.call_args_list]
    assert any("cycles 1-10" in msg for msg in infos)
    assert any("cycles 11-20" in msg for msg in infos)
    assert not any("cycles 12-21" in msg for msg in infos)
    assert any("Wrote optimization chart artifacts after jobex cycles 1-10" in msg for msg in infos)


@pytest.mark.asyncio
async def test_run_optimization_chunks_flushes_on_early_convergence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    flush_steps = []

    async def fake_persist(energy_data, grad_data, kwargs, existing=(None, None)):
        flush_steps.append([row["step"] for row in energy_data])
        return (MagicMock(), MagicMock())

    monkeypatch.setattr(
        "molecular_qm_turbomole.lib.opt_artifacts.persist_opt_charts",
        fake_persist,
    )

    def fake_subprocess(name, command, cwd=""):
        _write_energy(tmp_path, 7)
        _write_gradient(tmp_path, 7)
        (tmp_path / "GEO_OPT_CONVERGED").write_text("CONVERGED\n", encoding="utf-8")
        return True

    node_runner = MagicMock()
    node_runner.subprocess.side_effect = fake_subprocess
    qm_input = SimpleNamespace(
        basis_set=SimpleNamespace(basis_set="def2-SVP"),
        max_opt_cycles=100,
    )
    await _run_optimization_chunks(qm_input, node_runner, {"node_runner": node_runner})
    assert all(steps[-1] == 7 for steps in flush_steps)
    assert flush_steps


@pytest.mark.asyncio
async def test_run_optimization_chunks_honors_max_opt_cycles(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def fake_persist(energy_data, grad_data, kwargs, existing=(None, None)):
        return (MagicMock(), MagicMock())

    monkeypatch.setattr(
        "molecular_qm_turbomole.lib.opt_artifacts.persist_opt_charts",
        fake_persist,
    )

    seen_limits = []

    def fake_subprocess(name, command, cwd=""):
        seen_limits.append(command)
        _write_energy(tmp_path, 5)
        _write_gradient(tmp_path, 5)
        (tmp_path / "GEO_OPT_FAILED").write_text(
            "OPTIMIZATION DID NOT CONVERGE WITHIN MAXCYCLES\n",
            encoding="utf-8",
        )
        return True

    node_runner = MagicMock()
    node_runner.subprocess.side_effect = fake_subprocess
    qm_input = SimpleNamespace(
        basis_set=SimpleNamespace(basis_set="def2-SVP"),
        max_opt_cycles=5,
    )
    with pytest.raises(RuntimeError, match="did not converge in 5 cycles"):
        await _run_optimization_chunks(qm_input, node_runner, {"node_runner": node_runner})
    assert len(seen_limits) == 1
    assert "-c 5" in seen_limits[0]


def test_with_runner_output_appends_stderr_once():
    runner = SimpleNamespace(last_stderr="[TM ERROR] jobex failed.", last_stdout="")
    message = _with_runner_output(runner, "Geometry optimization failed: jobex did not end properly")
    assert message.startswith("Geometry optimization failed: jobex did not end properly")
    assert "[TM ERROR] jobex failed." in message
    assert message.count("[TM ERROR] jobex failed.") == 1


@pytest.mark.asyncio
async def test_run_optimization_chunks_includes_subprocess_output_on_running_marker(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    async def fake_persist(energy_data, grad_data, kwargs, existing=(None, None)):
        return (MagicMock(), MagicMock())

    monkeypatch.setattr(
        "molecular_qm_turbomole.lib.opt_artifacts.persist_opt_charts",
        fake_persist,
    )

    def fake_subprocess(name, command, cwd=""):
        _write_energy(tmp_path, 1)
        _write_gradient(tmp_path, 1)
        (tmp_path / "GEO_OPT_RUNNING").write_text("running\n", encoding="utf-8")
        return False

    node_runner = MagicMock()
    node_runner.subprocess.side_effect = fake_subprocess
    node_runner.last_stdout = "[TM ERROR] jobex failed.\n  dscf ended abnormally"
    node_runner.last_stderr = ""
    qm_input = SimpleNamespace(
        basis_set=SimpleNamespace(basis_set="def2-SVP"),
        max_opt_cycles=100,
    )
    with pytest.raises(RuntimeError, match="jobex did not end properly") as exc_info:
        await _run_optimization_chunks(qm_input, node_runner, {"node_runner": node_runner})
    assert "dscf ended abnormally" in str(exc_info.value)
