import importlib
import math
import re
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
from molecular_qm_turbomole.lib.optimization_timing import (
    attach_optimizer_timings,
    optimization_timing_table,
)
from molecular_qm_turbomole.lib.output_parser import parse_energy_history, parse_gradient_history
from molecular_qm_turbomole.nodes.turbomole2 import (
    HEARTBEAT_INTERVAL_S,
    HEARTBEAT_LOG,
    _collect_turbomole_info_files,
    _collect_turbomole_restart_files,
    _run_optimization_chunks,
    _with_runner_output,
)

turbomole2_module = importlib.import_module("molecular_qm_turbomole.nodes.turbomole2")


@pytest.fixture
def patch_heartbeat(monkeypatch):
    instance = MagicMock()
    cls = MagicMock(return_value=instance)
    monkeypatch.setattr(turbomole2_module, "ProcessHeartbeat", cls)
    return cls, instance


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
async def test_run_optimization_chunks_flushes_at_ten_and_twenty(
    tmp_path, monkeypatch, patch_heartbeat
):
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
async def test_run_optimization_chunks_flushes_when_energy_has_initial_scf(
    tmp_path, monkeypatch, patch_heartbeat
):
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
async def test_run_optimization_chunks_flushes_on_early_convergence(
    tmp_path, monkeypatch, patch_heartbeat
):
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
async def test_run_optimization_chunks_honors_max_opt_cycles(
    tmp_path, monkeypatch, patch_heartbeat
):
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
    tmp_path, monkeypatch, patch_heartbeat
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


def test_collect_turbomole_info_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Create info files
    (tmp_path / "job.last").write_text("job last content\n", encoding="utf-8")
    (tmp_path / "job.1").write_text("job 1 content\n", encoding="utf-8")
    (tmp_path / "job.2").write_text("job 2 content\n", encoding="utf-8")
    (tmp_path / "job.start").write_text("job start content\n", encoding="utf-8")
    (tmp_path / "turbomole_define.log").write_text("define log\n", encoding="utf-8")
    (tmp_path / "turbomole_exe_c010.log").write_text("opt chunk log\n", encoding="utf-8")
    (tmp_path / "turbomole_exe.log").write_text("exe log\n", encoding="utf-8")
    (tmp_path / "not.converged").write_text("not converged\n", encoding="utf-8")
    (tmp_path / "GEO_OPT_FAILED").write_text("failed\n", encoding="utf-8")
    (tmp_path / "statistics").write_text("stats\n", encoding="utf-8")
    (tmp_path / "define.inp").write_text("define inp\n", encoding="utf-8")
    (tmp_path / "define.out").write_text("define out\n", encoding="utf-8")
    (tmp_path / "jobex.out").write_text("jobex out\n", encoding="utf-8")
    (tmp_path / HEARTBEAT_LOG).write_text("heartbeat\n", encoding="utf-8")

    # Create restart files (which must NOT be added to info_files)
    (tmp_path / "control").write_text("$title test\n", encoding="utf-8")
    (tmp_path / "coord").write_text("$coord\n$end\n", encoding="utf-8")
    (tmp_path / "basis").write_text("$basis\n$end\n", encoding="utf-8")
    (tmp_path / "auxbasis").write_text("$auxbasis\n$end\n", encoding="utf-8")
    (tmp_path / "mos").write_text("$scfmo\n$end\n", encoding="utf-8")
    (tmp_path / "energy").write_text("$energy\n$end\n", encoding="utf-8")
    (tmp_path / "gradient").write_text("$grad\n$end\n", encoding="utf-8")

    node_runner = SimpleNamespace(info_files=[], info=MagicMock(), warning=MagicMock())
    _collect_turbomole_info_files(node_runner)

    collected_names = {fs.name for fs in node_runner.info_files}

    # Info files that must be present
    assert "job.last" in collected_names
    assert "job.1" in collected_names
    assert "job.2" in collected_names
    assert "job.start" in collected_names
    assert "turbomole_define.log" in collected_names
    assert "turbomole_exe_c010.log" in collected_names
    assert "turbomole_exe.log" in collected_names
    assert "not.converged" in collected_names
    assert "GEO_OPT_FAILED" in collected_names
    assert "statistics" in collected_names
    assert "define.inp" in collected_names
    assert "define.out" in collected_names
    assert "jobex.out" in collected_names
    assert HEARTBEAT_LOG in collected_names

    # Restart files must NOT be in info_files
    for restart_name in ["control", "coord", "basis", "auxbasis", "mos", "energy", "gradient"]:
        assert restart_name not in collected_names


@pytest.mark.asyncio
async def test_collect_turbomole_restart_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Create restart files
    (tmp_path / "control").write_text("$title test\n", encoding="utf-8")
    (tmp_path / "coord").write_text("$coord\n$end\n", encoding="utf-8")
    (tmp_path / "basis").write_text("$basis\n$end\n", encoding="utf-8")
    (tmp_path / "auxbasis").write_text("$auxbasis\n$end\n", encoding="utf-8")
    (tmp_path / "mos").write_text("$scfmo\n$end\n", encoding="utf-8")
    (tmp_path / "energy").write_text("$energy\n$end\n", encoding="utf-8")
    (tmp_path / "gradient").write_text("$grad\n$end\n", encoding="utf-8")
    (tmp_path / "hessapprox").write_text("$hess\n$end\n", encoding="utf-8")
    (tmp_path / "optinfo").write_text("$optinfo\n$end\n", encoding="utf-8")
    (tmp_path / "final_geometry.xyz").write_text("3\n\nC 0 0 0\n", encoding="utf-8")

    # Create info files (which must NOT be added to files)
    (tmp_path / "job.last").write_text("job last\n", encoding="utf-8")
    (tmp_path / "turbomole_exe_c010.log").write_text("chunk log\n", encoding="utf-8")

    node_runner = SimpleNamespace(files=[], info=MagicMock(), warning=MagicMock())
    qm_result = SimpleNamespace(files=[])

    await _collect_turbomole_restart_files(node_runner, qm_result)

    runner_files = {fs.name for fs in node_runner.files}
    result_files = {fs.name for fs in qm_result.files}

    assert runner_files == result_files
    assert "control" in runner_files
    assert "coord" in runner_files
    assert "basis" in runner_files
    assert "auxbasis" in runner_files
    assert "mos" in runner_files
    assert "energy" in runner_files
    assert "gradient" in runner_files
    assert "hessapprox" in runner_files
    assert "optinfo" in runner_files
    assert "final_geometry.xyz" in runner_files

    # Info files must NOT be in runner.files
    assert "job.last" not in runner_files
    assert "turbomole_exe_c010.log" not in runner_files


def test_record_iteration_logs_energy_gradient_and_timings(tmp_path):
    node_runner = MagicMock()
    tracker = OptimizationChartTracker({"node_runner": node_runner})
    _write_energy(tmp_path, 3)
    _write_gradient(tmp_path, 3)
    tracker.update_from_directory(tmp_path)
    tracker.record_iteration(2.5, 1.25)
    assert len(tracker.timing_history) == 1
    row = tracker.timing_history[0]
    assert row["step"] == 3
    assert row["wall_time_s"] == 2.5
    assert row["cpu_time_s"] == 1.25
    assert row["energy"] == pytest.approx(-76.003)
    assert row["grad_norm"] == pytest.approx(0.0103)
    messages = [call.args[0] for call in node_runner.info.call_args_list]
    assert len(messages) == 1
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ", messages[0])
    assert "Optimization step 3: energy=-76.003000000000 Ha, |g|=1.030000e-02 Ha/Bohr" in messages[0]
    assert "wall=2.50s, cpu=1.25s" in messages[0]
    logged = [call.args[0] for call in node_runner.log.call_args_list]
    assert logged == messages


def test_record_iteration_requires_wall_and_cpu():
    tracker = OptimizationChartTracker({})
    with pytest.raises(ValueError, match="wall_s"):
        tracker.record_iteration(None, 1.0)
    with pytest.raises(ValueError, match="cpu_s"):
        tracker.record_iteration(1.0, None)


def test_optimization_timing_table_has_iteration_and_summary_rows():
    snap = SimpleNamespace(
        timing_history=[
            {"step": 1, "wall_time_s": 2.0, "cpu_time_s": 1.5},
            {"step": 2, "wall_time_s": 4.0, "cpu_time_s": 3.0},
        ],
        opt_wall_s=7.0,
        opt_cpu_s=5.0,
    )
    table = optimization_timing_table(snap)
    assert table.name == "Optimization timing"
    assert table.row[0]["metric"] == "iteration"
    assert table.row[0]["step"] == 1
    assert table.row[1]["step"] == 2
    by_metric = {row["metric"]: row for row in table.row if row["metric"] != "iteration"}
    assert by_metric["total"]["wall_time_s"] == 6.0
    assert by_metric["mean"]["cpu_time_s"] == 2.25
    assert by_metric["min"]["wall_time_s"] == 2.0
    assert by_metric["max"]["cpu_time_s"] == 3.0
    assert by_metric["optimize"]["wall_time_s"] == 7.0


def test_optimization_timing_table_includes_frequencies_row():
    snap = SimpleNamespace(
        timing_history=[{"step": 1, "wall_time_s": 2.0, "cpu_time_s": 1.5}],
        opt_wall_s=2.0,
        opt_cpu_s=1.5,
    )
    table = optimization_timing_table(snap, freq_wall_s=4.5, freq_cpu_s=9.0)
    by_metric = {row["metric"]: row for row in table.row}
    assert by_metric["optimize"]["wall_time_s"] == 2.0
    assert by_metric["frequencies"]["wall_time_s"] == 4.5
    assert by_metric["frequencies"]["cpu_time_s"] == 9.0


def test_optimization_timing_table_frequencies_only():
    table = optimization_timing_table(None, freq_wall_s=4.5, freq_cpu_s=9.0)
    assert table.row[0]["metric"] == "frequencies"
    assert table.row[0]["wall_time_s"] == 4.5
    assert table.row[0]["cpu_time_s"] == 9.0


def test_optimization_timing_table_skips_empty_snapshotter():
    assert optimization_timing_table(None) is None
    snap = SimpleNamespace(timing_history=[], opt_wall_s=None, opt_cpu_s=None)
    assert optimization_timing_table(snap) is None


def test_optimization_timing_table_requires_cpu_when_wall_set():
    snap = SimpleNamespace(timing_history=[], opt_wall_s=1.0, opt_cpu_s=None)
    with pytest.raises(ValueError, match="opt_cpu_s"):
        optimization_timing_table(snap)


def test_optimization_timing_table_requires_freq_cpu_when_wall_set():
    with pytest.raises(ValueError, match="freq_cpu_s"):
        optimization_timing_table(None, freq_wall_s=1.0)


@pytest.mark.asyncio
async def test_run_optimization_chunks_records_timings_heartbeat_and_logs(
    tmp_path, monkeypatch, patch_heartbeat
):
    heartbeat_cls, heartbeat = patch_heartbeat
    monkeypatch.chdir(tmp_path)

    async def fake_persist(energy_data, grad_data, kwargs, existing=(None, None)):
        return (MagicMock(), MagicMock())

    monkeypatch.setattr(
        "molecular_qm_turbomole.lib.opt_artifacts.persist_opt_charts",
        fake_persist,
    )

    def fake_subprocess(name, command, cwd=""):
        _write_energy(tmp_path, 10)
        _write_gradient(tmp_path, 10)
        (tmp_path / "GEO_OPT_CONVERGED").write_text("CONVERGED\n", encoding="utf-8")
        return True

    node_runner = MagicMock()
    node_runner.subprocess.side_effect = fake_subprocess
    qm_input = SimpleNamespace(
        basis_set=SimpleNamespace(basis_set="def2-SVP"),
        max_opt_cycles=100,
    )
    tracker = await _run_optimization_chunks(
        qm_input, node_runner, {"node_runner": node_runner}
    )
    assert heartbeat_cls.call_count == 1
    assert heartbeat_cls.call_args.kwargs["interval_s"] == HEARTBEAT_INTERVAL_S
    assert HEARTBEAT_INTERVAL_S == 1800.0
    assert heartbeat.start.call_count == 1
    assert heartbeat.stop.call_count == 1
    assert [row["step"] for row in tracker.timing_history] == [10]
    assert tracker.timing_history[0]["wall_time_s"] >= 0
    assert tracker.timing_history[0]["cpu_time_s"] >= 0
    assert tracker.opt_wall_s is not None
    assert tracker.opt_cpu_s is not None
    metrics = [row["metric"] for row in node_runner.optimization_timing.row]
    assert metrics.count("iteration") == 1
    assert "total" in metrics
    assert "optimize" in metrics
    infos = [call.args[0] for call in node_runner.info.call_args_list]
    assert any("Starting jobex chunk cycles 1-10" in msg for msg in infos)
    step_logs = [msg for msg in infos if "Optimization step 10:" in msg]
    assert len(step_logs) == 1
    assert "energy=-76.010000000000 Ha" in step_logs[0]
    assert "|g|=1.100000e-02 Ha/Bohr" in step_logs[0]
    assert any("Optimization timings:" in msg for msg in infos)


def test_attach_optimizer_timings_copies_iteration_table():
    node_runner = SimpleNamespace()
    node_runner.info = MagicMock()
    snapshotter = SimpleNamespace(
        timing_history=[
            {"step": 1, "wall_time_s": 2.0, "cpu_time_s": 1.5},
            {"step": 2, "wall_time_s": 4.0, "cpu_time_s": 3.0},
        ],
        opt_wall_s=6.5,
        opt_cpu_s=15.25,
    )
    attach_optimizer_timings(node_runner, snapshotter, freq_wall_s=3.25, freq_cpu_s=8.5)
    assert node_runner.optimization_timing.row[0]["step"] == 1
    by_metric = {row["metric"]: row for row in node_runner.optimization_timing.row}
    assert by_metric["optimize"]["cpu_time_s"] == 15.25
    assert by_metric["frequencies"]["wall_time_s"] == 3.25
    node_runner.info.assert_called()


def test_process_heartbeat_appends_until_stopped(tmp_path):
    import time

    from molecular_qm_turbomole.lib.process_heartbeat import ProcessHeartbeat

    path = tmp_path / "heartbeat.log"
    extra = tmp_path / "turbomole_exe_c010.log"
    heartbeat = ProcessHeartbeat(
        str(path),
        "jobex chunk cycles 1-10",
        interval_s=0.2,
        task_id="abc",
        extra_paths=[str(extra)],
    )
    heartbeat.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not path.exists():
            time.sleep(0.05)
        time.sleep(0.45)
    finally:
        heartbeat.stop()
    text = path.read_text(encoding="utf-8")
    assert "jobex chunk cycles 1-10" in text
    assert "still running" in text
    assert extra.exists()
    assert "jobex chunk cycles 1-10" in extra.read_text(encoding="utf-8")
