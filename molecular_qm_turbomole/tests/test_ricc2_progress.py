import time
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from odmantic import ObjectId

from molecular_qm_turbomole.lib.ricc2_progress import NODE_RUNNER_LOG, Ricc2CrashProgress
from simstack.models.file_list import FileList


class _Registry:
    def __init__(self):
        self.info_files = FileList()


class _DB:
    def __init__(self):
        self.registry = _Registry()
        self.saved = []

    async def save(self, obj):
        self.saved.append(obj)
        return obj

    async def load_task_by_id(self, task_id):
        if task_id is None or not str(task_id).strip():
            raise ValueError("task_id is required")
        return self.registry


def _progress(interval_s=30.0):
    runner = SimpleNamespace(
        info=MagicMock(),
        log=MagicMock(),
        scratch_dir=None,
        task_id=str(ObjectId()),
    )
    progress = Ricc2CrashProgress(
        runner, {"task_id": runner.task_id}, interval_s=interval_s
    )
    return runner, progress


def _patch_db(monkeypatch, db):
    monkeypatch.setattr(
        "molecular_qm_turbomole.lib.ricc2_progress.context",
        SimpleNamespace(db=db),
    )


def test_progress_requires_task_id_and_interval():
    runner = SimpleNamespace(info=MagicMock(), log=MagicMock())
    with pytest.raises(ValueError, match="task_id"):
        Ricc2CrashProgress(runner, {}, interval_s=30.0)
    with pytest.raises(ValueError, match="interval_s"):
        Ricc2CrashProgress(runner, {"task_id": "abc"}, interval_s=None)
    with pytest.raises(ValueError, match="interval_s must be positive"):
        Ricc2CrashProgress(runner, {"task_id": "abc"}, interval_s=0)


@pytest.mark.asyncio
async def test_publish_exports_inputs_and_refreshes_growing_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = _DB()
    _patch_db(monkeypatch, db)
    runner, progress = _progress()
    (tmp_path / "define.inp").write_text("a coord\n", encoding="utf-8")
    (tmp_path / "control").write_text("$ricc2\nadc(2)\n$end\n", encoding="utf-8")
    (tmp_path / "dscf.out").write_text("ITER 1\n", encoding="utf-8")
    progress.note("Input files generated")
    await progress.publish()

    exported = {stack.name: stack for stack in db.saved if getattr(stack, "name", None)}
    assert set(exported) >= {"define.inp", "control", "dscf.out", NODE_RUNNER_LOG}
    assert b"a coord" in zlib.decompress(exported["define.inp"].content)
    assert b"$ricc2" in zlib.decompress(exported["control"].content)
    assert "Input files generated" in zlib.decompress(exported[NODE_RUNNER_LOG].content).decode()
    registered = {stack.name for stack in db.registry.info_files}
    assert registered >= {"define.inp", "control", "dscf.out", NODE_RUNNER_LOG}
    saved_after_first = len(db.saved)
    dscf_id = exported["dscf.out"].id

    await progress.publish()
    assert len(db.saved) == saved_after_first

    (tmp_path / "dscf.out").write_text("ITER 1\nITER 2 energy=-76.1\n", encoding="utf-8")
    await progress.publish(force=True)
    assert progress._stacks["dscf.out"].id == dscf_id
    assert b"ITER 2" in zlib.decompress(progress._stacks["dscf.out"].content)
    assert [stack.id for stack in db.registry.info_files].count(dscf_id) == 1


@pytest.mark.asyncio
async def test_publish_failure_is_written_to_node_runner_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class BoomDB(_DB):
        async def save(self, obj):
            raise RuntimeError("mongo down")

    _patch_db(monkeypatch, BoomDB())
    _, progress = _progress()
    (tmp_path / "define.inp").write_text("a coord\n", encoding="utf-8")
    await progress.publish_safely()
    text = (tmp_path / NODE_RUNNER_LOG).read_text(encoding="utf-8")
    assert "Failed to export info files" in text
    assert "mongo down" in text


@pytest.mark.asyncio
async def test_run_monitored_uploads_output_while_subprocess_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = _DB()
    _patch_db(monkeypatch, db)
    _, progress = _progress(interval_s=0.05)
    (tmp_path / "control").write_text("$end\n", encoding="utf-8")

    def fake_run(node_runner, name, prefix, kwargs, command="run_command"):
        assert name == "turbomole_dscf"
        assert command == "dscf_command"
        Path("dscf.out").write_text(
            "ITER 1\n dscf ended abnormally\n", encoding="utf-8"
        )
        deadline = time.time() + 3
        while time.time() < deadline:
            if any(getattr(obj, "name", None) == "dscf.out" for obj in db.saved):
                return False, 0.3, 0.1
            time.sleep(0.05)
        raise AssertionError("dscf.out was not exported while dscf was still running")

    monkeypatch.setattr(
        "molecular_qm_turbomole.lib.ricc2_progress._run_monitored_subprocess",
        fake_run,
    )
    ok, wall_s, cpu_s = await progress.run_monitored(
        "turbomole_dscf",
        "TURBOMOLE HF reference (dscf)",
        command="dscf_command",
        output_name="dscf.out",
    )
    assert ok is False
    assert wall_s == 0.3
    assert cpu_s == 0.1
    log_text = (tmp_path / NODE_RUNNER_LOG).read_text(encoding="utf-8")
    assert "dscf.out not written yet" in log_text
    assert "dscf ended abnormally" in log_text
    dscf = progress._stacks["dscf.out"]
    assert b"dscf ended abnormally" in zlib.decompress(dscf.content)
    assert dscf.name in {stack.name for stack in db.registry.info_files}
    assert "control" in {stack.name for stack in db.registry.info_files}
