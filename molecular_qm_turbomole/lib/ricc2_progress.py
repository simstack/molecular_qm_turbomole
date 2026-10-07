"""Export ricc2 inputs, outputs, and a crash-safe progress log while the job runs."""

import asyncio
import os
import time
import zlib
from datetime import datetime
from pathlib import Path

from simstack.core.context import context
from simstack.models.files import MONGODB_MAX_DOCUMENT_SIZE, FileStack
from simstack.util.file_hashing import hash_file

from molecular_qm_turbomole.nodes.turbomole2 import _run_monitored_subprocess

NODE_RUNNER_LOG = "node_runner.log"
_POLL_S = 1.0
_RESYNC_S = 5.0
_MAX_COMPRESSED = int(0.9 * MONGODB_MAX_DOCUMENT_SIZE)
_PATH_ONLY_FILES = {"mos", "alpha", "beta"}

# Text inputs and outputs small enough to store on the node registry. Large
# orbital files are registered by path so a killed job can still be inspected.
RICC2_PROGRESS_FILES = (
    "define.inp",
    "define.out",
    "coord",
    "control",
    "basis",
    "auxbasis",
    "energy",
    "statistics",
    "mos",
    "alpha",
    "beta",
    "dscf.out",
    "ricc2.out",
    "turbomole_define.log",
    "turbomole_dscf.log",
    "turbomole_ricc2.log",
    "heartbeat.log",
    NODE_RUNNER_LOG,
)


class Ricc2CrashProgress:
    """Write node_runner.log and publish input/output files before a kill."""

    def __init__(self, node_runner, kwargs, interval_s):
        if node_runner is None:
            raise ValueError("node_runner is required")
        if kwargs is None:
            raise ValueError("kwargs is required")
        if interval_s is None:
            raise ValueError("interval_s is required")
        interval = float(interval_s)
        if interval <= 0:
            raise ValueError("interval_s must be positive")
        task_id = kwargs.get("task_id")
        if task_id is None:
            task_id = getattr(node_runner, "task_id", None)
        if task_id is None or not str(task_id).strip():
            raise ValueError("task_id is required")
        self.node_runner = node_runner
        self.kwargs = kwargs
        self.interval_s = interval
        self.task_id = str(task_id)
        self._started = time.monotonic()
        self._stacks = {}
        self._signatures = {}
        self._pending_registry = []
        self._last_detail = None
        self._last_note_at = None
        self._uploaded_at = {}
        self._oversized = set()

    def note(self, message):
        if message is None or not str(message).strip():
            raise ValueError("message is required")
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} {message}"
        self.node_runner.info(line)
        log = getattr(self.node_runner, "log", None)
        if callable(log):
            log(line)
        path = Path(NODE_RUNNER_LOG)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def record_output(self, prefix, output_name):
        if prefix is None or not str(prefix).strip():
            raise ValueError("prefix is required")
        if output_name is None or not str(output_name).strip():
            raise ValueError("output_name is required")
        now = time.monotonic()
        elapsed = now - self._started
        path = self._candidate(str(output_name))
        if path is None:
            detail = f"{output_name} not written yet"
        else:
            detail = self._last_output_line(path)
        detail_changed = detail != self._last_detail
        due = self._last_note_at is None or now - self._last_note_at >= self.interval_s
        if not detail_changed and not due:
            return
        self._last_detail = detail
        self._last_note_at = now
        self.note(f"{prefix} elapsed={elapsed:.0f}s {detail}")

    async def publish(self, force=False):
        if not isinstance(force, bool):
            raise ValueError(f"force must be a bool, got {force!r}")
        db = context.db
        if db is None:
            raise ValueError("context.db is required")
        changed = []
        for name in RICC2_PROGRESS_FILES:
            path = self._candidate(name)
            if path is None:
                continue
            stack = self._capture(path, force=force)
            if stack is not None:
                changed.append(stack)
        for stack in changed:
            await db.save(stack)
        if not self._pending_registry:
            return
        registry = await db.load_task_by_id(self.task_id)
        if registry is None:
            raise ValueError(f"NodeRegistry {self.task_id} not found")
        info_files = getattr(registry, "info_files", None)
        if info_files is None:
            raise ValueError("NodeRegistry.info_files is missing")
        for stack in self._pending_registry:
            info_files.append(stack)
        await db.save(registry)
        self._pending_registry = []

    async def publish_safely(self, force=False):
        try:
            await self.publish(force=force)
        except Exception as exc:
            self.note(f"Failed to export info files: {exc}")

    async def run_monitored(self, name, prefix, command, output_name):
        if name is None or not str(name).strip():
            raise ValueError("name is required")
        if prefix is None or not str(prefix).strip():
            raise ValueError("prefix is required")
        if command is None or not str(command).strip():
            raise ValueError("command is required")
        if output_name is None or not str(output_name).strip():
            raise ValueError("output_name is required")
        self.record_output(str(prefix), str(output_name))
        await self.publish_safely(force=True)
        stop = asyncio.Event()

        async def watch():
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=_POLL_S)
                except asyncio.TimeoutError:
                    self.record_output(str(prefix), str(output_name))
                    await self.publish_safely()

        watcher = asyncio.create_task(watch())
        try:
            return await asyncio.to_thread(
                _run_monitored_subprocess,
                self.node_runner,
                name,
                prefix,
                self.kwargs,
                command,
            )
        finally:
            stop.set()
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
            self.record_output(str(prefix), str(output_name))
            await self.publish_safely(force=True)

    def _candidate(self, name):
        scratch = getattr(self.node_runner, "scratch_dir", None)
        if scratch is not None:
            scratch_path = Path(scratch) / name
            if scratch_path.is_file():
                return scratch_path
        local = Path(name)
        if local.is_file():
            return local
        return None

    def _capture(self, path: Path, force=False):
        path = path.resolve()
        name = path.name
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        stack = self._stacks.get(name)
        if stack is not None and self._signatures.get(name) == signature:
            return None
        uploaded = self._uploaded_at.get(name)
        if (
            stack is not None
            and not force
            and uploaded is not None
            and time.monotonic() - uploaded < _RESYNC_S
        ):
            return None
        in_memory = name not in _PATH_ONLY_FILES
        if stack is None:
            stack = FileStack.from_local_file(
                path,
                in_memory=in_memory,
                is_hashable=in_memory,
                secure_source=True,
                task_id=self.task_id,
            )
            self._stacks[name] = stack
            self._pending_registry.append(stack)
            self._signatures[name] = signature
            self._uploaded_at[name] = time.monotonic()
            return stack
        if not stack.in_memory:
            self._signatures[name] = signature
            self._uploaded_at[name] = time.monotonic()
            return None
        raw = path.read_bytes()
        compressed = zlib.compress(raw)
        if len(compressed) > _MAX_COMPRESSED:
            if name not in self._oversized:
                self._oversized.add(name)
                self.note(
                    f"{name} is too large to store in the database ({stat.st_size} bytes)"
                )
            self._signatures[name] = signature
            self._uploaded_at[name] = time.monotonic()
            return None
        stack.content = compressed
        stack.size = stat.st_size
        stack.hash = hash_file(path)
        self._signatures[name] = signature
        self._uploaded_at[name] = time.monotonic()
        return stack

    def _last_output_line(self, path: Path) -> str:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(size - 4096, 0))
            chunk = handle.read()
        text = chunk.decode("utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return f"{path.name} is empty ({size} bytes)"
        last = lines[-1]
        if len(last) > 240:
            last = last[:240]
        return f"{path.name} {size} bytes: {last}"
