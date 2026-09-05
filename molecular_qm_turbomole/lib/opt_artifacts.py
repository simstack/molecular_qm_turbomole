import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from odmantic import ObjectId

from molecular_qm_turbomole.lib.output_parser import parse_energy_history, parse_gradient_history
from simstack.core.context import context
from simstack.models.charts_artifact import (
    AGChartAxisConfig,
    AGChartTitleConfig,
    AGLineSeriesConfig,
    ChartArtifactModel,
)

logger = logging.getLogger("TurbomoleOptArtifacts")

OPT_CHART_INTERVAL = 10
MAX_OPT_CYCLES = 100


def task_id_from_kwargs(kwargs: dict) -> str:
    task_id = kwargs.get("task_id")
    if task_id is None:
        node_runner = kwargs.get("node_runner")
        task_id = getattr(node_runner, "task_id", None)
    return "" if task_id is None else str(task_id)


def task_parent_id(kwargs: dict):
    task_id = task_id_from_kwargs(kwargs)
    if not task_id:
        return None
    try:
        return ObjectId(str(task_id))
    except Exception:
        return None


def _get_db():
    try:
        return context.db
    except RuntimeError:
        return None


def opt_line_chart(data, y_key, title, y_label, parent_id, existing=None):
    series = AGLineSeriesConfig(
        type="line",
        xKey="step",
        yKey=y_key,
        title=y_label,
        data=data,
        marker={"enabled": False},
    )
    axes = [
        AGChartAxisConfig(type="number", position="bottom", title="Optimization step"),
        AGChartAxisConfig(type="number", position="left", title=y_label),
    ]
    if existing is not None:
        existing.data = data
        existing.title = AGChartTitleConfig(text=title)
        existing.series = [series]
        existing.axes = axes
        existing.parent_id = parent_id
        return existing
    return ChartArtifactModel(
        parent_id=parent_id,
        data=data,
        title=AGChartTitleConfig(text=title),
        series=[series],
        axes=axes,
    )


async def persist_opt_charts(energy_data, grad_data, kwargs, existing=(None, None)):
    node_runner = None if not kwargs else kwargs.get("node_runner")
    parent_id = task_parent_id(kwargs)
    if parent_id is None:
        if node_runner is not None:
            node_runner.warning("Skipping optimization charts: missing task_id")
        return existing
    db = _get_db()
    if db is None:
        return existing
    energy_chart = opt_line_chart(
        list(energy_data),
        "energy",
        "TURBOMOLE optimization energy",
        "Energy (Ha)",
        parent_id,
        existing[0],
    )
    grad_chart = opt_line_chart(
        list(grad_data),
        "grad_norm",
        "TURBOMOLE optimization gradient norm",
        "|g| (Ha/Bohr)",
        parent_id,
        existing[1],
    )
    try:
        await db.save(energy_chart)
        await db.save(grad_chart)
    except Exception as exc:
        if node_runner is not None:
            node_runner.warning(f"Failed to store optimization charts: {exc}")
        else:
            logger.warning("Failed to store optimization charts: %s", exc)
        return existing
    if node_runner is not None and energy_data:
        node_runner.info(
            f"Saved optimization charts at step {energy_data[-1]['step']} "
            f"(task_id={parent_id})"
        )
    return energy_chart, grad_chart


def inspect_geometry_optimization(directory: str | Path = ".") -> tuple[str, Optional[str]]:
    """Classify a jobex workdir as converged, continue, or failed."""
    root = Path(directory)
    if (root / "GEO_OPT_CONVERGED").is_file():
        return "converged", None
    if (root / "GEO_OPT_RUNNING").is_file():
        return "failed", "jobex did not end properly"
    failed = root / "GEO_OPT_FAILED"
    if not failed.is_file():
        return "continue", "opt status unknown"
    text = failed.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    if "your energy calculation did not converge" in lowered:
        return "failed", "SCF did not converge during optimization"
    if "OPTIMIZATION DID NOT CONVERGE" in text:
        return "continue", "opt not converged"
    return "failed", "geometry optimization failed"


class OptimizationChartTracker:
    """Accumulate energy/|g| traces, wall/CPU timings, and persist ChartArtifactModels."""

    def __init__(self, kwargs: dict, interval: int = OPT_CHART_INTERVAL):
        self.kwargs = kwargs
        self.interval = interval
        self.energy_history: list[dict] = []
        self.grad_history: list[dict] = []
        self.timing_history: list[dict] = []
        self.opt_wall_s = None
        self.opt_cpu_s = None
        self.charts = (None, None)

    @property
    def latest_step(self) -> int:
        if not self.energy_history:
            return 0
        return int(self.energy_history[-1]["step"])

    def _node_runner(self):
        if not self.kwargs:
            return None
        return self.kwargs.get("node_runner")

    def record_iteration(self, wall_s, cpu_s):
        """Log energy/|g| and append one timing row for the latest energy cycle."""
        if wall_s is None:
            raise ValueError("wall_s is required")
        if cpu_s is None:
            raise ValueError("cpu_s is required")
        step = self.latest_step
        energy = self.energy_history[-1]["energy"] if self.energy_history else None
        grad_norm = self.grad_history[-1]["grad_norm"] if self.grad_history else None
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if energy is None or grad_norm is None:
            msg = f"{stamp} Optimization step {step}: energy/gradient unavailable"
        else:
            msg = (
                f"{stamp} Optimization step {step}: "
                f"energy={float(energy):.12f} Ha, |g|={float(grad_norm):.6e} Ha/Bohr"
            )
        msg = f"{msg}, wall={float(wall_s):.2f}s, cpu={float(cpu_s):.2f}s"
        node_runner = self._node_runner()
        if node_runner is not None:
            node_runner.info(msg)
            log = getattr(node_runner, "log", None)
            if callable(log):
                log(msg)
        else:
            logger.info(msg)
        if step < 1:
            return stamp
        self.timing_history.append(
            {
                "step": int(step),
                "wall_time_s": float(wall_s),
                "cpu_time_s": float(cpu_s),
                "timestamp": stamp,
                "energy": energy,
                "grad_norm": grad_norm,
            }
        )
        return stamp

    def update_from_directory(self, directory: str | Path = ".") -> None:
        root = Path(directory)
        self.energy_history = parse_energy_history(root / "energy")
        self.grad_history = parse_gradient_history(root / "gradient")

    def should_flush(self, force: bool = False) -> bool:
        if not self.energy_history:
            return False
        if force:
            return True
        return self.latest_step % self.interval == 0

    async def maybe_flush(self, force: bool = False):
        if not self.should_flush(force=force):
            return self.charts
        saved = await persist_opt_charts(
            self.energy_history,
            self.grad_history,
            self.kwargs,
            self.charts,
        )
        if saved is not None:
            self.charts = saved
        return self.charts
