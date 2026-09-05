import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from odmantic import ObjectId

from molecular_qm_models import (
    BasisSet,
    BasisSetEnum,
    Functional,
    FunctionalEnum,
    Molecule,
    MoleculeSnapshot,
    QMInput,
    geometry_hash_from_molecule,
)
from molecular_qm_turbomole.lib.output_parser import (
    parse_coord_file,
    parse_energy_history,
    parse_gradient_history,
)
from simstack.core.context import context
from simstack.models.charts_artifact import (
    AGChartAxisConfig,
    AGChartTitleConfig,
    AGLineSeriesConfig,
    ChartArtifactModel,
)
from simstack.models.file_list import FileList
from simstack.models.files import FileStack

logger = logging.getLogger("TurbomoleOptArtifacts")

OPT_CHART_INTERVAL = 10
SNAPSHOT_INTERVAL = 10
MAX_OPT_CYCLES = 100
SNAPSHOT_ARCHIVE_PREFIX = "snapshot_restart"


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


def snapshot_marker(iteration, interval: int = SNAPSHOT_INTERVAL) -> int:
    if iteration is None:
        raise ValueError("iteration is required")
    if interval is None:
        raise ValueError("interval is required")
    interval = int(interval)
    if interval < 1:
        raise ValueError("interval must be >= 1")
    return (int(iteration) // interval) * interval


def should_snapshot(iteration, interval: int = SNAPSHOT_INTERVAL, seen=None) -> bool:
    if iteration is None:
        return False
    try:
        iteration = int(iteration)
    except (TypeError, ValueError):
        return False
    if interval is None:
        raise ValueError("interval is required")
    interval = int(interval)
    if interval < 1 or iteration < 1:
        return False
    marker = snapshot_marker(iteration, interval)
    if marker < interval:
        return False
    if seen is not None and marker in seen:
        return False
    return True


def _enum_from_text(enum_cls, text, name: str):
    if text is None:
        raise ValueError(f"{name} is required")
    raw = str(getattr(text, "value", text)).strip()
    if not raw:
        raise ValueError(f"{name} is required")
    for item in enum_cls:
        if item.value.lower() == raw.lower() or item.name.lower() == raw.lower():
            return item
    compact = raw.replace("-", "").replace("_", "").replace(" ", "")
    for item in enum_cls:
        item_value = item.value.replace("-", "").replace("_", "").replace(" ", "")
        item_name = item.name.replace("_", "")
        if item_value.lower() == compact.lower() or item_name.lower() == compact.lower():
            return item
    raise ValueError(f"Unsupported {name} for MoleculeSnapshot: {raw!r}")


def _snapshot_qm_input(qm_input, molecule: Molecule, restart_files: FileList) -> QMInput:
    if qm_input is None:
        raise ValueError("qm_input is required")
    basis_name = getattr(getattr(qm_input, "basis_set", None), "basis_set", None)
    func_keyword = qm_input.functional.keyword()
    return QMInput(
        molecule=molecule,
        charge=int(qm_input.charge),
        basis_set=BasisSet(basis_set=_enum_from_text(BasisSetEnum, basis_name, "basis_set")),
        functional=Functional(
            functional=_enum_from_text(FunctionalEnum, func_keyword, "functional")
        ),
        optimization=True,
        gradients=True,
        frequencies=bool(qm_input.frequencies),
        max_optimization_iterations=int(qm_input.max_opt_cycles),
        max_scf_iterations=int(qm_input.scfiterlimit),
        restart_files=restart_files,
    )


async def persist_opt_snapshot(
    directory,
    restart_names,
    qm_input,
    kwargs,
    geom_iter: int,
    energy_hartree=None,
):
    """Write a MoleculeSnapshot plus a zip of TURBOMOLE restart files."""
    if restart_names is None:
        raise ValueError("restart_names is required")
    if geom_iter is None:
        raise ValueError("geom_iter is required")
    node_runner = None if not kwargs else kwargs.get("node_runner")
    root = Path(directory)
    existing = [name for name in restart_names if (root / name).is_file()]
    if not existing:
        if node_runner is not None:
            node_runner.warning(
                f"Skipping MoleculeSnapshot at geom_iter={geom_iter}: no restart files"
            )
        return None
    molecule = parse_coord_file(root / "coord")
    if molecule is None or not molecule.atoms:
        if node_runner is not None:
            node_runner.warning(
                f"Skipping MoleculeSnapshot at geom_iter={geom_iter}: no coord geometry"
            )
        return None
    source = getattr(qm_input, "molecule", None)
    molecule.smiles = getattr(source, "smiles", None)
    molecule.formula = getattr(source, "formula", None)
    task_id = task_id_from_kwargs(kwargs or {})
    archive = root / f"{SNAPSHOT_ARCHIVE_PREFIX}_c{int(geom_iter):04d}.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name in existing:
            handle.write(root / name, arcname=name)
    restart_files = FileList()
    stacks = []
    for name in existing:
        fs = FileStack.from_local_file(
            str(root / name),
            in_memory=False,
            is_hashable=True,
            secure_source=True,
            task_id=task_id,
        )
        restart_files.append(fs)
        stacks.append(fs)
    wfn_fs = FileStack.from_local_file(
        str(archive),
        in_memory=False,
        is_hashable=True,
        secure_source=True,
        task_id=task_id,
    )
    snap_input = _snapshot_qm_input(qm_input, molecule, restart_files)
    snapshot = MoleculeSnapshot(
        date_created=datetime.now(),
        task_id=task_id,
        smiles=molecule.smiles,
        formula=molecule.formula,
        call_path=None if not kwargs else kwargs.get("call_path"),
        geom_iter=int(geom_iter),
        scf_iter=int(geom_iter),
        final_structure=False,
        energy_hartree=energy_hartree,
        has_forces=False,
        geometry_hash=geometry_hash_from_molecule(molecule),
        qm_input=snap_input,
        molecule=molecule,
        wavefunction=wfn_fs,
    )
    db = _get_db()
    if db is None:
        if node_runner is not None:
            node_runner.warning(
                f"Skipping MoleculeSnapshot persist at geom_iter={geom_iter}: "
                "context.db is unavailable"
            )
        return snapshot
    await db.save(wfn_fs)
    for fs in stacks:
        await db.save(fs)
    await db.save(molecule)
    await db.save(snap_input)
    await db.save(snapshot)
    if node_runner is not None:
        node_runner.info(
            f"Saved MoleculeSnapshot geom_iter={geom_iter} "
            f"restart_files={len(existing)} (task_id={task_id})"
        )
    return snapshot


async def cleanup_opt_snapshots(snapshots, kwargs, directory="."):
    """Delete intermediate MoleculeSnapshot checkpoints after a successful job."""
    node_runner = None if not kwargs else kwargs.get("node_runner")
    db = _get_db()
    records = list(snapshots or [])
    deleted = 0
    for snap in records:
        if db is None:
            break
        try:
            wfn = getattr(snap, "wavefunction", None)
            if wfn is not None:
                await db.delete(wfn)
            snap_input = getattr(snap, "qm_input", None)
            if snap_input is not None:
                for fs in list(getattr(snap_input, "restart_files", None) or []):
                    try:
                        await db.delete(fs)
                    except Exception:
                        pass
                await db.delete(snap_input)
            mol = getattr(snap, "molecule", None)
            if mol is not None:
                await db.delete(mol)
            await db.delete(snap)
            deleted += 1
        except Exception as exc:
            if node_runner is not None:
                node_runner.warning(f"Failed to delete MoleculeSnapshot: {exc}")
            else:
                logger.warning("Failed to delete MoleculeSnapshot: %s", exc)
    if isinstance(snapshots, list):
        snapshots.clear()
    root = Path(directory)
    for path in root.glob(f"{SNAPSHOT_ARCHIVE_PREFIX}_*.zip"):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except Exception as exc:
            if node_runner is not None:
                node_runner.warning(f"Failed to delete snapshot archive {path}: {exc}")
            else:
                logger.warning("Failed to delete snapshot archive %s: %s", path, exc)
    if node_runner is not None and deleted:
        node_runner.info(
            f"Removed {deleted} intermediate MoleculeSnapshot checkpoint(s)"
        )


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
        self.seen_snapshots: set[int] = set()
        self.snapshots: list = []

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

    async def maybe_snapshot(self, qm_input, restart_names, directory: str | Path = "."):
        step = self.latest_step
        if not should_snapshot(step, self.interval, self.seen_snapshots):
            return None
        energy = self.energy_history[-1]["energy"] if self.energy_history else None
        try:
            snap = await persist_opt_snapshot(
                directory,
                restart_names,
                qm_input,
                self.kwargs,
                geom_iter=step,
                energy_hartree=energy,
            )
        except Exception as exc:
            node_runner = self._node_runner()
            if node_runner is not None:
                node_runner.warning(f"Failed to store MoleculeSnapshot: {exc}")
            else:
                logger.warning("Failed to store MoleculeSnapshot: %s", exc)
            return None
        if snap is not None:
            self.seen_snapshots.add(snapshot_marker(step, self.interval))
            self.snapshots.append(snap)
        return snap
