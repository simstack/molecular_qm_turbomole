"""Per-chunk optimization timeout and oscillation checks, matching Psi4/PySCF."""

WATCHDOG_SIDECAR = "optimization_watchdog_timeout.txt"

# TURBOMOLE DFT is typically faster than Psi4 (180 s/atom) and PySCF (120 s/atom).
_TIMEOUT_SECONDS_PER_ATOM = 90
_TIMEOUT_MIN_SECONDS = 600
_TIMEOUT_MAX_SECONDS = 86400
_DEFAULT_BASIS_WEIGHT = 2.0

_OSC_WARMUP_STEPS = 10
_OSC_WINDOW_STEPS = 10
_OSC_MIN_SIGN_FLIPS = 4
_OSC_MEAN_DELTA_FLOOR = -1e-5
_OSC_AMPLITUDE_MIN = 1e-4
_OSC_GRAD_NORM_MIN = 1e-3


class OptimizationTimeoutError(RuntimeError):
    """Raised when a jobex chunk exceeds the iteration timeout."""


class OptimizationOscillationError(RuntimeError):
    """Raised when optimization energy oscillates with no net downward trend."""


def n_atoms_from_molecule(molecule) -> int:
    if molecule is None:
        raise ValueError("molecule is required")
    atoms = getattr(molecule, "atoms", None)
    if atoms is None:
        raise ValueError("molecule.atoms is required")
    try:
        n_atoms = len(atoms)
    except TypeError as exc:
        raise ValueError("molecule.atoms is required") from exc
    if n_atoms < 1:
        raise ValueError("molecule must contain at least one atom")
    return n_atoms


def basis_name_from_qm_input(qm_input) -> str:
    if qm_input is None:
        raise ValueError("qm_input is required")
    basis_set = getattr(qm_input, "basis_set", None)
    if basis_set is None:
        raise ValueError("basis_set is required")
    inner = getattr(basis_set, "basis_set", basis_set)
    raw = inner.value if hasattr(inner, "value") else inner
    if raw is None:
        raise ValueError("basis_set is required")
    name = str(raw).strip()
    if not name:
        raise ValueError("basis_set is required")
    return name


def basis_weight(basis_name) -> float:
    """Relative cost weight for the iteration-timeout heuristic."""
    if basis_name is None:
        raise ValueError("basis_name is required")
    name = str(basis_name).strip().lower().replace("_", "-")
    if not name:
        raise ValueError("basis_name is required")
    compact = name.replace("-", "")
    if compact in {"sto3g", "sto6g"}:
        return 1.0
    if "5z" in compact or "v5z" in compact:
        return 10.0
    if "qz" in compact or "vqz" in compact:
        return 8.0
    if "tz" in compact or "vtz" in compact:
        if "aug" in compact or compact.endswith("d"):
            return 6.0
        if "pp" in compact:
            return 5.0
        return 4.0
    if "aug" in compact or "svpd" in compact:
        return 3.0
    if "svp" in compact or "vdz" in compact or "631" in compact:
        return 2.0
    return _DEFAULT_BASIS_WEIGHT


def iteration_timeout_seconds(n_atoms, basis_name) -> float:
    """TURBOMOLE per-chunk timeout: 90 s × n_atoms × basis_weight, clamped to [10 min, 24 h]."""
    if n_atoms is None:
        raise ValueError("n_atoms is required")
    if basis_name is None:
        raise ValueError("basis_name is required")
    n = int(n_atoms)
    if n < 1:
        raise ValueError("n_atoms must be >= 1")
    raw = _TIMEOUT_SECONDS_PER_ATOM * n * basis_weight(basis_name)
    return min(_TIMEOUT_MAX_SECONDS, max(_TIMEOUT_MIN_SECONDS, raw))


def energy_oscillation_stats(energies, grad_norm, *, warmup=_OSC_WARMUP_STEPS, window=_OSC_WINDOW_STEPS):
    """Return oscillation stats if the last window has stalled, else None."""
    if energies is None:
        raise ValueError("energies is required")
    if warmup is None:
        raise ValueError("warmup is required")
    if window is None:
        raise ValueError("window is required")
    if len(energies) <= warmup or len(energies) < window:
        return None
    try:
        grad = float(grad_norm)
    except (TypeError, ValueError):
        return None
    if grad <= _OSC_GRAD_NORM_MIN:
        return None
    recent = [float(e) for e in energies[-window:]]
    deltas = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    if not deltas:
        return None
    flips = 0
    prev = None
    for delta in deltas:
        if delta == 0:
            continue
        sign = 1 if delta > 0 else -1
        if prev is not None and sign != prev:
            flips += 1
        prev = sign
    mean_delta = sum(deltas) / len(deltas)
    amplitude = max(recent) - min(recent)
    net = abs(recent[-1] - recent[0])
    stats = {
        "mean_delta": mean_delta,
        "amplitude": amplitude,
        "net": net,
        "sign_flips": flips,
        "grad_norm": grad,
        "n_steps": len(energies),
    }
    if mean_delta < _OSC_MEAN_DELTA_FLOOR:
        return None
    if flips < _OSC_MIN_SIGN_FLIPS:
        return None
    if amplitude <= _OSC_AMPLITUDE_MIN:
        return None
    if net >= 0.5 * amplitude:
        return None
    return stats


def energy_is_oscillating(energies, grad_norm, **kwargs) -> bool:
    return energy_oscillation_stats(energies, grad_norm, **kwargs) is not None
