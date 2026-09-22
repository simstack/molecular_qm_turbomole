import math
import os
import re
from pathlib import Path
from typing import Optional

from molecular_qm_models.molecule import Atom, Molecule
from simstack.core.node_runner import NodeRunner
from simstack.models.simple_table import SimpleTable

BOHR_TO_ANGSTROM = 1.0 / 1.8897259886
_PROGRAM_OUTPUT_TAIL_LINES = 40

_VIBSPECTRUM_ROW_RE = re.compile(
    r"^\s*(\d+)(?:\s+([A-Za-z0-9'\"]+))?\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([\w-]+)\s+([\w-]+)",
    re.MULTILINE,
)
_CYCLE_HEADER_RE = re.compile(r"cycle\s*=\s*(\d+)", re.IGNORECASE)
_GRAD_NORM_RE = re.compile(r"\|dE/dxyz\|\s*=\s*([-+0-9.EeDd]+)", re.IGNORECASE)
_RICC2_TM8_STATE_ROW = re.compile(
    r"^\s*\|\s*([A-Za-z0-9\"']+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*"
    r"([-+]?\d+\.\d+)\s*\|\s*([-+]?\d+\.\d+)\s*\|\s*([-+]?\d+\.?\d*)\s*\|\s*"
    r"([-+]?\d+\.\d+)\s*\|\s*([-+]?\d+\.\d+)\s*\|"
)
_RICC2_LEGACY_STATE_ROW = re.compile(
    r"^\s*(\d+)\s+([A-Za-z0-9\"']+)\s+"
    r"([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*$"
)
_RICC2_STATE_HEADER = re.compile(
    r"symmetry:\s*([A-Za-z0-9\"']+)\s+state:\s*(\d+)",
    re.IGNORECASE,
)
_RICC2_ORB_ROW = re.compile(
    r"^\s*\|\s*(\d+)\s+([A-Za-z0-9\"']+)\s+(\d+)\s*\|"
    r"\s*(\d+)\s+([A-Za-z0-9\"']+)\s+(\d+)\s*\|"
    r"\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*\|"
)


class TurbomoleOutputParser:
    """Parse energy and geometry from a TURBOMOLE working directory."""

    def __init__(self, directory: str = ".", node_runner: Optional[NodeRunner] = None):
        self.directory = directory
        self.node_runner = node_runner
        self.properly_terminated = False
        self.final_energy: Optional[float] = None
        self.final_structure: Optional[Molecule] = None
        self.energies: list[float] = []
        self.vibrational_frequencies: Optional[SimpleTable] = None

    def parse(self) -> None:
        energy_file = os.path.join(self.directory, "energy")
        if os.path.exists(energy_file):
            if self.node_runner:
                self.node_runner.info(f"Parsing energy from {energy_file}")
            with open(energy_file, encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("$") or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            self.final_energy = float(parts[1])
                            self.energies.append(self.final_energy)
                        except ValueError:
                            continue
            if self.final_energy is not None:
                self.properly_terminated = True

        coord_path = Path(self.directory) / "coord"
        if coord_path.exists():
            self.final_structure = parse_coord_file(coord_path)

        vibspectrum_path = Path(self.directory) / "vibspectrum"
        if vibspectrum_path.exists():
            if self.node_runner:
                self.node_runner.info(f"Parsing vibrational frequencies from {vibspectrum_path}")
            self.vibrational_frequencies = parse_vibspectrum_file(vibspectrum_path)


def _parse_fortran_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def parse_energy_history(path: str | Path) -> list[dict]:
    """Parse TURBOMOLE ``energy`` into ``[{step, energy}, ...]``."""
    energy_path = Path(path)
    if not energy_path.is_file():
        return []
    history: list[dict] = []
    for line in energy_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("$"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            step = int(float(parts[0]))
            energy = _parse_fortran_float(parts[1])
        except ValueError:
            continue
        history.append({"step": step, "energy": energy})
    return history


def parse_gradient_history(path: str | Path) -> list[dict]:
    """Parse TURBOMOLE ``gradient`` into ``[{step, grad_norm}, ...]``.

    Prefer ``|dE/dxyz|`` on the cycle header. If it is missing, use the RMS of
    Cartesian gradient components in that cycle's block.
    """
    gradient_path = Path(path)
    if not gradient_path.is_file():
        return []

    history: list[dict] = []
    current_step: Optional[int] = None
    current_norm: Optional[float] = None
    current_grads: list[float] = []

    def flush() -> None:
        nonlocal current_step, current_norm, current_grads
        if current_step is None:
            return
        grad_norm = current_norm
        if grad_norm is None and current_grads:
            grad_norm = _rms(current_grads)
        if grad_norm is not None:
            history.append({"step": current_step, "grad_norm": float(grad_norm)})
        current_step = None
        current_norm = None
        current_grads = []

    for line in gradient_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("$") and not stripped.lower().startswith("$grad"):
            flush()
            break
        cycle_match = _CYCLE_HEADER_RE.search(line)
        if cycle_match:
            flush()
            current_step = int(cycle_match.group(1))
            norm_match = _GRAD_NORM_RE.search(line)
            if norm_match:
                try:
                    current_norm = _parse_fortran_float(norm_match.group(1))
                except ValueError:
                    current_norm = None
            continue
        if current_step is None:
            continue
        parts = stripped.split()
        if len(parts) != 3:
            continue
        try:
            current_grads.extend(_parse_fortran_float(part) for part in parts)
        except ValueError:
            continue
    flush()
    return history


def parse_coord_file(coord_path: Path) -> Optional[Molecule]:
    molecule = Molecule()
    in_coord = False
    with open(coord_path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("$coord"):
                in_coord = True
                continue
            if stripped.startswith("$") and in_coord:
                break
            if not in_coord or not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            try:
                x = float(parts[0]) * BOHR_TO_ANGSTROM
                y = float(parts[1]) * BOHR_TO_ANGSTROM
                z = float(parts[2]) * BOHR_TO_ANGSTROM
            except ValueError:
                continue
            molecule.add_atom(Atom(element=parts[3].capitalize(), x=x, y=y, z=z))
    if not molecule.atoms:
        return None
    return molecule


def require_turbomole_normal_termination(path: str | Path, program: str) -> None:
    output_path = Path(path)
    if not output_path.is_file():
        raise ValueError(f"Missing {program} output file: {output_path}")
    text = output_path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    if (
        f"{program} ended abnormally" in lowered
        or "ended abnormally" in lowered
        or "abend of dscf" in lowered
    ):
        tail = "\n".join(text.splitlines()[-_PROGRAM_OUTPUT_TAIL_LINES:])
        raise ValueError(
            f"{program} ended abnormally. Last lines of {output_path.name}:\n{tail}"
        )


def parse_ricc2_file(
    path: str | Path,
) -> tuple[float, Optional[SimpleTable], Optional[SimpleTable]]:
    """Parse correlated energy, excited states, and occ→vir amplitudes from ricc2.out."""
    ricc2_path = Path(path)
    if not ricc2_path.is_file():
        raise ValueError(f"Missing ricc2 output file: {ricc2_path}")
    require_turbomole_normal_termination(ricc2_path, "ricc2")
    text = ricc2_path.read_text(encoding="utf-8", errors="replace")
    energy: Optional[float] = None
    final_matches = re.findall(
        r"Final\s+(?:MP2|CC2)\s+energy\s*[:=]\s*([-+0-9.EeDd]+)",
        text,
        flags=re.IGNORECASE,
    )
    method_matches = re.findall(
        r"^\s*(?:MP2|CC2)\s+energy\s*[:=]\s*([-+0-9.EeDd]+)\s*$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if final_matches:
        energy = _parse_fortran_float(final_matches[-1])
    elif method_matches:
        energy = _parse_fortran_float(method_matches[-1])
    if energy is None:
        raise ValueError(f"Failed to parse correlated energy from {ricc2_path}.")

    tm8_states: list[dict] = []
    legacy_states: list[dict] = []
    transitions: list[dict] = []
    current_state: Optional[int] = None
    current_symmetry: Optional[str] = None
    capturing_legacy = False
    for line in text.splitlines():
        tm8 = _RICC2_TM8_STATE_ROW.match(line)
        if tm8:
            tm8_states.append(
                {
                    "state": int(tm8.group(3)),
                    "symmetry": tm8.group(1),
                    "multiplicity": int(tm8.group(2)),
                    "energy_hartree": float(tm8.group(4)),
                    "energy_ev": float(tm8.group(5)),
                    "energy_cm_1": float(tm8.group(6)),
                    "percent_t1": float(tm8.group(7)),
                    "percent_t2": float(tm8.group(8)),
                }
            )
            continue
        header = _RICC2_STATE_HEADER.search(line)
        if header:
            current_symmetry = header.group(1)
            current_state = int(header.group(2))
            continue
        orb = _RICC2_ORB_ROW.match(line)
        if orb and current_state is not None:
            transitions.append(
                {
                    "state": current_state,
                    "symmetry": current_symmetry or "-",
                    "occ_orbital": f"{orb.group(1)} {orb.group(2)}",
                    "occ_index": int(orb.group(3)),
                    "vir_orbital": f"{orb.group(4)} {orb.group(5)}",
                    "vir_index": int(orb.group(6)),
                    "coefficient": float(orb.group(7)),
                    "percent": float(orb.group(8)),
                }
            )
            continue
        lowered = line.lower()
        if "excitation energy" in lowered and "final" not in lowered and not tm8_states:
            capturing_legacy = True
            continue
        if capturing_legacy:
            legacy = _RICC2_LEGACY_STATE_ROW.match(line)
            if legacy:
                legacy_states.append(
                    {
                        "state": int(legacy.group(1)),
                        "symmetry": legacy.group(2),
                        "energy_ev": float(legacy.group(3)),
                        "oscillator_strength": float(legacy.group(5)),
                    }
                )
                continue
            if legacy_states and (
                line.strip().startswith("$") or line.strip().startswith("=")
            ):
                capturing_legacy = False

    state_rows = tm8_states or legacy_states
    states_table: Optional[SimpleTable] = None
    if state_rows:
        states_table = SimpleTable(name="Excited States")
        for column, column_type in (
            ("state", "int"),
            ("symmetry", "str"),
            ("multiplicity", "int"),
            ("energy_hartree", "float"),
            ("energy_ev", "float"),
            ("energy_cm_1", "float"),
            ("percent_t1", "float"),
            ("percent_t2", "float"),
            ("oscillator_strength", "float"),
        ):
            if any(column in row for row in state_rows):
                states_table.add_column(column, column_type)
        for row in state_rows:
            states_table.add_row(row)

    transitions_table: Optional[SimpleTable] = None
    if transitions:
        transitions_table = SimpleTable(name="Excited State Transitions")
        transitions_table.add_column("state", "int")
        transitions_table.add_column("symmetry", "str")
        transitions_table.add_column("occ_orbital", "str")
        transitions_table.add_column("occ_index", "int")
        transitions_table.add_column("vir_orbital", "str")
        transitions_table.add_column("vir_index", "int")
        transitions_table.add_column("coefficient", "float")
        transitions_table.add_column("percent", "float")
        for row in transitions:
            transitions_table.add_row(row)
    return energy, states_table, transitions_table


def parse_vibspectrum_file(path: Path) -> Optional[SimpleTable]:
    content = path.read_text(encoding="utf-8", errors="replace")
    if "$vibrational spectrum" not in content.lower():
        return None
    matches = _VIBSPECTRUM_ROW_RE.findall(content)
    if not matches:
        return None
    table = SimpleTable(name="Vibrational Frequencies")
    table.add_column("mode", "int")
    table.add_column("frequency_cm_1", "float")
    table.add_column("symmetry", "str")
    table.add_column("ir_intensity_km_mol", "float")
    for mode, symmetry, frequency, intensity, _ir_selection, _raman_selection in matches:
        table.add_row(
            {
                "mode": int(mode),
                "frequency_cm_1": float(frequency),
                "symmetry": symmetry or "-",
                "ir_intensity_km_mol": float(intensity),
            }
        )
    if not table.row:
        return None
    return table


def write_final_geometry_xyz(molecule: Optional[Molecule], path: str = "final_geometry.xyz") -> Optional[str]:
    if molecule is None or not molecule.atoms:
        return None
    lines = [str(len(molecule.atoms)), "final geometry from turbomole2"]
    for atom in molecule.atoms:
        lines.append(f"{atom.element} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
