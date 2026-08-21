import os
from pathlib import Path
from typing import Optional

from molecular_qm_models.molecule import Atom, Molecule
from simstack.core.node_runner import NodeRunner

BOHR_TO_ANGSTROM = 1.0 / 1.8897259886


class TurbomoleOutputParser:
    """Parse energy and geometry from a TURBOMOLE working directory."""

    def __init__(self, directory: str = ".", node_runner: Optional[NodeRunner] = None):
        self.directory = directory
        self.node_runner = node_runner
        self.properly_terminated = False
        self.final_energy: Optional[float] = None
        self.final_structure: Optional[Molecule] = None
        self.energies: list[float] = []

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


def write_final_geometry_xyz(molecule: Optional[Molecule], path: str = "final_geometry.xyz") -> Optional[str]:
    if molecule is None or not molecule.atoms:
        return None
    lines = [str(len(molecule.atoms)), "final geometry from turbomole2"]
    for atom in molecule.atoms:
        lines.append(f"{atom.element} {atom.x:.10f} {atom.y:.10f} {atom.z:.10f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
