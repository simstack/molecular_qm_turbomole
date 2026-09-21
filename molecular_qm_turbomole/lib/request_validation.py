import math
import re

from molecular_qm_turbomole.models.turbomole_input import TurbomoleQMInput2


def validate_molecule_geometry(qm_input: TurbomoleQMInput2) -> None:
    atoms = getattr(qm_input.molecule, "atoms", None)
    if not atoms:
        raise ValueError(
            "TURBOMOLE requires a non-empty molecule with 3D coordinates."
        )
    for index, atom in enumerate(atoms, start=1):
        element = str(getattr(atom, "element", "") or "").strip()
        if not re.fullmatch(r"[A-Za-z]{1,3}", element):
            raise ValueError(f"Atom {index} has an invalid element symbol {element!r}.")
        coordinates = (atom.x, atom.y, atom.z)
        try:
            finite = all(math.isfinite(float(value)) for value in coordinates)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(
                f"Atom {index} ({element}) has non-finite coordinates: {coordinates!r}."
            )
