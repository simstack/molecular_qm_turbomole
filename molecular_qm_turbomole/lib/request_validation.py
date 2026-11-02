import math
import re

from molecular_qm_models.dispersion_correction import DispersionCorrectionEnum
from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctionalEnum
from molecular_qm_turbomole.models.turbomole_input import TurbomoleQMInput2

# Published DFT-D3 parameters for these functionals are zero-damping only.
# TURBOMOLE then stops ridft with "functional name unknown in DFT-D3 BJ".
_D3BJ_UNSUPPORTED = frozenset(
    {
        TurbomoleFunctionalEnum.M05,
        TurbomoleFunctionalEnum.M05_2X,
        TurbomoleFunctionalEnum.M06,
        TurbomoleFunctionalEnum.M06_2X,
        TurbomoleFunctionalEnum.M06_L,
        TurbomoleFunctionalEnum.M11_L,
        TurbomoleFunctionalEnum.MN15_L,
    }
)


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


def validate_dispersion_parametrization(qm_input: TurbomoleQMInput2) -> None:
    if qm_input.dispersion_enum() != DispersionCorrectionEnum.D3BJ:
        return
    functional = qm_input.functional_enum()
    if functional not in _D3BJ_UNSUPPORTED:
        return
    raise ValueError(
        f"TURBOMOLE has no DFT-D3(BJ) parameters for {functional.value}. "
        "ridft stops with 'functional name unknown in DFT-D3 BJ'. "
        "Use zero-damping D3 or NONE for this functional."
    )
