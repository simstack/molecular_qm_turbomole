import logging
import os
from typing import Optional

from molecular_qm_turbomole.models.turbomole_input import TurbomoleQMInput2

logger = logging.getLogger(__name__)

ANGSTROM_TO_BOHR = 1.8897259886
DEFAULT_TURBOMOLE_RI_MEMORY_MB = 1000

TURBOMOLE_BASIS_NAME_MAPPING = {
    "x2c-SV(P)": "x2c-SV(P)all",
    "x2c-SVP": "x2c-SVPall",
    "x2c-TZVP": "x2c-TZVPall",
    "x2c-TZVPP": "x2c-TZVPPall",
}


def ri_memory_mb() -> int:
    raw_value = os.environ.get("SIMSTACK_TURBOMOLE_RI_MEMORY_MB", "").strip()
    if raw_value:
        try:
            parsed = int(raw_value)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_TURBOMOLE_RI_MEMORY_MB


def tm_basis_name(raw_basis: str) -> str:
    mapping = {
        "STO3G": "sto-3g",
        "STO6G": "sto-6g",
        "DEF2-SVP": "def2-SVP",
        "DEF2-TZVP": "def2-TZVP",
        "DEF2-QZVP": "def2-QZVP",
        "CC-PVDZ": "cc-pVDZ",
        "CC-PVTZ": "cc-pVTZ",
        "CC-PVQZ": "cc-pVQZ",
        "AUG-CC-PVDZ": "aug-cc-pVDZ",
        "AUG-CC-PVTZ": "aug-cc-pVTZ",
    }
    if raw_basis in TURBOMOLE_BASIS_NAME_MAPPING:
        return TURBOMOLE_BASIS_NAME_MAPPING[raw_basis]
    return mapping.get(raw_basis.upper(), raw_basis)


def tm_functional_name(raw_functional: str) -> str:
    mapping = {
        "PBE": "pbe",
        "BLYP": "b-lyp",
        "BP86": "b-p",
        "B3LYP": "b3-lyp",
        "PBE0": "pbe0",
        "TPSS": "tpss",
        "TPSSH": "tpssh",
        "M06": "m06",
        "M06-2X": "m06-2x",
        "CAM-B3LYP": "cam-b3lyp",
        "ωB97X-D": "wb97x-d",
        "WB97X-D": "wb97x-d",
    }
    mapped = mapping.get(raw_functional.upper())
    if mapped is None:
        raise ValueError(f"Unsupported functional: {raw_functional}")
    return mapped


def tm_dispersion_name(raw_dispersion: str) -> Optional[str]:
    mapping = {
        "NONE": None,
        "D2": "d2",
        "D3": "d3",
        "D3BJ": "bj",
        "D4": "d4",
        "NL": "nl",
    }
    return mapping.get(raw_dispersion.upper(), raw_dispersion.lower())


def should_use_ri(qm_input: TurbomoleQMInput2) -> bool:
    return bool(str(qm_input.basis_set.basis_set).strip())


class TurbomoleInputWriter:
    """Write coord + define.inp for turbomole2."""

    def __init__(self, qm_input: TurbomoleQMInput2):
        self.qm_input = qm_input

    def write_coord(self, path: str = "coord") -> None:
        lines = ["$coord"]
        for atom in self.qm_input.molecule.atoms:
            x = atom.x * ANGSTROM_TO_BOHR
            y = atom.y * ANGSTROM_TO_BOHR
            z = atom.z * ANGSTROM_TO_BOHR
            lines.append(f"{x:20.14f} {y:20.14f} {z:20.14f} {atom.element.lower()}")
        lines.append("$end")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def write_define_input(self, path: str = "define.inp") -> None:
        use_ri = should_use_ri(self.qm_input)
        basis_name = tm_basis_name(str(self.qm_input.basis_set.basis_set))
        functional_name = tm_functional_name(self.qm_input.functional.functional.value)
        dispersion = tm_dispersion_name(
            self.qm_input.functional.dispersion_correction.value.value
        )
        memory_mb = ri_memory_mb()

        lines = [
            f"\n{self.qm_input.name}",
            "a coord",
        ]
        if self.qm_input.use_desy:
            lines.append("desy")
        lines.extend(
            [
                "*",
                "no",
                f"b all {basis_name}",
                "*",
                "eht",
                "",
                str(self.qm_input.charge),
            ]
        )
        if self.qm_input.multiplicity < 3:
            lines.extend(["", "", ""])
        else:
            lines.extend(["n", f"u {self.qm_input.multiplicity - 1}", "*", ""])

        lines.extend(
            ["scf", "iter", "100", "conv", str(int(self.qm_input.scfconv)), ""]
        )
        lines.extend(
            [
                "dft",
                "on",
                f"func {functional_name}",
                f"grid {self.qm_input.gridsize}",
                "",
            ]
        )
        logger.info(
            "Configured TURBOMOLE functional=%s grid=%s scfconv=%s",
            functional_name,
            self.qm_input.gridsize,
            self.qm_input.scfconv,
        )

        if dispersion:
            lines.extend(["dsp", dispersion, ""])

        lines.append("ri")
        if use_ri:
            lines.extend(["on", f"m {memory_mb}"])
        else:
            lines.append("off")
        lines.append("")

        if self.qm_input.states > 0:
            lines.extend(["ex", "a " + str(self.qm_input.states), "*"])

        lines.append("*")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def write_files(self) -> None:
        self.write_coord()
        self.write_define_input()
