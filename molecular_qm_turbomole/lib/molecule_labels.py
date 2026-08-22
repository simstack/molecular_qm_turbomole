"""Fill Molecule.smiles / formula via molecular_qm_util (Open Babel)."""

from typing import Any, Optional


def _is_missing_label(value: Any) -> bool:
    text = str(value or "").strip()
    return not text or text.lower().startswith("error")


def fill_molecule_labels(molecule: Any) -> Any:
    """Compute SMILES and formula when missing or previously recorded as util-missing."""
    if molecule is None:
        return molecule
    if _is_missing_label(getattr(molecule, "smiles", None)) and hasattr(molecule, "make_smiles"):
        molecule.smiles = molecule.make_smiles()
    if _is_missing_label(getattr(molecule, "formula", None)) and hasattr(molecule, "make_formula"):
        molecule.formula = molecule.make_formula()
    return molecule


def molecule_section_name(molecule: Optional[Any]) -> str:
    fill_molecule_labels(molecule)
    if molecule is None:
        return "molecule"
    formula = str(getattr(molecule, "formula", None) or "").strip()
    if formula and not formula.lower().startswith("error"):
        return formula
    smiles = str(getattr(molecule, "smiles", None) or "").strip()
    if smiles and not smiles.lower().startswith("error"):
        return smiles
    return "molecule"
