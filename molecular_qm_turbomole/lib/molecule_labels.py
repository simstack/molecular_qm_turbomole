"""Fill Molecule.smiles / formula via molecular_qm_util (Open Babel)."""

from typing import Any, Optional


def _is_missing_label(value: Any) -> bool:
    text = str(value or "").strip()
    return not text or text.lower().startswith("error")


def _usable_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower().startswith("error"):
        return None
    return text


def fill_molecule_labels(molecule: Any) -> Any:
    """Compute SMILES and formula when missing or previously recorded as util-missing."""
    if molecule is None:
        return molecule
    if _is_missing_label(getattr(molecule, "smiles", None)) and hasattr(molecule, "make_smiles"):
        computed = molecule.make_smiles()
        if _usable_label(computed) is not None:
            molecule.smiles = computed
    if _is_missing_label(getattr(molecule, "formula", None)) and hasattr(molecule, "make_formula"):
        computed = molecule.make_formula()
        if _usable_label(computed) is not None:
            molecule.formula = computed
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
