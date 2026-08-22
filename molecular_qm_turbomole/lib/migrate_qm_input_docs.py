"""Rewrite stored TURBOMOLE input / hyperpol records after the input shape change.

``TurbomoleBasisSet2`` and ``DispersionCorrection`` are required embedded
payloads (not optional Models). ``functional`` is a ``FunctionalEnum`` string;
legacy docs nested dispersion under ``functional``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bson import ObjectId

from molecular_qm_turbomole.models.turbomole_input import (
    TURBOMOLE_DEFAULT_BASIS_SET,
    DispersionCorrection,
    TurbomoleBasisSet2,
    TurbomoleQMInput2,
)

logger = logging.getLogger("migrate_qm_input_docs")

QM_INPUT_FIELD_NAMES = frozenset({"TurbomoleQMInput", "TurbomoleQMInput2"})
RECORD_FIELD_NAME = "HyperPolarizationRecord"
KNOWN_FIELD_NAMES = QM_INPUT_FIELD_NAMES | {RECORD_FIELD_NAME}

HYPERPOL_MODE_ALIASES = {
    "off": "none",
    "none": "none",
    "static": "static",
    "dc": "static",
    "dynamic": "dynamic",
}

QM_INPUT2_TOP_LEVEL_KEYS = set(TurbomoleQMInput2.model_fields) | {"_id"}
BASIS_SET_KEYS = {"field_name", "basis_set"}
DISPERSION_KEYS = {"field_name", "value"}


def _nested_id(payload: Dict[str, Any]) -> Optional[ObjectId]:
    value = payload.get("id", payload.get("_id"))
    if isinstance(value, ObjectId):
        return value
    if value in (None, ""):
        return None
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _enum_value(raw: Any, default: str) -> str:
    if raw is None or raw == "":
        return default
    if hasattr(raw, "value"):
        raw = raw.value
    return str(raw)


def as_basis_set_model_doc(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        basis = raw.get("basis_set") or TURBOMOLE_DEFAULT_BASIS_SET
    elif raw in (None, ""):
        basis = TURBOMOLE_DEFAULT_BASIS_SET
    else:
        basis = str(raw)
    return {
        "field_name": TurbomoleBasisSet2.__name__,
        "basis_set": str(basis),
    }


def as_dispersion_model_doc(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        value = _enum_value(raw.get("value"), "NONE")
    elif raw in (None, ""):
        value = "NONE"
    else:
        value = _enum_value(raw, "NONE")
    return {
        "field_name": DispersionCorrection.__name__,
        "value": value,
    }


def _dispersion_from_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    existing = doc.get("dispersion_correction")
    if isinstance(existing, dict) and existing.get("value") not in (None, ""):
        return as_dispersion_model_doc(existing)
    functional = doc.get("functional")
    nested = functional.get("dispersion_correction") if isinstance(functional, dict) else None
    return as_dispersion_model_doc(nested)


def _hyperpolarizability_mode(doc: Dict[str, Any]) -> str:
    current = doc.get("hyperpolarizability")
    if isinstance(current, str):
        mapped = HYPERPOL_MODE_ALIASES.get(current.strip().casefold())
        if mapped:
            return mapped
    mode = doc.get("hyperpolarizability_mode")
    if isinstance(mode, str):
        mapped = HYPERPOL_MODE_ALIASES.get(mode.strip().casefold())
        if mapped:
            return mapped
    frequency = doc.get("hyperpol_frequency_nm")
    if frequency in (None, ""):
        frequency = doc.get("hyperpol_frequency_nm_ui") or 0.0
    try:
        frequency_value = float(frequency or 0.0)
    except (TypeError, ValueError):
        frequency_value = 0.0
    if current is True:
        return "dynamic" if frequency_value > 0.0 else "static"
    return "none"


def _hyperpol_frequency_nm(doc: Dict[str, Any]) -> float:
    def _as_float(raw: Any) -> float:
        if raw in (None, ""):
            return 0.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    stored = _as_float(doc.get("hyperpol_frequency_nm"))
    if stored > 0.0:
        return stored
    return _as_float(doc.get("hyperpol_frequency_nm_ui"))


def _docs_equal(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return left == right


def upgrade_turbomole_qm_input_doc(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Return a TurbomoleQMInput2-shaped document that ODMantic can parse."""
    upgraded = copy.deepcopy(doc)
    upgraded["field_name"] = TurbomoleQMInput2.__name__
    upgraded["basis_set"] = as_basis_set_model_doc(doc.get("basis_set"))
    upgraded["dispersion_correction"] = _dispersion_from_doc(doc)
    functional = doc.get("functional")
    if isinstance(functional, dict):
        upgraded["functional"] = functional.get("functional") or "B3LYP"
    elif hasattr(functional, "functional"):
        upgraded["functional"] = getattr(functional.functional, "value", functional.functional)
    upgraded["hyperpolarizability"] = _hyperpolarizability_mode(doc)
    upgraded["hyperpol_frequency_nm"] = _hyperpol_frequency_nm(doc)
    stripped = {key: value for key, value in upgraded.items() if key in QM_INPUT2_TOP_LEVEL_KEYS}
    if isinstance(stripped.get("basis_set"), dict):
        stripped["basis_set"] = {
            key: value for key, value in stripped["basis_set"].items() if key in BASIS_SET_KEYS
        }
    if isinstance(stripped.get("dispersion_correction"), dict):
        stripped["dispersion_correction"] = {
            key: value
            for key, value in stripped["dispersion_correction"].items()
            if key in DISPERSION_KEYS
        }
    return stripped, not _docs_equal(doc, stripped)


def _ensure_nested_model_id(payload: Any, *, field_name: str) -> Any:
    if not isinstance(payload, dict):
        return payload
    upgraded = dict(payload)
    if _nested_id(upgraded) is None:
        upgraded["id"] = ObjectId()
    upgraded.setdefault("field_name", field_name)
    if "_id" in upgraded and "id" in upgraded:
        upgraded.pop("_id", None)
    return upgraded


def upgrade_hyperpolarization_record_doc(doc: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    upgraded = copy.deepcopy(doc)
    upgraded["field_name"] = RECORD_FIELD_NAME
    if upgraded.get("basis_set") not in (None, {}):
        upgraded["basis_set"] = as_basis_set_model_doc(upgraded.get("basis_set"))
    else:
        upgraded["basis_set"] = as_basis_set_model_doc(None)
    upgraded["dispersion_correction"] = _dispersion_from_doc(upgraded)
    functional = upgraded.get("functional")
    if isinstance(functional, dict):
        upgraded["functional"] = functional.get("functional") or "B3LYP"
    if isinstance(upgraded.get("hyperpol"), dict):
        upgraded["hyperpol"] = _ensure_nested_model_id(
            upgraded["hyperpol"], field_name="SimpleTable"
        )
    return upgraded, not _docs_equal(doc, upgraded)


def upgrade_stored_document(doc: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], bool]:
    field_name = doc.get("field_name")
    if field_name in QM_INPUT_FIELD_NAMES:
        return upgrade_turbomole_qm_input_doc(doc)
    if field_name == RECORD_FIELD_NAME:
        return upgrade_hyperpolarization_record_doc(doc)
    return None, False


def known_collections() -> List[str]:
    names = [
        getattr(TurbomoleQMInput2, "__collection__", None),
        "turbomole_qm_input",
        "turbomole_qm_input2",
        "hyper_polarization_record",
        "turbomole_basis_set2",
        "dispersion_correction",
    ]
    return [name for name in names if name]


def collections_with_legacy_docs(db, extra_names: Sequence[str] = ()) -> List[str]:
    names = []
    seen = set()
    candidates = list(known_collections())
    candidates.extend(extra_names)
    try:
        candidates.extend(db.list_collection_names())
    except Exception as exc:
        logger.warning("Could not list collections: %s", exc)
    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        collection = db[name]
        try:
            hit = collection.find_one({"field_name": {"$in": list(KNOWN_FIELD_NAMES)}})
        except Exception:
            continue
        if hit is not None:
            names.append(name)
    return names


def upgrade_collection(collection, dry_run: bool = False) -> int:
    query = {"field_name": {"$in": list(KNOWN_FIELD_NAMES)}}
    count = 0
    copied_to_input2 = 0
    db = collection.database
    input2_name = TurbomoleQMInput2.__collection__
    for doc in collection.find(query):
        upgraded, changed = upgrade_stored_document(doc)
        if upgraded is None:
            continue
        needs_copy = (
            upgraded.get("field_name") == TurbomoleQMInput2.__name__
            and collection.name != input2_name
        )
        if not changed and not needs_copy:
            continue
        count += 1
        if dry_run:
            continue
        if changed:
            collection.replace_one({"_id": doc["_id"]}, upgraded)
        if needs_copy:
            payload = copy.deepcopy(upgraded)
            db[input2_name].replace_one({"_id": payload["_id"]}, payload, upsert=True)
            copied_to_input2 += 1
        if count % 100 == 0:
            logger.info("Migrated %s documents in %s", count, collection.name)
    if copied_to_input2:
        logger.info(
            "Copied %s converted TurbomoleQMInput document(s) into %s",
            copied_to_input2,
            input2_name,
        )
    return count


def upgrade(db, dry_run: bool = False) -> int:
    total = 0
    collections = collections_with_legacy_docs(db)
    if not collections:
        logger.info("No TurbomoleQMInput / HyperPolarizationRecord collections found")
        return 0
    for name in collections:
        n = upgrade_collection(db[name], dry_run=dry_run)
        logger.info(
            "%s %s: %s document(s)",
            "Would migrate" if dry_run else "Migrated",
            name,
            n,
        )
        total += n
    logger.info("BSON migration completed. Total %s: %s", "matched" if dry_run else "migrated", total)
    return total


def record_row_name(record_id: Any) -> str:
    return str(record_id)


async def migrate_hyperpolarization_records_to_dataset(db, dry_run: bool = False) -> int:
    """Rebuild a DataSet from HyperPolarizationRecord docs so they appear in the Dataset UI."""
    from hyperpolarizibility.hyperpol_runner import (
        _hyperpol_dataset_row,
        _molecule_section_name,
    )
    from hyperpolarizibility.workflows import HyperPolarizationRecord
    from molecular_qm_models.density_functional import Functional, FunctionalEnum
    from simstack.models import StringData
    from simstack.models.dataset import DataSet
    from simstack.models.dataset_metadata import DataSetMetadata

    records = await db.find(HyperPolarizationRecord)
    logger.info("Found %s HyperPolarizationRecord document(s) for Dataset migration", len(records))
    if not records:
        return 0

    dataset = DataSet(
        field_name="hyperpol_runner.migrated",
        metadata=DataSetMetadata(
            field_name="hyperpol_runner",
            data={"formula": "migrated"},
        ),
    )
    existing = await db.find_one(DataSet, DataSet.field_name == "hyperpol_runner.migrated")
    if existing is not None:
        dataset.id = existing.id

    added = 0
    for record in records:
        molecule = record.molecule
        section_name = _molecule_section_name(molecule) if molecule is not None else "molecule"
        section = dataset[section_name]
        basis = getattr(record.basis_set, "basis_set", None) or TURBOMOLE_DEFAULT_BASIS_SET
        functional_value = record.functional
        if isinstance(functional_value, Functional):
            functional = functional_value
        elif isinstance(functional_value, FunctionalEnum):
            functional = Functional(functional=functional_value)
        else:
            functional = Functional()
        row = _hyperpol_dataset_row(
            basis_set=str(basis),
            functional=functional,
            frequency_nm=0.0,
            hyperpol_table=record.hyperpol,
        )
        if record.error:
            row["error"] = StringData(field_name="error", value=str(record.error))
        section.add_row(row, name=record_row_name(record.id))
        added += 1

    if dry_run:
        logger.info("Would write %s Dataset row(s) to hyperpol_runner.migrated", added)
        return added
    await dataset.save(db)
    logger.info("Saved DataSet hyperpol_runner.migrated with %s row(s)", added)
    return added
