"""Rewrite stored TURBOMOLE input / hyperpol records after the input shape change.

``TurbomoleBasisSet2``, ``TurbomoleFunctional``, and ``DispersionCorrection`` are
required embedded payloads (not optional Models). Legacy docs nested dispersion
under ``functional`` or stored a chemistry-name string such as ``B3LYP``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bson import ObjectId

from molecular_qm_turbomole.lib.molecule_labels import fill_molecule_labels, molecule_section_name
from molecular_qm_turbomole.models.turbomole_functional import (
    as_turbomole_functional_doc,
)
from molecular_qm_turbomole.models.turbomole_input import (
    TURBOMOLE_DEFAULT_BASIS_SET,
    TurbomoleDispersionCorrection,
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

QM_INPUT2_TOP_LEVEL_KEYS = set(TurbomoleQMInput2.model_fields) | {"_id", "field_name"}
BASIS_SET_KEYS = {"field_name", "basis_set"}
FUNCTIONAL_KEYS = {"field_name", "functional"}
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


def as_functional_model_doc(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        raw = raw.get("functional", raw)
    return as_turbomole_functional_doc(raw)


def as_dispersion_model_doc(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        value = _enum_value(raw.get("value"), "NONE")
    elif raw in (None, ""):
        value = "NONE"
    else:
        value = _enum_value(raw, "NONE")
    return {
        "field_name": TurbomoleDispersionCorrection.__name__,
        "value": value,
    }


def _dispersion_from_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    existing = doc.get("dispersion_correction")
    if isinstance(existing, dict) and existing.get("value") not in (None, ""):
        return as_dispersion_model_doc(existing)
    if existing not in (None, ""):
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
    upgraded["functional"] = as_functional_model_doc(doc.get("functional"))
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
    if isinstance(stripped.get("functional"), dict):
        stripped["functional"] = {
            key: value for key, value in stripped["functional"].items() if key in FUNCTIONAL_KEYS
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
    upgraded["functional"] = as_functional_model_doc(upgraded.get("functional"))
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


def is_hyperpol_runner_dataset(dataset: Any) -> bool:
    metadata = getattr(dataset, "metadata", None)
    if metadata is None:
        return False
    from hyperpolarizibility.hyperpol_runner import HYPERPOL_DATASET_TYPE

    return getattr(metadata, "field_name", None) == HYPERPOL_DATASET_TYPE


def is_hyperpol_aggregate_dataset(dataset: Any) -> bool:
    """True for rebuilt archive datasets, not per-run hyperpol_runner results."""
    from hyperpolarizibility.hyperpol_runner import HYPERPOL_RECORDS_DATASET_NAME

    name = getattr(dataset, "field_name", None)
    return name in {HYPERPOL_RECORDS_DATASET_NAME, "hyperpol_runner.migrated"}


def is_hyperpol_runner_template(template: Any) -> bool:
    from hyperpolarizibility.hyperpol_runner import HYPERPOL_DATASET_TYPE

    return getattr(template, "dataset_type", None) == HYPERPOL_DATASET_TYPE


def _iter_named_references(owner: Any, variable_name: str):
    refs = getattr(owner, "results_references", None)
    if refs is None and isinstance(owner, dict):
        refs = owner.get("results_references")
    for ref in refs or []:
        if isinstance(ref, dict):
            name = ref.get("variable_name")
            reference = ref.get("reference")
        else:
            name = getattr(ref, "variable_name", None)
            reference = getattr(ref, "reference", None)
        if name == variable_name and reference is not None:
            yield reference


def dataset_reference_from_runner(runner: Any) -> Any:
    return next(_iter_named_references(runner, "dataset"), None)


def record_references_from_nodes(nodes: Sequence[Any]) -> List[Any]:
    return [ref for node in nodes for ref in _iter_named_references(node, "record")]


async def delete_hyperpol_runner_datasets(db, dry_run: bool = False) -> Tuple[int, int]:
    """Drop the rebuilt hyperpol_runner archive DataSet and its metadata template.

    Per-run DataSets attached to finished ``hyperpol_runner`` nodes are left in
    place so those node results can still load.
    """
    from simstack.models.dataset import DataSet
    from simstack.models.dataset_metadata import DataSetMetadataTemplate

    deleted_datasets = 0
    for dataset in await db.find(DataSet):
        if not is_hyperpol_aggregate_dataset(dataset):
            continue
        deleted_datasets += 1
        if not dry_run:
            await db.delete(dataset)

    deleted_templates = 0
    for template in await db.find(DataSetMetadataTemplate):
        if not is_hyperpol_runner_template(template):
            continue
        deleted_templates += 1
        if not dry_run:
            await db.delete(template)
    return deleted_datasets, deleted_templates


def build_hyperpol_dataset_from_records(
    records: Sequence[Any],
    *,
    field_name: Optional[str] = None,
    formula: Optional[str] = None,
    dataset_id: Any = None,
):
    """Build a DataSet of the hyperpol_runner row shape without saving it."""
    from hyperpolarizibility.hyperpol_runner import (
        HYPERPOL_DATASET_TYPE,
        HYPERPOL_RECORDS_DATASET_NAME,
        dataset_row_from_record,
    )
    from simstack.models.dataset import DataSet
    from simstack.models.dataset_metadata import DataSetMetadata

    payload: Dict[str, Any] = {
        "field_name": field_name or HYPERPOL_RECORDS_DATASET_NAME,
        "metadata": DataSetMetadata(
            field_name=HYPERPOL_DATASET_TYPE,
            data={"formula": formula or "records"},
        ),
    }
    if dataset_id is not None:
        payload["id"] = dataset_id
    dataset = DataSet(**payload)
    added = 0
    for record in records:
        molecule = fill_molecule_labels(getattr(record, "molecule", None))
        section_name = molecule_section_name(molecule)
        section = dataset[section_name]
        section.add_row(dataset_row_from_record(record), name=record_row_name(getattr(record, "id", None)))
        added += 1
    return dataset, added


async def restore_missing_hyperpol_runner_datasets(db, dry_run: bool = False) -> int:
    """Recreate deleted per-run DataSets using the ObjectIds finished runners still reference."""
    from hyperpolarizibility.hyperpol_runner import HYPERPOL_DATASET_TYPE
    from hyperpolarizibility.hyperpolarization_record import HyperPolarizationRecord
    from simstack.models.dataset import DataSet
    from simstack.models.node_registry import NodeRegistry

    restored = 0
    runners = await db.find(NodeRegistry, NodeRegistry.name == "hyperpol_runner")
    for runner in runners:
        dataset_id = dataset_reference_from_runner(runner)
        if dataset_id is None:
            continue
        existing = await db.find_one(DataSet, DataSet.id == dataset_id)
        if existing is not None:
            continue
        children = await db.find(NodeRegistry, {"parent_ids": runner.id})
        records = []
        for record_id in record_references_from_nodes(children):
            record = await db.find_one(HyperPolarizationRecord, HyperPolarizationRecord.id == record_id)
            if record is not None:
                records.append(record)
        if not records:
            logger.warning(
                "Runner %s references missing DataSet %s and has no child HyperPolarizationRecord docs",
                getattr(runner, "id", None),
                dataset_id,
            )
            continue
        molecule = fill_molecule_labels(getattr(records[0], "molecule", None))
        section_name = molecule_section_name(molecule)
        dataset, added = build_hyperpol_dataset_from_records(
            records,
            field_name=f"{HYPERPOL_DATASET_TYPE}.{section_name}",
            formula=section_name,
            dataset_id=dataset_id,
        )
        if dry_run:
            logger.info(
                "Would restore DataSet %s (%s row(s)) for hyperpol_runner %s",
                dataset_id,
                added,
                getattr(runner, "id", None),
            )
            restored += 1
            continue
        await dataset.save(db)
        if dataset.id != dataset_id:
            raise RuntimeError(
                f"Restored DataSet id {dataset.id} does not match runner reference {dataset_id}"
            )
        logger.info(
            "Restored DataSet %s (%s row(s)) for hyperpol_runner %s",
            dataset.id,
            added,
            getattr(runner, "id", None),
        )
        restored += 1
    return restored


async def migrate_hyperpolarization_records_to_dataset(db, dry_run: bool = False) -> int:
    """Rebuild the archive DataSet and restore missing per-run hyperpol_runner results."""
    from hyperpolarizibility.hyperpol_runner import HYPERPOL_RECORDS_DATASET_NAME
    from hyperpolarizibility.hyperpolarization_record import HyperPolarizationRecord

    deleted_datasets, deleted_templates = await delete_hyperpol_runner_datasets(db, dry_run=dry_run)
    logger.info(
        "%s %s archive hyperpol_runner DataSet(s) and %s metadata template(s)",
        "Would delete" if dry_run else "Deleted",
        deleted_datasets,
        deleted_templates,
    )

    records = await db.find(HyperPolarizationRecord)
    logger.info("Found %s HyperPolarizationRecord document(s) for Dataset migration", len(records))
    added = 0
    if records:
        if not dry_run:
            for record in records:
                molecule = fill_molecule_labels(getattr(record, "molecule", None))
                if molecule is None:
                    continue
                molecule = await db.save(molecule)
                record.molecule = molecule
        dataset, added = build_hyperpol_dataset_from_records(records)
        if dry_run:
            logger.info("Would write %s Dataset row(s) to %s", added, HYPERPOL_RECORDS_DATASET_NAME)
        else:
            await dataset.save(db)
            logger.info("Saved DataSet %s with %s row(s)", HYPERPOL_RECORDS_DATASET_NAME, added)

    restored = await restore_missing_hyperpol_runner_datasets(db, dry_run=dry_run)
    logger.info(
        "%s %s finished hyperpol_runner DataSet(s)",
        "Would restore" if dry_run else "Restored",
        restored,
    )
    return added
