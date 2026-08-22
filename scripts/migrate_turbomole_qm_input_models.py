"""Migrate stored TURBOMOLE QM inputs and hyperpolarizability records.

Rewrites nested EmbeddedModel payloads (basis set, dispersion) into Model
documents with ``id`` so they still parse in the UI, copies legacy
``TurbomoleQMInput`` docs into ``turbomole_qm_input2``, and rebuilds a Dataset
from ``HyperPolarizationRecord``.

Uses the database named in simstack.toml. Does not touch test_database.

    uv run python scripts/migrate_turbomole_qm_input_models.py
    uv run python scripts/migrate_turbomole_qm_input_models.py --dry-run
    uv run python scripts/migrate_turbomole_qm_input_models.py --skip-dataset
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import pymongo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simstack.util.toml_reader import TomlReader  # noqa: E402

from molecular_qm_turbomole.lib.migrate_qm_input_docs import (  # noqa: E402
    migrate_hyperpolarization_records_to_dataset,
    upgrade,
)


logger = logging.getLogger("migration_runner")


def _mongo_db():
    toml_reader = TomlReader(PROJECT_ROOT)
    connection_string = toml_reader.get("parameters.db.connection_string")
    db_name = toml_reader.get("parameters.db.database")
    if not connection_string or not db_name:
        raise RuntimeError("Database connection string or database name not found in simstack.toml")
    client = pymongo.MongoClient(connection_string)
    client.admin.command("ping")
    return client, client[db_name]


async def _migrate_dataset(dry_run: bool) -> int:
    from simstack.core.context import context

    await context.initialize()
    return await migrate_hyperpolarization_records_to_dataset(context.db, dry_run=dry_run)


def run_migration(*, dry_run: bool = False, skip_dataset: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    client, db = _mongo_db()
    try:
        logger.info("Connected to database %s%s", db.name, " (dry-run)" if dry_run else "")
        bson_count = upgrade(db, dry_run=dry_run)
        logger.info("BSON rewrite %s %s document(s)", "would change" if dry_run else "changed", bson_count)
        if skip_dataset:
            return
        dataset_count = asyncio.run(_migrate_dataset(dry_run))
        logger.info(
            "Dataset migration %s %s row(s)",
            "would write" if dry_run else "wrote",
            dataset_count,
        )
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count matching documents without writing",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Only rewrite QM input / record BSON; do not rebuild the migrated DataSet",
    )
    args = parser.parse_args()
    try:
        run_migration(dry_run=args.dry_run, skip_dataset=args.skip_dataset)
    except Exception as exc:
        logger.exception("Migration failed: %s", exc)
        sys.exit(1)
