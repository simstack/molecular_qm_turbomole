from datetime import datetime

from hyperpolarizibility.workflows import HyperPolarizationRecord, beta_zzz_by_pair
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import StringData
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType
from simstack.models.table_artifact import AGGridColumnDef, TableArtifactModel


def _functional_label(record: HyperPolarizationRecord) -> str:
    functional = getattr(record, "functional", None)
    value = getattr(functional, "functional", None)
    if value is None:
        return "N/A"
    return str(getattr(value, "value", value))


def _basis_label(record: HyperPolarizationRecord) -> str:
    basis_set = getattr(record, "basis_set", None)
    value = getattr(basis_set, "basis_set", None)
    return str(value) if value else "N/A"


@node
async def hyperpolarization_records_to_table(
    date_info: StringData, **kwargs
) -> SimstackResult:
    """
    Convert stored HyperPolarizationRecord documents into a table of beta_zzz values.

    Parameters:
        date_info (StringData): Optional ISO datetime used only for logging.

    SimstackResult:
        table_artifact (TableArtifactModel): Grid view of the records.
        simple_table (SimpleTable): Same rows as a SimpleTable.
    """
    node_runner: NodeRunner = kwargs["node_runner"]

    try:
        dt = datetime.fromisoformat(date_info.value)
        node_runner.info(f"Processing records for date: {dt}")
    except (ValueError, TypeError) as exc:
        node_runner.info(f"Invalid date format: {date_info.value}. Error: {exc}")

    records = await context.db.find(HyperPolarizationRecord)
    node_runner.info(f"Found {len(records)} HyperPolarizationRecords.")

    columns = [
        AGGridColumnDef(field="started_at", headerName="Started At", sortable=True),
        AGGridColumnDef(field="molecule_smiles", headerName="Molecule (SMILES)"),
        AGGridColumnDef(field="functional", headerName="Functional"),
        AGGridColumnDef(field="basis_set", headerName="Basis Set"),
        AGGridColumnDef(field="beta_pair_1_zzz_1e30_esu", headerName="beta_pair_1_zzz_1e30_esu"),
        AGGridColumnDef(field="beta_pair_2_zzz_1e30_esu", headerName="beta_pair_2_zzz_1e30_esu"),
        AGGridColumnDef(field="beta_pair_3_zzz_1e30_esu", headerName="beta_pair_3_zzz_1e30_esu"),
        AGGridColumnDef(field="success", headerName="Success", sortable=True),
        AGGridColumnDef(field="error", headerName="Error"),
    ]

    row_data = []
    for record in records:
        betas = beta_zzz_by_pair(record.hyperpol)
        row = {
            "started_at": record.started_at.isoformat() if record.started_at else None,
            "molecule_smiles": record.molecule.smiles if record.molecule else "N/A",
            "functional": _functional_label(record),
            "basis_set": _basis_label(record),
            "beta_pair_1_zzz_1e30_esu": betas.get(1),
            "beta_pair_2_zzz_1e30_esu": betas.get(2),
            "beta_pair_3_zzz_1e30_esu": betas.get(3),
            "success": record.success,
            "error": record.error or "",
        }
        node_runner.info(
            f"Record: started_at={row['started_at']}, "
            f"molecule_smiles={row['molecule_smiles']}, "
            f"functional={row['functional']}, "
            f"basis_set={row['basis_set']}, "
            f"beta_pair_1_zzz_1e30_esu={row['beta_pair_1_zzz_1e30_esu']}, "
            f"beta_pair_2_zzz_1e30_esu={row['beta_pair_2_zzz_1e30_esu']}, "
            f"beta_pair_3_zzz_1e30_esu={row['beta_pair_3_zzz_1e30_esu']}, "
            f"success={row['success']}, "
            f"error={row['error']}"
        )
        row_data.append(row)

    table_artifact = TableArtifactModel(
        columns_defs=columns,
        row_data=row_data,
    )
    simple_table = SimpleTable(
        name="Hyperpolarization Results",
        heading=[col.headerName for col in columns],
        row=[
            {col.headerName: row[col.field] for col in columns}
            for row in row_data
        ],
        type=[SimpleTableColumnType.STRING] * len(columns),
    )

    node_runner.table_artifact = table_artifact
    node_runner.simple_table = simple_table
    return node_runner.succeed()
