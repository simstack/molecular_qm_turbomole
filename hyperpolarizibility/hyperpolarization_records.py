from datetime import datetime

from applications.electronic_structure.hyperpolarizibility.workflows import HyperPolarizationRecord
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import StringData
from simstack.models.simple_table import SimpleTable, SimpleTableColumnType
from simstack.models.table_artifact import AGGridColumnDef, TableArtifactModel


@node
async def hyperpolarization_records_to_table(
    date_info: StringData, **kwargs
) -> SimstackResult:
    """
    Converts HyperPolarizationRecords into a TableArtifactModel.

    Args:
        date_info (StringData): A string representing a datetime (e.g., ISO format).
        kwargs (dict): Additional keyword arguments, including 'node_runner'.

    Returns:
        SimstackResult: A result containing the generated TableArtifactModel.
    """
    node_runner: NodeRunner = kwargs["node_runner"]

    try:
        # 1. Convert StringData to datetime
        # If the value is empty or invalid, this might raise an exception.
        # Handled by the try-except block.
        dt = datetime.fromisoformat(date_info.value)
        node_runner.log(f"Processing records for date: {dt}")
    except (ValueError, TypeError) as e:
        node_runner.log(f"Invalid date format: {date_info.value}. Error: {e}")
        # We continue anyway if the date is just for logging/filtering,
        # but the prompt implies it should be converted.
        # If the intention was to filter by date, we'd use it in the query.
        pass

    # 2. Query all HyperPolarizationRecords
    records = await context.db.find(HyperPolarizationRecord)
    node_runner.log(f"Found {len(records)} HyperPolarizationRecords.")

    # 3. Create TableArtifactModel
    columns = [
        AGGridColumnDef(field="started_at", headerName="Started At", sortable=True),
        AGGridColumnDef(field="molecule_smiles", headerName="Molecule (SMILES)"),
        AGGridColumnDef(field="functional", headerName="Functional"),
        AGGridColumnDef(field="basis_set", headerName="Basis Set"),
        AGGridColumnDef(field="beta_static_zzz_esu", headerName="beta_static_zzz_esu"),
        AGGridColumnDef(field="beta_dynamic_1_zzz_esu", headerName="beta_dynamic_1_zzz_esu"),
        AGGridColumnDef(field="beta_dynamic_2_zzz_esu", headerName="beta_dynamic_2_zzz_esu"),
        AGGridColumnDef(field="beta_static_parallel_esu", headerName="beta_static_parallel_esu"),
        AGGridColumnDef(field="beta_dynamic_1_parallel_esu", headerName="beta_dynamic_1_parallel_esu"),
        AGGridColumnDef(field="beta_dynamic_2_parallel_esu", headerName="beta_dynamic_2_parallel_esu"),
        AGGridColumnDef(field="success", headerName="Success", sortable=True),
        AGGridColumnDef(field="error", headerName="Error"),
    ]

    row_data = []
    for record in records:
        row = {
            "started_at": record.started_at.isoformat() if record.started_at else None,
            "molecule_smiles": record.molecule.smiles if record.molecule else "N/A",
            "functional": record.functional.functional if record.functional else "N/A",
            "basis_set": record.basis_set.basis_set if record.basis_set else "N/A",
            "beta_static_zzz_esu": record.beta_tensor.beta_static_zzz_esu if record.beta_tensor else None,
            "beta_dynamic_1_zzz_esu": record.beta_tensor.beta_dynamic_1_zzz_esu if record.beta_tensor else None,
            "beta_dynamic_2_zzz_esu": record.beta_tensor.beta_dynamic_2_zzz_esu if record.beta_tensor else None,
            "beta_static_parallel_esu": record.beta_tensor.beta_static_parallel_esu if record.beta_tensor else None,
            "beta_dynamic_1_parallel_esu": record.beta_tensor.beta_dynamic_1_parallel_esu if record.beta_tensor else None,
            "beta_dynamic_2_parallel_esu": record.beta_tensor.beta_dynamic_2_parallel_esu if record.beta_tensor else None,
            "success": record.success,
            "error": record.error or "",
        }
        node_runner.log(
            f"Record: started_at={row['started_at']}, "
            f"molecule_smiles={row['molecule_smiles']}, "
            f"functional={row['functional']}, "
            f"basis_set={row['basis_set']}, "
            f"beta_static_zzz_esu={row['beta_static_zzz_esu']}, "
            f"beta_dynamic_1_zzz_esu={row['beta_dynamic_1_zzz_esu']}, "
            f"beta_dynamic_2_zzz_esu={row['beta_dynamic_2_zzz_esu']}, "
            f"beta_static_parallel_esu={row['beta_static_parallel_esu']}, "
            f"beta_dynamic_1_parallel_esu={row['beta_dynamic_1_parallel_esu']}, "
            f"beta_dynamic_2_parallel_esu={row['beta_dynamic_2_parallel_esu']}, "
            f"success={row['success']}, "
            f"error={row['error']}"
        )
        row_data.append(row)

    table_artifact = TableArtifactModel(
        columns_defs=columns,
        row_data=row_data,
    )


    # For now, return a SimpleTable as requested, but keep TableArtifactModel logic above
    simple_table = SimpleTable(
        name="Hyperpolarization Results",
        heading=[col.headerName for col in columns],
        row=[
            {col.headerName: row[col.field] for col in columns}
            for row in row_data
        ],
        type=[SimpleTableColumnType.STRING] * len(columns)  # Simplified types
    )

    node_runner.table_artifact = table_artifact
    node_runner.simple_table = simple_table

    return node_runner.succeed()
