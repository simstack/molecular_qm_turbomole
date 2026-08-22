from datetime import datetime

from hyperpolarizibility.workflows import HyperPolarizationRecord, beta_zzz_by_pair
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import Parameters, StringData
from simstack.models.simple_table import SimpleTable

# Host-side DB dump: int-nano does not install the hyperpolarizibility package.
_table_parameters = Parameters(resource="self", queue="default", force_rerun=True)


def _scalar_label(value, nested_attr: str) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, dict):
        nested = value.get(nested_attr)
        if nested is not None:
            value = nested
    else:
        nested = getattr(value, nested_attr, None)
        if nested is not None:
            value = nested
    text = str(getattr(value, "value", value) or "").strip()
    return text or "N/A"


def _functional_label(record: HyperPolarizationRecord) -> str:
    return _scalar_label(getattr(record, "functional", None), "functional")


def _basis_label(record: HyperPolarizationRecord) -> str:
    return _scalar_label(getattr(record, "basis_set", None), "basis_set")


@node(parameters=_table_parameters)
async def hyperpolarization_records_to_table(
    date_info: StringData, **kwargs
) -> SimstackResult:
    """
    Convert stored HyperPolarizationRecord documents into a table of beta_zzz values.

    Runs on the host (resource self). This node only reads MongoDB; it does not
    need TURBOMOLE or int-nano.

    Parameters:
        date_info (StringData): Optional ISO datetime used only for logging.

    SimstackResult:
        simple_table (SimpleTable): Rows of stored hyperpolarization records.
    """
    node_runner: NodeRunner = kwargs["node_runner"]

    try:
        dt = datetime.fromisoformat(date_info.value)
        node_runner.info(f"Processing records for date: {dt}")
    except (ValueError, TypeError) as exc:
        node_runner.info(f"Invalid date format: {date_info.value}. Error: {exc}")

    records = await context.db.find(HyperPolarizationRecord)
    node_runner.info(f"Found {len(records)} HyperPolarizationRecords.")

    simple_table = SimpleTable(name="Hyperpolarization Results")
    simple_table.add_column("Started At", "string")
    simple_table.add_column("Molecule (SMILES)", "string")
    simple_table.add_column("Functional", "string")
    simple_table.add_column("Basis Set", "string")
    simple_table.add_column("beta_pair_1_zzz_1e30_esu", "float")
    simple_table.add_column("beta_pair_2_zzz_1e30_esu", "float")
    simple_table.add_column("beta_pair_3_zzz_1e30_esu", "float")
    simple_table.add_column("Success", "string")
    simple_table.add_column("Error", "string")

    for record in records:
        betas = beta_zzz_by_pair(record.hyperpol)
        row = {
            "Started At": record.started_at.isoformat() if record.started_at else None,
            "Molecule (SMILES)": record.molecule.smiles if record.molecule else "N/A",
            "Functional": _functional_label(record),
            "Basis Set": _basis_label(record),
            "beta_pair_1_zzz_1e30_esu": betas.get(1),
            "beta_pair_2_zzz_1e30_esu": betas.get(2),
            "beta_pair_3_zzz_1e30_esu": betas.get(3),
            "Success": record.success,
            "Error": record.error or "",
        }
        node_runner.info(
            f"Record: started_at={row['Started At']}, "
            f"molecule_smiles={row['Molecule (SMILES)']}, "
            f"functional={row['Functional']}, "
            f"basis_set={row['Basis Set']}, "
            f"beta_pair_1_zzz_1e30_esu={row['beta_pair_1_zzz_1e30_esu']}, "
            f"beta_pair_2_zzz_1e30_esu={row['beta_pair_2_zzz_1e30_esu']}, "
            f"beta_pair_3_zzz_1e30_esu={row['beta_pair_3_zzz_1e30_esu']}, "
            f"success={row['Success']}, "
            f"error={row['Error']}"
        )
        simple_table.add_row(row)

    node_runner.simple_table = simple_table
    return node_runner.succeed()
