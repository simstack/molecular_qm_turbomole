import asyncio
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple

from odmantic import Field, Model
from pydantic import field_validator

from hyperpolarizibility.workflows import (
    HyperpolarizabilitySettings,
    _child_kwargs,
    _copy_qm_input,
    _ensure_db_molecule,
    _extract_hyperpolarizability_table,
    _is_completed,
    beta_zzz_by_pair,
    hyperpolarizibility,
    workflow_parameters,
)
from molecular_qm_models.density_functional import Functional, FunctionalEnum, FunctionalModel
from molecular_qm_models.molecule import Molecule
from molecular_qm_turbomole.lib.hyperpol import hyperpolarizability_wavelength_nm
from molecular_qm_turbomole.models.turbomole_input import (
    TURBOMOLE_BASIS_SET_VALUES,
    HyperpolarizabilityModeEnum,
    TurbomoleBasisSet2,
    TurbomoleQMInput2,
)
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import FloatData, StringData, simstack_model
from simstack.models.dataset import DataSet, DataSetSection
from simstack.models.dataset_metadata import DataSetMetadata
from simstack.models.simple_table import SimpleTable
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema


@simstack_model
class HyperpolRunnerModel(Model):
    field_name: str = "HyperpolRunnerModel"
    functionals: List[FunctionalEnum] = Field(
        default_factory=list,
        json_schema_extra={"description": "Functionals to sweep. Combined with basis_sets."},
    )
    basis_sets: List[str] = Field(
        default_factory=list,
        json_schema_extra={"description": "TURBOMOLE basis sets to sweep. Combined with functionals."},
    )

    @field_validator("basis_sets")
    @classmethod
    def validate_basis_sets(cls, value: List[str]) -> List[str]:
        invalid = [item for item in value if item not in TURBOMOLE_BASIS_SET_VALUES]
        if invalid:
            raise ValueError(f"Unsupported TURBOMOLE basis set(s): {invalid}")
        return value

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        properties = schema.setdefault("properties", {})
        if "basis_sets" in properties:
            properties["basis_sets"]["items"] = {
                "type": "string",
                "enum": list(TURBOMOLE_BASIS_SET_VALUES),
            }
        if "functionals" in properties:
            properties["functionals"]["items"] = {
                "type": "string",
                "enum": [item.value for item in FunctionalEnum],
            }
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        return ui_schema


def _unique(items: Iterable) -> list:
    unique: list = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def _molecule_section_name(molecule: Molecule) -> str:
    formula = getattr(molecule, "formula", None)
    if not formula and hasattr(molecule, "make_formula"):
        formula = molecule.make_formula()
    formula = str(formula or "").strip()
    if formula and not formula.lower().startswith("error"):
        return formula
    smiles = str(getattr(molecule, "smiles", None) or "").strip()
    return smiles or "molecule"


def _settings_from_qm_input(qm_input: TurbomoleQMInput2) -> HyperpolarizabilitySettings:
    mode = qm_input.hyperpolarizability
    if mode == HyperpolarizabilityModeEnum.NONE:
        mode = HyperpolarizabilityModeEnum.STATIC
    return HyperpolarizabilitySettings(
        hyperpolarizability=mode,
        hyperpol_frequency_nm=float(qm_input.hyperpol_frequency_nm or 0.0),
    )


def _functional_for_enum(functional_enum: FunctionalEnum) -> Functional:
    return Functional(functional=functional_enum)


def _qm_input_for_combo(
    base: TurbomoleQMInput2,
    *,
    basis_set: str,
    functional_enum: FunctionalEnum,
) -> TurbomoleQMInput2:
    qm_input = _copy_qm_input(base)
    qm_input.basis_set = TurbomoleBasisSet2(basis_set=basis_set)
    qm_input.functional = _functional_for_enum(functional_enum)
    qm_input.name = f"{base.name}_{functional_enum.value}_{basis_set}"
    return qm_input


def _empty_hyperpol_table() -> SimpleTable:
    table = SimpleTable(name="Hyperpolarizability")
    table.add_column("pair", "int")
    table.add_column("beta_zzz_1e30_esu", "float")
    return table


def _hyperpol_dataset_row(
    *,
    basis_set: str,
    functional: Functional,
    frequency_nm: float,
    hyperpol_table: Optional[SimpleTable],
) -> Dict[str, Model]:
    table = hyperpol_table if hyperpol_table is not None else _empty_hyperpol_table()
    row: Dict[str, Model] = {
        "basis_set": StringData(field_name="basis_set", value=basis_set),
        "functional": FunctionalModel(functional=functional),
        "frequency": FloatData(field_name="frequency", value=float(frequency_nm)),
        "hyperpolarizability": table,
    }
    for pair, value in sorted(beta_zzz_by_pair(table).items()):
        field = f"beta_pair_{pair}_zzz_1e30_esu"
        row[field] = FloatData(field_name=field, value=value)
    return row


def _child_hyperpol_table(result: Any) -> Optional[SimpleTable]:
    table = _extract_hyperpolarizability_table(result)
    if table is not None:
        return table
    record = getattr(result, "record", None)
    record_table = getattr(record, "hyperpol", None) if record is not None else None
    if isinstance(record_table, SimpleTable):
        return record_table
    return None


def _sweep_combos(model: HyperpolRunnerModel) -> List[Tuple[FunctionalEnum, str]]:
    functionals = _unique(model.functionals)
    basis_sets = _unique(model.basis_sets)
    if not functionals:
        raise ValueError("HyperpolRunnerModel.functionals must contain at least one functional.")
    if not basis_sets:
        raise ValueError("HyperpolRunnerModel.basis_sets must contain at least one basis set.")
    return list(product(functionals, basis_sets))


@node(parameters=workflow_parameters)
async def hyperpol_runner(
    qm_input: TurbomoleQMInput2,
    model: HyperpolRunnerModel,
    **kwargs,
) -> SimstackResult:
    """
    Sweep basis sets and functionals on a TurbomoleQMInput2 and collect beta results.

    Parameters:
        qm_input (TurbomoleQMInput2): Template calculation (molecule, solvent, hyperpol mode, grid, ...).
        model (HyperpolRunnerModel): Functionals and basis sets to combine.

    Called Nodes:
        hyperpolarizibility

    SimstackResult:
        dataset (DataSet): One section named by molecular formula; rows are basis, functional, frequency, beta.
    """
    node_runner: NodeRunner = kwargs["node_runner"]

    try:
        combos = _sweep_combos(model)
        molecule = await _ensure_db_molecule(qm_input.molecule)
        if hasattr(molecule, "make_smiles") and not getattr(molecule, "smiles", None):
            molecule.smiles = molecule.make_smiles()
        if hasattr(molecule, "make_formula") and not getattr(molecule, "formula", None):
            molecule.formula = molecule.make_formula()
        qm_input = _copy_qm_input(qm_input)
        qm_input.molecule = molecule

        settings = _settings_from_qm_input(qm_input)
        frequency_nm = hyperpolarizability_wavelength_nm(qm_input)
        section_name = _molecule_section_name(molecule)
        dataset = DataSet(
            field_name=f"hyperpol_runner.{section_name}",
            metadata=DataSetMetadata(
                field_name="hyperpol_runner",
                data={"formula": section_name},
            ),
        )
        section = DataSetSection()
        dataset[section_name] = section

        tasks = []
        for functional_enum, basis_set in combos:
            child_input = _qm_input_for_combo(
                qm_input,
                basis_set=basis_set,
                functional_enum=functional_enum,
            )
            tasks.append(
                hyperpolarizibility(
                    optimization_qm_input=child_input,
                    hyperpolarizability_settings=settings,
                    **_child_kwargs(kwargs),
                )
            )

        node_runner.info(
            f"Starting {len(tasks)} hyperpolarizability jobs "
            f"({len(_unique(model.functionals))} functionals x {len(_unique(model.basis_sets))} basis sets) "
            f"for {section_name}."
        )
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (functional_enum, basis_set), result in zip(combos, results):
            functional = _functional_for_enum(functional_enum)
            if isinstance(result, Exception):
                node_runner.info(
                    f"{functional_enum.value} / {basis_set} failed with exception: {result}"
                )
                row = _hyperpol_dataset_row(
                    basis_set=basis_set,
                    functional=functional,
                    frequency_nm=frequency_nm,
                    hyperpol_table=None,
                )
                row["error"] = StringData(field_name="error", value=str(result))
            else:
                table = _child_hyperpol_table(result)
                row = _hyperpol_dataset_row(
                    basis_set=basis_set,
                    functional=functional,
                    frequency_nm=frequency_nm,
                    hyperpol_table=table,
                )
                if not _is_completed(result):
                    error = getattr(result, "error_message", None) or "hyperpolarizibility did not complete"
                    node_runner.info(f"{functional_enum.value} / {basis_set} did not complete: {error}")
                    row["error"] = StringData(field_name="error", value=str(error))
                else:
                    node_runner.info(f"{functional_enum.value} / {basis_set} completed.")
            section.add_row(row, name=f"{functional_enum.value}_{basis_set}")

        await dataset.save(context.db)
        node_runner.dataset = dataset
        return node_runner.succeed()
    except Exception as exc:
        node_runner.error(str(exc))
        return node_runner.fail(str(exc))
