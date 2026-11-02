import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from odmantic import EmbeddedModel, Field, Model
from pydantic import field_validator, model_validator

from hyperpolarizibility.hyperpol_runner import _hyperpol_dataset_row
from hyperpolarizibility.hyperpolarization_record import HyperPolarizationRecord2
from hyperpolarizibility.workflows import (
    HyperpolarizabilitySettings,
    WorkflowFailure,
    _build_hyperpol_input,
    _child_kwargs,
    _copy_qm_input,
    _ensure_db_molecule,
    _extract_hyperpolarizability_table,
    _extract_qm_result,
    _has_hyperpol_output,
    _is_completed,
    _result_debug,
    _run_turbomole_inline,
    child_exception_text,
    optimize_geometry_m3_m5,
    workflow_parameters,
)
from molecular_qm_models.dispersion_correction import DispersionCorrectionEnum
from molecular_qm_models.molecule import Molecule
from molecular_qm_turbomole.lib.molecule_labels import fill_molecule_labels, molecule_section_name
from molecular_qm_turbomole.lib.request_validation import validate_dispersion_parametrization
from molecular_qm_turbomole.models.turbomole_functional import (
    TURBOMOLE_FUNCTIONAL_VALUES,
    TurbomoleFunctional,
    TurbomoleFunctionalEnum,
)
from molecular_qm_turbomole.models.turbomole_input import (
    TURBOMOLE_BASIS_SET_VALUES,
    TurbomoleBasisSet2,
    TurbomoleDispersionCorrection,
    TurbomoleQMInput2,
)
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import StringData, simstack_model
from simstack.models.dataset import DataSet, DataSetSection
from simstack.models.dataset_metadata import DataSetMetadata
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema

HYPERPOL2_DATASET_TYPE = "hyperpolarizibility2"
_DISPERSION_VALUES = [item.value for item in DispersionCorrectionEnum]


def _dispersion_choice(raw: Any) -> DispersionCorrectionEnum:
    if isinstance(raw, dict):
        if raw.get("value") in (None, ""):
            raise ValueError("dispersion_correction.value is required.")
        raw = raw["value"]
    elif not isinstance(raw, str):
        nested = getattr(raw, "value", None)
        if nested in (None, ""):
            raise ValueError("dispersion_correction is required.")
        raw = nested
    if isinstance(raw, DispersionCorrectionEnum):
        return raw
    text = str(raw).strip()
    if not text:
        raise ValueError("dispersion_correction is required.")
    for item in DispersionCorrectionEnum:
        if item.value == text or item.name == text:
            return item
    raise ValueError(f"Unsupported dispersion correction: {raw}")


def _functional_choice(raw: Any) -> TurbomoleFunctionalEnum:
    if raw in (None, ""):
        raise ValueError("functional is required.")
    if isinstance(raw, dict) and raw.get("functional") in (None, ""):
        raise ValueError("functional is required.")
    return TurbomoleFunctionalEnum.coerce(raw)


@simstack_model
class HyperpolFunctionalChoice(EmbeddedModel):
    """Functional and dispersion correction for one hyperpolarizability calculation."""

    field_name: str = "HyperpolFunctionalChoice"
    functional: TurbomoleFunctionalEnum = Field(
        json_schema_extra={
            "enum": TURBOMOLE_FUNCTIONAL_VALUES,
            "title": "Functional",
            "description": "TURBOMOLE density functional for the hyperpolarizability step.",
        },
    )
    dispersion_correction: DispersionCorrectionEnum = Field(
        json_schema_extra={
            "enum": _DISPERSION_VALUES,
            "title": "Dispersion Correction",
            "description": (
                "Dispersion correction for the hyperpolarizability step. "
                "Overwrites the dispersion correction on the QM input."
            ),
        },
    )

    @model_validator(mode="before")
    @classmethod
    def require_functional_and_dispersion(cls, data):
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise ValueError(
                "Each functionals entry must set functional and dispersion_correction."
            )
        data = dict(data)
        data.pop("id", None)
        data.pop("_id", None)
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        if data.get("functional") in (None, ""):
            raise ValueError("Each functionals entry must set functional.")
        if data.get("dispersion_correction") in (None, ""):
            raise ValueError("Each functionals entry must set dispersion_correction.")
        data["functional"] = _functional_choice(data["functional"])
        data["dispersion_correction"] = _dispersion_choice(data["dispersion_correction"])
        return data

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        schema["description"] = (
            "Functional and dispersion correction. This pair overwrites the QM input "
            "for the hyperpolarizability step."
        )
        properties = schema.setdefault("properties", {})
        properties["functional"] = {
            "type": "string",
            "enum": list(TURBOMOLE_FUNCTIONAL_VALUES),
            "title": "Functional",
            "description": "TURBOMOLE density functional for the hyperpolarizability step.",
        }
        properties["dispersion_correction"] = {
            "type": "string",
            "enum": list(_DISPERSION_VALUES),
            "title": "Dispersion Correction",
            "description": (
                "Dispersion correction for the hyperpolarizability step. "
                "Overwrites the dispersion correction on the QM input."
            ),
        }
        required = schema.setdefault("required", [])
        for name in ("functional", "dispersion_correction"):
            if name not in required:
                required.append(name)
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        ui_schema.setdefault("functional", {})["ui:widget"] = "select"
        ui_schema.setdefault("dispersion_correction", {})["ui:widget"] = "select"
        return ui_schema


@simstack_model
class HyperpolarizabilityMethods(Model):
    """Basis sets and functional/dispersion pairs for the hyperpolarizability step."""

    field_name: str = "HyperpolarizabilityMethods"
    functionals: List[HyperpolFunctionalChoice] = Field(
        default_factory=list,
        json_schema_extra={
            "description": (
                "Functional and dispersion-correction pairs for the hyperpolarizability step. "
                "Each pair overwrites the functional and dispersion correction on the QM input. "
                "Combined with basis_sets."
            ),
        },
    )
    basis_sets: List[str] = Field(
        default_factory=list,
        json_schema_extra={
            "description": "TURBOMOLE basis sets for the hyperpolarizability step. Combined with functionals.",
        },
    )

    @field_validator("functionals", mode="before")
    @classmethod
    def reject_functional_without_dispersion(cls, value):
        if not isinstance(value, list):
            raise ValueError(
                "functionals must be a list of functional and dispersion correction pairs."
            )
        for index, item in enumerate(value):
            if isinstance(item, (str, TurbomoleFunctionalEnum)):
                raise ValueError(
                    f"functionals[{index}] must set functional and dispersion_correction."
                )
        return value

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
        properties["functionals"] = {
            "type": "array",
            "title": "Functionals",
            "description": (
                "Functional and dispersion-correction pairs for the hyperpolarizability step. "
                "Each pair overwrites the functional and dispersion correction on the QM input."
            ),
            "items": HyperpolFunctionalChoice.json_schema(),
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        return ui_schema


def _basis_label(basis_set: Any) -> str:
    nested = getattr(basis_set, "basis_set", basis_set)
    text = str(nested or "").strip()
    if not text:
        raise ValueError("hyperpolarizability record is missing a basis set")
    return text


def _functional_enum(functional: Any) -> TurbomoleFunctionalEnum:
    nested = getattr(functional, "functional", functional)
    return TurbomoleFunctionalEnum.coerce(nested)


def _build_hyperpol_method_input(
    base_input: TurbomoleQMInput2,
    optimized_structure: Molecule,
    settings: HyperpolarizabilitySettings,
    *,
    basis_set: str,
    functional_enum: TurbomoleFunctionalEnum,
    dispersion_correction: DispersionCorrectionEnum,
) -> TurbomoleQMInput2:
    dispersion = _dispersion_choice(dispersion_correction)
    hyper_input = _build_hyperpol_input(base_input, optimized_structure, settings)
    hyper_input.basis_set = TurbomoleBasisSet2(basis_set=basis_set)
    hyper_input.functional = TurbomoleFunctional(functional=functional_enum)
    hyper_input.dispersion_correction = TurbomoleDispersionCorrection(value=dispersion)
    validate_dispersion_parametrization(hyper_input)
    hyper_input.name = (
        f"{base_input.name}_hyperpol_{functional_enum.value}_{dispersion.value}_{basis_set}"
    )
    return hyper_input


def _hyperpol_method_combos(
    model: HyperpolarizabilityMethods,
) -> List[Tuple[TurbomoleFunctionalEnum, DispersionCorrectionEnum, str]]:
    choices: List[Tuple[TurbomoleFunctionalEnum, DispersionCorrectionEnum]] = []
    for item in model.functionals:
        pair = (_functional_choice(item.functional), _dispersion_choice(item.dispersion_correction))
        if pair not in choices:
            choices.append(pair)
    basis_sets: List[str] = []
    for basis_set in model.basis_sets:
        if basis_set not in basis_sets:
            basis_sets.append(basis_set)
    if not choices:
        raise ValueError(
            "functionals must contain at least one functional and dispersion correction."
        )
    if not basis_sets:
        raise ValueError("basis_sets must contain at least one basis set.")
    return [
        (functional, dispersion, basis_set)
        for functional, dispersion in choices
        for basis_set in basis_sets
    ]


def dataset_row_from_record2(record: HyperPolarizationRecord2) -> Dict[str, Model]:
    dispersion = _dispersion_choice(record.hyperpolarizability_dispersion_correction)
    row = _hyperpol_dataset_row(
        basis_set=_basis_label(record.hyperpolarizability_basis_set),
        functional=_functional_enum(record.hyperpolarizability_functional),
        frequency_nm=float(record.wavelength or 0.0),
        hyperpol_table=record.hyperpol,
        error=record.error,
    )
    row["dispersion_correction"] = StringData(
        field_name="dispersion_correction",
        value=dispersion.value,
    )
    row["optimization_basis_set"] = StringData(
        field_name="optimization_basis_set",
        value=_basis_label(record.optimization_basis_set),
    )
    row["optimization_functional"] = StringData(
        field_name="optimization_functional",
        value=_functional_enum(record.optimization_functional).value,
    )
    return row


@node(parameters=workflow_parameters)
async def hyperpolarizibility2(
    optimization_qm_input: TurbomoleQMInput2,
    hyperpolarizability_settings: HyperpolarizabilitySettings,
    hyperpolarizability_methods: HyperpolarizabilityMethods,
    **kwargs,
) -> SimstackResult:
    """
    Optimize once with the QMInput m3/m5 frequency protocol, then compute
    hyperpolarizability for each functional, dispersion correction, and basis
    combination. Each functionals entry overwrites the QM input functional and
    dispersion correction for the hyperpolarizability step. Each job writes a
    HyperPolarizationRecord2 with both the optimization method and the
    hyperpolarizability method.

    Parameters:
        optimization_qm_input (TurbomoleQMInput2): Ground-state optimization settings and molecule.
        hyperpolarizability_settings (HyperpolarizabilitySettings): Static or dynamic beta, plus frequency_tolerance.
        hyperpolarizability_methods (HyperpolarizabilityMethods): Functional/dispersion pairs and basis sets for the hyperpolarizability step.

    Called Nodes:
        turbomole2

    SimstackResult:
        dataset (DataSet): Rows with optimization and hyperpolarizability method settings plus beta.
    """
    node_runner: NodeRunner = kwargs["node_runner"]

    try:
        if optimization_qm_input.states > 0:
            raise ValueError("Workflow expects ground-state path for optimization. Please set states=0.")
        combos = _hyperpol_method_combos(hyperpolarizability_methods)
        molecule = fill_molecule_labels(await _ensure_db_molecule(optimization_qm_input.molecule))
        molecule = await context.db.save(molecule)
        optimization_qm_input = _copy_qm_input(optimization_qm_input)
        optimization_qm_input.molecule = molecule

        optimization_functional = TurbomoleFunctional(
            functional=optimization_qm_input.functional.functional
        )
        optimization_dispersion = TurbomoleDispersionCorrection(
            value=optimization_qm_input.dispersion_correction.value
        )
        optimization_basis_set = TurbomoleBasisSet2(
            basis_set=optimization_qm_input.basis_set.basis_set
        )
        wavelength = float(hyperpolarizability_settings.hyperpol_frequency_nm or 0.0)
        record_wavelength = wavelength if wavelength > 0.0 else None
        started_at = datetime.now(timezone.utc)
        grids_used: List[str] = []

        node_runner.info(
            f"Running hyperpolarizibility2 '{optimization_qm_input.name}' "
            f"with {len(combos)} hyperpolarizability method(s)."
        )
        optimized_structure = await optimize_geometry_m3_m5(
            optimization_qm_input,
            hyperpolarizability_settings.frequency_tolerance,
            node_runner,
            _child_kwargs(kwargs),
            grids_used,
        )

        tasks = []
        for functional_enum, dispersion_enum, basis_set in combos:
            hyper_input = _build_hyperpol_method_input(
                optimization_qm_input,
                optimized_structure,
                hyperpolarizability_settings,
                basis_set=basis_set,
                functional_enum=functional_enum,
                dispersion_correction=dispersion_enum,
            )
            custom_name = (
                f"hyper_{functional_enum.value}_{dispersion_enum.value}_{basis_set}"
                .replace("(", "")
                .replace(")", "")
                .replace(" ", "_")
            )
            tasks.append(
                _run_turbomole_inline(
                    hyper_input,
                    _child_kwargs(kwargs),
                    custom_name=custom_name,
                )
            )

        node_runner.info(
            f"Starting {len(tasks)} hyperpolarizability jobs on the optimized geometry."
        )
        results = await asyncio.gather(*tasks, return_exceptions=True)

        section_name = molecule_section_name(molecule)
        dataset = DataSet(
            field_name=f"{HYPERPOL2_DATASET_TYPE}.{section_name}",
            metadata=DataSetMetadata(
                field_name=HYPERPOL2_DATASET_TYPE,
                data={"formula": section_name},
            ),
        )
        section = DataSetSection()
        dataset[section_name] = section

        failures: List[str] = []
        for (functional_enum, dispersion_enum, basis_set), result in zip(combos, results):
            error = None
            table = None
            combo_name = f"{functional_enum.value} / {dispersion_enum.value} / {basis_set}"
            if isinstance(result, Exception):
                error = child_exception_text(result, node_name="turbomole2")
                node_runner.error(f"{combo_name} failed: {error}")
            else:
                table = _extract_hyperpolarizability_table(result)
                hyperpol_result = _extract_qm_result(result) or result
                if not (_is_completed(result) or _is_completed(hyperpol_result)):
                    error = (
                        getattr(result, "error_message", None)
                        or f"Hyperpolarizability step did not complete successfully ({_result_debug(hyperpol_result)})."
                    )
                    node_runner.error(f"{combo_name} did not complete: {error}")
                elif not _has_hyperpol_output(result) and not _has_hyperpol_output(hyperpol_result):
                    error = "Hyperpolarizability step completed but no beta output was detected."
                    node_runner.error(f"{combo_name}: {error}")
                else:
                    node_runner.info(f"{combo_name} completed.")
            if error:
                failures.append(f"{combo_name}: {error}")

            record = HyperPolarizationRecord2(
                molecule=molecule,
                optimization_functional=TurbomoleFunctional(
                    functional=optimization_functional.functional
                ),
                optimization_dispersion_correction=TurbomoleDispersionCorrection(
                    value=optimization_dispersion.value
                ),
                optimization_basis_set=TurbomoleBasisSet2(
                    basis_set=optimization_basis_set.basis_set
                ),
                hyperpolarizability_functional=TurbomoleFunctional(functional=functional_enum),
                hyperpolarizability_dispersion_correction=TurbomoleDispersionCorrection(
                    value=dispersion_enum
                ),
                hyperpolarizability_basis_set=TurbomoleBasisSet2(basis_set=basis_set),
                grids_used=list(grids_used),
                started_at=started_at,
                wavelength=record_wavelength,
                hyperpol=table,
                success=error is None,
                error=str(error) if error else None,
            )
            record = await context.db.save(record)
            section.add_row(
                dataset_row_from_record2(record),
                name=f"{functional_enum.value}_{dispersion_enum.value}_{basis_set}",
            )

        await dataset.save(context.db)
        node_runner.dataset = dataset
        if failures:
            raise RuntimeError(
                f"{len(failures)} of {len(combos)} hyperpolarizability jobs failed: "
                + "; ".join(failures)
            )
        node_runner.info("Workflow completed successfully.")
        return node_runner.succeed()
    except WorkflowFailure:
        raise
    except Exception as exc:
        message = child_exception_text(exc)
        node_runner.error(message)
        node_runner.fail(message)
        raise RuntimeError(message) from exc
