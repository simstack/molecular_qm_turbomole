import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List

from odmantic import Model

from hyperpolarizibility.hyperpol_runner import (
    HyperpolRunnerModel,
    _hyperpol_dataset_row,
    _sweep_combos,
)
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
from molecular_qm_models.molecule import Molecule
from molecular_qm_turbomole.lib.molecule_labels import fill_molecule_labels, molecule_section_name
from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctional, TurbomoleFunctionalEnum
from molecular_qm_turbomole.models.turbomole_input import (
    TurbomoleDispersionCorrection,
    TurbomoleBasisSet2,
    TurbomoleQMInput2,
)
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import StringData
from simstack.models.dataset import DataSet, DataSetSection
from simstack.models.dataset_metadata import DataSetMetadata

HYPERPOL2_DATASET_TYPE = "hyperpolarizibility2"


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
) -> TurbomoleQMInput2:
    hyper_input = _build_hyperpol_input(base_input, optimized_structure, settings)
    hyper_input.basis_set = TurbomoleBasisSet2(basis_set=basis_set)
    hyper_input.functional = TurbomoleFunctional(functional=functional_enum)
    hyper_input.name = f"{base_input.name}_hyperpol_{functional_enum.value}_{basis_set}"
    return hyper_input


def dataset_row_from_record2(record: HyperPolarizationRecord2) -> Dict[str, Model]:
    row = _hyperpol_dataset_row(
        basis_set=_basis_label(record.hyperpolarizability_basis_set),
        functional=_functional_enum(record.hyperpolarizability_functional),
        frequency_nm=float(record.wavelength or 0.0),
        hyperpol_table=record.hyperpol,
        error=record.error,
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
    hyperpolarizability_methods: HyperpolRunnerModel,
    **kwargs,
) -> SimstackResult:
    """
    Optimize once with the QMInput m3/m5 frequency protocol, then compute
    hyperpolarizability for each functional × basis combination. Each job writes
    a HyperPolarizationRecord2 with both the optimization method and the
    hyperpolarizability method.

    Parameters:
        optimization_qm_input (TurbomoleQMInput2): Ground-state optimization settings and molecule.
        hyperpolarizability_settings (HyperpolarizabilitySettings): Static or dynamic beta, plus frequency_tolerance.
        hyperpolarizability_methods (HyperpolRunnerModel): Functionals and basis sets for the hyperpolarizability step.

    Called Nodes:
        turbomole2

    SimstackResult:
        dataset (DataSet): Rows with optimization and hyperpolarizability method settings plus beta.
    """
    node_runner: NodeRunner = kwargs["node_runner"]

    try:
        if optimization_qm_input.states > 0:
            raise ValueError("Workflow expects ground-state path for optimization. Please set states=0.")
        combos = _sweep_combos(hyperpolarizability_methods)
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
        for functional_enum, basis_set in combos:
            hyper_input = _build_hyperpol_method_input(
                optimization_qm_input,
                optimized_structure,
                hyperpolarizability_settings,
                basis_set=basis_set,
                functional_enum=functional_enum,
            )
            custom_name = (
                f"hyper_{functional_enum.value}_{basis_set}"
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
        for (functional_enum, basis_set), result in zip(combos, results):
            error = None
            table = None
            combo_name = f"{functional_enum.value} / {basis_set}"
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
                    value=optimization_qm_input.dispersion_correction.value
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
                name=f"{functional_enum.value}_{basis_set}",
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
