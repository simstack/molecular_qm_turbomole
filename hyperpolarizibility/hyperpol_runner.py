from typing import List

from odmantic import Model, Field

from applications.electronic_structure import FunctionalEnum, Molecule, Functional, SolventModeEnum
from applications.electronic_structure.hyperpolarizibility.hyperpolarizability_defaults import HYPERPOL_MODE_STATIC
from applications.electronic_structure.turbomole import TurbomoleQMInput, TurbomoleBasisSet
from applications.electronic_structure.hyperpolarizibility.workflows import workflow_parameters, _ensure_db_molecule, \
    TurbomoleHyperpolarizabilityQMInputAuto, TurbomoleOptHyperpolWorkflowInputAuto, turbomole_opt_hyperpol_workflow, \
    _child_kwargs
from molecular_qm_models import DispersionCorrectionEnum, DispersionCorrection
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import simstack_model, FloatData


@simstack_model
class HyperpolRunnerModel(Model):
    functionals: List[FunctionalEnum] = Field(
        default_factory=list,
        json_schema_extra={
            "description": "List of functionals to run."
        }
    )
    tolerance: float = Field(
        default_factory=lambda: FloatData(value=1e-6),
        json_schema_extra={
            "description": "Tolerance threshold for the calculations."
        }
    )
    dispersion_correction: DispersionCorrectionEnum = Field(
        default=DispersionCorrectionEnum.NONE,
        json_schema_extra={
            "enum": [e.value for e in DispersionCorrectionEnum],
            "description": "Version of the dispersion correction to use"
        }
    )


@node(parameters=workflow_parameters)
async def hyperpol_runner(
    molecule: Molecule,
    model: HyperpolRunnerModel,
    **kwargs,
) -> SimstackResult:
    """
    Executes a hyperpolarizability calculation workflow for a given molecule and set of functionals.

    This function performs a Turbomole-based quantum mechanical calculation including geometry
    optimization and hyperpolarizability computation. Each specified functional is processed in
    parallel using asynchronous tasks. The function dynamically constructs input data for the
    workflow and ensures the molecule exists in the database prior to execution.

    Arguments:
        molecule (Molecule): The molecule for which to perform the calculation.
        model (HyperpolRunnerModel): The model specifying the calculation parameters such as
            functionals to use, basis set, dispersion correction, and tolerance.
        **kwargs: Additional keyword arguments expected by the function, including `node_runner`.

    Returns:
        SimstackResult: The result of the workflow execution, containing the hyperpolarizability
        calculation outputs for each functional.
        result: (BetaTensorZDipoleESUResult)

    Called Nodes:
        turbomole_opt_hyperpol_workflow

    Raises:
        Any exceptions encountered during the asynchronous tasks will be captured and logged,
        but not explicitly raised by this function.
    """
    node_runner: NodeRunner = kwargs["node_runner"]
    import asyncio

    # Ensure molecule is in DB
    molecule = await _ensure_db_molecule(molecule)

    # Calculate scfconv from tolerance
    # tolerance is a float in HyperpolRunnerModel now, but let's check if it's wrapped
    tol_value = model.tolerance
    if hasattr(tol_value, "value"):
        tol_value = tol_value.value

    tasks = []
    for functional_enum in model.functionals:
        # Construct TurbomoleOptHyperpolWorkflowInputAuto
        # We need to build optimization_qm_input and hyperpolarizability_qm_input

        # Base QM input for optimization
        opt_qm_input = TurbomoleQMInput(
            molecule=molecule,
            functional=Functional(
                functional=functional_enum,
                dispersion_correction=DispersionCorrection(value=model.dispersion_correction)
            ),
            optimization=True,
            basis_set=TurbomoleBasisSet(value="6-31g"),
            solvent_mode=SolventModeEnum.IMPLICIT,
            solvent="choloroform"
        )

        # Base QM input for hyperpol (it will be coerced/projected anyway)
        hyperpol_qm_input = TurbomoleHyperpolarizabilityQMInputAuto(
            functional=Functional(
                functional=functional_enum,
                dispersion_correction=DispersionCorrection(value=model.dispersion_correction)
            ),
            hyperpolarizability=True,
            # We probably want some default frequency or static
            hyperpolarizability_mode=HYPERPOL_MODE_STATIC,
            solvent_mode=SolventModeEnum.IMPLICIT,
            solvent="choloroform"
        )

        workflow_input = TurbomoleOptHyperpolWorkflowInputAuto(
            name=f"hyperpol_{functional_enum.value}",
            optimization_qm_input=opt_qm_input,
            hyperpolarizability_qm_input=hyperpol_qm_input,
            run_beta_tensor=True
        )

        # Create a task for each functional
        tasks.append(
            turbomole_opt_hyperpol_workflow(
                workflow_input=workflow_input,
                freqency_tolerance=FloatData(value=tol_value),
                **_child_kwargs(kwargs)
            )
        )

    node_runner.log(f"Starting parallel execution for {len(tasks)} functionals.")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for functional_enum, result in zip(model.functionals, results):
        key = f"result_{functional_enum.value}"
        if isinstance(result, Exception):
            node_runner.log(f"Functional {functional_enum.value} failed with exception: {result}")
            # How to represent failure in results?
            # SimstackResult might not be directly assignable like this,
            # but usually we set it on the node_runner's result object.
            setattr(node_runner, key, str(result))
        else:
            node_runner.log(f"Functional {functional_enum.value} completed.")
            setattr(node_runner, key, result)

    return node_runner.succeed()
