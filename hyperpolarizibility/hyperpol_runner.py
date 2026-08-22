import asyncio
from typing import List

from odmantic import Field, Model

from hyperpolarizibility.workflows import (
    HyperpolarizabilitySettings,
    _child_kwargs,
    _ensure_db_molecule,
    hyperpolarizibility,
    workflow_parameters,
)
from molecular_qm_models.density_functional import Functional, FunctionalEnum
from molecular_qm_models.dispersion_correction import DispersionCorrection, DispersionCorrectionEnum
from molecular_qm_models.molecule import Molecule
from molecular_qm_turbomole.models.turbomole_input import (
    HyperpolarizabilityModeEnum,
    SolventModeEnum,
    TurbomoleBasisSet2,
    TurbomoleQMInput2,
)
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import StringData, simstack_model


@simstack_model
class HyperpolRunnerModel(Model):
    field_name: str = "HyperpolRunnerModel"
    functionals: List[FunctionalEnum] = Field(
        default_factory=list,
        json_schema_extra={"description": "List of functionals to run."},
    )
    dispersion_correction: DispersionCorrectionEnum = Field(
        default=DispersionCorrectionEnum.NONE,
        json_schema_extra={
            "enum": [e.value for e in DispersionCorrectionEnum],
            "description": "Version of the dispersion correction to use",
        },
    )


@node(parameters=workflow_parameters)
async def hyperpol_runner(
    molecule: Molecule,
    model: HyperpolRunnerModel,
    **kwargs,
) -> SimstackResult:
    """
    Run hyperpolarizibility once per requested functional.

    Parameters:
        molecule (Molecule): Molecule to optimize and evaluate.
        model (HyperpolRunnerModel): Functionals and dispersion correction.

    Called Nodes:
        hyperpolarizibility
    """
    node_runner: NodeRunner = kwargs["node_runner"]

    molecule = await _ensure_db_molecule(molecule)
    dispersion = DispersionCorrection(value=model.dispersion_correction)
    settings = HyperpolarizabilitySettings(
        hyperpolarizability=HyperpolarizabilityModeEnum.STATIC,
        hyperpol_frequency_nm=0.0,
        frequency_tolerance=1e-6,
    )

    tasks = []
    for functional_enum in model.functionals:
        functional = Functional(
            functional=functional_enum,
            dispersion_correction=dispersion,
        )
        opt_qm_input = TurbomoleQMInput2(
            molecule=molecule,
            name=f"hyperpol_{functional_enum.value}",
            functional=functional,
            optimization=True,
            basis_set=TurbomoleBasisSet2(),
            solvent_mode=SolventModeEnum.IMPLICIT,
            solvent="chloroform",
        )
        tasks.append(
            hyperpolarizibility(
                optimization_qm_input=opt_qm_input,
                hyperpolarizability_settings=settings,
                **_child_kwargs(kwargs),
            )
        )

    node_runner.info(f"Starting parallel execution for {len(tasks)} functionals.")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for functional_enum, result in zip(model.functionals, results):
        key = f"result_{functional_enum.value}"
        if isinstance(result, Exception):
            node_runner.info(f"Functional {functional_enum.value} failed with exception: {result}")
            setattr(node_runner, key, StringData(value=str(result)))
        else:
            node_runner.info(f"Functional {functional_enum.value} completed.")
            setattr(node_runner, key, result)

    return node_runner.succeed()
