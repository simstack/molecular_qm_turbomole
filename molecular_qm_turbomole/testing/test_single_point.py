import asyncio

from molecular_qm_models.molecule import Atom, Molecule
from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctionalEnum
from molecular_qm_turbomole.models.turbomole_input import (
    TurbomoleDispersionCorrection,
    TurbomoleBasisSet2,
    TurbomoleQMInput2,
)
from molecular_qm_turbomole.nodes.turbomole2 import turbomole2
from simstack.core.context import context
from simstack.models import Parameters


def make_water() -> Molecule:
    coords = [
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ]
    molecule = Molecule()
    for element, xyz in zip(["O", "H", "H"], coords):
        molecule.add_atom(Atom.from_coords(element=element, coords=xyz))
    molecule.formula = "H2O"
    return molecule


def make_h2() -> Molecule:
    molecule = Molecule()
    molecule.add_atom(Atom.from_coords(element="H", coords=[0.0, 0.0, 0.0]))
    molecule.add_atom(Atom.from_coords(element="H", coords=[0.0, 0.0, 0.74]))
    molecule.formula = "H2"
    return molecule


def make_qm_input(molecule: Molecule, *, optimization: bool = False) -> TurbomoleQMInput2:
    return TurbomoleQMInput2(
        molecule=molecule,
        name="turbomole2-test",
        charge=0,
        multiplicity=1,
        basis_set=TurbomoleBasisSet2(basis_set="def2-SVP"),
        functional=TurbomoleFunctionalEnum.PBE,
        dispersion_correction=TurbomoleDispersionCorrection(),
        optimization=optimization,
        gradients=False,
    )


async def run_single_point():
    await context.initialize()
    parameters = Parameters(resource="int-nano", queue="slurm-queue",force_rerun=True)
    for molecule in (make_h2(), make_water()):
        qm_input = make_qm_input(molecule, optimization=False)
        result = await turbomole2(qm_input, parameters=parameters)
        print(f"single-point {molecule.formula}: {result}")


if __name__ == "__main__":
    asyncio.run(run_single_point())
