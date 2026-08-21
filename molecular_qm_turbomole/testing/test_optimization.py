import asyncio

from molecular_qm_turbomole.testing.test_single_point import make_h2, make_qm_input, make_water
from molecular_qm_turbomole.nodes.turbomole2 import turbomole2
from simstack.core.context import context
from simstack.models import Parameters


async def run_optimization():
    await context.initialize()
    parameters = Parameters(resource="local", force_rerun=True)
    for molecule in (make_h2(), make_water()):
        qm_input = make_qm_input(molecule, optimization=True)
        result = await turbomole2(qm_input, parameters=parameters)
        print(f"optimization {molecule.formula}: {result}")


if __name__ == "__main__":
    asyncio.run(run_optimization())
