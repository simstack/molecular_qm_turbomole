import logging
import os
from pathlib import Path
from typing import Optional, Union

from molecular_qm_models.molecule import Molecule
from molecular_qm_models.qm_result import QMResult
from molecular_qm_turbomole.lib.input_writer import ANGSTROM_TO_BOHR
from molecular_qm_turbomole.lib.output_parser import TurbomoleOutputParser
from molecular_qm_turbomole.lib.request_validation import validate_molecule_geometry
from molecular_qm_turbomole.models.semiempirical import (
    TurbomoleSemiempiricalInput,
    TurbomoleSemiempiricalMethod,
)
from molecular_qm_turbomole.models.turbomole_functional import TurbomoleFunctional
from molecular_qm_turbomole.models.turbomole_input import (
    TurbomoleBasisSet2,
    TurbomoleMethodEnum,
    TurbomoleQMInput2,
)
from molecular_qm_turbomole.nodes.turbomole2 import (
    _collect_turbomole_info_files,
    _collect_turbomole_restart_files,
    _fail,
    _run_monitored_subprocess,
    _run_optimization_chunks,
    parameters,
)
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult

logger = logging.getLogger("TurbomoleSemiempiricalNode")


def _write_semiempirical_coord(molecule: Molecule, path: str = "coord") -> None:
    lines = ["$coord"]
    for atom in molecule.atoms:
        x = atom.x * ANGSTROM_TO_BOHR
        y = atom.y * ANGSTROM_TO_BOHR
        z = atom.z * ANGSTROM_TO_BOHR
        lines.append(f"{x:20.14f} {y:20.14f} {z:20.14f} {atom.element.lower()}")
    lines.append("$end")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_semiempirical_define_input(
    molecule: Molecule,
    opts: TurbomoleSemiempiricalInput,
    path: str = "define.inp",
    name: str = "semiempirical",
) -> None:
    method_name = opts.method.value if hasattr(opts.method, "value") else str(opts.method)
    lines = [
        f"\n{name}",
        "a coord",
        "*",
        "no",
        f"b all {method_name}",
        "*",
        "eht",
        "",
        str(opts.charge),
    ]
    if opts.multiplicity < 3:
        lines.extend(["", "", ""])
    else:
        lines.extend(["n", f"u {opts.multiplicity - 1}", "*", ""])

    lines.extend(
        [
            "scf",
            "iter",
            str(int(opts.scfiterlimit)),
            "conv",
            str(int(opts.scfconv)),
            "soghf",
            "off",
            "",
            "dft",
            "off",
            "",
            "*",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


@node(parameters=parameters)
async def turbomole_semiempirical(
    molecule: Molecule,
    opts: Optional[TurbomoleSemiempiricalInput] = None,
    **kwargs,
) -> SimstackResult:
    """
    TURBOMOLE semiempirical calculation node (AM1, PM3, MNDO, MNDOD).

    Parameters:
        molecule (Molecule): Input geometry.
        opts (TurbomoleSemiempiricalInput): Semiempirical settings (method, charge, multiplicity, opt).

    SimstackResult:
        result (QMResult): Final energy, structure, and convergence status.
        optimization_timing (SimpleTable): Per-chunk and summary wall/CPU times when optimizing.
    """
    node_runner = kwargs["node_runner"]
    if opts is None:
        opts = TurbomoleSemiempiricalInput()
    elif isinstance(opts, (str, TurbomoleSemiempiricalMethod)):
        opts = TurbomoleSemiempiricalInput(method=TurbomoleSemiempiricalMethod.coerce(opts))

    method_label = opts.method.value if hasattr(opts.method, "value") else str(opts.method)
    node_runner.info(
        f"Starting TURBOMOLE semiempirical calculation: method={method_label}, "
        f"optimization={opts.optimization}, charge={opts.charge}, multiplicity={opts.multiplicity}, "
        f"max_opt_cycles={opts.max_opt_cycles}, scfiterlimit={opts.scfiterlimit}"
    )

    enter_scratch = getattr(node_runner, "enter_scratch", None)
    if callable(enter_scratch):
        enter_scratch("turbomole_semiempirical")

    try:
        try:
            validate_molecule_geometry(opts.model_copy(update={"molecule": molecule}))
        except Exception:
            # fallback direct validation on molecule
            if not getattr(molecule, "atoms", None):
                _fail(node_runner, "Input molecule contains no atoms.")

        _write_semiempirical_coord(molecule)
        _write_semiempirical_define_input(molecule, opts, name=f"semiempirical_{method_label}")
        node_runner.info("Input files generated")

        ok, _, _ = _run_monitored_subprocess(
            node_runner,
            "turbomole_define",
            "TURBOMOLE define",
            kwargs,
            command="define_command",
        )
        if not ok:
            _fail(node_runner, "TURBOMOLE define failed. Check turbomole_define.log.")

        # Construct a proxy QMInput for optimization tracking and chunks
        qm_input_proxy = TurbomoleQMInput2(
            molecule=molecule,
            name=f"semi_{method_label}",
            method=TurbomoleMethodEnum.HF,
            basis_set=TurbomoleBasisSet2(basis_set=method_label),
            charge=opts.charge,
            multiplicity=opts.multiplicity,
            optimization=opts.optimization,
            max_opt_cycles=opts.max_opt_cycles,
            scfiterlimit=opts.scfiterlimit,
            scfconv=opts.scfconv,
            gradients=opts.gradients,
        )

        if opts.optimization:
            await _run_optimization_chunks(qm_input_proxy, node_runner, kwargs)
        else:
            Path("turbomole_opt_cycles").unlink(missing_ok=True)
            if opts.gradients:
                Path("turbomole_gradients").write_text("1", encoding="utf-8")
            else:
                Path("turbomole_gradients").unlink(missing_ok=True)
            ok, _, _ = _run_monitored_subprocess(
                node_runner,
                "turbomole_exe",
                "TURBOMOLE ground-state calculation",
                kwargs,
            )
            if not ok:
                _fail(node_runner, "TURBOMOLE ground-state calculation failed. Check turbomole_exe.log.")

        parser = TurbomoleOutputParser(".", node_runner=node_runner)
        parser.parse()
        if not parser.properly_terminated:
            _fail(node_runner, "Calculation failed to terminate properly.")

        final_structure = parser.final_structure or molecule
        qm_result = QMResult(
            molecule=molecule,
            properly_terminated=parser.properly_terminated,
            final_energy=parser.final_energy,
            final_structure=final_structure,
            energies=parser.energies,
        )
        try:
            db = context.db
            if db is not None:
                qm_result = await db.save(qm_result)
        except Exception:
            pass

        await _collect_turbomole_restart_files(node_runner, qm_result)
        _collect_turbomole_info_files(node_runner)

        node_runner.result = qm_result
        node_runner.qm_result = qm_result
        node_runner.info(f"TURBOMOLE semiempirical finished successfully. Energy: {parser.final_energy}")
        return node_runner.succeed()
    except Exception as exc:
        _fail(node_runner, f"TURBOMOLE semiempirical node failed: {exc}")
