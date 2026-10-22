"""Translate effective Simstack allocation settings into TURBOMOLE SMP cores."""
import os
from collections.abc import Mapping


def _positive(value, label):
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if isinstance(value, bool) or number < 1 or str(number) != str(value):
        raise ValueError(f"{label} must be a positive integer")
    return number


def turbomole_parallel_env(kwargs, node_runner, environ: Mapping[str, str] | None = None):
    """Resolve this node's effective allocation (forwarded as parent_parameters).

    Only explicitly supplied Slurm fields count as resource requests; model
    defaults alone must not change execution on a local workstation.
    """
    environ = os.environ if environ is None else environ
    parameters = kwargs.get("parent_parameters")
    slurm = getattr(parameters, "slurm_parameters", None)
    fields = getattr(slurm, "model_fields_set", set())

    def requested(name):
        return _positive(getattr(slurm, name, None), name) if name in fields else None

    nodes = requested("nodes")
    allocated_nodes = _positive(environ.get("SLURM_JOB_NUM_NODES") or environ.get("SLURM_NNODES"), "Slurm nodes")
    if (nodes or 1) > 1 or (allocated_nodes or 1) > 1:
        raise ValueError("TURBOMOLE SMP requires one node; request nodes=1 or use a separate MPI launcher.")
    tasks = requested("tasks")
    if tasks is None:
        tasks_per_node = requested("tasks_per_node")
        if tasks_per_node is not None:
            tasks = (nodes or 1) * tasks_per_node
    cpus = requested("cpus_per_task")
    cores = tasks * (cpus or 1) if tasks is not None else cpus
    source = "effective slurm_parameters"

    allocated_tasks = _positive(environ.get("SLURM_NTASKS"), "SLURM_NTASKS")
    allocated_cpus = _positive(environ.get("SLURM_CPUS_PER_TASK"), "SLURM_CPUS_PER_TASK")
    allocated = allocated_tasks * (allocated_cpus or 1) if allocated_tasks else allocated_cpus
    if allocated is not None:
        if cores is None:
            cores, source = allocated, "Slurm environment"
        elif cores > allocated:
            cores, source = allocated, "effective slurm_parameters capped by Slurm allocation"
    if cores is None:
        return {}
    node_runner.info(f"TURBOMOLE SMP cores={cores}; source={source}")
    return {
        "PARA_ARCH": "SMP",
        "PARNODES": str(cores),
        "OMP_NUM_THREADS": str(cores),
        # Module loading can overwrite PARNODES/OMP_NUM_THREADS. Launchers
        # reapply this invocation-specific value after loading the module.
        "SIMSTACK_TURBOMOLE_CORES": str(cores),
    }
