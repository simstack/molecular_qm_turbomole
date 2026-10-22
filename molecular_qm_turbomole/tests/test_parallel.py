from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from molecular_qm_turbomole.lib.parallel import turbomole_parallel_env
from simstack.core.node import inherit_parent_slurm_parameters_for_self
from simstack.models.parameters import Parameters, SlurmParameters


def resolve(slurm=None, environ=None):
    kwargs = {} if slurm is None else {"parent_parameters": SimpleNamespace(slurm_parameters=slurm)}
    return turbomole_parallel_env(kwargs, Mock(), environ={} if environ is None else environ)


@pytest.mark.parametrize("values, expected", [
    ({"nodes": 1, "tasks": 12, "cpus_per_task": 1}, 12),
    ({"tasks": 3, "cpus_per_task": 4}, 12),
    ({"tasks_per_node": 12}, 12),
    ({"tasks": 4, "tasks_per_node": 12}, 4),
    ({"cpus_per_task": 6}, 6),
])
def test_effective_settings(values, expected):
    env = resolve(SlurmParameters(**values))
    assert env == {key: str(expected) for key in (
        "PARNODES", "OMP_NUM_THREADS", "SIMSTACK_TURBOMOLE_CORES"
    )} | {"PARA_ARCH": "SMP"}


def test_nested_self_children():
    parent = Parameters(resource="self", slurm_parameters=SlurmParameters(nodes=1, tasks=12))
    child = Parameters(resource="self")
    grandchild = Parameters(resource="self")
    inherit_parent_slurm_parameters_for_self(child, parent)
    inherit_parent_slurm_parameters_for_self(grandchild, child)
    assert resolve(grandchild.slurm_parameters)["PARNODES"] == "12"
    assert grandchild.slurm_parameters is not parent.slurm_parameters


def test_child_limit_is_preserved():
    parent = Parameters(resource="self", slurm_parameters=SlurmParameters(tasks=12))
    child = Parameters(resource="self", slurm_parameters=SlurmParameters(tasks=4))
    inherit_parent_slurm_parameters_for_self(child, parent)
    assert resolve(child.slurm_parameters, {"SLURM_NTASKS": "12"})["PARNODES"] == "4"


def test_allocation_bounds_request():
    assert resolve(SlurmParameters(tasks=32), {"SLURM_NTASKS": "12"})["PARNODES"] == "12"


def test_slurm_environment_fallback():
    assert resolve(environ={"SLURM_NTASKS": "3", "SLURM_CPUS_PER_TASK": "4"})["PARNODES"] == "12"


def test_local_defaults_do_not_change_environment():
    assert resolve() == {}
    assert resolve(SlurmParameters()) == {}


@pytest.mark.parametrize("slurm, environ", [
    (SlurmParameters(nodes=2, tasks=12), {}),
    (None, {"SLURM_JOB_NUM_NODES": "2"}),
])
def test_multinode_smp_rejected(slurm, environ):
    with pytest.raises(ValueError, match="SMP requires one node"):
        resolve(slurm, environ)


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_invalid_allocation_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        resolve(environ={"SLURM_NTASKS": value})


@pytest.mark.parametrize("command", ["run_command", "escf_command", "aoforce_command", "dscf_command", "ricc2_command"])
def test_monitored_launch_receives_effective_cores(monkeypatch, command):
    import importlib
    module = importlib.import_module("molecular_qm_turbomole.nodes.turbomole2")
    resource_config = Mock()
    monkeypatch.setattr(module, "context", SimpleNamespace(resource_config=resource_config))
    monkeypatch.setattr(module, "ProcessHeartbeat", Mock())
    for key in ("SLURM_NTASKS", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NUM_NODES", "SLURM_NNODES"):
        monkeypatch.delenv(key, raising=False)
    module._run_monitored_subprocess(Mock(), "tm", "test", {
        "parent_parameters": SimpleNamespace(slurm_parameters=SlurmParameters(tasks=12)),
    }, command=command)
    assert resource_config.run.call_args.kwargs["env"]["PARNODES"] == "12"
    assert resource_config.run.call_args.kwargs["command"] == command
