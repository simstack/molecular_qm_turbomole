# TURBOMOLE SMP resources

Simstack supplies the executing node's effective parameters in
`kwargs["parent_parameters"]`. The hyperpolarizability workflows forward these
kwargs to Turbomole self-children. Simstack copies parent Slurm settings onto
self-children with no explicit Slurm settings; this also works through nested
self-children. An explicit child allocation is preserved.

Turbomole converts total `tasks * cpus_per_task` to SMP cores. If total tasks
are absent, it uses `nodes * tasks_per_node`; CPUs per task default to one.
The actual Slurm environment bounds the requested count and supplies a fallback
when effective parameters are unavailable. With neither source, local launcher
behavior remains unchanged. Multi-node allocations are rejected for SMP.

Every launcher receives invocation-specific `PARA_ARCH=SMP`, `PARNODES`,
`OMP_NUM_THREADS`, and `SIMSTACK_TURBOMOLE_CORES`. No global environment is
modified. Module-based launchers must re-export `PARNODES` and
`OMP_NUM_THREADS` from `SIMSTACK_TURBOMOLE_CORES` after loading Turbomole,
as shown in config.toml, since modules may reset their defaults.

Deploy the Simstack environment-forwarding change together with this package
and the launcher configuration. No database migration is needed. Validate on
HPC with a fresh single-node 12-task job: compare the Slurm allocation, the
`TURBOMOLE SMP cores=12` log entry, and Turbomole's reported parallel count.
