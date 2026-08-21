# TURBOMOLE capability package for Simstack.

## Layout

- `models/` — `TurbomoleQMInput2` and related embedded models
- `nodes/` — `turbomole2` node (single-point and geometry optimization)
- `lib/` — input writer, output parser, control helpers, env bootstrap
- `testing/` — manual/integration scripts for small molecules
- `tests/` — unit tests that do not require TURBOMOLE binaries

## Host usage

`create_*_table --dir` only accepts directories **inside this repo**. It cannot
scan `../simstack`. Installed packages (`simstack`, `molecular_qm_models`) are
already registered via entry points, so this is enough:

```bash
uv run create_model_table --dir molecular_qm_turbomole
uv run create_node_table --dir molecular_qm_turbomole
```

To scan sister source trees as well, make local junctions (gitignored, not submodules):

```powershell
New-Item -ItemType Junction -Path molecular_qm_models -Target ..\molecular_qm_models
New-Item -ItemType Junction -Path molecular_qm_util -Target ..\molecular_qm_util
New-Item -ItemType Junction -Path simstack -Target ..\simstack\src\simstack
```

```bash
uv run create_model_table --dir molecular_qm_turbomole --dir molecular_qm_models --dir simstack
uv run create_node_table --dir molecular_qm_turbomole --dir simstack
```

```bash
uv run python -m molecular_qm_turbomole.testing.test_single_point
uv run python -m molecular_qm_turbomole.testing.test_optimization
```

## Notes

`TurbomoleQMInput2` keeps the historical field surface (without GW) but does not
infer or rewrite solvent / hyperpolarizability / toggle fields on validation.
