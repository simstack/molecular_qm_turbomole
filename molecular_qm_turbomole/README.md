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
uv run create_model_table --dir molecular_qm_turbomole --dir hyperpolarizibility
uv run create_node_table --dir molecular_qm_turbomole --dir hyperpolarizibility
```

To scan sister source trees as well, make local junctions (gitignored, not submodules):

```powershell
New-Item -ItemType Junction -Path molecular_qm_models -Target ..\molecular_qm_models
New-Item -ItemType Junction -Path molecular_qm_util -Target ..\molecular_qm_util
New-Item -ItemType Junction -Path simstack -Target ..\simstack\src\simstack
```

```bash
uv run create_model_table --dir molecular_qm_turbomole --dir hyperpolarizibility --dir molecular_qm_models --dir simstack
uv run create_node_table --dir molecular_qm_turbomole --dir hyperpolarizibility --dir simstack
```

```bash
uv run python -m molecular_qm_turbomole.testing.test_single_point
uv run python -m molecular_qm_turbomole.testing.test_optimization
```

## Versioning

The package version comes from git tags (`hatch-vcs`). Tags and GitHub releases
are created on push to `main` by [python-semantic-release](https://python-semantic-release.readthedocs.io/)
from [Conventional Commits](https://www.conventionalcommits.org/):

- `fix:` → patch (`0.1.0` → `0.1.1`)
- `feat:` → minor (`0.1.0` → `0.2.0`)
- `BREAKING CHANGE:` → minor while the version is `0.x` (`0.2.0` → `0.3.0`)

Several `fix:` commits since the last tag become **one** new tag, not one tag
per commit. Preview the next version locally:

```bash
uvx --from python-semantic-release semantic-release version --print
```

## Notes

`TurbomoleQMInput2` keeps the historical field surface (without GW) but does not
infer or rewrite solvent / hyperpolarizability / toggle fields on validation.
