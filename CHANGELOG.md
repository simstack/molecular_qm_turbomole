# CHANGELOG

<!-- version list -->

## v0.9.0 (2026-09-05)


## v0.8.0 (2026-09-05)

### Chores

- Simplify active_dirs configuration by removing unused entries
  ([`48538fe`](https://github.com/simstack/molecular_qm_turbomole/commit/48538fe1294c02bb8cb7c3b62dc8b499372b6702))

- Update dependencies and add upgrade script
  ([`8b611d7`](https://github.com/simstack/molecular_qm_turbomole/commit/8b611d7c032efb20a814ce92ec823a3fcb53d830))

- Update simstack and molecular_qm_util references in pyproject.toml
  ([`7dfe18b`](https://github.com/simstack/molecular_qm_turbomole/commit/7dfe18b11f5f3628db6965aebe8d749a49567248))

### Features

- Add timing and heartbeat tracking for Turbomole subprocesses
  ([`1cde310`](https://github.com/simstack/molecular_qm_turbomole/commit/1cde310dd301d654764dd1e9c99566e32076483e))


## v0.7.0 (2026-08-23)

### Chores

- Enhance hyperpolarization dataset management and restore functionality
  ([`c2f4f1f`](https://github.com/simstack/molecular_qm_turbomole/commit/c2f4f1ff1b1997f34d7d9e17ec8d4efdaa76f455))

### Features

- Add functionality to collect Turbomole restart and info files
  ([`497664b`](https://github.com/simstack/molecular_qm_turbomole/commit/497664bc70a911066c6934d89e30f134e6c9416c))

### Refactoring

- Remove unused hyperpolarizability table and update dataset row construction
  ([`aff03ee`](https://github.com/simstack/molecular_qm_turbomole/commit/aff03ee568294a92becb333cd3b63c7c1a53efae))

- Rename DispersionCorrection to TurbomoleDispersionCorrection across models and tests
  ([`d08a619`](https://github.com/simstack/molecular_qm_turbomole/commit/d08a619427751a9e554de6635a0c2ea9f370cf1b))


## v0.6.0 (2026-08-22)

### Features

- Enhance hyperpolarization handling and dataset management
  ([`32df224`](https://github.com/simstack/molecular_qm_turbomole/commit/32df2243d0938f093ef824b187026307d53771d1))


## v0.5.0 (2026-08-22)

### Features

- Add openbabel-wheel dependency and enhance molecule handling
  ([`ad7a9f2`](https://github.com/simstack/molecular_qm_turbomole/commit/ad7a9f291aa9704d2ceece32459ae2e44f03a97b))


## v0.4.5 (2026-08-22)

### Bug Fixes

- Stop duplicating TURBOMOLE scripts in the Hatch wheel
  ([`e79f71b`](https://github.com/simstack/molecular_qm_turbomole/commit/e79f71bc554d7efe34c6d618a8fed959999fa746))

### Refactoring

- Update Turbomole input models and migration logic
  ([`353d1e2`](https://github.com/simstack/molecular_qm_turbomole/commit/353d1e295a239765c5a938148ddd4b490924df1a))


## v0.4.4 (2026-08-22)

### Bug Fixes

- Simplify label retrieval and enhance JSON schema for HyperPolarizationRecord
  ([`1f82b76`](https://github.com/simstack/molecular_qm_turbomole/commit/1f82b762624daa41176b2285e87a07eef6896503))

### Refactoring

- Enhance error handling and logging in hyperpolarizability workflow
  ([`7c4627f`](https://github.com/simstack/molecular_qm_turbomole/commit/7c4627f4461cecaebada19d115d6201a99d363f6))

- Update functional handling in TURBOMOLE input models
  ([`e4b58fe`](https://github.com/simstack/molecular_qm_turbomole/commit/e4b58fe986ed9e5e4e9efcf459c98a53f24d69c8))


## v0.4.3 (2026-08-22)

### Bug Fixes

- Introduce dispersion correction as a top-level model in TURBOMOLE input
  ([`0a4778e`](https://github.com/simstack/molecular_qm_turbomole/commit/0a4778ea663c010fddddab9aaf4dce9756ff8028))


## v0.4.2 (2026-08-22)

### Bug Fixes

- Update dispersion_correction field in HyperPolarizationRecord model
  ([`ed16bbe`](https://github.com/simstack/molecular_qm_turbomole/commit/ed16bbe4eecb7bcbcfba007c1b07b77b754f0f98))


## v0.4.1 (2026-08-22)

### Bug Fixes

- Improve optimization chunk handling in TURBOMOLE integration
  ([`11ccd72`](https://github.com/simstack/molecular_qm_turbomole/commit/11ccd72cc4c48172c4707d74a65f1004b034d9c2))


## v0.4.0 (2026-08-22)

### Features

- Enhance TURBOMOLE input model with SCF iteration limits and optimization cycles
  ([`29c4f58`](https://github.com/simstack/molecular_qm_turbomole/commit/29c4f58a4d2bc533f6c8ccdfd18f02388f83e47b))


## v0.3.0 (2026-08-22)


## v0.2.0 (2026-08-22)

### Features

- Introduce hyperpolarizability calculation framework
  ([`35cfedc`](https://github.com/simstack/molecular_qm_turbomole/commit/35cfedceabdbbb2b67906812f38181c649055ec3))


## v0.1.1 (2026-08-22)


## v0.1.0 (2026-08-22)

- Initial Release
