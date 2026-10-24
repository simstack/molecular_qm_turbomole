"""TURBOMOLE capabilities for molecular quantum mechanics in Simstack."""

from molecular_qm_turbomole.models.multistep_optimizer import (
    OptimizationStepInput,
    PreOptimizerInput,
)
from molecular_qm_turbomole.models.semiempirical import (
    TurbomoleSemiempiricalInput,
    TurbomoleSemiempiricalMethod,
)
from molecular_qm_turbomole.models.turbomole_functional import (
    TurbomoleFunctional,
    TurbomoleFunctionalEnum,
)
from molecular_qm_turbomole.models.turbomole_input import (
    TURBOMOLE_DEFAULT_BASIS_SET,
    TURBOMOLE_DEFAULT_GRID_SIZE,
    TURBOMOLE_DEFAULT_MAX_OPT_CYCLES,
    TURBOMOLE_DEFAULT_SCFCONV,
    TURBOMOLE_DEFAULT_SCFITERLIMIT,
    TurbomoleDispersionCorrection,
    HyperpolarizabilityModeEnum,
    SolventModeEnum,
    TurbomoleBasisSet2,
    TurbomoleMethodEnum,
    TurbomoleQMInput2,
)
from molecular_qm_turbomole.nodes.multistep_optimizer import multistep_optimizer
from molecular_qm_turbomole.nodes.turbomole2 import turbomole2
from molecular_qm_turbomole.nodes.turbomole_ricc2 import turbomole_ricc2
from molecular_qm_turbomole.nodes.turbomole_semiempirical import turbomole_semiempirical

try:
    from molecular_qm_turbomole._version import __version__
except ImportError:
    __version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    "TURBOMOLE_DEFAULT_BASIS_SET",
    "TURBOMOLE_DEFAULT_GRID_SIZE",
    "TURBOMOLE_DEFAULT_MAX_OPT_CYCLES",
    "TURBOMOLE_DEFAULT_SCFCONV",
    "TURBOMOLE_DEFAULT_SCFITERLIMIT",
    "SolventModeEnum",
    "HyperpolarizabilityModeEnum",
    "TurbomoleDispersionCorrection",
    "TurbomoleBasisSet2",
    "TurbomoleFunctional",
    "TurbomoleFunctionalEnum",
    "TurbomoleMethodEnum",
    "TurbomoleQMInput2",
    "TurbomoleSemiempiricalMethod",
    "TurbomoleSemiempiricalInput",
    "OptimizationStepInput",
    "PreOptimizerInput",
    "multistep_optimizer",
    "turbomole2",
    "turbomole_ricc2",
    "turbomole_semiempirical",
]
