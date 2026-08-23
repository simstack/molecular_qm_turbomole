"""TURBOMOLE capabilities for molecular quantum mechanics in Simstack."""

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
    TurbomoleQMInput2,
)
from molecular_qm_turbomole.nodes.turbomole2 import turbomole2

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
    "TurbomoleQMInput2",
    "turbomole2",
]
