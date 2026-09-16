"""Real-time bath compression into coupled Lindblad representations."""

from ._version import __version__
from .api import fit_correlation, fit_fermionic
from .exceptions import (
    BandLimitError,
    BandLimitWarning,
    CoupledLindbladError,
    InvalidSamplesError,
    OptionalDependencyError,
    RealizationError,
)
from .exponential import fit_exponentials
from .optimization import optimize_after_sdp
from .route_sdp import cvxpy_available, realize_sdp
from .types import (
    ExponentialFit,
    FermionicLindbladFit,
    FitDiagnostics,
    LindbladFit,
    LindbladModel,
)

__all__ = [
    "__version__",
    "BandLimitError",
    "BandLimitWarning",
    "CoupledLindbladError",
    "ExponentialFit",
    "FermionicLindbladFit",
    "FitDiagnostics",
    "InvalidSamplesError",
    "LindbladFit",
    "LindbladModel",
    "OptionalDependencyError",
    "RealizationError",
    "cvxpy_available",
    "fit_correlation",
    "fit_exponentials",
    "fit_fermionic",
    "optimize_after_sdp",
    "realize_sdp",
]
