"""mmquote: closed-form market-making quotes (Bergault-Evangelista-Gueant-Vieira proxy).

v1 scope: single CUSIP, exponential intensity, Model B default (xi = 0),
matrix-shaped Riccati exercised at d = 1. Seams: IntensityModel interface,
matrix Riccati, separated bid/ask paths, config layer.
"""

from .calibration import (
    calibrate_gamma_to_spread,
    estimate_sigma,
    fit_exponential_intensity,
)
from .config import ModelConfig, build_single_cusip_engine
from .hamiltonian import HamiltonianSide
from .intensity import ExponentialIntensity, IntensityModel
from .quotes import Quote, QuoteEngine
from .riccati import ProxyCoefficients, RiccatiSolution, solve_riccati, theta_check

__all__ = [
    "IntensityModel", "ExponentialIntensity",
    "HamiltonianSide",
    "ProxyCoefficients", "RiccatiSolution", "solve_riccati", "theta_check",
    "Quote", "QuoteEngine",
    "ModelConfig", "build_single_cusip_engine",
    "fit_exponential_intensity", "estimate_sigma", "calibrate_gamma_to_spread",
]
