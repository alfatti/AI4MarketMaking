"""Config layer: the policy knobs, separated from estimated parameters.

Estimated (calibration.py): A_b, k_b, A_a, k_a, sigma.
Chosen (here): gamma, z, Q_limit, T, xi (Model A vs B), ergodic flag.

These are the dimensions that multiply under tiering / multi-size later, so
they live in one dataclass rather than scattered constructor args.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hamiltonian import HamiltonianSide
from .intensity import IntensityModel
from .quotes import QuoteEngine
from .riccati import ProxyCoefficients, solve_riccati


@dataclass(frozen=True)
class ModelConfig:
    gamma: float          # risk aversion (the calibrate-to-spread knob)
    z: float              # transaction size
    T: float              # horizon; irrelevant if ergodic=True
    xi: float = 0.0       # 0.0 => Model B; gamma => Model A
    Q_limit: float | None = None
    ergodic: bool = True
    n_steps: int = 500


def build_single_cusip_engine(
    intensity_b: IntensityModel,
    intensity_a: IntensityModel,
    sigma: float,
    config: ModelConfig,
) -> QuoteEngine:
    """End-to-end wiring for d = 1: intensities -> Hamiltonians -> alphas ->
    Riccati -> QuoteEngine. The d > 1 version differs only in array shapes.
    """
    ham_b = HamiltonianSide(intensity=intensity_b, z=config.z, xi=config.xi)
    ham_a = HamiltonianSide(intensity=intensity_a, z=config.z, xi=config.xi)

    a0b, a1b, a2b = ham_b.alphas()
    a0a, a1a, a2a = ham_a.alphas()

    coeffs = ProxyCoefficients(
        alpha0_b=np.array([a0b]), alpha1_b=np.array([a1b]), alpha2_b=np.array([a2b]),
        alpha0_a=np.array([a0a]), alpha1_a=np.array([a1a]), alpha2_a=np.array([a2a]),
        z=np.array([config.z]),
    )
    Sigma = np.array([[sigma**2]])
    sol = solve_riccati(coeffs, Sigma, gamma=config.gamma, T=config.T,
                        n_steps=config.n_steps)
    return QuoteEngine(
        ham_b=ham_b, ham_a=ham_a, solution=sol,
        asset=0, Q_limit=config.Q_limit, ergodic=config.ergodic,
    )
