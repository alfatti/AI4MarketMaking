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


@dataclass(frozen=True)
class SizeDistribution:
    """Discrete request-size distribution nu (Section 5). Point mass = fixed size."""

    sizes: np.ndarray
    probs: np.ndarray

    def __post_init__(self) -> None:
        s = np.asarray(self.sizes, dtype=float)
        p = np.asarray(self.probs, dtype=float)
        object.__setattr__(self, "sizes", s)
        object.__setattr__(self, "probs", p)
        if s.shape != p.shape or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
            raise ValueError("sizes/probs must match and probs must sum to 1.")

    @classmethod
    def point(cls, z: float) -> "SizeDistribution":
        return cls(sizes=np.array([z]), probs=np.array([1.0]))

    @property
    def m1(self) -> float:
        return float(self.probs @ self.sizes)

    @property
    def m2(self) -> float:
        return float(self.probs @ self.sizes**2)


def build_multi_asset_engine(
    intensities_b: list[IntensityModel],
    intensities_a: list[IntensityModel],
    Sigma: np.ndarray,
    config: ModelConfig,
    size_dists: list[SizeDistribution] | None = None,
    mu: np.ndarray | None = None,
    Q_limits: list[float | None] | None = None,
) -> list[QuoteEngine]:
    """Wire d assets end-to-end (Section 5 model: drift, distributed sizes).

    Returns one QuoteEngine per asset; all share the same RiccatiSolution, so
    cross-asset inventory effects flow through the matrix A automatically.
    Model B only for distributed sizes (alphas are z-independent; Remark 2).
    """
    d = len(intensities_b)
    if size_dists is None:
        size_dists = [SizeDistribution.point(config.z)] * d
    if config.xi != 0.0 and any(len(sd.sizes) > 1 for sd in size_dists):
        raise NotImplementedError("Distributed sizes implemented for Model B (xi=0) only.")

    hams_b = [HamiltonianSide(intensity=lb, z=sd.m1, xi=config.xi)
              for lb, sd in zip(intensities_b, size_dists)]
    hams_a = [HamiltonianSide(intensity=la, z=sd.m1, xi=config.xi)
              for la, sd in zip(intensities_a, size_dists)]

    ab = np.array([h.alphas() for h in hams_b])   # (d, 3)
    aa = np.array([h.alphas() for h in hams_a])
    coeffs = ProxyCoefficients(
        alpha0_b=ab[:, 0], alpha1_b=ab[:, 1], alpha2_b=ab[:, 2],
        alpha0_a=aa[:, 0], alpha1_a=aa[:, 1], alpha2_a=aa[:, 2],
        z=np.array([sd.m1 for sd in size_dists]),
        m2=np.array([sd.m2 for sd in size_dists]),
    )
    sol = solve_riccati(coeffs, np.asarray(Sigma, dtype=float), gamma=config.gamma,
                        T=config.T, n_steps=config.n_steps, mu=mu)
    if Q_limits is None:
        Q_limits = [config.Q_limit] * d
    return [
        QuoteEngine(ham_b=hams_b[i], ham_a=hams_a[i], solution=sol,
                    asset=i, Q_limit=Q_limits[i], ergodic=config.ergodic)
        for i in range(d)
    ]


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
