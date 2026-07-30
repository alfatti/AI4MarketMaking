"""Bridge between mmquote (numpy closed-form engine) and hjbpinn (JAX PINN stack).

mmquote supplies the production-shaped closed-form path (intensity models,
matrix Riccati, QuoteEngine, d-dim FD benchmark). hjbpinn supplies the exact-B
quadrature proxy, the finite-Q PINN solver, and the certificate. This module
converts between the two so Section-6 style studies can run both benchmarks
from one specification, and cross-validate the stacks against each other.
"""
from __future__ import annotations

import numpy as np

from .spec import MarketSpec


def spec_from_mmquote(intensities_b, intensities_a, Sigma, gamma, T, size_dists,
                      Q_limits, mu=None, delta_inf=5.0, costs=None) -> MarketSpec:
    """Build a MarketSpec from mmquote objects (single tier, costless by default).

    Restrictions inherited from MarketSpec: atoms shared across sides per asset
    (mmquote's SizeDistribution is per-asset anyway); one tier.
    """
    from mmquote.intensity import ExponentialIntensity, LogisticIntensity

    d = len(intensities_b)
    K = max(len(sd.sizes) for sd in size_dists)
    kind = np.zeros((d, 1, 2), int)
    ip = np.zeros((d, 1, 2, 3))
    z_atoms = np.ones((d, 1, 2, K))
    p_atoms = np.zeros((d, 1, 2, K))
    for i, sd in enumerate(size_dists):
        if len(sd.sizes) != K:
            raise ValueError("all assets must share the atom count (pad with prob 0)")
        z_atoms[i, 0, :, :] = sd.sizes[None, :]
        p_atoms[i, 0, :, :] = sd.probs[None, :]
    for side, ints in ((0, intensities_b), (1, intensities_a)):
        for i, lam in enumerate(ints):
            if isinstance(lam, ExponentialIntensity):
                kind[i, 0, side] = 0
                ip[i, 0, side] = (lam.A, lam.k, 0.0)
            elif isinstance(lam, LogisticIntensity):
                kind[i, 0, side] = 1
                ip[i, 0, side] = (lam.lam_rfq, lam.alpha, lam.beta)
            else:
                raise TypeError(f"unsupported intensity {type(lam)}")
    c = np.zeros((d, 1, 2)) if costs is None else np.asarray(costs, float).reshape(d, 1, 2)
    return MarketSpec(d=d, n_tiers=1, T=float(T), gamma=float(gamma),
                      mu=np.zeros(d) if mu is None else np.asarray(mu, float),
                      Sigma=np.asarray(Sigma, float), Q=np.asarray(Q_limits, float),
                      delta_inf=float(delta_inf), kind=kind, ip=ip,
                      z_atoms=z_atoms, p_atoms=p_atoms, c=c)


def section6_spec(delta_inf: float = 5.0) -> MarketSpec:
    """The two-asset Section 6 specification (verbatim from the mmquote
    replication notebook): logistic lam_rfq=30, alpha=0.7, beta=30 on every
    (asset, side); sigma = (1.2, 0.6), rho = 0.5; mu = (+0.1, -0.1);
    gamma = 8e-6; T = 7; atoms {6250, 12500, 18750, 25000} with probs
    {.534, .35, .097, .019}; Q = (75000, 300000); costless, one tier."""
    d, K = 2, 4
    kind = np.ones((d, 1, 2), int)
    ip = np.zeros((d, 1, 2, 3)); ip[..., 0] = 30.0; ip[..., 1] = 0.7; ip[..., 2] = 30.0
    z = np.broadcast_to(np.array([6250., 12500., 18750., 25000.]), (d, 1, 2, K)).copy()
    p = np.broadcast_to(np.array([0.534, 0.350, 0.097, 0.019]), (d, 1, 2, K)).copy()
    return MarketSpec(d=d, n_tiers=1, T=7.0, gamma=8e-6, mu=np.array([0.1, -0.1]),
                      Sigma=np.array([[1.44, 0.36], [0.36, 0.36]]),
                      Q=np.array([75_000.0, 300_000.0]), delta_inf=delta_inf,
                      kind=kind, ip=ip, z_atoms=z, p_atoms=p,
                      c=np.zeros((d, 1, 2)))
