"""MarketSpec: general multi-asset Model B market description.

Conventions
-----------
side axis: 0 = bid, 1 = ask.
Intensity families (kind): 0 = exponential  Lambda(delta) = A * exp(-k * delta),
                           1 = logistic     Lambda(delta) = lam / (1 + exp(alpha + beta*delta)).
Intensity params ip[..., :]: exponential -> (A, k, unused), logistic -> (lam, alpha, beta).
Sizes are atomic: z_atoms (d, n_tiers, 2, K) with probabilities p_atoms summing to 1
over the K axis (pad unused atoms with prob 0 and z = 1.0 to keep shapes rectangular).

Model B (xi = 0) throughout: risk-adjusted expectation, dense running penalty.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import jax.numpy as jnp


@dataclasses.dataclass
class MarketSpec:
    d: int                      # number of assets
    n_tiers: int                # number of client tiers
    T: float                    # horizon
    gamma: float                # risk aversion
    mu: np.ndarray              # (d,) price drifts
    Sigma: np.ndarray           # (d, d) covariance (PSD)
    Q: np.ndarray               # (d,) risk limits (|q_i| <= Q_i)
    delta_inf: float            # quotes bounded below by -delta_inf
    kind: np.ndarray            # (d, n_tiers, 2) int intensity family
    ip: np.ndarray              # (d, n_tiers, 2, 3) intensity params
    z_atoms: np.ndarray         # (d, n_tiers, 2, K) request-size atoms
    p_atoms: np.ndarray         # (d, n_tiers, 2, K) atom probabilities
    c: np.ndarray               # (d, n_tiers, 2) fixed transaction costs

    def __post_init__(self):
        d, n, K = self.d, self.n_tiers, self.z_atoms.shape[-1]
        self.mu = np.asarray(self.mu, float).reshape(d)
        self.Sigma = np.asarray(self.Sigma, float).reshape(d, d)
        assert np.allclose(self.Sigma, self.Sigma.T), "Sigma must be symmetric"
        self.Q = np.asarray(self.Q, float).reshape(d)
        self.kind = np.asarray(self.kind, int).reshape(d, n, 2)
        # Branch safety (avoiding NaN/inf in unselected jnp.where branches) is handled
        # at the use-site in hamiltonians.py; params are stored as given.
        self.ip = np.asarray(self.ip, float).reshape(d, n, 2, 3).copy()
        self.z_atoms = np.asarray(self.z_atoms, float).reshape(d, n, 2, K)
        self.p_atoms = np.asarray(self.p_atoms, float).reshape(d, n, 2, K)
        assert np.all(self.z_atoms > 0), "size atoms must be positive"
        s = self.p_atoms.sum(-1)
        assert np.allclose(s, 1.0), "atom probabilities must sum to 1 per (i,n,side)"
        self.c = np.asarray(self.c, float).reshape(d, n, 2)
        assert self.delta_inf > 0

    # ---- moments of the size distribution, m_k = E[z^k], k in {-1,0,1,2,3} ----
    def moments(self) -> dict:
        m = {}
        for k in (-1, 0, 1, 2, 3):
            m[k] = (self.p_atoms * self.z_atoms ** k).sum(-1)   # (d, n_tiers, 2)
        return m

    def zbar(self) -> np.ndarray:
        """Mean request size per asset (averaged over tiers/sides/atoms) — boundary-layer length scale."""
        m1 = (self.p_atoms * self.z_atoms).sum(-1)              # (d, n, 2)
        return m1.mean(axis=(1, 2))                             # (d,)

    def to_jax(self) -> dict:
        """Device-array view used by jitted code."""
        return dict(
            mu=jnp.asarray(self.mu), Sigma=jnp.asarray(self.Sigma), Q=jnp.asarray(self.Q),
            kind=jnp.asarray(self.kind), ip=jnp.asarray(self.ip),
            z=jnp.asarray(self.z_atoms), pz=jnp.asarray(self.p_atoms), c=jnp.asarray(self.c),
            gamma=self.gamma, T=self.T, delta_inf=self.delta_inf,
        )


def single_asset_demo_spec() -> MarketSpec:
    """d=1 demo: asymmetric exponential intensities, drift, fixed costs, one tier, unit atom."""
    d, n, K = 1, 1, 1
    kind = np.zeros((d, n, 2), int)                 # exponential both sides
    ip = np.zeros((d, n, 2, 3))
    ip[0, 0, 0] = (1.0, 2.5, 0.0)                   # bid: A=1.0, k=2.5
    ip[0, 0, 1] = (1.2, 2.2, 0.0)                   # ask: A=1.2, k=2.2 (asymmetric)
    z = np.ones((d, n, 2, K))
    p = np.ones((d, n, 2, K))
    c = np.full((d, n, 2), 0.01)
    return MarketSpec(d=d, n_tiers=n, T=1.0, gamma=0.3, mu=np.array([0.05]),
                      Sigma=np.array([[1.4 ** 2]]), Q=np.array([6.0]), delta_inf=3.0,
                      kind=kind, ip=ip, z_atoms=z, p_atoms=p, c=c)


def random_general_spec(seed: int = 0, d: int = 3, n_tiers: int = 2, K: int = 3) -> MarketSpec:
    """Random asymmetric, multi-tier, mixed-family spec for Phase-0 structural tests."""
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 2, size=(d, n_tiers, 2))
    ip = np.zeros((d, n_tiers, 2, 3))
    # exponential slots
    ip[..., 0] = rng.uniform(0.5, 2.0, size=(d, n_tiers, 2))     # A or lam
    ip[..., 1] = rng.uniform(1.5, 4.0, size=(d, n_tiers, 2))     # k or alpha
    ip[..., 2] = rng.uniform(2.0, 6.0, size=(d, n_tiers, 2))     # beta (logistic)
    # keep logistic alpha moderate
    ip[..., 1] = np.where(kind == 1, rng.uniform(0.2, 1.0, size=(d, n_tiers, 2)), ip[..., 1])
    z = rng.uniform(0.5, 2.0, size=(d, n_tiers, 2, K))
    p = rng.uniform(0.2, 1.0, size=(d, n_tiers, 2, K)); p = p / p.sum(-1, keepdims=True)
    c = rng.uniform(0.0, 0.05, size=(d, n_tiers, 2))
    B = rng.normal(size=(d, d)); Sigma = B @ B.T / d + 0.5 * np.eye(d)
    mu = rng.normal(scale=0.05, size=d)
    return MarketSpec(d=d, n_tiers=n_tiers, T=1.5, gamma=0.2, mu=mu, Sigma=Sigma,
                      Q=rng.uniform(4.0, 8.0, size=d), delta_inf=3.0,
                      kind=kind, ip=ip, z_atoms=z, p_atoms=p, c=c)
