"""Shared per-channel arrays and the A.1-transposed risk coefficients.

Channels: instrument x side, ask block first (psi = +1, sell => n -> n - e_i)
then bid block (psi = -1). Identical convention to the Heston build.

Risk layer (docs/derivations.md section 2): hedge fraction family
e = eta * Delta^pi with
    pen(t, S)   = (gamma / 2) * eta^2 * sigma(t,S)^2 * S^2,
    carry(t, S) = mu * eta * S      (per unit of Delta^pi),
both state-dependent in general; the Merton intercept is exhibited in the
note and zeroed by mandate.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .hamiltonian import LogisticIntensity


def _bc(intensity: LogisticIntensity, extra: int) -> LogisticIntensity:
    """Reshape per-channel params to broadcast over `extra` trailing dims."""
    sh = lambda a: jnp.reshape(a, a.shape + (1,) * extra)
    return LogisticIntensity(lam=sh(intensity.lam), alpha=sh(intensity.alpha),
                             k=sh(intensity.k))


@dataclass(frozen=True)
class Channels:
    z: jnp.ndarray
    psi: jnp.ndarray
    intensity: LogisticIntensity

    @classmethod
    def from_book(cls, book) -> "Channels":
        rep = lambda x: jnp.concatenate([x, x])
        psi = jnp.concatenate([jnp.ones_like(book.z), -jnp.ones_like(book.z)])
        inten = LogisticIntensity(lam=rep(book.lam), alpha=rep(book.alpha),
                                  k=rep(book.k))
        return cls(z=rep(book.z), psi=psi, intensity=inten)


@dataclass(frozen=True)
class SpotMarket:
    """One-factor spot model. sigma_fn(t, S) -> sigma; BS: constant."""

    S0: float
    sigma0: float
    mu: float = 0.0
    beta_cev: float = 1.0    # 1.0 => Black-Scholes; else CEV local vol

    def sigma(self, t, S):
        if self.beta_cev == 1.0:
            return jnp.full_like(jnp.asarray(S, jnp.float64), self.sigma0)
        return self.sigma0 * (jnp.asarray(S, jnp.float64) / self.S0) ** (
            self.beta_cev - 1.0)


def pen_coef(market: SpotMarket, gamma: float, eta: float, t, S):
    """(gamma/2) eta^2 sigma(t,S)^2 S^2 — multiplies (Delta^pi)^2."""
    sig = market.sigma(t, S)
    return 0.5 * gamma * eta * eta * sig * sig * jnp.asarray(S) ** 2


def carry_coef(market: SpotMarket, eta: float, t, S):
    """mu eta S — multiplies Delta^pi (directional lean; zero at mu = 0)."""
    return market.mu * eta * jnp.asarray(S, jnp.float64)


BS_MARKET = SpotMarket(S0=10.0, sigma0=0.15, mu=0.0, beta_cev=1.0)
CEV_MARKET = SpotMarket(S0=10.0, sigma0=0.15, mu=0.0, beta_cev=0.5)
