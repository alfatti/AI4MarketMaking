"""Risk layer: hedge-tilt family and the coefficients it feeds the HJB.

Proportional hedge family h(c) = -c * xi * Vpi / (2 sqrt(nu) S): the quoting
problem sees the hedge layer only through
    m(c) = 1 - 2 rho c + c^2         (m(0)=1 delta-neutral, m(rho)=1-rho^2),
which scales the vega penalty coefficient
    pen_coef(c) = gamma * xi^2 * m(c) / 8,
so the running penalty is  pen_coef * (Vpi)^2.

Carry coefficient (per unit portfolio vega, state-dependent in nu):
    carry(t, nu) = (a_P(t,nu) - a_Q(t,nu)) / (2 sqrt(nu)).

Derivations: docs/derivations.md section 2. The Merton term mu/(gamma nu S) is
deliberately zeroed (mandate separation) — see the note.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


def hedge_variance_multiplier(c: float, rho: float) -> float:
    """m(c) = 1 - 2 rho c + c^2."""
    return 1.0 - 2.0 * rho * c + c * c


@dataclass(frozen=True)
class MarketParams:
    """Underlying dynamics parameters (Heston under P and Q)."""

    S0: float
    nu0: float
    kappa_P: float
    theta_P: float
    kappa_Q: float
    theta_Q: float
    xi: float
    rho: float
    mu: float = 0.0

    def a_P(self, nu):
        return self.kappa_P * (self.theta_P - nu)

    def a_Q(self, nu):
        return self.kappa_Q * (self.theta_Q - nu)

    def carry_coef(self, nu):
        """(a_P - a_Q) / (2 sqrt(nu)) — per unit portfolio vega."""
        return (self.a_P(nu) - self.a_Q(nu)) / (2.0 * jnp.sqrt(nu))

    def feller_ok(self) -> bool:
        return (2 * self.kappa_P * self.theta_P > self.xi**2) and (
            2 * self.kappa_Q * self.theta_Q > self.xi**2
        )


def penalty_coef(gamma: float, xi: float, rho: float, c: float) -> float:
    """gamma * xi^2 * m(c) / 8, multiplying (Vpi)^2 in the running penalty."""
    return gamma * xi * xi * hedge_variance_multiplier(c, rho) / 8.0


BBG_MARKET = MarketParams(
    S0=10.0, nu0=0.0225, kappa_P=2.0, theta_P=0.04,
    kappa_Q=3.0, theta_Q=0.0225, xi=0.2, rho=-0.5, mu=0.0,
)
