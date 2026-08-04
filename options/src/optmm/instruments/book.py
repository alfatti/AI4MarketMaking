"""Frozen-Greeks option book (stages 1-2) and the BBG section-4 universe.

FrozenBook holds everything the HJB needs per instrument:
    O0     model mid at (0, S0, nu0)
    vega   V^i = d O^i / d sqrt(nu) at (0, S0, nu0)
    z      trade size in contracts (Dirac), BBG: 5e5 / O0
    w      per-trade vega, w_i = z_i * vega_i  (the natural Vpi jump unit)
    lam, alpha, k   logistic intensity per instrument (k_i = beta / vega_i);
                    both sides share the same intensity in BBG.

Interface stubs reserved for later composability:
    events() — lifecycle events (barriers, calls, expiries): deferred.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..core.risk import MarketParams
from .heston import price_vega_delta


@dataclass(frozen=True)
class FrozenBook:
    K: jnp.ndarray
    tau: jnp.ndarray
    O0: jnp.ndarray
    vega: jnp.ndarray
    delta: jnp.ndarray
    z: jnp.ndarray
    w: jnp.ndarray          # z * vega
    lam: jnp.ndarray
    alpha: jnp.ndarray
    k: jnp.ndarray          # beta / vega

    @property
    def n_options(self) -> int:
        return int(self.K.shape[0])

    def events(self):
        """Lifecycle events (deferred; interface reserved)."""
        return ()


def build_bbg_book(market: MarketParams, strikes=None, maturities=None,
                   notional_per_trade: float = 5e5, beta: float = 150.0,
                   alpha: float = 0.7, rfq_base: float = 252.0 * 30.0,
                   rfq_decay: float = 0.7) -> FrozenBook:
    strikes = np.array([8.0, 9.0, 10.0, 11.0, 12.0] if strikes is None
                       else strikes, dtype=np.float64)
    maturities = np.array([1.0, 1.5, 2.0, 3.0] if maturities is None
                          else maturities, dtype=np.float64)
    Ks, taus, O0s, vegas, deltas = [], [], [], [], []
    for T in maturities:
        for K in strikes:
            p, v, d = price_vega_delta(market.S0, market.nu0, float(K),
                                       float(T), market.kappa_Q,
                                       market.theta_Q, market.xi, market.rho)
            Ks.append(K); taus.append(T)
            O0s.append(float(p)); vegas.append(float(v)); deltas.append(float(d))
    K = jnp.array(Ks); tau = jnp.array(taus)
    O0 = jnp.array(O0s); vega = jnp.array(vegas); delta = jnp.array(deltas)
    z = notional_per_trade / O0
    lam = rfq_base / (1.0 + rfq_decay * jnp.abs(market.S0 - K))
    return FrozenBook(K=K, tau=tau, O0=O0, vega=vega, delta=delta, z=z,
                      w=z * vega, lam=lam,
                      alpha=jnp.full_like(O0, alpha), k=beta / vega)
