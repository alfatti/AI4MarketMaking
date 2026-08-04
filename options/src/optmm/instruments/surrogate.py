"""Stage-3 instruments: state-dependent Greeks via a precomputed (S, nu) grid.

`build_stage3_book` prices a small universe — vanilla calls plus a tight
call spread (long K_lo, short K_hi) — on an (S, nu) grid with the COS pricer
and AD vega, and returns:

- a FrozenBook anchored at (S0, nu0) (sizes z = notional / |price|,
  intensities with k_i = beta / vega_ref where vega_ref = max |vega| over
  the grid: the BBG per-instrument slope calibration divides by vega, which
  is near zero for a tight spread at the money — documented choice);
- a jittable `vega_fn(t, S, nu) -> (N,)` bilinear interpolant.

The call spread is the sign-flip instrument: its vega is positive below the
strikes and negative above, flipping inside the sampled S band — the
barrier-like feature the stage-3 machinery must handle.  Greeks are frozen
in t over the short horizon (time decay over T ~ days is negligible next to
the S/nu dependence; documented, and trivially upgradable to a (t, S, nu)
grid).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..core.risk import MarketParams
from .book import FrozenBook
from .heston import price_vega_delta


@dataclass(frozen=True)
class VegaGrid:
    S_grid: jnp.ndarray            # (nS,)
    nu_grid: jnp.ndarray           # (nnu,)
    vega_tab: jnp.ndarray          # (N, nS, nnu)

    def __call__(self, t, S, nu):
        fS = jnp.clip((S - self.S_grid[0])
                      / (self.S_grid[1] - self.S_grid[0]),
                      0.0, self.S_grid.shape[0] - 1.001)
        fn = jnp.clip((nu - self.nu_grid[0])
                      / (self.nu_grid[1] - self.nu_grid[0]),
                      0.0, self.nu_grid.shape[0] - 1.001)
        iS = jnp.floor(fS).astype(jnp.int32)
        inu = jnp.floor(fn).astype(jnp.int32)
        aS, an = fS - iS, fn - inu
        v00 = self.vega_tab[:, iS, inu]
        v10 = self.vega_tab[:, iS + 1, inu]
        v01 = self.vega_tab[:, iS, inu + 1]
        v11 = self.vega_tab[:, iS + 1, inu + 1]
        return ((1 - aS) * (1 - an) * v00 + aS * (1 - an) * v10
                + (1 - aS) * an * v01 + aS * an * v11)


def _priced_leg(market: MarketParams, S, nu, K, tau):
    return price_vega_delta(S, nu, K, tau, market.kappa_Q, market.theta_Q,
                            market.xi, market.rho)


def build_stage3_book(market: MarketParams,
                      vanillas: Sequence[Tuple[float, float]] = ((9.0, 1.0),
                                                                (10.0, 1.0),
                                                                (11.0, 1.0)),
                      spread: Tuple[float, float, float] = (9.75, 10.25, 1.0),
                      S_band: float = 0.08, n_S: int = 33, n_nu: int = 17,
                      nu_lo: float = 0.0144, nu_hi: float = 0.0324,
                      notional_per_trade: float = 5e5, beta: float = 150.0,
                      alpha: float = 0.7, rfq_base: float = 252.0 * 30.0,
                      rfq_decay: float = 0.7):
    """Returns (FrozenBook anchored at (S0, nu0), vega_fn, vega_ref)."""
    S_grid = np.linspace(market.S0 * (1 - S_band), market.S0 * (1 + S_band),
                         n_S)
    nu_grid = np.linspace(nu_lo, nu_hi, n_nu)
    legs = [("call", K, T) for (K, T) in vanillas] + [("spread",) + spread]
    N = len(legs)
    tab = np.zeros((N, n_S, n_nu))
    O0 = np.zeros(N); vega0 = np.zeros(N); delta0 = np.zeros(N)
    Ks = np.zeros(N); taus = np.zeros(N)
    for i, leg in enumerate(legs):
        if leg[0] == "call":
            _, K, T = leg
            for a, S in enumerate(S_grid):
                for b, nu in enumerate(nu_grid):
                    _, v, _ = _priced_leg(market, float(S), float(nu), K, T)
                    tab[i, a, b] = float(v)
            p, v, d = _priced_leg(market, market.S0, market.nu0, K, T)
            O0[i], vega0[i], delta0[i] = float(p), float(v), float(d)
            Ks[i], taus[i] = K, T
        else:
            _, K1, K2, T = leg
            for a, S in enumerate(S_grid):
                for b, nu in enumerate(nu_grid):
                    _, v1, _ = _priced_leg(market, float(S), float(nu), K1, T)
                    _, v2, _ = _priced_leg(market, float(S), float(nu), K2, T)
                    tab[i, a, b] = float(v1) - float(v2)
            p1, v1, d1 = _priced_leg(market, market.S0, market.nu0, K1, T)
            p2, v2, d2 = _priced_leg(market, market.S0, market.nu0, K2, T)
            O0[i] = float(p1) - float(p2)
            vega0[i] = float(v1) - float(v2)
            delta0[i] = float(d1) - float(d2)
            Ks[i], taus[i] = 0.5 * (K1 + K2), T

    vega_ref = np.max(np.abs(tab), axis=(1, 2))       # slope calibration base
    z = notional_per_trade / np.abs(O0)
    lam = rfq_base / (1.0 + rfq_decay * np.abs(market.S0 - Ks))
    book = FrozenBook(K=jnp.asarray(Ks), tau=jnp.asarray(taus),
                      O0=jnp.asarray(O0), vega=jnp.asarray(vega0),
                      delta=jnp.asarray(delta0), z=jnp.asarray(z),
                      w=jnp.asarray(z * vega_ref), lam=jnp.asarray(lam),
                      alpha=jnp.full(N, alpha),
                      k=jnp.asarray(beta / vega_ref))
    vega_fn = VegaGrid(S_grid=jnp.asarray(S_grid),
                       nu_grid=jnp.asarray(nu_grid),
                       vega_tab=jnp.asarray(tab))
    return book, vega_fn, jnp.asarray(vega_ref)
