"""Instruments for the one-factor build.

- Black-Scholes closed forms (r = 0): price, delta, vega, for calls and
  linear combos (the 9.5/10/10.5 butterfly is the delta-sign-flip
  instrument — the one-factor twin of the Heston build's vega-flip spread).
- CEV local-vol pricing by Crank-Nicolson on a log-spot grid (numpy; a
  precompute step, not in any training loop), producing price and delta
  *tables over S* per instrument, exposed through the same grid-interpolant
  pattern as the Heston build's surrogate. sigma(t,S) is taken as given
  (parametric CEV); Dupire calibration is deferred.
- Book construction transposing the BBG section-4 calibration: sizes
  z = 5e5/|O_0|, request intensities decaying in |S0 - K|, logistic slopes
  k = beta / V_ref with V_ref an anchor *vega* scale (client behavior is
  calibrated in implied-vol space; the risk Greek is delta — deliberately
  distinct objects). Greeks are frozen in t over the short horizon
  (tau >= 0.25 vs T = 0.006) and live in S: the binding BBG assumption is
  the one dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm

from ..core.channels import SpotMarket


# ----------------------------- Black-Scholes ------------------------------

def bs_call(S, K, sigma, tau):
    S = jnp.asarray(S, jnp.float64)
    st = sigma * jnp.sqrt(tau)
    d1 = (jnp.log(S / K) + 0.5 * sigma * sigma * tau) / st
    return S * norm.cdf(d1) - K * norm.cdf(d1 - st)


def bs_delta(S, K, sigma, tau):
    S = jnp.asarray(S, jnp.float64)
    st = sigma * jnp.sqrt(tau)
    d1 = (jnp.log(S / K) + 0.5 * sigma * sigma * tau) / st
    return norm.cdf(d1)


def bs_vega(S, K, sigma, tau):
    S = jnp.asarray(S, jnp.float64)
    st = sigma * jnp.sqrt(tau)
    d1 = (jnp.log(S / K) + 0.5 * sigma * sigma * tau) / st
    return S * jnp.sqrt(tau) * norm.pdf(d1)


# legs: list of (weight, K, tau); a call is [(1, K, tau)], the butterfly
# [(1, Kl, tau), (-2, Km, tau), (1, Kh, tau)].
Leg = Tuple[float, float, float]


def combo_price_bs(S, legs: Sequence[Leg], sigma):
    return sum(w * bs_call(S, K, sigma, tau) for (w, K, tau) in legs)


def combo_delta_bs(S, legs: Sequence[Leg], sigma):
    return sum(w * bs_delta(S, K, sigma, tau) for (w, K, tau) in legs)


def combo_vega_bs(S, legs: Sequence[Leg], sigma):
    return sum(w * bs_vega(S, K, sigma, tau) for (w, K, tau) in legs)


# ------------------------- CEV local vol (CN, numpy) -----------------------

def cn_call_surface(market: SpotMarket, K: float, tau: float,
                    x_half: float = 0.5, n_x: int = 401, n_t: int = 240):
    """Crank-Nicolson solve of the r=0 local-vol pricing PDE on x = ln S.

    Returns (S_nodes, price_at_t0, delta_at_t0) as numpy arrays. The PDE is
        dO/dt + 0.5 sigma(S)^2 (d2O/dx2 - dO/dx) = 0,  O(tau, x) = payoff.
    Dirichlet far-field boundaries (O = 0 below, O = S - K above), grid far
    wider than the trading band so boundary error is negligible there.
    """
    x0 = np.log(market.S0)
    x = np.linspace(x0 - x_half, x0 + x_half, n_x)
    dx = x[1] - x[0]
    S = np.exp(x)
    sig = np.asarray(market.sigma(0.0, jnp.asarray(S)))
    a = 0.5 * sig * sig
    dt = tau / n_t
    # theta-scheme (CN): (I - dt/2 L) O_new = (I + dt/2 L) O_old, L applied
    # to interior nodes; L O = a (O'' - O') with central differences.
    lower = a / dx**2 * (1.0 + 0.5 * dx)      # coeff of O_{j-1}: a(1/dx^2 + 1/(2dx))
    diag = -2.0 * a / dx**2
    upper = a / dx**2 * (1.0 - 0.5 * dx)      # coeff of O_{j+1}
    O = np.maximum(S - K, 0.0)
    lo_bc, hi_bc = 0.0, S[-1] - K
    A_l = -0.5 * dt * lower[1:-1]
    A_d = 1.0 - 0.5 * dt * diag[1:-1]
    A_u = -0.5 * dt * upper[1:-1]
    for _ in range(n_t):
        rhs = (O[1:-1] + 0.5 * dt * (lower[1:-1] * O[:-2]
                                     + diag[1:-1] * O[1:-1]
                                     + upper[1:-1] * O[2:]))
        rhs[0] += 0.5 * dt * lower[1] * lo_bc
        rhs[-1] += 0.5 * dt * upper[-2] * hi_bc
        # Thomas solve
        n = rhs.shape[0]
        cp = np.empty(n); dp = np.empty(n)
        cp[0] = A_u[0] / A_d[0]; dp[0] = (rhs[0] - A_l[0] * lo_bc) / A_d[0]
        for j in range(1, n):
            m = A_d[j] - A_l[j] * cp[j - 1]
            cp[j] = A_u[j] / m
            dp[j] = (rhs[j] - A_l[j] * dp[j - 1]) / m
        dp[-1] = (rhs[-1] - A_u[-1] * hi_bc - A_l[-1] * dp[-2]) \
            / (A_d[-1] - A_l[-1] * cp[-2])
        sol = np.empty(n); sol[-1] = dp[-1]
        for j in range(n - 2, -1, -1):
            sol[j] = dp[j] - cp[j] * sol[j + 1]
        O = np.concatenate([[lo_bc], sol, [hi_bc]])
    dOdx = np.gradient(O, dx)
    return S, O, dOdx / S                      # delta = (1/S) dO/dx


def combo_tables_lv(market: SpotMarket, legs: Sequence[Leg], **cn_kwargs):
    S = None; O = None; D = None
    for (w, K, tau) in legs:
        Sn, On, Dn = cn_call_surface(market, K, tau, **cn_kwargs)
        S = Sn if S is None else S
        O = w * On if O is None else O + w * On
        D = w * Dn if D is None else D + w * Dn
    return S, O, D


# ------------------------------- delta grids -------------------------------

@dataclass(frozen=True)
class DeltaGrid:
    """Per-instrument live delta Delta_i(S), linear interpolation, jittable."""

    S_grid: jnp.ndarray            # (nS,)
    delta_tab: jnp.ndarray         # (N, nS)

    def __call__(self, t, S):
        f = jnp.clip((S - self.S_grid[0]) / (self.S_grid[1] - self.S_grid[0]),
                     0.0, self.S_grid.shape[0] - 1.001)
        i = jnp.floor(f).astype(jnp.int32)
        a = f - i
        return (1 - a) * self.delta_tab[:, i] + a * self.delta_tab[:, i + 1]


# --------------------------------- book -----------------------------------

@dataclass(frozen=True)
class SpotBook:
    legs: tuple                    # tuple of leg-tuples per instrument
    O0: jnp.ndarray
    delta0: jnp.ndarray            # anchor deltas at (0, S0)
    vega_ref: jnp.ndarray          # anchor vega scale (slope calibration)
    z: jnp.ndarray
    w: jnp.ndarray                 # z * delta0 (signed anchor risk vector)
    lam: jnp.ndarray
    alpha: jnp.ndarray
    k: jnp.ndarray
    K_body: jnp.ndarray            # body strike per instrument (for lambda)

    @property
    def n_options(self) -> int:
        return int(self.O0.shape[0])


DEFAULT_LEGS = (
    ((1.0, 9.0, 1.0),),
    ((1.0, 10.0, 0.25),),
    ((1.0, 11.0, 1.0),),
    ((1.0, 9.5, 0.25), (-2.0, 10.0, 0.25), (1.0, 10.5, 0.25)),  # butterfly
)


def build_book(market: SpotMarket, legs_list=DEFAULT_LEGS,
               S_band: float = 0.08, n_S: int = 41,
               notional_per_trade: float = 5e5, beta: float = 150.0,
               alpha: float = 0.7, rfq_base: float = 252.0 * 30.0,
               rfq_decay: float = 0.7):
    """Returns (SpotBook, DeltaGrid) for BS or CEV local vol."""
    S_lo, S_hi = market.S0 * (1 - S_band), market.S0 * (1 + S_band)
    S_grid = np.linspace(S_lo, S_hi, n_S)
    N = len(legs_list)
    tab = np.zeros((N, n_S))
    O0 = np.zeros(N); d0 = np.zeros(N); vref = np.zeros(N); Kb = np.zeros(N)
    for i, legs in enumerate(legs_list):
        Kb[i] = legs[0][1] if len(legs) == 1 else legs[len(legs) // 2][1]
        if market.beta_cev == 1.0:
            tab[i] = np.asarray(combo_delta_bs(jnp.asarray(S_grid), legs,
                                               market.sigma0))
            O0[i] = float(combo_price_bs(market.S0, legs, market.sigma0))
            d0[i] = float(combo_delta_bs(market.S0, legs, market.sigma0))
        else:
            S_cn, O_cn, D_cn = combo_tables_lv(market, legs)
            tab[i] = np.interp(S_grid, S_cn, D_cn)
            O0[i] = float(np.interp(market.S0, S_cn, O_cn))
            d0[i] = float(np.interp(market.S0, S_cn, D_cn))
        # slope calibration: anchor BS vega scale at sigma(0, S0); for
        # combos with near-zero anchor vega, max |vega| over the band
        sig_anchor = float(market.sigma(0.0, jnp.asarray(market.S0)))
        vg = np.asarray(combo_vega_bs(jnp.asarray(S_grid), legs, sig_anchor))
        v_at = abs(float(combo_vega_bs(market.S0, legs, sig_anchor)))
        vref[i] = v_at if v_at > 0.1 else np.max(np.abs(vg))
    z = notional_per_trade / np.abs(O0)
    lam = rfq_base / (1.0 + rfq_decay * np.abs(market.S0 - Kb))
    book = SpotBook(legs=tuple(tuple(l) for l in legs_list),
                    O0=jnp.asarray(O0), delta0=jnp.asarray(d0),
                    vega_ref=jnp.asarray(vref), z=jnp.asarray(z),
                    w=jnp.asarray(z * d0), lam=jnp.asarray(lam),
                    alpha=jnp.full(N, alpha), k=jnp.asarray(beta / vref),
                    K_body=jnp.asarray(Kb))
    return book, DeltaGrid(S_grid=jnp.asarray(S_grid),
                           delta_tab=jnp.asarray(tab))
