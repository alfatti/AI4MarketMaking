"""Exact policy evaluation on the (t, S, n) lattice: the regret machinery.

For a *fixed* quoting policy delta(t, S, n) the value V^pi solves the
linear backward equation

    dV/dtau = mu S dV/dS + 0.5 sigma^2 S^2 d2V/dS2 + carry Dpi - pen Dpi^2
              + sum_ch z 1{adm} Lambda(delta_ch) (delta_ch - [V - V(n')]/z),

the same equation as the HJB with the Hamiltonian sup replaced by the
policy's own term. Solving it prices any implementable policy exactly (to
RK4 tolerance), so the regret v* - V^pi is a pointwise EUR field with a
built-in validity check (nonnegativity). Policies are `PolicyTables`:
quotes stored at forward-time slices and held piecewise-constant between
them — "refreshed every stride" is the *definition* of the policy being
priced, not an approximation of a continuous one.

Builders: `from_lattice` (the exact policy), `from_pinn` (greedy from a
trained value network, via value tables on the same grid), `static` (the
inventory-blind best margin, argmax at p = 0), `frozen_anchor` (quotes
keyed to the frozen risk snapshot x = n . w0 through the reduced1d value,
held from the t = 0 sheet — the rolling-refresh-free incumbent).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from ..core.channels import Channels, SpotMarket, _bc, carry_coef, pen_coef
from ..core.state import Box
from ..instruments.spot_instruments import DeltaGrid, SpotBook
from .lattice import (LatticeSolution, _build_neighbors, quotes_from_vtable)
from .pinn import solve_reduced1d


@dataclass
class PolicyTables:
    t_slices: jnp.ndarray          # (n_slices,) forward time, ascending
    delta: jnp.ndarray             # (n_slices, n_ch, n_S, M) float32


@dataclass
class EvalResult:
    V0: jnp.ndarray                # (n_S, M) at t = 0
    n_pts: np.ndarray
    S_grid: jnp.ndarray


# ------------------------------ builders ----------------------------------

def from_lattice(lat: LatticeSolution, delta_inf=-5.0) -> PolicyTables:
    # slice-by-slice: peak memory = one slice's argmax intermediates
    q1 = jax.jit(lambda v: jnp.where(
        lat.adm_tgt[:, None, :],
        quotes_from_vtable(v, lat.nb_idx, lat.channels, delta_inf),
        delta_inf).astype(jnp.float32))
    d = jnp.stack([q1(lat.v_hist[k])
                   for k in range(int(lat.t_hist.shape[0]))])
    return PolicyTables(t_slices=lat.t_hist, delta=d)


def from_pinn(params, spec, value_fn, lat: LatticeSolution,
              delta_inf=-5.0):
    """Returns (PolicyTables, v_hist_pinn) on the lattice's own slices."""
    n_pts_j = jnp.asarray(lat.n_pts, jnp.float64)

    def v_slice(t):
        return jax.vmap(lambda s: jax.vmap(
            lambda m: value_fn(params, spec, t, s, m))(n_pts_j))(lat.S_grid)

    q1 = jax.jit(lambda v: jnp.where(
        lat.adm_tgt[:, None, :],
        quotes_from_vtable(v, lat.nb_idx, lat.channels, delta_inf),
        delta_inf).astype(jnp.float32))
    vs, ds = [], []
    for t in np.asarray(lat.t_hist):
        v = v_slice(float(t))
        vs.append(v.astype(jnp.float32)); ds.append(q1(v))
    return PolicyTables(t_slices=lat.t_hist,
                        delta=jnp.stack(ds)), jnp.stack(vs)


def static(book: SpotBook, n_S: int, M: int, delta_inf=-5.0) -> PolicyTables:
    ch = Channels.from_book(book)
    d0 = ch.intensity.argmax_delta(jnp.zeros_like(ch.z), delta_inf)
    d = jnp.broadcast_to(d0[None, :, None, None], (1, d0.shape[0], n_S, M))
    return PolicyTables(t_slices=jnp.zeros(1), delta=d.astype(jnp.float32))


def frozen_anchor(book: SpotBook, market: SpotMarket, gamma: float,
                  eta: float, box: Box, T: float, n_S: int,
                  n_pts: np.ndarray, delta_inf=-5.0) -> PolicyTables:
    w0 = np.asarray(book.w)
    x_pts = n_pts @ w0                              # frozen snapshot risk
    Dbar = 1.2 * float(np.max(np.abs(x_pts))) + 1.0
    red = solve_reduced1d(book, market, gamma, Dbar, T, eta=eta,
                          n_x=401, nt=1200)
    ch = red.channels
    xg = np.asarray(red.x_grid); v = np.asarray(red.v0)
    w_ch = np.concatenate([w0, w0])
    psi = np.asarray(ch.psi)
    d = np.full((2 * len(w0), len(x_pts)), delta_inf, dtype=np.float64)
    for c_i in range(2 * len(w0)):
        tgt = x_pts - psi[c_i] * w_ch[c_i]
        p = (np.interp(x_pts, xg, v) - np.interp(tgt, xg, v)) / float(ch.z[c_i])
        one = Channels(z=ch.z[c_i:c_i+1], psi=ch.psi[c_i:c_i+1],
                       intensity=type(ch.intensity)(
                           lam=ch.intensity.lam[c_i:c_i+1],
                           alpha=ch.intensity.alpha[c_i:c_i+1],
                           k=ch.intensity.k[c_i:c_i+1]))
        d[c_i] = np.asarray(_bc(one.intensity, 1).argmax_delta(
            jnp.asarray(p)[None, :], delta_inf))[0]
    dj = jnp.broadcast_to(jnp.asarray(d)[None, :, None, :],
                          (1, d.shape[0], n_S, d.shape[1]))
    return PolicyTables(t_slices=jnp.zeros(1),
                        delta=dj.astype(jnp.float32))


# ------------------------------ evaluator ---------------------------------

def evaluate_policy(book: SpotBook, dgrid: DeltaGrid, market: SpotMarket,
                    gamma: float, box: Box, T: float, pol: PolicyTables,
                    eta: float = 1.0, nt: int = 200, n_S: int = 41,
                    S_band: float = 0.08, periodic: bool = False
                    ) -> EvalResult:
    ch = Channels.from_book(book)
    n_pts, flat_index, nb_idx, adm_tgt = _build_neighbors(book, box, periodic)
    M = n_pts.shape[0]
    S = jnp.linspace(market.S0 * (1 - S_band), market.S0 * (1 + S_band), n_S)
    dS = float(S[1] - S[0])
    dt = T / nt
    sig = market.sigma(0.0, S)
    adv = market.mu * S
    diff_c = 0.5 * sig * sig * S * S
    penS = pen_coef(market, gamma, eta, 0.0, S)
    carS = carry_coef(market, eta, 0.0, S)
    dtab = jax.vmap(lambda s: dgrid(0.0, s))(S)
    Dpi = (dtab * book.z[None, :]) @ jnp.asarray(n_pts.T, jnp.float64)
    run_src = carS[:, None] * Dpi - penS[:, None] * Dpi**2
    nb_idx_j = jnp.asarray(nb_idx)
    adm_j = jnp.asarray(adm_tgt)
    z3 = ch.z[:, None, None]
    inten3 = _bc(ch.intensity, 2)
    t_sl = pol.t_slices
    pol_d = pol.delta

    def rhs(v, d_slice):
        d1 = jnp.zeros_like(v)
        d1 = d1.at[1:-1].set((v[2:] - v[:-2]) / (2 * dS))
        d2 = jnp.zeros_like(v)
        d2 = d2.at[1:-1].set((v[2:] - 2 * v[1:-1] + v[:-2]) / dS**2)
        d2 = d2.at[0].set(2 * (v[1] - v[0]) / dS**2)
        d2 = d2.at[-1].set(2 * (v[-2] - v[-1]) / dS**2)
        v_nb = v[:, nb_idx_j]
        p = jnp.moveaxis((v[:, None, :] - v_nb) / ch.z[None, :, None], 1, 0)
        lam = inten3(d_slice)
        jump = jnp.sum(jnp.where(adm_j[:, None, :],
                                 z3 * lam * (d_slice - p), 0.0), axis=0)
        return adv[:, None] * d1 + diff_c[:, None] * d2 + run_src + jump

    def rk4(v, j):
        # slice at the step's forward-time start: matches the point in the
        # backward march at which the reference policy's quotes were taken
        t_cur = T - j * dt
        k_sl = jnp.argmin(jnp.abs(t_sl - t_cur))
        d_slice = pol_d[k_sl].astype(jnp.float64)
        k1 = rhs(v, d_slice); k2 = rhs(v + 0.5 * dt * k1, d_slice)
        k3 = rhs(v + 0.5 * dt * k2, d_slice); k4 = rhs(v + dt * k3, d_slice)
        return v + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

    v0, _ = jax.lax.scan(rk4, jnp.zeros((n_S, M)), jnp.arange(nt))
    return EvalResult(V0=v0, n_pts=n_pts, S_grid=S)
