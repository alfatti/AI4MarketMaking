"""Exact-in-q lattice solver: method of lines on (nu-grid) x (n-lattice).

State: v(t, nu, n), n in trade units on the representation box
[-nbar_i, nbar_i]^N (admissible set + one-jump neighborhood).  The HJB jump
terms are exact gathers n -> n - psi_ch e_i (no interpolation) — this solver
is the q-space ground truth against which the reduced 2D solver's Vpi
interpolation error is measured (stage 1), and the only exact reference in
stage 2 (box limits, no reduction).

    dv/dtau = RHS(v),  tau = T - t,  v(tau=0) = 0,
    RHS = a_P dv/dnu + 0.5 xi^2 nu d2v/dnu2 + carry(nu) Vpi - pen Vpi^2
          + sum_ch z_ch 1{n' admissible} H_ch([v(n) - v(n')]/z_ch),

integrated by RK4 under lax.scan.  nu: central differences + Neumann ghosts
(as in reduced2d).  Inadmissible representation states evolve harmlessly
(their values are never read through masked jump terms).

Implementation: the lattice is flattened to M points; per channel we
precompute the neighbor's flat index (sentinel -1 if outside the box) and the
post-trade admissibility mask.  Generic in N.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from ..core.risk import MarketParams, penalty_coef
from ..core.state import Box, Slab
from ..instruments.book import FrozenBook
from .reduced2d import Channels, _bc


@dataclass
class LatticeSolution:
    nu_grid: jnp.ndarray
    n_pts: np.ndarray                 # (M, N) int lattice points
    flat_index: dict                  # tuple(n) -> flat idx
    v0: jnp.ndarray                   # (n_nu, M) at t = 0
    v_hist: Optional[jnp.ndarray]     # (nt+1, n_nu, M), forward-time order
    t_hist: Optional[jnp.ndarray]
    channels: Channels
    nb_idx: jnp.ndarray               # (n_ch, M)
    adm_tgt: jnp.ndarray              # (n_ch, M)
    adm_self: jnp.ndarray             # (M,)
    Vpi: jnp.ndarray                  # (M,)
    dt: float

    def idx_of(self, n) -> int:
        return self.flat_index[tuple(int(x) for x in n)]

    def quotes(self, v: Optional[jnp.ndarray] = None, delta_inf=-5.0):
        """(n_ch, n_nu, M) optimal quotes; NaN where inadmissible."""
        v = self.v0 if v is None else v
        ch = self.channels
        v_nb = v[:, self.nb_idx]                       # (n_nu, n_ch, M)
        p = (v[:, None, :] - v_nb) / ch.z[None, :, None]
        d = _bc(ch.intensity, 2).argmax_delta(jnp.moveaxis(p, 1, 0), delta_inf)
        return jnp.where(self.adm_tgt[:, None, :], d, jnp.nan)


def _enumerate_lattice(nbar: np.ndarray):
    axes = [np.arange(-b, b + 1) for b in nbar]
    grids = np.meshgrid(*axes, indexing="ij")
    n_pts = np.stack([g.ravel() for g in grids], axis=-1)  # (M, N)
    flat_index = {tuple(int(x) for x in row): i for i, row in enumerate(n_pts)}
    return n_pts, flat_index


def solve_lattice(book: FrozenBook, market: MarketParams, gamma: float,
                  adm: Union[Slab, Box], T: float, c: float = 0.0,
                  nt: int = 600, n_nu: int = 31,
                  nu_lo: float = 0.0144, nu_hi: float = 0.0324,
                  delta_inf: float = -5.0,
                  store_history: bool = False) -> LatticeSolution:
    ch = Channels.from_book(book)
    N = book.n_options
    w_np = np.asarray(book.w)
    pen = penalty_coef(gamma, market.xi, market.rho, c)

    nbar = adm.representation_nbar(w_np)
    n_pts, flat_index = _enumerate_lattice(nbar)
    M = n_pts.shape[0]

    # neighbor indices and masks per channel (ask first, then bid, as Channels)
    psi = np.concatenate([np.ones(N), -np.ones(N)])
    inst = np.concatenate([np.arange(N), np.arange(N)])
    nb_idx = np.full((2 * N, M), -1, dtype=np.int32)
    adm_tgt = np.zeros((2 * N, M), dtype=bool)
    w_j = jnp.asarray(w_np)
    for c_i in range(2 * N):
        tgt = n_pts.copy()
        tgt[:, inst[c_i]] -= int(psi[c_i])
        inside = np.all(np.abs(tgt) <= nbar[None, :], axis=1)
        adm_t = np.asarray(adm.admissible(jnp.asarray(tgt), w_j))
        for m in range(M):
            if inside[m]:
                nb_idx[c_i, m] = flat_index[tuple(int(x) for x in tgt[m])]
        adm_tgt[c_i] = inside & adm_t
    adm_self = np.asarray(adm.admissible(jnp.asarray(n_pts), w_j))
    # every admissible state's admissible target must be representable:
    assert not np.any(adm_tgt & (nb_idx < 0)), "representation box too small"

    nu = jnp.linspace(nu_lo, nu_hi, n_nu)
    dnu = float(nu[1] - nu[0])
    dt = T / nt
    a_P = market.a_P(nu)
    carry = market.carry_coef(nu)
    diff_c = 0.5 * market.xi**2 * nu
    Vpi = jnp.asarray(n_pts @ w_np)                       # (M,)
    run_src = carry[:, None] * Vpi[None, :] - pen * Vpi[None, :] ** 2

    nb_idx_j = jnp.asarray(nb_idx)
    adm_tgt_j = jnp.asarray(adm_tgt)
    z3 = ch.z[:, None, None]
    inten3 = _bc(ch.intensity, 2)

    def rhs(v):
        d1 = jnp.zeros_like(v)
        d1 = d1.at[1:-1].set((v[2:] - v[:-2]) / (2 * dnu))
        d2 = jnp.zeros_like(v)
        d2 = d2.at[1:-1].set((v[2:] - 2 * v[1:-1] + v[:-2]) / dnu**2)
        d2 = d2.at[0].set(2 * (v[1] - v[0]) / dnu**2)
        d2 = d2.at[-1].set(2 * (v[-2] - v[-1]) / dnu**2)
        v_nb = v[:, nb_idx_j]                             # (n_nu, n_ch, M)
        p = jnp.moveaxis((v[:, None, :] - v_nb) / ch.z[None, :, None], 1, 0)
        Hs = inten3.H(p, delta_inf)                       # (n_ch, n_nu, M)
        jump = jnp.sum(jnp.where(adm_tgt_j[:, None, :], z3 * Hs, 0.0), axis=0)
        return a_P[:, None] * d1 + diff_c[:, None] * d2 + run_src + jump

    def rk4(v, _):
        k1 = rhs(v)
        k2 = rhs(v + 0.5 * dt * k1)
        k3 = rhs(v + 0.5 * dt * k2)
        k4 = rhs(v + dt * k3)
        v_new = v + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return v_new, (v_new if store_history else None)

    v_init = jnp.zeros((n_nu, M))
    v0, hist = jax.lax.scan(rk4, v_init, None, length=nt)

    if store_history:
        # hist is tau-ordered (tau = dt ... T); prepend tau=0 and flip to t-order
        v_hist = jnp.concatenate([v_init[None], hist], axis=0)[::-1]
        t_hist = jnp.linspace(0.0, T, nt + 1)
    else:
        v_hist, t_hist = None, None

    return LatticeSolution(nu_grid=nu, n_pts=n_pts, flat_index=flat_index,
                           v0=v0, v_hist=v_hist, t_hist=t_hist, channels=ch,
                           nb_idx=nb_idx_j, adm_tgt=adm_tgt_j,
                           adm_self=jnp.asarray(adm_self), Vpi=Vpi, dt=dt)


def with_trade_vega(book: FrozenBook, w_target) -> FrozenBook:
    """Override trade sizes so per-trade vegas w are exactly w_target.

    Used to build commensurable test books: with w on a common grid the
    reduced problem's Vpi jumps land exactly on atoms and the reduced2d code
    run on the atom grid is *exact*, giving a reference that separates the 2D
    solver's interpolation error from the lattice's box-truncation error.
    """
    w_t = jnp.asarray(w_target, dtype=jnp.float64)
    z_new = w_t / book.vega
    return FrozenBook(K=book.K, tau=book.tau, O0=book.O0, vega=book.vega,
                      delta=book.delta, z=z_new, w=w_t, lam=book.lam,
                      alpha=book.alpha, k=book.k)


def subbook(book: FrozenBook, idx) -> FrozenBook:
    """Restrict a FrozenBook to instrument indices `idx`."""
    take = lambda a: a[jnp.asarray(idx)]
    return FrozenBook(K=take(book.K), tau=take(book.tau), O0=take(book.O0),
                      vega=take(book.vega), delta=take(book.delta),
                      z=take(book.z), w=take(book.w), lam=take(book.lam),
                      alpha=take(book.alpha), k=take(book.k))
