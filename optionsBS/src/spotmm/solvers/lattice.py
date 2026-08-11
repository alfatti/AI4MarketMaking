"""Exact-in-inventory lattice on (t, S-grid, n): the full unsimplified
problem's reference solver (the structural gift of one factor).

    dv/dtau = mu S dv/dS + 0.5 sigma(S)^2 S^2 d2v/dS2
              + carry(S) Dpi(S,n) - pen(S) Dpi(S,n)^2
              + sum_ch z 1{n' in box} H([v(n) - v(n')]/z),
    Dpi(S, n) = sum_i n_i z_i Delta_i(S)   (live deltas from a DeltaGrid),

RK4 under lax.scan, backward from v(T) = 0. S: central differences with
Neumann ghost second derivatives at the band edges. Coefficients are
time-independent (Greeks frozen in t over the short horizon, live in S —
docs/derivations.md section 3), so one coefficient precompute serves the
whole march.

`periodic=True` wraps the inventory lattice and disables admissibility
masking; it exists solely for the eta = 0 analytic anchor, whose truth is
constant in n so that wraparound gathers are exact (section 4 of the note).
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


@dataclass
class LatticeSolution:
    S_grid: jnp.ndarray
    n_pts: np.ndarray
    flat_index: dict
    v0: jnp.ndarray                   # (n_S, M) at t = 0
    v_hist: Optional[jnp.ndarray]     # (n_stored, n_S, M), forward-time
    t_hist: Optional[jnp.ndarray]
    channels: Channels
    nb_idx: jnp.ndarray
    adm_tgt: jnp.ndarray
    Dpi: jnp.ndarray                  # (n_S, M) live portfolio delta
    dt: float

    def idx_of(self, n) -> int:
        return self.flat_index[tuple(int(x) for x in n)]

    def quotes(self, v: Optional[jnp.ndarray] = None, delta_inf=-5.0):
        v = self.v0 if v is None else v
        ch = self.channels
        v_nb = v[:, self.nb_idx]
        p = jnp.moveaxis((v[:, None, :] - v_nb) / ch.z[None, :, None], 1, 0)
        d = _bc(ch.intensity, 2).argmax_delta(p, delta_inf)
        return jnp.where(self.adm_tgt[:, None, :], d, jnp.nan)


def _enumerate(nbar: np.ndarray):
    axes = [np.arange(-b, b + 1) for b in nbar]
    grids = np.meshgrid(*axes, indexing="ij")
    n_pts = np.stack([g.ravel() for g in grids], axis=-1)
    return n_pts, {tuple(int(x) for x in r): i for i, r in enumerate(n_pts)}


def analytic_eta0_value(book: SpotBook, T: float, t=0.0, delta_inf=-5.0):
    """eta = 0, no limits: v = (T - t) * sum_ch z_ch H_ch(0)."""
    ch = Channels.from_book(book)
    H0 = ch.intensity.H(jnp.zeros_like(ch.z), delta_inf)
    return (T - t) * float(jnp.sum(ch.z * H0))


def solve_lattice(book: SpotBook, dgrid: DeltaGrid, market: SpotMarket,
                  gamma: float, box: Box, T: float, eta: float = 1.0,
                  nt: int = 200, n_S: int = 41, S_band: float = 0.08,
                  delta_inf: float = -5.0, store_stride: Optional[int] = None,
                  periodic: bool = False) -> LatticeSolution:
    ch = Channels.from_book(book)
    N = book.n_options
    nbar = np.asarray(box.nbar, dtype=int) + (0 if periodic else 1)
    n_pts, flat_index = _enumerate(nbar)
    M = n_pts.shape[0]
    dims = 2 * nbar + 1

    psi = np.concatenate([np.ones(N), -np.ones(N)])
    inst = np.concatenate([np.arange(N), np.arange(N)])
    nb_idx = np.full((2 * N, M), -1, dtype=np.int32)
    adm_tgt = np.zeros((2 * N, M), dtype=bool)
    for c_i in range(2 * N):
        tgt = n_pts.copy()
        tgt[:, inst[c_i]] -= int(psi[c_i])
        if periodic:
            b = nbar[inst[c_i]]
            tgt[:, inst[c_i]] = ((tgt[:, inst[c_i]] + b) % dims[inst[c_i]]) - b
            inside = np.ones(M, dtype=bool)
            adm = np.ones(M, dtype=bool)
        else:
            inside = np.all(np.abs(tgt) <= nbar[None, :], axis=1)
            adm = np.asarray(box.admissible(jnp.asarray(tgt),
                                            jnp.asarray(book.w)))
        for m in range(M):
            if inside[m]:
                nb_idx[c_i, m] = flat_index[tuple(int(x) for x in tgt[m])]
        adm_tgt[c_i] = inside & adm
    assert not np.any(adm_tgt & (nb_idx < 0))

    S = jnp.linspace(market.S0 * (1 - S_band), market.S0 * (1 + S_band), n_S)
    dS = float(S[1] - S[0])
    dt = T / nt
    sig = market.sigma(0.0, S)
    adv = market.mu * S
    diff_c = 0.5 * sig * sig * S * S
    penS = pen_coef(market, gamma, eta, 0.0, S)          # (n_S,)
    carS = carry_coef(market, eta, 0.0, S)
    dtab = jax.vmap(lambda s: dgrid(0.0, s))(S)           # (n_S, N)
    Dpi = (dtab * book.z[None, :]) @ jnp.asarray(n_pts.T, jnp.float64)
    run_src = carS[:, None] * Dpi - penS[:, None] * Dpi**2

    nb_idx_j = jnp.asarray(nb_idx)
    adm_j = jnp.asarray(adm_tgt)
    z3 = ch.z[:, None, None]
    inten3 = _bc(ch.intensity, 2)

    def rhs(v):
        d1 = jnp.zeros_like(v)
        d1 = d1.at[1:-1].set((v[2:] - v[:-2]) / (2 * dS))
        d2 = jnp.zeros_like(v)
        d2 = d2.at[1:-1].set((v[2:] - 2 * v[1:-1] + v[:-2]) / dS**2)
        d2 = d2.at[0].set(2 * (v[1] - v[0]) / dS**2)
        d2 = d2.at[-1].set(2 * (v[-2] - v[-1]) / dS**2)
        v_nb = v[:, nb_idx_j]
        p = jnp.moveaxis((v[:, None, :] - v_nb) / ch.z[None, :, None], 1, 0)
        Hs = inten3.H(p, delta_inf)
        jump = jnp.sum(jnp.where(adm_j[:, None, :], z3 * Hs, 0.0), axis=0)
        return adv[:, None] * d1 + diff_c[:, None] * d2 + run_src + jump

    def rk4(v, _):
        k1 = rhs(v); k2 = rhs(v + 0.5 * dt * k1)
        k3 = rhs(v + 0.5 * dt * k2); k4 = rhs(v + dt * k3)
        v_new = v + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return v_new, (v_new if store_stride else None)

    v_init = jnp.zeros((n_S, M))
    v0, hist = jax.lax.scan(rk4, v_init, None, length=nt)
    if store_stride:
        keep = jnp.arange(nt) % store_stride == (store_stride - 1)
        v_hist = jnp.concatenate([v_init[None],
                                  hist[keep]], axis=0)[::-1]
        taus = jnp.concatenate([jnp.array([0.0]),
                                (jnp.arange(nt, dtype=jnp.float64) + 1)[keep]
                                * dt])
        t_hist = (T - taus)[::-1]
    else:
        v_hist, t_hist = None, None
    return LatticeSolution(S_grid=S, n_pts=n_pts, flat_index=flat_index,
                           v0=v0, v_hist=v_hist, t_hist=t_hist, channels=ch,
                           nb_idx=nb_idx_j, adm_tgt=adm_j, Dpi=Dpi, dt=dt)
