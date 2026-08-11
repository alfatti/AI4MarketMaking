"""CRN simulator on (S, n): value identity vs the exact lattice, the
eta-parabola, q^S hedge-path reporting (the A.1 deliverable), and the
Dynkin-term accumulator for PINN closure.

- Policy: quote tables from stored lattice slices (nearest-t lookup), or
  greedy-from-PINN when `pinn` is supplied.
- Reward: sum of fills z * delta plus [carry(S) Dpi - pen(S) Dpi^2] dt at
  the pre-trade state (docs/derivations.md section 3).
- eta enters only the P&L legs for the parabola (fixed policy, CRN):
      M(eta) = int eta sigma(t,S) S Dpi dW,   Var(eta)/Var(1) = eta^2.
- q^S = -(1 - eta) Dpi is materialized: terminal notional and turnover
  (sum |d q^S|) are reported per path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..core.channels import Channels, SpotMarket, _bc, carry_coef, pen_coef
from ..instruments.spot_instruments import DeltaGrid, SpotBook
from ..solvers.lattice import LatticeSolution


@dataclass
class SimResult:
    reward: jnp.ndarray
    M: jnp.ndarray                  # (n_eta, n_paths)
    corr: Optional[jnp.ndarray]
    fills: jnp.ndarray
    qS_final: jnp.ndarray           # hedge position at T (eta = eta_hedge)
    qS_turnover: jnp.ndarray
    occupancy: Optional[jnp.ndarray] = None   # (n_S, M) visit counts


def simulate(book: SpotBook, dgrid: DeltaGrid, market: SpotMarket,
             gamma: float, box, T: float, lat: LatticeSolution,
             eta: float = 1.0, eta_list: Sequence[float] = (1.0,),
             eta_hedge: float = 1.0, n_steps: int = 1200,
             n_paths: int = 10000, seed: int = 0, delta_inf: float = -5.0,
             res_fn=None, res_stride: int = 4,
             count_occupancy: bool = False):
    """res_fn(t, S, n) -> HJB residual (batched by vmap here) enables the
    Dynkin accumulator for PINN closure; None disables it."""
    assert lat.v_hist is not None, "lattice solved with store_stride"
    ch = lat.channels
    N = book.n_options
    n_ch = 2 * N
    dt = T / n_steps
    psi = np.concatenate([np.ones(N), -np.ones(N)])
    inst = np.concatenate([np.arange(N), np.arange(N)])
    shifts = jnp.asarray(-psi[:, None] * np.eye(N)[inst])
    nbar = jnp.asarray(box.nbar, jnp.float64)
    S_grid = lat.S_grid
    nS = int(S_grid.shape[0])
    dS = float(S_grid[1] - S_grid[0])
    t_hist = lat.t_hist
    n_slices = int(t_hist.shape[0])
    flat = lat.flat_index
    dims = jnp.asarray(2 * (np.abs(lat.n_pts).max(axis=0)) + 1)
    # flat index of an integer inventory vector (row-major as enumerated)
    strides = np.cumprod(np.concatenate([[1], np.asarray(dims)[::-1][:-1]]))[::-1]
    strides_j = jnp.asarray(strides.copy())
    nb_off = jnp.asarray(np.abs(lat.n_pts).max(axis=0))
    etas = jnp.asarray(eta_list)
    use_corr = res_fn is not None
    res_b = jax.vmap(res_fn) if use_corr else None

    def flat_of(n):
        return jnp.sum(((n + nb_off).astype(jnp.int64)) * strides_j, axis=-1
                       ).astype(jnp.int32)

    def step(carry, k):
        S, n, reward, M, corr, fills, qS_prev, turn, occ, key = carry
        t = k * dt
        key, k1, k2 = jax.random.split(key, 3)
        k_slice = jnp.argmin(jnp.abs(t_hist - t))
        v = lat.v_hist[k_slice]                        # (nS, M)
        f = jnp.clip((S - S_grid[0]) / dS, 0.0, nS - 1.001)
        iS = jnp.floor(f).astype(jnp.int32); a = f - iS
        idx = flat_of(n)                               # (P,)
        n_tgt = n[:, None, :] + shifts[None]
        idx_t = flat_of(n_tgt.reshape(-1, N)).reshape(-1, n_ch)
        adm = jnp.all(jnp.abs(n_tgt) <= nbar[None, None, :] + 1e-9, axis=-1)
        if count_occupancy:
            occ = occ.at[iS, idx].add(1.0)
        v0 = (1 - a) * v[iS, idx] + a * v[iS + 1, idx]
        vt = (1 - a)[:, None] * v[iS[:, None], idx_t] \
            + a[:, None] * v[iS[:, None] + 1, idx_t]
        p = (v0[:, None] - vt) / ch.z[None, :]
        delta = _bc(ch.intensity, 1).argmax_delta(p.T, delta_inf).T
        lam = _bc(ch.intensity, 1)(delta.T).T
        u = jax.random.uniform(k1, (S.shape[0], n_ch))
        fill = (u < lam * dt) & adm
        dl = jax.vmap(lambda s: dgrid(0.0, s))(S)      # (P, N)
        Dpi = jnp.sum(n * book.z[None, :] * dl, axis=1)
        sig = market.sigma(t, S)
        reward = reward + (carry_coef(market, eta, t, S) * Dpi
                           - pen_coef(market, gamma, eta, t, S) * Dpi**2) * dt
        reward = reward + jnp.sum(jnp.where(fill, ch.z[None, :] * delta, 0.0),
                                  axis=1)
        if use_corr:
            corr = jax.lax.cond((k % res_stride) == 0,
                                lambda c: c + res_b(jnp.full_like(S, t), S,
                                                    n) * (dt * res_stride),
                                lambda c: c, corr)
        zN = jax.random.normal(k2, (S.shape[0],))
        dW = jnp.sqrt(dt) * zN
        M = M + (etas[:, None] * (sig * S * Dpi)[None, :]) * dW[None, :]
        qS = -(1.0 - eta_hedge) * Dpi
        turn = turn + jnp.abs(qS - qS_prev)
        n = n + jnp.sum(jnp.where(fill[:, :, None], shifts[None], 0.0), axis=1)
        S = S * jnp.exp((market.mu - 0.5 * sig * sig) * dt + sig * dW)
        fills = fills + jnp.sum(fill, axis=1).astype(jnp.int32)
        return (S, n, reward, M, corr, fills, qS, turn, occ, key), None

    P = n_paths
    occ0 = jnp.zeros((nS, lat.n_pts.shape[0]))
    init = (jnp.full((P,), market.S0), jnp.zeros((P, N)), jnp.zeros(P),
            jnp.zeros((etas.shape[0], P)), jnp.zeros(P),
            jnp.zeros(P, dtype=jnp.int32), jnp.zeros(P), jnp.zeros(P),
            occ0, jax.random.PRNGKey(seed))
    (S, n, reward, M, corr, fills, qS, turn, occ, _), _ = jax.lax.scan(
        step, init, jnp.arange(n_steps))
    return SimResult(reward=reward, M=M, corr=corr if use_corr else None,
                     fills=fills, qS_final=qS, qS_turnover=turn,
                     occupancy=occ if count_occupancy else None)
