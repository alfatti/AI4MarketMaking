"""Finite-difference benchmark for the Model-B HJ equation (Section 6 setup).

Solves, on the inventory grid q in Prod_i {-Q_i, ..., Q_i} (step = base size),
    d(theta)/dt = (gamma/2) q'Sigma q - mu'q
                  - sum_i sum_k p^k z_k [ H^{i,b}(p_b^{i,k}) 1_b + H^{i,a}(p_a^{i,k}) 1_a ],
backward from theta(T, q) = 0 by explicit Euler, where
    p_b^{i,k} = (theta(q) - theta(q + z_k e_i)) / z_k,   z_k = k-th discrete size,
and the indicators switch a term off when the trade would breach the risk limit
(equivalently: when the shifted grid point does not exist).

This is the "exact" (up to time discretization) benchmark the closed-form proxy
is judged against: it uses the true Hamiltonians, the same discrete size mixture
nu, drift, correlation, and risk limits. Sizes must be integer multiples of the
per-asset base grid step.

Quotes from theta: delta*(q, z_k) = ham.delta_star(p) with p as above (Thm 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hamiltonian import HamiltonianSide
from .config import SizeDistribution


def _shift(theta: np.ndarray, axis: int, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """theta(q + cells * step * e_axis) and a validity mask (False past boundary)."""
    nd = theta.ndim
    out = np.zeros_like(theta)
    valid = np.zeros(theta.shape, dtype=bool)
    src = [slice(None)] * nd
    dst = [slice(None)] * nd
    if cells > 0:
        dst[axis] = slice(0, -cells)
        src[axis] = slice(cells, None)
    elif cells < 0:
        dst[axis] = slice(-cells, None)
        src[axis] = slice(0, cells)
    out[tuple(dst)] = theta[tuple(src)]
    valid[tuple(dst)] = True
    return out, valid


@dataclass
class FDHJBSolution:
    """theta at t=0 plus snapshots; grids in inventory units."""

    theta0: np.ndarray                 # theta(0, q), shape (n_1, ..., n_d)
    q_grids: list[np.ndarray]          # per-asset inventory grids
    t_snaps: np.ndarray                # snapshot times
    theta_snaps: np.ndarray            # (n_snaps, n_1, ..., n_d)
    hams_b: list[HamiltonianSide]
    hams_a: list[HamiltonianSide]
    size_dists: list[SizeDistribution]
    base_steps: list[float]

    def quotes(self, theta: np.ndarray, asset: int, size: float
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(delta_b, delta_a, valid_b, valid_a) arrays on the grid for one size."""
        cells = int(round(size / self.base_steps[asset]))
        th_up, ok_up = _shift(theta, asset, cells)
        th_dn, ok_dn = _shift(theta, asset, -cells)
        with np.errstate(invalid="ignore"):
            p_b = (theta - th_up) / size
            p_a = (theta - th_dn) / size
        d_b = np.where(ok_up, self.hams_b[asset].delta_star(np.where(ok_up, p_b, 0.0)), np.nan)
        d_a = np.where(ok_dn, self.hams_a[asset].delta_star(np.where(ok_dn, p_a, 0.0)), np.nan)
        return d_b, d_a, ok_up, ok_dn


def solve_fd_hjb(
    hams_b: list[HamiltonianSide],
    hams_a: list[HamiltonianSide],
    Sigma: np.ndarray,
    gamma: float,
    T: float,
    size_dists: list[SizeDistribution],
    Q_limits: list[float],
    base_steps: list[float],
    mu: np.ndarray | None = None,
    dt: float = 1e-3,
    n_snaps: int = 100,
) -> FDHJBSolution:
    d = len(hams_b)
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    mu = np.zeros(d) if mu is None else np.asarray(mu, dtype=float)

    # inventory grids: -Q .. Q in base steps (sizes must be integer multiples)
    q_grids, cells_per_size = [], []
    for i in range(d):
        n_half = int(round(Q_limits[i] / base_steps[i]))
        if not np.isclose(n_half * base_steps[i], Q_limits[i]):
            raise ValueError("Q_limit must be an integer multiple of base_step.")
        q_grids.append(np.arange(-n_half, n_half + 1) * base_steps[i])
        cells = [int(round(s / base_steps[i])) for s in size_dists[i].sizes]
        if not all(np.isclose(c * base_steps[i], s)
                   for c, s in zip(cells, size_dists[i].sizes)):
            raise ValueError("All sizes must be integer multiples of base_step.")
        cells_per_size.append(cells)

    mesh = np.meshgrid(*q_grids, indexing="ij")
    # source term: (gamma/2) q'Sigma q - mu'q, precomputed on the grid
    quad = np.zeros(mesh[0].shape)
    for i in range(d):
        for j in range(d):
            quad += 0.5 * gamma * Sigma[i, j] * mesh[i] * mesh[j]
        quad -= mu[i] * mesh[i]

    # enumerate (asset, side, size) combos once; H evals stacked per step
    combos = []
    for i in range(d):
        for k, (zk, pk) in enumerate(zip(size_dists[i].sizes, size_dists[i].probs)):
            combos.append((i, +cells_per_size[i][k], zk, pk, hams_b[i]))   # bid: q -> q+z
            combos.append((i, -cells_per_size[i][k], zk, pk, hams_a[i]))   # ask: q -> q-z

    n_steps = int(round(T / dt))
    snap_every = max(1, n_steps // n_snaps)
    theta = np.zeros(mesh[0].shape)
    t_snaps, theta_snaps = [], []

    def theta_dot(th: np.ndarray) -> np.ndarray:
        out = quad.copy()
        ps, masks, metas = [], [], []
        for (i, cells, zk, pk, ham) in combos:
            th_s, ok = _shift(th, i, cells)
            with np.errstate(invalid="ignore"):
                p = np.where(ok, (th - th_s) / zk, 0.0)
            ps.append(p.ravel()); masks.append(ok); metas.append((zk, pk, ham))
        flat = np.concatenate(ps)
        # all combos share the same intensity family per side; evaluate blockwise
        # (H is cheap and vectorized; a single call per combo on the raveled block)
        off = 0
        n_cell = mesh[0].size
        for (ok, (zk, pk, ham)) in zip(masks, metas):
            Hv = np.asarray(ham.H(flat[off:off + n_cell])).reshape(mesh[0].shape)
            out -= pk * zk * np.where(ok, Hv, 0.0)
            off += n_cell
        return out

    for step in range(n_steps):
        if step % snap_every == 0:
            t_snaps.append(T - step * dt)
            theta_snaps.append(theta.copy())
        theta = theta - dt * theta_dot(theta)     # backward: theta(t) = theta(t+dt) - dt*theta_dot

    t_snaps.append(0.0)
    theta_snaps.append(theta.copy())

    return FDHJBSolution(
        theta0=theta, q_grids=q_grids,
        t_snaps=np.array(t_snaps), theta_snaps=np.array(theta_snaps),
        hams_b=hams_b, hams_a=hams_a, size_dists=size_dists, base_steps=base_steps,
    )
