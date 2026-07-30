"""Lockstep CRN PnL simulation for GPU (H200): one lax.scan over proposal events,
all paths vectorized, ALL policies carried in the same step so common random
numbers hold by construction (same gap/category/atom/acceptance-uniform/Brownian
arrays feed every policy). Counter-based PRNG (fold_in per step) — no O(paths x
events) random prealloc. Generic in d via strided flat indices into quote tables.

Scaling: cost ~ n_paths x max_events gathers; 1e6 paths x 1200 events is ~1e9
gather-ops — seconds-to-minutes on an H200, vs hours for the host-loop version.
Accounting matches mc.pnl_simulation: PnL = sum_fills z*delta + sum_seg q'dS;
penalty = (gamma/2) int q'Sigma q dt; objective = PnL - penalty."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .mc import lam_bar_table


def pnl_lockstep(spec, tables, t_tab, grids, n_paths, seed=0, max_events=None,
                 q0_idx=None):
    """tables: {name: (n_t, *shape, d, K, 2)} quote tables (NaN = side off).
    q0_idx: (d,) or (n_paths, d) start indices (default: grid centers).
    Returns {name: dict(pnl, spread, inv, penalty, fills)} numpy arrays."""
    d = spec.d
    K = spec.z_atoms.shape[-1]
    shape = tables[next(iter(tables))].shape[1:1 + d]
    n_t = tables[next(iter(tables))].shape[0]
    lb = np.asarray(lam_bar_table(spec))[:, 0, :]                # (d, 2)
    Rbar = float(lb.sum())
    if max_events is None:
        max_events = int(Rbar * spec.T + 8.0 * np.sqrt(Rbar * spec.T)) + 16
    # category tables (flat over d*2, ordered (i, side))
    pcat = (lb / Rbar).ravel()
    cum_cat = jnp.asarray(np.cumsum(pcat))
    cat_i = jnp.asarray(np.repeat(np.arange(d), 2))
    cat_s = jnp.asarray(np.tile(np.arange(2), d))
    cum_atom = jnp.asarray(np.cumsum(spec.p_atoms[:, 0, :, :], axis=-1))  # (d,2,K)
    zt = jnp.asarray(spec.z_atoms[:, 0, :, :])                    # (d,2,K)
    steps = np.array([g[1] - g[0] for g in grids])
    cells_t = jnp.asarray(np.round(spec.z_atoms[:, 0, :, :] / steps[:, None, None]
                                   ).astype(np.int32))            # (d,2,K)
    sgn_side = jnp.asarray(np.array([1, -1], np.int32))
    nvec = jnp.asarray(np.array(shape, np.int32))
    strides = jnp.asarray(np.array([int(np.prod(shape[i+1:])) for i in range(d)], np.int32))
    NS = int(np.prod(shape))
    kind_t = jnp.asarray(spec.kind[:, 0, :])                      # (d,2)
    ip_t = jnp.asarray(spec.ip[:, 0, :, :])                       # (d,2,3)
    lb_j = jnp.asarray(lb)
    L = jnp.asarray(np.linalg.cholesky(spec.Sigma))
    mu = jnp.asarray(spec.mu); Sig = jnp.asarray(spec.Sigma)
    gvals = [jnp.asarray(g) for g in grids]
    t_tab_j = jnp.asarray(t_tab)
    names = list(tables)
    tabs_flat = {nm: jnp.asarray(tables[nm]).reshape(n_t * NS * d * K * 2)
                 for nm in names}
    if q0_idx is None:
        q0_idx = np.array([len(g) // 2 for g in grids], np.int32)
    q0_idx = np.broadcast_to(np.asarray(q0_idx, np.int32), (n_paths, d)).copy()

    def lam_val(delta, i, s):
        kd = kind_t[i, s]                                   # (n_paths,)
        ip = ip_t[i, s]                                     # (n_paths, 3)
        lam_exp = ip[:, 0] * jnp.exp(-ip[:, 1] * delta)
        lam_log = ip[:, 0] / (1.0 + jnp.exp(ip[:, 1] + ip[:, 2] * delta))
        return jnp.where(kd == 0, lam_exp, lam_log)

    base = jax.random.PRNGKey(seed)

    def step(carry, j):
        t, states, acc = carry
        k = jax.random.fold_in(base, j)
        k1, k2, k3, k4, k5 = jax.random.split(k, 5)
        gap = jax.random.exponential(k1, (n_paths,)) / Rbar
        u_cat = jax.random.uniform(k2, (n_paths,))
        u_atom = jax.random.uniform(k3, (n_paths,))
        u_acc = jax.random.uniform(k4, (n_paths,))
        xi = jax.random.normal(k5, (n_paths, d))
        t_next = t + gap
        seg = jnp.clip(jnp.minimum(t_next, spec.T) - jnp.minimum(t, spec.T), 0.0)
        dS = mu[None, :] * seg[:, None] + (xi * jnp.sqrt(seg)[:, None]) @ L.T
        live = t_next <= spec.T
        cat = (u_cat[:, None] > cum_cat[None, :]).sum(1)
        i_a = cat_i[cat]; s_a = cat_s[cat]
        cum_k = cum_atom[i_a, s_a]                                # (n_paths, K)
        k_a = (u_atom[:, None] > cum_k).sum(1).astype(jnp.int32)
        z = zt[i_a, s_a, k_a]
        cells = cells_t[i_a, s_a, k_a] * sgn_side[s_a]
        ti = jnp.clip((t_next[:, None] >= t_tab_j[None, :]).sum(1) - 1, 0, n_t - 1)
        new_states, new_acc = {}, {}
        for nm in names:
            idx = states[nm]                                      # (n_paths, d) int32
            qv = jnp.stack([gvals[i][idx[:, i]] for i in range(d)], axis=1)
            inv = (qv * dS).sum(1)
            pen = 0.5 * spec.gamma * seg * jnp.einsum("bi,ij,bj->b", qv, Sig, qv)
            sflat = (idx * strides[None, :]).sum(1)
            flat = (((ti * NS + sflat) * d + i_a) * K + k_a) * 2 + s_a
            dlt = tabs_flat[nm][flat]
            newi = idx[jnp.arange(n_paths), i_a] + cells
            ok = live & jnp.isfinite(dlt) & (newi >= 0) & (newi < nvec[i_a])
            acc_p = ok & (u_acc < lam_val(dlt, i_a, s_a) / lb_j[i_a, s_a])
            oh = jax.nn.one_hot(i_a, d, dtype=jnp.int32)
            idx = idx + oh * (cells * acc_p.astype(jnp.int32))[:, None]
            a = acc[nm]
            new_acc[nm] = dict(
                spread=a["spread"] + jnp.where(acc_p, z * dlt, 0.0),
                inv=a["inv"] + inv, penalty=a["penalty"] + pen,
                fills=a["fills"] + acc_p.astype(jnp.float64),
                segsum=a["segsum"] + seg)
            new_states[nm] = idx
        return (t_next, new_states, new_acc), None

    zero = lambda: dict(spread=jnp.zeros(n_paths), inv=jnp.zeros(n_paths),
                        penalty=jnp.zeros(n_paths), fills=jnp.zeros(n_paths),
                        segsum=jnp.zeros(n_paths))
    carry0 = (jnp.zeros(n_paths),
              {nm: jnp.asarray(q0_idx) for nm in names},
              {nm: zero() for nm in names})
    (tf, _, acc), _ = jax.lax.scan(jax.jit(step), carry0, jnp.arange(max_events))
    out = {}
    for nm in names:
        a = acc[nm]
        out[nm] = dict(pnl=np.asarray(a["spread"] + a["inv"]),
                       spread=np.asarray(a["spread"]), inv=np.asarray(a["inv"]),
                       penalty=np.asarray(a["penalty"]), fills=np.asarray(a["fills"]),
                       segsum=np.asarray(a["segsum"]))
    return out
