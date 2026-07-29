"""Monte-Carlo correction and policy-value machinery (paper Section 4 analogue).

Identity (policy evaluation / Dynkin): for ANY ansatz theta_ans with
theta_ans(T,.) = 0 and its greedy quote policy pi, the value of pi satisfies

    v_pi(t0, q0) = theta_ans(t0, q0) + E[ integral_t0^T r(s, Q_s) ds ]

where Q_s is the inventory under pi (true intensities, risk limits enforced) and
r is the TRUE-equation residual of theta_ans,

    r = dt(theta_ans) + mu'q - (gamma/2) q'Sigma q
        + sum_feasible p_k z_k H0((theta_ans(q) - theta_ans(q +- z e_i) + c)/z_k),

because the sup in H0 is attained at the greedy quote, so the linear
policy-evaluation operator applied to theta_ans coincides with the HJ operator.
The ansatz itself is the Dynkin control variate: only the (small) residual is
integrated by MC, so the estimator variance scales with the residual, not with
the PnL.  v_pi <= theta_true always; for theta_ans = theta_true, r = 0 and the
estimate is exact with zero variance.

This gives grid-free, pointwise, unbiased policy values at ANY dimension —
the validation instrument that survives when the lattice does not.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from . import hamiltonians as ham
from . import network
from . import residual as rm


def lam_bar_table(spec):
    """Upper bound Lambda(-delta_inf) per (i, n, side): exact thinning envelope."""
    out = np.zeros((spec.d, spec.n_tiers, 2))
    for i in range(spec.d):
        for n in range(spec.n_tiers):
            for s in range(2):
                A_or_l, b_or_a, c_or_b = spec.ip[i, n, s]
                if spec.kind[i, n, s] == 0:
                    out[i, n, s] = A_or_l * np.exp(b_or_a * spec.delta_inf)
                else:
                    out[i, n, s] = A_or_l / (1.0 + np.exp(b_or_a - c_or_b * spec.delta_inf))
    return out


def make_greedy_quote_fn(spec, proxy, params, fs, theta_scale=1.0):
    """Jitted (t, q) -> delta (d, n_tiers, 2, K) for the greedy policy of
    theta_check + eta (params=None handled by caller passing zero-init params)."""
    sj = spec.to_jax()
    d, nt, K = spec.d, spec.n_tiers, spec.z_atoms.shape[-1]
    z = sj["z"]                                           # (d, n, 2, K)
    sgn = jnp.array([1.0, -1.0]).reshape(1, 1, 2, 1)
    eye = jnp.eye(d).reshape(d, 1, 1, 1, d)
    shift = sgn[..., None] * z[..., None] * eye           # (d, n, 2, K, d)

    def quote_all(t, q):
        At, Bt = proxy.A(t), proxy.B(t)
        qA = q @ At                                       # (d,)
        diffs_chk = (2.0 * sgn * z * qA[:, None, None, None]
                     + z ** 2 * jnp.diagonal(At)[:, None, None, None]
                     + sgn * z * Bt[:, None, None, None])  # (d, n, 2, K)
        eta0 = network.eta(params, t, q, fs, theta_scale)
        qs = q[None, None, None, None, :] + shift          # (d, n, 2, K, d)
        etaS = jax.vmap(lambda qq: network.eta(params, t, qq, fs, theta_scale)
                        )(qs.reshape(-1, d)).reshape(d, nt, 2, K)
        p = (diffs_chk + (eta0 - etaS) + sj["c"][..., None]) / z
        return ham.delta_star(p, sj["kind"][..., None], sj["ip"][..., None, :],
                              spec.delta_inf)

    return jax.jit(quote_all)


def zero_eta_params(spec, widths=(8,)):
    """Zero-init network => eta identically 0: proxy-greedy via the same code path."""
    fs = network.feature_spec(spec)
    return network.init_params(jax.random.PRNGKey(0), network.n_features(fs), widths), fs


def simulate_greedy_paths(spec, proxy, params, fs, t0, q0, n_paths, rng,
                          theta_scale=1.0):
    """Exact thinning under the greedy policy of theta_check + theta_scale*eta_raw.
    NOTE: params must already produce eta in physical units via the caller's
    closure convention: here we take quotes from make_greedy_quote_fn on params
    whose mlp output IS eta_physical (wrap scaling into params upstream by
    passing scaled output weights, or use theta_scale=1 with physical params).
    Returns per path: list of segments (t_start, t_end, q_vec) and fill log."""
    qfn = make_greedy_quote_fn(spec, proxy, params, fs, theta_scale)
    lb = lam_bar_table(spec)
    flat_lb = lb.ravel()
    Rbar = flat_lb.sum()
    cats = np.stack(np.unravel_index(np.arange(flat_lb.size), lb.shape), axis=1)
    pcat = flat_lb / Rbar
    K = spec.z_atoms.shape[-1]
    sj_ip = spec.ip; sj_kind = spec.kind
    paths = []
    for _ in range(n_paths):
        t, q = t0, np.asarray(q0, float).copy()
        segs, t_seg = [], t0
        while True:
            t_next = t + rng.exponential(1.0 / Rbar)
            if t_next >= spec.T:
                segs.append((t_seg, spec.T, q.copy()))
                break
            i, n, s = cats[rng.choice(len(pcat), p=pcat)]
            k = rng.choice(K, p=spec.p_atoms[i, n, s])
            z = spec.z_atoms[i, n, s, k]
            sgn = 1.0 if s == 0 else -1.0
            feasible = abs(q[i] + sgn * z) <= spec.Q[i] + 1e-9
            if feasible:
                dlt = float(np.asarray(qfn(t_next, jnp.asarray(q)))[i, n, s, k])
                lam_val = _lam_np(dlt, sj_kind[i, n, s], sj_ip[i, n, s])
                if rng.uniform() < lam_val / lb[i, n, s]:
                    segs.append((t_seg, t_next, q.copy()))
                    q[i] += sgn * z
                    t_seg = t_next
            t = t_next
        paths.append(segs)
    return paths


def _lam_np(delta, kind, ip):
    if kind == 0:
        return ip[0] * np.exp(-ip[1] * delta)
    return ip[0] / (1.0 + np.exp(ip[1] + ip[2] * delta))


_GL3_X = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
_GL3_W = np.array([5.0, 8.0, 5.0]) / 9.0


def mc_policy_value(spec, proxy, params, fs, theta_scale, t0, q0, n_paths, rng,
                    theta_at=None):
    """v_pi(t0, q0) with the Dynkin control variate. params: physical-eta network
    params (zero-init => proxy policy). Returns (v, stderr, correction_mean)."""
    paths = simulate_greedy_paths(spec, proxy, params, fs, t0, q0, n_paths, rng,
                                  theta_scale=theta_scale)
    ts, qs, meta = [], [], []
    for ipath, segs in enumerate(paths):
        for (a, b, qv) in segs:
            if b - a < 1e-14:
                continue
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            for xg, wg in zip(_GL3_X, _GL3_W):
                ts.append(mid + half * xg)
                qs.append(qv)
                meta.append((ipath, wg * half))
    ts = np.asarray(ts); qs = np.asarray(qs)
    prep = rm.prepare_batch(spec, proxy, ts, qs)
    res_scale, _ = rm.scales(spec)
    r = np.asarray(rm.residual_fn(params, prep)) * res_scale
    corr = np.zeros(n_paths)
    for (ipath, w), rv in zip(meta, r):
        corr[ipath] += w * rv
    if theta_at is None:
        th0 = float(proxy.theta_fast(t0, jnp.asarray(q0)))
        th0 += float(network.eta(params, jnp.asarray(t0),
                                 jnp.asarray(q0, dtype=jnp.float64), fs, theta_scale))
    else:
        th0 = theta_at
    v = th0 + corr.mean()
    return v, corr.std(ddof=1) / np.sqrt(n_paths), corr.mean(), th0


# ---------------------------- CRN-paired PnL simulation ----------------------------

def pnl_simulation(spec, delta_tables, t_tab, grids, n_paths, rng, n_events_cap=4000):
    """Paired PnL under common random numbers. delta_tables: {name: (n_t, *shape, d, K, 2)}
    (NaN = side off). Same proposal stream (times, category, atom, accept-u) and the
    same Brownian increments drive every policy; only delta differs => paired
    differences isolate policy quality from market noise.
    PnL = sum_fills z*delta + sum_segs q'(mu dt + L dW); penalty = (gamma/2) int q'Sigma q dt.
    Returns {name: dict(pnl, spread, inv, penalty, fills)} arrays of shape (n_paths,)."""
    d = spec.d
    lb = lam_bar_table(spec)
    flat_lb = lb.ravel(); Rbar = flat_lb.sum()
    cats = np.stack(np.unravel_index(np.arange(flat_lb.size), lb.shape), axis=1)
    pcat = flat_lb / Rbar
    K = spec.z_atoms.shape[-1]
    L = np.linalg.cholesky(spec.Sigma)
    names = list(delta_tables)
    out = {nm: {k: np.zeros(n_paths) for k in ("pnl", "spread", "inv", "penalty", "fills")}
           for nm in names}
    n_t = len(t_tab)
    centers = [np.searchsorted(g, 0.0) for g in grids]
    steps = [g[1] - g[0] for g in grids]

    for ip_ in range(n_paths):
        n_ev = rng.poisson(Rbar * spec.T)
        n_ev = min(n_ev, n_events_cap)
        times = np.sort(rng.uniform(0.0, spec.T, n_ev))
        cat_idx = rng.choice(len(pcat), size=n_ev, p=pcat)
        u_atom = rng.uniform(size=n_ev)
        u_acc = rng.uniform(size=n_ev)
        seg_t = np.diff(np.concatenate([[0.0], times, [spec.T]]))
        xi = rng.standard_normal((n_ev + 1, d))
        dS = (spec.mu[None, :] * seg_t[:, None]
              + (xi * np.sqrt(seg_t)[:, None]) @ L.T)

        for nm in names:
            tab = delta_tables[nm]
            idx = list(centers)
            spread = inv = pen = 0.0; fills = 0
            for j in range(n_ev + 1):
                qv = np.array([grids[i][idx[i]] for i in range(d)])
                inv += qv @ dS[j]
                pen += 0.5 * spec.gamma * (qv @ spec.Sigma @ qv) * seg_t[j]
                if j == n_ev:
                    break
                i, n, s = cats[cat_idx[j]]
                pk = spec.p_atoms[i, n, s]
                k = int(np.searchsorted(np.cumsum(pk), u_atom[j]))
                z = spec.z_atoms[i, n, s, k]
                ti = min(int(np.searchsorted(t_tab, times[j])), n_t - 1)
                dlt = tab[(ti, *idx, i, k, s)]
                if np.isnan(dlt):
                    continue
                lam_val = _lam_np(dlt, spec.kind[i, n, s], spec.ip[i, n, s])
                if u_acc[j] < lam_val / lb[i, n, s]:
                    cells = int(round(z / steps[i])) * (1 if s == 0 else -1)
                    new = idx[i] + cells
                    if 0 <= new < len(grids[i]):
                        spread += z * dlt
                        idx[i] = new
                        fills += 1
            out[nm]["spread"][ip_] = spread
            out[nm]["inv"][ip_] = inv
            out[nm]["penalty"][ip_] = pen
            out[nm]["pnl"][ip_] = spread + inv
            out[nm]["fills"][ip_] = fills
    return out
