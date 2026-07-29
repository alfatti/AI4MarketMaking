"""Adversarial certificate machinery (H200-scale gap closing).

The train-vs-certificate gap is a *sampling* problem: switching surfaces
q_i = +-(Q_i - z_k) are (d-1)-dimensional manifolds x time, and fixed random
budgets undersample them as d grows. Two GPU-friendly attacks:

  1. rar_stream    — streamed residual evaluation over pools of 1e6-1e7 points
                     (chunked vmap; embarrassingly parallel; pool size is the
                     GPU knob).
  2. surface_ascent — projected gradient ascent of r(t,q)^2 restricted to a
                     band around each switching surface (free coordinates roam
                     the box), vmapped over thousands of starts, lax.fori_loop
                     inside jit: zero host round-trips. A two-sided kink
                     refinement evaluates both sides of each surface at the end
                     (the max often sits exactly on the discontinuity, where
                     the a.e. gradient is one-sided).

residual_tq is the self-contained differentiable pointwise residual (same
formula as residual.residual_point, but built from (t, q) directly rather than
from precomputed tensors) — verified against residual_fn at machine precision
in tests. Ascent needs d/dq, which prepare_batch's frozen tensors cannot give.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from . import hamiltonians as ham
from . import network
from . import residual as rm
from . import sampling


def residual_tq_factory(spec, proxy, fs):
    """Returns jitted scalar r(params, t, q) — scaled residual, differentiable in (t,q)."""
    sj = spec.to_jax()
    res_scale, theta_scale = rm.scales(spec)
    d, nt, K = spec.d, spec.n_tiers, spec.z_atoms.shape[-1]
    z = sj["z"]                                            # (d,n,2,K)
    pz = sj["pz"]; c = sj["c"]
    sgn = jnp.array([1.0, -1.0]).reshape(1, 1, 2, 1)
    eye = jnp.eye(d).reshape(d, 1, 1, 1, d)
    shift = (sgn[..., None] * z[..., None] * eye)          # (d,n,2,K,d)
    T = spec.T; Q = sj["Q"]

    def r(params, t, q):
        At, Bt = proxy.A(t), proxy.B(t)
        qA = q @ At
        diff_check = (2.0 * sgn * z * qA[:, None, None, None]
                      + z ** 2 * jnp.diagonal(At)[:, None, None, None]
                      + sgn * z * Bt[:, None, None, None])
        q_new_i = q[:, None, None, None] + sgn * z         # post-trade inventory, asset i
        mask = ((q_new_i <= Q[:, None, None, None] + 1e-12)
                & (q_new_i >= -Q[:, None, None, None] - 1e-12)).astype(q.dtype)
        f0 = network.features(t, q, fs)
        nn0 = network.mlp(params, f0)
        qs = (q[None, None, None, None, :] + shift).reshape(-1, d)
        nnS = jax.vmap(lambda qq: network.mlp(params, network.features(t, qq, fs))
                       )(qs).reshape(d, nt, 2, K)
        tfac = theta_scale * (1.0 - t / T)
        d_eta = tfac * (nn0 - nnS)
        tang = jnp.zeros_like(f0).at[0].set(1.0 / T)
        _, dnn_dt = jax.jvp(lambda x: network.mlp(params, x), (f0,), (tang,))
        dt_eta = -(theta_scale / T) * nn0 + tfac * dnn_dt
        p = (diff_check + d_eta + c[..., None]) / z
        H = ham.H0(p, sj["kind"][..., None], sj["ip"][..., None, :], spec.delta_inf)
        jump = (mask * pz * z * H).sum()
        static = q @ sj["mu"] - 0.5 * sj["gamma"] * q @ (sj["Sigma"] @ q)
        dt_check = proxy.dtheta_dt(t, q, At, Bt)
        return (dt_check + dt_eta + static + jump) / res_scale

    return jax.jit(r)


def rar_stream(spec, proxy, params, fs, n_pool, rng, chunk=200_000, keep=0):
    """Streamed pool residual sup (and optionally the worst-`keep` points).
    GPU: raise n_pool to 1e6-1e7 and chunk to ~1e6; memory is O(chunk)."""
    r_tq = residual_tq_factory(spec, proxy, fs)
    rb = jax.jit(jax.vmap(lambda t, q: r_tq(params, t, q)))
    best_v = -1.0; worst = []
    for start in range(0, n_pool, chunk):
        m = min(chunk, n_pool - start)
        n_u = int(0.5 * m); n_b = int(0.2 * m); n_s = m - n_u - n_b
        t1, q1 = sampling.uniform(spec, n_u, rng)
        t2, q2 = sampling.boundary_band(spec, n_b, rng)
        t3, q3 = sampling.surface_band(spec, n_s, rng)
        t = np.concatenate([t1, t2, t3]); q = np.vstack([q1, q2, q3])
        rv = np.abs(np.asarray(rb(jnp.asarray(t), jnp.asarray(q))))
        if keep:
            idx = np.argpartition(rv, -keep)[-keep:]
            worst.append((rv[idx], t[idx], q[idx]))
        best_v = max(best_v, float(rv.max()))
    if keep:
        rv = np.concatenate([w[0] for w in worst])
        tt = np.concatenate([w[1] for w in worst]); qq = np.vstack([w[2] for w in worst])
        idx = np.argpartition(rv, -keep)[-keep:]
        return best_v, tt[idx], qq[idx]
    return best_v, None, None


def surface_ascent(spec, proxy, params, fs, n_per_surface=64, n_steps=120,
                   lr=0.02, eps_band=0.15, rng=None, two_sided_eps=1e-4):
    """Projected gradient ascent of r^2 on bands around every switching surface.

    Returns (sup_found, t_pts, q_pts) — the refined adversarial set, suitable both
    for certification (sup) and for folding back into training (gap closing).
    GPU knobs: n_per_surface (thousands), n_steps; everything vmapped+jitted."""
    rng = np.random.default_rng(0) if rng is None else rng
    r_tq = residual_tq_factory(spec, proxy, fs)
    zb = np.asarray(spec.zbar())
    surfaces = []                                          # (axis, s_val, band_halfwidth)
    for i in range(spec.d):
        for zk in np.unique(spec.z_atoms[i]):
            for s in (1.0, -1.0):
                surfaces.append((i, s * (spec.Q[i] - zk), eps_band * zb[i]))
    starts_t, starts_q, ax_arr, sv_arr, bw_arr = [], [], [], [], []
    for (i, sval, bw) in surfaces:
        tt = rng.uniform(0.0, spec.T * 0.999, n_per_surface)
        qq = rng.uniform(-1.0, 1.0, (n_per_surface, spec.d)) * spec.Q[None, :]
        qq[:, i] = sval + rng.uniform(-bw, bw, n_per_surface)
        starts_t.append(tt); starts_q.append(qq)
        ax_arr += [i] * n_per_surface; sv_arr += [sval] * n_per_surface
        bw_arr += [bw] * n_per_surface
    t0 = jnp.asarray(np.concatenate(starts_t))
    q0 = jnp.asarray(np.vstack(starts_q))
    ax = jnp.asarray(ax_arr); sv = jnp.asarray(sv_arr); bw = jnp.asarray(bw_arr)
    Q = jnp.asarray(spec.Q); T = spec.T
    step_t = lr * T
    step_q = lr * Q                                        # coordinate-normalized steps

    g_fn = jax.grad(lambda t, q: r_tq(params, t, q) ** 2, argnums=(0, 1))

    def one(t, q, a, s, b):
        onehot = jax.nn.one_hot(a, spec.d)
        def body(_, carry):
            t_, q_ = carry
            gt, gq = g_fn(t_, q_)
            nrm = jnp.sqrt(gt ** 2 + (gq ** 2).sum()) + 1e-30
            t_ = jnp.clip(t_ + step_t * gt / nrm, 0.0, T * (1 - 1e-9))
            q_ = q_ + step_q * gq / nrm
            q_ = jnp.clip(q_, -Q, Q)
            qi = jnp.clip(q_ @ onehot, s - b, s + b)       # stay in the surface band
            q_ = q_ * (1 - onehot) + qi * onehot
            return (t_, q_)
        t_, q_ = jax.lax.fori_loop(0, n_steps, body, (t, q))
        # two-sided kink refinement: the sup often sits on the discontinuity
        cands = jnp.stack([q_,
                           q_ * (1 - onehot) + (s + two_sided_eps) * onehot,
                           q_ * (1 - onehot) + (s - two_sided_eps) * onehot])
        rv = jax.vmap(lambda qq: jnp.abs(r_tq(params, t_, qq)))(cands)
        j = jnp.argmax(rv)
        return rv[j], t_, cands[j]

    rv, tf, qf = jax.jit(jax.vmap(one))(t0, q0, ax, sv, bw)
    rv = np.asarray(rv)
    return float(rv.max()), np.asarray(tf), np.asarray(qf), rv


def certificate_v2(spec, proxy, params, fs, rng, n_random=50_000,
                   n_per_surface=64, ascent_steps=120):
    """Stratified random pool + surface ascent; returns dict of per-instrument sups.
    The certified bound uses the max over all instruments."""
    sup_pool, _, _ = rar_stream(spec, proxy, params, fs, n_random, rng)
    sup_asc, t_a, q_a, rv = surface_ascent(spec, proxy, params, fs,
                                           n_per_surface=n_per_surface,
                                           n_steps=ascent_steps, rng=rng)
    res_scale, _ = rm.scales(spec)
    sup = max(sup_pool, sup_asc)
    return dict(sup=sup, sup_pool=sup_pool, sup_ascent=sup_asc,
                sup_physical=sup * res_scale,
                theta_bound_t0=sup * res_scale * spec.T,
                ascent_points=(t_a, q_a, rv))
