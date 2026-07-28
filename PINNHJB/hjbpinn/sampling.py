"""Collocation sampling.

- uniform over [0, T] x prod_i [-Q_i, Q_i]
- boundary bands: within a few mean trade sizes of a risk limit (the censoring layer)
- model-native path sampling: exact thinning of the marked point process under the
  proxy-greedy policy (this is the policy-relevant measure; NOT rfqsim's auction
  mechanics, which belong to the calibration layer)
- residual-adaptive refinement (RAR): top-k residual points from a candidate pool
"""
from __future__ import annotations

import numpy as np

from . import hamiltonians as ham
import jax.numpy as jnp


def uniform(spec, n, rng):
    t = rng.uniform(0.0, spec.T, size=n)
    q = rng.uniform(-spec.Q, spec.Q, size=(n, spec.d))
    return t, q


def boundary_band(spec, n, rng, widths=3.0):
    t = rng.uniform(0.0, spec.T, size=n)
    q = rng.uniform(-spec.Q, spec.Q, size=(n, spec.d))
    zb = spec.zbar()
    for row in range(n):
        i = rng.integers(spec.d)
        side = rng.choice([-1.0, 1.0])
        depth = rng.uniform(0.0, widths * zb[i])
        q[row, i] = side * (spec.Q[i] - depth)
    return t, q


def surface_band(spec, n, rng, width=0.5):
    """Points concentrated around the switching surfaces q_i = +-(Q_i - z_k), where the
    residual of any smooth-in-q ansatz concentrates (indicator-jump kinks)."""
    surfaces = []
    for i in range(spec.d):
        for z in np.unique(spec.z_atoms[i]):
            s = float(spec.Q[i] - z)
            surfaces.append((i, s))
            if s > 1e-12:
                surfaces.append((i, -s))
    t = rng.uniform(0.0, spec.T, size=n)
    q = rng.uniform(-spec.Q, spec.Q, size=(n, spec.d))
    zb = spec.zbar()
    for row in range(n):
        i, s = surfaces[rng.integers(len(surfaces))]
        q[row, i] = np.clip(s + rng.normal(scale=width * zb[i]), -spec.Q[i], spec.Q[i])
    return t, q


def surface_straddle(spec, n_t=21, eps_fracs=(1e-4, 0.03, 0.12)):
    """Deterministic points straddling every switching surface at multiple distances,
    across a t-grid. The residual AT the straddle points reads the learned kink
    amplitude directly; Gaussian bands almost never sample the 1e-4*zbar scale, so
    without these the jump-amplitude error is invisible to training (but not to the
    certificate grid, which straddles by construction)."""
    surfaces = []
    for i in range(spec.d):
        for z in np.unique(spec.z_atoms[i]):
            s = float(spec.Q[i] - z)
            surfaces.append((i, s))
            if s > 1e-12:
                surfaces.append((i, -s))
    zb = spec.zbar()
    ts, qs = [], []
    rng_local = np.random.default_rng(0)
    for tt in np.linspace(0.0, spec.T * 0.999, n_t):
        for i, s in surfaces:
            for f in eps_fracs:
                for sgn in (+1.0, -1.0):
                    q = rng_local.uniform(-spec.Q, spec.Q)
                    q[i] = np.clip(s + sgn * f * zb[i], -spec.Q[i], spec.Q[i])
                    ts.append(tt); qs.append(q)
    return np.asarray(ts), np.asarray(qs)


def rar(residual_eval, spec, n_pool, n_keep, rng, surface_frac=0.3):
    """Residual-adaptive refinement: evaluate |r| on a mixed candidate pool
    (uniform + boundary + surface bands + surface straddles) and keep the worst."""
    n_s = int(surface_frac * n_pool); n_b = n_pool // 5
    t1, q1 = uniform(spec, n_pool - n_s - n_b, rng)
    t2, q2 = boundary_band(spec, n_b, rng)
    t3, q3 = surface_band(spec, n_s, rng)
    t4, q4 = surface_straddle(spec, n_t=15)
    t = np.concatenate([t1, t2, t3, t4]); q = np.vstack([q1, q2, q3, q4])
    r = np.abs(np.asarray(residual_eval(t, q)))
    idx = np.argsort(-r)[:n_keep]
    return t[idx], q[idx]


def _root_np(p, kind, ip, delta_inf):
    """Unconstrained argmax root, numpy scalar. Exponential closed form; logistic via
    Lambert-W in log space (asymptotic branch for large arguments)."""
    if kind == 0:
        r = p + 1.0 / ip[1]
    else:
        alpha, beta = ip[1], ip[2]
        L = -(alpha + beta * p + 1.0)
        if L > 30.0:                                   # W(e^L) ~ L - log L + log L / L
            w = L - np.log(L) + np.log(L) / L
            for _ in range(3):
                w = w - (w + np.log(w) - L) * w / (w + 1.0)
        else:
            from scipy.special import lambertw
            w = float(np.real(lambertw(np.exp(L))))
        r = p + (1.0 + w) / beta
    return max(r, -delta_inf)


def _lam_np(delta, kind, ip):
    if kind == 0:
        return ip[0] * np.exp(-ip[1] * delta)
    return ip[0] / (1.0 + np.exp(np.clip(ip[1] + ip[2] * delta, -60.0, 60.0)))


def proxy_policy_paths(spec, proxy, n_paths, rng, record_dt=None, n_grid=512):
    """Exact thinning under the proxy-greedy quotes, pure numpy in the event loop:
    A(t), B(t) precomputed on a t-grid (one jitted batch call), linearly interpolated."""
    import jax
    d, n, K = spec.d, spec.n_tiers, spec.z_atoms.shape[-1]
    tg = np.linspace(0.0, spec.T, n_grid)
    AB = jax.jit(jax.vmap(lambda tt: (proxy.A(tt), proxy.B(tt))))(jnp.asarray(tg))
    A_g = np.asarray(AB[0]); B_g = np.asarray(AB[1])           # (G,d,d), (G,d)

    def AB_at(t):
        x = np.clip(t / spec.T * (n_grid - 1), 0, n_grid - 1 - 1e-9)
        j = int(x); w = x - j
        return ((1 - w) * A_g[j] + w * A_g[j + 1], (1 - w) * B_g[j] + w * B_g[j + 1])

    lam_bar = np.zeros((d, n, 2))
    for i in range(d):
        for nn_ in range(n):
            for s in range(2):
                lam_bar[i, nn_, s] = _lam_np(-spec.delta_inf, spec.kind[i, nn_, s],
                                             spec.ip[i, nn_, s])
    R = lam_bar.sum()
    record_dt = record_dt if record_dt is not None else spec.T / 20.0
    flat_p = (lam_bar / R).ravel()
    ts, qs = [], []
    for _ in range(n_paths):
        t, q = 0.0, np.zeros(d)
        next_rec = 0.0
        while t < spec.T:
            t += rng.exponential(1.0 / R)
            if t >= spec.T:
                break
            while next_rec <= t:
                ts.append(next_rec); qs.append(q.copy()); next_rec += record_dt
            idx = rng.choice(lam_bar.size, p=flat_p)
            i, nn_, side = np.unravel_index(idx, (d, n, 2))
            k = rng.choice(K, p=spec.p_atoms[i, nn_, side])
            z = spec.z_atoms[i, nn_, side, k]
            sgn = 1.0 if side == 0 else -1.0
            if abs(q[i] + sgn * z) > spec.Q[i]:
                continue
            At, Bt = AB_at(t)
            diff = 2.0 * sgn * z * float(q @ At[:, i]) + z * z * At[i, i] + sgn * z * Bt[i]
            p_arg = (diff + spec.c[i, nn_, side]) / z
            delta = _root_np(p_arg, int(spec.kind[i, nn_, side]), spec.ip[i, nn_, side],
                             spec.delta_inf)
            if rng.uniform() < _lam_np(delta, int(spec.kind[i, nn_, side]),
                                       spec.ip[i, nn_, side]) / lam_bar[i, nn_, side]:
                q[i] += sgn * z
                ts.append(t); qs.append(q.copy())
        while next_rec < spec.T:
            ts.append(next_rec); qs.append(q.copy()); next_rec += record_dt
    return np.asarray(ts), np.asarray(qs)



