"""Validation utilities.

1) Method-of-lines reference (d = 1, atomic sizes commensurate with a lattice):
   when every atom z is an integer multiple of the grid spacing h and Q is a multiple
   of h, the lattice ODE system IS the exact equation restricted to reachable states —
   index shifts are exact, censoring is the lattice edge. Integrated with stiff
   solve_ivp at tight tolerances; this is ground truth up to ODE tolerance.

2) Certificate: the linearization of the HJ operator around any theta is a censored
   Markov jump generator (rates Lambda(delta*) >= 0, constant-shift invariant), so the
   flow is a sup-norm contraction and a uniform residual bound eps gives
   |theta_hat - theta|_inf(t) <= eps * (T - t). Estimated as a grid sup with points
   straddling the switching surfaces q_i = +-(Q_i - z_k) where the residual is
   discontinuous in q.

3) Fill-weighted quote metrics: quote errors weighted by the fill intensity of the
   reference policy — the policy-relevant norm (errors where trades never happen
   shouldn't dominate the score).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
import jax
import jax.numpy as jnp

from . import hamiltonians as ham
from . import residual as res_mod


# ------------------------- method of lines (d = 1) -------------------------

def mol_solve(spec, t_eval, h=None):
    """Solve the exact d=1 lattice ODE. Returns (q_grid, theta[t, j])."""
    assert spec.d == 1
    z_all = spec.z_atoms.ravel()
    h = float(z_all.min()) if h is None else h
    assert np.allclose(z_all / h, np.round(z_all / h)), "atoms must be lattice-commensurate"
    assert abs(spec.Q[0] / h - round(spec.Q[0] / h)) < 1e-12, "Q must be lattice-commensurate"
    M = int(round(spec.Q[0] / h))
    qg = h * np.arange(-M, M + 1)                     # (2M+1,)
    J = qg.size
    sj = spec.to_jax()
    mu, gam, sig2 = float(spec.mu[0]), spec.gamma, float(spec.Sigma[0, 0])
    n, K = spec.n_tiers, spec.z_atoms.shape[-1]

    stencil = []                                       # (shift, pz, z, kind, ip, c)
    for nn_ in range(n):
        for side, sgn in ((0, 1.0), (1, -1.0)):
            for k in range(K):
                z = float(spec.z_atoms[0, nn_, side, k])
                stencil.append((int(round(sgn * z / h)), float(spec.p_atoms[0, nn_, side, k]),
                                z, int(spec.kind[0, nn_, side]),
                                np.asarray(spec.ip[0, nn_, side]), float(spec.c[0, nn_, side])))

    def rhs(tau, th):
        out = mu * qg - 0.5 * gam * sig2 * qg ** 2
        for shift, pz, z, kd, ip, c in stencil:
            j = np.arange(J)
            j_new = j + shift
            ok = (j_new >= 0) & (j_new < J)
            p = np.zeros(J)
            p[ok] = (th[j[ok]] - th[j_new[ok]] + c) / z
            Hv = np.asarray(ham.H0(jnp.asarray(p), jnp.asarray(kd), jnp.asarray(ip),
                                   spec.delta_inf))
            out = out + np.where(ok, pz * z * Hv, 0.0)
        return out

    taus = np.sort(spec.T - np.asarray(t_eval))
    sol = solve_ivp(rhs, (0.0, spec.T), np.zeros(J), t_eval=taus,
                    method="LSODA", rtol=1e-11, atol=1e-12)
    assert sol.success
    theta = sol.y.T[::-1]                              # rows ordered as ascending t_eval
    return qg, theta


# ------------------------- certificate -------------------------

def certificate_grid(spec, n_t=25, n_q_uniform=2000, rng=None):
    """(t, q) grid for the residual sup: uniform + points straddling switching surfaces."""
    rng = np.random.default_rng(0) if rng is None else rng
    t = np.repeat(np.linspace(0.0, spec.T * 0.999, n_t), n_q_uniform // n_t)
    q = rng.uniform(-spec.Q, spec.Q, size=(t.size, spec.d))
    extras_q, extras_t = [], []
    eps_vals = np.array([-1e-4, 1e-4])
    z_unique = np.unique(spec.z_atoms)
    for tt in np.linspace(0.0, spec.T * 0.999, n_t):
        for i in range(spec.d):
            for z in z_unique:
                for s in (+1.0, -1.0):
                    for e in eps_vals:
                        qq = rng.uniform(-spec.Q, spec.Q, size=spec.d)
                        qq[i] = np.clip(s * (spec.Q[i] - z) + e, -spec.Q[i], spec.Q[i])
                        extras_q.append(qq); extras_t.append(tt)
    t = np.concatenate([t, np.asarray(extras_t)])
    q = np.vstack([q, np.asarray(extras_q)])
    return t, q


def certificate(spec, proxy, params, t, q, batch=4096):
    """sup |residual| over the grid (physical units) and the implied value bound at t=0."""
    sup = 0.0
    for s in range(0, t.size, batch):
        prep = res_mod.prepare_batch(spec, proxy, t[s:s + batch], q[s:s + batch])
        r = np.asarray(res_mod.residual_fn(params, prep)) * prep["res_scale"]
        sup = max(sup, float(np.abs(r).max()))
    return sup, sup * spec.T


# ------------------------- quote metrics -------------------------

def quote_table(spec, theta_fn, t, qg):
    """delta quotes on the lattice for each (n, side, atom): NaN where infeasible."""
    sj = spec.to_jax()
    n, K = spec.n_tiers, spec.z_atoms.shape[-1]
    th = np.asarray([theta_fn(t, np.array([q])) for q in qg])
    out = np.full((qg.size, n, 2, K), np.nan)
    hgrid = qg[1] - qg[0]
    for nn_ in range(n):
        for side, sgn in ((0, 1.0), (1, -1.0)):
            for k in range(K):
                z = float(spec.z_atoms[0, nn_, side, k])
                shift = int(round(sgn * z / hgrid))
                for j in range(qg.size):
                    jn = j + shift
                    if 0 <= jn < qg.size:
                        p = (th[j] - th[jn] + float(spec.c[0, nn_, side])) / z
                        out[j, nn_, side, k] = float(ham.delta_star(
                            jnp.asarray(p), sj["kind"][0, nn_, side], sj["ip"][0, nn_, side],
                            spec.delta_inf))
    return out


def fill_weighted_quote_error(spec, d_ref, d_cand):
    """RMSE of quotes weighted by the reference policy's fill intensity."""
    sj = spec.to_jax()
    n, K = spec.n_tiers, spec.z_atoms.shape[-1]
    num = den = 0.0
    for nn_ in range(n):
        for side in (0, 1):
            for k in range(K):
                m = np.isfinite(d_ref[:, nn_, side, k]) & np.isfinite(d_cand[:, nn_, side, k])
                if not m.any():
                    continue
                lam_w = np.asarray(ham.lam(jnp.asarray(d_ref[m, nn_, side, k]),
                                           sj["kind"][0, nn_, side], sj["ip"][0, nn_, side]))
                w = lam_w * float(spec.p_atoms[0, nn_, side, k])
                err = d_cand[m, nn_, side, k] - d_ref[m, nn_, side, k]
                num += float((w * err ** 2).sum()); den += float(w.sum())
    return np.sqrt(num / max(den, 1e-300))


# ------------------- method of lines, arbitrary d (tensor lattice) -------------------

def mol_solve_nd(spec, t_eval, n_steps=1400):
    """Exact-in-q lattice reference at any d: per-asset base step = min atom (all atoms
    must be integer multiples), grid spans [-Q_i, Q_i], nonlocal shifts are exact index
    shifts and risk-limit censoring is the lattice edge. RK4 in tau = T - t, jitted.
    State count prod_i(2 Q_i/h_i + 1) grows exponentially in d — this is the method the
    PINN exists to replace above d ~ 3-4."""
    d = spec.d
    hs, grids = [], []
    for i in range(d):
        z_all = spec.z_atoms[i].ravel()
        h = float(z_all.min())
        assert np.allclose(z_all / h, np.round(z_all / h)), "atoms lattice-commensurate"
        M = int(round(spec.Q[i] / h))
        assert abs(M * h - spec.Q[i]) < 1e-9 * spec.Q[i], "Q lattice-commensurate"
        hs.append(h); grids.append(h * np.arange(-M, M + 1))
    mesh = np.meshgrid(*grids, indexing="ij")
    shape = mesh[0].shape
    static = np.zeros(shape)
    for i in range(d):
        static -= 0.5 * spec.gamma * sum(spec.Sigma[i, j] * mesh[i] * mesh[j]
                                         for j in range(d))
        static += spec.mu[i] * mesh[i]
    static = jnp.asarray(static)

    combos = []          # (axis, cells signed, z, pz, kind, ip, c, mask)
    for i in range(d):
        for nn_ in range(spec.n_tiers):
            for side, sgn in ((0, 1), (1, -1)):
                for k in range(spec.z_atoms.shape[-1]):
                    z = float(spec.z_atoms[i, nn_, side, k])
                    cells = sgn * int(round(z / hs[i]))
                    idx = np.arange(shape[i])
                    ok_ax = (idx + cells >= 0) & (idx + cells < shape[i])
                    ok = np.ones(shape, bool)
                    sl = [None] * d; sl[i] = slice(None)
                    ok = ok & ok_ax[tuple(sl)]
                    combos.append((i, cells, z, float(spec.p_atoms[i, nn_, side, k]),
                                   int(spec.kind[i, nn_, side]),
                                   jnp.asarray(spec.ip[i, nn_, side]),
                                   float(spec.c[i, nn_, side]), jnp.asarray(ok)))

    dinf = spec.delta_inf

    def F(theta):
        out = static
        for (ax, cells, z, pz, kd, ipv, c, ok) in combos:
            th_s = jnp.roll(theta, -cells, axis=ax)          # theta(q + cells*h e_ax)
            p = jnp.where(ok, (theta - th_s + c) / z, 0.0)
            Hv = ham.H0(p, jnp.asarray(kd), ipv, dinf)
            out = out + pz * z * jnp.where(ok, Hv, 0.0)
        return out

    F = jax.jit(F)
    dt = spec.T / n_steps
    tau_snap = np.sort(spec.T - np.asarray(t_eval))
    theta = jnp.zeros(shape)
    out = {}
    tau = 0.0
    for _ in range(n_steps):
        for ts_ in tau_snap:
            if abs(tau - ts_) < 0.5 * dt and ts_ not in out:
                out[ts_] = np.asarray(theta)
        k1 = F(theta); k2 = F(theta + 0.5 * dt * k1)
        k3 = F(theta + 0.5 * dt * k2); k4 = F(theta + dt * k3)
        theta = theta + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        tau += dt
    out[spec.T] = np.asarray(theta)
    thetas = np.stack([out[min(out, key=lambda x: abs(x - (spec.T - te)))]
                       for te in t_eval])
    return grids, thetas


def lattice_quotes_nd(spec, theta_grid, grids):
    """Quotes for every lattice state and (asset, atom, side) from exact index-shift
    differences. Returns delta (states..., d, K, 2) and validity mask (same shape)."""
    d, K = spec.d, spec.z_atoms.shape[-1]
    shape = theta_grid.shape
    delta = np.full(shape + (d, K, 2), np.nan)
    okall = np.zeros(shape + (d, K, 2), bool)
    sj = spec.to_jax()
    for i in range(d):
        h = grids[i][1] - grids[i][0]
        for k in range(K):
            z = float(spec.z_atoms[i, 0, 0, k])
            for side, sgn in ((0, 1), (1, -1)):
                cells = sgn * int(round(z / h))
                th_s = np.roll(theta_grid, -cells, axis=i)
                idx = np.arange(shape[i])
                ok_ax = (idx + cells >= 0) & (idx + cells < shape[i])
                sl = [None] * d; sl[i] = slice(None)
                ok = np.broadcast_to(ok_ax[tuple(sl)], shape)
                p = np.where(ok, (theta_grid - th_s + float(spec.c[i, 0, side])) / z, 0.0)
                dd = np.asarray(ham.delta_star(jnp.asarray(p), sj["kind"][i, 0, side],
                                               sj["ip"][i, 0, side], spec.delta_inf))
                delta[..., i, k, side] = np.where(ok, dd, np.nan)
                okall[..., i, k, side] = ok
    return delta, okall


def mol_solve_nd_scan(spec, t_eval, n_steps=1400):
    """lax.scan version of mol_solve_nd for GPU: the time loop runs on-device in
    equal-length scan segments between snapshots (one compiled scan, reused), with
    a donated carry — no per-step host dispatch. Identical math to mol_solve_nd.
    H200 sizing: d=5 (9.8e6 states, 40-term stencil) ~ minutes; d=6 (2.4e8 states,
    ~2 GB/buffer) feasible; d=7 (49 GB/buffer, 5-6 RK4 buffers) exceeds HBM."""
    d = spec.d
    hs, grids = [], []
    for i in range(d):
        z_all = spec.z_atoms[i].ravel(); h = float(z_all.min())
        M = int(round(spec.Q[i] / h))
        hs.append(h); grids.append(h * np.arange(-M, M + 1))
    mesh = np.meshgrid(*grids, indexing="ij")
    shape = mesh[0].shape
    static = np.zeros(shape)
    for i in range(d):
        static -= 0.5 * spec.gamma * sum(spec.Sigma[i, j] * mesh[i] * mesh[j] for j in range(d))
        static += spec.mu[i] * mesh[i]
    static = jnp.asarray(static)
    combos = []
    for i in range(d):
        for nn_ in range(spec.n_tiers):
            for side, sgn in ((0, 1), (1, -1)):
                for k in range(spec.z_atoms.shape[-1]):
                    z = float(spec.z_atoms[i, nn_, side, k])
                    cells = sgn * int(round(z / hs[i]))
                    idx = np.arange(shape[i])
                    ok_ax = (idx + cells >= 0) & (idx + cells < shape[i])
                    sl = [None] * d; sl[i] = slice(None)
                    ok = np.broadcast_to(ok_ax[tuple(sl)], shape)
                    combos.append((i, cells, z, float(spec.p_atoms[i, nn_, side, k]),
                                   int(spec.kind[i, nn_, side]),
                                   jnp.asarray(spec.ip[i, nn_, side]),
                                   float(spec.c[i, nn_, side]),
                                   jnp.asarray(np.ascontiguousarray(ok))))
    dinf = spec.delta_inf

    def F(theta):
        out = static
        for (ax, cells, z, pz, kd, ipv, cc, ok) in combos:
            th_s = jnp.roll(theta, -cells, axis=ax)
            p = jnp.where(ok, (theta - th_s + cc) / z, 0.0)
            out = out + pz * z * jnp.where(ok, ham.H0(p, jnp.asarray(kd), ipv, dinf), 0.0)
        return out

    n_seg = len(t_eval) - 1 if len(t_eval) > 1 else 1
    dt = spec.T / n_steps
    tau_targets = np.sort(spec.T - np.asarray(t_eval, float))
    seg_lens = np.diff(np.concatenate([[0.0], tau_targets]))
    Ls = np.round(seg_lens / dt).astype(int)
    assert np.allclose(Ls * dt, seg_lens, atol=1e-9 * spec.T), \
        "each snapshot interval must be an integer number of steps"

    def make_segment(L):
        @jax.jit
        def segment(theta):
            def body(th, _):
                k1 = F(th); k2 = F(th + 0.5 * dt * k1)
                k3 = F(th + 0.5 * dt * k2); k4 = F(th + dt * k3)
                return th + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None
            th, _ = jax.lax.scan(body, theta, None, length=L)
            return th
        return segment
    seg_cache = {}

    theta = jnp.zeros(shape)
    tau = 0.0
    snaps = {0.0: np.asarray(theta)}
    for L in Ls:
        if L > 0:
            if int(L) not in seg_cache:
                seg_cache[int(L)] = make_segment(int(L))
            theta = seg_cache[int(L)](theta)
            tau += L * dt
        snaps[round(tau, 12)] = np.asarray(theta)
    thetas = np.stack([snaps[min(snaps, key=lambda x: abs(x - (spec.T - te)))]
                       for te in t_eval])
    return grids, thetas
