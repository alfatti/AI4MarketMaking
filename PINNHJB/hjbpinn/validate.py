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
