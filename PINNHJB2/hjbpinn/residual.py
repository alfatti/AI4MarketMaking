"""Residual of the true (finite-Q) Model B Hamilton-Jacobi equation, organized so that
everything parameter-independent is precomputed once per collocation batch:

  resid(t,q) = [ dt(theta) + mu'q - (gamma/2) q'Sigma q
                 + sum_{i,n,side,atom} 1{feasible} * p_atom * z * H0((Dtheta + c)/z) ] / res_scale

with theta = theta_check + eta. The proxy differences and dt(theta_check) are analytic
(A', B', C' from the ODE right-hand sides — no AD through eigendecompositions or
quadrature), so Gauss-Newton Jacobians only ever traverse the small MLP.

Masks are applied to contributions; shifted inputs for infeasible stencil entries are
clamped into the domain so the network never sees out-of-range inputs.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from . import hamiltonians as ham
from . import network


def scales(spec):
    """Characteristic residual and value scales for nondimensionalizing the loss."""
    sj = spec.to_jax()
    a0, _, _ = ham.alphas(sj["kind"], sj["ip"], spec.delta_inf)
    m1 = jnp.asarray(spec.moments()[1])
    myopic = float((a0 * m1).sum())
    risk = float(0.5 * spec.gamma * abs(spec.Q @ spec.Sigma @ spec.Q))
    drift = float(np.abs(spec.mu) @ spec.Q)
    res_scale = max(myopic, risk, drift, 1e-12)
    theta_scale = 0.5 * res_scale * spec.T
    return res_scale, theta_scale


def prepare_batch(spec, proxy, t, q):
    """Precompute all parameter-independent tensors for a fixed collocation set.

    t: (N,), q: (N, d). Stencil axes: (i, n, side, k).
    """
    sj = spec.to_jax()
    t = jnp.asarray(t); q = jnp.asarray(q)
    N, d = q.shape
    n, K = spec.n_tiers, spec.z_atoms.shape[-1]
    res_scale, theta_scale = scales(spec)

    AB = jax.jit(jax.vmap(lambda tt: (proxy.A(tt), proxy.B(tt))))(t)
    At, Bt = AB                                            # (N,d,d), (N,d)
    dt_check = jax.jit(jax.vmap(proxy.dtheta_dt))(t, q, At, Bt)     # (N,)
    static = q @ sj["mu"] - 0.5 * sj["gamma"] * jnp.einsum("bi,ij,bj->b", q, sj["Sigma"], q)

    z = sj["z"]                                            # (d,n,2,K)
    sgn = jnp.array([1.0, -1.0])[None, None, :, None]      # bid +, ask -
    # proxy differences: 2 sgn z (q'A)_i + z^2 A_ii + sgn z B_i   -> (N,d,n,2,K)
    qA = jnp.einsum("bi,bij->bj", q, At)                   # (N,d)
    diff_check = (2.0 * sgn * z[None] * qA[:, :, None, None, None]
                  + z[None] ** 2 * jnp.einsum("bii->bi", At)[:, :, None, None, None]
                  + sgn * z[None] * Bt[:, :, None, None, None])

    # feasibility masks and clamped shifted inventories
    Q = sj["Q"]
    qi = q[:, :, None, None, None]                         # (N,d,1,1,1)
    q_new_i = qi + sgn * z[None]                           # post-trade inventory in asset i
    mask = (q_new_i <= Q[None, :, None, None, None] + 1e-12) & \
           (q_new_i >= -Q[None, :, None, None, None] - 1e-12)
    q_new_i_safe = jnp.clip(q_new_i, -Q[None, :, None, None, None], Q[None, :, None, None, None])

    # shifted full inventory vectors -> features for the MLP
    eye = jnp.eye(d)
    q_shift = (q[:, None, None, None, None, :]
               + (q_new_i_safe - qi)[..., None] * eye[None, :, None, None, None, :])  # (N,d,n,2,K,d)
    zbar = jnp.asarray(spec.zbar())
    fs = network.feature_spec(spec)
    feat = jax.vmap(lambda tt, qq: network.features(tt, qq, fs))
    feat_base = feat(t, q)                                                       # (N,F)
    S = d * n * 2 * K
    q_shift_flat = q_shift.reshape(N * S, d)
    t_rep = jnp.repeat(t, S)
    feat_shift = feat(t_rep, q_shift_flat)                                       # (N*S,F)

    kind_b = jnp.broadcast_to(sj["kind"][None, :, :, :, None], (N, d, n, 2, K))
    ip_b = jnp.broadcast_to(sj["ip"][None, :, :, :, None, :], (N, d, n, 2, K, 3))
    c_b = jnp.broadcast_to(sj["c"][None, :, :, :, None], (N, d, n, 2, K))
    pz = jnp.broadcast_to(sj["pz"][None], (N, d, n, 2, K))
    zz = jnp.broadcast_to(z[None], (N, d, n, 2, K))

    # flatten the stencil axes (d, n, 2, K) -> S per point: the residual is POINTWISE,
    # which lets the Jacobian be assembled as a per-sample gradient under vmap
    # (single-point graph per row) instead of jacrev over a coupled batch vector.
    def fl(x):
        return x.reshape(N, S, *x.shape[5:])
    return dict(
        t=t, feat_base=feat_base, feat_shift=feat_shift.reshape(N, S, -1),
        diff_check=fl(diff_check), dt_check=dt_check, static=static,
        mask=fl(mask.astype(feat_base.dtype)), kind=fl(kind_b), ip=fl(ip_b),
        c=fl(c_b), pz=fl(pz), z=fl(zz),
        T=spec.T, delta_inf=spec.delta_inf,
        res_scale=res_scale, theta_scale=theta_scale,
    )


_POINT_KEYS = ("t", "feat_base", "feat_shift", "diff_check", "dt_check", "static",
               "mask", "kind", "ip", "c", "pz", "z")
_SCALAR_KEYS = ("T", "delta_inf", "res_scale", "theta_scale")


def residual_point(params, pt, aux):
    """Scaled residual at ONE collocation point. pt: per-point slices; aux: scalars."""
    T, ts = aux["T"], aux["theta_scale"]
    tfac = ts * (1.0 - pt["t"] / T)

    nn_base = network.mlp(params, pt["feat_base"])
    nn_shift = jax.vmap(lambda f: network.mlp(params, f))(pt["feat_shift"])   # (S,)
    d_eta = tfac * (nn_base - nn_shift)

    tangent = jnp.zeros_like(pt["feat_base"]).at[0].set(1.0 / T)
    _, dnn_dt = jax.jvp(lambda x: network.mlp(params, x), (pt["feat_base"],), (tangent,))
    dt_eta = -(ts / T) * nn_base + tfac * dnn_dt

    p = (pt["diff_check"] + d_eta + pt["c"]) / pt["z"]
    H = ham.H0(p, pt["kind"], pt["ip"], aux["delta_inf"])
    jump = (pt["mask"] * pt["pz"] * pt["z"] * H).sum()

    return (pt["dt_check"] + dt_eta + pt["static"] + jump) / aux["res_scale"]


def split_prep(prep):
    pts = {k: prep[k] for k in _POINT_KEYS}
    aux = {k: prep[k] for k in _SCALAR_KEYS}
    return pts, aux


def residual_fn(params, prep):
    """(N,) scaled residual. Only the MLP depends on params."""
    pts, aux = split_prep(prep)
    return jax.vmap(lambda pt: residual_point(params, pt, aux))(pts)


# ----------------------- Phase-0 check evaluators -----------------------

def quadratic_residual_production_path(spec, proxy, t, q):
    """Residual of the QUADRATIC HJ (Q = inf, eta = 0) using the production code path
    (analytic diffs + A',B',C' RHS). Must be ~0: validates aggregation + derivations."""
    sj = spec.to_jax()
    a0, a1, a2 = ham.alphas(sj["kind"], sj["ip"], spec.delta_inf)
    prep = prepare_batch(spec, proxy, t, q)
    N = prep["diff_check"].shape[0]
    pq = (prep["diff_check"] + prep["c"]) / prep["z"]                       # (N, S)
    K = spec.z_atoms.shape[-1]
    def bc(a):
        return jnp.broadcast_to(a[None, :, :, :, None], (N,) + a.shape + (K,)).reshape(N, -1)
    a0b, a1b, a2b = bc(a0), bc(a1), bc(a2)
    Hq = a0b + a1b * pq + 0.5 * a2b * pq ** 2
    jump = (prep["pz"] * prep["z"] * Hq).sum(axis=1)              # NO masks: Q = inf
    return (prep["dt_check"] + prep["static"] + jump) / prep["res_scale"]


def quadratic_residual_independent_path(spec, proxy, t, q, h=None):
    """Same quantity, maximally independent implementation: theta_check differences by
    direct subtraction of proxy.theta (with C), time derivative by central finite
    difference. Validates that the closed forms (spectral A, quadrature B, C) actually
    solve the derived ODE system."""
    sj = spec.to_jax()
    h = 1e-5 * spec.T if h is None else h
    a0, a1, a2 = ham.alphas(sj["kind"], sj["ip"], spec.delta_inf)
    res_scale, _ = scales(spec)
    d, n, K = spec.d, spec.n_tiers, spec.z_atoms.shape[-1]
    out = []
    for tt, qq in zip(np.asarray(t), np.asarray(q)):
        tt = float(tt); qq = jnp.asarray(qq)
        th0 = proxy.theta(tt, qq)
        dt_fd = (proxy.theta(min(tt + h, spec.T * (1 - 1e-12)), qq)
                 - proxy.theta(max(tt - h, 0.0), qq)) / (
                    min(tt + h, spec.T * (1 - 1e-12)) - max(tt - h, 0.0))
        static = float(qq @ sj["mu"] - 0.5 * sj["gamma"] * qq @ sj["Sigma"] @ qq)
        jump = 0.0
        for i in range(d):
            ei = np.zeros(d); ei[i] = 1.0
            for nn_ in range(n):
                for side, sgn in ((0, 1.0), (1, -1.0)):
                    for k in range(K):
                        z = float(spec.z_atoms[i, nn_, side, k])
                        pz = float(spec.p_atoms[i, nn_, side, k])
                        th1 = proxy.theta(tt, qq + sgn * z * jnp.asarray(ei))
                        p = (float(th0 - th1) + float(spec.c[i, nn_, side])) / z
                        Hq = float(a0[i, nn_, side] + a1[i, nn_, side] * p
                                   + 0.5 * a2[i, nn_, side] * p * p)
                        jump += pz * z * Hq
        out.append((float(dt_fd) + static + jump) / res_scale)
    return np.asarray(out)
