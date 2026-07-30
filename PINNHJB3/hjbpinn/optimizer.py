"""Gauss-Newton with Levenberg-Marquardt trust-region control, Jacobi column scaling,
and the push-through (kernel-trick) solve in residual space:

  minimize (1/2)||r(theta)||^2
  step:  Dt = -D^{-1/2} Jt' (Jt Jt' + lam I)^{-1} r,  Jt = J D^{-1/2},  D = diag(J'J)

Deterministic residual assumed (fixed collocation + quadrature per phase), per the
curvature-aware paper's guidance; lam adapted by actual-vs-predicted reduction.
"""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def gauss_newton(residual, params0, n_iters=150, lam0=1e-3, lam_min=1e-12, lam_max=1e8,
                 tol_r=1e-12, verbose=True, log_every=10, per_sample=None):
    """residual: callable(params) -> (N,). Returns (params, history).

    If per_sample = (residual_point, pts, aux) is given, the Jacobian is assembled as
    vmap-of-grad of the scalar per-point residual — one single-point graph per row,
    O(N * P) memory — instead of jacrev over the coupled batch vector (which batches
    the full graph per output row and exhausts memory for N in the thousands).
    """
    pvec0, unravel = ravel_pytree(params0)

    r_flat = jax.jit(lambda pv: residual(unravel(pv)))
    if per_sample is not None:
        res_pt, pts, aux = per_sample

        def _row(pv, pt):
            g = jax.grad(lambda w: res_pt(unravel(w), pt, aux))(pv)
            return g
        J_flat = jax.jit(lambda pv: jax.vmap(lambda pt: _row(pv, pt))(pts))
    else:
        J_flat = jax.jit(jax.jacrev(r_flat))

    pvec = pvec0
    lam = lam0
    hist = []
    r = np.asarray(r_flat(pvec))
    loss = 0.5 * float(r @ r)
    for it in range(n_iters):
        t0 = time.time()
        J = np.asarray(J_flat(pvec))                       # (N, P)
        d = np.sqrt((J * J).sum(0)) + 1e-30                # Jacobi column scales
        Jt = J / d[None, :]
        G = Jt @ Jt.T
        accepted = False
        for _ in range(25):
            K = G + lam * np.eye(G.shape[0])
            try:
                w = np.linalg.solve(K, r)
            except np.linalg.LinAlgError:
                lam = min(lam * 10.0, lam_max); continue
            step = -(Jt.T @ w) / d
            pred = 0.5 * float(r @ r) - 0.5 * float((r + J @ step) @ (r + J @ step))
            r_new = np.asarray(r_flat(pvec + jnp.asarray(step)))
            loss_new = 0.5 * float(r_new @ r_new)
            actual = loss - loss_new
            rho = actual / max(pred, 1e-300)
            if actual > 0 and pred > 0:
                pvec = pvec + jnp.asarray(step)
                r, loss = r_new, loss_new
                lam = max(lam / 3.0, lam_min) if rho > 0.75 else (
                    min(lam * 2.0, lam_max) if rho < 0.25 else lam)
                accepted = True
                break
            lam = min(lam * 10.0, lam_max)
        sup_r = float(np.max(np.abs(r)))
        hist.append(dict(it=it, loss=loss, sup_r=sup_r, lam=lam,
                         accepted=accepted, dt=time.time() - t0))
        if verbose and (it % log_every == 0 or it == n_iters - 1):
            print(f"  GN it {it:4d}  loss {loss:.6e}  sup|r| {sup_r:.3e}  "
                  f"lam {lam:.1e}  {'ok' if accepted else 'REJ'}")
        if not accepted and lam >= lam_max:
            break
        if loss < tol_r:
            break
    return unravel(pvec), hist


def gauss_newton_gpu(residual, params0, n_iters=150, lam0=1e-3, lam_min=1e-12,
                     lam_max=1e8, verbose=True, log_every=10, per_sample=None,
                     row_chunk=8192):
    """Device-resident GN/LM for GPU (H200) scale: J assembled in row chunks and kept
    on device; Gram via one fp64 DGEMM (tensor cores); Cholesky solve on device; only
    per-iteration scalars sync to host. Memory: J is N*P fp64 (N=P=5e4 -> 20 GB);
    Gram N^2 (20 GB); both fit HBM3e. The LM retry loop re-solves with a new lambda
    without recomputing J.  API-compatible with gauss_newton."""
    from jax.scipy.linalg import cho_factor, cho_solve
    pvec0, unravel = ravel_pytree(params0)
    r_flat = jax.jit(lambda pv: residual(unravel(pv)))
    assert per_sample is not None, "GPU path requires the per-sample kernel"
    res_pt, pts, aux = per_sample
    N = next(iter(pts.values())).shape[0]

    def _rows(pv, pts_chunk):
        return jax.vmap(lambda pt: jax.grad(
            lambda w: res_pt(unravel(w), pt, aux))(pv))(pts_chunk)
    _rows = jax.jit(_rows)

    def J_flat(pv):
        outs = []
        for s in range(0, N, row_chunk):
            sl = jax.tree_util.tree_map(lambda a: a[s:s + row_chunk], pts)
            outs.append(_rows(pv, sl))
        return jnp.concatenate(outs, axis=0)

    @jax.jit
    def _prep(J):
        d = jnp.sqrt((J * J).sum(0)) + 1e-30
        Jt = J / d[None, :]
        G = Jt @ Jt.T
        return d, Jt, G

    @jax.jit
    def _try(G, Jt, d, r, lam, pvec):
        K = G + lam * jnp.eye(G.shape[0], dtype=G.dtype)
        w = cho_solve(cho_factor(K, lower=True), r)
        step = -(Jt.T @ w) / d
        r_lin = r + (Jt * d[None, :]) @ step               # J @ step, reusing Jt
        pred = 0.5 * (r @ r) - 0.5 * (r_lin @ r_lin)
        pv_new = pvec + step
        r_new = r_flat(pv_new)
        loss_new = 0.5 * (r_new @ r_new)
        return pv_new, r_new, loss_new, pred

    pvec = jnp.asarray(pvec0); lam = lam0; hist = []
    r = r_flat(pvec); loss = float(0.5 * (r @ r))
    for it in range(n_iters):
        t0 = time.time()
        J = J_flat(pvec)
        d, Jt, G = _prep(J)
        accepted = False
        for _ in range(25):
            pv_new, r_new, loss_new_j, pred_j = _try(G, Jt, d, r, lam, pvec)
            loss_new, pred = float(loss_new_j), float(pred_j)
            actual = loss - loss_new
            rho = actual / max(pred, 1e-300)
            if actual > 0 and pred > 0:
                pvec, r, loss = pv_new, r_new, loss_new
                lam = max(lam / 3.0, lam_min) if rho > 0.75 else (
                    min(lam * 2.0, lam_max) if rho < 0.25 else lam)
                accepted = True
                break
            lam = min(lam * 10.0, lam_max)
        sup_r = float(jnp.max(jnp.abs(r)))
        hist.append(dict(it=it, loss=loss, sup_r=sup_r, lam=lam,
                         accepted=accepted, dt=time.time() - t0))
        if verbose and (it % log_every == 0 or it == n_iters - 1):
            print(f"  GNg it {it:4d}  loss {loss:.6e}  sup|r| {sup_r:.3e}  "
                  f"lam {lam:.1e}  {'ok' if accepted else 'REJ'}  {time.time()-t0:.2f}s")
        if not accepted and lam >= lam_max:
            break
    return unravel(pvec), hist
