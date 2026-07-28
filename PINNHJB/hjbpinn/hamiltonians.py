"""Model B (xi = 0) Hamiltonians.

H0(p) = sup_{delta >= -delta_inf} Lambda(delta) * (delta - p)

Uniform treatment of the floor: delta*(p) = max(root(p), -delta_inf), where root(p) is
the unconstrained argmax; since the objective is unimodal in delta, clamping at the floor
is exact. H0 is evaluated with delta* under lax.stop_gradient, so autodiff of H0 w.r.t. p
returns exactly -Lambda(delta*) (Danskin / envelope theorem) — correct for Gauss-Newton,
which never needs second derivatives of the residual.

Branch safety: both intensity-family branches are always evaluated under jnp.where, so
each branch substitutes benign parameters for entries belonging to the other family
(_safe_* helpers). This avoids the classic where-NaN gradient trap.

All functions broadcast over arbitrary leading batch shapes; (kind, ip) select the
family per (asset, tier, side): 0 exponential (A, k, -), 1 logistic (lam, alpha, beta).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_NEWTON_ITERS = 40


def _safe_exp_params(kind, ip):
    A = jnp.where(kind == 0, ip[..., 0], 1.0)
    k = jnp.where(kind == 0, ip[..., 1], 1.0)
    return A, k


def _safe_log_params(kind, ip):
    lmb = jnp.where(kind == 1, ip[..., 0], 1.0)
    alpha = jnp.where(kind == 1, ip[..., 1], 0.0)
    beta = jnp.where(kind == 1, ip[..., 2], 1.0)
    return lmb, alpha, beta


# ------------------------- intensity families -------------------------

def lam(delta, kind, ip):
    A, k = _safe_exp_params(kind, ip)
    l_exp = A * jnp.exp(jnp.clip(-k * delta, -60.0, 60.0))
    lmb, alpha, beta = _safe_log_params(kind, ip)
    x = jnp.clip(alpha + beta * delta, -60.0, 60.0)
    l_log = lmb / (1.0 + jnp.exp(x))
    return jnp.where(kind == 0, l_exp, l_log)


def lam_prime(delta, kind, ip):
    A, k = _safe_exp_params(kind, ip)
    lp_exp = -k * A * jnp.exp(jnp.clip(-k * delta, -60.0, 60.0))
    lmb, alpha, beta = _safe_log_params(kind, ip)
    x = jnp.clip(alpha + beta * delta, -60.0, 60.0)
    u = jax.nn.sigmoid(x)
    lp_log = -lmb * beta * u * (1.0 - u)
    return jnp.where(kind == 0, lp_exp, lp_log)


# ------------------------- unconstrained argmax root -------------------------

def _root_exp(p, kind, ip):
    _, k = _safe_exp_params(kind, ip)
    return p + 1.0 / k


def _lambertw0_from_log(L):
    """w = W0(exp(L)) computed stably in log space: solve f(w) = w + log w - L = 0, w > 0.

    Newton: w <- w - (w + log w - L) * w / (w + 1); quadratic convergence from
    w0 = L - log L (L large) or w0 = exp(L) (L small), overflow-free for any L.
    """
    w0_big = L - jnp.log(jnp.maximum(L, 1.1))
    w0_small = jnp.exp(jnp.clip(L, -60.0, 1.0))
    w = jnp.where(L > 1.0, jnp.maximum(w0_big, 1e-12), w0_small)

    def body(_, ww):
        f = ww + jnp.log(ww) - L
        ww_new = ww - f * ww / (ww + 1.0)
        return jnp.maximum(ww_new, 0.2 * ww)            # guard positivity
    return jax.lax.fori_loop(0, _NEWTON_ITERS, body, w)


def _root_logistic(p, kind, ip):
    """Closed form via Lambert-W: with y = beta*(delta - p) - 1, the FOC
    Lambda'(delta)(delta - p) + Lambda(delta) = 0 becomes y e^y = exp(-(alpha+beta*p+1)),
    so delta* = p + (1 + W0(exp(-(alpha + beta*p + 1)))) / beta.
    """
    _, alpha, beta = _safe_log_params(kind, ip)
    L = -(alpha + beta * p + 1.0)
    return p + (1.0 + _lambertw0_from_log(L)) / beta


def root(p, kind, ip):
    return jnp.where(kind == 0, _root_exp(p, kind, ip), _root_logistic(p, kind, ip))


def delta_star(p, kind, ip, delta_inf):
    """Optimal offset with the floor applied (exact by unimodality)."""
    return jnp.maximum(root(p, kind, ip), -delta_inf)


# ------------------------- Hamiltonian and derivative -------------------------

def H0(p, kind, ip, delta_inf):
    ds = jax.lax.stop_gradient(delta_star(p, kind, ip, delta_inf))
    return lam(ds, kind, ip) * (ds - p)


def H0_prime(p, kind, ip, delta_inf):
    """Exact: H0'(p) = -Lambda(delta*(p)), both branches (envelope / boundary)."""
    ds = delta_star(p, kind, ip, delta_inf)
    return -lam(ds, kind, ip)


# ------------------------- Taylor coefficients at p = 0 -------------------------

def alphas(kind, ip, delta_inf):
    """alpha_j = d^j/dp^j H0(0), j = 0,1,2 — z-independent for Model B (xi = 0).

    alpha2 via implicit differentiation of the FOC:
      exponential: delta*'(p) = 1                     -> alpha2 = k * Lambda(delta*)
      logistic:    delta*'(p) = sigmoid(alpha + beta*delta*)
    If the floor binds at p = 0 (pathological calibration), H0 is locally linear:
    alpha2 = 0, alpha1 = -Lambda(-delta_inf).
    """
    p0 = jnp.zeros(kind.shape)
    r = root(p0, kind, ip)
    clamped = r < -delta_inf
    ds = jnp.maximum(r, -delta_inf)
    a0 = lam(ds, kind, ip) * (ds - p0)
    a1 = -lam(ds, kind, ip)
    _, alpha_l, beta_l = _safe_log_params(kind, ip)
    dsp_log = jax.nn.sigmoid(jnp.clip(alpha_l + beta_l * ds, -60.0, 60.0))
    dsp = jnp.where(kind == 0, 1.0, dsp_log)
    a2_free = -lam_prime(ds, kind, ip) * dsp
    a2 = jnp.where(clamped, 0.0, a2_free)
    return a0, a1, a2
