"""Correction network: theta(t, q) = theta_check(t, q) + (1 - t/T) * theta_scale * NN(features).

Features (feature_spec builds the static data once per market):
  [ t/T,  q/Q,  exp(-(Q-q)/zbar),  exp(-(Q+q)/zbar),  -expm1(-|q_i - s|/zbar_i) per surface ]

The last group are KINK features, one per switching surface s = +-(Q_i - z) (z over the
unique atoms of asset i; a surface at the origin appears once). Rationale (Phase-1
diagnosis): the equation's right-hand side jumps across each surface (an indicator
switches), so the true theta has a kink in q there and d_t theta is discontinuous in q —
a smooth-in-q ansatz carries an irreducible half-jump residual AT the surface regardless
of capacity. A t-modulated kink feature gives theta_hat a kink whose amplitude the net
controls, so d_t theta_hat inherits the jump. The -expm1 form is |x| near the surface
(unit-slope kink) but saturates within ~3 zbar — bounded, well-conditioned inputs.

Exponential boundary features handle the (smooth) censoring boundary layer amplitude;
affine distances would be vacuous through the affine first layer.

Output layer is zero-initialized: eta == 0 at init — training starts exactly at the proxy.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def feature_spec(spec):
    """Static feature data: box scales and the switching-surface list."""
    si, sp_ = [], []
    for i in range(spec.d):
        for z in np.unique(spec.z_atoms[i]):
            s = float(spec.Q[i] - z)
            if s > 1e-12:
                si += [i, i]; sp_ += [s, -s]
            elif abs(s) <= 1e-12:
                si += [i]; sp_ += [0.0]
    return dict(T=spec.T, Q=jnp.asarray(spec.Q), zbar=jnp.asarray(spec.zbar()),
                si=jnp.asarray(si, dtype=int), sp=jnp.asarray(sp_))


def n_features(fs) -> int:
    return 1 + 3 * int(fs["Q"].shape[0]) + int(fs["si"].shape[0])


def features(t, q, fs):
    T, Q, zbar = fs["T"], fs["Q"], fs["zbar"]
    tt = jnp.atleast_1d(t / T)
    qt = q / Q
    fb_hi = jnp.exp(-(Q - q) / zbar)
    fb_lo = jnp.exp(-(Q + q) / zbar)
    kink = -jnp.expm1(-jnp.abs(q[fs["si"]] - fs["sp"]) / zbar[fs["si"]])
    return jnp.concatenate([tt, qt, fb_hi, fb_lo, kink])


def init_params(key, n_in, widths=(64, 64), dtype=jnp.float64):
    sizes = [n_in, *widths, 1]
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (a, b) in zip(keys, zip(sizes[:-1], sizes[1:])):
        W = jax.random.normal(k, (a, b), dtype) * jnp.sqrt(1.0 / a)
        params.append((W, jnp.zeros((b,), dtype)))
    W_last, b_last = params[-1]
    params[-1] = (jnp.zeros_like(W_last), b_last)          # eta == 0 at init
    return params


def mlp(params, x):
    h = x
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[0]


def eta(params, t, q, fs, theta_scale):
    return theta_scale * (1.0 - t / fs["T"]) * mlp(params, features(t, q, fs))
