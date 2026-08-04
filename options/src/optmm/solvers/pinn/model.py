"""PINN value model for stage 1: a pure MLP over (t, nu, n).

Deliberate design constraints (see docs/derivations.md section 6):
- NO w^T n feature, NO constraint-distance features: the hyperplane structure
  grad_n v || w is the stage-1 *discovery metric* and must not be leaked into
  the architecture.  Stage 2/3 may add axis-aligned box-distance features
  (constraint geometry is public; risk geometry is not).
- Hard terminal condition: v_theta(t, nu, n) = (1 - t/T) * V_SCALE * NN(x),
  so v(T, ., .) = 0 exactly and the residual loss carries no BC penalty term.
- Inputs normalized to O(1): t -> 2t/T - 1, nu -> (nu - mid)/half,
  n_i -> n_i * w_i / Vbar (per-dim band half-width).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PinnSpec:
    T: float
    nu_lo: float
    nu_hi: float
    w: jnp.ndarray            # (N,) per-trade vegas
    Vbar: float
    hidden: Sequence[int] = (128, 128, 128)
    v_scale: float = 3.0e4


def init_params(spec: PinnSpec, key):
    dims = [2 + int(spec.w.shape[0])] + list(spec.hidden) + [1]
    params = []
    for i in range(len(dims) - 1):
        key, k = jax.random.split(key)
        Wm = jax.random.normal(k, (dims[i], dims[i + 1])) \
            * jnp.sqrt(2.0 / dims[i])
        params.append({"W": Wm, "b": jnp.zeros(dims[i + 1])})
    return params


def _features(spec: PinnSpec, t, nu, n):
    nu_mid = 0.5 * (spec.nu_lo + spec.nu_hi)
    nu_half = 0.5 * (spec.nu_hi - spec.nu_lo)
    return jnp.concatenate([
        jnp.atleast_1d(2.0 * t / spec.T - 1.0),
        jnp.atleast_1d((nu - nu_mid) / nu_half),
        n * spec.w / spec.Vbar,
    ])


def value(params, spec: PinnSpec, t, nu, n):
    """Scalar v_theta(t, nu, n); n is a float vector (trade units)."""
    x = _features(spec, t, nu, n)
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["W"] + layer["b"])
    out = (x @ params[-1]["W"] + params[-1]["b"])[0]
    return (1.0 - t / spec.T) * spec.v_scale * out
