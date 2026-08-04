"""Stage-3 PINN value model: MLP over (t, S, nu, n).

Same design rules as stage 1 (docs/derivations.md section 6): hard terminal
condition v = (1 - t/T) * scale * NN, inputs normalized to O(1), and no
risk-structure features — the per-dim n normalization uses the *frozen
anchor* per-trade vegas w_ref (public book data at t = 0), never the live
state-dependent vegas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Pinn3Spec:
    T: float
    S0: float
    S_half: float                 # half-width of the S training band
    nu_lo: float
    nu_hi: float
    w_ref: jnp.ndarray            # (N,) frozen anchor per-trade vegas
    n_max: jnp.ndarray            # (N,) box limits (trade units)
    Vbar: float
    hidden: Sequence[int] = (128, 128, 128)
    v_scale: float = 3.0e4


def init_params3(spec: Pinn3Spec, key):
    dims = [3 + int(spec.w_ref.shape[0])] + list(spec.hidden) + [1]
    params = []
    for i in range(len(dims) - 1):
        key, k = jax.random.split(key)
        Wm = jax.random.normal(k, (dims[i], dims[i + 1])) \
            * jnp.sqrt(2.0 / dims[i])
        params.append({"W": Wm, "b": jnp.zeros(dims[i + 1])})
    return params


def value3(params, spec: Pinn3Spec, t, S, nu, n):
    nu_mid = 0.5 * (spec.nu_lo + spec.nu_hi)
    nu_half = 0.5 * (spec.nu_hi - spec.nu_lo)
    x = jnp.concatenate([
        jnp.atleast_1d(2.0 * t / spec.T - 1.0),
        jnp.atleast_1d((S - spec.S0) / spec.S_half),
        jnp.atleast_1d((nu - nu_mid) / nu_half),
        n / spec.n_max,
    ])
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["W"] + layer["b"])
    out = (x @ params[-1]["W"] + params[-1]["b"])[0]
    return (1.0 - t / spec.T) * spec.v_scale * out


def sample_batch3(key, spec: Pinn3Spec, batch: int, nu_pad: float = 0.2):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    t = jax.random.uniform(k1, (batch,)) * spec.T
    S = spec.S0 + spec.S_half * jax.random.uniform(k2, (batch,),
                                                   minval=-1.0, maxval=1.0)
    pad = nu_pad * (spec.nu_hi - spec.nu_lo)
    nu = jax.random.uniform(k3, (batch,)) \
        * (spec.nu_hi - spec.nu_lo + 2 * pad) + spec.nu_lo - pad
    n = jax.random.uniform(k4, (batch, spec.w_ref.shape[0]),
                           minval=-1.0, maxval=1.0) * spec.n_max[None, :]
    return t, S, nu, n
