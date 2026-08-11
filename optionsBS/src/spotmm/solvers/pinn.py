"""Frozen-anchor 1-D reduction (test fixture) and the (t, S, n) PINN.

reduced1d: freeze Delta_i and sigma S at (0, S0) and impose a slab
|Delta^pi| <= Dbar; the state collapses to (t, x = Delta^pi). Explicit
Euler on an x-grid with linear-interpolated jump landings (the optmm
reduced2d pattern minus the second factor). Purpose: the frozen-regression
control — the S-featured PINN trained on the frozen problem must match
this fixture and show emergent S-independence — plus atom-exact
self-consistency via a commensurable z-override.

PINN: value model v_theta(t, S, n) with hard terminal condition and inputs
normalized by public constraint/book data only (T, the S band, the box
nbar); residual assembling the full HJB of docs/derivations.md section 3
with live deltas from a DeltaGrid and Box admissibility; a compact Adam
trainer with pluggable samplers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..core.channels import (Channels, SpotMarket, _bc, carry_coef, pen_coef)
from ..core.state import Box
from ..instruments.spot_instruments import DeltaGrid, SpotBook


# ------------------------------ reduced1d ---------------------------------

@dataclass
class Reduced1DSolution:
    x_grid: jnp.ndarray
    v0: jnp.ndarray                # (n_x,)
    channels: Channels


def solve_reduced1d(book: SpotBook, market: SpotMarket, gamma: float,
                    Dbar: float, T: float, eta: float = 1.0,
                    nt: int = 1200, n_x: int = 201,
                    delta_inf: float = -5.0) -> Reduced1DSolution:
    ch = Channels.from_book(book)
    w0 = jnp.concatenate([book.w, book.w])          # frozen anchor z*delta0
    pen0 = float(pen_coef(market, gamma, eta, 0.0, jnp.asarray(market.S0)))
    car0 = float(carry_coef(market, eta, 0.0, jnp.asarray(market.S0)))
    x = jnp.linspace(-Dbar, Dbar, n_x)
    dx = float(x[1] - x[0])
    dt = T / nt
    tgt = x[None, :] - ch.psi[:, None] * w0[:, None]
    adm = (tgt >= -Dbar * (1 + 1e-12)) & (tgt <= Dbar * (1 + 1e-12))
    f = jnp.clip((tgt + Dbar) / dx, 0.0, n_x - 1.0 - 1e-9)
    i0 = jnp.floor(f).astype(jnp.int32)
    fr = f - i0
    run = car0 * x - pen0 * x * x
    z2 = ch.z[:, None]
    inten2 = _bc(ch.intensity, 1)

    def step(v, _):
        v_sh = (1 - fr) * v[i0] + fr * v[i0 + 1]
        p = (v[None, :] - v_sh) / z2
        H = inten2.H(p, delta_inf)
        jump = jnp.sum(jnp.where(adm, z2 * H, 0.0), axis=0)
        return v + dt * (run + jump), None

    v0, _ = jax.lax.scan(step, jnp.zeros(n_x), None, length=nt)
    return Reduced1DSolution(x_grid=x, v0=v0, channels=ch)


# --------------------------------- PINN -----------------------------------

@dataclass(frozen=True)
class PinnSpec:
    T: float
    S0: float
    S_half: float
    n_max: jnp.ndarray             # (N,) box limits (public constraint data)
    hidden: Sequence[int] = (64, 64, 64)
    v_scale: float = 8.0e5


def init_params(spec: PinnSpec, key):
    dims = [2 + int(spec.n_max.shape[0])] + list(spec.hidden) + [1]
    params = []
    for i in range(len(dims) - 1):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (dims[i], dims[i + 1])) * jnp.sqrt(2.0 / dims[i])
        params.append({"W": W, "b": jnp.zeros(dims[i + 1])})
    return params


def value(params, spec: PinnSpec, t, S, n):
    x = jnp.concatenate([jnp.atleast_1d(2.0 * t / spec.T - 1.0),
                         jnp.atleast_1d((S - spec.S0) / spec.S_half),
                         n / spec.n_max])
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["W"] + layer["b"])
    return (1.0 - t / spec.T) * spec.v_scale * (x @ params[-1]["W"]
                                                + params[-1]["b"])[0]


def residual(params, spec: PinnSpec, book: SpotBook, dgrid: DeltaGrid,
             market: SpotMarket, gamma: float, eta: float, box: Box,
             t, S, n, delta_inf=-5.0, ch: Optional[Channels] = None,
             greeks=None):
    """Full HJB residual at one (t, S, n); greeks defaults to the live
    DeltaGrid, and can be pinned to a frozen callable for the regression
    control."""
    ch = Channels.from_book(book) if ch is None else ch
    N = book.n_options
    inst_idx = jnp.arange(2 * N) % N
    eye = jnp.eye(N)
    shifts = -ch.psi[:, None] * eye[inst_idx]
    g = (lambda tt, SS: dgrid(tt, SS)) if greeks is None else greeks
    delta = g(t, S)                                    # (N,)
    Dpi = jnp.sum(n * book.z * delta)
    v_t = jax.grad(lambda a: value(params, spec, a, S, n))(t)
    v_S = jax.grad(lambda a: value(params, spec, t, a, n))(S)
    v_SS = jax.jacfwd(jax.grad(lambda a: value(params, spec, t, a, n)))(S)
    sig = market.sigma(t, S)
    run = carry_coef(market, eta, t, S) * Dpi \
        - pen_coef(market, gamma, eta, t, S) * Dpi**2
    n_tgt = n[None, :] + shifts
    adm = box.admissible(n_tgt, book.w)
    v0 = value(params, spec, t, S, n)
    v_tgt = jax.vmap(lambda m: value(params, spec, t, S, m))(n_tgt)
    p = (v0 - v_tgt) / ch.z
    H = _bc(ch.intensity, 0).H(p, delta_inf)
    jumps = jnp.sum(jnp.where(adm, ch.z * H, 0.0))
    return (v_t + market.mu * S * v_S + 0.5 * sig * sig * S * S * v_SS
            + run + jumps)


def make_loss(spec, book, dgrid, market, gamma, eta, box, greeks=None):
    def loss(params, t, S, n):
        r = jax.vmap(lambda a, b, m: residual(params, spec, book, dgrid,
                                              market, gamma, eta, box,
                                              a, b, m, greeks=greeks)
                     )(t, S, n)
        return jnp.mean(r * r)
    return loss


def box_sampler(key, spec: PinnSpec, batch: int):
    k1, k2, k3 = jax.random.split(key, 3)
    t = jax.random.uniform(k1, (batch,)) * spec.T
    S = spec.S0 + spec.S_half * jax.random.uniform(k2, (batch,),
                                                   minval=-1.0, maxval=1.0)
    n = jax.random.uniform(k3, (batch, spec.n_max.shape[0]),
                           minval=-1.0, maxval=1.0) * spec.n_max[None, :]
    return t, S, n


def adam_train(params, loss_fn, spec, key, steps, batch=256, lr=2e-3,
               log_every=1500, log=print, sampler=box_sampler):
    import optax
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(params, state, key):
        key, ks = jax.random.split(key)
        args = sampler(ks, spec, batch)
        l, g = jax.value_and_grad(loss_fn)(params, *args)
        up, state = opt.update(g, state)
        return optax.apply_updates(params, up), state, key, l

    hist = []
    for i in range(steps):
        params, state, key, l = step(params, state, key)
        if i % log_every == 0 or i == steps - 1:
            hist.append((i, float(l)))
            log(f"  adam {i:5d}: loss {float(l):.4e}")
    return params, hist
