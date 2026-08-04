"""HJB residuals for the PINN, per collocation point, vmapped by the trainer.

Stage 1 (frozen Greeks, slab):
    R = dv/dt + a_P dv/dnu + 0.5 xi^2 nu d2v/dnu2
        + carry(nu) (w.n) - pen (w.n)^2
        + sum_ch z_ch 1{|w.(n - psi e_i)| <= Vbar} H_ch([v(n) - v(n')] / z_ch)

Stage 3 (state-dependent Greeks via a `greeks(t, S, nu) -> (N,) vega` callable,
S in the state, P-generator with the rho cross term):
    R = dv/dt + mu S dv/dS + a_P dv/dnu
        + 0.5 nu S^2 d2v/dSS + rho xi nu S d2v/dSnu + 0.5 xi^2 nu d2v/dnunu
        + carry(nu) (vega(t,S,nu) . q) - pen (vega . q)^2 + jumps
where q = n * z in contracts, so vega . q = sum_i n_i z_i vega_i(t,S,nu).

The stage-3 assembly on an S-independent v with frozen greeks reduces
*identically* to stage 1 (the dS terms vanish under AD) — this is the
consistency test that certifies the stage-3 residual before anything exact
exists at stage 3.

Second derivatives via forward-over-reverse (jax.jacfwd of jax.grad).
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from ...core.risk import MarketParams
from ...core.state import Slab
from ..reduced2d import Channels, _bc


def _jump_sum(vfun: Callable, n, z, psi, inst_idx, w, adm_set, intensity,
              delta_inf):
    """sum_ch z 1{adm} H([v(n) - v(n - psi e_i)]/z); vfun: n -> scalar.

    adm_set: a core.state admissibility object (Slab or Box); the post-trade
    indicator is adm_set.admissible(n', w) — for the Slab this is the
    aggregate-vega band on the (possibly state-dependent) per-trade vegas w,
    for the Box it is per-instrument position limits.
    """
    n_ch = z.shape[0]
    eye = jnp.eye(n.shape[0])
    shifts = -psi[:, None] * eye[inst_idx]            # (n_ch, N)
    n_tgt = n[None, :] + shifts
    adm = adm_set.admissible(n_tgt, w)
    v0 = vfun(n)
    v_tgt = jax.vmap(vfun)(n_tgt)                     # (n_ch,)
    p = (v0 - v_tgt) / z
    H = _bc(intensity, 0).H(p, delta_inf)
    return jnp.sum(jnp.where(adm, z * H, 0.0))


def residual_stage1(params, spec, value_fn, ch: Channels, market: MarketParams,
                    pen: float, Vbar: float, t, nu, n, delta_inf=-5.0,
                    adm_set=None):
    adm_set = Slab(Vbar) if adm_set is None else adm_set
    inst_idx = jnp.arange(ch.z.shape[0]) % (ch.z.shape[0] // 2)
    w = spec.w
    v_t = jax.grad(lambda tt: value_fn(params, spec, tt, nu, n))(t)
    v_nu = jax.grad(lambda nn: value_fn(params, spec, t, nn, n))(nu)
    v_nunu = jax.jacfwd(jax.grad(
        lambda nn: value_fn(params, spec, t, nn, n)))(nu)
    Vpi = n @ w
    run = market.carry_coef(nu) * Vpi - pen * Vpi**2
    jumps = _jump_sum(lambda m: value_fn(params, spec, t, nu, m),
                      n, ch.z, ch.psi, inst_idx, w, adm_set, ch.intensity,
                      delta_inf)
    return (v_t + market.a_P(nu) * v_nu + 0.5 * market.xi**2 * nu * v_nunu
            + run + jumps)


def residual_stage3(params, spec, value_fn, ch: Channels, market: MarketParams,
                    pen: float, Vbar: float, greeks: Callable, z_contracts,
                    t, S, nu, n, delta_inf=-5.0, adm_set=None):
    """value_fn signature: (params, spec, t, S, nu, n) -> scalar."""
    adm_set = Slab(Vbar) if adm_set is None else adm_set
    inst_idx = jnp.arange(ch.z.shape[0]) % (ch.z.shape[0] // 2)
    vega = greeks(t, S, nu)                           # (N,) per-contract vega
    w_state = vega * z_contracts                      # per-trade vega now
    v_t = jax.grad(lambda x: value_fn(params, spec, x, S, nu, n))(t)
    v_S = jax.grad(lambda x: value_fn(params, spec, t, x, nu, n))(S)
    v_nu = jax.grad(lambda x: value_fn(params, spec, t, S, x, n))(nu)
    v_SS = jax.jacfwd(jax.grad(
        lambda x: value_fn(params, spec, t, x, nu, n)))(S)
    v_nunu = jax.jacfwd(jax.grad(
        lambda x: value_fn(params, spec, t, S, x, n)))(nu)
    v_Snu = jax.jacfwd(
        lambda y: jax.grad(lambda x: value_fn(params, spec, t, x, y, n))(S))(nu)
    Vpi = n @ w_state
    run = market.carry_coef(nu) * Vpi - pen * Vpi**2
    jumps = _jump_sum(lambda m: value_fn(params, spec, t, S, nu, m),
                      n, ch.z, ch.psi, inst_idx, w_state, adm_set,
                      ch.intensity, delta_inf)
    gen = (v_t + market.mu * S * v_S + market.a_P(nu) * v_nu
           + 0.5 * nu * S**2 * v_SS + market.rho * market.xi * nu * S * v_Snu
           + 0.5 * market.xi**2 * nu * v_nunu)
    return gen + run + jumps
