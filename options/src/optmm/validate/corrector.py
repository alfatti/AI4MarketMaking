"""Dynkin-identity machinery, pre-validated where exact references exist.

For a candidate value table v~ and the policy pi~ its quotes induce, the
policy value satisfies (Dynkin / Feynman-Kac for the controlled jump
diffusion):

    V^{pi~}(0, x0) = v~(0, x0) + E[ int_0^T R~(t, X_t) dt ],

where R~ is the *linear* (fixed-policy) residual of v~:

    R~ = dv~/dt + a_P dv~/dnu + 0.5 xi^2 nu d2v~/dnu2
         + carry Vpi - pen Vpi^2
         + sum_ch z Lambda(delta~) (delta~ - [v~ - v~(shift)]/z) 1{adm}.

Three independent evaluations of V^{pi~} must agree (the triangle test):
  (1) direct simulation of the reward under pi~;
  (2) the linear policy-evaluation PDE (this module, `linear_policy_eval`);
  (3) the corrector estimate v~(0) + mean(int R~ along simulated paths).
Validated at stage 1 on the atom chain; deployed at stage 3 where nothing
exact exists.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core.risk import MarketParams, penalty_coef
from ..solvers.reduced2d import Reduced2DSolution


def quotes_tables(sol: Reduced2DSolution, v_hist, delta_inf=-5.0):
    """Per-slice quote tables for an arbitrary value history."""
    return jax.vmap(lambda v: sol.quotes(v, delta_inf))(v_hist)


def _interp_setup(sol: Reduced2DSolution):
    ch = sol.channels
    V = sol.V_grid
    n_V = V.shape[0]
    Vbar = float(V[-1])
    dV = float(V[1] - V[0])
    tgt = V[None, :] - ch.psi[:, None] * ch.w[:, None]
    adm = (tgt >= -Vbar - 1e-9 * Vbar) & (tgt <= Vbar + 1e-9 * Vbar)
    f = jnp.clip((tgt + Vbar) / dV, 0.0, n_V - 1.0 - 1e-9)
    i0 = jnp.floor(f).astype(jnp.int32)
    return adm, i0, f - i0


def _spatial(sol: Reduced2DSolution, market: MarketParams, pen, v, tab,
             adm, i0, fr):
    """a_P dv/dnu + diff + run + fixed-policy jump terms, one slice."""
    ch = sol.channels
    nu = sol.nu_grid
    dnu = nu[1] - nu[0]
    d1 = jnp.zeros_like(v)
    d1 = d1.at[1:-1].set((v[2:] - v[:-2]) / (2 * dnu))
    d2 = jnp.zeros_like(v)
    d2 = d2.at[1:-1].set((v[2:] - 2 * v[1:-1] + v[:-2]) / dnu**2)
    d2 = d2.at[0].set(2 * (v[1] - v[0]) / dnu**2)
    d2 = d2.at[-1].set(2 * (v[-2] - v[-1]) / dnu**2)
    V = sol.V_grid
    run = market.carry_coef(nu)[:, None] * V[None, :] - pen * V[None, :] ** 2
    v_sh = (1 - fr)[None] * v[:, i0] + fr[None] * v[:, i0 + 1]
    p = jnp.moveaxis((v[:, None, :] - v_sh) / ch.z[None, :, None], 1, 0)
    d = tab                                       # (n_ch, n_nu, n_V), NaN=off
    ok = jnp.isfinite(d) & adm[:, None, :]
    d_safe = jnp.where(ok, d, 0.0)
    lam = sol.channels.intensity.lam[:, None, None] * jax.nn.sigmoid(
        -(sol.channels.intensity.alpha[:, None, None]
          + sol.channels.intensity.k[:, None, None] * d_safe))
    flow = jnp.where(ok, sol.channels.z[:, None, None] * lam * (d_safe - p),
                     0.0)
    return (market.a_P(nu)[:, None] * d1
            + 0.5 * market.xi**2 * nu[:, None] * d2 + run
            + jnp.sum(flow, axis=0))


def linear_policy_eval(sol: Reduced2DSolution, tabs, market: MarketParams,
                       gamma: float, T: float, c: float = 0.0):
    """Backward Euler evaluation of the fixed policy in `tabs`.

    tabs[k] is the policy on [t_k, t_{k+1}); returns u with u[k] ~ V^pi(t_k).
    """
    pen = penalty_coef(gamma, market.xi, market.rho, c)
    adm, i0, fr = _interp_setup(sol)
    nt = tabs.shape[0] - 1
    dt = T / nt

    def body(u, k):
        kk = nt - 1 - k
        u_new = u + dt * _spatial(sol, market, pen, u, tabs[kk], adm, i0, fr)
        return u_new, u_new

    u_T = jnp.zeros_like(sol.v0)
    u0, hist = jax.lax.scan(body, u_T, jnp.arange(nt))
    u_hist = jnp.concatenate([hist[::-1], u_T[None]], axis=0)
    return u0, u_hist


def residual_tables(sol: Reduced2DSolution, v_hist, tabs,
                    market: MarketParams, gamma: float, T: float,
                    c: float = 0.0):
    """Linear residual R~ of v_hist under the policy in tabs: (nt, nnu, nV).

    R~[k] uses the forward time difference (v[k+1]-v[k])/dt and spatial terms
    on slice k+1 — aligned with the backward Euler update, so for the
    solver's own optimal history (with its own quote tables) R~ vanishes to
    machine precision, and for any other v~ it measures true suboptimality
    plus discretization.
    """
    pen = penalty_coef(gamma, market.xi, market.rho, c)
    adm, i0, fr = _interp_setup(sol)
    nt = v_hist.shape[0] - 1
    dt = T / nt

    def one(k):
        vt = (v_hist[k + 1] - v_hist[k]) / dt
        return vt + _spatial(sol, market, pen, v_hist[k + 1], tabs[k + 1],
                             adm, i0, fr)

    return jax.vmap(one)(jnp.arange(nt))
