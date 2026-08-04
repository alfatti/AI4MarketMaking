"""Stage-3 CRN simulator and the Dynkin closure test.

State per path: (S, nu, n) with n on the integer box; quotes are computed
*greedily from the PINN* each step (Lambert-W at p = value differences),
which is exactly the policy the network's value implies.  Because the greedy
policy attains the Hamiltonian sup, the fixed-policy (linear) residual of
v_theta equals its HJB residual, so the Dynkin identity closes with the
already-implemented `residual_stage3`:

    E[sum reward] = v_theta(0, x0) + E[ int_0^T R(t, S_t, nu_t, n_t) dt ].

Both sides are estimated on the same paths (CRN); their agreement — with no
exact reference anywhere — is the stage-3 validation instrument, itself
pre-validated at stage 1 (tests/test_corrector.py).  The residual integral
is sampled every `res_stride` steps (Riemann, documented bias O(stride*dt)).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..core.risk import MarketParams, penalty_coef
from ..core.state import Box
from ..solvers.pinn.model3 import Pinn3Spec, value3
from ..solvers.pinn.residual import residual_stage3
from ..solvers.reduced2d import Channels, _bc


@dataclass
class Stage3SimResult:
    reward: jnp.ndarray            # (n_paths,)
    corr: jnp.ndarray              # (n_paths,) int R dt (Dynkin term)
    S_final: jnp.ndarray
    Vpi_path_rms: jnp.ndarray      # (n_paths,) rms live portfolio vega
    fills: jnp.ndarray


def simulate_stage3(params, spec: Pinn3Spec, ch: Channels, vega_fn,
                    z_contracts, market: MarketParams, gamma: float,
                    box: Box, T: float, n_steps: int, n_paths: int = 1500,
                    res_stride: int = 4, seed: int = 0, c: float = 0.0,
                    delta_inf: float = -5.0):
    N = spec.w_ref.shape[0]
    n_ch = 2 * N
    dt = T / n_steps
    pen = penalty_coef(gamma, market.xi, market.rho, c)
    rho = market.rho
    sq = np.sqrt(1.0 - rho * rho)
    psi = jnp.concatenate([jnp.ones(N), -jnp.ones(N)])
    inst = jnp.arange(n_ch) % N
    eye = jnp.eye(N)
    shifts = -psi[:, None] * eye[inst]                 # (n_ch, N)
    nbar = jnp.asarray(box.nbar, dtype=jnp.float64)

    v_batch = jax.vmap(lambda t, S, nu, n: value3(params, spec, t, S, nu, n))

    res_pt = partial(residual_stage3, params, spec, value3, ch, market, pen,
                     spec.Vbar, vega_fn, z_contracts, delta_inf=delta_inf,
                     adm_set=box)
    res_batch = jax.vmap(lambda t, S, nu, n: res_pt(t=t, S=S, nu=nu, n=n))

    def step(carry, k):
        S, nu, n, reward, corr, fills, key = carry
        t = k * dt
        key, k1, k2, k3 = jax.random.split(key, 4)
        tt = jnp.full((n_paths,), t)
        # greedy quotes from value differences
        v0 = v_batch(tt, S, nu, n)                                  # (P,)
        n_tgt = n[:, None, :] + shifts[None, :, :]                  # (P,nch,N)
        v_tgt = jax.vmap(lambda t_, S_, nu_, m: jax.vmap(
            lambda mm: value3(params, spec, t_, S_, nu_, mm))(m)
        )(tt, S, nu, n_tgt)                                         # (P,nch)
        p = (v0[:, None] - v_tgt) / ch.z[None, :]
        delta = _bc(ch.intensity, 1).argmax_delta(p.T, delta_inf).T  # (P,nch)
        adm = jnp.all(jnp.abs(n_tgt) <= nbar[None, None, :] + 1e-9, axis=-1)
        lam = _bc(ch.intensity, 1)(delta.T).T
        u = jax.random.uniform(k1, (n_paths, n_ch))
        fill = (u < lam * dt) & adm
        # dt-integrand at pre-trade state (live Greeks)
        vega = jax.vmap(lambda t_, S_, nu_: vega_fn(t_, S_, nu_))(tt, S, nu)
        Vpi = jnp.sum(n * z_contracts[None, :] * vega, axis=1)
        reward = reward + (market.carry_coef(nu) * Vpi - pen * Vpi**2) * dt
        reward = reward + jnp.sum(
            jnp.where(fill, ch.z[None, :] * delta, 0.0), axis=1)
        # Dynkin term, strided
        corr = jax.lax.cond(
            (k % res_stride) == 0,
            lambda c_: c_ + res_batch(tt, S, nu, n) * (dt * res_stride),
            lambda c_: c_, corr)
        # state updates
        n = n + jnp.sum(jnp.where(fill[:, :, None], shifts[None], 0.0), axis=1)
        zn = jax.random.normal(k2, (n_paths,))
        zS = rho * zn + sq * jax.random.normal(k3, (n_paths,))
        nup = jnp.maximum(nu, 0.0)
        S = S * jnp.exp(-0.5 * nup * dt + jnp.sqrt(nup * dt) * zS)
        nu = nu + market.a_P(nup) * dt + market.xi * jnp.sqrt(nup * dt) * zn
        fills = fills + jnp.sum(fill, axis=1).astype(jnp.int32)
        return (S, nu, n, reward, corr + 0.0,
                fills, key), Vpi**2

    S0 = jnp.full((n_paths,), market.S0)
    nu0 = jnp.full((n_paths,), market.nu0)
    n0 = jnp.zeros((n_paths, N))
    key = jax.random.PRNGKey(seed)
    (S, nu, n, reward, corr, fills, _), Vpi2 = jax.lax.scan(
        step, (S0, nu0, n0, jnp.zeros(n_paths), jnp.zeros(n_paths),
               jnp.zeros(n_paths, dtype=jnp.int32), key),
        jnp.arange(n_steps))
    return Stage3SimResult(reward=reward, corr=corr, S_final=S,
                           Vpi_path_rms=jnp.sqrt(jnp.mean(Vpi2, axis=0)),
                           fills=fills)
