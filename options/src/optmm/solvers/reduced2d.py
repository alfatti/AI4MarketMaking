"""BBG reduced 2D solver: v(t, nu, Vpi) on [0,T] x [nu_lo, nu_hi] x [-Vbar, Vbar].

    0 = dv/dt + a_P dv/dnu + 0.5 xi^2 nu d2v/dnu2
        + Vpi * carry(nu) - pen * Vpi^2
        + sum_ch z_ch * 1{|Vpi - psi_ch w_ch| <= Vbar}
                 * H_ch( [v(Vpi) - v(Vpi - psi_ch w_ch)] / z_ch ),
    v(T, ., .) = 0,  psi(ask) = +1, psi(bid) = -1.

Explicit Euler backward: v(t - dt) = v(t) + dt * RHS(v(t)).
- nu: central differences (cell Peclet << 1 for BBG params), Neumann BCs via
  ghost reflection (first derivative 0 at edges; second = 2(v1 - v0)/dnu^2).
- Vpi jumps land off-grid: linear interpolation with precomputed indices and
  weights per channel; the grid spans [-Vbar, Vbar] exactly, so out-of-grid
  targets are exactly the inadmissible ones (masked).
- Stability numbers (diffusion CFL, dt * sum_ch Lambda_ch(delta*)) are
  computed and stored on the solution for the tests to assert on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

from ..core.hamiltonian import LogisticIntensity


def _bc(intensity: "LogisticIntensity", extra: int) -> "LogisticIntensity":
    """Reshape per-channel params to broadcast over `extra` trailing dims."""
    sh = lambda a: jnp.reshape(a, a.shape + (1,) * extra)
    return LogisticIntensity(lam=sh(intensity.lam), alpha=sh(intensity.alpha),
                             k=sh(intensity.k))
from ..core.risk import MarketParams, penalty_coef
from ..instruments.book import FrozenBook


@dataclass(frozen=True)
class Channels:
    """Per-channel (instrument x side) arrays, ask first then bid."""

    z: jnp.ndarray
    w: jnp.ndarray
    psi: jnp.ndarray
    intensity: LogisticIntensity

    @classmethod
    def from_book(cls, book: FrozenBook) -> "Channels":
        rep = lambda x: jnp.concatenate([x, x])
        psi = jnp.concatenate([jnp.ones_like(book.z), -jnp.ones_like(book.z)])
        inten = LogisticIntensity(lam=rep(book.lam), alpha=rep(book.alpha),
                                  k=rep(book.k))
        return cls(z=rep(book.z), w=rep(book.w), psi=psi, intensity=inten)


@dataclass
class Reduced2DSolution:
    nu_grid: jnp.ndarray
    V_grid: jnp.ndarray
    t_grid: jnp.ndarray
    v0: jnp.ndarray                      # (Nnu, NV) at t = 0
    v_hist: Optional[jnp.ndarray]        # (n_stored, Nnu, NV) incl. t=0 & T
    hist_t: Optional[jnp.ndarray]
    channels: Channels
    diffusion_cfl: float
    monotonicity_number: float           # max over state of dt*sum Lambda(d*)

    def quotes(self, v: Optional[jnp.ndarray] = None, delta_inf=-5.0):
        """Optimal per-channel quotes on the grid: (n_ch, Nnu, NV)."""
        v = self.v0 if v is None else v
        ch = self.channels
        tgt = self.V_grid[None, :] - ch.psi[:, None] * ch.w[:, None]
        Vmin, Vmax = self.V_grid[0], self.V_grid[-1]
        dV = self.V_grid[1] - self.V_grid[0]
        adm = (tgt >= Vmin - 1e-9 * Vmax) & (tgt <= Vmax + 1e-9 * Vmax)
        f = jnp.clip((tgt - Vmin) / dV, 0.0, self.V_grid.shape[0] - 1.001)
        i0 = jnp.floor(f).astype(jnp.int32)
        fr = f - i0
        v_sh = (1 - fr)[None] * v[:, i0] + fr[None] * v[:, i0 + 1]  # (Nnu,nch,NV)
        p = (v[:, None, :] - v_sh) / ch.z[None, :, None]
        d = _bc(ch.intensity, 2).argmax_delta(jnp.moveaxis(p, 1, 0), delta_inf)
        return jnp.where(adm[:, None, :], d, jnp.nan)


def solve_reduced2d(book: FrozenBook, market: MarketParams, gamma: float,
                    Vbar: float, T: float, c: float = 0.0,
                    nt: int = 720, n_nu: int = 31, n_V: int = 201,
                    nu_lo: float = 0.0144, nu_hi: float = 0.0324,
                    delta_inf: float = -5.0,
                    store_stride: Optional[int] = None,
                    rho_override: Optional[float] = None) -> Reduced2DSolution:
    ch = Channels.from_book(book)
    rho = market.rho if rho_override is None else rho_override
    pen = penalty_coef(gamma, market.xi, rho, c)

    nu = jnp.linspace(nu_lo, nu_hi, n_nu)
    V = jnp.linspace(-Vbar, Vbar, n_V)
    dnu = float(nu[1] - nu[0])
    dV = float(V[1] - V[0])
    dt = T / nt
    a_P = market.a_P(nu)
    carry = market.carry_coef(nu)
    diff_c = 0.5 * market.xi**2 * nu

    # Precomputed interpolation: target Vpi' = V - psi_ch * w_ch
    tgt = V[None, :] - ch.psi[:, None] * ch.w[:, None]            # (nch, NV)
    adm = (tgt >= -Vbar - 1e-9 * Vbar) & (tgt <= Vbar + 1e-9 * Vbar)
    f = jnp.clip((tgt + Vbar) / dV, 0.0, n_V - 1.0 - 1e-9)
    i0 = jnp.floor(f).astype(jnp.int32)
    fr = f - i0

    run_src = carry[:, None] * V[None, :] - pen * V[None, :] ** 2  # (Nnu, NV)

    def rhs(v):
        # nu derivatives with Neumann ghosts
        d1 = jnp.zeros_like(v)
        d1 = d1.at[1:-1].set((v[2:] - v[:-2]) / (2 * dnu))
        d2 = jnp.zeros_like(v)
        d2 = d2.at[1:-1].set((v[2:] - 2 * v[1:-1] + v[:-2]) / dnu**2)
        d2 = d2.at[0].set(2 * (v[1] - v[0]) / dnu**2)
        d2 = d2.at[-1].set(2 * (v[-2] - v[-1]) / dnu**2)
        # jump terms
        v_sh = (1 - fr)[None] * v[:, i0] + fr[None] * v[:, i0 + 1]  # (Nnu,nch,NV)
        p = (v[:, None, :] - v_sh) / ch.z[None, :, None]
        Hs = _bc(ch.intensity, 2).H(jnp.moveaxis(p, 1, 0), delta_inf)  # (nch,Nnu,NV)
        jump = jnp.sum(jnp.where(adm[:, None, :], ch.z[:, None, None] * Hs, 0.0),
                       axis=0)
        return a_P[:, None] * d1 + diff_c[:, None] * d2 + run_src + jump

    stride = store_stride if store_stride else nt + 1

    def step(carry_, k):
        v = carry_
        v_new = v + dt * rhs(v)
        out = jnp.where((k % stride) == 0, 1, 0)
        return v_new, (out, v_new)

    v_T = jnp.zeros((n_nu, n_V))
    if store_stride:
        vs = [v_T]
        v = v_T
        step_j = jax.jit(lambda v: v + dt * rhs(v))
        for k in range(nt):
            v = step_j(v)
            if (k + 1) % store_stride == 0 or (k + 1) == nt:
                vs.append(v)
        v0 = v
        v_hist = jnp.stack(vs[::-1])     # forward-time order, [t=0 ... t=T]
        hist_t = jnp.array([T - min(k * store_stride, nt) * dt
                            for k in range(len(vs))][::-1])
    else:
        def body(v, _):
            return v + dt * rhs(v), None
        v0, _ = jax.lax.scan(body, v_T, None, length=nt)
        v_hist, hist_t = None, None

    # diagnostics at t=0
    v_sh = (1 - fr)[None] * v0[:, i0] + fr[None] * v0[:, i0 + 1]
    p = jnp.moveaxis((v0[:, None, :] - v_sh) / ch.z[None, :, None], 1, 0)
    inten3 = _bc(ch.intensity, 2)
    lam_star = inten3(inten3.argmax_delta(p, delta_inf))
    lam_star = jnp.where(adm[:, None, :], lam_star, 0.0)
    mono = float(dt * jnp.max(jnp.sum(lam_star, axis=0)))
    cfl = float(dt * jnp.max(diff_c) / dnu**2)

    return Reduced2DSolution(nu_grid=nu, V_grid=V, t_grid=jnp.linspace(0, T, nt + 1),
                             v0=v0, v_hist=v_hist, hist_t=hist_t, channels=ch,
                             diffusion_cfl=cfl, monotonicity_number=mono)
