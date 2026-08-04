"""Heston vanilla pricing under Q via the COS method (Fang-Oosterlee 2008).

- Characteristic function in the Albrecher "little trap" form (no branch
  crossing for our parameter ranges; validated against MC in tests).
- Truncation range from cumulants c1, c2 of ln S_T obtained by AD of log(phi)
  at u = 0 (log phi(u) = i u c1 - u^2 c2 / 2 + ...), NOT from the hand-typed
  closed-form c2 monster — one fewer transcription-error surface.
- Greeks by AD: vega = d price / d sqrt(nu0) = 2 sqrt(nu0) * d price / d nu0;
  delta = d price / d S0.

Zero rates and dividends throughout (BBG setting).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

N_COS = 512
L_TRUNC = 14.0


def _log_cf(u, tau, S0, nu0, kappa, theta, xi, rho):
    """log characteristic function of X = ln S_T under Q, r = 0.

    Cancellation-free form: (beta - d)/xi^2 = -(iu + u^2)/(beta + d) exactly
    (from (beta-d)(beta+d) = -xi^2 (iu + u^2)), so the small-xi limit is
    numerically clean; the C log-term uses a guarded log(1+w) with a series
    for tiny |w|.  Little-trap branch (g = (beta-d)/(beta+d)).
    """
    u = u + 0.0j
    iuu = 1j * u + u * u
    beta = kappa - 1j * rho * xi * u
    d = jnp.sqrt(beta * beta + xi * xi * iuu)
    bpd = beta + d
    bmd_over_xi2 = -iuu / bpd            # (beta - d) / xi^2, exact
    g = -xi * xi * iuu / (bpd * bpd)     # (beta - d) / (beta + d), exact
    edt = jnp.exp(-d * tau)
    D = bmd_over_xi2 * (1.0 - edt) / (1.0 - g * edt)
    w = g * (1.0 - edt) / (1.0 - g)      # (1 - g edt)/(1 - g) = 1 + w
    logterm = jnp.where(jnp.abs(w) < 1e-4,
                        w - 0.5 * w * w + w * w * w / 3.0,
                        jnp.log(1.0 + w))
    C = kappa * theta * (bmd_over_xi2 * tau - 2.0 * logterm / (xi * xi))
    return C + D * nu0 + 1j * u * jnp.log(S0)


def _cumulants(tau, S0, nu0, kappa, theta, xi, rho):
    """c1 = E[ln S_T], c2 = Var[ln S_T] via AD of log phi at u = 0."""
    f = lambda u: _log_cf(u, tau, S0, nu0, kappa, theta, xi, rho)
    d1 = jax.jacfwd(f)(0.0)
    d2 = jax.jacfwd(jax.jacfwd(f))(0.0)
    c1 = jnp.imag(d1)
    c2 = -jnp.real(d2)
    return c1, c2


def _chi_psi(k, a, b, c, d):
    """COS payoff integrals of e^y cos(...) and cos(...) over [c, d] in [a, b]."""
    om = k * jnp.pi / (b - a)
    chi = (jnp.cos(om * (d - a)) * jnp.exp(d) - jnp.cos(om * (c - a)) * jnp.exp(c)
           + om * (jnp.sin(om * (d - a)) * jnp.exp(d)
                   - jnp.sin(om * (c - a)) * jnp.exp(c))) / (1.0 + om * om)
    psi_k = jnp.where(k > 0,
                      (jnp.sin(om * (d - a)) - jnp.sin(om * (c - a)))
                      / jnp.where(k > 0, om, 1.0),
                      d - c)
    return chi, psi_k


@partial(jax.jit, static_argnums=(7,))
def call_price(S0, nu0, K, tau, kappa, theta, xi_rho, is_call=True):
    """European option price under Q-Heston, COS method.

    xi_rho = (xi, rho) packed so the signature stays short.
    """
    xi, rho = xi_rho
    c1, c2 = _cumulants(tau, S0, nu0, kappa, theta, xi, rho)
    y0 = c1 - jnp.log(K)  # cumulants of y = ln(S_T / K)
    a = y0 - L_TRUNC * jnp.sqrt(jnp.abs(c2))
    b = y0 + L_TRUNC * jnp.sqrt(jnp.abs(c2))
    k = jnp.arange(N_COS, dtype=jnp.float64)
    u = k * jnp.pi / (b - a)
    phi_y = jnp.exp(_log_cf(u, tau, S0, nu0, kappa, theta, xi, rho)
                    - 1j * u * jnp.log(K))
    if is_call:
        chi, psi = _chi_psi(k, a, b, jnp.zeros_like(a), b)
        V = 2.0 / (b - a) * K * (chi - psi)
    else:
        chi, psi = _chi_psi(k, a, b, a, jnp.zeros_like(a))
        V = 2.0 / (b - a) * K * (psi - chi)
    terms = jnp.real(phi_y * jnp.exp(-1j * u * a)) * V
    return jnp.sum(terms) - 0.5 * terms[0]


def price_vega_delta(S0, nu0, K, tau, kappa, theta, xi, rho, is_call=True):
    """(price, vega = d/d sqrt(nu0), delta = d/dS0), scalar inputs."""
    f_nu = lambda v: call_price(S0, v, K, tau, kappa, theta, (xi, rho), is_call)
    f_S = lambda s: call_price(s, nu0, K, tau, kappa, theta, (xi, rho), is_call)
    price = f_nu(nu0)
    vega = 2.0 * jnp.sqrt(nu0) * jax.grad(f_nu)(nu0)
    delta = jax.grad(f_S)(S0)
    return price, vega, delta
