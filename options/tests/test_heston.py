import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from optmm.core.risk import BBG_MARKET as M
from optmm.instruments import heston


def bs_call(S, K, sigma, tau):
    d1 = (np.log(S / K) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def test_bs_limit():
    """xi -> 0, theta = nu0: deterministic variance => exact Black-Scholes."""
    for K in [8.0, 10.0, 12.0]:
        for tau in [0.5, 1.0, 2.0]:
            p = float(heston.call_price(10.0, 0.0225, K, tau, 3.0, 0.0225,
                                        (1e-8, -0.5)))
            assert p == pytest.approx(bs_call(10.0, K, 0.15, tau), abs=2e-8)


def test_put_call_parity():
    for K in [8.0, 10.0, 12.0]:
        c = float(heston.call_price(M.S0, M.nu0, K, 1.5, M.kappa_Q, M.theta_Q,
                                    (M.xi, M.rho), True))
        p = float(heston.call_price(M.S0, M.nu0, K, 1.5, M.kappa_Q, M.theta_Q,
                                    (M.xi, M.rho), False))
        assert c - p == pytest.approx(M.S0 - K, abs=1e-9)


def test_truncation_insensitivity():
    import optmm.instruments.heston as h
    p14 = float(h.call_price(M.S0, M.nu0, 11.0, 2.0, M.kappa_Q, M.theta_Q,
                             (M.xi, M.rho)))
    old_L, old_N = h.L_TRUNC, h.N_COS
    try:
        h.L_TRUNC, h.N_COS = 10.0, 1024
        h.call_price.clear_cache()
        p10 = float(h.call_price(M.S0, M.nu0, 11.0, 2.0, M.kappa_Q, M.theta_Q,
                                 (M.xi, M.rho)))
    finally:
        h.L_TRUNC, h.N_COS = old_L, old_N
        h.call_price.clear_cache()
    assert p14 == pytest.approx(p10, abs=5e-9)


def test_vega_ad_vs_fd():
    _, vega, delta = heston.price_vega_delta(M.S0, M.nu0, 10.0, 1.0,
                                             M.kappa_Q, M.theta_Q, M.xi, M.rho)
    eps = 1e-6
    up = float(heston.call_price(M.S0, M.nu0 + eps, 10.0, 1.0, M.kappa_Q,
                                 M.theta_Q, (M.xi, M.rho)))
    dn = float(heston.call_price(M.S0, M.nu0 - eps, 10.0, 1.0, M.kappa_Q,
                                 M.theta_Q, (M.xi, M.rho)))
    vega_fd = 2.0 * np.sqrt(M.nu0) * (up - dn) / (2 * eps)
    assert float(vega) == pytest.approx(vega_fd, rel=1e-6)
    assert 0.0 < float(delta) < 1.0


def test_monotone_in_strike():
    Ks = jnp.linspace(7.0, 13.0, 25)
    ps = jnp.array([heston.call_price(M.S0, M.nu0, float(K), 1.0, M.kappa_Q,
                                      M.theta_Q, (M.xi, M.rho)) for K in Ks])
    assert bool(jnp.all(jnp.diff(ps) < 0))


@pytest.mark.slow
def test_mc_agreement():
    """Full-truncation Euler MC under Q vs COS, one representative option."""
    n_paths, n_steps, tau = 200_000, 400, 1.0
    dt = tau / n_steps
    sq1mr2 = np.sqrt(1 - M.rho**2)

    def step(carry, key):
        lnS, nu = carry
        k1, k2 = jax.random.split(key)
        zn = jax.random.normal(k1, (n_paths,))
        zS = M.rho * zn + sq1mr2 * jax.random.normal(k2, (n_paths,))
        nup = jnp.maximum(nu, 0.0)
        lnS = lnS + (-0.5 * nup) * dt + jnp.sqrt(nup * dt) * zS
        nu = nu + M.kappa_Q * (M.theta_Q - nup) * dt              + M.xi * jnp.sqrt(nup * dt) * zn
        return (lnS, nu), None

    keys = jax.random.split(jax.random.PRNGKey(0), n_steps)
    (lnS, _), _ = jax.lax.scan(step, (jnp.full(n_paths, np.log(M.S0)),
                                      jnp.full(n_paths, M.nu0)), keys)
    payoff = jnp.maximum(jnp.exp(lnS) - 10.0, 0.0)
    mc, se = float(jnp.mean(payoff)), float(jnp.std(payoff) / np.sqrt(n_paths))
    cos = float(heston.call_price(M.S0, M.nu0, 10.0, tau, M.kappa_Q, M.theta_Q,
                                  (M.xi, M.rho)))
    assert abs(mc - cos) < 4 * se + 2e-3  # MC error + O(dt) bias allowance
