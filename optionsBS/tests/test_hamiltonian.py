import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spotmm.core.hamiltonian import (ExponentialIntensity, LogisticIntensity,
                                    foc_residual)


def _logistic_grid():
    # BBG-like scales: lam up to 7560/yr, k = beta/vega in [~100, ~400]
    lam = jnp.array(7560.0)
    alpha = jnp.array(0.7)
    ks = jnp.array([100.0, 277.0, 365.0, 1000.0])
    ps = jnp.linspace(-0.5, 0.5, 41)  # EUR per contract, wide
    return lam, alpha, ks, ps


def test_logistic_newton_foc_residual():
    lam, alpha, ks, ps = _logistic_grid()
    for k in ks:
        inten = LogisticIntensity(lam=lam, alpha=alpha, k=k)
        d = inten.argmax_delta(ps)
        res = foc_residual(inten, ps, d)
        # scale-free check: residual relative to the local scale Lambda(d*)
        rel = jnp.abs(res) / inten(d)
        assert float(jnp.max(rel)) < 1e-10


def test_logistic_matches_brute_force():
    lam, alpha, ks, _ = _logistic_grid()
    inten = LogisticIntensity(lam=lam, alpha=alpha, k=ks[1])
    for p in [-0.3, -0.05, 0.0, 0.02, 0.2]:
        d_star = float(inten.argmax_delta(jnp.array(p)))
        grid = jnp.linspace(p - 0.2, p + 0.5, 200001)
        vals = inten(grid) * (grid - p)
        d_brute = float(grid[jnp.argmax(vals)])
        assert abs(d_star - d_brute) < 5e-6
        assert float(inten.H(jnp.array(p))) >= float(jnp.max(vals)) - 1e-9


def test_exponential_closed_form():
    inten = ExponentialIntensity(A=jnp.array(1000.0), k=jnp.array(50.0))
    ps = jnp.linspace(-0.2, 0.4, 31)
    d = inten.argmax_delta(ps)
    np.testing.assert_allclose(np.asarray(d), np.asarray(ps + 1.0 / 50.0), rtol=1e-14)
    H = inten.H(ps)
    H_closed = (1000.0 / 50.0) * jnp.exp(-50.0 * ps - 1.0)
    np.testing.assert_allclose(np.asarray(H), np.asarray(H_closed), rtol=1e-12)
    res = foc_residual(inten, ps, d)
    rel = jnp.abs(res) / inten(d)
    assert float(jnp.max(rel)) < 1e-10


def test_H_decreasing_convex_and_envelope():
    lam, alpha, ks, _ = _logistic_grid()
    inten = LogisticIntensity(lam=lam, alpha=alpha, k=ks[2])
    ps = jnp.linspace(-0.3, 0.3, 401)
    H = inten.H(ps)
    dH = jnp.diff(H)
    assert bool(jnp.all(dH < 0))            # H strictly decreasing
    assert bool(jnp.all(jnp.diff(dH) > -1e-9))  # convex (numerically)
    # envelope: H'(p) = -Lambda(delta*(p))
    Hp = jax.vmap(jax.grad(lambda p: inten.H(p)))(ps)
    lam_at_opt = inten(inten.argmax_delta(ps))
    np.testing.assert_allclose(np.asarray(Hp), -np.asarray(lam_at_opt), rtol=1e-8)


def test_delta_inf_binding():
    lam, alpha, ks, _ = _logistic_grid()
    inten = LogisticIntensity(lam=lam, alpha=alpha, k=ks[1])
    p = jnp.array(-0.4)
    d_free = float(inten.argmax_delta(p))
    d_inf = d_free + 0.05  # force binding
    d = float(inten.argmax_delta(p, delta_inf=d_inf))
    assert d == pytest.approx(d_inf)
    H = float(inten.H(p, delta_inf=d_inf))
    assert H == pytest.approx(float(inten(jnp.array(d_inf)) * (d_inf - p)))


def test_bbg_fill_prob_calibration():
    """alpha=0.7 -> 33.2% at mid; one vol point through mid -> 69.0%."""
    vega = 1.3
    inten = LogisticIntensity(lam=jnp.array(7560.0), alpha=jnp.array(0.7),
                              k=jnp.array(150.0 / vega))
    at_mid = float(inten(jnp.array(0.0)) / 7560.0)
    assert at_mid == pytest.approx(1.0 / (1.0 + np.exp(0.7)), rel=1e-12)
    one_vol_better = float(inten(jnp.array(-0.01 * vega)) / 7560.0)
    assert one_vol_better == pytest.approx(1.0 / (1.0 + np.exp(-0.8)), rel=1e-12)
