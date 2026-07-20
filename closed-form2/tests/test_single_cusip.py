"""Tests: analytic cross-checks against the paper's closed forms (d = 1)."""

import numpy as np
import pytest

from mmquote import (
    ExponentialIntensity,
    HamiltonianSide,
    ModelConfig,
    ProxyCoefficients,
    build_single_cusip_engine,
    calibrate_gamma_to_spread,
    estimate_sigma,
    fit_exponential_intensity,
    solve_riccati,
)

A_, K_, SIG, GAM, Z = 2.0, 5.0, 0.3, 0.1, 1.0


def make_engine(gamma=GAM, **kw):
    cfg = ModelConfig(gamma=gamma, z=Z, T=10.0, xi=0.0, **kw)
    lam = ExponentialIntensity(A=A_, k=K_)
    return build_single_cusip_engine(lam, lam, SIG, cfg)


# ---------------------------------------------------------------- intensity

def test_exponential_wellposedness_is_exactly_one():
    lam = ExponentialIntensity(A=A_, k=K_)
    lam.validate()
    assert lam.wellposedness_sup(np.linspace(-5, 5, 11)) == 1.0


def test_inverse_roundtrip():
    lam = ExponentialIntensity(A=A_, k=K_)
    d = np.linspace(-1, 3, 7)
    np.testing.assert_allclose(lam.inverse(lam.value(d)), d, rtol=1e-12)


# ---------------------------------------------------------------- hamiltonian

def test_model_b_hamiltonian_closed_form():
    """xi=0: H(p) = (A/k) e^{-1} e^{-kp}; delta*(p) = p + 1/k."""
    ham = HamiltonianSide(ExponentialIntensity(A=A_, k=K_), z=Z, xi=0.0)
    p = np.linspace(-0.5, 0.5, 11)
    np.testing.assert_allclose(ham.H(p), (A_ / K_) * np.exp(-1.0) * np.exp(-K_ * p), rtol=1e-12)
    np.testing.assert_allclose(ham.delta_star(p), p + 1.0 / K_, rtol=1e-10, atol=1e-12)


def test_alphas_signs_and_ratios():
    ham = HamiltonianSide(ExponentialIntensity(A=A_, k=K_), z=Z, xi=0.0)
    a0, a1, a2 = ham.alphas()
    assert a0 > 0 and a1 < 0 and a2 > 0
    np.testing.assert_allclose(a1, -K_ * a0, rtol=1e-12)   # H' = -k H
    np.testing.assert_allclose(a2, K_**2 * a0, rtol=1e-12)  # H'' = k^2 H


def test_numeric_fallback_matches_closed_form():
    """Force the numeric sup path and compare to the exponential closed form."""

    class OpaqueExp(ExponentialIntensity):
        pass

    ham_cf = HamiltonianSide(ExponentialIntensity(A=A_, k=K_), z=Z, xi=0.0)
    ham_num = HamiltonianSide.__new__(HamiltonianSide)
    object.__setattr__(ham_num, "intensity", ExponentialIntensity(A=A_, k=K_))
    object.__setattr__(ham_num, "z", Z)
    object.__setattr__(ham_num, "xi", 0.0)
    p = np.array([-0.2, 0.0, 0.3])
    np.testing.assert_allclose(ham_num._H_numeric(p), ham_cf.H(p), rtol=1e-6)


# ---------------------------------------------------------------- riccati

def test_scalar_riccati_matches_tanh_formula():
    """d=1: A(t) = (1/2) sqrt(gamma sigma^2 / (Dp)) * ... = (Ahat/(2Dp)) tanh(Ahat (T-t))."""
    ham = HamiltonianSide(ExponentialIntensity(A=A_, k=K_), z=Z, xi=0.0)
    a0, a1, a2 = ham.alphas()
    coeffs = ProxyCoefficients(
        alpha0_b=np.array([a0]), alpha1_b=np.array([a1]), alpha2_b=np.array([a2]),
        alpha0_a=np.array([a0]), alpha1_a=np.array([a1]), alpha2_a=np.array([a2]),
        z=np.array([Z]),
    )
    T = 10.0
    sol = solve_riccati(coeffs, np.array([[SIG**2]]), gamma=GAM, T=T)
    Dp = 2 * a2 * Z
    Ahat = np.sqrt(GAM) * np.sqrt(Dp) * SIG
    expected = 0.5 * (Ahat / Dp) * np.tanh(Ahat * (T - sol.t_grid))
    np.testing.assert_allclose(sol.A_path[:, 0, 0], expected, rtol=1e-10)


def test_ergodic_limit_formula():
    """A_erg = (1/2) sqrt(gamma) * sigma / sqrt((a2_b + a2_a) z)."""
    engine = make_engine()
    sol = engine.solution
    ham = engine.ham_b
    _, _, a2 = ham.alphas()
    expected = 0.5 * np.sqrt(GAM) * SIG / np.sqrt(2 * a2 * Z)
    np.testing.assert_allclose(sol.A_erg[0, 0], expected, rtol=1e-12)
    # finite-T A(0) approaches the ergodic value: rerun with T large enough
    # that tanh(Ahat*T) has saturated (Ahat ~ 0.26 here, so T = 60 suffices)
    cfg_long = ModelConfig(gamma=GAM, z=Z, T=60.0, xi=0.0)
    lam = ExponentialIntensity(A=A_, k=K_)
    sol_long = build_single_cusip_engine(lam, lam, SIG, cfg_long).solution
    np.testing.assert_allclose(sol_long.A_path[0, 0, 0], sol_long.A_erg[0, 0], rtol=1e-4)


def test_symmetric_book_has_zero_B():
    engine = make_engine()
    np.testing.assert_allclose(engine.solution.B_erg, 0.0, atol=1e-12)
    np.testing.assert_allclose(engine.solution.B_path, 0.0, atol=1e-10)


def test_asymmetric_B_path_converges_to_ergodic():
    """Regression for the backward-integration sign: on an asymmetric book,
    finite-T B(0) must converge to B_erg (would diverge with the wrong sign)."""
    lam_b = ExponentialIntensity(A=1.5, k=6.0)
    lam_a = ExponentialIntensity(A=2.5, k=4.5)
    cfg = ModelConfig(gamma=GAM, z=Z, T=80.0, xi=0.0, n_steps=4000)
    engine = build_single_cusip_engine(lam_b, lam_a, SIG, cfg)
    sol = engine.solution
    assert np.all(np.isfinite(sol.B_path))
    np.testing.assert_allclose(sol.B_path[0], sol.B_erg, rtol=2e-3)


# ---------------------------------------------------------------- quotes

def test_quotes_lean_against_inventory():
    engine = make_engine()
    q_long, q_flat, q_short = np.array([3.0]), np.array([0.0]), np.array([-3.0])
    qt_l, qt_f, qt_s = engine.quote(q_long), engine.quote(q_flat), engine.quote(q_short)
    # long inventory: back off the bid (wider delta_b), tighten the ask
    assert qt_l.delta_b > qt_f.delta_b > qt_s.delta_b
    assert qt_l.delta_a < qt_f.delta_a < qt_s.delta_a
    # symmetric book at flat inventory: symmetric quotes, zero skew
    np.testing.assert_allclose(qt_f.delta_b, qt_f.delta_a, rtol=1e-10)
    np.testing.assert_allclose(engine.skew(q_flat), 0.0, atol=1e-10)


def test_marginal_value_affine_in_q():
    engine = make_engine()
    qs = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    p_b = np.array([engine.marginal_values(np.array([q]))[0] for q in qs])
    slopes = np.diff(p_b) / np.diff(qs)
    np.testing.assert_allclose(slopes, slopes[0], rtol=1e-10)


def test_inventory_limit_switches_side_off():
    engine = make_engine(Q_limit=2.0)
    qt = engine.quote(np.array([2.0]))
    assert qt.delta_b is None and qt.delta_a is not None
    qt = engine.quote(np.array([-2.0]))
    assert qt.delta_a is None and qt.delta_b is not None


def test_wider_quotes_with_more_risk_aversion():
    s_lo = make_engine(gamma=0.05).quote(np.zeros(1))
    s_hi = make_engine(gamma=0.5).quote(np.zeros(1))
    assert s_hi.delta_b + s_hi.delta_a > s_lo.delta_b + s_lo.delta_a


# ---------------------------------------------------------------- calibration

def test_intensity_mle_recovers_truth():
    rng = np.random.default_rng(0)
    n, tau = 50_000, 0.05  # ~1,600 expected fills: enough to identify (A, k)
    deltas = rng.uniform(0.0, 0.6, n)
    lam_true = ExponentialIntensity(A=A_, k=K_)
    fills = rng.poisson(tau * lam_true.value(deltas))
    fit = fit_exponential_intensity(deltas, fills, tau)
    np.testing.assert_allclose(fit.model.A, A_, rtol=0.1)
    np.testing.assert_allclose(fit.model.k, K_, rtol=0.1)


def test_sigma_estimator():
    rng = np.random.default_rng(1)
    dt = 1.0 / 252
    prices = 100 + np.cumsum(SIG * np.sqrt(dt) * rng.standard_normal(50_000))
    np.testing.assert_allclose(estimate_sigma(prices, dt), SIG, rtol=0.02)


def test_gamma_calibration_hits_target_spread():
    lam = ExponentialIntensity(A=A_, k=K_)

    def make(gamma):
        return build_single_cusip_engine(
            lam, lam, SIG, ModelConfig(gamma=gamma, z=Z, T=10.0)
        )

    target = 0.55
    gam = calibrate_gamma_to_spread(make, target)
    qt = make(gam).quote(np.zeros(1))
    np.testing.assert_allclose(qt.delta_b + qt.delta_a, target, rtol=1e-8)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
