import jax.numpy as jnp
import numpy as np
import pytest

from optmm.core.risk import BBG_MARKET, MarketParams
from optmm.instruments.book import build_bbg_book
from optmm.solvers.reduced2d import solve_reduced2d

GAMMA, VBAR, T = 1e-3, 1e7, 0.0012


@pytest.fixture(scope="module")
def book():
    return build_bbg_book(BBG_MARKET)


SYM = MarketParams(S0=10.0, nu0=0.0225, kappa_P=3.0, theta_P=0.0225,
                   kappa_Q=3.0, theta_Q=0.0225, xi=0.2, rho=-0.5)


def test_stability_numbers(book):
    sol = solve_reduced2d(book, BBG_MARKET, GAMMA, VBAR, T,
                          nt=240, n_nu=21, n_V=101)
    assert sol.diffusion_cfl < 0.5
    assert sol.monotonicity_number < 1.0


def test_grid_convergence(book):
    ref = solve_reduced2d(book, BBG_MARKET, GAMMA, VBAR, T,
                          nt=960, n_nu=21, n_V=401)
    lo = solve_reduced2d(book, BBG_MARKET, GAMMA, VBAR, T,
                         nt=240, n_nu=21, n_V=101)
    mid = solve_reduced2d(book, BBG_MARKET, GAMMA, VBAR, T,
                          nt=480, n_nu=21, n_V=201)
    v_ref = float(ref.v0[10, 200])
    e_lo = abs(float(lo.v0[10, 50]) - v_ref) / abs(v_ref)
    e_mid = abs(float(mid.v0[10, 100]) - v_ref) / abs(v_ref)
    assert e_mid < e_lo
    assert e_mid < 2e-3


def test_symmetric_market_even_value_and_quote_symmetry(book):
    sol = solve_reduced2d(book, SYM, GAMMA, VBAR, T, nt=240, n_nu=11, n_V=101)
    v = np.asarray(sol.v0)
    # nu-independence (Remark 6) to machine precision
    assert np.max(np.abs(v - v[0][None, :])) < 1e-8 * np.max(np.abs(v))
    # even in Vpi
    assert np.max(np.abs(v - v[:, ::-1])) < 1e-8 * np.max(np.abs(v))
    # quote symmetry: bid(Vpi) == ask(-Vpi) channel-wise
    q = np.asarray(sol.quotes())
    n = book.n_options
    bid = q[n:, 5, :]
    ask = q[:n, 5, ::-1]
    m = np.isfinite(bid) & np.isfinite(ask)
    assert np.nanmax(np.abs(bid[m] - ask[m])) < 1e-10


def test_remark6_equivalence(book):
    """a_P = a_Q: mv-hedge at (xi, rho) == delta-neutral at xi*sqrt(1-rho^2)."""
    rho = SYM.rho
    A = solve_reduced2d(book, SYM, GAMMA, VBAR, T, c=rho,
                        nt=240, n_nu=11, n_V=101)
    xi_eff = SYM.xi * np.sqrt(1 - rho**2)
    SYM_dn = MarketParams(S0=SYM.S0, nu0=SYM.nu0, kappa_P=SYM.kappa_P,
                          theta_P=SYM.theta_P, kappa_Q=SYM.kappa_Q,
                          theta_Q=SYM.theta_Q, xi=xi_eff, rho=0.0)
    B = solve_reduced2d(book, SYM_dn, GAMMA, VBAR, T, c=0.0,
                        nt=240, n_nu=11, n_V=101)
    vA, vB = np.asarray(A.v0), np.asarray(B.v0)
    assert np.max(np.abs(vA - vB)) < 1e-8 * np.max(np.abs(vA))
    # ... and with a_P != a_Q the two differ (nu-coupling breaks it)
    A2 = solve_reduced2d(book, BBG_MARKET, GAMMA, VBAR, T, c=BBG_MARKET.rho,
                         nt=240, n_nu=11, n_V=101)
    xi_eff2 = BBG_MARKET.xi * np.sqrt(1 - BBG_MARKET.rho**2)
    B2_mkt = MarketParams(S0=10.0, nu0=0.0225, kappa_P=2.0, theta_P=0.04,
                          kappa_Q=3.0, theta_Q=0.0225, xi=xi_eff2, rho=0.0)
    B2 = solve_reduced2d(book, B2_mkt, GAMMA, VBAR, T, c=0.0,
                         nt=240, n_nu=11, n_V=101)
    # The gap lives only in the nu-curvature channel; over BBG's 0.3-day
    # horizon it is ~0.08 EUR at the Vpi extremes (vs 1.7e5 value scale) —
    # i.e. m(c) is essentially the whole hedge story at this horizon.  Assert
    # it is resolved above discretization noise but do not overstate it.
    gap = np.max(np.abs(np.asarray(A2.v0) - np.asarray(B2.v0)))
    assert 0.02 < gap < 1.0


def test_quote_monotonicity_in_Vpi(book):
    sol = solve_reduced2d(book, BBG_MARKET, GAMMA, VBAR, T,
                          nt=240, n_nu=11, n_V=101)
    q = np.asarray(sol.quotes())
    n = book.n_options
    tol = 1e-9
    for ich in range(n):
        ask = q[ich, 5, :]
        bid = q[n + ich, 5, :]
        a = ask[np.isfinite(ask)]
        b = bid[np.isfinite(bid)]
        assert np.all(np.diff(a) <= tol), f"ask ch {ich} not nonincreasing"
        assert np.all(np.diff(b) >= -tol), f"bid ch {ich} not nondecreasing"


def test_carry_sign_leans_long_vega(book):
    """Amplified a_P - a_Q > 0: at Vpi = 0 the MM bids tighter than it asks."""
    amp = MarketParams(S0=10.0, nu0=0.0225, kappa_P=2.0, theta_P=0.2,
                       kappa_Q=3.0, theta_Q=0.0225, xi=0.2, rho=-0.5)
    sol = solve_reduced2d(book, amp, GAMMA, VBAR, T, nt=240, n_nu=11, n_V=101)
    q = np.asarray(sol.quotes())
    n = book.n_options
    mid_idx = 50
    bid0 = q[n:, 5, mid_idx]
    ask0 = q[:n, 5, mid_idx]
    assert np.all(bid0 < ask0)
