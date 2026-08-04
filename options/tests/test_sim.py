"""Simulator validation: the two strongest end-to-end checks in the repo.

1. Value identity: E[int risk-adjusted reward] under the stored optimal
   policy equals v(0, nu0, Vpi=0) within MC error.  This exercises every
   sign convention at once (carry, penalty, psi/jump direction, reward
   accounting, Hamiltonian, policy extraction); a sign error anywhere shows
   up as a many-sigma gap.  Measured: gap = +0.65 SE at 20k paths.
2. Hedge parabola: Var[M(c)]/Var[M(0)] = m(c) = 1 - 2 rho c + c^2 under CRN,
   with the fitted vertex at c = rho.  Measured: all ratios within 1%,
   vertex -0.5023 vs -0.5.
"""
import numpy as np
import pytest

from optmm.core.risk import BBG_MARKET
from optmm.instruments.book import build_bbg_book
from optmm.solvers.lattice import subbook, with_trade_vega
from optmm.solvers.reduced2d import solve_reduced2d
from optmm.sim.lockstep import simulate_atoms

C_LIST = [-1.0, -0.5, 0.0, 0.5, 1.0]


@pytest.fixture(scope="module")
def sol():
    book = build_bbg_book(BBG_MARKET)
    sb = with_trade_vega(subbook(book, [2, 8]), [1.0e6, 1.5e6])
    return solve_reduced2d(sb, BBG_MARKET, 1e-3, 1e7, 0.0012,
                           nt=1200, n_nu=31, n_V=41, store_stride=1)


@pytest.fixture(scope="module")
def res(sol):
    return simulate_atoms(sol, BBG_MARKET, 1e-3, 0.0012, c=0.0,
                          c_list=C_LIST, n_paths=20000, seed=1)


def test_value_identity(sol, res):
    r = np.asarray(res.reward)
    v0 = float(sol.v0[15, 20])
    mean, se = r.mean(), r.std() / np.sqrt(len(r))
    assert abs(mean - v0) < 3.5 * se + 0.005 * abs(v0)


def test_hedge_parabola(res):
    rho = BBG_MARKET.rho
    var = np.asarray(res.M).var(axis=1)
    theory = np.array([1.0 - 2 * rho * c + c * c for c in C_LIST])
    ratios = var / var[C_LIST.index(0.0)]
    assert np.max(np.abs(ratios - theory) / theory) < 0.02
    coef = np.polyfit(C_LIST, var, 2)
    vertex = -coef[1] / (2 * coef[0])
    assert abs(vertex - rho) < 0.02


def test_paths_actually_trade(res):
    assert float(np.asarray(res.fills).mean()) > 1.0
    assert float(np.asarray(res.Vpi_final).std()) > 1e5
