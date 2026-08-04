"""Dynkin corrector triangle test on the stage-1 atom chain.

Take v~ = optimal value deliberately perturbed by a smooth known bump; let
pi~ be the (suboptimal) policy its quotes induce.  Three independent
evaluations of V^{pi~}(0, nu0, 0) must agree:
  (1) direct simulation of the reward under pi~;
  (2) the linear policy-evaluation PDE;
  (3) the corrector estimate v~(0) + E[int R~ dt] along simulated paths.
And V^{pi~} must be below the optimal value (suboptimality sanity).  This
pre-validates the exact machinery that stage 3 relies on where no exact
reference exists.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from optmm.core.risk import BBG_MARKET
from optmm.instruments.book import build_bbg_book
from optmm.solvers.lattice import subbook, with_trade_vega
from optmm.solvers.reduced2d import solve_reduced2d
from optmm.sim.lockstep import simulate_atoms
from optmm.validate.corrector import (linear_policy_eval, quotes_tables,
                                      residual_tables)

GAMMA, VBAR, T = 1e-3, 1e7, 0.0012


@pytest.fixture(scope="module")
def stack():
    book = build_bbg_book(BBG_MARKET)
    sb = with_trade_vega(subbook(book, [2, 8]), [1.0e6, 1.5e6])
    sol = solve_reduced2d(sb, BBG_MARKET, GAMMA, VBAR, T,
                          nt=1200, n_nu=31, n_V=41, store_stride=1)
    # smooth known perturbation, zero at t = T (terminal condition kept)
    tt = sol.hist_t[:, None, None] / T
    bump = (1.0 - tt) * 1500.0 * jnp.cos(
        np.pi * sol.V_grid[None, None, :] / (2 * VBAR))
    v_pert = sol.v_hist + bump
    tabs = quotes_tables(sol, v_pert)
    u0, _ = linear_policy_eval(sol, tabs, BBG_MARKET, GAMMA, T)
    res = residual_tables(sol, v_pert, tabs, BBG_MARKET, GAMMA, T)
    sim = simulate_atoms(sol, BBG_MARKET, GAMMA, T, n_paths=20000, seed=3,
                         tabs=tabs, res_tab=res)
    return sol, v_pert, u0, sim


def test_triangle_sim_vs_linear_pde(stack):
    sol, _, u0, sim = stack
    r = np.asarray(sim.reward)
    mean, se = r.mean(), r.std() / np.sqrt(len(r))
    u_val = float(u0[15, 20])
    assert abs(mean - u_val) < 3.5 * se + 0.005 * abs(u_val)


def test_triangle_corrector_vs_linear_pde(stack):
    sol, v_pert, u0, sim = stack
    c = np.asarray(sim.corr)
    vhat = float(v_pert[0, 15, 20]) + c.mean()
    se = c.std() / np.sqrt(len(c))
    u_val = float(u0[15, 20])
    assert abs(vhat - u_val) < 3.5 * se + 0.01 * abs(u_val)


def test_suboptimality(stack):
    sol, _, u0, _ = stack
    assert float(u0[15, 20]) < float(sol.v0[15, 20]) + 1.0


def test_linear_eval_recovers_optimal_value(stack):
    """Evaluating the *optimal* policy linearly must return the optimal v."""
    sol, _, _, _ = stack
    from optmm.sim.lockstep import policy_tables
    tabs_opt = policy_tables(sol)
    u0, _ = linear_policy_eval(sol, tabs_opt, BBG_MARKET, GAMMA, T)
    assert abs(float(u0[15, 20]) - float(sol.v0[15, 20])) < 5.0
