"""Stage-3 residual certification before anything exact exists at stage 3.

The stage-3 assembly (S in the state, rho cross term, state-dependent Greeks)
evaluated on an S-independent value function with frozen Greeks must equal the
stage-1 residual *identically*: the dS terms vanish under AD, mu = 0, and
vega(t,S,nu) == frozen vega implies the same run/jump terms.  This holds for
ANY network parameters, so the test uses random ones.  With genuinely
S-dependent Greeks the two must differ — the cross terms are live.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optmm.core.risk import BBG_MARKET, penalty_coef
from optmm.instruments.book import build_bbg_book
from optmm.solvers.lattice import subbook, with_trade_vega
from optmm.solvers.pinn.model import PinnSpec, init_params, value
from optmm.solvers.pinn.residual import residual_stage1, residual_stage3
from optmm.solvers.reduced2d import Channels
from optmm.solvers.pinn.train import sample_batch


@pytest.fixture(scope="module")
def setup():
    book = build_bbg_book(BBG_MARKET)
    sb = with_trade_vega(subbook(book, [2, 8]), [1.0e6, 1.5e6])
    ch = Channels.from_book(sb)
    spec = PinnSpec(T=0.0012, nu_lo=0.0144, nu_hi=0.0324,
                    w=jnp.asarray(sb.w), Vbar=1e7, hidden=(32, 32))
    params = init_params(spec, jax.random.PRNGKey(7))
    pen = penalty_coef(1e-3, BBG_MARKET.xi, BBG_MARKET.rho, 0.0)
    return book, sb, ch, spec, params, pen


def test_stage3_reduces_to_stage1(setup):
    _, sb, ch, spec, params, pen = setup
    lift = lambda p, s, t, S, nu, n: value(p, s, t, nu, n)  # S-independent
    frozen = lambda t, S, nu: sb.vega
    t_, nu_, n_ = sample_batch(jax.random.PRNGKey(1), spec, 64)
    S_ = jnp.full((64,), BBG_MARKET.S0) * jnp.linspace(0.9, 1.1, 64)
    r1 = jax.vmap(lambda t, nu, n: residual_stage1(
        params, spec, value, ch, BBG_MARKET, pen, 1e7, t, nu, n))(t_, nu_, n_)
    r3 = jax.vmap(lambda t, S, nu, n: residual_stage3(
        params, spec, lift, ch, BBG_MARKET, pen, 1e7, frozen, sb.z,
        t, S, nu, n))(t_, S_, nu_, n_)
    scale = jnp.max(jnp.abs(r1))
    assert float(jnp.max(jnp.abs(r1 - r3)) / scale) < 1e-12


def test_stage3_differs_with_live_greeks(setup):
    _, sb, ch, spec, params, pen = setup
    lift = lambda p, s, t, S, nu, n: value(p, s, t, nu, n)
    live = lambda t, S, nu: sb.vega * (S / BBG_MARKET.S0)  # toy S-dependence
    t_, nu_, n_ = sample_batch(jax.random.PRNGKey(2), spec, 64)
    S_ = jnp.full((64,), 1.1 * BBG_MARKET.S0)
    r1 = jax.vmap(lambda t, nu, n: residual_stage1(
        params, spec, value, ch, BBG_MARKET, pen, 1e7, t, nu, n))(t_, nu_, n_)
    r3 = jax.vmap(lambda t, S, nu, n: residual_stage3(
        params, spec, lift, ch, BBG_MARKET, pen, 1e7, live, sb.z,
        t, S, nu, n))(t_, S_, nu_, n_)
    assert float(jnp.max(jnp.abs(r1 - r3))) > 1.0
