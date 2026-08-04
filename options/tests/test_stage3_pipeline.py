"""Stage-2/3 pipeline: surrogate Greeks, box residual semantics, stage-3
model and simulator smoke.  Full-scale results live in
notebooks/stage2_stage3_ladder_and_frontier.ipynb and docs/VALIDATION.md.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optmm.core.risk import BBG_MARKET, penalty_coef
from optmm.core.state import Box, Slab
from optmm.instruments.surrogate import build_stage3_book
from optmm.solvers.pinn.model import PinnSpec, init_params, value
from optmm.solvers.pinn.model3 import (Pinn3Spec, init_params3, sample_batch3,
                                       value3)
from optmm.solvers.pinn.residual import residual_stage1, residual_stage3
from optmm.solvers.reduced2d import Channels
from optmm.sim.stage3 import simulate_stage3


@pytest.fixture(scope="module")
def coarse_stage3():
    # coarse grid keeps the fixture fast; the notebook uses the fine grid
    return build_stage3_book(BBG_MARKET, n_S=9, n_nu=5)


def test_spread_vega_sign_flip(coarse_stage3):
    book3, vega_fn, vega_ref = coarse_stage3
    v_lo = float(vega_fn(0.0, 9.4, 0.0225)[3])
    v_hi = float(vega_fn(0.0, 10.6, 0.0225)[3])
    assert v_lo > 0.05 and v_hi < -0.05
    assert float(vega_ref[3]) > 0.1


def test_surrogate_matches_anchor(coarse_stage3):
    book3, vega_fn, _ = coarse_stage3
    v = np.asarray(vega_fn(0.0, BBG_MARKET.S0, BBG_MARKET.nu0))
    # bilinear interp on the coarse grid vs exact anchor vegas
    np.testing.assert_allclose(v, np.asarray(book3.vega), atol=5e-3)


def test_box_residual_semantics():
    """Box == Slab deep inside; differ where box faces mask channels."""
    from optmm.instruments.book import build_bbg_book
    from optmm.solvers.lattice import subbook, with_trade_vega
    bk = with_trade_vega(subbook(build_bbg_book(BBG_MARKET), [2, 8]),
                         [1.0e6, 1.5e6])
    ch = Channels.from_book(bk)
    spec = PinnSpec(T=0.0012, nu_lo=0.0144, nu_hi=0.0324,
                    w=jnp.asarray(bk.w), Vbar=1e7, hidden=(16, 16))
    p = init_params(spec, jax.random.PRNGKey(1))
    pen = penalty_coef(1e-3, 0.2, -0.5, 0.0)
    box = Box(np.array([3, 3]))
    args = (p, spec, value, ch, BBG_MARKET, pen, 1e7)
    r_slab = residual_stage1(*args, 0.0005, 0.02, jnp.array([1.0, -1.0]))
    r_box = residual_stage1(*args, 0.0005, 0.02, jnp.array([1.0, -1.0]),
                            adm_set=box)
    assert abs(float(r_slab - r_box)) < 1e-9
    r_slab_f = residual_stage1(*args, 0.0005, 0.02, jnp.array([3.0, 0.0]))
    r_box_f = residual_stage1(*args, 0.0005, 0.02, jnp.array([3.0, 0.0]),
                              adm_set=box)
    assert abs(float(r_slab_f - r_box_f)) > 1.0


def test_model3_terminal_and_sampler(coarse_stage3):
    book3, _, _ = coarse_stage3
    spec3 = Pinn3Spec(T=0.006, S0=10.0, S_half=0.6, nu_lo=0.0144,
                      nu_hi=0.0324, w_ref=book3.z * book3.vega,
                      n_max=jnp.array([4., 4., 4., 8.]), Vbar=1e7,
                      hidden=(16, 16))
    p3 = init_params3(spec3, jax.random.PRNGKey(0))
    assert float(value3(p3, spec3, 0.006, 10.3, 0.02, jnp.ones(4))) == 0.0
    t, S, nu, n = sample_batch3(jax.random.PRNGKey(1), spec3, 512)
    assert np.all(np.abs(np.asarray(S) - 10.0) <= 0.6 + 1e-12)
    assert np.all(np.abs(np.asarray(n)) <= np.array([4, 4, 4, 8]) + 1e-12)


def test_stage3_sim_smoke_and_closure_finite(coarse_stage3):
    book3, vega_fn, _ = coarse_stage3
    ch3 = Channels.from_book(book3)
    box3 = Box(np.array([2, 2, 2, 3]))
    spec3 = Pinn3Spec(T=0.001, S0=10.0, S_half=0.6, nu_lo=0.0144,
                      nu_hi=0.0324, w_ref=book3.z * book3.vega,
                      n_max=jnp.array([2., 2., 2., 3.]), Vbar=1e7,
                      hidden=(16, 16))
    p3 = init_params3(spec3, jax.random.PRNGKey(2))
    sim = simulate_stage3(p3, spec3, ch3, vega_fn, book3.z, BBG_MARKET,
                          1e-3, box3, 0.001, n_steps=50, n_paths=40,
                          res_stride=5, seed=0)
    assert bool(jnp.all(jnp.isfinite(sim.reward)))
    assert bool(jnp.all(jnp.isfinite(sim.corr)))
    v0 = float(value3(p3, spec3, 0.0, 10.0, 0.0225, jnp.zeros(4)))
    gap = float(jnp.mean(sim.reward)) - (v0 + float(jnp.mean(sim.corr)))
    assert np.isfinite(gap)
