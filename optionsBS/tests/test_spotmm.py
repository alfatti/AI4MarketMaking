"""spotmm consolidated suite (small configs; headline numbers in
docs/VALIDATION.md come from the full-size runs and the notebook).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spotmm.core.channels import BS_MARKET, CEV_MARKET, Channels
from spotmm.core.state import Box
from spotmm.instruments.spot_instruments import (bs_call, bs_delta,
                                                 build_book, cn_call_surface)
from spotmm.sim.lockstep import simulate
from spotmm.solvers.lattice import analytic_eta0_value, solve_lattice
from spotmm.solvers.pinn import (PinnSpec, box_sampler, init_params,
                                 make_loss, residual, value)
from spotmm.solvers.reduced1d import solve_reduced1d  # noqa: F401  (import path)

T3, GAMMA = 0.006, 3e-5


@pytest.fixture(scope="module")
def bs():
    return build_book(BS_MARKET)


def test_bs_delta_ad_equals_closed_form():
    d_ad = float(jax.grad(lambda s: bs_call(s, 10.0, 0.15, 1.0))(10.0))
    assert d_ad == pytest.approx(float(bs_delta(10.0, 10.0, 0.15, 1.0)),
                                 abs=1e-12)


def test_butterfly_flip_both_models(bs):
    book, dg = bs
    _, dg_lv = build_book(CEV_MARKET)
    for g in (dg, dg_lv):
        assert float(g(0.0, 9.3)[3]) > 0.05
        assert float(g(0.0, 10.7)[3]) < -0.05


def test_cn_sigma_const_matches_bs():
    S, O, D = cn_call_surface(BS_MARKET, 10.0, 0.25, n_x=201, n_t=120)
    assert abs(float(np.interp(10.0, S, O))
               - float(bs_call(10.0, 10.0, 0.15, 0.25))) < 2e-4
    assert abs(float(np.interp(10.0, S, D))
               - float(bs_delta(10.0, 10.0, 0.15, 0.25))) < 2e-4


def test_analytic_anchor_periodic(bs):
    book, dg = bs
    lat = solve_lattice(book, dg, BS_MARKET, GAMMA, Box(np.array([1, 1, 1, 1])),
                        T3, eta=0.0, nt=100, n_S=11, periodic=True)
    v_an = analytic_eta0_value(book, T3)
    assert float(jnp.max(jnp.abs(lat.v0 - v_an))) < 1e-6 * v_an


def test_eta0_box_gap_nonnegative_and_face_localized(bs):
    book, dg = bs
    NB = np.array([1, 1, 1, 2])
    lat = solve_lattice(book, dg, BS_MARKET, GAMMA, Box(NB), T3, eta=0.0,
                        nt=100, n_S=11)
    v_an = analytic_eta0_value(book, T3)
    v = np.asarray(lat.v0[5])
    adm = np.all(np.abs(lat.n_pts) <= NB[None, :], axis=1)
    gap = v_an - v[adm]
    face = np.min(NB[None, :] - np.abs(lat.n_pts), axis=1)[adm]
    assert (gap >= -1e-6 * v_an).all()
    assert gap[face == 0].max() > gap[face == 1].max() > 0


def test_lattice_step_halving_small(bs):
    book, dg = bs
    box = Box(np.array([1, 1, 1, 2]))
    # nt = 50 sits outside RK4's jump-Lipschitz stability bound
    # (dt * sum Lambda ~ 5.8 > 2.78); compare the stable pair instead.
    a = solve_lattice(book, dg, BS_MARKET, GAMMA, box, T3, nt=200, n_S=11)
    b = solve_lattice(book, dg, BS_MARKET, GAMMA, box, T3, nt=100, n_S=11)
    assert float(jnp.max(jnp.abs(a.v0 - b.v0))) < 1.0


def test_flip_reversal_exact_small(bs):
    """ATM-call skew at a long-butterfly inventory crosses zero at the
    delta flip — the corrected reversal statement, as an exact lattice
    fact on a small hermetic solve."""
    book, dg = bs
    lat = solve_lattice(book, dg, BS_MARKET, GAMMA,
                        Box(np.array([1, 1, 1, 3])), T3, nt=100, n_S=21)
    q = np.asarray(lat.quotes())
    S = np.asarray(lat.S_grid)
    i = lat.idx_of([0, 0, 0, 3])
    skew = q[1, :, i] - q[5, :, i]
    assert skew[0] < 0 and skew[-1] > 0
    cross = S[np.where(np.diff(np.sign(skew)))[0]]
    assert len(cross) == 1 and 9.85 < cross[0] < 10.15


def test_sim_identity_and_eta_parabola(bs):
    book, dg = bs
    box = Box(np.array([1, 1, 1, 2]))
    lat = solve_lattice(book, dg, BS_MARKET, GAMMA, box, T3, nt=100, n_S=21,
                        store_stride=2)
    v0 = float(lat.v0[10, lat.idx_of([0, 0, 0, 0])])
    res = simulate(book, dg, BS_MARKET, GAMMA, box, T3, lat,
                   eta_list=[0.0, 0.5, 1.0], n_steps=800, n_paths=4000,
                   seed=2)
    r = np.asarray(res.reward)
    mean, se = r.mean(), r.std() / np.sqrt(len(r))
    assert abs(mean - v0) < 3.5 * se + 0.02 * abs(v0)
    var = np.asarray(res.M).var(axis=1)
    np.testing.assert_allclose(var / var[-1], [0.0, 0.25, 1.0], atol=1e-3)
    assert float(np.asarray(res.fills).mean()) > 3.0


def test_pinn_structural(bs):
    book, dg = bs
    NB = np.array([1, 1, 1, 2])
    spec = PinnSpec(T=T3, S0=10.0, S_half=0.8,
                    n_max=jnp.asarray(NB, jnp.float64), hidden=(16, 16))
    p = init_params(spec, jax.random.PRNGKey(0))
    assert float(value(p, spec, T3, 10.4, jnp.ones(4))) == 0.0
    t, S, n = box_sampler(jax.random.PRNGKey(1), spec, 256)
    assert np.all(np.abs(np.asarray(S) - 10.0) <= 0.8 + 1e-12)
    assert np.all(np.abs(np.asarray(n)) <= NB[None, :] + 1e-12)
    loss = make_loss(spec, book, dg, BS_MARKET, GAMMA, 1.0, Box(NB))
    assert np.isfinite(float(loss(p, t[:32], S[:32], n[:32])))
    r = residual(p, spec, book, dg, BS_MARKET, GAMMA, 1.0, Box(NB),
                 0.001, 10.1, jnp.array([1.0, 0.0, -1.0, 2.0]))
    assert np.isfinite(float(r))


def test_reduced1d_symmetry_and_refinement(bs):
    book, _ = bs
    a = solve_reduced1d(book, BS_MARKET, GAMMA, 4e6, T3, n_x=101, nt=600)
    b = solve_reduced1d(book, BS_MARKET, GAMMA, 4e6, T3, n_x=201, nt=1200)
    va = np.asarray(a.v0)
    # mu = 0 and symmetric intensities: v even in x
    assert np.max(np.abs(va - va[::-1])) < 1e-6 * np.max(np.abs(va))
    vb_on_a = np.interp(np.asarray(a.x_grid), np.asarray(b.x_grid),
                        np.asarray(b.v0))
    assert np.max(np.abs(va - vb_on_a)) < 0.01 * np.max(np.abs(va))
