"""PINN stage 1: structural checks (fast) and a short-training smoke (slow).

Full-budget training is an H200 job (scripts/run_stage1_pinn.py).  The CPU
smoke recorded in docs/VALIDATION.md (6000 Adam steps, ~4 min) reached
median |cos(grad_n v, w)| = 1.0000 with 99% of interior samples > 0.99 —
the pure MLP discovers the hyperplane structure untold — and 0.8% value
error at the book center vs the reduced2d reference.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from optmm.core.risk import BBG_MARKET
from optmm.instruments.book import build_bbg_book
from optmm.solvers.pinn.model import PinnSpec, init_params, value
from optmm.solvers.pinn.train import (adam_train, collinearity, make_loss,
                                      sample_batch)
from optmm.solvers.reduced2d import Channels


@pytest.fixture(scope="module")
def setup():
    book = build_bbg_book(BBG_MARKET)
    ch = Channels.from_book(book)
    spec = PinnSpec(T=0.0012, nu_lo=0.0144, nu_hi=0.0324,
                    w=jnp.asarray(book.w), Vbar=1e7, hidden=(64, 64, 64))
    params = init_params(spec, jax.random.PRNGKey(0))
    return book, ch, spec, params


def test_terminal_condition_hard(setup):
    _, _, spec, params = setup
    n = jnp.ones(20) * 2.0
    assert float(value(params, spec, spec.T, 0.02, n)) == 0.0


def test_sampler_in_slab(setup):
    _, _, spec, _ = setup
    t, nu, n = sample_batch(jax.random.PRNGKey(1), spec, 2048)
    Vpi = np.asarray(n @ spec.w)
    assert np.all(np.abs(Vpi) <= 0.9500001 * 1e7)
    assert np.all((np.asarray(t) >= 0) & (np.asarray(t) <= spec.T))


def test_residual_finite(setup):
    book, ch, spec, params = setup
    loss_fn, batched = make_loss(spec, ch, BBG_MARKET, 1e-3, 0.0, 1e7)
    t, nu, n = sample_batch(jax.random.PRNGKey(2), spec, 32)
    r = batched(params, t, nu, n)
    assert bool(jnp.all(jnp.isfinite(r)))


@pytest.mark.slow
def test_short_training_reduces_loss_and_discovers_hyperplane(setup):
    book, ch, spec, params = setup
    loss_fn, batched = make_loss(spec, ch, BBG_MARKET, 1e-3, 0.0, 1e7)
    t, nu, n = sample_batch(jax.random.PRNGKey(3), spec, 256)
    l0 = float(loss_fn(params, t, nu, n))
    params, _ = adam_train(params, loss_fn, spec, jax.random.PRNGKey(4),
                           steps=2500, batch=256, lr=2e-3, log_every=1250,
                           log=lambda *_: None)
    l1 = float(loss_fn(params, t, nu, n))
    assert l1 < l0 / 100.0        # measured: 618x at this budget
    med_cos, _ = collinearity(params, spec, jax.random.PRNGKey(5))
    assert med_cos > 0.95         # measured: 1.0000
