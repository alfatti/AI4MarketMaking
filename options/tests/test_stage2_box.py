"""Stage 2: box admissibility on the lattice (no reduction exists).

Checks: (a) channel masking at box faces; (b) quote monotonicity in own
inventory; (c) RK4 step-halving; (d) negative control — under the box the
value is NOT a function of Vpi alone (the reduction genuinely dies), in
contrast to the slab case where the collapse held to the reference floor.
"""
import numpy as np
import pytest

from optmm.core.risk import BBG_MARKET
from optmm.core.state import Box
from optmm.instruments.book import build_bbg_book
from optmm.solvers.lattice import solve_lattice, subbook, with_trade_vega

GAMMA, T = 1e-3, 0.0012


@pytest.fixture(scope="module")
def sb():
    book = build_bbg_book(BBG_MARKET)
    return with_trade_vega(subbook(book, [2, 8]), [1.0e6, 1.5e6])


@pytest.fixture(scope="module")
def lat(sb):
    return solve_lattice(sb, BBG_MARKET, GAMMA, Box(np.array([6, 6])), T,
                         nt=400, n_nu=21)


def test_box_face_masking(sb, lat):
    q = np.asarray(lat.quotes())          # (4, n_nu, M); ask ch 0..1, bid 2..3
    at_hi = lat.idx_of([6, 0])
    at_lo = lat.idx_of([-6, 0])
    assert np.isnan(q[2, 10, at_hi])      # bid instrument 1 at +6: blocked
    assert np.isfinite(q[0, 10, at_hi])   # ask instrument 1 at +6: allowed
    assert np.isnan(q[0, 10, at_lo])      # ask instrument 1 at -6: blocked
    assert np.isfinite(q[2, 10, at_lo])


def test_quote_monotone_in_own_inventory(sb, lat):
    q = np.asarray(lat.quotes())
    ns = range(-6, 7)
    ask1 = [q[0, 10, lat.idx_of([n1, 0])] for n1 in ns]
    bid1 = [q[2, 10, lat.idx_of([n1, 0])] for n1 in ns]
    a = np.array([x for x in ask1 if np.isfinite(x)])
    b = np.array([x for x in bid1 if np.isfinite(x)])
    assert np.all(np.diff(a) <= 1e-9)
    assert np.all(np.diff(b) >= -1e-9)


def test_step_halving(sb):
    l1 = solve_lattice(sb, BBG_MARKET, GAMMA, Box(np.array([6, 6])), T,
                       nt=200, n_nu=21)
    l2 = solve_lattice(sb, BBG_MARKET, GAMMA, Box(np.array([6, 6])), T,
                       nt=400, n_nu=21)
    assert float(np.max(np.abs(np.asarray(l1.v0) - np.asarray(l2.v0)))) < 1e-4


def test_reduction_dies_under_box(sb, lat):
    """States with (near-)equal Vpi have materially different values."""
    v = np.asarray(lat.v0[10])
    Vpi = np.asarray(lat.Vpi)
    i_a = lat.idx_of([6, -4])             # Vpi = 0, near the box corner
    i_b = lat.idx_of([0, 0])              # Vpi = 0, center
    assert abs(Vpi[i_a] - Vpi[i_b]) < 1e-6
    assert abs(v[i_a] - v[i_b]) > 1000.0  # box geometry breaks the collapse
