"""M4: lattice vs reduced2d cross-validation on a commensurable sub-book.

Design: with per-trade vegas w = (1.0e6, 1.5e6) every Vpi jump lands exactly
on the 0.5e6-atom grid, so reduced2d run on the 41-atom grid is *exact in
Vpi* and serves as the reference that separates
  (i)  the 2D solver's Vpi-interpolation error (generic, non-divisor NV), and
  (ii) the lattice's representation-box truncation pollution (the slab is an
       infinite diagonal band; a finite box cuts genuinely admissible states,
       and the damage decays with distance from the artificial faces).
Measured reference numbers (nt=11520 atoms): Euler time error is exactly
first order; interp max error 153 -> 50 -> 9.8 EUR over NV = 97/193/385;
box+4 padding kills pollution at the base eval set to < 10 EUR; the lattice
matches the exact reduction at n=0 to the reference's own dt error (~0.3 EUR
on a 26.5k scale).
"""
import numpy as np
import pytest

from optmm.core.risk import BBG_MARKET
from optmm.core.state import Slab
from optmm.instruments.book import build_bbg_book
from optmm.solvers.lattice import solve_lattice, subbook, with_trade_vega
from optmm.solvers.reduced2d import solve_reduced2d

GAMMA, VBAR, T = 1e-3, 1e7, 0.0012


@pytest.fixture(scope="module")
def sb():
    book = build_bbg_book(BBG_MARKET)
    return with_trade_vega(subbook(book, [2, 8]), [1.0e6, 1.5e6])


@pytest.fixture(scope="module")
def atoms(sb):
    return {nt: np.asarray(solve_reduced2d(sb, BBG_MARKET, GAMMA, VBAR, T,
                                           nt=nt, n_nu=31, n_V=41).v0)
            for nt in (720, 1440, 2880, 5760)}


def test_atom_grid_exactness(sb):
    """NV=41 and NV=81 both land jumps on nodes -> identical solutions."""
    a = np.asarray(solve_reduced2d(sb, BBG_MARKET, GAMMA, VBAR, T,
                                   nt=1440, n_nu=31, n_V=41).v0)
    b = np.asarray(solve_reduced2d(sb, BBG_MARKET, GAMMA, VBAR, T,
                                   nt=1440, n_nu=31, n_V=81).v0)
    assert np.max(np.abs(a - b[:, ::2])) < 1e-6


def test_euler_first_order_in_time(atoms):
    d1 = np.max(np.abs(atoms[720] - atoms[5760]))
    d2 = np.max(np.abs(atoms[1440] - atoms[5760]))
    d3 = np.max(np.abs(atoms[2880] - atoms[5760]))
    # exact first order: ratios (1/720-1/5760)/(1/1440-1/5760) = 7/3, then 3
    assert d1 / d2 == pytest.approx(7.0 / 3.0, rel=0.15)
    assert d2 / d3 == pytest.approx(3.0, rel=0.15)


def test_interp_error_ladder(sb, atoms):
    ref = atoms[2880][15]
    Vat = np.linspace(-VBAR, VBAR, 41)
    errs = []
    for NV in (97, 193, 385):
        r2 = solve_reduced2d(sb, BBG_MARKET, GAMMA, VBAR, T,
                             nt=2880, n_nu=31, n_V=NV)
        v2 = np.interp(Vat, np.asarray(r2.V_grid), np.asarray(r2.v0[15]))
        errs.append(np.max(np.abs(v2 - ref)))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 15.0


def test_lattice_matches_exact_reduction(sb, atoms):
    """Padded box: lattice value == exact atoms on the base admissible set."""
    class SlabPad(Slab):
        def representation_nbar(self, w):
            return np.ceil(self.Vbar / np.asarray(w)).astype(int) + 5

    lat = solve_lattice(sb, BBG_MARKET, GAMMA, SlabPad(VBAR), T,
                        nt=600, n_nu=31)
    base = Slab(VBAR)
    nbar0 = base.representation_nbar(np.asarray(sb.w))
    adm = np.asarray(lat.adm_self)
    inside0 = np.all(np.abs(lat.n_pts) <= nbar0[None, :], axis=1)
    m = adm & inside0
    Vat = np.linspace(-VBAR, VBAR, 41)
    v_ref = np.interp(np.asarray(lat.Vpi)[m], Vat, atoms[5760][15])
    err = np.abs(np.asarray(lat.v0[15])[m] - v_ref)
    assert np.max(err) < 12.0           # reference dt floor + margin
    i0 = lat.idx_of([0, 0])
    e0 = abs(float(lat.v0[15, i0]) - atoms[5760][15, 20])
    assert e0 < 1.0                     # ~0.3 EUR on 26.5k = 1e-5 relative


def test_box_truncation_decays_with_padding(sb, atoms):
    errs = []
    for extra in (0, 4):
        class SlabX(Slab):
            def representation_nbar(self, w, _e=extra):
                return np.ceil(self.Vbar / np.asarray(w)).astype(int) + 1 + _e

        lat = solve_lattice(sb, BBG_MARKET, GAMMA, SlabX(VBAR), T,
                            nt=300, n_nu=31)
        adm = np.asarray(lat.adm_self)
        nbar0 = np.ceil(VBAR / np.asarray(sb.w)).astype(int) + 1
        m = adm & np.all(np.abs(lat.n_pts) <= nbar0[None, :], axis=1)
        Vat = np.linspace(-VBAR, VBAR, 41)
        v_ref = np.interp(np.asarray(lat.Vpi)[m], Vat, atoms[5760][15])
        errs.append(np.max(np.abs(np.asarray(lat.v0[15])[m] - v_ref)))
    assert errs[0] > 1000.0             # unpadded: face pollution is huge
    assert errs[1] < 15.0               # +4 padding: at the reference floor
