# optmm

Option market making in the Baldacci–Bergault–Guéant (2020) framework
(arXiv:1907.12433): one underlying with Heston stochastic volatility under
ℙ and ℚ, Δ-hedged RFQ market making in N options, vega as the residual risk.
Three solver families over one shared definition of the HJB:

- **reduced2d** — the paper's dimensionality reduction v(t, ν, 𝒱π), explicit
  monotone Euler with linear interpolation (exact reduction, inexact solver:
  incommensurable jumps force interpolation);
- **lattice** — exact-in-inventory method of lines on (ν-grid) × (n-lattice),
  RK4 under `lax.scan`, generic in N, slab or box admissibility;
- **pinn** — a structure-blind MLP trained on the HJB residual over the full
  (t, ν, q) state, with hard terminal condition and *no* 𝒱ᵀq or
  constraint-distance features: the hyperplane structure ∇_q v ∥ 𝒱 is a
  discovery metric, not an input.

Plus a CRN lockstep simulator (value identity, hedge parabola) and the
Dynkin-corrector machinery (`validate/`), pre-validated where exact
references exist.

Mathematical source of truth: `docs/derivations.md`.  Measured validation
numbers: `docs/VALIDATION.md`.

## The three-stage arc

| Stage | Assumptions | Exact reference | Status (CPU smoke; H200 = same code, bigger budgets) |
|---|---|---|---|
| 1 | frozen Greeks + aggregate-vega slab | 2D reduction (exact on commensurable atoms) + q-lattice | **done**: solvers, cross-checks, sim identities, PINN trained + analysed (notebook 1) |
| 2 | frozen Greeks + per-option box (reduction dies) | q-lattice, N = 3/4/5 ladder **run** | **done**: box-residual PINN trained, validated on every admissible lattice state (notebook 2) |
| 3 | state-dependent Greeks (incl. sign-flipping spread), S in state, ρ cross term, box | none — frozen-Greeks regression + Dynkin closure + residual ascent | **done at smoke scale**: trained, closure gap +0.83 SE on CRN paths (notebook 2) |

The hedge layer (paper appendix A.1, generalized): proportional tilt c with
q^S = −Δ^π − c·ξ𝒱π/(2√ν S); the quoting problem sees it only through
m(c) = 1 − 2ρc + c² scaling the vega penalty.  The Merton term μ/(γνS) is
derived and deliberately zeroed (mandate separation).  Verified in-solver
(Remark-6 equivalence at machine precision) and in-simulator (parabola with
vertex at c = ρ).

## Layout

```
src/optmm/
  core/        hamiltonian (Lambert-W semi-closed logistic), risk (m(c),
               MarketParams, penalty), state (Slab/Box admissibility)
  instruments/ heston (cancellation-free COS CF, AD cumulants + greeks),
               book (BBG §4 universe, FrozenBook, events() stub),
               surrogate (live (S,ν) vega grids incl. sign-flipping spread)
  solvers/     reduced2d, lattice (+ subbook / with_trade_vega helpers),
               pinn/ (model, residual stage-1 & stage-3, trainer + GN-CG)
  sim/         lockstep (CRN, atom chain, reward + hedge-leg martingales,
               corrector integration), stage3 (S,ν,q sim, greedy PINN
               quotes, Dynkin closure)
  validate/    corrector (linear policy eval, residual tables, Dynkin)
tests/         38 fast + 2 slow-marked; see docs/VALIDATION.md
notebooks/     stage1_pinn_train_and_analysis.ipynb and
               stage2_stage3_ladder_and_frontier.ipynb — self-contained,
               re-entrant train + analysis for the full three-stage arc
               (both ship executed on the CPU profile; auto-switch to the
               H200 budget on CUDA)
scripts/       run_stage1_pinn.py (same budgets, headless)
```

## Quickstart

```bash
pip install -e . && pip install optax pytest matplotlib
pytest -q -m "not slow"        # ~4 min on CPU
jupyter lab notebooks/stage1_pinn_train_and_analysis.ipynb   # train + analysis
```

The notebook is the primary entry point: it builds/loads the cached
reference, resumes PINN training from `checkpoints/`, and produces the
outcome analysis (value collapse, hyperplane discovery, quote-level
comparison, residual diagnostics + adversarial ascent).  On a CUDA machine
it automatically runs the full H200 budget including the GN-CG polish.

## Conventions

Inventory in integer trade units n (contracts q = n ⊙ z); per-trade vega
w_i = z_i 𝒱_i; the 2D grid spans exactly [−V̄, V̄] so off-grid jump targets
are precisely the inadmissible ones; float64 enforced at package import;
quotes are per-contract mid-to-bid / ask-to-mid distances.

## Deliberately deferred

Distributed trade sizes (breaks lattice closure) · client tiering ·
two-factor vol · hedge frictions/caps (breaks the pointwise A.1 decoupling —
a real extension, not a toggle) · lifecycle events (`FrozenBook.events()`
stub reserves the interface) · multi-underlying · tensor-train lattice
compression (named as the honest competitor at large N).
