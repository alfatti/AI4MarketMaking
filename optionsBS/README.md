# spotmm

The Baldacci-Bergault-Gueant options-market-making example **without the
two simplifying assumptions** (per-line box limits; Greeks live in S),
under one-factor spot models: **Black-Scholes** and **CEV local vol**.
Standalone companion to `optmm` (the Heston/vega build).

The structural gift of one factor: the full unsimplified problem lives on
(t, S, n) and is **exactly lattice-solvable** at small N — so the neural
solver is validated against the exact answer of the problem that matters,
and the Dynkin closure is the scale-out instrument rather than the only
one. The hedge layer transposes paper appendix A.1: the pointwise
decoupling survives, the optimum collapses to full hedging (completeness),
and the modeled channel is the *residual delta* family e = eta * Delta^pi
with a state-dependent penalty (gamma/2) eta^2 sigma(t,S)^2 S^2 (Delta^pi)^2
(docs/derivations.md section 2; Merton intercept exhibited and zeroed).

Sign-flip instrument: a tight 9.5/10/10.5 butterfly (delta flips inside
the trading band — the one-factor twin of optmm's vega-flip spread).
Measured flip theorem (exact on the lattice): hold the butterfly and your
*vanilla* skew reverses through the flip, crossing at S = 10.0.

```
src/spotmm/
  core/        hamiltonian (Lambert-W, ported verbatim), state (Slab/Box,
               ported), channels (SpotMarket, eta-risk coefficients)
  instruments/ spot_instruments (BS closed forms, CEV Crank-Nicolson,
               DeltaGrid live-Greeks, book with butterfly)
  solvers/     lattice ((t,S,n) exact reference; periodic anchor mode),
               pinn (model/residual/trainer + reduced1d frozen fixture),
               evaluate (exact policy pricing: regret vs the optimum,
               static and frozen-anchor baselines)
  sim/         lockstep (value identity, eta-parabola, q^S reporting,
               Dynkin accumulator)
tests/         18 tests, ~4 min CPU; numbers in docs/VALIDATION.md
checkpoints/   lattice_bs / lattice_lv caches, pinn_bs smoke checkpoint
```

Quickstart: `pip install -e . optax pytest && pytest -q`. Headline runs:
see docs/VALIDATION.md; deferred items in docs/derivations.md section 8.
