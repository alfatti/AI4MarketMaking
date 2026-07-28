# hjbpinn — PINN solver for the general Model B market-making HJ equation

Solves the reduced equation (Eq. 21 of Bergault–Evangelista–Guéant–Vieira) for
`theta(t, q)` under **Model B (xi = 0)**: multi-asset, asymmetric bid/ask intensities,
client tiers, atomic request-size distributions, fixed transaction costs, drift, and
**hard risk limits** (the indicators are kept — `Q = infinity` lives only in the
closed-form scaffold).

## Architecture

`theta = theta_check + (1 - t/T) * theta_scale * NN(t/T, q/Q, boundary features)`

- **`spec.py`** — `MarketSpec`: the single source of truth. Families: exponential
  `A e^{-k delta}` and logistic `lam / (1 + e^{alpha + beta delta})` per (asset, tier, side).
- **`hamiltonians.py`** — `H0(p) = sup_{delta >= -delta_inf} Lambda(delta)(delta - p)`.
  Logistic argmax in **closed form via Lambert-W**: `delta* = p + (1 + W0(e^{-(alpha+beta p+1)}))/beta`,
  evaluated in log space (overflow-free). `H0` carries `stop_gradient(delta*)` so autodiff
  yields the exact Danskin derivative `-Lambda(delta*)`; Gauss-Newton needs nothing more
  (H0 is C^1 but not C^2 at the floor — a reason to avoid second-order residual terms).
- **`proxy.py`** — quadratic proxy with the **general** coefficient system (tiers, costs,
  drift): spectral tanh Riccati for `A(t)`, variation-of-parameters `B(t)` with stable
  cosh-ratio Gauss–Legendre quadrature, and `C'(t)` **derived from scratch by degree
  matching** — note the `c^2 E[1/z]` (inverse-moment) cost term. `A'`, `B'`, `C'` are
  algebraic ODE right-hand sides: the residual never differentiates through
  eigendecompositions or quadrature. Exact difference formulas (`diffs`) avoid
  large-value subtraction. `theta_fast` uses a Hermite-cached `C` for reporting; the
  exact nested-quadrature `C` is retained for the independent-path Phase-0 test.
- **`network.py`** — hand-rolled MLP; zero-init output layer (`eta == 0` at init: training
  starts exactly at the proxy); exponential boundary-layer features `exp(-(Q -+ q)/zbar)`
  (affine distances would be vacuous through an affine first layer); and **kink features**
  `-expm1(-|q_i - s|/zbar_i)` at every switching surface `s = +-(Q_i - z_k)`. The kink
  features are the v2 fix for a diagnosed structural ceiling: the RHS jumps across each
  surface, so `d_t theta` is discontinuous in q and a smooth-in-q ansatz carries an
  irreducible residual there regardless of capacity (see RUN_REPORT).
- **`residual.py`** — `prepare_batch` precomputes everything parameter-independent
  (proxy diffs, `dt theta_check`, masks, features); `residual_point` is the **pointwise**
  scaled residual — Jacobians assemble as vmap-of-grad (one single-point graph per row).
  Also the two Phase-0 quadratic-HJ evaluators (production path / independent path).
- **`optimizer.py`** — Gauss-Newton + LM trust region, Jacobi column scaling, kernel-trick
  solve `(Jt Jt' + lam I)` in residual space (N < P regime). Deterministic collocation per
  phase (stochastic batching corrupts curvature).
- **`sampling.py`** — uniform, boundary bands, **surface bands** and deterministic
  **surface straddles** (points at `+-{1e-4, 0.03, 0.12} * zbar` from each switching
  surface — Gaussian bands never sample the 1e-4 scale, so without these the learned kink
  amplitude error is invisible to training but not to the certificate grid), **model-native
  exact thinning** under the proxy-greedy policy (numpy event loop over grid-interpolated
  `A`, `B`), and RAR over a mixed candidate pool.
- **`validate.py`** — exact method-of-lines lattice reference (atoms commensurate with the
  grid: index shifts exact, censoring = lattice edge); sup-norm contraction certificate
  `|theta_hat - theta|_inf <= eps (T - t)` estimated on a grid that straddles the switching
  surfaces `q_i = +-(Q_i - z_k)`; fill-intensity-weighted quote metrics.
- **`policy.py`** — oracle-compatible `quote(t, q, z, i, n, side)`; `use_eta=False` gives
  the pure closed-form proxy policy.

## Verified invariants (tests/test_phase0.py, tests/test_wiring.py)

- Lambert-W root: cancellation-free FOC < 3e-15, all families; floor semantics per family
  (exponential root runs to the floor; bounded logistic saturates and need not).
- Danskin: AD grad of `H0` == `-Lambda(delta*)` to machine precision.
- `A(T) = B(T) = 0`; `A'`, `B'` RHS match finite differences of the closed forms (~1e-10).
- **Machine-zero**: `theta_check` zeros the quadratic HJ at 1.3e-16 (production path) and
  3.3e-11 (independent path: direct subtraction + FD time derivative) on a random d=3,
  2-tier, mixed-family spec with drift and costs — this pins the entire derivation,
  including the `E[1/z]` cost term.
- d=1 collapse to the scalar tanh formula; symmetric no-drift => `B == 0`.
- `B(t)` from the cosh-ratio quadrature vs a tight ODE integration at `lam*T = 66`
  (1.1e-13 relative) — covers the Phase-2 regime, where `lam*T ~ 26`.
- **Wiring**: the fast residual equals a brute-force independent implementation (direct
  subtraction, FD in t, explicit feasibility test) at random nonzero `eta`, including
  points straddling the switching surfaces where masks flip (5.7e-7 / 5.2e-7). Phase 0's
  machine-zero tests only exercise the `eta = 0` path; this covers the rest.

## Usage

```bash
python tests/test_phase0.py            # structural verification
python tests/test_wiring.py            # residual assembly vs brute force at nonzero eta

# Phase 1 v2, one bounded stage per invocation (state persisted to npz).
# scenario is {exp|logi}; k is the GN phase index 0..2.
python tests/run_v2.py phase exp 0 states/v2_exp.npz
python tests/run_v2.py phase exp 1 states/v2_exp.npz
python tests/run_v2.py phase exp 2 states/v2_exp.npz
python tests/run_v2.py cert  exp states/v2_exp.npz   # certificate + residual localization
python tests/run_v2.py mol   exp states/v2_exp.npz   # theta / quote errors vs exact MOL

python tests/demo_phase1_v2.py exp     # same run as a single process (needs ~4 GB / ~10 min)
```

`states/v2_exp.npz` and `states/v2_logi.npz` ship with the trained parameters behind the
RUN_REPORT numbers, so `cert` and `mol` reproduce them without retraining.

Requires `jax` (CPU fine; float64 enforced) and `scipy`. On GPU nothing changes.

## Current state

Phase 1 v2 is validated on both intensity families against the exact MOL reference; see
RUN_REPORT.md for the full table. Headline (exponential scenario, vs MOL at t = 0):
theta sup 2.19e-4, fill-weighted quote RMSE 6.05e-5 — 10x and 15x below v1. The
switching-surface residual is no longer binding; the mid-horizon slice (t = 0.5) is, and
a polish phase confirmed that is an ansatz limit (the global `(1 - t/T)` prefactor
throttling kink amplitude), not undertraining. Documented fix: a separate kink-amplitude
head with its own terminal enforcement.

## Roadmap seams (deliberately not built)

Phase 2: replicate the paper's d=2 benchmark (sigma = 1.2/0.6, rho = 0.5, gamma = 8e-6,
T = 7, logistic lam=30/alpha=0.7/beta=30, 4 Gamma-atoms, Q = 75000/300000) against their
FD numbers. Phase 3: scale d with the Section-4 Monte-Carlo correction term as pointwise
ground truth. Deferred: continuous-nu quadrature, RAR loop wiring, permutation-symmetric
encoders, matrix-free CG (only needed once N ~ P), Howard-iteration regression variant.
