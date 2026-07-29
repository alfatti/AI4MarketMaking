# Run report — hjbpinn v2 (CPU container, float64, JAX 0.11)

All numbers below are from runs in this container. MOL = exact method-of-lines lattice
reference (atoms commensurate with the grid: index shifts exact, censoring = lattice edge,
no interpolation anywhere in the reference). Quote errors are fill-intensity weighted.

## Phase 0 — structural verification: ALL PASS

| check | result |
|---|---|
| root FOC, cancellation-free form, all families | 2.66e-15 |
| Danskin: AD grad of H0 == -Lambda(delta*) | exact (< 1e-12) |
| floor semantics per family (exp binds; bounded logistic saturates) | pass |
| H0 sup-property against a delta-grid | gap 0.00 |
| A(T) = B(T) = 0 | pass |
| A', B' ODE RHS vs FD of closed forms | 2.21e-10 / 8.34e-11 (rel) |
| **B quadrature vs tight ODE reference at lam*T = 66.4** | **1.10e-13 (rel)** |
| **machine-zero quadratic HJ, production path** (d=3, 2 tiers, mixed families, drift, costs) | **1.29e-16** |
| **machine-zero quadratic HJ, independent path** (direct subtraction + FD in t) | **3.29e-11** |
| d=1 scalar tanh collapse (3 t-values) | exact |
| d=1 symmetric no-drift: B == 0 | exact |
| GN solves linear LSQ in <= 3 iterations | 2.47e-11 |

Diagnostic (unchanged, as predicted): the proxy defect on the TRUE equation is bimodal —
interior sup 2.5e-2 vs boundary-band sup 4.9e-1 (scaled units).

## Wiring regression (tests/test_wiring.py) — ALL PASS

Fast residual (analytic proxy diffs, analytic d_t theta_check, precomputed masks,
flattened stencil) vs a brute-force independent implementation (direct subtraction,
central FD in t, explicit feasibility test) at RANDOM NONZERO eta, including points
straddling the switching surfaces where masks flip:

| scenario | max diff |
|---|---|
| exp, 1 tier, 1 atom | 5.71e-07 |
| logistic, 2 tiers, 2 atoms | 5.19e-07 |

(Phase-0's machine-zero tests only exercise the eta = 0 path; this covers the rest.)

## The v1 ceiling, diagnosed

Dense-grid localization of the v1 trained residual put the sup **at the switching
surfaces** q = +-(Q - z_k), 25x the interior level (1.39e-2 vs 5.67e-4). The equation's
RHS jumps across each surface (an indicator switches), so d_t theta is discontinuous in q
and theta has a kink there. **No smooth-in-q ansatz can represent that** — an irreducible
floor independent of capacity or iteration count.

v2 recipe: (i) bounded kink features -expm1(-|q - s|/zbar) per surface, t-modulated so
d_t theta_hat inherits the jump; (ii) surface-band collocation; (iii) deterministic
surface-STRADDLE collocation at +-{1e-4, 0.03, 0.12}*zbar (Gaussian bands never sample
the 1e-4 scale, so the learned kink amplitude error was invisible to training but not to
the certificate grid); (iv) two RAR phases with straddles in the candidate pool;
(v) widths (64, 64).

## Phase 1 v2 — d=1 asymmetric exponential (mu = 0.05, c = 0.01, Q = 6, T = 1)

3 GN phases (90/60/60) + 1 polish phase, ~2200 -> ~2840 collocation points, ~350 s CPU.

| metric | proxy (eta = 0) | v1 | **v2** | gain vs v1 |
|---|---|---|---|---|
| training rms (scaled) | 3.78e-1 | 3.39e-4 | 6.50e-4 | — (different set) |
| certificate sup residual (physical) | 5.71e+1 | 1.59e-1 | **9.46e-2** | 1.7x |
| theta sup error vs MOL, t = 0 | 1.58 | 2.20e-3 | **2.19e-4** | **10x** |
| quote RMSE, t = 0, all q | 5.25e-1 | 9.24e-4 | **6.05e-5** | **15x** |
| quote RMSE, t = 0, boundary band | 5.68e-1 | 9.96e-4 | **6.31e-5** | **16x** |
| theta sup error vs MOL, t = 0.5 | 2.07e-1 | — | 4.26e-3 | — |
| quote RMSE, t = 0.5, all q | 7.79e-2 | 1.94e-3 | 1.68e-3 | 1.2x |

Residual localization after v2 (scaled, dense 40x400 grid):

| distance to nearest switching surface | sup abs(r) |
|---|---|
| [0, 0.1) | 2.95e-3 |
| [0.1, 0.5) | 3.57e-4 |
| [0.5, 1.5) | 4.10e-3 |
| [1.5, 7) | 4.11e-4 |

**The surface bin fell 1.39e-2 -> 2.95e-3 and is no longer the binding constraint** — the
domain corners are (4.10e-3). And the certificate's scaled sup (8.93e-3) now EQUALS the
training sup: the off-training gap the straddle points targeted is closed.

## Phase 1 v2 — d=1 logistic, 2 tiers, 2 atoms (rfqsim demand family, mu = -0.04, c = 0.02)

Same recipe, 4 surfaces (Q - z for z in {1, 2}), ~2200 -> ~2840 points, ~380 s CPU.

| metric | proxy | v1 | **v2** |
|---|---|---|---|
| certificate sup residual (physical) | 1.06 | 6.65e-2 | **5.18e-2** |
| theta sup error vs MOL, t = 0 | 3.22e-1 | 6.73e-3 | **5.82e-3** |
| quote RMSE, t = 0 | 2.51e-2 | 8.48e-4 | 9.22e-4 |
| quote RMSE, t = 0.5 | 7.30e-3 | 2.16e-3 | 2.09e-3 |

Localization: surface bin 2.50e-3, interior 9.44e-5 — surfaces suppressed here too, and
the certificate/training gap closed. But **the MOL quote metrics are flat vs v1**, and a
polish phase confirmed convergence at that level. Honest read: the logistic proxy defect
was already ~30x smaller than the exponential one, and with bounded intensities the
fill-weighted metric de-emphasizes the deep-inventory surface region — v1 was never
surface-limited *in this metric*. The v2 gains here show up in the certificate and the
residual localization, not the quote RMSE.

## What is now binding (both scenarios)

The t = 0.5 slice. A 60-iteration polish phase on the exponential scenario moved
theta sup 4.21e-3 -> 4.26e-3 and quote RMSE 1.79e-3 -> 1.68e-3 — **flat**, which rules
out undertraining and points at the ansatz. Suspect: the global (1 - t/T) prefactor
enforcing the terminal condition also throttles kink amplitude at mid-horizon, where the
censoring correction is largest relative to the remaining horizon. Documented fix (not
built): a separate kink-amplitude head with its own terminal enforcement, so the
prefactor stops scaling all capacity uniformly; corners want corner-band collocation.

Context for the intended use: both scenarios sit 50-1000x below 9-bin quote resolution,
so for oracle-label generation this is already well inside tolerance.

## Caveats

- The certificate is a **grid-sup estimate, not a proof**: the residual is discontinuous
  in q across the switching surfaces; the grid straddles them by construction, but no
  claim of uniformity between grid points is made.
- Certificate remains conservative vs realized error (9.46e-2 bound vs 2.19e-4 realized
  theta sup at t = 0) — expected for a sup-norm contraction bound.
- v1 numbers are quoted from the earlier report for comparison; the v1 network (no kink
  features) no longer exists in the code, so the v1 demo scripts were removed rather than
  left broken against the current API.

## Section-6 2D benchmark: PINN vs closed form vs exact HJB (+ MC correction)

Merged mmquote (vendored verbatim) with hjbpinn via `hjbpinn/bridge.py`; cross-stack agreement at the
Section-6 spec: Hamiltonian 3.9e-16 rel (Newton-on-x vs Lambert-W), A(t) 2.6e-12 abs, B(0) vs tight ODE
5.1e-14 / 2.1e-8 rel (quadrature / Euler), q=0 quotes identical to 6dp.

Exact reference: `validate.mol_solve_nd` (jitted RK4, arbitrary d) cross-checked against `mmquote.solve_fd_hjb`
(independent Euler impl) at 2.6e-5 rel on a shrunken spec; full 25x97 solve, 71 slices, step-halving 8.4e-7 EUR.

PINN (23 features incl. 16 surface kinks, widths 48x48, 5 phases, 2 RAR rounds, final N=3392):
- theta vs exact: t=0 sup 34,257 -> 147 EUR (rms 8,866 -> 20); never worse than ~50x better on any slice;
  worst slice sup 644 EUR at t=6.5; eta(T)=0 structural (3e-12).
- quotes at t=0: inner-band median 1.051 -> 0.013 cents, boundary median 1.285 -> 0.018, sup 29.2 -> 2.4.
- certificate (fresh grid): scaled sup 4.96e-1 -> 2.74e-2 (18x); physical 6.2e3 EUR/day; generalization gap
  cert/train = 10.2x (surface manifolds undersampled at fixed straddle budget — known limitation).

MC correction (Dynkin policy evaluation, `hjbpinn/mc.py`): v_pi = theta_ans + E[int r ds]; validated vs exact
theta at d=2 (v <= theta at every probe). Gaps at (0,0): proxy 94.5 +- 62.8; PINN 13.2 +- 7.9 EUR / 7 days.
Estimator noise scales with the residual: +-8 EUR at 24 paths for the PINN policy.

CRN-paired PnL (400 paths): 85.0k/85.4k/85.7k +- ~83k raw (matches paper scale); paired objective diffs all
|t| < 1 with SE ~660 EUR — even paired realized PnL cannot resolve ~10-100 EUR policy gaps (needs O(1e5)
paths); the Dynkin estimator is ~4 orders of magnitude more sample-efficient.

d=5 (25^5 = 9.8e6-state lattice infeasible): trains in ~3.5 min; sup-certificate nearly vacuous at 85 iters
(honest negative — surface manifolds are 4-D); MC instrument still sharp: PINN value estimate 20x more
self-consistent (|theta-v| 211 vs 4613 EUR), policy value +363 +- 279 vs proxy.

Notebook: `notebooks/section6_pinn_benchmark.ipynb` (executed; pre-registered metrics, training-quality
battery: phase table, generalization gap, region decomposition, residual-vs-surface-distance, residual
heatmaps, per-slice error curves, terminal check). Artifacts under `states/`. Known bug fixed during the run:
early diagnostics evaluated raw MLP without the (1-t/T) prefactor; all reported numbers use `network.eta`.

## H200 scaling programs (built + CPU-smoke-validated; sized for the GPU)

New modules (every code path exercised in-container at reduced scale):
- `hjbpinn/adversarial.py` — residual_tq (differentiable pointwise residual, verified vs residual_fn at
  1.3e-16), streamed mega-pool RAR (chunked; GPU knob 2e6/round), vmapped projected surface ascent with
  two-sided kink refinement, certificate_v2. Smoke: ascent at 8 starts/surface already beat the shipped
  4800-pt certificate (3.47e-2 > 2.74e-2); a 60k pool found 5.38e-2 — the shipped certificate was an
  underestimate; certification is sampling-limited, which is the H200's job. One adversarial round + 6
  GN iters collapsed the ascent sup 3.47e-2 -> 4.9e-3 (mechanism validated).
- `hjbpinn/optimizer.gauss_newton_gpu` — device-resident GN/LM: chunked J assembly, single fp64 DGEMM Gram
  (tensor cores), on-device Cholesky, scalar-only host syncs; sized for N=P=5e4 (20 GB J + 20 GB Gram).
- `hjbpinn/validate.mol_solve_nd_scan` — lax.scan lattice RK4 with snapshot-exact segmentation; matches the
  loop solver to 9.1e-13. d=3 exact solves run here (17^3 in 154 s CPU); H200: d=4 seconds, d=5 minutes,
  d=7 exceeds HBM.
- `hjbpinn/gpu_sim.pnl_lockstep` — one lax.scan, all policies in one carry (CRN by construction),
  counter-based PRNG, strided gathers, generic d, q0_idx for gap maps. Validated vs the host-loop sim
  (fills/spread within MC error; segment invariant exact; paired diffs consistent with the Dynkin
  prediction). ~175x per-path faster than the host loop on CPU alone.
- `tests/run_h200.py` — stages gap / ladder / pnl with smoke and H200 configs, HBM/flops sizing notes.

Validator ladder extended: d=3 exact lattice (Q=50k box) + grid-free MC: v = 69,138 +- 307 <= theta = 71,292;
proxy gap 2,154 +- 307 (significant — limits-near-everywhere regime, where the closed form should degrade).

Open check (pre-registered for the H200 run): lockstep paired optimal-pinn objective = -270 +- 114 and
-340 +- 137 at two independent seeds — the lattice-optimal policy underperforming the PINN policy is
consistent with near-T staleness of 71-slice piecewise-constant quote tables; stage_pnl prescribes n_t=283
tables + 1e6 paths to resolve it.
