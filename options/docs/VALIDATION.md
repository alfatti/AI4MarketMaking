# optmm — Validation Log

Every number below was measured in this build (CPU, float64, jax 0.11).
Reproduce with the referenced test or script.  Values in EUR unless noted.

## M1 — Hamiltonian (tests/test_hamiltonian.py, 6/6)

- Logistic FOC residual via the Lambert-W semi-closed form: < 1e-14 relative
  to the local scale Lambda(delta*), for k in [100, 1000], p in [-3, 3],
  including the asymptotic-branch switch region (L = 30).
- **Bug caught by the brute-force test**: a plain Newton in delta-space takes
  O(|alpha + k p|) iterations through the flat logistic tail (k = 277,
  p = -0.3: returned -0.083 vs true -0.018 after 60 iterations).  Fix: the
  FOC in x = alpha + k delta is x - b - 1 = e^{-x}, b = alpha + k p, with the
  exact root x* = b + 1 + W(e^{-(b+1)}).
- Exponential closed form, envelope H'(p) = -Lambda(delta*), convexity,
  delta_inf binding, BBG calibration (33.2% fill at mid, 69.0% one vol point
  through): all pass.
- Lambda evaluation rewritten as lam * sigmoid(-x); survives x ~ 700 where
  Lambda underflows below the smallest float64 subnormal (correctly to 0).

## M2 — Heston COS pricer (tests/test_heston.py, 6/6)

- **Bug caught by the BS-limit test**: catastrophic cancellation in
  (beta - d)/xi^2 at small xi.  Fix: exact identity
  beta - d = -xi^2 (iu + u^2)/(beta + d) plus a guarded log(1 + w).
- CF vs 50-digit mpmath reference: <= 2.3e-13 at xi = 0.2 and xi = 1e-6.
- BS limit (xi = 1e-8, theta = nu0): <= 2e-8 absolute.  At xi = 1e-6 the
  residual 3e-8..2e-7 errors are the *genuine* O(xi) model gap (sign flips
  with moneyness like a rho-xi skew) — converged in N and L.
- Put-call parity: 1.8e-15.  AD vega = FD vega to 1e-6 rel.  MC (200k paths,
  full-truncation Euler) agrees within 4 SE + O(dt).
- Cumulants by AD of log phi at u = 0: c1 to 1e-13, c2 to 2.6e-9 of exact in
  the deterministic-variance limit.
- Book vs paper legends: K=12,T=1 price 0.0613 / vega 0.559 (paper 0.06 /
  0.54); K=8,T=1 price 2.0597 / vega 0.4167 (paper 2.06 / 0.41).  ATM T=1
  trade moves Vpi by 10.7% of Vbar; K=12,T=1 by 45.6%.

## M3 — reduced2d (tests/test_reduced2d.py, 6/6)

- Full 20-option book, 720x31x201: v(0, nu0, 0) = 164,028; carry asymmetry
  v(+Vbar) = 54,663 > v(-Vbar) = 54,147 (long-vega lean, correct sign).
  Diffusion CFL 0.003; monotonicity number dt * sum Lambda(delta*) = 0.084.
- Symmetric market (a_P = a_Q): v nu-independent and even in Vpi to <= 1e-8
  relative; bid(Vpi) = ask(-Vpi) to 1e-10.
- **Remark-6 equivalence**: mv-hedge at (xi, rho, c = rho) vs delta-neutral
  at xi sqrt(1 - rho^2): max |vA - vB| = 4.4e-11 on a 1.7e5 scale (machine
  precision).  With a_P != a_Q the gap is real but tiny: ~0.08 EUR at the
  Vpi extremes over the 0.3-day horizon — m(c) is essentially the entire
  hedge story at this timescale.
- Quote monotonicity in Vpi (all 40 channels) and the amplified-carry
  directional lean: pass.

## M4 — lattice vs reduced2d (tests/test_lattice_vs_reduced2d.py, 5/5)

Design note: the stage-1 slab is an **infinite diagonal band** in gross
inventory; any finite representation box truncates genuinely admissible
states, and the damage propagates inward through the jump coupling.  A first
naive comparison "saturated" at 8.6k EUR — that was box pollution, not
interpolation error.  The clean design uses a commensurable sub-book
(w = (1.0e6, 1.5e6) via z-override) so the reduced2d code on the 41-atom
grid is *exact in Vpi* (checked: NV = 41 vs NV = 81 agree to 1e-9 EUR).

- Euler time error of the atom reference is **exactly first order**: diffs to
  nt = 11520 give ratios 2.33 and 3.00 (theory 7/3 and 3).
- Pure Vpi-interpolation error of reduced2d (non-divisor NV, matched nt):
  max 153 -> 50 -> 9.8 -> 3.5 EUR over NV = 97 / 193 / 385 / 769
  (center scale 26,489).
- Box-truncation pollution at a fixed evaluation set: 10,441 EUR unpadded ->
  9.2 EUR with +4 padding -> reference-floor 5.4 EUR at +8.
- Lattice vs exact reduction at n = 0: 0.28 EUR on 26.5k (1e-5 relative),
  limited by the reference's dt, not the lattice.  RK4 step-halving
  (600 vs 1200): 2.6e-7 EUR.

## Simulator (tests/test_sim.py, 3/3)

- **Value identity** (validates every sign convention at once): sim mean
  26,566.6 +/- 116.7 vs v(0, nu0, 0) = 26,491.0 — gap +0.65 SE
  (20k paths x 1200 steps, 12 s).
- **Hedge parabola**: Var[M(c)]/Var[M(0)] = [0.9954, 0.7517, 1, 1.7403,
  2.9725] vs m(c) = [1, 0.75, 1, 1.75, 3] for c = [-1, -.5, 0, .5, 1]
  (all within 1%); fitted vertex c* = -0.5023 vs rho = -0.5.

## Stage 2 — box limits (tests/test_stage2_box.py, 4/4)

- Face masking and own-inventory quote monotonicity: pass.
- **Reduction death (negative control)**: states (6,-4) and (0,0) share
  Vpi = 0 but differ by 5,417 EUR — under the box the value is genuinely not
  a function of Vpi, unlike the slab case where the collapse held to the
  reference floor.

## Stage 3 — residual certification (tests/test_stage3_consistency.py, 2/2)

- The stage-3 assembly (S in state, rho cross term, Greeks callable) on an
  S-independent value with frozen Greeks equals the stage-1 residual to
  < 1e-12 relative, for arbitrary network parameters (dS terms vanish under
  AD).  With S-dependent Greeks the residuals differ (cross terms live).

## Dynkin corrector (tests/test_corrector.py, 4/4)

- **Sign bug caught by the triangle test**: the identity is
  V^pi = v~(0) **+** E[int R~ dt] (an early draft had minus), and the
  residual must pair (v[k+1] - v[k])/dt with spatial terms on slice k+1 —
  then the solver's own optimal history has machine-zero residual.
- Triangle on a deliberately perturbed v~ (+1500 EUR smooth bump):
  linear-PDE policy value 26,490.9; direct sim 26,505.4 +/- 116.2;
  corrector estimate agrees within tolerance; suboptimality and
  optimal-policy-recovery checks pass.

## PINN stage 1 — CPU smoke (scripts/run_stage1_pinn.py --smoke)

6000 Adam steps, batch 256, hidden (64, 64, 64), 245 s CPU; pure MLP, no
w^T q feature, no constraint features:

- **Hyperplane discovery**: median |cos(grad_n v, w)| = 1.0000; 99% of
  interior samples above 0.99.  The network recovered grad_n v || w untold.
- Value error vs the reduced2d reference (NV = 801): median 1,317 EUR, max
  6,722 EUR on a 164k scale; center error 0.8%.  Loss RMS 2.7e8 -> 5.1e6
  and still descending — full-budget training is the H200 job.

## Notebook run (notebooks/stage1_pinn_train_and_analysis.ipynb)

Executed on the CPU profile, resuming the smoke checkpoint + 500 steps:

- Value collapse: median |err| 997 EUR / max 7,997 EUR on the 164k scale;
  within-bin spread at fixed Vpi (spurious gross-composition dependence):
  1,923 EUR median.
- Hyperplane discovery: median |cos| = 0.99999, 99.0% of samples > 0.99.
- **Quote level** (what the desk consumes): ATM ask along the ray, median
  |err| 0.016 cents, max 0.030; quote collapse across 196 random gross
  compositions at Vpi = 0.4 Vbar: 0.226c +/- 0.014c (2D ref 0.214c).
- Residual pool: median 3.1e6 / max 3.9e7 EUR/yr; 25 adversarial ascent
  steps amplify the pool-worst by x1.89 — the pool max underestimates the
  sup, as expected; the real certificate is stage-2 machinery.

## Stage 2 — dimension ladder + box PINN
## (notebooks/stage2_stage3_ladder_and_frontier.ipynb; tests/test_stage3_pipeline.py)

- Exact lattice ladder (box limits, ATM-ish sub-books, CPU):
  N=3: M=1,573 states, 23.5 s, step-halving 8.3e-4, v(0,nu0,0) = 39,872;
  N=4: M=5,103, 37.3 s, 6.7e-3, 49,311;
  N=5: M=16,807, 82.3 s, 9.8e-3, 54,872.
  The measured cardinality curve the PINN must eventually beat.
- Box-admissibility PINN (residuals are now admissibility-generic; Slab
  default preserves all stage-1 behavior): trained at N=3, validated on all
  891 admissible lattice states — a true error vs an exact reference:
  median 1,726 EUR, max 12,239 (center 39,872).  Quote level vs exact:
  median 0.021 cents, max 0.085.
- Reduction death, quantified over the whole box: exact-value spread at
  fixed Vpi (median over bins) = 75,832 EUR.

## Stage 3 — frontier (no exact reference)

- Universe: 3 vanillas + tight 9.75/10.25 call spread; live COS-priced
  (S, nu) vega grids (spread vega flips sign: +0.165 at S=9.3 to -0.207 at
  S=10.7; interpolant matches direct COS to 6e-5).  Horizon T = 0.006 so S
  traverses the flip region; box limits; rho cross term live.
- Frozen-Greeks regression (S-featured net, signed-w 2D reference): median
  |err| 1,555 EUR on a 267,574 scale, and **emergent S-independence** —
  median std of v_theta over S in [9.4, 10.6] is 717 EUR with S as an
  input and nothing telling the network S is irrelevant.
- Policy through the sign flip: the spread's quote skew at a long-spread
  inventory reverses as S crosses the flip — the live-Greeks signature at
  the level the desk consumes.
- **Dynkin closure** (greedy policy attains the sup => linear residual =
  HJB residual; machinery pre-validated at stage 1): 800 CRN paths x 1200
  steps: E[reward] = 263,690 +/- 2,585 vs v_theta(0) + E[int R dt] =
  265,111 - 3,568 = 261,543; closure gap +2,147 EUR = **+0.83 SE**.
  34.2 fills/path, rms live Vpi 1.33e6.
- Residual pool median 4.5e6 / max 7.3e8 EUR/yr; 20-step ascent amplifies
  the pool-worst x1.57 (pool max underestimates the sup, as at stage 1).

## Open items (H200 / future)

- Full-budget trainings on H200 (all three stages; the GN-CG path remains
  compiled-but-unexercised); rigorous residual certificates beyond the
  ascent preview; stage-3 CRN PnL vs the rolling-refresh 2D incumbent;
  larger-N ladder points; tensor-train lattice methods as the honest
  competitor at large N.
