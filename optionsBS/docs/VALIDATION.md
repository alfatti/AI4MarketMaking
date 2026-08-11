# spotmm — Validation Log

All numbers measured in this build (CPU, float64). Headline configuration:
BBG-transposed book (3 calls + 9.5/10/10.5 butterfly), box nbar = (2,2,2,4),
T = 0.006, gamma = 3e-5, eta = 1, mu = 0.

## Instruments (tests 1-3 of tests/test_spotmm.py)

- BS AD delta == closed form to machine zero.
- CN local-vol pricer, sigma == const degeneracy: price 4.2e-5 / delta
  6.8e-6 vs BS at production grid (2e-4 at test grid).
- Butterfly delta flips sign inside the band in both models
  (BS: +0.108 -> -0.092 across S in [9.3, 10.7]; CEV similar, flip point
  shifted by the skew).

## Exact (t, S, n) lattice — the full unsimplified problem

- **Analytic anchor** (eta = 0, no limits, periodic mini-lattice; replaces
  optmm's Remark-6 as the structural identity): max error 5.6e-9 on a
  1,055,328 scale — machine precision.
- Headline solve: M = 3,773 x 41 S-nodes, ~110 s; v(0, S0, 0) = 829,904
  vs the riskless bound 1,055,328 (risk costs ~21% of gross at
  gamma = 3e-5 — the recorded balance calibration).
- RK4 step-halving (stable pair): 0.18 EUR (2e-7 relative). Note: nt = 50
  violates the jump-Lipschitz RK4 bound dt * sum(Lambda) <= 2.78 and is
  excluded by design.
- eta = 0 **with** the box: gaps to the analytic value all nonnegative and
  face-localized (294k / 250k at face distance 0 / 1 on the headline box)
  — the box-truncation mechanism, here a feature of the true problem
  rather than a representation artifact.
- LV (CEV beta = 0.5): v(0, S0, 0) = 847,736; same pipeline verbatim.

## The flip reversal — corrected statement, exact on the lattice

A long position in the sign-flip instrument does NOT reverse its own
skew (you shed what you hold on both sides; an earlier framing in the
optmm whitepaper figure caption is wrong and flagged). The correct dual
statements, verified as exact lattice facts:

- ATM-call skew at long-3-butterfly inventory: -1.35 / -0.15 / +4.03
  cents at S = 9.3 / 10.0 / 10.7, **crossing zero at S = 10.0** (BS) and
  at 10.0 +/- one grid cell (LV) — hold the flip instrument and your
  *vanilla* quoting reverses through the flip.
- Butterfly skew at long-2-ATM inventory: -0.37 / -0.04 / +1.07 cents,
  crossing at S = 10.0 — hold vanillas and the *flip instrument's* skew
  reverses.

## Simulator (A.1 deliverables)

- **Value identity vs the exact lattice of the full problem**: sim
  830,921 +/- 1,957 vs 829,904 — gap **+0.52 SE** (8k paths x 1200 steps).
- **eta-parabola**: Var(eta)/Var(1) = [0, 0.0625, 0.25, 0.5625, 1.0] to
  four decimals against eta^2 (CRN, fixed policy).
- q^S = -(1 - eta) Delta^pi materialized: at eta = 0.5, mean turnover
  8.6e6 contracts/path, terminal q^S mean ~0.

## PINN (CPU smoke, 14k Adam steps total)

- vs the exact lattice on all 1,125 admissible states: median 52,781 EUR
  (6.4% of the 830k center), max 80,842; loss still descending — full
  budget is the GPU run.
- Flip reversal reproduced: ATM-skew crossing at S = 9.9 (lattice 10.0).
- **Dynkin closure, corrector earning its keep**: raw v_theta(0) is off
  by 44k, yet v_theta(0) + E[int R dt] = 833,120 vs simulated 830,098
  under the lattice policy — gap **-0.97 SE**. Precision note: paths ran
  under the lattice policy while R is v_theta's sup-residual, so the
  estimate upper-bounds the identity for mismatched policies (observed
  consistent, +0.97 SE above the sim mean); the exact-identity variant
  simulates under the PINN-greedy policy and is the notebook/GPU
  follow-up.

## Suite

16/16 (tests/test_spotmm.py + ported test_hamiltonian.py), ~2 min CPU.

## Policy evaluation (notebooks/performance_eval.ipynb; tests/test_evaluate.py)

Regret machinery: freeze any policy's quotes and solve the linear
backward equation on the same lattice — exact pricing of implementable
(stride-refreshed) policies. Gates: eta=0 periodic + static policy
reproduces the analytic value to 0.00e+00; fixed point converges at
second order in dt (74.5 -> 17.9 EUR under halving, ratio 4.2), giving
the ~18 EUR discretization floor at nt=200; dominance and baseline
ordering hold.

Headline (r = 8 refresh, smoke PINN): exact-policy regret 157 EUR (the
refresh cost — the resolution statement); PINN 3,502 EUR = 98.9% capture
of the achievable improvement over the static desk and 88.8% over the
frozen-anchor desk; frozen-anchor 31,166; static 315,343. Regret
concentrates at face-distance-0 corner states with large opposing
inventories. Quotes: median 0.085c raw; occupancy-weighted mean 0.123c
under BOTH the exact-policy and PINN-policy visit distributions (the
weighting choice resolved by measurement: they coincide). Fill
distortion 0.7% of lambda; ~1.8 differing fill decisions per episode of
~32 fills. CRN paired MC regret 4,181 +/- 677 vs lattice 3,344 net:
+1.24 SE agreement.

Suite: 18/18 (~4 min CPU).
