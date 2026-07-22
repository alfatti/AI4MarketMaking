# mmquote

Closed-form market-making quotes for a single CUSIP, implementing the quadratic
value-function proxy of Bergault, Evangelista, Guéant & Vieira, *Closed-form
approximations in multi-asset market making* (arXiv:1810.04383), with the repo
shaped so that the paper's named generalizations bolt on rather than force a
rewrite.

## Scope

**v2 adds the Section 5/6 features**: logistic intensities (RFQ arrival x hit-rate,
with a semi-closed Hamiltonian via a scalar Newton solve), price drifts mu (carried by
B(t)), distributed request sizes (Model B: only the moments E[z], E[z^2] enter), per-size
quoting, a d-dimensional finite-difference HJ benchmark (`fd_hjb.py`), and the full
two-asset Section 6 replication (`notebooks/section6_replication.ipynb`) validating the
closed forms against the exact solve. Deferred: tiering, fixed costs (V~ hook wired),
MMPP regimes.

## v1 scope (all paper assumptions)

- Single asset (d = 1), driftless reference price dS = σ dW
- Exponential intensity Λ(δ) = A e^{−kδ} per side (Avellaneda–Stoikov)
- Model B objective by default (ξ = 0: expected PnL − ½γ∫q'Σq); Model A (CARA,
  ξ = γ) available through the same Hamiltonian interface
- Quadratic proxy θ̌(t,q) = −q'A(t)q − q'B(t) − C(t); matrix-tanh Riccati for
  A(t), backward integration for B(t), ergodic limits A → ½√γ Γ
- Greedy quotes δ* from the proxy via Λ⁻¹; single size z; inventory cap as a
  policy overlay (proxy derived in the Q → ∞ limit)

## Estimated vs chosen parameters

| Estimated (calibration.py) | Chosen (config.py) |
|---|---|
| A_b, k_b, A_a, k_a — Poisson MLE on (δ, fill) log | γ — calibrated to a target spread |
| σ — realized vol on the reference series | z, Q_limit, T, ξ, ergodic flag |

## Layout

```
mmquote/
  intensity.py     Seam 1: abstract IntensityModel; ExponentialIntensity v1.
                   Later: logistic-hit-rate × arrival, tier/size-indexed.
  hamiltonian.py   H_ξ(p), derivatives, δ*(p); closed form for exponential,
                   numeric sup fallback for any IntensityModel. ξ lives here.
  riccati.py       Seam 2: matrix-shaped (A, B, C) system exercised at d = 1.
                   B is wired in even when zero — it is the asymmetry/drift
                   carrier where κ(λ^a−λ^b) lands later.
  quotes.py        Greedy quote map; separate bid/ask paths; inventory limits.
  calibration.py   Intensity MLE, σ estimator, γ-to-spread root-finder.
  config.py        Policy knobs + build_single_cusip_engine() wiring.
examples/
  run_single_cusip.py   v1 milestone: synthetic log → calibrate → solve → quote.
  fd_hjb.py        d-dim finite-difference HJ benchmark (sizes, limits, drift).
notebooks/
  behavioral_tests.ipynb       9-test behavioral suite (comparative statics,
                               optimality vs exact HJB, closed-loop simulation).
  section6_replication.ipynb   Two-asset Section 6 replication: Figs 1-11 program,
                               error metrics, behavioral sign checks (6/6 PASS).
tests/
  test_single_cusip.py  Analytic cross-checks vs the paper's closed forms + Section 5/6
                        feature tests (23 total).
```

## Quick start

```bash
pip install -e ".[dev]"
pytest tests/ -q
python examples/run_single_cusip.py
```

```python
import numpy as np
from mmquote import ExponentialIntensity, ModelConfig, build_single_cusip_engine

lam = ExponentialIntensity(A=2.0, k=5.0)
cfg = ModelConfig(gamma=0.1, z=1.0, T=30.0)          # Model B, ergodic
engine = build_single_cusip_engine(lam, lam, sigma=0.3, config=cfg)
engine.quote(np.array([2.0]))                         # quotes at q = +2
```

## Extension roadmap (papers' named directions)

Nearly free (seams already in place):
- **Multi-asset**: pass d > 1 arrays and a full Σ to `ProxyCoefficients` /
  `solve_riccati`; the solver is already matrix-form.
- **Model A**: set ξ = γ in `ModelConfig`.
- **Alternative demand curves**: subclass `IntensityModel` (numeric Hamiltonian
  path already works); e.g., logistic hit-rate × RFQ arrival from the liquidity
  paper (Bergault et al., *Liquidity Dynamics in RFQ Markets*).

Interfaces stubbed, machinery deliberately not built:
- **Price drift / κ(λ^a − λ^b)**: enters the B(t) ODE as an extra linear term.
- **Client tiering / multiple sizes**: index intensities by (tier, size);
  multiplies the α-coefficient set (paper Section 5).
- **MMPP liquidity regimes**: turns θ into a state-indexed vector coupled by
  the generator Q — a different solver structure (coupled Riccatis). Deferred;
  the quoting interface would take a state distribution π.
- **Fixed transaction costs**: Section 5 of the paper.

## Model-risk notes (read before trusting quotes)

Exponential Λ is a convenience, not a law; the quadratic proxy degrades at
large |q| (exactly where precision matters most); parameters are assumed
stationary; fill data is endogenous to historical quoting policy; the driftless
price ignores adverse selection — monitor post-fill markouts. Intended use:
assistive / warm-start / decision-support quoting with a human in the loop,
recalibrated as fill data accumulates.
