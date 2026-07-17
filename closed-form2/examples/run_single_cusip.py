"""v1 milestone: single-CUSIP end-to-end on synthetic data.

Pipeline: simulate an RFQ/fill log and a reference-price path from known
ground-truth parameters -> estimate {A_b, k_b, A_a, k_a, sigma} -> calibrate
gamma to a target spread -> solve the Riccati proxy -> print the quote ladder
vs inventory. Swap the synthetic block for a real RFQ log to go live.

Run:  python examples/run_single_cusip.py
"""

import numpy as np

from mmquote import (
    ExponentialIntensity,
    ModelConfig,
    build_single_cusip_engine,
    calibrate_gamma_to_spread,
    estimate_sigma,
    fit_exponential_intensity,
)

rng = np.random.default_rng(42)

# ------------------------------------------------------------------ ground truth
TRUE = dict(A_b=1.8, k_b=6.0, A_a=2.2, k_a=5.0, sigma=0.4)  # slightly asymmetric book
Z = 1.0               # quote size (e.g., 1 = reference notional unit)
TARGET_SPREAD = 0.50  # desired zero-inventory total spread (price units)

# ------------------------------------------------------------------ synthetic data
# RFQ log: quoted distance + Poisson fill count per observation window
n_obs, tau = 40_000, 0.05
deltas_b = rng.uniform(0.0, 0.6, n_obs)
deltas_a = rng.uniform(0.0, 0.6, n_obs)
fills_b = rng.poisson(tau * TRUE["A_b"] * np.exp(-TRUE["k_b"] * deltas_b))
fills_a = rng.poisson(tau * TRUE["A_a"] * np.exp(-TRUE["k_a"] * deltas_a))

# reference-price series for sigma
dt = 1.0 / 252
prices = 100 + np.cumsum(TRUE["sigma"] * np.sqrt(dt) * rng.standard_normal(30_000))

# ------------------------------------------------------------------ estimation
fit_b = fit_exponential_intensity(deltas_b, fills_b, tau)
fit_a = fit_exponential_intensity(deltas_a, fills_a, tau)
sigma_hat = estimate_sigma(prices, dt)

print("=== Estimated parameter set {A_b, k_b, A_a, k_a, sigma} ===")
print(f"  bid:  A = {fit_b.model.A:.3f} (true {TRUE['A_b']}),  "
      f"k = {fit_b.model.k:.3f} (true {TRUE['k_b']})   [{fit_b.n_fills} fills]")
print(f"  ask:  A = {fit_a.model.A:.3f} (true {TRUE['A_a']}),  "
      f"k = {fit_a.model.k:.3f} (true {TRUE['k_a']})   [{fit_a.n_fills} fills]")
print(f"  sigma = {sigma_hat:.4f} (true {TRUE['sigma']})")

# ------------------------------------------------------------------ gamma knob
def make(gamma: float):
    cfg = ModelConfig(gamma=gamma, z=Z, T=30.0, xi=0.0, ergodic=True)
    return build_single_cusip_engine(fit_b.model, fit_a.model, sigma_hat, cfg)

gamma_hat = calibrate_gamma_to_spread(make, TARGET_SPREAD)
engine = make(gamma_hat)
print(f"\n=== gamma calibrated to target spread {TARGET_SPREAD} ===")
print(f"  gamma = {gamma_hat:.5f}")
print(f"  ergodic Gamma = {engine.solution.Gamma[0, 0]:.5f}, "
      f"A_erg = {engine.solution.A_erg[0, 0]:.5f}, "
      f"B_erg = {engine.solution.B_erg[0]:+.5f}  (asymmetry carrier)")

# ------------------------------------------------------------------ quote ladder
print("\n=== Ergodic greedy quotes vs inventory ===")
print(f"{'q':>5} | {'delta_b':>8} {'delta_a':>8} | {'spread':>7} {'skew a-b':>9}")
for q in [-4, -2, -1, 0, 1, 2, 4]:
    qt = engine.quote(np.array([float(q)]))
    print(f"{q:>5} | {qt.delta_b:>8.4f} {qt.delta_a:>8.4f} | "
          f"{qt.delta_b + qt.delta_a:>7.4f} {qt.delta_a - qt.delta_b:>+9.4f}")

print("\nReading: long inventory -> bid backs off, ask tightens (negative skew);")
print("spread is inventory-invariant under the proxy (affine p, symmetric shift).")
