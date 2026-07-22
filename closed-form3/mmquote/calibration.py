"""Calibration: the estimated set {A_b, k_b, A_a, k_a, sigma} plus the gamma knob.

- Exponential intensity (A, k): Poisson MLE from an RFQ/fill log. With exposure
  time per delta level, log-intensity is linear in delta -> equivalent to a
  Poisson GLM; here a direct 2-parameter MLE.
- sigma: realized-vol estimator on the reference (composite) price series.
- gamma: not estimated. calibrate_gamma_to_spread() roots gamma so the model's
  zero-inventory spread matches an observed/target spread (liquidity-paper
  practice).

Endogeneity caveat: observed deltas come from someone's quoting policy. If delta
was not varied independently of demand conditions, k is biased. Check the spread
of observed deltas before trusting the slope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize

from .intensity import ExponentialIntensity


# ---------------------------------------------------------------- intensity fit

@dataclass(frozen=True)
class IntensityFit:
    model: ExponentialIntensity
    loglik: float
    n_fills: int
    exposure: float


def fit_exponential_intensity(
    deltas: np.ndarray,
    filled: np.ndarray,
    exposure_per_obs: float | np.ndarray,
) -> IntensityFit:
    """MLE of Lambda(delta) = A exp(-k delta) from (delta, fill) observations.

    Model: over an observation window with exposure tau at quoted distance delta,
    fills ~ Poisson(tau * A * exp(-k * delta)). `filled` is the fill count per
    observation (0/1 for per-RFQ rows under the thinning interpretation).

    Log-likelihood: sum_i [ n_i (log A - k d_i) - tau_i A e^{-k d_i} ] + const.
    """
    deltas = np.asarray(deltas, dtype=float)
    filled = np.asarray(filled, dtype=float)
    tau = np.broadcast_to(np.asarray(exposure_per_obs, dtype=float), deltas.shape)

    def negll(params: np.ndarray) -> float:
        logA, k = params
        if k <= 0:
            return np.inf
        lam = np.exp(logA - k * deltas)
        return float(np.sum(tau * lam) - np.sum(filled * (logA - k * deltas)))

    n_fills = float(filled.sum())
    total_tau = float(tau.sum())
    x0 = np.array([np.log(max(n_fills, 1.0) / total_tau), 1.0])
    res = minimize(negll, x0, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20_000})
    logA, k = res.x
    return IntensityFit(
        model=ExponentialIntensity(A=float(np.exp(logA)), k=float(k)),
        loglik=-float(res.fun),
        n_fills=int(n_fills),
        exposure=total_tau,
    )


# ---------------------------------------------------------------- sigma

def estimate_sigma(prices: np.ndarray, dt: float) -> float:
    """Realized volatility per sqrt(unit time) from a reference-price series."""
    prices = np.asarray(prices, dtype=float)
    increments = np.diff(prices)
    return float(np.std(increments, ddof=1) / np.sqrt(dt))


# ---------------------------------------------------------------- gamma knob

def calibrate_gamma_to_spread(
    make_engine,  # Callable[[float], QuoteEngine]: gamma -> configured engine
    target_spread: float,
    gamma_lo: float = 1e-6,
    gamma_hi: float = 1e2,
) -> float:
    """Root gamma so that the zero-inventory total spread matches target.

    Spread(0) = delta_b(0) + delta_a(0), evaluated on the ergodic engine.
    Monotone increasing in gamma (more risk aversion -> wider quotes).
    """

    def spread_err(gamma: float) -> float:
        engine = make_engine(gamma)
        q0 = np.zeros(1)
        qt = engine.quote(q0)
        return (qt.delta_b + qt.delta_a) - target_spread

    return float(brentq(spread_err, gamma_lo, gamma_hi, xtol=1e-10))
