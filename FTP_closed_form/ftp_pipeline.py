# Fix the module by avoiding escaped triple quotes inside the code string.
from textwrap import dedent

code = '''
# rfq_quotes_pipeline.py
# -------------------------------------------------------------
# Closed-form(ish) pipeline for optimal quotes in an RFQ market
# using a quadratic approximation for the value function (A,B,C)
# as described in the user's methodology (one state, inventory caps).
#
# Dependencies: numpy only.
#
# Author: ChatGPT (GPT-5 Thinking)
# -------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, Dict
import numpy as np

Array = np.ndarray

# --------------------------
# Utilities
# --------------------------

def _polyfit_quadratic(x: Array, y: Array) -> Tuple[float, float, float]:
    """Fit y ~= c0 + c1 x + c2 x^2 (least squares). Returns (c0, c1, c2)."""
    X = np.vstack([np.ones_like(x), x, x**2]).T
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    c0, c1, c2 = coeffs
    return float(c0), float(c1), float(c2)


def _rk4_step(fun, t, y, h):
    """Classic RK4 step."""
    k1 = fun(t, y)
    k2 = fun(t + 0.5*h, y + 0.5*h*k1)
    k3 = fun(t + 0.5*h, y + 0.5*h*k2)
    k4 = fun(t + h, y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)


# --------------------------
# S-curve definitions
# --------------------------

@dataclass
class Scurve:
    """A smooth, decreasing fill-probability curve and its domain."""
    f: Callable[[Array], Array]               # f(δ) in (0,1)
    name: str = "custom"
    delta_min: float = -10.0
    delta_max: float = 10.0

    def grid(self, n: int = 2001) -> Array:
        return np.linspace(self.delta_min, self.delta_max, n)


def logistic_scurve(a: float, b: float,
                    delta_min: float = -10.0, delta_max: float = 10.0) -> Scurve:
    """Logistic fill prob: f(δ) = 1 / (1 + exp(a + b δ)), with b>0 typically."""
    def f(delta: Array) -> Array:
        x = a + b*delta
        # numerically stable sigmoid
        out = np.empty_like(x, dtype=float)
        mask = x >= 0
        out[mask] = np.exp(-x[mask]) / (1.0 + np.exp(-x[mask]))
        out[~mask] = 1.0 / (1.0 + np.exp(x[~mask]))
        return out
    return Scurve(f=f, name=f"logistic(a={a},b={b})", delta_min=delta_min, delta_max=delta_max)


# --------------------------
# Quadratic fit for Hamiltonian H(p) near p=0
# --------------------------

def hamiltonian_from_scurve(s: Scurve, p: float, n_delta: int = 2001) -> float:
    """Compute H(p) = max_δ f(δ) * (δ - p) via dense grid search on δ."""
    delta = s.grid(n_delta)
    vals = s.f(delta) * (delta - p)
    vals = np.nan_to_num(vals, neginf=-1e300, posinf=1e300)
    j = np.argmax(vals)
    return float(vals[j])


def quadratic_alpha_from_scurve(s: Scurve,
                                p_max: float = 0.5,
                                n_p: int = 41,
                                n_delta: int = 2001) -> Tuple[float, float, float]:
    """
    Fit H(p) ~= α0 + α1 p + 1/2 α2 p^2 near p=0.
    We sample p in [-p_max, p_max], compute H(p) by grid search over δ,
    then least-squares fit a quadratic.
    Returns (α0, α1, α2).
    """
    p_grid = np.linspace(-p_max, p_max, n_p)
    H_vals = np.array([hamiltonian_from_scurve(s, p, n_delta=n_delta) for p in p_grid])
    c0, c1, c2 = _polyfit_quadratic(p_grid, H_vals)   # H ~= c0 + c1 p + c2 p^2
    alpha0 = c0
    alpha1 = c1
    alpha2 = 2.0 * c2   # because 1/2 α2 p^2 ↔ c2 p^2
    return alpha0, alpha1, alpha2


# --------------------------
# Model parameters
# --------------------------

@dataclass
class ModelParams:
    # Market/flow
    lambda_b: float
    lambda_a: float
    z: float
    sigma: float
    kappa: float
    gamma: float
    # Horizon and inventory caps
    T: float
    qbar: float
    # Discretization
    n_time: int = 2000


@dataclass
class AlphaCoeffs:
    # α0, α1, α2 for bid and ask
    alpha_b: Tuple[float, float, float]
    alpha_a: Tuple[float, float, float]


# --------------------------
# ODE system for A, B, C (quadratic value function coefficients)
# --------------------------

def _dABC_dt(t: float, Y: np.ndarray, m: ModelParams, a: AlphaCoeffs) -> np.ndarray:
    """
    Right-hand side of the ODE system for Y = [A, B, C].
    Uses the Δ-notation simplification: Δ^b_{i,k} = α^b_i z^k, Δ^a_{i,k} = α^a_i z^k.
    """
    A, B, C = Y
    z = m.z

    # Unpack alphas
    a0b, a1b, a2b = a.alpha_b
    a0a, a1a, a2a = a.alpha_a

    # Build Δ's needed
    D2b1 = a2b * z**1
    D2a1 = a2a * z**1

    D1b1 = a1b * z**1
    D1a1 = a1a * z**1

    D2b2 = a2b * z**2
    D2a2 = a2a * z**2

    D0b1 = a0b * z**1
    D0a1 = a0a * z**1

    D1b2 = a1b * z**2
    D1a2 = a1a * z**2

    D2b3 = a2b * z**3
    D2a3 = a2a * z**3

    # Short-hands
    Lb, La = m.lambda_b, m.lambda_a

    # A' equation
    dA = 2.0 * (Lb * D2b1 + La * D2a1) * A**2 - 0.5 * m.gamma * (m.sigma**2)

    # B' equation
    dB = 2.0 * (Lb * D1b1 - La * D1a1) * A \
         + 2.0 * (Lb * D2b2 - La * D2a2) * A**2 \
         + m.kappa * (La - Lb) \
         + 2.0 * (Lb * D2b1 + La * D2a1) * A * B

    # C' equation
    dC = (Lb * D0b1 + La * D0a1) \
         + (Lb * D1b2 + La * D1a2) * A \
         + (Lb * D1b1 - La * D1a1) * B \
         + 0.5 * (Lb * D2b3 + La * D2a3) * A**2 \
         + 0.5 * (Lb * D2b1 + La * D2a1) * B**2 \
         + (Lb * D2b2 - La * D2a2) * A * B

    return np.array([dA, dB, dC], dtype=float)


def integrate_ABC(params: ModelParams, alpha: AlphaCoeffs) -> Dict[str, np.ndarray]:
    """Backward-integrate A,B,C from T to 0 using RK4 on a uniform grid."""
    T = params.T
    N = params.n_time
    t_grid = np.linspace(T, 0.0, N+1)  # backward
    h = - (T / N)                       # negative step

    # Terminal conditions
    Y = np.array([0.0, 0.0, 0.0], dtype=float)   # [A(T), B(T), C(T)]

    A = np.empty_like(t_grid)
    B = np.empty_like(t_grid)
    C = np.empty_like(t_grid)

    for i, t in enumerate(t_grid):
        A[i], B[i], C[i] = Y
        if i == len(t_grid) - 1:
            break
        Y = _rk4_step(lambda tt, yy: _dABC_dt(tt, yy, params, alpha), t, Y, h)

    # Reverse arrays to increasing time (0 -> T) for convenience
    t_out = t_grid[::-1].copy()
    A_out = A[::-1].copy()
    B_out = B[::-1].copy()
    C_out = C[::-1].copy()

    return dict(t=t_out, A=A_out, B=B_out, C=C_out)


# --------------------------
# Quote computation
# --------------------------

def _p_b(A: float, B: float, q: float, z: float) -> float:
    return 2.0*A*q + A*z + B

def _p_a(A: float, B: float, q: float, z: float) -> float:
    return -2.0*A*q + A*z - B


def optimal_delta_grid(s: Scurve, p: float, n_delta: int = 2001) -> float:
    """Argmax over δ of f(δ)*(δ - p) via dense grid (robust, no root-finding)."""
    delta = s.grid(n_delta)
    vals = s.f(delta) * (delta - p)
    j = np.argmax(vals)
    return float(delta[j])


@dataclass
class QuoteEngine:
    params: ModelParams
    scurve_b: Scurve
    scurve_a: Scurve
    ABC: Dict[str, np.ndarray]

    def _interp_coeffs(self, t: float) -> Tuple[float, float, float]:
        """Linear interpolation of A(t), B(t), C(t) on stored grid."""
        t_grid = self.ABC['t']
        A = np.interp(t, t_grid, self.ABC['A'])
        B = np.interp(t, t_grid, self.ABC['B'])
        C = np.interp(t, t_grid, self.ABC['C'])
        return float(A), float(B), float(C)

    def value_theta(self, t: float, q: float) -> float:
        A, B, C = self._interp_coeffs(t)
        return -A*q*q - B*q - C

    def quotes(self, t: float, q: float, n_delta: int = 2001) -> Tuple[Optional[float], Optional[float]]:
        """Return (δ_b*, δ_a*) at (t,q), honoring inventory caps."""
        A, B, C = self._interp_coeffs(t)
        z = self.params.z

        # Inventory cap logic
        allow_bid = (q + z) <= self.params.qbar
        allow_ask = (q - z) >= -self.params.qbar

        db = None
        da = None

        if allow_bid:
            pb = _p_b(A, B, q, z)
            db = optimal_delta_grid(self.scurve_b, pb, n_delta=n_delta)
        if allow_ask:
            pa = _p_a(A, B, q, z)
            da = optimal_delta_grid(self.scurve_a, pa, n_delta=n_delta)

        return db, da

    def ftp(self, t: float, n_delta: int = 2001) -> float:
        """Fair Transfer Price offset at q=0: mid of optimal ask/bid offsets."""
        db, da = self.quotes(t, q=0.0, n_delta=n_delta)
        if db is None or da is None:
            raise ValueError("FTP needs both bid and ask quotes available at q=0.")
        return 0.5*(da - db)  # offset relative to S_t (ask above, bid below)


# --------------------------
# High-level Builder
# --------------------------

def build_pipeline(params: ModelParams,
                   scurve_b: Scurve,
                   scurve_a: Scurve,
                   alpha_b: Optional[Tuple[float, float, float]] = None,
                   alpha_a: Optional[Tuple[float, float, float]] = None,
                   alpha_fit_kwargs: Optional[dict] = None) -> Tuple[QuoteEngine, AlphaCoeffs, Dict[str, np.ndarray]]:
    """
    Build the quote engine:
      - If alphas are not provided, fit (α0,α1,α2) for H_b and H_a from the S-curves.
      - Integrate A,B,C backward in time.
      - Return a QuoteEngine with interpolation over the A,B,C grid.
    """
    if alpha_fit_kwargs is None:
        alpha_fit_kwargs = dict(p_max=0.5, n_p=41, n_delta=2001)

    # Fit alphas if needed
    if alpha_b is None:
        alpha_b = quadratic_alpha_from_scurve(scurve_b, **alpha_fit_kwargs)
    if alpha_a is None:
        alpha_a = quadratic_alpha_from_scurve(scurve_a, **alpha_fit_kwargs)

    alpha = AlphaCoeffs(alpha_b=alpha_b, alpha_a=alpha_a)
    ABC = integrate_ABC(params, alpha)
    engine = QuoteEngine(params=params, scurve_b=scurve_b, scurve_a=scurve_a, ABC=ABC)
    return engine, alpha, ABC


# --------------------------
# Demo / Self-test
# --------------------------

def _demo():
    """Run a small demo with plausible parameters and print a few quotes."""
    # Parameters (illustrative only)
    params = ModelParams(
        lambda_b=0.8, lambda_a=1.0,    # RFQ intensities
        z=1.0,                         # trade size
        sigma=0.02,                    # ref price vol (per sqrt time)
        kappa=0.0,                     # drift from flow imbalance
        gamma=5.0,                     # risk aversion
        T=5.0,                         # horizon (e.g., hours)
        qbar=10.0,                     # inventory cap (units of z)
        n_time=2000
    )

    # Symmetric but not identical S-curves (logistic)
    sc_b = logistic_scurve(a=0.0, b=1.2, delta_min=-6, delta_max=6)
    sc_a = logistic_scurve(a=0.2, b=1.0, delta_min=-6, delta_max=6)

    engine, alpha, ABC = build_pipeline(params, sc_b, sc_a)

    # Print alpha coefficients
    print("Alpha (bid):", alpha.alpha_b)
    print("Alpha (ask):", alpha.alpha_a)

    # Sample quotes at t=0
    t0 = 0.0
    for q in [-4, -2, 0, 2, 4]:
        db, da = engine.quotes(t=t0, q=float(q))
        print(f"t={t0:.2f}, q={q:+d} -> δ_b*={db:.4f} | δ_a*={da:.4f}")

    ftp0 = engine.ftp(t=t0)
    print(f"FTP offset at q=0, t=0: {ftp0:.4f}")

if __name__ == "__main__":
    _demo()
'''
with open("/mnt/data/rfq_quotes_pipeline.py", "w") as f:
    f.write(dedent(code))

print("Saved to /mnt/data/rfq_quotes_pipeline.py (fixed)")

# Run the demo again to confirm it works
import subprocess, sys
try:
    out = subprocess.check_output([sys.executable, "/mnt/data/rfq_quotes_pipeline.py"], stderr=subprocess.STDOUT, timeout=60)
    print(out.decode("utf-8"))
except subprocess.CalledProcessError as e:
    print("Script error:\n", e.output.decode("utf-8"))
except Exception as e:
    print("Could not run demo:", repr(e))
#================================================================================
from rfq_quotes_pipeline import (
    ModelParams, logistic_scurve, build_pipeline
)

# 1) Set model parameters
params = ModelParams(
    lambda_b=0.8, lambda_a=1.0,  # RFQ intensities
    z=1.0,                       # trade size
    sigma=0.02,                  # ref-price vol
    kappa=0.0,                   # drift from flow imbalance
    gamma=5.0,                   # risk aversion
    T=5.0,                       # horizon
    qbar=10.0,                   # inventory cap (in units of z)
    n_time=2000                  # ODE grid
)

# 2) Choose S-curves (logistic example; you can plug your own)
sc_b = logistic_scurve(a=0.0, b=1.2, delta_min=-6, delta_max=6)
sc_a = logistic_scurve(a=0.2, b=1.0, delta_min=-6, delta_max=6)

# 3) Build pipeline (fits α’s for H_b/H_a, integrates A,B,C)
engine, alpha, ABC = build_pipeline(params, sc_b, sc_a)

# 4) Quotes and FTP
t = 0.0
for q in [-4, -2, 0, 2, 4]:
    db, da = engine.quotes(t=t, q=float(q))
    print(f"q={q:+d} -> delta_b*={db:.4f}, delta_a*={da:.4f}")

ftp0 = engine.ftp(t=t)
print("FTP offset at q=0:", ftp0)
