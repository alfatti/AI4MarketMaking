"""Value-function proxy: theta_check(t, q) = -q'A(t)q - q'B(t) - C(t).

Seam 2 of the repo: everything here is written in the matrix form of the
closed-form paper (Prop. 1-3) even though v1 only exercises d = 1. The scalar
single-CUSIP case is the 1x1 collapse; multi-asset later means passing d > 1
inputs, not rewriting the solver.

System (Eq. 11), with D_+ = diag((alpha_2^{i,b}+alpha_2^{i,a}) z^i), etc.:
    A'(t) = 2 A(t) D_+ A(t) - (gamma/2) Sigma,            A(T) = 0
    B'(t) = 2 A(t) [V_- + D_- D(A(t)) + D_+ B(t)],        B(T) = 0
    C'(t) = ... (level term; irrelevant for quotes, kept for completeness)

Closed form (Prop. 2):
    A(t) = (1/2) D_+^{-1/2} Ahat tanh(Ahat (T-t)) D_+^{-1/2},
    Ahat = sqrt(gamma) (D_+^{1/2} Sigma D_+^{1/2})^{1/2}

Ergodic limit (Prop. 3):
    A -> (1/2) sqrt(gamma) Gamma,
    Gamma = D_+^{-1/2} (D_+^{1/2} Sigma D_+^{1/2})^{1/2} D_+^{-1/2}
    B -> -D_+^{-1/2} Ahat Ahat^+ D_+^{-1/2} (V_- + (1/2) sqrt(gamma) D_- D(Gamma))

B carries all bid/ask asymmetry (V_-, D_-) and is the slot where a kappa-style
drift term lands later; it is wired in even when zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import sqrtm, tanhm


def _sym_sqrt(M: np.ndarray) -> np.ndarray:
    S = np.real_if_close(sqrtm(M))
    return np.asarray(S, dtype=float)


@dataclass(frozen=True)
class ProxyCoefficients:
    """Per-asset Taylor coefficients of the two Hamiltonians and sizes.

    All arrays have shape (d,). For single-CUSIP, d = 1.
    """

    alpha0_b: np.ndarray
    alpha1_b: np.ndarray
    alpha2_b: np.ndarray
    alpha0_a: np.ndarray
    alpha1_a: np.ndarray
    alpha2_a: np.ndarray
    z: np.ndarray

    def __post_init__(self) -> None:
        arrs = [self.alpha0_b, self.alpha1_b, self.alpha2_b,
                self.alpha0_a, self.alpha1_a, self.alpha2_a, self.z]
        d = arrs[0].shape[0]
        for a in arrs:
            if a.shape != (d,):
                raise ValueError("All coefficient arrays must share shape (d,).")
        if np.any(self.alpha2_b + self.alpha2_a <= 0):
            raise ValueError("Need alpha2_b + alpha2_a > 0 per asset (Prop. 2).")

    # Matrices of Prop. 1 (k-th powers of z folded in)
    @property
    def D_plus(self) -> np.ndarray:  # (alpha2_b + alpha2_a) * z, diagonal
        return np.diag((self.alpha2_b + self.alpha2_a) * self.z)

    @property
    def D_minus(self) -> np.ndarray:  # (alpha2_b - alpha2_a) * z^2, diagonal
        return np.diag((self.alpha2_b - self.alpha2_a) * self.z**2)

    @property
    def V_minus(self) -> np.ndarray:  # (alpha1_b - alpha1_a) * z, vector
        return (self.alpha1_b - self.alpha1_a) * self.z


@dataclass
class RiccatiSolution:
    """A(t), B(t) on a time grid, plus ergodic limits. C omitted (level only)."""

    t_grid: np.ndarray
    A_path: np.ndarray  # (n_t, d, d)
    B_path: np.ndarray  # (n_t, d)
    A_erg: np.ndarray   # (d, d)
    B_erg: np.ndarray   # (d,)
    Gamma: np.ndarray   # (d, d)

    def A(self, t: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.t_grid, t), 0, len(self.t_grid) - 1))
        return self.A_path[i]

    def B(self, t: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.t_grid, t), 0, len(self.t_grid) - 1))
        return self.B_path[i]


def solve_riccati(
    coeffs: ProxyCoefficients,
    Sigma: np.ndarray,
    gamma: float,
    T: float,
    n_steps: int = 500,
) -> RiccatiSolution:
    """Closed-form A(t) via matrix tanh; B(t) by backward integration; ergodic limits."""
    d = coeffs.z.shape[0]
    Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
    if Sigma.shape != (d, d):
        raise ValueError(f"Sigma must be ({d},{d}).")

    Dp = coeffs.D_plus
    Dm = coeffs.D_minus
    Vm = coeffs.V_minus
    Dp_h = _sym_sqrt(Dp)
    Dp_ih = np.linalg.inv(Dp_h)

    Ahat = np.sqrt(gamma) * _sym_sqrt(Dp_h @ Sigma @ Dp_h)

    t_grid = np.linspace(0.0, T, n_steps + 1)

    # A(t) closed form (Prop. 2, tanh representation)
    A_path = np.empty((len(t_grid), d, d))
    for i, t in enumerate(t_grid):
        Th = np.real_if_close(tanhm(Ahat * (T - t)))
        A_path[i] = 0.5 * Dp_ih @ Ahat @ Th @ Dp_ih

    # B(t): backward integration of B' = 2A(V_- + D_- diag(A)) + 2A D_+ B, B(T) = 0
    # Backward in time: B(t) = B(t+dt) - dt * B'(t+dt).
    B_path = np.zeros((len(t_grid), d))
    dt = T / n_steps
    for i in range(len(t_grid) - 2, -1, -1):
        A_next = A_path[i + 1]
        rhs = 2.0 * A_next @ (Vm + Dm @ np.diag(A_next)) + 2.0 * A_next @ Dp @ B_path[i + 1]
        B_path[i] = B_path[i + 1] - dt * rhs

    # Ergodic limits (Prop. 3)
    Gamma = Dp_ih @ _sym_sqrt(Dp_h @ Sigma @ Dp_h) @ Dp_ih
    A_erg = 0.5 * np.sqrt(gamma) * Gamma
    Ahat_pinv = np.linalg.pinv(Ahat)
    B_erg = -(Dp_ih @ Ahat @ Ahat_pinv @ Dp_ih) @ (
        Vm + 0.5 * np.sqrt(gamma) * Dm @ np.diag(Gamma)
    )

    return RiccatiSolution(
        t_grid=t_grid, A_path=A_path, B_path=B_path,
        A_erg=A_erg, B_erg=B_erg, Gamma=Gamma,
    )


def theta_check(A: np.ndarray, B: np.ndarray, q: np.ndarray, C: float = 0.0) -> float:
    """theta_check(q) = -q'Aq - q'B - C. C irrelevant for quotes (differences only)."""
    q = np.asarray(q, dtype=float)
    return float(-q @ A @ q - q @ B - C)
