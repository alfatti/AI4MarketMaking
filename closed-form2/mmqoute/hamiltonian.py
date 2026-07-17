"""Hamiltonian functions H_xi(p) built from an IntensityModel.

Definitions (closed-form paper, Eqs. (3)-(4)), per side:
    xi > 0 (Model A, xi = gamma):
        H_xi(p) = sup_delta Lambda(delta)/(xi z) * (1 - exp(-xi z (delta - p)))
    xi = 0 (Model B):
        H_0(p)  = sup_delta Lambda(delta) * (delta - p)

For ExponentialIntensity everything is closed-form:
    delta*(p) = p + (1/k) * log(1 + xi z / k)          (xi > 0)
    delta*(p) = p + 1/k                                 (xi = 0)
    H_xi(p)   = C_xi * exp(-k p),  with
      C_0  = (A / k) * e^{-1}
      C_xi = (A / (xi z)) * (1 - (1 + xi z / k)^{-1}) * (1 + xi z / k)^{-k/(xi z)}
Hence H^{(j)}(0) = (-k)^j * C_xi, and the Taylor coefficients alpha_j fall out
analytically. For a generic IntensityModel we fall back to numerical sup / finite
differences behind the same interface.

The quadratic proxy (Section 3) uses alpha_j = H^{(j)}(0), j in {0,1,2}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from .intensity import ExponentialIntensity, IntensityModel


@dataclass(frozen=True)
class HamiltonianSide:
    """H_xi for one side (bid or ask), for one asset.

    Parameters
    ----------
    intensity : the side's IntensityModel.
    z         : transaction size for the asset.
    xi        : objective switch. xi = gamma for Model A (CARA), xi = 0 for Model B.
    """

    intensity: IntensityModel
    z: float
    xi: float = 0.0

    # ----- closed-form path (exponential intensity) -----

    def _exp_params(self) -> tuple[float, float] | None:
        if isinstance(self.intensity, ExponentialIntensity):
            return self.intensity.A, self.intensity.k
        return None

    def _exp_C(self) -> float:
        A, k = self._exp_params()
        if self.xi == 0.0:
            return (A / k) * np.exp(-1.0)
        r = 1.0 + self.xi * self.z / k
        return (A / (self.xi * self.z)) * (1.0 - 1.0 / r) * r ** (-k / (self.xi * self.z))

    # ----- public interface -----

    def H(self, p: np.ndarray | float) -> np.ndarray | float:
        if self._exp_params() is not None:
            _, k = self._exp_params()
            return self._exp_C() * np.exp(-k * np.asarray(p, dtype=float))
        return self._H_numeric(p)

    def dH(self, p: np.ndarray | float) -> np.ndarray | float:
        if self._exp_params() is not None:
            _, k = self._exp_params()
            return -k * self.H(p)
        return self._fd(self.H, p, order=1)

    def d2H(self, p: np.ndarray | float) -> np.ndarray | float:
        if self._exp_params() is not None:
            _, k = self._exp_params()
            return k**2 * self.H(p)
        return self._fd(self.H, p, order=2)

    def delta_star(self, p: np.ndarray | float) -> np.ndarray | float:
        """Optimal quote distance for marginal inventory value p (Thm 1 / Thm 2)."""
        if self.xi == 0.0:
            target = -self.dH(p)
        else:
            target = self.xi * self.z * self.H(p) - self.dH(p)
        return self.intensity.inverse(target)

    def alphas(self) -> tuple[float, float, float]:
        """(alpha_0, alpha_1, alpha_2) = (H(0), H'(0), H''(0)) for the quadratic proxy."""
        return float(self.H(0.0)), float(self.dH(0.0)), float(self.d2H(0.0))

    # ----- numeric fallbacks for non-exponential intensities -----

    def _objective(self, delta: float, p: float) -> float:
        lam = float(self.intensity.value(delta))
        if self.xi == 0.0:
            return lam * (delta - p)
        return lam / (self.xi * self.z) * (1.0 - np.exp(-self.xi * self.z * (delta - p)))

    def _H_numeric(self, p):
        p_arr = np.atleast_1d(np.asarray(p, dtype=float))
        out = np.empty_like(p_arr)
        for i, pi in enumerate(p_arr):
            res = minimize_scalar(
                lambda d: -self._objective(d, pi),
                bounds=(pi - 50.0, pi + 50.0),
                method="bounded",
            )
            out[i] = -res.fun
        return out if np.ndim(p) else float(out[0])

    @staticmethod
    def _fd(f, p, order: int, h: float = 1e-4):
        p = np.asarray(p, dtype=float)
        if order == 1:
            return (f(p + h) - f(p - h)) / (2 * h)
        return (f(p + h) - 2 * f(p) + f(p - h)) / h**2
