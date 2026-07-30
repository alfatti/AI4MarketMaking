"""Intensity (demand-curve) models: Lambda(delta) = fill intensity at quoted distance delta.

Seam 1 of the repo. v1 ships ExponentialIntensity (Avellaneda-Stoikov, closed-form
Hamiltonian). Later variants slot in behind the same interface:
  - LogisticHitRateIntensity: lambda_rfq * f(delta), f logistic  (liquidity paper)
  - tier/size-indexed intensities (closed-form paper Section 5)
All downstream code (Hamiltonian, calibration) consumes only this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class IntensityModel(ABC):
    """Abstract fill-intensity curve Lambda(delta) for one side (bid or ask).

    Requirements (closed-form paper, Section 2.1):
      - twice continuously differentiable
      - strictly decreasing, Lambda' < 0
      - Lambda(delta) -> 0 as delta -> +inf
      - sup_delta Lambda * Lambda'' / (Lambda')**2 < 2   (well-posedness)
    """

    @abstractmethod
    def value(self, delta: np.ndarray | float) -> np.ndarray | float:
        """Lambda(delta)."""

    @abstractmethod
    def deriv(self, delta: np.ndarray | float) -> np.ndarray | float:
        """Lambda'(delta)."""

    @abstractmethod
    def deriv2(self, delta: np.ndarray | float) -> np.ndarray | float:
        """Lambda''(delta)."""

    @abstractmethod
    def inverse(self, level: np.ndarray | float) -> np.ndarray | float:
        """Lambda^{-1}(level): the delta at which intensity equals `level`.

        Used by the quote map: delta* = Lambda^{-1}(xi z H - H') (Model A)
        or Lambda^{-1}(-H') (Model B).
        """

    def wellposedness_sup(self, grid: np.ndarray) -> float:
        """max over grid of Lambda*Lambda''/(Lambda')**2; must be < 2."""
        lam = np.asarray(self.value(grid), dtype=float)
        d1 = np.asarray(self.deriv(grid), dtype=float)
        d2 = np.asarray(self.deriv2(grid), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = lam * d2 / d1**2
        return float(np.nanmax(ratio))

    def validate(self, grid: np.ndarray | None = None) -> None:
        """Raise if the curve violates the structural conditions on `grid`."""
        if grid is None:
            grid = np.linspace(-5.0, 10.0, 501)
        lam = np.asarray(self.value(grid), dtype=float)
        d1 = np.asarray(self.deriv(grid), dtype=float)
        if np.any(lam < 0):
            raise ValueError("Intensity must be nonnegative.")
        if np.any(d1 >= 0):
            raise ValueError("Intensity must be strictly decreasing (Lambda' < 0).")
        sup = self.wellposedness_sup(grid)
        if not sup < 2.0:
            raise ValueError(
                f"Well-posedness violated: sup Lambda*Lambda''/(Lambda')^2 = {sup:.4f} >= 2."
            )


@dataclass(frozen=True)
class LogisticIntensity(IntensityModel):
    """RFQ arrival x logistic hit-rate: Lambda(delta) = lam_rfq / (1 + exp(alpha + beta*delta)).

    The demand form of the liquidity-dynamics paper and of Section 6 of the
    closed-form paper. Well-posedness holds strictly:
    Lambda*Lambda''/(Lambda')^2 = 1 - e^{-(alpha+beta*delta)} < 1 < 2.

    Parameters
    ----------
    lam_rfq : RFQ arrival intensity (requests per unit time).
    alpha   : logit intercept (hit probability at delta=0 is 1/(1+e^alpha)).
    beta    : logit slope / client price-sensitivity (1 / price units).
    """

    lam_rfq: float
    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.lam_rfq <= 0 or self.beta <= 0:
            raise ValueError("LogisticIntensity requires lam_rfq > 0 and beta > 0.")

    def _u(self, delta):
        return np.exp(np.clip(self.alpha + self.beta * np.asarray(delta, dtype=float), -700, 700))

    def value(self, delta):
        return self.lam_rfq / (1.0 + self._u(delta))

    def deriv(self, delta):
        u = self._u(delta)
        return -self.lam_rfq * self.beta * u / (1.0 + u) ** 2

    def deriv2(self, delta):
        u = self._u(delta)
        return self.lam_rfq * self.beta**2 * u * (u - 1.0) / (1.0 + u) ** 3

    def inverse(self, level):
        level = np.asarray(level, dtype=float)
        return (np.log(self.lam_rfq / level - 1.0) - self.alpha) / self.beta

    def wellposedness_sup(self, grid: np.ndarray) -> float:  # analytic: sup = 1
        return 1.0


@dataclass(frozen=True)
class ExponentialIntensity(IntensityModel):
    """Avellaneda-Stoikov exponential: Lambda(delta) = A * exp(-k * delta).

    Saturates the well-posedness condition at equality
    (Lambda*Lambda''/(Lambda')^2 == 1 < 2) and yields the fully closed-form
    Hamiltonian used throughout v1.

    Parameters
    ----------
    A : intensity at delta = 0 (fills per unit time).
    k : decay rate / client price-sensitivity (1 / price units).
    """

    A: float
    k: float

    def __post_init__(self) -> None:
        if self.A <= 0 or self.k <= 0:
            raise ValueError("ExponentialIntensity requires A > 0 and k > 0.")

    def value(self, delta):
        return self.A * np.exp(-self.k * np.asarray(delta, dtype=float))

    def deriv(self, delta):
        return -self.k * self.value(delta)

    def deriv2(self, delta):
        return self.k**2 * self.value(delta)

    def inverse(self, level):
        level = np.asarray(level, dtype=float)
        return -np.log(level / self.A) / self.k

    def wellposedness_sup(self, grid: np.ndarray) -> float:  # exact, no grid needed
        return 1.0
