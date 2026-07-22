"""Greedy quote map: from the proxy value function to bid/ask quote distances.

Optimal quotes (Thms 1-2) evaluate the side Hamiltonian's delta_star at the
marginal inventory value
    p_b(t, q) = [theta(t, q) - theta(t, q + z e_i)] / z_i
    p_a(t, q) = [theta(t, q) - theta(t, q - z e_i)] / z_i
with theta replaced by the quadratic proxy. Because theta_check is quadratic,
p is affine in q -- the interpretable inventory-linear skew.

Bid and ask paths are kept separate end-to-end (asymmetry / tiering seam).
Inventory limits are enforced here as a policy overlay (the proxy itself is
derived in the Q -> infinity limit).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hamiltonian import HamiltonianSide
from .riccati import RiccatiSolution, theta_check


@dataclass(frozen=True)
class Quote:
    delta_b: float | None  # None => side switched off (inventory limit)
    delta_a: float | None
    p_b: float | None
    p_a: float | None


@dataclass
class QuoteEngine:
    """Greedy quoting for asset i given the Riccati proxy solution.

    Parameters
    ----------
    ham_b, ham_a : per-side Hamiltonians (carry intensity, z, xi).
    solution     : RiccatiSolution (finite-T paths + ergodic limits).
    asset        : index i of the asset being quoted (0 for single-CUSIP).
    Q_limit      : inventory cap in units of the asset; None disables.
    ergodic      : if True, use (A_erg, B_erg) -- time-independent quotes.
    """

    ham_b: HamiltonianSide
    ham_a: HamiltonianSide
    solution: RiccatiSolution
    asset: int = 0
    Q_limit: float | None = None
    ergodic: bool = True

    def _AB(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if self.ergodic:
            return self.solution.A_erg, self.solution.B_erg
        return self.solution.A(t), self.solution.B(t)

    def marginal_values(self, q: np.ndarray, t: float = 0.0,
                        z: float | None = None) -> tuple[float, float]:
        """(p_b, p_a) at inventory vector q for request size z; affine in q.

        z defaults to the ham's stored size (E[z] under distributed sizes).
        Per-size quoting (Section 5): p = 2q'Ae_i + z e_i'Ae_i +/- e_i'B.
        """
        A, B = self._AB(t)
        q = np.asarray(q, dtype=float)
        z = self.ham_b.z if z is None else float(z)
        e = np.zeros_like(q)
        e[self.asset] = 1.0
        th0 = theta_check(A, B, q)
        p_b = (th0 - theta_check(A, B, q + z * e)) / z
        p_a = (th0 - theta_check(A, B, q - z * e)) / z
        return p_b, p_a

    def quote(self, q: np.ndarray | float, t: float = 0.0,
              z: float | None = None) -> Quote:
        q = np.atleast_1d(np.asarray(q, dtype=float))
        qi = q[self.asset]
        z = self.ham_b.z if z is None else float(z)

        bid_on = self.Q_limit is None or qi + z <= self.Q_limit
        ask_on = self.Q_limit is None or qi - z >= -self.Q_limit

        p_b, p_a = self.marginal_values(q, t, z)
        return Quote(
            delta_b=float(self.ham_b.delta_star(p_b)) if bid_on else None,
            delta_a=float(self.ham_a.delta_star(p_a)) if ask_on else None,
            p_b=p_b if bid_on else None,
            p_a=p_a if ask_on else None,
        )

    def skew(self, q: np.ndarray | float, t: float = 0.0) -> float:
        """delta_a - delta_b: the quote asymmetry (liquidity-paper skew at q=0)."""
        qt = self.quote(q, t)
        if qt.delta_a is None or qt.delta_b is None:
            raise ValueError("Skew undefined when a side is off.")
        return qt.delta_a - qt.delta_b
