"""Inventory state: trade-unit lattice n (q = n * z) and admissible sets.

Stage 1: Slab  {|w^T n| <= Vbar}  (aggregate-vega risk limit; w_i = z_i vega_i)
Stage 2: Box   {|n_i| <= nbar_i}  (per-instrument position limits)

The lattice *representation* box must contain the admissible set plus its
one-jump neighborhood so that every jump target of an admissible state is
representable; `representation_nbar` computes and asserts this.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Slab:
    Vbar: float

    def admissible(self, n_grid: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """n_grid: (..., N) integer lattice points -> bool mask (...)."""
        Vpi = jnp.tensordot(n_grid, w, axes=([-1], [0]))
        return jnp.abs(Vpi) <= self.Vbar * (1 + 1e-12)

    def representation_nbar(self, w: np.ndarray) -> np.ndarray:
        return np.ceil(self.Vbar / np.asarray(w)).astype(int) + 1


@dataclass(frozen=True)
class Box:
    nbar: np.ndarray  # per-instrument, in trade units

    def admissible(self, n_grid: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        nb = jnp.asarray(self.nbar)
        return jnp.all(jnp.abs(n_grid) <= nb + 1e-12, axis=-1)

    def representation_nbar(self, w: np.ndarray) -> np.ndarray:
        return np.asarray(self.nbar, dtype=int) + 1
