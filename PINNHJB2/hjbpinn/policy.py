"""Quote extraction: delta*(t, q, z, i, n, side) from a trained theta = theta_check + eta.

Oracle-compatible surface: `quote(...)` mirrors the BG oracle signature so the SFT label
generator can swap oracles behind a flag. `use_eta=False` gives the pure proxy policy.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from . import hamiltonians as ham
from . import network


class Policy:
    def __init__(self, spec, proxy, params=None, theta_scale=None):
        self.spec, self.proxy = spec, proxy
        self.params = params
        self.theta_scale = theta_scale
        self.sj = spec.to_jax()
        self.fs = network.feature_spec(spec)

    def _eta(self, t, q):
        if self.params is None:
            return 0.0
        return float(network.eta(self.params, t, jnp.asarray(q), self.fs, self.theta_scale))

    def per_unit_diff(self, t, q, z, i, n, side, use_eta=True):
        sgn = 1.0 if side == 0 else -1.0
        At, Bt = self.proxy.A(t), self.proxy.B(t)
        diff = float(self.proxy.diffs(jnp.asarray(q), At, Bt, z, int(i), sgn))
        if use_eta and self.params is not None:
            qs = np.array(q, float); qs[i] += sgn * z
            diff += self._eta(t, np.asarray(q, float)) - self._eta(t, qs)
        return (diff + float(self.spec.c[i, n, side])) / z

    def quote(self, t, q, z, i=0, n=0, side=0, use_eta=True):
        """Optimal offset delta (>= -delta_inf). side: 0 bid, 1 ask. Feasibility-checked."""
        sgn = 1.0 if side == 0 else -1.0
        if abs(q[i] + sgn * z) > self.spec.Q[i] + 1e-12:
            return np.nan                                     # trade not admissible
        p = self.per_unit_diff(t, q, z, i, n, side, use_eta)
        return float(ham.delta_star(jnp.asarray(p), self.sj["kind"][i, n, side],
                                    self.sj["ip"][i, n, side], self.spec.delta_inf))
