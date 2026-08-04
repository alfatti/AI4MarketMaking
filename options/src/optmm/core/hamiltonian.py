"""Hamiltonians H(p) = sup_{delta >= delta_inf} Lambda(delta) (delta - p).

Families (both satisfy sup Lam Lam''/Lam'^2 < 2 => unique FOC root):

- exponential: Lam(d) = A exp(-k d).  Closed form d* = p + 1/k.
- logistic:    Lam(d) = lam / (1 + exp(alpha + k d)).  Semi-closed form via
  Lambert W: with x = alpha + k d and b = alpha + k p, the FOC
      Lam + Lam'(d - p) = 0   <=>   x - b - 1 = exp(-x)
  has the unique root
      x* = b + 1 + W( exp(-(b+1)) ),         W = principal branch,
  hence d* = (x* - alpha)/k, and (from the FOC) d* - p = (1 + e^{-x*})/k,
      H(p) = Lam(d*) (1 + e^{-x*}) / k.
  For b + 1 << 0 the W argument overflows; there we use the standard
  large-argument asymptotics W(e^L) = L - log L + ... refined by Newton on
  f(w) = w + log w - L.  A plain-Newton implementation in delta-space is
  *wrong in practice*: from d0 = p its steps are ~1/k while sigma is tiny, so
  it needs O(|alpha + k p|) iterations — caught by the brute-force test.

Derivations: docs/derivations.md section 4.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

_HALLEY_ITERS = 12
_LOG_SWITCH = 30.0  # use asymptotic branch when L = -(b+1) exceeds this


def _lambertw_principal(a):
    """W0(a) for a >= 0, Halley iteration, machine precision."""
    a = jnp.asarray(a, jnp.float64)
    w = jnp.where(a > 2.0, jnp.log(jnp.maximum(a, 2.0))
                  - jnp.log(jnp.log(jnp.maximum(a, 2.0))), a / (1.0 + a))

    def body(_, w):
        ew = jnp.exp(w)
        f = w * ew - a
        denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0)
        return w - f / denom

    return jax.lax.fori_loop(0, _HALLEY_ITERS, body, w)


def _lambertw_of_expL(L):
    """W(exp(L)) for real L, overflow-safe."""
    L = jnp.asarray(L, jnp.float64)
    a_safe = jnp.exp(jnp.minimum(L, _LOG_SWITCH))
    w_direct = _lambertw_principal(a_safe)
    Ls = jnp.maximum(L, 2.0)
    w_asym = Ls - jnp.log(Ls) + jnp.log(Ls) / Ls

    def corr(_, w):
        f = w + jnp.log(w) - Ls
        return w - f / (1.0 + 1.0 / w)

    w_asym = jax.lax.fori_loop(0, 3, corr, w_asym)
    return jnp.where(L > _LOG_SWITCH, w_asym, w_direct)


@dataclass(frozen=True)
class LogisticIntensity:
    """Lambda(delta) = lam / (1 + exp(alpha + k * delta)). Arrays broadcast."""

    lam: jnp.ndarray
    alpha: jnp.ndarray
    k: jnp.ndarray

    def __call__(self, delta):
        # lam / (1 + e^x) = lam * sigmoid(-x): overflow-safe for large |x|
        return self.lam * jax.nn.sigmoid(-(self.alpha + self.k * delta))

    def _xstar(self, p):
        b = self.alpha + self.k * p
        return b + 1.0 + _lambertw_of_expL(-(b + 1.0))

    def argmax_delta(self, p, delta_inf=-jnp.inf):
        d = (self._xstar(p) - self.alpha) / self.k
        return jnp.maximum(d, delta_inf)

    def H(self, p, delta_inf=-jnp.inf):
        x = self._xstar(p)
        d_free = (x - self.alpha) / self.k
        H_free = self(d_free) * (1.0 + jnp.exp(-x)) / self.k
        H_bound = self(delta_inf) * (delta_inf - p)
        return jnp.where(d_free >= delta_inf, H_free, H_bound)


@dataclass(frozen=True)
class ExponentialIntensity:
    """Lambda(delta) = A * exp(-k * delta)."""

    A: jnp.ndarray
    k: jnp.ndarray

    def __call__(self, delta):
        return self.A * jnp.exp(-self.k * delta)

    def argmax_delta(self, p, delta_inf=-jnp.inf):
        return jnp.maximum(p + 1.0 / self.k, delta_inf)

    def H(self, p, delta_inf=-jnp.inf):
        d = self.argmax_delta(p, delta_inf)
        return self(d) * (d - p)


def foc_residual(intensity, p, delta):
    """Lambda(d) + Lambda'(d)(d - p); zero at the unconstrained optimum."""
    lam_fn = lambda d: intensity(d)
    lam = lam_fn(delta)
    flat = jnp.ravel(jnp.asarray(delta, jnp.float64))
    dlam = jax.vmap(jax.grad(lam_fn))(flat).reshape(jnp.shape(delta))
    return lam + dlam * (delta - p)
