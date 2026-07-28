"""Closed-form quadratic proxy theta_check(t, q) = -q'A(t)q - q'B(t) - C(t)
for the general Model B market (asymmetric intensities, tiers, atomic sizes,
fixed costs, drift). Q = infinity scaffold — indicators dropped by construction.

Coefficient aggregation (Prop. 4 of Bergault et al., specialized to xi = 0 where the
Taylor coefficients alpha_j are z-independent so Delta_{j,k} = alpha_j * m_k):

  D_plus  = diag_i  sum_n [ a2_b m1_b + a2_a m1_a ]
  D_minus = diag_i  sum_n [ a2_b m2_b - a2_a m2_a ]
  V_minus = vec_i   sum_n [ a1_b m1_b - a1_a m1_a ]
  Vt_minus= vec_i   sum_n [ c_b a2_b - c_a a2_a ]          (cost-weighted, m0 = 1)

ODE system (derived from scratch; degree-0/1/2 identification verified by the
machine-zero Phase-0 test):

  A'(t) = 2 A D+ A - (gamma/2) Sigma,                         A(T) = 0
  B'(t) = mu + 2 A [ V- + D- diag(A) + D+ B + Vt- ],          B(T) = 0
  C'(t) = sum_{i,n,side±} [ a0 m1 + a1 (m2 A_ii ± m1 B_i + c)
            + (a2/2) (m3 A_ii^2 + m1 B_i^2 + c^2 m_{-1}
                      ± 2 m2 A_ii B_i + 2 c m1 A_ii ± 2 c B_i) ],   C(T) = 0

(± = + for bid, - for ask; note the c^2 * E[1/z] term — the inverse moment.)

Closed forms: with Ahat = sqrt(gamma) (D+^{1/2} Sigma D+^{1/2})^{1/2} = P diag(lam) P',
  A(t) = (1/2) D+^{-1/2} P diag(lam_i tanh(lam_i (T-t))) P' D+^{-1/2}
  B(t) = -D+^{-1/2} P diag(1/cosh(lam_i(T-t))) ∫_t^T diag(cosh(lam_i(T-s))) P' D+^{1/2} f(s) ds
         with f(s) = mu + 2 A(s) (V- + Vt- + D- diag(A(s))),
computed by Gauss-Legendre in s with the stable cosh ratio
  cosh(x_s)/cosh(x_t) = exp(x_s - x_t) (1+e^{-2 x_s})/(1+e^{-2 x_t}),  x_s <= x_t.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from . import hamiltonians as ham

_GL_NODES = 48


class Proxy:
    def __init__(self, spec):
        self.spec = spec
        sj = spec.to_jax()
        self.gamma, self.T = spec.gamma, spec.T
        self.mu = sj["mu"]; self.Sigma = sj["Sigma"]
        kind, ip, dinf = sj["kind"], sj["ip"], spec.delta_inf
        a0, a1, a2 = ham.alphas(kind, ip, dinf)          # (d, n, 2)
        m = spec.moments()
        m1 = jnp.asarray(m[1]); m2 = jnp.asarray(m[2]); m3 = jnp.asarray(m[3])
        mm1 = jnp.asarray(m[-1])
        c = sj["c"]
        b, a = 0, 1                                       # side indices
        self.Dp = jnp.diag((a2[:, :, b] * m1[:, :, b] + a2[:, :, a] * m1[:, :, a]).sum(1))
        self.Dm = jnp.diag((a2[:, :, b] * m2[:, :, b] - a2[:, :, a] * m2[:, :, a]).sum(1))
        self.Vm = (a1[:, :, b] * m1[:, :, b] - a1[:, :, a] * m1[:, :, a]).sum(1)
        self.Vtm = (c[:, :, b] * a2[:, :, b] - c[:, :, a] * a2[:, :, a]).sum(1)
        # stash per-(i,n,side) tensors for C'
        self._a0, self._a1, self._a2 = a0, a1, a2
        self._m1, self._m2, self._m3, self._mm1 = m1, m2, m3, mm1
        self._c = c
        # spectral data
        dp = jnp.diag(self.Dp)
        assert bool((np.asarray(dp) > 0).all()), "need alpha2-weighted mean sizes > 0 per asset"
        self.dp_sqrt = jnp.sqrt(dp)
        S = (self.dp_sqrt[:, None] * self.Sigma) * self.dp_sqrt[None, :]
        w, P = jnp.linalg.eigh(S)
        w = jnp.clip(w, 0.0, None)
        self.lams = jnp.sqrt(self.gamma * w)              # eigenvalues of Ahat
        self.P = P
        # Gauss-Legendre nodes on [0, 1]
        x, wq = np.polynomial.legendre.leggauss(_GL_NODES)
        self.gl_x = jnp.asarray(0.5 * (x + 1.0)); self.gl_w = jnp.asarray(0.5 * wq)

    # ------------- A(t) and helpers (all pure jnp, vmap/jit friendly) -------------

    def A(self, t):
        th = self.lams * (self.T - t)
        core = self.P @ jnp.diag(self.lams * jnp.tanh(th)) @ self.P.T
        return 0.5 * core / self.dp_sqrt[:, None] / self.dp_sqrt[None, :]

    def _f(self, s):
        As = self.A(s)
        return self.mu + 2.0 * As @ (self.Vm + self.Vtm + jnp.diag(self.Dm) * jnp.diag(As))

    def B(self, t):
        """Variation-of-parameters with stable cosh ratios, GL quadrature on [t, T]."""
        span = self.T - t
        s_nodes = t + span * self.gl_x                                   # (G,)
        xt = self.lams * (self.T - t)                                    # (d,)

        def integrand(s):
            xs = self.lams * (self.T - s)
            ratio = jnp.exp(xs - xt) * (1.0 + jnp.exp(-2.0 * xs)) / (1.0 + jnp.exp(-2.0 * xt))
            v = self.P.T @ (self.dp_sqrt * self._f(s))
            return ratio * v                                             # eigen-coords

        vals = jax.vmap(integrand)(s_nodes)                              # (G, d)
        integ = span * jnp.einsum("g,gk->k", self.gl_w, vals)
        return -(self.P @ integ) / self.dp_sqrt

    def A_prime(self, t):
        At = self.A(t)
        return 2.0 * At @ self.Dp @ At - 0.5 * self.gamma * self.Sigma

    def B_prime(self, t, At=None, Bt=None):
        At = self.A(t) if At is None else At
        Bt = self.B(t) if Bt is None else Bt
        return self.mu + 2.0 * At @ (self.Vm + jnp.diag(self.Dm) * jnp.diag(At)
                                     + jnp.diag(self.Dp) * Bt + self.Vtm)

    def C_prime(self, t, At=None, Bt=None):
        At = self.A(t) if At is None else At
        Bt = self.B(t) if Bt is None else Bt
        Aii = jnp.diag(At)[:, None]                                      # (d, 1) -> broadcast over tiers
        Bi = Bt[:, None]
        a0, a1, a2 = self._a0, self._a1, self._a2
        m1, m2, m3, mm1 = self._m1, self._m2, self._m3, self._mm1
        c = self._c
        total = 0.0
        for side, sgn in ((0, 1.0), (1, -1.0)):
            t0 = a0[:, :, side] * m1[:, :, side]
            t1 = a1[:, :, side] * (m2[:, :, side] * Aii + sgn * m1[:, :, side] * Bi + c[:, :, side])
            t2 = 0.5 * a2[:, :, side] * (
                m3[:, :, side] * Aii ** 2 + m1[:, :, side] * Bi ** 2
                + c[:, :, side] ** 2 * mm1[:, :, side]
                + sgn * 2.0 * m2[:, :, side] * Aii * Bi
                + 2.0 * c[:, :, side] * m1[:, :, side] * Aii
                + sgn * 2.0 * c[:, :, side] * Bi)
            total = total + (t0 + t1 + t2).sum()
        return total

    def C(self, t):
        """Only needed for reporting theta values; GL quadrature of C' on [t, T].
        Exact nested version — kept as-is for the independent-path Phase-0 test."""
        span = self.T - t
        s_nodes = t + span * self.gl_x
        vals = jax.vmap(lambda s: self.C_prime(s))(s_nodes)
        return -span * jnp.einsum("g,g->", self.gl_w, vals)

    # -------- cached C for fast reporting (validation / demos, not tests) --------

    def _build_C_cache(self, n_seg=256):
        """C on a grid by composite 4-point Gauss-Legendre per segment (error (dt)^8),
        plus C' at the nodes for cubic Hermite interpolation (error (dt)^4)."""
        import numpy as _np
        ts = _np.linspace(0.0, self.T, n_seg + 1)
        x4, w4 = _np.polynomial.legendre.leggauss(4)
        Cp_fn = jax.jit(jax.vmap(lambda s: self.C_prime(s)))
        seg_nodes = (ts[:-1, None] + 0.5 * (ts[1:] - ts[:-1])[:, None] * (x4[None, :] + 1.0))
        seg_vals = _np.asarray(Cp_fn(jnp.asarray(seg_nodes.ravel()))).reshape(n_seg, 4)
        seg_int = 0.5 * (ts[1:] - ts[:-1]) * (seg_vals @ w4)
        C = _np.zeros(n_seg + 1)
        for k in range(n_seg - 1, -1, -1):          # C(t) = C(t_next) - int C'
            C[k] = C[k + 1] - seg_int[k]
        Cp = _np.asarray(Cp_fn(jnp.asarray(ts)))
        self._C_ts, self._C_vals, self._C_derivs = ts, C, Cp

    def C_fast(self, t):
        if not hasattr(self, "_C_ts"):
            self._build_C_cache()
        import numpy as _np
        ts, C, Cp = self._C_ts, self._C_vals, self._C_derivs
        k = min(int(_np.searchsorted(ts, float(t)) - 1) if t > ts[0] else 0, len(ts) - 2)
        h = ts[k + 1] - ts[k]; x = (float(t) - ts[k]) / h
        h00 = (1 + 2 * x) * (1 - x) ** 2; h10 = x * (1 - x) ** 2
        h01 = x * x * (3 - 2 * x); h11 = x * x * (x - 1)
        return h00 * C[k] + h * h10 * Cp[k] + h01 * C[k + 1] + h * h11 * Cp[k + 1]

    def theta_fast(self, t, q):
        """theta_check via cached C — for validation and demos (reporting path)."""
        At, Bt = self.A(t), self.B(t)
        return float(-q @ At @ q - q @ Bt) - float(self.C_fast(t))

    # ------------- values, exact finite differences, time derivative -------------

    def theta(self, t, q):
        At, Bt, Ct = self.A(t), self.B(t), self.C(t)
        return -q @ At @ q - q @ Bt - Ct

    def diffs(self, q, At, Bt, z, i, sgn):
        """theta_check(q) - theta_check(q + sgn * z * e_i), exact (no subtraction of large values).

        sgn = +1 for bid (inventory up), -1 for ask.
        = 2 sgn z q'A e_i + z^2 A_ii + sgn z B_i
        Broadcasts: q (d,), z any shape, returns z.shape.
        """
        qAei = (q @ At)[i]
        return 2.0 * sgn * z * qAei + z ** 2 * At[i, i] + sgn * z * Bt[i]

    def dtheta_dt(self, t, q, At=None, Bt=None):
        At = self.A(t) if At is None else At
        Bt = self.B(t) if Bt is None else Bt
        return (-q @ self.A_prime(t) @ q - q @ self.B_prime(t, At, Bt)
                - self.C_prime(t, At, Bt))
