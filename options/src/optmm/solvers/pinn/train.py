"""Stage-1 PINN trainer: collocation sampler, Adam loop, optional GN-CG
polish, and the two headline metrics (hyperplane collinearity, value error
vs the reduced2d reference).  Re-entrant via a plain checkpoint dict.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ...core.risk import MarketParams, penalty_coef
from ..reduced2d import Channels
from .model import PinnSpec, init_params, value
from .residual import residual_stage1


def sample_batch(key, spec: PinnSpec, batch: int, nu_pad: float = 0.2,
                 slab_frac: float = 0.95):
    """(t, nu, n) collocation points in the slab interior.

    n is drawn uniformly on the per-dim band box [-Vbar/w_i, Vbar/w_i]^N,
    a fraction shrunk toward the origin (near-diagonal states matter), then
    radially projected into {|w.n| <= slab_frac * Vbar}.  The projection
    slightly over-weights the slab boundary; acceptable for training and
    documented.  nu is sampled on a padded band so no boundary condition is
    imposed (interior collocation of the free problem).
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    t = jax.random.uniform(k1, (batch,)) * spec.T
    pad = nu_pad * (spec.nu_hi - spec.nu_lo)
    nu = jax.random.uniform(k2, (batch,)) \
        * (spec.nu_hi - spec.nu_lo + 2 * pad) + spec.nu_lo - pad
    N = spec.w.shape[0]
    b = spec.Vbar / spec.w
    x = jax.random.uniform(k3, (batch, N), minval=-1.0, maxval=1.0)
    shrink = jax.random.uniform(k4, (batch, 1)) ** 0.5
    n = x * b[None, :] * shrink
    Vpi = n @ spec.w
    scale = jnp.minimum(1.0, slab_frac * spec.Vbar / (jnp.abs(Vpi) + 1e-30))
    return t, nu, n * scale[:, None]


def make_loss(spec, ch: Channels, market: MarketParams, gamma: float,
              c: float, Vbar: float, delta_inf=-5.0, adm_set=None):
    pen = penalty_coef(gamma, market.xi, market.rho, c)
    res1 = partial(residual_stage1, spec=spec, value_fn=value, ch=ch,
                   market=market, pen=pen, Vbar=Vbar, delta_inf=delta_inf,
                   adm_set=adm_set)

    def batched_residual(params, t, nu, n):
        return jax.vmap(lambda a, b_, m: res1(params, t=a, nu=b_, n=m)
                        )(t, nu, n)

    def loss(params, t, nu, n):
        r = batched_residual(params, t, nu, n)
        return jnp.mean(r * r)

    return loss, batched_residual


def adam_train(params, loss_fn, spec, key, steps: int, batch: int = 512,
               lr: float = 1e-3, log_every: int = 250, log=print,
               sampler=None):
    """sampler(key, spec, batch) -> loss_fn point args; defaults to the
    stage-1 slab sampler.  Stage-2 box problems pass a box sampler; stage-3
    losses pass a (t, S, nu, n) sampler paired with a matching loss_fn."""
    import optax  # local import; only the trainer needs it
    sampler = sample_batch if sampler is None else sampler
    opt = optax.adam(lr)
    state = opt.init(params)

    @jax.jit
    def step(params, state, key):
        key, ks = jax.random.split(key)
        args = sampler(ks, spec, batch)
        l, g = jax.value_and_grad(loss_fn)(params, *args)
        updates, state = opt.update(g, state)
        return optax.apply_updates(params, updates), state, key, l

    hist = []
    for i in range(steps):
        params, state, key, l = step(params, state, key)
        if i % log_every == 0 or i == steps - 1:
            hist.append((i, float(l)))
            log(f"  adam {i:5d}: loss {float(l):.4e} (rms {float(l)**0.5:.3e})")
    return params, hist


def gn_polish(params, batched_residual, spec, key, iters: int = 10,
              batch: int = 1024, damping: float = 1e-3, cg_iters: int = 50,
              log=print):
    """Matrix-free Gauss-Newton with CG on (J^T J + damping * I) d = -J^T r."""
    flat, unravel = jax.flatten_util.ravel_pytree(params)

    def r_of_flat(fl, t, nu, n):
        return batched_residual(unravel(fl), t, nu, n)

    for it in range(iters):
        key, ks = jax.random.split(key)
        t, nu, n = sample_batch(ks, spec, batch)
        r0 = r_of_flat(flat, t, nu, n)
        scale = jnp.sqrt(r0.shape[0] * 1.0)

        def jvp(d):
            return jax.jvp(lambda fl: r_of_flat(fl, t, nu, n), (flat,), (d,))[1]

        def vjp(rr):
            return jax.vjp(lambda fl: r_of_flat(fl, t, nu, n), flat)[1](rr)[0]

        def mv(d):
            return vjp(jvp(d)) / scale**2 + damping * d

        b = -vjp(r0) / scale**2
        d, _ = jax.scipy.sparse.linalg.cg(mv, b, maxiter=cg_iters)
        # backtracking on the sampled-batch loss
        l0 = float(jnp.mean(r0 * r0))
        stepsize = 1.0
        for _ in range(8):
            cand = flat + stepsize * d
            lc = float(jnp.mean(r_of_flat(cand, t, nu, n) ** 2))
            if lc < l0:
                flat = cand
                break
            stepsize *= 0.5
        log(f"  gn {it:3d}: loss {l0:.4e} -> {lc:.4e} (step {stepsize})")
    return unravel(flat)


def collinearity(params, spec, key, n_samples: int = 512, t_eval: float = 0.0,
                 nu_eval: float | None = None):
    """Median cos(grad_n v, w) over interior samples — discovery metric."""
    nu_eval = 0.5 * (spec.nu_lo + spec.nu_hi) if nu_eval is None else nu_eval
    _, _, n = sample_batch(key, spec, n_samples, slab_frac=0.8)
    g = jax.vmap(lambda m: jax.grad(
        lambda mm: value(params, spec, t_eval, nu_eval, mm))(m))(n)
    w = spec.w
    cos = (g @ w) / (jnp.linalg.norm(g, axis=1) * jnp.linalg.norm(w) + 1e-30)
    return float(jnp.median(jnp.abs(cos))), np.asarray(cos)


def value_error_vs_reference(params, spec, key, V_grid, v_ref_row,
                             n_samples: int = 512, t_eval: float = 0.0,
                             nu_eval: float | None = None):
    """|v_theta - v_2d(w.n)| over sampled interior states (EUR)."""
    nu_eval = 0.5 * (spec.nu_lo + spec.nu_hi) if nu_eval is None else nu_eval
    _, _, n = sample_batch(key, spec, n_samples, slab_frac=0.9)
    vp = jax.vmap(lambda m: value(params, spec, t_eval, nu_eval, m))(n)
    Vpi = np.asarray(n @ spec.w)
    v2 = np.interp(Vpi, np.asarray(V_grid), np.asarray(v_ref_row))
    err = np.abs(np.asarray(vp) - v2)
    return float(np.median(err)), float(np.max(err))
