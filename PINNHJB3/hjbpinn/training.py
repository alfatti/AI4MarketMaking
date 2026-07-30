"""Re-entrant multi-phase training, callable from drivers and notebooks alike.

train() runs GN phases from wherever the checkpoint left off, honoring an
optional wall-clock budget (it never starts a phase it may not finish cleanly;
each completed phase checkpoints). Checkpoints are self-describing (widths,
feature count, scenario name, next phase index)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from . import network, residual as res_mod, sampling
from .proxy import Proxy
from .optimizer import gauss_newton, gauss_newton_gpu
from .scenarios import ScenarioCfg


def template(spec, cfg: ScenarioCfg):
    fs = network.feature_spec(spec)
    return network.init_params(jax.random.PRNGKey(0), network.n_features(fs),
                               tuple(cfg.widths)), fs


def load_checkpoint(spec, cfg, path):
    """(params, fs, t_all, q_all, k_next); fresh template if no checkpoint."""
    tmpl, fs = template(spec, cfg)
    p = Path(path)
    if not p.exists():
        return tmpl, fs, None, None, 0
    st = np.load(p, allow_pickle=True)
    if "widths" in st.files:
        assert tuple(st["widths"]) == tuple(cfg.widths), \
            f"checkpoint widths {tuple(st['widths'])} != cfg {tuple(cfg.widths)}"
    _, unravel = ravel_pytree(tmpl)
    return (unravel(jnp.asarray(st["pvec"])), fs, st["t_all"], st["q_all"],
            int(st["k"]) + 1 if "k" in st.files else len(cfg.iters))


def initial_collocation(spec, proxy, cfg: ScenarioCfg):
    rng = np.random.default_rng(cfg.seed)
    t_u, q_u = sampling.uniform(spec, cfg.n_uniform, rng)
    t_b, q_b = sampling.boundary_band(spec, cfg.n_boundary, rng)
    t_s, q_s = sampling.surface_band(spec, cfg.n_surface, rng)
    t_p, q_p = sampling.proxy_policy_paths(spec, proxy, n_paths=8, rng=rng)
    if t_p.size > cfg.n_paths_cap:
        idx = rng.choice(t_p.size, cfg.n_paths_cap, replace=False)
        t_p, q_p = t_p[idx], q_p[idx]
    t_x, q_x = sampling.surface_straddle(spec, n_t=cfg.straddle_nt)
    return (np.concatenate([t_u, t_b, t_s, t_p, t_x]),
            np.vstack([q_u, q_b, q_s, q_p, q_x]))


def train(spec, cfg: ScenarioCfg, path, max_seconds=None, use_gpu_gn=False,
          phase_time_estimate=180.0, max_phases=None, log=print):
    """Run remaining phases; returns dict(status='done'|'paused', k_next, N)."""
    px = Proxy(spec)
    params, fs, t_all, q_all, k0 = load_checkpoint(spec, cfg, path)
    if k0 == 0:
        t_all, q_all = initial_collocation(spec, px, cfg)
        log(f"[{cfg.name}] features {network.n_features(fs)}, N0={t_all.size}, "
            f"widths {cfg.widths}")
    gn = gauss_newton_gpu if use_gpu_gn else gauss_newton
    start = time.time()
    done_ct = 0
    for k in range(k0, len(cfg.iters)):
        if max_phases is not None and done_ct >= max_phases:
            return dict(status='paused', k_next=k, N=t_all.size)
        if max_seconds is not None and time.time() - start > max_seconds - phase_time_estimate:
            return dict(status="paused", k_next=k, N=t_all.size)
        prep = res_mod.prepare_batch(spec, px, t_all, q_all)
        pts, aux = res_mod.split_prep(prep)
        r0 = np.asarray(res_mod.residual_fn(params, prep))
        log(f"[{cfg.name}] phase {k}: N={t_all.size}, start rms "
            f"{np.sqrt((r0**2).mean()):.3e} sup {np.abs(r0).max():.3e}")
        t0 = time.time()
        params, _ = gn(lambda pp, _p=prep: res_mod.residual_fn(pp, _p), params,
                       n_iters=cfg.iters[k], verbose=True, log_every=30,
                       per_sample=(res_mod.residual_point, pts, aux))
        r1 = np.asarray(res_mod.residual_fn(params, prep))
        log(f"[{cfg.name}] phase {k} done: rms {np.sqrt((r1**2).mean()):.3e} "
            f"sup {np.abs(r1).max():.3e} ({time.time()-t0:.0f}s)")
        if k < len(cfg.iters) - 1:
            rng = np.random.default_rng(100 + k)
            def reval(tt, qq, _params=params):
                prep_p = res_mod.prepare_batch(spec, px, tt, qq)
                return jax.jit(lambda: res_mod.residual_fn(_params, prep_p))()
            t_r, q_r = sampling.rar(reval, spec, n_pool=cfg.rar_pool,
                                    n_keep=cfg.rar_keep, rng=rng)
            log(f"[{cfg.name}] RAR: +{t_r.size} pts, pool worst "
                f"{np.abs(np.asarray(reval(t_r[:8], q_r[:8]))).max():.3e}")
            t_all = np.concatenate([t_all, t_r]); q_all = np.vstack([q_all, q_r])
        pvec, _ = ravel_pytree(params)
        np.savez(path, pvec=np.asarray(pvec), t_all=t_all, q_all=q_all,
                 name=cfg.name, k=k, widths=np.array(cfg.widths),
                 n_features=network.n_features(fs))
        done_ct += 1
    return dict(status="done", k_next=len(cfg.iters), N=t_all.size)
