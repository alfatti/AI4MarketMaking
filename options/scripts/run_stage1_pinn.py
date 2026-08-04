#!/usr/bin/env python3
"""Stage-1 PINN training driver.  --smoke runs the CPU-sized budget used in
docs/VALIDATION.md; the default is the H200 full budget (Adam warmup +
Gauss-Newton-CG polish, large batches).  Re-entrant: pass --resume to
continue from the checkpoint.

Usage:
    python scripts/run_stage1_pinn.py [--smoke] [--resume] [--out CKPT]
"""
import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from optmm.core.risk import BBG_MARKET
from optmm.instruments.book import build_bbg_book
from optmm.solvers.pinn.model import PinnSpec, init_params, value
from optmm.solvers.pinn.train import (adam_train, collinearity, gn_polish,
                                      make_loss, value_error_vs_reference)
from optmm.solvers.reduced2d import Channels, solve_reduced2d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="checkpoints/pinn_stage1.pkl")
    a = ap.parse_args()

    if a.smoke:
        hidden, steps, batch, gn_iters, gn_batch = (64, 64, 64), 6000, 256, 0, 0
    else:
        hidden, steps, batch, gn_iters, gn_batch = (128,) * 3, 60000, 4096, 40, 8192

    book = build_bbg_book(BBG_MARKET)
    ch = Channels.from_book(book)
    spec = PinnSpec(T=0.0012, nu_lo=0.0144, nu_hi=0.0324,
                    w=jnp.asarray(book.w), Vbar=1e7, hidden=hidden)
    loss_fn, batched = make_loss(spec, ch, BBG_MARKET, 1e-3, 0.0, 1e7)
    key = jax.random.PRNGKey(0)
    key, k0 = jax.random.split(key)
    if a.resume and os.path.exists(a.out):
        with open(a.out, "rb") as f:
            params = jax.tree_util.tree_map(
                jnp.asarray, pickle.load(f)["params"])
        print(f"resumed from {a.out}")
    else:
        params = init_params(spec, k0)

    t0 = time.time()
    params, hist = adam_train(params, loss_fn, spec, key, steps=steps,
                              batch=batch, log_every=max(steps // 20, 1))
    if gn_iters:
        params = gn_polish(params, batched, spec, key, iters=gn_iters,
                           batch=gn_batch)
    print(f"training: {time.time() - t0:.0f}s")

    ref_path = "checkpoints/ref2d_full.npz"
    if not os.path.exists(ref_path):
        ref = solve_reduced2d(book, BBG_MARKET, 1e-3, 1e7, 0.0012,
                              nt=1440, n_nu=31, n_V=801)
        os.makedirs("checkpoints", exist_ok=True)
        np.savez(ref_path, V_grid=np.asarray(ref.V_grid),
                 v_row=np.asarray(ref.v0[15]))
    ref = np.load(ref_path)
    key, k1, k2 = jax.random.split(key, 3)
    med_cos, _ = collinearity(params, spec, k1)
    med_e, max_e = value_error_vs_reference(params, spec, k2,
                                            ref["V_grid"], ref["v_row"])
    print(f"collinearity median |cos|: {med_cos:.4f}")
    print(f"value err vs 2D ref: median {med_e:.0f} / max {max_e:.0f} EUR")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params),
                     "hidden": hidden,
                     "metrics": {"med_cos": med_cos, "med_err": med_e,
                                 "max_err": max_e},
                     "loss_hist": hist}, f)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
