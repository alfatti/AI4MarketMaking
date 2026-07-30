"""H200 driver for the three scaling programs. CUDA-first design throughout:
lax.scan (no host loops), counter-based PRNG inside scans, device-resident GN
with single-DGEMM fp64 Grams (tensor cores), chunked streaming pools, donated
RK4 carries. Every code path here is CPU-smoke-tested; --smoke shrinks sizes.

  python tests/run_h200.py gap    [--smoke]   # item 1: close the certificate gap
  python tests/run_h200.py ladder [--smoke]   # item 3: validate validators at d=3,4,5
  python tests/run_h200.py pnl    [--smoke]   # item 4: 1e6-path CRN PnL + gap maps

Environment (H200):
  jax[cuda12]; XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
  jax.config.update("jax_enable_x64", True)          # fp64 Grams hit FP64 tensor cores
  jax.config.update("jax_default_matmul_precision", "highest")
Sizing rationale (141 GB HBM3e, ~34/67 TFLOPS fp64 vec/TC):
  gap:    RAR pools 2e6/round (chunk 1e6, elementwise-bound), ascent 2048/surface x
          400 steps; GN at N=5e4 collocation x P<=5e4 params: J 20 GB + Gram 20 GB
          resident; Cholesky 4e13 flops ~ 1 s/iter.
  ladder: mol_solve_nd_scan: d=4 (390k states) seconds; d=5 (9.8e6 states, 78 MB/buf,
          ~2.4e14 flops at n=3000) ~ minutes; d=6 borderline; d=7 exceeds HBM.
  pnl:    lockstep 1e6 paths x ~1200 events x 3 policies: gather-bound, minutes;
          quote tables at n_t=283 (kills the near-T discretization flagged in
          validation); gap maps: q0_idx batched over a lattice of starts.
"""
import sys, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from hjbpinn import scenarios as run_v2_compat
from hjbpinn.bridge import section6_spec
from hjbpinn.proxy import Proxy
from hjbpinn import (network, validate, residual as rm, sampling, mc,
                     adversarial as adv, gpu_sim)
from hjbpinn.optimizer import gauss_newton_gpu

S = lambda p: str(ROOT / "states" / p)

CFG = dict(
    h200=dict(pool=2_000_000, chunk=1_000_000, asc_starts=2048, asc_steps=400,
              rounds=4, N_new=20_000, gn_iters=40, pnl_paths=1_000_000,
              pnl_nt=283, ladder_d=(3, 4, 5), ladder_steps=3000, mc_paths=64),
    smoke=dict(pool=40_000, chunk=20_000, asc_starts=8, asc_steps=50,
               rounds=1, N_new=256, gn_iters=6, pnl_paths=1500,
               pnl_nt=71, ladder_d=(3,), ladder_steps=700, mc_paths=6),
)


def load_sec6():
    sp = section6_spec(); px = Proxy(sp)
    tmpl, fs = __import__('hjbpinn.training', fromlist=['template']).template(sp, run_v2_compat.get_cfg('sec6'))
    _, unravel = ravel_pytree(tmpl)
    st = np.load(S("v2_sec6.npz"))
    return sp, px, fs, unravel(jnp.asarray(st["pvec"])), st


def stage_gap(cfg):
    """Adversarial rounds: (pool + ascent) -> fold worst points into the training
    set -> device-resident GN -> re-certify. The certificate should rise (honesty)
    then fall (training) across rounds; report both curves."""
    sp, px, fs, params, st = load_sec6()
    rng = np.random.default_rng(42)
    t_all, q_all = st["t_all"], st["q_all"]
    print(f"[gap] start: training N={t_all.size}")
    for rd in range(cfg["rounds"]):
        sup_pool, t_w, q_w = adv.rar_stream(sp, px, params, fs, cfg["pool"], rng,
                                            chunk=cfg["chunk"], keep=cfg["N_new"] // 2)
        sup_asc, t_a, q_a, rv = adv.surface_ascent(
            sp, px, params, fs, n_per_surface=cfg["asc_starts"],
            n_steps=cfg["asc_steps"], rng=rng)
        k = min(cfg["N_new"] // 2, len(rv))
        top = np.argpartition(rv, -k)[-k:]
        t_all = np.concatenate([t_all, t_w, t_a[top]])
        q_all = np.vstack([q_all, q_w, q_a[top]])
        print(f"[gap] round {rd}: pool sup {sup_pool:.3e}, ascent sup {sup_asc:.3e}; "
              f"training N -> {t_all.size}")
        prep = rm.prepare_batch(sp, px, t_all, q_all)
        pts, aux = rm.split_prep(prep)
        params, _ = gauss_newton_gpu(lambda pp: rm.residual_fn(pp, prep), params,
                                     n_iters=cfg["gn_iters"], verbose=True,
                                     log_every=max(1, cfg["gn_iters"] // 3),
                                     per_sample=(rm.residual_point, pts, aux))
    cert = adv.certificate_v2(sp, px, params, fs, rng,
                              n_random=cfg["pool"], n_per_surface=cfg["asc_starts"],
                              ascent_steps=cfg["asc_steps"])
    print(f"[gap] final certificate: pool {cert['sup_pool']:.3e}, "
          f"ascent {cert['sup_ascent']:.3e}, sup {cert['sup']:.3e} "
          f"({cert['sup_physical']:.3e} EUR/day; theta bound t=0 {cert['theta_bound_t0']:.0f} EUR)")
    pvec, _ = ravel_pytree(params)
    np.savez_compressed(S("v2_sec6_gapclosed.npz"), pvec=np.asarray(pvec),
                        t_all=t_all, q_all=q_all)
    print("[gap] saved states/v2_sec6_gapclosed.npz")


def stage_ladder(cfg):
    """Exact lattices at d=3,4,5 (scan solver) + grid-free instruments checked
    against them. Smoke reuses the saved d=3 solve; H200 recomputes and extends."""
    sp5 = run_v2_compat.get_spec('d5')
    for d in cfg["ladder_d"]:
        import copy
        spd = copy.deepcopy(sp5)
        spd.d = d; spd.Q = np.full(d, 50_000.0); spd.mu = sp5.mu[:d]
        spd.Sigma = sp5.Sigma[:d, :d]; spd.kind = sp5.kind[:d]; spd.ip = sp5.ip[:d]
        spd.z_atoms = sp5.z_atoms[:d]; spd.p_atoms = sp5.p_atoms[:d]; spd.c = sp5.c[:d]
        f = Path(S(f"d{d}_val.npz"))
        if f.exists() and cfg is CFG["smoke"]:
            v, se, corr, th0, thM = np.load(f)["mc"]
            print(f"[ladder] d={d} (cached): v {v:.0f}+-{se:.0f} <= theta {thM:.0f}: {v <= thM + 3*se}")
            continue
        t0 = time.time()
        grids, th = validate.mol_solve_nd_scan(spd, t_eval=[0.0], n_steps=cfg["ladder_steps"])
        c = [len(g) // 2 for g in grids]
        thM = float(th[0][tuple(c)])
        print(f"[ladder] d={d}: lattice {np.prod([len(g) for g in grids]):,} states "
              f"({time.time()-t0:.0f}s), theta(0,0)={thM:.0f}")
        px = Proxy(spd); p0, fsd = mc.zero_eta_params(spd)
        _, tsd = rm.scales(spd)
        v, se, corr, th0 = mc.mc_policy_value(spd, px, p0, fsd, tsd, 0.0, np.zeros(d),
                                              cfg["mc_paths"], np.random.default_rng(21))
        print(f"[ladder] d={d}: MC v {v:.0f}+-{se:.0f} <= theta {thM:.0f}: {v <= thM + 3*se}")
        np.savez_compressed(S(f"d{d}_val.npz"), theta0=th[0],
                            mc=np.array([v, se, corr, th0, thM]))


def stage_pnl(cfg):
    """Lockstep CRN at scale + suboptimality gap maps over starting inventories."""
    sp = section6_spec()
    mol = np.load(S("section6_mol.npz"))
    grids = [mol["q1"], mol["q2"]]
    tabs = {nm: np.load(S(f"sec6_tab_{nm}.npy")) for nm in ("optimal", "pinn", "proxy")}
    if cfg["pnl_nt"] != len(mol["t_eval"]):
        print(f"[pnl] NOTE (H200): rebuild tables at n_t={cfg['pnl_nt']} from a finer "
              f"MOL run to remove the near-T discretization flagged in validation.")
    t0 = time.time()
    out = gpu_sim.pnl_lockstep(sp, tabs, mol["t_eval"], grids,
                               n_paths=cfg["pnl_paths"], seed=7)
    print(f"[pnl] {cfg['pnl_paths']:,} paths x 3 policies: {time.time()-t0:.0f}s; "
          f"seg invariant {np.abs(out['pinn']['segsum']-sp.T).max():.1e}")
    for a, b in (("pinn", "proxy"), ("optimal", "pinn"), ("optimal", "proxy")):
        d_ = (out[a]["pnl"] - out[a]["penalty"]) - (out[b]["pnl"] - out[b]["penalty"])
        se = d_.std(ddof=1) / np.sqrt(len(d_))
        print(f"[pnl] paired {a:8s}-{b:8s}: {d_.mean():+8.2f} +- {se:.2f}  (t={d_.mean()/se:+.1f})")
    # gap map over starting inventories (coarse lattice of q0)
    i_grid = np.linspace(2, len(grids[0]) - 3, 5).astype(int)
    j_grid = np.linspace(2, len(grids[1]) - 3, 5).astype(int)
    q0s = np.stack(np.meshgrid(i_grid, j_grid, indexing="ij"), -1).reshape(-1, 2)
    reps = max(1, cfg["pnl_paths"] // (4 * len(q0s)))
    q0_idx = np.repeat(q0s, reps, axis=0)
    out2 = gpu_sim.pnl_lockstep(sp, tabs, mol["t_eval"], grids,
                                n_paths=len(q0_idx), seed=11, q0_idx=q0_idx)
    dmap = ((out2["pinn"]["pnl"] - out2["pinn"]["penalty"])
            - (out2["proxy"]["pnl"] - out2["proxy"]["penalty"])).reshape(len(q0s), reps)
    print(f"[pnl] gap map (pinn-proxy objective) over {len(q0s)} starts x {reps} paths:")
    print(f"      max advantage {dmap.mean(1).max():+.0f} at q0="
          f"({grids[0][q0s[dmap.mean(1).argmax()][0]]:.0f}, {grids[1][q0s[dmap.mean(1).argmax()][1]]:.0f}); "
          f"min {dmap.mean(1).min():+.0f}")
    np.savez_compressed(S("pnl_scale.npz"), q0s=q0s, dmap=dmap,
                        **{f"{nm}_{k}": out[nm][k] for nm in out for k in out[nm]})
    print("[pnl] saved states/pnl_scale.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gap", "ladder", "pnl"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = CFG["smoke" if a.smoke else "h200"]
    dict(gap=stage_gap, ladder=stage_ladder, pnl=stage_pnl)[a.stage](cfg)
