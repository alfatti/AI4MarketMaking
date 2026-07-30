"""Phase 1 v2 — break the v1 ceilings.

Diagnosis (documented in RUN_REPORT): the trained-residual sup concentrated AT the
switching surfaces q = +-(Q - z_k) with a 25x interior gradient — an irreducible floor
for any smooth-in-q ansatz, because the equation's RHS (hence d_t theta) jumps there.
Recipe changes vs v1: (i) kink features -expm1(-|q - s|/zbar) per surface, (ii) surface-
band collocation, (iii) two RAR phases, (iv) widths (64, 64).

v1 ceilings (for reference, same specs):
  exp   : train rms 3.39e-4, sup 5.95e-3 | theta sup 2.20e-3 | quote RMSE 9.24e-4 (t=0)
  logist: train rms 5.18e-4, sup 6.64e-3 | theta sup 6.73e-3 | quote RMSE 8.48e-4 (t=0)

Run: python tests/demo_phase1_v2.py
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

from hjbpinn import spec as spec_mod, network, residual as res_mod, sampling, validate
from hjbpinn.proxy import Proxy
from hjbpinn.optimizer import gauss_newton


def logistic_spec():
    d, n, K = 1, 2, 2
    kind = np.ones((d, n, 2), int)
    ip = np.zeros((d, n, 2, 3))
    ip[0, 0, 0] = (2.0, 0.7, 3.0); ip[0, 0, 1] = (1.8, 0.6, 3.4)
    ip[0, 1, 0] = (0.9, 0.9, 2.2); ip[0, 1, 1] = (1.1, 0.8, 2.6)
    z = np.zeros((d, n, 2, K)); z[..., 0] = 1.0; z[..., 1] = 2.0
    p = np.zeros((d, n, 2, K)); p[..., 0] = 0.7; p[..., 1] = 0.3
    c = np.full((d, n, 2), 0.02)
    return spec_mod.MarketSpec(d=d, n_tiers=n, T=1.0, gamma=0.25, mu=np.array([-0.04]),
                               Sigma=np.array([[1.2 ** 2]]), Q=np.array([6.0]),
                               delta_inf=3.0, kind=kind, ip=ip, z_atoms=z, p_atoms=p, c=c)


def run(name, sp, iters=(120, 80, 80), rar_keep=320, widths=(64, 64), seed=42):
    print(f"\n================ {name} ================")
    px = Proxy(sp)
    fs = network.feature_spec(sp)
    res_scale, theta_scale = res_mod.scales(sp)
    print(f"features: {network.n_features(fs)} "
          f"(surfaces at {np.asarray(fs['sp']).tolist()})")
    rng = np.random.default_rng(seed)

    t_u, q_u = sampling.uniform(sp, 1024, rng)
    t_b, q_b = sampling.boundary_band(sp, 224, rng)
    t_s, q_s = sampling.surface_band(sp, 288, rng)
    t_p, q_p = sampling.proxy_policy_paths(sp, px, n_paths=8, rng=rng)
    if t_p.size > 160:
        idx = rng.choice(t_p.size, 160, replace=False); t_p, q_p = t_p[idx], q_p[idx]
    t_x, q_x = sampling.surface_straddle(sp, n_t=21)
    t_all = np.concatenate([t_u, t_b, t_s, t_p, t_x])
    q_all = np.vstack([q_u, q_b, q_s, q_p, q_x])
    params = network.init_params(jax.random.PRNGKey(0), network.n_features(fs), widths)

    t0 = time.time()
    for phase, n_it in enumerate(iters):
        prep = res_mod.prepare_batch(sp, px, t_all, q_all)
        pts, aux = res_mod.split_prep(prep)
        r0 = np.asarray(res_mod.residual_fn(params, prep))
        print(f"phase {phase}: N={t_all.size}, start rms {np.sqrt((r0**2).mean()):.3e} "
              f"sup {np.abs(r0).max():.3e}")
        params, _ = gauss_newton(lambda pp, _prep=prep: res_mod.residual_fn(pp, _prep),
                                 params, n_iters=n_it, verbose=True, log_every=max(n_it // 3, 1),
                                 per_sample=(res_mod.residual_point, pts, aux))
        if phase < len(iters) - 1:
            def reval(tt, qq, _params=params):
                prep_p = res_mod.prepare_batch(sp, px, tt, qq)
                return res_mod.residual_fn(_params, prep_p)
            t_r, q_r = sampling.rar(reval, sp, n_pool=6000, n_keep=rar_keep, rng=rng)
            t_all = np.concatenate([t_all, t_r]); q_all = np.vstack([q_all, q_r])
        import gc; jax.clear_caches(); gc.collect()
    train_s = time.time() - t0

    prep = res_mod.prepare_batch(sp, px, t_all, q_all)
    r1 = np.asarray(res_mod.residual_fn(params, prep))
    print(f"final train rms {np.sqrt((r1**2).mean()):.3e}, sup {np.abs(r1).max():.3e} "
          f"({train_s:.0f}s)")

    t_c, q_c = validate.certificate_grid(sp, n_t=25, n_q_uniform=4000)
    eps1, _ = validate.certificate(sp, px, params, t_c, q_c)
    eps0, _ = validate.certificate(sp, px, network.init_params(
        jax.random.PRNGKey(0), network.n_features(fs), widths), t_c, q_c)
    print(f"certificate sup|r| (physical): proxy {eps0:.4e} -> PINN {eps1:.4e} "
          f"(bound at t=0: {eps1 * sp.T:.4e})")

    t_slices = [0.0, 0.5 * sp.T]
    qg, th_mol = validate.mol_solve(sp, t_slices)

    def theta_pinn(t, q):
        return px.theta_fast(t, jnp.asarray(q)) + float(
            network.eta(params, t, jnp.asarray(q), fs, theta_scale))

    for ti, tt in enumerate(t_slices):
        th_p = np.array([theta_pinn(tt, np.array([qq])) for qq in qg])
        th_c = np.array([px.theta_fast(tt, np.array([qq])) for qq in qg])
        ref = th_mol[ti]
        d_ref = validate.quote_table(sp, lambda t, q, _ti=ti: float(
            th_mol[_ti][int(round((q[0] + sp.Q[0]) / (qg[1] - qg[0])))]), tt, qg)
        d_pinn = validate.quote_table(sp, lambda t, q: theta_pinn(tt, q), tt, qg)
        d_prox = validate.quote_table(sp, lambda t, q: px.theta_fast(tt, q), tt, qg)
        band = np.abs(qg) >= sp.Q[0] - 2 * sp.zbar()[0]
        print(f"t={tt}: theta sup  proxy {np.abs(th_c-ref).max():.3e} -> "
              f"PINN {np.abs(th_p-ref).max():.3e} | quote RMSE "
              f"proxy {validate.fill_weighted_quote_error(sp, d_ref, d_prox):.4e} -> "
              f"PINN {validate.fill_weighted_quote_error(sp, d_ref, d_pinn):.4e} "
              f"(band {validate.fill_weighted_quote_error(sp, d_ref[band], d_pinn[band]):.4e})")
    return params


if __name__ == "__main__":
    import resource
    which = sys.argv[1] if len(sys.argv) > 1 else "exp"
    if which == "exp":
        run("d=1 asymmetric exponential (v1 baseline spec)", spec_mod.single_asset_demo_spec())
    else:
        run("d=1 logistic 2-tier 2-atom (rfqsim family)", logistic_spec())
    print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
    print("\nPHASE 1 v2 SCENARIO COMPLETE")
