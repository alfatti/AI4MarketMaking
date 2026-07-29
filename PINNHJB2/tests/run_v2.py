"""Re-entrant Phase-1 v2 driver, one bounded stage per invocation, state on disk.

  python tests/run_v2.py phase {exp|logi} <k> <state.npz>   # GN phase k (builds collocation at k=0)
  python tests/run_v2.py cert  {exp|logi} <state.npz>       # certificate + residual localization
  python tests/run_v2.py mol   {exp|logi} <state.npz>       # MOL theta/quote comparison

Recipe (v2): kink features at switching surfaces, surface-band + deterministic surface-
STRADDLE collocation (straddles read the learned kink amplitude at the +-1e-4*zbar scale
the certificate probes), RAR with straddles in the pool, widths (64, 64).
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from hjbpinn import spec as spec_mod, network, residual as res_mod, sampling, validate
from hjbpinn.proxy import Proxy
from hjbpinn.optimizer import gauss_newton

WIDTHS = (64, 64)
ITERS = (90, 60, 60, 60, 60)
ITERS_D5 = (45, 40)
RAR_KEEP = 320


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


def get_spec(name):
    if name == "exp":
        return spec_mod.single_asset_demo_spec()
    if name == "sec6":
        from hjbpinn.bridge import section6_spec
        return section6_spec()
    if name == "d5":
        return d5_spec()
    return logistic_spec()


def d5_spec():
    """Five-asset Section-6-like book: same logistic demand and atom mixture per
    (asset, side); sigma = (1.2,.6,.9,1.05,.75), equicorrelated rho=0.5; small
    mixed drifts; Q_i = 75000. Lattice reference would need 25^5 ~ 9.8e6 states
    x 40-term stencil -- out of reach on this box; the PINN + certificate + MC
    correction are the only validators that survive."""
    d, K = 5, 4
    sig = np.array([1.2, 0.6, 0.9, 1.05, 0.75])
    R = 0.5 * np.ones((d, d)) + 0.5 * np.eye(d)
    Sigma = np.outer(sig, sig) * R
    kind = np.ones((d, 1, 2), int)
    ip = np.zeros((d, 1, 2, 3)); ip[..., 0] = 30.0; ip[..., 1] = 0.7; ip[..., 2] = 30.0
    z = np.broadcast_to(np.array([6250., 12500., 18750., 25000.]), (d, 1, 2, K)).copy()
    p = np.broadcast_to(np.array([0.534, 0.350, 0.097, 0.019]), (d, 1, 2, K)).copy()
    return spec_mod.MarketSpec(d=d, n_tiers=1, T=7.0, gamma=8e-6,
                               mu=np.array([0.1, -0.1, 0.05, 0.0, -0.05]),
                               Sigma=Sigma, Q=np.full(d, 75_000.0), delta_inf=5.0,
                               kind=kind, ip=ip, z_atoms=z, p_atoms=p,
                               c=np.zeros((d, 1, 2)))


def scenario_widths(name):
    return (48, 48) if name in ("sec6", "d5") else WIDTHS


def scenario_straddle_nt(name):
    return {"sec6": 6, "d5": 4}.get(name, 21)


def template(sp, name="exp"):
    fs = network.feature_spec(sp)
    return network.init_params(jax.random.PRNGKey(0), network.n_features(fs),
                               scenario_widths(name)), fs


def run_phase(name, k, path):
    sp = get_spec(name); px = Proxy(sp)
    tmpl, fs = template(sp, name)
    _, unravel = ravel_pytree(tmpl)
    if k == 0:
        rng = np.random.default_rng(42)
        t_u, q_u = sampling.uniform(sp, 1024, rng)
        t_b, q_b = sampling.boundary_band(sp, 224, rng)
        t_s, q_s = sampling.surface_band(sp, 288, rng)
        t_p, q_p = sampling.proxy_policy_paths(sp, px, n_paths=8, rng=rng)
        if t_p.size > 160:
            idx = rng.choice(t_p.size, 160, replace=False); t_p, q_p = t_p[idx], q_p[idx]
        t_x, q_x = sampling.surface_straddle(sp, n_t=scenario_straddle_nt(name))
        t_all = np.concatenate([t_u, t_b, t_s, t_p, t_x])
        q_all = np.vstack([q_u, q_b, q_s, q_p, q_x])
        params = tmpl
        print(f"[{name}] features {network.n_features(fs)}, "
              f"surfaces {np.asarray(fs['sp']).tolist()}")
    else:
        st = np.load(path)
        params = unravel(jnp.asarray(st["pvec"]))
        t_all, q_all = st["t_all"], st["q_all"]

    prep = res_mod.prepare_batch(sp, px, t_all, q_all)
    pts, aux = res_mod.split_prep(prep)
    r0 = np.asarray(res_mod.residual_fn(params, prep))
    print(f"phase {k}: N={t_all.size}, start rms {np.sqrt((r0**2).mean()):.3e} "
          f"sup {np.abs(r0).max():.3e}", flush=True)
    t0 = time.time()
    params, _ = gauss_newton(lambda pp, _p=prep: res_mod.residual_fn(pp, _p), params,
                             n_iters=(ITERS_D5 if name == 'd5' else ITERS)[k], verbose=True, log_every=30,
                             per_sample=(res_mod.residual_point, pts, aux))
    r1 = np.asarray(res_mod.residual_fn(params, prep))
    print(f"phase {k} done: rms {np.sqrt((r1**2).mean()):.3e} sup {np.abs(r1).max():.3e} "
          f"({time.time()-t0:.0f}s)")
    if k < len(ITERS_D5 if name == 'd5' else ITERS) - 1:
        rng = np.random.default_rng(100 + k)
        def reval(tt, qq, _params=params):
            prep_p = res_mod.prepare_batch(sp, px, tt, qq)
            return jax.jit(lambda: res_mod.residual_fn(_params, prep_p))()
        t_r, q_r = sampling.rar(reval, sp, n_pool=6000 if sp.d == 1 else 9000,
                                n_keep=RAR_KEEP if sp.d == 1 else 480, rng=rng)
        print(f"RAR: added {t_r.size} points, pool worst |r| = "
              f"{np.abs(np.asarray(reval(t_r[:8], q_r[:8]))).max():.3e}")
        t_all = np.concatenate([t_all, t_r]); q_all = np.vstack([q_all, q_r])
    pvec, _ = ravel_pytree(params)
    np.savez(path, pvec=np.asarray(pvec), t_all=t_all, q_all=q_all, name=name, k=k)
    print(f"saved {path}")


def load_params(sp, path):
    tmpl, fs = template(sp, name)
    _, unravel = ravel_pytree(tmpl)
    return unravel(jnp.asarray(np.load(path)["pvec"])), tmpl, fs


def run_cert(name, path):
    sp = get_spec(name); px = Proxy(sp)
    params, tmpl, fs = load_params(sp, path)
    res_scale, _ = res_mod.scales(sp)
    t_c, q_c = validate.certificate_grid(sp, n_t=25, n_q_uniform=4000)
    eps1, _ = validate.certificate(sp, px, params, t_c, q_c)
    eps0, _ = validate.certificate(sp, px, tmpl, t_c, q_c)
    print(f"[{name}] certificate sup|r| physical: proxy {eps0:.4e} -> PINN {eps1:.4e} "
          f"(scaled {eps1/res_scale:.3e})")
    tg = np.repeat(np.linspace(0, sp.T * 0.999, 40), 400)
    qg_d = np.tile(np.linspace(-sp.Q[0], sp.Q[0], 400), 40)[:, None]
    rs = []
    for s in range(0, tg.size, 8000):
        prep_d = res_mod.prepare_batch(sp, px, tg[s:s + 8000], qg_d[s:s + 8000])
        rs.append(np.asarray(jax.jit(lambda p=prep_d: res_mod.residual_fn(params, p))()))
    a = np.abs(np.concatenate(rs))
    surfs = np.concatenate([[sp.Q[0] - z, -(sp.Q[0] - z)] for z in np.unique(sp.z_atoms[0])])
    dist = np.min(np.abs(qg_d[:, 0][:, None] - surfs[None, :]), axis=1)
    for lo, hi in [(0, 0.1), (0.1, 0.5), (0.5, 1.5), (1.5, 7)]:
        m = (dist >= lo) & (dist < hi)
        print(f"  dist to surface [{lo},{hi}): sup|r| = {a[m].max():.3e} (n={m.sum()})")


def run_mol(name, path):
    sp = get_spec(name); px = Proxy(sp)
    params, tmpl, fs = load_params(sp, path)
    _, theta_scale = res_mod.scales(sp)
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


if __name__ == "__main__":
    cmd, name = sys.argv[1], sys.argv[2]
    if cmd == "phase":
        run_phase(name, int(sys.argv[3]), sys.argv[4])
    elif cmd == "cert":
        run_cert(name, sys.argv[3])
    elif cmd == "mol":
        run_mol(name, sys.argv[3])
    import resource
    print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
