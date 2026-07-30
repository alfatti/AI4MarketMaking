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
from hjbpinn import scenarios as scen_mod, training as train_mod

from hjbpinn.scenarios import d5_spec, logistic_spec
from hjbpinn.scenarios import get_cfg as scenario_cfg
from hjbpinn.training import train as _train


def template_for(sp, widths):
    fs = network.feature_spec(sp)
    return network.init_params(jax.random.PRNGKey(0), network.n_features(fs),
                               tuple(widths)), fs


def get_spec(name):
    return scen_mod.get_spec(name)

WIDTHS = (64, 64)
ITERS = (90, 60, 60, 60, 60)
ITERS_D5 = (45, 40)
RAR_KEEP = 320


def scenario_widths(name):
    return scenario_cfg(name).widths


def scenario_straddle_nt(name):
    return scenario_cfg(name).straddle_nt


def template(sp, name="exp"):
    return train_mod.template(sp, scen_mod.get_cfg(name))


def _template_legacy(sp, name="exp"):
    return template_for(sp, scenario_cfg(name).widths)


def run_phase(name, k, path):
    sp = scen_mod.get_spec(name)
    st = train_mod.train(sp, scen_mod.get_cfg(name), path, max_phases=1)
    print(f'saved {path} ({st})')
    return


def _run_phase_legacy(name, k, path):
    """One phase per invocation (bounded-command workflow): delegates to
    hjbpinn.training.train with max_seconds=0, which runs exactly the next
    pending phase then pauses. `k` is kept for CLI compatibility and checked
    against the checkpoint's own phase counter."""
    import os
    if k == 0 and os.path.exists(path):
        os.remove(path)
    _train(name, path, max_seconds=0)


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
