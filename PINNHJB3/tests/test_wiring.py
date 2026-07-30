"""Wiring regression: the fast residual (analytic proxy differences, analytic d_t
theta_check from the ODE right-hand sides, precomputed masks, flattened stencil) must
equal a brute-force independent implementation at RANDOM NONZERO eta — direct
subtraction of theta values, central finite difference in t, explicit feasibility test.

This is the test that would catch a stencil/mask/broadcast/sign error anywhere in
prepare_batch or residual_point; Phase 0's machine-zero tests only exercise the eta = 0
path. Run: python tests/test_wiring.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

from hjbpinn import spec as spec_mod, network, residual as res_mod
from hjbpinn import hamiltonians as ham
from hjbpinn.proxy import Proxy

sys.path.insert(0, os.path.dirname(__file__))
from run_v2 import logistic_spec

failures = []


def check(name, ok, detail=""):
    print(f"{'[PASS]' if ok else '[FAIL]'} {name} {detail}")
    if not ok:
        failures.append(name)


def brute_residual(sp, px, params, fs, theta_scale, res_scale, t, q):
    """Independent evaluation: theta by direct value, d_t by central FD, explicit masks."""
    sj = sp.to_jax()

    def theta_full(tt, qq):
        return float(px.theta(tt, jnp.asarray(qq))) + float(
            network.eta(params, tt, jnp.asarray(qq), fs, theta_scale))

    h = 1e-6 * sp.T
    tp, tm = min(t + h, sp.T * (1 - 1e-12)), max(t - h, 0.0)
    dt = (theta_full(tp, q) - theta_full(tm, q)) / (tp - tm)
    stat = float(q @ sp.mu - 0.5 * sp.gamma * q @ sp.Sigma @ q)
    th0 = theta_full(t, q)
    jump = 0.0
    for i in range(sp.d):
        for n in range(sp.n_tiers):
            for side, sgn in ((0, 1.0), (1, -1.0)):
                for k in range(sp.z_atoms.shape[-1]):
                    z = float(sp.z_atoms[i, n, side, k])
                    pz = float(sp.p_atoms[i, n, side, k])
                    qn = np.array(q, float); qn[i] += sgn * z
                    if abs(qn[i]) > sp.Q[i] + 1e-12:
                        continue                                  # censored
                    p = (th0 - theta_full(t, qn) + float(sp.c[i, n, side])) / z
                    Hv = float(ham.H0(jnp.asarray(p), sj["kind"][i, n, side],
                                      sj["ip"][i, n, side], sp.delta_inf))
                    jump += pz * z * Hv
    return (dt + stat + jump) / res_scale


for name, sp in (("exp (asymmetric, 1 tier, 1 atom)", spec_mod.single_asset_demo_spec()),
                 ("logistic (2 tiers, 2 atoms)", logistic_spec())):
    px = Proxy(sp)
    fs = network.feature_spec(sp)
    res_scale, theta_scale = res_mod.scales(sp)
    key = jax.random.PRNGKey(11)
    params = network.init_params(key, network.n_features(fs), (32, 32))
    W, b = params[-1]                                             # break the zero init
    params[-1] = (jax.random.normal(jax.random.PRNGKey(12), W.shape) * 0.4, b)

    rng = np.random.default_rng(5)
    # include points straddling the switching surfaces, where masks flip
    t_s = np.concatenate([rng.uniform(0, sp.T * 0.99, 8), [0.0, 0.3 * sp.T]])
    q_s = np.vstack([rng.uniform(-sp.Q, sp.Q, (8, sp.d)),
                     [sp.Q - sp.z_atoms.max() + 1e-5], [sp.Q - 1e-9]])
    prep = res_mod.prepare_batch(sp, px, t_s, q_s)
    r_fast = np.asarray(res_mod.residual_fn(params, prep))
    r_brute = np.array([brute_residual(sp, px, params, fs, theta_scale, res_scale,
                                       float(tt), qq) for tt, qq in zip(t_s, q_s)])
    err = np.abs(r_fast - r_brute).max()
    check(f"residual wiring, {name}", err < 1e-6, f"max diff {err:.2e}")

print("\n" + ("WIRING TESTS PASSED" if not failures else f"FAILURES: {failures}"))
sys.exit(0 if not failures else 1)
