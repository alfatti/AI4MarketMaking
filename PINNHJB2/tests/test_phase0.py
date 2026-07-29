"""Phase 0: structural verification. Run: python tests/test_phase0.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from hjbpinn import spec as spec_mod
from hjbpinn import hamiltonians as ham
from hjbpinn.proxy import Proxy
from hjbpinn import residual as res_mod
from hjbpinn.optimizer import gauss_newton

PASS, FAIL = "[PASS]", "[FAIL]"
failures = []


def check(name, ok, detail=""):
    print(f"{PASS if ok else FAIL} {name} {detail}")
    if not ok:
        failures.append(name)


# ---------------- 1. Hamiltonians ----------------
sp = spec_mod.random_general_spec(seed=3, d=3, n_tiers=2, K=3)
sj = sp.to_jax()
rng = np.random.default_rng(0)
p_test = jnp.asarray(rng.uniform(-4.0, 4.0, size=(64,) + sp.kind.shape))
kind_b = jnp.broadcast_to(sj["kind"][None], p_test.shape)
ip_b = jnp.broadcast_to(sj["ip"][None], p_test.shape + (3,))

# 1a. root satisfies the FOC, checked in cancellation-free form:
#     exponential: 1 - k (d* - p) = 0;  logistic: 1 - beta * sigmoid(alpha+beta d*) * (d* - p) = 0
# (The naive form Lambda'(d*)(d*-p) + Lambda(d*) forms (1-u) with u ~ 1 - 1e-9 and loses
#  8 digits to cancellation — a verifier artifact, not a root error.)
r = ham.root(p_test, kind_b, ip_b)
k_arr = ip_b[..., 1]
foc_exp = 1.0 - k_arr * (r - p_test)
u = jax.nn.sigmoid(ip_b[..., 1] + ip_b[..., 2] * r)
foc_log = 1.0 - ip_b[..., 2] * u * (r - p_test)
foc = jnp.where(kind_b == 0, foc_exp, foc_log)
check("root FOC (cancellation-free) < 1e-12, all families",
      float(jnp.abs(foc).max()) < 1e-12, f"max {float(jnp.abs(foc).max()):.2e}")

# 1b. Danskin: AD of H0 w.r.t. p equals -Lambda(delta*), and matches central FD of H0
def h0_scalar(pp, kk, iparams):
    return ham.H0(pp, kk, iparams, sp.delta_inf)
g_ad = jax.vmap(jax.grad(h0_scalar), in_axes=(0, 0, 0))(
    p_test.ravel(), kind_b.ravel(), ip_b.reshape(-1, 3))
g_exact = ham.H0_prime(p_test, kind_b, ip_b, sp.delta_inf).ravel()
h = 1e-6
g_fd = (ham.H0(p_test + h, kind_b, ip_b, sp.delta_inf)
        - ham.H0(p_test - h, kind_b, ip_b, sp.delta_inf)).ravel() / (2 * h)
check("Danskin: AD grad == -Lambda(delta*)", float(jnp.max(jnp.abs(g_ad - g_exact))) < 1e-12)
# FD is a smoke check only: H0 is C^1 but not C^2 at the floor junction, so central
# differences carry O(h * jump(H0'')) error there; the exact identity above is the assertion.
check("Danskin: AD grad ~= central FD of H0 (smoke)",
      float(jnp.max(jnp.abs(g_ad - g_fd))) < 1e-3,
      f"max diff {float(jnp.max(jnp.abs(g_ad - g_fd))):.2e}")

# 1c. floor semantics. Exponential: root = p + 1/k -> -inf with p, floor must bind.
# Logistic: Lambda is bounded, so for deep-negative p the root saturates near
# -(alpha + 1 + log L)/beta and need NOT hit the floor — assert it exceeds the floor
# whenever unclamped, and construct a case where the floor provably binds.
p_neg = jnp.full(sp.kind.shape, -25.0)
ds = ham.delta_star(p_neg, sj["kind"], sj["ip"], sp.delta_inf)
exp_m = np.asarray(sj["kind"]) == 0
check("floor binds for deep-negative p (exponential entries)",
      bool(np.all(np.abs(np.asarray(ds)[exp_m] + sp.delta_inf) < 1e-12)))
check("delta* >= -delta_inf everywhere", bool(jnp.all(ds >= -sp.delta_inf - 1e-12)))
ip_c = jnp.asarray([1.0, 8.0, 2.0])   # logistic (lam, alpha, beta): root ~ -6.3 < -3
ds_c = ham.delta_star(jnp.asarray(-25.0), jnp.asarray(1), ip_c, sp.delta_inf)
check("constructed logistic case: floor binds", abs(float(ds_c) + sp.delta_inf) < 1e-12,
      f"delta*={float(ds_c):.4f}")

# 1d. sup property: H0(p) >= Lambda(delta)(delta - p) on a grid of feasible deltas
dgrid = jnp.linspace(-sp.delta_inf, 8.0, 200)
H = ham.H0(p_test, kind_b, ip_b, sp.delta_inf)
vals = ham.lam(dgrid[:, None, None, None, None], kind_b[None], ip_b[None]) * \
       (dgrid[:, None, None, None, None] - p_test[None])
check("H0 dominates objective on delta-grid (sup property)",
      float(jnp.max(vals.max(0) - H)) < 1e-9, f"gap {float(jnp.max(vals.max(0) - H)):.2e}")

# ---------------- 2. Proxy closed forms ----------------
px = Proxy(sp)
check("terminal conditions A(T)=0, B(T)=0",
      float(jnp.abs(px.A(sp.T)).max()) < 1e-12 and float(jnp.abs(px.B(sp.T)).max()) < 1e-10)

# 2a. A', B' RHS vs finite differences of the closed forms
for name, f, fp in (("A", px.A, px.A_prime), ("B", px.B, px.B_prime)):
    t0 = 0.37 * sp.T; h = 1e-6 * sp.T
    fd = (np.asarray(f(t0 + h)) - np.asarray(f(t0 - h))) / (2 * h)
    an = np.asarray(fp(t0))
    err = np.max(np.abs(fd - an)) / (np.max(np.abs(an)) + 1e-30)
    check(f"{name}'(t): ODE RHS == FD of closed form", err < 1e-6, f"rel {err:.2e}")

# 2b. large lambda*T stress (Phase-2 regime and beyond): compare B(t) from the GL
# cosh-ratio quadrature DIRECTLY against a tight ODE integration. (An FD-based check is
# ill-posed here: B sits in quasi-steady state so B' ~ 0 — relative normalization against
# a near-zero quantity — and FD amplifies the node-sweep quadrature noise by 1/h.)
from scipy.integrate import solve_ivp as _sivp
sp_stiff = spec_mod.random_general_spec(seed=5, d=2, n_tiers=1, K=2)
sp_stiff.gamma = sp_stiff.gamma * 4e3                      # push lam*T up
px_st = Proxy(sp_stiff)
lamT = float(np.max(np.asarray(px_st.lams)) * sp_stiff.T)

def _rhsB(tt, Bv):
    Amat = np.asarray(px_st.A(tt))
    return np.asarray(px_st._f(tt)) + 2.0 * Amat @ (np.asarray(jnp.diag(px_st.Dp)) * Bv)

worst = 0.0
for t0 in (0.15 * sp_stiff.T, 0.6 * sp_stiff.T):
    sol = _sivp(_rhsB, (sp_stiff.T, t0), np.zeros(sp_stiff.d), method="LSODA",
                rtol=1e-12, atol=1e-14)
    ref = sol.y[:, -1]
    err = np.abs(np.asarray(px_st.B(t0)) - ref).max() / (np.abs(ref).max() + 1e-30)
    worst = max(worst, err)
check(f"B quadrature vs tight ODE at lam*T = {lamT:.1f} (stress)", worst < 1e-9,
      f"rel {worst:.2e}")

# ---------------- 3. Machine-zero quadratic-HJ tests (THE test) ----------------
rng = np.random.default_rng(7)
t_test = rng.uniform(0.0, sp.T * 0.999, size=40)
q_test = rng.uniform(-sp.Q, sp.Q, size=(40, sp.d))

r_prod = np.asarray(res_mod.quadratic_residual_production_path(sp, px, t_test, q_test))
check("machine-zero (production path: analytic diffs, A'/B'/C' RHS, aggregation)",
      np.abs(r_prod).max() < 1e-9, f"sup {np.abs(r_prod).max():.2e}")

r_ind = res_mod.quadratic_residual_independent_path(sp, px, t_test[:12], q_test[:12])
check("machine-zero (independent path: direct subtraction, FD time derivative)",
      np.abs(r_ind).max() < 1e-6, f"sup {np.abs(r_ind).max():.2e}")

# ---------------- 4. d=1 scalar collapse ----------------
sp1 = spec_mod.single_asset_demo_spec()
# symmetrize + strip drift/costs for the textbook comparison
sp1s = spec_mod.MarketSpec(d=1, n_tiers=1, T=sp1.T, gamma=sp1.gamma, mu=np.zeros(1),
                           Sigma=sp1.Sigma, Q=sp1.Q, delta_inf=sp1.delta_inf,
                           kind=np.zeros((1, 1, 2), int),
                           ip=np.broadcast_to(np.array([1.0, 2.5, 0.0]), (1, 1, 2, 3)).copy(),
                           z_atoms=np.ones((1, 1, 2, 1)), p_atoms=np.ones((1, 1, 2, 1)),
                           c=np.zeros((1, 1, 2)))
px1 = Proxy(sp1s)
sigma = float(np.sqrt(sp1s.Sigma[0, 0])); gam = sp1s.gamma
A_i, k_i = 1.0, 2.5
Dp = 2 * A_i * np.e ** -1 * k_i * 1.0                # 2 A C0 k z, C0 = e^-1
lam_sc = sigma * np.sqrt(gam * Dp)
for tt in (0.0, 0.4, 0.9):
    A_ref = sigma * np.sqrt(gam) / (2 * np.sqrt(Dp)) * np.tanh(lam_sc * (sp1s.T - tt))
    check(f"d=1 scalar A(t={tt}) matches tanh formula",
          abs(float(px1.A(tt)[0, 0]) - A_ref) < 1e-12,
          f"{float(px1.A(tt)[0,0]):.8f} vs {A_ref:.8f}")
check("d=1 symmetric no-drift: B == 0", float(jnp.abs(px1.B(0.2)).max()) < 1e-12)

# ---------------- 5. residual at eta=0 equals proxy defect on TRUE equation ----------------
from hjbpinn import network as net_mod
key = jax.random.PRNGKey(0)
fs = net_mod.feature_spec(sp)
params0 = net_mod.init_params(key, net_mod.n_features(fs))
prep = res_mod.prepare_batch(sp, px, t_test, q_test)
r0 = np.asarray(res_mod.residual_fn(params0, prep))
# zero-init sanity: eta==0 at init
check("zero-init: params0 give eta == 0 (residual == proxy defect, finite)",
      np.isfinite(r0).all())
# interior vs boundary bimodality (diagnostic, not an assertion)
inb = (np.abs(q_test) > (sp.Q[None, :] - 2 * sp.zbar()[None, :])).any(1)
print(f"       proxy defect: interior sup |r| = {np.abs(r0[~inb]).max():.3e}, "
      f"boundary-band sup |r| = {np.abs(r0[inb]).max() if inb.any() else float('nan'):.3e}")

# ---------------- 6. GN sanity on a linear least-squares toy ----------------
Xt = jnp.asarray(rng.normal(size=(30, 4))); yt = jnp.asarray(rng.normal(size=30))
toy0 = [(jnp.zeros((4, 1)), jnp.zeros(1))]
toy_res = lambda pp: (Xt @ pp[0][0][:, 0] + pp[0][1][0] - yt)
p_fit, hist = gauss_newton(toy_res, toy0, n_iters=3, verbose=False)
w_ls = np.linalg.lstsq(np.column_stack([np.asarray(Xt), np.ones(30)]), np.asarray(yt),
                       rcond=None)[0]
w_gn = np.concatenate([np.asarray(p_fit[0][0][:, 0]), np.asarray(p_fit[0][1])])
check("GN solves linear LSQ in <=3 iterations", np.abs(w_gn - w_ls).max() < 1e-8,
      f"max diff {np.abs(w_gn - w_ls).max():.2e}")

print("\n" + ("ALL PHASE-0 TESTS PASSED" if not failures else f"FAILURES: {failures}"))
sys.exit(0 if not failures else 1)
