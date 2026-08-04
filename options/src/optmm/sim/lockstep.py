"""CRN lockstep simulator on the commensurable Vpi atom chain.

State per path: atom index a (Vpi = atom_grid[a]), instantaneous variance nu
(CIR full-truncation Euler under P).  Fills per channel are Bernoulli with
probability Lambda_ch(delta) * dt, quotes gathered from a precomputed
time-dependent policy table (NaN = inadmissible => fill prob 0).

Accumulated per path:
- reward: sum_fills z_ch * delta + [carry(nu) * Vpi - pen * Vpi^2] * dt,
  the risk-adjusted running objective, whose expectation under the optimal
  policy equals v(0, nu_0, Vpi_0) — the value-identity test.
- M(c) for each hedge tilt c in c_list: the martingale P&L leg
      dM = (xi/2) * Vpi * (dW_nu - c dW_S),   dW_S = rho dW_nu + sqrt(1-rho^2) dW_perp,
  under common random numbers across c, so Var[M(c)] / Var[M(0)] = m(c)
  exactly in expectation — the hedge-parabola test.

Accounting conventions: dt-integrand terms use the pre-trade state; fills are
applied after.  Multiple channels may fill in one step (all applied); the
joint-admissibility edge case at the slab boundary is O((Lambda dt)^2) and
negligible at the default dt.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..core.risk import MarketParams, penalty_coef
from ..solvers.reduced2d import Reduced2DSolution, _bc


@dataclass
class SimResult:
    reward: jnp.ndarray          # (n_paths,)
    M: jnp.ndarray               # (n_c, n_paths) hedge-leg martingales
    Vpi_final: jnp.ndarray       # (n_paths,)
    fills: jnp.ndarray           # (n_paths,) total fill count
    corr: jnp.ndarray | None = None   # (n_paths,) int R~ dt along the path


def policy_tables(sol: Reduced2DSolution, delta_inf=-5.0):
    """Quotes for every stored time slice: (n_t, n_ch, n_nu, n_V)."""
    assert sol.v_hist is not None, "solve with store_stride=1"
    return jax.vmap(lambda v: sol.quotes(v, delta_inf))(sol.v_hist)


def simulate_atoms(sol: Reduced2DSolution, market: MarketParams, gamma: float,
                   T: float, c: float = 0.0, c_list=(0.0,),
                   n_paths: int = 20000, seed: int = 0,
                   a0: int | None = None, nu0: float | None = None,
                   tabs=None, res_tab=None):
    """Simulate under a (time-dependent) policy.

    tabs: (n_t, n_ch, n_nu, n_V) quote tables; defaults to the optimal policy
    stored in `sol`.  res_tab: optional (n_t-1, n_nu, n_V) linear-residual
    table integrated along the path into `corr` (the Dynkin corrector term).
    The number of sim steps equals the table length minus one, so policy
    lookup is exact in time.
    """
    if tabs is None:
        tabs = policy_tables(sol)                  # (n_t, n_ch, n_nu, n_V)
    n_t = tabs.shape[0]
    nt = n_t - 1
    dt = T / nt
    ch = sol.channels
    atom = float(sol.V_grid[1] - sol.V_grid[0])
    steps = np.round(np.asarray(ch.w) / atom).astype(int)
    assert np.allclose(steps * atom, np.asarray(ch.w), rtol=1e-9), \
        "book not commensurable with the atom grid"
    dstep = jnp.asarray(-np.asarray(ch.psi) * steps, dtype=jnp.int32)  # per ch
    n_ch = int(ch.z.shape[0])
    nV = int(sol.V_grid.shape[0])
    nu_lo = float(sol.nu_grid[0])
    dnu = float(sol.nu_grid[1] - sol.nu_grid[0])
    n_nu = int(sol.nu_grid.shape[0])
    pen = penalty_coef(gamma, market.xi, market.rho, c)
    rho = market.rho
    sq = np.sqrt(1.0 - rho * rho)
    cs = jnp.asarray(c_list)                        # (n_c,)
    V_grid = sol.V_grid

    a0 = nV // 2 if a0 is None else a0
    nu0 = market.nu0 if nu0 is None else nu0

    inten = ch.intensity                            # params shape (n_ch,)

    use_corr = res_tab is not None
    res_tab_j = res_tab if use_corr else jnp.zeros((1, n_nu, nV))

    def step(carry, inp):
        a, nu, reward, M, fills, corr, key = carry
        k_idx = inp
        key, k1, k2, k3 = jax.random.split(key, 4)
        # policy lookup: interp over nu at (t=k, channel, atom=a)
        f = jnp.clip((nu - nu_lo) / dnu, 0.0, n_nu - 1.001)
        i = jnp.floor(f).astype(jnp.int32)
        fr = f - i
        tab = tabs[k_idx]                           # (n_ch, n_nu, n_V)
        d_lo = tab[:, i, a]                         # (n_ch, n_paths)
        d_hi = tab[:, i + 1, a]
        delta = (1 - fr)[None, :] * d_lo + fr[None, :] * d_hi
        ok = jnp.isfinite(d_lo) & jnp.isfinite(d_hi)
        lam = jnp.where(ok, inten(jnp.where(ok, delta, 0.0).T).T, 0.0)
        u = jax.random.uniform(k1, (n_ch, a.shape[0]))
        fill = (u < lam * dt) & ok
        # dt-integrand at pre-trade state
        Vpi = V_grid[a]
        carry_c = market.carry_coef(nu)
        reward = reward + (carry_c * Vpi - pen * Vpi**2) * dt
        reward = reward + jnp.sum(jnp.where(fill, ch.z[:, None] * delta, 0.0),
                                  axis=0)
        # hedge-leg martingales, CRN across c
        zn = jax.random.normal(k2, (a.shape[0],))
        zp = jax.random.normal(k3, (a.shape[0],))
        dWn = jnp.sqrt(dt) * zn
        dWS = rho * dWn + sq * jnp.sqrt(dt) * zp
        M = M + (market.xi / 2.0) * Vpi[None, :] * (dWn[None, :]
                                                    - cs[:, None] * dWS[None, :])
        # Dynkin corrector integrand at the pre-trade state
        if use_corr:
            rt = res_tab_j[jnp.minimum(k_idx, res_tab_j.shape[0] - 1)]
            r_lo = rt[i, a]
            r_hi = rt[i + 1, a]
            corr = corr + ((1 - fr) * r_lo + fr * r_hi) * dt
        # state updates
        a = (a + jnp.sum(jnp.where(fill, dstep[:, None], 0), axis=0)
             .astype(jnp.int32))
        a = jnp.clip(a, 0, nV - 1)
        nup = jnp.maximum(nu, 0.0)
        nu = nu + market.a_P(nup) * dt + market.xi * jnp.sqrt(nup * dt) * zn
        fills = fills + jnp.sum(fill, axis=0).astype(jnp.int32)
        return (a, nu, reward, M, fills, corr, key), None

    a_init = jnp.full((n_paths,), a0, dtype=jnp.int32)
    nu_init = jnp.full((n_paths,), nu0)
    r_init = jnp.zeros((n_paths,))
    M_init = jnp.zeros((cs.shape[0], n_paths))
    f_init = jnp.zeros((n_paths,), dtype=jnp.int32)
    c_init = jnp.zeros((n_paths,))
    key = jax.random.PRNGKey(seed)
    (a, nu, reward, M, fills, corr, _), _ = jax.lax.scan(
        step, (a_init, nu_init, r_init, M_init, f_init, c_init, key),
        jnp.arange(nt))
    return SimResult(reward=reward, M=M, Vpi_final=V_grid[a], fills=fills,
                     corr=corr if use_corr else None)
