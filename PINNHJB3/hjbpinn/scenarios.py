"""Scenario registry: specs + training configurations, importable by drivers,
notebooks, and tests alike (no dependency on the tests/ folder). Widths and
sampling budgets live here so trained checkpoints are reconstructible from the
scenario name alone; train-time metadata is also written into every checkpoint."""
from __future__ import annotations

from dataclasses import dataclass, replace, field

import numpy as np

from . import spec as spec_mod


@dataclass(frozen=True)
class ScenarioCfg:
    name: str
    widths: tuple = (64, 64)
    straddle_nt: int = 21
    iters: tuple = (90, 60, 60, 60, 60)
    rar_pool: int = 6000
    rar_keep: int = 320
    n_uniform: int = 1024
    n_boundary: int = 224
    n_surface: int = 288
    n_paths_cap: int = 160
    seed: int = 42


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


def d5_spec():
    """Five-asset Section-6-like book (see run_h200 docstring for sizing)."""
    d, K = 5, 4
    sig = np.array([1.2, 0.6, 0.9, 1.05, 0.75])
    R = 0.5 * np.ones((d, d)) + 0.5 * np.eye(d)
    kind = np.ones((d, 1, 2), int)
    ip = np.zeros((d, 1, 2, 3)); ip[..., 0] = 30.0; ip[..., 1] = 0.7; ip[..., 2] = 30.0
    z = np.broadcast_to(np.array([6250., 12500., 18750., 25000.]), (d, 1, 2, K)).copy()
    p = np.broadcast_to(np.array([0.534, 0.350, 0.097, 0.019]), (d, 1, 2, K)).copy()
    return spec_mod.MarketSpec(d=d, n_tiers=1, T=7.0, gamma=8e-6,
                               mu=np.array([0.1, -0.1, 0.05, 0.0, -0.05]),
                               Sigma=np.outer(sig, sig) * R, Q=np.full(d, 75_000.0),
                               delta_inf=5.0, kind=kind, ip=ip, z_atoms=z, p_atoms=p,
                               c=np.zeros((d, 1, 2)))


def truncated_spec(base, d, Q=None):
    """First-d-assets truncation of a spec (validator-ladder boxes)."""
    import copy
    sp = copy.deepcopy(base)
    sp.d = d; sp.mu = base.mu[:d]; sp.Sigma = base.Sigma[:d, :d]
    sp.kind = base.kind[:d]; sp.ip = base.ip[:d]; sp.z_atoms = base.z_atoms[:d]
    sp.p_atoms = base.p_atoms[:d]; sp.c = base.c[:d]
    sp.Q = np.full(d, Q) if Q is not None else base.Q[:d].copy()
    return sp


_CFG = {
    "exp":  ScenarioCfg("exp"),
    "logi": ScenarioCfg("logi"),
    "sec6": ScenarioCfg("sec6", widths=(48, 48), straddle_nt=6,
                        rar_pool=9000, rar_keep=480),
    "d5":   ScenarioCfg("d5", widths=(48, 48), straddle_nt=4, iters=(45, 40),
                        rar_pool=9000, rar_keep=480),
}


def get_cfg(name) -> ScenarioCfg:
    return _CFG[name]


def get_spec(name):
    if name == "exp":
        return spec_mod.single_asset_demo_spec()
    if name == "sec6":
        from .bridge import section6_spec
        return section6_spec()
    if name == "d5":
        return d5_spec()
    return logistic_spec()
