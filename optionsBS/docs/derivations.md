# spotmm — Derivation Note (M0)

Single source of mathematical truth for the one-factor build. Any
discrepancy between code and this note is a bug in the code.

Companion to the `optmm` (Heston/vega) build; reference: Baldacci,
Bergault, Gueant, *Algorithmic market making for options*
(arXiv:1907.12433), whose section-4 example this repo solves **without the
two simplifying assumptions** (per-line box limits instead of the
aggregate-Greek band; Greeks live in S instead of frozen).

## 1. Model and the modeling premise

Underlying under P:  dS = mu S dt + sigma(t,S) S dW,  with
- Black-Scholes: sigma(t,S) = sigma0 (constant);
- Local vol (CEV parametric, calibration deferred):
  sigma(t,S) = sigma0 (S/S0)^(beta_cev - 1).
Zero rates; pricing measure = P with mu = 0 in the pricer.

**The risk channel (deliberate premise).** Under a one-factor complete
model, continuous frictionless Delta-hedging leaves *no* residual risk:
the market-making problem degenerates to static margin optimization. The
nondegenerate desk problem models the *residual delta*: the desk hedges a
fraction (1 - eta) of the book's aggregate option delta, eta in [0,1]
("eta = 1: unhedged; eta = 0: fully hedged"). This replaces optmm's vega
channel and is stated as the premise, not a footnote.

## 2. Hedge layer (paper appendix A.1, transposed)

Free spot position q^S, costless, unconstrained; write e := Delta^pi + q^S
with Delta^pi(t,S,q) = sum_i q_i Delta_i(t,S). The MtM dynamics contribute
    mu S e dt + sigma(t,S) S e dW.
Pointwise mean-variance optimization over e (A.1's argument verbatim):
    e* = mu / (gamma sigma^2 S)  — a pure Merton position, book-independent.
At mu = 0: e* = 0, full hedging, zero residual variance — the completeness
collapse. **What transposes from A.1 is the pointwise-decoupling
structure, not the optimum.** The restricted family
    e = eta * Delta^pi,   eta in [0,1]  (no intercept; mandate),
substitutes exactly (e is a function of (t,S,q); no new state, no control
coupling), leaving the quoting HJB unchanged except:
- penalty coefficient  pen(t,S) = (gamma/2) eta^2 sigma(t,S)^2 S^2
  multiplying (Delta^pi)^2   — **state-dependent** under LV and live-BS;
- inventory carry  mu eta S * Delta^pi  at mu != 0 (directional lean; the
  one-factor analog of the vol-risk-premium carry).
The Merton term is exhibited (a free intercept e0 would optimize to
mu/(gamma sigma^2 S), undoing the restriction) and zeroed by mandate.
Honesty note: unlike Heston (where the family contained the unrestricted
optimum c = rho), here the optimum is the boundary eta = 0; eta > 0 is a
genuine restriction standing in for unmodeled hedge frictions. Caps/costs
that would *derive* eta break the pointwise decoupling and stay deferred.
Spot-leg P&L increment: hedged part -(1-eta) sigma S Delta^pi dW; net
martingale exposure eta sigma(t,S) S Delta^pi dW  (the simulator's
parabola: Var scales as eta^2). The hedge position q^S = -(1-eta) Delta^pi
is materialized and reported by the simulator (notional, turnover).

## 3. The HJB on (t, S, n)

Instruments i = 1..N, Dirac trade size z_i contracts, inventory n in Z^N
(q = n z), per-line box admissibility Q = {|n_i| <= nbar_i}. Live Greeks
Delta_i(S) (frozen in t over the short horizon: tau_i >= 0.25 yr vs
T = 0.006 yr, so time decay of Delta over the episode is negligible; the
binding BBG assumption — the S-freeze — is the one dropped). Client fill
curves logistic with per-instrument slope k_i = beta / V_i^ref: client
behavior is calibrated in *implied-vol* space (V^ref = anchor BS vega
scale; for combos with near-zero anchor vega, V^ref = max |vega| over the
S band). The client-behavior Greek (vega) and the risk Greek (delta) are
deliberately distinct objects.

Value v(t, S, n), terminal v(T) = 0:
    0 = dv/dt + mu S dv/dS + (1/2) sigma(t,S)^2 S^2 d2v/dS2
        + mu eta S Delta^pi(S,n) - pen(t,S) (Delta^pi(S,n))^2
        + sum_{i,+-} z_i 1{n' in Q} H_i([v(n) - v(n')]/z_i),
n' = n - psi e_i, psi(ask) = +1, psi(bid) = -1,
Delta^pi(S,n) = sum_i n_i z_i Delta_i(S). Headline runs: eta = 1, mu = 0.
This state is **lattice-solvable**: the full unsimplified problem has an
exact reference at small N (the structural gift of one factor), and the
Dynkin closure is the scale-out instrument rather than the only one.

## 4. Exact anchors and degeneracies

- **eta = 0, no limits** (the analytic identity replacing Remark 6): the
  HJB decouples; v(t) = (T - t) * sum_{i,+-} z_i H_i(0), n- and
  S-independent. With limits ON this is *false* (masked channels at faces
  propagate inward — the box-pollution mechanism measured in optmm M4):
  the with-limits variant is a behavioral test (v <= analytic, gap at
  faces), the no-limits variant is machine-precision (residual-level, and
  full-solve on a periodic mini-lattice where wraparound is exact for the
  constant-in-n truth).
- **sigma == const under LV** reproduces BS end-to-end (pricer to ~CN
  accuracy; lattice/PINN exactly, same coefficients).
- **Frozen-anchor reduction** (test fixture): freeze Delta_i and sigma S
  at (0, S0) and impose a slab |Delta^pi| <= Dbar: state collapses to
  (t, x = Delta^pi); 1-D explicit-Euler solver `reduced1d`; commensurable
  z-override gives atom-exact landings (the optmm M4 pattern). Used for
  the frozen-regression control: the S-featured PINN on the frozen problem
  must match reduced1d and show emergent S-independence.

## 5. Book (BBG section 4, transposed)

S0 = 10, sigma0 = 0.15 (= sqrt(nu0)); mu = 0. Universe: calls
(K=9, tau=1), (K=10, tau=0.25) [short-dated ATM: gamma makes live Greeks
bite], (K=11, tau=1), plus a tight **butterfly** 9.5/10/10.5, tau=0.25 —
the delta-sign-flip instrument (long-delta below the body, short above,
flip inside the trading band; the one-factor twin of optmm's vega-flip
spread). z_i = 5e5/|O_i(0)|; lambda_i = 252*30/(1 + 0.7 |S0 - K_i|)
(butterfly at its body K=10); alpha = 0.7, beta = 150, k_i = beta/V_i^ref.
T = 0.006 yr (~1.5 days: S traverses the flip). Box nbar = (2,2,2,4).
gamma chosen for risk-reward balance at this scale (recorded at
calibration in VALIDATION.md). delta_inf = -5.

## 6. Conventions

float64 at import; backward time via RK4/lax.scan on dv/dtau = RHS;
S-grid central differences with Neumann ghost second derivatives at the
band edges (band S0 * (1 +- 0.08), 41 nodes; boundary-insensitivity
checked); channels ordered ask-block then bid-block; quotes are
per-contract; PINN: hard terminal condition, inputs (t, S, n) normalized
by (T, S-band, nbar) — public constraint/book data only, no risk features.

## 7. Test map

1. Hamiltonian suite (ported verbatim, 6 tests).
2. BS: parity, AD-vs-closed-form delta, butterfly flip sign; LV: CN
   sigma==const -> BS to grid accuracy; MC cross-check; flip under CEV.
3. Lattice: RHS-on-analytic (eta = 0, no limits) machine precision;
   periodic mini-lattice full-solve exactness; box behavioral (v <=
   analytic, face-localized gap); RK4 step-halving; S-band insensitivity.
4. reduced1d: atom-exact self-consistency; frozen-fixture cross-checks.
5. Sim: value identity vs the exact (t,S,n) lattice; eta-parabola
   Var(eta)/Var(1) = eta^2 under CRN with fixed policy; q^S reporting;
   mu != 0 directional-lean sign test.
6. PINN: terminal/sampler/residual-finite; trained-vs-exact-lattice on all
   admissible states; butterfly-skew reversal (exact on lattice, matched
   by PINN); frozen-regression control; Dynkin closure.

## 8. Deferral list

Dupire calibration of sigma(t,S) (parametric CEV given) · hedge
frictions/caps (breaks A.1 decoupling; would endogenize eta) · distributed
trade sizes · client tiering · multi-underlying · stochastic-vol overlay
(that is optmm) · lifecycle events.
