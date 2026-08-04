# optmm — Derivation Note (M0)

Single source of mathematical truth. Every solver implements the equations in this
note; any discrepancy between code and this note is a bug in the code.

Reference: Baldacci, Bergault, Guéant, *Algorithmic market making for options*
(arXiv:1907.12433v7), hereafter **BBG**.

## 1. Model

Underlying under ℙ:
    dS = μ S dt + √ν S dW^S,     dν = a^ℙ(t,ν) dt + ξ √ν dW^ν,   d⟨W^S,W^ν⟩ = ρ dt.
Under ℚ (zero rates): dS = √ν S dŴ^S, dν = a^ℚ(t,ν) dt + ξ √ν dŴ^ν.
Heston: a^m(t,ν) = κ^m(θ^m − ν), m ∈ {ℙ,ℚ}. Feller 2κθ > ξ² required under both.

N European options, prices O^i(t,S,ν) solving the ℚ-pricing PDE. Vega
    𝒱^i(t,S,ν) := ∂_{√ν} O^i = 2√ν ∂_ν O^i.
Using the pricing PDE, the ℙ-dynamics of each option price is
    dO^i = ∂_S O^i dS + 𝒱^i (a^ℙ − a^ℚ)/(2√ν) dt + (ξ/2) 𝒱^i dW^ν.        (†)

RFQ fills: marked point processes per instrument and side j ∈ {a,b}, ψ(a)=+1,
ψ(b)=−1. Trade size Dirac at z^i contracts (closure of the inventory lattice
depends on this; distributed sizes are explicitly deferred). Post-trade
inventory q' = q − ψ(j) z^i e_i. Intensity Λ^{i,j}(δ) · 1{q' ∈ Q}, with
δ = quote distance to model mid per contract (bid: O−δ^b, ask: O+δ^a).

Intensity families (both satisfy sup Λ Λ'' / Λ'^2 < 2):
- logistic (BBG §4): Λ(δ) = λ / (1 + e^{α + k δ}), with per-instrument slope
  k_i = β / 𝒱^i (clients decide in IV space: δ of one vol point ⇒ k_i δ = β·0.01).
- exponential: Λ(δ) = A e^{−k δ}.

## 2. Objective and hedge layer (A.1, generalized)

The MM also holds q^S in the underlying (costless, continuous, unconstrained).
Write h := Δ^π + q^S with Δ^π = Σ q^i ∂_S O^i. MtM dynamics (using (†)):
    dV = Σ_{i,j} z δ^{i,j} dN^{i,j} + 𝒱^π (a^ℙ−a^ℚ)/(2√ν) dt
         + (ξ/2) 𝒱^π dW^ν + √ν S h dW^S + μ S h dt,
with 𝒱^π := Σ q^i 𝒱^i (state-dependent in general).

Mean-variance pointwise optimization over h (h enters only pointwise, no state
coupling, no cost):
    max_h  μ S h − (γ/2)[ ν S² h² + ρ ξ 𝒱^π √ν S h ]
    ⇒ h* = μ/(γ ν S) − ρ ξ 𝒱^π / (2 √ν S).
The μ/(γνS) term is a Merton speculative position. **Mandate separation choice:
we set μ = 0 in the hedge layer** (the hedging book hedges; it does not take
spot views). Documented deliberately; the code keeps μ as a generator parameter
for stage 3 state dynamics but the hedge layer derivation assumes μ = 0.

Proportional hedge family, tilt c ∈ ℝ:
    h(c) = −c ξ 𝒱^π / (2 √ν S)   (c = 0: Δ-neutral; c = ρ: A.1 optimum).
Residual martingale variance rate:
    (ξ 𝒱^π/2)² · m(c),   m(c) := 1 − 2ρc + c²,   m(0)=1, m(ρ)=1−ρ².
The quoting problem depends on the hedge layer **only** through m(c) scaling the
vega penalty. Under the optimal hedge the spot-leg P&L increment is
    √ν S h dW^S = −c (ξ/2) 𝒱^π dW^S      (S cancels — used by the simulator).

Running risk-adjusted objective (Cartea–Jaimungal form, matches BBG body via
Itô isometry):
    sup_δ E ∫_0^T [ Σ_{i,j} z^i δ^{i,j} Λ^{i,j}(δ^{i,j}) 1{adm}
                    + 𝒱^π (a^ℙ−a^ℚ)/(2√ν)
                    − (γ ξ² m(c) / 8) (𝒱^π)² ] dt,     value v(T,·) = 0.

## 3. The three stage HJBs

Hamiltonian per channel: H^{i,j}(p) := sup_{δ ≥ δ_∞} Λ^{i,j}(δ)(δ − p);
jump term contribution z^i · 1{q' ∈ Q} · H^{i,j}( [v(x,q) − v(x,q')] / z^i ),
with q' = q − ψ(j) z^i e_i. Units: p and δ in € per contract, v in €.

**Stage 1** (frozen Greeks 𝒱^i := 𝒱^i(0,S₀,ν₀); slab Q = {|𝒱ᵀq| ≤ V̄}), state (t,ν,q):
    0 = ∂_t u + a^ℙ ∂_ν u + ½ ξ² ν ∂²_νν u
        + (𝒱ᵀq)(a^ℙ−a^ℚ)/(2√ν) − (γ ξ² m(c)/8)(𝒱ᵀq)²
        + Σ_{i,j} z^i 1{|𝒱ᵀq − ψ(j) z^i 𝒱^i| ≤ V̄} H^{i,j}([u(q)−u(q−ψ(j)z^i e_i)]/z^i).
Exact reduction u(t,ν,q) = v(t,ν,𝒱ᵀq) with v solving BBG eq. (4) (2D). The
reduction is exact in continuum; the 2D *solver* is not exact (incommensurable
jumps z^i𝒱^i force interpolation), while the q-lattice solver is exact in q.
Remark 6 limit: a^ℙ = a^ℚ ⇒ v ν-independent.

**Stage 2**: identical except Q = box {|q^i| ≤ q̄^i}. Reduction dies (indicator
not a function of 𝒱ᵀq).

**Stage 3** (state-dependent Greeks; S in the state), state (t,S,ν,q):
    0 = ∂_t u + μS ∂_S u + a^ℙ ∂_ν u
        + ½ ν S² ∂²_SS u + ρ ξ ν S ∂²_Sν u + ½ ξ² ν ∂²_νν u
        + (𝒱(t,S,ν)ᵀq)(a^ℙ−a^ℚ)/(2√ν) − (γ ξ² m(c)/8)(𝒱(t,S,ν)ᵀq)²
        + Σ_{i,j} z^i 1{q' ∈ Q} H^{i,j}([u(q)−u(q')]/z^i).
A.1 extension holds verbatim: the pointwise h-optimization never used constant
vega, so m(c) multiplies the state-dependent penalty. If the Greeks are frozen
and coefficients S-independent, any S-independent u reduces stage 3 to stage 1
(consistency test: stage-3 residual on a lifted stage-1 solution ≡ stage-1
residual; ∂_S terms vanish identically under AD).

## 4. Hamiltonian solve

FOC: Λ(δ) + Λ'(δ)(δ − p) = 0  ⇔  g(δ) := δ − p + Λ(δ)/Λ'(δ) = 0.
g'(δ) = 2 − Λ Λ''/Λ'² > 0 by hypothesis ⇒ unique root.
- Exponential: Λ/Λ' = −1/k ⇒ δ* = p + 1/k, H(p) = (A/k) e^{−kp−1}, closed form.
- Logistic Λ = λ/(1+e^{x}), x = α + kδ: with σ(δ) := 1/(1+e^{−x}) = 1 − Λ/λ,
  Λ/Λ' = −1/(k σ), so g(δ) = δ − p − 1/(k σ(δ)), g' = 1/σ, and Newton is
      δ ← δ − g(δ) σ(δ).
  g is increasing and concave (g'' = −k(1−σ)/σ < 0); starting at δ₀ = p gives
  g(δ₀) < 0 and Newton converges monotonically from below (tangent of a concave
  function lies above it ⇒ no overshoot). Envelope: H'(p) = −Λ(δ*(p)).
Constraint: δ = max(δ_∞, δ*); if binding, H = Λ(δ_∞)(δ_∞ − p). δ_∞ chosen deep.
Optimal quotes read off directly: δ^{i,j*}(x) = argmax at p^{i,j}(x) =
[v(x) − v(x')]/z^i (no Λ⁻¹∘H' composition needed).

## 5. BBG §4 book (replication target)

S₀=10, ν₀=0.0225, κ^ℙ=2, θ^ℙ=0.04, κ^ℚ=3, θ^ℚ=0.0225, ξ=0.2, ρ=−0.5.
20 calls, K ∈ {8,9,10,11,12} × T^i ∈ {1,1.5,2,3}. λ^i = 252·30/(1+0.7|S₀−K_i|),
α=0.7, β=150, z^i = 5·10⁵/O^i₀ contracts (≈ €500k notional), V̄ = 10⁷,
T = 0.0012 yr, γ = 10⁻³. Grid 180 × 30 × 40 on [0,T]×[0.0144,0.0324]×[−V̄,V̄],
Neumann in ν. Calibration self-checks: fill prob at mid = 1/(1+e^{0.7}) ≈ 33.2%;
at one vol point through mid: 1/(1+e^{−0.8}) ≈ 69.0%.

## 6. Conventions

- q in contracts; lattice index n ∈ ℤ^N with q = n ⊙ z; per-trade vega
  w_i := z^i 𝒱^i; 𝒱^π = wᵀn. Grid bound of the 2D solver = ±V̄ exactly, so
  off-grid jump targets are precisely the inadmissible ones.
- float64 everywhere (enforced at package import).
- Backward time: reduced2d explicit Euler v(t−dt) = v(t) + dt·RHS; lattice RK4
  on dv/dt = −RHS via lax.scan. Monotonicity/CFL: dt·ΣΛ(δ*) < 1 and
  ½ξ²ν dt/dν² < ½ (checked at runtime).
- ν advection: central differences (cell Péclet ≪ 1 for BBG params); Neumann via
  ghost reflection: ∂²νν at edge = 2(v₁−v₀)/dν².
- PINN: terminal condition hard-enforced, v_θ = (1−t/T)·NN(·). Stage-1 net is a
  pure MLP (no 𝒱ᵀq feature, no constraint-distance feature) — hyperplane
  discovery is the stage-1 success metric and must not be leaked. Stage-2/3 may
  add axis-aligned box-distance features (constraint geometry is public
  knowledge; risk geometry is not). Documented asymmetry.

## 7. Test map (behavioral spine)

1. Hamiltonian: Newton FOC residual < 1e−10 across wide p, k; exponential
   closed form matches generic path; H decreasing convex.
2. Heston COS: BS limit (ξ→0, θ=ν₀) to ~1e−8; put–call parity; AD vega = FD
   vega; MC agreement; monotone in K; cumulants via AD of log φ (no hand-typed
   c₂ formula).
3. reduced2d: grid convergence; a^ℙ=a^ℚ symmetric book ⇒ v even in 𝒱^π and
   δ^b(𝒱^π) = δ^a(−𝒱^π); δ^b nondecreasing / δ^a nonincreasing in 𝒱^π.
4. Remark-6 equivalence: with a^ℙ=a^ℚ, mv-mode (ξ,ρ,c=ρ) ≡ Δ-neutral at
   ξ√(1−ρ²) to machine precision on the same grid; with a^ℙ≠a^ℚ they differ.
5. Lattice vs reduced2d (N=2, slab): quantify 2D interpolation error; lattice
   step-halving.
6. Simulator: value identity E[∫ risk-adj reward] under the lattice-optimal
   policy = v_lattice(0,ν₀,0) within MC error (validates every sign in one
   shot); hedge parabola Var[M(c)]/Var[M(0)] = m(c) under CRN, vertex at c=ρ.
7. Stage-3 residual on lifted stage-1 solution ≡ stage-1 residual.
8. Dynkin corrector: policy value = ṽ(0) − E[∫ linear residual] on problems
   with exact references, before use at stage 3.
9. PINN stage 1: ∇_q v collinearity with 𝒱 (discovery), projected error vs
   reduced2d/lattice.

## 8. Deferral list

Distributed trade sizes (breaks lattice closure) · client tiering · two-factor
vol · hedge frictions/caps (breaks pointwise decoupling — a real extension, not
a toggle) · lifecycle events (interface stub `events()` reserved) ·
multi-underlying · imperfect-Δ transaction costs.
