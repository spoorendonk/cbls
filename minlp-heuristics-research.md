# MINLP Heuristics Research — Conversation Summary

**Date:** 2026-03-11
**Context:** mip-heuristics codebase (LP-free MIP heuristics: FJ, Local-MIP, FPR, Diving, Scylla, Adaptive Portfolio)

---

## Question 1: Do These Heuristics Make Sense for Highly Nonlinear Cases?

### What's Implemented

The codebase implements five MIP heuristics — **Feasibility Jump**, **Local-MIP**, **Fix-Propagate-Repair (FPR)**, **Diving**, and **Scylla** — orchestrated by a **Thompson Sampling adaptive portfolio**. All assume strictly linear constraints and a linear objective.

### Why They Break on Nonlinear Problems

| Heuristic | Linear Assumption | What Breaks on Nonlinear |
|-----------|-------------------|--------------------------|
| **Feasibility Jump** | Piecewise-linear score function over variable moves | Violation delta is no longer linear in the move — the "jump" to minimum violation has no closed-form |
| **Local-MIP** | Tight moves solve `delta = (rhs - activity) / a[x]` | Nonlinear constraints make "tightening" a root-finding / NLP subproblem |
| **FPR** | Constraint propagation via linear activity bounds | Implied bounds from `min/max activity` require interval arithmetic over nonlinear expressions |
| **Diving** | LP re-solves at each depth | Needs NLP re-solves; loses convexity guarantees that make LP diving robust |
| **Scylla** | PDLP for approximate LP relaxation | PDLP is a first-order LP method — does not handle nonlinear relaxations |

**Short answer: No, these heuristics do not directly make sense for highly nonlinear cases.** They would need substantial generalization.

### What MINLP Solvers Actually Do

#### Feasibility Pump for MINLP
- **Convex MINLP**: Bonmin implements FP by alternating between NLP solves (via IPOPT) and MIP rounding.
- **Nonconvex MINLP**: Belotti & Berthold (2017) implemented FP in Couenne with: hierarchy of rounding procedures, generalized distance function, linearization cuts for nonconvex constraints.
- D'Ambrosio et al. proposed a "storm" of feasibility pump variants for nonconvex MINLP using successive projection methods with norm constraints.

#### RINS / LNS for MINLP
- Berthold's thesis (ZIB) provides a generic recipe for generalizing LNS heuristics from MIP to MINLP: fix a subset of variables (using LP/NLP relaxation agreement), solve the residual sub-MINLP. Implemented in SCIP.
- The **undercover heuristic** in SCIP fixes enough variables to make the sub-problem *linear* — a workaround for the nonlinear cost.
- Knitro implements LNS by fixing a large fraction of integer variables and solving the residual MINLP subproblem via its NLP solver.

#### Diving for MINLP
- SCIP's diving heuristics extended to MINLP via spatial branch-and-bound. Instead of LP re-solves, they use NLP re-solves with polyhedral outer approximations.

#### What Would It Take to Extend This Codebase to MINLP?
1. **Nonlinear constraint representation** — expression trees or factorable programming
2. **Gradient/subgradient evaluation** — for move scoring in FJ and tight-move computation in Local-MIP
3. **NLP subproblem solver** — replace LP re-solves (HiGHS) with NLP solves (IPOPT, Knitro)
4. **Interval arithmetic** — for constraint propagation in FPR over nonlinear expressions
5. **Linearization cuts** — outer approximation at each iterate for nonconvex constraints
6. **Convergence safeguards** — norm constraints or trust regions to prevent divergence

#### Solvers with MINLP Heuristic Suites

| Solver | Type | MINLP Heuristics |
|--------|------|-------------------|
| **SCIP 10** | Open-source | FP, RINS, RENS, Undercover, Shift-and-Propagate, diving, LNS |
| **BARON 2025** | Commercial | Proprietary primal heuristics |
| **Bonmin** | Open-source (convex) | FP, NLP-BB, OA, hybrid |
| **Couenne** | Open-source (global) | Nonconvex FP, spatial B&B |
| **Knitro** | Commercial | FP, LNS, NLP-based B&B |
| **Juniper** | Open-source (Julia) | NLP-based B&B heuristic |

---

## Question 2: Are There LPs Too Big to Solve Where LP-Free Heuristics Make Sense?

**Yes, concretely.** The PDLP paper (NeurIPS 2021 / MPC 2025) introduced a benchmark of 11 large-scale LPs with 125M to 6.3 billion nonzeros:

- Gurobi **barrier** solved only 3/11 — exceeded 1TB RAM limit on the other 8
- Simplex solved only 3/11 within 6 days (memory-efficient but too slow)
- PDLP solved 8/11 to 1% gap within 6 days

For the 3 instances PDLP also can't solve, LP-free methods are all you have.

That said, **when the LP relaxation is solvable, LP-based heuristics dominate.** Even a rough LP solution (from PDLP) gives FPR the same quality as a full interior-point solve. LP-free methods occupy a niche — very large instances, time-critical settings (pre-root), or zero solver dependencies.

---

## Question 3: The MINLP Opportunity — NLP Relaxation Is Expensive

**This is the strongest case for NLP-free heuristics.**

- For MIP, the LP relaxation is a **polynomial-time** subproblem. Solving it is routine.
- For MINLP, the NLP relaxation is itself **NP-hard** (nonconvex NLP). Even for convex MINLP, each NLP solve involves iterative methods (IPOPT) that are orders of magnitude slower than simplex.

Berthold's thesis documents this: RINS for MINLP requires solving an NLP at the root + an NLP subproblem for the neighborhood search. For large nonconvex problems, this is often **prohibitively expensive**.

**The ratio of "relaxation cost" to "heuristic cost" is vastly worse for MINLP than MIP.** An NLP-free heuristic has a much larger potential payoff.

Concrete opportunity: a Feasibility-Jump-like weighted-violation local search that evaluates nonlinear constraints directly (via function evaluation, not via solving a relaxation) could find feasible solutions to nonconvex MINLPs **before** the root NLP even finishes.

---

## Question 4: Complementing a CBLS Approach

ViolationLS (Davies, Didier & Perron, CPAIOR 2024) demonstrated that Feasibility Jump generalizes well to CBLS — they built a CBLS solver for MiniZinc models on top of FJ's weighted-violation framework, significantly outperforming dedicated CBLS solvers (OscaR, Yuck). Integrating it into CP-SAT's portfolio improved performance on >50% of instances.

Key insight: CBLS works on an **abstract constraint model** with violation functions — it doesn't care whether the underlying constraint is linear, quadratic, or a global constraint. The violation function is a black box returning a scalar.

Layered architecture for MINLP:

```
Symbolic model (AMPL/Pyomo/MiniZinc)
    │
    ├─ NLP-based heuristics (RINS, FP)  ← expensive, high quality
    │
    └─ CBLS / FJ-style local search     ← cheap, fast feasible solutions
         │
         ├─ violation = f(x) - rhs      (nonlinear constraint eval)
         ├─ gradient = ∇f(x)            (for move scoring)
         └─ no matrix, no factorization
```

Thompson Sampling (already in this codebase) orchestrates the portfolio.

---

## Question 5: Working on the Symbolic/Abstract Model (Matrix-Free)

The most novel direction. Current MINLP solvers build explicit constraint representations. A local search heuristic fundamentally needs only:

1. **Constraint evaluation**: given x, compute violation
2. **Move scoring**: given candidate Δx, estimate improvement

Neither requires a constraint matrix. From an algebraic modeling language (AMPL, Pyomo, JuMP), you have:
- A **symbolic expression graph** for each constraint
- **Automatic differentiation** for gradients (move scoring)
- **Incremental evaluation** — only re-evaluate constraints involving the changed variable (the column view in this codebase already does this for linear constraints)

This means you could build an FJ/Local-MIP-style heuristic that:
- Takes a symbolic model + data as input
- Never builds the full constraint matrix or Jacobian
- Evaluates violations and gradients **incrementally** via AD
- Works on instances too large to instantiate as explicit NLP problems
- Scales to millions of constraints if the expression graph is sparse

This is essentially what CBLS systems (COMET/OscaR) do for combinatorial problems. Matrix-free NLP solvers (Newton-Krylov methods) show the same approach works for continuous nonlinear using only Jacobian-vector products. A heuristic would need even less — just function evaluations and directional derivatives.

---

## Opportunity Summary

| Setting | LP/NLP-based heuristics | LP/NLP-free heuristics | Opportunity |
|---------|------------------------|------------------------|-------------|
| MIP, normal scale | Dominant | Niche (pre-root, fast incumbents) | Small |
| MIP, very large scale | PDLP struggles | Only option | Moderate |
| **Convex MINLP** | NLP solve feasible but slow | **Large advantage in speed** | **Significant** |
| **Nonconvex MINLP** | NLP solve may fail/be local | **Avoids the hardest subproblem entirely** | **Large** |
| **Symbolic/matrix-free** | Impossible (needs explicit matrix) | **Natural fit** | **Novel** |

The strongest case: **nonconvex MINLP on symbolic models** — sidestep the NLP relaxation entirely, work incrementally on the expression graph, deliver feasible solutions fast, complementing exact methods in a portfolio.

---

## Key References

- [PDLP: Large-Scale LP (MPC 2025)](https://arxiv.org/html/2501.07018v1)
- [Low-precision FOMs for MIP heuristics (2025)](https://arxiv.org/html/2503.10344)
- [Scylla: Matrix-free FPR (Mexi et al., 2025)](https://link.springer.com/chapter/10.1007/978-3-031-58405-3_9)
- [GPU-Accelerated Primal Heuristics for MIP (2025)](https://arxiv.org/html/2510.20499)
- [Berthold - Heuristics in global MINLP solvers (ZIB thesis)](https://www.zib.de/userpage//berthold/Berthold2014.pdf)
- [SCIP 8 global optimization of MINLP](https://link.springer.com/article/10.1007/s10898-023-01345-1)
- [ViolationLS: CBLS in CP-SAT (CPAIOR 2024)](https://link.springer.com/chapter/10.1007/978-3-031-60597-0_16)
- [Matrix-free convex optimization modeling (Boyd et al.)](https://link.springer.com/chapter/10.1007/978-3-319-42056-1_7)
- [Matrix-free NLP algorithm (IBM/SIAM)](https://www.researchgate.net/publication/303150814_A_Matrix-Free_Algorithm_for_Equality_Constrained_Optimization_Problems_with_Rank-Deficient_Jacobians)
- [Feasibility Jump (Luteberget & Sartor, MPC 2023)](https://link.springer.com/article/10.1007/s12532-023-00234-8)
- [Knitro MINLP Documentation](https://www.artelys.com/app/docs/knitro/2_userGuide/minlp.html)
- [Juniper: Nonlinear Branch-and-Bound in Julia](https://arxiv.org/pdf/1804.07332)
- [BARON Solver](https://minlp.com/baron-solver)
- [Belotti & Berthold - FP for nonconvex MINLP](https://link.springer.com/article/10.1007/s11590-016-1046-0)
- [D'Ambrosio et al. - Storm of feasibility pumps for nonconvex MINLP](https://link.springer.com/article/10.1007/s10107-012-0608-x)
- [MINLP Solver Technology (Vigerske, 2017)](https://www.gams.com/~svigerske/2017_minlp.pdf)
- [Learning to Optimize for MINLP (2024)](https://arxiv.org/html/2410.11061)
