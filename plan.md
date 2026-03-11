# CBLS Engine — Implementation Plan

## Context

Build an open-source **Constraint-Based Local Search engine** with a Hexaly-style modeling API. UB-oriented (no lower bounds), targeting nonlinear, nonconvex, mixed discrete+continuous problems that LP-based MIP solvers and CP-SAT can't handle. Enriched with LP/NLP-free ideas from the mip-heuristics codebase (FJ, Local-MIP).

**What this is:** Open-source Hexaly (SA on expression DAG) + MIP heuristic enrichments (FJ-NL warm-start, Newton/gradient moves, structured inner solver hook).

**What this is NOT:** An exact solver, a CP propagation engine, a competitor to Gurobi/CP-SAT on their home turf.

See full plan details in the conversation that generated this file.
