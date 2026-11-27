# cbls_scip_traj: SCIP's incumbent trajectory on a pinned SCIP

`cbls_scip_traj` solves one problem file with a pinned, unreleased SCIP and logs
every incumbent as `(CLOCK_MONOTONIC seconds, objective)`, so a comparison can
score SCIP by primal integral (`benchmarks/common/primal_integral.py`). It is
#442's SCIP backend, chosen on #444 (option A): SCIP-Conc as GAMS's MIPfeas post
describes it, i.e. SCIP's concurrent mode on scipopt/scip `master` (11.0.0-dev),
every setting at its default.

PySCIPOpt cannot drive SCIP 11, so the PySCIPOpt wheel stays the modeller: it
writes each model as CIP (with SOS2 and `sin` intact) and a start as a `.sol`
file, and this binary solves them.

## Pin

- **SCIP:** the commit in `SCIP_COMMIT` (master, 2026-11-26), reported as
  `SCIP 11.0.0 [GitHash: 124040dbc]`. A table built on it labels SCIP an
  unreleased SCIP 11 preview at that commit.
- **SoPlex 8.1.0, Ipopt 3.14.19 over MUMPS (coin-or-tools/ThirdParty-Mumps at
  the commit in `build_scip.sh`) with METIS 5.2.1**: scipoptsuite-deploy
  `main` at e4f0441, which moved Linux builds to METIS 5.2.1 (deploy #45).
- **TPI=tny**, shared libraries, the system's LAPACK/BLAS. `build_scip.sh` has
  the full configure lines and how they depart from deploy's.

The binary refuses to run unless libscip's git hash is a prefix of
`SCIP_COMMIT`; CMake reads that file at configure time.

## Build

```sh
xk lock -- nice -n 19 taskset -c 0-3 benchmarks/common/scip_driver/build_scip.sh ~/data/<session>/scip-442
cmake -B build -DCBLS_SCIP_DIR=$HOME/data/<session>/scip-442
xk lock -- taskset -c 0-3 cmake --build build --target cbls_scip_traj -j4
```

About 10 min at `JOBS=4`, 0.5 GB; the script checks each archive's sha256.
`CBLS_SCIP_DIR` is empty by default, so the gate builds no driver and needs no
SCIP. CMake caches `SCIP_DIR` on first configure: a build directory links the
first prefix it was given, so a new prefix needs a fresh build directory.

## Use

```sh
cbls_scip_traj PROBLEM --threads N --time-limit S --traj-out TRAJ.jsonl \
    --result-out RESULT.json [--set PARAMS.set] [--partial-sol START.sol] [--sol-out BEST.sol]
```

- `--threads N`: `parallel/maxnthreads` N and `SCIPsolveConcurrent` at N > 1
  (opportunistic, master's 10-setting portfolio); `SCIPsolve` at N = 1.
- `--time-limit S`: wall seconds from the driver's start, completion included,
  set as `limits/time` to what is left before the solve. At N > 1 each
  concurrent solver gets that value and counts it from its own start, after the
  main's transform and the N copies, so a run can overrun by that setup time:
  a caller scoring to a deadline clips the records there.
- `--partial-sol`: a `.sol` file; variables it does not list are unknown. At
  N = 1 it goes to the main SCIP as a partial solution. At N > 1 a concurrent
  solver gets only a complete solution (`concsolver_scip.c:636`), so an
  auxiliary single-thread SCIP completes it first with SCIP's `completesol`
  (`presolving/maxrounds` 0, every other heuristic off) and completesol's best
  solution goes to the main by variable name. If
  completesol finds none (it skips a start past
  `heuristics/completesol/maxunknownrate`), nothing goes to the main and
  `start_completed` is false.
- `TRAJ.jsonl`: one `{"mono_s", "solver", "objective"}` per incumbent of the
  main SCIP (`solver` 0) or a concurrent solver (1..N), in `mono_s` order. A
  concurrent solver is a copy made from the main while it sits at `PRESOLVED`,
  not one at depth 0: `heur_dualval` builds a depth-0 sub-SCIP too. Once the
  solvers exist the main logs no more, since it only re-adds their solutions.
  `mono_s` is absolute `CLOCK_MONOTONIC`, Python's `time.monotonic()` on Linux,
  so a caller subtracts its own start. Objectives are in the original space
  (`null` if infinite); take the running best in the result's `sense`
  (solvers interleave, and a shared solution is logged once per solver).
- `RESULT.json`: `ok`, `scip_retcode`, `status` (PySCIPOpt's names), `sense`,
  `primal_bound`, `dual_bound` (`null` while infinite), `n_sols`, `records`,
  `problem`, `threads`, `time_limit_s`, `wall_s`, `solving_time_s`,
  `start_mono_s`, `start_completed` (`null` without `--partial-sol` or at
  N = 1, where SCIP's own completesol takes the partial solution),
  `completion_s` (N > 1 with a start), `scip_version`, `scip_git_hash`,
  `freed`. It is written (by rename) before `SCIPfree`, `freed: false`, and
  again after it, so a driver killed in a long free still leaves its result.
- `BEST.sol`: the best solution with zeros printed, read by variable name.
