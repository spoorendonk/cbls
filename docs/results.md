# Benchmark results

The full-benchmark tables of the four sets in [docs/benchmarks/README.md](benchmarks/README.md):
MIPfeas, MINLPLib, UC-CHPED and CVRP. Each table is copied from the issue
comment that scored it, with that comment's run record and caveats. The comment
is linked, so the rest of its record is one click away.

**Read each table at its engine commit.** A search-trajectory change
invalidates every table ([measuring.md](measuring.md)). Since MIPfeas's
`8246efd`, 112 commits have touched `src/` or `include/`. Since `02c78ad`, the
engine of the MINLPLib tables and of #257's UC-CHPED runs, none has, as of
`13a793d`; #442's cbls runs at `2eb27278` have the same `src/` and `include/`.

**The per-row files are not in the repository.** They stay with each run's
results directory on simon-workstation. The sha256 of each file is given
where its source comment states one. Every table is an output, regenerable from
those records ([measuring.md](measuring.md)).

Every comparison solver ran at its own defaults, unless a table says otherwise
(Simon, 2026-11-01,
[#87](https://github.com/spoorendonk/cbls/issues/87#issuecomment-6558356494)).

| Set | Compared against | What the table shows |
|---|---|---|
| [MIPfeas](#mipfeas) | CP-SAT at its defaults | CP-SAT is ahead: 175 feasible against cbls's 140 of 211; primal integral 0.168 against 0.390 |
| [MINLPLib](#minlplib) | SCIP 10 at its defaults; MINLPLib's best known | SCIP is ahead: GM(gap+1) ratio cbls/SCIP 1.68; 169 feasible against 187 of 195. On open instances, 3 clear improvements on MINLPLib's best known |
| [UC-CHPED](#uc-chped) | Pedroso et al. 2014's method ported to SCIP 11; SCIP's exact MINLP | at equal time and threads, cbls has the lowest primal integral on 19 of 20 rows; Pedroso's method ends with the best final gap |
| [CVRP](#cvrp) | - | not run: no table |

## MIPfeas

Source: [#255, comment 6566503731](https://github.com/spoorendonk/cbls/issues/255#issuecomment-6566503731)
(2026-11-02). The runner and the scoring rules are in
[benchmarks/instances/mipfeas/README.md](../benchmarks/instances/mipfeas/README.md).

**Result: CP-SAT at its defaults is ahead of cbls on this roster; parity is not reached.**

| 211 held-out instances, 60 s | cbls | CP-SAT (defaults) |
|---|---|---|
| feasible | 140 | 175 |
| matched the MIPLIB reference | 22 | 87 |
| Primal Integral, shifted geometric mean (lower is better) | **0.390** | **0.168** |
| PI mean / median | 0.948 / 0.750 | 0.605 / 0.172 |
| verification_failed / below_reference | 0 / 0 | 0 / 0 |
| did not search (MODEL_INVALID / killed) | 0 | 9 / 1 |

- **Feasibility parity** is measured on the 201 instances where both engines searched:
  - the two agree on 151 (130 both feasible, 21 neither);
  - **cbls only: 5** (app1-1, app1-2, neos-5195221-niemur, neos-873061, ns1116954);
  - **CP-SAT only: 45**.
- **Defects: 13.** None is a solution defect:
  - 9 CP-SAT MODEL_INVALID, a baseline limitation;
  - 1 CP-SAT kill (s100, below);
  - 3 CP-SAT traces that collapsed to their final objective (cryptanalysiskb128n5obj16, fhnw-binpack4-48, neos-3004026-krka).
- 9 cbls rows are `verification_marginal`, which the scorer does not count as a defect.
- **Model-shape cross-check:** 210 of 211 instances checked; the one difference (neos-933966, one free row) is benign.

### Run record

- **Engine commit:** `8246efdca55776b958d22322eaf7fac2ba19d67f`, frozen for the full stage. **Driver:** `c0cf3f16433a903231499571cedb52274133c5a2`. OR-Tools 9.15.6755, PySCIPOpt 6.3.0, Python 3.13.9; MIPLIB references from the `.solu` of miplib2017 v36.
- **Roster:** `roster.csv` with `--held-out tuned.txt`, which leaves out the 22 tuning instances: 211 of 233.
- **Budget:** 60 s per instance and engine, one job at a time.
- **cbls:** 16 threads, pinned to cores 0-15, seed 1.
- **CP-SAT:** at its defaults (presolve on, full portfolio, unpinned, `workers_started` 32).
- **Machine:** simon-workstation, 32 hardware threads, 121.5 GiB.
- **Run units:** `cbls-heimdall-255-campaign` and `-campaign2`, with the failed rows redone in `cbls-heimdall-night256`. A cgroup `MemoryMax`, not an address-space cap, was raised from 48G (instances 1-36) to 64G (37-156) and 110G (157 on, and the redo).
- **Redo of the failed rows** at 110G:
  - **eilA101-2, CP-SAT:** feasible, obj 1371.943, verified. The solve took 191.6 s wall at 8.56 cores, 131.6 s past the 60 s budget.
  - **s100, CP-SAT:** killed by the driver at its 960 s wall-clock limit, in the first run and the redo. It is scored as no-solution (PI 2.0) and left out of the parity sets.
- **Model difference:** CP-SAT solves its presolved model; cbls solves the file's rows, with activity-based bound tightening and infinite bounds clamped at 1e7 ([#255, comment 6558313900](https://github.com/spoorendonk/cbls/issues/255#issuecomment-6558313900)).
- **Scored by:**

  ```
  primal_integral.py --results-dir <campaign> --roster benchmarks/instances/mipfeas/roster.csv \
      --held-out benchmarks/instances/mipfeas/tuned.txt --budget 60 --out <campaign>/comparison.csv
  ```

- **Outputs**, in `cbls-heimdall/mipfeas255/campaign/` on simon-workstation:
  - `comparison.csv`, 211 x 2 rows, sha256 `f2e75b74e8878175bc87f242d6a25c7739d4f142e25890319236f6e10a760193`;
  - `comparison_report.md`, sha256 `517c774d707ec5ea5985ef2b1dcd07368390a6f4bb15e8a0c10fc09c3cc1672a`. The source comment carries it in full.

## MINLPLib

Two tables, both at engine `02c78ad`, 60 s, seed 1, 16 threads on cores 0-15.
The runner and its rules are in
[benchmarks/instances/minlplib/README.md](../benchmarks/instances/minlplib/README.md).

### cbls against SCIP, Mittelmann's nontrivial roster

Source: [#256, comment 7181284914](https://github.com/spoorendonk/cbls/issues/256#issuecomment-7181284914)
(2026-11-26).

| measure | value |
|---|---|
| roster (bounds.csv) | 196 |
| counted (roster less claim-excluded) | 195 |
| scored | 195 |
| cbls feasible | 169 |
| SCIP feasible | 187 |
| both feasible | 165 |
| GM pool (both, finite BKS) | 165 |
| GM(gap+1) cbls | 3.6021 |
| GM(gap+1) SCIP | 2.1490 |
| ratio cbls/SCIP | **1.6762** |
| cbls wins (only-cbls) | 27 (4) |
| cbls losses (only-SCIP) | 62 (22) |
| ties (within 1%) | 76 |

- rule: elec25, elec50 (claim-excluded) out of every count and aggregate. GM = geometric mean of (max(gap_to_bks%, 0) + 1) over instances both solvers solved, with a finite BKS; ratio = GM_cbls / GM_SCIP (below 1: cbls ahead). Win/loss: gap+1 more than 1% smaller than the other's, same pool; brackets: instances only that solver solved (no BKS needed). Solved = a feasible row (cbls: and not VERIFY-FAILED); anything else is that solver's no-solution, listed with its reason, on both sides alike.
- claim-excluded (out of every count): 1 (elec50)
- no finite BKS (out of GM and both-solved win/loss): 0
- cbls no solution: 26
  - infeasible: 26 (camshape100, carton7, carton9, crudeoil_lee1_07, crudeoil_pooling_ct2, eigena2, feedtray, fin2bb, gabriel02, gastrans582_cold13_95, gastrans582_mild11, gilbert, hvb11, meanvar-orl400_05_e_7, multiplants_mtg2, nd_netgen-3000-1-1-b-b-ns_7, nous2, oil2, optmass, powerflow0014r, product2, product, topopt-mbb_60x40_50, wager, waste, waterno2_03)
- SCIP no solution: 8
  - no-solution(timelimit): 8 (eigena2, feedtray, johnall, meanvar-orl400_05_e_7, meanvar-orl400_05_e_8, radar-2000-10-a-6_lat_7, radar-3000-10-a-8_lat_7, topopt-mbb_60x40_50)

**Reading:** on this set cbls is behind SCIP. Its GM(gap+1) is 1.68x SCIP's,
and it finds a solution on 169 instances against SCIP's 187. Only cbls found
one on 4: johnall, meanvar-orl400_05_e_8, radar-2000-10-a-6_lat_7,
radar-3000-10-a-8_lat_7. Only SCIP found one on 22. There is no primal
integral: the SCIP shards record no trace.

Run record:
- **Roster:** Mittelmann's `nontrivial.txt` (200 names, sha256 `4a754551…97e8`, fetched 2026-11-26), matched against the pinned 2026-11-01 MINLPLib archive: 196 modelled (`bounds.csv` sha256 `2a06e488…fdda`). `cardqp_inlp` and `netmod_kar2` are gone from MINLPLib. `ball_mk3_10` and `powerflow0057r` have no known primal bound, so no gap to score; they are in the next table.
- **cbls:** the engine's defaults at `02c78ad` (`float_hook=on;lns=on;lns_interval=3;compound_moves=on;novelty_prob=0.5;unproductive_iters=300;perturbation_period=100`). Nothing was tuned on this roster. Run 2026-11-25 23:40Z to 2026-11-26 02:58Z on simon-workstation, in run unit `cbls-gefjon-256-chain`. Every one of the 196 calls exited 0, and SCIP checked all 196 rows, demoting none. Memory cap: a 110G cgroup.
- **SCIP:** reused from the 2026-11-01 run on the same host, not re-run: SCIP 10.0.2 / PySCIPOpt 6.3.0 at its defaults, seed 0, 60 s, single-threaded, 16 solves at once, one per physical core. SCIP's default clock is wall time: `timing/clocktype` defaults to 2, measured ([benchmarks/instances/minlplib/README.md](../benchmarks/instances/minlplib/README.md)). The source comment calls it CPU time; that is wrong.
- **Files:** `~/data/cbls-gefjon/256-scip8/` on simon-workstation (`table.md`; `summary.json`, sha256 `4cbcb37e…2f23`; cbls staging, SCIP shards, `.nl` files, run scripts and logs).

### cbls against MINLPLib's best known, open and no-known instances

Source: [#440, comments 7222891355 and 7222893856](https://github.com/spoorendonk/cbls/issues/440#issuecomment-7222891355)
(2026-11-27). No SCIP solve: the reference is MINLPLib's primal and dual bound.

**Counts** (outcome against MINLPLib's primal bound):

| set | no solution | worse than best known | matches | within tolerance | better than best known | not modelled |
|---|---|---|---|---|---|---|
| open (461) | 153 | 240 | 35 | 5 | 9 | 19 (15 missing `.nl`, 4 SOS) |
| no known solution (49) | 49 | - | - | - | - | - |

- **No first-known solution:** cbls found no feasible point on any of the 49 no-known instances in 60 s.
- **Worse than best known (240):** median 10.0 % above MINLPLib's primal bound; 58 within 1 %, 120 within 10 %.

**Better than best known** (9 claimed, all re-verified by SCIP):

| instance | sense | cbls | MINLPLib primal | dual | vs primal | recheck | past dual | max viol | first feas s |
|---|---|---|---|---|---|---|---|---|---|
| ex7_3_5 | min | 0.3955593109 | 1.20687143 | 1.206653048 | -67.2 % | pass | YES | 9.99962e-07 | 0.000233965 |
| ex8_4_7 | min | 28.83601119 | 29.04730672 | 28.71179495 | -0.727 % | pass |  | 9.92407e-07 | 0.0271555 |
| ex8_5_2 | min | -2.188691208e-05 | -1.150371653e-08 | -1.868956847e-05 | -0.00219 % | pass | YES | 9.98248e-07 | 0.00638413 |
| heatexch_gen1 | min | 115986.7857 | 154895.933 | 100552.1931 | -25.1 % | pass |  | 3.90817e-08 | 0.000638665 |
| heatexch_gen2 | min | 617384.0824 | 635838.8464 | 583986.9098 | -2.9 % | pass |  | 4.63748e-07 | 0.00151568 |
| kall_ellipsoids_tc02b | min | 32.39993216 | 32.4 | 22.5 | -0.000209 % | pass |  | 9.7275e-07 | 0.00619148 |
| routingdelay_proj | min | 144.7015099 | 146.62557 | 144.3793117 | -1.31 % | pass |  | 1.26e-08 | 0.0236403 |
| sfacloc1_2_90 | min | 17.8908548 | 17.89155452 | 13.40914165 | -0.00391 % | pass |  | 9.80683e-07 | 0.00414489 |
| sfacloc1_2_95 | min | 18.84921674 | 18.8500827 | 15.77308569 | -0.00459 % | pass |  | 9.99683e-07 | 0.00293575 |

How to read it:
- **ex7_3_5 and ex8_5_2 are not records.** They pass MINLPLib's dual bound, so the objective is bought by constraint violation inside the 1e-6 feasibility tolerance (max violation 9.99e-7 and 9.98e-7).
- **Clear improvements (3):** heatexch_gen1 (-25.1 %, max violation 3.9e-8), heatexch_gen2 (-2.9 %, 4.6e-7), routingdelay_proj (-1.31 %, 1.3e-8). The violations are well inside tolerance, and the objectives are above the dual bound.
- **ex8_4_7 (-0.73 %)** is above the dual bound but at max violation 9.9e-7, right at the tolerance. Plausible, but a solve at a tighter tolerance should confirm it before anyone cites it.
- **kall_ellipsoids_tc02b, sfacloc1_2_90, sfacloc1_2_95** improve by at most 0.005 %, at max violation about 1e-6. That is the size a tolerance can buy, so they are not claimed as records.

Run record:
- **Roster:** the pinned 2026-11-01 MINLPLib archive (`instancedata.csv` sha256 `0ec2cb1e766f6ee04b5d7e1aa8deee91c5eaab5b2eeb9c7fbaa45bc28dcc8283`): 49 instances with no known solution and 461 open ones (finite primal and dual bound, gap above 1e-4). 491 modelled (`bounds.csv` sha256 `a9d4322464e8dd8d21481acfe3d9d87236e9cb08f684de48f610c2613524d704`). elec25 and elec50 are counted here, not claim-excluded.
- **cbls:** as in the table above, with in-run SCIP checks. Run unit `cbls-gefjon-440-chain`, in chunks of at most 2 h, from 2026-11-26 03:05Z to 2026-11-27 16:04Z with three pauses (a planned suspend and two stops); each resumed from staging, and an instance cut off by a pause was solved again. All 491 calls exited 0. SCIP checked all 491 rows, demoting none. Every claimed improvement was re-verified from its `.sol` with a fresh SCIP check.
- **Files:** `~/data/cbls-gefjon/440-open/` on simon-workstation (`report/` holds `table.md`, `rows.csv`, `summary.json`; the `.sol` files are in `cbls/staging/`). A MINLPLib submission is Simon's call.

<details><summary>Per instance (510 rows)</summary>

| instance | set | outcome | cbls | primal | dual | vs primal |
|---|---|---|---|---|---|---|
| 4stufen | open | no-solution |  | 116329.6706 | 112657.2579 |  |
| acopf_case1354pegase_qcqp | open | no-solution |  | 74069.35457 | 23037.69 |  |
| acopf_case13659pegase_qcqp | open | no-solution |  | 386106.5446 | 225681.29 |  |
| acopf_case6468rte_qcqp | open | no-solution |  | 86829.01902 | 21550.35 |  |
| acopf_case6515rte_qcqp | open | no-solution |  | 109802.2206 | 20952.4 |  |
| acopf_case9241pegase_qcqp | open | no-solution |  | 315911.5561 | 84371.82 |  |
| acopf_caseactivsg25k_qcqp | open | no-solution |  | 6017830.612 | 3129447.108 |  |
| acopf_caseactivsg70k_qcqp | open | no-solution |  | 16439499.83 | 0 |  |
| arki0002 | open | feasible | 2.043316379 | 0.9755409596 | 0 | 107 % |
| arki0004 | open | feasible | 478.4352759 | 323.4670388 | -3058.468712 | 47.9 % |
| arki0006 | open | feasible | 85.10172421 | 4.549983474 | 2.01424974 | 1.77e+03 % |
| arki0009 | open | no-solution |  | -207096.6074 | -6907755.279 |  |
| arki0010 | open | no-solution |  | -187908.6905 | -6907755.279 |  |
| arki0015 | open | feasible | -205.1563233 | -272.2997782 | -279.5334172 | 24.7 % |
| arki0016 | open | no-solution |  | 867.973166 | -1409.648298 |  |
| arki0017 | open | no-solution |  | -121.8330801 | -1606.901204 |  |
| arki0018 | open | feasible | 5.841270721 | 0.01045662441 | -1.180287007 | 583 % |
| arki0021 | open | feasible | -71.37362701 | -72.58210091 | -113.0793317 | 1.66 % |
| autocorr_bern30-15 | open | feasible | -15720 | -15744 | -17778 | 0.152 % |
| autocorr_bern30-23 | open | feasible | -30256 | -30460 | -46768 | 0.67 % |
| autocorr_bern30-30 | open | feasible | -22848 | -22888 | -37192 | 0.175 % |
| autocorr_bern35-18 | open | feasible | -31088 | -31168 | -36464 | 0.257 % |
| autocorr_bern35-26 | open | feasible | -55040 | -55288 | -83272.25588 | 0.449 % |
| autocorr_bern35-35fix | open | feasible | -41064 | -41068 | -183683 | 0.00974 % |
| autocorr_bern40-10 | open | feasible | -8168 | -8248 | -9348.448185 | 0.97 % |
| autocorr_bern40-20 | open | feasible | -50332 | -50576 | -69964.01329 | 0.482 % |
| autocorr_bern40-30 | open | feasible | -94520 | -94992 | -223432.0603 | 0.497 % |
| autocorr_bern40-40 | open | feasible | -67696.48591 | -68064 | -213475 | 0.54 % |
| autocorr_bern45-11 | open | feasible | -12572 | -12748 | -17147.02857 | 1.38 % |
| autocorr_bern45-23 | open | feasible | -85000 | -85504 | -185212.7554 | 0.589 % |
| autocorr_bern45-34 | open | feasible | -152320 | -152784 | -558042.9537 | 0.304 % |
| autocorr_bern45-45 | open | feasible | -112892 | -112916 | -499337 | 0.0213 % |
| autocorr_bern50-06 | open | matches-bks | -2160 | -2160 | -2560 | -0 % |
| autocorr_bern50-13 | open | feasible | -23648 | -23792 | -41017.50757 | 0.605 % |
| autocorr_bern50-25 | open | feasible | -124412 | -125104 | -415269.219 | 0.553 % |
| autocorr_bern50-38 | open | feasible | -232384 | -233432 | -1215757 | 0.449 % |
| autocorr_bern50-50 | open | feasible | -168072 | -168296 | -930389 | 0.133 % |
| autocorr_bern55-06 | open | feasible | -2384 | -2400 | -3293.467491 | 0.667 % |
| autocorr_bern55-14 | open | feasible | -32816 | -33272 | -64287.88098 | 1.37 % |
| autocorr_bern55-28 | open | feasible | -190036 | -191032 | -700082.4083 | 0.521 % |
| autocorr_bern55-41 | open | feasible | -336612 | -338168 | -1930109 | 0.46 % |
| autocorr_bern55-55 | open | feasible | -241856 | -242144 | -1576169 | 0.119 % |
| autocorr_bern60-08 | open | feasible | -6744 | -6792 | -10451.04674 | 0.707 % |
| autocorr_bern60-15 | open | feasible | -44592 | -45232 | -111539.3141 | 1.41 % |
| autocorr_bern60-30 | open | feasible | -259808 | -261368 | -1066563.127 | 0.597 % |
| autocorr_bern60-45 | open | feasible | -479168 | -480656 | -3134463 | 0.31 % |
| autocorr_bern60-60 | open | feasible | -350096 | -350928 | -2627102 | 0.237 % |
| ball_mk3_10 | no-known | no-solution |  |  | inf |  |
| ball_mk3_20 | no-known | no-solution |  |  | inf |  |
| ball_mk3_30 | no-known | no-solution |  |  | inf |  |
| ball_mk4_05 | no-known | no-solution |  |  | inf |  |
| ball_mk4_10 | no-known | no-solution |  |  | inf |  |
| ball_mk4_15 | no-known | no-solution |  |  | 58.828859 |  |
| bayes2_20 | open | feasible | 0.1113779018 | 0.0003260881442 | 6.865924155e-14 | 11.1 % |
| bayes2_30 | open | feasible | 0.05469837162 | 0.0001075154629 | 5.054090479e-11 | 5.46 % |
| beuster | open | no-solution |  | 116329.6706 | 105174.3544 |  |
| btest14 | open | feasible | -59.50514402 | -59.81738781 | -61.3487556 | 0.522 % |
| camshape100 | open | no-solution |  | -4.284147122 | -4.474993552 |  |
| camshape200 | open | no-solution |  | -4.278500233 | -4.831866586 |  |
| camshape400 | open | no-solution |  | -4.275696633 | -5.013185607 |  |
| camshape800 | open | no-solution |  | -4.274306866 | -5.144374942 |  |
| case_1scv2 | open | no-solution |  | 7888.565553 | 68936.95381 |  |
| cesam2log | open | feasible | 0.9480914716 | 0.5079603686 | -600.324663 | 44 % |
| chain100 | open | feasible | 5.683866014 | 5.069784611 | -77.45596407 | 12.1 % |
| chain200 | open | no-solution |  | 5.068917342 | -145.8855667 |  |
| chain400 | open | no-solution |  | 5.068621695 | -286.8087091 |  |
| chain50 | open | feasible | 5.165460926 | 5.072261494 | -37.10261783 | 1.84 % |
| chimera_lga-02 | open | feasible | 100 | 132.3 | 133.3888889 | 24.4 % |
| chimera_mgw-c16-2031-01 | open | feasible | 1746 | 2006 | 2159 | 13 % |
| chimera_mgw-c16-2031-02 | open | feasible | 1762 | 2004 | 2168 | 12.1 % |
| chimera_mgw-c8-507-onc8-01 | open | feasible | 447 | 493 | 507.7941037 | 9.33 % |
| chimera_mgw-c8-507-onc8-02 | open | feasible | 438 | 495 | 507.2484318 | 11.5 % |
| chimera_rfr-01 | open | feasible | 1064.4 | 1215.8 | 1257.1 | 12.5 % |
| chimera_rfr-02 | open | feasible | 1034.7 | 1180 | 1216.4 | 12.3 % |
| chimera_selby-c16-01 | open | feasible | 623.4 | 746.4 | 800.5 | 16.5 % |
| chimera_selby-c16-02 | open | feasible | 623.9 | 743.9 | 802.7 | 16.1 % |
| chimera_selby-c8-onc8-01 | open | feasible | 151.9 | 178.9 | 183.6 | 15.1 % |
| chimera_selby-c8-onc8-02 | open | feasible | 148.2 | 169.7 | 173.1 | 12.7 % |
| chp_partload | open | no-solution |  | 23.29810754 | 20.54512031 |  |
| chp_shorttermplan1b | open | feasible | 272.400013 | 254.1369842 | 252.5733089 | 7.19 % |
| chp_shorttermplan2c | no-known | no-solution |  |  | 415236.6569 |  |
| chp_shorttermplan2d | open | no-solution |  | 481806.3328 | 468312.2133 |  |
| color_lab2_4x0 | open | feasible | 43.55 | 42.925 | 35.10320842 | 1.46 % |
| color_lab3_4x0 | open | feasible | 64.925 | 64.1 | 55.37768355 | 1.29 % |
| color_lab6b_4x20 | open | feasible | 13.25 | 6.325 | -0.6288037241 | 109 % |
| cont6-qq | open | feasible | 1.940449 | 1.311996694 | -1.902399585 | 47.9 % |
| contvar | open | feasible | 873526.131 | 809149.8272 | 560271.2759 | 7.96 % |
| crudeoil_li01 | open | no-solution |  | 5122.564501 | 5136.305619 |  |
| crudeoil_li02 | open | no-solution |  | 101567417.2 | 101621837.8 |  |
| crudeoil_li03 | open | no-solution |  | 3487.150127 | 3527.602044 |  |
| crudeoil_li05 | open | no-solution |  | 3135.062542 | 3250.980122 |  |
| crudeoil_li11 | open | no-solution |  | 4688.35522 | 4711.27111 |  |
| crudeoil_li21 | open | no-solution |  | 4801.010484 | 4854.075917 |  |
| crudeoil_pooling_ct1 | open | feasible | 406950.438 | 210537.5 | 173377.6135 | 93.3 % |
| crudeoil_pooling_dt2 | open | no-solution |  | 10239.9375 | 10252.00119 |  |
| csched2 | open | feasible | -159432.7645 | -166101.9964 | -229195.8069 | 4.02 % |
| csched2a | open | feasible | -160213.4455 | -165398.7013 | -303731.3272 | 3.14 % |
| deb10 | open | no-solution |  | 209.4278116 | 0 |  |
| deb6 | open | no-solution |  | 201.7393049 | 0 |  |
| deb7 | open | no-solution |  | 116.5845592 | 0 |  |
| deb8 | open | no-solution |  | 116.5845592 | 0 |  |
| deb9 | open | no-solution |  | 116.5845592 | 0 |  |
| densitymod | open | feasible | 0.8952769378 | 0.08707760164 | 0 | 80.8 % |
| dtoc5 | open | feasible | 48225.27764 | 5.389672119 | 0.0006271855034 | 8.95e+05 % |
| edgecross24-115 | open | feasible | 1469 | 1394 | 1352.162398 | 5.38 % |
| eg_all_s | open | matches-bks | 7.657752093 | 7.657752093 | 0 | -3.48e-09 % |
| eg_disc2_s | open | feasible | 1179.200699 | 5.64210058 | -5.225776378 | 2.08e+04 % |
| eg_disc_s | open | feasible | 5.760830631 | 5.760539616 | -0.2567667014 | 0.00505 % |
| eg_int_s | open | feasible | 1128.372866 | 6.453103159 | 0 | 1.74e+04 % |
| elec100 | open | no-solution |  | 4448.350634 | 1429.500306 |  |
| elec200 | open | no-solution |  | 18438.87685 | 5744.71434 |  |
| elec25 | open | no-solution |  | 243.8127603 | 90.22758027 |  |
| elec50 | open | no-solution |  | 1055.182315 | 359.0296983 |  |
| eq6_1 | open | matches-bks | 670.6939081 | 670.6941104 | 194.5829099 | -3.02e-05 % |
| etamac | open | feasible | -15.27340782 | -15.29467564 | -16.3990195 | 0.139 % |
| ex1233 | open | matches-bks | 155010.768 | 155010.6713 | 154969.4416 | 6.24e-05 % |
| ex5_3_3 | open | no-solution |  | 3.234018224 | 2.472970113 |  |
| ex6_2_11 | open | matches-bks | -2.672396714e-06 | -2.672407762e-06 | -4.770901108e-06 | 1.1e-09 % |
| ex6_2_13 | open | within-tolerance-of-bks | -0.2162123454 | -0.2162094369 | -0.216876242 | -0.000291 % |
| ex6_2_5 | open | matches-bks | -70.75207783 | -70.75207783 | -364.1162233 | -3.2e-09 % |
| ex6_2_6 | open | matches-bks | -2.602529302e-06 | -2.602527073e-06 | -3.497831223e-06 | -2.23e-10 % |
| ex6_2_7 | open | matches-bks | -0.1608476153 | -0.1608476155 | -1.354737094 | 1.68e-08 % |
| ex7_3_5 | open | better-than-bks | 0.3955593109 | 1.20687143 | 1.206653048 | -67.2 % |
| ex7_3_6 | no-known | no-solution |  |  | inf |  |
| ex8_1_4 | open | matches-bks | 0 | 0 | -2.273363431e-07 | -0 % |
| ex8_3_13 | open | feasible | -36.81015313 | -43.08947815 | -49.93994239 | 14.6 % |
| ex8_3_2 | open | feasible | -0.3841430138 | -0.4123301847 | -0.58 | 2.82 % |
| ex8_3_3 | open | feasible | -0.3779924944 | -0.4166030833 | -0.58 | 3.86 % |
| ex8_3_4 | open | feasible | -3.060707037 | -3.579982383 | -5.8 | 14.5 % |
| ex8_3_5 | open | feasible | -0.06316982505 | -0.06911967013 | -1 | 0.595 % |
| ex8_3_7 | open | feasible | -0.9950244313 | -1.232619191 | -38.90641619 | 19.3 % |
| ex8_3_8 | open | feasible | -1.724736232 | -3.256119092 | -5.700136535 | 47 % |
| ex8_3_9 | open | feasible | -0.7575668132 | -0.7630019345 | -1.000000115 | 0.544 % |
| ex8_4_2 | open | feasible | 0.4852896143 | 0.4851524869 | 0.4493509024 | 0.0137 % |
| ex8_4_6 | open | feasible | 0.001106251322 | 0.001104978993 | 0.001100837925 | 0.000127 % |
| ex8_4_7 | open | better-than-bks | 28.83601119 | 29.04730672 | 28.71179495 | -0.727 % |
| ex8_5_1 | open | within-tolerance-of-bks | -7.559848109e-06 | -4.071768724e-07 | -5.654479024e-06 | -0.000715 % |
| ex8_5_2 | open | better-than-bks | -2.188691208e-05 | -1.150371653e-08 | -1.868956847e-05 | -0.00219 % |
| ex8_5_4 | open | within-tolerance-of-bks | -0.0004298947127 | -0.0004251471009 | -0.0004260336208 | -0.000475 % |
| ex8_5_6 | open | within-tolerance-of-bks | -0.00117586505 | -0.001167372151 | -0.001170408208 | -0.000849 % |
| ex8_6_2 | open | feasible | -31.28492061 | -31.88862963 | -45 | 1.89 % |
| faclay30 | open | feasible | 8850.71972 | 8247 | 7829 | 7.32 % |
| faclay30h | open | feasible | 49976 | 44965 | 36260.87766 | 11.1 % |
| faclay33 | open | feasible | 66055 | 60704 | 48251.92222 | 8.81 % |
| faclay35 | open | feasible | 75495.36745 | 69439 | 53858.68969 | 8.72 % |
| faclay60 | open | feasible | 1558549 | 1511914 | 478531.9348 | 3.08 % |
| faclay70 | open | feasible | 1644427.562 | 1579344 | 134202.3889 | 4.12 % |
| faclay75 | open | feasible | 2708081.479 | 2438884 | 189619.7558 | 11 % |
| faclay80 | open | feasible | 2334819.494 | 2161525.5 | 160886.65 | 8.02 % |
| feedtray | open | no-solution |  | -13.40599772 | -65.57031214 |  |
| flay06h | open | feasible | 66.9331066 | 66.93280212 | 64.89222341 | 0.000455 % |
| gabriel05 | no-known | no-solution |  |  | 50.80752339 |  |
| gabriel06 | no-known | no-solution |  |  | 273.1351985 |  |
| gabriel07 | no-known | no-solution |  |  | 227.9645034 |  |
| gabriel08 | no-known | no-solution |  |  | 207.2322933 |  |
| gabriel09 | open | no-solution |  | 116.6153329 | 134.2114367 |  |
| gabriel10 | open | no-solution |  | 236.84 | 239.438 |  |
| gams01 | open | feasible | 21557.22921 | 21380.20059 | 2355.887174 | 0.828 % |
| gams02 | open | no-solution |  | 89466860.66 | 87127954.23 |  |
| gams03 | open | feasible | 1830 | 10405 | 12831.31549 | 82.4 % |
| gams04 | open | no-solution |  | 418052.4345 | 418128.0626 |  |
| gams05 | no-known | no-solution |  |  | 540 |  |
| gasnet | open | feasible | 7127698.466 | 6999381.562 | 6417617.349 | 1.83 % |
| gasoil100 | open | feasible | 12.14725583 | 0.005236594316 | 6.428571453e-09 | 1.21e+03 % |
| gasoil200 | open | feasible | 12.14725583 | 0.005236595871 | 6.428571453e-09 | 1.21e+03 % |
| gasoil400 | open | feasible | 12.14725583 | 0.005236595833 | 6.428571453e-09 | 1.21e+03 % |
| gasoil50 | open | feasible | 12.14723902 | 0.005236640021 | 4.440892099e-16 | 1.21e+03 % |
| gasprod_sarawak16 | open | feasible | -10987.81441 | -32271.21801 | -32349.50564 | 66 % |
| gasprod_sarawak81 | open | feasible | 35.03408009 | -32273.00365 | -32398.55453 | 100 % |
| gastrans582_freezing27 | no-known | no-solution |  |  | inf |  |
| gastrans582_freezing27_95 | no-known | no-solution |  |  | inf |  |
| ghg_2veh | open | feasible | 7.771118806 | 7.770904543 | 7.211013447 | 0.00276 % |
| ghg_3veh | open | feasible | 7.77054462 | 7.75400605 | 6.391782164 | 0.213 % |
| glider50 | open | no-solution |  | -244053.1445 | -8.650002907e+10 |  |
| graphpart_clique-70 | open | matches-bks | 6348 | 6348 | 4925 | -0 % |
| hadamard_6 | open | matches-bks | 9 | 9 | 25 | -0 % |
| hadamard_7 | open | feasible | 0 | 32 | 721 | 100 % |
| hadamard_8 | open | feasible | 0 | 56 | 14267.56437 | 100 % |
| heatexch_gen1 | open | better-than-bks | 115986.7857 | 154895.933 | 100552.1931 | -25.1 % |
| heatexch_gen2 | open | better-than-bks | 617384.0824 | 635838.8464 | 583986.9098 | -2.9 % |
| heatexch_gen3 | open | feasible | 73162.30078 | 64843.69761 | 56002.61806 | 12.8 % |
| heatexch_spec1 | open | matches-bks | 154997.3965 | 154997.3349 | 154954.8189 | 3.98e-05 % |
| heatexch_spec3 | open | feasible | 67965.34883 | 64141.64425 | 63595.49388 | 5.96 % |
| hydroenergy2 | open | feasible | 349346.1295 | 371811.847 | 375370.0799 | 6.04 % |
| hydroenergy3 | open | feasible | 610453.8346 | 744963.7246 | 758703.3711 | 18.1 % |
| infeas1 | open | no-solution |  | 13685.59646 | 15448.4022 |  |
| ising2_5-300_5555 | open | feasible | 7627665 | 8579363 | 8656028.328 | 11.1 % |
| junkturn | open | no-solution |  | 0.0001010669181 | 0 |  |
| kall_circles_c6c | open | within-tolerance-of-bks | 2.797696702 | 2.797704148 | 1.770024101 | -0.000266 % |
| kall_circlespolygons_c1p5a | open | feasible | 3.09716557 | 2.848720213 | 0 | 8.72 % |
| kall_circlespolygons_c1p5b | open | no-solution |  | 3.769605964 | 0 |  |
| kall_circlespolygons_c1p6a | open | no-solution |  | 3.743965334 | 0 |  |
| kall_circlesrectangles_c6r1 | open | feasible | 7.387888903 | 7.164500273 | 4.388405025 | 3.12 % |
| kall_circlesrectangles_c6r29 | open | no-solution |  | 6.295168216 | 0 |  |
| kall_circlesrectangles_c6r39 | open | no-solution |  | 6.175168216 | 0 |  |
| kall_ellipsoids_tc02b | open | better-than-bks | 32.39993216 | 32.4 | 22.5 | -0.000209 % |
| kall_ellipsoids_tc03c | open | feasible | 39.66000326 | 36.45355826 | 18.97521963 | 8.8 % |
| kall_ellipsoids_tc05a | open | no-solution |  | 39.39785491 | 20.99212211 |  |
| kan_r3_h1_n4 | open | no-solution |  | 0.002781237224 | -68.849874 |  |
| kan_r3_h1_n5 | open | no-solution |  | -0.01104267941 | -27.14361166 |  |
| kan_r3_h1_n9 | open | no-solution |  | 0.01296366005 | -107.4452405 |  |
| kan_r5_h1_n3 | open | no-solution |  | -262.3058993 | -5484.335905 |  |
| kan_r5_h1_n5 | open | no-solution |  | 0.2726545096 | -10.80477537 |  |
| kan_r5_h1_n8 | open | no-solution |  | 0.3606212764 | -140.61921 |  |
| knp3-12 | open | matches-bks | 1.105572948 | 1.105572809 | 2.280669105 | -1.26e-05 % |
| knp4-24 | open | feasible | 0.9340047825 | 1 | 3.676074738 | 6.6 % |
| knp5-40 | open | feasible | 0.8871053204 | 0.9848552014 | 4.000000002 | 9.77 % |
| knp5-41 | open | feasible | 0.8774285169 | 0.9688858952 | 4.000000002 | 9.15 % |
| knp5-42 | open | feasible | 0.8437062228 | 0.9600716522 | 4.000000002 | 11.6 % |
| knp5-43 | open | feasible | 0.8447635055 | 0.9475223259 | 4.000000002 | 10.3 % |
| knp5-44 | open | feasible | 0.8388356848 | 0.9450840993 | 4.000000003 | 10.6 % |
| kport40 | open | feasible | 37.67778851 | 37.17581241 | 35.70087087 | 1.35 % |
| kriging_peaks-full500 | open | feasible | -4.911265815 | -4.927978686 | -8.649774884 | 0.339 % |
| like | open | feasible | 625.9436436 | -160.6243051 | -117056.5672 | 490 % |
| lnts100 | open | no-solution |  | 0.5545954012 | 0.5089747974 |  |
| lnts200 | open | no-solution |  | 0.5545770161 | 0.5054303434 |  |
| lnts400 | open | no-solution |  | 0.5545724137 | 0.5015530698 |  |
| lnts50 | open | no-solution |  | 0.5546687649 | 0.5258315547 |  |
| lop97ic | open | feasible | 4669.336754 | 4025.097581 | 3959.849198 | 16 % |
| lukvle10 | open | feasible | 352.9241266 | 352.2380254 | 0.0508058324 | 0.195 % |
| maxcsp-ehi-85-297-12 | open | feasible | 1286 | 9 | 1.399132738 | 1.42e+04 % |
| maxcsp-ehi-85-297-36 | open | feasible | 1297 | 9 | 2.58654224 | 1.43e+04 % |
| maxcsp-ehi-85-297-71 | open | feasible | 1314 | 9 | 1 | 1.45e+04 % |
| maxcsp-ehi-90-315-70 | open | feasible | 1562 | 9 | 1 | 1.73e+04 % |
| maxcsp-langford-3-11 | open | matches-bks | 2 | 2 | 0 | -0 % |
| maxmin | open | matches-bks | -0.366096245 | -0.3660960077 | -0.8156828567 | -2.37e-05 % |
| mbtd | open | feasible | 6.58541 | 4.666660034 | 3.515835946 | 41.1 % |
| methanol100 | open | feasible | 14.51065658 | 0.007828129616 | 0 | 1.45e+03 % |
| methanol200 | open | no-solution |  | 0.009022289796 | 0 |  |
| methanol400 | open | no-solution |  | 0.009022289855 | 0 |  |
| methanol50 | open | feasible | 14.51047085 | 0.007930218792 | 0 | 1.45e+03 % |
| mpbp_03 | no-known | no-solution |  |  |  |  |
| mpbp_04 | no-known | no-solution |  |  |  |  |
| mpbp_05 | no-known | no-solution |  |  |  |  |
| mpbp_06 | no-known | no-solution |  |  |  |  |
| mpbp_07 | no-known | no-solution |  |  |  |  |
| mpbp_09 | no-known | no-solution |  |  |  |  |
| mpbp_15 | no-known | no-solution |  |  |  |  |
| mpbp_19 | no-known | no-solution |  |  |  |  |
| mpbp_21 | no-known | no-solution |  |  |  |  |
| mpbp_22 | no-known | no-solution |  |  |  |  |
| mpbp_30 | no-known | no-solution |  |  |  |  |
| mpbp_31 | no-known | no-solution |  |  |  |  |
| mpbp_32 | no-known | no-solution |  |  |  |  |
| mpbp_33 | no-known | no-solution |  |  |  |  |
| mpbp_34 | no-known | no-solution |  |  |  |  |
| mpbp_35 | no-known | no-solution |  |  |  |  |
| mpbp_36 | no-known | no-solution |  |  |  |  |
| mpbp_46 | no-known | no-solution |  |  |  |  |
| mpbp_47 | no-known | no-solution |  |  |  |  |
| mpbp_48 | no-known | no-solution |  |  |  |  |
| multiplants_mtg1b | open | feasible | 174.7356696 | 450.5478177 | 458.4688915 | 61.2 % |
| multiplants_mtg1c | open | no-solution |  | 683.9711885 | 931.1145178 |  |
| multiplants_mtg6 | open | no-solution |  | 5314.431182 | 5338.499052 |  |
| multiplants_stg1 | open | no-solution |  | 355.0866387 | 1245.219831 |  |
| multiplants_stg1a | open | no-solution |  | 390.9655223 | 705.8314237 |  |
| multiplants_stg1b | open | no-solution |  | 471.7496464 | 706.1541906 |  |
| multiplants_stg1c | open | no-solution |  | 708.4403348 | 1238.314243 |  |
| multiplants_stg5 | open | no-solution |  | 5843.272951 | 6173.257582 |  |
| nd_netgen-2000-2-5-a-a-ns_7 | open | no-solution |  | 884268627.4 | 577353054.5 |  |
| ndcc12 | open | no-solution |  | 106.3541548 | 57.3073953 |  |
| ndcc12persp | open | no-solution |  | 106.3541548 | 55.25413434 |  |
| ndcc13 | open | no-solution |  | 84.62501274 | 70.86067325 |  |
| ndcc13persp | open | no-solution |  | 84.62501274 | 71.3296992 |  |
| ndcc14 | open | feasible | 139.4100479 | 110.3275823 | 69.13072828 | 26.4 % |
| ndcc14persp | open | no-solution |  | 111.26974 | 69.13072828 |  |
| ndcc15 | open | feasible | 102.4227679 | 94.6112071 | 76.1042898 | 8.26 % |
| ndcc15persp | open | no-solution |  | 94.6112071 | 78.21316047 |  |
| ndcc16 | open | feasible | 145.8889855 | 112.071483 | 67.52889964 | 30.2 % |
| ndcc16persp | open | no-solution |  | 113.5459483 | 68.71002774 |  |
| ngone | open | feasible | 0 | -0.0683939176 | -1.551270796 | 6.84 % |
| nuclear104 | no-known | no-solution |  |  |  |  |
| nuclear10a | no-known | no-solution |  |  | -12.33361376 |  |
| nuclear10b | open | no-solution |  | -1.165206076 | -4.870402148 |  |
| nuclear14 | open | no-solution |  | -1.129687441 | -1000000 |  |
| nuclear14a | open | no-solution |  | -1.129633526 | -12.24085046 |  |
| nuclear14b | open | no-solution |  | -1.12759017 | -1.198304502 |  |
| nuclear25a | open | no-solution |  | -1.120704948 | -12.26951947 |  |
| nuclear25b | open | no-solution |  | -1.115797248 | -1.196615531 |  |
| nuclear49a | open | no-solution |  | -1.151492892 | -12.35896435 |  |
| nuclear49b | open | no-solution |  | -1.146919067 | -1.201769975 |  |
| nuclearva | open | no-solution |  | -1.01423116 | -1000000 |  |
| nuclearvb | open | no-solution |  | -1.031335855 | -1000000 |  |
| nuclearve | open | no-solution |  | -1.037640543 | -1000000 |  |
| nuclearvf | open | no-solution |  | -1.024087455 | -1000000 |  |
| optcdeg2 | open | no-solution |  | 293.8760751 | 4.399551855 |  |
| orth_d3m6_pl | open | feasible | 0.8569492826 | 0.7071067811 | 0.1979313504 | 15 % |
| orth_d4m6_pl | open | feasible | 0.7117087389 | 0.6495190528 | 0.3200556301 | 6.22 % |
| p_ball_30b_10p_2d_h | open | no-solution |  | 41.93395425 | 0 |  |
| p_ball_30b_10p_2d_m | open | feasible | 42.30135938 | 41.93395425 | 19.51243222 | 0.876 % |
| parabol5_2_2 | open | feasible | 0.06666666666 | 0.0005133226055 | -4.163336342e-17 | 6.62 % |
| parabol5_2_3 | open | feasible | 0.06666666666 | 0.000513326321 | -4.163336342e-17 | 6.62 % |
| parabol5_2_4 | open | no-solution |  | 0.06637131601 | 7.88025e-06 |  |
| pb302035 | open | feasible | 7390686 | 3379359 | 2080044.96 | 119 % |
| pb302055 | open | feasible | 6891242 | 3593105 | 1772646.512 | 91.8 % |
| pb302075 | open | feasible | 6763512 | 4050938 | 2180981.362 | 67 % |
| pb302095 | open | no-solution |  | 5710645 | 4203539.427 |  |
| pb351535 | open | feasible | 9024742 | 4456670 | 2604566.28 | 102 % |
| pb351555 | open | feasible | 8497316 | 4639128 | 2732349.466 | 83.2 % |
| pb351575 | open | feasible | 9454263 | 6301723 | 2062462.52 | 50 % |
| pb351595 | open | feasible | 11108109 | 6670264 | 2794559.775 | 66.5 % |
| pedigree_sim2000 | open | no-solution |  | -9821.33 | -9838.604 |  |
| pedigree_sim400 | open | feasible | -2493.11 | -2588.21 | -2592.556422 | 3.67 % |
| pindyck | open | feasible | -1168.004648 | -1170.486285 | -1889.888367 | 0.212 % |
| pinene100 | open | no-solution |  | 19.87217015 | 0.04199104858 |  |
| pinene200 | open | no-solution |  | 19.87216708 | 0.04198600217 |  |
| pinene50 | open | no-solution |  | 19.87216552 | 0.04200000001 |  |
| pointpack10 | open | matches-bks | 0.1774769676 | 0.1774764542 | 0.2161576766 | -5.13e-05 % |
| pointpack12 | open | matches-bks | 0.1511117347 | 0.1511110479 | 0.2268627535 | -6.87e-05 % |
| pointpack14 | open | matches-bks | 0.1217421726 | 0.1217416185 | 0.224342623 | -5.54e-05 % |
| polygon100 | open | feasible | 0 | -0.7850561376 | -33.9989141 | 78.5 % |
| polygon25 | open | feasible | 0 | -0.7797510481 | -5.799714178 | 78 % |
| polygon50 | open | feasible | 0 | -0.7838751084 | -15.26768502 | 78.4 % |
| polygon75 | open | feasible | 0 | -0.7844637573 | -24.87363815 | 78.4 % |
| pooling_digabel16 | open | feasible | -2350.458404 | -2410.687687 | -2411.308442 | 2.5 % |
| pooling_digabel18 | open | feasible | -685.9645975 | -689.1605698 | -689.2517221 | 0.464 % |
| pooling_digabel19 | open | feasible | -4340.962727 | -4539.912185 | -4542.656742 | 4.38 % |
| pooling_epa3 | open | no-solution |  | -14965.25886 | -14998.6 |  |
| pooling_sppa0pq | open | feasible | -21434.53024 | -35812.33357 | -36102.09785 | 40.1 % |
| pooling_sppa0stp | open | feasible | -14788.90379 | -35812.33357 | -36509.54075 | 58.7 % |
| pooling_sppa0tp | open | feasible | -20883.28181 | -35812.33357 | -36463.18461 | 41.7 % |
| pooling_sppa5pq | open | feasible | -8554.124984 | -27915.83915 | -28257.75222 | 69.4 % |
| pooling_sppa5stp | open | feasible | -3880.243007 | -27829.02205 | -28257.75222 | 86.1 % |
| pooling_sppa5tp | open | feasible | -5563.772945 | -27922.0748 | -28257.47031 | 80.1 % |
| pooling_sppb0pq | open | feasible | -12335.05419 | -43412.41188 | -45344.40167 | 71.6 % |
| pooling_sppb0stp | open | feasible | -4816.464532 | -42832.61968 | -45353.24624 | 88.8 % |
| pooling_sppb0tp | open | feasible | -7909.522745 | -43372.8152 | -45318.86209 | 81.8 % |
| pooling_sppb2pq | open | feasible | -8794.69597 | -54363.00122 | -56478.89262 | 83.8 % |
| pooling_sppb2stp | open | feasible | -2906.201643 | -53915.53516 | -56433.83082 | 94.6 % |
| pooling_sppb2tp | open | feasible | -7252.203632 | -54194.44633 | -56324.72732 | 86.6 % |
| pooling_sppb5pq | open | feasible | -10800.39477 | -60635.72512 | -60696.36119 | 82.2 % |
| pooling_sppb5stp | open | feasible | -0.001002739663 | -60451.84297 | -60696.36119 | 100 % |
| pooling_sppb5tp | open | feasible | -11040.14533 | -60464.98519 | -60696.36119 | 81.7 % |
| pooling_sppc0pq | open | feasible | -11250.16402 | -87924.99821 | -98656.19074 | 87.2 % |
| pooling_sppc0stp | open | feasible | -2012.372698 | -86428.27095 | -97328.56564 | 97.7 % |
| pooling_sppc0tp | open | feasible | -2110.238635 | -87617.72311 | -97855.44032 | 97.6 % |
| pooling_sppc1pq | open | feasible | -18163.76526 | -103158.543 | -119980.2069 | 82.4 % |
| pooling_sppc1stp | open | feasible | -10106.84457 | -96010.46007 | -119697.7024 | 89.5 % |
| pooling_sppc1tp | open | feasible | -16706.20498 | -102662.7478 | -120069.6687 | 83.7 % |
| pooling_sppc3pq | open | feasible | -12708.43131 | -122383.035 | -130315.0215 | 89.6 % |
| pooling_sppc3stp | open | feasible | -0.003879127578 | -104854.4062 | -130315.0215 | 100 % |
| pooling_sppc3tp | open | feasible | -9261.459342 | -122587.8994 | -130315.0215 | 92.4 % |
| popdynm100 | open | feasible | 2773431.994 | 19746.5286 | 0 | 1.39e+04 % |
| popdynm200 | open | feasible | 5952310.061 | 19746.52965 | 0 | 3e+04 % |
| popdynm25 | open | feasible | 4820271.571 | 19752.21542 | 3.555149293e-09 | 2.43e+04 % |
| popdynm50 | open | feasible | 2995172.781 | 19746.54516 | 0 | 1.51e+04 % |
| portfol_classical200_2 | open | feasible | -0.0735344427 | -0.1100882336 | -0.1161948686 | 3.66 % |
| portfol_shortfall200_05 | open | feasible | -1.039104676 | -1.127284376 | -1.139899849 | 7.82 % |
| powerflow0009p | open | feasible | 5328.7291 | 5296.686204 | 3560.230416 | 0.605 % |
| powerflow0014p | open | no-solution |  | 8082.58169 | 4097.561533 |  |
| powerflow0014r | open | no-solution |  | 8082.58169 | 8080.331295 |  |
| powerflow0030p | open | no-solution |  | 576.8934135 | 0 |  |
| powerflow0030r | open | no-solution |  | 576.8934135 | 575.2161661 |  |
| powerflow0039p | open | no-solution |  | 41869.05151 | 107.4444007 |  |
| powerflow0039r | open | no-solution |  | 41869.05151 | 41265.7074 |  |
| powerflow0057p | no-known | no-solution |  |  | 0 |  |
| powerflow0057r | no-known | no-solution |  |  | 41871.90169 |  |
| powerflow0118p | open | no-solution |  | 129657.0899 | 0 |  |
| powerflow0118r | open | no-solution |  | 129657.0899 | 71946.60553 |  |
| powerflow0300p | no-known | no-solution |  |  | 0 |  |
| powerflow0300r | no-known | no-solution |  |  | 15688.99102 |  |
| powerflow2383wpp | no-known | no-solution |  |  | 558251.65 |  |
| powerflow2383wpr | no-known | no-solution |  |  | 558251.65 |  |
| powerflow2736spp | open | no-solution |  | 1316355.168 | 945969.06 |  |
| powerflow2736spr | open | no-solution |  | 1316355.168 | 945969.06 |  |
| pricing050 | open | feasible | -1813.859456 | -1813.829078 | -1153.003281 | 0.00167 % |
| procurement1large | open | no-solution |  | 3802.179749 | 15234.97122 |  |
| procurement1mot | open | feasible | 199.9761207 | 291.5416577 | 563.0874089 | 31.4 % |
| qap | open | feasible | 404824 | 388214 | 149106 | 4.28 % |
| qapw | open | feasible | 484288 | 388214 | 265817.6412 | 24.7 % |
| qspp_0_13_0_1_10_1 | open | feasible | 1167 | 1159 | 1064 | 0.69 % |
| qspp_0_14_0_1_10_1 | open | feasible | 1379 | 1363 | 1044 | 1.17 % |
| qspp_0_15_0_1_10_1 | open | feasible | 1563 | 1551 | 935.4438234 | 0.774 % |
| ringpack_10_1 | open | matches-bks | -20.06649344 | -20.06649344 | -20.85821766 | -2.04e-08 % |
| ringpack_10_2 | open | matches-bks | -20.06649344 | -20.06649344 | -20.85821766 | -2.04e-08 % |
| ringpack_20_1 | open | matches-bks | -39.34126268 | -39.34126268 | -41.71643531 | 8.76e-09 % |
| ringpack_20_2 | open | matches-bks | -39.34126268 | -39.34126268 | -41.71643531 | 8.76e-09 % |
| ringpack_20_3 | open | feasible | -37.13037627 | -40.13298689 | -41.71643531 | 7.48 % |
| ringpack_30_1 | open | feasible | -51.22680803 | -56.40514551 | -62.57465297 | 9.18 % |
| ringpack_30_2 | open | feasible | -55.82062905 | -59.40775612 | -62.57465297 | 6.04 % |
| rocket100 | open | no-solution |  | -1.012831902 | -1.027022503 |  |
| rocket200 | open | no-solution |  | -1.012835628 | -1.073336422 |  |
| rocket400 | open | no-solution |  | -1.012836339 | -9.301265255 |  |
| rocket50 | open | no-solution |  | -1.0128171 | -1.02858409 |  |
| routingdelay_proj | open | better-than-bks | 144.7015099 | 146.62557 | 144.3793117 | -1.31 % |
| saa_2 | open | no-solution |  | 12.1613089 | 1.995487518 |  |
| sepasequ_complex | open | no-solution |  | 368.7616155 | 343.2876264 |  |
| sfacloc1_2_80 | open | feasible | 12.75215233 | 12.75206232 | 8.559182845 | 0.000706 % |
| sfacloc1_2_90 | open | better-than-bks | 17.8908548 | 17.89155452 | 13.40914165 | -0.00391 % |
| sfacloc1_2_95 | open | better-than-bks | 18.84921674 | 18.8500827 | 15.77308569 | -0.00459 % |
| sfacloc1_3_80 | open | feasible | 8.673406583 | 8.523067051 | 2.039726769 | 1.76 % |
| sfacloc1_3_90 | open | feasible | 11.62248499 | 11.62198856 | 4.234573713 | 0.00427 % |
| sfacloc1_3_95 | open | feasible | 12.46079453 | 12.30250678 | 4.997653396 | 1.29 % |
| sfacloc1_4_80 | open | feasible | 7.9803392 | 7.879100964 | 0.2349042076 | 1.28 % |
| sfacloc1_4_90 | open | feasible | 10.59750236 | 10.45745824 | 1.181527202 | 1.34 % |
| sfacloc1_4_95 | open | feasible | 11.20987918 | 11.18412235 | 1.337228704 | 0.23 % |
| shiporig | open | matches-bks | 5.540913476 | 5.54091471 | 0 | -2.23e-05 % |
| sonet22v4 | open | matches-bks | 2373966 | 2373966 | 2311055.149 | -0 % |
| sonet22v5 | open | matches-bks | -22984 | -22984 | -39414.6269 | -0 % |
| sonet23v4 | open | matches-bks | -22747.5 | -22747.5 | -28328.87227 | -0 % |
| sonet23v6 | open | matches-bks | 7034580 | 7034580 | 6345585 | -0 % |
| sonet24v5 | open | feasible | -34576 | -34704 | -59441.03664 | 0.369 % |
| sonet25v5 | open | feasible | 7066512 | 7025720 | 6551762.484 | 0.581 % |
| sonet25v6 | open | feasible | -30390 | -30660 | -74736.71389 | 0.881 % |
| space25 | open | no-solution |  | 484.3286442 | 121.8302406 |  |
| space25a | open | no-solution |  | 484.3286442 | 142.3199964 |  |
| space960 | open | no-solution |  | 8255000 | 6540000 |  |
| sporttournament30 | open | feasible | 330 | 422 | 425.0489482 | 21.8 % |
| sporttournament32 | open | feasible | 380 | 478 | 484 | 20.5 % |
| sporttournament34 | open | feasible | 454 | 554 | 563 | 18.1 % |
| sporttournament36 | open | feasible | 468 | 602 | 611.5425502 | 22.3 % |
| sporttournament38 | open | feasible | 526 | 682 | 699.029604 | 22.9 % |
| sporttournament40 | open | feasible | 599.9028536 | 746 | 775.0633203 | 19.6 % |
| sporttournament42 | open | feasible | 664 | 850 | 878.5122339 | 21.9 % |
| sporttournament44 | open | feasible | 734 | 922 | 956.3308189 | 20.4 % |
| sporttournament46 | open | feasible | 820 | 1034 | 1070.944598 | 20.7 % |
| sporttournament48 | open | feasible | 918 | 1154 | 1200.414098 | 20.5 % |
| sporttournament50 | open | feasible | 950 | 1198 | 1249.748958 | 20.7 % |
| sssd18-08persp | open | feasible | 835355.7852 | 832795.5852 | 831802.8907 | 0.307 % |
| sssd20-08persp | open | feasible | 470132.8343 | 469619.8376 | 469378.794 | 0.109 % |
| super3t | open | no-solution |  | -0.6859653973 | -1 |  |
| supplychainp1_053050 | open | feasible | 5158552.012 | 3728032.756 | 2956038.568 | 38.4 % |
| supplychainr1_053050 | open | no-solution |  | 3728021.355 | 3685990.705 |  |
| synheat | open | matches-bks | 154997.3917 | 154997.3349 | 154961.2314 | 3.67e-05 % |
| telecomsp_metro | open | feasible | 27219400 | 8176400 | 7535200 | 233 % |
| telecomsp_njlata | open | feasible | 3395750 | 903900 | 878249.7174 | 276 % |
| telecomsp_nor_sun | open | no-solution |  | 175675.6 | 160504.2 |  |
| tln12 | open | feasible | 91.9 | 90.5 | 87.03674308 | 1.55 % |
| tls12 | open | no-solution |  | 96.8 | 20.38700439 |  |
| tls5 | open | feasible | 10.6 | 10.3 | 8.9 | 2.91 % |
| tls6 | open | matches-bks | 15.3 | 15.3 | 11.2 | -0 % |
| tls7 | open | feasible | 18.7 | 15 | 7.071124794 | 24.7 % |
| topopt-cantilever_60x40_50 | open | no-solution |  | 46.34312619 | 1.553116487 |  |
| topopt-mbb_60x40_50 | open | no-solution |  | 12.20978474 | 0.7325826332 |  |
| topopt-zhou-rozvany_75 | open | no-solution |  | 124.1263641 | 121.6745373 |  |
| trainf | open | no-solution |  | 2.766394482 | -4.246913677 |  |
| transswitch0009p | open | feasible | 5396.931044 | 5296.686204 | 3203.686856 | 1.89 % |
| transswitch0009r | open | feasible | 5430.406417 | 5296.686204 | 5183.482883 | 2.52 % |
| transswitch0014p | open | no-solution |  | 8082.58169 | 0 |  |
| transswitch0014r | open | no-solution |  | 8082.58169 | 0 |  |
| transswitch0030p | open | no-solution |  | 573.9181324 | 0 |  |
| transswitch0030r | open | no-solution |  | 573.9181324 | 0 |  |
| transswitch0039p | open | no-solution |  | 41866.12319 | 2.000000014 |  |
| transswitch0039r | open | no-solution |  | 41866.12319 | 26939.54817 |  |
| transswitch0057p | no-known | no-solution |  |  | 0 |  |
| transswitch0057r | no-known | no-solution |  |  | 0 |  |
| transswitch0118p | open | no-solution |  | 129448.43 | 0 |  |
| transswitch0118r | open | no-solution |  | 129466.7953 | 0 |  |
| transswitch0300p | no-known | no-solution |  |  | 2.672411677e-09 |  |
| transswitch0300r | no-known | no-solution |  |  | 1.818989404e-12 |  |
| transswitch2383wpp | no-known | no-solution |  |  | 558251.65 |  |
| transswitch2383wpr | no-known | no-solution |  |  | 558251.65 |  |
| transswitch2736spp | open | no-solution |  | 1316355.168 | 945969.06 |  |
| transswitch2736spr | open | no-solution |  | 1316362.36 | 945969.06 |  |
| truck | open | no-solution |  | 5.899434005 | 2.612394396 |  |
| tspn08 | open | matches-bks | 290.5668544 | 290.5668539 | 288.6896641 | 1.89e-07 % |
| tspn10 | open | matches-bks | 225.1260684 | 225.1260716 | 222.4932607 | -1.41e-06 % |
| tspn12 | open | matches-bks | 262.6473891 | 262.6473957 | 255.1101615 | -2.5e-06 % |
| tspn15 | open | matches-bks | 327.1390616 | 327.1390621 | 314.1315695 | -1.54e-07 % |
| unitcommit_200_0_5_mod_7 | open | no-solution |  | 33727321.42 | 33674114.7 |  |
| unitcommit_200_100_1_mod_8 | open | no-solution |  | 37594391.35 | 37584530.95 |  |
| unitcommit_200_100_2_mod_7 | open | no-solution |  | 32736736.16 | 32731550.87 |  |
| unitcommit_200_100_2_mod_8 | open | no-solution |  | 30598698.07 | 30593199.31 |  |
| var_con10 | open | no-solution |  | 444.213954 | 0 |  |
| var_con5 | open | no-solution |  | 278.1448937 | 0 |  |
| wastepaper5 | open | feasible | 0.007182681696 | 0.0008212036779 | 2.365036897e-05 | 0.636 % |
| wastepaper6 | open | feasible | 0.001741558343 | 0.0001263745072 | 0 | 0.162 % |
| wastewater11m2 | open | feasible | 3705.981757 | 2127.115385 | 1492.570481 | 74.2 % |
| wastewater12m2 | open | feasible | 3658.91403 | 1201.038462 | 887.5216412 | 205 % |
| wastewater13m2 | open | feasible | 196896.5567 | 1564.957983 | 1192.021608 | 1.25e+04 % |
| wastewater14m1 | open | feasible | 518.2703942 | 513.0008681 | 512.9495713 | 1.03 % |
| wastewater14m2 | open | feasible | 1983.921271 | 513.0008681 | 403.9116969 | 287 % |
| water | open | feasible | 923.7111843 | 904.9021593 | 670.3632024 | 2.08 % |
| water3 | open | not-modelled(reader-unsupported) |  | 907.0169959 | 796.4438847 |  |
| waterful2 | open | not-modelled(reader-unsupported) |  | 933.1243953 | 466.8534217 |  |
| waternd2 | open | no-solution |  | 1062685.084 | 1059307.033 |  |
| waternd_blacksburg | open | not-modelled(missing-nl) |  | 118251.0932 | 107758.746 |  |
| waternd_fossiron | open | not-modelled(missing-nl) |  | 178494.141 | 175664.3624 |  |
| waternd_fosspoly0 | open | not-modelled(missing-nl) |  | 70680507.9 | 67559218.09 |  |
| waternd_fosspoly1 | open | not-modelled(missing-nl) |  | 322922.2382 | 25578.52896 |  |
| waternd_hanoi | open | not-modelled(missing-nl) |  | 6109620.9 | 5809007.699 |  |
| waternd_modena | open | not-modelled(missing-nl) |  | 2580379.532 | 2082121.462 |  |
| waternd_pescara | open | not-modelled(missing-nl) |  | 1837440.41 | 1570815.671 |  |
| waterno1_02 | open | not-modelled(missing-nl) |  | 300.8423531 | 300.4318489 |  |
| waterno1_03 | open | not-modelled(missing-nl) |  | 402.3825104 | 398.1038027 |  |
| waterno1_04 | open | not-modelled(missing-nl) |  | 472.097085 | 457.815738 |  |
| waterno1_06 | open | not-modelled(missing-nl) |  | 687.673008 | 619.1432623 |  |
| waterno1_09 | open | not-modelled(missing-nl) |  | 1065.137445 | 892.2281587 |  |
| waterno1_12 | open | not-modelled(missing-nl) |  | 1423.113367 | 1191.824145 |  |
| waterno1_18 | open | not-modelled(missing-nl) |  | 1905.38107 | 1542.536401 |  |
| waterno1_24 | open | not-modelled(missing-nl) |  | 2578.037134 | 2063.83729 |  |
| waterno2_06 | open | no-solution |  | 282.8880374 | 108.4043018 |  |
| waterno2_09 | open | no-solution |  | 922.5952898 | 134.2240446 |  |
| waterno2_12 | open | no-solution |  | 2263.358374 | 280.5539705 |  |
| waterno2_18 | open | no-solution |  | 5269.638815 | 383.6649097 |  |
| waterno2_24 | open | no-solution |  | 7332.721691 | 474.8317637 |  |
| waters | open | not-modelled(reader-unsupported) |  | 907.0169959 | 220.0215765 |  |
| watersbp | open | not-modelled(reader-unsupported) |  | 907.0169959 | 335.9366372 |  |
| waterund01 | open | matches-bks | 86.83333142 | 86.83333333 | 86.78902985 | -2.2e-06 % |
| waterund14 | open | feasible | 332.8012248 | 329.5698382 | 329.4744403 | 0.98 % |
| waterund17 | open | feasible | 157.0965708 | 157.0944444 | 155.5355289 | 0.00135 % |
| waterund18 | open | matches-bks | 238.7333346 | 238.7333333 | 237.143172 | 5.54e-07 % |
| waterund22 | open | feasible | 328.3256009 | 323.5051042 | 323.4213724 | 1.49 % |
| waterund25 | open | feasible | 411.5122051 | 410.6353929 | 395.1435286 | 0.214 % |
| waterund27 | open | feasible | 668.4782244 | 556.6751626 | 513.7306959 | 20.1 % |
| waterund28 | open | no-solution |  | 1812.170327 | 1719.48867 |  |
| waterund32 | open | no-solution |  | 638.7167531 | 330.2692926 |  |
| waterund36 | open | feasible | 716.8695324 | 662.8070384 | 607.7760507 | 8.16 % |
| waterx | open | feasible | 927.2748223 | 909.0278626 | 807.0073387 | 2.01 % |
| waterz | open | feasible | 923.7509938 | 907.0169959 | 484.9586856 | 1.84 % |

</details>

## UC-CHPED

Pedroso et al. 2014's unit-commitment instances with the valve-point cost
`|d·sin(e·(Pmin−P))|`. The model, its fidelity audit and the SCIP references
are in [docs/benchmarks/uc-chped/README.md](benchmarks/uc-chped/README.md),
[benchmarks/uc-chped/FIDELITY.md](../benchmarks/uc-chped/FIDELITY.md) and
[benchmarks/uc-chped/pedroso/README.md](../benchmarks/uc-chped/pedroso/README.md).

Two models appear below. The source's demand row is an equality. The first
three tables solve it as one: cbls runs with `--demand-equality`. The last two,
#441's and #257's, have cbls on `demand >=`, a relaxation that could lower the
optimum (FIDELITY.md, section 1.4), while their SCIP references keep the `=`.

### cbls against SCIP at equal time and threads

Source: [#442, comment 7227877261](https://github.com/spoorendonk/cbls/issues/442#issuecomment-7227877261)
(2026-11-27), the corrected 20-row table. It replaces the table in
[comment 7225825260](https://github.com/spoorendonk/cbls/issues/442#issuecomment-7225825260).

| instance | T | budget (s) | reference | cbls PI | cbls final gap % | pwl PI | pwl final gap % | exact PI | exact final gap % |
|---|---|---|---|---|---|---|---|---|---|
| ucp13 | 1 | 10 | 11701.00 | 0.0002 [0.0002, 0.0004] | 0.00 [0.00, 0.00] | 0.0184 | 0.00 | 0.0315 | 0.00 |
| ucp13 | 3 | 30 | 38849.84 | 0.0014 [0.0014, 0.0022] | 0.00 [0.00, 0.00] | 0.0131 | 0.00 | 0.0173 | 0.00 |
| ucp13 | 6 | 60 | 91600.01 | 0.0055 [0.0043, 0.0099] | 0.32 [0.12, 0.61] | 0.0092 | 0.00 | 0.0168 | 0.14 |
| ucp13 | 12 | 120 | 232326.66 | 0.0056 [0.0038, 0.0059] | 0.42 [0.23, 0.46] | 0.0083 | 0.00 | 0.1294 | 0.37 |
| ucp13 | 24 | 300 | 465933.05 | 0.0085 [0.0072, 0.0091] | 0.60 [0.57, 0.74] | 0.0123 | 0.00 | 0.5273 | 0.57 |
| ucp40 | 1 | 10 | 55644.79 | 0.0040 [0.0023, 0.0042] | 0.00 [0.00, 0.04] | 0.0313 | 0.00 | 0.0523 | 0.00 † |
| ucp40 | 3 | 30 | 178517.92 | 0.0214 [0.0189, 0.0223] | 1.80 [1.41, 1.88] | **0.0210** | 0.00 | 0.0677 | 0.08 |
| ucp40 | 6 | 60 | 416606.00 | 0.0118 [0.0109, 0.0138] | 0.96 [0.96, 1.07] | 0.0248 | 0.05 | 0.1503 | 0.36 † |
| ucp40 | 12 | 120 | 1113255.99 | 0.0177 [0.0152, 0.0181] | 1.30 [1.18, 1.41] | 0.0181 | 0.00 | 0.3688 | 10.04 † |
| ucp40 | 24 | 300 | 2238504.00 | 0.0307 [0.0291, 0.0344] | 2.46 [2.38, 2.79] | 0.0364 | 0.00 | 0.4129 | 8.12 |
| ucp100 | 1 | 10 | 140000.18 | 0.0224 [0.0212, 0.0265] | 1.13 [0.81, 1.37] | 0.0484 | 0.00 | 0.2510 | 0.10 † |
| ucp100 | 3 | 30 | 449468.07 | 0.0262 [0.0231, 0.0272] | 2.39 [2.18, 2.57] | 0.0389 | 0.00 | 0.3770 | 0.45 † |
| ucp100 | 6 | 60 | 1053010.60 | 0.0155 [0.0136, 0.0162] | 1.24 [1.13, 1.31] | 0.0677 | 0.00 | 0.6931 | 0.61 |
| ucp100 | 12 | 120 | 2824715.46 | 0.0299 [0.0251, 0.0304] | 2.51 [2.06, 2.84] | 0.2276 | 0.00 | 0.9962 | 0.89 |
| ucp100 | 24 | 300 | 5678343.24 | 0.0516 [0.0459, 0.0581] | 4.29 [4.28, 5.09] | 0.4058 | 0.00 | 0.6230 | 0.88 |
| ucp200 | 1 | 10 | 277781.64 | 0.0581 [0.0553, 0.0591] | 4.57 [3.53, 4.58] | 0.1606 | 0.00 | 0.1580 | 0.41 † |
| ucp200 | 3 | 30 | 890277.18 | 0.0351 [0.0349, 0.0361] | 3.05 [3.02, 3.39] | 0.1366 | 0.00 | 0.5173 | 1.75 |
| ucp200 | 6 | 60 | 2079415.51 | 0.0226 [0.0219, 0.0282] | 1.66 [1.32, 1.93] | 0.2307 | 0.00 | 0.9627 | 17.49 † |
| ucp200 | 12 | 120 | 5554265.66 | 0.0425 [0.0414, 0.0456] | 3.36 [3.16, 3.58] | 0.7539 | 0.00 | 0.6402 | 0.74 |
| ucp200 | 24 | 300 | 11153209.99 | 0.0699 [0.0673, 0.0703] | 6.04 [5.71, 6.78] | 0.8342 | 0.00 | 1.0674 | 0.77 |

- rule: benchmarks/common/primal_integral.py's MIPfeas primal gap |z - ref| / max(|z|, |ref|), 2 before the first incumbent, and PI its time average over [0, budget]. cbls is median [min, max] over 3 seeds; its PI is over every seed, its final gap over the seeds that solved it. pwl and exact are one run each. The reference is the lowest of Pedroso's UB, every final objective and every trajectory point, over all seeds and methods.
- cbls: its `--trace`, seconds from the solve's start. SCIP: the driver's running best, wall seconds from the method's start (model build and file I/O included). exact's points are SCIP's objective, an upper bound on that incumbent's true cost; its final gap uses the true cost.
- † marks the 7 exact rows whose best came after the budget (SCIP's overshoot). Their final gap uses the last in-budget SCIP objective.

**Read-out:**
- **Primal integral:** cbls is lowest on 19 of 20 rows. It is not lowest on ucp40 T=3: pwl's 0.0210 is below cbls's median 0.0214 and inside cbls's seed range [0.0189, 0.0223].
- **Final gap:** pwl is 0.00 on 19 rows and 0.05 % on ucp40 T=6. exact is best on none. ucp13 T=1/T=3 and ucp40 T=1 are 0.00 for all three methods.
- At equal time and threads, cbls gets good solutions fastest, and Pedroso's pwl ends best.

Run record:
- **Arms**, all at 16 threads and the same wall budget per horizon (10/30/60/120/300 s), on simon-workstation cores 0-15, one row per `xk lock`, nothing else timed on those cores:
  - **cbls:** [#445](https://github.com/spoorendonk/cbls/issues/445)'s baseline runs, engine at main `2eb27278`, `--demand-equality --trace`, seeds 1, 2 and 42. No commit has touched `src/` or `include/` between `02c78ad` and `2eb27278`.
  - **pwl:** Pedroso et al.'s Algorithm 1 ported to SCIP, `ucp_valve_scip.py --backend driver --threads 16`, rerun at `13a793d`: points are stamped when SCIP finds them, and `ub` counts only in-budget finds.
  - **exact:** SCIP's MINLP with the true valve-point term, run at `f252ac51` and re-derived at `13a793d`'s in-budget rule from those records.
  - **SCIP:** 11 preview at pinned commit `124040dbc` (unreleased master, opportunistic concurrent mode), through the `cbls_scip_traj` driver. SCIP's own concurrent default is 10 threads; these runs use 16 (Simon, [#442, comment 7220251785](https://github.com/spoorendonk/cbls/issues/442#issuecomment-7220251785)).
- **Scored by:** `benchmarks/uc-chped/score.py --roster <20 rows> --scip pwl-small --scip exact-small --trace ...`.
- **Files:** `~/data/cbls-gefjon/442-step3/` on simon-workstation (`pwl-13a793d.jsonl`, `exact-rederived.jsonl`, `score2/small.md`).
- **Not scored yet: the 4 long rows** (ucp100 and ucp200 at T=48 and T=168, 600 s). Three of their SCIP runs have no incumbent within 600 s (pwl ucp200 T=168, exact ucp100 T=168, exact ucp200 T=168), and pwl ucp200 T=48 has no record yet. They are open on [#442](https://github.com/spoorendonk/cbls/issues/442).

### Best known bounds

Source: [#443, comment 7192831491](https://github.com/spoorendonk/cbls/issues/443#issuecomment-7192831491)
(2026-11-26). The best UB and LB known for each row, from dedicated methods
run for this purpose, not from a timed comparison.

▲ = strictly better than #441's best (lowest UB over pwl/exact; highest LB that is a bound: exact, or pwl on ucp13 only), = = equal to the cent. #441's pwl LB is listed separately because on ucp40/100/200 it is not a bound (units 26–28 break `c <= e*f^2/2`, #441). Pedroso's Table 2 comes from the same pwl method, so its ucp40 LBs are not proven bounds either. Pedroso columns use `data.py`'s pinned integers, and the paper's two decimals for the three optima.

| Row | Best UB | Best LB (SCIP) | Gap % | #441 best UB | #441 best valid LB | #441 pwl LB (✗ = not a bound) | Pedroso UB | Pedroso LB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ucp13-1p | 11,701.28 = | 11,701.28 = | 0.00001 | 11,701.28 | 11,701.28 | 11,701.28 | 11,701.28 | 11,701.28 |
| ucp13-3p | 38,849.84 = | 38,849.84 = | 0.00001 | 38,849.84 | 38,849.84 | 38,849.84 | 38,849.84 | 38,849.84 |
| ucp13-6p | 91,568.93 ▲ | 91,568.92 ▲ | 0.00001 | 91,590.44 | 91,270.15 | 91,270.15 | 91,784.00 | 91,406.00 |
| ucp13-12p | 232,177.23 ▲ | 232,048.91 ▲ | 0.05527 | 232,315.20 | 231,580.50 | 231,580.50 | 232,537.00 | 231,587.00 |
| ucp13-24p | 465,530.93 ▲ | 464,981.37 ▲ | 0.11805 | 465,849.14 | 464,144.42 | 464,144.42 | 466,187.00 | 464,053.00 |
| ucp40-1p | 55,644.79 = | 55,644.79 = | 0.00001 | 55,644.79 | 55,644.79 | 55,644.79 ✗ | 55,644.79 | 55,644.79 |
| ucp40-3p | 178,445.07 ▲ | 178,445.05 ▲ | 0.00001 | 178,472.16 | 178,041.58 | 178,386.44 ✗ | 178,547.00 | 178,396.00 |
| ucp40-6p | 416,303.47 ▲ | 416,206.62 ▲ | 0.02326 | 416,699.43 | 415,325.75 | 416,063.52 ✗ | 416,606.00 | 416,108.00 |
| ucp40-12p | 1,112,697.66 ▲ | 1,112,411.53 ▲ | 0.02572 | 1,113,361.51 | 1,109,292.49 | 1,112,317.63 ✗ | 1,113,801.00 | 1,112,371.00 |
| ucp40-24p | 2,236,705.09 ▲ | 2,235,927.45 ▲ | 0.03477 | 2,238,724.94 | 2,231,030.97 | 2,235,956.00 ✗ | 2,238,504.00 | 2,235,971.00 |
| ucp100-1p | 139,802.09 = | 139,775.96 ▲ | 0.01869 | 139,802.09 | 139,639.91 | 139,802.09 ✗ | – | – |
| ucp100-3p | 449,135.56 ▲ | 448,938.89 ▲ | 0.04379 | 449,258.30 | 448,152.89 | 448,928.30 ✗ | – | – |
| ucp100-6p | 1,052,830.13 ▲ | 1,052,505.46 ▲ | 0.03084 | 1,053,252.46 | 1,050,741.16 | 1,052,506.74 ✗ | – | – |
| ucp100-12p | 2,824,119.63 ▲ | 2,823,712.99 | 0.01440 | 2,824,746.44 | – | 2,823,667.82 ✗ | – | – |
| ucp100-24p | 5,676,332.99 ▲ | 5,675,078.43 | 0.02210 | 5,678,575.25 | – | 5,675,095.65 ✗ | – | – |
| ucp100-48p | 11,049,341.43 ▲ | 11,044,543.47 ▲ | 0.04342 | 11,054,462.60 | 10,836,484.21 | 11,045,035.16 ✗ | – | – |
| ucp100-168p | 37,966,440.39 ▲ | 37,945,629.72 ▲ | 0.05481 | 38,051,403.60 | 37,249,952.29 | 37,943,520.26 ✗ | – | – |
| ucp200-1p | 277,649.58 = | 277,605.62 ▲ | 0.01583 | 277,649.58 | 277,388.47 | 277,612.81 ✗ | – | – |
| ucp200-3p | 890,009.30 ▲ | 889,841.21 ▲ | 0.01889 | 890,277.18 | 888,106.56 | 889,824.75 ✗ | – | – |
| ucp200-6p | 2,078,841.66 ▲ | 2,078,456.82 | 0.01851 | 2,079,424.10 | – | 2,078,435.20 ✗ | – | – |
| ucp200-12p | 5,553,011.57 ▲ | 5,552,607.86 | 0.00727 | 5,554,007.95 | – | 5,552,467.41 ✗ | – | – |
| ucp200-24p | 11,150,910.56 ▲ | 11,149,711.95 | 0.01075 | 11,153,060.42 | – | 11,149,808.51 ✗ | – | – |
| ucp200-48p | 21,697,353.94 ▲ | 21,689,590.98 | 0.03578 | 21,706,712.86 | – | 21,689,167.76 ✗ | – | – |
| ucp200-168p | 74,519,886.78 ▲ | 74,491,273.76 ▲ | 0.03840 | 82,336,484.29 | 122,548.65 | 74,476,585.60 ✗ | – | – |

- **Two more rows proven optimal (to SCIP tolerance):** ucp13-6p (91,568.93) and ucp40-3p (178,445.07), which neither Pedroso's Table 2 nor #441 closed. The three rows already known optimal are matched to the cent.
- **UB:** at or below #441's best on all 24 rows. It ties on ucp13-1p/3p, ucp40-1p, ucp100-1p and ucp200-1p, and is lower on the other 19. Every ucp13/ucp40 UB lies inside Pedroso's [LB, UB] and is below his UB, except the three known optima, which it equals.
- **LB:** at or above every *valid* reference LB on every row that has one; on 6 constructed rows #441 has no valid LB at all, as its exact runs crashed. It is below some LBs that are not proven bounds, by 7–97: #441's pwl LB on ucp100-1p, -24p, ucp200-1p, -24p, and both pwl and Pedroso on ucp40-24p. On ucp100-1p and ucp200-1p, pwl converged to LB = UB at our UB, which suggests our UB is optimal there but does not prove it.
  - That list misses two rows, by this table: the best LB is also below #441's pwl LB, not a bound, on ucp100-6p (by 1.28) and ucp100-48p (by 492).

Run record:
- **UB:** each solution file checked against Pedroso's model by `verify.py`, written once and not changed (equality demand within 1e-6 relative, reserve, min up/down from the initial state, true valve-point cost and start-up costs).
- **LB:** each a SCIP 10 dual bound (1 thread, feastol 1e-7, gap 1e-6) on a model that relaxes Pedroso's problem, as the source comment's Methods section argues. The bounds are valid up to SCIP's tolerances.
- **Run:** 2026-11-26 09:02Z-13:02Z on simon-workstation; the source comment gives the method and time behind every bound.
- **Code and files:** the methods' code is not in the repository. It was never committed: it is archived as `refs/wcp-archive/20261126T130420Z/uncommitted` in its session's clone and in that archive's bundle on simon-workstation ([#443, comment 7192840644](https://github.com/spoorendonk/cbls/issues/443#issuecomment-7192840644)). The same code, with the run's files, is in `~/data/cbls-idun/` there (`code/`, `code.patch`, `results.jsonl`, `best_final.jsonl`, solutions and SCIP logs).

### cbls against the best known bounds

**Computed for this page** from posted numbers, not by a scorer: cbls's
median objective over seeds 1, 2 and 42 from [#442, comment
7225825260](https://github.com/spoorendonk/cbls/issues/442#issuecomment-7225825260)'s
LB/UB table (the same #445 baseline runs as the table above), against the best
known bounds of the previous table. Gap = 100 · (cbls − bound) / bound. The
cbls medians are posted to one decimal, so a gap below 0.005 % is not resolved.

| instance | T | cbls objective (median) | best known UB | best known LB | cbls gap to UB % | cbls gap to LB % |
|---|---|---|---|---|---|---|
| ucp13 | 1 | 11701.3 | 11701.28 | 11701.28 | 0.00 | 0.00 |
| ucp13 | 3 | 38849.8 | 38849.84 | 38849.84 | 0.00 | 0.00 |
| ucp13 | 6 | 91895.3 | 91568.93 | 91568.92 | 0.36 | 0.36 |
| ucp13 | 12 | 233298.0 | 232177.23 | 232048.91 | 0.48 | 0.54 |
| ucp13 | 24 | 468757.1 | 465530.93 | 464981.37 | 0.69 | 0.81 |
| ucp40 | 1 | 55644.8 | 55644.79 | 55644.79 | 0.00 | 0.00 |
| ucp40 | 3 | 181786.0 | 178445.07 | 178445.05 | 1.87 | 1.87 |
| ucp40 | 6 | 420662.9 | 416303.47 | 416206.62 | 1.05 | 1.07 |
| ucp40 | 12 | 1127940.9 | 1112697.66 | 1112411.53 | 1.37 | 1.40 |
| ucp40 | 24 | 2294894.9 | 2236705.09 | 2235927.45 | 2.60 | 2.64 |
| ucp100 | 1 | 141603.8 | 139802.09 | 139775.96 | 1.29 | 1.31 |
| ucp100 | 3 | 460483.1 | 449135.56 | 448938.89 | 2.53 | 2.57 |
| ucp100 | 6 | 1066260.6 | 1052830.13 | 1052505.46 | 1.28 | 1.31 |
| ucp100 | 12 | 2897429.1 | 2824119.63 | 2823712.99 | 2.60 | 2.61 |
| ucp100 | 24 | 5932793.3 | 5676332.99 | 5675078.43 | 4.52 | 4.54 |
| ucp200 | 1 | 291075.2 | 277649.58 | 277605.62 | 4.84 | 4.85 |
| ucp200 | 3 | 918241.7 | 890009.30 | 889841.21 | 3.17 | 3.19 |
| ucp200 | 6 | 2114433.1 | 2078841.66 | 2078456.82 | 1.71 | 1.73 |
| ucp200 | 12 | 5747520.2 | 5553011.57 | 5552607.86 | 3.50 | 3.51 |
| ucp200 | 24 | 11869941.0 | 11150910.56 | 11149711.95 | 6.45 | 6.46 |

### SCIP references at one hour (cbls on `>=`)

Source: [#441, comment 7177744991](https://github.com/spoorendonk/cbls/issues/441#issuecomment-7177744991)
(2026-11-25). SCIP at Pedroso's one-hour budget. The cbls column is #257's
three seeds on `demand >=` (next table), scored beside it; it is not an
equal-time comparison.

| instance | T | cbls objective (median) | pwl LB | pwl UB | pwl gap % | exact LB | exact UB | exact gap % | cbls gap to best SCIP LB %: median [min, max] | cbls gap to best SCIP UB %: median [min, max] |
|---|---|---|---|---|---|---|---|---|---|---|
| ucp13 | 1 | 11701.3 | 11701.28 | 11701.28 | 0.00 | 11701.28 | 11701.28 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| ucp13 | 3 | 38849.8 | 38849.84 | 38849.84 | 0.00 | 38849.84 | 38849.84 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| ucp13 | 6 | 92957.8 | 91270.15 | 91600.01 | 0.36 | 91190.35 | 91590.44 | 0.44 | 1.85 [1.84, 1.94] | 1.49 [1.49, 1.59] |
| ucp13 | 12 | 233633.3 | 231580.50 | 232315.20 | 0.32 | 230621.18 | 233088.07 | 1.07 | 0.89 [0.72, 0.90] | 0.57 [0.40, 0.58] |
| ucp13 | 24 | 469201.5 | 464144.42 | 465849.14 | 0.37 | 462271.29 | 467479.38 | 1.13 | 1.09 [1.07, 1.09] | 0.72 [0.70, 0.72] |
| ucp40 | 1 | 55644.8 | 55644.79* | 55644.79 | - | 55644.79 | 55644.79 | 0.00 | 0.00 [0.00, 0.06] | 0.00 [0.00, 0.06] |
| ucp40 | 3 | 181437.4 | 178386.44* | 178521.74 | - | 178041.58 | 178472.16 | 0.24 | 1.91 [1.51, 1.97] | 1.66 [1.26, 1.73] |
| ucp40 | 6 | 419638.9 | 416063.52* | 416699.43 | - | 415325.75 | 416810.48 | 0.36 | 1.04 [0.96, 1.16] | 0.71 [0.63, 0.83] |
| ucp40 | 12 | 1125206.9 | 1112317.63* | 1113361.51 | - | 1109292.49 | 1114661.41 | 0.48 | 1.43 [1.38, 1.61] | 1.06 [1.01, 1.24] |
| ucp40 | 24 | 2300356.6 | 2235956.00* | 2238724.94 | - | 2231030.97 | 2247084.55 | 0.72 | 3.11 [2.93, 3.21] | 2.75 [2.58, 2.85] |
| ucp100 | 1 | 140957.3 | 139802.09* | 139802.09 | - | 139639.91 | 139802.09 | 0.12 | 0.94 [0.90, 1.00] | 0.83 [0.78, 0.89] |
| ucp100 | 3 | 459622.8 | 448928.30* | 449258.30 | - | 448152.89 | 449477.82 | 0.30 | 2.56 [2.35, 2.73] | 2.31 [2.09, 2.48] |
| ucp100 | 6 | 1065568.0 | 1052506.74* | 1053252.46 | - | 1050741.16 | 1054090.12 | 0.32 | 1.41 [1.28, 1.55] | 1.17 [1.04, 1.30] |
| ucp100 | 12 | 2903736.3 | 2823667.82* | 2824746.44 | - | SCIP crashed (reproduced) | - | - | - | 2.80 [2.54, 2.89] |
| ucp100 | 24 | 5937300.7 | 5675095.65* | 5678575.25 | - | SCIP crashed (reproduced) | - | - | - | 4.56 [4.33, 4.68] |
| ucp100-48p | 48 | 11664698.4 | 11045035.16* | 11054462.60 | - | 10836484.21 | - | - | 7.64 [7.33, 7.67] | 5.52 [5.21, 5.54] |
| ucp100-168p | 168 | 40788216.7 | 37943520.26* | 38051403.60 | - | 37249952.29 | - | - | 9.50 [9.24, 9.83] | 7.19 [6.94, 7.51] |
| ucp200 | 1 | 287562.9 | 277612.81* | 277649.58 | - | 277388.47 | 277700.05 | 0.11 | 3.67 [2.92, 4.12] | 3.57 [2.82, 4.03] |
| ucp200 | 3 | 916644.6 | 889824.75* | 890277.18 | - | 888106.56 | 891795.28 | 0.42 | 3.21 [3.20, 3.55] | 2.96 [2.95, 3.29] |
| ucp200 | 6 | 2109094.3 | 2078435.20* | 2079424.10 | - | SCIP crashed (reproduced) | - | - | - | 1.43 [1.34, 1.72] |
| ucp200 | 12 | 5742321.5 | 5552467.41* | 5554007.95 | - | SCIP crashed (reproduced) | - | - | - | 3.39 [2.65, 3.95] |
| ucp200 | 24 | 11798460.1 | 11149808.51* | 11153060.42 | - | SCIP crashed (reproduced) | - | - | - | 5.79 [5.41, 5.90] |
| ucp200-48p | 48 | 23138729.1 | 21689167.76* | 21706712.86 | - | SCIP crashed (reproduced) | - | - | - | 6.60 [6.59, 6.88] |
| ucp200-168p | 168 | 80255310.0 | 74476585.60* | 82336484.29 | - | 122548.65 | - | - | 65388.53 [65249.06, 65490.27] | -2.53 [-2.74, -2.38] |

- SCIP 10.0; budget (process CPU s): pwl 3600; exact 3600 (ucp_valve_scip.py records). Each SCIP solve is limited in wall time to what is left, as ucp_valve.py limited Gurobi: SCIP's CPU clock is a times() syscall costly enough to overrun the budget by a third (#441).
- rule: SCIP gap = 100 * (UB - LB) / LB per method; cbls's gaps as above, against the highest LB that is a bound and the lowest UB over the methods. `pwl` is Pedroso et al.'s Algorithm 1 ported to SCIP, `exact` SCIP's MINLP with the true valve-point term. Both solve the source's model (demand =); cbls's has demand >= (FIDELITY.md section 1.4).
- `*`: pwl LB not a bound, as some unit breaks `c <= e*f^2/2`. That is every ucp40, ucp100 and ucp200 row (19).
- no SCIP solution: 3 (exact on ucp100-48p, ucp100-168p and ucp200-168p: timelimit).
- **SCIP crashed (reproduced), 6 exact rows:** each aborted inside `m.optimize()` with a glibc heap check, and again when rerun once under `gdb`. All six backtraces run through an NLP heuristic into Ipopt, MUMPS and METIS in the `libscip` that PySCIPOpt 6.3.0's wheel bundles (scipopt/scip#211).
- pwl completed fewer refinement iterations in the hour than Pedroso's Table 2 reports for Gurobi 5.0.1 (for example 22 against 30 on ucp40 T=3). The misses against Pedroso's [LB, UB] come from that, not from the port.

Run record: SCIP 10.0 and PySCIPOpt 6.3.0, run at repository commit `87eb3390`,
with `scip_clock: wall` in every record; at most 14 solves at once, one per
core, `MemoryMax=8G`, a 3600 s budget and a 3900 s wall stop, which no record
hit. Scored with main's `score.py` over #257's three seeds and the 24 pwl and
18 exact records; the six crash marks are a hand edit.

### cbls against Pedroso's published bounds (cbls on `>=`)

Source: [#257, comment 7172717156](https://github.com/spoorendonk/cbls/issues/257#issuecomment-7172717156)
(2026-11-25). For the source's `=` model, the tables above supersede it
([#442](https://github.com/spoorendonk/cbls/issues/442)).

| instance | T | budget (s) | solved | objective (median) | gap to LB %: median [min, max] | gap to UB %: median [min, max] | Pedroso gap % |
|---|---|---|---|---|---|---|---|
| ucp13 | 1 | 10 | 3/3 | 11701.3 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 |
| ucp13 | 3 | 30 | 3/3 | 38849.8 | -0.00 [-0.00, -0.00] | -0.00 [-0.00, -0.00] | 0.00 |
| ucp13 | 6 | 60 | 3/3 | 92957.8 | 1.70 [1.69, 1.79] | 1.28 [1.27, 1.37] | 0.41 |
| ucp13 | 12 | 120 | 3/3 | 233633.3 | 0.88 [0.71, 0.89] | 0.47 [0.30, 0.48] | 0.41 |
| ucp13 | 24 | 300 | 3/3 | 469201.5 | 1.11 [1.09, 1.11] | 0.65 [0.62, 0.65] | 0.46 |
| ucp40 | 1 | 10 | 3/3 | 55644.8 | -0.00 [-0.00, 0.06] | -0.00 [-0.00, 0.06] | 0.00 |
| ucp40 | 3 | 30 | 3/3 | 181437.4 | 1.70 [1.30, 1.77] | 1.62 [1.22, 1.68] | 0.08 |
| ucp40 | 6 | 60 | 3/3 | 419638.9 | 0.85 [0.77, 0.97] | 0.73 [0.65, 0.85] | 0.12 |
| ucp40 | 12 | 120 | 3/3 | 1125206.9 | 1.15 [1.10, 1.33] | 1.02 [0.97, 1.20] | 0.13 |
| ucp40 | 24 | 300 | 3/3 | 2300356.6 | 2.88 [2.71, 2.98] | 2.76 [2.59, 2.86] | 0.11 |
| ucp100 | 1 | 10 | 3/3 | 140957.3 | - | - | - |
| ucp100 | 3 | 30 | 3/3 | 459622.8 | - | - | - |
| ucp100 | 6 | 60 | 3/3 | 1065568.0 | - | - | - |
| ucp100 | 12 | 120 | 3/3 | 2903736.3 | - | - | - |
| ucp100 | 24 | 300 | 3/3 | 5937300.7 | - | - | - |
| ucp100-48p | 48 | 600 | 3/3 | 11664698.4 | - | - | - |
| ucp100-168p | 168 | 600 | 3/3 | 40788216.7 | - | - | - |
| ucp200 | 1 | 10 | 3/3 | 287562.9 | - | - | - |
| ucp200 | 3 | 30 | 3/3 | 916644.6 | - | - | - |
| ucp200 | 6 | 60 | 3/3 | 2109094.3 | - | - | - |
| ucp200 | 12 | 120 | 3/3 | 5742321.5 | - | - | - |
| ucp200 | 24 | 300 | 3/3 | 11798460.1 | - | - | - |
| ucp200-48p | 48 | 600 | 3/3 | 23138729.1 | - | - | - |
| ucp200-168p | 168 | 600 | 3/3 | 80255310.0 | - | - | - |

- engine commit: 02c78ad (commit_sha column)
- seeds: 1, 2, 3 (seed column)
- cbls threads: 16 (search_config column)
- search config: float_hook=on;lns=on;lns_interval=3;compound_moves=off;novelty_prob=0.5;unproductive_iters=300;perturbation_period=100;max_iterations=0;time_limit=on;threads=16 (search_config column)
- feasibility tolerance: 1e-06 (feas_tol column)
- rule: gap = 100 * (objective - bound) / bound over the seeds that solved it, and a seed that did not is counted in `solved` and named below. UB below 0 beats Pedroso's one-hour incumbent. Pedroso's gap is a cited result (Pedroso et al. 2014, Table 2), not a measurement. `-`: no published bound, or no seed solved it.
- no solution: 0
- The five runs below Pedroso's LB (ucp13 T=3, all three seeds; ucp40 T=1, seeds 1 and 3) are rounding, not a fault: `data.py` pins Table 2's bounds as integers (38850, 55645), and cbls finds the true optima 38849.84 and 55644.79 that the paper prints to two decimals.

Run record: engine frozen at `02c78ade`, with FPR, LNS repair and DAG
propagation off. Seeds 1, 2 and 3 ran one after another, 2026-11-25
16:00Z-19:44Z on simon-workstation, each its own capped run unit under `xk
lock`, `taskset -c 0-15`, 16 threads. All 72 runs solved, were feasible and
passed the runner's verifier. Files: `table.md` beside the seeds'
`comparison.{csv,log}` in the session's `257-uc-chped/` data directory.

## CVRP

No table: the full CVRPLIB campaign
([#204](https://github.com/spoorendonk/cbls/issues/204)) has not been run, so
there is no `List` result to publish. The only run so far is a 5-instance,
30 s, single-thread smoke at engine `3e54d34`
([#204, comment 6506019842](https://github.com/spoorendonk/cbls/issues/204#issuecomment-6506019842)),
a check of the pipeline, not a result.
