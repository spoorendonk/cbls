"""MIPfeas baseline: OR-Tools CP-SAT restricted to its violation-based local search.

This is the reference implementation of the algorithm CBLS reimplements — the
`fj` (Feasibility Jump) and `ls` (ViolationLS) workers described in Davies, Didier
& Perron, *ViolationLS: Constraint-Based Local Search in CP-SAT*, CPAIOR 2024
(https://link.springer.com/chapter/10.1007/978-3-031-60597-0_16). Comparing against
them answers "is the reimplementation competent"; comparing against CP-SAT's default
portfolio would be a different (and rejected) question — see epic #87.

Configuration notes, all established empirically against ortools 9.15:

* `filter_subsolvers` is the only parameter that accepts `fj`/`ls`. `subsolvers` and
  `ignore_subsolvers` validate against full-problem subsolver names only and reject
  both, so they cannot express "LS only".
* `ls` alone never bootstraps a first solution (status stays UNKNOWN); it needs `fj`.
  That pairing mirrors CBLS, which runs Feasibility Jump to reach feasibility and then
  ViolationLS with the objective folded in as `obj <= bound`.
* `num_workers: 1` runs both of them — the log reports `1 first solution subsolver: [fj]`
  and `1 interleaved subsolver: [ls]`. One worker is therefore the default here, so the
  baseline gets the same single thread CBLS does. Raising it multiplies *both* workers
  (`num_workers: 2` gives `fj(2)` and `ls(2)`, ~2x the CPU in the same wall time), so
  the count is recorded per result rather than assumed.
* Presolve is left at its default (on), i.e. the worker as it actually ships.
* `ModelSolver.log_callback` raises TypeError on this ortools/Python combination
  (pybind11 std::function caster unregistered), so the log is captured by redirecting
  fd 1 around the solve call. That works precisely because this script runs one
  instance per process.

With --solution-dir, a feasible run also writes its solution vector as
`<instance>.sol`, for `verify_solution.py` to check against the original
instance file with a reader that is not OR-Tools' (issue #138). The objective
this script reports is otherwise taken entirely on trust.

Every fact above is empirical, and the dependency range it was established
against is pinned in `pyproject.toml` (`ortools>=9.7,<9.16`). A release that renames a
subsolver flag or reformats a log line does not crash -- it silently produces a
degraded or empty baseline across the whole roster, visible only at scoring time.
`--preflight` is the cheap check for exactly that: one tiny in-memory model, no
instance and no network, asserting that the restriction is in force and that the
log still carries the lines the trace is recovered from, and naming which of the
two broke. The run driver runs it once before dispatching any job.

Usage:
    python cpsat_solve.py --preflight
    python cpsat_solve.py --instance pk1 --out-dir results/cpsat --budget 600
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import resource
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from ortools.linear_solver.python import model_builder

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

#: CP-SAT logs each improving solution as e.g.
#:   `#12      3.30s best:6908.97 next:[5726.32,6908.97] ls_restart_compound(...)`
#: Bound-only lines (`#Bound`) and model lines (`#Model`) do not match.
SOLUTION_LINE = re.compile(r"^#(\d+)\s+([0-9.]+)s\s+best:(-?[0-9.eE+-]+)\b")

#: CP-SAT announces which subsolvers a filtered run actually started, e.g.
#:   `1 first solution subsolver: [fj]`
#:   `1 interleaved subsolver: [ls]`
#: This is the only place the *effect* of `filter_subsolvers` is observable, so the
#: preflight reads the restriction off it rather than trusting that the parameter
#: was accepted. At higher worker counts a name carries a multiplicity, `fj(2)`.
#:
#: The role is captured rather than whitelisted. A whitelist misses the single
#: announcement that means the restriction is gone -- an unrestricted CP-SAT run
#: says `full problem subsolvers`, not `full` -- and a line that does not match is
#: a silent pass, which is the failure mode this whole check exists to close.
SUBSOLVER_LINE = re.compile(r"^\d+\s+([a-z][a-z ]*?)\s+subsolvers?:\s*\[(.*)\]\s*$")

#: `Starting search at 0.00s with 1 workers.`
SEARCH_WORKERS_LINE = re.compile(r"^Starting search at [0-9.]+s with (\d+) workers?\.?$")

DEFAULT_WORKERS = 1

#: The OR-Tools range every parameter and log-format fact in this file was
#: established against, mirrored by the `ortools` bound in `pyproject.toml`. The
#: bound is what stops a resolver silently installing a release this parsing was
#: never checked on; the preflight is what catches one installed anyway.
SUPPORTED_ORTOOLS_RANGE = ">=9.7,<9.16"

#: The two things the preflight can find broken, named in the failure so a run
#: that refuses to start says which. The split is by *what moved*: the shape of a
#: line the harness reads is the log format; what the line says is the restriction.
WORKER_RESTRICTION_CHECK = "worker restriction"
LOG_FORMAT_CHECK = "log format"

#: Which subsolvers the restriction is supposed to leave running, by announcement
#: role. `ls` alone never bootstraps a first solution, so both are required and
#: anything else running means the filter stopped expressing "LS only".
EXPECTED_SUBSOLVERS: dict[str, frozenset[str]] = {
    "first solution": frozenset({"fj"}),
    "interleaved": frozenset({"ls"}),
}

#: Roles whose contents are not the restriction: helpers are bookkeeping agents and
#: `ignored` is the complement of the filter, i.e. the evidence it is working.
UNRESTRICTED_ROLES = frozenset({"helper", "ignored"})

#: Seconds the preflight solve is given. It has to reach a first solution and log
#: at least one improving line on a model that presolve cannot crack; it does that
#: in milliseconds, and the cap is what keeps a broken configuration from hanging
#: the check it exists to make fast.
PREFLIGHT_BUDGET_SECONDS = 2.0


def build_parameters(workers: int, seed: int) -> str:
    """CP-SAT parameter string restricting the solve to the fj + ls workers.

    The budget is not set here: `set_time_limit_in_seconds` already populates
    `max_time_in_seconds`, and this string merges on top of it rather than
    replacing it, so stating the limit twice would only invite the two to drift.
    """
    return (
        f"num_workers:{workers},"
        f"num_violation_ls:1,"
        f"filter_subsolvers:'fj',"
        f"filter_subsolvers:'ls',"
        f"random_seed:{seed},"
        f"log_search_progress:true"
    )


def parse_trace(log_text: str) -> list[tuple[float, float]]:
    """Extract (seconds, objective) for each improving solution in a CP-SAT log."""
    return _parse_lines(log_text.splitlines())


def parse_trace_file(path: Path) -> list[tuple[float, float]]:
    """Same, streaming the file rather than loading it.

    A 600s run on an instance that keeps improving writes tens of thousands of log
    lines (mas76 manages 3.3 MB in 5s), and this job runs under an address-space cap
    alongside a multi-GB model.
    """
    with open(path) as fh:
        return _parse_lines(fh)


def _parse_lines(lines: Iterable[str]) -> list[tuple[float, float]]:
    trace: list[tuple[float, float]] = []
    for line in lines:
        match = SOLUTION_LINE.match(line.strip())
        if match is None:
            continue
        trace.append((float(match.group(2)), float(match.group(3))))
    return trace


class PreflightFailure(NamedTuple):
    """One thing the preflight found broken, and which of the two checks found it."""

    #: `WORKER_RESTRICTION_CHECK` or `LOG_FORMAT_CHECK`.
    check: str
    message: str


def parse_subsolvers(log_text: str) -> dict[str, frozenset[str]]:
    """Role -> the subsolver names CP-SAT announced for it, multiplicities stripped."""
    found: dict[str, frozenset[str]] = {}
    for raw in log_text.splitlines():
        match = SUBSOLVER_LINE.match(raw.strip())
        if match is None:
            continue
        names = {
            re.sub(r"\(\d+\)$", "", name.strip())
            for name in match.group(2).split(",")
            if name.strip()
        }
        found[match.group(1)] = frozenset(names)
    return found


def _announcement_failures(announced: dict[str, frozenset[str]]) -> list[PreflightFailure]:
    """What the subsolver announcement block says about the restriction.

    Split out of `check_preflight_log` because it is one question with its own
    attribution rule, and because that rule is the part most likely to need
    revisiting when a release moves the block.
    """
    failures: list[PreflightFailure] = []
    block_parsed = bool(announced)
    # BOTH expected roles missing while other roles parsed is the case the
    # shape/content split cannot decide. A real unrestricted run still announces
    # `first solution` -- captured against live 9.15, which logs
    # `2 first solution subsolvers: [fj, fs_random_no_lp]` -- so losing both roles
    # at once is as likely to be the role LABELS being renamed as the restriction
    # being gone. Naming one check exonerates the other, and sends the reader to
    # the wrong place; criterion 3 asks which broke, and the honest answer here is
    # that the log cannot say. So both are named and the ambiguity is stated.
    both_absent = all(role not in announced for role in EXPECTED_SUBSOLVERS)
    ambiguous = block_parsed and both_absent
    for role, expected in EXPECTED_SUBSOLVERS.items():
        if role not in announced:
            failures.append(
                PreflightFailure(
                    WORKER_RESTRICTION_CHECK if block_parsed else LOG_FORMAT_CHECK,
                    f"the `N {role} subsolver: [...]` announcement is absent from the log. "
                    + (
                        f"Other subsolver roles were announced ({sorted(announced)}), so the "
                        "log format is intact and this worker is simply not running."
                        if block_parsed
                        else "No subsolver announcement parsed at all, so the line this "
                        "harness reads the restriction off has moved or gone."
                    ),
                )
            )
        elif announced[role] != expected:
            failures.append(
                PreflightFailure(
                    WORKER_RESTRICTION_CHECK,
                    f"{role} subsolvers are {sorted(announced[role])}, expected "
                    f"{sorted(expected)}. The solve is no longer restricted to the "
                    "fj + ls workers, so it would not be the baseline this benchmark "
                    "compares against.",
                )
            )
    if ambiguous:
        failures.append(
            PreflightFailure(
                LOG_FORMAT_CHECK,
                f"neither expected role was announced, but others were ({sorted(announced)}). "
                "An unrestricted run still announces `first solution`, so losing BOTH roles "
                "at once is as likely to be these role labels having been renamed as the "
                "restriction having gone. Both checks are named because the log does not say "
                "which: compare the announcement block against EXPECTED_SUBSOLVERS by hand.",
            )
        )
    return failures


def check_preflight_log(
    log_text: str, *, status: str, workers: int, found_solution: bool
) -> list[PreflightFailure]:
    """Everything the preflight can conclude from one solve, without solving again.

    Pure, so the failure modes it exists for can be tested against a log captured
    from the release that is broken rather than by installing that release.
    """
    failures: list[PreflightFailure] = []
    if status == "INVALID_SOLVER_PARAMETERS":
        return [
            PreflightFailure(
                WORKER_RESTRICTION_CHECK,
                "CP-SAT rejected the parameter string outright "
                f"(status {status}). `filter_subsolvers` no longer accepts these "
                "names, or a parameter in the string was renamed or removed.",
            )
        ]

    announced = parse_subsolvers(log_text)
    # The announcement block is one block. If any line in it parsed, the format is
    # intact and a *role* missing from it is the restriction having gone, not the
    # parser having gone stale -- an unrestricted run announces `full problem` and
    # neither of the two below. Only a block that parsed nothing at all is a format
    # break. That keeps the shape/content split of the two checks checkable rather
    # than a guess: what is read is shape, what it says is content.
    failures += _announcement_failures(announced)
    extra = {
        role: names
        for role, names in announced.items()
        if role not in EXPECTED_SUBSOLVERS and role not in UNRESTRICTED_ROLES and names
    }
    if extra:
        failures.append(
            PreflightFailure(
                WORKER_RESTRICTION_CHECK,
                f"subsolvers outside the fj + ls pairing are running: "
                f"{ {role: sorted(names) for role, names in sorted(extra.items())} }.",
            )
        )

    worker_line = [SEARCH_WORKERS_LINE.match(line.strip()) for line in log_text.splitlines()]
    matched = [m for m in worker_line if m is not None]
    if not matched:
        failures.append(
            PreflightFailure(
                LOG_FORMAT_CHECK,
                "the `Starting search at Xs with N workers.` line is absent, so the "
                "thread count the baseline ran at cannot be confirmed from the log.",
            )
        )
    elif int(matched[0].group(1)) != workers:
        failures.append(
            PreflightFailure(
                WORKER_RESTRICTION_CHECK,
                f"CP-SAT started {matched[0].group(1)} workers, not the {workers} asked "
                "for. The baseline would get a different share of CPU than CBLS.",
            )
        )

    if not found_solution:
        failures.append(
            PreflightFailure(
                WORKER_RESTRICTION_CHECK,
                "the restricted configuration found no solution at all on a model the "
                "fj + ls pairing solves in milliseconds. The workers the filter leaves "
                "running cannot search.",
            )
        )
    elif not parse_trace(log_text):
        failures.append(
            PreflightFailure(
                LOG_FORMAT_CHECK,
                "the solve found a solution but no improving-solution line matched "
                f"`{SOLUTION_LINE.pattern}`. Every CP-SAT incumbent profile is "
                "recovered from those lines, so the whole roster would score as if "
                "the baseline never improved.",
            )
        )
    return failures


#: Shape of the model the preflight solves: enough covering rows over enough
#: binaries that presolve cannot close it, small enough to build and search in
#: milliseconds. Built in memory, so the check needs no instance file and no
#: network -- it has to be runnable before the roster is even fetched.
PREFLIGHT_COLUMNS = 40
PREFLIGHT_ROWS = 25


def build_preflight_model() -> model_builder.ModelBuilder:
    """A tiny set-covering model, generated deterministically rather than read.

    The coefficients come from a fixed integer recurrence rather than a random
    seed so that the preflight solves the same model on every machine and every
    release: a check whose input drifts cannot say whether the solver moved.
    """
    model = model_builder.ModelBuilder()  # type: ignore[no-untyped-call]
    columns = [model.new_bool_var(f"x{i}") for i in range(PREFLIGHT_COLUMNS)]
    for row in range(PREFLIGHT_ROWS):
        weights = [1 + (7 * row + 13 * i + row * i) % 20 for i in range(PREFLIGHT_COLUMNS)]
        model.add(sum(w * x for w, x in zip(weights, columns, strict=True)) >= 60)
    model.minimize(sum((1 + (5 * i) % 30) * x for i, x in enumerate(columns)))
    return model


def run_preflight(workers: int = DEFAULT_WORKERS, seed: int = 42) -> list[PreflightFailure]:
    """Solve the preflight model and report what the release broke, if anything.

    Run once before a roster, not once per instance: an OR-Tools release that
    renames a subsolver flag or reformats a log line does not crash, it silently
    produces a degraded or empty baseline across every instance, discovered only
    at scoring time. This is the same failure at second zero.
    """
    model = build_preflight_model()
    solver = model_builder.ModelSolver("SAT")
    solver.enable_output(True)
    solver.set_time_limit_in_seconds(PREFLIGHT_BUDGET_SECONDS)
    solver.set_solver_specific_parameters(build_parameters(workers, seed))
    with tempfile.TemporaryDirectory(prefix="cpsat-preflight-") as tmpdir:
        log_path = Path(tmpdir) / "preflight.log"
        with capture_stdout_fd(log_path):
            status = solver.solve(model)
        log_text = log_path.read_text()
    return check_preflight_log(
        log_text,
        status=status.name,
        workers=workers,
        found_solution=status
        in (model_builder.SolveStatus.OPTIMAL, model_builder.SolveStatus.FEASIBLE),
    )


def report_preflight(failures: list[PreflightFailure], workers: int) -> int:
    """Print the verdict; 0 when the baseline is the one this harness documents."""
    installed = version("ortools")
    if not failures:
        print(
            f"preflight OK: ortools {installed}, {workers} worker(s), restricted to "
            f"{sorted(EXPECTED_SUBSOLVERS['first solution'] | EXPECTED_SUBSOLVERS['interleaved'])}"
            ", improving-solution lines parse."
        )
        return 0
    broken = sorted({failure.check for failure in failures})
    print(
        f"PREFLIGHT FAILED on ortools {installed}: {' and '.join(broken)} broke.",
        file=sys.stderr,
    )
    for failure in failures:
        print(f"  [{failure.check}] {failure.message}", file=sys.stderr)
    print(
        f"\nThis harness's parsing was established against ortools {SUPPORTED_ORTOOLS_RANGE}. "
        "A run started now would produce a degraded or empty CP-SAT baseline across the "
        "whole roster and only show it at scoring time, so it is refused here.",
        file=sys.stderr,
    )
    return 3


#: Magnitude at or past which ModelBuilder reports a constraint bound as infinite.
#: Its own sentinel is 1e30; compared as a threshold rather than for equality so a
#: bound the reader scaled or rounded still reads as infinite.
MODEL_BUILDER_INFINITY = 1e30


def count_free_constraints(model: model_builder.ModelBuilder) -> int:
    """Linear constraints CP-SAT holds that bound nothing on either side.

    These are the MPS `N` rows after the first: the first is the objective, and
    every reader treats it as such, but ModelBuilder keeps the rest as
    unconstrained linear constraints where the CBLS adapter (and SCIP) drop them.
    That is the whole of the known-benign way the two engines' constraint counts
    differ, so it is recorded rather than reasoned about after the fact -- the
    scorer's model-shape cross-check subtracts exactly this and flags whatever is
    left (issue #139).

    Read off the model helper rather than `get_linear_constraints()`, which
    materialises a pandas Index holding one Python proxy per constraint. This is
    instrumentation for a cross-check, and on a roster whose largest models run to
    a million rows it must not cost what it measures.
    """
    helper = model.helper
    return sum(
        1
        for i in range(helper.num_constraints())
        if helper.constraint_lower_bound(i) <= -MODEL_BUILDER_INFINITY
        and helper.constraint_upper_bound(i) >= MODEL_BUILDER_INFINITY
    )


def status_note(status_name: str, has_solution: bool, parameters: str) -> tuple[str, str] | None:
    """`(status, message)` for a verdict that means "did not search", else None.

    CP-SAT scales continuous columns to integers and rejects what it cannot express
    (MODEL_INVALID), and ABNORMAL means it errored out. Both are "did not search",
    not "searched and found nothing", so they are tallied apart rather than counted
    against the baseline as a search failure.

    NOT_SOLVED deliberately gets no note: it is the ordinary outcome of a
    time-limited run that found nothing, which is exactly what the metric is asking
    about. The precise verdict survives in `cpsat_status` and reaches the comparison
    table as `solver_status`.

    Each carries a message, because the scorer publishes one per withheld row and a
    row reading "no message recorded" is the least useful form of the most important
    defect class this baseline has -- and it is the class the preflight exists to
    catch before a roster rather than after it.
    """
    if status_name == "INVALID_SOLVER_PARAMETERS":
        return (
            "invalid_parameters",
            f"CP-SAT rejected the parameter string `{parameters}`. The worker "
            "restriction this baseline is defined by is not in force; run "
            "`cpsat_solve.py --preflight` to see which part of it moved.",
        )
    if not has_solution and status_name in ("MODEL_INVALID", "ABNORMAL"):
        return (
            "invalid_model",
            f"CP-SAT returned {status_name}: it did not search. MODEL_INVALID means "
            "it could not express the model (it scales continuous columns to "
            "integers and rejects what will not scale); ABNORMAL means it errored "
            "out. Either way this row is a baseline limitation, not a search result.",
        )
    return None


@contextlib.contextmanager
def capture_stdout_fd(sink_path: Path) -> Iterator[None]:
    """Redirect fd 1 (including writes from C++) to `sink_path` for the block's duration.

    The caller owns `sink_path`. Holding it in a TemporaryDirectory is what makes the
    log disappear on every exit path, including a solve that raises; deleting it only
    after a clean return leaks one log per failed job, and these logs are not small.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with open(sink_path, "w") as sink:
            os.dup2(sink.fileno(), 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def solve(
    mps_path: Path, budget: float, workers: int, seed: int
) -> tuple[dict[str, object], list[tuple[float, float]], dict[str, float]]:
    """Run the LS-only CP-SAT configuration.

    Returns (result record, incumbent trace, solution values). The values are
    keyed by the MPS column name, empty when the run found nothing; they are what
    `verify_solution.py` checks against the instance file.
    """
    # ortools ships no annotations for ModelBuilder's constructor; the rest of the
    # model_builder surface used here is typed.
    model = model_builder.ModelBuilder()  # type: ignore[no-untyped-call]
    # Read and build are one call here -- ModelBuilder parses the MPS straight
    # into its own protobuf -- so this is the whole of the setup CBLS splits into
    # `read_seconds` + `build_seconds`. Timed on both sides because the published
    # wall-clock column has only ever bracketed the solve.
    setup_started = time.monotonic()
    if not model.import_from_mps_file(str(mps_path)):
        failed_after = time.monotonic() - setup_started
        # Carries every key main() and the scorer read unconditionally. A key
        # missing here crashes the job *after* write_outputs has written its
        # result, and the driver — seeing a result file — reports that as a clean
        # run. The scorer .get()s the rest.
        return (
            {
                "status": "read_error",
                "message": f"CP-SAT could not import {mps_path.name}",
                "wall_seconds": 0.0,
                # One reading, not two: the same failure must not report two
                # different durations for the same measurement.
                "read_seconds": failed_after,
                "setup_seconds": failed_after,
                "objective": None,
            },
            [],
            {},
        )

    read_seconds = time.monotonic() - setup_started

    solver = model_builder.ModelSolver("SAT")
    solver.enable_output(True)
    solver.set_time_limit_in_seconds(budget)
    parameters = build_parameters(workers, seed)
    solver.set_solver_specific_parameters(parameters)
    setup_seconds = time.monotonic() - setup_started
    # Counted after the setup clock stops: this is the scorer's cross-check, not
    # work the baseline needs, and charging it to `setup_seconds` would inflate
    # exactly the number the timing split exists to publish.
    free_cons = count_free_constraints(model)

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cpsat-log-") as tmpdir:
        log_path = Path(tmpdir) / "solve.log"
        with capture_stdout_fd(log_path):
            status = solver.solve(model)
        wall = time.monotonic() - started
        trace = parse_trace_file(log_path)
    has_solution = status in (
        model_builder.SolveStatus.OPTIMAL,
        model_builder.SolveStatus.FEASIBLE,
    )

    record: dict[str, object] = {
        "status": "feasible" if has_solution else "no_solution",
        "cpsat_status": status.name,
        "wall_seconds": wall,
        "read_seconds": read_seconds,
        "setup_seconds": setup_seconds,
        "n_vars": model.num_variables,
        "n_cons": model.num_constraints,
        "n_free_cons": free_cons,
        "objective": None,
    }
    note = status_note(status.name, has_solution, parameters)
    if note is not None:
        record["status"], record["message"] = note

    # Whether the incumbent profile came from the log or is a single end-point.
    # A systematic regex miss after an OR-Tools log-format change would otherwise
    # score every CP-SAT instance ~2.0, indistinguishable from "CP-SAT is bad".
    record["trace_source"] = "log" if trace else "final_only"

    values: dict[str, float] = {}
    if has_solution:
        objective = float(solver.objective_value)
        record["objective"] = objective
        # Keyed by name, not by index: the verifier re-reads the instance with a
        # different reader, so a positional dump would verify clean against a
        # differently ordered parse of the same file.
        values = {var.name: float(solver.value(var)) for var in model.get_variables()}
        # The log is the source of truth for *when* each incumbent appeared, but it
        # prints rounded values and the final solution can land after the last logged
        # line. Append the exact final objective so the tail of the profile is right.
        if not trace or trace[-1][1] != objective:
            trace.append((min(wall, budget), objective))
    return record, trace, values


def solution_text(instance: str, objective: object, values: dict[str, float]) -> str:
    """The MIPLIB-style solution format, shared with the CBLS runner.

    `repr` rather than a format string: a rounded value can violate a row the true
    one satisfies, which would read as a solver defect rather than a lossy dump.
    """
    lines = [f"# instance {instance}", "# engine cpsat", f"=obj= {objective!r}"]
    lines += [f"{name} {value!r}" for name, value in values.items()]
    return "\n".join(lines) + "\n"


def write_outputs(
    out_dir: Path,
    instance: str,
    record: dict[str, object],
    trace: list[tuple[float, float]],
    args: argparse.Namespace,
    values: dict[str, float] | None = None,
) -> dict[str, object]:
    """Write the trace, the solution vector and the result; return what was written.

    Returns the record rather than mutating the caller's, because a solution the
    job could not write downgrades the status and withholds the objective — and
    main() has to report and exit on what actually landed on disk.
    """
    record = dict(record)
    record.update(
        engine="cpsat",
        instance=instance,
        # Reported per result so a full-roster run's concurrency can be sized from
        # measurement rather than guessed.
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        budget_seconds=args.budget,
        seed=args.seed,
        workers=args.workers,
        ortools_version=version("ortools"),
        parameters=build_parameters(args.workers, args.seed),
    )
    lines = ["time_seconds,objective"]
    lines += [f"{t},{obj}" for t, obj in trace]
    (out_dir / f"{instance}.trace.csv").write_text("\n".join(lines) + "\n")

    if args.solution_dir and values:
        solution_dir = Path(args.solution_dir)
        solution_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Temp-then-rename, like every other file this benchmark writes: a job
            # killed mid-write must leave no solution rather than a truncated one,
            # which would verify as an infeasible point.
            tmp_solution = solution_dir / f"{instance}.sol.tmp"
            tmp_solution.write_text(solution_text(instance, record["objective"], values))
            tmp_solution.replace(solution_dir / f"{instance}.sol")
        except OSError as exc:
            # No solution file means no independent verdict, and a row with no
            # verdict must not publish a number (#138).
            record["status"] = "solution_write_error"
            record["message"] = f"could not write the solution vector: {exc}"
            record["objective"] = None

    # Result last, and via a rename. The driver resumes on the result file's
    # existence, so it must not appear before the trace it is scored with, and it
    # must never appear truncated.
    tmp = out_dir / f"{instance}.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_dir / f"{instance}.json")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="assert the worker restriction and the log format on one tiny in-memory "
        "model, then exit. No instance, no network. The run driver does this once "
        "before any real solving, so a release that broke either fails at second zero "
        "instead of after the roster has burned its budget",
    )
    parser.add_argument("--instance", default=None)
    parser.add_argument("--inst-dir", default="benchmarks/instances/mipfeas")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--budget", type=float, default=600.0, help="seconds (MIPfeas uses 600)")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="CP-SAT threads; 1 runs both the fj and ls workers, matching CBLS",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--solution-dir",
        default=None,
        help="write the solution vector of a feasible run here, for "
        "verify_solution.py to check against the original instance file",
    )
    args = parser.parse_args()

    if args.preflight:
        return report_preflight(run_preflight(args.workers, args.seed), args.workers)
    missing = [f for f in ("instance", "out_dir") if getattr(args, f) is None]
    if missing:
        parser.error(
            f"{', '.join('--' + f.replace('_', '-') for f in missing)} "
            "required unless --preflight is given"
        )

    mps_path = Path(args.inst_dir) / f"{args.instance}.mps.gz"
    if not mps_path.exists():
        # No result file: an absent instance is an incomplete run, not a zero score.
        print(
            f"{mps_path} not found. Fetch the roster first:\n  python {args.inst_dir}/download.py",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record, trace, values = solve(mps_path, args.budget, args.workers, args.seed)
    record = write_outputs(out_dir, args.instance, record, trace, args, values)

    objective = record.get("objective")
    print(
        f"{args.instance:<28} {str(record['status']):<12} "
        f"obj={objective if objective is not None else 'n/a':<16} "
        f"{float(record['wall_seconds']):8.2f}s"  # type: ignore[arg-type]
    )
    # A rejected parameter string or an unreadable instance is a harness fault, not
    # a search outcome. An OR-Tools release renaming `filter_subsolvers` would
    # otherwise score every CP-SAT instance at 2.0 across a 39 CPU-hour run, at exit
    # 0. Matches the CBLS runner, which also exits 1 on a read error — and on a
    # solution it could not write, which leaves a row nothing can verify.
    return (
        1 if record["status"] in ("invalid_parameters", "read_error", "solution_write_error") else 0
    )


if __name__ == "__main__":
    sys.exit(main())
