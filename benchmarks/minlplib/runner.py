"""The `cbls_minlplib` runner's contract, as every Python consumer reads it.

One row per instance, one exit status per invocation, one vocabulary of notes,
shared by both drivers and the ablation report (#160). `minlplib.cpp` is the
source of truth; tests pin this module against its literals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

#: The runner target, and the executable `cmake --build` produces for it.
RUNNER_TARGET = "cbls_minlplib"

#: Exit status when at least one instance threw while read, built or solved
#: (#153). 1 is "no roster" or an exception escaping `main`; 2 is a bad flag or
#: unopenable output (`benchmarks/common/runner_args.h`). A coverage gap
#: (unsupported, `.nl` not downloaded) is not an error and exits 0. Pinned
#: against `kExitErrored` in `minlplib.cpp`.
RUNNER_EXIT_ERRORED = 3

#: Result-row columns, in order, one row per instance in `--out`. Pinned against
#: `minlplib.cpp` (`test_the_published_header_still_matches_what_the_runner_writes`)
#: and the committed `comparison.csv` header
#: (`test_the_committed_table_uses_the_columns_the_driver_assembles`).
RUNNER_COLUMNS: tuple[str, ...] = (
    "instance",
    "objective",
    "primal_bks",
    "dual_bound",
    "gap_to_bks%",
    "gap_to_dual%",
    "wall_seconds",
    "feasible",
    "note",
    "commit_sha",
    "max_violation",
    "n_int_vars",
    "lns_repairs",
    "lns_repairs_accepted",
    "first_feasible_objective",
    "time_to_first_feasible",
    "search_config",
)

#: The columns of the anytime trace the runner writes under `--trace`.
TRACE_COLUMNS: tuple[str, ...] = ("instance", "time_seconds", "objective", "new_best")

#: Rows published as documented failures, excluded from every quality aggregate
#: and claim until #110 lands and #116's criterion can be checked (#87). They
#: stay in the roster and its counts, so the table agrees with `bounds.csv`
#: (#123); `campaign_report.AGGREGATION_RULE` says which aggregates are which (#142).
CLAIM_EXCLUDED = ("elec25", "elec50")

#: `note` prefix of a `run_ablation.failed_row` (runner exited nonzero): no
#: measurement, so held out of every count rather than read as infeasible.
RUNNER_FAILED_NOTE = "runner-failed"

#: Notes a completed search produces: the ALLOWLIST deciding whether a row is a
#: scored measurement; anything else is held out and disclosed (#153). A
#: denylist would fail open, scoring a new runner outcome as lost feasibility
#: (#151); an allowlist that lags the runner discards real measurements instead,
#: so `test_every_completed_search_note_the_runner_writes_is_allowlisted` sweeps
#: `minlplib.cpp` for them.
#:
#: Sources in `minlplib.cpp`: `classify_against_bks` (`better-than-bks`,
#: `matches-bks`, `within-tolerance-of-bks`, `feasible`; `run_instance` also
#: writes `feasible` when there is no bound), `verify_assignment`
#: (`VERIFY-FAILED(...)`), `describe_infeasible` (`infeasible(...)`), the
#: non-finite-objective guard (`non-finite`). Matched on the cell's START
#: because each may carry `; integrality-mismatch(...)`, `; stale-analysis-note`
#: or ` | <curated note>`. `note_policy.h` enumerates no note strings, so
#: nothing is shared with it.
COMPLETED_SEARCH_NOTES: tuple[str, ...] = (
    "better-than-bks",
    "matches-bks",
    "within-tolerance-of-bks",
    "feasible",
    "non-finite",
    "VERIFY-FAILED",
    "infeasible",
)

#: Coverage-gap notes: declined by reader or adapter, or `.nl` never downloaded.
#: Exit 0, bucketed apart, published as documented rows.
COVERAGE_GAP_NOTES: tuple[str, ...] = ("unsupported", "not-found")

#: Notes a staged row may carry and still stand in for a fresh solve: a
#: completed search or a coverage gap. An allowlist so it cannot fail open; the
#: rest (`read-error`, `build-error`, `solve-error`, any later note) come with
#: `RUNNER_EXIT_ERRORED` and are refused, since such a row passes every
#: structural check.
STAGEABLE_NOTES: tuple[str, ...] = (*COMPLETED_SEARCH_NOTES, *COVERAGE_GAP_NOTES)


def completed_search(note: str) -> bool:
    """Whether this row's note is one a COMPLETED search produces.

    Anything else (a recorded crash, a row written before any search ran, an
    unknown note) is held out of every count.
    """
    return note.startswith(COMPLETED_SEARCH_NOTES)


def stageable_note(note: str) -> bool:
    """Whether a staged row carrying this note may stand in for a fresh solve."""
    return note.startswith(STAGEABLE_NOTES)


def runner_command(
    build_dir: Path,
    inst_dir: Path,
    *,
    time_limit: float,
    seed: int,
    sha: str,
    instance: str,
    out: Path,
    solution_dir: Path,
    threads: int = 1,
    extra: Sequence[str] = (),
) -> list[str]:
    """The `cbls_minlplib` invocation for one instance.

    Always passes `--instance` (the runner refuses to write `comparison.csv`
    from a subset run), `--solution-dir` (where verified assignments go;
    `independent_check.check_rows` refuses a verified row without one, #205)
    and `--threads` (states the portfolio size even at 1, #299).
    """
    return [
        str(build_dir / RUNNER_TARGET),
        str(inst_dir),
        "--time-limit",
        f"{time_limit:g}",
        "--seed",
        str(seed),
        "--commit",
        sha,
        "--instance",
        instance,
        "--out",
        str(out),
        "--solution-dir",
        str(solution_dir),
        "--threads",
        str(threads),
        *extra,
    ]


def build_command(build_dir: Path, jobs: int) -> list[str]:
    """Rebuild the runner, so the binary cannot lag the commit it is labelled with."""
    return ["cmake", "--build", str(build_dir), "--target", RUNNER_TARGET, "-j", str(jobs)]
