"""The `cbls_minlplib` runner's contract, as every Python consumer reads it.

One row per instance, one exit status per invocation, one vocabulary of notes.
Both drivers (`run_benchmark.py`, `run_ablation.py`) and both reports
(`ablation_report.py`, `first_feasible_report.py`) read it from here; until
#160 each spelled out its own copy and a test pinned the copies equal.
`benchmarks/minlplib/minlplib.cpp` is the source of truth, and the tests pin
this module against its literals rather than against a run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

#: The runner target, and the executable `cmake --build` produces for it.
RUNNER_TARGET = "cbls_minlplib"

#: The runner's exit status when its error tally is nonzero -- at least one
#: instance THREW while being read, built or solved (#153). Distinct from the
#: codes already in use: 1 is "no roster to run" and an exception escaping the
#: runner's `main`, 2 is a bad flag or an output file that would not open
#: (`benchmarks/common/runner_args.h`).
#:
#: A coverage gap is NOT an error and does not reach this: an instance skipped as
#: unsupported, or one whose `.nl` was never downloaded, is bucketed apart by the
#: runner and leaves the exit status at 0. `kExitErrored` in `minlplib.cpp` is
#: the source of truth and a test pins this constant against it.
RUNNER_EXIT_ERRORED = 3

#: The columns of every result row the runner writes, in order: one row per
#: instance in `--out`. `test_the_published_header_still_matches_what_the_runner_writes`
#: pins each name against `minlplib.cpp`, and
#: `test_the_committed_table_uses_the_columns_the_driver_assembles` pins the
#: committed `comparison.csv` header against this tuple.
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

#: Instances whose rows are published as documented failures and are excluded
#: from every aggregate and every quality claim, per issue #87 ("Do not publish
#: `elec` rows until #110 lands and #116's criterion can actually be checked").
#: They stay *in* the roster: #123 asks for 50 instances, the roster of record is
#: `bounds.csv`, and dropping the rows would make the table disagree with it.
CLAIM_EXCLUDED = ("elec25", "elec50")

#: Prefix `run_ablation.failed_row` puts in a row's `note` when the runner
#: process exited nonzero. Such a row records no measurement of anything, so it
#: is held out of every count rather than read as an infeasible run.
RUNNER_FAILED_NOTE = "runner-failed"

#: Notes a COMPLETED search produces -- the ALLOWLIST that decides whether a row
#: is a measurement. A row whose note starts with one of these ran a search and
#: reported something, so it is scored; anything else is held out and disclosed.
#:
#: The polarity is deliberate (#153). This was a denylist of the six notes that
#: mean "no search completed", and a denylist FAILS OPEN: a seventh outcome added
#: to the runner later would be scored as a lost feasibility -- exactly the
#: defect #151 was filed for, re-armed. An allowlist fails the other way: an
#: unrecognised note is held out, counted, and named on the report's disclosure
#: line, so it announces itself instead of moving a number. The cost of that
#: polarity is that an allowlist which falls behind the runner DISCARDS real
#: measurements, which is why
#: `test_every_completed_search_note_the_runner_writes_is_allowlisted` sweeps
#: `minlplib.cpp` for the literals a completed search can write.
#:
#: From `minlplib.cpp`, in full: `classify_against_bks` returns `better-than-bks`,
#: `matches-bks`, `within-tolerance-of-bks` or `feasible` (which `run_instance`
#: also writes directly when there is no published bound to classify against);
#: `verify_assignment` writes `VERIFY-FAILED(...)`; `describe_infeasible` writes
#: `infeasible(...)`; and the non-finite-objective guard writes `non-finite`.
#: Each may carry an appended `; integrality-mismatch(...)`,
#: `; stale-analysis-note` or ` | <curated note>`, which is why the match is on
#: the START of the cell.
#:
#: NOT shared with `benchmarks/minlplib/note_policy.h`, which was checked: that
#: header is the three-way merge policy for `analysis_notes.csv` (kNone/kMerge/
#: kStale) and enumerates no note strings at all.
COMPLETED_SEARCH_NOTES: tuple[str, ...] = (
    "better-than-bks",
    "matches-bks",
    "within-tolerance-of-bks",
    "feasible",
    "non-finite",
    "VERIFY-FAILED",
    "infeasible",
)

#: The notes of a coverage gap: an instance the reader or adapter declined, and
#: one whose `.nl` was never downloaded. The runner exits 0 on both and buckets
#: them apart; they are documented rows published like any other.
COVERAGE_GAP_NOTES: tuple[str, ...] = ("unsupported", "not-found")

#: The notes a staged row may carry and still stand in for a fresh solve: a
#: completed search, or a coverage gap. An ALLOWLIST, for the same reason
#: `COMPLETED_SEARCH_NOTES` is one: the first draft of this guard named the three
#: notes a throw writes, which fails open the moment the runner grows a fourth.
#: Everything else -- `read-error`, `build-error`, `solve-error`, and any note
#: added later -- comes WITH `RUNNER_EXIT_ERRORED` and is refused: such a row is
#: complete by every structural check, which is exactly why it has to be named.
STAGEABLE_NOTES: tuple[str, ...] = (*COMPLETED_SEARCH_NOTES, *COVERAGE_GAP_NOTES)


def completed_search(note: str) -> bool:
    """Whether this row's note is one a COMPLETED search produces.

    The scorer's one classification question. Everything else -- a crash the
    driver recorded, a row the runner wrote before any search ran, and any note
    neither list has heard of -- is held out of every count.
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
    extra: Sequence[str] = (),
) -> list[str]:
    """The `cbls_minlplib` invocation for one instance.

    `--instance` is always present, which is itself a lock on the published
    table: the runner refuses to write `comparison.csv` from a subset run
    whatever else is passed.
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
        *extra,
    ]


def build_command(build_dir: Path, jobs: int) -> list[str]:
    """Rebuild the runner, so the binary cannot lag the commit it is labelled with."""
    return ["cmake", "--build", str(build_dir), "--target", RUNNER_TARGET, "-j", str(jobs)]
