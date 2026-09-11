"""Score the MINLPLib ablation campaign of issue #143.

Reads the `results.csv` that `run_ablation.py` appends to and reports, for each
arm, its per-instance and aggregate effect against **the control from the same
sitting** -- never against the published table, which was measured at another
time on a differently-loaded machine.

THE NOISE FLOOR IS MEASURED, NOT ASSUMED. Issue #143 quotes a "3-4 gap point"
noise floor from an earlier campaign; nothing here uses that number. The only
measurement of run-to-run noise this campaign contains is the control's own
spread across its seeds, and that is what the floor is built from:

* For each instance whose control produced at least two feasible runs with a
  finite gap, `s_i` is the sample standard deviation (ddof=1) of those gaps.
* The effect statistic for an instance is a difference of per-arm mean gaps, so
  its noise scale is `se_i = s_i * sqrt(1/k_arm + 1/k_control)`, with `k` the
  number of comparable runs on each side.
* The **per-instance floor** is `t * se_i`, a two-sided 95% band, with `t` the
  Student multiplier for `k_control - 1` degrees of freedom -- 4.30 at three
  seeds, not 1.96. Three seeds is a small sample and the band has to say so;
  using 2.0 there would print a band about half its nominal width.
* That band is then RAISED, where it falls below it, to the runner's own tie
  band for the instance -- `min_move_points`, the one number in this module that
  is not measured. A control that returned ~1e-7 on every seed measures a
  spread, but not one the gap it bounds can resolve, and scoring a 1e-5 delta
  against it is reading noise the runner would not call a difference at all.
  The instances this touches are counted and disclosed on their own line,
  because for them "the floor is measured" stops being the whole truth. It is
  a per-instance quantity and not a constant for the reason the next bullet
  gives about imputation: a gap point is a percentage of the published bound,
  so the same objective difference is a different number of points at every
  scale on this roster.
* An instance whose control produced **fewer than two** comparable runs, or an
  identical gap on every seed, has no measured spread and is **not scored**. It
  is counted and listed in its own line instead. Noise cannot be imputed onto it
  from other instances: this
  roster's gaps span six orders of magnitude, so borrowing the median absolute
  spread (~1 gap point) onto an instance whose gap is ~1e6 hands it a floor it
  clears automatically, which manufactures a significant result rather than
  measuring one.

THE VERDICT IS SCALE-FREE, AND THIS IS THE POINT ON WHICH AN EARLIER CUT WAS
WRONG. Gap-to-BKS is a percentage of wildly differing magnitudes across this
roster: in the committed table `gear4` alone is 1.65e6 gap points, 98.5% of the
sum over all finite gaps, where the median instance is 1.00. An unweighted mean
of per-instance deltas is therefore not an average over the roster at all -- it
is `gear4`'s seed draw, and a uniform 10-point regression on all 45 other
instances moves it by less than its own noise. So the verdict is built from
statistics that do not care about scale:

* Each instance is judged against **its own** floor. `moved worse` and
  `moved better` count the instances whose delta exceeds it.
* Those counts are read with an exact two-sided **sign test**. An arm that
  moved nothing outside its own floor is "inside the noise", with the typical
  floor quoted; an arm that moved instances in both directions in comparable
  numbers is reported as mixed rather than as an effect.
* The **median** delta is the location statistic quoted, never the mean.

The mean is still printed, immediately followed by the share of it contributed
by the single largest instance, so a reader who reaches for it can see at once
whether it describes the roster or one instance.

WHAT DOES NOT GET AVERAGED. A gap is defined only for a run that was feasible
AND has a finite published bound to be scored against, so an arm that is
feasible where the control is not has no delta at all -- there is no control gap
to subtract. Silently averaging that instance in (as a NaN, or by pretending the
missing side scored zero) is the failure this module is written to avoid: every
instance lands in exactly one of six buckets, the buckets are counted in the
report, and only `both-feasible` contributes to a mean. The sixth,
`no-runs-recorded`, is the same refusal applied to an absence of RUNS: a cell
whose every row crashed or errored has `feasible_runs == 0` without that being
a measurement of anything, so it is bucketed apart rather than read as a side
losing feasibility.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from benchmarks.minlplib.run_benchmark import CLAIM_EXCLUDED

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

#: The arm every other arm is measured against.
CONTROL_ARM = "control"

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
#: measurements, which is why `test_every_completed_search_note_is_allowlisted`
#: sweeps `minlplib.cpp` for the literals a completed search can write.
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

#: Notes known to mean NO SEARCH COMPLETED. Nothing is classified by this tuple
#: any more -- `completed_search` decides that from `COMPLETED_SEARCH_NOTES`
#: alone -- but a held-out row still has to be REPORTED, and "3 run(s) completed
#: no search (solve-error)" is a different sentence from
#: "(unsupported: NL_UNKNOWN_OPCODE 42; at row 3)" repeated three times. So these
#: are the labels `no_search_label` collapses a known note to, in the order the
#: report lists them; a note matching none of them keeps its own text and is
#: called out as unrecognised.
#:
#: `RUNNER_FAILED_NOTE` is deliberately first: the report lists the runner's own
#: notes as `NO_SEARCH_NOTES[1:]`, because the driver's crashes get their own
#: line. The runner's five are, by writer:
#:
#:   * `not-found`, `read-error`, `build-error`  -- `write_preread_row`, before
#:     a model exists at all;
#:   * `unsupported`/`unsupported: <reason>`     -- the reader or the adapter
#:     declined the instance;
#:   * `solve-error`                             -- `cbls::solve` threw.
NO_SEARCH_NOTES: tuple[str, ...] = (
    RUNNER_FAILED_NOTE,
    "solve-error",
    "read-error",
    "build-error",
    "not-found",
    "unsupported",
)

#: What `no_search_label` prefixes a note no list recognises with. Kept as a
#: constant because the report keys its third disclosure clause off it and a test
#: asserts on it.
UNRECOGNISED_NOTE = "unrecognised-note"


def completed_search(note: str) -> bool:
    """Whether this row's note is one a COMPLETED search produces.

    The scorer's one classification question. Everything else -- a crash the
    driver recorded, a row the runner wrote before any search ran, and any note
    neither list has heard of -- is held out of every count.
    """
    return any(note.startswith(prefix) for prefix in COMPLETED_SEARCH_NOTES)


def no_search_label(note: str) -> str:
    """A held-out row's note collapsed to the label the report lists it under.

    A known no-search note collapses to its prefix, so the long
    `unsupported: <reason>` form and the driver's `runner-failed-exit-139` do
    not each become their own histogram bucket. Anything else keeps its text and
    is marked unrecognised: it is held out either way, and a reader who is being
    told a row was not scored needs to be able to see what the row said.
    """
    for prefix in NO_SEARCH_NOTES:
        if note.startswith(prefix):
            return prefix
    return f"{UNRECOGNISED_NOTE}({note.strip() or '<empty>'})"


def note_order(label: str) -> tuple[int, str]:
    """Sort key for the disclosure line: the known labels first, in list order."""
    if label in NO_SEARCH_NOTES:
        return (NO_SEARCH_NOTES.index(label), "")
    return (len(NO_SEARCH_NOTES), label)


#: The relative band inside which the RUNNER ITSELF calls two objectives equal.
#: `classify_against_bks` (minlplib.cpp) returns `matches-bks` when the two
#: differ by at most `tie_band = 1e-6 * (|primal_bks| + 1)` objective units, and
#: says in its own comment why that band is relative rather than absolute --
#: reusing an absolute one published `ex8_4_5` (BKS 3.07e-4) as a match when it
#: was 1.38% worse. This module reuses the runner's number rather than inventing
#: a second one.
OBJ_TIE_RELATIVE = 1e-6

#: Where `safe_gap` stops dividing. Below it a "gap point" is a raw objective
#: residual rather than a percentage, so the floor is in objective units too.
GAP_ABSOLUTE_BRANCH = 1e-12

#: The tie band in GAP POINTS as |primal_bks| grows without bound, and the floor
#: used for an instance whose recorded bound cannot say better. See
#: `min_move_points` for why this is the infimum and never over-suppresses.
MIN_MOVE_POINTS = 1e-4


def min_move_points(primal_bks: float) -> float:
    """The smallest gap-point delta that can be a move on this instance.

    WHY A FLOOR AT ALL. The measured band is the control's own across-seed
    spread, and the spread guard below rejects only an EXACT zero. This roster's
    near-zero instances are not exact: a control at gap ~1e-7 on all three seeds
    measures a spread of ~1e-7 and a band of ~3.5e-7 gap points, which a 1e-5
    delta clears -- and four such instances clearing it printed "the arm is
    WORSE than the control" beside "median gap delta +0.00" (issue #151). A
    floor must be meaningful in the units of the thing it bounds.

    WHY THIS FLOOR AND NOT ONE OF THE OTHER TWO #151 LISTS. Both of the others
    were checked against the numbers in the report that prompted it and neither
    fixes it:

    * a RELATIVE-SPREAD test -- score only where `s_i` is a large enough
      fraction of the gap -- passes the exact instances at issue, whose gaps
      are ~1e-7 and whose spreads are ~1e-7: the ratio is order 1, so every
      relative test waves them through;
    * REFUSING to score an instance whose spread is below a multiple of its own
      gap's NUMERICAL resolution fails from the other side -- a double near 1e-7
      resolves to ~1e-23, so a 1e-7 spread is ~1e16 ulps and clears any sane
      multiple.

    So the floor is absolute, which is #151's first option -- but "absolute" has
    to mean absolute in OBJECTIVE units, and a gap point is not one. `safe_gap`
    reports `100*(obj-ref)/|ref|` wherever `|ref| >= GAP_ABSOLUTE_BRANCH`, so
    the runner's own tie band of `OBJ_TIE_RELATIVE * (|ref| + 1)` objective
    units is `1e-4 * (1 + 1/|ref|)` gap points -- a function of the instance,
    not a constant. Pushing the runner's band through `safe_gap` per instance is
    what this returns, and `primal_bks` is already a recorded column, so it
    costs no re-measurement and no schema change.

    A single constant would not do. Its only defensible value is the infimum of
    that expression, `MIN_MOVE_POINTS` = 1e-4, reached as `|ref|` grows: safe
    (it can never suppress a move the runner would call a real difference) but
    loose wherever `|ref| < 1`, and this roster goes far below 1. `ex6_2_6` and
    `ex6_2_11` have `|primal_bks|` ~2.6e-6, where the runner's tie band is ~38
    GAP POINTS -- their whole recorded gaps (-8.3e-5, +9.5e-6) sit inside it.
    A flat 1e-4 there is five orders of magnitude too permissive, which is #151
    unfixed on the campaign's own roster.

    The fallback is that infimum, for a row whose `primal_bks` is missing or
    non-finite. Such an instance has no finite gap either, so it is
    `no-comparable-gap` and never reaches a floor; the fallback is there so that
    a malformed cell errs toward the wider band rather than raising.
    """
    if not math.isfinite(primal_bks):
        return MIN_MOVE_POINTS
    reference = abs(primal_bks)
    if reference < GAP_ABSOLUTE_BRANCH:
        # `safe_gap`'s absolute-residual branch: a gap point IS an objective
        # unit here, so the tie band needs no conversion.
        return OBJ_TIE_RELATIVE * (reference + 1.0)
    return 100.0 * OBJ_TIE_RELATIVE * (reference + 1.0) / reference


#: The gate probe's rows. Same configuration as the control, run over the roster
#: hours earlier for the sole purpose of reading `lns_repairs`, so they are held
#: out of every effect estimate and reported on their own as a drift check.
PROBE_ARM_NAME = "gate-probe"

#: Two-sided 95% Student multipliers by degrees of freedom (`k_control - 1`).
#: At three seeds df = 2 and the multiplier is 4.30, not the 1.96 a normal
#: approximation would use -- an earlier cut used 2.0 flat and printed a band
#: roughly half its nominal width. Beyond the table `t_multiplier` holds the
#: last tabulated value rather than relaxing to the normal one, which is the
#: wider band of the two.
T_95: dict[int, float] = {
    1: 12.71,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
}
NORMAL_Z_95 = 1.96

#: Unanimous movers required before a direction is named without a significant
#: sign test. Below it the movers are reported individually instead: one or two
#: instances clearing their own floors is a fact about those instances, not an
#: arm effect over the roster.
MIN_UNANIMOUS_MOVERS = 4


def t_multiplier(df: int) -> float:
    """The two-sided 95% multiplier for `df` degrees of freedom."""
    if df < 1:
        return math.nan
    if df in T_95:
        return T_95[df]
    # Between AND past the tabulated points take the next LOWER df's multiplier,
    # which is the wider band: erring toward a floor that is too generous keeps a
    # marginal move from being called a result. `>= 10` and not `> 10` because
    # df 11-14 otherwise fell through to NORMAL_Z_95 -- 1.96, NARROWER than both
    # the true t (2.20-2.15) and the tabulated df=10 value, i.e. the opposite of
    # what this comment promises. Every df >= 1 now finds a tabulated multiplier,
    # so NORMAL_Z_95 is a guard against a future edit to T_95 rather than a
    # branch this table can reach; past df=30 the returned 2.042 is deliberately
    # the wider of it and the asymptotic 1.96.
    # `wider` is tested before it is indexed: `max([])` raises, so writing
    # this as a conditional on `max(wider)` made NORMAL_Z_95 a fallback that
    # could not actually fire. The tabulated value is always the wider band, so
    # there is no df at which falling back to 1.96 is the safer answer.
    wider = [k for k in T_95 if k < df]
    return T_95[max(wider)] if wider else NORMAL_Z_95


def sign_test_p(worse: int, better: int) -> float:
    """Exact two-sided binomial p-value for `worse` vs `better` under p = 1/2.

    Written out rather than pulled from scipy: the campaign's dependencies are
    the repository's, and this is six lines.
    """
    n = worse + better
    if n == 0:
        return 1.0
    extreme = min(worse, better)
    tail = math.fsum(math.comb(n, k) for k in range(extreme + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def format_points(value: float, *, signed: bool = True) -> str:
    """A gap-point quantity with enough digits to be visible.

    `%.2f` was the only format this report used, and below 0.1 it shows fewer
    than two significant figures -- rendering as `0.00` every quantity between
    a floor (1e-4 and up, see `min_move_points`) and 0.005. That is how a
    verdict naming a direction came to sit beside "median gap delta +0.00
    points" with four movers whose deltas all printed as `+0.00 / 0.00`,
    leaving a reader unable to identify the instances that drove it.

    Two decimals where two decimals carry two significant figures, three
    significant figures where they would not. An exact zero keeps the familiar
    `+0.00`, because there it is the truth rather than a rounding of one -- and
    a NEGATIVE exact zero is normalised into it, so an underflowed difference
    cannot print as `-0.00` and read as a direction.
    """
    if value == 0.0:
        return "+0.00" if signed else "0.00"
    if abs(value) >= 0.1:
        return f"{value:+.2f}" if signed else f"{value:.2f}"
    return f"{value:+.3g}" if signed else f"{value:.3g}"


#: Per-instance comparison buckets. Every compared instance lands in exactly one.
BOTH_FEASIBLE = "both-feasible"
ARM_ONLY_FEASIBLE = "arm-only-feasible"
CONTROL_ONLY_FEASIBLE = "control-only-feasible"
NEITHER_FEASIBLE = "neither-feasible"
NO_COMPARABLE_GAP = "no-comparable-gap"
#: One side recorded no completed run at all -- every row there crashed, or the
#: runner wrote one of `NO_SEARCH_NOTES`. There is nothing to compare, in either
#: direction: an absence of runs is not an absence of feasibility.
NO_RUNS_RECORDED = "no-runs-recorded"

BUCKETS = (
    BOTH_FEASIBLE,
    ARM_ONLY_FEASIBLE,
    CONTROL_ONLY_FEASIBLE,
    NEITHER_FEASIBLE,
    NO_COMPARABLE_GAP,
    NO_RUNS_RECORDED,
)


@dataclass(frozen=True)
class RunRow:
    """One completed solve, as `results.csv` records it."""

    instance: str
    arm: str
    seed: int
    feasible: bool
    gap: float
    #: The published best-known primal value this row's gap was computed
    #: against. Read only to size the floor: `safe_gap` divides by it, so the
    #: same objective difference is a wildly different number of gap points at
    #: different scales. Already a recorded column -- nothing new is measured.
    primal_bks: float
    lns_repairs: float
    #: Of those repairs, the ones LNS actually KEPT (#150). Read alongside
    #: `lns_repairs` because the two answer different questions: the attempt
    #: count says LNS spent budget, this one says whether any of it bought a
    #: better search state. NaN carries the same "no reading" meaning here as
    #: everywhere else -- a row where no solve ran has neither number.
    lns_repairs_accepted: float
    wall: float
    #: The runner's note cell. Read because a row the DRIVER wrote for a process
    #: that crashed carries `feasible=false` like any infeasible run, and
    #: nothing else distinguishes them: three segfaults on one instance under
    #: one arm would otherwise be scored as that arm losing feasibility there.
    note: str = ""

    @property
    def runner_failed(self) -> bool:
        """Whether this row records a crashed process rather than a solve."""
        return self.note.startswith(RUNNER_FAILED_NOTE)

    @property
    def no_search(self) -> bool:
        """Whether NO SEARCH COMPLETED on this row, crash or exit-0 alike.

        A superset of `runner_failed`. The two are counted separately in the
        report because the reader needs to know which happened, but they are
        held out of the scoring identically: neither is a measurement.

        Decided by the ALLOWLIST, not by a list of failures (#153): a note this
        module has never heard of is not a measurement of anything, so it is
        held out and disclosed rather than read as a lost feasibility.
        """
        return not completed_search(self.note)


def _number(text: str | None) -> float:
    """A CSV cell as a float, with anything unparseable reading as NaN.

    "NaN" is what the runner writes for a cell it has no value for, and an empty
    cell means the same thing; neither may become a 0 that an aggregate would
    then treat as a measurement.
    """
    try:
        return float(text) if text is not None and text.strip() else math.nan
    except ValueError:
        return math.nan


def load_rows(path: Path) -> list[RunRow]:
    """Every row of a campaign's `results.csv`."""
    with path.open(newline="") as fh:
        return [
            RunRow(
                instance=row["instance"],
                arm=row["arm"],
                seed=int(row["seed"]),
                feasible=row["feasible"] == "true",
                gap=_number(row.get("gap_to_bks%")),
                primal_bks=_number(row.get("primal_bks")),
                lns_repairs=_number(row.get("lns_repairs")),
                lns_repairs_accepted=_number(row.get("lns_repairs_accepted")),
                wall=_number(row.get("wall_seconds")),
                note=row.get("note") or "",
            )
            for row in csv.DictReader(fh)
        ]


@dataclass(frozen=True)
class Cell:
    """One arm's runs on one instance."""

    instance: str
    arm: str
    runs: int
    feasible_runs: int
    #: Finite gaps of the feasible runs -- the only values a mean may be taken of.
    gaps: tuple[float, ...]
    repairs: tuple[float, ...]
    #: Rows whose runner process crashed. Excluded from `runs` and
    #: `feasible_runs` so they cannot read as a lost feasibility, and surfaced
    #: in the report so they are not silently dropped either.
    failed_runs: int = 0
    #: Rows the RUNNER wrote for a solve that never completed -- `solve-error`
    #: and the rest of `NO_SEARCH_NOTES`, plus any note this module does not
    #: recognise. Held out exactly as `failed_runs` is, and counted separately
    #: only so the report can say which of the two happened.
    no_search_runs: int = 0
    #: What those rows' notes collapse to under `no_search_label`, in
    #: `note_order`. The distinction is not cosmetic: `not-found` and
    #: `unsupported` are roster problems that hit every arm alike, where
    #: `solve-error` can depend on the configuration and so is a property OF THE
    #: ARM -- and an `unrecognised-note(...)` entry says the scorer held a row
    #: out because it could not tell which of those it was.
    no_search_notes: tuple[str, ...] = ()
    #: The instance's published primal bound, for sizing the floor. NaN where no
    #: row recorded one.
    primal_bks: float = math.nan
    #: The `repairs` entries' accepted halves, in the same row order. Defaulted
    #: for the same reason `seed_gaps` is: only the repair line reads it, and a
    #: hand-built cell that says nothing about acceptance contributes nothing to
    #: that line -- which the line then reports as "no accepted reading", not as
    #: a fabricated zero. See `_repair_cell`.
    repairs_accepted: tuple[float, ...] = ()
    #: The same gaps keyed by seed, so a same-seed comparison is possible. The
    #: drift check needs it: averaging the seed away first is what turned an
    #: earlier drift line into a measurement of seed variance. Defaulted because
    #: only that check reads it, and every other construction site would
    #: otherwise carry a field it never uses.
    seed_gaps: dict[int, float] = field(default_factory=dict)

    @property
    def mean_gap(self) -> float | None:
        return statistics.fmean(self.gaps) if self.gaps else None


def build_cells(rows: Iterable[RunRow]) -> dict[tuple[str, str], Cell]:
    """Group runs into `(instance, arm)` cells."""
    grouped: dict[tuple[str, str], list[RunRow]] = {}
    for row in rows:
        grouped.setdefault((row.instance, row.arm), []).append(row)
    return {
        key: Cell(
            instance=key[0],
            arm=key[1],
            # Rows where no search completed are not runs. Counting them would
            # put an arm that segfaulted -- or an arm whose configuration
            # provoked an exception, which exits 0 and looks well-formed -- into
            # `control-only-feasible`, the bucket that means "this arm lost
            # feasibility here", which is a claim about the search rather than
            # about the process table.
            runs=sum(1 for r in group if not r.no_search),
            feasible_runs=sum(1 for r in group if r.feasible and not r.no_search),
            failed_runs=sum(1 for r in group if r.runner_failed),
            no_search_runs=sum(1 for r in group if r.no_search and not r.runner_failed),
            no_search_notes=tuple(
                sorted(
                    {no_search_label(r.note) for r in group if r.no_search and not r.runner_failed},
                    key=note_order,
                )
            ),
            primal_bks=next((r.primal_bks for r in group if math.isfinite(r.primal_bks)), math.nan),
            gaps=tuple(
                r.gap for r in group if r.feasible and not r.no_search and math.isfinite(r.gap)
            ),
            repairs=tuple(r.lns_repairs for r in group if not r.no_search),
            repairs_accepted=tuple(r.lns_repairs_accepted for r in group if not r.no_search),
            seed_gaps={
                r.seed: r.gap
                for r in group
                if r.feasible and not r.no_search and math.isfinite(r.gap)
            },
        )
        for key, group in grouped.items()
    }


@dataclass(frozen=True)
class Comparison:
    """One instance's verdict for one arm against the control."""

    instance: str
    arm: str
    bucket: str
    control: Cell
    treatment: Cell
    #: arm mean gap minus control mean gap, in gap points. POSITIVE MEANS WORSE:
    #: `gap_to_bks%` is a distance from the best known solution. None unless the
    #: bucket is `both-feasible`.
    delta: float | None
    #: Feasible-run count difference, which is a result in its own right on the
    #: instances where no gap comparison exists. **None where either side
    #: recorded no run at all**: a difference taken against an absence is the
    #: same manufactured claim `classify` refuses to bucket, and leaving a
    #: number here would leave defect 1 sitting in the data model for the next
    #: summariser to find.
    feasibility_delta: int | None


def classify(control: Cell, treatment: Cell) -> str:
    """Which bucket an instance falls in for this arm."""
    if control.runs == 0 or treatment.runs == 0:
        # NO RUNS IS NOT NO FEASIBILITY, and this test has to come first.
        # `feasible_runs == 0` is true of a cell whose every row crashed or
        # errored, so deciding the bucket from it manufactures a verdict out of
        # a process table: three segfaults on the arm read as
        # `control-only-feasible` ("this arm lost feasibility here") and three
        # on the CONTROL read as `arm-only-feasible` -- an arm win produced by
        # three crashes. Neither claim is measured, so neither is made.
        return NO_RUNS_RECORDED
    if control.feasible_runs == 0 and treatment.feasible_runs == 0:
        return NEITHER_FEASIBLE
    if control.feasible_runs == 0:
        return ARM_ONLY_FEASIBLE
    if treatment.feasible_runs == 0:
        return CONTROL_ONLY_FEASIBLE
    if not control.gaps or not treatment.gaps:
        # Feasible on both sides, but at least one has no finite gap -- the
        # instance has no published bound, or the objective was not finite there.
        return NO_COMPARABLE_GAP
    return BOTH_FEASIBLE


def compare(control: Cell, treatment: Cell) -> Comparison:
    """One instance's comparison, with a delta only where one is defined."""
    bucket = classify(control, treatment)
    control_mean = control.mean_gap
    treatment_mean = treatment.mean_gap
    delta: float | None = None
    if bucket == BOTH_FEASIBLE and control_mean is not None and treatment_mean is not None:
        delta = treatment_mean - control_mean
    return Comparison(
        instance=control.instance,
        arm=treatment.arm,
        bucket=bucket,
        control=control,
        treatment=treatment,
        delta=delta,
        feasibility_delta=(
            None if bucket == NO_RUNS_RECORDED else treatment.feasible_runs - control.feasible_runs
        ),
    )


@dataclass(frozen=True)
class NoiseFloor:
    """The measured noise floor for one arm's comparison against the control."""

    per_instance: dict[str, float]
    #: Instances whose control supplied a real `s_i`, and those whose control
    #: gave fewer than two comparable runs so no floor could be measured. The
    #: second group is NOT scored -- see the module docstring for why imputing
    #: onto it manufactures results on this roster.
    measured: int
    unmeasured: int
    #: Measured instances whose Student band came out below their own
    #: `min_move_points` and were raised to it. Reported, because "the floor is
    #: measured" stops being the whole truth for them.
    floored: int
    #: The typical per-instance floor -- the headline number to quote.
    median_floor: float
    #: The median control standard deviation across the campaign. Not restricted
    #: to the instances that got a floor: it describes the run-to-run noise of
    #: the sitting, which is what a reader wants beside the typical floor.
    median_spread: float


def control_spreads(
    cells: dict[tuple[str, str], Cell], instances: Sequence[str]
) -> dict[str, float]:
    """Per-instance control standard deviation, where two or more runs allow one.

    Computed over every instance the control covers, not only the compared ones,
    so the median quoted in the report describes the campaign's run-to-run noise
    rather than whichever subset happened to be comparable. Nothing BORROWS it:
    an instance with no spread of its own is not scored at all -- see the module
    docstring for why imputing onto this roster manufactures results.
    """
    spreads: dict[str, float] = {}
    for instance in instances:
        control = cells.get((instance, CONTROL_ARM))
        if control is None or len(control.gaps) < 2:
            continue
        spread = statistics.stdev(control.gaps)
        # A spread of exactly zero is NOT a measurement of "no noise" -- it is
        # three seeds that happened to land on the same value, which on this
        # roster is the normal state of an instance solved to its bound. The
        # published table has four instances at gap exactly 0 and around
        # fourteen more within 1e-7. Scored with a floor of 0.0, any nonzero
        # delta clears it, so a 1e-9 move became "outside the measured floor"
        # and, with the unanimity rule below, a roster-wide verdict. Treated as
        # unmeasured instead: counted, listed, not scored.
        if spread > 0.0:
            spreads[instance] = spread
    return spreads


def noise_floor(comparisons: Sequence[Comparison], spreads: dict[str, float]) -> NoiseFloor:
    """The floor for this arm, from the control's own across-seed spread.

    One floor per instance and no aggregate floor: an aggregate would only ever
    be read against an aggregate effect, and on a roster whose gaps span six
    orders of magnitude the only aggregate worth quoting is a rank statistic,
    which has no gap-point floor to be compared with.
    """
    compared = [c for c in comparisons if c.bucket == BOTH_FEASIBLE]
    per_instance: dict[str, float] = {}
    measured = unmeasured = floored = 0
    for comparison in compared:
        spread = spreads.get(comparison.instance)
        if spread is None:
            # No measured spread, so no floor. Not imputed: see the module
            # docstring. The instance is counted and listed, never scored.
            unmeasured += 1
            continue
        measured += 1
        scale = math.sqrt(1.0 / len(comparison.treatment.gaps) + 1.0 / len(comparison.control.gaps))
        band = t_multiplier(len(comparison.control.gaps) - 1) * spread * scale
        # A floor must be meaningful in the UNITS OF THE THING IT BOUNDS, and a
        # measured spread finer than the runner's own tie band is not one. See
        # `min_move_points`.
        reference = comparison.control.primal_bks
        if not math.isfinite(reference):
            reference = comparison.treatment.primal_bks
        smallest = min_move_points(reference)
        if band < smallest:
            floored += 1
            band = smallest
        per_instance[comparison.instance] = band
    return NoiseFloor(
        per_instance=per_instance,
        measured=measured,
        unmeasured=unmeasured,
        floored=floored,
        median_floor=statistics.median(per_instance.values()) if per_instance else math.nan,
        median_spread=statistics.median(spreads.values()) if spreads else math.nan,
    )


@dataclass(frozen=True)
class ArmSummary:
    """One arm's whole story: buckets, aggregate effect, floor and verdict."""

    arm: str
    comparisons: tuple[Comparison, ...]
    counts: dict[str, int]
    #: Printed, never used for the verdict, and always beside `largest_delta`
    #: -- see the module docstring on why a mean over this roster is one
    #: instance's number wearing the roster's name.
    mean_delta: float | None
    #: The single largest |delta| and the instance it belongs to, printed beside
    #: the mean so a reader can see at once whether the mean describes the
    #: roster. NOT a share: `max|d| / sum|d|` is bounded in [1/n, 1] and so
    #: reads as reassuring exactly when mixed signs cancel -- deltas of +100
    #: and -99 give a mean of +0.5 and a "50%" share, when that instance
    #: contributes two hundred times the mean. The raw number cannot mislead
    #: that way.
    largest_delta: float
    largest_delta_instance: str
    median_delta: float | None
    #: Median delta over the MOVERS alone. Quoted whenever a direction is named,
    #: so that "the arm is WORSE" always carries a number that is nonzero by
    #: construction: every mover exceeds its own floor, which is at least
    #: `MIN_MOVE_POINTS`. The roster median is quoted beside it -- it can be a
    #: true `+0.00` when most scored instances did not move, and saying so is
    #: the point -- but it is no longer the only figure the sentence offers.
    mover_median_delta: float | None
    #: Median of the MOVERS' OWN floors. Quoted beside the roster's typical
    #: floor whenever anything moved, because those two can differ by orders of
    #: magnitude on this roster and #151 was filed on a report that printed only
    #: the first: "typical per-instance floor +/-3.51 points" while the four
    #: instances that produced the verdict had floors of ~3.5e-7.
    mover_median_floor: float | None
    #: Instances whose delta exceeds THEIR OWN floor, by direction.
    moved_worse: int
    moved_better: int
    #: Exact two-sided sign-test p-value over those two counts.
    sign_p: float
    feasibility_delta: int
    #: How many instances the balanced delta above is summed over -- the
    #: denominator that makes a `+0` on it readable as "no difference over N"
    #: rather than as "no difference" over an empty set.
    feasibility_balanced: int
    #: Feasible-run delta restricted to instances with equal run counts on both
    #: sides, and how many instances that leaves out. An unequal-`k` instance
    #: (the in-progress one on an interrupted campaign) otherwise contributes a
    #: difference that is bookkeeping rather than a result.
    feasibility_delta_balanced: int
    feasibility_unbalanced: int
    floor: NoiseFloor
    #: Scored instances with rows on only one side, so in no bucket at all. An
    #: interrupted campaign always ends mid-instance-block, so its last instance
    #: has control rows and not every arm's. It has to be visible: without it the
    #: bucket counts sit quietly short of the "instances scored" line above, and
    #: this module's own promise that every instance lands in exactly one bucket
    #: is false for precisely the instances a reader would want to ask about.
    uncompared: int = 0

    @property
    def comparable(self) -> int:
        """Instances with a delta at all -- the denominator of `mean_delta`.

        Distinct from `scored`, which is the denominator of `median_delta`, and
        the two were printed as parallel lines with neither naming its own. Ten
        scored instances at delta 0 beside ten unscored at +900 prints a median
        of +0.00 and a mean of +450.00, and a reader with no denominators cannot
        tell that they are statistics over different sets.
        """
        return sum(1 for c in self.comparisons if c.delta is not None)

    @property
    def scored(self) -> int:
        """Instances with both a delta and a measured floor to judge it by."""
        return self.moved_worse + self.moved_better + self.held

    @property
    def held(self) -> int:
        """Scored instances whose delta stayed inside their own floor."""
        return sum(
            1
            for c in self.comparisons
            if c.delta is not None
            and c.instance in self.floor.per_instance
            and abs(c.delta) <= self.floor.per_instance[c.instance]
        )

    @property
    def verdict(self) -> str:
        """The arm's effect, judged instance by instance and then by sign test.

        Never from the mean: this roster's gaps span six orders of magnitude, so
        an unweighted mean is the largest instance's seed draw. See the module
        docstring.
        """
        comparable = sum(1 for c in self.comparisons if c.delta is not None)
        if comparable == 0:
            return "no instance is comparable on gap; read the feasibility counts instead"
        if not self.floor.per_instance:
            # Comparable, but nothing to judge the comparison BY. Distinct from
            # the line above and it has to say so: "no instance is comparable"
            # would be false here and would send a reader to the feasibility
            # counts for an arm whose gaps are all present.
            return (
                f"{comparable} instance(s) are comparable on gap, but none has a measurable "
                "noise floor -- every control either produced fewer than two comparable runs "
                "or returned an identical gap on every seed, which is not a measurement that "
                "there is no noise. The effect cannot be called"
            )
        if self.median_delta is None:
            return "no scored instance has a delta; read the buckets below"
        band = format_points(self.floor.median_floor, signed=False)
        typical = f"typical per-instance floor +/-{band} points"
        moved = self.moved_worse + self.moved_better
        if moved == 0:
            return (
                f"INSIDE THE NOISE: none of {self.scored} scored instance(s) moved outside its "
                f"own measured floor ({typical}); median gap delta over those {self.scored} "
                f"instance(s) {format_points(self.median_delta)} points"
            )
        # THE DELTA A NAMED DIRECTION IS A CLAIM ABOUT IS THE MOVERS' OWN.
        # The roster-wide median is legitimately +0.00 whenever most scored
        # instances held inside their floors, and printing it in the same
        # sentence as "the arm is WORSE than the control" is exactly the
        # contradiction issue #151 forbids -- a verdict naming a direction may
        # not coexist with a median delta of +0.00. It is not suppressed, only
        # moved: `median per-instance gap delta` in the report below still
        # reports it over the scored instances, with its denominator named.
        #
        # The movers' median CANNOT be zero here, which is what makes this a
        # fix rather than a relabelling: every mover's |delta| exceeds its own
        # floor, which is itself bounded below by the runner's tie band, and a
        # direction is named only when the movers are unanimous or the sign
        # test resolves -- either way more than half of them share a sign, so
        # the median sits on that side.
        mover_delta = self.mover_median_delta
        summary = (
            f"{moved} of {self.scored} scored instance(s) moved outside their own floor "
            f"({self.moved_worse} worse, {self.moved_better} better; sign test p = "
            f"{self.sign_p:.3f}, {typical}); median gap delta over the {moved} mover(s) "
            f"{format_points(mover_delta if mover_delta is not None else 0.0)} points against "
            f"their own median floor "
            f"+/-{format_points(self.mover_median_floor or 0.0, signed=False)}; the other "
            f"{self.held} scored instance(s) held inside their own floor"
        )
        # A roster-level direction needs either a significant sign test or
        # enough unanimous movers to be worth the name. One mover is not a
        # roster verdict: an earlier cut named a direction off a single
        # instance, printing "the arm is WORSE than the control" in the same
        # sentence as "sign test p = 1.000" and "median gap delta +0.00".
        # MIN_UNANIMOUS_MOVERS unanimous movers give p = 0.125 -- short of
        # significance, which is why the line says so rather than claiming it.
        unanimous = self.moved_worse == 0 or self.moved_better == 0
        if not unanimous and self.sign_p > 0.05:
            return f"MIXED, no consistent direction: {summary}"
        if not unanimous or moved >= MIN_UNANIMOUS_MOVERS or self.sign_p <= 0.05:
            direction = "WORSE than" if self.moved_worse > self.moved_better else "BETTER than"
            qualifier = "" if self.sign_p <= 0.05 else " (unanimous, but too few movers for the "
            if qualifier:
                qualifier += "sign test to resolve)"
            return f"the arm is {direction} the control{qualifier}: {summary}"
        return (
            f"ISOLATED MOVERS, not a roster-level effect: {summary}. Fewer than "
            f"{MIN_UNANIMOUS_MOVERS} instances moved; read them individually below"
        )


def summarize_arm(
    arm: str, cells: dict[tuple[str, str], Cell], instances: Sequence[str]
) -> ArmSummary:
    """Compare one arm against the control over `instances`."""
    comparisons: list[Comparison] = []
    uncompared = 0
    for instance in instances:
        control = cells.get((instance, CONTROL_ARM))
        treatment = cells.get((instance, arm))
        if control is None or treatment is None:
            uncompared += 1
            continue
        comparisons.append(compare(control, treatment))
    deltas = [c.delta for c in comparisons if c.delta is not None]
    floor = noise_floor(comparisons, control_spreads(cells, instances))
    moved_worse = moved_better = 0
    for comparison in comparisons:
        band = floor.per_instance.get(comparison.instance)
        if comparison.delta is None or band is None:
            continue
        if comparison.delta > band:
            moved_worse += 1
        elif comparison.delta < -band:
            moved_better += 1
    # The mean is reported only alongside the largest single delta, so a reader
    # can see immediately whether it describes the roster.
    mean_delta = statistics.fmean(deltas) if deltas else None
    scored_deltas = [
        c.delta for c in comparisons if c.delta is not None and c.instance in floor.per_instance
    ]
    movers = [
        c
        for c in comparisons
        if c.delta is not None
        and c.instance in floor.per_instance
        and abs(c.delta) > floor.per_instance[c.instance]
    ]
    mover_deltas = [c.delta for c in movers if c.delta is not None]
    mover_floors = [floor.per_instance[c.instance] for c in movers]
    largest = max(
        ((abs(c.delta), c.instance) for c in comparisons if c.delta is not None),
        default=(math.nan, ""),
    )
    balanced = [
        c
        for c in comparisons
        if c.bucket != NO_RUNS_RECORDED
        and c.control.runs == c.treatment.runs
        and c.control.runs > 0
    ]
    return ArmSummary(
        arm=arm,
        comparisons=tuple(comparisons),
        counts={bucket: sum(1 for c in comparisons if c.bucket == bucket) for bucket in BUCKETS},
        mean_delta=mean_delta,
        largest_delta=largest[0],
        largest_delta_instance=largest[1],
        # The MEDIAN quoted in the verdict is over the SCORED instances only,
        # so the sentence "none of N scored instances moved" cannot sit beside
        # a median drawn from a wider set that includes an unscored +900.
        median_delta=statistics.median(scored_deltas) if scored_deltas else None,
        mover_median_delta=statistics.median(mover_deltas) if mover_deltas else None,
        mover_median_floor=statistics.median(mover_floors) if mover_floors else None,
        moved_worse=moved_worse,
        moved_better=moved_better,
        sign_p=sign_test_p(moved_worse, moved_better),
        # A cell with no completed run on one side contributes no feasibility
        # comparison EITHER WAY -- summing its `feasible_runs` difference is the
        # same manufactured claim `classify` refuses to bucket.
        feasibility_delta=sum(
            c.feasibility_delta for c in comparisons if c.feasibility_delta is not None
        ),
        feasibility_delta_balanced=sum(
            c.feasibility_delta for c in balanced if c.feasibility_delta is not None
        ),
        feasibility_balanced=len(balanced),
        feasibility_unbalanced=sum(1 for c in comparisons if c.bucket != NO_RUNS_RECORDED)
        - len(balanced),
        floor=floor,
        uncompared=uncompared,
    )


def scored_instances(rows: Sequence[RunRow]) -> list[str]:
    """Roster order, minus the instances published as documented failures.

    `elec25`/`elec50` are excluded from every quality claim under epic #87. They
    are reported separately rather than dropped: an arm that changes them is
    still a finding, just not one that can be folded into an aggregate.
    """
    seen: list[str] = []
    for row in rows:
        if row.instance not in seen and row.instance not in CLAIM_EXCLUDED:
            seen.append(row.instance)
    return seen


def drift_check(cells: dict[tuple[str, str], Cell], instances: Sequence[str]) -> str:
    """The probe against the campaign control -- one configuration, two times.

    This is the only direct measurement of machine drift the campaign contains,
    and it is why the probe's rows are kept rather than thrown away: the two sets
    are the same configuration run hours apart, so their disagreement is drift
    plus seed-free run-to-run noise, and it can be read against the same floor
    the arms are judged by.
    """
    pairs: list[float] = []
    for instance in instances:
        probe = cells.get((instance, PROBE_ARM_NAME))
        control = cells.get((instance, CONTROL_ARM))
        if probe is None or control is None:
            continue
        # SAME SEED ON BOTH SIDES, and this is what makes the line a drift
        # measurement rather than a restatement of seed variance. The probe is
        # one seed; comparing it against the control's mean over all three
        # would differ by ~s*sqrt(2/3) from seeds alone, which is not drift and
        # which an earlier cut printed as though it were, under a closing
        # sentence telling the reader to discount arm effects smaller than it.
        shared = set(probe.seed_gaps) & set(control.seed_gaps)
        for seed in sorted(shared):
            pairs.append(abs(probe.seed_gaps[seed] - control.seed_gaps[seed]))
    if not pairs:
        return (
            "drift check: no instance has a probe row and a control row at the same seed, "
            "so this campaign measured no drift"
        )
    return (
        f"drift check: {len(pairs)} same-seed pair(s) ran the control configuration twice at "
        f"different points in the sitting (gate probe vs campaign control). Mean absolute gap "
        f"difference {format_points(statistics.fmean(pairs), signed=False)} points, median "
        f"{format_points(statistics.median(pairs), signed=False)}, "
        f"max {format_points(max(pairs), signed=False)}. Same configuration, same seed, "
        "hours apart: the difference is machine drift. It is descriptive -- no verdict above "
        "is computed from it."
    )


def _repair_totals(cells: Iterable[Cell]) -> tuple[float, float]:
    """LNS destroy-repairs summed over cells: attempted, then accepted.

    A NaN is skipped rather than read as zero: the runner writes it on a row
    where no solve completed, and "no reading" is not "no repairs" -- that is
    the same distinction the gate turns on. The two sums are taken
    independently for that reason; they are not a count and a subset of the
    same rows once unreadable cells are dropped from each. (So a row that
    somehow carried one reading and not the other could make accepted exceed
    attempted. The runner cannot write such a row -- `LnsCells` gives a row
    both cells or neither -- and the alternative, dropping a row from both
    sums, would turn a whole missing COLUMN into "0 attempted, 0 accepted",
    which is the louder lie.)

    A side on which NOTHING carried a reading totals NaN, not 0.0 -- see
    `_repair_cell`. `math.fsum(())` is 0.0, and 0 here is the strongest claim
    either counter can make.
    """
    materialised = list(cells)
    attempted = [r for cell in materialised for r in cell.repairs if math.isfinite(r)]
    accepted = [r for cell in materialised for r in cell.repairs_accepted if math.isfinite(r)]
    return (
        math.fsum(attempted) if attempted else math.nan,
        math.fsum(accepted) if accepted else math.nan,
    )


def _repair_cell(total: float, noun: str) -> str:
    """One half of the repair line, or the fact that nothing measured it.

    NaN is not zero. It is a side on which no row carried that counter at all --
    every campaign written before the column existed, which `--report-only`
    re-scores from its own `results.csv`. `lns_repairs` predates
    `lns_repairs_accepted` by seven issues, so the two columns went missing at
    different times and both cases are live. Printing 0 for either would
    publish the strongest claim the counter can make ("LNS repaired nothing",
    "LNS kept nothing") out of a file that never measured it.
    """
    return f"no {noun} reading" if math.isnan(total) else f"{total:g} {noun}"


def _instance_lines(summary: ArmSummary) -> list[str]:
    lines = [
        f"  {'instance':<22} {'ctl feas':>8} {'ctl gap':>11} {'arm feas':>8} "
        f"{'arm gap':>11} {'delta':>11} {'floor':>10}  bucket"
    ]
    for comparison in summary.comparisons:
        control_mean = comparison.control.mean_gap
        treatment_mean = comparison.treatment.mean_gap
        floor = summary.floor.per_instance.get(comparison.instance, math.nan)
        lines.append(
            f"  {comparison.instance:<22} "
            f"{comparison.control.feasible_runs:>3}/{comparison.control.runs:<4} "
            f"{'-' if control_mean is None else format_points(control_mean, signed=False):>11} "
            f"{comparison.treatment.feasible_runs:>3}/{comparison.treatment.runs:<4} "
            f"{'-' if treatment_mean is None else format_points(treatment_mean, signed=False):>11} "
            f"{'-' if comparison.delta is None else format_points(comparison.delta):>11} "
            f"{'-' if math.isnan(floor) else format_points(floor, signed=False):>10}  "
            f"{comparison.bucket}"
        )
    return lines


def _floor_lines(summary: ArmSummary) -> list[str]:
    floor = summary.floor
    if not floor.per_instance:
        # The reason is TWO-PART, and the verdict on the next line has said so
        # since the exact-zero-spread guard went in: an instance is unmeasurable
        # either because its control produced fewer than two comparable runs OR
        # because every seed returned an identical gap. This string named only
        # the first, and so contradicted the verdict beneath it on a campaign
        # where every control had three feasible runs with finite gaps.
        return [
            "  noise floor: not measurable -- every control either produced fewer than two "
            "comparable runs or returned an identical gap on every seed"
        ]
    lines = [
        f"  noise floor: typical per-instance "
        f"+/-{format_points(floor.median_floor, signed=False)} points",
        "    from the control's own across-seed spread (median s = "
        f"{format_points(floor.median_spread, signed=False)} "
        f"gap points, two-sided 95% Student band); {floor.measured} instance(s) measured",
    ]
    if floor.floored:
        lines.append(
            f"    {floor.floored} instance(s) measured a band finer than the runner's own tie "
            "band for their published bound, and were raised to it: a control that returned "
            "~1e-7 on every seed measures a spread, but not one the gap it bounds can resolve. "
            "That bound is the only assumed number here -- see `min_move_points`"
        )
    if floor.unmeasured:
        lines.append(
            f"    {floor.unmeasured} comparable instance(s) NOT SCORED: their control either "
            "produced fewer than two comparable runs or returned an identical gap on every "
            "seed, so no floor could be measured. Their deltas are listed below with a '-' "
            "floor. Imputing one from other instances would be unsound here -- the roster's "
            "gaps span six orders of magnitude, so a borrowed ~1-point spread lent to a "
            "1e6-point instance is a floor it clears automatically."
        )
    return lines


def _excluded_lines(cells: dict[tuple[str, str], Cell], arms: Sequence[str]) -> list[str]:
    lines: list[str] = []
    for instance in CLAIM_EXCLUDED:
        for arm in arms:
            cell = cells.get((instance, arm))
            if cell is None:
                continue
            mean = cell.mean_gap
            lines.append(
                f"  {instance:<10} {arm:<16} feasible {cell.feasible_runs}/{cell.runs}  "
                f"gap {'-' if mean is None else format_points(mean, signed=False)}"
            )
    return lines or ["  (no rows)"]


def render_report(results: Path, gate: dict[str, object] | None = None) -> str:
    """The whole campaign report, as text."""
    rows = load_rows(results)
    if not rows:
        return f"{results} holds no rows"
    cells = build_cells(rows)
    instances = scored_instances(rows)
    arms = [a for a in dict.fromkeys(r.arm for r in rows) if a not in (CONTROL_ARM, PROBE_ARM_NAME)]
    # Rows where no search completed carry a 0.0 wall (`write_preread_row` and
    # the `solve-error` `write_unsolved_row` both pass one), so averaging them in
    # drags the headline toward zero while every other line insists they record
    # nothing.
    walls = [r.wall for r in rows if math.isfinite(r.wall) and not r.no_search]
    mean_wall = statistics.fmean(walls) if walls else math.nan
    lines = [
        "=== MINLPLib ablation campaign (issue #143) ===",
        f"rows recorded:        {len(rows)} "
        f"({sum(1 for r in rows if not r.no_search)} completed a search)",
        f"instances scored:     {len(instances)} "
        f"(excluding {', '.join(CLAIM_EXCLUDED)}, published as documented failures)",
        f"seeds:                {sorted({r.seed for r in rows})}",
        f"arms:                 {', '.join([CONTROL_ARM, *arms])}",
        "mean wall per run:    "
        + (
            f"{mean_wall:.1f}s"
            if math.isfinite(mean_wall)
            else "not measured (no row completed a search)"
        ),
        "",
        "Sign convention: delta = arm mean gap-to-BKS minus control mean gap-to-BKS,",
        "in gap points. POSITIVE IS WORSE. Only `both-feasible` instances have one.",
        "",
    ]
    if gate is not None:
        lines += [
            f"LNS gate: {'ran the arm' if gate.get('run_arm') else 'skipped the arm'}",
            f"  {gate.get('reason')}",
            "",
        ]
    lines += [drift_check(cells, instances), ""]
    for arm in arms:
        summary = summarize_arm(arm, cells, instances)
        lines.append(f"--- {arm} ---")
        lines.append("  " + "  ".join(f"{k}={v}" for k, v in summary.counts.items()))
        # The denominator is printed even when it is zero. `+0 over instances
        # with equal run counts` is unreadable on an all-held-out campaign,
        # where the honest reading is "+0 over 0 instances" -- a sum over an
        # empty set, not a measurement that the arm changed nothing.
        lines.append(
            f"  feasible-run delta over the roster: {summary.feasibility_delta_balanced:+d} "
            f"over the {summary.feasibility_balanced} instance(s) with equal run counts on "
            "both sides"
            + (
                ""
                if not summary.feasibility_unbalanced
                else f" ({summary.feasibility_unbalanced} instance(s) left out for unequal run "
                f"counts; counting them gives {summary.feasibility_delta:+d}, which is "
                "bookkeeping on run counts that do not correspond -- either an incomplete "
                "campaign or rows held out above -- rather than a result)"
            )
        )
        # Issue #143 asks for the repair counts wherever the LNS arm is RUN, not
        # only for the reading that would justify skipping it. Both sides are
        # printed for every arm: the control's is the campaign's own answer to
        # "is LNS doing work at this budget", measured over all three seeds
        # rather than the gate probe's one.
        #
        # The accepted half (#150) is printed beside it because the attempt
        # count alone cannot separate "LNS is working" from "LNS is spending".
        # It is reported and not gated on: a rejected repair still randomises,
        # still repairs and still costs seconds, so zero acceptances does NOT
        # make an LNS arm equivalent to a no-LNS one -- see run_ablation.py's
        # LNS_GATE_MIN_REPAIRS note.
        control_attempted, control_accepted = _repair_totals(c.control for c in summary.comparisons)
        arm_attempted, arm_accepted = _repair_totals(c.treatment for c in summary.comparisons)
        lines.append(
            "  LNS repairs over the roster: control "
            f"{_repair_cell(control_attempted, 'attempted')}, "
            f"{_repair_cell(control_accepted, 'accepted')}; "
            f"arm {_repair_cell(arm_attempted, 'attempted')}, "
            f"{_repair_cell(arm_accepted, 'accepted')}"
        )
        # SPLIT BY SIDE. The reason these rows are held out is that whether a
        # solve crashes or throws can depend on the configuration, which makes
        # the side the single most informative thing about them -- a total hides
        # exactly the arm property it is reporting.
        crashed_control = sum(c.control.failed_runs for c in summary.comparisons)
        crashed_arm = sum(c.treatment.failed_runs for c in summary.comparisons)
        if crashed_control or crashed_arm:
            lines.append(
                f"  {crashed_control + crashed_arm} run(s) crashed (control {crashed_control}, "
                f"arm {crashed_arm}) and are held out of every count above -- they record no "
                "measurement, so reading them as infeasible would score a process failure as an "
                "arm losing feasibility"
            )
        unsearched_control = sum(c.control.no_search_runs for c in summary.comparisons)
        unsearched_arm = sum(c.treatment.no_search_runs for c in summary.comparisons)
        observed = sorted(
            {
                note
                for c in summary.comparisons
                for note in c.control.no_search_notes + c.treatment.no_search_notes
            },
            key=note_order,
        )
        if unsearched_control or unsearched_arm:
            # The justification differs by note and the line has to say which
            # it is making. `solve-error` is the arm-dependent one -- whether a
            # solve throws can depend on the configuration -- while
            # `not-found`/`read-error`/`build-error`/`unsupported` hit every arm
            # identically and are roster problems, not arm properties. Printing
            # the arm-dependence argument under a list of `not-found` would be
            # the same overreach in miniature.
            arm_dependent = [n for n in observed if n == "solve-error"]
            unrecognised = [n for n in observed if n.startswith(UNRECOGNISED_NOTE)]
            roster_wide = [
                n for n in observed if n != "solve-error" and not n.startswith(UNRECOGNISED_NOTE)
            ]
            why = "; ".join(
                part
                for part in (
                    "whether a solve throws can depend on the arm, so scoring "
                    f"{', '.join(arm_dependent)} would read an exception as a lost feasibility"
                    if arm_dependent
                    else "",
                    f"{', '.join(roster_wide)} is a roster problem that hits every arm alike, "
                    "so scoring it would read a missing or unsupported instance as one"
                    if roster_wide
                    else "",
                    # The allowlist's whole point: a note nobody taught this
                    # module cannot be classified, so it is held out and SAID,
                    # rather than scored as an infeasibility on the strength of
                    # its `feasible=false` cell.
                    f"{', '.join(unrecognised)} is a note this report does not recognise, and it "
                    "scores only the notes a completed search is known to write -- so an "
                    "unfamiliar one is held out and named here rather than counted"
                    if unrecognised
                    else "",
                )
                if part
            )
            lines.append(
                f"  {unsearched_control + unsearched_arm} run(s) completed no search "
                f"(control {unsearched_control}, arm {unsearched_arm}; {', '.join(observed)}) and "
                "are held out of every count above. The row is well-formed and "
                f"`feasible=false`, so nothing but the note says so -- and {why}"
            )
        held_out = [c.instance for c in summary.comparisons if c.bucket == NO_RUNS_RECORDED]
        if held_out:
            lines.append(
                f"  {len(held_out)} instance(s) recorded no completed run on at least one side "
                f"and are in no feasibility bucket and in no feasible-run delta: "
                f"{', '.join(held_out)}"
            )
        if summary.uncompared:
            lines.append(
                f"  {summary.uncompared} scored instance(s) have rows on only one side and are "
                "in no bucket -- the campaign is incomplete for this arm"
            )
        if summary.median_delta is not None:
            lines.append(
                f"  median per-instance gap delta: {format_points(summary.median_delta)} points "
                f"over the {summary.scored} SCORED instance(s)"
            )
        if summary.mean_delta is not None:
            largest = (
                ""
                if math.isnan(summary.largest_delta)
                else "; largest single-instance |delta| "
                f"{format_points(summary.largest_delta, signed=False)} points "
                f"({summary.largest_delta_instance})"
            )
            lines.append(
                f"  mean per-instance gap delta: {format_points(summary.mean_delta)} points over "
                f"the {summary.comparable} COMPARABLE instance(s){largest} "
                "-- NOT the verdict statistic, see the module docstring"
            )
        lines += _floor_lines(summary)
        lines.append(f"  VERDICT: {summary.verdict}")
        lines.append("")
        lines += _instance_lines(summary)
        lines.append("")
    lines.append(
        "NOTE ON UNITS: `gap_to_bks%` is a percentage where the reference value is nonzero "
        "and an ABSOLUTE residual where it is not (see the runner's safe_gap). Per-instance "
        "floors and per-instance deltas are unit-consistent, so the moved/held verdict above "
        "is sound; the roster-level median and mean pool the two and are quoted as 'points' "
        "for want of a better word."
    )
    lines.append("")
    lines.append("--- instances excluded from every claim ---")
    lines += _excluded_lines(cells, [CONTROL_ARM, *arms])
    return "\n".join(lines)
