"""Shell-level smoke tests for the `cbls` CLI binary.

They run the built executable because the exit status and stderr a shell sees
are only observable there: an exception escaping `main` (a malformed flag's
`std::stod` did, pre-fix: SIGABRT, `what(): stod`, no flag named) is
indistinguishable in-process from a clean non-zero exit.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

CBLS_BINARY = Path(__file__).resolve().parents[2] / "build" / "cbls"
MODEL = Path(__file__).resolve().parents[2] / "examples" / "simple.cbls"

# Every flag in the CLI's option loop that parses its value as a number.
NUMERIC_FLAGS = [
    "--time-limit",
    "--seed",
    "--lns",
    "--lns-interval",
    "--threads",
]


def _run_cbls(*args: str) -> subprocess.CompletedProcess[str]:
    if not CBLS_BINARY.exists():
        pytest.skip("cbls not built")
    return subprocess.run([str(CBLS_BINARY), *args], capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("flag", NUMERIC_FLAGS)
def test_a_malformed_numeric_flag_is_reported_and_names_the_flag(flag: str) -> None:
    result = _run_cbls(flag, "abc")

    # subprocess reports death by signal as a negative returncode, so this bound
    # is what separates a reported error from an abort. Pre-fix this was -6.
    assert 0 < result.returncode < 128, f"{flag}: returncode {result.returncode}"
    assert "terminate" not in result.stderr, f"{flag}: {result.stderr}"
    assert flag in result.stderr, f"{flag}: {result.stderr}"
    # The value is named too: a flag fallen through to `unknown option` exits
    # the same and also names the flag.
    assert "'abc' is not a" in result.stderr, f"{flag}: {result.stderr}"


# Only the int-width flags carry a range check; the int64 and double flags accept
# whatever stoll/stod do.
INT_WIDTH_FLAGS = ["--lns-interval", "--threads"]


@pytest.mark.parametrize("flag", INT_WIDTH_FLAGS)
def test_a_value_wider_than_int_is_reported_as_out_of_range(flag: str) -> None:
    # Pre-fix, std::stoi's out_of_range escaped main like a malformed value; the
    # widened parse's explicit bound is covered only here.
    result = _run_cbls(flag, "2147483648")

    assert 0 < result.returncode < 128, f"{flag}: returncode {result.returncode}"
    assert f"{flag}: '2147483648' is out of range" in result.stderr, result.stderr


def test_well_formed_numeric_flags_are_all_accepted() -> None:
    # Valid values and no model file: reaching the model check proves every flag
    # parsed (an inverted int bound or a wrong uint64 conversion fails here).
    result = _run_cbls(
        "--time-limit",
        "0.1",
        "--seed",
        "7",
        "--threads",
        "2",
        "--lns",
        "0.3",
        "--lns-interval",
        "2",
    )

    assert result.returncode == 1, result.stderr
    assert "no model file specified" in result.stderr


def test_the_seed_the_cli_prints_can_be_parsed_back() -> None:
    # --seed is unsigned and echoed unsigned (`--seed -1` prints 2**64-1), so an
    # int64 parse could not read back the seed it printed. A real model runs
    # because the header that prints the seed needs one; that also catches a
    # formatter printing it signed.
    printed = _run_cbls(str(MODEL), "--seed", "-1", "--time-limit", "0.1")
    assert printed.returncode == 0, printed.stderr

    echoed = str(2**64 - 1)
    assert f"Seed: {echoed}" in printed.stdout, printed.stdout

    reparsed = _run_cbls(str(MODEL), "--seed", echoed, "--time-limit", "0.1")
    assert reparsed.returncode == 0, reparsed.stderr
    assert f"Seed: {echoed}" in reparsed.stdout, reparsed.stdout


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--threads", "2147483648"),
        ("--seed", "18446744073709551616"),
        ("--time-limit", "1e400"),
    ],
)
def test_an_out_of_range_value_is_reported_as_such_not_as_a_typo(flag: str, value: str) -> None:
    # A number too large to hold is not a mistyped one. Reporting it as "not an
    # integer" sends the reader hunting for a wrong digit that is not there.
    result = _run_cbls(flag, value)

    assert 0 < result.returncode < 128, result.stderr
    assert f"{flag}: '{value}' is out of range" in result.stderr, result.stderr


def test_a_nan_time_limit_is_rejected() -> None:
    # "nan" is a well-formed double that yields a solve that never searched, and
    # nothing downstream in the CLI guards it, so the parse rejects it.
    result = _run_cbls("--time-limit", "nan")

    assert 0 < result.returncode < 128, result.stderr
    assert "--time-limit: 'nan' is not a number" in result.stderr


# In the long-running benchmark runners a mistyped flag that parses to a default
# yields a result-shaped artifact, not an error (pre-fix, setcover's `strtoull`
# ran `--seed abc` over the whole roster at seed 0).
SETCOVER_BINARY = Path(__file__).resolve().parents[2] / "build" / "cbls_setcover"

SETCOVER_BAD_FLAGS = [
    ("--seed", "abc"),
    ("--seeds", "2x"),
    ("--time", "60s"),
    ("--time", "nan"),
    ("--structural-prob", "abc"),
    ("--structural-sample-size", "abc"),
]


@pytest.mark.parametrize(("flag", "value"), SETCOVER_BAD_FLAGS)
def test_setcover_refuses_a_malformed_numeric_flag(flag: str, value: str) -> None:
    if not SETCOVER_BINARY.exists():
        pytest.skip("cbls_setcover not built")
    # Parsing precedes any disk access, so this needs no instance data.
    result = subprocess.run(
        [str(SETCOVER_BINARY), flag, value], capture_output=True, text=True, timeout=60
    )

    assert 0 < result.returncode < 128, f"{flag} {value}: returncode {result.returncode}"
    assert f"{flag}: '{value}'" in result.stderr, result.stderr


def test_setcover_refuses_an_unknown_selection() -> None:
    """`--structural-selection` is a word with its own reject path: a silent
    fall-back to the default would publish rows naming an arm nobody ran."""
    if not SETCOVER_BINARY.exists():
        pytest.skip("cbls_setcover not built")
    result = subprocess.run(
        [str(SETCOVER_BINARY), "--structural-selection", "bogus"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert 0 < result.returncode < 128, result.stderr
    assert "unknown selection 'bogus'" in result.stderr, result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ["--structural-prob", "0", "--structural-selection", "violation_guided"],
        ["--structural-prob", "-0", "--structural-selection", "best_of_sample"],
        ["--structural-sample-size", "4"],  # inert under first_improving
    ],
)
def test_setcover_refuses_a_structural_flag_read_by_nothing(args: list[str]) -> None:
    """The shared table's no-effect rules, which setcover takes with its spellings
    (#198): a row must not record a policy no structural batch ever ran."""
    if not SETCOVER_BINARY.exists():
        pytest.skip("cbls_setcover not built")
    result = subprocess.run(
        [str(SETCOVER_BINARY), *args], capture_output=True, text=True, timeout=60
    )

    assert 0 < result.returncode < 128, result.stderr
    assert "read by nothing" in result.stderr or "no effect" in result.stderr, result.stderr


# The uc-chped runner writes a published table; its guards judge which file
# `--out` names, not whether it was given, so `--out <the published table>`
# spelled out (the README's form) is refused too. Pre-fix, a shortened run that
# way rewrote the table and dropped the unrun instances' reference rows, exit 0.
UC_CHPED_BINARY = Path(__file__).resolve().parents[2] / "build" / "cbls_uc_chped"


def _uc_chped_scratch(tmp_path: Path) -> Path:
    src = Path(__file__).resolve().parents[2] / "benchmarks" / "instances" / "uc-chped"
    dst = tmp_path / "uc-chped"
    dst.mkdir()
    for f in src.iterdir():
        if f.is_file():
            (dst / f.name).write_bytes(f.read_bytes())
    return dst


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--time-limit", "0.2"),
        ("--instance", "ucp13"),
        # One horizon per run and the portfolio are tuning shapes (#262), not
        # the single-threaded full-roster protocol the table publishes.
        ("--periods", "3"),
        ("--threads", "2"),
    ],
)
def test_uc_chped_refuses_a_partial_run_onto_the_published_table(
    flag: str, value: str, tmp_path: Path
) -> None:
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    published = inst_dir / "comparison.csv"
    before = published.read_bytes()

    # --out spelled out, which is the shape the README documents.
    result = subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), flag, value, "--out", str(published)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert "cannot write the published table" in result.stderr, result.stderr
    assert published.read_bytes() == before, "the published table was modified"


# The published protocol must also say which engine it measured and re-check
# what it measured: without --verify every `verified` cell is empty, which
# fails one of the run's own acceptance criteria while exiting 0 (#146). Each
# case satisfies every rung before the one it pins.
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "requires an explicit --commit"),
        (["--commit", "deadbee"], "requires --verify"),
    ],
)
def test_uc_chped_requires_a_commit_and_verify_to_write_the_published_table(
    extra: list[str], message: str, tmp_path: Path
) -> None:
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    published = inst_dir / "comparison.csv"
    before = published.read_bytes()

    result = subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), "--out", str(published), *extra],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert message in result.stderr, result.stderr
    assert "pass --out elsewhere" in result.stderr, result.stderr
    assert published.read_bytes() == before, "the published table was modified"


# The search-configuration flags (#136): an ablation arm must not rewrite the
# published table, and every row the runner writes names the arm.
@pytest.mark.parametrize(
    "arm",
    [
        ["--no-lns"],
        ["--lns-interval", "5"],
        ["--no-float-hook"],
        ["--no-time-limit", "--max-iterations", "100"],
        # The engine default is the arm here: uc-chped publishes with compound
        # moves OFF, so the guard must judge against that, not SearchFlags{}.
        ["--compound-moves"],
    ],
)
def test_uc_chped_refuses_an_ablation_arm_onto_the_published_table(
    arm: list[str], tmp_path: Path
) -> None:
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    published = inst_dir / "comparison.csv"
    before = published.read_bytes()

    result = subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), *arm, "--out", str(published)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert "cannot write the published table" in result.stderr, result.stderr
    assert published.read_bytes() == before, "the published table was modified"


def test_uc_chped_rejects_a_clock_free_run_with_no_iteration_budget(tmp_path: Path) -> None:
    """--no-time-limit alone is a run with no budget at all, which solve() ends
    immediately -- a full-looking table of empty results."""
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)

    result = subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), "--no-time-limit", "--out", str(tmp_path / "o.csv")],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert "--max-iterations" in result.stderr, result.stderr
    assert not (tmp_path / "o.csv").exists()


# Every flag of the shared table's structural families (#198). The scalar
# runners' models have no List or Set, so the engine resolves the structural
# batch to probability 0 and each of these would be recorded and read by
# nothing -- refused instead, even restated at its default.
STRUCTURAL_SEARCH_FLAGS = [
    ["--structural-prob", "1"],
    ["--structural-selection", "first_improving"],
    ["--structural-sample-size", "8"],
    ["--neighbours", "5"],
]


@pytest.mark.parametrize("flag", STRUCTURAL_SEARCH_FLAGS)
def test_uc_chped_refuses_a_structural_search_flag(flag: list[str], tmp_path: Path) -> None:
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    out = tmp_path / "o.csv"

    result = subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), "--out", str(out), *flag],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert f"{flag[0]} is a " in result.stderr, result.stderr
    assert not out.exists()


def test_uc_chped_records_the_arm_on_every_measured_row(tmp_path: Path) -> None:
    """The iteration-budgeted arm end to end: it runs without a wall clock, and
    every row it measures carries the configuration it was measured under."""
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    out = tmp_path / "arm.csv"

    result = subprocess.run(
        [
            str(UC_CHPED_BINARY),
            str(inst_dir),
            "--instance",
            "ucp13",
            "--no-time-limit",
            "--max-iterations",
            "200",
            "--lns-interval",
            "5",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    lines = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    header = lines[0].split(",")
    assert header[-1] == "search_config"
    expected = (
        "float_hook=on;lns=on;lns_interval=5;compound_moves=off;novelty_prob=0.5;"
        "unproductive_iters=300;perturbation_period=100;max_iterations=200;time_limit=off"
    )
    measured = 0
    for line in lines[1:]:
        cells = line.split(",")
        # No short rows: a reader's column count must not depend on the row.
        assert len(cells) == len(header), line
        if "CBLS ViolationLS" in line:
            assert cells[-1] == expected, line
            measured += 1
        else:
            # A cited Pedroso row is not our measurement, so it carries no
            # configuration -- as it carries no seed or tolerance either.
            assert cells[-1] == "", line
    assert measured > 0, "no measured rows in the table"


# The anytime trace (#147). tests/test_uc_chped_trace.cpp pins the recorder;
# only the shell shows the runner hands it to solve(): with `nullptr` as the
# SolveCallback (the pre-#147 call), the file gets a header and no rows.
UC_CHPED_TRACE_HEADER = (
    "instance,periods,time_limit_s,time_seconds,batches,objective,new_best,commit_sha"
)


def test_uc_chped_records_an_anytime_trace(tmp_path: Path) -> None:
    """--trace writes an incumbent-versus-time profile that names the (instance,
    horizon) row it belongs to and the engine commit it was measured on."""
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    out = tmp_path / "run.csv"
    trace = tmp_path / "trace.csv"

    # Iteration-budgeted, so bounded by work, not machine speed. The trace's
    # wall-clock cells are real seconds; nothing asserts on their values.
    result = subprocess.run(
        [
            str(UC_CHPED_BINARY),
            str(inst_dir),
            "--instance",
            "ucp13",
            "--no-time-limit",
            "--max-iterations",
            "200",
            "--commit",
            "deadbee",
            "--out",
            str(out),
            "--trace",
            str(trace),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert str(trace) in result.stdout, "the tally does not say where the trace went"

    lines = trace.read_text().splitlines()
    assert lines[0] == UC_CHPED_TRACE_HEADER
    rows = [ln.split(",") for ln in lines[1:] if ln]
    # Rows, not just a header: header-only is what a runner passing `nullptr` writes.
    assert rows, "the trace has no rows, so no callback reached solve()"

    horizons = set()
    for cells in rows:
        assert len(cells) == len(UC_CHPED_TRACE_HEADER.split(",")), cells
        assert cells[0] == "ucp13"
        # The horizon makes a trace row attributable to its (instance, horizon).
        assert cells[1] in {"1", "3", "6", "12", "24"}, cells
        # The row's time budget: 0 on this clock-free arm.
        assert float(cells[2]) == 0.0, cells
        assert float(cells[3]) >= 0.0
        assert int(cells[4]) >= 0
        assert math.isfinite(float(cells[5]))
        assert cells[6] in {"0", "1"}
        assert cells[7] == "deadbee"
        horizons.add(cells[1])
    # Every solved horizon appears: a budget map has a number per horizon.
    assert horizons == {"1", "3", "6", "12", "24"}, horizons


def test_uc_chped_refuses_a_trace_onto_the_published_trace(tmp_path: Path) -> None:
    """`--trace` truncates on open before any solving, so an unpublished run
    aimed at the reserved trace name would replace the published profile at exit
    0 while --out pointed somewhere harmless."""
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    published_trace = inst_dir / "anytime_trace.csv"
    published_trace.write_text("published rows nobody may overwrite\n")
    before = published_trace.read_bytes()

    result = subprocess.run(
        [
            str(UC_CHPED_BINARY),
            str(inst_dir),
            "--instance",
            "ucp13",
            "--out",
            str(tmp_path / "elsewhere.csv"),
            "--trace",
            str(published_trace),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert "cannot write the published anytime trace" in result.stderr, result.stderr
    assert published_trace.read_bytes() == before, "the published trace was modified"


@pytest.mark.parametrize("flag", ["--out", "--trace"])
def test_uc_chped_refuses_either_flag_onto_either_published_artifact(
    flag: str, tmp_path: Path
) -> None:
    """The guard is about the files, not which flag names them.

    Pre-fix, `--trace <the comparison table>` passed a per-flag guard and its
    truncating open replaced the ten cited Pedroso rows, exit 0 (a 3789-byte
    comparison.csv came back as 688 bytes of trace rows).
    """
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    published_table = inst_dir / "comparison.csv"
    published_trace = inst_dir / "anytime_trace.csv"
    published_trace.write_text("published rows nobody may overwrite\n")
    before = {p: p.read_bytes() for p in (published_table, published_trace)}

    for target in (published_table, published_trace):
        args = [str(UC_CHPED_BINARY), str(inst_dir), "--instance", "ucp13", flag, str(target)]
        # Keep the OTHER flag pointed somewhere harmless, so the only thing that
        # can trip the guard is the one under test.
        if flag == "--trace":
            args += ["--out", str(tmp_path / "elsewhere.csv")]
        result = subprocess.run(args, capture_output=True, text=True, timeout=300)

        assert result.returncode == 2, f"{flag} {target.name}: {result.stdout}"
        assert "cannot write the published" in result.stderr, result.stderr
        assert flag in result.stderr, result.stderr
        for path, content in before.items():
            assert path.read_bytes() == content, f"{path.name} was modified"


def test_uc_chped_refuses_one_path_for_both_outputs(tmp_path: Path) -> None:
    """The table's closing rename lands on top of the trace, so one path for
    both flags silently destroys the profile at exit 0."""
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    both = tmp_path / "both.csv"

    result = subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), "--out", str(both), "--trace", str(both)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert "name the same file" in result.stderr, result.stderr
    assert not both.exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "requires an explicit --commit"),
        # Only the verified published run writes the trace, so a trace-only
        # publish is refused too, with a trace-specific reason (#146).
        (["--commit", "deadbee"], "requires --verify: the published trace belongs"),
    ],
)
def test_uc_chped_requires_a_commit_and_verify_to_write_the_published_trace(
    extra: list[str], message: str, tmp_path: Path
) -> None:
    """A full-roster run may write the published trace, but only while saying
    which engine it profiled and having verified it -- and the diagnostic names
    --trace, the flag that redirects the file it is refusing, not --out."""
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    published_trace = inst_dir / "anytime_trace.csv"
    published_trace.write_text("published rows nobody may overwrite\n")
    before = published_trace.read_bytes()

    # No --instance or budget override, so this is the published protocol,
    # refused for the missing flag alone during argument resolution (cheap).
    result = subprocess.run(
        [
            str(UC_CHPED_BINARY),
            str(inst_dir),
            "--out",
            str(tmp_path / "elsewhere.csv"),
            "--trace",
            str(published_trace),
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert message in result.stderr, result.stderr
    assert "pass --trace elsewhere" in result.stderr, result.stderr
    assert published_trace.read_bytes() == before, "the published trace was modified"


def test_uc_chped_rejects_a_trace_flag_with_no_value(tmp_path: Path) -> None:
    """The shared ArgCursor rule: a trailing --trace must not read past argv nor
    silently run without the trace it was asked for."""
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)

    result = subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), "--out", str(tmp_path / "o.csv"), "--trace"],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert "Unknown or incomplete option: --trace" in result.stderr, result.stderr
    assert not (tmp_path / "o.csv").exists()


def test_uc_chped_refuses_a_horizon_no_selected_file_has(tmp_path: Path) -> None:
    """A run asked for one row and finding none would write an empty table at
    exit 0: a tuning study would read that as a missing value, not a typo."""
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    out = tmp_path / "o.csv"

    result = subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), "--instance", "ucp13", "--periods", "48"]
        + ["--out", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert "--periods 48 is not a horizon of the selected instances" in result.stderr
    assert not out.exists()


def test_uc_chped_marks_a_row_that_lost_portfolio_workers(tmp_path: Path) -> None:
    """The exit code alone is gone once a scorer globs the tables (#363). A
    1-microsecond budget starves every worker: no test seam, the deadline is real."""
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    out = tmp_path / "o.csv"

    result = subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), "--instance", "ucp13", "--periods", "3"]
        + ["--threads", "3", "--time-limit", "0.000001", "--verify", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 1, result.stderr

    lines = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    measured = [r for r in csv.DictReader(lines) if r["method"] == "CBLS ViolationLS"]
    assert [r["note"] for r in measured] == ["lost-workers"]


def test_uc_chped_runs_one_horizon_on_the_portfolio(tmp_path: Path) -> None:
    """The tuning shape (#262): one (file, horizon) on `ParallelSearch` gives one
    verified measured row, the portfolio's stream reaches the trace, and the
    row reports the best incumbent that stream saw."""
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    out = tmp_path / "o.csv"
    trace = tmp_path / "t.csv"

    result = subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), "--instance", "ucp13", "--periods", "3"]
        # 1 s, not less: a worker that starts after the deadline counts as
        # lost (#170) and fails the run, which a loaded machine can cause.
        + ["--threads", "2", "--time-limit", "1", "--verify"]
        + ["--out", str(out), "--trace", str(trace)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    lines = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    rows = list(csv.DictReader(lines))
    measured = [r for r in rows if r["method"] == "CBLS ViolationLS"]
    assert [(r["instance"], r["periods"]) for r in measured] == [("ucp13", "3")]
    assert measured[0]["verified"] == "true", measured[0]
    assert "threads=2" in measured[0]["search_config"].split(";"), measured[0]
    incumbents = list(csv.DictReader(trace.read_text().splitlines()))
    assert incumbents, "the portfolio's stream never reached the trace"
    assert float(measured[0]["objective"]) == pytest.approx(
        min(float(p["objective"]) for p in incumbents if p["objective"]), rel=1e-6
    )


# The same two properties on the MINLPLib runner. Its rows are cheap to provoke:
# a roster naming an instance whose .nl is absent writes the runner's
# "not found" row without solving anything, and that row is exactly the
# early-exit case a new trailing column can silently turn into a short row.
MINLPLIB_BINARY = Path(__file__).resolve().parents[2] / "build" / "cbls_minlplib"

MINLPLIB_DEFAULT_ARM = (
    "float_hook=on;lns=on;lns_interval=3;compound_moves=on;novelty_prob=0.5;"
    "unproductive_iters=300;perturbation_period=100;max_iterations=0;time_limit=on"
)


def _minlplib_scratch(tmp_path: Path) -> Path:
    inst_dir = tmp_path / "minlplib"
    inst_dir.mkdir()
    (inst_dir / "bounds.csv").write_text(
        "instance,structure,nvars,ncons,objsense,primal_bks,dual_bound\nnosuch,NLP,1,1,min,1,1\n"
    )
    return inst_dir


def _run_minlplib(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(MINLPLIB_BINARY), *args], capture_output=True, text=True, timeout=timeout
    )


@pytest.mark.parametrize("explicit_out", [True, False])
def test_minlplib_refuses_an_ablation_arm_onto_the_published_table(
    explicit_out: bool, tmp_path: Path
) -> None:
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    published = inst_dir / "comparison.csv"
    published.write_text("published rows nobody may overwrite\n")
    before = published.read_bytes()

    args = [str(inst_dir), "--no-float-hook"]
    if explicit_out:
        # Naming the published table explicitly must not buy a way past the
        # guard; that is the shape the documented full-roster command uses.
        args += ["--out", str(published)]
    result = _run_minlplib(*args)

    assert result.returncode == 2, result.stdout
    assert "cannot write the published table" in result.stderr, result.stderr
    assert published.read_bytes() == before, "the published table was modified"


@pytest.mark.parametrize(
    "bad",
    [
        ["--lns-interval"],  # a value flag with no value: the ArgCursor rule
        ["--lns-interval", "x"],  # not an integer
        ["--max-iterations", "1.5"],  # trailing characters, so not an integer
        ["--novelty-prob", "2"],  # a number, but not one the engine can use
        ["--no-time-limit"],  # no iteration budget, so no budget at all
        ["--no-lns", "--lns-interval", "5"],  # the interval is read by nothing
        ["--no-compound-moves", "--novelty-prob", "0.25"],  # read by nothing then
    ],
)
def test_minlplib_rejects_a_bad_search_flag_value(bad: list[str], tmp_path: Path) -> None:
    """A bad value reports and exits rather than silently keeping the default --
    the shared policy the runners' other flags already follow."""
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    out = tmp_path / "scratch.csv"

    # The bad flag goes last, or `--lns-interval --out` swallows "--out" and
    # exits on the integer parse, not the trailing-flag path the first case pins.
    result = _run_minlplib(str(inst_dir), "--out", str(out), *bad)

    assert result.returncode == 2, result.stdout
    assert result.stderr.strip(), "a rejected value must say what was wrong"
    assert not out.exists()


@pytest.mark.parametrize("flag", STRUCTURAL_SEARCH_FLAGS)
def test_minlplib_refuses_a_structural_search_flag(flag: list[str], tmp_path: Path) -> None:
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    out = tmp_path / "scratch.csv"

    result = _run_minlplib(str(inst_dir), "--out", str(out), *flag)

    assert result.returncode == 2, result.stdout
    assert f"{flag[0]} is a " in result.stderr, result.stderr
    assert not out.exists()


def test_minlplib_records_the_arm_on_an_early_exit_row(tmp_path: Path) -> None:
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    out = tmp_path / "arm.csv"

    result = _run_minlplib(
        str(inst_dir), "--no-compound-moves", "--lns-interval", "5", "--out", str(out)
    )
    assert result.returncode == 0, result.stderr

    lines = out.read_text().splitlines()
    header = lines[0].split(",")
    assert header[-1] == "search_config"
    assert len(lines) == 2, lines  # the roster's one instance, which has no .nl
    cells = lines[1].split(",")
    assert len(cells) == len(header), lines[1]
    assert cells[-1] == MINLPLIB_DEFAULT_ARM.replace(
        "compound_moves=on", "compound_moves=off"
    ).replace("lns_interval=3", "lns_interval=5")
    # No solve ran, so the #143 and #150 LNS counters read NaN together: a 0
    # would read as "LNS ran and never repaired", what the ablation's LNS gate
    # decides on.
    row = dict(zip(header, cells, strict=True))
    assert row["lns_repairs"] == "NaN"
    assert row["lns_repairs_accepted"] == "NaN"
    # Likewise #149's pair: a 0 `time_to_first_feasible` would read as "feasible
    # instantly", and the #149 correlation would consume it as data.
    assert row["first_feasible_objective"] == "NaN"
    assert row["time_to_first_feasible"] == "NaN"


def test_minlplib_refuses_an_ablation_arm_onto_the_published_trace(tmp_path: Path) -> None:
    """`--trace` opens its file with a truncating stream before any solving, and
    `anytime_trace.csv` is published too -- so an arm aimed at it would replace
    the published anytime profile at exit 0 while `--out` pointed somewhere
    harmless."""
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    trace = inst_dir / "anytime_trace.csv"
    trace.write_text("published trace nobody may overwrite\n")
    before = trace.read_bytes()

    result = _run_minlplib(
        str(inst_dir),
        "--no-lns",
        "--out",
        str(tmp_path / "scratch.csv"),
        "--trace",
        str(trace),
    )

    assert result.returncode == 2, result.stdout
    # Names the artifact it refuses, so a --trace refusal does not point at --out.
    assert "cannot write the published anytime trace" in result.stderr, result.stderr
    assert "--trace" in result.stderr, result.stderr
    assert trace.read_bytes() == before, "the published trace was modified"


def test_minlplib_help_lists_the_search_flags() -> None:
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    result = _run_minlplib("--help")
    assert result.returncode == 0
    for flag in (
        "--no-float-hook",
        "--no-lns",
        "--lns-interval",
        "--novelty-prob",
        "--unproductive-iters",
        "--perturbation-period",
        "--max-iterations",
        "--no-time-limit",
        # A help text that omits a shipped arm is how an ablation runs without it.
        "--compound-moves",
    ):
        assert flag in result.stdout, flag
    # ...and none it would refuse (#198).
    for refused in STRUCTURAL_SEARCH_FLAGS:
        assert refused[0] not in result.stdout, refused[0]


@pytest.mark.parametrize(
    ("flag", "target_name", "other"),
    [
        ("--trace", "comparison.csv", "--out"),
        ("--out", "anytime_trace.csv", "--trace"),
    ],
)
def test_uc_chped_refuses_a_crossed_artifact_on_a_published_run(
    flag: str, target_name: str, other: str, tmp_path: Path
) -> None:
    """Only --out ever writes the table and only --trace ever writes the trace.

    The protocol ladder cannot police this: the run is the legitimate published
    one, and --commit satisfies every rung. With the refusal keyed on the
    protocol, this command truncated a 3789-byte comparison.csv to 3362 bytes of
    trace rows (the ten cited Pedroso rows gone) before any solve, exit 0.
    Refused during argument resolution, so the test is cheap.
    """
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    (inst_dir / "anytime_trace.csv").write_text("published trace rows nobody may overwrite\n")
    target = inst_dir / target_name
    before = {p: p.read_bytes() for p in inst_dir.glob("*.csv")}

    result = subprocess.run(
        [
            str(UC_CHPED_BINARY),
            str(inst_dir),
            "--commit",
            "deadbee",
            flag,
            str(target),
            other,
            str(tmp_path / "elsewhere.csv"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert flag in result.stderr, result.stderr
    for path, content in before.items():
        assert path.read_bytes() == content, f"{path.name} was modified"


def test_uc_chped_refuses_a_trace_onto_the_tables_temp_path(tmp_path: Path) -> None:
    """`--trace <out>.tmp` is the same file as the table, one rename later.

    The table is written to `<out>.tmp` and renamed into place, so aiming the
    trace at that path puts two truncating streams on one inode and then
    publishes the interleave under the table's name at exit 0.
    """
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    out = tmp_path / "run.csv"

    result = subprocess.run(
        [
            str(UC_CHPED_BINARY),
            str(inst_dir),
            "--instance",
            "ucp13",
            "--out",
            str(out),
            "--trace",
            str(out) + ".tmp",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert "temp file" in result.stderr, result.stderr
    assert not out.exists()


def test_uc_chped_trace_ends_every_solve_at_its_budget(tmp_path: Path) -> None:
    """Every (instance, horizon) the runner starts appears in the trace, and its
    last row is the one at the run's final time.

    The engine samples at most once a second and `record_best` resets that
    timer, so without a closing row a record ends up to ~1.5s short and can read
    as "still improving"; and a solve with no valued incumbent would write no
    rows, indistinguishable from a callback regressed to nullptr.
    """
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    trace = tmp_path / "trace.csv"
    budget = 3.0

    result = subprocess.run(
        [
            str(UC_CHPED_BINARY),
            str(inst_dir),
            "--instance",
            "ucp13",
            "--time-limit",
            str(budget),
            "--out",
            str(tmp_path / "run.csv"),
            "--trace",
            str(trace),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr

    rows = list(csv.DictReader(trace.read_text().splitlines()))
    assert rows, "no trace rows, so no callback reached solve()"

    # ucp13 runs five horizons; every one of them must be present.
    horizons = {int(r["periods"]) for r in rows}
    assert horizons == {1, 3, 6, 12, 24}, horizons

    for periods in sorted(horizons):
        block = [r for r in rows if int(r["periods"]) == periods]
        last = block[-1]
        # The closing row is at the run's final time, not at the last sample
        # that happened to be taken, and it does not claim an improvement.
        assert last["new_best"] == "0", block
        assert float(last["time_seconds"]) == max(float(r["time_seconds"]) for r in block)
        assert float(last["time_seconds"]) >= budget - 0.5, (
            f"the record for {periods}p stops {budget - float(last['time_seconds']):.2f}s "
            "short of the budget"
        )


def test_minlplib_records_a_probability_it_can_be_read_back_from(tmp_path: Path) -> None:
    """The recorded cell must name the value the search actually received.

    `%g`'s six significant digits recorded `0.1234567` and `0.1234568` both as
    `novelty_prob=0.123457`.
    """
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    cells = []
    for value in ("0.1234567", "0.1234568"):
        sub = tmp_path / value
        sub.mkdir()
        out = tmp_path / f"arm{value}.csv"
        result = _run_minlplib(
            str(_minlplib_scratch(sub)),
            "--compound-moves",
            "--novelty-prob",
            value,
            "--out",
            str(out),
        )
        assert result.returncode == 0, result.stderr
        cells.append(out.read_text().splitlines()[1].split(",")[-1])

    assert cells[0] != cells[1], cells
    assert "0.1234567" in cells[0], cells[0]


@pytest.mark.parametrize("value", ["-1", "-300"])
def test_minlplib_rejects_a_negative_unproductive_iters(value: str, tmp_path: Path) -> None:
    """A negative and a zero are the same run (the unproductive-batch exit arms
    only on `> 0`), so accepting both would give one configuration two
    `search_config` cells, the invariant the no-effect refusals rest on. 0 is
    the documented spelling of the fixed-cadence arm.
    """
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    result = _run_minlplib(
        "--unproductive-iters", value, "--instance", "alkylation", "--out", str(tmp_path / "o.csv")
    )
    assert result.returncode == 2, result.stdout
    assert "--unproductive-iters must be >= 0" in result.stderr, result.stderr


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--time-limit", "1"), ("--seed", "1"), ("--feas-tol", "1"), ("--threads", "2")],
)
def test_minlplib_refuses_an_off_protocol_run_onto_the_published_table(
    flag: str, value: str, tmp_path: Path
) -> None:
    """The budget and the numerics belong in the guard, not just the roster.

    Unguarded, `--time-limit 1` over the full roster republished the table with
    one-second results at exit 0 (the #88 hazard); `--seed` and `--feas-tol`
    change rows in a way no column records. uc-chped guards these too.
    """
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    published = inst_dir / "comparison.csv"
    published.write_text("published rows nobody may overwrite\n")
    before = published.read_bytes()

    result = subprocess.run(
        [str(MINLPLIB_BINARY), str(inst_dir), flag, value],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert flag in result.stderr, result.stderr
    assert published.read_bytes() == before, "the published table was modified"


def test_minlplib_runs_an_instance_on_the_portfolio(tmp_path: Path) -> None:
    """`--threads 2` (#262): one row, feasible, its residual read from the
    portfolio's best restored into the model, and the portfolio's stream in the
    trace."""
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = Path(__file__).resolve().parents[2] / "benchmarks" / "instances" / "minlplib"
    out = tmp_path / "o.csv"
    trace = tmp_path / "t.csv"

    result = _run_minlplib(
        # 1 s, not less: a worker that starts after the deadline counts as
        # lost (#170) and fails the run, which a loaded machine can cause.
        *[str(inst_dir), "--instance", "alkylation", "--threads", "2", "--time-limit", "1"],
        *["--out", str(out), "--trace", str(trace)],
    )
    assert result.returncode == 0, result.stderr

    rows = list(csv.DictReader(out.read_text().splitlines()))
    assert [(r["instance"], r["feasible"]) for r in rows] == [("alkylation", "true")]
    # Which verdict depends on how far a 1 s search gets: a lucky run reaches
    # the published bound and is `matches-bks`, not `feasible`.
    feasible_verdicts = ("better-than-bks", "matches-bks", "within-tolerance-of-bks", "feasible")
    assert rows[0]["note"] in feasible_verdicts, rows[0]["note"]
    # The row records its portfolio size: one table holds one thread count (#299).
    assert rows[0]["search_config"].endswith(";threads=2"), rows[0]["search_config"]
    incumbents = list(csv.DictReader(trace.read_text().splitlines()))
    assert incumbents, "the portfolio's stream never reached the trace"


def test_minlplib_requires_a_commit_to_write_the_published_table(tmp_path: Path) -> None:
    """A published row whose provenance reads "unknown" is one a later reader
    cannot tell engine drift from a bug with."""
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    published = inst_dir / "comparison.csv"
    published.write_text("published rows nobody may overwrite\n")
    before = published.read_bytes()

    result = subprocess.run(
        [str(MINLPLIB_BINARY), str(inst_dir)], capture_output=True, text=True, timeout=300
    )

    assert result.returncode == 2, result.stdout
    assert "requires an explicit --commit" in result.stderr, result.stderr
    assert published.read_bytes() == before


def test_minlplib_refuses_a_crossed_artifact(tmp_path: Path) -> None:
    """Only --out writes the table and only --trace writes the trace.

    `--trace` truncates its file on open before any solving, so `--trace <the
    comparison table>` would empty the published results at exit 0 (reproduced
    on uc-chped).
    """
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    published = inst_dir / "comparison.csv"
    published.write_text("published rows nobody may overwrite\n")
    before = published.read_bytes()

    result = subprocess.run(
        [
            str(MINLPLIB_BINARY),
            str(inst_dir),
            "--commit",
            "deadbee",
            "--trace",
            str(published),
            "--out",
            str(tmp_path / "elsewhere.csv"),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert "--trace cannot write the published table" in result.stderr, result.stderr
    assert published.read_bytes() == before


def test_a_hardlink_to_the_published_table_is_still_the_published_table(tmp_path: Path) -> None:
    """A hardlink is the same file under two names that no path comparison can
    equate: `weakly_canonical` resolves symlinks, but a hardlink is not an
    indirection to resolve -- both names ARE the file.

    Measured against the path-only guard: `ln comparison.csv alias.csv` then
    `--out alias.csv` rewrote the published inode at exit 0 with every guard
    satisfied.
    """
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    published = inst_dir / "comparison.csv"
    published.write_text("published rows nobody may overwrite\n")
    alias = inst_dir / "alias.csv"
    os.link(published, alias)
    before = published.read_bytes()

    result = subprocess.run(
        [str(MINLPLIB_BINARY), str(inst_dir), "--no-lns", "--out", str(alias)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert published.read_bytes() == before, "the published inode was rewritten through an alias"


def _parse_human_solution(stdout: str) -> dict[str, float]:
    """The `name = value` lines under `Solution:` in the human format."""
    values: dict[str, float] = {}
    in_block = False
    for line in stdout.splitlines():
        if line.startswith("Solution:"):
            in_block = True
            continue
        if in_block:
            if "=" not in line:
                break
            name, _, raw = line.partition("=")
            values[name.strip()] = float(raw.strip())
    return values


@pytest.mark.parametrize("threads", ["1", "4", "0"])
def test_the_cli_prints_the_assignment_it_reports_an_objective_for(threads: str) -> None:
    # Both formatters print the assignment by iterating the CLI's own Model, not
    # SearchResult.best_state. On the single-threaded path that is right because
    # solve() restores its winner into the model it was handed; the portfolio's
    # workers search their OWN models, so the CLI has to restore the winning
    # state itself. Without that it printed a real objective next to this
    # model's untouched initial values -- x = y = 0 here, which does not even
    # satisfy the constraint it just called feasible.
    #
    # Parametrised over both paths so the assertion is the same one in each: the
    # printed numbers must be a solution of the printed problem.
    result = _run_cbls(str(MODEL), "--threads", threads, "--time-limit", "0.3")
    assert result.returncode == 0, result.stderr

    values = _parse_human_solution(result.stdout)
    assert set(values) == {"x", "y"}, result.stdout
    # examples/simple.cbls: minimise x^2 + y^2 subject to x + y >= 5.
    assert values["x"] + values["y"] >= 5.0 - 1e-6, result.stdout

    objective = float(
        next(line for line in result.stdout.splitlines() if line.startswith("Objective:"))
        .removeprefix("Objective:")
        .strip()
    )
    # The printed objective must be the objective OF the printed assignment.
    # Two decimal places is all the human format gives, so the tolerance is the
    # rounding, not the solver's.
    assert objective == pytest.approx(values["x"] ** 2 + values["y"] ** 2, abs=0.2), result.stdout


@pytest.mark.parametrize("threads", ["1", "4", "0"])
def test_the_jsonl_solution_object_is_a_solution_of_the_printed_problem(threads: str) -> None:
    # The human format's half of this is pinned above. JSONL is the machine
    # contract -- it is what a driver script reads -- and it goes through a
    # different formatter (`JsonlFormatter::print_result`), which builds its
    # `solution` object by iterating the CLI's own Model exactly as the human
    # one does. So it fails in exactly the same way if the parallel path stops
    # restoring the winning state, and nothing covered it.
    result = _run_cbls(str(MODEL), "--threads", threads, "--format", "jsonl", "--time-limit", "0.3")
    assert result.returncode == 0, result.stderr

    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    finals = [r for r in records if "solution" in r]
    assert len(finals) == 1, result.stdout
    final = finals[0]

    solution = final["solution"]
    assert set(solution) == {"x", "y"}, result.stdout
    # examples/simple.cbls: minimise x^2 + y^2 subject to x + y >= 5.
    assert solution["x"] + solution["y"] >= 5.0 - 1e-6, result.stdout
    # Full double precision here, unlike the human format's two decimals, so the
    # objective must match the assignment to within the solver's own tolerance.
    assert final["objective"] == pytest.approx(solution["x"] ** 2 + solution["y"] ** 2, rel=1e-6)


@pytest.mark.parametrize("fmt", ["human", "jsonl"])
def test_a_maximize_objective_is_printed_in_the_models_own_sense(fmt: str, tmp_path: Path) -> None:
    # The engine minimises internally: `maximize(e)` makes its objective `-e`,
    # and SearchResult/SolveProgress carry that negated value. The CLI printed
    # it as it was, so maximising x^2 + y^2 over [0, 10]^2 reported -200 at
    # x = y = 10 -- in the result and in every progress row (#225).
    model = tmp_path / "max.cbls"
    model.write_text(MODEL.read_text().replace('"minimize"', '"maximize"'))
    result = _run_cbls(str(model), "--threads", "1", "--format", fmt, "--time-limit", "0.3")
    assert result.returncode == 0, result.stderr

    if fmt == "jsonl":
        records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        final = next(r for r in records if "solution" in r)
        values = final["solution"]
        objective = final["objective"]
        progress = [r["objective"] for r in records if r.get("event") == "progress"]
    else:
        values = _parse_human_solution(result.stdout)
        objective = float(
            next(line for line in result.stdout.splitlines() if line.startswith("Objective:"))
            .removeprefix("Objective:")
            .strip()
        )
        # Progress rows: "<time>s <iter> <objective|-> ...", one per tick.
        progress = [
            float(fields[2])
            for fields in (line.split() for line in result.stdout.splitlines())
            if len(fields) >= 3
            and fields[0].endswith("s")
            and fields[2] != "-"
            and fields[0][:-1].replace(".", "", 1).isdigit()
        ]
        assert progress, result.stdout
    assert objective == pytest.approx(values["x"] ** 2 + values["y"] ** 2, abs=0.2), result.stdout
    assert objective > 0.0, result.stdout
    assert all(p is None or p > 0.0 for p in progress), result.stdout


NQUEENS = Path(__file__).resolve().parents[2] / "examples" / "nqueens.cbls"


def test_the_cli_warns_when_portfolio_workers_did_not_complete() -> None:
    """A lost worker is not silent in the CLI either (#170).

    A 1-microsecond budget is gone before any worker has copied its model, so
    every worker is starved before its first solve and the warning lists each
    one with its reason. No test seam: the deadline is real.
    """
    result = _run_cbls(str(MODEL), "--quiet", "--threads", "3", "--time-limit", "0.000001")
    assert "Warning: 0 of 3 portfolio workers completed" in result.stderr, result.stderr
    assert result.stderr.count("before this worker's first solve") == 3, result.stderr


def test_the_cli_does_not_warn_when_a_peer_answered_first() -> None:
    """Workers a solved feasibility model never needed are not lost (#170).

    nqueens has no objective, so the first worker to a feasible point stops its
    peers, most before their first solve. Nothing failed, and the CLI once warned
    "3 of 8 portfolio workers completed" on every such run.
    """
    result = _run_cbls(str(NQUEENS), "--quiet", "--threads", "8", "--time-limit", "5")
    assert result.returncode == 0, result.stderr
    assert "Warning" not in result.stderr, result.stderr
