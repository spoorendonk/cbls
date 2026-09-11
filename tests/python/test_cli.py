"""Shell-level smoke tests for the `cbls` CLI binary.

These run the built executable as a subprocess rather than calling into the
bindings, because what they check is only observable at that level: the exit
status and stderr a shell sees. A malformed numeric flag used to let
`std::stod`'s exception escape `main`, which is `std::terminate` -- the process
died on SIGABRT with `terminate called after throwing an instance of
'std::invalid_argument' / what(): stod` and no mention of which flag was wrong.
Nothing in-process can distinguish that from a clean non-zero exit.
"""

from __future__ import annotations

import csv
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
    "--epoch-iters",
    "--max-epochs",
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
    # Naming the flag is the point: `what(): stod` told the user nothing about
    # which of seven flags they mistyped.
    assert flag in result.stderr, f"{flag}: {result.stderr}"
    # ...and says what was wrong with the *value*. Without this, a refactor that
    # dropped the parse and let the flag fall through to `unknown option
    # '--threads'` would keep the test green: same exit code, and that message
    # names the flag too.
    assert "'abc' is not a" in result.stderr, f"{flag}: {result.stderr}"


# Only the int-width flags carry a range check; the int64 and double flags accept
# whatever stoll/stod do.
INT_WIDTH_FLAGS = ["--lns-interval", "--threads", "--max-epochs"]


@pytest.mark.parametrize("flag", INT_WIDTH_FLAGS)
def test_a_value_wider_than_int_is_reported_as_out_of_range(flag: str) -> None:
    # std::stoi threw out_of_range here, which escaped main exactly like a
    # malformed value did. The widened parse plus an explicit bound is what
    # replaced it, and nothing else covers that branch.
    result = _run_cbls(flag, "2147483648")

    assert 0 < result.returncode < 128, f"{flag}: returncode {result.returncode}"
    assert f"{flag}: '2147483648' is out of range" in result.stderr, result.stderr


def test_well_formed_numeric_flags_are_all_accepted() -> None:
    # The negative tests cannot show the flags still work. Every numeric flag
    # gets a valid value and no model file, so the CLI stops at the model check
    # -- reaching that message proves each flag before it parsed. Without this,
    # an inverted bound in the int overload or a wrong conversion in the uint64
    # one would be caught by nothing.
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
        "--epoch-iters",
        "10",
        "--max-epochs",
        "1",
    )

    assert result.returncode == 1, result.stderr
    assert "no model file specified" in result.stderr


def test_the_seed_the_cli_prints_can_be_parsed_back() -> None:
    # --seed is unsigned and the CLI echoes it back unsigned, so `--seed -1` is
    # recorded as 2**64-1. Parsing the seed as int64 would make the tool unable
    # to read back the seed it just printed, defeating the flag's purpose.
    #
    # This has to run a real model: the seed is printed by the header, which
    # only runs once a model has loaded. Asserting on the parse alone would
    # leave the round trip -- the reason the flag is unsigned at all -- unpinned,
    # and a formatter that printed the seed signed would keep such a test green.
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
        ("--epoch-iters", "99999999999999999999999"),
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
    # "nan" is a well-formed double, so strict syntax alone accepts it -- and it
    # yields a solve that never searched. There is no downstream guard in the
    # CLI to catch it, so the parse rejects it.
    result = _run_cbls("--time-limit", "nan")

    assert 0 < result.returncode < 128, result.stderr
    assert "--time-limit: 'nan' is not a number" in result.stderr


# The benchmark runners have the same exposure with a worse consequence: they
# run for minutes to hours, so a mistyped flag that parses to a default produces
# a result-shaped artifact rather than an error. setcover's `--seed` went
# through `strtoull`, which reports nothing at all -- `--seed abc` ran the whole
# roster at seed 0 and wrote a CSV row naming it.
SETCOVER_BINARY = Path(__file__).resolve().parents[2] / "build" / "cbls_setcover"

SETCOVER_BAD_FLAGS = [
    ("--seed", "abc"),
    ("--seeds", "2x"),
    ("--time", "60s"),
    ("--time", "nan"),
    ("--struct-prob", "abc"),
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


# The uc-chped runner writes a published results table, so its guards are the
# thing most worth pinning: a cold review found they tested whether `--out` was
# OMITTED rather than which file it named, which meant the command the README
# documents -- `--out <the published table>` spelled out -- satisfied every
# guard while doing exactly the damage they exist to stop. A shortened run
# rewrote the table and deleted the cited reference rows for every instance it
# did not run, at exit 0.
UC_CHPED_BINARY = Path(__file__).resolve().parents[2] / "build" / "cbls_uc_chped"


def _uc_chped_scratch(tmp_path: Path) -> Path:
    src = Path(__file__).resolve().parents[2] / "benchmarks" / "instances" / "uc-chped"
    dst = tmp_path / "uc-chped"
    dst.mkdir()
    for f in src.iterdir():
        if f.is_file():
            (dst / f.name).write_bytes(f.read_bytes())
    return dst


@pytest.mark.parametrize("flag", ["--time-limit", "--instance"])
def test_uc_chped_refuses_a_partial_run_onto_the_published_table(flag: str, tmp_path: Path) -> None:
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    published = inst_dir / "comparison.csv"
    before = published.read_bytes()
    value = "0.2" if flag == "--time-limit" else "ucp13"

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


def test_uc_chped_requires_a_commit_to_write_the_published_table(tmp_path: Path) -> None:
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    published = inst_dir / "comparison.csv"
    before = published.read_bytes()

    result = subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), "--out", str(published)],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert "requires an explicit --commit" in result.stderr, result.stderr
    assert published.read_bytes() == before, "the published table was modified"


# The search-configuration flags (#136). An ablation arm is a command-line change
# recorded in the row, which puts two things at the shell level: the arm must not
# be able to regenerate the published table as a side effect, and every row the
# runner writes has to say which arm produced it.
@pytest.mark.parametrize(
    "arm",
    [
        ["--no-lns"],
        ["--lns-interval", "5"],
        ["--no-float-hook"],
        ["--no-time-limit", "--max-iterations", "100"],
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


# The anytime trace (#147). The defect was that this runner passed `nullptr`
# where solve() takes a SolveCallback, so nothing recorded what the incumbent was
# doing when the clock stopped and every published gap was a number at an
# undefended budget. The Catch2 side (tests/test_uc_chped_trace.cpp) pins the
# recorder's contract; what only the shell can prove is that the RUNNER hands it
# to solve() -- with the argument back at `nullptr` the file below still gets its
# header and no rows at all.
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

    # Iteration-budgeted, so the run is bounded by work rather than by this
    # machine's speed: the same reason test_uc_chped_records_the_arm_on_every
    # _measured_row uses it. The wall-clock cells in the trace are still real
    # seconds; nothing below asserts on their values.
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
    # The point of the test: rows, not just a header. A header-only file is
    # exactly what the unfixed runner produces.
    assert rows, "the trace has no rows, so no callback reached solve()"

    horizons = set()
    for cells in rows:
        assert len(cells) == len(UC_CHPED_TRACE_HEADER.split(",")), cells
        assert cells[0] == "ucp13"
        # One row per (instance, horizon) pair, so the horizon is what makes a
        # trace row attributable at all.
        assert cells[1] in {"1", "3", "6", "12", "24"}, cells
        # The budget the row was measured against. This arm is clock-free, so it
        # is 0 -- which is itself the honest statement of what bounded the run.
        assert float(cells[2]) == 0.0, cells
        assert float(cells[3]) >= 0.0
        assert int(cells[4]) >= 0
        assert math.isfinite(float(cells[5]))
        assert cells[6] in {"0", "1"}
        assert cells[7] == "deadbee"
        horizons.add(cells[1])
    # Every horizon this run solved appears, not just the first: a trace that
    # covered one row could not defend a budget map with a number per horizon.
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
    """The guard is about the FILES, not about which flag names them.

    An earlier cut tested --out only against comparison.csv and --trace only
    against anytime_trace.csv, so `--trace <the comparison table>` fell through
    both: the trace's truncating open emptied the published results and the run
    exited 0, deleting the ten cited Pedroso rows. Verified against that cut --
    a 3789-byte comparison.csv came back as 688 bytes of trace rows.
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


def test_uc_chped_requires_a_commit_to_write_the_published_trace(tmp_path: Path) -> None:
    """A full-roster run may write the published trace, but only while saying
    which engine it profiled -- and the diagnostic names --trace, the flag that
    redirects the file it is refusing, not --out."""
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = _uc_chped_scratch(tmp_path)
    published_trace = inst_dir / "anytime_trace.csv"
    published_trace.write_text("published rows nobody may overwrite\n")
    before = published_trace.read_bytes()

    # No --instance and no budget override, so the run IS the published protocol
    # and is refused for the commit alone. It exits during argument resolution,
    # before a single instance is read, which is what keeps this cheap.
    result = subprocess.run(
        [
            str(UC_CHPED_BINARY),
            str(inst_dir),
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
    assert "requires an explicit --commit" in result.stderr, result.stderr
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


# The same two properties on the MINLPLib runner. Its rows are cheap to provoke:
# a roster naming an instance whose .nl is absent writes the runner's
# "not found" row without solving anything, and that row is exactly the
# early-exit case a new trailing column can silently turn into a short row.
MINLPLIB_BINARY = Path(__file__).resolve().parents[2] / "build" / "cbls_minlplib"

MINLPLIB_DEFAULT_ARM = (
    "float_hook=on;lns=on;lns_interval=3;compound_moves=off;novelty_prob=0.5;"
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
        ["--novelty-prob", "0.25"],  # read by nothing without --compound-moves
    ],
)
def test_minlplib_rejects_a_bad_search_flag_value(bad: list[str], tmp_path: Path) -> None:
    """A bad value reports and exits rather than silently keeping the default --
    the shared policy the runners' other flags already follow."""
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    out = tmp_path / "scratch.csv"

    # The bad flag goes LAST. With it in the middle, `--lns-interval --out` is
    # a value flag that finds a value: it swallows "--out", and the exit 2 comes
    # from the integer parse rather than from the trailing-flag path the first
    # case is meant to exercise.
    result = _run_minlplib(str(inst_dir), "--out", str(out), *bad)

    assert result.returncode == 2, result.stdout
    assert result.stderr.strip(), "a rejected value must say what was wrong"
    assert not out.exists()


def test_minlplib_records_the_arm_on_an_early_exit_row(tmp_path: Path) -> None:
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    out = tmp_path / "arm.csv"

    result = _run_minlplib(
        str(inst_dir), "--compound-moves", "--novelty-prob", "0.25", "--out", str(out)
    )
    assert result.returncode == 0, result.stderr

    lines = out.read_text().splitlines()
    header = lines[0].split(",")
    assert header[-1] == "search_config"
    assert len(lines) == 2, lines  # the roster's one instance, which has no .nl
    cells = lines[1].split(",")
    assert len(cells) == len(header), lines[1]
    assert cells[-1] == MINLPLIB_DEFAULT_ARM.replace(
        "compound_moves=off", "compound_moves=on"
    ).replace("novelty_prob=0.5", "novelty_prob=0.25")
    # The #143 LNS counter is on the row too, and on THIS row it must read NaN:
    # no solve ran, and a 0 here would be indistinguishable from "LNS ran and
    # never repaired" -- the reading the ablation's LNS gate is decided on. The
    # #150 acceptance counter obeys the same rule for the same reason, and is
    # asserted beside it so a row cannot end up half-NaN.
    row = dict(zip(header, cells, strict=True))
    assert row["lns_repairs"] == "NaN"
    assert row["lns_repairs_accepted"] == "NaN"
    # #149's pair obeys the same rule, and for the sharper version of the same
    # reason: a 0 in `time_to_first_feasible` would read as "reached feasibility
    # instantly", which is the single most favourable reading a row that never
    # ran could be given, and the #149 correlation would consume it as data.
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
    # Names the artifact it is actually refusing. It said "table" for both
    # before, which sent a reader of a --trace refusal looking at --out.
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
        # The ninth. Listed last because it was the one the original assertion
        # missed, and a help text that omits a shipped arm is how an ablation
        # gets run without it.
        "--compound-moves",
    ):
        assert flag in result.stdout, flag


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

    The protocol ladder cannot police this pairing, because the run it lets
    through is the legitimate published one: no --instance, no --time-limit, no
    arm flag, and an explicit --commit satisfies every rung. Against the cut
    that keyed the refusal on the protocol, this command truncated a 3789-byte
    published comparison.csv to 3362 bytes of trace rows -- the ten cited
    Pedroso rows gone -- within a second of launch, before any solve started,
    and would have exited 0.

    The run is refused during argument resolution, so no solving happens and
    the test stays cheap.
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

    Two failures this pins, both of which make a trace unreadable rather than
    wrong: the engine samples at most once a second and `record_best` resets
    that timer, so without a closing row the record ends up to ~1.5s short of
    the budget and -- when the last thing before the gap was an improvement --
    reads as "still improving when the clock stopped"; and a solve that never
    finds a valued incumbent writes no rows at all, which is indistinguishable
    from a filtered roster or a callback that regressed to nullptr.
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

    `%g`'s six significant digits could not: `--novelty-prob 0.1234567` and
    `--novelty-prob 0.1234568` are two different runs that both recorded
    `novelty_prob=0.123457`, so a results file did not, in fact, state the
    configuration it was produced under.
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
    """A negative and a zero are the SAME run -- the engine arms the
    unproductive-batch exit only on `> 0` -- so accepting both would put two
    different `search_config` cells on one configuration.

    That invariant is what justifies refusing the two no-effect flag
    combinations, so it has to hold. 0 stays as the documented spelling for the
    fixed-cadence arm.
    """
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    result = _run_minlplib(
        "--unproductive-iters", value, "--instance", "alkylation", "--out", str(tmp_path / "o.csv")
    )
    assert result.returncode == 2, result.stdout
    assert "--unproductive-iters must be >= 0" in result.stderr, result.stderr


@pytest.mark.parametrize("flag", ["--time-limit", "--seed", "--feas-tol"])
def test_minlplib_refuses_an_off_protocol_run_onto_the_published_table(
    flag: str, tmp_path: Path
) -> None:
    """The budget and the numerics belong in the guard, not just the roster.

    `--time-limit 1` over the full roster satisfied every other rung and
    republished the published table with one-second results at exit 0 -- the #88
    hazard by name. `--seed` and `--feas-tol` did the same with rows differing
    from the published protocol in a way no column records. The sibling uc-chped
    runner has always guarded these; minlplib was the odd one out.
    """
    if not MINLPLIB_BINARY.exists():
        pytest.skip("cbls_minlplib not built")
    inst_dir = _minlplib_scratch(tmp_path)
    published = inst_dir / "comparison.csv"
    published.write_text("published rows nobody may overwrite\n")
    before = published.read_bytes()

    result = subprocess.run(
        [str(MINLPLIB_BINARY), str(inst_dir), flag, "1"],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 2, result.stdout
    assert flag in result.stderr, result.stderr
    assert published.read_bytes() == before, "the published table was modified"


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
    comparison table>` empties the published results at exit 0 -- reproduced on
    the sibling uc-chped runner against a real table.
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
