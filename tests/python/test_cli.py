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
