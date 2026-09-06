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
