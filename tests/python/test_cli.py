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
