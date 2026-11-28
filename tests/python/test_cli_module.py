"""The `cbls` console script (`python -m cbls`) against the C++ CLI (#465).

Both run as processes on the same files and command lines, and their output
and exit codes are compared. Where a run is reproducible (--threads 1 on a
model that stops on its own, or a zero budget) the comparison is exact up to
wall times; the formats the C++ CLI cannot read are checked on their own.
"""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

import cbls
import pytest
from cbls import __main__ as cli

ROOT = Path(__file__).resolve().parents[2]
CBLS_BINARY = ROOT / "build" / "cbls"
SIMPLE = ROOT / "examples" / "simple.cbls"
NQUEENS = ROOT / "examples" / "nqueens.cbls"
FILTER_NL = ROOT / "tests" / "data" / "filter.nl"

# min x + 2y over integers, x + y >= 3, x <= 2: optimum (2, 1).
TINY_MPS = """\
NAME          TINY
ROWS
 N  COST
 G  C1
 L  C2
COLUMNS
    MARKER    'MARKER'    'INTORG'
    X         COST        1.0   C1   1.0
    X         C2          1.0
    Y         COST        2.0   C1   1.0
    MARKER    'MARKER'    'INTEND'
RHS
    RHS       C1          3.0   C2   2.0
BOUNDS
 UP BND       X           10.0
 UP BND       Y           10.0
ENDATA
"""


def _run_python(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONPATH=str(Path(cbls.__file__).resolve().parents[1]))
    return subprocess.run(
        [sys.executable, "-m", "cbls", *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
        env=env,
    )


def _run_cpp(*args: str) -> subprocess.CompletedProcess[str]:
    if not CBLS_BINARY.exists():
        pytest.skip("cbls not built")
    return subprocess.run(
        [str(CBLS_BINARY), *args], capture_output=True, text=True, timeout=60, cwd=ROOT
    )


def _without_times(stdout: str) -> str:
    """`stdout` with wall times masked: the human format's `1.23s`, JSONL's "time"."""
    lines = []
    for line in stdout.splitlines():
        if line.startswith("{"):
            record = json.loads(line)
            record.pop("time", None)
            lines.append(json.dumps(record, sort_keys=True))
        else:
            lines.append(re.sub(r"\d+\.\d\ds\b", "<t>s", line))
    return "\n".join(lines)


def _assert_same_run(args: list[str]) -> subprocess.CompletedProcess[str]:
    cpp = _run_cpp(*args)
    py = _run_python(*args)
    assert py.returncode == cpp.returncode, (py.stderr, cpp.stderr)
    assert py.stderr == cpp.stderr
    assert _without_times(py.stdout) == _without_times(cpp.stdout)
    return py


def _options_block(help_text: str) -> str:
    return help_text[help_text.index("Options:") :]


def test_help_lists_the_cpp_clis_options() -> None:
    py = _run_python("--help")
    cpp = _run_cpp("--help")

    assert py.returncode == cpp.returncode == 0
    assert _options_block(py.stdout) == _options_block(cpp.stdout)
    # The one difference, the inputs it reads, is said.
    assert ".mps.gz" in py.stdout and ".nl" in py.stdout


def test_version_is_the_cpp_clis() -> None:
    _assert_same_run(["--version"])
    assert _run_python("--version").stdout == f"cbls {cbls.__version__}\n"


@pytest.mark.parametrize(
    "args",
    [
        ["--time-limit", "abc"],
        ["--seed", "abc"],
        ["--lns", "abc"],
        ["--lns-interval", "abc"],
        ["--threads", "abc"],
        ["--threads", "2147483648"],
        ["--lns-interval", "-2147483649"],
        ["--seed", "18446744073709551616"],
        ["--time-limit", "1e400"],
        ["--time-limit", "nan"],
        ["--time-limit", "60s"],
        ["--format", "xml"],
        ["--bogus"],
        ["--time-limit"],
        [],
        ["--time-limit", " 0.1", "--seed", "-1", "--threads", "2", "--lns", "0.3"],
        ["examples/no_such_model.cbls"],
    ],
)
def test_a_command_line_that_does_not_solve_ends_as_the_cpp_clis(args: list[str]) -> None:
    py = _assert_same_run(args)
    assert py.returncode == 1


@pytest.mark.parametrize("fmt", ["human", "jsonl"])
@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--quiet"],
        ["--seed", "-1"],
        ["--lns", "0.3", "--lns-interval", "2", "--intensify"],
    ],
)
def test_a_reproducible_solve_prints_what_the_cpp_cli_does(fmt: str, extra: list[str]) -> None:
    # nqueens stops at its first feasible point, so one thread is deterministic.
    py = _assert_same_run(
        [str(NQUEENS), "--threads", "1", "--time-limit", "5", "--format", fmt, *extra]
    )
    assert py.returncode == 0


@pytest.mark.parametrize("fmt", ["human", "jsonl"])
@pytest.mark.parametrize("threads", ["1", "2", "-3"])
def test_a_zero_budget_prints_what_the_cpp_cli_does(fmt: str, threads: str) -> None:
    # No budget: an infeasible result at the start point (exit 1), on both paths.
    py = _assert_same_run([str(SIMPLE), "--threads", threads, "--time-limit", "0", "--format", fmt])
    assert py.returncode == 1
    assert "no_budget" in py.stdout


def test_lost_portfolio_workers_are_reported_as_the_cpp_cli_does() -> None:
    py = _assert_same_run([str(SIMPLE), "--quiet", "--threads", "3", "--time-limit", "0.000001"])
    assert "Warning: 0 of 3 portfolio workers completed" in py.stderr


def test_a_timed_solve_reports_as_the_cpp_cli_does() -> None:
    # Not reproducible (stopped by the clock), so the shape is compared.
    args = [str(SIMPLE), "--threads", "2", "--time-limit", "0.3", "--format", "jsonl"]
    cpp = _run_cpp(*args)
    py = _run_python(*args)

    assert py.returncode == cpp.returncode == 0, py.stderr
    cpp_records = [json.loads(line) for line in cpp.stdout.splitlines()]
    py_records = [json.loads(line) for line in py.stdout.splitlines()]
    assert py_records[0] == cpp_records[0]  # the start record
    assert set(py_records[-1]) == set(cpp_records[-1])
    solution = py_records[-1]["solution"]
    # examples/simple.cbls: minimise x^2 + y^2 subject to x + y >= 5.
    assert solution["x"] + solution["y"] >= 5.0 - 1e-6


def _result_record(stdout: str) -> dict[str, object]:
    record = json.loads(stdout.splitlines()[-1])
    assert record["event"] == "result", stdout
    return dict(record)


def _write_mps(path: Path, text: bytes = TINY_MPS.encode()) -> None:
    path.write_bytes(gzip.compress(text) if path.name.lower().endswith(".gz") else text)


@pytest.mark.parametrize("name", ["tiny.mps", "tiny.mps.gz"])
def test_an_mps_file_is_read_by_its_extension(name: str, tmp_path: Path) -> None:
    path = tmp_path / name
    _write_mps(path)
    result = _run_python(str(path), "--threads", "1", "--time-limit", "1", "--format", "jsonl")

    assert result.returncode == 0, result.stderr
    record = _result_record(result.stdout)
    solution = record["solution"]
    assert isinstance(solution, dict)
    assert solution["X"] + solution["Y"] >= 3.0 and solution["X"] <= 2.0, solution
    # The C++ CLI reads .cbls only.
    assert _run_cpp(str(path), "--threads", "1", "--time-limit", "1").returncode == 1


@pytest.mark.parametrize("name", ["TINY.MPS", "TINY.MPS.GZ"])
def test_an_upper_case_extension_is_not_taken_for_mps(name: str, tmp_path: Path) -> None:
    # Extensions are case-sensitive, as read_mps's .gz test: an upper-case name
    # is read as .cbls and fails to load (#465).
    path = tmp_path / name
    _write_mps(path)
    result = _run_python(str(path), "--threads", "1", "--time-limit", "1")

    assert result.returncode == 1, result.stdout
    assert result.stderr.startswith("Error loading model: "), result.stderr


def _run_python_bytes(*args: str | bytes) -> subprocess.CompletedProcess[bytes]:
    env = dict(os.environ, PYTHONPATH=str(Path(cbls.__file__).resolve().parents[1]))
    return subprocess.run(
        [sys.executable, "-m", "cbls", *args], capture_output=True, timeout=60, env=env
    )


@pytest.mark.parametrize(
    ("fmt", "printed"), [("human", b"\n  X\xe9 = "), ("jsonl", '"X�":'.encode())]
)
def test_a_name_that_is_not_utf8_is_printed(fmt: str, printed: bytes, tmp_path: Path) -> None:
    # MPS names are raw bytes: the human format prints them as they are, JSON
    # lines as U+FFFD, so the output stays JSON (#465).
    path = tmp_path / "latin1.mps"
    _write_mps(path, TINY_MPS.replace("    X  ", "    X\xe9 ").encode("latin-1"))
    result = _run_python_bytes(str(path), "--threads", "1", "--time-limit", "0.5", "--format", fmt)

    assert result.returncode == 0, result.stderr
    assert printed in result.stdout, result.stdout


@pytest.mark.skipif(os.name == "nt", reason="a POSIX argv is bytes")
@pytest.mark.parametrize(
    ("args", "error"),
    [
        ([b"q\xe9.cbls"], b"Error loading model: the path is not valid UTF-8"),
        ([b"--time-limit", b"\xff"], b"Error: --time-limit: '"),
    ],
)
def test_an_argument_that_is_not_utf8_is_an_error_not_a_crash(
    args: list[bytes], error: bytes
) -> None:
    result = _run_python_bytes(*args)

    assert result.returncode == 1, result.stderr
    assert result.stderr.startswith(error), result.stderr
    assert b"Traceback" not in result.stderr and b"incompatible" not in result.stderr


def test_an_nl_file_is_read_by_its_extension() -> None:
    result = _run_python(str(FILTER_NL), "--threads", "1", "--time-limit", "1", "--format", "jsonl")

    assert result.returncode == 0, result.stderr
    assert _result_record(result.stdout)["status"] == "feasible"


def test_a_locale_that_cannot_encode_the_header_changes_nothing() -> None:
    # The header's em dash goes out as UTF-8 bytes, whatever stdout's encoding (#465).
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(cbls.__file__).resolve().parents[1]),
        PYTHONIOENCODING="latin-1",
    )
    args = [str(NQUEENS), "--threads", "1"]
    py = subprocess.run(
        [sys.executable, "-m", "cbls", *args], capture_output=True, timeout=60, env=env
    )

    assert py.returncode == 0, py.stderr
    assert py.stdout.startswith(f"cbls {cbls.__version__} — ".encode()), py.stdout


def _start_python(*args: str) -> subprocess.Popen[bytes]:
    env = dict(os.environ, PYTHONPATH=str(Path(cbls.__file__).resolve().parents[1]))
    return subprocess.Popen(
        [sys.executable, "-m", "cbls", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


@pytest.mark.skipif(not hasattr(signal, "SIGPIPE"), reason="POSIX signals")
def test_ctrl_c_ends_the_run_at_once_as_the_cpp_cli() -> None:
    # SIGINT kills, as the C++ CLI (#465). On the portfolio no Python runs on the
    # main thread mid-search, so a Python handler would wait out the budget.
    args = [str(SIMPLE), "--threads", "2", "--time-limit", "60", "--format", "jsonl"]
    proc = _start_python(*args)
    try:
        assert proc.stdout is not None
        assert b'"start"' in proc.stdout.readline()  # main has run; the solve is next
        proc.send_signal(signal.SIGINT)
        _, err = proc.communicate(timeout=10)
    finally:
        proc.kill()
    assert proc.returncode == -signal.SIGINT, err


@pytest.mark.skipif(not hasattr(signal, "SIGPIPE"), reason="POSIX signals")
def test_a_closed_pipe_ends_the_run_quietly_as_the_cpp_cli() -> None:
    # As `cbls ... | head -n 0`: the reader is gone before the result is written.
    proc = _start_python(str(SIMPLE), "--threads", "1", "--time-limit", "1", "--format", "jsonl")
    try:
        assert proc.stdout is not None and proc.stderr is not None
        proc.stdout.close()
        err = proc.stderr.read()
        proc.wait(timeout=30)
    finally:
        proc.kill()
    assert proc.returncode == -signal.SIGPIPE, err
    assert err == b""


def _binary_stdout(monkeypatch: pytest.MonkeyPatch, raw: io.BytesIO) -> None:
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="utf-8"))


def test_main_writes_to_the_python_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    # The formatters write through sys.stdout's buffer, not the process's fd 1.
    raw = io.BytesIO()
    _binary_stdout(monkeypatch, raw)
    code = cli.main([str(NQUEENS), "--threads", "1", "--format", "jsonl"])

    assert code == 0
    events = [json.loads(line)["event"] for line in raw.getvalue().splitlines()]
    assert events[0] == "start" and events[-1] == "result", raw.getvalue()


def test_main_writes_to_a_text_only_stdout() -> None:
    # A StringIO has no binary buffer; the bytes are decoded into it.
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main([str(NQUEENS), "--threads", "1"])

    assert code == 0
    assert out.getvalue().startswith(f"cbls {cbls.__version__} — "), out.getvalue()
    assert "Status:     feasible" in out.getvalue()


class _FailingStream(io.BytesIO):
    """Fails on the first progress row: the header is written first."""

    def write(self, data: object) -> int:
        assert isinstance(data, bytes)
        if b'"progress"' in data:
            raise OSError("stream closed")
        return super().write(data)


def test_a_failing_stream_ends_the_run_with_its_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The write runs under the solve's released GIL; its error ends the solve.
    _binary_stdout(monkeypatch, _FailingStream())
    code = cli.main([str(NQUEENS), "--threads", "1", "--format", "jsonl"])

    assert code == 1
    assert capsys.readouterr().err == "Error: stream closed\n"
