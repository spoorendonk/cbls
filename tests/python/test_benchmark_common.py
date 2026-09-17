"""Tests for `benchmarks/common/`: the job runner, durable records, provenance.

Each benchmark driver's own tests pin its policy -- what it skips, what a failure
records. These pin the mechanics every driver runs through, once.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING

import pytest

from benchmarks.common.jobs import run_jobs, run_process, with_memory_limit
from benchmarks.common.provenance import build_dir_problems, commit_sha
from benchmarks.common.records import (
    atomic_write,
    csv_number,
    read_json_object,
    repair_torn_tail,
    stamp_mismatch,
    write_json,
)

if TYPE_CHECKING:
    from pathlib import Path

# --- jobs: one process ----------------------------------------------------------


def test_with_memory_limit_is_a_no_op_without_a_limit() -> None:
    assert with_memory_limit(["prog", "--flag"], None) == ["prog", "--flag"]
    assert with_memory_limit(["prog"], 0) == ["prog"]


def test_with_memory_limit_actually_caps_the_child() -> None:
    # Asserting on the wrapper's shape would pass even if the quoting were wrong,
    # so run it and ask the child what its own limit is. 2 GB in KiB.
    outcome = run_process(["/bin/sh", "-c", "ulimit -v"], mem_limit_gb=2.0)
    assert outcome.stdout.strip() == str(2 * 1024 * 1024)


def test_with_memory_limit_preserves_arguments_containing_spaces() -> None:
    outcome = run_process(["/bin/echo", "two words", "--flag=a b"], mem_limit_gb=1.0)
    assert outcome.stdout.rstrip("\n") == "two words --flag=a b"


def test_a_timeout_is_an_outcome_not_an_exception(tmp_path: Path) -> None:
    """The caller branches on it to write the record a killed job needs; an
    exception would be one a new caller could forget to catch."""
    outcome = run_process(["/bin/sleep", "10"], timeout=0.2, log=tmp_path / "job.log")
    assert outcome.timed_out
    assert outcome.returncode == -9
    assert outcome.elapsed < 5.0
    assert (tmp_path / "job.log").read_text() == ""


def test_a_failed_process_still_leaves_its_log(tmp_path: Path) -> None:
    """The failed run is the one whose log gets read."""
    log = tmp_path / "job.log"
    outcome = run_process(["/bin/sh", "-c", "echo out; echo err >&2; exit 3"], log=log)
    assert outcome.returncode == 3
    assert not outcome.timed_out
    assert log.read_text() == "out\nerr\n"


@pytest.mark.parametrize("own_session", [True, False])
def test_own_session_is_what_keeps_a_ctrl_c_off_the_child(own_session: bool) -> None:
    """Without its own session a Ctrl-C at the driver reaches every in-flight
    child, and each leaves a failure record resume then treats as done."""
    probe = "import os; print(os.getsid(0))"
    outcome = run_process([sys.executable, "-c", probe], own_session=own_session)
    assert (int(outcome.stdout) != os.getsid(0)) is own_session


# --- jobs: a plan ---------------------------------------------------------------


def test_a_serial_run_stops_before_the_job_after_a_failure() -> None:
    """A driver that must stop on its first failure raises from `run_one`; the
    next job must not already be running when it does."""
    started: list[int] = []

    def run_one(job: int) -> int:
        started.append(job)
        if job == 2:
            raise RuntimeError("job 2 failed")
        return job

    with pytest.raises(RuntimeError, match="job 2 failed"):
        list(run_jobs([1, 2, 3], run_one, serial_tail=[4]))
    assert started == [1, 2]


def test_the_batch_runs_bounded_in_plan_order_and_the_tail_runs_alone() -> None:
    lock = threading.Lock()
    inflight: list[int] = []
    peak_batch = 0
    tail_overlap: list[int] = []

    def run_one(job: int) -> int:
        nonlocal peak_batch
        with lock:
            inflight.append(job)
            if job >= 100 and len(inflight) > 1:
                tail_overlap.append(job)
            if job < 100:
                peak_batch = max(peak_batch, len(inflight))
        time.sleep(0.02)
        with lock:
            inflight.remove(job)
        return job

    results = list(run_jobs(range(8), run_one, workers=3, serial_tail=[100, 101]))
    assert results == [*range(8), 100, 101]
    assert 1 < peak_batch <= 3
    assert tail_overlap == [], "a large job ran beside another job"


# --- records --------------------------------------------------------------------


def test_an_interrupted_write_leaves_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`open(path, "w")` truncates before the first byte: a kill there replaces a
    published table with nothing, at exit 0."""
    path = tmp_path / "table.csv"
    path.write_text("the published table\n")

    def die(*args: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", die)
    with pytest.raises(KeyboardInterrupt):
        atomic_write(path, "half a new table")
    assert path.read_text() == "the published table\n"


def test_an_atomic_write_is_byte_exact_and_leaves_nothing_beside_it(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    atomic_write(path, "a,b\r\nc,d\n")
    assert path.read_bytes() == b"a,b\r\nc,d\n"
    assert [p.name for p in tmp_path.iterdir()] == ["table.csv"]


@pytest.mark.parametrize(
    "contents",
    ['{"status": "feas', b"\xff\xfe\x00binary", "[1, 2]"],
    ids=["truncated-by-a-kill", "corrupted-to-binary", "not-an-object"],
)
def test_a_record_a_kill_could_have_left_reads_as_absent(
    tmp_path: Path, contents: str | bytes
) -> None:
    path = tmp_path / "inst.json"
    if isinstance(contents, bytes):
        path.write_bytes(contents)
    else:
        path.write_text(contents)
    assert read_json_object(path) is None
    assert read_json_object(tmp_path / "absent.json") is None


def test_write_json_round_trips_and_creates_its_directory(tmp_path: Path) -> None:
    path = tmp_path / "fresh" / "cbls" / "inst.json"
    write_json(path, {"b": 1, "a": None})
    assert path.read_text() == '{\n  "a": null,\n  "b": 1\n}\n'
    assert read_json_object(path) == {"a": None, "b": 1}


def test_a_torn_final_line_is_dropped_once(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("instance,arm,seed\na,control,1\nb,control,")
    assert repair_torn_tail(path) is True
    assert path.read_text() == "instance,arm,seed\na,control,1\n"
    assert repair_torn_tail(path) is False
    assert repair_torn_tail(tmp_path / "absent.csv") is False


def test_a_stamp_refuses_only_a_resume_into_another_configuration(tmp_path: Path) -> None:
    path = tmp_path / "stamp.txt"
    assert stamp_mismatch(path, "commit=a\n", resume=True) is None  # fresh: stamped
    assert stamp_mismatch(path, "commit=a\n", resume=True) is None  # matching
    assert stamp_mismatch(path, "commit=b\n", resume=True) == "commit=a\n"
    assert path.read_text() == "commit=a\n", "a refused stamp must not be overwritten"
    # Not resuming is starting over, so the stamp is rewritten rather than checked.
    assert stamp_mismatch(path, "commit=b\n", resume=False) is None
    assert path.read_text() == "commit=b\n"


@pytest.mark.parametrize(
    ("cell", "finite"),
    [("1.5", 1.5), ("-0", 0.0), ("NaN", None), ("", None), ("  ", None), (None, None), ("x", None)],
)
def test_a_cell_with_no_value_is_nan_never_zero(cell: str | None, finite: float | None) -> None:
    value = csv_number(cell)
    assert value == finite if finite is not None else value != value


# --- provenance -----------------------------------------------------------------


def _git(repo: Path, *argv: str) -> None:
    subprocess.run(["git", *argv], cwd=repo, check=True, capture_output=True)


def test_the_commit_is_marked_dirty_only_for_modified_tracked_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain SHA from a modified checkout claims a reproducibility the result
    does not have; an untracked scratch file says nothing about the code."""
    # A git hook runs this suite with GIT_DIR and GIT_INDEX_FILE exported, and
    # every git command below would inherit them: `init` then REINITIALISES the
    # repository being committed to (setting core.bare) and `add` stages into
    # its index. Clear them before the first command, not after.
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)
    _git(tmp_path, "init", "-q")
    (tmp_path / "engine.cpp").write_text("int main() {}\n")
    _git(tmp_path, "add", "engine.cpp")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")

    clean = commit_sha(tmp_path)
    assert len(clean) == 7 and all(c in "0123456789abcdef" for c in clean)
    (tmp_path / "scratch.txt").write_text("untracked\n")
    assert commit_sha(tmp_path) == clean
    (tmp_path / "engine.cpp").write_text("int main() { return 1; }\n")
    assert commit_sha(tmp_path) == f"{clean}-dirty"


def _cache(tmp_path: Path, entries: str) -> Path:
    build = tmp_path / "build"
    build.mkdir(exist_ok=True)
    (build / "CMakeCache.txt").write_text(
        "// a comment line that is not a cache entry\nCMAKE_PROJECT_NAME:STATIC=cbls\n" + entries
    )
    return build


RELEASE_HERE = "CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_HOME_DIRECTORY:INTERNAL={home}\n"


@pytest.mark.parametrize(
    ("extra", "refusal"),
    [
        ("", None),
        # An ordinary gated build records both options empty/OFF; that must pass.
        ("CBLS_SANITIZE:STRING=\nCBLS_PROFILE:BOOL=OFF\n", None),
        # CBLS_SANITIZE is a sticky cache entry that leaves CMAKE_BUILD_TYPE=Release,
        # so the Release check alone would pass a build measured several-fold slow.
        ("CBLS_SANITIZE:STRING=address,undefined\n", "CBLS_SANITIZE=address,undefined"),
        ("CBLS_PROFILE:BOOL=ON\n", "CBLS_PROFILE=ON"),
    ],
    ids=["release", "options-off", "sanitizer", "frame-pointers"],
)
def test_a_build_dir_is_refused_for_what_it_would_measure(
    tmp_path: Path, extra: str, refusal: str | None
) -> None:
    home = (tmp_path / "checkout").resolve()
    build = _cache(tmp_path, RELEASE_HERE.format(home=home) + extra)
    problems = build_dir_problems(build, home)
    assert problems == [] if refusal is None else any(refusal in p for p in problems)


@pytest.mark.parametrize(
    ("entries", "refusal"),
    [
        (None, "CMakeCache.txt not found"),
        ("CMAKE_BUILD_TYPE:STRING=Debug\n", "not Release"),
        # The SHA is read from this checkout; the binary must come from it too.
        (RELEASE_HERE.format(home="/elsewhere"), "was configured from"),
    ],
    ids=["unconfigured", "debug", "another-checkout"],
)
def test_a_build_dir_is_refused_for_what_it_is(
    tmp_path: Path, entries: str | None, refusal: str
) -> None:
    build = tmp_path / "build" if entries is None else _cache(tmp_path, entries)
    assert any(refusal in p for p in build_dir_problems(build, tmp_path / "checkout"))
