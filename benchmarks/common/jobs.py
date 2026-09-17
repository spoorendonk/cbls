"""Running solver processes: one process per job, bounded, and never silently.

Every driver runs each job as its own process, so a job that dies takes only
itself. What differs between them is policy -- which jobs to skip, what a failure
records, whether a failure stops the run -- and that stays with each driver.
What they share is the mechanics, here:

* `run_process` -- one process with an optional wall-clock timeout, an optional
  address-space cap, optionally its own process group, and its output captured
  and optionally logged. A timeout is an `Outcome`, not an exception, so no
  caller can forget to record it.
* `run_jobs` -- a plan run `workers` at a time with a serial tail, in plan
  order. A serial run is a plain loop: nothing is started ahead of the job in
  hand, so an exception stops the run before the next job begins.
"""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from pathlib import Path


def with_memory_limit(command: Sequence[str], limit_gb: float | None) -> list[str]:
    """Wrap a command so the child caps its own address space before exec.

    Not `preexec_fn`: a driver may run jobs from a thread pool, and preexec_fn in
    a multithreaded parent can deadlock the child between fork and exec -- the one
    failure mode an unattended multi-hour run must not have. `ulimit` in the
    intermediate shell does the same job with no fork-safety question. `"$0" "$@"`
    passes the argv through without re-quoting it.
    """
    if not limit_gb:
        return list(command)
    limit_kb = int(limit_gb * 1024 * 1024)
    # `&&`, not `;`: if the limit cannot be set (a lower hard limit already in
    # force), the job must fail loudly rather than run uncapped.
    return ["/bin/sh", "-c", f'ulimit -v {limit_kb} && exec "$0" "$@"', *command]


@dataclass(frozen=True)
class Outcome:
    """How one process ended."""

    #: None when the timeout killed it: no exit status was ever observed.
    returncode: int | None
    stdout: str
    stderr: str
    elapsed: float

    @property
    def timed_out(self) -> bool:
        return self.returncode is None


def run_process(
    command: Sequence[str],
    *,
    timeout: float | None = None,
    mem_limit_gb: float | None = None,
    own_session: bool = False,
    log: Path | None = None,
) -> Outcome:
    """Run one job's process to completion or to its timeout.

    `own_session` puts the child in its own process group. Without it a Ctrl-C
    at the driver reaches every in-flight child, each of which then leaves a
    failure record that resume treats as done.

    `log`, when given, receives stdout then stderr -- written whatever the exit
    status, because the failed run is the one whose log gets read.
    """
    started = time.monotonic()
    try:
        completed = subprocess.run(
            with_memory_limit(command, mem_limit_gb),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            start_new_session=own_session,
        )
    except subprocess.TimeoutExpired:
        outcome = Outcome(None, "", "", time.monotonic() - started)
    else:
        outcome = Outcome(
            completed.returncode, completed.stdout, completed.stderr, time.monotonic() - started
        )
    if log is not None:
        log.write_text(outcome.stdout + outcome.stderr)
    return outcome


def run_jobs[J, R](
    jobs: Iterable[J],
    run_one: Callable[[J], R],
    *,
    workers: int = 1,
    serial_tail: Iterable[J] = (),
) -> Iterator[R]:
    """`run_one` over `jobs`, `workers` at a time, then over `serial_tail` one at a time.

    Results are yielded in plan order as they become available, so a driver can
    print each line while an hours-long run is still going. The tail is for the
    jobs that must not share the machine -- the largest instances, run alone
    rather than four-up against a memory limit -- and starts only once every
    job before it has finished.

    `run_one` should not raise on a job's own failure: in a pool, an exception
    is re-raised where the caller iterates, which cancels every job still queued
    and the whole tail -- on an unattended run, one transient OSError costing the
    rest of the roster. A serial run (`workers` of 1, and the tail) is a plain
    loop, so there an exception stops the run before the next job starts, which
    is what a driver that must stop on its first failure wants.
    """
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(run_one, jobs)
    else:
        yield from map(run_one, jobs)
    yield from map(run_one, serial_tail)
