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
  order.
* `wallclock_lock` -- the one lock every serial wall-clock-budgeted driver
  holds for its whole run, so two of them never share the machine.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import socket
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

#: The lock `wallclock_lock` takes. ONE fixed path, resolved once at import:
#: deliberately not `$XDG_STATE_HOME`, which two shells can set differently and
#: which would then hand them two locks. Per user -- another user's run on the
#: same machine is not excluded by it (the ablation driver's load-average gate
#: is what notices that). Named by host, so a home directory shared between
#: machines (NFS, where flock is enforced across clients) does not make one
#: machine's run refuse another's. Tests point it elsewhere by patching this name.
WALLCLOCK_LOCK = (
    Path.home() / ".local" / "state" / "cbls" / f"wallclock-{socket.gethostname()}.lock"
)


class LockHeldError(RuntimeError):
    """`wallclock_lock` is held by another process; the message names it."""


@contextlib.contextmanager
def wallclock_lock(owner: str) -> Iterator[None]:
    """Hold the machine's wall-clock lock for the whole run, or raise `LockHeldError`.

    Every driver that runs wall-clock-budgeted solves serially and publishes or
    compares their numbers takes it (`minlplib/run_benchmark.py`,
    `minlplib/run_ablation.py`): two such runs sharing the machine halve each
    other's iteration counts with nothing in either record saying so, which is
    what makes a run record's "one solve at a time" true rather than asserted.
    Non-blocking: a second driver refuses rather than waiting. `flock` is held on
    the inode and dropped by the kernel when the holder exits, so a lock file a
    crash left behind is not stale -- never delete it to get past a refusal.
    `owner` is written into the file so the refusal can name who holds it.
    """
    path = WALLCLOCK_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    # "a+", not "w": truncating before the flock fails would wipe the holder's line.
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.seek(0)
            holder = handle.read().strip() or "holder unknown"
            raise LockHeldError(
                f"another wall-clock benchmark holds {path} ({holder}); timed solves must not "
                "share the machine. Wait for it to finish."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()} {owner}\n")
        handle.flush()
        yield


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

    #: The exit status; negative for a signal, as `subprocess` reports it. A
    #: process the timeout killed reports SIGKILL, which is what killed it.
    returncode: int
    stdout: str
    stderr: str
    elapsed: float
    #: Whether the timeout, rather than the process itself, ended it. Checked
    #: before `returncode` by any caller that set a timeout.
    timed_out: bool = False
    #: User plus system CPU seconds of the child and every descendant it reaped,
    #: from `wait4`. Divided by `elapsed` it is the number of cores the job
    #: actually used (#240). None only for an `Outcome` built by hand.
    cpu_seconds: float | None = None
    #: The child's peak resident set in KiB (`ru_maxrss`, which Linux reports in
    #: KiB). None only for an `Outcome` built by hand.
    peak_rss_kib: int | None = None


def with_cpuset(command: Sequence[str], cpuset: Sequence[int] | None) -> list[str]:
    """Wrap a command so it runs pinned to `cpuset`, via `taskset`, which execs it.

    Not `os.sched_setaffinity` in a `preexec_fn`, for the same fork-safety reason
    as `with_memory_limit`. Affinity is inherited, so every thread the solver
    starts stays on the set.
    """
    if not cpuset:
        return list(command)
    return ["taskset", "-c", ",".join(str(cpu) for cpu in cpuset), *command]


def parse_cpu_list(text: str) -> list[int]:
    """Read a kernel CPU list (`0,2,4-6`), the format `taskset -c` and sysfs use.

    Raises ValueError on a malformed entry, a descending range or a repeated CPU:
    a set named twice would pin fewer cores than it claims to.
    """
    cpus: list[int] = []
    for part in text.strip().split(","):
        low, _, high = part.partition("-")
        first, last = int(low), int(high or low)
        if first < 0 or last < first:
            raise ValueError(f"bad CPU range {part!r} in {text!r}")
        cpus.extend(range(first, last + 1))
    if len(set(cpus)) != len(cpus):
        raise ValueError(f"CPU list {text!r} names a CPU twice")
    return cpus


#: Where the kernel publishes CPU topology. A parameter of `cpuset_problems` only
#: so a test can hand it a made-up machine.
SYS_CPU = Path("/sys/devices/system/cpu")


def cpuset_problems(cpus: Sequence[int], sys_cpu: Path = SYS_CPU) -> list[str]:
    """Why `cpus` is not a set of distinct physical cores this process may use.

    Two SMT siblings are one core's execution units shared by two threads, so an
    N-thread run pinned to them gets less than N cores, and its CPU/wall ratio
    would still read N (#240). Topology that cannot be read is a problem too: the
    claim cannot be checked.
    """
    problems: list[str] = []
    allowed = os.sched_getaffinity(0)
    if outside := sorted(set(cpus) - allowed):
        problems.append(f"CPUs {outside} are not in this process's affinity {sorted(allowed)}")
    for cpu in cpus:
        path = sys_cpu / f"cpu{cpu}" / "topology" / "thread_siblings_list"
        try:
            siblings = set(parse_cpu_list(path.read_text()))
        except (OSError, ValueError) as exc:
            problems.append(f"cannot read CPU {cpu}'s SMT siblings: {exc}")
            continue
        if shared := sorted((siblings - {cpu}) & set(cpus)):
            problems.append(f"CPU {cpu} shares a physical core with {shared}")
    return problems


#: How often `run_process` polls for the child's exit. The cost is at most this
#: much added to a job's wall time, against budgets of seconds to minutes.
_POLL_SECONDS = 0.01


def run_process(
    command: Sequence[str],
    *,
    timeout: float | None = None,
    mem_limit_gb: float | None = None,
    own_session: bool = False,
    log: Path | None = None,
    cpuset: Sequence[int] | None = None,
    measure: bool = False,
) -> Outcome:
    """Run one job's process to completion or to its timeout.

    `own_session` puts the child in its own process group. Without it a Ctrl-C
    at the driver reaches every in-flight child, each of which then leaves a
    failure record that resume treats as done.

    `log`, when given, receives stdout then stderr -- written whatever the exit
    status, because the failed run is the one whose log gets read.

    `cpuset` pins the child (`with_cpuset`). `measure` fills the `Outcome`'s CPU
    seconds and peak RSS (`_run_measured`); off, the child runs under plain
    `subprocess.run`, which is the seam the drivers' tests fake.
    """
    argv = with_cpuset(with_memory_limit(command, mem_limit_gb), cpuset)
    run = _run_measured if measure else _run_plain
    outcome = run(argv, timeout, own_session)
    if log is not None:
        log.write_text(outcome.stdout + outcome.stderr)
    return outcome


def _run_plain(argv: list[str], timeout: float | None, own_session: bool) -> Outcome:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            start_new_session=own_session,
        )
    except subprocess.TimeoutExpired:
        return Outcome(-signal.SIGKILL, "", "", time.monotonic() - started, timed_out=True)
    return Outcome(
        completed.returncode, completed.stdout, completed.stderr, time.monotonic() - started
    )


def _run_measured(argv: list[str], timeout: float | None, own_session: bool) -> Outcome:
    """`_run_plain`, reaping the child with `wait4` for its resource usage.

    `wait4` is the only call that returns one child's usage: `RUSAGE_CHILDREN`
    sums every child this process ever reaped, which concurrent jobs share, and
    its peak RSS is the largest of them. Output goes to temporary files so the
    poll can never stall on a full pipe.
    """
    started = time.monotonic()
    deadline = None if timeout is None else started + timeout
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        process = subprocess.Popen(argv, stdout=out, stderr=err, start_new_session=own_session)
        timed_out = False
        try:
            while True:
                pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if pid:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    process.kill()
                    _, status, usage = os.wait4(process.pid, 0)
                    break
                time.sleep(_POLL_SECONDS)
        except BaseException:
            process.kill()
            process.wait()
            raise
        elapsed = time.monotonic() - started
        # Reaped here, so `Popen` must not try to wait for the pid again.
        process.returncode = os.waitstatus_to_exitcode(status)
        out.seek(0)
        err.seek(0)
        stdout = out.read().decode(errors="replace")
        stderr = err.read().decode(errors="replace")
    if timed_out:
        # The output of a killed job is dropped, as `subprocess.run` drops it.
        process.returncode, stdout, stderr = -signal.SIGKILL, "", ""
    return Outcome(
        process.returncode,
        stdout,
        stderr,
        elapsed,
        timed_out=timed_out,
        cpu_seconds=usage.ru_utime + usage.ru_stime,
        peak_rss_kib=usage.ru_maxrss,
    )


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

    Every job runs on a pool thread, one-worker runs and the tail included, never
    on the caller's. That is what a Ctrl-C relies on: it lands in the caller, and
    leaving the pool waits for the jobs in flight, so a solve that is running
    finishes and writes its record. On the caller's own thread the interrupt
    would land inside `subprocess.run`, which kills the child and leaves nothing.

    `run_one` should not raise on a job's own failure: an exception is re-raised
    where the caller iterates, which cancels every job still queued and the whole
    tail -- on an unattended run, one transient OSError costing the rest of the
    roster.
    """
    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        yield from pool.map(run_one, jobs)
    with ThreadPoolExecutor(max_workers=1) as pool:
        yield from pool.map(run_one, serial_tail)
