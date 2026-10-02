"""What produced a result, beyond the code that ran.

A wall-clock-budgeted result is a statement about a commit, a build and a
machine as much as about an algorithm. Everything here is read, never inferred,
and says so when it could not be read.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(repo_root: Path, *argv: str) -> str:
    out = subprocess.run(["git", *argv], cwd=repo_root, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def commit_sha(repo_root: Path = REPO_ROOT, *, ignore: Iterable[Path] = ()) -> str:
    """The commit a run is attributed to, marked `-dirty` when the tree is modified.

    A plain SHA from a modified checkout claims a reproducibility the result does
    not have -- the code that ran is not the code at that commit.

    `git rev-parse --short=7` and not `git describe`, whose output format changes
    the moment the repository gains its first tag -- a recorded commit would
    silently change shape mid-history. (The mipfeas driver used `describe`
    until #160; with no tag in the repository the two print the same string.)

    Dirtiness comes from `git status --porcelain --untracked-files=no`, which
    covers modifications to tracked files. Untracked files are deliberately not
    counted: a scratch file beside the source says nothing about the code that
    ran. The corollary is that a *new*, never-added source file does not mark the
    tree dirty, so this is a guard against edited code, not against every
    difference from HEAD.

    `ignore` names files whose modification does not count (`modified_paths`
    says exactly which entries are dropped). The default, nothing ignored, is
    every caller's behaviour but the MINLPLib re-run driver's, which passes the
    tables it publishes itself (#123).

    Raises `subprocess.CalledProcessError` or `OSError` when git cannot answer;
    whether that is fatal is the caller's decision.
    """
    sha = _git(repo_root, "rev-parse", "--short=7", "HEAD")
    return f"{sha}-dirty" if modified_paths(repo_root, ignore=ignore) else sha


def _unresolved_name(path: Path) -> Path:
    """`path` absolute, with its directories' symlinks resolved but not its own name.

    Resolving the last component too would let a tracked symlink named like an
    ignored file -- `comparison.csv -> ../src/x.cpp` -- exempt every edit to its
    target. Git tracks the link, not the target, so the name is what is compared.
    """
    absolute = path if path.is_absolute() else Path.cwd() / path
    return absolute.parent.resolve() / absolute.name


#: Porcelain XY codes of an unmerged path. Always dirty, whatever `ignore` says:
#: a conflict in a published table is not output a driver wrote.
_UNMERGED = frozenset({b"DD", b"AU", b"UD", b"UA", b"DU", b"AA", b"UU"})


def modified_paths(repo_root: Path = REPO_ROOT, *, ignore: Iterable[Path] = ()) -> list[str]:
    """The modified tracked files that make `commit_sha` say `-dirty`, as git names them.

    A rename or copy is reported `old -> new`. An entry is dropped only when every
    path it names is in `ignore` -- both sides of a rename, so a file moved onto
    an ignored name still counts -- and never when it is unmerged. Relative
    `ignore` paths resolve against the process's working directory, not
    `repo_root`; paths outside the work tree can never appear in the status, and
    are moot. Names are compared without resolving their own final symlink
    (`_unresolved_name`).

    `-z` rather than the line format, which C-quotes a path holding a space, a
    quote or a non-ASCII byte, so that it would match nothing. Porcelain paths
    are relative to the top of the work tree whatever directory git runs in,
    hence `--show-toplevel`. A rename or copy entry is `XY new` followed by a
    second NUL-terminated field holding the original path.
    """
    raw = subprocess.run(
        ["git", "status", "--porcelain", "-z", "--untracked-files=no"],
        cwd=repo_root,
        capture_output=True,
        check=True,
    ).stdout
    top = Path(_git(repo_root, "rev-parse", "--show-toplevel")).resolve()
    ignored = {_unresolved_name(path) for path in ignore}
    fields = raw.split(b"\0")
    modified: list[str] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        status, names = field[:2], [os.fsdecode(field[3:])]
        if b"R" in status or b"C" in status:
            names.insert(0, os.fsdecode(fields[index]))
            index += 1
        exempt = status not in _UNMERGED and all(
            _unresolved_name(top / name) in ignored for name in names
        )
        if not exempt:
            modified.append(" -> ".join(names))
    return modified


def cmake_cache(build_dir: Path) -> dict[str, str]:
    """The `NAME:TYPE=VALUE` entries of a build directory's cache, by name."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return {}
    entries: dict[str, str] = {}
    for line in cache.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and ":" in name and not name.startswith(("#", "//")):
            entries[name.split(":", 1)[0]] = value.strip()
    return entries


#: The values CMake's `if(<variable>)` reads as false, upper-cased. Anything else
#: -- `address`, `ON`, `yes` -- switches CBLS_SANITIZE/CBLS_PROFILE on in
#: CMakeLists.txt, so anything else must be refused here.
_CMAKE_FALSE = frozenset({"", "0", "OFF", "NO", "FALSE", "N", "IGNORE", "NOTFOUND"})


def cmake_true(value: str) -> bool:
    """Whether CMake's `if()` would read the cache value `value` as on."""
    upper = value.strip().upper()
    return upper not in _CMAKE_FALSE and not upper.endswith("-NOTFOUND")


def build_dir_problems(
    build_dir: Path, cache: dict[str, str], repo_root: Path = REPO_ROOT
) -> list[str]:
    """Refusals about a build directory whose binary would be measured and published.

    `cache` is `cmake_cache(build_dir)`, read once by the caller. Empty when the
    directory is a configured, optimised, uninstrumented build of `repo_root`.
    Each refusal is a way to publish a wall-clock-budgeted number measured on an
    engine nobody runs.
    """
    if not cache:
        return [
            f"{build_dir}/CMakeCache.txt not found; configure first, e.g.\n"
            f'    cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python"'
        ]
    problems: list[str] = []
    if cache.get("CMAKE_BUILD_TYPE") != "Release":
        problems.append(
            f"{build_dir} is CMAKE_BUILD_TYPE={cache.get('CMAKE_BUILD_TYPE') or '(empty)'}, "
            "not Release; these are wall-clock-budgeted solves and an unoptimised build "
            "measures a different engine"
        )
    # Both are sticky cache entries, so a build dir configured once with either
    # keeps it through every later flag-less `cmake -B build` while
    # CMAKE_BUILD_TYPE still reads Release. A sanitizer binary runs several-fold
    # slower and -fno-omit-frame-pointer costs throughput, so either would
    # publish wall-clock-budgeted rows measured on an engine nobody runs. See
    # docs/profiling.md.
    if cmake_true(cache.get("CBLS_SANITIZE", "")):
        problems.append(
            f"{build_dir} is configured with CBLS_SANITIZE={cache['CBLS_SANITIZE']}; "
            "these are wall-clock-budgeted solves and a sanitizer build measures a "
            "different engine. Use a separate build directory for sanitizers."
        )
    if cmake_true(cache.get("CBLS_PROFILE", "")):
        problems.append(
            f"{build_dir} is configured with CBLS_PROFILE={cache['CBLS_PROFILE']}; "
            "frame pointers cost throughput and docs/profiling.md says a build-profile "
            "wall-clock is not a benchmark number. Use a separate build directory."
        )
    # Every configured cache records it, so a cache without one is not one this
    # check can vouch for -- and skipping the comparison would pass it.
    home = cache.get("CMAKE_HOME_DIRECTORY")
    if not home:
        problems.append(
            f"{build_dir}/CMakeCache.txt has no CMAKE_HOME_DIRECTORY, so it cannot be "
            f"shown to be a build of {repo_root}"
        )
    elif Path(home).resolve() != repo_root:
        problems.append(
            f"{build_dir} was configured from {home}, but the commit SHA is read from "
            f"{repo_root}; the rows would name one checkout and measure another"
        )
    return problems


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def memory_total_kib() -> int | None:
    """Total RAM, or None off Linux. The record says what it could not measure."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def cpu_model(cpuinfo: Path = Path("/proc/cpuinfo")) -> str | None:
    """The CPU's marketing name, or None where /proc/cpuinfo has no "model name" line.

    That is off Linux, and on many ARM kernels too.

    `platform.processor()` is usually just the architecture on Linux, which says
    nothing about how fast a wall-clock-budgeted run could go.
    """
    try:
        for line in cpuinfo.read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() == "model name":
                return value.strip()
    except (OSError, ValueError):  # ValueError: a non-UTF-8 file (UnicodeDecodeError)
        return None
    return None


def machine_record() -> dict[str, object]:
    """What produced a wall-clock-limited result, beyond the code that ran.

    The same roster on half the cores, or four-up instead of one-up, is a
    different measurement. Cores are reported twice because they differ under
    cgroup or taskset confinement, and that difference is exactly the sort of
    thing that makes two runs of "the same" benchmark disagree.
    """
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "memory_total_kib": memory_total_kib(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }
