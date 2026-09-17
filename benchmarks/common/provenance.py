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

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(repo_root: Path, *argv: str) -> str:
    out = subprocess.run(["git", *argv], cwd=repo_root, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def commit_sha(repo_root: Path = REPO_ROOT) -> str:
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

    Raises `subprocess.CalledProcessError` or `OSError` when git cannot answer;
    whether that is fatal is the caller's decision.
    """
    sha = _git(repo_root, "rev-parse", "--short=7", "HEAD")
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    return f"{sha}-dirty" if dirty else sha


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


def build_dir_problems(build_dir: Path, repo_root: Path = REPO_ROOT) -> list[str]:
    """Refusals about a build directory whose binary would be measured and published.

    Empty when the directory is a configured, optimised, uninstrumented build of
    `repo_root`. Each refusal is a way to publish a wall-clock-budgeted number
    measured on an engine nobody runs.
    """
    cache = cmake_cache(build_dir)
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
    if cache.get("CBLS_SANITIZE"):
        problems.append(
            f"{build_dir} is configured with CBLS_SANITIZE={cache['CBLS_SANITIZE']}; "
            "these are wall-clock-budgeted solves and a sanitizer build measures a "
            "different engine. Use a separate build directory for sanitizers."
        )
    if cache.get("CBLS_PROFILE", "OFF") not in ("OFF", "FALSE", "0", ""):
        problems.append(
            f"{build_dir} is configured with CBLS_PROFILE={cache['CBLS_PROFILE']}; "
            "frame pointers cost throughput and docs/profiling.md says a build-profile "
            "wall-clock is not a benchmark number. Use a separate build directory."
        )
    home = cache.get("CMAKE_HOME_DIRECTORY")
    if home and Path(home).resolve() != repo_root:
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
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "memory_total_kib": memory_total_kib(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }
