"""Refuse a UC-CHPED published run whose rows would carry a false provenance (#146).

The runner's own guards police what a run may WRITE; they cannot see what it was
built from. `--commit` is a free-form string, so a SHA passed from a modified
tree names code that did not run, and a Debug, sanitizer or frame-pointer build
measures a different engine at a wall-clock budget. This checks exactly those,
with the checks `benchmarks/common/provenance.py` already holds for the other
drivers, and nothing else -- no staging, resume or assembly (the issue's scope
note).

On success it prints the commit SHA, so the `--commit` the runner records comes
from the same check that found the tree clean; the documented command in
`docs/benchmarks/uc-chped/README.md` reads it that way. Otherwise it prints every
refusal to stderr and exits 2, before any solving starts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.provenance import (  # noqa: E402
    REPO_ROOT,
    build_dir_problems,
    cmake_cache,
    commit_sha,
    modified_paths,
)


def preflight_problems(build_dir: Path, repo_root: Path = REPO_ROOT) -> list[str]:
    """Every reason `build_dir`'s runner may not produce a published table from `repo_root`."""
    problems: list[str] = []
    try:
        dirty = modified_paths(repo_root)
    except (subprocess.CalledProcessError, OSError) as exc:
        problems.append(f"cannot read the working tree's state at {repo_root}: {exc}")
    else:
        if dirty:
            problems.append(
                f"the working tree has modified tracked files ({', '.join(dirty)}); "
                "the recorded --commit would name code that did not run"
            )
    problems += build_dir_problems(build_dir, cmake_cache(build_dir), repo_root)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    # Relative to the working directory, like the `./build/cbls_uc_chped` the
    # documented command runs; the checkout it must belong to is this script's.
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    problems = preflight_problems(args.build_dir, repo_root)
    if problems:
        print("\n".join(f"refusing: {p}" for p in problems), file=sys.stderr)
        return 2
    print(commit_sha(repo_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
