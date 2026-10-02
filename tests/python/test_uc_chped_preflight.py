"""The UC-CHPED publish preflight (#146): refusals that need no benchmark run.

`build_dir_problems` and `modified_paths` are pinned in test_benchmark_common.py;
what is pinned here is that the preflight wires both, refuses with exit 2, and
prints the SHA only from a tree it found clean. Runs `git init` in a tmp dir,
which is safe only because conftest strips the GIT_* variables a hook exports.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PREFLIGHT = Path(__file__).resolve().parents[2] / "benchmarks" / "uc-chped" / "preflight.py"
RELEASE = "CMAKE_BUILD_TYPE:STRING=Release\nCBLS_SANITIZE:STRING=\nCBLS_PROFILE:BOOL=OFF\n"


def _git(repo: Path, *argv: str) -> None:
    subprocess.run(["git", *argv], cwd=repo, check=True, capture_output=True)


def _checkout(tmp_path: Path, cache: str, *, dirty: bool = False) -> Path:
    repo = (tmp_path / "checkout").resolve()
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "engine.cpp").write_text("int main() {}\n")
    _git(repo, "add", "engine.cpp")
    identity = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    _git(repo, *identity, "-c", "core.hooksPath=/dev/null", "commit", "-qm", "c")
    if dirty:
        (repo / "engine.cpp").write_text("int main() { return 1; }\n")
    build = repo / "build"
    build.mkdir()
    home = cache.replace("{home}", str(repo))
    (build / "CMakeCache.txt").write_text(home)
    return repo


def _preflight(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PREFLIGHT),
            "--build-dir",
            str(repo / "build"),
            "--repo-root",
            str(repo),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )


HERE = "CMAKE_HOME_DIRECTORY:INTERNAL={home}\n"


def test_a_clean_release_checkout_passes_and_prints_its_sha(tmp_path: Path) -> None:
    repo = _checkout(tmp_path, RELEASE + HERE)
    result = _preflight(repo)
    assert result.returncode == 0, result.stderr
    sha = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    assert result.stdout.strip() == sha


@pytest.mark.parametrize(
    ("cache", "dirty", "refusal"),
    [
        (RELEASE + HERE, True, "modified tracked files (engine.cpp)"),
        (HERE + "CMAKE_BUILD_TYPE:STRING=Debug\n", False, "not Release"),
        (RELEASE + HERE + "CBLS_SANITIZE:STRING=address\n", False, "CBLS_SANITIZE=address"),
        (RELEASE + HERE + "CBLS_PROFILE:BOOL=ON\n", False, "CBLS_PROFILE=ON"),
        (RELEASE + "CMAKE_HOME_DIRECTORY:INTERNAL=/elsewhere\n", False, "was configured from"),
    ],
    ids=["dirty-tree", "debug", "sanitizer", "frame-pointers", "another-checkout"],
)
def test_the_preflight_refuses_a_run_whose_provenance_would_be_false(
    tmp_path: Path, cache: str, dirty: bool, refusal: str
) -> None:
    result = _preflight(_checkout(tmp_path, cache, dirty=dirty))
    assert result.returncode == 2, result.stdout
    assert refusal in result.stderr, result.stderr
    # No SHA on a refusal, so `sha=$(preflight) && cbls_uc_chped --commit "$sha"`
    # cannot go on to record one.
    assert result.stdout == ""
