"""Wheel smoke test (#463), registered in ctest as `wheel_smoke`.

Builds the wheel from the checkout, installs it into a fresh venv and uses it
there, isolated from the source and build trees, as a module and through the
`cbls` console script (#465). Offline: the build runs
without isolation in the calling interpreter (the dev venv carries
scikit-build-core and nanobind), and json comes from the gated build's copy.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

# Run in the fresh venv under -I, so only the installed wheel can satisfy it.
# argv: a .cbls, an MPS and an .nl file, each solved through cbls.solve (#464).
USE_INSTALLED = """
import pathlib, sys
import cbls

installed = pathlib.Path(cbls.__file__).resolve()
assert pathlib.Path(sys.prefix).resolve() in installed.parents, installed

m = cbls.Model()
x = m.int_var(0, 10, "x")
m.add_constraint(m.sum([m.constant(3.0), m.prod(m.constant(-1.0), x)]))  # x >= 3
m.minimize(m.sum([x]))
m.close()
result = cbls.solve(m, 1.0, 42, threads=1)
assert result.feasible, "tiny model not solved"
assert result.value(x) == m.var(-(x + 1)).value == 3.0, result.values

cbls_file, mps_file, nl_file = sys.argv[1:]
models = {
    "cbls": cbls.load_model(cbls_file),
    "mps": cbls.mps_to_model(cbls.read_mps(mps_file)).model,
    "nl": cbls.nl_to_model(cbls.read_nl(nl_file)).model,
}
for kind, model in models.items():
    solved = cbls.solve(model, 1.0, 42, threads=1)
    assert solved.feasible, kind + " model not solved"
    print(kind, solved.objective)
print("OK", installed)
"""

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


def build_wheel(source: Path, work: Path, args: argparse.Namespace) -> Path:
    """The PEP 517 hook pip would call, without isolation; returns the wheel."""
    if importlib.util.find_spec("scikit_build_core") is None:
        sys.exit(
            f"scikit-build-core is not installed in {sys.prefix}; "
            "run `uv sync --extra dev --extra benchmarks`"
        )
    dist = work / "dist"
    dist.mkdir(parents=True)
    settings = {"build-dir": str(work / "build")}
    if args.json_src:
        settings["cmake.define.FETCHCONTENT_SOURCE_DIR_JSON"] = args.json_src
    if args.cxx:  # the gated build's compiler, not whatever `c++` is
        settings["cmake.define.CMAKE_CXX_COMPILER"] = args.cxx
    jobs = args.jobs
    hook = (
        "import sys\n"
        "from scikit_build_core.build import build_wheel\n"
        "print(build_wheel(sys.argv[1], dict(a.split('=', 1) for a in sys.argv[2:])))\n"
    )
    env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=str(jobs))
    subprocess.run(
        [sys.executable, "-c", hook, str(dist), *(f"{k}={v}" for k, v in settings.items())],
        cwd=source,
        env=env,
        check=True,
    )
    wheels = sorted(dist.glob("cbls-*.whl"))
    if len(wheels) != 1:
        sys.exit(f"expected one wheel in {dist}, found {wheels}")
    return wheels[0]


def check_contents(wheel: Path) -> None:
    """The package and its metadata only: no C++ binary, headers or CMake files."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        entry_points = [n for n in names if n.endswith(".dist-info/entry_points.txt")]
        scripts = archive.read(entry_points[0]).decode() if entry_points else ""
    stray = [
        n
        for n in names
        if not n.startswith("cbls/") and not n.split("/", 1)[0].endswith(".dist-info")
    ]
    if stray:
        sys.exit(f"wheel carries files outside the package: {stray}")
    for required in ("__init__.py", "__main__.py", "_solve.py", "_cbls_core.pyi", "py.typed"):
        if f"cbls/{required}" not in names:
            sys.exit(f"wheel has no cbls/{required}: {names}")
    if not any(n.startswith("cbls/_cbls_core.") and n.endswith((".so", ".pyd")) for n in names):
        sys.exit(f"wheel has no compiled cbls/_cbls_core: {names}")
    if "cbls = cbls.__main__:main" not in scripts:
        sys.exit(f"wheel declares no cbls console script: {scripts!r}")


def install(wheel: Path, env_dir: Path) -> tuple[Path, Path | None]:
    """Install into a fresh venv; returns its interpreter and `cbls` console script.

    With pip when the base interpreter has ensurepip. A Debian python3 without
    python3-venv lacks it, and there the wheel, which check_contents has shown
    to be the package and its dist-info only, is unpacked into site-packages,
    which generates no console script: None.
    """
    bin_dir = env_dir / ("Scripts" if os.name == "nt" else "bin")
    python = bin_dir / ("python.exe" if os.name == "nt" else "python")
    try:
        venv.create(env_dir, with_pip=True, clear=True)
    except subprocess.CalledProcessError:
        print("ensurepip unavailable: unpacking the wheel instead of pip-installing it")
        venv.create(env_dir, with_pip=False, clear=True)
        query = "import sysconfig; print(sysconfig.get_paths()['platlib'])"
        site = subprocess.run(
            [str(python), "-I", "-c", query], capture_output=True, text=True, check=True
        ).stdout.strip()
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(site)
        return python, None
    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", "--no-index", "--no-deps", str(wheel)],
        check=True,
    )
    return python, bin_dir / ("cbls.exe" if os.name == "nt" else "cbls")


def run_cli(python: Path, script: Path | None, model: Path, work: Path) -> None:
    """Both entry points (#465): `cbls --help` and `python -m cbls MODEL --format jsonl`.

    Unpacked, with no script, the script's target is called as its launcher would.
    """
    launcher = "import sys; from cbls.__main__ import main; sys.exit(main())"
    help_cmd = [str(script)] if script else [str(python), "-I", "-c", launcher]
    shown = subprocess.run(
        [*help_cmd, "--help"], cwd=work, capture_output=True, text=True, check=True
    )
    if not shown.stdout.startswith("Usage: cbls"):
        sys.exit(f"cbls --help printed no usage: {shown.stdout!r}")
    args = [str(model), "--format", "jsonl", "--threads", "1", "--time-limit", "1"]
    ran = subprocess.run(
        [str(python), "-I", "-m", "cbls", *args], cwd=work, capture_output=True, text=True
    )
    last = ran.stdout.splitlines()[-1] if ran.stdout else ""
    if ran.returncode != 0 or '"event":"result"' not in last:
        sys.exit(f"python -m cbls failed ({ran.returncode}): {ran.stdout}{ran.stderr}")
    print("CLI OK", last)


def install_and_use(wheel: Path, work: Path, source: Path) -> None:
    python, script = install(wheel, work / "venv")
    mps = work / "tiny.mps"
    mps.write_text(TINY_MPS)
    files = [source / "examples" / "simple.cbls", mps, source / "tests" / "data" / "filter.nl"]
    subprocess.run([str(python), "-I", "-c", USE_INSTALLED, *map(str, files)], cwd=work, check=True)
    run_cli(python, script, files[0], work)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--json-src", default="", help="FETCHCONTENT_SOURCE_DIR_JSON")
    parser.add_argument("--cxx", default="", help="CMAKE_CXX_COMPILER")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()

    shutil.rmtree(args.work, ignore_errors=True)
    args.work.mkdir(parents=True)
    wheel = build_wheel(args.source.resolve(), args.work, args)
    check_contents(wheel)
    install_and_use(wheel, args.work, args.source.resolve())


if __name__ == "__main__":
    main()
