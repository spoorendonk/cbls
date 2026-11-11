"""Helpers the binding tests share. No `_cbls_core` import here: conftest.py
finds binding tests by their own imports, so a test file that needs the
bindings must still import them itself."""

import os
import subprocess
import sys


def vid(handle: int) -> int:
    """Variable id from a variable handle (negative, as the builders return)."""
    return -(handle + 1)


def run_scenario(script: str, name: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """Run scenario `name` of test file `script` (its `__main__` block) in a
    child interpreter, under a hard deadline of `timeout` seconds."""
    env = dict(os.environ)
    # The child imports _cbls_core directly, without conftest.py to place the
    # build directory on the path -- hand it this process's search path instead.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    # Nearly every scenario assertion lives in the child, so an inherited
    # PYTHONOPTIMIZE would strip them and leave a child that prints OK and
    # exits 0 without having checked anything.
    env["PYTHONOPTIMIZE"] = "0"
    return subprocess.run(
        [sys.executable, os.path.abspath(script), name],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
