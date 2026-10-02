"""#148: the Table 2 bounds describe ucp13 but not this repository's ucp40.

`benchmarks/uc-chped/instance_identity.py` established it (FIDELITY.md section
7): ucp40 caps units 19-20 at 500 MW where the authors' instance has 550, and the
authors' 1-period optimum is infeasible on ours. What the runner must therefore
do, and what this pins, is mark ucp40's cited rows as bounds for a related system
and publish no gap for a measured ucp40 row -- while ucp13 keeps both.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UC_CHPED_BINARY = ROOT / "build" / "cbls_uc_chped"


def _rows(path: Path) -> list[dict[str, str]]:
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(lines))


def test_uc_chped_marks_ucp40_bounds_as_a_related_system(tmp_path: Path) -> None:
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    src = ROOT / "benchmarks" / "instances" / "uc-chped"
    inst_dir = tmp_path / "uc-chped"
    inst_dir.mkdir()
    for f in src.iterdir():
        if f.is_file():
            (inst_dir / f.name).write_bytes(f.read_bytes())
    out = tmp_path / "o.csv"

    result = subprocess.run(
        [
            str(UC_CHPED_BINARY),
            str(inst_dir),
            *("--instance", "ucp13", "--instance", "ucp40"),
            *("--no-time-limit", "--max-iterations", "50", "--out", str(out)),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    rows = _rows(out)
    cited = {(r["instance"], r["method"]) for r in rows if r["source"] != "this work"}
    assert {m for i, m in cited if i == "ucp13"} == {"Pedroso MIP (1hr)"}
    assert {m for i, m in cited if i == "ucp40"} == {"Pedroso MIP (1hr) [related system]"}

    measured = [r for r in rows if r["source"] == "this work"]
    ucp40 = [r for r in measured if r["instance"] == "ucp40"]
    assert ucp40, "no measured ucp40 rows"
    for r in ucp40:
        assert r["gap_pct"] == "" and r["lb"] == "", r
    # Every ucp13 horizon has a published bound, so a feasible row carries a gap.
    assert any(r["gap_pct"] != "" for r in measured if r["instance"] == "ucp13"), measured
