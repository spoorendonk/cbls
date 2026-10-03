"""#148/#193: the Table 2 bounds describe this repository's ucp13 and ucp40.

`benchmarks/uc-chped/instance_identity.py` settles it against the authors' GPL
`ucp_data.py` (FIDELITY.md section 7). #148 found ucp40's units 19-20 capped at
500 MW where the source has 550, and relabelled its cited rows as bounds for a
related system; #193 corrected the data in `benchmarks/chped/` and removed the
relabel. What this pins is the corrected state: the committed instances carry
the source's limits, and the runner scores ucp40 against its own Table 2 bounds
exactly as it does ucp13.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UC_CHPED_BINARY = ROOT / "build" / "cbls_uc_chped"
INSTANCES = ROOT / "benchmarks" / "instances" / "uc-chped"


def _rows(path: Path) -> list[dict[str, str]]:
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(lines))


@pytest.mark.parametrize("name", ["ucp40", "ucp100", "ucp200"])
def test_uc_chped_40_unit_copies_carry_the_sources_pmax_at_units_19_20(name: str) -> None:
    """Every copy of the 40-unit system (the `i % 40` cycle) has 550 MW at
    units 19-20, as the authors' `ucp40()` does, and so no base 24-period
    instance asks for more demand + reserve than its total P_max (#152)."""
    inst = json.loads((INSTANCES / f"{name}.jsonl").read_text())
    p_max = inst["P_max"]
    copies = range(0, inst["n_units"], 40)
    assert [(p_max[k + 18], p_max[k + 19]) for k in copies] == [(550.0, 550.0)] * len(copies)
    capacity = sum(p_max)
    short = [
        t + 1 for t in range(inst["n_periods"]) if inst["demand"][t] + inst["reserve"][t] > capacity
    ]
    assert short == [], f"{name}: capacity-short periods {short}"


def test_uc_chped_scores_ucp40_against_its_table2_bounds(tmp_path: Path) -> None:
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    inst_dir = tmp_path / "uc-chped"
    inst_dir.mkdir()
    for f in INSTANCES.iterdir():
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
    cited = [r for r in rows if r["source"] != "this work"]
    assert {r["method"] for r in cited} == {"Pedroso MIP (1hr)"}, cited
    assert {r["instance"] for r in cited} == {"ucp13", "ucp40"}
    lbs = {(r["instance"], r["periods"]): r["lb"] for r in cited}

    measured = [r for r in rows if r["source"] == "this work"]
    for name in ("ucp13", "ucp40"):
        mine = [r for r in measured if r["instance"] == name]
        assert mine, f"no measured {name} rows"
        for r in mine:
            # Every horizon run here has a published bound, so every row carries it.
            assert r["lb"] == lbs[(name, r["periods"])], r
            assert (r["gap_pct"] != "") == (r["feasible"] == "true" and r["objective"] != ""), r


def test_instance_identity_refuses_an_upstream_file_with_another_hash(tmp_path: Path) -> None:
    """A revised or tampered `ucp_data.py` must not silently become the yardstick:
    the check loads only the file whose sha256 FIDELITY.md section 7.1 records."""
    (tmp_path / "ucp_data.py").write_text("def ucp13(periods):\n    return None\n")
    script = ROOT / "benchmarks" / "uc-chped" / "instance_identity.py"
    result = subprocess.run(
        [sys.executable, str(script), "--upstream-dir", str(tmp_path), "--skip-solves"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "refusing to compare" in result.stderr, result.stderr
