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
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
UC_CHPED_BINARY = ROOT / "build" / "cbls_uc_chped"
INSTANCES = ROOT / "benchmarks" / "instances" / "uc-chped"


def _rows(path: Path) -> list[dict[str, str]]:
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
    return list(csv.DictReader(lines))


@pytest.mark.parametrize("name", ["ucp40", "ucp100", "ucp200"])
def test_uc_chped_40_unit_copies_carry_the_sources_pmax_at_units_19_20(name: str) -> None:
    """Every copy of the 40-unit system (the `i % 40` cycle) has 550 MW at
    units 19-20, as the authors' `ucp40()` does (#193)."""
    inst = json.loads((INSTANCES / f"{name}.jsonl").read_text())
    p_max = inst["P_max"]
    copies = range(0, inst["n_units"], 40)
    assert [(p_max[k + 18], p_max[k + 19]) for k in copies] == [(550.0, 550.0)] * len(copies)


_COMMITTED = sorted(INSTANCES.glob("*.jsonl"))


def test_uc_chped_committed_instances_are_all_checked() -> None:
    """The glob below must see the roster, or the capacity test proves nothing."""
    assert {f.stem for f in _COMMITTED} >= {
        "ucp13",
        "ucp40",
        "ucp100",
        "ucp200",
        "ucp100-48p",
        "ucp100-168p",
        "ucp200-48p",
        "ucp200-168p",
    }


@pytest.mark.parametrize("path", _COMMITTED, ids=[f.stem for f in _COMMITTED])
def test_uc_chped_instance_has_capacity_for_demand_plus_reserve_in_every_period(
    path: Path,
) -> None:
    """No committed instance asks, in any period, for more demand + spinning
    reserve than the sum of every unit's P_max. Such a period is infeasible
    under any assignment, so a run on it says nothing about the search. #152
    found it at ucp40/ucp200 period 12 (the P_max error, #193) and #194 in the
    48h/168h instances `extend_horizon()` builds."""
    inst = json.loads(path.read_text())
    capacity = sum(inst["P_max"])
    short = {
        t + 1: inst["demand"][t] + inst["reserve"][t] - capacity
        for t in range(inst["n_periods"])
        if inst["demand"][t] + inst["reserve"][t] > capacity
    }
    assert short == {}, f"{path.stem}: MW short per (1-indexed) period {short}"


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
        assert any(r["gap_pct"] != "" for r in mine), f"no scored {name} row: {mine}"


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


def _uc_chped_data() -> ModuleType:
    # The module sits in a hyphenated directory, so it is loaded by path.
    spec = importlib.util.spec_from_file_location("uc_chped_data", INSTANCES / "data.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("base_name", "n_periods"), [("ucp100", 48), ("ucp100", 168), ("ucp200", 48), ("ucp200", 168)]
)
def test_uc_chped_extend_horizon_never_exceeds_the_base_day_and_matches_the_committed_file(
    base_name: str, n_periods: int
) -> None:
    """#194's construction rule, pinned at the generator rather than only on its
    output: every extended period's demand and reserve is at most the base
    profile's in the same hour, no two days of a week are exact copies, and the
    committed jsonl is what `extend_horizon()` builds today (so a rule change
    that is not regenerated fails here, not silently)."""
    data = _uc_chped_data()
    base: dict[str, Any] = getattr(data, f"UCP_{base_name[3:]}UNIT")
    ext: dict[str, Any] = data.extend_horizon(base, n_periods)
    hours = base["n_periods"]
    for key in ("demand", "reserve"):
        over = [t + 1 for t in range(n_periods) if ext[key][t] > base[key][t % hours]]
        assert over == [], f"{key} above the base day at periods {over}"
    days = [tuple(ext["demand"][d * hours : (d + 1) * hours]) for d in range(n_periods // hours)]
    assert len(set(days)) == len(days), "two extended days are exact copies"

    committed = json.loads((INSTANCES / f"{base_name}-{n_periods}p.jsonl").read_text())
    for key in ("n_periods", "demand", "reserve", "P_max"):
        assert committed[key] == ext[key], (
            f"{key} differs from extend_horizon(); rerun gen_jsonl.py"
        )
