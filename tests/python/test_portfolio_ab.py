"""The #179 portfolio A/B driver's scoring (benchmarks/minlplib/portfolio_ab.py).

The runs themselves are the engine's; what this pins is the statistics the
published reading rests on -- in particular that the instance level treats
seeds as nested in instances rather than as independent pairs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmarks.minlplib.portfolio_ab import _sign_test, analyze, t975

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, diffs: dict[str, list[float]]) -> None:
    rows = []
    for instance, ds in diffs.items():
        for seed, d in enumerate(ds):
            rows.append({"instance": instance, "seed": seed, "share": False, "pi": 0.5, "gap": 0.5})
            rows.append(
                {"instance": instance, "seed": seed, "share": True, "pi": 0.5 + d, "gap": 0.5}
            )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_instance_level_uses_one_mean_per_instance(tmp_path: Path) -> None:
    """Many tight seeds on one instance must not buy precision at the instance level."""
    path = tmp_path / "ab.jsonl"
    _write(path, {"a": [-0.1] * 6, "b": [0.05] * 6, "c": [-0.02] * 6})
    lines = analyze(path)
    pairs = next(line for line in lines if line.startswith("pi pairs"))
    instances = next(line for line in lines if line.startswith("pi instances"))
    assert "n=18" in pairs
    assert "n=3" in instances
    assert "better=2 worse=1 flat=0" in instances


def test_sign_test_is_two_sided() -> None:
    assert _sign_test(10, 0) == 2 / 1024
    assert _sign_test(5, 5) == 1.0
    assert _sign_test(0, 0) == 1.0


def test_t_quantile_falls_back_to_the_nearest_tabulated_df_below() -> None:
    assert t975(9) == 2.262
    assert t975(12) == 2.228
    assert t975(1000) == 1.96
