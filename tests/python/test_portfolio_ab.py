"""The #179 portfolio A/B driver's scoring (benchmarks/minlplib/portfolio_ab.py).

The runs themselves are the engine's; what this pins is the statistics the
published reading rests on -- in particular that the instance level treats
seeds as nested in instances rather than as independent pairs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmarks.minlplib.portfolio_ab import REPO, _sign_test, analyze, main, t975

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


def test_analyze_excludes_a_pair_whose_arm_lost_a_worker(tmp_path: Path) -> None:
    """#170's rule: an arm that ran fewer workers than asked is a smaller portfolio."""
    rows = []
    for seed, completed in ((1, 4), (2, 3), (3, 4)):
        for share, pi in ((False, 0.5), (True, 0.4)):
            rows.append(
                {
                    "instance": "a" if seed != 3 else "b",
                    "seed": seed,
                    "share": share,
                    "pi": pi,
                    "gap": 0.1,
                    "threads": 4,
                    "workers_completed": completed if share else 4,
                }
            )
    path = tmp_path / "ab.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    lines = analyze(path)
    assert lines[0].startswith("excluded 1 pair(s)")
    assert "a/2" in lines[0]
    assert "n=2" in next(line for line in lines if line.startswith("pi pairs"))


def _fake_build(tmp_path: Path, build_type: str) -> Path:
    """A build directory whose binary prints one fixed result line."""
    build = tmp_path / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_BUILD_TYPE:STRING={build_type}\nCMAKE_HOME_DIRECTORY:INTERNAL={REPO}\n"
    )
    binary = build / "cbls_minlplib_portfolio"
    result = {
        "feasible": True,
        "objective": 2.0,
        "workers_completed": 4,
        "iterations": 1,
        "shared_bound_tightenings": 0,
        "own_best_behind_global": 0,
        "bound_behind_global_batches": 0,
        "bound_behind_global_seconds": 0.0,
        "trace": [[0.1, 2.0]],
    }
    binary.write_text(f"#!/bin/sh\necho '{json.dumps(result)}'\n")
    binary.chmod(0o755)
    return build


def _inst_dir(tmp_path: Path) -> Path:
    inst = tmp_path / "inst"
    inst.mkdir()
    (inst / "bounds.csv").write_text("instance,objsense,primal_bks\nx,min,1.0\n")
    (inst / "x.nl").write_text("")
    return inst


def _run(tmp_path: Path, build: Path, out: Path, budget: str = "1") -> int:
    return main(
        [
            "run",
            *("--instances", "x", "--seeds", "1", "--budget", budget),
            *(
                "--out",
                str(out),
                "--build-dir",
                str(build),
                "--inst-dir",
                str(_inst_dir_once(tmp_path)),
            ),
        ]
    )


def _inst_dir_once(tmp_path: Path) -> Path:
    inst = tmp_path / "inst"
    return inst if inst.exists() else _inst_dir(tmp_path)


def test_run_refuses_an_unoptimised_build(tmp_path: Path) -> None:
    out = tmp_path / "ab.jsonl"
    assert _run(tmp_path, _fake_build(tmp_path, "Debug"), out) == 2
    assert not out.exists()


def test_run_stamps_rows_repairs_a_torn_tail_and_refuses_another_campaign(tmp_path: Path) -> None:
    build = _fake_build(tmp_path, "Release")
    out = tmp_path / "ab.jsonl"
    out.write_text('{"instance": "x", "se')  # a kill mid-append
    assert _run(tmp_path, build, out) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2, "the torn line was dropped, both arms appended"
    assert {r["share"] for r in rows} == {True, False}
    assert all(r["commit_sha"] and r["budget"] == 1.0 and r["threads"] == 4 for r in rows)
    machine = [
        json.loads(line) for line in (tmp_path / "ab.jsonl.machine.jsonl").read_text().splitlines()
    ]
    assert machine[0]["machine"]["cpu_count"]
    assert _run(tmp_path, build, out) == 0  # resume: nothing left to do
    assert len(out.read_text().splitlines()) == 2
    assert _run(tmp_path, build, out, budget="2") == 2  # another campaign's file
