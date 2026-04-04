#!/usr/bin/env python3
"""Parse ROADEF 2010 competition data files and output summary / JSONL.

Usage:
    python parse_roadef.py data0.txt              # print summary
    python parse_roadef.py data0.txt --json        # output JSONL to stdout
"""

import json
import sys
from pathlib import Path
from typing import Any


def tokenize_line(line: str) -> list[str]:
    """Split a line into whitespace-separated tokens."""
    return line.strip().split()


class ROADEFParser:
    def __init__(self, path: str) -> None:
        with open(path) as f:
            self.lines = [line.rstrip() for line in f]
        self.pos = 0

    def has_next(self) -> bool:
        return self.pos < len(self.lines)

    def next_line(self) -> str:
        if self.pos >= len(self.lines):
            raise RuntimeError("Unexpected end of file")
        line = self.lines[self.pos]
        self.pos += 1
        return line

    def next_tokens(self) -> list[str]:
        return tokenize_line(self.next_line())

    def expect(self, keyword: str) -> list[str]:
        toks = self.next_tokens()
        if not toks or toks[0] != keyword:
            raise RuntimeError(
                f"Expected '{keyword}' but got '{toks[0] if toks else ''}' at line {self.pos}"
            )
        return toks[1:]

    def expect_int(self, keyword: str) -> int:
        return int(self.expect(keyword)[0])

    def expect_float(self, keyword: str) -> float:
        return float(self.expect(keyword)[0])

    def expect_floats(self, keyword: str) -> list[float]:
        return [float(x) for x in self.expect(keyword)]

    def expect_ints(self, keyword: str) -> list[int]:
        return [int(x) for x in self.expect(keyword)]


def parse_profile(p: ROADEFParser) -> list[tuple[float, float]]:
    n_points = p.expect_int("profile_points")
    toks = p.expect("decrease_profile")
    points: list[tuple[float, float]] = []
    for i in range(n_points):
        if 2 * i + 1 < len(toks):
            points.append((float(toks[2 * i]), float(toks[2 * i + 1])))
    return points


def parse_roadef(path: str) -> dict[str, Any]:
    p = ROADEFParser(path)
    inst: dict[str, Any] = {"name": Path(path).stem}

    # Main section
    p.expect("begin")  # begin main
    inst["T"] = p.expect_int("timesteps")
    inst["H"] = p.expect_int("weeks")
    inst["K"] = p.expect_int("campaigns")
    inst["S"] = p.expect_int("scenario")
    inst["epsilon"] = p.expect_float("epsilon")
    inst["n_type1"] = p.expect_int("powerplant1")
    inst["n_type2"] = p.expect_int("powerplant2")

    constraint_counts: dict[int, int] = {}
    for ct in range(13, 22):
        constraint_counts[ct] = p.expect_int(f"constraint{ct}")

    inst["timestep_durations"] = p.expect_floats("durations")
    inst["demand"] = []
    for _ in range(inst["S"]):
        inst["demand"].append(p.expect_floats("demand"))
    p.expect("end")  # end main

    # Power plants
    inst["type1_plants"] = []
    inst["type2_plants"] = []
    total_plants: int = inst["n_type1"] + inst["n_type2"]

    for _ in range(total_plants):
        p.expect("begin")  # begin powerplant
        name = p.expect("name")[0]
        plant_type = int(p.expect("type")[0])

        if plant_type == 1:
            t1: dict[str, Any] = {"name": name, "index": p.expect_int("index")}
            s_count = p.expect_int("scenario")
            _t_count = p.expect_int("timesteps")
            t1["pmin"] = []
            t1["pmax"] = []
            t1["cost"] = []
            for _ in range(s_count):
                t1["pmin"].append(p.expect_floats("pmin"))
                t1["pmax"].append(p.expect_floats("pmax"))
                t1["cost"].append(p.expect_floats("cost"))
            p.expect("end")
            inst["type1_plants"].append(t1)

        elif plant_type == 2:
            t2: dict[str, Any] = {"name": name, "index": p.expect_int("index")}
            t2["initial_stock"] = p.expect_float("stock")
            t2["n_cycles"] = p.expect_int("campaigns")
            t2["durations"] = p.expect_ints("durations")

            cur_mmax = p.expect_float("current_campaign_max_modulus")
            cycle_mmax = p.expect_floats("max_modulus")
            t2["mmax"] = [cur_mmax] + cycle_mmax

            t2["rmax"] = p.expect_floats("max_refuel")
            t2["rmin"] = p.expect_floats("min_refuel")
            t2["q"] = p.expect_floats("refuel_ratio")

            cur_bo = p.expect_float("current_campaign_stock_threshold")
            bo_vals = p.expect_floats("stock_threshold")
            t2["bo"] = bo_vals if len(bo_vals) >= t2["n_cycles"] + 1 else [cur_bo] + bo_vals

            t2["pmax_t"] = p.expect_floats("pmax")
            t2["amax"] = p.expect_floats("max_stock_before_refueling")
            t2["smax"] = p.expect_floats("max_stock_after_refueling")
            t2["refuel_cost"] = p.expect_floats("refueling_cost")

            fp_toks = p.expect("fuel_price")
            t2["fuel_price_end"] = float(fp_toks[-1]) if fp_toks else 0.0

            # Profiles
            t2["profiles"] = []
            # Current campaign profile
            p.expect("begin")  # begin current_campaign_profile
            t2["profiles"].append(parse_profile(p))
            p.expect("end")  # end current_campaign_profile
            # Per-cycle profiles
            for _ in range(t2["n_cycles"]):
                p.expect("begin")  # begin profile
                p.expect_int("campaign_profile")
                t2["profiles"].append(parse_profile(p))
                p.expect("end")  # end profile

            p.expect("end")  # end powerplant
            inst["type2_plants"].append(t2)

    # Constraints
    inst["ct13"] = []
    inst["spacing_constraints"] = []
    inst["ct19"] = []
    inst["ct20"] = []
    inst["ct21"] = []

    total_constraints = sum(constraint_counts.values())
    for _ in range(total_constraints):
        p.expect("begin")  # begin constraint
        ctype = p.expect_int("type")

        if ctype == 13:
            ct: dict[str, Any] = {"type": 13}
            while p.has_next():
                toks = p.next_tokens()
                if not toks:
                    continue
                if toks[0] == "end":
                    break
                if toks[0] == "index":
                    ct["index"] = int(toks[1])
                elif toks[0] == "powerplant":
                    ct["plant_idx"] = int(toks[1])
                elif toks[0] == "campaign":
                    ct["cycle"] = int(toks[1])
                elif toks[0] == "earliest_stop_time":
                    ct["TO"] = int(toks[1])
                elif toks[0] == "latest_stop_time":
                    ct["TA"] = int(toks[1])
            inst["ct13"].append(ct)

        elif 14 <= ctype <= 18:
            sc: dict[str, Any] = {"type": ctype}
            while p.has_next():
                toks = p.next_tokens()
                if not toks:
                    continue
                if toks[0] == "end" and (len(toks) < 2 or toks[1] == "constraint"):
                    break
                if toks[0] == "end" and len(toks) >= 2:
                    # CT15: "end <number>" = IF_m (period end)
                    try:
                        sc["period_end"] = int(toks[1])
                        continue
                    except ValueError:
                        break
                if toks[0] == "index":
                    sc["index"] = int(toks[1])
                elif toks[0] == "set":
                    sc["plant_set"] = [int(x) for x in toks[1:]]
                elif toks[0] == "spacing":
                    sc["spacing"] = float(toks[1])
                elif toks[0] == "start":
                    sc["period_start"] = int(toks[1])
            inst["spacing_constraints"].append(sc)

        elif ctype == 19:
            ct = {"type": 19, "usages": []}
            while p.has_next():
                toks = p.next_tokens()
                if not toks:
                    continue
                if toks[0] == "end" and (len(toks) < 2 or toks[1] == "constraint"):
                    break
                if toks[0] == "index":
                    ct["index"] = int(toks[1])
                elif toks[0] == "quantity":
                    ct["quantity"] = float(toks[1])
                elif toks[0] == "set":
                    ct["plant_set"] = [int(x) for x in toks[1:]]
                elif toks[0] == "begin" and len(toks) >= 2 and toks[1] == "period":
                    usage: dict[str, Any] = {}
                    while p.has_next():
                        ptoks = p.next_tokens()
                        if not ptoks:
                            continue
                        if ptoks[0] == "end":
                            break
                        if ptoks[0] == "powerplant":
                            usage["plant_idx"] = int(ptoks[1])
                        elif ptoks[0] == "start":
                            usage["start"] = [int(x) for x in ptoks[1:]]
                        elif ptoks[0] == "duration":
                            usage["duration"] = [int(x) for x in ptoks[1:]]
                    ct["usages"].append(usage)
            inst["ct19"].append(ct)

        elif ctype == 20:
            ct = {"type": 20}
            while p.has_next():
                toks = p.next_tokens()
                if not toks:
                    continue
                if toks[0] == "end":
                    break
                if toks[0] == "index":
                    ct["index"] = int(toks[1])
                elif toks[0] == "week":
                    ct["week"] = int(toks[1])
                elif toks[0] == "set":
                    ct["plant_set"] = [int(x) for x in toks[1:]]
                elif toks[0] == "max":
                    ct["max_allowed"] = int(toks[1])
            inst["ct20"].append(ct)

        elif ctype == 21:
            ct = {"type": 21}
            while p.has_next():
                toks = p.next_tokens()
                if not toks:
                    continue
                if toks[0] == "end":
                    break
                if toks[0] == "index":
                    ct["index"] = int(toks[1])
                elif toks[0] == "set":
                    ct["plant_set"] = [int(x) for x in toks[1:]]
                elif toks[0] == "startend":
                    ct["time_range"] = [int(x) for x in toks[1:]]
                elif toks[0] == "max":
                    ct["imax"] = float(toks[1])
            inst["ct21"].append(ct)

        else:
            while p.has_next():
                toks = p.next_tokens()
                if toks and toks[0] == "end":
                    break

    return inst


def print_summary(inst: dict[str, Any]) -> None:
    print(f"Instance: {inst['name']}")
    print(f"  Timesteps: {inst['T']}, Weeks: {inst['H']}, "
          f"Timesteps/week: {inst['T'] // inst['H']}")
    print(f"  Scenarios: {inst['S']}, Epsilon: {inst['epsilon']}")
    print(f"  Type 1 plants: {inst['n_type1']}")
    for plant in inst["type1_plants"]:
        print(f"    {plant['name']} (index {plant['index']}): "
              f"pmax_range=[{min(min(r) for r in plant['pmax']):.0f}, "
              f"{max(max(r) for r in plant['pmax']):.0f}]")

    print(f"  Type 2 plants: {inst['n_type2']}")
    for plant in inst["type2_plants"]:
        print(f"    {plant['name']} (index {plant['index']}): "
              f"stock={plant['initial_stock']:.0f}, cycles={plant['n_cycles']}, "
              f"durations={plant['durations']}")

    print(f"  CT13 windows: {len(inst['ct13'])}")
    for ct in inst["ct13"]:
        print(f"    Plant {ct['plant_idx']} cycle {ct['cycle']}: "
              f"[{ct['TO']}, {ct['TA']}]")

    print(f"  CT14-18: {len(inst['spacing_constraints'])}")
    for sc in inst["spacing_constraints"]:
        print(f"    CT{sc['type']} plants={sc.get('plant_set', [])}, "
              f"spacing={sc.get('spacing', '?')}")

    print(f"  CT19: {len(inst['ct19'])}")
    print(f"  CT20: {len(inst['ct20'])}")
    print(f"  CT21: {len(inst['ct21'])}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data_file> [--json]")
        sys.exit(1)

    inst = parse_roadef(sys.argv[1])

    if "--json" in sys.argv:
        json.dump(inst, sys.stdout, indent=None)
        print()
    else:
        print_summary(inst)
