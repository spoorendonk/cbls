"""Generate JSONL instance files for UC-CHPED benchmarks.

Reads from data.py and dumps each instance as a single JSON object per file.
"""
import json
import os
from data import (
    UCP_13UNIT, UCP_40UNIT, UCP_100UNIT, UCP_200UNIT,
    extend_horizon,
)

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def write_instance(inst):
    """Serialize a single instance to JSONL."""
    serializable = dict(inst)
    serializable["known_bounds"] = {
        str(k): list(v) for k, v in inst["known_bounds"].items()
    }
    filename = f"{inst['name']}.jsonl"
    with open(filename, "w") as f:
        json.dump(serializable, f)
        f.write("\n")
    print(f"Wrote {filename} ({inst['n_units']} units, {inst['n_periods']} periods)")


# Base 24h instances
for inst in [UCP_13UNIT, UCP_40UNIT, UCP_100UNIT, UCP_200UNIT]:
    write_instance(inst)

# Extended horizons for larger instances
for base in [UCP_100UNIT, UCP_200UNIT]:
    for T in [48, 168]:
        write_instance(extend_horizon(base, T))
