"""Generate JSONL instance files for UC-CHPED benchmarks.

Reads from data.py and dumps each instance as a single JSON object per file.
"""
import json
import os
from data import UCP_13UNIT, UCP_40UNIT

os.chdir(os.path.dirname(os.path.abspath(__file__)))

for inst in [UCP_13UNIT, UCP_40UNIT]:
    # Convert tuple values in known_bounds to lists for JSON serialization
    serializable = dict(inst)
    serializable["known_bounds"] = {
        str(k): list(v) for k, v in inst["known_bounds"].items()
    }
    filename = f"{inst['name']}.jsonl"
    with open(filename, "w") as f:
        json.dump(serializable, f)
        f.write("\n")
    print(f"Wrote {filename}")
