"""Generate JSONL instance files for nuclear outage scheduling benchmarks."""
import json
import os
from data import MINI, SMALL, MEDIUM

os.chdir(os.path.dirname(os.path.abspath(__file__)))

for inst in [MINI, SMALL, MEDIUM]:
    serializable = dict(inst)
    # Convert tuple values in known_bounds to lists for JSON serialization
    serializable["known_bounds"] = {
        str(k): list(v) for k, v in inst["known_bounds"].items()
    }
    filename = f"{inst['name']}.jsonl"
    with open(filename, "w") as f:
        json.dump(serializable, f)
        f.write("\n")
    print(f"Wrote {filename} ({inst['n_units']} units, {inst['n_periods']} periods, "
          f"{inst['n_scenarios']} scenarios, {inst['n_outages']} outages)")
