"""Generate JSONL instance files for GLSP-RP benchmark.

Generates 150 instances (3 classes x 50) as individual JSONL files,
plus 2 scaled classes (D, E) with 10 instances each.
"""
import json
import os
from dataclasses import asdict

from data import generate_all

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Standard classes: 50 each
instances = generate_all(classes=("A", "B", "C"), n_per_class=50)

# Scaled classes: 10 each
scaled = generate_all(classes=("D", "E"), n_per_class=10, base_seed=1337)
instances.extend(scaled)

# Write one JSONL file per class
by_class = {}
for inst in instances:
    by_class.setdefault(inst.cls, []).append(inst)

for cls, class_instances in sorted(by_class.items()):
    filename = f"class_{cls.lower()}.jsonl"
    with open(filename, "w") as f:
        for inst in class_instances:
            json.dump(asdict(inst), f)
            f.write("\n")
    print(f"Wrote {filename} ({len(class_instances)} instances)")

# Also write a single combined file
with open("all_instances.jsonl", "w") as f:
    for inst in instances:
        json.dump(asdict(inst), f)
        f.write("\n")
print(f"Wrote all_instances.jsonl ({len(instances)} instances total)")
