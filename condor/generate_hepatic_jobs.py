#!/usr/bin/env python3
"""Generate HTCondor job list and submit file for hepatic experiments."""

METHODS = ["baseline", "dp", "dp_synthetic", "dp_public", "dp_shampoo", "dp_shampoo_synth"]
SEEDS = [42, 123, 456]
EPSILONS = [1.0, 2.0, 4.0, 8.0]

# Tuned LRs per method (adam optimizer)
LRS = {
    "baseline": 0.001,
    "dp": 0.001,
    "dp_synthetic": 0.001,
    "dp_public": 0.001,
    "dp_shampoo": 0.001,
    "dp_shampoo_synth": 0.001,
}

OPTIMIZER = "adam"

jobs = []
for method in METHODS:
    lr = LRS[method]
    if method == "baseline":
        for seed in SEEDS:
            jobs.append(f"{method} 8.0 {seed} {lr} {OPTIMIZER}")
    else:
        for eps in EPSILONS:
            for seed in SEEDS:
                jobs.append(f"{method} {eps} {seed} {lr} {OPTIMIZER}")

# Write job list
with open("hepatic_jobs.txt", "w") as f:
    for job in jobs:
        f.write(job + "\n")

print(f"Generated {len(jobs)} jobs -> hepatic_jobs.txt")
for m in METHODS:
    n = sum(1 for j in jobs if j.startswith(m + " "))
    print(f"  {m}: {n} jobs")
