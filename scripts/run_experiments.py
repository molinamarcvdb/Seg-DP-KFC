#!/usr/bin/env python
"""
Run all 4 comparison experiments.

Methods:
  1. baseline        - No DP (upper bound on utility)
  2. dp_no_precond   - DP-SGD without preconditioning
  3. dp_adadps_public - DP-SGD + AdaDPS from public DRIVE data
  4. dp_adadps_oracle - DP-SGD + AdaDPS from private data (oracle)

Usage:
    uv run python scripts/run_experiments.py
    uv run python scripts/run_experiments.py --methods baseline dp_adadps_public
    uv run python scripts/run_experiments.py --epsilons 1.0 2.0 4.0 8.0
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


METHODS = ["baseline", "dp_no_precond", "dp_adadps_public", "dp_adadps_oracle"]


def run_experiment(method: str, epsilon: float = None, extra_args: list = None):
    """Run a single experiment."""
    cmd = [sys.executable, "scripts/train.py", "--method", method]

    if epsilon is not None and method != "baseline":
        cmd.append(f"dp.epsilon={epsilon}")

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"Running: {method}" + (f" (ε={epsilon})" if epsilon else ""))
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all DP segmentation experiments")
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS,
                        help="Methods to run")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[2.0],
                        help="Epsilon values to test (for DP methods)")
    parser.add_argument("extra", nargs="*", help="Extra args passed to train.py")
    args = parser.parse_args()

    results = []
    start_time = datetime.now()

    for method in args.methods:
        if method == "baseline":
            # Baseline doesn't use epsilon
            success = run_experiment(method, extra_args=args.extra)
            results.append({"method": method, "epsilon": None, "success": success})
        else:
            # Run DP methods for each epsilon
            for eps in args.epsilons:
                success = run_experiment(method, epsilon=eps, extra_args=args.extra)
                results.append({"method": method, "epsilon": eps, "success": success})

    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {datetime.now() - start_time}")
    print()

    for r in results:
        status = "OK" if r["success"] else "FAILED"
        eps_str = f"ε={r['epsilon']}" if r["epsilon"] else "no DP"
        print(f"  {r['method']:<20} ({eps_str:<10}) - {status}")

    # Save summary
    summary_path = Path(__file__).parent.parent / "outputs" / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    summary_path.parent.mkdir(exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
