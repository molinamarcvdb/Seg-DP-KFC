#!/usr/bin/env python3
"""
Analyze all experiment results and generate tables for MICCAI paper.

Scans outputs/ for results.json files, aggregates multi-seed runs,
and produces:
  1. Main results table (all datasets x methods, eps=8.0)
  2. Epsilon sweep table/data
  3. LaTeX-formatted tables ready for paper
  4. Summary statistics with mean +/- std

Usage:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --format latex
    python scripts/analyze_results.py --dataset kvasir
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from collections import defaultdict
import math


def collect_results(output_dir="outputs"):
    """Scan all results.json files and return structured data."""
    results = []
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"Output directory {output_dir} not found!")
        return results

    for results_file in sorted(output_path.rglob("results.json")):
        try:
            with open(results_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Infer metadata from directory name if not in results
        dir_name = results_file.parent.name

        # Determine dataset
        dataset = data.get("dataset")
        if dataset is None:
            if "pathmnist" in dir_name:
                dataset = "pathmnist"
            elif "kvasir" in dir_name:
                dataset = "kvasir"
            elif "pet" in dir_name:
                dataset = "pet"
            elif "fundus" in dir_name:
                dataset = "fundus"
            elif "brats" in dir_name:
                dataset = "brats"
            elif "hepatic" in dir_name:
                dataset = "hepatic"
            else:
                dataset = "unknown"

        # Determine method
        method = data.get("method")
        if method is None:
            for m in ["dp_shampoo_synth", "dp_shampoo", "dp_synthetic", "dp_momentum",
                       "dp_public", "dp_no_precond", "dp_adadps_public",
                       "dp_adadps_oracle", "baseline", "dp"]:
                if m in dir_name:
                    method = m
                    break
            if method is None:
                method = "unknown"

        # Normalize method names
        method_map = {
            "dp_no_precond": "dp",
            "dp_adadps_public": "dp_public",
            "dp_adadps_oracle": "dp_oracle",
        }
        method = method_map.get(method, method)

        # Get epsilon
        epsilon = data.get("epsilon")
        if epsilon is None and method != "baseline":
            # Try to extract from dir name
            import re
            eps_match = re.search(r"eps([\d.]+)", dir_name)
            if eps_match:
                epsilon = float(eps_match.group(1))

        # Get seed
        seed = data.get("seed", 42)

        # Get metric value
        if dataset == "pathmnist":
            metric_name = "val_acc"
            best_val = data.get("best_val_acc", data.get("best_val_acc"))
            test_val = data.get("test_acc")
        else:
            metric_name = "val_dice"
            best_val = data.get("best_val_dice", data.get("best_dice"))
            test_val = None

        if best_val is None:
            continue

        # Get epochs and lr
        epochs = data.get("epochs")
        lr = data.get("lr")

        results.append({
            "dir": str(results_file.parent),
            "dir_name": dir_name,
            "dataset": dataset,
            "method": method,
            "epsilon": epsilon,
            "seed": seed,
            "metric_name": metric_name,
            "best_val": best_val,
            "test_val": test_val,
            "epochs": epochs,
            "lr": lr,
        })

    return results


def aggregate_results(results, min_epochs=None):
    """Group by (dataset, method, epsilon) and compute mean +/- std."""
    groups = defaultdict(list)

    for r in results:
        if min_epochs and r["epochs"] and r["epochs"] < min_epochs:
            continue
        key = (r["dataset"], r["method"], r["epsilon"])
        groups[key].append(r["best_val"])

    aggregated = {}
    for key, values in groups.items():
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((v - mean) ** 2 for v in values) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0.0
        aggregated[key] = {
            "mean": mean,
            "std": std,
            "n": n,
            "values": values,
        }

    return aggregated


def print_main_table(aggregated, fmt="markdown"):
    """Print main results table (best eps per method)."""

    datasets = sorted(set(k[0] for k in aggregated))
    methods_order = ["baseline", "dp", "dp_synthetic", "dp_public", "dp_shampoo", "dp_shampoo_synth"]
    methods = [m for m in methods_order if any(k[1] == m for k in aggregated)]

    # For each dataset, find the most common epsilon (usually 8.0)
    def get_best_key(ds, method):
        """Get key with highest epsilon (or None for baseline)."""
        candidates = [(k, v) for k, v in aggregated.items()
                      if k[0] == ds and k[1] == method]
        if not candidates:
            return None, None
        # Prefer eps=8.0, then highest
        for eps in [8.0, 4.0, 2.0, 1.0, None]:
            for k, v in candidates:
                if k[2] == eps:
                    return k, v
        return candidates[-1]

    # Header
    metric_labels = {
        "pathmnist": "Val Acc (%)",
        "kvasir": "Val Dice",
        "pet": "Val Dice",
        "fundus": "Val Dice",
        "brats": "Val Dice",
        "hepatic": "Val Dice",
    }

    print("\n" + "="*80)
    print("MAIN RESULTS TABLE (eps=8.0 for DP methods)")
    print("="*80)

    if fmt == "latex":
        # LaTeX table
        ncols = len(methods) + 1
        print(f"\\begin{{tabular}}{{l{'c' * len(methods)}}}")
        print("\\toprule")
        header = "Dataset & " + " & ".join(
            m.replace("_", "\\_") for m in methods
        ) + " \\\\"
        print(header)
        print("\\midrule")

        for ds in datasets:
            row = [ds.replace("_", "\\_")]
            best_val = -1
            for method in methods:
                key, agg = get_best_key(ds, method)
                if agg is None:
                    row.append("---")
                else:
                    val = agg["mean"]
                    if val > best_val and method != "baseline":
                        best_val = val
                    if agg["n"] > 1:
                        row.append(f"${val:.1f} \\pm {agg['std']:.1f}$")
                    else:
                        row.append(f"${val:.1f}$")

            # Bold the best DP method
            print(" & ".join(row) + " \\\\")

        print("\\bottomrule")
        print("\\end{tabular}")

    else:
        # Markdown table
        header = f"| {'Dataset':12s} |"
        separator = f"|{'-'*14}|"
        for m in methods:
            header += f" {m:18s} |"
            separator += f"{'-'*20}|"
        print(header)
        print(separator)

        for ds in datasets:
            row = f"| {ds:12s} |"
            for method in methods:
                key, agg = get_best_key(ds, method)
                if agg is None:
                    row += f" {'---':18s} |"
                else:
                    val = agg["mean"]
                    if agg["n"] > 1:
                        cell = f"{val:.2f} +/- {agg['std']:.2f} (n={agg['n']})"
                    else:
                        cell = f"{val:.2f} (n=1)"
                    row += f" {cell:18s} |"
            print(row)


def print_epsilon_sweep(aggregated, dataset="kvasir", fmt="markdown"):
    """Print epsilon sweep table for one dataset."""
    sweep_methods = ["dp", "dp_synthetic", "dp_shampoo", "dp_shampoo_synth"]
    epsilons = sorted(set(k[2] for k in aggregated if k[0] == dataset and k[2] is not None))

    if not epsilons:
        print(f"\nNo epsilon sweep data for {dataset}")
        return

    print(f"\n{'='*80}")
    print(f"EPSILON SWEEP: {dataset}")
    print(f"{'='*80}")

    # Include baseline for reference
    baseline_key = (dataset, "baseline", None)
    baseline = aggregated.get(baseline_key)

    if fmt == "latex":
        ncols = len(sweep_methods) + 1
        print(f"\\begin{{tabular}}{{l{'c' * len(sweep_methods)}}}")
        print("\\toprule")
        header = "$\\epsilon$ & " + " & ".join(
            m.replace("_", "\\_") for m in sweep_methods
        ) + " \\\\"
        print(header)
        print("\\midrule")

        if baseline:
            row = ["Baseline (no DP)"]
            for _ in sweep_methods:
                if baseline["n"] > 1:
                    row.append(f"\\multicolumn{{1}}{{c}}{{${baseline['mean']:.2f} \\pm {baseline['std']:.2f}$}}")
                else:
                    row.append(f"${baseline['mean']:.2f}$")
            print(" & ".join(row) + " \\\\")
            print("\\midrule")

        for eps in epsilons:
            row = [f"{eps}"]
            for method in sweep_methods:
                key = (dataset, method, eps)
                agg = aggregated.get(key)
                if agg is None:
                    row.append("---")
                elif agg["n"] > 1:
                    row.append(f"${agg['mean']:.2f} \\pm {agg['std']:.2f}$")
                else:
                    row.append(f"${agg['mean']:.2f}$")
            print(" & ".join(row) + " \\\\")

        print("\\bottomrule")
        print("\\end{tabular}")

    else:
        # Markdown
        header = f"| {'Epsilon':10s} |"
        separator = f"|{'-'*12}|"
        for m in sweep_methods:
            header += f" {m:18s} |"
            separator += f"{'-'*20}|"
        print(header)
        print(separator)

        if baseline:
            row = f"| {'Baseline':10s} |"
            for _ in sweep_methods:
                if baseline["n"] > 1:
                    cell = f"{baseline['mean']:.2f} +/- {baseline['std']:.2f}"
                else:
                    cell = f"{baseline['mean']:.2f}"
                row += f" {cell:18s} |"
            print(row)
            print(separator)

        for eps in epsilons:
            row = f"| {eps:<10.1f} |"
            for method in sweep_methods:
                key = (dataset, method, eps)
                agg = aggregated.get(key)
                if agg is None:
                    row += f" {'---':18s} |"
                elif agg["n"] > 1:
                    cell = f"{agg['mean']:.2f} +/- {agg['std']:.2f}"
                    row += f" {cell:18s} |"
                else:
                    row += f" {agg['mean']:.2f} (n=1)        |"
            print(row)


def print_all_runs(results, dataset=None):
    """Print every individual run for debugging."""
    if dataset:
        results = [r for r in results if r["dataset"] == dataset]

    results = sorted(results, key=lambda r: (r["dataset"], r["method"], r.get("epsilon") or 0, r["seed"]))

    print(f"\n{'='*80}")
    print("ALL INDIVIDUAL RUNS")
    print(f"{'='*80}")
    print(f"| {'Dataset':10s} | {'Method':15s} | {'Eps':5s} | {'Seed':5s} | {'Epochs':6s} | {'LR':6s} | {'Best Val':10s} |")
    print(f"|{'-'*12}|{'-'*17}|{'-'*7}|{'-'*7}|{'-'*8}|{'-'*8}|{'-'*12}|")

    for r in results:
        eps = f"{r['epsilon']:.1f}" if r["epsilon"] else "---"
        epochs = str(r["epochs"]) if r["epochs"] else "?"
        lr = f"{r['lr']:.4f}" if r["lr"] else "?"
        val = f"{r['best_val']:.4f}"
        print(f"| {r['dataset']:10s} | {r['method']:15s} | {eps:5s} | {r['seed']:<5d} | {epochs:6s} | {lr:6s} | {val:10s} |")


def print_gap_analysis(aggregated):
    """Show gap between methods vs baseline for each dataset."""
    print(f"\n{'='*80}")
    print("GAP ANALYSIS (method vs baseline)")
    print(f"{'='*80}")

    datasets = sorted(set(k[0] for k in aggregated))
    methods = ["dp", "dp_synthetic", "dp_public", "dp_shampoo", "dp_shampoo_synth"]

    for ds in datasets:
        baseline_key = (ds, "baseline", None)
        baseline = aggregated.get(baseline_key)
        if baseline is None:
            # Try with eps
            for k, v in aggregated.items():
                if k[0] == ds and k[1] == "baseline":
                    baseline = v
                    break

        if baseline is None:
            print(f"\n{ds}: No baseline found")
            continue

        bl_val = baseline["mean"]
        is_pct = ds == "pathmnist"
        unit = "pp" if is_pct else "dice"

        print(f"\n{ds} (baseline: {bl_val:.2f}):")
        for method in methods:
            # Find best epsilon for this method
            best_agg = None
            best_eps = None
            for k, v in aggregated.items():
                if k[0] == ds and k[1] == method:
                    if best_agg is None or v["mean"] > best_agg["mean"]:
                        best_agg = v
                        best_eps = k[2]

            if best_agg is None:
                print(f"  {method:15s}: ---")
            else:
                gap = best_agg["mean"] - bl_val
                pct_gap = (gap / bl_val) * 100
                print(f"  {method:15s}: {best_agg['mean']:.2f} (gap: {gap:+.2f} {unit}, {pct_gap:+.1f}%) [eps={best_eps}]")

        # Show precond benefits vs dp
        dp_key = None
        synth_key = None
        shampoo_key = None
        shampoo_synth_key = None
        for k in aggregated:
            if k[0] == ds and k[1] == "dp":
                dp_key = k
            if k[0] == ds and k[1] == "dp_synthetic":
                synth_key = k
            if k[0] == ds and k[1] == "dp_shampoo":
                shampoo_key = k
            if k[0] == ds and k[1] == "dp_shampoo_synth":
                shampoo_synth_key = k

        if dp_key:
            dp_val = aggregated[dp_key]["mean"]
            if synth_key:
                benefit = aggregated[synth_key]["mean"] - dp_val
                print(f"  >> AdaDPS(synth) benefit vs dp: {benefit:+.2f} {unit}")
            if shampoo_key:
                benefit = aggregated[shampoo_key]["mean"] - dp_val
                print(f"  >> Shampoo(public) benefit vs dp: {benefit:+.2f} {unit}")
            if shampoo_synth_key:
                benefit = aggregated[shampoo_synth_key]["mean"] - dp_val
                print(f"  >> Shampoo(synth) benefit vs dp: {benefit:+.2f} {unit}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MICCAI experiment results")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--format", type=str, default="markdown",
                        choices=["markdown", "latex"])
    parser.add_argument("--dataset", type=str, default=None,
                        help="Filter to specific dataset")
    parser.add_argument("--min_epochs", type=int, default=None,
                        help="Exclude runs with fewer epochs")
    parser.add_argument("--all_runs", action="store_true",
                        help="Show every individual run")
    args = parser.parse_args()

    # Collect
    print("Scanning for results...")
    results = collect_results(args.output_dir)
    print(f"Found {len(results)} experiment runs")

    if args.dataset:
        results = [r for r in results if r["dataset"] == args.dataset]
        print(f"  Filtered to {len(results)} runs for {args.dataset}")

    if not results:
        print("No results found!")
        return

    # Show individual runs
    if args.all_runs:
        print_all_runs(results, args.dataset)

    # Aggregate
    aggregated = aggregate_results(results, min_epochs=args.min_epochs)

    # Print tables
    print_main_table(aggregated, fmt=args.format)

    # Epsilon sweep for each dataset
    for ds in sorted(set(r["dataset"] for r in results)):
        eps_keys = [k for k in aggregated if k[0] == ds and k[2] is not None]
        eps_values = set(k[2] for k in eps_keys)
        if len(eps_values) > 1:
            print_epsilon_sweep(aggregated, dataset=ds, fmt=args.format)

    # Gap analysis
    print_gap_analysis(aggregated)

    # Summary
    print(f"\n{'='*80}")
    print("EXPERIMENT COVERAGE SUMMARY")
    print(f"{'='*80}")
    datasets = sorted(set(r["dataset"] for r in results))
    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        methods = sorted(set(r["method"] for r in ds_results))
        seeds = sorted(set(r["seed"] for r in ds_results))
        epsilons = sorted(set(r["epsilon"] for r in ds_results if r["epsilon"] is not None))
        max_epochs = max((r["epochs"] or 0) for r in ds_results)

        print(f"\n{ds}:")
        print(f"  Methods:  {', '.join(methods)}")
        print(f"  Seeds:    {seeds}")
        print(f"  Epsilons: {epsilons or ['N/A']}")
        print(f"  Runs:     {len(ds_results)}")
        print(f"  Max epochs: {max_epochs}")

        # Show what's missing
        all_methods = {"baseline", "dp", "dp_synthetic", "dp_public", "dp_shampoo", "dp_shampoo_synth"}
        missing = all_methods - set(methods)
        if missing:
            print(f"  MISSING:  {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
