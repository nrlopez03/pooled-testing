#!/usr/bin/env python3
"""Recompute Appendix C's archived means and standard errors from raw rows.

The script uses only the Python standard library.  It reproduces the numerical
table, but it does not retrain or reevaluate the historical PPO policies.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_reported_results"
REFERENCE = HERE.parent / "current_exact" / "results" / "archived_large_summary.csv"

FILES = {
    5: RAW / "sample_N50_d2_B5_G5_Utils3.csv",
    3: RAW / "sample_N50_d2_B5_G3_Utils3.csv",
}

METHODS = (
    ("static_milp", "Static MILP benchmark", "solveMILP"),
    (
        "gibbs_marginal_greedy",
        "Gibbs-Marginal Greedy",
        "solveConicGibbsGreedyDynamic",
    ),
    ("ppo_4", "PPO 4", "PPO_4Bucket_20000000"),
    ("ppo_4_plus", "PPO 4+", "PPO_4Bucket_50000000"),
    ("ppo_5", "PPO 5", "PPO_5Bucket_20000000"),
    ("ppo_5_plus", "PPO 5+", "PPO_5Bucket_50000000"),
)


def standard_error(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values))


def summarize(path: Path) -> tuple[int, dict[str, tuple[float, float]]]:
    values = {method: [] for method, _label, _column in METHODS}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            for method, _label, column in METHODS:
                values[method].append(float(row[column]))
    counts = {len(method_values) for method_values in values.values()}
    if len(counts) != 1:
        raise AssertionError(f"columns in {path.name} have inconsistent lengths")
    count = counts.pop()
    return count, {
        method: (statistics.fmean(method_values), standard_error(method_values))
        for method, method_values in values.items()
    }


def load_reference() -> dict[tuple[int, str], tuple[int, float, float]]:
    with REFERENCE.open(newline="") as stream:
        return {
            (int(row["pool_cap"]), row["method"]): (
                int(row["instances"]),
                float(row["mean_realized_welfare"]),
                float(row["se_realized_welfare"]),
            )
            for row in csv.DictReader(stream)
        }


def main() -> None:
    reference = load_reference()
    print("Appendix C LaTeX rows recomputed from the two raw 10,000-row files:")

    for pool_cap in (5, 3):
        count, results = summarize(FILES[pool_cap])
        if count != 10_000:
            raise AssertionError(f"expected 10,000 rows for G={pool_cap}, found {count}")
        print(f"  % (n,B,G)=(50,5,{pool_cap}); n={count}")
        for method, label, _column in METHODS:
            mean, se = results[method]
            expected_count, expected_mean, expected_se = reference[(pool_cap, method)]
            if expected_count != count:
                raise AssertionError(f"reference count mismatch for G={pool_cap}, {method}")
            if not math.isclose(mean, expected_mean, rel_tol=0.0, abs_tol=5e-9):
                raise AssertionError(f"reference mean mismatch for G={pool_cap}, {method}")
            if not math.isclose(se, expected_se, rel_tol=0.0, abs_tol=5e-9):
                raise AssertionError(f"reference SE mismatch for G={pool_cap}, {method}")
            print(f"    & {label} & {mean:.2f} ({se:.4f}) " + r"\\")

    print("\nAll means and standard errors match archived_large_summary.csv.")
    print(
        "Caution: this verifies the reported aggregation only. The historical PPO "
        "checkpoint files are not present in the archived repository, so these rows "
        "cannot currently be regenerated end to end from trained policies."
    )


if __name__ == "__main__":
    main()
