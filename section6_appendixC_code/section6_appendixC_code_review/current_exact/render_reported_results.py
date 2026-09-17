#!/usr/bin/env python3
"""Render and validate the exact-experiment numbers reported in the paper."""

from __future__ import annotations

import csv
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUMMARY = HERE / "results" / "exact_small_instances_summary.csv"

SETTINGS = (
    ("random_n4_b2_g4", "(4,2,4)"),
    ("random_n5_b3_g5", "(5,3,5)"),
    ("random_n6_b3_g3", "(6,3,3)"),
)

EXPECTED_MAIN_ROWS = (
    "(4,2,4) & .9948 (.0007) & .9932 (.0009) & .9529 (.0043) & .9881 (.0015) & .9880 (.0015) & 1.0744 \\\\",
    "(5,3,5) & .9908 (.0009) & .9895 (.0010) & .9242 (.0069) & .9879 (.0014) & .9876 (.0014) & 1.0501 \\\\",
    "(6,3,3) & .9894 (.0013) & .9838 (.0019) & .9126 (.0090) & .9863 (.0019) & .9858 (.0020) & 1.0507 \\\\",
)

EXPECTED_HEAD_TO_HEAD_ROWS = (
    "(4,2,4) & 300 & 38.3\\% & 61.7\\% & 0.0\\% \\\\",
    "(5,3,5) & 200 & 52.5\\% & 47.5\\% & 0.0\\% \\\\",
    "(6,3,3) & 100 & 71.0\\% & 29.0\\% & 0.0\\% \\\\",
)


def no_leading_zero(value: float, digits: int) -> str:
    rendered = f"{value:.{digits}f}"
    if rendered.startswith("0."):
        return rendered[1:]
    if rendered.startswith("-0."):
        return "-" + rendered[2:]
    return rendered


def load_rows() -> dict[str, dict[str, str]]:
    with SUMMARY.open(newline="") as stream:
        return {row["setting"]: row for row in csv.DictReader(stream)}


def metric(row: dict[str, str], name: str) -> float:
    return float(row[name])


def mean_and_se(row: dict[str, str], stem: str) -> str:
    return (
        f"{no_leading_zero(metric(row, f'mean_{stem}'), 4)} "
        f"({no_leading_zero(metric(row, f'se_{stem}'), 4)})"
    )


def main() -> None:
    rows = load_rows()
    main_rows: list[str] = []
    head_to_head_rows: list[str] = []

    for key, label in SETTINGS:
        row = rows[key]
        main_rows.append(
            " & ".join(
                (
                    label,
                    mean_and_se(row, "static_fraction_dynamic"),
                    mean_and_se(row, "static_nonoverlap_fraction_dynamic"),
                    mean_and_se(row, "static_greedy_fraction_dynamic"),
                    mean_and_se(row, "exact_joint_fraction_dynamic"),
                    mean_and_se(row, "marginal_product_fraction_dynamic"),
                    f"{metric(row, 'max_adaptivity_gap'):.4f}",
                )
            )
            + r" \\"
        )

        exact_rate = metric(row, "exact_better_than_static_greedy_rate")
        static_rate = metric(row, "static_greedy_better_than_exact_rate")
        tie_rate = 1.0 - exact_rate - static_rate
        head_to_head_rows.append(
            f"{label} & {int(float(row['instances']))} & {100 * exact_rate:.1f}\\% & "
            f"{100 * tie_rate:.1f}\\% & {100 * static_rate:.1f}\\% " + r"\\"
        )

    if tuple(main_rows) != EXPECTED_MAIN_ROWS:
        raise AssertionError("generated Section 6 table rows differ from the manuscript")
    if tuple(head_to_head_rows) != EXPECTED_HEAD_TO_HEAD_ROWS:
        raise AssertionError("generated greedy head-to-head rows differ from the manuscript")

    print("Section 6 main-table rows:")
    for row in main_rows:
        print("   ", row)

    print("\nGreedy head-to-head rows:")
    for row in head_to_head_rows:
        print("   ", row)

    print("\nAdditional narrative checks:")
    for key, label in SETTINGS:
        row = rows[key]
        exact_count = round(metric(row, "exact_better_rate") * float(row["instances"]))
        marginal_count = round(
            metric(row, "marginal_better_rate") * float(row["instances"])
        )
        print(
            f"  {label}: min Greedy/OPTdyn="
            f"{metric(row, 'min_exact_joint_fraction_dynamic'):.4f}; "
            f"Greedy>OPT*={100 * metric(row, 'exact_better_than_static_rate'):.1f}%; "
            f"OPT*>Greedy={100 * metric(row, 'static_better_than_exact_rate'):.1f}%; "
            f"Greedy>MPGreedy on {exact_count}, reverse on {marginal_count}"
        )

    guarantee = 1.0 / (2.0 * (math.e + 1.0))
    print(f"  theoretical fraction 1/[2(e+1)] = {guarantee:.7f}")
    print("\nAll rounded values agree with the audited manuscript tables.")


if __name__ == "__main__":
    main()
