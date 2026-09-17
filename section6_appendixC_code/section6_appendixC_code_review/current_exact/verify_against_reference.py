#!/usr/bin/env python3
"""Compare two experiment CSVs while ignoring machine-dependent runtimes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    reference = read_rows(args.reference)
    candidate = read_rows(args.candidate)
    if len(reference) != len(candidate):
        raise SystemExit(
            f"row-count mismatch: reference={len(reference)}, candidate={len(candidate)}"
        )

    mismatches: list[str] = []
    for row_number, (expected, observed) in enumerate(
        zip(reference, candidate), start=2
    ):
        if expected.keys() != observed.keys():
            raise SystemExit(f"column mismatch at CSV row {row_number}")
        for field in expected:
            if "runtime_seconds" in field:
                continue
            if expected[field] != observed[field]:
                mismatches.append(
                    f"row {row_number}, {field}: "
                    f"expected {expected[field]!r}, observed {observed[field]!r}"
                )

    if mismatches:
        preview = "\n".join(mismatches[:20])
        suffix = "" if len(mismatches) <= 20 else f"\n... {len(mismatches) - 20} more"
        raise SystemExit(f"deterministic-field mismatches:\n{preview}{suffix}")

    print("All non-runtime fields match exactly.")


if __name__ == "__main__":
    main()
