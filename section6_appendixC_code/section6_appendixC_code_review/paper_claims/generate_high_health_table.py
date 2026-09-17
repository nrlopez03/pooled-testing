#!/usr/bin/env python3
"""Generate and audit the paper's boundary-sensitive high-health table."""

from __future__ import annotations

import argparse


POOL_CAPS = (2, 3, 4, 5, 8, 10)
THETAS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)

# Entries currently printed in the audited v12 manuscript.
MANUSCRIPT_ROWS = {
    2: ("2.00", "1.67", "1.43", "1.25", "1.18", "1.11", "1.05"),
    3: ("2.00", "2.00", "2.00", "1.56", "1.38", "1.24", "1.11"),
    4: ("2.00", "2.00", "2.00", "1.95", "1.63", "1.37", "1.17"),
    5: ("2.00", "2.00", "2.00", "2.00", "1.92", "1.52", "1.23"),
    8: ("2.00", "2.00", "2.00", "2.00", "2.00", "2.00", "1.43"),
    10: ("2.00", "2.00", "2.00", "2.00", "2.00", "2.00", "1.59"),
}


def phi(pool_cap: int, theta: float) -> tuple[float, int]:
    candidates = [
        (pool_cap / (pool_size * theta ** (pool_size - 1)), pool_size)
        for pool_size in range(1, pool_cap + 1)
    ]
    return min(candidates)


def table_value(pool_cap: int, theta: float) -> tuple[float, int]:
    value, minimizing_pool_size = phi(pool_cap, theta)
    return min(2.0, value), minimizing_pool_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-manuscript",
        action="store_true",
        help="exit unsuccessfully if a printed manuscript entry differs from the formula",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated: dict[int, tuple[str, ...]] = {}

    print("LaTeX rows generated from min{2, phi_G(theta)}:")
    for pool_cap in POOL_CAPS:
        row = tuple(
            f"{table_value(pool_cap, theta)[0]:.2f}" for theta in THETAS
        )
        generated[pool_cap] = row
        print(f"    {pool_cap:<2} & " + " & ".join(row) + r" \\")

    mismatches: list[str] = []
    for pool_cap in POOL_CAPS:
        for theta, generated_entry, manuscript_entry in zip(
            THETAS, generated[pool_cap], MANUSCRIPT_ROWS[pool_cap]
        ):
            if generated_entry != manuscript_entry:
                exact, minimizing_pool_size = table_value(pool_cap, theta)
                mismatches.append(
                    f"G={pool_cap}, theta={theta:.2f}: formula={exact:.12f} "
                    f"(s={minimizing_pool_size}), rounds to {generated_entry}; "
                    f"manuscript prints {manuscript_entry}"
                )

    if mismatches:
        print("\nManuscript discrepancies:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        if args.strict_manuscript:
            raise SystemExit(1)
    else:
        print("\nEvery manuscript entry matches the defining formula.")


if __name__ == "__main__":
    main()
