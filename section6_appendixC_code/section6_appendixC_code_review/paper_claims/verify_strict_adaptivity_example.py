#!/usr/bin/env python3
"""Reproduce the three-agent strict-adaptivity example in the paper.

This is a focused, dependency-free certificate for the claims made in the
strict-adaptivity example.  It imports the same exact solver used for the
Section 6 experiment, exhaustively enumerates the static allocations, and
inspects the optimal dynamic policy returned by backward induction.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE_ROOT / "current_exact"))

from exact_small_instances import ExactSolver, Instance, TOL  # noqa: E402


AGENT_NAMES = ("A", "B", "C")


def format_pool(mask: int) -> str:
    members = [AGENT_NAMES[i] for i in range(len(AGENT_NAMES)) if mask & (1 << i)]
    return "{" + ",".join(members) + "}"


def format_allocation(allocation: tuple[int, ...]) -> str:
    return "(" + ", ".join(format_pool(pool) for pool in allocation) + ")"


def best_static_allocations(
    solver: ExactSolver, *, require_disjoint: bool
) -> tuple[float, list[tuple[int, ...]]]:
    best_value = -math.inf
    best: list[tuple[int, ...]] = []
    for number_of_pools in range(1, solver.instance.budget + 1):
        for allocation in itertools.combinations(solver.pools, number_of_pools):
            if require_disjoint and any(
                first & second
                for first, second in itertools.combinations(allocation, 2)
            ):
                continue
            value = solver.static_allocation_value(allocation)
            if value > best_value + TOL:
                best_value = value
                best = [allocation]
            elif abs(value - best_value) <= TOL:
                best.append(allocation)
    return best_value, best


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-11):
        raise AssertionError(f"{label}: expected {expected!r}, found {actual!r}")


def main() -> None:
    instance = Instance(
        utilities=(0.129, 0.17483, 0.569),
        health_probabilities=(0.5562, 1.0, 0.12),
        budget=2,
        pool_cap=3,
    )
    solver = ExactSolver(instance)

    static_value, static_allocations = best_static_allocations(
        solver, require_disjoint=False
    )
    nonoverlap_value, nonoverlap_allocations = best_static_allocations(
        solver, require_disjoint=True
    )
    dynamic_value, state_count, policy = solver.optimal_dynamic_solution()

    root_state = (solver.full_support, 0, instance.budget)
    root_pool = policy[root_state]
    negative_support, positive_support, negative_probability = solver.split(
        solver.full_support, root_pool
    )
    negative_child = policy[(negative_support, root_pool, 1)]
    positive_child = policy[(positive_support, 0, 1)]
    gap = dynamic_value / static_value

    # These masks encode {A,B}, {B,C}, {C}, {B}, {A}, respectively.
    assert static_allocations == [(0b011, 0b110)]
    assert nonoverlap_allocations == [(0b001, 0b010)]
    assert root_pool == 0b011
    assert negative_child == 0b100
    assert positive_child == 0b010

    assert_close(static_value, 0.24658099248, "optimal static overlapping value")
    assert_close(nonoverlap_value, 0.2465798, "optimal static non-overlapping value")
    assert_close(dynamic_value, 0.284557136, "optimal dynamic value")
    assert_close(gap, 1.1540108308351475, "adaptivity gap")

    print("Strict-adaptivity example: exhaustive verification passed")
    print(f"  static overlapping optimum = {static_value:.11f}")
    print(f"    allocation: {format_allocation(static_allocations[0])}")
    print(f"  static non-overlapping optimum = {nonoverlap_value:.10f}")
    print(f"    allocation: {format_allocation(nonoverlap_allocations[0])}")
    print(f"  dynamic optimum = {dynamic_value:.9f}")
    print(f"    first test: {format_pool(root_pool)}")
    print(
        f"    after a negative first test (probability {negative_probability:.4f}): "
        f"{format_pool(negative_child)}"
    )
    print(f"    after a positive first test: {format_pool(positive_child)}")
    print(f"  adaptivity gap = {gap:.12f}")
    print(f"  posterior states evaluated = {state_count}")


if __name__ == "__main__":
    main()
