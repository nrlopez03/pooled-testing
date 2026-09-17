#!/usr/bin/env python3
"""Exact small-instance experiments for dynamic welfare-maximizing pooled testing.

The solver represents a posterior by the set of health profiles consistent with
the observed pooled-test outcomes. Prior product probabilities are retained as
unnormalized weights on that set. It computes, without sampling:

* the optimal dynamic policy by backward induction;
* Exact-Joint Greedy and its true expected policy value;
* Marginal-Product Greedy and its true expected policy value; and
* optimal static overlapping and non-overlapping allocations; and
* the static non-overlapping greedy allocation.

The implementation is intentionally dependency-free and designed for n <= 7.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import math
import random
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence


TOL = 1e-10


@dataclass(frozen=True)
class Instance:
    utilities: tuple[float, ...]
    health_probabilities: tuple[float, ...]
    budget: int
    pool_cap: int

    @property
    def n(self) -> int:
        return len(self.utilities)


def masks_of_size_at_most(n: int, cap: int) -> tuple[int, ...]:
    return tuple(mask for mask in range(1, 1 << n) if mask.bit_count() <= cap)


class ExactSolver:
    """Exact posterior-state solver for one instance."""

    def __init__(self, instance: Instance):
        self.instance = instance
        self.n = instance.n
        self.full_agent_mask = (1 << self.n) - 1
        self.worlds = tuple(range(1 << self.n))
        self.pools = masks_of_size_at_most(self.n, instance.pool_cap)

        self.world_mass = tuple(self._world_probability(world) for world in self.worlds)
        self.full_support = (1 << len(self.worlds)) - 1
        self.utility_sum = tuple(
            sum(instance.utilities[i] for i in range(self.n) if mask & (1 << i))
            for mask in range(1 << self.n)
        )
        self.q_product = tuple(
            math.prod(
                instance.health_probabilities[i]
                for i in range(self.n)
                if mask & (1 << i)
            )
            for mask in range(1 << self.n)
        )

        negative_support: dict[int, int] = {}
        for pool in self.pools:
            support = 0
            for world in self.worlds:
                if world & pool == pool:
                    support |= 1 << world
            negative_support[pool] = support
        self.negative_support = negative_support

    def _world_probability(self, world: int) -> float:
        probability = 1.0
        for i, q_i in enumerate(self.instance.health_probabilities):
            probability *= q_i if world & (1 << i) else 1.0 - q_i
        return probability

    @lru_cache(maxsize=None)
    def support_mass(self, support: int) -> float:
        total = 0.0
        remaining = support
        while remaining:
            least_bit = remaining & -remaining
            world = least_bit.bit_length() - 1
            total += self.world_mass[world]
            remaining ^= least_bit
        return total

    def feasible_pools(self, cleared: int) -> Iterable[int]:
        # Cleared agents are known healthy and can be deleted from later pools.
        return (pool for pool in self.pools if not (pool & cleared))

    def split(self, support: int, pool: int) -> tuple[int, int, float]:
        negative = support & self.negative_support[pool]
        positive = support & (self.full_support ^ self.negative_support[pool])
        total_mass = self.support_mass(support)
        p_negative = self.support_mass(negative) / total_mass
        return negative, positive, p_negative

    def conditional_marginal(self, support: int, agent: int) -> float:
        healthy_support = support & self.negative_support[1 << agent]
        return self.support_mass(healthy_support) / self.support_mass(support)

    def exact_joint_score(self, support: int, cleared: int, pool: int) -> float:
        negative = support & self.negative_support[pool]
        p_negative = self.support_mass(negative) / self.support_mass(support)
        return p_negative * self.utility_sum[pool & ~cleared]

    def marginal_product_score(self, support: int, cleared: int, pool: int) -> float:
        healthy_probability = math.prod(
            self.conditional_marginal(support, i)
            for i in range(self.n)
            if pool & (1 << i)
        )
        return healthy_probability * self.utility_sum[pool & ~cleared]

    def _selected_pool(
        self,
        support: int,
        cleared: int,
        score: Callable[[int, int, int], float],
    ) -> int | None:
        best_pool = None
        best_score = -1.0
        for pool in self.feasible_pools(cleared):
            candidate = score(support, cleared, pool)
            if candidate > best_score + TOL or (
                abs(candidate - best_score) <= TOL
                and (best_pool is None or pool < best_pool)
            ):
                best_pool = pool
                best_score = candidate
        return best_pool

    def greedy_value(self, kind: str) -> tuple[float, int]:
        if kind == "exact_joint":
            score = self.exact_joint_score
        elif kind == "marginal_product":
            score = self.marginal_product_score
        else:
            raise ValueError(f"unknown greedy kind: {kind}")

        states = 0

        @lru_cache(maxsize=None)
        def value(support: int, cleared: int, remaining_budget: int) -> float:
            nonlocal states
            states += 1
            if remaining_budget == 0 or cleared == self.full_agent_mask:
                return 0.0
            pool = self._selected_pool(support, cleared, score)
            if pool is None:
                return 0.0
            negative, positive, p_negative = self.split(support, pool)
            negative_value = 0.0
            positive_value = 0.0
            if p_negative > 0.0:
                negative_value = self.utility_sum[pool & ~cleared] + value(
                    negative, cleared | pool, remaining_budget - 1
                )
            if p_negative < 1.0:
                positive_value = value(positive, cleared, remaining_budget - 1)
            return p_negative * negative_value + (1.0 - p_negative) * positive_value

        result = value(self.full_support, 0, self.instance.budget)
        return result, states

    def optimal_dynamic_solution(self) -> tuple[float, int, dict[tuple[int, int, int], int]]:
        """Return the optimal value, state count, and optimal action at every state."""
        states = 0
        policy: dict[tuple[int, int, int], int] = {}

        @lru_cache(maxsize=None)
        def value(support: int, cleared: int, remaining_budget: int) -> float:
            nonlocal states
            states += 1
            if remaining_budget == 0 or cleared == self.full_agent_mask:
                return 0.0
            best = 0.0
            best_pool = None
            for pool in self.feasible_pools(cleared):
                negative, positive, p_negative = self.split(support, pool)
                negative_value = 0.0
                positive_value = 0.0
                if p_negative > 0.0:
                    negative_value = self.utility_sum[pool & ~cleared] + value(
                        negative, cleared | pool, remaining_budget - 1
                    )
                if p_negative < 1.0:
                    positive_value = value(positive, cleared, remaining_budget - 1)
                candidate = (
                    p_negative * negative_value
                    + (1.0 - p_negative) * positive_value
                )
                if candidate > best + TOL or (
                    abs(candidate - best) <= TOL
                    and (best_pool is None or pool < best_pool)
                ):
                    best = candidate
                    best_pool = pool
            if best_pool is not None:
                policy[(support, cleared, remaining_budget)] = best_pool
            return best

        result = value(self.full_support, 0, self.instance.budget)
        return result, states, policy

    def static_allocation_value(self, allocation: Sequence[int]) -> float:
        total = 0.0
        for i, u_i in enumerate(self.instance.utilities):
            containing = tuple(pool for pool in allocation if pool & (1 << i))
            probability_cleared = 0.0
            for number in range(1, len(containing) + 1):
                sign = 1.0 if number % 2 else -1.0
                for chosen in itertools.combinations(containing, number):
                    union = 0
                    for pool in chosen:
                        union |= pool
                    probability_cleared += sign * self.q_product[union]
            total += u_i * probability_cleared
        return total

    def optimal_static_value(self) -> float:
        number_selected = min(self.instance.budget, len(self.pools))
        best = 0.0
        for allocation in itertools.combinations(self.pools, number_selected):
            candidate = self.static_allocation_value(allocation)
            if candidate > best:
                best = candidate
        return best

    def optimal_static_nonoverlap_value(self) -> float:
        """Optimal value of at most B pairwise-disjoint pools."""

        @lru_cache(maxsize=None)
        def value(available: int, remaining_budget: int) -> float:
            if remaining_budget == 0 or available == 0:
                return 0.0
            best = 0.0
            for pool in self.pools:
                if pool & ~available:
                    continue
                candidate = (
                    self.q_product[pool] * self.utility_sum[pool]
                    + value(available & ~pool, remaining_budget - 1)
                )
                if candidate > best:
                    best = candidate
            return best

        return value(self.full_agent_mask, self.instance.budget)

    def static_nonoverlap_greedy_value(self) -> float:
        """Repeatedly select the best prior-value pool among unused agents."""
        available = self.full_agent_mask
        total = 0.0
        for _step in range(self.instance.budget):
            best_pool = None
            best_value = -1.0
            for pool in self.pools:
                if pool & ~available:
                    continue
                candidate = self.q_product[pool] * self.utility_sum[pool]
                if candidate > best_value + TOL or (
                    abs(candidate - best_value) <= TOL
                    and (best_pool is None or pool < best_pool)
                ):
                    best_pool = pool
                    best_value = candidate
            if best_pool is None:
                break
            total += best_value
            available &= ~best_pool
        return total

    def root_choices(self) -> tuple[int | None, int | None]:
        return (
            self._selected_pool(self.full_support, 0, self.exact_joint_score),
            self._selected_pool(self.full_support, 0, self.marginal_product_score),
        )


def solve_instance(instance: Instance) -> dict[str, float | int]:
    started = time.perf_counter()
    solver = ExactSolver(instance)
    optimal_dynamic, dynamic_states, _optimal_policy = solver.optimal_dynamic_solution()
    exact_joint, exact_states = solver.greedy_value("exact_joint")
    marginal_product, marginal_states = solver.greedy_value("marginal_product")
    optimal_static = solver.optimal_static_value()
    optimal_static_nonoverlap = solver.optimal_static_nonoverlap_value()
    static_nonoverlap_greedy = solver.static_nonoverlap_greedy_value()
    exact_root, marginal_root = solver.root_choices()

    if optimal_dynamic + TOL < max(
        exact_joint,
        marginal_product,
        optimal_static,
        optimal_static_nonoverlap,
        static_nonoverlap_greedy,
    ):
        raise AssertionError("optimal dynamic value is below a feasible policy")
    if optimal_dynamic > 2.0 * optimal_static + 1e-8:
        raise AssertionError("computed instance violates the factor-two theorem")

    elapsed = time.perf_counter() - started
    return {
        "n": instance.n,
        "budget": instance.budget,
        "pool_cap": instance.pool_cap,
        "optimal_dynamic": optimal_dynamic,
        "optimal_static": optimal_static,
        "optimal_static_nonoverlap": optimal_static_nonoverlap,
        "static_nonoverlap_greedy": static_nonoverlap_greedy,
        "exact_joint_greedy": exact_joint,
        "marginal_product_greedy": marginal_product,
        "adaptivity_gap": optimal_dynamic / optimal_static if optimal_static else 1.0,
        "static_fraction_dynamic": (
            optimal_static / optimal_dynamic if optimal_dynamic else 1.0
        ),
        "static_nonoverlap_fraction_dynamic": (
            optimal_static_nonoverlap / optimal_dynamic if optimal_dynamic else 1.0
        ),
        "static_greedy_fraction_dynamic": (
            static_nonoverlap_greedy / optimal_dynamic if optimal_dynamic else 1.0
        ),
        "exact_joint_fraction_dynamic": exact_joint / optimal_dynamic if optimal_dynamic else 1.0,
        "marginal_product_fraction_dynamic": (
            marginal_product / optimal_dynamic if optimal_dynamic else 1.0
        ),
        "exact_minus_marginal": exact_joint - marginal_product,
        "root_choices_differ": int(exact_root != marginal_root),
        "dynamic_states": dynamic_states,
        "exact_greedy_states": exact_states,
        "marginal_greedy_states": marginal_states,
        "runtime_seconds": elapsed,
    }


def archived_instances(path: Path, budget: int, pool_cap: int) -> Iterable[Instance]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            agents = ast.literal_eval(row["agents"])
            yield Instance(
                utilities=tuple(float(agent[1]) for agent in agents),
                health_probabilities=tuple(float(agent[2]) for agent in agents),
                budget=budget,
                pool_cap=pool_cap,
            )


def random_instance(rng: random.Random, n: int, budget: int, pool_cap: int) -> Instance:
    # Broad interior priors and heterogeneous positive utilities stress both
    # posterior dependence and welfare asymmetry without numerical degeneracy.
    q = tuple(rng.uniform(0.05, 0.95) for _ in range(n))
    utilities = tuple(math.exp(rng.uniform(math.log(0.1), math.log(2.0))) for _ in range(n))
    return Instance(utilities, q, budget, pool_cap)


def write_rows(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standard_error(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def summarize(label: str, rows: Sequence[dict[str, float | int]]) -> dict[str, float | int | str]:
    metric_names = (
        "optimal_dynamic",
        "optimal_static",
        "optimal_static_nonoverlap",
        "static_nonoverlap_greedy",
        "exact_joint_greedy",
        "marginal_product_greedy",
        "adaptivity_gap",
        "static_fraction_dynamic",
        "static_nonoverlap_fraction_dynamic",
        "static_greedy_fraction_dynamic",
        "exact_joint_fraction_dynamic",
        "marginal_product_fraction_dynamic",
        "runtime_seconds",
        "dynamic_states",
    )
    result: dict[str, float | int | str] = {"setting": label, "instances": len(rows)}
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        result[f"mean_{metric}"] = statistics.fmean(values)
        result[f"se_{metric}"] = standard_error(values)
        result[f"min_{metric}"] = min(values)
        result[f"max_{metric}"] = max(values)
    result["root_disagreement_rate"] = statistics.fmean(
        float(row["root_choices_differ"]) for row in rows
    )
    result["exact_better_rate"] = statistics.fmean(
        float(row["exact_minus_marginal"] > TOL) for row in rows
    )
    result["marginal_better_rate"] = statistics.fmean(
        float(row["exact_minus_marginal"] < -TOL) for row in rows
    )
    result["exact_better_than_static_rate"] = statistics.fmean(
        float(float(row["exact_joint_greedy"]) > float(row["optimal_static"]) + TOL)
        for row in rows
    )
    result["static_better_than_exact_rate"] = statistics.fmean(
        float(float(row["optimal_static"]) > float(row["exact_joint_greedy"]) + TOL)
        for row in rows
    )
    result["exact_better_than_static_greedy_rate"] = statistics.fmean(
        float(
            float(row["exact_joint_greedy"])
            > float(row["static_nonoverlap_greedy"]) + TOL
        )
        for row in rows
    )
    result["static_greedy_better_than_exact_rate"] = statistics.fmean(
        float(
            float(row["static_nonoverlap_greedy"])
            > float(row["exact_joint_greedy"]) + TOL
        )
        for row in rows
    )
    return result


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    all_rows: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str]] = []

    archived_settings = ()
    if args.archived_n3 and args.archived_n5:
        archived_settings = (
            ("archived_n3_b2_g3", Path(args.archived_n3), 2, 3),
            ("archived_n5_b3_g5", Path(args.archived_n5), 3, 5),
        )
    for label, path, budget, pool_cap in archived_settings:
        instances = list(archived_instances(path, budget, pool_cap))
        if args.archived_limit:
            instances = instances[: args.archived_limit]
        rows = []
        for index, instance in enumerate(instances):
            row = solve_instance(instance)
            row.update(
                {
                    "setting": label,
                    "instance": index,
                    "seed": "archived",
                    "utilities": repr(instance.utilities),
                    "health_probabilities": repr(instance.health_probabilities),
                }
            )
            rows.append(row)
        all_rows.extend(rows)
        summaries.append(summarize(label, rows))

    rng = random.Random(args.seed)
    random_settings = (
        ("random_n4_b2_g4", 4, 2, 4, args.random_n4),
        ("random_n5_b3_g5", 5, 3, 5, args.random_n5),
        ("random_n6_b3_g3", 6, 3, 3, args.random_n6),
    )
    for label, n, budget, pool_cap, count in random_settings:
        rows = []
        for index in range(count):
            instance = random_instance(rng, n, budget, pool_cap)
            row = solve_instance(instance)
            row.update(
                {
                    "setting": label,
                    "instance": index,
                    "seed": args.seed,
                    "utilities": repr(instance.utilities),
                    "health_probabilities": repr(instance.health_probabilities),
                }
            )
            rows.append(row)
        all_rows.extend(rows)
        summaries.append(summarize(label, rows))

    write_rows(output / "exact_small_instances_raw.csv", all_rows)
    write_rows(output / "exact_small_instances_summary.csv", summaries)

    for row in summaries:
        print(
            row["setting"],
            "instances=", row["instances"],
            "max_gap=", f"{float(row['max_adaptivity_gap']):.6f}",
            "min_exact/opt=", f"{float(row['min_exact_joint_fraction_dynamic']):.6f}",
            "mean_runtime=", f"{float(row['mean_runtime_seconds']):.4f}s",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archived-n3")
    parser.add_argument("--archived-n5")
    parser.add_argument("--output", default="results")
    parser.add_argument("--archived-limit", type=int, default=0)
    parser.add_argument("--random-n4", type=int, default=300)
    parser.add_argument("--random-n5", type=int, default=200)
    parser.add_argument("--random-n6", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if bool(args.archived_n3) != bool(args.archived_n5):
        parser.error("--archived-n3 and --archived-n5 must be supplied together")
    return args


if __name__ == "__main__":
    run(parse_args())
