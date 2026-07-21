from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .canonical import canonical_hash
from .schema import FAMILY_ORDER, family_schemas, generation_levels, search_axes


@dataclass(frozen=True)
class FamilyBudgetInput:
    family: str
    valid_domain_upper_bound: int
    searchable_axis_count: int
    largest_axis_cardinality: int
    largest_priority_pair_cardinality: int
    marginal_coverage_minimum: int
    pairwise_coverage_minimum: int
    measured_seconds_per_production_dispatch: float
    measured_output_bytes_per_dispatch: float
    existing_executable_multiplicity: int


def _round_up(value: int, unit: int = 64) -> int:
    return unit * math.ceil(value / unit)


def _measurement_rows(capacity_measurement: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if capacity_measurement.get("status") != "pass" or int(capacity_measurement.get("workers", 0)) != 4:
        raise ValueError("budget optimizer requires a passing four-worker capacity measurement")
    rows = {str(row["family"]): row for row in capacity_measurement.get("families", ())}
    if set(rows) != set(FAMILY_ORDER):
        raise ValueError("capacity measurement does not cover every family")
    for family, row in rows.items():
        if float(row.get("seconds_per_dispatch", 0)) <= 0 or float(row.get("output_bytes_per_dispatch", 0)) <= 0:
            raise ValueError(f"capacity measurement is nonpositive for {family}")
    return rows


def budget_inputs(existing_counts: dict[str, int], capacity_measurement: Mapping[str, Any]) -> list[FamilyBudgetInput]:
    measurements = _measurement_rows(capacity_measurement)
    rows: list[FamilyBudgetInput] = []
    for family in FAMILY_ORDER:
        if family == "KDA02B_SURVIVOR_ADJUDICATION_V1":
            continue
        axes = search_axes(family)
        domain = math.prod(len(generation_levels(family, axis)) for axis in axes)
        largest_axis = max(len(generation_levels(family, axis)) for axis in axes)
        pair_sizes = [
            len(generation_levels(family, family_schemas[family].axis_map[left]))
            * len(generation_levels(family, family_schemas[family].axis_map[right]))
            for left, right in family_schemas[family].priority_pairs
        ]
        rows.append(FamilyBudgetInput(
            family=family,
            valid_domain_upper_bound=domain,
            searchable_axis_count=len(axes),
            largest_axis_cardinality=largest_axis,
            largest_priority_pair_cardinality=max(pair_sizes),
            marginal_coverage_minimum=50,
            pairwise_coverage_minimum=20,
            measured_seconds_per_production_dispatch=float(measurements[family]["seconds_per_dispatch"]),
            measured_output_bytes_per_dispatch=float(measurements[family]["output_bytes_per_dispatch"]),
            existing_executable_multiplicity=existing_counts.get(family, 0),
        ))
    return rows


def optimize_budget(existing_counts: dict[str, int], capacity_measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Allocate from coverage plus a measured four-worker production-path capacity gate."""
    inputs = budget_inputs(existing_counts, capacity_measurement)
    legacy_total = sum(existing_counts.values())
    family_measurements = _measurement_rows(capacity_measurement)
    average_seconds = sum(float(row["seconds_per_dispatch"]) for row in family_measurements.values()) / len(family_measurements)
    average_output = sum(float(row["output_bytes_per_dispatch"]) for row in family_measurements.values()) / len(family_measurements)
    dispatch_multiplier = int(capacity_measurement.get("development_dispatches_per_address", 0))
    if dispatch_multiplier <= 0:
        raise ValueError("capacity measurement lacks the frozen fold-dispatch multiplier")
    compute_capacity = math.floor((7 * 86400 * 4 * 0.50) / (average_seconds * dispatch_multiplier))
    output_capacity = math.floor((24 * 1024**3 * 0.50) / (average_output * dispatch_multiplier))
    bounded_capacity = min(12000, compute_capacity, output_capacity)
    if bounded_capacity < max(8000, legacy_total):
        raise ValueError("measured capacity cannot support the minimum substantial campaign or inherited executable multiplicity")
    target_total = max(8000, 64 * (bounded_capacity // 64))
    if target_total > 12000 or target_total < legacy_total:
        raise ValueError("measured capacity cannot support the frozen planning range")
    broad_target = target_total - legacy_total
    floors: dict[str, int] = {}
    weights: dict[str, float] = {}
    for row in inputs:
        coverage_floor = max(
            row.largest_axis_cardinality * row.marginal_coverage_minimum,
            row.largest_priority_pair_cardinality * row.pairwise_coverage_minimum,
        )
        floors[row.family] = _round_up(coverage_floor)
        weights[row.family] = (
            coverage_floor
            * math.sqrt(row.searchable_axis_count)
            * math.sqrt(math.log2(max(2, row.valid_domain_upper_bound)))
            / math.sqrt(row.measured_seconds_per_production_dispatch)
        )
    floor_total = sum(floors.values())
    if floor_total > broad_target:
        raise ValueError("capacity target cannot meet frozen coverage floors")
    remaining_units = (broad_target - floor_total) // 64
    raw_units = {family: remaining_units * weight / sum(weights.values()) for family, weight in weights.items()}
    extra_units = {family: int(math.floor(value)) for family, value in raw_units.items()}
    unallocated = remaining_units - sum(extra_units.values())
    for family in sorted(raw_units, key=lambda key: (-(raw_units[key] - extra_units[key]), key))[:unallocated]:
        extra_units[family] += 1
    allocations = {family: floors[family] + 64 * extra_units[family] for family in floors}
    if sum(allocations.values()) != broad_target:
        raise AssertionError("budget optimizer allocation arithmetic diverged")
    return {
        "schema": "stage22_outcome_free_budget_optimizer_v2",
        "inputs": [asdict(row) for row in inputs],
        "rules": {
            "campaign_total_range": [8000, 12000],
            "target_total": "floor64(min(12000,50pct seven-day four-worker measured dispatch capacity,50pct 24GiB measured artifact capacity)); minimum 8000",
            "coverage_floor": "ceil64(max(largest_axis_cardinality*50,largest_priority_pair_cardinality*20))",
            "headroom_weight": "coverage_floor*sqrt(searchable_axis_count)*sqrt(log2(valid_domain_upper_bound))/sqrt(measured_seconds_per_production_dispatch)",
            "allocation": "coverage floors plus largest-remainder 64-address blocks; family lexical tie-break",
            "prohibited_inputs": ["return", "PnL", "candidate rank", "winner", "partial outcome"],
        },
        "legacy_executable_projections": legacy_total,
        "capacity_measurement_sha256": canonical_hash(capacity_measurement),
        "measured_compute_capacity_attempts": compute_capacity,
        "measured_output_capacity_attempts": output_capacity,
        "development_dispatches_per_address": dispatch_multiplier,
        "target_total_attempt_rows": target_total,
        "new_broad_target": broad_target,
        "coverage_floors": floors,
        "headroom_weights": weights,
        "new_broad_counts": allocations,
        "status": "pass",
    }


__all__ = ["FamilyBudgetInput", "budget_inputs", "optimize_budget"]
