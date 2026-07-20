from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

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
    deterministic_engine_work_units: int
    existing_executable_multiplicity: int


# Operation counts for the longest registered raw-history/state-machine path.
# They are not economic measurements and do not depend on returns or ranks.
ENGINE_WORK_UNITS = {
    "A4_TSMOM_V7": 240,
    "A1_COMPRESSION_V2": 7200,
    "A2_PRIOR_HIGH_RS_CONTEXT_V1": 370,
    "A3_STARTER_RETEST_V3": 924,
}


def _round_up(value: int, unit: int = 64) -> int:
    return unit * math.ceil(value / unit)


def budget_inputs(existing_counts: dict[str, int]) -> list[FamilyBudgetInput]:
    rows: list[FamilyBudgetInput] = []
    for family in FAMILY_ORDER:
        if family not in ENGINE_WORK_UNITS:
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
            deterministic_engine_work_units=ENGINE_WORK_UNITS[family],
            existing_executable_multiplicity=existing_counts.get(family, 0),
        ))
    return rows


def optimize_budget(existing_counts: dict[str, int]) -> dict[str, Any]:
    """Allocate an outcome-free 8k-12k campaign from coverage and compute inputs."""
    inputs = budget_inputs(existing_counts)
    legacy_total = sum(existing_counts.values())
    # Existing multiplicity provides the only scale input.  The 2.4x expansion
    # is rounded to a 64-address generation block and clipped to Stage-22's
    # substantial-campaign planning interval.
    target_total = min(12000, max(8000, _round_up(math.ceil(legacy_total * 2.4))))
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
            / math.sqrt(row.deterministic_engine_work_units)
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
        "schema": "stage22_outcome_free_budget_optimizer_v1",
        "inputs": [asdict(row) for row in inputs],
        "rules": {
            "campaign_total_range": [8000, 12000],
            "target_total": "ceil64(clip(existing_executable_multiplicity*2.4,8000,12000))",
            "coverage_floor": "ceil64(max(largest_axis_cardinality*50,largest_priority_pair_cardinality*20))",
            "headroom_weight": "coverage_floor*sqrt(searchable_axis_count)*sqrt(log2(valid_domain_upper_bound))/sqrt(deterministic_engine_work_units)",
            "allocation": "coverage floors plus largest-remainder 64-address blocks; family lexical tie-break",
            "prohibited_inputs": ["return", "PnL", "candidate rank", "winner", "partial outcome"],
        },
        "legacy_executable_projections": legacy_total,
        "target_total_attempt_rows": target_total,
        "new_broad_target": broad_target,
        "coverage_floors": floors,
        "headroom_weights": weights,
        "new_broad_counts": allocations,
        "status": "pass",
    }


__all__ = ["ENGINE_WORK_UNITS", "FamilyBudgetInput", "budget_inputs", "optimize_budget"]
