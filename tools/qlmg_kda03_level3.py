"""Pure frozen KDA03 Level-3 arithmetic, inference, and policy-v1.0 routing."""

from __future__ import annotations

import math
from typing import Any

from tools.qlmg_kda01_level3_economic import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    cluster_bootstrap,
    equal_cluster_returns,
    positive_contribution_share,
    score_open_prices,
)


ROUTE_VOCABULARY = (
    "translation_rejected",
    "sample_limited_prospective_candidate",
    "execution_sensitive_candidate",
    "narrow_sleeve_candidate",
    "conditional_context_candidate_unvalidated",
    "unconditional_control_candidate",
)


def branch_side(branch_id: str) -> int:
    mapping = {
        "negative_reference_led_catchup": 1,
        "positive_reference_led_catchup": -1,
        "negative_basis_impulse_continuation": -1,
        "positive_basis_impulse_continuation": 1,
        "negative_completed_basis_impulse_rejection": 1,
        "positive_completed_basis_impulse_rejection": -1,
    }
    suffix = branch_id.removeprefix("primary_").removeprefix("robustness_")
    if suffix not in mapping:
        raise ValueError(f"unknown frozen KDA03 branch side: {branch_id}")
    return mapping[suffix]


def route_flags(row: dict[str, Any]) -> dict[str, bool]:
    finite = lambda name: math.isfinite(float(row[name]))
    flags = {
        "base_market_day_mean_positive": finite("equal_day_base_mean_bps") and float(row["equal_day_base_mean_bps"]) > 0,
        "base_market_day_median_positive": finite("equal_day_base_median_bps") and float(row["equal_day_base_median_bps"]) > 0,
        "bootstrap_lower_ge_minus5": finite("bootstrap_lower_bps") and float(row["bootstrap_lower_bps"]) >= -5,
        "stress_mean_ge_minus10": finite("equal_day_stress_mean_bps") and float(row["equal_day_stress_mean_bps"]) >= -10,
        "symbol_contribution_le_25pct": finite("symbol_positive_share") and float(row["symbol_positive_share"]) <= .25,
        "year_contribution_le_70pct": finite("year_positive_share") and float(row["year_positive_share"]) <= .70,
        "day_contribution_le_10pct": finite("market_day_positive_share") and float(row["market_day_positive_share"]) <= .10,
        "adequate_market_day_clusters": int(row["market_day_clusters"]) >= 50,
        "no_material_estimand_or_context_dependence": not bool(row.get("material_estimand_or_context_dependence", False)),
        "no_single_event_or_defect_explanation": not bool(row.get("single_event_or_defect_explanation", False)),
    }
    flags["control_eligible"] = all((
        flags["base_market_day_mean_positive"], flags["base_market_day_median_positive"],
        flags["bootstrap_lower_ge_minus5"], flags["adequate_market_day_clusters"],
        flags["no_single_event_or_defect_explanation"],
    ))
    return flags


def assign_route(row: dict[str, Any]) -> str:
    """Assign exactly one route in the task's frozen priority order."""
    flags = route_flags(row)
    if not (flags["base_market_day_mean_positive"] and flags["base_market_day_median_positive"]):
        return "translation_rejected"
    if not flags["bootstrap_lower_ge_minus5"] or not flags["adequate_market_day_clusters"]:
        return "sample_limited_prospective_candidate"
    if not flags["stress_mean_ge_minus10"]:
        return "execution_sensitive_candidate"
    if not flags["symbol_contribution_le_25pct"]:
        return "narrow_sleeve_candidate"
    if not all((
        flags["year_contribution_le_70pct"], flags["day_contribution_le_10pct"],
        flags["no_material_estimand_or_context_dependence"],
    )):
        return "conditional_context_candidate_unvalidated"
    return "unconditional_control_candidate"


__all__ = [
    "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED", "ROUTE_VOCABULARY", "assign_route",
    "branch_side", "cluster_bootstrap", "equal_cluster_returns",
    "positive_contribution_share", "route_flags", "score_open_prices",
]
