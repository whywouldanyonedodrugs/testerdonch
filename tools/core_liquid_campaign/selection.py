from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from .family_engines.common import type7_quantile
from .schema import gower_distance


INNER_FOLD_EMPTY = None


@dataclass(frozen=True)
class EventObservation:
    event_id: str
    symbol: str
    entry_day: str
    month: str
    year: int
    base_net_bps: float
    stress_net_bps: float
    market_day: str
    eligible_days: int | None = None
    threshold_eligible: bool = True
    component_metrics: tuple[tuple[str, float], ...] = ()
    holding_seconds_weighted: float = 0.0
    eligible_symbol_seconds: float | None = None


def aggregate_materialized(events: Sequence[EventObservation]) -> dict[str, Any]:
    ordered = sorted(events, key=lambda item: (item.market_day, item.symbol, item.event_id))
    if len({item.event_id for item in ordered}) != len(ordered):
        raise ValueError("duplicate economic event ID")
    by_day: dict[str, list[EventObservation]] = {}
    by_symbol: dict[str, float] = {}
    by_month: dict[str, float] = {}
    by_year: dict[int, float] = {}
    component_values: dict[str, list[float]] = {}
    for item in ordered:
        by_day.setdefault(item.market_day, []).append(item)
        by_symbol[item.symbol] = by_symbol.get(item.symbol, 0.0) + item.base_net_bps
        by_month[item.month] = by_month.get(item.month, 0.0) + item.base_net_bps
        by_year[item.year] = by_year.get(item.year, 0.0) + item.base_net_bps
        for name, value in item.component_metrics:
            component_values.setdefault(name, []).append(float(value))
    day_base = [sum(item.base_net_bps for item in rows) / len(rows) for _, rows in sorted(by_day.items())]
    day_stress = [sum(item.stress_net_bps for item in rows) / len(rows) for _, rows in sorted(by_day.items())]
    total_abs = sum(abs(item.base_net_bps) for item in ordered)
    concentration = lambda values: (max((abs(value) for value in values), default=0.0) / total_abs if total_abs else 0.0)
    eligible_day_values = {item.eligible_days for item in ordered if item.eligible_days is not None}
    if len(eligible_day_values) > 1:
        raise ValueError("inconsistent eligible-day denominator")
    eligible_days = next(iter(eligible_day_values)) if eligible_day_values else None
    occupancy_denominators = {item.eligible_symbol_seconds for item in ordered if item.eligible_symbol_seconds is not None}
    if len(occupancy_denominators) > 1:
        raise ValueError("inconsistent occupancy denominator")
    occupancy_denominator = next(iter(occupancy_denominators)) if occupancy_denominators else None
    return {
        "event_count": len(ordered),
        "market_days": len(by_day),
        "base_net_bps": sum(day_base) / len(day_base) if day_base else None,
        "stress_net_bps": sum(day_stress) / len(day_stress) if day_stress else None,
        "median_event_base_bps": median([item.base_net_bps for item in ordered]) if ordered else None,
        "trimmed_mean_event_base_bps": _trimmed_mean([item.base_net_bps for item in ordered]),
        "symbol_concentration": concentration(by_symbol.values()),
        "month_concentration": concentration(by_month.values()),
        "year_concentration": concentration(by_year.values()),
        "symbol_contributions": dict(sorted(by_symbol.items())),
        "month_contributions": dict(sorted(by_month.items())),
        "year_contributions": {str(key): value for key, value in sorted(by_year.items())},
        "threshold_coverage": sum(item.threshold_eligible for item in ordered) / len(ordered) if ordered else None,
        "opportunity_frequency_per_30d": 30.0 * len(ordered) / eligible_days if eligible_days else None,
        "occupancy": sum(item.holding_seconds_weighted for item in ordered) / occupancy_denominator if occupancy_denominator else None,
        "component_metrics": {name: sum(values) / len(values) for name, values in sorted(component_values.items())},
    }


def aggregate_streaming(events: Iterable[EventObservation]) -> dict[str, Any]:
    return aggregate_materialized(list(events))


def _trimmed_mean(values: Sequence[float], fraction: float = 0.1) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    selected = ordered[trim:len(ordered) - trim] if trim and 2 * trim < len(ordered) else ordered
    return sum(selected) / len(selected)


def inner_fold_summary(values: Sequence[float | None]) -> dict[str, Any]:
    explicit = [None if value is None else float(value) for value in values]
    finite = [value for value in explicit if value is not None and math.isfinite(value)]
    return {
        "vector": explicit,
        "fold_count": len(explicit),
        "nonempty_count": len(finite),
        "nonempty_fraction": len(finite) / len(explicit) if explicit else 0.0,
        "p20_with_empties_unavailable": type7_quantile(finite, 0.2) if finite and len(finite) == len(explicit) else None,
        "empty_count": len(explicit) - len(finite),
    }


def stable_neighborhoods(family_id: str, rows: Sequence[Mapping[str, Any]], radius: float = 0.15) -> list[dict[str, Any]]:
    neighborhoods: list[dict[str, Any]] = []
    for center in rows:
        members = [row for row in rows if gower_distance(family_id, center["config"], row["config"]) <= radius]
        positive = sum(float(row["base_net_bps"]) > 0 for row in members)
        inner_ok = [float(row["inner_nonempty_fraction"]) >= 0.75 for row in members]
        varied = sum(len({str(row["config"].get(key)) for row in members}) > 1 for key in center["config"])
        passed = (
            len({row["canonical_economic_address_sha256"] for row in members}) >= 5
            and varied >= 2
            and positive / len(members) >= 0.60
            and median(float(row["base_net_bps"]) for row in members) > 0
            and median(float(row["stress_net_bps"]) for row in members) > -18
            and all(inner_ok)
        )
        neighborhoods.append({"center": center["canonical_economic_address_sha256"], "support": len(members), "passed": passed})
    return neighborhoods


def beam_order_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["plateau_support_count"]),
        -float(row["day_cluster_bootstrap_q05"]),
        -float(row["p20_inner_fold"]),
        -float(row["stress_net_bps"]),
        -float(row["opportunity_frequency"]),
        float(row["complexity"]),
        str(row["canonical_economic_address_sha256"]),
    )


def select_beam(rows: Sequence[Mapping[str, Any]], width: int = 5) -> list[Mapping[str, Any]]:
    eligible = [
        row for row in rows
        if row.get("stable_region")
        and int(row.get("accepted_trades", 0)) >= 30
        and int(row.get("market_days", 0)) >= 20
        and math.isfinite(float(row.get("base_net_bps", math.nan)))
        and float(row["base_net_bps"]) > 0
        and float(row.get("threshold_coverage", 0.0)) >= 0.70
    ]
    return sorted(eligible, key=beam_order_key)[:width]


def deduplicate_event_overlap(rows: Sequence[Mapping[str, Any]], threshold: float = 0.80) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    retained: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in sorted(rows, key=beam_order_key):
        current = set(row.get("event_ids", ()))
        duplicate_of = None
        overlap = 0.0
        for prior in retained:
            prior_events = set(prior.get("event_ids", ()))
            union = current | prior_events
            score = len(current & prior_events) / len(union) if union else 1.0
            if score > threshold:
                duplicate_of = prior["canonical_economic_address_sha256"]
                overlap = score
                break
        if duplicate_of is None:
            retained.append(row)
        else:
            rejected.append({"canonical_economic_address_sha256": row["canonical_economic_address_sha256"], "duplicate_of": duplicate_of, "jaccard": overlap, "status": "rejected_event_overlap"})
    return retained, rejected


def family_outer_vector(fold_values: Mapping[str, Sequence[float]], fold_order: Sequence[str]) -> list[float]:
    return [median(fold_values[fold]) if fold in fold_values and fold_values[fold] else -math.inf for fold in fold_order]


def materialization_policy(rows: Sequence[Mapping[str, Any]], failed_audit_modulus: int = 97) -> list[str]:
    selected: set[str] = set()
    for row in rows:
        address = str(row["canonical_economic_address_sha256"])
        if row.get("beam_survivor") or row.get("mechanism_anchor") or row.get("main_component_null"):
            selected.add(address)
        if row.get("near_miss") and row.get("near_miss_rule") == "one_failed_nonintegrity_gate_within_10pct":
            selected.add(address)
        if not row.get("passed") and int(address[:8], 16) % failed_audit_modulus == 0:
            selected.add(address)
    return sorted(selected)


def safe_pruning_policy() -> dict[str, Any]:
    return {
        "preoutcome": ["invalid_combination", "duplicate_economic_address", "inactive_axis_duplicate", "required_data_unavailable", "source_or_hash_integrity_failure", "impossible_minimum_sample_after_complete_enumeration"],
        "stage_boundary": ["no_eligible_plateau_under_frozen_rule", "component_null_exact_economic_equality", "family_specific_integrity_failure"],
        "prohibited": ["early_unprofitability", "first_fold_loss", "symbol_loss", "month_loss", "stochastic_successive_halving"],
        "independent_family_continuation": True,
    }
