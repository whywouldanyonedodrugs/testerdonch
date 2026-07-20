from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .canonical import canonical_json_bytes
from .family_engines.common import type7_quantile_with_negative_infinity
from .schema import gower_distance


INNER_FOLD_EMPTY = -math.inf


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
    decision_ts: datetime
    entry_ts: datetime
    exit_ts: datetime
    platform: str = "kraken_native_linear_pf"
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
        if item.platform != "kraken_native_linear_pf" or not (item.decision_ts <= item.entry_ts < item.exit_ts) or item.exit_ts.year >= 2026:
            raise ValueError("event observation violates platform or rankable timestamp firewall")
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
    """Independent one-pass aggregate; does not call/materialize the ledger path."""
    seen: set[str] = set()
    day_base: dict[str, list[float]] = {}
    day_stress: dict[str, list[float]] = {}
    symbol_sums: dict[str, float] = {}
    month_sums: dict[str, float] = {}
    year_sums: dict[int, float] = {}
    event_values: list[float] = []
    component_sums: dict[str, tuple[float, int]] = {}
    eligible_days_values: set[int] = set()
    occupancy_denominators: set[float] = set()
    threshold_count = 0
    holding = 0.0
    count = 0
    for item in events:
        if item.platform != "kraken_native_linear_pf" or not (item.decision_ts <= item.entry_ts < item.exit_ts) or item.exit_ts.year >= 2026:
            raise ValueError("event observation violates platform or rankable timestamp firewall")
        if item.event_id in seen:
            raise ValueError("duplicate economic event ID")
        seen.add(item.event_id)
        count += 1
        event_values.append(item.base_net_bps)
        day_base.setdefault(item.market_day, []).append(item.base_net_bps)
        day_stress.setdefault(item.market_day, []).append(item.stress_net_bps)
        symbol_sums[item.symbol] = symbol_sums.get(item.symbol, 0.0) + item.base_net_bps
        month_sums[item.month] = month_sums.get(item.month, 0.0) + item.base_net_bps
        year_sums[item.year] = year_sums.get(item.year, 0.0) + item.base_net_bps
        threshold_count += int(item.threshold_eligible)
        holding += item.holding_seconds_weighted
        if item.eligible_days is not None:
            eligible_days_values.add(item.eligible_days)
        if item.eligible_symbol_seconds is not None:
            occupancy_denominators.add(item.eligible_symbol_seconds)
        for name, value in item.component_metrics:
            subtotal, n = component_sums.get(name, (0.0, 0))
            component_sums[name] = (subtotal + float(value), n + 1)
    if len(eligible_days_values) > 1 or len(occupancy_denominators) > 1:
        raise ValueError("inconsistent aggregate denominator")
    base_days = [sum(day_base[key]) / len(day_base[key]) for key in sorted(day_base)]
    stress_days = [sum(day_stress[key]) / len(day_stress[key]) for key in sorted(day_stress)]
    total_abs = sum(abs(value) for value in event_values)
    concentration = lambda values: max((abs(value) for value in values), default=0.0) / total_abs if total_abs else 0.0
    eligible_days = next(iter(eligible_days_values)) if eligible_days_values else None
    occupancy_denominator = next(iter(occupancy_denominators)) if occupancy_denominators else None
    return {
        "event_count": count,
        "market_days": len(day_base),
        "base_net_bps": sum(base_days) / len(base_days) if base_days else None,
        "stress_net_bps": sum(stress_days) / len(stress_days) if stress_days else None,
        "median_event_base_bps": median(event_values) if event_values else None,
        "trimmed_mean_event_base_bps": _trimmed_mean(event_values),
        "symbol_concentration": concentration(symbol_sums.values()),
        "month_concentration": concentration(month_sums.values()),
        "year_concentration": concentration(year_sums.values()),
        "symbol_contributions": dict(sorted(symbol_sums.items())),
        "month_contributions": dict(sorted(month_sums.items())),
        "year_contributions": {str(key): value for key, value in sorted(year_sums.items())},
        "threshold_coverage": threshold_count / count if count else None,
        "opportunity_frequency_per_30d": 30.0 * count / eligible_days if eligible_days else None,
        "occupancy": holding / occupancy_denominator if occupancy_denominator else None,
        "component_metrics": {name: subtotal / n for name, (subtotal, n) in sorted(component_sums.items())},
    }


def _trimmed_mean(values: Sequence[float], fraction: float = 0.1) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    trim = int(len(ordered) * fraction)
    selected = ordered[trim:len(ordered) - trim] if trim and 2 * trim < len(ordered) else ordered
    return sum(selected) / len(selected)


def inner_fold_summary(values: Sequence[float | None]) -> dict[str, Any]:
    explicit = [INNER_FOLD_EMPTY if value is None else float(value) for value in values]
    if any(math.isnan(value) or value == math.inf for value in explicit):
        raise ValueError("invalid inner-fold value")
    finite = [value for value in explicit if value != -math.inf]
    return {
        "vector": explicit,
        "fold_count": len(explicit),
        "nonempty_count": len(finite),
        "nonempty_fraction": len(finite) / len(explicit) if explicit else 0.0,
        "p20_with_negative_infinity": type7_quantile_with_negative_infinity(explicit, 0.2) if explicit else -math.inf,
        "empty_count": len(explicit) - len(finite),
    }


def _medoid(family_id: str, members: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted(members, key=lambda row: str(row["canonical_economic_address_sha256"]))
    return min(
        ordered,
        key=lambda candidate: (
            sum(gower_distance(family_id, candidate["config"], other["config"]) for other in ordered),
            str(candidate["canonical_economic_address_sha256"]),
        ),
    )["canonical_economic_address_sha256"]


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
        member_ids = sorted(str(row["canonical_economic_address_sha256"]) for row in members)
        neighborhoods.append({
            "center": center["canonical_economic_address_sha256"],
            "medoid": _medoid(family_id, members),
            "support": len(set(member_ids)),
            "member_ids": member_ids,
            "passed": passed,
        })
    return neighborhoods


def resolve_region_overlap(regions: Sequence[Mapping[str, Any]], threshold: float = 0.50) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    retained: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    eligible = sorted((row for row in regions if row.get("passed")), key=lambda row: (-int(row["support"]), str(row["medoid"])))
    for region in eligible:
        members = set(region["member_ids"])
        conflict = None
        for prior in retained:
            prior_members = set(prior["member_ids"])
            union = members | prior_members
            overlap = len(members & prior_members) / len(union) if union else 1.0
            if overlap >= threshold:
                conflict = (prior["medoid"], overlap)
                break
        if conflict is None:
            retained.append(region)
        else:
            rejected.append({"medoid": region["medoid"], "overlaps": conflict[0], "jaccard": conflict[1], "status": "rejected_region_overlap"})
    return retained, rejected


def refinement_neighbors(family_id: str, center: Mapping[str, Any], priority_pairs: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    from .schema import family_schemas, normalize_config
    schema = family_schemas[family_id]
    normalized = normalize_config(family_id, center)
    moves: dict[str, list[Any]] = {}
    for spec in schema.axes:
        if not spec.search_new_broad or spec.allowed_values is None or normalized[spec.name] is None:
            continue
        levels = list(spec.allowed_values)
        index = levels.index(normalized[spec.name])
        moves[spec.name] = [levels[item] for item in (index - 1, index + 1) if 0 <= item < len(levels)]
    candidates: dict[bytes, dict[str, Any]] = {}
    for field, values in moves.items():
        for value in values:
            raw = dict(normalized); raw[field] = value
            try:
                candidate = normalize_config(family_id, raw)
            except Exception:
                continue
            candidates[canonical_json_bytes(candidate)] = candidate
    for left, right in priority_pairs:
        for left_value in moves.get(left, []):
            for right_value in moves.get(right, []):
                raw = dict(normalized); raw[left] = left_value; raw[right] = right_value
                try:
                    candidate = normalize_config(family_id, raw)
                except Exception:
                    continue
                candidates[canonical_json_bytes(candidate)] = candidate
    return [candidates[key] for key in sorted(candidates)]


def allocate_refinements(family_id: str, retained_regions: Sequence[Mapping[str, Any]], rows_by_address: Mapping[str, Mapping[str, Any]], reserved_ids: Sequence[str], existing_addresses: set[str]) -> list[dict[str, Any]]:
    from .schema import economic_address, family_schemas
    generated: list[tuple[str, dict[str, Any], str]] = []
    for region in retained_regions:
        medoid = str(region["medoid"])
        for config in refinement_neighbors(family_id, rows_by_address[medoid]["config"], family_schemas[family_id].priority_pairs):
            _, address = economic_address(family_id, config)
            if address not in existing_addresses:
                generated.append((address, config, medoid))
                existing_addresses.add(address)
    generated.sort(key=lambda item: item[0])
    output = []
    for index, reserved in enumerate(reserved_ids):
        if index >= len(generated):
            output.append({"reserved_attempt_id": reserved, "status": "unavailable_no_plateau"})
        else:
            address, config, medoid = generated[index]
            output.append({"reserved_attempt_id": reserved, "status": "registered_refinement", "canonical_economic_address_sha256": address, "center_medoid": medoid, "config": config})
    return output


def day_cluster_bootstrap_q05(day_values: Sequence[float], seed: int, replicates: int = 5000) -> float:
    if not day_values:
        return -math.inf
    values = np.asarray(day_values, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        means[index] = values[generator.integers(0, len(values), size=len(values))].mean()
    return float(np.quantile(means, 0.05, method="linear"))


def paired_control_pass(uplifts_by_day: Sequence[float], coverage: float, seed: int) -> dict[str, Any]:
    mean = sum(uplifts_by_day) / len(uplifts_by_day) if uplifts_by_day else -math.inf
    q05 = day_cluster_bootstrap_q05(uplifts_by_day, seed)
    return {"mean_paired_uplift": mean, "bootstrap_q05": q05, "coverage": coverage, "pass": coverage >= 0.70 and mean > 0 and q05 > 0}


def adjudicate_route(family_id: str, common_gate: bool, main_null: bool, component_passes: Mapping[str, bool], *, base_positive: bool, stress_positive: bool, delay_positive: bool, sample_sufficient: bool, add_fraction: float | None = None) -> str:
    if base_positive and (not stress_positive or not delay_positive):
        return "execution_sensitive_candidate"
    if base_positive and not sample_sufficient:
        return "sample_limited_context_candidate" if family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1" else "sample_limited_candidate"
    components_ok = all(component_passes.values())
    if family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
        return "context_uplift_candidate" if common_gate and main_null and components_ok else "overlay_rejected"
    if family_id == "A3_STARTER_RETEST_V3":
        if common_gate and (add_fraction == 0 or (main_null and components_ok)):
            return "starter_only_candidate" if add_fraction == 0 else "starter_plus_add_candidate"
        return "translation_rejected"
    if family_id == "KDA02B_SURVIVOR_ADJUDICATION_V1":
        return "component_supported_survivor" if common_gate and main_null and components_ok and stress_positive and delay_positive else "survivor_rejected"
    return "control_supported_rolling_candidate" if common_gate and main_null and components_ok else "translation_rejected"


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
