from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from typing import Any, Mapping, Sequence

from .canonical import canonical_hash
from .engine_types import FamilyInput
from .family_engines.common import require_utc
from .selection import EventObservation, aggregate_materialized, aggregate_streaming


STAGE20_KDA02B_OUTER_FOLDS = (
    "2023Q4",
    "2024Q1",
    "2024Q2",
    "2024Q3",
    "2024Q4",
    "2025Q1",
    "2025Q2",
    "2025Q3",
    "2025Q4",
)
STAGE20_ELIGIBLE_SYMBOLS = 187


class KDA02BDenominatorError(ValueError):
    pass


def _utc(value: object) -> datetime:
    if isinstance(value, datetime):
        return require_utc(value)
    return require_utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))


def _partition(frame: FamilyInput) -> dict[str, Any]:
    raw = frame.metadata.get("campaign_partition")
    if not isinstance(raw, Mapping) or raw.get("phase") != "kda02b_adjudication":
        raise KDA02BDenominatorError("KDA02B denominator frame lacks its Stage20 outer-fold partition")
    fold = str(raw.get("outer_fold_id", ""))
    start = _utc(raw.get("evaluation_start"))
    end = _utc(raw.get("evaluation_end_exclusive"))
    if fold not in STAGE20_KDA02B_OUTER_FOLDS or end <= start:
        raise KDA02BDenominatorError("KDA02B denominator frame has an invalid Stage20 evaluation interval")
    seconds = (end - start).total_seconds()
    days = int(frame.metadata.get("eligible_days", -1))
    symbol_seconds = float(frame.metadata.get("eligible_symbol_seconds", math.nan))
    if seconds % 86400 != 0 or days != int(seconds // 86400):
        raise KDA02BDenominatorError("KDA02B frame eligible-day denominator differs from Stage20")
    eligible_symbols = symbol_seconds / seconds
    if not math.isfinite(symbol_seconds) or not math.isclose(
        eligible_symbols, STAGE20_ELIGIBLE_SYMBOLS, rel_tol=0.0, abs_tol=1e-12
    ):
        raise KDA02BDenominatorError("KDA02B frame occupancy denominator differs from Stage20")
    return {
        "outer_fold_id": fold,
        "evaluation_start": start,
        "evaluation_end_exclusive": end,
        "eligible_days": days,
        "eligible_symbols": STAGE20_ELIGIBLE_SYMBOLS,
        "eligible_symbol_seconds": symbol_seconds,
    }


def stage20_kda02b_denominator_contract(frames: Sequence[FamilyInput]) -> dict[str, Any]:
    """Reconcile the nine Stage20 fold denominators before cross-fold aggregation."""
    by_fold: dict[str, dict[str, Any]] = {}
    frame_counts: dict[str, int] = {}
    for frame in frames:
        part = _partition(frame)
        fold = str(part["outer_fold_id"])
        prior = by_fold.get(fold)
        if prior is not None and prior != part:
            raise KDA02BDenominatorError("KDA02B frames disagree on one Stage20 fold denominator")
        by_fold[fold] = part
        frame_counts[fold] = frame_counts.get(fold, 0) + 1
    if tuple(sorted(by_fold, key=STAGE20_KDA02B_OUTER_FOLDS.index)) != STAGE20_KDA02B_OUTER_FOLDS:
        missing = sorted(set(STAGE20_KDA02B_OUTER_FOLDS) - set(by_fold))
        extra = sorted(set(by_fold) - set(STAGE20_KDA02B_OUTER_FOLDS))
        raise KDA02BDenominatorError(f"KDA02B Stage20 denominator fold coverage differs: missing={missing}, extra={extra}")
    ordered = [by_fold[fold] for fold in STAGE20_KDA02B_OUTER_FOLDS]
    for left, right in zip(ordered, ordered[1:]):
        if left["evaluation_end_exclusive"] != right["evaluation_start"]:
            raise KDA02BDenominatorError("KDA02B Stage20 denominator intervals overlap or contain a gap")
    eligible_days = sum(int(part["eligible_days"]) for part in ordered)
    eligible_symbol_seconds = sum(float(part["eligible_symbol_seconds"]) for part in ordered)
    serialized = [{
        **part,
        "evaluation_start": part["evaluation_start"].isoformat(),
        "evaluation_end_exclusive": part["evaluation_end_exclusive"].isoformat(),
        "frame_count": frame_counts[str(part["outer_fold_id"])],
    } for part in ordered]
    return {
        "schema": "stage24_kda02b_stage20_denominator_contract_v1",
        "stage20_opportunity_formula": "30 * accepted_event_count / eligible_calendar_days",
        "stage20_occupancy_formula": "sum(actual_exposure_seconds) / (187 * evaluation_interval_seconds)",
        "cross_fold_rule": "sum the nine disjoint Stage20 outer-fold denominators before aggregate reconstruction",
        "eligible_days": eligible_days,
        "eligible_symbols": STAGE20_ELIGIBLE_SYMBOLS,
        "eligible_symbol_seconds": eligible_symbol_seconds,
        "evaluation_start": serialized[0]["evaluation_start"],
        "evaluation_end_exclusive": serialized[-1]["evaluation_end_exclusive"],
        "partitions": serialized,
        "contract_sha256": canonical_hash(serialized),
    }


def _partition_for_observation(item: EventObservation, contract: Mapping[str, Any]) -> Mapping[str, Any]:
    decision = require_utc(item.decision_ts)
    matches = [
        part for part in contract["partitions"]
        if _utc(part["evaluation_start"]) <= decision < _utc(part["evaluation_end_exclusive"])
    ]
    if len(matches) != 1:
        raise KDA02BDenominatorError("KDA02B observation does not map to exactly one Stage20 denominator interval")
    return matches[0]


def reconcile_stage20_kda02b_aggregate(
    observations: Sequence[EventObservation],
    contract: Mapping[str, Any],
) -> tuple[tuple[EventObservation, ...], dict[str, Any], dict[str, Any]]:
    """Rebind per-fold observations to the exact additive nine-fold denominator.

    The strict aggregate invariant remains unchanged: every observation passed
    to either aggregate implementation carries one identical job denominator.
    Before rebinding, each observation must match its original Stage20 fold.
    """
    normalized: list[EventObservation] = []
    for item in observations:
        part = _partition_for_observation(item, contract)
        if item.eligible_days != int(part["eligible_days"]) or not math.isclose(
            float(item.eligible_symbol_seconds),
            float(part["eligible_symbol_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise KDA02BDenominatorError("KDA02B observation denominator differs from its Stage20 fold")
        normalized.append(replace(
            item,
            eligible_days=int(contract["eligible_days"]),
            eligible_symbol_seconds=float(contract["eligible_symbol_seconds"]),
        ))
    normalized_rows = tuple(sorted(normalized, key=lambda item: (item.market_day, item.symbol, item.event_id)))
    streaming = aggregate_streaming(iter(normalized_rows))
    materialized = aggregate_materialized(normalized_rows)
    component_values: dict[str, list[float]] = {}
    for item in normalized_rows:
        for name, value in item.component_metrics:
            component_values.setdefault(name, []).append(float(value))
    stable_components = {
        name: math.fsum(values) / len(values)
        for name, values in sorted(component_values.items())
    }
    # Both independent paths use the same order-independent final reduction for
    # forensic component means. This removes only binary addition-order noise;
    # event values, weights and economic formulas are unchanged.
    streaming["component_metrics"] = stable_components
    materialized["component_metrics"] = stable_components
    streaming_sha = canonical_hash(streaming)
    materialized_sha = canonical_hash(materialized)
    if streaming_sha != materialized_sha:
        differing = sorted(
            key for key in set(streaming) | set(materialized)
            if canonical_hash(streaming.get(key)) != canonical_hash(materialized.get(key))
        )
        raise KDA02BDenominatorError(
            f"KDA02B aggregate/materialized reconstruction differs: fields={differing}"
        )
    primary_days = len(streaming.get("day_base_net_bps", {}))
    primary_numerator = sum(float(value) for value in streaming.get("day_base_net_bps", {}).values())
    holding_seconds = sum(float(item.holding_seconds_weighted) for item in normalized_rows)
    trace = {
        "schema": "stage24_kda02b_denominator_reconciliation_v1",
        "input_observation_count": len(observations),
        "normalized_observation_count": len(normalized_rows),
        "zero_exposure_observation_count": sum(item.holding_seconds_weighted == 0 for item in normalized_rows),
        "aggregate_numerators": {
            "primary_sum_of_utc_day_means_bps": primary_numerator,
            "opportunity_30x_event_count": 30 * len(normalized_rows),
            "occupancy_actual_exposure_seconds": holding_seconds,
        },
        "aggregate_denominators": {
            "primary_market_days": primary_days,
            "opportunity_eligible_calendar_days": int(contract["eligible_days"]),
            "occupancy_eligible_symbol_seconds": float(contract["eligible_symbol_seconds"]),
        },
        "streaming_aggregate_sha256": streaming_sha,
        "materialized_aggregate_sha256": materialized_sha,
        "aggregate_materialized_equal": True,
        "denominator_contract_sha256": contract["contract_sha256"],
    }
    return normalized_rows, streaming, trace


__all__ = [
    "KDA02BDenominatorError",
    "STAGE20_ELIGIBLE_SYMBOLS",
    "STAGE20_KDA02B_OUTER_FOLDS",
    "reconcile_stage20_kda02b_aggregate",
    "stage20_kda02b_denominator_contract",
]
