"""Outcome-free KDA01 v2 causal features, episodes, and event identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tools.qlmg_kraken_derivatives_state import (
    COHORT_VERSION,
    FEATURE_VERSION,
    causal_daily_normalization,
    stable_hash,
    validate_rankable_times,
)


TRANSLATION_ID = "KDA01_v2_episode_level_crowding_price_progress_bifurcation"
FEATURE_EXTENSION_VERSION = "kda01_v2_feature_extension_v1_20260719"
GENERATOR_VERSION = "kda01_v2_episode_generator_v1_20260719"
ATTEMPTS = ("primary", "robustness")
DIRECTIONS = (1, -1)

FEATURE_EXTENSION_CONTRACT: dict[str, Any] = {
    "translation_id": TRANSLATION_ID,
    "version": FEATURE_EXTENSION_VERSION,
    "grid": "completed_5m",
    "normalization": {
        "lookback_calendar_days": 60,
        "minimum_valid_days": 30,
        "minimum_expected_fraction": 0.70,
        "availability": "prior_UTC_day_only",
        "daily_aggregation": "median",
        "zero_mad": "fail_closed",
    },
    "oi_expansion": {
        "primary": "oi_change_robust_z >= 2",
        "robustness": "oi_change_percentile >= 0.95",
    },
    "price_progress_per_oi": "abs(trade_return_1h) / oi_log_change_1h only on directionally coherent primary-or-robustness pre-progress parent rows with finite positive denominator",
    "outcomes_authorized": False,
}
FEATURE_EXTENSION_HASH = stable_hash(FEATURE_EXTENSION_CONTRACT)

GENERATOR_CONTRACT: dict[str, Any] = {
    "translation_id": TRANSLATION_ID,
    "version": GENERATOR_VERSION,
    "feature_extension_hash": FEATURE_EXTENSION_HASH,
    "parent_absence_reset": "12 contiguous completed 5m rows / 60 minutes",
    "hysteresis": "oi_change_robust_z >= 1 and direction*basis_level_robust_z >= 1",
    "hysteresis_exit": "6 consecutive completed 5m rows outside or six-hour maximum",
    "initial_impulse": "first 12 completed 5m rows",
    "failure": "first completed trade+mark structural close-through after deterioration and impulse completion",
    "parent_required_validity": "finite current price-progress percentile and path_efficiency_1h in addition to causal normalization and coverage fields",
    "candidate_limit": "one efficient onset and one completed failure per parent episode",
    "outcomes_authorized": False,
}
GENERATOR_HASH = stable_hash(GENERATOR_CONTRACT)


def extend_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add OI and price-progress scores using distributions ending prior UTC day."""
    required = {
        "timestamp_utc", "oi_log_change_1h", "trade_return_1h", "path_efficiency_1h",
        "mark_return_1h", "basis_decimal", "basis_level_robust_z", "basis_level_percentile",
        "basis_level_normalization_valid", "eligible", "known_lifecycle_mask",
        "trade_coverage", "mark_coverage", "analytics_coverage",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing KDA01 v2 feature inputs: {missing}")
    out = frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True).copy()
    validate_rankable_times(out.timestamp_utc)
    oi = pd.to_numeric(out.oi_log_change_1h, errors="coerce")
    oi_stats = causal_daily_normalization(out.timestamp_utc, oi)
    out["oi_change_robust_z"] = oi_stats.robust_z
    out["oi_change_percentile"] = oi_stats.empirical_percentile
    out["oi_change_normalization_valid"] = oi_stats.normalization_valid
    preprogress_parent = pd.Series(False, index=out.index)
    for attempt in ATTEMPTS:
        for direction in DIRECTIONS:
            preprogress_parent |= _preprogress_parent_mask(out, attempt, direction)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (pd.to_numeric(out.trade_return_1h, errors="coerce").abs() / oi.where(oi > 0)).where(preprogress_parent)
    ratio = ratio.where(np.isfinite(ratio))
    progress_stats = causal_daily_normalization(out.timestamp_utc, ratio)
    out["price_progress_per_oi"] = ratio
    out["price_progress_robust_z"] = progress_stats.robust_z
    out["price_progress_percentile"] = progress_stats.empirical_percentile
    out["price_progress_normalization_valid"] = progress_stats.normalization_valid
    return out


def progress_classification(frame: pd.DataFrame) -> pd.Series:
    percentile = pd.to_numeric(frame.price_progress_percentile, errors="coerce")
    efficiency = pd.to_numeric(frame.path_efficiency_1h, errors="coerce")
    valid = frame.price_progress_normalization_valid.fillna(False) & percentile.notna() & efficiency.notna()
    result = pd.Series("invalid", index=frame.index, dtype="object")
    result.loc[valid] = "intermediate"
    result.loc[valid & ((percentile <= 0.25) | (efficiency <= 0.25))] = "deteriorating_progress"
    result.loc[valid & (percentile >= 0.75) & (efficiency >= 0.50)] = "efficient_progress"
    return result


def _preprogress_parent_mask(frame: pd.DataFrame, attempt: str, direction: int) -> pd.Series:
    if attempt not in ATTEMPTS or direction not in DIRECTIONS:
        raise ValueError("unsupported attempt or direction")
    trade_direction = np.sign(pd.to_numeric(frame.trade_return_1h, errors="coerce"))
    mark_direction = np.sign(pd.to_numeric(frame.mark_return_1h, errors="coerce"))
    basis = pd.to_numeric(frame.basis_decimal, errors="coerce")
    basis_z = pd.to_numeric(frame.basis_level_robust_z, errors="coerce")
    base = (
        frame.eligible.fillna(False)
        & frame.known_lifecycle_mask.fillna(False)
        & frame.trade_coverage.fillna(False)
        & frame.mark_coverage.fillna(False)
        & frame.analytics_coverage.fillna(False)
        & frame.basis_level_normalization_valid.fillna(False)
        & frame.oi_change_normalization_valid.fillna(False)
        & trade_direction.eq(direction)
        & mark_direction.eq(direction)
        & (direction * basis > 0)
    )
    if attempt == "primary":
        return base & (frame.oi_change_robust_z >= 2) & (direction * basis_z >= 2)
    basis_percentile = pd.to_numeric(frame.basis_level_percentile, errors="coerce")
    coherent_tail = basis_percentile.ge(0.95) if direction > 0 else basis_percentile.le(0.05)
    return base & (frame.oi_change_percentile >= 0.95) & coherent_tail


def parent_mask(frame: pd.DataFrame, attempt: str, direction: int) -> pd.Series:
    percentile = pd.to_numeric(frame.price_progress_percentile, errors="coerce")
    efficiency = pd.to_numeric(frame.path_efficiency_1h, errors="coerce")
    return (
        _preprogress_parent_mask(frame, attempt, direction)
        & frame.price_progress_normalization_valid.fillna(False)
        & percentile.notna() & np.isfinite(percentile)
        & efficiency.notna() & np.isfinite(efficiency)
    )


def hysteresis_mask(frame: pd.DataFrame, direction: int) -> pd.Series:
    return (
        frame.oi_change_normalization_valid.fillna(False)
        & frame.basis_level_normalization_valid.fillna(False)
        & (frame.oi_change_robust_z >= 1)
        & (direction * frame.basis_level_robust_z >= 1)
    )


def _contiguous(ts: pd.Series, left: int, right: int) -> bool:
    return right >= left and pd.Timestamp(ts.iloc[right]) - pd.Timestamp(ts.iloc[left]) == pd.Timedelta(minutes=5 * (right-left))


def deterministic_episode_id(symbol: str, attempt: str, direction: int, onset_ts: Any) -> str:
    return "kda01v2_episode_" + stable_hash({
        "symbol": symbol, "attempt": attempt, "direction": direction,
        "onset_ts": pd.Timestamp(onset_ts).isoformat(), "generator_hash": GENERATOR_HASH,
    })[:32]


def deterministic_event_ids(row: Mapping[str, Any]) -> tuple[str, str]:
    payload = {
        key: str(row[key]) for key in (
            "branch_id", "attempt", "symbol", "parent_direction", "trade_direction",
            "parent_episode_id", "decision_ts", "feature_extension_hash", "generator_hash",
        )
    }
    event_id = "kda01v2_event_" + stable_hash(payload)[:32]
    address = "kda01v2_addr_" + stable_hash({**payload, "identity_type": "pre_exit_candidate"})[:32]
    return event_id, address


@dataclass(frozen=True)
class EpisodeBounds:
    onset: int
    end: int
    maximum: int


def _episode_bounds(frame: pd.DataFrame, onset_index: int, direction: int) -> EpisodeBounds:
    ts = frame.timestamp_utc
    hysteresis = hysteresis_mask(frame, direction).fillna(False).to_numpy(dtype=bool)
    maximum = onset_index
    while maximum + 1 < len(frame) and maximum + 1 - onset_index <= 72 and _contiguous(ts, onset_index, maximum + 1):
        maximum += 1
    outside = 0
    end = maximum
    for index in range(onset_index, maximum + 1):
        outside = 0 if hysteresis[index] else outside + 1
        if outside >= 6:
            end = index
            break
    return EpisodeBounds(onset_index, end, maximum)


def generate_parent_episodes_and_events(
    frame: pd.DataFrame, *, symbol: str, semantic_hash: str, analytics_manifest_hash: str,
    cohort_hash: str, source_refs: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate complete parent episodes and at most two candidate events per episode."""
    required_ohlc = {"trade_high", "trade_low", "mark_high", "mark_low", "trade_close", "mark_close"}
    if missing := sorted(required_ohlc - set(frame.columns)):
        raise ValueError(f"missing structural OHLC: {missing}")
    ordered = frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True).copy()
    ordered["progress_class"] = progress_classification(ordered)
    ts = ordered.timestamp_utc
    episode_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for attempt in ATTEMPTS:
        for direction in DIRECTIONS:
            parent = parent_mask(ordered, attempt, direction).fillna(False).to_numpy(dtype=bool)
            last_episode_end = -1
            for index in np.flatnonzero(parent):
                if index <= last_episode_end or index < 12 or not _contiguous(ts, index-12, index):
                    continue
                if parent[index-12:index].any():
                    continue
                bounds = _episode_bounds(ordered, index, direction)
                last_episode_end = bounds.end
                episode_id = deterministic_episode_id(symbol, attempt, direction, ts.iloc[index])
                impulse_end = index + 11
                impulse_complete = impulse_end <= bounds.end and _contiguous(ts, index, impulse_end)
                onset_class = ordered.progress_class.iloc[index]
                episode_slice = ordered.iloc[index:bounds.end+1]
                deterioration_indices = episode_slice.index[episode_slice.progress_class.eq("deteriorating_progress")].tolist()
                episode = {
                    "translation_id": TRANSLATION_ID, "parent_episode_id": episode_id,
                    "attempt": attempt, "symbol": symbol, "parent_direction": direction,
                    "parent_onset_ts": ts.iloc[index], "parent_decision_ts": ts.iloc[index] + pd.Timedelta(minutes=5),
                    "episode_end_ts": ts.iloc[bounds.end] + pd.Timedelta(minutes=5),
                    "maximum_deadline_ts": ts.iloc[index] + pd.Timedelta(hours=6, minutes=5),
                    "onset_progress_class": onset_class,
                    "entered_deteriorating_progress": bool(deterioration_indices),
                    "impulse_complete": bool(impulse_complete),
                    "feature_extension_hash": FEATURE_EXTENSION_HASH, "generator_hash": GENERATOR_HASH,
                    "semantic_contract_hash": semantic_hash, "analytics_manifest_hash": analytics_manifest_hash,
                    "cohort_hash": cohort_hash, "source_path_refs": source_refs,
                    "protected_row_count": 0,
                }
                episode_rows.append(episode)
                branch_prefix = f"{attempt}_{'positive' if direction > 0 else 'negative'}"
                if onset_class == "efficient_progress":
                    event = {
                        **{k: episode[k] for k in ("parent_episode_id", "attempt", "symbol", "parent_direction", "feature_extension_hash", "generator_hash", "semantic_contract_hash", "analytics_manifest_hash", "cohort_hash", "source_path_refs", "protected_row_count")},
                        "branch_id": branch_prefix + "_efficient_continuation",
                        "event_type": "efficient_crowding_continuation", "trade_direction": direction,
                        "decision_ts": episode["parent_decision_ts"], "state_ts": episode["parent_onset_ts"],
                    }
                    event["event_id"], event["economic_address"] = deterministic_event_ids(event)
                    event_rows.append(event)
                if not impulse_complete or not deterioration_indices:
                    continue
                impulse = ordered.iloc[index:impulse_end+1]
                trade_low, trade_high = float(impulse.trade_low.min()), float(impulse.trade_high.max())
                mark_low, mark_high = float(impulse.mark_low.min()), float(impulse.mark_high.max())
                first_deterioration = min(deterioration_indices)
                search_start = max(impulse_end + 1, first_deterioration)
                failure_index: int | None = None
                for candidate in range(search_start, bounds.maximum + 1):
                    if not _contiguous(ts, index, candidate):
                        break
                    trade_close = float(ordered.trade_close.iloc[candidate])
                    mark_close = float(ordered.mark_close.iloc[candidate])
                    crossed = (trade_close < trade_low and mark_close < mark_low) if direction > 0 else (trade_close > trade_high and mark_close > mark_high)
                    if crossed:
                        failure_index = candidate
                        break
                if failure_index is not None:
                    event = {
                        **{k: episode[k] for k in ("parent_episode_id", "attempt", "symbol", "parent_direction", "feature_extension_hash", "generator_hash", "semantic_contract_hash", "analytics_manifest_hash", "cohort_hash", "source_path_refs", "protected_row_count")},
                        "branch_id": branch_prefix + "_completed_failure",
                        "event_type": "completed_structural_failure", "trade_direction": -direction,
                        "decision_ts": ts.iloc[failure_index] + pd.Timedelta(minutes=5), "state_ts": ts.iloc[failure_index],
                    }
                    event["event_id"], event["economic_address"] = deterministic_event_ids(event)
                    event_rows.append(event)
    episodes = pd.DataFrame(episode_rows)
    events = pd.DataFrame(event_rows)
    for result, identifier in ((episodes, "parent_episode_id"), (events, "event_id")):
        if not result.empty:
            validate_rankable_times(result.parent_decision_ts if "parent_decision_ts" in result else result.decision_ts)
            if result[identifier].duplicated().any():
                raise ValueError(f"duplicate {identifier}")
    if not events.empty and events.economic_address.duplicated().any():
        raise ValueError("duplicate economic address")
    return episodes, events
