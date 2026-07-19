"""Outcome-free KDA02 v2 liquidation/OI purge features and episode identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tools.qlmg_kraken_derivatives_state import (
    causal_daily_normalization,
    stable_hash,
    validate_rankable_times,
)


TRANSLATION_ID = "KDA02A_v2_liquidation_oi_purge_state_machine"
INACTIVE_LINEAGE_ID = "KDA02B_oi_vacuum_without_liquidation_candidate_library"
FEATURE_EXTENSION_VERSION = "kda02_v2_feature_extension_v2_20260719"
GENERATOR_VERSION = "kda02_v2_episode_generator_v2_20260719"
ATTEMPTS = ("primary", "robustness")
DIRECTIONS = (-1, 1)

FEATURE_EXTENSION_CONTRACT: dict[str, Any] = {
    "translation_id": TRANSLATION_ID,
    "version": FEATURE_EXTENSION_VERSION,
    "grid": "completed_5m",
    "window": "three exact contiguous completed 5m bars ending at state_ts",
    "timestamp_validation": "every adjacent timestamp must be a distinct five-minute successor; duplicates fail closed",
    "raw_fields": {
        "trade_return_15m": "trade_close / trade_close.shift(3) - 1",
        "mark_return_15m": "mark_close / mark_close.shift(3) - 1",
        "liquidation_base_units_15m": "sum liquidation_base_units_5m over current and prior two bars",
        "liquidation_to_lagged_oi_15m": "liquidation_base_units_15m / oi_close immediately before window (shift 3)",
        "oi_log_change_15m": "log(oi_close / oi_close.shift(3))",
        "price_displacement_15m": "abs(trade_return_15m)",
    },
    "normalization": {
        "lookback_calendar_days": 60,
        "minimum_valid_days": 30,
        "minimum_expected_fraction": 0.70,
        "availability": "prior_UTC_day_only",
        "robust_z": "(value-prior_median)/(1.4826*prior_MAD)",
        "empirical_percentile": "right-inclusive against prior daily aggregates",
        "daily_aggregation": {
            "liquidation_to_lagged_oi_15m": "max",
            "oi_log_change_15m": "median",
            "price_displacement_15m": "median",
        },
        "zero_or_nonfinite_scale": "fail_closed",
    },
    "semantic_status": "inferred_authoritative_v1; liquidation side is price-inferred proxy only",
    "outcomes_authorized": False,
}
FEATURE_EXTENSION_HASH = stable_hash(FEATURE_EXTENSION_CONTRACT)

GENERATOR_CONTRACT: dict[str, Any] = {
    "translation_id": TRANSLATION_ID,
    "version": GENERATOR_VERSION,
    "feature_extension_hash": FEATURE_EXTENSION_HASH,
    "parent_reset": "12 exact completed bars / 60 minutes without the attempt-direction parent and no active same-state episode",
    "parent_primary": "aligned nonzero trade+mark direction; liquidation z>=2; OI z<=-2; displacement z>=1",
    "parent_robustness": "aligned nonzero trade+mark direction; liquidation pct>=.95; OI pct<=.05; displacement pct>=.75",
    "hysteresis_primary": "liquidation z>=1 OR OI z<=-1",
    "hysteresis_robustness": "liquidation pct>=.75 OR OI pct<=.25",
    "episode_end": "third consecutive completed bar outside both hysteresis conditions, capped so decision<=onset+6h",
    "initial_impulse": "first three completed bars beginning at onset",
    "continuation": "first trade+mark close beyond impulse extreme within 60m while active, liquidation above hysteresis, OI below pre-onset",
    "reversal": "first trade+mark close through onset opens after three liquidation-cooldown bars within 6h; cumulative OI reduction remains at least as deep as the parent-onset reset (current OI <= frozen onset OI)",
    "candidate_limit": "at most one continuation and one reversal per episode",
    "outcomes_authorized": False,
}
GENERATOR_HASH = stable_hash(GENERATOR_CONTRACT)


def strict_contiguous_mask(timestamps: pd.Series, bars: int) -> pd.Series:
    """Require every one of the prior ``bars`` transitions to be exactly five minutes."""
    ts = pd.to_datetime(timestamps, utc=True, errors="raise")
    steps = ts.diff().eq(pd.Timedelta(minutes=5))
    return steps.rolling(bars, min_periods=bars).sum().eq(bars)


def _exact_return(values: pd.Series, timestamps: pd.Series, bars: int = 3) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    lagged = numeric.shift(bars)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (numeric / lagged - 1).where(strict_contiguous_mask(timestamps, bars))
    return result.where((numeric > 0) & (lagged > 0) & np.isfinite(result))


def extend_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build exact 15-minute inputs and prior-day-only causal normalizations."""
    required = {
        "timestamp_utc", "trade_close", "mark_close", "oi_close_base_units",
        "liquidation_base_units_5m", "eligible", "known_lifecycle_mask",
        "trade_coverage", "mark_coverage", "analytics_coverage",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"missing KDA02 v2 feature inputs: {missing}")
    out = frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True).copy()
    out["timestamp_utc"] = pd.to_datetime(out.timestamp_utc, utc=True, errors="raise")
    validate_rankable_times(out.timestamp_utc)
    if out.timestamp_utc.duplicated().any():
        raise ValueError("duplicate timestamp in KDA02 v2 feature grid")
    exact = strict_contiguous_mask(out.timestamp_utc, 3)
    out["trade_return_15m"] = _exact_return(out.trade_close, out.timestamp_utc)
    out["mark_return_15m"] = _exact_return(out.mark_close, out.timestamp_utc)
    liquidation = pd.to_numeric(out.liquidation_base_units_5m, errors="coerce")
    out["liquidation_base_units_15m"] = liquidation.rolling(3, min_periods=3).sum().where(exact)
    oi = pd.to_numeric(out.oi_close_base_units, errors="coerce")
    lagged_oi = oi.shift(3)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["liquidation_to_lagged_oi_15m"] = (out.liquidation_base_units_15m / lagged_oi).where(exact & (lagged_oi > 0))
        out["oi_log_change_15m"] = np.log(oi.where(oi > 0) / lagged_oi.where(lagged_oi > 0)).where(exact)
    out["price_displacement_15m"] = out.trade_return_15m.abs()
    liq_stats = causal_daily_normalization(
        out.timestamp_utc, out.liquidation_to_lagged_oi_15m, daily_aggregation="max"
    )
    oi_stats = causal_daily_normalization(out.timestamp_utc, out.oi_log_change_15m)
    price_stats = causal_daily_normalization(out.timestamp_utc, out.price_displacement_15m)
    for prefix, stats in (("liquidation_intensity_15m", liq_stats), ("oi_change_15m", oi_stats), ("price_displacement_15m", price_stats)):
        out[f"{prefix}_robust_z"] = stats.robust_z
        out[f"{prefix}_percentile"] = stats.empirical_percentile
        out[f"{prefix}_normalization_valid"] = stats.normalization_valid
        out[f"{prefix}_normalization_valid_days"] = stats.prior_valid_days
    finite = (
        out[["trade_return_15m", "mark_return_15m", "liquidation_base_units_15m",
             "liquidation_to_lagged_oi_15m", "oi_log_change_15m",
             "price_displacement_15m"]].apply(np.isfinite).all(axis=1)
    )
    out["exact_contiguous_15m_valid"] = exact & finite & (lagged_oi > 0)
    out["pre_window_oi_close_base_units"] = lagged_oi.where(out.exact_contiguous_15m_valid)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def _base_valid(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.eligible.fillna(False)
        & frame.known_lifecycle_mask.fillna(False)
        & frame.trade_coverage.fillna(False)
        & frame.mark_coverage.fillna(False)
        & frame.analytics_coverage.fillna(False)
        & frame.exact_contiguous_15m_valid.fillna(False)
        & frame.liquidation_intensity_15m_normalization_valid.fillna(False)
        & frame.oi_change_15m_normalization_valid.fillna(False)
        & frame.price_displacement_15m_normalization_valid.fillna(False)
    )


def parent_mask(frame: pd.DataFrame, attempt: str, direction: int) -> pd.Series:
    if attempt not in ATTEMPTS or direction not in DIRECTIONS:
        raise ValueError("unsupported attempt or direction")
    aligned = (
        np.sign(pd.to_numeric(frame.trade_return_15m, errors="coerce")).eq(direction)
        & np.sign(pd.to_numeric(frame.mark_return_15m, errors="coerce")).eq(direction)
    )
    base = _base_valid(frame) & aligned
    if attempt == "primary":
        return base & (frame.liquidation_intensity_15m_robust_z >= 2) & (frame.oi_change_15m_robust_z <= -2) & (frame.price_displacement_15m_robust_z >= 1)
    return base & (frame.liquidation_intensity_15m_percentile >= .95) & (frame.oi_change_15m_percentile <= .05) & (frame.price_displacement_15m_percentile >= .75)


def liquidation_above_hysteresis(frame: pd.DataFrame, attempt: str) -> pd.Series:
    if attempt == "primary":
        return frame.liquidation_intensity_15m_normalization_valid.fillna(False) & (frame.liquidation_intensity_15m_robust_z >= 1)
    if attempt == "robustness":
        return frame.liquidation_intensity_15m_normalization_valid.fillna(False) & (frame.liquidation_intensity_15m_percentile >= .75)
    raise ValueError("unsupported attempt")


def hysteresis_mask(frame: pd.DataFrame, attempt: str) -> pd.Series:
    liq = liquidation_above_hysteresis(frame, attempt)
    if attempt == "primary":
        oi = frame.oi_change_15m_normalization_valid.fillna(False) & (frame.oi_change_15m_robust_z <= -1)
    elif attempt == "robustness":
        oi = frame.oi_change_15m_normalization_valid.fillna(False) & (frame.oi_change_15m_percentile <= .25)
    else:
        raise ValueError("unsupported attempt")
    return liq | oi


def _contiguous(ts: pd.Series, left: int, right: int) -> bool:
    if right < left:
        return False
    values = pd.DatetimeIndex(pd.to_datetime(ts.iloc[left:right + 1], utc=True, errors="raise"))
    return len(values) <= 1 or bool(((values[1:] - values[:-1]) == pd.Timedelta(minutes=5)).all())


def deterministic_episode_id(symbol: str, attempt: str, direction: int, onset_ts: Any) -> str:
    return "kda02v2_episode_" + stable_hash({
        "symbol": symbol, "attempt": attempt, "direction": direction,
        "onset_ts": pd.Timestamp(onset_ts).isoformat(), "generator_hash": GENERATOR_HASH,
    })[:32]


def deterministic_event_ids(row: Mapping[str, Any]) -> tuple[str, str]:
    payload = {key: str(row[key]) for key in (
        "branch_id", "attempt", "symbol", "parent_direction", "trade_direction",
        "parent_episode_id", "decision_ts", "feature_extension_hash", "generator_hash",
    )}
    return (
        "kda02v2_event_" + stable_hash(payload)[:32],
        "kda02v2_addr_" + stable_hash({**payload, "identity_type": "pre_exit_candidate"})[:32],
    )


@dataclass(frozen=True)
class EpisodeBounds:
    active_end: int
    maximum: int


def _episode_bounds(frame: pd.DataFrame, onset: int, attempt: str) -> EpisodeBounds:
    ts = frame.timestamp_utc
    hysteresis = hysteresis_mask(frame, attempt).fillna(False).to_numpy(dtype=bool)
    maximum = onset
    # Last state bar whose completion/decision is no later than onset + six hours.
    while maximum + 1 < len(frame) and maximum + 1 - onset <= 71 and _contiguous(ts, onset, maximum + 1):
        maximum += 1
    outside = 0
    active_end = maximum
    for index in range(onset, maximum + 1):
        outside = 0 if hysteresis[index] else outside + 1
        if outside >= 3:
            active_end = index
            break
    return EpisodeBounds(active_end=active_end, maximum=maximum)


def _event_base(episode: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "parent_episode_id", "attempt", "symbol", "parent_direction",
        "parent_onset_ts",
        "feature_extension_hash", "generator_hash", "semantic_contract_hash",
        "analytics_manifest_hash", "cohort_hash", "source_path_refs", "protected_row_count",
    )
    return {key: episode[key] for key in keys}


def generate_parent_episodes_and_events(
    frame: pd.DataFrame, *, symbol: str, semantic_hash: str,
    analytics_manifest_hash: str, cohort_hash: str, source_refs: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate complete episodes and at most one candidate per branch per episode."""
    required = {
        "trade_open", "trade_high", "trade_low", "trade_close",
        "mark_open", "mark_high", "mark_low", "mark_close", "oi_close_base_units",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"missing structural KDA02 OHLC/OI: {missing}")
    ordered = frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True).copy()
    ts = ordered.timestamp_utc
    episodes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for attempt in ATTEMPTS:
        for direction in DIRECTIONS:
            parent = parent_mask(ordered, attempt, direction).fillna(False).to_numpy(dtype=bool)
            last_active_end = -1
            for onset in np.flatnonzero(parent):
                if onset <= last_active_end or onset < 12 or not _contiguous(ts, onset - 12, onset):
                    continue
                if parent[onset - 12:onset].any():
                    continue
                bounds = _episode_bounds(ordered, onset, attempt)
                last_active_end = bounds.active_end
                episode_id = deterministic_episode_id(symbol, attempt, direction, ts.iloc[onset])
                impulse_end = onset + 2
                impulse_complete = impulse_end <= bounds.maximum and _contiguous(ts, onset, impulse_end)
                pre_onset_oi = float(ordered.pre_window_oi_close_base_units.iloc[onset])
                episode = {
                    "translation_id": TRANSLATION_ID,
                    "parent_episode_id": episode_id,
                    "attempt": attempt,
                    "symbol": symbol,
                    "parent_direction": direction,
                    "price_inferred_liquidation_side": "long_liquidation_proxy" if direction < 0 else "short_liquidation_proxy",
                    "parent_onset_ts": ts.iloc[onset],
                    "parent_decision_ts": ts.iloc[onset] + pd.Timedelta(minutes=5),
                    "episode_active_end_ts": ts.iloc[bounds.active_end] + pd.Timedelta(minutes=5),
                    "maximum_deadline_ts": ts.iloc[onset] + pd.Timedelta(hours=6),
                    "pre_onset_oi_close_base_units": pre_onset_oi,
                    "onset_oi_close_base_units": float(ordered.oi_close_base_units.iloc[onset]),
                    "onset_trade_open": float(ordered.trade_open.iloc[onset]),
                    "onset_mark_open": float(ordered.mark_open.iloc[onset]),
                    "impulse_complete": bool(impulse_complete),
                    "feature_extension_hash": FEATURE_EXTENSION_HASH,
                    "generator_hash": GENERATOR_HASH,
                    "semantic_contract_hash": semantic_hash,
                    "analytics_manifest_hash": analytics_manifest_hash,
                    "cohort_hash": cohort_hash,
                    "source_path_refs": source_refs,
                    "protected_row_count": 0,
                }
                episodes.append(episode)
                branch_prefix = f"{attempt}_{'negative' if direction < 0 else 'positive'}"
                if impulse_complete:
                    impulse = ordered.iloc[onset:impulse_end + 1]
                    trade_extreme = float(impulse.trade_low.min() if direction < 0 else impulse.trade_high.max())
                    mark_extreme = float(impulse.mark_low.min() if direction < 0 else impulse.mark_high.max())
                    confirmation_limit = min(bounds.active_end, onset + 11)
                    for index in range(impulse_end + 1, confirmation_limit + 1):
                        if not _contiguous(ts, onset, index):
                            break
                        trade_close = float(ordered.trade_close.iloc[index])
                        mark_close = float(ordered.mark_close.iloc[index])
                        broke = (trade_close < trade_extreme and mark_close < mark_extreme) if direction < 0 else (trade_close > trade_extreme and mark_close > mark_extreme)
                        oi_below = float(ordered.oi_close_base_units.iloc[index]) < pre_onset_oi
                        liq_active = bool(liquidation_above_hysteresis(ordered.iloc[[index]], attempt).iloc[0])
                        if broke and oi_below and liq_active:
                            event = {
                                **_event_base(episode),
                                "branch_id": branch_prefix + "_active_purge_continuation",
                                "event_type": "active_purge_continuation",
                                "trade_direction": direction,
                                "state_ts": ts.iloc[index],
                                "decision_ts": ts.iloc[index] + pd.Timedelta(minutes=5),
                            }
                            event["event_id"], event["economic_address"] = deterministic_event_ids(event)
                            events.append(event)
                            break
                liq_active = liquidation_above_hysteresis(ordered, attempt).fillna(False).to_numpy(dtype=bool)
                cooldown = 0
                for index in range(onset, bounds.maximum + 1):
                    if not _contiguous(ts, onset, index):
                        break
                    cooldown = 0 if liq_active[index] else cooldown + 1
                    if cooldown < 3:
                        continue
                    trade_close = float(ordered.trade_close.iloc[index])
                    mark_close = float(ordered.mark_close.iloc[index])
                    reclaimed = (trade_close > episode["onset_trade_open"] and mark_close > episode["onset_mark_open"]) if direction < 0 else (trade_close < episode["onset_trade_open"] and mark_close < episode["onset_mark_open"])
                    # Retain at least the full parent-onset reset; equality is the frozen boundary.
                    cumulative_oi_material = float(ordered.oi_close_base_units.iloc[index]) <= episode["onset_oi_close_base_units"]
                    if reclaimed and cumulative_oi_material:
                        event = {
                            **_event_base(episode),
                            "branch_id": branch_prefix + "_completed_purge_reversal",
                            "event_type": "completed_purge_reversal",
                            "trade_direction": -direction,
                            "state_ts": ts.iloc[index],
                            "decision_ts": ts.iloc[index] + pd.Timedelta(minutes=5),
                        }
                        event["event_id"], event["economic_address"] = deterministic_event_ids(event)
                        events.append(event)
                        break
    episode_frame = pd.DataFrame(episodes)
    event_frame = pd.DataFrame(events)
    if episode_frame.empty:
        episode_frame = pd.DataFrame(columns=[
            "translation_id", "parent_episode_id", "attempt", "symbol",
            "parent_direction", "price_inferred_liquidation_side", "parent_onset_ts",
            "parent_decision_ts", "episode_active_end_ts", "maximum_deadline_ts",
            "pre_onset_oi_close_base_units", "onset_oi_close_base_units",
            "onset_trade_open", "onset_mark_open", "impulse_complete",
            "feature_extension_hash", "generator_hash", "semantic_contract_hash",
            "analytics_manifest_hash", "cohort_hash", "source_path_refs",
            "protected_row_count",
        ])
    if event_frame.empty:
        event_frame = pd.DataFrame(columns=[
            "parent_episode_id", "attempt", "symbol", "parent_direction",
            "parent_onset_ts",
            "feature_extension_hash", "generator_hash", "semantic_contract_hash",
            "analytics_manifest_hash", "cohort_hash", "source_path_refs",
            "protected_row_count", "branch_id", "event_type", "trade_direction",
            "state_ts", "decision_ts", "event_id", "economic_address",
        ])
    if not episode_frame.empty:
        validate_rankable_times(episode_frame.parent_decision_ts)
        if episode_frame.parent_episode_id.duplicated().any():
            raise ValueError("duplicate KDA02 v2 parent episode ID")
    if not event_frame.empty:
        validate_rankable_times(event_frame.decision_ts)
        if event_frame.event_id.duplicated().any() or event_frame.economic_address.duplicated().any():
            raise ValueError("duplicate KDA02 v2 event identity")
        if event_frame.groupby(["parent_episode_id", "event_type"]).size().gt(1).any():
            raise ValueError("more than one KDA02 v2 branch candidate per episode")
    return episode_frame, event_frame
