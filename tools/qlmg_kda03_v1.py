"""Outcome-free causal KDA03 basis-shock features and episode identities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tools.qlmg_kraken_derivatives_state import (
    causal_daily_normalization,
    stable_hash,
    validate_rankable_times,
)


TRANSLATION_ID = "KDA03_basis_shock_v1"
FEATURE_EXTENSION_VERSION = "kda03_feature_extension_v1_20260719"
GENERATOR_VERSION = "kda03_episode_generator_v1_20260719"
ATTEMPTS = ("primary", "robustness")
DIRECTIONS = (-1, 1)
PARENT_KINDS = ("catchup", "impulse")

FEATURE_EXTENSION_CONTRACT: dict[str, Any] = {
    "translation_id": TRANSLATION_ID,
    "version": FEATURE_EXTENSION_VERSION,
    "grid": "completed_5m_interval_start_timestamps",
    "window": "three exact contiguous bars ending at state_ts; pre-window state is shift(3)",
    "decision": "state_ts plus five minutes",
    "raw_fields": {
        "basis_change_15m": "basis_decimal - basis_decimal.shift(3)",
        "prior_basis_level": "basis_decimal.shift(3)",
        "trade_return_15m": "trade_close / trade_open.shift(2) - 1",
        "mark_return_15m": "mark_close / mark_open.shift(2) - 1",
        "oi_log_change_15m": "log(oi_close / oi_close.shift(3))",
        "liquidation_to_lagged_oi_15m": "sum(current and prior two liquidation rows) / oi_close.shift(3)",
    },
    "normalization": {
        "lookback_calendar_days": 60,
        "minimum_valid_days": 30,
        "minimum_expected_fraction": 0.70,
        "availability": "prior_UTC_day_only",
        "prior_observation_distribution": ["basis_change_15m"],
        "daily_median": [
            "prior_basis_level", "trade_abs_displacement_15m",
            "mark_abs_displacement_15m", "oi_log_change_15m",
        ],
        "daily_max": ["liquidation_to_lagged_oi_15m"],
        "robust_z": "(value-prior_median)/(1.4826*prior_MAD)",
        "empirical_percentile": "right-inclusive against prior daily aggregates",
        "zero_or_nonfinite_scale": "fail_closed",
    },
    "semantic_status": "basis inferred_authoritative_v1 signed decimal; positive=futures_above_reference",
    "outcomes_authorized": False,
}
FEATURE_EXTENSION_HASH = stable_hash(FEATURE_EXTENSION_CONTRACT)

GENERATOR_CONTRACT: dict[str, Any] = {
    "translation_id": TRANSLATION_ID,
    "version": GENERATOR_VERSION,
    "feature_extension_hash": FEATURE_EXTENSION_HASH,
    "parents": {
        "KDA03A_primary": "abs basis-change z>=2; trade/mark displacement pct<=.25; abs OI z<=1; liquidation z<1",
        "KDA03A_robustness": "signed basis-change pct>=.95 or <=.05; remaining KDA03A rules unchanged",
        "KDA03BC_primary": "abs basis-change z>=2; trade/mark signs equal shock; displacement pct>=.75; OI z>=2; abs prior-basis z<1.5; liquidation z<2",
        "KDA03BC_robustness": "signed basis-change pct>=.95 or <=.05; OI pct>=.95; remaining KDA03BC rules unchanged",
    },
    "episode_reset": "twelve exact completed bars / 60 minutes without the same attempt-kind-direction parent",
    "episode_end": "six consecutive completed bars / 30 minutes outside the same parent, or six hours",
    "immediate_candidates": {
        "KDA03A": "opposite shock direction at completed-parent availability",
        "KDA03B": "same shock direction at completed-parent availability",
    },
    "rejection_candidate": "first completed bar in the impulse episode where basis crosses the frozen pre-window level and both trade and mark closes cross the frozen shock-window opens; opposite shock direction",
    "candidate_limit": "one immediate and at most one rejection candidate per parent episode",
    "outcomes_authorized": False,
}
GENERATOR_HASH = stable_hash(GENERATOR_CONTRACT)


def strict_contiguous_mask(timestamps: pd.Series, transitions: int) -> pd.Series:
    """Require every prior transition to be an exact distinct five-minute successor."""
    ts = pd.to_datetime(timestamps, utc=True, errors="raise")
    steps = ts.diff().eq(pd.Timedelta(minutes=5))
    return steps.rolling(transitions, min_periods=transitions).sum().eq(transitions)


def _normalize(out: pd.DataFrame, source: str, prefix: str, aggregation: str = "median") -> None:
    stats = causal_daily_normalization(
        out.timestamp_utc, out[source], daily_aggregation=aggregation
    )
    out[f"{prefix}_robust_z"] = stats.robust_z
    out[f"{prefix}_percentile"] = stats.empirical_percentile
    out[f"{prefix}_normalization_valid"] = stats.normalization_valid
    out[f"{prefix}_normalization_valid_days"] = stats.prior_valid_days


def _normalize_prior_observations(out: pd.DataFrame, source: str, prefix: str) -> None:
    """Score against all valid observations from prior UTC days in the 60-day window."""
    ts = pd.to_datetime(out.timestamp_utc, utc=True, errors="raise")
    values = pd.to_numeric(out[source], errors="coerce")
    days = ts.dt.floor("D")
    grouped = {
        day: group[np.isfinite(group)].to_numpy(dtype=float)
        for day, group in values.groupby(days, sort=True)
        if np.isfinite(group).any()
    }
    result = pd.DataFrame(index=out.index)
    result["robust_z"] = np.nan
    result["empirical_percentile"] = np.nan
    result["normalization_valid"] = False
    result["prior_valid_days"] = np.nan
    if not grouped:
        out[f"{prefix}_robust_z"] = result.robust_z
        out[f"{prefix}_percentile"] = result.empirical_percentile
        out[f"{prefix}_normalization_valid"] = result.normalization_valid
        out[f"{prefix}_normalization_valid_days"] = result.prior_valid_days
        return
    first = min(grouped)
    for day, indices in days.groupby(days, sort=True).groups.items():
        history_days = [key for key in grouped if day - pd.Timedelta(days=60) <= key < day]
        expected = min(60, max(0, (day - first).days))
        required = max(30, int(np.ceil(expected * .70)))
        result.loc[indices, "prior_valid_days"] = len(history_days)
        if len(history_days) < required or len(history_days) < 30:
            continue
        history = np.concatenate([grouped[key] for key in sorted(history_days)])
        median = float(np.median(history))
        mad = float(np.median(np.abs(history - median)))
        if not np.isfinite(mad) or mad <= 0:
            continue
        current = values.loc[indices]
        finite = current.notna() & np.isfinite(current)
        ordered = np.sort(history)
        result.loc[indices, "normalization_valid"] = True
        result.loc[current.index[finite], "robust_z"] = (current.loc[finite] - median) / (1.4826 * mad)
        result.loc[current.index[finite], "empirical_percentile"] = np.searchsorted(
            ordered, current.loc[finite].to_numpy(dtype=float), side="right"
        ) / len(ordered)
    out[f"{prefix}_robust_z"] = result.robust_z
    out[f"{prefix}_percentile"] = result.empirical_percentile
    out[f"{prefix}_normalization_valid"] = result.normalization_valid
    out[f"{prefix}_normalization_valid_days"] = result.prior_valid_days


def extend_causal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Construct exact 15-minute KDA03 inputs using prior-day-only normalizers."""
    required = {
        "timestamp_utc", "basis_decimal", "trade_open", "trade_close", "mark_open",
        "mark_close", "oi_close_base_units", "liquidation_base_units_5m", "eligible",
        "known_lifecycle_mask", "trade_coverage", "mark_coverage", "analytics_coverage",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"missing KDA03 feature inputs: {missing}")
    out = frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True).copy()
    out["timestamp_utc"] = pd.to_datetime(out.timestamp_utc, utc=True, errors="raise")
    validate_rankable_times(out.timestamp_utc)
    if out.timestamp_utc.duplicated().any():
        raise ValueError("duplicate timestamp in KDA03 feature grid")
    exact = strict_contiguous_mask(out.timestamp_utc, 3)
    basis = pd.to_numeric(out.basis_decimal, errors="coerce")
    oi = pd.to_numeric(out.oi_close_base_units, errors="coerce")
    lagged_oi = oi.shift(3)
    trade_open = pd.to_numeric(out.trade_open, errors="coerce").shift(2)
    mark_open = pd.to_numeric(out.mark_open, errors="coerce").shift(2)
    trade_close = pd.to_numeric(out.trade_close, errors="coerce")
    mark_close = pd.to_numeric(out.mark_close, errors="coerce")
    out["basis_change_15m"] = (basis - basis.shift(3)).where(exact)
    out["prior_basis_level"] = basis.shift(3).where(exact)
    out["onset_trade_open"] = trade_open.where(exact & (trade_open > 0))
    out["onset_mark_open"] = mark_open.where(exact & (mark_open > 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["trade_return_15m"] = (trade_close / trade_open - 1).where(
            exact & (trade_close > 0) & (trade_open > 0)
        )
        out["mark_return_15m"] = (mark_close / mark_open - 1).where(
            exact & (mark_close > 0) & (mark_open > 0)
        )
        out["oi_log_change_15m"] = np.log(oi.where(oi > 0) / lagged_oi.where(lagged_oi > 0)).where(exact)
    liquidation = pd.to_numeric(out.liquidation_base_units_5m, errors="coerce")
    out["liquidation_base_units_15m"] = liquidation.rolling(3, min_periods=3).sum().where(exact)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["liquidation_to_lagged_oi_15m"] = (
            out.liquidation_base_units_15m / lagged_oi
        ).where(exact & (lagged_oi > 0))
    out["trade_abs_displacement_15m"] = out.trade_return_15m.abs()
    out["mark_abs_displacement_15m"] = out.mark_return_15m.abs()
    _normalize_prior_observations(out, "basis_change_15m", "basis_change_15m")
    for source, prefix, aggregation in (
        ("prior_basis_level", "prior_basis_level", "median"),
        ("trade_abs_displacement_15m", "trade_displacement_15m", "median"),
        ("mark_abs_displacement_15m", "mark_displacement_15m", "median"),
        ("oi_log_change_15m", "oi_change_15m", "median"),
        ("liquidation_to_lagged_oi_15m", "liquidation_intensity_15m", "max"),
    ):
        _normalize(out, source, prefix, aggregation)
    finite = out[[
        "basis_change_15m", "prior_basis_level", "trade_return_15m", "mark_return_15m",
        "oi_log_change_15m", "liquidation_to_lagged_oi_15m", "onset_trade_open",
        "onset_mark_open",
    ]].apply(np.isfinite).all(axis=1)
    out["exact_contiguous_15m_valid"] = exact & finite & (lagged_oi > 0)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def _base_valid(frame: pd.DataFrame) -> pd.Series:
    validity = [
        "basis_change_15m_normalization_valid", "prior_basis_level_normalization_valid",
        "trade_displacement_15m_normalization_valid", "mark_displacement_15m_normalization_valid",
        "oi_change_15m_normalization_valid", "liquidation_intensity_15m_normalization_valid",
    ]
    return (
        frame.eligible.fillna(False)
        & frame.known_lifecycle_mask.fillna(False)
        & frame.trade_coverage.fillna(False)
        & frame.mark_coverage.fillna(False)
        & frame.analytics_coverage.fillna(False)
        & frame.exact_contiguous_15m_valid.fillna(False)
        & frame[validity].fillna(False).all(axis=1)
    )


def _signed_basis_shock(frame: pd.DataFrame, attempt: str, direction: int) -> pd.Series:
    if attempt == "primary":
        return (
            np.sign(pd.to_numeric(frame.basis_change_15m, errors="coerce")).eq(direction)
            & (frame.basis_change_15m_robust_z.abs() >= 2)
        )
    if attempt == "robustness":
        threshold = frame.basis_change_15m_percentile >= .95 if direction == 1 else frame.basis_change_15m_percentile <= .05
        return np.sign(pd.to_numeric(frame.basis_change_15m, errors="coerce")).eq(direction) & threshold
    raise ValueError("unsupported KDA03 attempt")


def parent_mask(frame: pd.DataFrame, attempt: str, parent_kind: str, direction: int) -> pd.Series:
    """Return the exact frozen KDA03A or KDA03B/C parent state."""
    if attempt not in ATTEMPTS or parent_kind not in PARENT_KINDS or direction not in DIRECTIONS:
        raise ValueError("unsupported KDA03 parent coordinates")
    base = _base_valid(frame) & _signed_basis_shock(frame, attempt, direction)
    if parent_kind == "catchup":
        return (
            base
            & (frame.trade_displacement_15m_percentile <= .25)
            & (frame.mark_displacement_15m_percentile <= .25)
            & (frame.oi_change_15m_robust_z.abs() <= 1)
            & (frame.liquidation_intensity_15m_robust_z < 1)
        )
    price_aligned = (
        np.sign(pd.to_numeric(frame.trade_return_15m, errors="coerce")).eq(direction)
        & np.sign(pd.to_numeric(frame.mark_return_15m, errors="coerce")).eq(direction)
    )
    oi_rule = frame.oi_change_15m_robust_z >= 2 if attempt == "primary" else frame.oi_change_15m_percentile >= .95
    return (
        base
        & price_aligned
        & (frame.trade_displacement_15m_percentile >= .75)
        & (frame.mark_displacement_15m_percentile >= .75)
        & oi_rule
        & (frame.prior_basis_level_robust_z.abs() < 1.5)
        & (frame.liquidation_intensity_15m_robust_z < 2)
    )


def deterministic_episode_id(symbol: str, attempt: str, parent_kind: str, direction: int, onset_ts: Any) -> str:
    return "kda03_episode_" + stable_hash({
        "translation_id": TRANSLATION_ID, "symbol": symbol, "attempt": attempt,
        "parent_kind": parent_kind, "direction": direction,
        "onset_ts": pd.Timestamp(onset_ts).isoformat(),
    })[:32]


def _event_row(
    *, symbol: str, attempt: str, parent_kind: str, direction: int, mechanism: str,
    parent_episode_id: str, parent_onset_ts: pd.Timestamp, state_ts: pd.Timestamp,
    trade_direction: int, semantic_hash: str, analytics_manifest_hash: str,
    cohort_hash: str, source_refs: str, frozen: dict[str, Any],
) -> dict[str, Any]:
    direction_name = "positive" if direction == 1 else "negative"
    branch_id = f"{attempt}_{direction_name}_{mechanism}"
    payload = {
        "translation_id": TRANSLATION_ID, "symbol": symbol, "attempt": attempt,
        "parent_kind": parent_kind, "parent_direction": direction, "mechanism": mechanism,
        "parent_episode_id": parent_episode_id, "parent_onset_ts": parent_onset_ts.isoformat(),
        "state_ts": state_ts.isoformat(), "decision_ts": (state_ts + pd.Timedelta(minutes=5)).isoformat(),
        "trade_direction": trade_direction, "semantic_hash": semantic_hash,
        "analytics_manifest_hash": analytics_manifest_hash, "cohort_hash": cohort_hash,
        "source_refs": source_refs, "feature_hash": FEATURE_EXTENSION_HASH,
        "generator_hash": GENERATOR_HASH,
    }
    event_id = "kda03_event_" + stable_hash(payload)[:32]
    economic_address = "kda03_addr_" + stable_hash({**payload, "branch_id": branch_id})[:32]
    return {
        "event_id": event_id, "economic_address": economic_address, "branch_id": branch_id,
        "attempt": attempt, "symbol": symbol, "parent_kind": parent_kind,
        "parent_direction": direction, "mechanism": mechanism, "trade_direction": trade_direction,
        "parent_episode_id": parent_episode_id, "parent_onset_ts": parent_onset_ts,
        "state_ts": state_ts, "decision_ts": state_ts + pd.Timedelta(minutes=5),
        "semantic_contract_hash": semantic_hash, "analytics_manifest_hash": analytics_manifest_hash,
        "cohort_hash": cohort_hash, "source_refs": source_refs,
        "feature_extension_hash": FEATURE_EXTENSION_HASH, "generator_hash": GENERATOR_HASH,
        **frozen,
    }


def _quiet_parent(mask: pd.Series, timestamps: pd.Series, index: int) -> bool:
    if index < 12 or bool(mask.iloc[index - 12:index].any()):
        return False
    segment = pd.DatetimeIndex(timestamps.iloc[index - 12:index + 1])
    return bool(((segment[1:] - segment[:-1]) == pd.Timedelta(minutes=5)).all())


def _episode_end(mask: pd.Series, timestamps: pd.Series, onset: int) -> int:
    cap_ts = pd.Timestamp(timestamps.iloc[onset]) + pd.Timedelta(hours=6) - pd.Timedelta(minutes=5)
    outside = 0
    end = onset
    for index in range(onset + 1, len(mask)):
        ts = pd.Timestamp(timestamps.iloc[index])
        if ts - pd.Timestamp(timestamps.iloc[index - 1]) != pd.Timedelta(minutes=5) or ts > cap_ts:
            break
        end = index
        outside = 0 if bool(mask.iloc[index]) else outside + 1
        if outside >= 6:
            break
    return end


def generate_parent_episodes_and_events(
    frame: pd.DataFrame, *, symbol: str, semantic_hash: str,
    analytics_manifest_hash: str, cohort_hash: str, source_refs: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate parent-neutral KDA03A/B/C episodes and candidates without outcomes."""
    required = {
        "timestamp_utc", "basis_decimal", "trade_close", "mark_close", "prior_basis_level",
        "onset_trade_open", "onset_mark_open",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"missing KDA03 generator inputs: {missing}")
    work = frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    timestamps = pd.to_datetime(work.timestamp_utc, utc=True, errors="raise")
    validate_rankable_times(timestamps)
    episodes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for attempt in ATTEMPTS:
        for parent_kind in PARENT_KINDS:
            for direction in DIRECTIONS:
                mask = parent_mask(work, attempt, parent_kind, direction).fillna(False)
                for onset in [index for index in range(len(work)) if bool(mask.iloc[index]) and _quiet_parent(mask, timestamps, index)]:
                    onset_ts = pd.Timestamp(timestamps.iloc[onset])
                    end = _episode_end(mask, timestamps, onset)
                    episode_id = deterministic_episode_id(symbol, attempt, parent_kind, direction, onset_ts)
                    frozen = {
                        "pre_shock_basis": float(work.at[onset, "prior_basis_level"]),
                        "onset_trade_open": float(work.at[onset, "onset_trade_open"]),
                        "onset_mark_open": float(work.at[onset, "onset_mark_open"]),
                        "onset_basis_change_15m": float(work.at[onset, "basis_change_15m"]),
                    }
                    episode_events = 1
                    immediate = "reference_led_catchup" if parent_kind == "catchup" else "basis_impulse_continuation"
                    immediate_direction = -direction if parent_kind == "catchup" else direction
                    events.append(_event_row(
                        symbol=symbol, attempt=attempt, parent_kind=parent_kind, direction=direction,
                        mechanism=immediate, parent_episode_id=episode_id, parent_onset_ts=onset_ts,
                        state_ts=onset_ts, trade_direction=immediate_direction,
                        semantic_hash=semantic_hash, analytics_manifest_hash=analytics_manifest_hash,
                        cohort_hash=cohort_hash, source_refs=source_refs, frozen=frozen,
                    ))
                    if parent_kind == "impulse":
                        rejection = None
                        for index in range(onset + 1, end + 1):
                            basis = float(work.at[index, "basis_decimal"])
                            trade = float(work.at[index, "trade_close"])
                            mark = float(work.at[index, "mark_close"])
                            if direction == 1:
                                confirmed = basis <= frozen["pre_shock_basis"] and trade < frozen["onset_trade_open"] and mark < frozen["onset_mark_open"]
                            else:
                                confirmed = basis >= frozen["pre_shock_basis"] and trade > frozen["onset_trade_open"] and mark > frozen["onset_mark_open"]
                            if confirmed:
                                rejection = index
                                break
                        if rejection is not None:
                            episode_events += 1
                            events.append(_event_row(
                                symbol=symbol, attempt=attempt, parent_kind=parent_kind, direction=direction,
                                mechanism="completed_basis_impulse_rejection", parent_episode_id=episode_id,
                                parent_onset_ts=onset_ts, state_ts=pd.Timestamp(timestamps.iloc[rejection]),
                                trade_direction=-direction, semantic_hash=semantic_hash,
                                analytics_manifest_hash=analytics_manifest_hash, cohort_hash=cohort_hash,
                                source_refs=source_refs, frozen=frozen,
                            ))
                    episodes.append({
                        "parent_episode_id": episode_id, "symbol": symbol, "attempt": attempt,
                        "parent_kind": parent_kind, "parent_direction": direction,
                        "parent_onset_ts": onset_ts, "parent_decision_ts": onset_ts + pd.Timedelta(minutes=5),
                        "episode_active_end_ts": pd.Timestamp(timestamps.iloc[end]) + pd.Timedelta(minutes=5),
                        "episode_candidate_count": episode_events,
                        "rejection_candidate_count": max(0, episode_events - 1),
                        **frozen,
                    })
    episode_columns = [
        "parent_episode_id", "symbol", "attempt", "parent_kind", "parent_direction",
        "parent_onset_ts", "parent_decision_ts", "episode_active_end_ts",
        "episode_candidate_count", "rejection_candidate_count", "pre_shock_basis",
        "onset_trade_open", "onset_mark_open", "onset_basis_change_15m",
    ]
    event_columns = [
        "event_id", "economic_address", "branch_id", "attempt", "symbol", "parent_kind",
        "parent_direction", "mechanism", "trade_direction", "parent_episode_id",
        "parent_onset_ts", "state_ts", "decision_ts", "semantic_contract_hash",
        "analytics_manifest_hash", "cohort_hash", "source_refs", "feature_extension_hash",
        "generator_hash", "pre_shock_basis", "onset_trade_open", "onset_mark_open",
        "onset_basis_change_15m",
    ]
    episode_frame = pd.DataFrame(episodes, columns=episode_columns).sort_values(
        ["symbol", "parent_onset_ts", "attempt", "parent_kind", "parent_direction"], kind="mergesort"
    ).reset_index(drop=True)
    event_frame = pd.DataFrame(events, columns=event_columns).sort_values(
        ["symbol", "decision_ts", "branch_id", "event_id"], kind="mergesort"
    ).reset_index(drop=True)
    if episode_frame.parent_episode_id.duplicated().any() or event_frame.event_id.duplicated().any() or event_frame.economic_address.duplicated().any():
        raise ValueError("duplicate KDA03 episode/event/economic identity")
    return episode_frame, event_frame


__all__ = [
    "ATTEMPTS", "DIRECTIONS", "FEATURE_EXTENSION_CONTRACT", "FEATURE_EXTENSION_HASH",
    "GENERATOR_CONTRACT", "GENERATOR_HASH", "TRANSLATION_ID", "deterministic_episode_id",
    "extend_causal_features", "generate_parent_episodes_and_events", "parent_mask",
    "strict_contiguous_mask",
]
