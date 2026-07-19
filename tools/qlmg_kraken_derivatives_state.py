"""Causal, outcome-free Kraken derivatives-state semantics and identities."""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


SEMANTIC_STATUS = "inferred_authoritative_v1"
COHORT_VERSION = "current_roster_analytics_bar_existence_cohort_v1"
FEATURE_VERSION = "kda_shared_causal_features_v1_20260719"
TRAIN_START = pd.Timestamp("2023-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2026-01-01T00:00:00Z")
FORBIDDEN_OUTCOME_TOKENS = (
    "forward_return", "future_return", "net_r", "gross_r", "pnl", "profit",
    "mae", "mfe", "exit_price", "candidate_return", "control_return",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_semantic_decision(path: Path, *, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("semantic decision file hash mismatch")
    decision = json.loads(path.read_text(encoding="utf-8"))
    if decision.get("decision_id") != "DONCH_KRAKEN_ANALYTICS_SEMANTICS_20260719_V1":
        raise ValueError("unexpected analytics semantic decision version")
    if decision.get("status") != "project_authoritative_inference_v1":
        raise ValueError("semantic decision is not the approved inferred authority")
    for key in ("future_basis", "open_interest", "liquidation_volume"):
        if decision.get(key, {}).get("semantic_status") != SEMANTIC_STATUS:
            raise ValueError(f"{key} semantic status mismatch")
    expected = {
        "value_0_raw": "open", "value_1_raw": "high",
        "value_2_raw": "low", "value_3_raw": "close",
    }
    if decision["open_interest"].get("tuple_mapping") != expected:
        raise ValueError("open-interest OHLC tuple mapping mismatch")
    if decision["future_basis"].get("numerical_unit") != "decimal ratio":
        raise ValueError("future-basis decimal semantics mismatch")
    if decision["liquidation_volume"].get("directionality") != "unsigned aggregate; no native long/short side":
        raise ValueError("liquidation directionality mismatch")
    return decision, stable_hash(decision)


def decimal_text(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid exact decimal: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError("non-finite decimal")
    return parsed


def basis_fields(raw: Any) -> tuple[str, float, float, float]:
    value = decimal_text(raw)
    return str(raw), float(value), float(value * Decimal(100)), float(value * Decimal(10000))


def open_interest_fields(value_json: str) -> tuple[str, str, str, str, float, float, float, float]:
    raw = json.loads(value_json)
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("open-interest source value is not an OHLC tuple")
    exact = tuple(str(value) for value in raw)
    values = tuple(float(decimal_text(value)) for value in exact)
    if min(values) < 0 or values[1] < max(values[0], values[3]) or values[2] > min(values[0], values[3]) or values[1] < values[2]:
        raise ValueError("open-interest OHLC tuple violates structural inequalities")
    return (*exact, *values)


def liquidation_fields(raw: Any) -> tuple[str, float]:
    value = decimal_text(raw)
    if value < 0:
        raise ValueError("liquidation volume must be unsigned")
    return str(raw), float(value)


def price_inferred_liquidation_side(trade_return_1h: float) -> str:
    if not math.isfinite(trade_return_1h) or trade_return_1h == 0:
        return "ambiguous"
    return "long_liquidation_proxy" if trade_return_1h < 0 else "short_liquidation_proxy"


def assert_no_outcomes(columns: Iterable[str]) -> None:
    lowered = [str(column).lower() for column in columns]
    hits = sorted({token for column in lowered for token in FORBIDDEN_OUTCOME_TOKENS if token in column})
    if hits:
        raise ValueError(f"outcome fields prohibited in Stage 8A: {hits}")


def validate_rankable_times(values: Iterable[Any]) -> pd.DatetimeIndex:
    times = pd.to_datetime(list(values), utc=True, errors="raise")
    if (times < TRAIN_START).any() or (times >= PROTECTED_START).any():
        raise ValueError("non-rankable timestamp reached Stage 8A")
    return pd.DatetimeIndex(times)


def causal_daily_normalization(
    timestamps: pd.Series, values: pd.Series, *, lookback_days: int = 60,
    minimum_days: int = 30, minimum_fraction: float = 0.70,
    daily_aggregation: str = "median",
) -> pd.DataFrame:
    """Score each row against robust statistics frozen through the prior UTC day."""
    ts = pd.to_datetime(timestamps, utc=True, errors="raise")
    numeric = pd.to_numeric(values, errors="coerce")
    daily = pd.DataFrame({"day": ts.dt.floor("D"), "value": numeric}).dropna()
    if daily_aggregation not in {"median", "max"}:
        raise ValueError("unsupported daily normalization aggregation")
    grouped = daily.groupby("day", sort=True).value
    daily = grouped.median() if daily_aggregation == "median" else grouped.max()
    row_days = ts.dt.floor("D")
    day_indices = row_days.groupby(row_days, sort=True).groups
    days = pd.date_range(ts.min().floor("D"), ts.max().floor("D"), freq="D", tz="UTC")
    numeric_columns = [
        "prior_valid_days", "prior_expected_days", "prior_median", "prior_mad",
        "robust_z", "empirical_percentile",
    ]
    result = pd.DataFrame(np.nan, index=timestamps.index, columns=numeric_columns)
    result["normalization_valid"] = False
    result["normalization_stale_or_missing"] = True
    for day in days:
        history = daily[(daily.index < day) & (daily.index >= day - pd.Timedelta(days=lookback_days))]
        expected = min(lookback_days, max(0, (day - daily.index.min()).days)) if len(daily) else 0
        required = max(minimum_days, math.ceil(expected * minimum_fraction))
        valid = len(history) >= required and len(history) >= minimum_days
        median = float(history.median()) if valid else np.nan
        mad = float((history - median).abs().median()) if valid else np.nan
        scale_valid = valid and math.isfinite(mad) and mad > 0
        indices = day_indices.get(day, [])
        if not len(indices):
            continue
        current = numeric.loc[indices]
        result.loc[indices, [
            "prior_valid_days", "prior_expected_days", "prior_median", "prior_mad",
            "normalization_valid", "normalization_stale_or_missing",
        ]] = [len(history), expected, median, mad, scale_valid, not scale_valid]
        if scale_valid:
            result.loc[indices, "robust_z"] = (current - median) / (1.4826 * mad)
            finite = current.notna() & np.isfinite(current)
            ordered = np.sort(history.to_numpy(dtype=float))
            result.loc[current.index[finite], "empirical_percentile"] = (
                np.searchsorted(ordered, current.loc[finite].to_numpy(dtype=float), side="right") / len(ordered)
            )
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def exact_horizon_mask(timestamps: pd.Series, bars: int) -> pd.Series:
    """Require a complete five-minute horizon before using a lagged feature."""
    ts = pd.to_datetime(timestamps, utc=True, errors="raise")
    return ts.diff(bars).eq(pd.Timedelta(minutes=5 * bars))


def deterministic_event_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    required = (
        "family_id", "definition_id", "attempt_id", "symbol", "direction",
        "state_start", "decision_ts", "feature_window_start", "feature_window_end",
        "semantic_contract_hash", "analytics_data_manifest_hash",
        "trade_and_mark_authority_hashes", "cohort_version", "feature_version",
        "generator_contract_hash",
    )
    payload = {key: str(row[key]) for key in required}
    event_id = "kda_event_" + stable_hash(payload)[:32]
    address = "kda_addr_" + stable_hash({key: payload[key] for key in required if key not in {"attempt_id"}})[:32]
    return event_id, address


def cluster_canonical_episodes(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    ordered = events.sort_values(["symbol", "state_start", "feature_window_end", "event_id"], kind="mergesort").copy()
    episode_ids: dict[int, str] = {}
    sizes: dict[str, int] = {}
    for symbol, group in ordered.groupby("symbol", sort=True):
        members: list[int] = []
        end: pd.Timestamp | None = None
        for index, row in group.iterrows():
            start = pd.Timestamp(row.state_start)
            row_end = pd.Timestamp(row.feature_window_end)
            if end is None or start > end:
                if members:
                    identity = "kda_episode_" + stable_hash({"symbol": symbol, "events": sorted(ordered.loc[members, "event_id"])})[:32]
                    for member in members: episode_ids[member] = identity
                    sizes[identity] = len(members)
                members, end = [index], row_end
            else:
                members.append(index); end = max(end, row_end)
        if members:
            identity = "kda_episode_" + stable_hash({"symbol": symbol, "events": sorted(ordered.loc[members, "event_id"])})[:32]
            for member in members: episode_ids[member] = identity
            sizes[identity] = len(members)
    ordered["canonical_episode_id"] = pd.Series(episode_ids)
    ordered["canonical_episode_member_count"] = ordered.canonical_episode_id.map(sizes)
    return ordered.reset_index(drop=True)
