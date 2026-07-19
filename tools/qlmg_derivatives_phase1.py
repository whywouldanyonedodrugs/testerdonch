"""Outcome-free primitives for Stage 14 derivatives Phase-1 closure.

This module deliberately accepts only a named contemporaneous feature allow-list.
It contains no execution, forward-label, or economic-result reader.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROTECTED_START = pd.Timestamp("2026-01-01T00:00:00Z")
FIVE_MINUTES = pd.Timedelta(minutes=5)

FEATURE_COLUMNS = (
    "timestamp_utc", "trade_close", "basis_decimal", "basis_bps",
    "oi_close_base_units", "oi_log_change_1h", "trade_log_return_5m",
    "trade_return_1h", "mark_return_1h", "realized_vol_24h",
    "liquidation_base_units_1h", "liquidation_intensity_robust_z",
    "liquidation_intensity_percentile", "basis_level_robust_z",
    "basis_change_robust_z", "basis_level_normalization_valid",
    "basis_change_normalization_valid", "liquidation_normalization_valid",
    "trade_coverage", "mark_coverage", "analytics_coverage", "eligible",
    "known_lifecycle_mask", "rank", "major_vs_alt",
)

PARENT_EVENT_COLUMNS = (
    "symbol", "state_ts", "decision_ts", "event_id", "economic_address",
    "branch_id",
)
FEATURE_ROOT = Path("/opt/parquet/kraken_derivatives/analytics/stage8a_foundation_v1_exact/features")
PARENT_ROOTS = tuple(Path(x) for x in (
    "/opt/parquet/kraken_derivatives/analytics/stage8b_kda01_v2_prerun_v1_final",
    "/opt/parquet/kraken_derivatives/analytics/stage9_kda02_v2_prerun_v4",
    "/opt/parquet/kraken_derivatives/analytics/stage11_kda03_v1",
))


class OutcomeReadSpy:
    """Records every parquet request and rejects anything outside its allow-list."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def read(self, path: Path, columns: Iterable[str], *, kind: str) -> pd.DataFrame:
        columns = tuple(columns)
        if kind not in {"feature", "parent"}:
            raise ValueError(f"outcome firewall rejected unknown reader kind: {kind}")
        allowed = FEATURE_COLUMNS if kind == "feature" else PARENT_EVENT_COLUMNS
        resolved = path.resolve()
        roots = (FEATURE_ROOT,) if kind == "feature" else PARENT_ROOTS
        if not any(resolved.is_relative_to(root.resolve()) for root in roots):
            raise ValueError(f"outcome firewall rejected unauthorized {kind} path: {resolved}")
        unknown = sorted(set(columns) - set(allowed))
        if unknown:
            raise ValueError(f"outcome firewall rejected {kind} columns: {unknown}")
        parquet = pq.ParquetFile(resolved)
        time_column = "timestamp_utc" if kind == "feature" else "decision_ts"
        column_index = parquet.schema_arrow.names.index(time_column)
        for row_group in range(parquet.metadata.num_row_groups):
            stats = parquet.metadata.row_group(row_group).column(column_index).statistics
            if parquet.metadata.row_group(row_group).num_rows == 0:
                continue
            if stats is None or not stats.has_min_max:
                raise ValueError(f"outcome firewall lacks pre-read timestamp statistics: {resolved}")
            maximum=pd.Timestamp(stats.max)
            maximum=maximum.tz_localize("UTC") if maximum.tzinfo is None else maximum.tz_convert("UTC")
            if maximum >= PROTECTED_START:
                raise ValueError(f"protected row rejected before {kind} payload read")
        self.requests.append({"path": str(resolved), "kind": kind, "columns": list(columns), "pre_read_max_timestamp_check": "pass"})
        frame = pd.read_parquet(resolved, columns=list(columns))
        if time_column in frame:
            times = pd.to_datetime(frame[time_column], utc=True, errors="raise")
            if (times >= PROTECTED_START).any():
                raise ValueError(f"protected row reached {kind} reader")
        return frame


def stable_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def strict_base_valid(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["eligible"].fillna(False).astype(bool)
        & frame["known_lifecycle_mask"].fillna(False).astype(bool)
        & frame["trade_coverage"].fillna(False).astype(bool)
        & frame["mark_coverage"].fillna(False).astype(bool)
        & frame["analytics_coverage"].fillna(False).astype(bool)
    )


def causal_oi_normalization(timestamps: pd.Series, values: pd.Series) -> pd.DataFrame:
    """Prior-UTC-day 60-day robust statistics; at least 30 valid days."""
    ts = pd.to_datetime(timestamps, utc=True, errors="raise")
    val = pd.to_numeric(values, errors="coerce")
    daily = pd.DataFrame({"day": ts.dt.floor("D"), "value": val}).dropna()
    medians = daily.groupby("day", sort=True).value.median()
    row_days = ts.dt.floor("D")
    unique_days = pd.Index(row_days.unique()).sort_values()
    stats: dict[pd.Timestamp, tuple[float, float, np.ndarray] | None] = {}
    for day in unique_days:
        history = medians[(medians.index < day) & (medians.index >= day - pd.Timedelta(days=60))]
        expected = min(60, max(0, (day - medians.index.min()).days)) if len(medians) else 0
        required = max(30, int(np.ceil(expected * .70)))
        if len(history) < required:
            stats[day] = None
            continue
        median = float(history.median())
        mad = float((history - median).abs().median())
        stats[day] = (median, mad, np.sort(history.to_numpy(float))) if np.isfinite(mad) and mad > 0 else None
    z = np.full(len(ts), np.nan)
    pct = np.full(len(ts), np.nan)
    valid = np.zeros(len(ts), dtype=bool)
    day_positions = row_days.groupby(row_days, sort=True).groups
    raw = val.to_numpy(float)
    for day in unique_days:
        idx = np.asarray(day_positions[day], dtype=np.int64)
        item = stats[day]
        if item is None:
            continue
        median, mad, ordered = item
        finite = idx[np.isfinite(raw[idx])]
        z[finite] = (raw[finite] - median) / (1.4826 * mad)
        pct[finite] = np.searchsorted(ordered, raw[finite], side="right") / len(ordered)
        valid[finite] = True
    return pd.DataFrame({"oi_change_robust_z": z, "oi_change_percentile": pct,
                         "oi_normalization_valid": valid}, index=frame_index(timestamps))


def frame_index(series: pd.Series) -> pd.Index:
    return series.index


def onset_mask(state: pd.Series, valid: pd.Series, timestamps: pd.Series) -> pd.Series:
    """False-to-true only after a valid contiguous predecessor.

    A true state at the start of retention is intentionally not an onset.
    """
    state = state.fillna(False).astype(bool)
    valid = valid.fillna(False).astype(bool)
    ts = pd.to_datetime(timestamps, utc=True, errors="raise")
    contiguous = ts.diff().eq(FIVE_MINUTES)
    return state & valid & valid.shift(1, fill_value=False) & contiguous & ~state.shift(1, fill_value=False)


def oi_retention_gap_counts(timestamps: pd.Series, oi_values: pd.Series) -> tuple[int, int]:
    """Count discontinuities and missing five-minute bars within observed OI retention."""
    ts=pd.to_datetime(timestamps,utc=True,errors="raise").reset_index(drop=True)
    available=pd.to_numeric(oi_values,errors="coerce").reset_index(drop=True).notna()
    steps=ts[available].diff().dropna()
    return int(steps.gt(FIVE_MINUTES).sum()), int(((steps/FIVE_MINUTES)-1).clip(lower=0).sum())


def episode_table(symbol: str, timestamps: pd.Series, state: pd.Series,
                  valid: pd.Series, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="raise").reset_index(drop=True)
    s = state.fillna(False).astype(bool).reset_index(drop=True)
    v = valid.fillna(False).astype(bool).reset_index(drop=True)
    starts = onset_mask(s, v, ts)
    rows: list[dict[str, object]] = []
    for start in np.flatnonzero(starts.to_numpy()):
        end = start
        while end + 1 < len(s) and bool(s.iloc[end + 1]) and bool(v.iloc[end + 1]) and ts.iloc[end + 1] - ts.iloc[end] == FIVE_MINUTES:
            end += 1
        row: dict[str, object] = {
            "symbol": symbol, "onset_ts": ts.iloc[start],
            "decision_ts": ts.iloc[start] + FIVE_MINUTES,
            "episode_end_ts": ts.iloc[end] + FIVE_MINUTES,
            "duration_bars": end - start + 1,
            "duration_minutes": 5 * (end - start + 1),
        }
        if extra is not None:
            for column in extra.columns:
                row[column] = extra.iloc[start][column]
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["minutes_since_prior_episode"] = (
            out.onset_ts - out.episode_end_ts.shift(1)
        ).dt.total_seconds().div(60)
    return out


def completed_purge_states(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    base = strict_base_valid(frame)
    oi_reset = pd.to_numeric(frame.oi_log_change_1h, errors="coerce") < 0
    completed = (
        np.sign(pd.to_numeric(frame.trade_log_return_5m, errors="coerce")).ne(0)
        & np.sign(pd.to_numeric(frame.mark_return_1h, errors="coerce")).eq(
            np.sign(pd.to_numeric(frame.trade_log_return_5m, errors="coerce"))
        )
    )
    primary_liq = frame.liquidation_normalization_valid.fillna(False) & (frame.liquidation_intensity_robust_z >= 2)
    robust_liq = frame.liquidation_normalization_valid.fillna(False) & (frame.liquidation_intensity_percentile >= .95)
    return (base & primary_liq.shift(1, fill_value=False) & oi_reset & completed,
            base & robust_liq.shift(1, fill_value=False) & oi_reset & completed,
            np.sign(pd.to_numeric(frame.trade_log_return_5m, errors="coerce")))


GRAMMAR_COMPONENTS = ("trade_downside", "mark_downside", "structural_rejection", "oi", "liquidation", "basis_level", "basis_change", "breadth")
GRAMMAR_LADDERS = (
    ("trade_downside", "mark_downside", "structural_rejection"),
    ("trade_downside", "mark_downside", "structural_rejection", "oi"),
    ("trade_downside", "mark_downside", "structural_rejection", "oi", "liquidation"),
    ("trade_downside", "mark_downside", "structural_rejection", "oi", "basis_level"),
    ("trade_downside", "mark_downside", "structural_rejection", "oi", "basis_change"),
    ("trade_downside", "mark_downside", "structural_rejection", "oi", "breadth"),
    ("trade_downside", "mark_downside", "structural_rejection", "oi", "liquidation", "basis_change"),
)


def validate_grammar(ladders: Iterable[Iterable[str]] = GRAMMAR_LADDERS,
                     *, max_depth: int = 6) -> list[tuple[str, ...]]:
    result = [tuple(x) for x in ladders]
    if not result or len(result) != len(set(result)):
        raise ValueError("component grammar must be nonempty and unique")
    for cell in result:
        mandatory={"trade_downside","mark_downside","structural_rejection"}
        if not set(cell) <= set(GRAMMAR_COMPONENTS) or not mandatory.issubset(cell) or len(cell) > max_depth:
            raise ValueError(f"invalid component grammar cell: {cell}")
    return result


def reconcile_universe(inventory: pd.DataFrame, cache_symbols: set[str]) -> pd.DataFrame:
    out = inventory.copy()
    if out.PF_symbol.duplicated().any():
        raise ValueError("frozen inventory has duplicate PF identity")
    out["crypto_pf_identity"] = ~out.canonical_asset_id.astype(str).str.endswith("x")
    out["rankable_trade_coverage"] = out.trade_coverage_day_count.astype(int).gt(0)
    out["rankable_mark_coverage"] = out.mark_coverage_day_count.astype(int).gt(0)
    out["OI_coverage"] = out.PF_symbol.isin(cache_symbols)
    out["basis_coverage"] = out.PF_symbol.isin(cache_symbols)
    out["liquidation_coverage"] = out.PF_symbol.isin(cache_symbols)
    out["causal_lookback_eligibility"] = out.PF_symbol.isin(cache_symbols)
    out["ever_in_authorized_cohort"] = out.PF_symbol.isin(cache_symbols)
    out["final_campaign_eligible"] = out.PF_symbol.isin(cache_symbols)
    def reason(row: pd.Series) -> str:
        if row.final_campaign_eligible:
            return "included_authorized_stage8_cohort"
        if not bool(row.crypto_pf_identity):
            return "non_crypto_pf_identity"
        if not bool(row.get("included", False)):
            return str(row.get("exclusion_reason") or "frozen_inventory_excluded")
        if not bool(row.rankable_trade_coverage):
            return "no_rankable_trade_coverage_in_authorized_feature_cache"
        if not bool(row.rankable_mark_coverage):
            return "no_rankable_mark_coverage_in_authorized_feature_cache"
        return "not_in_authorized_stage8_cohort_requires_separate_cohort_rebuild"
    out["campaign_exclusion_reason"] = out.apply(reason, axis=1)
    if len(out) != 479 or int(out.included.fillna(False).sum()) != 460 or int(out.final_campaign_eligible.sum()) != len(cache_symbols):
        raise ValueError("universe reconciliation count mismatch")
    return out
