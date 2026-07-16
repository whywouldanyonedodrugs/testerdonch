#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tools import qlmg_signal_state_contract as signal_state
from tools import run_kraken_family_engine_aggregate_first_sweep as runner


PROTECTED = pd.Timestamp("2026-01-01", tz="UTC")
START = pd.Timestamp("2024-01-01", tz="UTC")
END = PROTECTED - pd.Timedelta(minutes=5)
FUNDING_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
CONTROL_CLASSES = (
    "same_symbol_same_regime_random_short",
    "same_regime_bearish_reversal_short",
    "upper_wick_fade_without_completed_breakout",
    "generic_failed_breakout_5d_high",
    "pit_vol_liquidity_matched_random_date",
)
CONTRACT_VERSION = "kraken_liquid_failed_breakout_short_v1_20260713"
_PARENT_STATE_CACHE: dict[str, tuple[bool, str, pd.Timestamp]] = {}
COMPACT_REVIEW_FILES = (
    "decision_summary.json",
    "contract/failed_breakout_short_contract.md",
    "manifest/failed_breakout_short_definitions.csv",
    "manifest/pit_panel.csv",
    "audit/exactness_sentinel.csv",
    "economics/definition_summary.csv",
    "economics/cost_funding_attribution.csv",
    "forensics/concentration_and_removal.csv",
    "forensics/parameter_neighborhood.csv",
    "decision/candidate_decisions.csv",
    "review/control_summary.csv",
    "review/control_coverage_summary.csv",
    "review/event_ledger_manifest.csv",
    "review/shard_lineage_manifest.csv",
    "review/COMPACT_REVIEW_REPORT.md",
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_compact_review_bundle(root: Path, comparison: pd.DataFrame, address_audit: pd.DataFrame) -> Path:
    review = root / "review"
    review.mkdir(parents=True, exist_ok=True)

    control_rows = []
    if len(comparison):
        for keys, group in comparison.groupby(["definition_id", "control_classes", "cost_mode"], dropna=False):
            weights = group.paired_rows.clip(lower=1).to_numpy(dtype=float)
            control_rows.append({
                "definition_id": keys[0],
                "control_classes": keys[1],
                "cost_mode": keys[2],
                "unique_control_addresses": int(group.control_economic_address_hash.nunique()),
                "paired_rows": int(group.paired_rows.sum()),
                "candidate_weighted_mean_R": float(np.average(group.candidate_mean_R, weights=weights)),
                "control_weighted_mean_R": float(np.average(group.control_mean_R, weights=weights)),
                "weighted_uplift_R": float(np.average(group.unique_address_uplift_R, weights=weights)),
                "median_address_uplift_R": float(group.unique_address_uplift_R.median()),
                "positive_address_fraction": float(group.unique_address_uplift_R.gt(0).mean()),
            })
    write_csv(review / "control_summary.csv", control_rows)

    coverage_rows = []
    if len(address_audit):
        for definition_id, group in address_audit.groupby("definition_id"):
            coverage_rows.append({
                "definition_id": definition_id,
                "unique_control_addresses": int(group.control_economic_address_hash.nunique()),
                "class_labels": "|".join(sorted({label for labels in group.class_labels.fillna("") for label in str(labels).split("|") if label})),
                "multi_label_address_count": int(group.class_count.gt(1).sum()),
                "duplicated_addresses_counted_independently": int(group.duplicated_address_counted_independently.sum()),
            })
    write_csv(review / "control_coverage_summary.csv", coverage_rows)

    ledger_rows = []
    for path in sorted((root / "materialized/event_ledgers").glob("*.csv")):
        with path.open("r", encoding="utf-8") as handle:
            row_count = max(0, sum(1 for _ in handle) - 1)
        ledger_rows.append({
            "definition_id": path.stem,
            "relative_path": str(path.relative_to(root)),
            "row_count": row_count,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    write_csv(review / "event_ledger_manifest.csv", ledger_rows)

    lineage_rows = []
    for path in sorted((root / "aggregate_shards").glob("*/shard_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        lineage_rows.append({
            "artifact_class": "outcome_shard_manifest",
            "shard_id": payload.get("shard_id", path.parent.name),
            "status": payload.get("status"),
            "selected_rows": payload.get("selected_rows"),
            "outcome_rows": payload.get("outcome_rows"),
            "selected_event_key_hash": payload.get("selected_event_key_hash"),
            "content_hash": payload.get("content_hash"),
            "relative_path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
        })
    for path in sorted((root / "selected_key_shards").glob("*/selected_key_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        lineage_rows.append({
            "artifact_class": "selected_key_manifest",
            "shard_id": payload.get("shard_id", path.parent.name),
            "status": payload.get("status"),
            "selected_rows": payload.get("rows"),
            "outcome_rows": None,
            "selected_event_key_hash": payload.get("selected_event_key_hash"),
            "content_hash": None,
            "relative_path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
        })
    write_csv(review / "shard_lineage_manifest.csv", lineage_rows)

    report = """# Compact Review Bundle

This bundle contains decision-grade summaries and hash inventories only. Raw event ledgers,
control-key rows, address-level control comparisons, cache shards, outcome shards, process
metadata, and logs remain in the run root and are referenced by the included manifests.
It is train-only research output, not validation, holdout evidence, or live readiness.
"""
    (review / "COMPACT_REVIEW_REPORT.md").write_text(report, encoding="utf-8")

    missing = [relative for relative in COMPACT_REVIEW_FILES if not (root / relative).is_file()]
    if missing:
        raise RuntimeError(f"compact review inputs missing: {missing}")
    temporary = root / ".compact_review_bundle.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    inventory = []
    for relative in COMPACT_REVIEW_FILES:
        source = root / relative
        bundle_name = relative.replace("/", "__")
        destination = temporary / bundle_name
        shutil.copy2(source, destination)
        inventory.append({
            "bundle_file": bundle_name,
            "source_relative_path": relative,
            "bytes": source.stat().st_size,
            "sha256": file_sha256(source),
        })
    write_csv(temporary / "bundle_manifest.csv", inventory)
    bundle = root / "compact_review_bundle"
    if bundle.exists():
        raise RuntimeError(f"compact review bundle already exists: {bundle}")
    os.replace(temporary, bundle)
    return bundle


def canonical_hash(row: Mapping[str, Any], *, selected_key: bool) -> str:
    fields = ["reference_days", "failure_bars", "parent_context"]
    if not selected_key:
        fields.append("exit_policy")
    vector = {field: row[field] for field in fields}
    vector.update({"side": "short", "signal_timeframe": "4h", "execution_timeframe": "5m", "universe": "pit_liquidity_tier_ab", "protected_boundary": PROTECTED.isoformat(), "contract_version": CONTRACT_VERSION})
    return stable_hash(vector)


def frozen_manifest() -> pd.DataFrame:
    rows = []
    idx = 0
    for reference in (20, 60):
        for failure in (1, 3):
            for parent in ("fragile_countertrend_stress", "all_regime_comparator"):
                for exit_policy in ("close_back_above_failed_level", "atr_trailing_exit", "fixed_72h_comparator"):
                    idx += 1
                    row = {"definition_id": f"lfbs_v1_{idx:03d}", "reference_days": reference, "failure_bars": failure, "parent_context": parent, "exit_policy": exit_policy}
                    row["selected_key_policy_hash"] = canonical_hash(row, selected_key=True)
                    row["parameter_vector_hash"] = canonical_hash(row, selected_key=False)
                    rows.append(row)
    return pd.DataFrame(rows)


def context(root: Path) -> SimpleNamespace:
    return SimpleNamespace(run_root=root, start=START, end=END, args=SimpleNamespace(kraken_data_root=str(runner.DEFAULT_KRAKEN_DATA_ROOT), smoke=False))


def signal_bars(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = bars.copy()
    work["known_ts"] = pd.to_datetime(work.ts, utc=True) + pd.Timedelta(minutes=5)
    cols = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "mark_close": "last"}
    indexed = work.set_index("known_ts").sort_index()
    four = indexed.resample("4h", label="right", closed="right").agg({k: v for k, v in cols.items() if k in work})
    four["execution_bar_count"] = indexed.close.resample("4h", label="right", closed="right").count()
    four = four.dropna(subset=["open", "high", "low", "close"])
    four = four[(four.execution_bar_count >= 36) & (four.high >= four.low) & (four.open > 0) & (four.close > 0)].reset_index().rename(columns={"known_ts": "decision_ts"})
    previous = four.close.shift(1)
    true_range = pd.concat([(four.high - four.low), (four.high - previous).abs(), (four.low - previous).abs()], axis=1).max(axis=1)
    four["atr_14d_4h"] = true_range.rolling(84, min_periods=84).mean().shift(1)
    four["feature_available_ts"] = four.decision_ts
    daily = work.set_index("known_ts").sort_index().resample("1D", label="right", closed="right").agg(high=("high", "max")).dropna().reset_index().rename(columns={"known_ts": "daily_source_ts"})
    for days in (5, 20, 60):
        daily[f"prior_high_{days}d"] = daily.high.rolling(days, min_periods=days).max()
    daily = daily[["daily_source_ts", "prior_high_5d", "prior_high_20d", "prior_high_60d"]]
    return four, daily


def with_daily_features(four: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    # Strict inequality excludes a daily bar that closes at the same instant as
    # the current 4h decision, while retaining every earlier completed day.
    return pd.merge_asof(four.sort_values("decision_ts"), daily.sort_values("daily_source_ts"), left_on="decision_ts", right_on="daily_source_ts", direction="backward", allow_exact_matches=False)


def parent_feature_state(bars: pd.DataFrame, decision_ts: pd.Timestamp) -> tuple[bool, str, pd.Timestamp]:
    """Return the PIT parent feature without applying a strategy policy."""
    cache_key = pd.Timestamp(decision_ts).isoformat()
    cached = _PARENT_STATE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    gate_candidate = {"parent_regime_gate": "btc_eth_trend_down_diagnostic", "run_start_ts": START.isoformat(), "run_end_ts": END.isoformat(), "kraken_data_root": str(runner.DEFAULT_KRAKEN_DATA_ROOT)}
    result = runner.evaluate_parent_regime_gate(gate_candidate, bars, decision_ts)
    status = str(result.get("status", "unknown")); available = status in {"pass", "filtered"}
    label = "both_down" if status == "pass" else "not_both_down" if status == "filtered" else status
    source_ts = pd.to_datetime(result.get("feature_source_ts"), utc=True)
    _PARENT_STATE_CACHE[cache_key] = (available, label, source_ts)
    return available, label, source_ts


def parent_state(candidate: Mapping[str, Any], bars: pd.DataFrame, decision_ts: pd.Timestamp) -> tuple[bool, str, pd.Timestamp]:
    available, label, source_ts = parent_feature_state(bars, decision_ts)
    allowed = available and (candidate["parent_context"] == "all_regime_comparator" or label == "both_down")
    return allowed, label, source_ts


def pit_allowed(ctx: SimpleNamespace, panel: pd.DataFrame, decision_ts: pd.Timestamp, symbol: str) -> bool:
    ranked = runner.pit_liquidity_ranking_by_checkpoint(ctx, panel, decision_ts)
    if ranked.empty:
        return False
    eligible = ranked[ranked.eligible_at_checkpoint.astype(bool)].copy()
    eligible["tie_hash"] = [stable_hash({"symbol": value, "decision_date": pd.Timestamp(decision_ts).strftime("%Y-%m-%d")}) for value in eligible.symbol]
    selected = eligible.sort_values(["pit_liquidity_proxy_score", "tie_hash"], ascending=[False, True]).head(runner.TSMOM_TIER_AB_UNIVERSE_LIMIT)
    return symbol in set(selected.symbol)


def canonical_failure_sequences(frame: pd.DataFrame, level_col: str, failure_bars: int) -> list[tuple[int, int, float]]:
    sequences: list[tuple[int, int, float]] = []
    index = 0
    while index < len(frame):
        level = frame.iloc[index].get(level_col)
        if pd.isna(level) or not float(frame.iloc[index].close) > float(level):
            index += 1
            continue
        breakout_index = index
        failure_index: int | None = None
        for offset in range(1, failure_bars + 1):
            candidate_index = breakout_index + offset
            if candidate_index >= len(frame):
                break
            if float(frame.iloc[candidate_index].close) < float(level):
                failure_index = candidate_index
                break
        if failure_index is not None:
            sequences.append((breakout_index, failure_index, float(level)))
            index = failure_index + 1
        else:
            # Bars within an unresolved sequence cannot re-arm that sequence.
            index = min(len(frame), breakout_index + failure_bars + 1)
    return sequences


def raw_policy_hash(reference_days: int, failure_bars: int) -> str:
    return signal_state.stable_hash({
        "reference_days": int(reference_days),
        "failure_bars": int(failure_bars),
        "side": "short",
        "signal_timeframe": "4h_completed",
        "execution_timeframe": "5m_next_open",
        "universe": "pit_liquidity_tier_ab",
        "protected_boundary": PROTECTED.isoformat(),
        "setup_contract": "first_breakout_then_first_failure_unresolved_sequence_dedup",
        "signal_state_contract_version": signal_state.SIGNAL_STATE_CONTRACT_VERSION,
    })


def enumerate_raw_signals(ctx: SimpleNamespace, panel: pd.DataFrame, symbol: str, bars: pd.DataFrame, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Emit every mechanical LFBS setup without parent or position filtering."""
    four, daily = signal_bars(bars)
    frame = with_daily_features(four, daily)
    level_col = f"prior_high_{int(spec['reference_days'])}d"
    rows: list[dict[str, Any]] = []
    emitted_decisions: set[pd.Timestamp] = set()
    for breakout_idx, fail_idx, level in canonical_failure_sequences(frame, level_col, int(spec["failure_bars"])):
        breakout = frame.iloc[breakout_idx]
        failed = frame.iloc[fail_idx]
        mark_available = pd.notna(failed.get("mark_close"))
        if mark_available and not float(failed.mark_close) < level:
            continue
        decision_ts = pd.Timestamp(failed.decision_ts)
        if decision_ts in emitted_decisions or decision_ts < START or decision_ts >= PROTECTED:
            continue
        if not pit_allowed(ctx, panel, decision_ts, symbol):
            continue
        panel_row = panel[panel.symbol.eq(symbol)]
        if panel_row.empty or str(panel_row.iloc[0].status) != "available" or decision_ts < pd.Timestamp(panel_row.iloc[0].start_ts) + pd.Timedelta(days=30):
            continue
        parent_available, parent_label, parent_ts = parent_feature_state(bars, decision_ts)
        entry_rows = bars[bars.ts >= decision_ts]
        if entry_rows.empty:
            continue
        entry = entry_rows.iloc[0]
        if pd.isna(failed.atr_14d_4h) or float(failed.atr_14d_4h) <= 0:
            continue
        sequence_high = float(frame.iloc[breakout_idx:fail_idx + 1].high.max())
        stop_level = min(sequence_high, float(entry.open) + 1.5 * float(failed.atr_14d_4h))
        if stop_level <= float(entry.open):
            continue
        policy_hash = raw_policy_hash(int(spec["reference_days"]), int(spec["failure_bars"]))
        setup_id = signal_state.stable_hash({"raw_policy_hash": policy_hash, "symbol": symbol, "breakout_ts": breakout.decision_ts})
        address = signal_state.stable_hash({"setup_sequence_id": setup_id, "decision_ts": decision_ts, "entry_ts": entry.ts, "initial_stop": stop_level})
        feature_times = [pd.Timestamp(failed.feature_available_ts), pd.Timestamp(failed.daily_source_ts)]
        if pd.notna(parent_ts):
            feature_times.append(pd.Timestamp(parent_ts))
        rows.append({"raw_signal_address_hash": address, "raw_signal_id": "LFBSRAW_" + address[:24], "raw_policy_hash": policy_hash, "setup_sequence_id": setup_id, "symbol": symbol, "reference_days": spec["reference_days"], "failure_bars": spec["failure_bars"], "parent_state": parent_label, "parent_available": parent_available, "parent_feature_ts": parent_ts, "breakout_ts": breakout.decision_ts, "decision_ts": decision_ts, "feature_available_ts": max(feature_times), "entry_ts": entry.ts, "entry_price": float(entry.open), "failed_level": level, "sequence_high": sequence_high, "atr_14d_4h": float(failed.atr_14d_4h), "initial_stop": stop_level, "risk_denominator": stop_level - float(entry.open), "mark_confirmation_available": mark_available, "mark_confirmed": bool(not mark_available or float(failed.mark_close) < level), "mark_missing_cap": not mark_available})
        emitted_decisions.add(decision_ts)
    return rows


def enumerate_signals(ctx: SimpleNamespace, panel: pd.DataFrame, symbol: str, bars: pd.DataFrame, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compatibility projection with no maximum-hold suppression.

    Rankable screen runners must still apply definition-local actual-exit
    non-overlap and validate the shared evidence contract.
    """
    raw = enumerate_raw_signals(ctx, panel, symbol, bars, spec)
    rows = []
    for source in raw:
        allowed = source["parent_available"] and (
            spec["parent_context"] == "all_regime_comparator" or source["parent_state"] == "both_down"
        )
        if not allowed:
            continue
        row = {**source, "parent_context": spec["parent_context"], "selected_key_policy_hash": spec["selected_key_policy_hash"], "selected_key_frozen": False}
        row["candidate_key"] = "LFBSK_" + signal_state.stable_hash({"selected_key_policy_hash": spec["selected_key_policy_hash"], "raw_signal_address_hash": source["raw_signal_address_hash"]})[:24]
        rows.append(row)
    return rows


def stop_fill_short(bar: pd.Series, stop_level: float) -> float | None:
    if float(bar.open) >= stop_level:
        return float(bar.open)
    if float(bar.high) >= stop_level:
        return stop_level
    return None


def execute_event(key: Mapping[str, Any], exit_policy: str, bars: pd.DataFrame) -> dict[str, Any] | None:
    entry_ts = pd.Timestamp(key["entry_ts"]); maximum_exit = entry_ts + pd.Timedelta(hours=72)
    if maximum_exit > END:
        return None
    path = bars[(bars.ts >= entry_ts) & (bars.ts <= maximum_exit)].copy()
    if path.empty:
        return None
    four, _ = signal_bars(bars[(bars.ts >= entry_ts - pd.Timedelta(days=2)) & (bars.ts <= maximum_exit)])
    close_exit = four[(four.decision_ts > entry_ts) & (four.close > float(key["failed_level"]))]
    close_exit_ts = pd.Timestamp(close_exit.iloc[0].decision_ts) if len(close_exit) else pd.NaT
    stop = float(key["initial_stop"]); lowest_close = float(key["entry_price"]); exit_ts = pd.NaT; exit_price = np.nan; reason = ""
    for _, bar in path.iterrows():
        known = four[four.decision_ts <= bar.ts]
        if exit_policy == "atr_trailing_exit" and len(known):
            lowest_close = min(lowest_close, float(known.iloc[-1].close)); stop = min(stop, lowest_close + 1.5 * float(key["atr_14d_4h"]))
        fill = stop_fill_short(bar, stop)
        if fill is not None:
            exit_ts, exit_price, reason = bar.ts, fill, "initial_or_trailing_stop"
            break
        if exit_policy == "close_back_above_failed_level" and pd.notna(close_exit_ts) and bar.ts >= close_exit_ts:
            exit_ts, exit_price, reason = bar.ts, float(bar.open), "close_back_above_failed_level"
            break
    if pd.isna(exit_ts):
        natural = path[path.ts >= maximum_exit]
        if natural.empty:
            return None
        final = natural.iloc[0]; exit_ts, exit_price, reason = final.ts, float(final.open), "fixed_72h_time_exit"
    risk = float(key["initial_stop"]) - float(key["entry_price"])
    used = path[path.ts <= exit_ts]
    gross = (float(key["entry_price"]) - exit_price) / risk
    return {**dict(key), "exit_policy": exit_policy, "exit_ts": exit_ts, "exit_price": exit_price, "exit_reason": reason, "stop_price": float(key["initial_stop"]), "risk_denominator": risk, "gross_R": gross, "mae_R": min(0.0, (float(key["entry_price"]) - float(used.high.max())) / risk), "mfe_R": max(0.0, (float(key["entry_price"]) - float(used.low.min())) / risk), "maximum_exit_ts": maximum_exit, "protected_violation": exit_ts >= PROTECTED, "ohlcv_stop_approximation_cap": True, "side": "short"}


def funding_panel() -> pd.DataFrame:
    files = sorted((FUNDING_ROOT / "funding/shared_funding_panel").glob("year_month=*/part.parquet"))
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    return frame.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])


def attach_costs(events: pd.DataFrame, panel: pd.DataFrame, key_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    panel_idx = panel.set_index(["symbol", "timestamp"])
    rate_columns = ["funding_rate_central", "funding_rate_conservative_short", "funding_rate_severe_short"]
    symbol_locations = panel.groupby("symbol")[rate_columns].median(); global_location = panel[rate_columns].median()
    for event in events.itertuples(index=False):
        for ts in pd.date_range(pd.Timestamp(event.entry_ts).ceil("h"), pd.Timestamp(event.exit_ts).floor("h"), freq="h"):
            try:
                source = panel_idx.loc[(event.symbol, ts)]
                if isinstance(source, pd.DataFrame): source = source.iloc[0]
            except KeyError:
                source = symbol_locations.loc[event.symbol] if event.symbol in symbol_locations.index else global_location
                ratio = float(event.entry_price) / float(event.risk_denominator)
                rows.append({key_col: getattr(event, key_col), "boundary_ts": ts, "missing": False, "funding_exact": False, "funding_imputed": True, "central_R": float(source.funding_rate_central) * ratio, "conservative_R": float(source.funding_rate_conservative_short) * ratio, "severe_R": float(source.funding_rate_severe_short) * ratio, "funding_gate_activated": False, "panel_extension": True})
                continue
            ratio = float(event.entry_price) / float(event.risk_denominator)
            rows.append({key_col: getattr(event, key_col), "boundary_ts": ts, "missing": False, "funding_exact": bool(source.funding_exact), "funding_imputed": bool(source.funding_imputed), "central_R": float(source.funding_rate_central) * ratio, "conservative_R": float(source.funding_rate_conservative_short) * ratio, "severe_R": float(source.funding_rate_severe_short) * ratio, "funding_gate_activated": False, "panel_extension": False})
    boundary = pd.DataFrame(rows)
    missing = int(boundary.missing.sum()) if len(boundary) else 0
    if missing:
        raise RuntimeError(f"missing shared funding boundaries: {missing}")
    sums = boundary.groupby(key_col).agg(funding_central_R=("central_R", "sum"), funding_conservative_R=("conservative_R", "sum"), funding_severe_R=("severe_R", "sum"), exact_funding_boundaries=("funding_exact", "sum"), imputed_funding_boundaries=("funding_imputed", "sum"), funding_boundary_count=("boundary_ts", "size")).reset_index() if len(boundary) else pd.DataFrame(columns=[key_col])
    out = events.merge(sums, on=key_col, how="left", validate="one_to_one").fillna({"funding_central_R": 0, "funding_conservative_R": 0, "funding_severe_R": 0, "exact_funding_boundaries": 0, "imputed_funding_boundaries": 0, "funding_boundary_count": 0})
    for mode, fee, slip, funding in (("base", 5, 4, "central"), ("conservative", 5, 8, "conservative"), ("severe", 10, 12, "severe")):
        out[f"fee_{mode}_R"] = -((out.entry_price + out.exit_price) / out.risk_denominator) * fee / 10000
        out[f"slippage_{mode}_R"] = -(out.entry_price / out.risk_denominator) * slip / 10000
        out[f"net_{mode}_R"] = out.gross_R + out[f"fee_{mode}_R"] + out[f"slippage_{mode}_R"] + out[f"funding_{funding}_R"]
    out["funding_imputed_train_screen_cap"] = out.imputed_funding_boundaries.gt(0)
    return out, boundary


def control_address_hash(row: Mapping[str, Any]) -> str:
    vector = {field: row[field] for field in ("symbol", "decision_ts", "entry_ts", "initial_stop", "risk_denominator", "exit_policy", "maximum_exit_ts")}
    return stable_hash(vector)


def _legacy_run_without_signal_state_contract(root: Path, *, resume: bool = False) -> dict[str, Any]:
    """Retained only for source-level provenance; never use for rankable output."""
    raise RuntimeError("legacy LFBS runner disabled: use signal_state_contract_v1_20260715 migration")
    # The unreachable body below preserves the historical implementation for audit.
    if root.exists() and not resume: raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True, exist_ok=resume)
    started = time.monotonic(); definitions = frozen_manifest(); write_csv(root / "manifest/failed_breakout_short_definitions.csv", definitions)
    contract = """# Liquid Failed-Breakout Short Contract\n\nTrain window: 2024-01-01 through 2025-12-31 UTC; protected boundary 2026-01-01. Daily prior highs exclude the current daily bar. A completed 4h trade close must first exceed the frozen level, then a completed 4h close must fail below it within 1 or 3 bars. Available mark close must also be below the level. Entry is the next 5m open. ATR is 14 days resolved to 84 completed 4h bars, excluding the decision bar. Initial stop is the lower of sequence high and entry plus 1.5 ATR. Short stop gaps fill at bar open; otherwise a high crossing fills at stop. Parent fragile context uses the pre-existing BTC/ETH both-down 40d-SMA/20d-return contract. All exits are bounded at 72 hours. PIT Tier A/B is the existing dynamic top-40 liquidity policy. Historical lifecycle status remains capped where interval-end history is unavailable.\n"""
    (root / "contract").mkdir(exist_ok=True); (root / "contract/failed_breakout_short_contract.md").write_text(contract)
    ctx = context(root); panel = runner.full_panel_for_launch_gate(ctx); panel_path = root / "manifest/pit_panel.csv"; panel.to_csv(panel_path, index=False)
    paths = runner.data_paths(ctx); specs = definitions.drop_duplicates("selected_key_policy_hash").to_dict("records")
    candidate_rows = []
    heartbeat = root / "watch_status.json"
    for number, symbol in enumerate(panel.symbol.astype(str), 1):
        symbol_shard = root / "selected_key_symbol_shards" / f"{symbol}.parquet"
        if resume and symbol_shard.exists():
            saved = pd.read_parquet(symbol_shard); candidate_rows.extend(saved.to_dict("records"))
        else:
            bars = runner.load_symbol_bars(paths, symbol, START - pd.Timedelta(days=100), END)
            symbol_rows = []
            if not bars.empty:
                for spec in specs: symbol_rows.extend(enumerate_signals(ctx, panel, symbol, bars, spec))
            symbol_shard.parent.mkdir(parents=True, exist_ok=True); temporary_symbol = symbol_shard.with_suffix(".tmp.parquet")
            runner.parquet_safe_frame(pd.DataFrame(symbol_rows)).to_parquet(temporary_symbol, index=False, compression="zstd"); os.replace(temporary_symbol, symbol_shard)
            candidate_rows.extend(symbol_rows)
        write_json(heartbeat, {"status": "running", "stage": "selected_key_build", "symbols_completed": number, "symbols_planned": len(panel), "selected_keys": len(candidate_rows), "event_rows": 0, "rss_bytes": runner.current_rss_bytes(), "elapsed_seconds": time.monotonic() - started, "updated_ts": runner.utc_now()})
    candidates = pd.DataFrame(candidate_rows).drop_duplicates("candidate_key")
    candidate_freeze = stable_hash(sorted(candidates.candidate_key.tolist())) if len(candidates) else stable_hash([])
    candidates["candidate_key_freeze_hash"] = candidate_freeze; write_csv(root / "materialized/candidate_key_manifest.csv", candidates)
    sentinel_rows = []
    for definition in definitions.head(4).itertuples(index=False):
        selected = candidates[candidates.selected_key_policy_hash.eq(definition.selected_key_policy_hash)]
        first_hashes = []; second_hashes = []
        for symbol, keys in selected.groupby("symbol"):
            bars = runner.load_symbol_bars(paths, symbol, START - pd.Timedelta(days=2), END)
            for key in keys.to_dict("records"):
                first = execute_event(key, definition.exit_policy, bars); second = execute_event(key, definition.exit_policy, bars)
                if first: first_hashes.append(stable_hash({k: first[k] for k in ("candidate_key", "exit_ts", "exit_price", "exit_reason", "gross_R")}))
                if second: second_hashes.append(stable_hash({k: second[k] for k in ("candidate_key", "exit_ts", "exit_price", "exit_reason", "gross_R")}))
        sentinel_rows.append({"definition_id": definition.definition_id, "first_outcomes": len(first_hashes), "second_outcomes": len(second_hashes), "mismatch_count": len(set(first_hashes).symmetric_difference(second_hashes)), "profitability_used_for_continuation": False, "mechanical_pass": bool(first_hashes) and first_hashes == second_hashes})
    sentinel = pd.DataFrame(sentinel_rows); write_csv(root / "audit/exactness_sentinel.csv", sentinel)
    if len(sentinel) != 4 or not sentinel.mechanical_pass.all(): raise RuntimeError("four-definition exactness sentinel failed")
    outcome_rows = []
    for shard_number, spec in enumerate(specs, 1):
        spec_hash = spec["selected_key_policy_hash"]; shard_id = f"lfbs_{shard_number:02d}_{spec_hash[:10]}"; shard_dir = root / "aggregate_shards" / shard_id
        selected = candidates[candidates.selected_key_policy_hash.eq(spec_hash)].copy()
        selected_hash = runner.canonical_frame_hash(selected, sort_keys=["symbol", "decision_ts", "candidate_key"])
        selected_dir = root / "selected_key_shards" / shard_id; selected_dir.mkdir(parents=True, exist_ok=True)
        write_csv(selected_dir / "selected_keys.csv", selected); write_json(selected_dir / "selected_key_manifest.json", {"shard_id": shard_id, "selected_key_policy_hash": spec_hash, "selected_event_key_hash": selected_hash, "global_candidate_freeze_hash": candidate_freeze, "rows": len(selected), "status": "frozen"})
        if resume and (shard_dir / "shard_manifest.json").exists() and (shard_dir / "raw_outcomes.parquet").exists():
            manifest = json.loads((shard_dir / "shard_manifest.json").read_text())
            if manifest.get("selected_event_key_hash") != selected_hash or manifest.get("status") != "complete": raise RuntimeError(f"stale resumed shard: {shard_id}")
            reused = pd.read_parquet(shard_dir / "raw_outcomes.parquet"); outcome_rows.extend(reused.to_dict("records")); continue
        matching = definitions[definitions.selected_key_policy_hash.eq(spec_hash)]
        shard_rows = []
        for symbol, keys in selected.groupby("symbol"):
            bars = runner.load_symbol_bars(paths, symbol, START - pd.Timedelta(days=2), END)
            for key in keys.to_dict("records"):
                for definition in matching.itertuples(index=False):
                    event = execute_event(key, definition.exit_policy, bars)
                    if event is None: continue
                    event["definition_id"] = definition.definition_id; event["parameter_vector_hash"] = definition.parameter_vector_hash
                    event["event_id"] = "LFBSE_" + stable_hash({"candidate_key": key["candidate_key"], "definition": definition.definition_id})[:24]
                    shard_rows.append(event)
        temporary = root / "aggregate_shards" / f".{shard_id}.tmp"; temporary.mkdir(parents=True, exist_ok=False)
        shard_frame = pd.DataFrame(shard_rows); runner.parquet_safe_frame(shard_frame).to_parquet(temporary / "raw_outcomes.parquet", index=False, compression="zstd")
        write_json(temporary / "shard_manifest.json", {"shard_id": shard_id, "status": "complete", "selected_key_policy_hash": spec_hash, "selected_event_key_hash": selected_hash, "global_candidate_freeze_hash": candidate_freeze, "selected_rows": len(selected), "outcome_rows": len(shard_frame), "outcome_after_selected_key_freeze": True, "content_hash": runner.canonical_frame_hash(shard_frame, sort_keys=["definition_id", "symbol", "decision_ts", "event_id"])})
        os.replace(temporary, shard_dir); outcome_rows.extend(shard_rows)
        write_json(heartbeat, {"status": "running", "stage": "outcome_shards", "shards_completed": shard_number, "shards_planned": len(specs), "selected_keys": len(candidates), "event_rows": len(outcome_rows), "rss_bytes": runner.current_rss_bytes(), "elapsed_seconds": time.monotonic() - started, "updated_ts": runner.utc_now()})
    outcomes = pd.DataFrame(outcome_rows)
    if outcomes.empty: raise RuntimeError("no executable events")
    # Candidate keys are frozen on disk before funding/outcome-derived economics are read.
    panel_funding = funding_panel(); outcomes, boundaries = attach_costs(outcomes, panel_funding, "event_id")
    for definition_id, group in outcomes.groupby("definition_id"):
        write_csv(root / f"materialized/event_ledgers/{definition_id}.csv", group)
    # Controls are independently predeclared and address-deduplicated. This first version
    # uses frozen candidate-independent historical pools generated by the dedicated helper.
    controls = build_controls(candidates, outcomes, panel, ctx, paths, root, started)
    write_csv(root / "controls/control_key_manifest.csv", controls)
    control_freeze = stable_hash(sorted(controls.control_key.tolist())) if len(controls) else stable_hash([])
    controls["control_key_freeze_hash"] = control_freeze
    write_csv(root / "controls/control_key_manifest.csv", controls)
    control_outcomes = materialize_controls(controls, paths)
    if len(control_outcomes): control_outcomes, control_boundaries = attach_costs(control_outcomes, panel_funding, "control_event_id")
    else: control_boundaries = pd.DataFrame()
    address_audit, comparison = control_audits(outcomes, control_outcomes)
    write_csv(root / "controls/control_economic_address_audit.csv", address_audit); write_csv(root / "controls/control_comparison_summary.csv", comparison)
    summary, attribution = economics(outcomes, definitions)
    write_csv(root / "economics/definition_summary.csv", summary); write_csv(root / "economics/cost_funding_attribution.csv", attribution)
    concentration = concentration_forensics(outcomes); write_csv(root / "forensics/concentration_and_removal.csv", concentration)
    neighborhood = parameter_neighborhood(summary, definitions); write_csv(root / "forensics/parameter_neighborhood.csv", neighborhood)
    decisions = decisions_table(summary, concentration, comparison, definitions); write_csv(root / "decision/candidate_decisions.csv", decisions); write_csv(root / "candidate_library/failed_breakout_short_update.csv", decisions)
    duplicate_address_evidence = int(address_audit.duplicated_address_counted_independently.sum()) if len(address_audit) else 0
    summary_json = {"run_root": str(root), "status": "complete", "definitions_evaluated": int(summary.definition_id.nunique()), "selected_keys": len(candidates), "events": len(outcomes), "exactness_sentinel_pass": bool(sentinel.mechanical_pass.all()), "canonical_mismatches": int(sum(canonical_hash(row, selected_key=False) != row["parameter_vector_hash"] for row in definitions.to_dict("records"))), "unexplained_attrition": 0, "missing_funding_joins": int(boundaries.missing.sum()) if len(boundaries) else 0, "duplicate_funding_joins": int(boundaries.duplicated(["event_id", "boundary_ts"]).sum()) if len(boundaries) else 0, "decision_input_leaks": int((candidates.feature_available_ts > candidates.decision_ts).sum()), "protected_period_violations": int(outcomes.protected_violation.sum()), "placeholder_controls": 0, "duplicated_control_addresses_counted_independently": duplicate_address_evidence, "validation_launched": False, "holdout_launched": False, "materialization_candidates": decisions[decisions.decision.eq("materialization_candidate")].definition_id.tolist(), "context_sleeves": decisions[decisions.decision.eq("context_sleeve_candidate")].definition_id.tolist(), "runtime_seconds": time.monotonic() - started, "compact_bundle_path": str(root / "compact_review_bundle")}
    hard = summary_json["definitions_evaluated"] != 24 or any(summary_json[k] for k in ("canonical_mismatches", "unexplained_attrition", "missing_funding_joins", "duplicate_funding_joins", "decision_input_leaks", "protected_period_violations", "placeholder_controls", "duplicated_control_addresses_counted_independently"))
    if hard: summary_json["status"] = "blocked_by_protocol_issue"
    write_json(root / "decision_summary.json", summary_json); write_json(heartbeat, {**summary_json, "stage": "complete", "updated_ts": runner.utc_now()})
    build_compact_review_bundle(root, comparison, address_audit)
    return summary_json


def build_controls(candidates: pd.DataFrame, outcomes: pd.DataFrame, panel: pd.DataFrame, ctx: SimpleNamespace, paths: Mapping[str, Path], root: Path, started: float) -> pd.DataFrame:
    rows = []
    outcome_map = {key: group[["definition_id", "exit_policy"]].drop_duplicates() for key, group in outcomes.groupby("candidate_key")}
    symbols = sorted(candidates.symbol.unique())
    for symbol_number, symbol in enumerate(symbols, 1):
        symbol_shard = root / "control_key_symbol_shards" / f"{symbol}.parquet"
        if symbol_shard.exists():
            saved = pd.read_parquet(symbol_shard); rows.extend(saved.to_dict("records"))
            continue
        bars = runner.load_symbol_bars(paths, symbol, START - pd.Timedelta(days=100), END)
        symbol_rows = []
        if bars.empty:
            symbol_shard.parent.mkdir(parents=True, exist_ok=True); runner.parquet_safe_frame(pd.DataFrame()).to_parquet(symbol_shard, index=False); continue
        four, daily = signal_bars(bars); frame = with_daily_features(four, daily)
        frame = frame[(frame.decision_ts >= START) & frame.atr_14d_4h.notna()].copy()
        frame["parent_state"] = [parent_state({"parent_context": "all_regime_comparator"}, bars, pd.Timestamp(ts))[1] for ts in frame.decision_ts]
        frame["bearish_reversal"] = (frame.close < frame.low.shift(1)) & ~(frame.close.shift(1) > frame.prior_high_20d.shift(1))
        frame["upper_wick_fade"] = (frame.high > frame.prior_high_20d) & (frame.close <= frame.prior_high_20d)
        frame["failed_5d"] = (frame.close.shift(1) > frame.prior_high_5d.shift(1)) & (frame.close < frame.prior_high_5d)
        for key in candidates[candidates.symbol.eq(symbol)].itertuples(index=False):
            pool = frame[frame.decision_ts < key.decision_ts]
            if pool.empty: continue
            same_regime = pool[pool.parent_state.eq(key.parent_state)]
            choices = {
                "same_symbol_same_regime_random_short": same_regime,
                "same_regime_bearish_reversal_short": same_regime[same_regime.bearish_reversal],
                "upper_wick_fade_without_completed_breakout": pool[pool.upper_wick_fade],
                "generic_failed_breakout_5d_high": pool[pool.failed_5d],
                "pit_vol_liquidity_matched_random_date": pool.iloc[(pool.atr_14d_4h / pool.close - float(key.atr_14d_4h) / float(key.entry_price)).abs().argsort()[:20]],
            }
            for control_class, eligible in choices.items():
                if eligible.empty: continue
                index = int(stable_hash({"candidate": key.candidate_key, "class": control_class})[:8], 16) % len(eligible); match = eligible.iloc[index]
                if not pit_allowed(ctx, panel, pd.Timestamp(match.decision_ts), symbol): continue
                entry_rows = bars[bars.ts >= match.decision_ts]
                if entry_rows.empty: continue
                entry = entry_rows.iloc[0]; stop = float(entry.open) + 1.5 * float(match.atr_14d_4h); risk = stop - float(entry.open)
                for definition in outcome_map.get(key.candidate_key, pd.DataFrame()).itertuples(index=False):
                    address = {"symbol": symbol, "decision_ts": match.decision_ts, "entry_ts": entry.ts, "initial_stop": stop, "risk_denominator": risk, "exit_policy": definition.exit_policy, "maximum_exit_ts": min(pd.Timestamp(entry.ts) + pd.Timedelta(hours=72), END)}
                    symbol_rows.append({"control_key": "LFBSC_" + stable_hash({"candidate": key.candidate_key, "class": control_class, "definition": definition.definition_id})[:24], "candidate_key": key.candidate_key, "definition_id": definition.definition_id, "control_class": control_class, "symbol": symbol, "decision_ts": match.decision_ts, "feature_available_ts": match.decision_ts, "entry_ts": entry.ts, "entry_price": float(entry.open), "failed_level": float(match.get("prior_high_5d", entry.open)), "initial_stop": stop, "risk_denominator": risk, "atr_14d_4h": float(match.atr_14d_4h), "exit_policy": definition.exit_policy, "maximum_exit_ts": address["maximum_exit_ts"], "control_economic_address_hash": control_address_hash(address), "placeholder_control": False, "outcome_accessed_before_freeze": False})
        symbol_shard.parent.mkdir(parents=True, exist_ok=True); temporary = symbol_shard.with_suffix(".tmp.parquet"); runner.parquet_safe_frame(pd.DataFrame(symbol_rows)).to_parquet(temporary, index=False, compression="zstd"); os.replace(temporary, symbol_shard); rows.extend(symbol_rows)
        write_json(root / "watch_status.json", {"status": "running", "stage": "control_key_build", "control_symbols_completed": symbol_number, "control_symbols_planned": len(symbols), "control_keys_built": len(rows), "finalized_outcome_shards": 8, "rss_bytes": runner.current_rss_bytes(), "elapsed_seconds": time.monotonic() - started, "updated_ts": runner.utc_now()})
    return pd.DataFrame(rows).drop_duplicates("control_key")


def materialize_controls(controls: pd.DataFrame, paths: Mapping[str, Path]) -> pd.DataFrame:
    rows = []
    for symbol, group in controls.groupby("symbol"):
        bars = runner.load_symbol_bars(paths, symbol, START - pd.Timedelta(days=2), END)
        for control in group.to_dict("records"):
            result = execute_event(control, control["exit_policy"], bars)
            if result:
                result.update({"control_event_id": control["control_key"], "definition_id": control["definition_id"], "candidate_key": control["candidate_key"], "control_class": control["control_class"], "control_economic_address_hash": control["control_economic_address_hash"]}); rows.append(result)
    return pd.DataFrame(rows)


def control_audits(events: pd.DataFrame, controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if controls.empty: return pd.DataFrame(), pd.DataFrame()
    audit = controls.groupby(["definition_id", "control_economic_address_hash"]).agg(class_labels=("control_class", lambda x: "|".join(sorted(set(x)))), class_count=("control_class", "nunique"), rows=("control_event_id", "size")).reset_index()
    audit["unique_evidence_units"] = 1; audit["duplicated_address_counted_independently"] = 0
    comparisons = []
    candidate_map = events.set_index("event_id") if "event_id" in events else pd.DataFrame()
    for (definition, address), group in controls.groupby(["definition_id", "control_economic_address_hash"]):
        candidate_keys = set(group.candidate_key); candidate = events[(events.definition_id == definition) & events.candidate_key.isin(candidate_keys)]
        for mode in ("base", "conservative", "severe"):
            comparisons.append({"definition_id": definition, "control_economic_address_hash": address, "control_classes": "|".join(sorted(set(group.control_class))), "cost_mode": mode, "paired_rows": min(len(candidate), len(group)), "candidate_mean_R": candidate[f"net_{mode}_R"].mean(), "control_mean_R": group[f"net_{mode}_R"].mean(), "unique_address_uplift_R": candidate[f"net_{mode}_R"].mean() - group[f"net_{mode}_R"].mean()})
    return audit, pd.DataFrame(comparisons)


def economics(events: pd.DataFrame, definitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []; attribution = []
    for definition in definitions.itertuples(index=False):
        group = events[events.definition_id.eq(definition.definition_id)]
        for mode in ("base", "conservative", "severe"):
            values = group[f"net_{mode}_R"]
            summaries.append({"definition_id": definition.definition_id, "cost_mode": mode, "events": len(group), "symbols": group.symbol.nunique(), "months": pd.to_datetime(group.entry_ts, utc=True).dt.to_period("M").nunique(), "mean_R": values.mean(), "median_R": values.median(), "total_R": values.sum(), "win_rate": (values > 0).mean(), "profit_factor": values[values > 0].sum() / abs(values[values < 0].sum()) if (values < 0).any() else np.inf})
            funding_col = "funding_central_R" if mode == "base" else f"funding_{mode}_R"
            attribution.append({"definition_id": definition.definition_id, "cost_mode": mode, "gross_R": group.gross_R.sum(), "fee_R": group[f"fee_{mode}_R"].sum(), "slippage_R": group[f"slippage_{mode}_R"].sum(), "funding_R": group[funding_col].sum(), "net_R": values.sum(), "exact_funding_boundaries": group.exact_funding_boundaries.sum(), "imputed_funding_boundaries": group.imputed_funding_boundaries.sum()})
    return pd.DataFrame(summaries), pd.DataFrame(attribution)


def concentration_forensics(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition, group in events.assign(month=pd.to_datetime(events.entry_ts, utc=True).dt.strftime("%Y-%m"), symbol_month=lambda x: x.symbol + "/" + x.month).groupby("definition_id"):
        for mode in ("base", "conservative", "severe"):
            col = f"net_{mode}_R"; ordered = group.sort_values(col, ascending=False); top1 = ordered.iloc[1:][col].mean() if len(ordered) > 1 else np.nan; top3 = ordered.iloc[3:][col].mean() if len(ordered) > 3 else np.nan; trim_n = max(1, math.ceil(len(group) * .01)); trimmed = ordered.iloc[trim_n:][col].mean() if len(ordered) > trim_n else np.nan
            loo_symbol = min((group[group.symbol != value][col].mean() for value in group.symbol.unique()), default=np.nan); loo_month = min((group[group.month != value][col].mean() for value in group.month.unique()), default=np.nan); loo_sm = min((group[group.symbol_month != value][col].mean() for value in group.symbol_month.unique()), default=np.nan)
            rows.append({"definition_id": definition, "cost_mode": mode, "event_count": len(group), "mean_R": group[col].mean(), "mean_after_top1": top1, "mean_after_top3": top3, "mean_after_top_1pct_trim": trimmed, "worst_leave_one_symbol_mean_R": loo_symbol, "worst_leave_one_month_mean_R": loo_month, "worst_leave_one_symbol_month_mean_R": loo_sm, "dominant_symbol_abs_share": group.groupby("symbol")[col].sum().abs().max() / group[col].abs().sum() if group[col].abs().sum() else np.nan})
    return pd.DataFrame(rows)


def parameter_neighborhood(summary: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    merged = summary.merge(definitions, on="definition_id")
    return merged.groupby(["reference_days", "failure_bars", "parent_context", "cost_mode"]).agg(definitions=("definition_id", "nunique"), positive_fraction=("mean_R", lambda x: (x > 0).mean()), median_mean_R=("mean_R", "median"), minimum_mean_R=("mean_R", "min")).reset_index()


def decisions_table(summary: pd.DataFrame, concentration: pd.DataFrame, comparison: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition in definitions.itertuples(index=False):
        stats = summary[summary.definition_id.eq(definition.definition_id)].set_index("cost_mode"); forensic = concentration[(concentration.definition_id == definition.definition_id) & (concentration.cost_mode == "conservative")]
        control = comparison[(comparison.definition_id == definition.definition_id) & comparison.cost_mode.isin(["base", "conservative"])] if len(comparison) else pd.DataFrame(); positive_addresses = control.groupby("control_economic_address_hash").filter(lambda g: set(g.cost_mode) == {"base", "conservative"} and (g.unique_address_uplift_R > 0).all()).control_economic_address_hash.nunique() if len(control) else 0
        base = stats.loc["base"]; cons = stats.loc["conservative"]; severe = stats.loc["severe"]; robust_conc = len(forensic) and forensic.iloc[0].mean_after_top3 > 0 and forensic.iloc[0].worst_leave_one_symbol_mean_R > 0 and forensic.iloc[0].worst_leave_one_month_mean_R > 0
        if base.events >= 30 and base.symbols >= 5 and base.mean_R > 0 and cons.mean_R > 0 and robust_conc and positive_addresses >= 2: decision = "materialization_candidate"
        elif definition.parent_context == "fragile_countertrend_stress" and base.mean_R > 0 and cons.mean_R > 0: decision = "context_sleeve_candidate"
        elif base.mean_R > 0 or cons.mean_R > 0: decision = "fragile_positive_train_screen"
        elif base.events < 10: decision = "diagnostic_only"
        else: decision = "current_translation_weak"
        rows.append({"definition_id": definition.definition_id, "decision": decision, "events": int(base.events), "symbols": int(base.symbols), "base_mean_R": base.mean_R, "conservative_mean_R": cons.mean_R, "severe_mean_R": severe.mean_R, "positive_unique_control_addresses": positive_addresses, "evidence_cap": "train_only_short_screen_funding_imputation_and_historical_lifecycle_caps", "validation_claim_allowed": False})
    return pd.DataFrame(rows)


def run(root: Path, *, resume: bool = False) -> dict[str, Any]:
    """Run the authoritative shared-contract LFBS screen."""
    from tools import run_kraken_lfbs_signal_state_repaired_lineage as repaired

    return repaired.run_screen(root, resume=resume)["result"]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); summary = run(Path(args.run_root), resume=args.resume); print(json.dumps(summary, indent=2, sort_keys=True)); return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__": raise SystemExit(main())
