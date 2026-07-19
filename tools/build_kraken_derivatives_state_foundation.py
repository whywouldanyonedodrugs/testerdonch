#!/usr/bin/env python3
"""Build the outcome-free Stage 8A Kraken derivatives-state foundation."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import build_kraken_c01_foundation as c01
from tools.telegram_notify import TelegramNotifier
from tools.qlmg_kraken_derivatives_state import (
    COHORT_VERSION, FEATURE_VERSION, PROTECTED_START, SEMANTIC_STATUS, TRAIN_START,
    assert_no_outcomes, basis_fields, causal_daily_normalization,
    cluster_canonical_episodes, decimal_text, deterministic_event_identity, exact_horizon_mask, liquidation_fields,
    load_semantic_decision, open_interest_fields, price_inferred_liquidation_side,
    sha256_file, stable_hash, validate_rankable_times,
)


TASK_ID = "donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1"
SEMANTIC_FILE_SHA256 = "c5ccd4f57981dfd949857016410fab87defaecc4635a6951fe5ee3e4965ede48"
ANALYTICS_MANIFEST_HASH = "f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d"
COHORT_SOURCE_HASH = "768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15"
METRICS = ("future-basis", "open-interest", "liquidation-volume")
MAJORS = ("PF_XBTUSD", "PF_ETHUSD")

ATTEMPTS = (
    ("KDA01", "kda01_primary_efficient", "crowding_with_efficient_price_progress", "robust_z_2"),
    ("KDA01", "kda01_primary_deteriorating", "crowding_with_deteriorating_price_progress", "robust_z_2"),
    ("KDA01", "kda01_primary_failure", "completed_failure_after_crowding_deterioration", "robust_z_2"),
    ("KDA01", "kda01_robust_efficient", "crowding_with_efficient_price_progress", "percentile_95"),
    ("KDA01", "kda01_robust_deteriorating", "crowding_with_deteriorating_price_progress", "percentile_95"),
    ("KDA01", "kda01_robust_failure", "completed_failure_after_crowding_deterioration", "percentile_95"),
    ("KDA02", "kda02_primary_active_purge", "active_liquidation_purge", "robust_z_2"),
    ("KDA02", "kda02_primary_completed", "completed_purge_reclaim_or_failure", "robust_z_2"),
    ("KDA02", "kda02_primary_oi_vacuum", "OI_vacuum_with_modest_price_displacement", "robust_z_2"),
    ("KDA02", "kda02_robust_active_purge", "active_liquidation_purge", "percentile_95"),
    ("KDA02", "kda02_robust_completed", "completed_purge_reclaim_or_failure", "percentile_95"),
    ("KDA02", "kda02_robust_oi_vacuum", "OI_vacuum_with_modest_price_displacement", "percentile_95"),
    ("KDA03", "kda03_basis_oi", "basis_expansion_with_OI_confirmation", "robust_z_2"),
    ("KDA03", "kda03_basis_no_price", "basis_expansion_without_price_confirmation", "robust_z_2"),
    ("KDA03", "kda03_basis_liq_reset", "extreme_basis_with_liquidation_and_OI_reset", "robust_z_2"),
)

GENERATOR_CONTRACT = {
    "version": "kda_outcome_free_generator_v1_20260719",
    "attempts": ATTEMPTS,
    "event_rule": "causal_false_to_true_onset_on_completed_five_minute_state",
    "decision_delay": "five_minutes_after_source_bucket_open",
    "semantic_duplicate_policy": {"kda02_robust_oi_vacuum": "killed_duplicate_of_kda02_primary_oi_vacuum"},
}
GENERATOR_CONTRACT_HASH = stable_hash(GENERATOR_CONTRACT)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def verify_analytics_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("content_hash") != ANALYTICS_MANIFEST_HASH or manifest.get("file_count") != 3672:
        raise ValueError("Stage 7C analytics manifest identity mismatch")
    if stable_hash(manifest["files"]) != manifest["content_hash"]:
        raise ValueError("Stage 7C manifest content hash mismatch")
    finals = [row for row in manifest["files"] if row.get("kind") == "final_parquet"]
    if len(finals) != 1836:
        raise ValueError("unexpected Stage 7C final parquet count")
    for row in finals:
        source = Path(row["path"])
        if not source.is_file() or sha256_file(source) != row["sha256"]:
            raise ValueError(f"Stage 7C object hash mismatch: {source}")
        if "/stage7c_v1/normalized/" not in str(source):
            raise ValueError("analytics object outside frozen root")
        schema = pq.ParquetFile(source).schema_arrow
        if row["rows"] and not {"timestamp_utc", "value_json", "analytics_type", "symbol", "interval_seconds"}.issubset(schema.names):
            raise ValueError(f"analytics schema mismatch: {source}")
    return manifest, finals


def build_wide_analytics_cache(finals: Sequence[Mapping[str, Any]], cache: Path) -> list[Path]:
    complete = cache / "analytics_wide_complete.json"
    if complete.exists():
        state = json.loads(complete.read_text())
        paths = [Path(item) for item in state["partitions"]]
        if state.get("manifest_hash") != ANALYTICS_MANIFEST_HASH or not paths or not all(path.is_file() for path in paths):
            raise ValueError("stale or incomplete analytics cache manifest")
        return paths
    cache.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[int, int, str], dict[str, Mapping[str, Any]]] = {}
    pattern = re.compile(r"analytics_type=([^/]+)/interval=300/year=(\d{4})/month=(\d{2})/shard=([^/]+)/data\.parquet$")
    for row in finals:
        match = pattern.search(str(row["path"]))
        if match:
            metric, year, month, shard = match.groups()
            grouped.setdefault((int(year), int(month), shard), {})[metric] = row
    connection = duckdb.connect(str(cache / "stage8a.duckdb"))
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='1GB'")
    connection.execute("SET preserve_insertion_order=false")
    connection.execute(f"SET temp_directory='{(cache / 'duckdb_tmp').as_posix()}'")
    published: list[Path] = []
    for order, (key, metrics) in enumerate(sorted(grouped.items())):
        if set(metrics) != set(METRICS) or any(int(metrics[metric]["rows"]) == 0 for metric in METRICS):
            continue
        year, month, shard = key
        unit = f"{year:04d}{month:02d}_{shard}"
        final_dir = cache / "wide_units" / unit
        temp_dir = cache / "wide_units" / f".{unit}.tmp"
        if final_dir.exists():
            marker = final_dir / "_COMPLETE.json"
            if not marker.is_file():
                raise ValueError(f"incomplete finalized analytics unit: {unit}")
            unit_paths = sorted(final_dir.glob("symbol=*/data_*.parquet"))
            if not unit_paths:
                raise ValueError(f"empty finalized analytics unit: {unit}")
            published.extend(unit_paths); continue
        if temp_dir.exists():
            raise ValueError(f"stale temporary analytics unit: {unit}")
        temp_dir.mkdir(parents=True)
        basis = str(metrics["future-basis"]["path"]).replace("'", "''")
        oi = str(metrics["open-interest"]["path"]).replace("'", "''")
        liquidation = str(metrics["liquidation-volume"]["path"]).replace("'", "''")
        query = f"""
          SELECT b.symbol, b.timestamp_utc,
            trim(both '"' from b.value_json) AS basis_raw,
            o.value_json AS oi_json,
            trim(both '"' from l.value_json) AS liquidation_raw
          FROM read_parquet('{basis}', hive_partitioning=false) b
          INNER JOIN read_parquet('{oi}', hive_partitioning=false) o USING (symbol, timestamp_utc)
          INNER JOIN read_parquet('{liquidation}', hive_partitioning=false) l USING (symbol, timestamp_utc)
          WHERE b.interval_seconds=300 AND o.interval_seconds=300 AND l.interval_seconds=300
            AND b.timestamp_utc >= TIMESTAMPTZ '2023-01-01 00:00:00+00'
            AND b.timestamp_utc < TIMESTAMPTZ '2026-01-01 00:00:00+00'
        """
        connection.execute(f"COPY ({query}) TO '{temp_dir.as_posix()}' (FORMAT PARQUET, PARTITION_BY (symbol), COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
        unit_paths = sorted(temp_dir.glob("symbol=*/data_*.parquet"))
        if not unit_paths:
            raise ValueError(f"analytics unit produced no exact intersections: {unit}")
        write_json(temp_dir / "_COMPLETE.json", {"unit": unit, "source_hashes": {metric: metrics[metric]["sha256"] for metric in METRICS}, "partition_count": len(unit_paths)})
        os.replace(temp_dir, final_dir)
        published.extend(sorted(final_dir.glob("symbol=*/data_*.parquet")))
        if order % 25 == 0:
            print(f"analytics units completed: {order + 1}/{len(grouped)}", flush=True)
    connection.close()
    if not published:
        raise ValueError("wide analytics cache is empty")
    write_json(complete, {"manifest_hash": ANALYTICS_MANIFEST_HASH, "partitions": [str(path) for path in sorted(published)]})
    return sorted(published)


def load_bar_frame(rows: Sequence[c01.AuthorityRow], symbol: str, dataset: str) -> tuple[pd.DataFrame, str]:
    selected = [row for row in rows if row.symbol == symbol and row.dataset == dataset]
    if not selected:
        raise ValueError(f"missing bar authority: {dataset}:{symbol}")
    parts: list[pd.DataFrame] = []
    columns = ["time", "close", "volume", "venue_symbol", "resolution", "rankable_pre_holdout", "contains_protected_period"]
    for row in selected:
        schema = pq.ParquetFile(row.parquet_path).schema_arrow.names
        if "time" not in schema:
            continue
        raw = pq.ParquetFile(row.parquet_path).read(columns=columns).to_pandas()
        if not raw.venue_symbol.eq(symbol).all() or not raw.resolution.eq("5m").all():
            raise ValueError("bar identity mismatch")
        if not raw.rankable_pre_holdout.map(c01._as_bool).all() or raw.contains_protected_period.map(c01._as_bool).any():
            raise ValueError("unsafe bar row")
        raw["timestamp_utc"] = pd.to_datetime(raw.time, unit="ms", utc=True)
        raw["close"] = pd.to_numeric(raw.close, errors="coerce")
        raw["volume"] = pd.to_numeric(raw.volume, errors="coerce")
        parts.append(raw[["timestamp_utc", "close", "volume"]])
    frame = pd.concat(parts, ignore_index=True).sort_values("timestamp_utc", kind="mergesort")
    frame = frame[(frame.timestamp_utc >= TRAIN_START) & (frame.timestamp_utc < PROTECTED_START)]
    duplicated = frame.duplicated("timestamp_utc", keep=False)
    if duplicated.any() and frame.loc[duplicated].groupby("timestamp_utc").close.nunique().gt(1).any():
        raise ValueError("conflicting duplicate bar")
    frame = frame.drop_duplicates("timestamp_utc", keep="first").reset_index(drop=True)
    validate_rankable_times(frame.timestamp_utc)
    ref = stable_hash([row.reference_id for row in selected])
    return frame, ref


def rolling_exact_return(close: pd.Series, ts: pd.Series, bars: int) -> pd.Series:
    result = np.log(close / close.shift(bars))
    result[ts.diff(bars).ne(pd.Timedelta(minutes=5 * bars))] = np.nan
    return result


def rolling_path(log_return: pd.Series, bars: int) -> tuple[pd.Series, pd.Series]:
    absolute = log_return.abs().rolling(bars, min_periods=bars).sum()
    cumulative = log_return.rolling(bars, min_periods=bars).sum().abs()
    efficiency = cumulative / absolute.replace(0, np.nan)
    largest = log_return.abs().rolling(bars, min_periods=bars).max() / absolute.replace(0, np.nan)
    return efficiency, largest


def add_symbol_features(raw: pd.DataFrame, trade: pd.DataFrame, mark: pd.DataFrame) -> pd.DataFrame:
    frame = raw.merge(trade.rename(columns={"close": "trade_close", "volume": "trade_volume"}), on="timestamp_utc", how="inner", validate="one_to_one")
    frame = frame.merge(mark[["timestamp_utc", "close"]].rename(columns={"close": "mark_close"}), on="timestamp_utc", how="inner", validate="one_to_one")
    frame = frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
    frame = frame.dropna(subset=["basis_raw", "oi_json", "liquidation_raw"])
    parsed_basis = [basis_fields(value) for value in frame.basis_raw]
    frame[["basis_raw_exact", "basis_decimal", "basis_percent", "basis_bps"]] = pd.DataFrame(parsed_basis, index=frame.index)
    parsed_oi = [open_interest_fields(value) for value in frame.oi_json]
    frame[["value_0_raw", "value_1_raw", "value_2_raw", "value_3_raw", "oi_open_base_units", "oi_high_base_units", "oi_low_base_units", "oi_close_base_units"]] = pd.DataFrame(parsed_oi, index=frame.index)
    parsed_liq = [liquidation_fields(value) for value in frame.liquidation_raw]
    frame[["liquidation_value_raw", "liquidation_base_units_5m"]] = pd.DataFrame(parsed_liq, index=frame.index)
    frame["trade_log_return_5m"] = rolling_exact_return(frame.trade_close, frame.timestamp_utc, 1)
    for name, bars in (("trade_return_15m", 3), ("trade_return_1h", 12), ("trade_return_6h", 72), ("trade_return_24h", 288)):
        frame[name] = rolling_exact_return(frame.trade_close, frame.timestamp_utc, bars)
    for name, bars in (("mark_return_1h", 12), ("mark_return_6h", 72)):
        frame[name] = rolling_exact_return(frame.mark_close, frame.timestamp_utc, bars)
    frame["realized_vol_1h"] = frame.trade_log_return_5m.rolling(12, min_periods=12).std(ddof=1)
    frame["realized_vol_24h"] = frame.trade_log_return_5m.rolling(288, min_periods=288).std(ddof=1)
    for label, bars in (("1h", 12), ("6h", 72)):
        frame[f"path_efficiency_{label}"], frame[f"largest_bar_share_{label}"] = rolling_path(frame.trade_log_return_5m, bars)
    for label, bars in (("15m", 3), ("1h", 12), ("6h", 72), ("24h", 288)):
        exact = exact_horizon_mask(frame.timestamp_utc, bars)
        current_oi = frame.oi_close_base_units.where(frame.oi_close_base_units > 0)
        lagged_oi = frame.oi_close_base_units.shift(bars).where(frame.oi_close_base_units.shift(bars) > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            frame[f"oi_log_change_{label}"] = np.log(current_oi / lagged_oi).where(exact)
        frame[f"basis_change_{label}"] = (frame.basis_decimal - frame.basis_decimal.shift(bars)).where(exact)
    exact_1h = exact_horizon_mask(frame.timestamp_utc, 12)
    frame["oi_range_fraction_1h"] = ((frame.oi_high_base_units.rolling(12, min_periods=12).max() - frame.oi_low_base_units.rolling(12, min_periods=12).min()) / frame.oi_close_base_units.shift(12)).where(exact_1h)
    frame["oi_notional_usd_proxy"] = frame.oi_close_base_units * frame.mark_close
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["oi_notional_log_change_1h"] = np.log(frame.oi_notional_usd_proxy.where(frame.oi_notional_usd_proxy > 0) / frame.oi_notional_usd_proxy.shift(12).where(frame.oi_notional_usd_proxy.shift(12) > 0)).where(exact_1h)
        frame["oi_notional_log_change_6h"] = np.log(frame.oi_notional_usd_proxy.where(frame.oi_notional_usd_proxy > 0) / frame.oi_notional_usd_proxy.shift(72).where(frame.oi_notional_usd_proxy.shift(72) > 0)).where(exact_horizon_mask(frame.timestamp_utc, 72))
    for label, bars in (("1h", 12), ("6h", 72)):
        exact = exact_horizon_mask(frame.timestamp_utc, bars)
        frame[f"liquidation_base_units_{label}"] = frame.liquidation_base_units_5m.rolling(bars, min_periods=bars).sum().where(exact)
        frame[f"liquidation_to_lagged_oi_{label}"] = (frame[f"liquidation_base_units_{label}"] / frame.oi_close_base_units.shift(bars)).where(exact)
    frame["liquidation_mark_notional_usd_proxy_1h"] = (frame.liquidation_base_units_5m * frame.mark_close).rolling(12, min_periods=12).sum().where(exact_1h)
    frame["liquidation_to_lagged_trade_volume_1h"] = (frame.liquidation_base_units_1h / frame.trade_volume.rolling(12, min_periods=12).sum().shift(12)).where(exact_horizon_mask(frame.timestamp_utc, 24))
    frame["price_inferred_liquidation_side"] = frame.trade_return_1h.map(price_inferred_liquidation_side)
    basis_stats = causal_daily_normalization(frame.timestamp_utc, frame.basis_decimal)
    change_stats = causal_daily_normalization(frame.timestamp_utc, frame.basis_change_1h)
    liq_stats = causal_daily_normalization(frame.timestamp_utc, frame.liquidation_to_lagged_oi_1h, daily_aggregation="max")
    frame["basis_level_robust_z"] = basis_stats.robust_z
    frame["basis_level_percentile"] = basis_stats.empirical_percentile
    frame["basis_change_robust_z"] = change_stats.robust_z
    frame["basis_change_percentile"] = change_stats.empirical_percentile
    frame["liquidation_intensity_robust_z"] = liq_stats.robust_z
    frame["liquidation_intensity_percentile"] = liq_stats.empirical_percentile
    frame["basis_level_normalization_valid"] = basis_stats.normalization_valid
    frame["basis_change_normalization_valid"] = change_stats.normalization_valid
    frame["liquidation_normalization_valid"] = liq_stats.normalization_valid
    frame["normalization_valid_days"] = pd.concat([basis_stats.prior_valid_days, change_stats.prior_valid_days, liq_stats.prior_valid_days], axis=1).min(axis=1)
    frame["normalization_stale_or_missing"] = ~(basis_stats.normalization_valid & change_stats.normalization_valid & liq_stats.normalization_valid)
    frame["trade_coverage"] = True; frame["mark_coverage"] = True; frame["analytics_coverage"] = True
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def register_attempts(semantic_hash: str, feature_hash: str) -> pd.DataFrame:
    nearest = {"KDA01": "FORUM H01/C04; RFBS; Backside; failed-breakdown; C01", "KDA02": "delayed-flush reclaim; PD04; FORUM H04; RFBS/Backside", "KDA03": "C08/local basis; C02 control"}
    return pd.DataFrame([{
        "family_id": family, "definition_id": definition, "attempt_id": definition,
        "state": state, "threshold": threshold, "lookback": "prior_60_calendar_days",
        "semantic_contract_hash": semantic_hash, "feature_contract_hash": feature_hash,
        "generator_contract_hash": GENERATOR_CONTRACT_HASH,
        "nearest_prior_families": nearest[family], "attempted_before_generation": True,
        "event_count": 0, "zero_count_branch": False,
        "killed_branch": definition == "kda02_robust_oi_vacuum",
        "kill_reason": "semantic_duplicate_of_kda02_primary_oi_vacuum" if definition == "kda02_robust_oi_vacuum" else "",
    } for family, definition, state, threshold in ATTEMPTS])


def one_minute_aggregation_diagnostics(finals: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    path_rows: dict[tuple[str, int, int, int], list[Path]] = {}
    pattern = re.compile(r"analytics_type=([^/]+)/interval=(60|300)/year=(\d{4})/month=(\d{2})/shard=([^/]+)/data\.parquet$")
    for row in finals:
        if not row.get("rows"):
            continue
        match = pattern.search(str(row["path"]))
        if match:
            metric, interval, year, month, _ = match.groups()
            path_rows.setdefault((metric, int(interval), int(year), int(month)), []).append(Path(row["path"]))
    counters = {(metric, symbol): {"overlap": 0, "mismatch": 0} for metric in METRICS for symbol in MAJORS}
    for metric in METRICS:
        for year in range(2023, 2026):
            for month in range(1, 13):
                minute_paths = path_rows.get((metric, 60, year, month), [])
                five_paths = path_rows.get((metric, 300, year, month), [])
                if not minute_paths or not five_paths:
                    continue
                minute = pd.concat([pq.ParquetFile(path).read(columns=["timestamp_utc", "symbol", "value_json"]).to_pandas() for path in minute_paths], ignore_index=True)
                five = pd.concat([pq.ParquetFile(path).read(columns=["timestamp_utc", "symbol", "value_json"]).to_pandas() for path in five_paths], ignore_index=True)
                minute = minute[minute.symbol.isin(MAJORS)].copy(); five = five[five.symbol.isin(MAJORS)].copy()
                minute["bucket"] = pd.to_datetime(minute.timestamp_utc, utc=True).dt.floor("5min")
                five["bucket"] = pd.to_datetime(five.timestamp_utc, utc=True)
                for symbol in MAJORS:
                    one = minute[minute.symbol.eq(symbol)].sort_values("timestamp_utc")
                    observed = five[five.symbol.eq(symbol)].set_index("bucket").value_json
                    if one.empty or observed.empty:
                        continue
                    if metric == "future-basis":
                        aggregate = one.groupby("bucket", sort=True).value_json.last().map(lambda value: str(decimal_text(json.loads(value))))
                        expected = observed.map(lambda value: str(decimal_text(json.loads(value))))
                    elif metric == "liquidation-volume":
                        aggregate = one.assign(parsed=one.value_json.map(lambda value: decimal_text(json.loads(value)))).groupby("bucket", sort=True).parsed.sum().map(str)
                        expected = observed.map(lambda value: str(decimal_text(json.loads(value))))
                    else:
                        parsed = one.value_json.map(lambda value: open_interest_fields(value))
                        expanded = pd.DataFrame(parsed.tolist(), index=one.index)
                        work = pd.concat([one[["bucket"]], expanded.iloc[:, 4:].set_axis(["open", "high", "low", "close"], axis=1)], axis=1)
                        aggregate = work.groupby("bucket", sort=True).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
                        expected = pd.DataFrame(observed.map(lambda value: open_interest_fields(value)[4:]).tolist(), index=observed.index, columns=["open", "high", "low", "close"])
                    common = aggregate.index.intersection(expected.index)
                    if metric == "open-interest":
                        mismatch = ~np.isclose(aggregate.loc[common].to_numpy(), expected.loc[common].to_numpy(), rtol=0, atol=1e-12).all(axis=1)
                    else:
                        tolerance = decimal_text("1e-12") if metric == "liquidation-volume" else decimal_text("0")
                        mismatch = np.array([
                            abs(decimal_text(left) - decimal_text(right)) > tolerance
                            for left, right in zip(aggregate.loc[common], expected.loc[common])
                        ])
                    counters[(metric, symbol)]["overlap"] += len(common)
                    counters[(metric, symbol)]["mismatch"] += int(mismatch.sum())
    return pd.DataFrame([{
        "metric": metric, "symbol": symbol, "overlapping_5m_rows": values["overlap"],
        "aggregation_mismatches": values["mismatch"],
        "aggregation_rule": "last" if metric == "future-basis" else "sum" if metric == "liquidation-volume" else "first_max_min_last",
        "comparison_tolerance": "1e-12" if metric != "future-basis" else "0",
        "state_agreement": values["mismatch"] == 0 and values["overlap"] > 0,
        "timing_finding": "completed_5m_state_available_only_after_bucket_close; no_1m_alt_proxy",
        "rankable_alt_proxy": False,
    } for (metric, symbol), values in counters.items()])


def onset(mask: pd.Series) -> pd.Series:
    return mask.fillna(False) & ~mask.shift(1, fill_value=False)


def generate_events(frame: pd.DataFrame, symbol: str, refs: str, semantic_hash: str, feature_hash: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_valid = frame.eligible & frame.trade_coverage & frame.mark_coverage & frame.analytics_coverage
    basis_valid = cohort_valid & frame.basis_level_normalization_valid
    liquidation_valid = cohort_valid & frame.liquidation_normalization_valid
    oi_primary = frame.oi_log_change_1h > 0
    basis_primary = frame.basis_level_robust_z.abs() >= 2
    crowd_primary = basis_valid & oi_primary & basis_primary
    oi_robust = frame.oi_log_change_1h > 0
    basis_robust = (frame.basis_level_percentile >= .95) | (frame.basis_level_percentile <= .05)
    crowd_robust = basis_valid & oi_robust & basis_robust
    efficient = (frame.path_efficiency_1h >= .50) & (frame.trade_return_1h.abs() > frame.realized_vol_24h)
    deteriorating = (frame.path_efficiency_1h <= .25) | (np.sign(frame.trade_return_1h) != np.sign(frame.basis_decimal))
    liq_primary = liquidation_valid & (frame.liquidation_intensity_robust_z >= 2)
    liq_robust = liquidation_valid & (frame.liquidation_intensity_percentile >= .95)
    oi_reset = frame.oi_log_change_1h < 0
    modest = frame.trade_return_1h.abs() <= (2 * frame.realized_vol_24h)
    prior_direction = np.sign(frame.trade_return_1h.shift(1))
    completed_failure = (
        prior_direction.ne(0)
        & np.sign(frame.trade_log_return_5m).eq(-prior_direction)
        & np.sign(frame.mark_return_1h).eq(-prior_direction)
    )
    completed_purge = (
        np.sign(frame.trade_log_return_5m).ne(0)
        & np.sign(frame.mark_return_1h).eq(np.sign(frame.trade_log_return_5m))
    )
    states = {
        "kda01_primary_efficient": crowd_primary & efficient,
        "kda01_primary_deteriorating": crowd_primary & deteriorating,
        "kda01_primary_failure": crowd_primary.shift(1, fill_value=False) & deteriorating.shift(1, fill_value=False) & completed_failure,
        "kda01_robust_efficient": crowd_robust & efficient,
        "kda01_robust_deteriorating": crowd_robust & deteriorating,
        "kda01_robust_failure": crowd_robust.shift(1, fill_value=False) & deteriorating.shift(1, fill_value=False) & completed_failure,
        "kda02_primary_active_purge": liq_primary & oi_reset,
        "kda02_primary_completed": liq_primary.shift(1, fill_value=False) & oi_reset & completed_purge,
        "kda02_primary_oi_vacuum": oi_reset & modest & cohort_valid,
        "kda02_robust_active_purge": liq_robust & oi_reset,
        "kda02_robust_completed": liq_robust.shift(1, fill_value=False) & oi_reset & completed_purge,
    }
    attempt_map = {definition: (family, state, threshold) for family, definition, state, threshold in ATTEMPTS}
    rows: list[dict[str, Any]] = []
    for definition, mask in states.items():
        family, state, _ = attempt_map[definition]
        for index in frame.index[onset(mask)]:
            ts = frame.at[index, "timestamp_utc"]
            direction = "positive" if frame.at[index, "trade_return_1h"] >= 0 else "negative"
            row = {
                "family_id": family, "definition_id": definition, "attempt_id": definition,
                "symbol": symbol, "direction": direction, "state": state,
                "state_start": ts - pd.Timedelta(hours=1), "decision_ts": ts + pd.Timedelta(minutes=5),
                "feature_window_start": ts - pd.Timedelta(days=60), "feature_window_end": ts + pd.Timedelta(minutes=5),
                "semantic_contract_hash": semantic_hash, "analytics_data_manifest_hash": ANALYTICS_MANIFEST_HASH,
                "trade_and_mark_authority_hashes": refs, "cohort_version": COHORT_VERSION,
                "feature_version": FEATURE_VERSION, "generator_contract_hash": GENERATOR_CONTRACT_HASH,
                "source_path_refs": frame.at[index, "source_path_refs"],
                "protected_row_count": 0, "major_vs_alt": frame.at[index, "major_vs_alt"],
                "prior_day_liquidity_rank": frame.at[index, "prior_day_liquidity_rank"],
                "known_lifecycle_mask": frame.at[index, "known_lifecycle_mask"],
            }
            row["event_id"], row["economic_address"] = deterministic_event_identity(row)
            rows.append(row)
    events = pd.DataFrame(rows)
    feasibility = pd.DataFrame({
        "definition_id": ["kda03_basis_oi", "kda03_basis_no_price", "kda03_basis_liq_reset"],
        "family_id": "KDA03",
        "state": ["basis_expansion_with_OI_confirmation", "basis_expansion_without_price_confirmation", "extreme_basis_with_liquidation_and_OI_reset"],
        "symbol": symbol,
        "feasible_row_count": [
            int((cohort_valid & frame.basis_change_normalization_valid & (frame.basis_change_robust_z.abs() >= 2) & (frame.oi_log_change_1h > 0)).sum()),
            int((cohort_valid & frame.basis_change_normalization_valid & (frame.basis_change_robust_z.abs() >= 2) & (frame.trade_return_1h.abs() < frame.realized_vol_24h)).sum()),
            int((basis_valid & liquidation_valid & (frame.basis_level_robust_z.abs() >= 2) & (frame.liquidation_intensity_robust_z >= 2) & oi_reset).sum()),
        ],
    })
    return events, feasibility


def safe_old_family_overlap(event_paths: Sequence[Path], repository_root: Path) -> pd.DataFrame:
    families = {
        "RFBS": repository_root / "results/rebaseline/phase_kraken_rfbs_signal_state_repaired_screen_20260715_v1",
        "Backside": repository_root / "results/rebaseline/phase_kraken_backside_blowoff_signal_state_repaired_screen_20260715_v1",
        "failed_breakdown": repository_root / "results/rebaseline/phase_kraken_failed_breakdown_squeeze_reclaim_long_screen_20260716_v1",
        "LFBS": repository_root / "results/rebaseline/phase_kraken_lfbs_signal_state_repaired_screen_20260715_v1",
        "H43": repository_root / "results/rebaseline/phase_kraken_btc_led_delayed_alt_diffusion_long_screen_20260716_v1",
        "C01": repository_root / "results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_20260717_103227",
        "C02": repository_root / "docs/agent/task_archive/20260717_donch_bt_stage_3b_c02_leadership_generator_20260717_v1",
    }
    safe = {"symbol", "PF_symbol", "decision_ts", "dominant_bar_close_ts", "impulse_onset_ts", "event_id", "candidate_id", "economic_address", "entry_ts"}
    rows = []
    for family, root in families.items():
        if family == "C01":
            candidates = [root / "CAUSAL_EVENT_ANCHOR_FREEZE.parquet"]
        elif family == "C02":
            candidates = [root / "C02_IMPULSE_EVENT_TAPE.parquet"]
        else:
            candidates = sorted(path for path in root.rglob("*.parquet") if "event" in path.name.lower() or "candidate" in path.name.lower()) if root.exists() else []
        candidates = [path for path in candidates if path.is_file()]
        source = ""; selected_path: Path | None = None; time_col = ""
        for path in candidates:
            names = set(pq.ParquetFile(path).schema_arrow.names)
            symbol_col = "symbol" if "symbol" in names else "PF_symbol" if "PF_symbol" in names else None
            candidate_time = next((column for column in ("decision_ts", "dominant_bar_close_ts", "impulse_onset_ts", "entry_ts") if column in names), None)
            if symbol_col and candidate_time:
                cols = [column for column in (symbol_col, candidate_time, "event_id", "candidate_id", "economic_address") if column in names]
                if not set(cols).issubset(safe):
                    raise ValueError("unsafe old-family overlap projection")
                source = str(path); selected_path = path; time_col = candidate_time; break
        if selected_path is None:
            rows.append({"old_family": family, "source": source or "safe_identity_tape_unavailable", "old_rows": 0, "exact_symbol_time_overlaps": 0})
            continue
        event_list = ",".join("'" + str(path).replace("'", "''") + "'" for path in event_paths)
        old = str(selected_path).replace("'", "''")
        connection = duckdb.connect()
        protected = connection.execute(f"SELECT count(*) FROM read_parquet('{old}') WHERE {time_col} >= TIMESTAMPTZ '2026-01-01 00:00:00+00'").fetchone()[0]
        if protected:
            raise ValueError("protected old-family identity timestamp")
        old_rows = connection.execute(f"SELECT count(*) FROM read_parquet('{old}')").fetchone()[0]
        exact = connection.execute(f"SELECT count(*) FROM read_parquet([{event_list}], union_by_name=true) e INNER JOIN read_parquet('{old}') o ON e.symbol=o.{symbol_col} AND e.decision_ts=o.{time_col}").fetchone()[0]
        connection.close()
        rows.append({"old_family": family, "source": source, "old_rows": old_rows, "exact_symbol_time_overlaps": exact})
    return pd.DataFrame(rows)


def artifact_manifest(root: Path, local_cache: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "ARTIFACT_MANIFEST.json"):
        if "attempts" in path.relative_to(root).parts or "handoff" in path.relative_to(root).parts:
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path), "drive_eligible": path.suffix != ".parquet"})
    local = []
    # Manifest finalized reusable evidence only. Aborted attempts and DuckDB
    # scratch state remain preserved locally but are not authoritative outputs.
    for path in sorted(item for item in local_cache.rglob("*") if item.is_file()):
        if any(part.startswith("aborted_") or part == "duckdb_tmp" for part in path.parts):
            continue
        if path.suffix in {".duckdb", ".tmp"}:
            continue
        local.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {"task_id": TASK_ID, "files": files, "local_cache_files": local}
    payload["manifest_content_hash"] = stable_hash(payload)
    write_json(root / "ARTIFACT_MANIFEST.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    archive = Path("docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=archive)
    parser.add_argument("--cache", type=Path, default=Path("/opt/parquet/kraken_derivatives/analytics/stage8a_foundation_v1_exact"))
    parser.add_argument(
        "--semantic-decision",
        type=Path,
        default=archive / "received" / "DONCH_DECISION_Kraken_Analytics_Inferred_Semantics_2026-07-19_v1.json",
    )
    parser.add_argument("--analytics-manifest", type=Path, default=Path("/opt/testerdonch/results/rebaseline/phase_kraken_futures_analytics_stage7c_20260717_v1_20260717_204226/KRAKEN_ANALYTICS_DATA_MANIFEST.json"))
    parser.add_argument("--market-manifest", type=Path, default=Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv"))
    parser.add_argument("--liquidity-cohort", type=Path, default=Path("/opt/testerdonch/docs/agent/task_archive/20260717_donch_bt_stage_2c1_volume_authority_resume_20260717_v1/C01_DAILY_LIQUIDITY_MEMBERSHIP.parquet"))
    parser.add_argument("--lifecycle-source", type=Path, default=Path("docs/agent/task_archive/20260717_donch_bt_stage_2a1_c01_reference_panel_20260717_v1/sources/terminal_lifecycle/kraken_derivatives_delistings.body"))
    parser.add_argument("--repository-root", type=Path, default=Path("/opt/testerdonch"))
    parser.add_argument("--symbol-limit", type=int, default=0, help="Synthetic/development only; forbidden for authoritative execution")
    parser.add_argument("--tg-bot-token", default="")
    parser.add_argument("--tg-chat-id", default="")
    parser.add_argument("--tg-auto-chat", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); started = time.monotonic(); args.output.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    notifier = TelegramNotifier.from_args(args, run_label="Stage 8A Kraken derivatives-state")
    notifier.send("started", f"output={args.output}; economic_run=no; protected_access=no")
    decision, semantic_hash = load_semantic_decision(args.semantic_decision, expected_sha256=SEMANTIC_FILE_SHA256)
    manifest, finals = verify_analytics_manifest(args.analytics_manifest)
    wide = build_wide_analytics_cache(finals, args.cache)
    feature_contract = {
        "version": FEATURE_VERSION, "cohort_version": COHORT_VERSION, "grid": "completed_5m",
        "normalization": {"lookback_calendar_days": 60, "minimum_valid_days": 30, "minimum_expected_fraction": .70, "availability": "prior_UTC_day_distribution_only_current_intraday_value", "basis_daily_aggregation": "median", "liquidation_intensity_daily_aggregation": "max", "validity": "family_specific_required_feature_scales", "zero_mad": "fail_closed", "zero_or_nonfinite_ratio": "fail_closed"},
        "horizon_alignment": "exact_contiguous_five_minute_lag_required",
        "semantic_contract_hash": semantic_hash, "analytics_manifest_hash": ANALYTICS_MANIFEST_HASH,
        "outcomes_authorized": False,
    }
    feature_hash = stable_hash(feature_contract)
    attempts = register_attempts(semantic_hash, feature_hash)
    write_csv(args.output / "KDA_FAMILY_AND_ATTEMPT_REGISTER.csv", attempts)
    authority_rows = c01.load_safe_manifest(args.market_manifest)
    liquidity = pq.ParquetFile(args.liquidity_cohort).read().to_pandas()
    if liquidity.cohort_hash.nunique() != 1 or liquidity.cohort_hash.iloc[0] != COHORT_SOURCE_HASH or liquidity.rank_uses_current_day.any():
        raise ValueError("Stage 2C liquidity cohort mismatch")
    lifecycle = c01.load_known_lifecycle_invalidations(args.lifecycle_source)
    partition_by_symbol: dict[str, list[Path]] = {}
    for path in wide:
        partition_by_symbol.setdefault(path.parent.name.split("=", 1)[1], []).append(path)
    symbols = sorted((set(liquidity.loc[liquidity.asset_identity_eligible & liquidity.top_100_eligible, "symbol"]) | set(MAJORS)) & set(partition_by_symbol))
    if args.symbol_limit:
        symbols = symbols[:args.symbol_limit]
    parent_parts = []
    for parent_symbol, prefix in (("PF_XBTUSD", "BTC"), ("PF_ETHUSD", "ETH")):
        parent, _ = load_bar_frame(authority_rows, parent_symbol, "historical_trade_candles_5m")
        parent_parts.append(pd.DataFrame({
            "timestamp_utc": parent.timestamp_utc,
            f"{prefix}_return_1h": rolling_exact_return(parent.close, parent.timestamp_utc, 12),
            f"{prefix}_return_6h": rolling_exact_return(parent.close, parent.timestamp_utc, 72),
        }))
    parent_features = parent_parts[0].merge(parent_parts[1], on="timestamp_utc", how="inner", validate="one_to_one")
    cohort_hash = stable_hash({"version": COHORT_VERSION, "source_cohort_hash": COHORT_SOURCE_HASH, "symbols": symbols})
    event_paths: list[Path] = []; feasibility_paths: list[Path] = []; event_total = 0
    cache_rows = []; unavailable = []; coverage_rows = []
    for number, symbol in enumerate(symbols, 1):
        source_refs = stable_hash([str(path) for path in sorted(partition_by_symbol[symbol])])
        feature_dir = args.cache / "features" / f"symbol={symbol}"
        feature_path = feature_dir / "data.parquet"
        feature_manifest_path = feature_dir / "manifest.json"
        if feature_dir.exists():
            if not feature_manifest_path.is_file() or not feature_path.is_file():
                raise ValueError(f"incomplete feature partition fails closed: {symbol}")
            feature_manifest = json.loads(feature_manifest_path.read_text())
            expected = {"feature_contract_hash": feature_hash, "analytics_manifest_hash": ANALYTICS_MANIFEST_HASH, "source_path_refs": source_refs}
            if any(feature_manifest.get(key) != value for key, value in expected.items()) or sha256_file(feature_path) != feature_manifest.get("sha256"):
                raise ValueError(f"stale feature partition fails closed: {symbol}")
            features = pq.ParquetFile(feature_path).read().to_pandas()
            features["timestamp_utc"] = pd.to_datetime(features.timestamp_utc, utc=True)
            trade_ref, mark_ref = feature_manifest["trade_ref"], feature_manifest["mark_ref"]
        else:
            raw = pd.concat([pq.ParquetFile(path).read().to_pandas() for path in sorted(partition_by_symbol[symbol])], ignore_index=True)
            raw["timestamp_utc"] = pd.to_datetime(raw.timestamp_utc, utc=True)
            validate_rankable_times(raw.timestamp_utc)
            trade, trade_ref = load_bar_frame(authority_rows, symbol, "historical_trade_candles_5m")
            mark, mark_ref = load_bar_frame(authority_rows, symbol, "historical_mark_candles_5m")
            features = add_symbol_features(raw, trade, mark)
            features = features.merge(parent_features, on="timestamp_utc", how="left", validate="one_to_one")
            features["utc_day"] = features.timestamp_utc.dt.floor("D")
            daily = liquidity[liquidity.symbol.eq(symbol)][["utc_day", "rank", "top_100_eligible"]]
            features = features.merge(daily, on="utc_day", how="left", validate="many_to_one")
            features["major_vs_alt"] = "major" if symbol in MAJORS else "alt"
            features["prior_day_liquidity_rank"] = features["rank"]
            features["eligible"] = True if symbol in MAJORS else features.top_100_eligible.eq(True)
            features["known_lifecycle_mask"] = True
            for left, right in lifecycle.get(symbol, []):
                features.loc[features.timestamp_utc.between(left, right, inclusive="left"), "known_lifecycle_mask"] = False
            features["eligible"] &= features.known_lifecycle_mask
            features["source_path_refs"] = source_refs
            feature_columns = [column for column in features.columns if column not in {"oi_json"}]
            assert_no_outcomes(feature_columns)
            temp_dir = feature_dir.with_name(f".{feature_dir.name}.tmp")
            if temp_dir.exists():
                raise ValueError(f"stale temporary feature partition fails closed: {symbol}")
            temp_dir.mkdir(parents=True)
            temp_path = temp_dir / "data.parquet"
            features[feature_columns].to_parquet(temp_path, index=False, compression="zstd")
            feature_manifest = {
                "symbol": symbol, "feature_contract_hash": feature_hash,
                "analytics_manifest_hash": ANALYTICS_MANIFEST_HASH, "source_path_refs": source_refs,
                "trade_ref": trade_ref, "mark_ref": mark_ref, "rows": len(features),
                "sha256": sha256_file(temp_path), "status": "complete",
            }
            write_json(temp_dir / "manifest.json", feature_manifest)
            os.replace(temp_dir, feature_dir)
        refs = stable_hash({"trade": trade_ref, "mark": mark_ref})
        event_dir = args.cache / "events" / f"symbol={symbol}"
        event_path = event_dir / "events.parquet"; feasibility_path = event_dir / "kda03.csv"; event_manifest_path = event_dir / "manifest.json"
        if event_dir.exists():
            if not event_manifest_path.is_file() or not feasibility_path.is_file():
                raise ValueError(f"incomplete event partition fails closed: {symbol}")
            event_manifest = json.loads(event_manifest_path.read_text())
            if (event_manifest.get("feature_sha256") != sha256_file(feature_path)
                    or event_manifest.get("semantic_contract_hash") != semantic_hash
                    or event_manifest.get("generator_contract_hash") != GENERATOR_CONTRACT_HASH):
                raise ValueError(f"stale event partition fails closed: {symbol}")
            if int(event_manifest["event_count"]) and (not event_path.is_file() or sha256_file(event_path) != event_manifest.get("event_sha256")):
                raise ValueError(f"event partition hash mismatch: {symbol}")
        else:
            events, kda03 = generate_events(features, symbol, refs, semantic_hash, feature_hash)
            if not events.empty:
                events = cluster_canonical_episodes(events)
                if events.event_id.duplicated().any() or events.economic_address.duplicated().any():
                    raise ValueError(f"duplicate KDA identity within symbol: {symbol}")
                validate_rankable_times(events.decision_ts)
            temp_dir = event_dir.with_name(f".{event_dir.name}.tmp")
            if temp_dir.exists():
                raise ValueError(f"stale temporary event partition fails closed: {symbol}")
            temp_dir.mkdir(parents=True)
            temp_event_path = temp_dir / "events.parquet"
            if not events.empty:
                events.to_parquet(temp_event_path, index=False, compression="zstd")
            kda03.to_csv(temp_dir / "kda03.csv", index=False, lineterminator="\n")
            event_manifest = {
                "symbol": symbol, "feature_sha256": sha256_file(feature_path),
                "semantic_contract_hash": semantic_hash, "generator_contract_hash": GENERATOR_CONTRACT_HASH,
                "event_count": len(events),
                "event_sha256": sha256_file(temp_event_path) if not events.empty else "",
                "kda03_sha256": sha256_file(temp_dir / "kda03.csv"), "status": "complete",
            }
            write_json(temp_dir / "manifest.json", event_manifest)
            os.replace(temp_dir, event_dir)
        if int(event_manifest["event_count"]):
            event_paths.append(event_path)
        feasibility_paths.append(feasibility_path)
        event_total += int(event_manifest["event_count"])
        cache_rows.append({"symbol": symbol, "path": str(feature_path), "rows": len(features), "sha256": sha256_file(feature_path), "trade_ref": trade_ref, "mark_ref": mark_ref, "manifest_path": str(feature_manifest_path)})
        for year, group in features.groupby(features.timestamp_utc.dt.year):
            coverage_rows.append({"year": int(year), "symbol": symbol, "exact_aligned_rows": len(group), "trade_coverage_rows": int(group.trade_coverage.sum()), "mark_coverage_rows": int(group.mark_coverage.sum()), "analytics_coverage_rows": int(group.analytics_coverage.sum()), "eligible_rows": int(group.eligible.sum()), "first_usable_ts": group.timestamp_utc.min(), "last_usable_ts": group.timestamp_utc.max()})
        missing = (~features.eligible).sum()
        unavailable.append({"symbol": symbol, "reason": "cohort_or_known_lifecycle_ineligible", "row_count": int(missing)})
        write_json(args.output / "watch_status.json", {"stage": "features_and_events", "symbols_completed": number, "symbols_total": len(symbols), "event_rows": event_total, "elapsed_seconds": time.monotonic() - started, "peak_rss_gib": rss_gib()})
        print(f"[{number}/{len(symbols)}] {symbol}: features={len(features)} events={event_manifest['event_count']}", flush=True)
        if number % 10 == 0 or number == len(symbols):
            notifier.send("progress", f"symbols={number}/{len(symbols)}; events={event_total}; rss={rss_gib():.2f} GiB")
    if not event_paths:
        raise ValueError("all KDA01/KDA02 branches produced zero events")
    event_list = ",".join("'" + str(path).replace("'", "''") + "'" for path in event_paths)
    reducer = duckdb.connect(str(args.cache / "stage8a_reducer.duckdb")); reducer.execute("SET memory_limit='1GB'"); reducer.execute("SET threads=2")
    source = f"read_parquet([{event_list}], union_by_name=true)"
    duplicate_ids = reducer.execute(f"SELECT count(*) FROM (SELECT event_id FROM {source} GROUP BY event_id HAVING count(*)>1)").fetchone()[0]
    duplicate_addresses = reducer.execute(f"SELECT count(*) FROM (SELECT economic_address FROM {source} GROUP BY economic_address HAVING count(*)>1)").fetchone()[0]
    if duplicate_ids or duplicate_addresses:
        raise ValueError("duplicate KDA identities across symbol shards")
    for family, output_name, count_name in (("KDA01", "KDA01_EVENT_TAPE.parquet", "KDA01_EVENT_COUNT_MATRIX.csv"), ("KDA02", "KDA02_EVENT_TAPE.parquet", "KDA02_EVENT_COUNT_MATRIX.csv")):
        reducer.execute(f"COPY (SELECT * FROM {source} WHERE family_id='{family}' ORDER BY symbol,decision_ts,definition_id) TO '{(args.output / output_name).as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
        matrix = reducer.execute(f"SELECT year(decision_ts) AS year,symbol,major_vs_alt,direction,family_id,definition_id,state,count(*) AS event_count FROM {source} WHERE family_id='{family}' GROUP BY ALL ORDER BY ALL").fetchdf()
        write_csv(args.output / count_name, matrix)
    definition_counts = reducer.execute(f"SELECT definition_id,count(*) AS event_count FROM {source} GROUP BY definition_id").fetchdf().set_index("definition_id").event_count
    episode_count = reducer.execute(f"SELECT count(DISTINCT canonical_episode_id) FROM {source}").fetchone()[0]
    kda01_count = reducer.execute(f"SELECT count(*) FROM {source} WHERE family_id='KDA01'").fetchone()[0]
    kda02_count = reducer.execute(f"SELECT count(*) FROM {source} WHERE family_id='KDA02'").fetchone()[0]
    reducer.close()
    kda03 = pd.concat([pd.read_csv(path) for path in feasibility_paths], ignore_index=True); write_csv(args.output / "KDA03_FEASIBILITY_MATRIX.csv", kda03)
    attempts["event_count"] = attempts.definition_id.map(definition_counts).fillna(0).astype(int)
    attempts.loc[attempts.family_id.eq("KDA03"), "event_count"] = attempts.loc[attempts.family_id.eq("KDA03"), "definition_id"].map(kda03.groupby("definition_id").feasible_row_count.sum()).fillna(0).astype(int)
    attempts["zero_count_branch"] = attempts.event_count.eq(0); write_csv(args.output / "KDA_FAMILY_AND_ATTEMPT_REGISTER.csv", attempts)
    write_csv(args.output / "KDA_UNAVAILABLE_FEATURE_COUNTS.csv", pd.DataFrame(unavailable))
    overlap = safe_old_family_overlap(event_paths, args.repository_root); write_csv(args.output / "KDA_OLD_FAMILY_OVERLAP.csv", overlap)
    one_minute = one_minute_aggregation_diagnostics(finals); write_csv(args.output / "KDA_ONE_MINUTE_TIMING_DIAGNOSTICS.csv", one_minute)
    write_json(args.output / "ANALYTICS_SEMANTIC_CONTRACT.json", decision)
    write_json(args.output / "KDA_GENERATOR_CONTRACT.json", {**GENERATOR_CONTRACT, "generator_contract_hash": GENERATOR_CONTRACT_HASH})
    write_json(args.output / "KDA_SHARED_FEATURE_SCHEMA.json", {"feature_version": FEATURE_VERSION, "feature_contract_hash": feature_hash, "columns": sorted(set(column for row in cache_rows for column in pq.ParquetFile(row["path"]).schema_arrow.names)), "outcome_columns": []})
    write_json(args.output / "KDA_FEATURE_CACHE_MANIFEST.json", {"feature_contract_hash": feature_hash, "analytics_manifest_hash": ANALYTICS_MANIFEST_HASH, "cohort_version": COHORT_VERSION, "cohort_hash": cohort_hash, "partitions": cache_rows, "protected_rows_opened": 0})
    write_csv(args.output / "KDA_DATA_AUTHORITY_AND_COVERAGE.csv", pd.DataFrame(cache_rows))
    write_csv(args.output / "KDA_ANALYTICS_COVERAGE_BY_YEAR_SYMBOL.csv", pd.DataFrame(coverage_rows))
    (args.output / "ANALYTICS_SEMANTIC_IMPLEMENTATION.md").write_text(f"# Analytics Semantic Implementation\n\nSemantic contract hash: `{semantic_hash}`. Status: `{SEMANTIC_STATUS}`. Decimal source strings are retained. Basis is last-state decimal ratio; OI is unsigned OHLC base-unit quantity; liquidation is unsigned summed base-unit flow.\n", encoding="utf-8")
    (args.output / "ANALYTICS_DATA_AUTHORITY_AND_COVERAGE.md").write_text(f"# Analytics Data Authority and Coverage\n\nStage 7C manifest content hash: `{ANALYTICS_MANIFEST_HASH}`. Verified final objects: `1836`. Cohort: `{COHORT_VERSION}`. Symbols processed: `{len(symbols)}`. This remains current-roster capped, not survivorship-free, and does not claim continuous tradeability. Protected rows opened: `0`.\n", encoding="utf-8")
    (args.output / "KDA_SHARED_FEATURE_CONTRACT.md").write_text(f"# KDA Shared Feature Contract\n\nHash: `{feature_hash}`. Completed five-minute grid. Daily robust normalization uses only the prior 60 UTC days, minimum 30 valid days and 70% expected observations. Zero MAD and missing exact trade/mark/analytics rows fail closed. No outcome field is produced.\n", encoding="utf-8")
    (args.output / "KDA_CANONICAL_EPISODE_REPORT.md").write_text(f"# KDA Canonical Episodes\n\nEvents: `{event_total}`. Family-neutral same-symbol overlapping causal episodes: `{episode_count}`. Outcome values are not inputs.\n", encoding="utf-8")
    status = "complete" if rss_gib() < 4 else "blocked_resource_limit"
    (args.output / "KDA_GENERATOR_REVIEW.md").write_text(f"# KDA Generator Review\n\nStatus: `{status}`. KDA01 events: `{kda01_count}`. KDA02 events: `{kda02_count}`. KDA03 feasibility rows: `{int(kda03.feasible_row_count.sum())}`. Peak RSS: `{rss_gib():.3f} GiB`. Protected rows and economic outputs: `0/0`.\n", encoding="utf-8")
    (args.output / "KDA_NEXT_ECONOMIC_CONTRACT_RECOMMENDATION.md").write_text("# Next Economic Contract Recommendation\n\nSubject to human approval after review: (1) KDA01 crowding-price-progress; (2) KDA02 liquidation/OI purge. Ranking uses causal clarity, coverage, counts, identity quality, distinctness and mechanical falsifiability only. No returns were computed.\n", encoding="utf-8")
    write_json(args.output / "completion_summary.json", {"status": status, "semantic_contract_hash": semantic_hash, "analytics_data_manifest_hash": ANALYTICS_MANIFEST_HASH, "cohort_version": COHORT_VERSION, "cohort_hash": cohort_hash, "feature_contract_hash": feature_hash, "generator_contract_hash": GENERATOR_CONTRACT_HASH, "KDA01_events": kda01_count, "KDA02_events": kda02_count, "KDA03_feasible_rows": int(kda03.feasible_row_count.sum()), "episodes": episode_count, "peak_rss_gib": rss_gib(), "one_minute_aggregation_mismatches": int(one_minute.aggregation_mismatches.sum()), "protected_rows_opened": 0, "economic_outputs_computed": 0, "runtime_seconds": time.monotonic()-started})
    artifact_manifest(args.output, args.cache)
    if status != "complete": raise RuntimeError(status)
    notifier.send("complete", f"KDA01={kda01_count}; KDA02={kda02_count}; KDA03_feasible={int(kda03.feasible_row_count.sum())}; rss={rss_gib():.2f} GiB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        failure_args = parse_args()
        TelegramNotifier.from_args(failure_args, run_label="Stage 8A Kraken derivatives-state").send(
            "failed", f"{type(exc).__name__}: {exc}"
        )
        raise
