from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import tools.kraken_funding_imputation as model_lib
import tools.run_kraken_family_engine_aggregate_first_sweep as runner


DEFAULT_RUN_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
EXACT_ROOT = Path("results/rebaseline/phase_kraken_exact_funding_coverage_acquisition_audit_20260709_v1")
A1_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_funding_policy_universe_repair_20260709_v1")
TSMOM_SELECTED_ROOT = Path("results/rebaseline/phase_kraken_tsmom_mask_selected_key_builder_20260707_v1_20260707_201216")
TSMOM_OUTCOME_ROOT = Path("results/rebaseline/phase_kraken_tsmom_outcome_grouped_aggregate_20260707_v1")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    frame.to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def load_required_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    a1_files = sorted((A1_ROOT / "aggregate_shards").glob("*/outcome_events.parquet"))
    if not a1_files:
        raise RuntimeError("A1 outcome ledgers are missing")
    a1 = pd.concat([pd.read_parquet(path, columns=["symbol", "entry_ts", "exit_interval_end_ts"]) for path in a1_files], ignore_index=True)
    a1 = a1.rename(columns={"exit_interval_end_ts": "interval_end_ts"})
    a1["family_requirement_source"] = "a1_compression_repaired_first_pack_outcomes"
    tsmom_path = TSMOM_OUTCOME_ROOT / "cache/tsmom_interval_outcome.parquet"
    if not tsmom_path.exists():
        raise RuntimeError("TSMOM interval outcomes are missing")
    tsmom = pd.read_parquet(tsmom_path, columns=["symbol", "entry_ts", "exit_interval_end_ts"])
    tsmom = tsmom.rename(columns={"exit_interval_end_ts": "interval_end_ts"})
    tsmom["family_requirement_source"] = "tsmom_selected_key_outcome_intervals"
    selected_manifest = TSMOM_SELECTED_ROOT / "cache/selected_event_key_manifest.csv"
    if not selected_manifest.exists():
        raise RuntimeError("TSMOM selected-key lineage manifest is missing")
    for frame in (a1, tsmom):
        frame["entry_ts"] = pd.to_datetime(frame["entry_ts"], utc=True, errors="coerce")
        frame["interval_end_ts"] = pd.to_datetime(frame["interval_end_ts"], utc=True, errors="coerce")
    return a1, tsmom


def required_hourly_boundaries(events: pd.DataFrame) -> pd.DataFrame:
    by_symbol: dict[str, set[pd.Timestamp]] = {}
    for row in events.itertuples(index=False):
        start = pd.Timestamp(row.entry_ts)
        end = pd.Timestamp(row.interval_end_ts)
        if pd.isna(start) or pd.isna(end) or end <= start:
            continue
        first = start.floor("h") + pd.Timedelta(hours=1)
        last = end.floor("h")
        if first > last:
            continue
        by_symbol.setdefault(str(row.symbol), set()).update(pd.date_range(first, last, freq="h", tz="UTC" if first.tzinfo is None else None).tolist())
    rows = [{"symbol": symbol, "timestamp": ts} for symbol, timestamps in sorted(by_symbol.items()) for ts in sorted(timestamps) if ts < model_lib.PROTECTED_TRAIN_BOUNDARY]
    return pd.DataFrame(rows)


def load_exact_rates(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = runner.local_data_paths_from_root(runner.DEFAULT_KRAKEN_DATA_ROOT)
    frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        files = runner.find_symbol_files(paths["funding"], symbol)
        for path in files:
            try:
                frame = pd.read_parquet(path, columns=["timestamp", "relativeFundingRate", "fundingRate", "venue_symbol"])
            except Exception:
                continue
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame["relativeFundingRate"] = pd.to_numeric(frame["relativeFundingRate"], errors="coerce")
            frame["symbol"] = symbol
            frame = frame[(frame["timestamp"] < model_lib.PROTECTED_TRAIN_BOUNDARY) & frame["relativeFundingRate"].notna()]
            if not frame.empty:
                frames.append(frame[["symbol", "timestamp", "relativeFundingRate"]])
            source_rows.append({"symbol": symbol, "path": str(path), "sha256": file_hash(path), "rows_pre_cutoff": int(len(frame)), "has_relativeFundingRate": "relativeFundingRate" in frame.columns})
    if not frames:
        raise RuntimeError("no exact relativeFundingRate rows found")
    exact = pd.concat(frames, ignore_index=True)
    exact = exact.sort_values(["symbol", "timestamp"], kind="mergesort").drop_duplicates(["symbol", "timestamp"], keep="last").reset_index(drop=True)
    return exact, pd.DataFrame(source_rows)


def load_daily_market_features(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    paths = runner.local_data_paths_from_root(runner.DEFAULT_KRAKEN_DATA_ROOT)
    daily_frames: list[pd.DataFrame] = []
    load_start = start - pd.Timedelta(days=40)
    for symbol in symbols:
        files = runner.pre_holdout_files(runner.find_symbol_files(paths["trade_5m"], symbol) or runner.find_symbol_files(paths["alt_trade"], symbol))
        chunks: list[pd.DataFrame] = []
        for path in files:
            frame = pd.DataFrame()
            for time_col in ("time", "timestamp"):
                try:
                    frame = pd.read_parquet(path, columns=[time_col, "close", "volume"])
                    break
                except Exception:
                    frame = pd.DataFrame()
            if frame.empty:
                continue
            time_col = "time" if "time" in frame.columns else "timestamp"
            frame["timestamp"] = runner.parse_time_ms_or_iso(frame[time_col])
            frame = frame[(frame["timestamp"] >= load_start) & (frame["timestamp"] < min(end + pd.Timedelta(days=1), model_lib.PROTECTED_TRAIN_BOUNDARY))]
            if frame.empty:
                continue
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
            chunks.append(frame[["timestamp", "close", "volume"]])
        if not chunks:
            continue
        bars = pd.concat(chunks, ignore_index=True).dropna(subset=["timestamp", "close"]).sort_values("timestamp", kind="mergesort").drop_duplicates("timestamp", keep="last")
        bars["date"] = bars["timestamp"].dt.floor("D")
        bars["quote_volume_proxy"] = bars["close"] * bars["volume"]
        daily = bars.groupby("date", sort=True).agg(close=("close", "last"), quote_volume_proxy=("quote_volume_proxy", "sum")).reset_index()
        daily["symbol"] = symbol
        daily["return_20d"] = daily["close"].pct_change(20, fill_method=None)
        daily["realized_vol_20d"] = daily["close"].pct_change(fill_method=None).rolling(20, min_periods=15).std() * np.sqrt(365.0)
        daily["trailing_quote_volume_30d"] = daily["quote_volume_proxy"].rolling(30, min_periods=10).median()
        daily_frames.append(daily)
    if not daily_frames:
        return pd.DataFrame(columns=["symbol", "date", "liquidity_tier", "realized_vol_state", "return_state", "parent_regime"])
    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_daily["liquidity_percentile"] = all_daily.groupby("date")["trailing_quote_volume_30d"].rank(method="first", pct=True, ascending=False)
    all_daily["liquidity_tier"] = np.select([all_daily["liquidity_percentile"] <= 0.20, all_daily["liquidity_percentile"] <= 0.50], ["tier_a", "tier_b"], default="tier_c_or_unknown")
    all_daily["realized_vol_state"] = np.select([all_daily["realized_vol_20d"] < 0.50, all_daily["realized_vol_20d"] < 1.00], ["low", "medium"], default="high_or_unknown")
    all_daily["return_state"] = np.select([all_daily["return_20d"] < -0.10, all_daily["return_20d"] > 0.10], ["down", "up"], default="neutral_or_unknown")
    parent = all_daily[all_daily["symbol"].isin(["PF_XBTUSD", "PF_ETHUSD"])].pivot(index="date", columns="symbol", values="return_20d")
    if {"PF_XBTUSD", "PF_ETHUSD"}.issubset(parent.columns):
        parent["parent_regime"] = np.select([(parent["PF_XBTUSD"] > 0) & (parent["PF_ETHUSD"] > 0), (parent["PF_XBTUSD"] < 0) & (parent["PF_ETHUSD"] < 0)], ["both_up", "both_down"], default="mixed_or_unknown")
        all_daily = all_daily.merge(parent[["parent_regime"]].reset_index(), on="date", how="left")
    else:
        all_daily["parent_regime"] = "unknown"
    return all_daily[["symbol", "date", "liquidity_tier", "parent_regime", "realized_vol_state", "return_state", "trailing_quote_volume_30d", "return_20d", "realized_vol_20d"]]


def attach_daily_features(rows: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["date"] = out["timestamp"].dt.floor("D")
    out = out.merge(daily, on=["symbol", "date"], how="left", validate="many_to_one")
    for col in ["liquidity_tier", "parent_regime", "realized_vol_state", "return_state"]:
        out[col] = out[col].fillna("unknown")
    return out.drop(columns=["date"])


def premium_availability_audit(symbols: list[str], exact: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    paths = runner.local_data_paths_from_root(runner.DEFAULT_KRAKEN_DATA_ROOT)
    rows = []
    for symbol in symbols:
        mark_files = runner.find_symbol_files(paths["mark_5m"], symbol)
        rows.append({"symbol": symbol, "historical_mark_files": len(mark_files), "historical_index_files": 0, "true_mark_index_premium_available": False, "reason": "historical mark candles exist for some symbols but no historical index-price dataset is present; mark-minus-trade is not treated as venue premium"})
    return False, pd.DataFrame(rows)


def aggregate_validation(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (validation, model), group in predictions.groupby(["validation", "model"], sort=True):
        rows.append({"validation": validation, "model": model, **model_lib.prediction_metrics(group["actual"], group["predicted"])})
    return pd.DataFrame(rows)


def write_partitioned_panel(run_root: Path, panel: pd.DataFrame) -> pd.DataFrame:
    panel_root = run_root / "funding/shared_funding_panel"
    panel_root.mkdir(parents=True, exist_ok=True)
    work = panel.copy()
    work["year_month"] = pd.to_datetime(work["timestamp"], utc=True).dt.strftime("%Y-%m")
    rows = []
    for year_month, part in work.groupby("year_month", sort=True):
        rel = Path(f"year_month={year_month}/part.parquet")
        path = panel_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        part = part.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
        tmp = path.with_suffix(".parquet.tmp")
        part.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        rows.append({"partition": year_month, "path": str(Path("funding/shared_funding_panel") / rel), "row_count": len(part), "min_timestamp": part["timestamp"].min(), "max_timestamp": part["timestamp"].max(), "content_hash": model_lib.canonical_frame_hash(part), "file_sha256": file_hash(path), "exact_rows": int(part["funding_exact"].sum()), "imputed_rows": int(part["funding_imputed"].sum()), "status": "pass"})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    args = parser.parse_args()
    run_root = Path(args.run_root)
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"run root already exists and is nonempty: {run_root}")
    started = time.monotonic()
    for rel in ["funding", "audit", "compact_review_bundle"]:
        (run_root / rel).mkdir(parents=True, exist_ok=True)

    a1, tsmom = load_required_events()
    all_events = pd.concat([a1, tsmom], ignore_index=True)
    boundaries = required_hourly_boundaries(all_events)
    symbols = sorted(boundaries["symbol"].unique())
    exact, exact_sources = load_exact_rates(symbols)
    daily = load_daily_market_features(symbols, boundaries["timestamp"].min(), boundaries["timestamp"].max())
    exact_features = attach_daily_features(exact, daily)
    boundary_features = attach_daily_features(boundaries, daily)
    model_lib.assert_no_strategy_outcome_features(exact_features.columns)

    premium_usable, premium_audit = premium_availability_audit(symbols, exact)
    blocked, blocked_predictions = model_lib.expanding_blocked_time_validation(exact_features)
    exact_features["year_month"] = exact_features["timestamp"].dt.strftime("%Y-%m")
    leave_month, leave_month_predictions = model_lib.leave_group_out_validation(exact_features, "year_month")
    leave_symbol, leave_symbol_predictions = model_lib.leave_group_out_validation(exact_features, "symbol")
    selected_name, comparison = model_lib.select_central_model(blocked_predictions, leave_symbol_predictions)
    selected_model = model_lib.fit_funding_model(exact_features, selected_name)
    selected_oof = blocked_predictions[blocked_predictions["model"] == selected_name]
    abs_residual = (selected_oof["predicted"] - selected_oof["actual"]).abs()
    residual_quantiles = (float(abs_residual.quantile(0.75)), float(abs_residual.quantile(0.95)))
    panel = model_lib.build_funding_scenarios(boundary_features, exact, boundary_features, selected_model, residual_quantiles)
    exact_audit = model_lib.exact_rows_unchanged_audit(panel)

    baseline = comparison[comparison["model"] == "global_robust_median"].iloc[0]
    selected = comparison[comparison["model"] == selected_name].iloc[0]
    central_gate = bool(
        float(selected["blocked_mae"]) <= float(baseline["blocked_mae"]) * 1.001
        and abs(float(selected["blocked_aggregate_bias_abs_rate_share"])) <= max(abs(float(baseline["blocked_aggregate_bias_abs_rate_share"])) * 1.05, 0.01)
    )
    partition_manifest = write_partitioned_panel(run_root, panel)

    training_manifest = pd.DataFrame([
        {"artifact": "required_A1_intervals", "path": str(A1_ROOT / "aggregate_shards/*/outcome_events.parquet"), "rows": len(a1), "symbols": a1["symbol"].nunique(), "min_entry_ts": a1["entry_ts"].min(), "max_interval_end_ts": a1["interval_end_ts"].max(), "content_hash": model_lib.canonical_frame_hash(a1), "strategy_outcome_fields_used": False},
        {"artifact": "required_TSMOM_intervals", "path": str(TSMOM_OUTCOME_ROOT / "cache/tsmom_interval_outcome.parquet"), "rows": len(tsmom), "symbols": tsmom["symbol"].nunique(), "min_entry_ts": tsmom["entry_ts"].min(), "max_interval_end_ts": tsmom["interval_end_ts"].max(), "content_hash": model_lib.canonical_frame_hash(tsmom), "strategy_outcome_fields_used": False},
        {"artifact": "exact_relativeFundingRate_training_rows", "path": str(runner.DEFAULT_KRAKEN_DATA_ROOT / "parquet/funding"), "rows": len(exact_features), "symbols": exact_features["symbol"].nunique(), "min_entry_ts": exact_features["timestamp"].min(), "max_interval_end_ts": exact_features["timestamp"].max(), "content_hash": model_lib.canonical_frame_hash(exact_features[[c for c in exact_features.columns if c not in {"year_month"}]]), "strategy_outcome_fields_used": False},
        {"artifact": "required_unique_hourly_boundaries", "path": "derived_union_A1_TSMOM_event_intervals", "rows": len(boundaries), "symbols": len(symbols), "min_entry_ts": boundaries["timestamp"].min(), "max_interval_end_ts": boundaries["timestamp"].max(), "content_hash": model_lib.canonical_frame_hash(boundaries), "strategy_outcome_fields_used": False},
    ])
    write_csv(run_root / "funding/model_training_dataset_manifest.csv", training_manifest)
    write_csv(run_root / "funding/model_comparison.csv", comparison)
    write_csv(run_root / "funding/blocked_time_validation.csv", blocked)
    write_csv(run_root / "funding/leave_month_out_validation.csv", leave_month)
    write_csv(run_root / "funding/leave_symbol_out_validation.csv", leave_symbol)
    residual_rows = []
    for source in [blocked_predictions, leave_month_predictions, leave_symbol_predictions]:
        for (validation, model), group in source.groupby(["validation", "model"], sort=True):
            residual = group["predicted"] - group["actual"]
            residual_rows.append({"validation": validation, "model": model, "rows": len(group), "residual_q01": residual.quantile(0.01), "residual_q05": residual.quantile(0.05), "residual_q25": residual.quantile(0.25), "residual_median": residual.median(), "residual_q75": residual.quantile(0.75), "residual_q95": residual.quantile(0.95), "residual_q99": residual.quantile(0.99), "mean_residual": residual.mean(), "mean_absolute_residual": residual.abs().mean()})
    write_csv(run_root / "funding/imputation_residual_distribution.csv", residual_rows)
    write_csv(run_root / "funding/shared_funding_panel_manifest.csv", partition_manifest)
    write_csv(run_root / "funding/exact_source_manifest.csv", exact_sources)
    write_csv(run_root / "funding/premium_availability_audit.csv", premium_audit)
    write_csv(run_root / "audit/exact_rows_unchanged_audit.csv", exact_audit)

    leak_audit = pd.DataFrame([
        {"audit": "forbidden_strategy_outcome_features", "checked_fields": ";".join(sorted(model_lib.FORBIDDEN_OUTCOME_FEATURES)), "violation_count": 0, "pass": True, "detail": "model inputs are exact funding and PIT market-state features only"},
        {"audit": "protected_train_boundary", "checked_fields": "timestamp", "violation_count": int((panel["timestamp"] >= model_lib.PROTECTED_TRAIN_BOUNDARY).sum()), "pass": bool((panel["timestamp"] < model_lib.PROTECTED_TRAIN_BOUNDARY).all()), "detail": str(model_lib.PROTECTED_TRAIN_BOUNDARY)},
        {"audit": "funding_gate_separation", "checked_fields": "funding_gate_eligible;funding_gate_use_policy", "violation_count": int((panel["funding_imputed"] & panel["funding_gate_eligible"]).sum()), "pass": bool(not (panel["funding_imputed"] & panel["funding_gate_eligible"]).any()), "detail": "imputed rates are outcome-cost only and cannot activate gates"},
    ])
    write_csv(run_root / "audit/no_strategy_outcome_leak_audit.csv", leak_audit)
    write_text(run_root / "funding/model_feature_contract.md", """# Shared Kraken Funding Imputation Model Feature Contract

The target is exact hourly `relativeFundingRate`. Inputs are symbol identity, point-in-time trailing liquidity tier, BTC/ETH parent return regime, symbol realized-volatility state, and symbol trailing-return state. Market-state features are built from completed trade bars available on or before the funding boundary date. Strategy returns, event PnL, MAE/MFE, candidate rank, and candidate performance are forbidden.

The fitted model is retrospective train-screen cost imputation. It is not a decision input and is never consumed by funding gates. Exact rows bypass prediction and remain unchanged. Every imputed row carries `funding_imputed_train_screen_cap`.

Historical mark candles exist, but no historical index-price dataset was found for the required symbols and interval. Mark-minus-trade is not silently substituted for venue mark/index premium, so the premium-derived candidate is unavailable in v1.
""")
    write_text(run_root / "funding/funding_scenario_contract.md", f"""# Funding Scenario Contract

- Central: `{selected_name}` estimate for missing rows; exact `relativeFundingRate` for exact rows.
- Conservative: central plus the blocked-time absolute residual 75th percentile for long-cost scenarios. Use `funding_rate_conservative_short` for short positions.
- Severe: central plus the blocked-time absolute residual 95th percentile for long-cost scenarios. Use `funding_rate_severe_short` for short positions.
- Exact rows are identical in every scenario because their boundary cost is known.
- The legacy `-0.05R` proxy is excluded from panel base modes. It remains only a separately named extreme event-level stress outside this panel.
- Imputation cannot activate, satisfy, or block a historical funding gate.

Residual magnitudes: q75={residual_quantiles[0]:.12g}, q95={residual_quantiles[1]:.12g}.
""")

    exact_rows_in_panel = int(panel["funding_exact"].sum())
    imputed_rows = int(panel["funding_imputed"].sum())
    status = "complete" if bool(exact_audit.iloc[0]["pass"]) and not int(leak_audit["violation_count"].sum()) and central_gate else "blocked_model_gate"
    summary = {
        "run_root": str(run_root),
        "status": status,
        "code_modified": True,
        "strategy_aggregate_shards_launched": False,
        "signals_rebuilt": False,
        "materialization_controls_validation_holdout_launched": False,
        "exact_rows_used_for_training": int(len(exact_features)),
        "exact_rows_in_required_panel": exact_rows_in_panel,
        "selected_central_model": selected_name,
        "selected_model_hash": selected_model.model_hash,
        "premium_derived_model_usable": premium_usable,
        "blocked_time_validation_pass": central_gate,
        "leave_symbol_out_rows": int(len(leave_symbol)),
        "blocked_time_selected_model_metrics": {k.replace("blocked_", ""): selected[k] for k in selected.index if str(k).startswith("blocked_")},
        "leave_symbol_selected_model_metrics": {k.replace("leave_symbol_", ""): selected[k] for k in selected.index if str(k).startswith("leave_symbol_")},
        "aggregate_bias_versus_exact_funding": float(selected["blocked_aggregate_bias"]),
        "aggregate_bias_abs_rate_share": float(selected["blocked_aggregate_bias_abs_rate_share"]),
        "panel_rows": int(len(panel)),
        "central_panel_rows": int(len(panel)),
        "conservative_panel_rows": int(len(panel)),
        "severe_panel_rows": int(len(panel)),
        "imputed_rows": imputed_rows,
        "exact_rows_unchanged": bool(exact_audit.iloc[0]["pass"]),
        "confidence_cap_policy": model_lib.IMPUTED_CAP,
        "reusable_by_A1": status == "complete",
        "reusable_by_TSMOM": status == "complete",
        "protected_train_boundary": str(model_lib.PROTECTED_TRAIN_BOUNDARY),
        "runtime_seconds": time.monotonic() - started,
        "next_recommended_phase": "integrate_shared_funding_panel_outcome_cost_mode_dry_run_next" if status == "complete" else "repair_shared_funding_model_validation_next",
        "compact_bundle_path": str(run_root / "compact_review_bundle"),
    }
    write_json(run_root / "decision_summary.json", summary)

    bundle_files = [
        "funding/model_training_dataset_manifest.csv", "funding/model_feature_contract.md", "funding/model_comparison.csv",
        "funding/blocked_time_validation.csv", "funding/leave_month_out_validation.csv", "funding/leave_symbol_out_validation.csv",
        "funding/imputation_residual_distribution.csv", "funding/shared_funding_panel_manifest.csv", "funding/funding_scenario_contract.md",
        "funding/premium_availability_audit.csv", "audit/no_strategy_outcome_leak_audit.csv", "audit/exact_rows_unchanged_audit.csv",
        "decision_summary.json",
    ]
    bundle_rows = []
    for rel in bundle_files:
        src = run_root / rel
        dst = run_root / "compact_review_bundle" / rel.replace("/", "__")
        shutil.copy2(src, dst)
        bundle_rows.append({"source": rel, "bundle_path": str(dst.relative_to(run_root)), "sha256": file_hash(dst), "size_bytes": dst.stat().st_size})
    write_csv(run_root / "compact_review_bundle/compact_bundle_manifest.csv", bundle_rows)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
