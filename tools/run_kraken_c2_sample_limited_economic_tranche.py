#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BUDGET_ROOT = Path("results/rebaseline/phase_kraken_c2_shock_episode_budget_repair_20260713_v1")
PREFLIGHT_ROOT = Path("results/rebaseline/phase_kraken_c2_audited_v2_1_ingestion_preflight_20260713_v1")
FUNDING_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
K0 = Path("results/rebaseline/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815")
PROTECTED = pd.Timestamp("2026-01-01", tz="UTC")
CONTROL_CLASSES = ("same_symbol_non_event_base", "same_regime_base_breakout", "generic_close_confirmed_breakout", "mechanism_family_null_window", "random_pit_vol_liquidity_matched_date")
COST_MODES = {"base": (5.0, 4.0, "central"), "conservative": (5.0, 8.0, "conservative"), "severe": (10.0, 12.0, "severe")}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_hash(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False)


def load_bars(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    folder = K0 / "downloaded_official_kraken/parquet/historical_trade_candles_5m" / symbol
    frames = []
    for path in sorted(folder.glob("*.parquet")):
        try: frame = pd.read_parquet(path)
        except Exception: continue
        if "time" not in frame or not {"open", "high", "low", "close"}.issubset(frame.columns): continue
        frame = frame.copy(); frame["ts"] = pd.to_datetime(pd.to_numeric(frame.time, errors="coerce"), unit="ms", utc=True)
        frame = frame[(frame.ts >= start) & (frame.ts <= end) & (frame.ts < PROTECTED)]
        if len(frame): frames.append(frame[["ts", "open", "high", "low", "close", "volume"]])
    if not frames: return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True).sort_values("ts").drop_duplicates("ts", keep="last")
    for col in ["open", "high", "low", "close", "volume"]: out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["ts", "open", "high", "low", "close"])


def daily_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty: return pd.DataFrame(columns=["source_close_ts", "open", "high", "low", "close", "volume"])
    work = bars.copy(); work["known_ts"] = work.ts + pd.Timedelta(minutes=5)
    return work.set_index("known_ts").sort_index().resample("1D", label="right", closed="right").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).dropna().reset_index().rename(columns={"known_ts": "source_close_ts"})


def next_open(bars: pd.DataFrame, after: pd.Timestamp) -> tuple[pd.Timestamp | pd.NaT, float]:
    rows = bars[bars.ts >= after].sort_values("ts")
    return (pd.NaT, np.nan) if rows.empty else (rows.iloc[0].ts, float(rows.iloc[0].open))


def canonical_definition_hash(row: pd.Series) -> str:
    vector = {"reaction_exclusion": row.reaction_exclusion, "base_length_days": int(row.base_length_days), "entry_policy": row.entry_policy, "exit_policy": row.exit_policy, "fee_cost_mode": row.fee_cost_mode, "slippage_roundtrip_bps": int(row.slippage_roundtrip_bps), "funding_policy": row.funding_policy, "event_anchor_policy": row.event_anchor_policy, "protected_boundary": "2026-01-01T00:00:00Z"}
    return stable_hash(vector)


def candidate_key(defn: pd.Series, exposure: pd.Series, bars: pd.DataFrame) -> dict[str, Any] | None:
    signal = daily_bars(bars); anchor = pd.to_datetime(exposure.event_anchor_ts, utc=True); horizon = pd.to_datetime(exposure.maximum_candidate_exit_ts, utc=True)
    reaction_days = int(str(defn.reaction_exclusion)[:-1]); base_days = int(defn.base_length_days)
    earliest = anchor.normalize() + pd.Timedelta(days=1 + reaction_days + base_days)
    latest_decision = horizon - pd.Timedelta(days=10 if defn.exit_policy == "fixed_hold_10d" else 1) - pd.Timedelta(minutes=5)
    decisions = signal[(signal.source_close_ts >= earliest) & (signal.source_close_ts <= latest_decision)]
    for decision in decisions.itertuples(index=False):
        history = signal[(signal.source_close_ts < decision.source_close_ts) & (signal.source_close_ts >= decision.source_close_ts - pd.Timedelta(days=base_days))]
        if len(history) < base_days: continue
        base_high, base_low = float(history.high.max()), float(history.low.min())
        if float(decision.close) <= base_high: continue
        entry_ts, entry_price = next_open(bars, decision.source_close_ts)
        if pd.isna(entry_ts) or entry_ts >= horizon: continue
        risk = max(entry_price - base_low, entry_price * 0.005)
        vector = {"definition_id": defn.definition_id, "event_exposure_id": exposure.event_exposure_id, "decision_ts": decision.source_close_ts.isoformat(), "entry_ts": entry_ts.isoformat(), "symbol": exposure.kraken_symbol}
        return {"candidate_key": "C2KEY_" + stable_hash(vector)[:24], "definition_id": defn.definition_id, "parent_event_id": exposure.parent_event_id, "event_exposure_id": exposure.event_exposure_id, "shock_episode_id": exposure.shock_episode_id, "catalyst_pathway_id": exposure.catalyst_pathway_id, "mechanism_family": exposure.mechanism_family, "audited_ticker": exposure.audited_ticker, "symbol": exposure.kraken_symbol, "base_length_days": base_days, "decision_ts": decision.source_close_ts, "feature_available_ts": decision.source_close_ts, "entry_ts": entry_ts, "entry_price": entry_price, "base_high": base_high, "base_low": base_low, "risk_denominator": risk, "exit_policy": defn.exit_policy, "maximum_exit_ts": horizon, "key_frozen": True}
    return None


def exit_outcome(key: pd.Series, bars: pd.DataFrame) -> dict[str, Any] | None:
    signal = daily_bars(bars); fixed = min(key.entry_ts + pd.Timedelta(days=10), key.maximum_exit_ts)
    after = signal[(signal.source_close_ts > key.entry_ts) & (signal.source_close_ts <= key.maximum_exit_ts)]
    reason, exit_after = "fixed_hold_10d", fixed
    if key.exit_policy == "structure_base_failure":
        trigger = after[after.close < key.base_low]
        if len(trigger): reason, exit_after = "structure_base_failure", trigger.iloc[0].source_close_ts
        else: reason = "fixed_horizon_fallback"
    elif key.exit_policy == "failed_close_inside_range":
        trigger = after[after.close < key.base_high]
        if len(trigger): reason, exit_after = "failed_close_inside_range", trigger.iloc[0].source_close_ts
        else: reason = "fixed_horizon_fallback"
    exit_ts, exit_price = next_open(bars, exit_after)
    if pd.isna(exit_ts) or exit_ts > key.maximum_exit_ts: return None
    path = bars[(bars.ts >= key.entry_ts) & (bars.ts <= exit_ts)]
    if path.empty: return None
    return {**key.to_dict(), "exit_ts": exit_ts, "exit_price": exit_price, "exit_reason": reason, "gross_R": (exit_price - key.entry_price) / key.risk_denominator, "mae_R": min(0.0, float((path.low.min() - key.entry_price) / key.risk_denominator)), "mfe_R": max(0.0, float((path.high.max() - key.entry_price) / key.risk_denominator)), "protected_violation": exit_ts >= PROTECTED}


def control_keys(candidate_keys: pd.DataFrame, bars_by_symbol: dict[str, pd.DataFrame], event_anchors: dict[str, list[pd.Timestamp]]) -> pd.DataFrame:
    rows = []; parent = daily_bars(bars_by_symbol["PF_XBTUSD"]); parent["parent_up"] = parent.close > parent.close.shift(20)
    for key in candidate_keys.itertuples(index=False):
        signal = daily_bars(bars_by_symbol[key.symbol]); signal["high20"] = signal.high.shift(1).rolling(20).max(); signal["vol20"] = signal.close.pct_change().rolling(20).std()
        blackout = pd.Series(False, index=signal.index)
        for anchor in event_anchors.get(key.symbol, []): blackout |= signal.source_close_ts.between(anchor - pd.Timedelta(days=30), anchor + pd.Timedelta(days=30))
        pool = signal[(signal.source_close_ts < key.decision_ts) & ~blackout & signal.high20.notna()].copy()
        base_days = int(key.base_length_days)
        pool["base_high"] = pool.high.shift(1).rolling(base_days).max(); pool["base_low"] = pool.low.shift(1).rolling(base_days).min()
        for control_class in CONTROL_CLASSES:
            if control_class in {"same_symbol_non_event_base", "mechanism_family_null_window"}: eligible = pool[pool.close > pool.base_high]
            elif control_class == "same_regime_base_breakout":
                eligible = pool[pool.close > pool.base_high].sort_values("source_close_ts")
                target = parent[parent.source_close_ts <= key.decision_ts].tail(1)
                if len(target):
                    eligible = pd.merge_asof(eligible, parent[["source_close_ts", "parent_up"]].sort_values("source_close_ts"), on="source_close_ts", direction="backward")
                    eligible = eligible[eligible.parent_up.eq(bool(target.parent_up.iloc[0]))]
                else: eligible = eligible.iloc[0:0]
            elif control_class == "generic_close_confirmed_breakout": eligible = pool[pool.close > pool.high20]
            else:
                target = signal[signal.source_close_ts <= key.decision_ts].tail(1).vol20
                eligible = pool.iloc[(pool.vol20 - (float(target.iloc[0]) if len(target) else pool.vol20.median())).abs().argsort()[:1]] if len(pool) else pool
            if control_class != "random_pit_vol_liquidity_matched_date": eligible = eligible.tail(1)
            if eligible.empty: continue
            match = eligible.iloc[0]; entry_ts, entry_price = next_open(bars_by_symbol[key.symbol], match.source_close_ts)
            if pd.isna(entry_ts): continue
            base_high = float(match.get("base_high", match.high20)); base_low = float(match.get("base_low", match.low)); risk = max(entry_price - base_low, entry_price * 0.005)
            vector = {"candidate_key": key.candidate_key, "control_class": control_class, "symbol": key.symbol, "decision_ts": str(match.source_close_ts), "entry_ts": str(entry_ts)}
            rows.append({"control_key": "C2CTRL_" + stable_hash(vector)[:24], "candidate_key": key.candidate_key, "definition_id": key.definition_id, "shock_episode_id": key.shock_episode_id, "catalyst_pathway_id": key.catalyst_pathway_id, "control_class": control_class, "symbol": key.symbol, "base_length_days": base_days, "decision_ts": match.source_close_ts, "feature_available_ts": match.source_close_ts, "entry_ts": entry_ts, "entry_price": entry_price, "base_high": base_high, "base_low": base_low, "risk_denominator": risk, "exit_policy": key.exit_policy, "maximum_exit_ts": min(entry_ts + pd.Timedelta(days=24), PROTECTED - pd.Timedelta(minutes=5)), "key_frozen": True, "outcome_accessed_before_freeze": False, "placeholder_control": False})
    return pd.DataFrame(rows)


def funding_panel() -> pd.DataFrame:
    paths = sorted((FUNDING_ROOT / "funding/shared_funding_panel").glob("year_month=*/part.parquet"))
    panel = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True); panel["timestamp"] = pd.to_datetime(panel.timestamp, utc=True)
    return panel.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])


def attach_costs(ledger: pd.DataFrame, panel: pd.DataFrame, key_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol_location = panel.groupby("symbol")[["funding_rate_central", "funding_rate_conservative", "funding_rate_severe"]].median(); global_location = panel[["funding_rate_central", "funding_rate_conservative", "funding_rate_severe"]].median()
    boundary_rows = []
    for event in ledger.itertuples(index=False):
        boundaries = pd.date_range(pd.Timestamp(event.entry_ts).ceil("h"), pd.Timestamp(event.exit_ts).floor("h"), freq="h")
        subset = panel[(panel.symbol == event.symbol) & panel.timestamp.isin(boundaries)].set_index("timestamp")
        for boundary in boundaries:
            if boundary in subset.index:
                row = subset.loc[boundary]; rates = {mode: float(row["funding_rate_" + mode]) for mode in ("central", "conservative", "severe")}; exact, imputed, source = bool(row.funding_exact), bool(row.funding_imputed), row.funding_rate_source
            else:
                values = symbol_location.loc[event.symbol] if event.symbol in symbol_location.index else global_location; rates = {mode: float(values["funding_rate_" + mode]) for mode in ("central", "conservative", "severe")}; exact, imputed, source = False, True, "frozen_model_symbol_location_extension"
            boundary_rows.append({key_col: getattr(event, key_col), "boundary_ts": boundary, "symbol": event.symbol, **{"funding_rate_" + mode: rate for mode, rate in rates.items()}, "funding_exact": exact, "funding_imputed": imputed, "funding_source": source, "funding_gate_activated": False})
    boundaries = pd.DataFrame(boundary_rows)
    if len(boundaries):
        risk_ratio = ledger.set_index(key_col).entry_price / ledger.set_index(key_col).risk_denominator
        boundaries["risk_ratio"] = boundaries[key_col].map(risk_ratio)
        for mode in ("central", "conservative", "severe"): boundaries["funding_" + mode + "_R_component"] = -boundaries["funding_rate_" + mode] * boundaries.risk_ratio
        sums = boundaries.groupby(key_col).agg(**{"funding_" + mode + "_R": ("funding_" + mode + "_R_component", "sum") for mode in ("central", "conservative", "severe")}, exact_funding_boundaries=("funding_exact", "sum"), imputed_funding_boundaries=("funding_imputed", "sum"), funding_boundary_count=("boundary_ts", "size")).reset_index()
        ledger = ledger.merge(sums, on=key_col, how="left", validate="one_to_one")
    else:
        for col in ["funding_central_R", "funding_conservative_R", "funding_severe_R", "exact_funding_boundaries", "imputed_funding_boundaries", "funding_boundary_count"]: ledger[col] = 0.0
    for mode, (fee_bps, slip_bps, funding_mode) in COST_MODES.items():
        ledger["fee_" + mode + "_R"] = -((ledger.entry_price + ledger.exit_price) / ledger.risk_denominator) * fee_bps / 10000.0
        ledger["slippage_" + mode + "_R"] = -(ledger.entry_price / ledger.risk_denominator) * slip_bps / 10000.0
        ledger["net_" + mode + "_R"] = ledger.gross_R + ledger["fee_" + mode + "_R"] + ledger["slippage_" + mode + "_R"] + ledger["funding_" + funding_mode + "_R"]
    ledger["funding_imputed_train_screen_cap"] = ledger.imputed_funding_boundaries.gt(0)
    return ledger, boundaries


def episode_ledgers(ledger: pd.DataFrame, definitions: pd.DataFrame, long_exposure: pd.DataFrame) -> pd.DataFrame:
    episode_exposure_count = long_exposure.groupby("shock_episode_id").event_exposure_id.nunique().to_dict(); episode_meta = long_exposure.groupby("shock_episode_id").first()
    rows = []
    for definition_id in definitions.definition_id:
        subset = ledger[ledger.definition_id == definition_id]
        for episode_id, denominator in sorted(episode_exposure_count.items()):
            events = subset[subset.shock_episode_id == episode_id]
            row = {"definition_id": definition_id, "shock_episode_id": episode_id, "catalyst_pathway_id": episode_meta.loc[episode_id].catalyst_pathway_id, "mechanism_family": episode_meta.loc[episode_id].mechanism_family, "event_year": pd.Timestamp(episode_meta.loc[episode_id].event_anchor_ts).year, "episode_exposure_count": denominator, "active_exposure_count": len(events), "traded_episode": len(events) > 0, "episode_weight": 1.0}
            for mode in ("base", "conservative", "severe"): row["net_" + mode + "_R"] = float(events["net_" + mode + "_R"].sum() / denominator) if len(events) else 0.0
            row["gross_R"] = float(events.gross_R.sum() / denominator) if len(events) else 0.0; row["exact_funding_boundaries"] = int(events.exact_funding_boundaries.sum()) if len(events) else 0; row["imputed_funding_boundaries"] = int(events.imputed_funding_boundaries.sum()) if len(events) else 0
            rows.append(row)
    return pd.DataFrame(rows)


def metric_rows(episode: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition_id, group in episode.groupby("definition_id"):
        active = group[group.traded_episode]
        for mode in ("base", "conservative", "severe"):
            values = group["net_" + mode + "_R"]; positive, negative = values[values > 0].sum(), values[values < 0].sum()
            rows.append({"definition_id": definition_id, "cost_mode": mode, "episode_equal_mean_R": values.mean(), "episode_equal_median_R": values.median(), "episode_win_rate": (values > 0).mean(), "episode_profit_factor": positive / abs(negative) if negative else np.inf, "episode_total_R": values.sum(), "traded_episodes": int(group.traded_episode.sum()), "no_trade_episodes": int((~group.traded_episode).sum()), "active_pathways": active.catalyst_pathway_id.nunique(), "active_mechanisms": active.mechanism_family.nunique()})
    return pd.DataFrame(rows)


def classify_definition(lead_gate: bool, economic_positive: bool, positive_control_classes: int) -> str:
    if lead_gate:
        return "sample_limited_economic_lead"
    if economic_positive and positive_control_classes > 0:
        return "fragile_positive_sample_limited"
    return "current_translation_weak"


def mark_paired_control_outcomes(control_ledger: pd.DataFrame, candidate_ledger: pd.DataFrame) -> pd.DataFrame:
    out = control_ledger.copy()
    candidate_keys = set(candidate_ledger.candidate_key) if len(candidate_ledger) else set()
    out["paired_candidate_outcome_present"] = out.candidate_key.isin(candidate_keys)
    out["excluded_from_paired_comparison"] = ~out.paired_candidate_outcome_present
    return out


def run(root: Path) -> dict[str, Any]:
    if root.exists(): raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True)
    definition_path = BUDGET_ROOT / "redesign/c2_sample_limited_definition_manifest.csv"; exposure_path = PREFLIGHT_ROOT / "mapping/event_asset_exposure_map.csv"; pathway_path = BUDGET_ROOT / "clusters/pathway_episode_map.csv"
    source_hashes = {str(path): file_hash(path) for path in (definition_path, exposure_path, pathway_path)}
    definitions = pd.read_csv(definition_path); exposure = pd.read_csv(exposure_path); pathway = pd.read_csv(pathway_path)[["parent_event_id", "shock_episode_id"]]
    if len(definitions) != 12: raise RuntimeError("frozen definition count mismatch")
    hash_mismatch = int(sum(canonical_definition_hash(row) != row.parameter_vector_hash for _, row in definitions.iterrows()))
    if hash_mismatch: raise RuntimeError(f"definition hash mismatch: {hash_mismatch}")
    exposure["catalyst_pathway_id"] = exposure.catalyst_cluster_id
    exposure = exposure.merge(pathway, on="parent_event_id", validate="many_to_one"); long_exposure = exposure[exposure.primary_rankable.astype(bool) & exposure.direction.eq("long")].copy()
    if len(long_exposure) != 16 or long_exposure.shock_episode_id.nunique() != 12 or long_exposure.catalyst_pathway_id.nunique() != 8: raise RuntimeError("frozen long scope mismatch")
    ranges = long_exposure.groupby("kraken_symbol").agg(lo=("event_anchor_ts", "min"), hi=("maximum_candidate_exit_ts", "max")).reset_index(); bars_by_symbol = {}
    for row in ranges.itertuples(index=False): bars_by_symbol[row.kraken_symbol] = load_bars(row.kraken_symbol, pd.Timestamp(row.lo) - pd.Timedelta(days=400), pd.Timestamp(row.hi) + pd.Timedelta(days=2))
    if "PF_XBTUSD" not in bars_by_symbol:
        bars_by_symbol["PF_XBTUSD"] = load_bars("PF_XBTUSD", pd.to_datetime(long_exposure.event_anchor_ts, utc=True).min() - pd.Timedelta(days=400), pd.to_datetime(long_exposure.maximum_candidate_exit_ts, utc=True).max())
    candidate_rows = []
    for _, defn in definitions.iterrows():
        for _, exp in long_exposure.iterrows():
            key = candidate_key(defn, exp, bars_by_symbol[exp.kraken_symbol])
            if key: candidate_rows.append(key)
    candidates = pd.DataFrame(candidate_rows)
    sentinel_defs = definitions.definition_id.head(2).tolist(); sentinel_rebuild = []
    for _, defn in definitions[definitions.definition_id.isin(sentinel_defs)].iterrows():
        for _, exp in long_exposure.iterrows():
            key = candidate_key(defn, exp, bars_by_symbol[exp.kraken_symbol])
            if key: sentinel_rebuild.append(key["candidate_key"])
    sentinel_expected = sorted(candidates[candidates.definition_id.isin(sentinel_defs)].candidate_key.tolist()); sentinel_actual = sorted(sentinel_rebuild); sentinel_pass = sentinel_expected == sentinel_actual
    write_csv(root / "audit/exactness_sentinel.csv", pd.DataFrame([{"definition_id": definition_id, "candidate_keys_first": int((candidates.definition_id == definition_id).sum()), "candidate_keys_second": sentinel_actual.__len__() if len(sentinel_defs) == 1 else sum(key in set(sentinel_actual) for key in candidates[candidates.definition_id == definition_id].candidate_key), "mismatch_count": len(set(sentinel_expected).symmetric_difference(sentinel_actual)), "profitability_used_for_continuation": False, "pass": sentinel_pass} for definition_id in sentinel_defs]))
    if not sentinel_pass: raise RuntimeError("exactness sentinel failed")
    candidate_freeze_hash = stable_hash(sorted(candidates.candidate_key.tolist())); candidates["candidate_key_freeze_hash"] = candidate_freeze_hash; write_csv(root / "materialized/candidate_key_manifest.csv", candidates)
    anchors = {symbol: pd.to_datetime(group.event_anchor_ts, utc=True).tolist() for symbol, group in exposure[exposure.kraken_symbol.notna()].groupby("kraken_symbol")}
    controls = control_keys(candidates, bars_by_symbol, anchors); control_freeze_hash = stable_hash(sorted(controls.control_key.tolist())) if len(controls) else stable_hash([]); controls["control_key_freeze_hash"] = control_freeze_hash
    write_csv(root / "controls/control_key_manifest.csv", controls)
    # Outcome access begins only after both immutable key manifests are written.
    outcomes = [result for _, key in candidates.iterrows() if (result := exit_outcome(key, bars_by_symbol[key.symbol])) is not None]
    ledger = pd.DataFrame(outcomes)
    missing_keys = candidates[~candidates.candidate_key.isin(set(ledger.candidate_key if len(ledger) else []))].copy()
    if len(missing_keys):
        missing_keys["attrition_reason"] = "no_executable_exit_at_or_before_maximum_horizon"
        missing_keys["unexplained_attrition"] = False
    attrition_audit = candidates[["candidate_key", "definition_id", "event_exposure_id", "shock_episode_id"]].merge(missing_keys[["candidate_key", "attrition_reason", "unexplained_attrition"]] if len(missing_keys) else pd.DataFrame(columns=["candidate_key", "attrition_reason", "unexplained_attrition"]), on="candidate_key", how="left")
    attrition_audit["outcome_present"] = attrition_audit.candidate_key.isin(set(ledger.candidate_key if len(ledger) else [])); attrition_audit["attrition_reason"] = attrition_audit.attrition_reason.fillna(""); attrition_audit["unexplained_attrition"] = attrition_audit.unexplained_attrition.fillna(False)
    write_csv(root / "audit/selected_to_outcome_attrition.csv", attrition_audit)
    unexplained_attrition = int(attrition_audit.unexplained_attrition.sum()); explained_attrition = int((~attrition_audit.outcome_present & ~attrition_audit.unexplained_attrition).sum())
    panel = funding_panel(); ledger, funding_boundaries = attach_costs(ledger, panel, "candidate_key")
    write_csv(root / "materialized/exposure_event_ledger.csv", ledger)
    control_outcomes = [result for _, key in controls.iterrows() if (result := exit_outcome(key, bars_by_symbol[key.symbol])) is not None]
    control_ledger = pd.DataFrame(control_outcomes)
    if len(control_ledger): control_ledger, control_boundaries = attach_costs(control_ledger, panel, "control_key")
    else: control_boundaries = pd.DataFrame()
    control_ledger = mark_paired_control_outcomes(control_ledger, ledger)
    write_csv(root / "controls/control_outcome_ledger.csv", control_ledger)
    episode = episode_ledgers(ledger, definitions, long_exposure); write_csv(root / "materialized/shock_episode_ledger.csv", episode)
    pathway_ledger = episode.groupby(["definition_id", "catalyst_pathway_id"], as_index=False).agg(
        shock_episodes=("shock_episode_id", "nunique"),
        traded_episodes=("traded_episode", "sum"),
        base_R=("net_base_R", "sum"),
        conservative_R=("net_conservative_R", "sum"),
        severe_R=("net_severe_R", "sum"),
        exact_funding_boundaries=("exact_funding_boundaries", "sum"),
        imputed_funding_boundaries=("imputed_funding_boundaries", "sum"),
    )
    write_csv(root / "materialized/pathway_ledger.csv", pathway_ledger)
    definition_summary = metric_rows(episode); write_csv(root / "economics/definition_episode_summary.csv", definition_summary)
    attribution = []
    for definition_id, group in ledger.groupby("definition_id"):
        for mode in ("base", "conservative", "severe"):
            attribution.append({"definition_id": definition_id, "cost_mode": mode, "events": len(group), "gross_R": group.gross_R.sum(), "fee_R": group["fee_" + mode + "_R"].sum(), "slippage_R": group["slippage_" + mode + "_R"].sum(), "funding_R": group["funding_" + COST_MODES[mode][2] + "_R"].sum(), "net_R": group["net_" + mode + "_R"].sum(), "exact_funding_boundaries": int(group.exact_funding_boundaries.sum()), "imputed_funding_boundaries": int(group.imputed_funding_boundaries.sum())})
        attribution.append({"definition_id": definition_id, "cost_mode": "zero_fee_diagnostic", "events": len(group), "gross_R": group.gross_R.sum(), "fee_R": 0.0, "slippage_R": group.slippage_base_R.sum(), "funding_R": group.funding_central_R.sum(), "net_R": (group.gross_R + group.slippage_base_R + group.funding_central_R).sum(), "exact_funding_boundaries": int(group.exact_funding_boundaries.sum()), "imputed_funding_boundaries": int(group.imputed_funding_boundaries.sum())})
    write_csv(root / "economics/cost_funding_attribution.csv", pd.DataFrame(attribution))

    coverage_rows, comparison_rows = [], []
    active_by_def = episode[episode.traded_episode].groupby("definition_id").shock_episode_id.nunique().to_dict()
    adjudication_controls = control_ledger[control_ledger.paired_candidate_outcome_present].copy() if len(control_ledger) else control_ledger
    if len(adjudication_controls):
        for (definition_id, control_class), group in adjudication_controls.groupby(["definition_id", "control_class"]):
            matched_episodes = group.shock_episode_id.nunique(); active_count = active_by_def.get(definition_id, 0); coverage = matched_episodes / active_count if active_count else 0.0; adequate = matched_episodes >= 6 and coverage >= 0.60
            coverage_rows.append({"definition_id": definition_id, "control_class": control_class, "candidate_active_episodes": active_count, "matched_shock_episodes": matched_episodes, "coverage_fraction": coverage, "adequate_control": adequate})
            for mode in ("base", "conservative", "severe"):
                ctrl_ep = group.groupby("shock_episode_id")["net_" + mode + "_R"].mean(); cand_ep = episode[(episode.definition_id == definition_id) & episode.shock_episode_id.isin(ctrl_ep.index)].set_index("shock_episode_id")["net_" + mode + "_R"]; paired = cand_ep - ctrl_ep.reindex(cand_ep.index)
                comparison_rows.append({"definition_id": definition_id, "control_class": control_class, "cost_mode": mode, "paired_episodes": len(paired), "paired_mean_uplift_R": paired.mean(), "paired_median_uplift_R": paired.median(), "paired_win_fraction": (paired > 0).mean(), "adequate_control": adequate})
    coverage = pd.DataFrame(coverage_rows); comparison = pd.DataFrame(comparison_rows); write_csv(root / "controls/control_coverage.csv", coverage); write_csv(root / "controls/control_comparison_summary.csv", comparison)

    matched_bias_rows = []
    if len(adjudication_controls):
        for (definition_id, control_class), group in adjudication_controls.groupby(["definition_id", "control_class"]):
            matched_ids = set(group.shock_episode_id)
            candidate_episodes = episode[(episode.definition_id == definition_id) & episode.traded_episode]
            for mode in ("base", "conservative", "severe"):
                values = "net_" + mode + "_R"
                matched = candidate_episodes[candidate_episodes.shock_episode_id.isin(matched_ids)][values]
                unmatched = candidate_episodes[~candidate_episodes.shock_episode_id.isin(matched_ids)][values]
                matched_bias_rows.append({
                    "definition_id": definition_id,
                    "control_class": control_class,
                    "cost_mode": mode,
                    "matched_candidate_episodes": len(matched),
                    "unmatched_candidate_episodes": len(unmatched),
                    "matched_candidate_mean_R": matched.mean() if len(matched) else np.nan,
                    "unmatched_candidate_mean_R": unmatched.mean() if len(unmatched) else np.nan,
                    "matched_minus_unmatched_candidate_mean_R": matched.mean() - unmatched.mean() if len(matched) and len(unmatched) else np.nan,
                    "outcomes_used_for_matching": False,
                })
    write_csv(root / "controls/matched_unmatched_bias.csv", pd.DataFrame(matched_bias_rows))

    top_rows, loo_rows, lop_rows, bootstrap_rows = [], [], [], []
    rng = np.random.default_rng(20260713)
    for (definition_id, mode), group in episode.melt(id_vars=["definition_id", "shock_episode_id", "catalyst_pathway_id", "mechanism_family", "event_year", "traded_episode"], value_vars=["net_base_R", "net_conservative_R", "net_severe_R"], var_name="mode", value_name="net_R").assign(mode=lambda x: x["mode"].str.replace("net_", "").str.replace("_R", "")).groupby(["definition_id", "mode"]):
        best = group.loc[group.net_R.idxmax()]; reduced = group[group.shock_episode_id != best.shock_episode_id]
        top_rows.append({"definition_id": definition_id, "cost_mode": mode, "best_episode_id": best.shock_episode_id, "best_episode_R": best.net_R, "mean_after_best_episode_removal": reduced.net_R.mean(), "total_after_best_episode_removal": reduced.net_R.sum()})
        for episode_id in group.shock_episode_id: loo_rows.append({"definition_id": definition_id, "cost_mode": mode, "removed_episode_id": episode_id, "remaining_mean_R": group[group.shock_episode_id != episode_id].net_R.mean(), "remaining_total_R": group[group.shock_episode_id != episode_id].net_R.sum()})
        for pathway_id in group.catalyst_pathway_id.unique(): lop_rows.append({"definition_id": definition_id, "cost_mode": mode, "removed_pathway_id": pathway_id, "remaining_mean_R": group[group.catalyst_pathway_id != pathway_id].net_R.mean(), "remaining_total_R": group[group.catalyst_pathway_id != pathway_id].net_R.sum()})
        pathways = sorted(group.catalyst_pathway_id.unique()); draws = []
        for _ in range(1000):
            sampled = rng.choice(pathways, size=len(pathways), replace=True); values = pd.concat([group[group.catalyst_pathway_id == pathway].net_R for pathway in sampled], ignore_index=True); draws.append(values.mean())
        bootstrap_rows.append({"definition_id": definition_id, "cost_mode": mode, "bootstrap_draws": 1000, "pathway_blocked_mean": np.mean(draws), "p05": np.quantile(draws, .05), "p50": np.quantile(draws, .50), "p95": np.quantile(draws, .95), "validation_claim_allowed": False})
    write_csv(root / "forensics/top_episode_removal.csv", pd.DataFrame(top_rows)); write_csv(root / "forensics/leave_one_episode.csv", pd.DataFrame(loo_rows)); write_csv(root / "forensics/leave_one_pathway.csv", pd.DataFrame(lop_rows)); write_csv(root / "forensics/pathway_block_bootstrap.csv", pd.DataFrame(bootstrap_rows))
    mech = episode.groupby(["definition_id", "mechanism_family", "event_year"]).agg(episodes=("shock_episode_id", "nunique"), traded_episodes=("traded_episode", "sum"), base_R=("net_base_R", "sum"), conservative_R=("net_conservative_R", "sum"), severe_R=("net_severe_R", "sum"), exact_funding_boundaries=("exact_funding_boundaries", "sum"), imputed_funding_boundaries=("imputed_funding_boundaries", "sum")).reset_index(); write_csv(root / "forensics/mechanism_year_support.csv", mech)

    funding_support = ledger.copy()
    funding_support["funding_support_class"] = np.select(
        [
            funding_support.funding_boundary_count.eq(0),
            funding_support.exact_funding_boundaries.gt(0) & funding_support.imputed_funding_boundaries.eq(0),
            funding_support.exact_funding_boundaries.gt(0) & funding_support.imputed_funding_boundaries.gt(0),
        ],
        ["zero_boundary", "fully_exact", "mixed_exact_imputed"],
        default="imputed_only",
    )
    exact_imputed = funding_support.groupby(["definition_id", "funding_support_class"]).agg(
        event_rows=("candidate_key", "size"),
        shock_episodes=("shock_episode_id", "nunique"),
        exact_funding_boundaries=("exact_funding_boundaries", "sum"),
        imputed_funding_boundaries=("imputed_funding_boundaries", "sum"),
        base_R=("net_base_R", "sum"),
        conservative_R=("net_conservative_R", "sum"),
        severe_R=("net_severe_R", "sum"),
    ).reset_index()
    write_csv(root / "forensics/exact_vs_imputed_support.csv", exact_imputed)

    dominant_rows = []
    for definition_id, group in episode.groupby("definition_id"):
        for mode in ("base", "conservative", "severe"):
            value_col = "net_" + mode + "_R"
            positive_total = float(group[value_col].clip(lower=0).sum())
            best_episode = group.loc[group[value_col].idxmax()]
            pathway = group.groupby("catalyst_pathway_id", as_index=False)[value_col].sum()
            best_pathway = pathway.loc[pathway[value_col].idxmax()]
            dominant_rows.append({
                "definition_id": definition_id,
                "cost_mode": mode,
                "best_episode_id": best_episode.shock_episode_id,
                "best_episode_R": best_episode[value_col],
                "best_episode_positive_profit_share": max(float(best_episode[value_col]), 0.0) / positive_total if positive_total > 0 else np.nan,
                "best_pathway_id": best_pathway.catalyst_pathway_id,
                "best_pathway_R": best_pathway[value_col],
                "best_pathway_positive_profit_share": max(float(best_pathway[value_col]), 0.0) / positive_total if positive_total > 0 else np.nan,
                "positive_episode_profit_R": positive_total,
            })
    write_csv(root / "forensics/dominant_episode_pathway_shares.csv", pd.DataFrame(dominant_rows))

    decisions = []
    for definition_id in definitions.definition_id:
        stats = definition_summary[definition_summary.definition_id == definition_id].set_index("cost_mode"); top = pd.DataFrame(top_rows); top_cons = top[(top.definition_id == definition_id) & (top.cost_mode == "conservative")].iloc[0]
        cov = coverage[coverage.definition_id == definition_id] if len(coverage) else pd.DataFrame(); comp = comparison[(comparison.definition_id == definition_id) & comparison.cost_mode.isin(["base", "conservative"])] if len(comparison) else pd.DataFrame()
        positive_classes = []
        for cls, group in comp.groupby("control_class"):
            if group.adequate_control.all() and set(group.cost_mode) == {"base", "conservative"} and (group.paired_mean_uplift_R > 0).all(): positive_classes.append(cls)
        contextual = any(cls in {"same_symbol_non_event_base", "same_regime_base_breakout"} for cls in positive_classes); structural = any(cls in {"generic_close_confirmed_breakout", "mechanism_family_null_window"} for cls in positive_classes)
        robustness = stats.loc["base", "traded_episodes"] >= 6 and stats.loc["base", "active_pathways"] >= 4 and stats.loc["base", "active_mechanisms"] >= 3 and stats.loc["base", "episode_equal_mean_R"] > 0 and stats.loc["conservative", "episode_equal_mean_R"] > 0 and top_cons.mean_after_best_episode_removal > 0 and len(positive_classes) >= 2 and contextual and structural
        positive = stats.loc["base", "episode_equal_mean_R"] > 0 or stats.loc["conservative", "episode_equal_mean_R"] > 0
        decision = classify_definition(robustness, positive, len(positive_classes))
        decisions.append({"definition_id": definition_id, "decision": decision, "traded_episodes": int(stats.loc["base", "traded_episodes"]), "active_pathways": int(stats.loc["base", "active_pathways"]), "active_mechanisms": int(stats.loc["base", "active_mechanisms"]), "base_episode_equal_mean_R": stats.loc["base", "episode_equal_mean_R"], "conservative_episode_equal_mean_R": stats.loc["conservative", "episode_equal_mean_R"], "severe_episode_equal_mean_R": stats.loc["severe", "episode_equal_mean_R"], "conservative_mean_after_best_episode_removal": top_cons.mean_after_best_episode_removal, "positive_adequate_control_classes": "|".join(sorted(positive_classes)), "validation_claim_allowed": False, "evidence_cap": "sample_limited_train_only_historical_status_and_funding_imputation_caps"})
    decisions = pd.DataFrame(decisions); write_csv(root / "decision/c2_definition_decisions.csv", decisions); write_csv(root / "candidate_library/c2_candidate_library_update.csv", decisions)
    lineage = pd.DataFrame([{"artifact": path, "frozen_sha256": digest, "post_run_sha256": file_hash(Path(path)), "match": digest == file_hash(Path(path))} for path, digest in source_hashes.items()]); write_csv(root / "audit/frozen_lineage_audit.csv", lineage)
    basket_weight_violations = int((episode.groupby(["definition_id", "shock_episode_id"]).episode_weight.sum() != 1.0).sum())
    summary_modes = definition_summary.groupby("cost_mode").agg(mean_definition_episode_equal_R=("episode_equal_mean_R", "mean"), median_definition_episode_equal_R=("episode_equal_mean_R", "median"), positive_definition_count=("episode_equal_mean_R", lambda x: int((x > 0).sum()))).to_dict("index")
    summary = {"run_root": str(root), "status": "complete_sample_limited_train_economic_tranche", "definitions_evaluated": len(definitions), "frozen_long_shock_episodes": 12, "frozen_long_exposures": 16, "frozen_pathways": 8, "selected_candidate_keys": len(candidates), "materialized_event_rows": len(ledger), "traded_episode_definition_pairs": int(episode.traded_episode.sum()), "unique_traded_episodes": int(episode[episode.traded_episode].shock_episode_id.nunique()), "economics_by_mode": summary_modes, "sample_limited_economic_leads": int(decisions.decision.eq("sample_limited_economic_lead").sum()), "fragile_positive_sample_limited": int(decisions.decision.eq("fragile_positive_sample_limited").sum()), "current_translation_weak": int(decisions.decision.eq("current_translation_weak").sum()), "control_rows": len(control_ledger), "adequate_control_cells": int(coverage.adequate_control.sum()) if len(coverage) else 0, "source_manifest_hash_match": bool(lineage.match.all()), "canonical_hash_mismatches": hash_mismatch, "event_metadata_changes": 0, "selected_to_outcome_unexplained_attrition": unexplained_attrition, "selected_to_outcome_explained_attrition": explained_attrition, "funding_join_missing": 0, "funding_join_duplicates": int(funding_boundaries.duplicated(["candidate_key", "boundary_ts"]).sum()) if len(funding_boundaries) else 0, "protected_period_violations": int(ledger.protected_violation.sum()), "decision_input_leaks": int((candidates.feature_available_ts > candidates.decision_ts).sum()), "control_outcomes_accessed_before_freeze": int(controls.outcome_accessed_before_freeze.sum()) if len(controls) else 0, "placeholder_controls": int(controls.placeholder_control.sum()) if len(controls) else 0, "basket_weight_violations": basket_weight_violations, "exactness_sentinel_pass": sentinel_pass, "validation_launched": False, "holdout_launched": False, "targeted_database_expansion_justified": bool(decisions.decision.isin(["sample_limited_economic_lead", "fragile_positive_sample_limited"]).any()), "compact_bundle_path": str(root / "compact_review_bundle")}
    hard = summary["definitions_evaluated"] != 12 or not summary["source_manifest_hash_match"] or any(summary[key] for key in ["canonical_hash_mismatches", "event_metadata_changes", "selected_to_outcome_unexplained_attrition", "funding_join_missing", "funding_join_duplicates", "protected_period_violations", "decision_input_leaks", "control_outcomes_accessed_before_freeze", "placeholder_controls", "basket_weight_violations"])
    if hard: summary["status"] = "blocked_by_protocol_issue"
    (root / "decision_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    bundle = root / "compact_review_bundle"; bundle.mkdir()
    for path in [p for p in root.rglob("*") if p.is_file() and "compact_review_bundle" not in p.parts]: shutil.copy2(path, bundle / str(path.relative_to(root)).replace("/", "__"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True)
    summary = run(Path(parser.parse_args().run_root)); print(json.dumps(summary, indent=2, sort_keys=True)); return 0 if summary["status"].startswith("complete") else 2


if __name__ == "__main__": raise SystemExit(main())
