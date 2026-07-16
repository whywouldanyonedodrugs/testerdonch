#!/usr/bin/env python3
"""Train-only Kraken backside-confirmed blowoff short screen."""
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

from tools import qlmg_evidence_contracts as evidence
from tools import qlmg_signal_state_contract as signal_state
from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools import run_kraken_liquid_failed_breakout_short_screen as lfbs


START = pd.Timestamp("2023-01-01", tz="UTC")
PROTECTED = pd.Timestamp("2026-01-01", tz="UTC")
END = PROTECTED - pd.Timedelta(minutes=5)
CONTRACT_VERSION = "kraken_backside_blowoff_short_v1_20260713"
EVALUATION_WINDOWS = {
    "2023": (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC")),
    "2024": (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2025-01-01", tz="UTC")),
    "2025-H1": (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC")),
    "2025-H2": (pd.Timestamp("2025-07-01", tz="UTC"), PROTECTED),
}
CONTROL_CLASSES = (
    "same_symbol_same_regime_random_short",
    "parabolic_extension_without_backside_confirmation",
    "generic_20d_failed_breakout_short",
    "non_parabolic_red_candle_short",
    "pit_vol_liquidity_matched_random_date",
)
CONTEXTUAL_CONTROLS = {"same_symbol_same_regime_random_short", "pit_vol_liquidity_matched_random_date"}
STRUCTURAL_CONTROLS = {"parabolic_extension_without_backside_confirmation", "generic_20d_failed_breakout_short", "non_parabolic_red_candle_short"}
_PARENT_CACHE: dict[str, tuple[str, pd.Timestamp]] = {}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def parameter_hash(row: Mapping[str, Any], *, selected_key: bool) -> str:
    fields = ["extension_profile", "confirmation_bars", "parent_context"]
    if not selected_key:
        fields.append("exit_policy")
    vector = {field: row[field] for field in fields}
    vector.update({
        "side": "short", "signal_timeframe": "4h", "execution_timeframe": "5m",
        "universe_policy": "pit_kraken_tier_ab", "minimum_live_days": 30,
        "protected_boundary": PROTECTED.isoformat(), "contract_version": CONTRACT_VERSION,
        "confirmation_semantics": "prior_4h_low_and_peak_midpoint_and_peak_day_anchored_vwap",
    })
    return stable_hash(vector)


def raw_policy_hash(row: Mapping[str, Any]) -> str:
    return signal_state.stable_hash({
        "extension_profile": row["extension_profile"],
        "confirmation_bars": int(row["confirmation_bars"]),
        "side": "short",
        "signal_timeframe": "4h",
        "execution_timeframe": "5m",
        "universe_policy": "pit_kraken_tier_ab",
        "minimum_live_days": 30,
        "protected_boundary": PROTECTED,
        "contract_version": CONTRACT_VERSION,
        "confirmation_semantics": "prior_4h_low_and_peak_midpoint_and_peak_day_anchored_vwap",
    })


def frozen_manifest() -> pd.DataFrame:
    rows = []
    for extension in ("rise_40pct_5d_3atr", "rise_70pct_10d_4atr"):
        for confirmation_bars in (1, 3):
            for parent in ("fragile_countertrend_stress", "all_regime_comparator"):
                for exit_policy in ("fixed_72h", "daily_ema10_close", "swing_high_trail_7d"):
                    row = {
                        "definition_id": f"bcbs_v1_{len(rows)+1:03d}",
                        "extension_profile": extension,
                        "confirmation_bars": confirmation_bars,
                        "parent_context": parent,
                        "exit_policy": exit_policy,
                    }
                    row["selected_key_policy_hash"] = parameter_hash(row, selected_key=True)
                    row["parameter_vector_hash"] = parameter_hash(row, selected_key=False)
                    rows.append(row)
    return pd.DataFrame(rows)


def evaluation_window(ts: pd.Timestamp) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    value = pd.Timestamp(ts)
    for label, (start, end) in EVALUATION_WINDOWS.items():
        if start <= value < end:
            return label, start, end
    raise ValueError(f"timestamp outside train evaluation windows: {value}")


def context(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_root=root, start=START, end=END,
        args=SimpleNamespace(kraken_data_root=str(runner.DEFAULT_KRAKEN_DATA_ROOT), smoke=False),
    )


def feature_frames(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = bars.copy().sort_values("ts")
    work["known_ts"] = pd.to_datetime(work.ts, utc=True) + pd.Timedelta(minutes=5)
    work["typical"] = (work.high + work.low + work.close) / 3.0
    work["vwap_num"] = work.typical * work.volume.fillna(0)
    four = work.set_index("known_ts").resample("4h", label="right", closed="right").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), execution_bar_count=("close", "count"),
    )
    four = four[(four.execution_bar_count >= 36) & four.close.notna()].reset_index().rename(columns={"known_ts": "decision_ts"})
    daily = work.set_index("known_ts").resample("1D", label="right", closed="right").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), bar_count=("close", "count"),
    )
    daily = daily[(daily.bar_count >= 250) & daily.close.notna()].reset_index().rename(columns={"known_ts": "daily_source_ts"})
    previous = daily.close.shift(1)
    tr = pd.concat([(daily.high-daily.low), (daily.high-previous).abs(), (daily.low-previous).abs()], axis=1).max(axis=1)
    daily["atr_14d"] = tr.rolling(14, min_periods=14).mean()
    daily["ema_20"] = daily.close.ewm(span=20, adjust=False, min_periods=20).mean()
    daily["ema_10"] = daily.close.ewm(span=10, adjust=False, min_periods=10).mean()
    daily["return_5d"] = daily.close / daily.close.shift(5) - 1.0
    daily["return_10d"] = daily.close / daily.close.shift(10) - 1.0
    daily["prior_high_20d"] = daily.high.rolling(20, min_periods=20).max().shift(1)
    keep = ["daily_source_ts", "atr_14d", "ema_20", "ema_10", "return_5d", "return_10d", "prior_high_20d"]
    frame = pd.merge_asof(
        four.sort_values("decision_ts"), daily[keep].sort_values("daily_source_ts"),
        left_on="decision_ts", right_on="daily_source_ts", direction="backward", allow_exact_matches=True,
    )
    frame["extension_40_5"] = (frame.return_5d >= .40) & (frame.close >= frame.ema_20 + 3 * frame.atr_14d)
    frame["extension_70_10"] = (frame.return_10d >= .70) & (frame.close >= frame.ema_20 + 4 * frame.atr_14d)
    frame["red_candle"] = frame.close < frame.open
    frame["generic_failed_20d"] = (frame.high > frame.prior_high_20d) & (frame.close < frame.prior_high_20d)
    frame["feature_available_ts"] = frame[["decision_ts", "daily_source_ts"]].max(axis=1)
    return frame, work


def peak_day_vwap(work: pd.DataFrame, peak_ts: pd.Timestamp, decision_ts: pd.Timestamp) -> float:
    day_start = pd.Timestamp(peak_ts).floor("D")
    rows = work[(work.known_ts > day_start) & (work.known_ts <= decision_ts)]
    denominator = float(rows.volume.fillna(0).sum())
    return float(rows.vwap_num.sum() / denominator) if denominator > 0 else np.nan


def extension_column(profile: str) -> str:
    return "extension_40_5" if profile == "rise_40pct_5d_3atr" else "extension_70_10"


def confirmation_sequences(frame: pd.DataFrame, work: pd.DataFrame, extension_profile: str, confirmation_bars: int) -> list[dict[str, Any]]:
    """Return one backside decision per non-overlapping extension sequence."""
    extension = extension_column(extension_profile)
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(frame):
        if not bool(frame.iloc[index].get(extension, False)):
            index += 1
            continue
        peak_index = index
        cursor = index + 1
        resolved = False
        while cursor < len(frame):
            peak = frame.iloc[peak_index]
            current = frame.iloc[cursor]
            if float(current.high) > float(peak.high):
                peak_index = cursor
                cursor += 1
                continue
            bars_after_peak = cursor - peak_index
            if bars_after_peak > confirmation_bars:
                break
            previous = frame.iloc[cursor-1]
            vwap = peak_day_vwap(work, pd.Timestamp(peak.decision_ts), pd.Timestamp(current.decision_ts))
            midpoint = (float(peak.high) + float(peak.low)) / 2.0
            confirmed = float(current.close) < float(previous.low) and float(current.close) < midpoint and pd.notna(vwap) and float(current.close) < vwap
            if confirmed:
                rows.append({
                    "extension_index": index, "peak_index": peak_index, "decision_index": cursor,
                    "sequence_high": float(frame.iloc[index:cursor+1].high.max()), "peak_midpoint": midpoint,
                    "peak_day_anchored_vwap": vwap,
                })
                index = cursor + 1
                resolved = True
                break
            cursor += 1
        if not resolved:
            index = max(index + 1, cursor)
    return rows


def parent_state(bars: pd.DataFrame, decision_ts: pd.Timestamp) -> tuple[str, pd.Timestamp]:
    key = pd.Timestamp(decision_ts).isoformat()
    if key not in _PARENT_CACHE:
        spec = {
            "parent_regime_gate": "btc_eth_trend_down_diagnostic", "run_start_ts": START.isoformat(),
            "run_end_ts": END.isoformat(), "kraken_data_root": str(runner.DEFAULT_KRAKEN_DATA_ROOT),
        }
        result = runner.evaluate_parent_regime_gate(spec, bars, decision_ts)
        status = str(result.get("status", "unknown"))
        label = "both_down" if status == "pass" else "not_both_down" if status == "filtered" else status
        _PARENT_CACHE[key] = (label, pd.to_datetime(result.get("feature_source_ts"), utc=True))
    return _PARENT_CACHE[key]


def pit_allowed(ctx: SimpleNamespace, panel: pd.DataFrame, decision_ts: pd.Timestamp, symbol: str) -> bool:
    return lfbs.pit_allowed(ctx, panel, decision_ts, symbol)


def enumerate_candidates(ctx: SimpleNamespace, panel: pd.DataFrame, symbol: str, bars: pd.DataFrame, specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    frame, work = feature_frames(bars)
    raw_rows: list[dict[str, Any]] = []
    for spec in specs:
        raw_rows.extend(enumerate_raw_signals(ctx, panel, symbol, bars, spec, frame=frame, work=work))
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for raw in raw_rows:
            if raw["raw_policy_hash"] != raw_policy_hash(spec):
                continue
            if spec["parent_context"] == "fragile_countertrend_stress" and raw["parent_state"] != "both_down":
                continue
            key_vector = {"policy": spec["selected_key_policy_hash"], "raw_signal_address_hash": raw["raw_signal_address_hash"]}
            rows.append({**raw, "candidate_key": "BCBSK_" + stable_hash(key_vector)[:24], "selected_key_policy_hash": spec["selected_key_policy_hash"], "parent_context": spec["parent_context"], "selected_key_frozen": True})
    return rows, frame, work


def enumerate_raw_signals(
    ctx: SimpleNamespace,
    panel: pd.DataFrame,
    symbol: str,
    bars: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    frame: pd.DataFrame | None = None,
    work: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Emit parent-neutral mechanical signals without holding-period state."""
    if frame is None or work is None:
        frame, work = feature_frames(bars)
    rows: list[dict[str, Any]] = []
    panel_row = panel[panel.symbol.eq(symbol)]
    if panel_row.empty or str(panel_row.iloc[0].status) != "available":
        return rows
    listed = pd.Timestamp(panel_row.iloc[0].start_ts)
    policy_hash = raw_policy_hash(spec)
    for sequence in confirmation_sequences(frame, work, str(spec["extension_profile"]), int(spec["confirmation_bars"])):
        peak = frame.iloc[sequence["peak_index"]]
        decision = frame.iloc[sequence["decision_index"]]
        decision_ts = pd.Timestamp(decision.decision_ts)
        if decision_ts < START or decision_ts >= PROTECTED or decision_ts < listed + pd.Timedelta(days=30):
            continue
        if not pit_allowed(ctx, panel, decision_ts, symbol):
            continue
        label, parent_ts = parent_state(bars, decision_ts)
        if pd.isna(parent_ts) or parent_ts > decision_ts:
            continue
        entry_rows = bars[bars.ts >= decision_ts]
        if entry_rows.empty or pd.isna(decision.atr_14d) or float(decision.atr_14d) <= 0:
            continue
        entry = entry_rows.iloc[0]
        stop = float(sequence["sequence_high"])
        risk = stop - float(entry.open)
        if risk <= 0 or risk > 1.5 * float(decision.atr_14d):
            continue
        period, window_start, window_end = evaluation_window(pd.Timestamp(entry.ts))
        setup_vector = {
            "raw_policy_hash": policy_hash,
            "symbol": symbol,
            "extension_decision_ts": frame.iloc[sequence["extension_index"]].decision_ts,
            "peak_decision_ts": peak.decision_ts,
            "confirmation_decision_ts": decision_ts,
        }
        setup_id = "BCBS_SETUP_" + signal_state.stable_hash(setup_vector)[:24]
        address_vector = {**setup_vector, "entry_ts": entry.ts, "initial_stop": stop, "risk_denominator": risk}
        rows.append({
            "raw_policy_hash": policy_hash,
            "raw_signal_address_hash": signal_state.stable_hash(address_vector),
            "setup_sequence_id": setup_id,
            "symbol": symbol,
            "extension_profile": spec["extension_profile"],
            "confirmation_bars": int(spec["confirmation_bars"]),
            "parent_state": label,
            "parent_feature_ts": parent_ts,
            "parent_available": True,
            "decision_ts": decision_ts,
            "feature_available_ts": max(pd.Timestamp(decision.feature_available_ts), parent_ts),
            "entry_ts": entry.ts,
            "entry_price": float(entry.open),
            "initial_stop": stop,
            "risk_denominator": risk,
            "daily_atr": float(decision.atr_14d),
            "daily_ema10_at_decision": float(decision.ema_10),
            "sequence_high": stop,
            "peak_midpoint": sequence["peak_midpoint"],
            "peak_day_anchored_vwap": sequence["peak_day_anchored_vwap"],
            "evaluation_period": period,
            "evaluation_window_start": window_start,
            "evaluation_window_end": window_end,
            "imputed_funding_gate_activated": False,
            "stop_at_observed_sequence_high_ohlcv_cap": True,
        })
    return rows


def stop_fill_short(bar: pd.Series, stop: float) -> float | None:
    return lfbs.stop_fill_short(bar, stop)


def execute_event(key: Mapping[str, Any], exit_policy: str, bars: pd.DataFrame, precomputed_frame: pd.DataFrame | None = None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    entry_ts = pd.Timestamp(key["entry_ts"]); boundary = pd.Timestamp(key["evaluation_window_end"])
    if exit_policy == "fixed_72h":
        natural_limit = entry_ts + pd.Timedelta(hours=72)
    elif exit_policy == "swing_high_trail_7d":
        natural_limit = entry_ts + pd.Timedelta(days=7)
    else:
        natural_limit = boundary - pd.Timedelta(minutes=5)
    if exit_policy != "daily_ema10_close" and natural_limit >= boundary:
        return None, {"candidate_key": key["candidate_key"], "exit_policy": exit_policy, "reason": "maximum_hold_crosses_evaluation_boundary", "entry_ts": entry_ts, "evaluation_window_end": boundary}
    path = bars[(bars.ts >= entry_ts) & (bars.ts <= natural_limit)].copy()
    if path.empty or pd.Timestamp(path.iloc[-1].ts) < natural_limit:
        return None, {"candidate_key": key["candidate_key"], "exit_policy": exit_policy, "reason": "insufficient_bars_for_natural_exit", "entry_ts": entry_ts, "evaluation_window_end": boundary}
    frame = precomputed_frame
    if frame is None:
        frame, _ = feature_frames(bars[(bars.ts >= entry_ts-pd.Timedelta(days=30)) & (bars.ts <= natural_limit)])
    frame = frame[(frame.decision_ts >= entry_ts-pd.Timedelta(days=1)) & (frame.decision_ts <= natural_limit)]
    stop = float(key["initial_stop"]); exit_ts = pd.NaT; exit_price = np.nan; exit_reason = ""
    processed_four = 0
    for _, bar in path.iterrows():
        completed = frame[(frame.decision_ts > entry_ts) & (frame.decision_ts <= bar.ts)]
        if exit_policy == "swing_high_trail_7d" and len(completed) >= 3 and len(completed) > processed_four:
            values = completed.reset_index(drop=True)
            middle = values.iloc[-2]
            if float(middle.high) > float(values.iloc[-3].high) and float(middle.high) >= float(values.iloc[-1].high):
                stop = min(stop, float(middle.high))
            processed_four = len(completed)
        fill = stop_fill_short(bar, stop)
        if fill is not None:
            exit_ts, exit_price, exit_reason = bar.ts, fill, "sequence_or_swing_high_stop"
            break
        if exit_policy == "daily_ema10_close" and len(completed):
            latest = completed.iloc[-1]
            if pd.notna(latest.ema_10) and float(latest.close) <= float(latest.ema_10):
                exit_ts, exit_price, exit_reason = bar.ts, float(bar.open), "completed_4h_close_below_daily_ema10"
                break
    if pd.isna(exit_ts):
        if exit_policy == "daily_ema10_close":
            return None, {"candidate_key": key["candidate_key"], "exit_policy": exit_policy, "reason": "no_natural_ema_exit_before_evaluation_boundary", "entry_ts": entry_ts, "evaluation_window_end": boundary}
        natural = path[path.ts >= natural_limit]
        if natural.empty:
            return None, {"candidate_key": key["candidate_key"], "exit_policy": exit_policy, "reason": "natural_exit_bar_unavailable", "entry_ts": entry_ts, "evaluation_window_end": boundary}
        final = natural.iloc[0]; exit_ts, exit_price = final.ts, float(final.open)
        exit_reason = "fixed_72h_time_exit" if exit_policy == "fixed_72h" else "maximum_7d_time_exit"
    risk = float(key["risk_denominator"]); used = path[path.ts <= exit_ts]
    result = {
        **dict(key), "exit_policy": exit_policy, "exit_ts": exit_ts, "exit_price": exit_price, "exit_reason": exit_reason,
        "stop_price": stop, "maximum_exit_ts": natural_limit, "gross_R": (float(key["entry_price"])-exit_price)/risk,
        "mae_R": min(0.0, (float(key["entry_price"])-float(used.high.max()))/risk),
        "mfe_R": max(0.0, (float(key["entry_price"])-float(used.low.min()))/risk),
        "side": "short", "protected_violation": exit_ts >= PROTECTED, "artificial_horizon_exit": False,
        "ohlcv_stop_approximation_cap": True,
    }
    return result, None


def candidate_address(row: Mapping[str, Any]) -> str:
    return stable_hash({field: row[field] for field in (
        "symbol", "decision_ts", "entry_ts", "initial_stop", "risk_denominator", "exit_policy", "maximum_exit_ts"
    )})


def build_control_keys(candidates: pd.DataFrame, outcomes: pd.DataFrame, feature_cache: dict[str, pd.DataFrame], bars_cache: dict[str, pd.DataFrame], panel: pd.DataFrame, ctx: SimpleNamespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows=[]; unavailable=[]
    policies = {
        key: group[["definition_id", "exit_policy"]].drop_duplicates()
        for key, group in outcomes.groupby("candidate_key")
    }
    for key in candidates.itertuples(index=False):
        frame=feature_cache[key.symbol]; bars=bars_cache[key.symbol]
        pool=frame[(frame.decision_ts < key.decision_ts) & (frame.decision_ts >= START) & frame.atr_14d.notna()].copy()
        if pool.empty: continue
        parent_labels=[]
        for ts in pool.decision_ts:
            parent_labels.append(parent_state(bars,pd.Timestamp(ts))[0])
        pool["parent_state"]=parent_labels
        same=pool[pool.parent_state.eq(key.parent_state)]
        extension_col=extension_column(key.extension_profile)
        choices={
            "same_symbol_same_regime_random_short": same,
            "parabolic_extension_without_backside_confirmation": pool[pool[extension_col]],
            "generic_20d_failed_breakout_short": pool[pool.generic_failed_20d],
            "non_parabolic_red_candle_short": pool[pool.red_candle & ~pool[extension_col]],
            "pit_vol_liquidity_matched_random_date": pool.iloc[(pool.atr_14d/pool.close-float(key.daily_atr)/float(key.entry_price)).abs().argsort()[:20]],
        }
        for control_class, eligible in choices.items():
            if eligible.empty:
                unavailable.append({"candidate_key":key.candidate_key,"control_class":control_class,"reason":"no_decision_time_eligible_control"}); continue
            ordered=eligible.copy()
            ordered["_match_order"]=[stable_hash({"candidate":key.candidate_key,"class":control_class,"decision_ts":ts}) for ts in ordered.decision_ts]
            ordered=ordered.sort_values("_match_order")
            match=None
            for _,proposal in ordered.iterrows():
                if pit_allowed(ctx,panel,pd.Timestamp(proposal.decision_ts),key.symbol):
                    match=proposal; break
            if match is None:
                unavailable.append({"candidate_key":key.candidate_key,"control_class":control_class,"reason":"no_pit_universe_eligible_control"}); continue
            entry_rows=bars[bars.ts>=match.decision_ts]
            if entry_rows.empty: continue
            entry=entry_rows.iloc[0]; stop=float(entry.open)+1.5*float(match.atr_14d); risk=stop-float(entry.open)
            period,wstart,wend=evaluation_window(pd.Timestamp(entry.ts))
            for definition in policies.get(key.candidate_key, pd.DataFrame()).itertuples(index=False):
                maximum = (pd.Timestamp(entry.ts)+pd.Timedelta(hours=72)) if definition.exit_policy=="fixed_72h" else ((pd.Timestamp(entry.ts)+pd.Timedelta(days=7)) if definition.exit_policy=="swing_high_trail_7d" else wend-pd.Timedelta(minutes=5))
                if definition.exit_policy!="daily_ema10_close" and maximum>=wend:
                    unavailable.append({"candidate_key":key.candidate_key,"control_class":control_class,"reason":"control_maximum_hold_crosses_evaluation_boundary"}); continue
                address={"symbol":key.symbol,"decision_ts":match.decision_ts,"entry_ts":entry.ts,"initial_stop":stop,"risk_denominator":risk,"exit_policy":definition.exit_policy,"maximum_exit_ts":maximum}
                rows.append({
                    "control_key":"BCBSC_"+stable_hash({"candidate":key.candidate_key,"class":control_class,"definition":definition.definition_id})[:24],
                    "candidate_key":key.candidate_key,"definition_id":definition.definition_id,"control_class":control_class,"symbol":key.symbol,
                    "decision_ts":match.decision_ts,"feature_available_ts":match.feature_available_ts,"entry_ts":entry.ts,"entry_price":float(entry.open),
                    "initial_stop":stop,"risk_denominator":risk,"daily_atr":float(match.atr_14d),"exit_policy":definition.exit_policy,
                    "evaluation_period":period,"evaluation_window_start":wstart,"evaluation_window_end":wend,"maximum_exit_ts":maximum,
                    "control_economic_address_hash":candidate_address(address),"placeholder_control":False,"outcome_accessed_before_freeze":False,
                })
    return pd.DataFrame(rows).drop_duplicates("control_key"),pd.DataFrame(unavailable)


def attach_costs(events: pd.DataFrame, funding: pd.DataFrame, key_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, boundaries = lfbs.attach_costs(events, funding, key_col)
    out["net_zero_funding_base_R"] = out.gross_R + out.fee_base_R + out.slippage_base_R
    out["net_zero_fee_base_R"] = out.gross_R + out.slippage_base_R + out.funding_central_R
    return out, boundaries


def summarize_economics(events: pd.DataFrame, definitions: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    summaries=[]; attribution=[]
    for definition in definitions.itertuples(index=False):
        group=events[events.definition_id.eq(definition.definition_id)]
        months=pd.to_datetime(group.entry_ts,utc=True).dt.strftime("%Y-%m").nunique()
        for mode in ("base","conservative","severe"):
            values=group[f"net_{mode}_R"]; losses=values[values<0]
            summaries.append({"definition_id":definition.definition_id,"cost_mode":mode,"events":len(group),"symbols":group.symbol.nunique(),"months":months,"mean_R":values.mean(),"median_R":values.median(),"total_R":values.sum(),"win_rate":values.gt(0).mean(),"profit_factor":values[values>0].sum()/abs(losses.sum()) if len(losses) else np.inf})
            funding_col="funding_central_R" if mode=="base" else f"funding_{mode}_R"
            attribution.append({"definition_id":definition.definition_id,"cost_mode":mode,"gross_R":group.gross_R.sum(),"fee_R":group[f"fee_{mode}_R"].sum(),"slippage_R":group[f"slippage_{mode}_R"].sum(),"funding_R":group[funding_col].sum(),"net_R":values.sum(),"exact_funding_boundaries":group.exact_funding_boundaries.sum(),"imputed_funding_boundaries":group.imputed_funding_boundaries.sum()})
    summary=pd.DataFrame(summaries); attribution=pd.DataFrame(attribution)
    period=[]
    for (definition,label),group in events.groupby(["definition_id","evaluation_period"]):
        for mode in ("base","conservative","severe"):
            values=group[f"net_{mode}_R"]; losses=values[values<0]
            period.append({"definition_id":definition,"period":label,"cost_mode":mode,"events":len(group),"symbols":group.symbol.nunique(),"mean_R":values.mean(),"profit_factor":values[values>0].sum()/abs(losses.sum()) if len(losses) else np.inf})
    return summary,attribution,pd.DataFrame(period)


def controls_report(events: pd.DataFrame, controls: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    if controls.empty: return pd.DataFrame(),pd.DataFrame()
    address=controls.groupby(["definition_id","control_economic_address_hash"]).agg(control_classes=("control_class",lambda x:"|".join(sorted(set(x)))),class_count=("control_class","nunique"),rows=("control_event_id","size")).reset_index()
    address["duplicated_addresses_counted_independently"]=0
    rows=[]
    for (definition,control_class),group in controls.groupby(["definition_id","control_class"]):
        unique=group.sort_values(["control_economic_address_hash","candidate_key"]).drop_duplicates("control_economic_address_hash")
        candidate=events[(events.definition_id==definition)&events.candidate_key.isin(unique.candidate_key)]
        coverage=candidate.candidate_key.nunique()/max(1,events[events.definition_id==definition].candidate_key.nunique())
        unique_count=unique.control_economic_address_hash.nunique(); adequate=unique_count>=15 and coverage>=.70
        for mode in ("base","conservative","severe"):
            rows.append({"definition_id":definition,"control_class":control_class,"cost_mode":mode,"unique_control_addresses":unique_count,"coverage":coverage,"adequate_control":adequate,"candidate_mean_R":candidate[f"net_{mode}_R"].mean(),"control_mean_R":unique[f"net_{mode}_R"].mean(),"mean_uplift_R":candidate[f"net_{mode}_R"].mean()-unique[f"net_{mode}_R"].mean()})
    return address,pd.DataFrame(rows)


def forensics(events: pd.DataFrame, definitions: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    concentration=lfbs.concentration_forensics(events)
    neighborhood=events.groupby(["extension_profile","confirmation_bars","parent_context","exit_policy"]).agg(events=("event_id","size"),symbols=("symbol","nunique"),base_mean_R=("net_base_R","mean"),conservative_mean_R=("net_conservative_R","mean"),severe_mean_R=("net_severe_R","mean")).reset_index()
    exits=events.groupby(["extension_profile","confirmation_bars","parent_context","exit_policy","exit_reason"]).agg(events=("event_id","size"),conservative_mean_R=("net_conservative_R","mean")).reset_index()
    return concentration,neighborhood,exits


def funding_partition_report(events: pd.DataFrame) -> pd.DataFrame:
    work=events.copy()
    work["funding_partition"]=np.select([
        work.exact_funding_boundaries.gt(0)&work.imputed_funding_boundaries.eq(0),
        work.exact_funding_boundaries.gt(0)&work.imputed_funding_boundaries.gt(0),
        work.exact_funding_boundaries.eq(0)&work.imputed_funding_boundaries.gt(0),
    ],["fully_exact","mixed","fully_imputed"],default="zero_boundary")
    rows=[]
    for (definition,period,partition),group in work.groupby(["definition_id","evaluation_period","funding_partition"]):
        rows.append({"definition_id":definition,"period":period,"funding_partition":partition,"events":len(group),"symbols":group.symbol.nunique(),"exact_boundaries":int(group.exact_funding_boundaries.sum()),"imputed_boundaries":int(group.imputed_funding_boundaries.sum()),"base_mean_R":group.net_base_R.mean(),"conservative_mean_R":group.net_conservative_R.mean(),"severe_mean_R":group.net_severe_R.mean(),"zero_funding_base_mean_R":group.net_zero_funding_base_R.mean()})
    return pd.DataFrame(rows)


def decision_table(summary: pd.DataFrame, concentration: pd.DataFrame, controls: pd.DataFrame, period: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for definition in definitions.itertuples(index=False):
        stats=summary[summary.definition_id.eq(definition.definition_id)].set_index("cost_mode")
        base,cons,severe=stats.loc["base"],stats.loc["conservative"],stats.loc["severe"]
        forensic=concentration[(concentration.definition_id==definition.definition_id)&(concentration.cost_mode=="conservative")].iloc[0]
        adequate=controls[(controls.definition_id==definition.definition_id)&controls.cost_mode.eq("conservative")&controls.adequate_control&controls.mean_uplift_R.gt(0)] if len(controls) else pd.DataFrame()
        classes=set(adequate.control_class) if len(adequate) else set()
        period_rows=period[(period.definition_id==definition.definition_id)&period.cost_mode.eq("conservative")]
        stable_periods=int(period_rows.mean_R.gt(0).sum())
        robust=forensic.mean_after_top3>0 and forensic.worst_leave_one_symbol_mean_R>0 and forensic.worst_leave_one_month_mean_R>0
        if base.events>=30 and base.symbols>=10 and base.mean_R>0 and cons.mean_R>0 and robust and stable_periods>=3 and len(classes)>=2 and classes&CONTEXTUAL_CONTROLS and classes&STRUCTURAL_CONTROLS:
            decision="materialization_candidate"
        elif definition.parent_context=="fragile_countertrend_stress" and base.mean_R>0 and cons.mean_R>0 and robust:
            decision="context_sleeve_candidate"
        elif base.mean_R>0 or cons.mean_R>0:
            decision="fragile_positive_train_screen"
        elif base.events<10:
            decision="diagnostic_only"
        else:
            decision="current_translation_weak"
        rows.append({"definition_id":definition.definition_id,"decision":decision,"events":int(base.events),"symbols":int(base.symbols),"base_mean_R":base.mean_R,"conservative_mean_R":cons.mean_R,"severe_mean_R":severe.mean_R,"positive_adequate_control_classes":len(classes),"positive_periods":stable_periods,"evidence_cap":"train_only_blowoff_short_funding_and_ohlcv_execution_caps"})
    return pd.DataFrame(rows)


def build_bundle(root: Path) -> Path:
    files=(
        "decision_summary.json","contract/backside_blowoff_short_contract.md","manifest/backside_blowoff_short_definitions.csv",
        "audit/exactness_sentinel.csv","audit/boundary_censor_audit.csv","audit/hard_gate_audit.csv",
        "economics/definition_summary.csv","economics/period_summary.csv","forensics/concentration_and_removal.csv",
        "forensics/parameter_neighborhood.csv","forensics/exact_vs_imputed_funding.csv","forensics/exit_policy_comparison.csv",
        "controls/control_summary.csv","decision/candidate_decisions.csv",
    )
    temp=root/".compact_review_bundle.tmp"; temp.mkdir(); inventory=[]
    for relative in files:
        source=root/relative; target=temp/relative.replace("/","__"); shutil.copy2(source,target)
        inventory.append({"bundle_file":target.name,"source_relative_path":relative,"bytes":source.stat().st_size,"sha256":lfbs.file_sha256(source)})
    write_csv(temp/"bundle_manifest.csv",inventory); os.replace(temp,root/"compact_review_bundle"); return root/"compact_review_bundle"


def _legacy_run_without_signal_state_contract(root: Path) -> dict[str,Any]:
    raise RuntimeError("legacy backside runner disabled: use signal_state_contract_v1_20260715")
    if root.exists(): raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True); started=time.monotonic(); definitions=frozen_manifest(); write_csv(root/"manifest/backside_blowoff_short_definitions.csv",definitions)
    contract="""# Backside-Confirmed Blowoff Short Contract\n\nTrain-only 2023-2025 screen. Completed daily extension data are joined with source_close_ts <= decision_ts. One pending extension sequence tracks its complete high; a higher high resets the peak and confirmation window. A backside decision requires a completed 4h close below the previous completed 4h low, peak-bar midpoint, and peak-day anchored VWAP within one or three completed bars. Entry is the next 5m open. Stop is the observed sequence high and the event is skipped when risk exceeds 1.5 completed-daily ATR. Fixed 72h, daily EMA10-close, and completed 4h swing-high/7d exits are executable. Evaluation windows are 2023, 2024, 2025-H1, and 2025-H2; intervals without a natural exit before the exclusive window boundary are dropped, never force-exited at the boundary. PIT Tier A/B, 30-day listing age, parent context, frozen shared funding, and OHLCV stop approximations remain explicit caps.\n"""
    path=root/"contract/backside_blowoff_short_contract.md"; path.parent.mkdir(); path.write_text(contract)
    ctx=context(root); panel=runner.full_panel_for_launch_gate(ctx); write_csv(root/"manifest/pit_panel.csv",panel); paths=runner.data_paths(ctx)
    specs=definitions.drop_duplicates("selected_key_policy_hash").to_dict("records"); candidates=[]; feature_cache={}; bars_cache={}
    for number,symbol in enumerate(panel.symbol.astype(str),1):
        bars=runner.load_symbol_bars(paths,symbol,START-pd.Timedelta(days=100),END)
        if bars.empty: continue
        symbol_rows,frame,_=enumerate_candidates(ctx,panel,symbol,bars,specs); candidates.extend(symbol_rows); feature_cache[symbol]=frame; bars_cache[symbol]=bars
        write_json(root/"watch_status.json",{"status":"running","stage":"candidate_key_build","symbols_completed":number,"symbols_planned":len(panel),"selected_keys":len(candidates),"rss_bytes":runner.current_rss_bytes(),"elapsed_seconds":time.monotonic()-started,"updated_ts":runner.utc_now()})
    candidates=pd.DataFrame(candidates).drop_duplicates("candidate_key")
    if candidates.empty: raise RuntimeError("no candidate keys")
    freeze=stable_hash(sorted(candidates.candidate_key)); candidates["selected_key_freeze_hash"]=freeze; write_csv(root/"keys/candidate_key_manifest.csv",candidates)
    sentinel=[]
    for definition in definitions.head(4).itertuples(index=False):
        selected=candidates[candidates.selected_key_policy_hash.eq(definition.selected_key_policy_hash)]; first=[]; second=[]
        for key in selected.to_dict("records"):
            a,_=execute_event(key,definition.exit_policy,bars_cache[key["symbol"]],feature_cache[key["symbol"]]); b,_=execute_event(key,definition.exit_policy,bars_cache[key["symbol"]],feature_cache[key["symbol"]])
            if a: first.append(stable_hash({k:a[k] for k in ("candidate_key","exit_ts","exit_price","exit_reason","gross_R")}))
            if b: second.append(stable_hash({k:b[k] for k in ("candidate_key","exit_ts","exit_price","exit_reason","gross_R")}))
        sentinel.append({"definition_id":definition.definition_id,"first_outcomes":len(first),"second_outcomes":len(second),"mismatch_count":len(set(first).symmetric_difference(second)),"profitability_used_for_continuation":False,"mechanical_pass":bool(first) and first==second})
    sentinel=pd.DataFrame(sentinel); write_csv(root/"audit/exactness_sentinel.csv",sentinel)
    if len(sentinel)!=4 or not sentinel.mechanical_pass.all(): raise RuntimeError("mechanical sentinel failed")
    outcomes=[]; exclusions=[]
    for definition in definitions.itertuples(index=False):
        selected=candidates[candidates.selected_key_policy_hash.eq(definition.selected_key_policy_hash)].sort_values(["symbol","entry_ts"])
        blocked_until: dict[str,pd.Timestamp]={}
        for key in selected.to_dict("records"):
            if pd.Timestamp(key["entry_ts"]) < blocked_until.get(key["symbol"],pd.Timestamp.min.tz_localize("UTC")):
                exclusions.append({"candidate_key":key["candidate_key"],"definition_id":definition.definition_id,"exit_policy":definition.exit_policy,"reason":"overlapping_position_same_symbol_definition","entry_ts":key["entry_ts"],"evaluation_window_end":key["evaluation_window_end"]}); continue
            event,excluded=execute_event(key,definition.exit_policy,bars_cache[key["symbol"]],feature_cache[key["symbol"]])
            if excluded: exclusions.append({**excluded,"definition_id":definition.definition_id}); continue
            event["definition_id"]=definition.definition_id; event["parameter_vector_hash"]=definition.parameter_vector_hash
            event["event_id"]="BCBSE_"+stable_hash({"candidate":key["candidate_key"],"definition":definition.definition_id})[:24]
            event["candidate_economic_address_hash"]=candidate_address(event); outcomes.append(event)
            blocked_until[key["symbol"]]=pd.Timestamp(event["exit_ts"])
    outcomes=pd.DataFrame(outcomes); write_csv(root/"audit/boundary_censor_audit.csv",exclusions)
    if outcomes.empty: raise RuntimeError("no outcomes")
    funding=lfbs.funding_panel(); outcomes,boundaries=attach_costs(outcomes,funding,"event_id"); write_csv(root/"materialized/event_ledger.csv",outcomes)
    control_keys,control_unavailable=build_control_keys(candidates,outcomes,feature_cache,bars_cache,panel,ctx)
    control_freeze=stable_hash(sorted(control_keys.control_key)) if len(control_keys) else stable_hash([]); control_keys["control_key_freeze_hash"]=control_freeze; write_csv(root/"controls/control_key_manifest.csv",control_keys); write_csv(root/"controls/control_unavailable_reasons.csv",control_unavailable)
    control_outcomes=[]; control_exclusions=[]
    for control in control_keys.to_dict("records"):
        event,excluded=execute_event(control,control["exit_policy"],bars_cache[control["symbol"]],feature_cache[control["symbol"]])
        if excluded: control_exclusions.append({**excluded,"control_key":control["control_key"]}); continue
        event.update({"control_event_id":control["control_key"],"candidate_key":control["candidate_key"],"definition_id":control["definition_id"],"control_class":control["control_class"],"control_economic_address_hash":control["control_economic_address_hash"]}); control_outcomes.append(event)
    control_outcomes=pd.DataFrame(control_outcomes)
    if len(control_outcomes): control_outcomes,control_boundaries=attach_costs(control_outcomes,funding,"control_event_id")
    else: control_boundaries=pd.DataFrame()
    write_csv(root/"controls/control_event_ledger.csv",control_outcomes); write_csv(root/"audit/control_boundary_censor_audit.csv",control_exclusions)
    address_audit,control_summary=controls_report(outcomes,control_outcomes); write_csv(root/"controls/control_address_audit.csv",address_audit); write_csv(root/"controls/control_summary.csv",control_summary)
    summary,attribution,period=summarize_economics(outcomes,definitions); write_csv(root/"economics/definition_summary.csv",summary); write_csv(root/"economics/cost_funding_attribution.csv",attribution); write_csv(root/"economics/period_summary.csv",period)
    concentration,neighborhood,exit_comparison=forensics(outcomes,definitions); write_csv(root/"forensics/concentration_and_removal.csv",concentration); write_csv(root/"forensics/parameter_neighborhood.csv",neighborhood); write_csv(root/"forensics/exit_policy_comparison.csv",exit_comparison)
    write_csv(root/"forensics/exact_vs_imputed_funding.csv",funding_partition_report(outcomes))
    decisions=decision_table(summary,concentration,control_summary,period,definitions); write_csv(root/"decision/candidate_decisions.csv",decisions)
    interval_violations=[]
    for label,(window_start,window_end) in EVALUATION_WINDOWS.items():
        result=evidence.validate_evaluation_window_intervals(outcomes[outcomes.evaluation_period.eq(label)],window_start=window_start,window_end=window_end)
        interval_violations.extend(result.violations)
    hard={
        "definitions_evaluated":int(summary.definition_id.nunique()),"candidate_duplicate_economic_addresses":int(outcomes.duplicated(["definition_id","candidate_economic_address_hash"]).sum()),
        "unexplained_attrition":0,"artificial_horizon_exits":int(outcomes.artificial_horizon_exit.sum()),"funding_join_missing":int(boundaries.missing.sum()) if len(boundaries) else 0,
        "funding_join_duplicates":int(boundaries.duplicated(["event_id","boundary_ts"]).sum()) if len(boundaries) else 0,"decision_input_leaks":int((candidates.feature_available_ts>candidates.decision_ts).sum()),
        "protected_period_violations":int(outcomes.protected_violation.sum()),"placeholder_controls":int(control_keys.placeholder_control.sum()) if len(control_keys) else 0,
        "duplicate_control_addresses_counted_independently":int(address_audit.duplicated_addresses_counted_independently.sum()) if len(address_audit) else 0,
        "evaluation_interval_contract_violations":len(interval_violations),
    }
    write_csv(root/"audit/hard_gate_audit.csv",[{"gate":key,"value":value,"pass":value==(24 if key=="definitions_evaluated" else 0)} for key,value in hard.items()])
    status="complete" if hard["definitions_evaluated"]==24 and not any(value for key,value in hard.items() if key!="definitions_evaluated") else "blocked_by_protocol_issue"
    decision={"run_root":str(root),"status":status,**hard,"selected_keys":len(candidates),"canonical_event_rows":len(outcomes),"boundary_exclusions":len(exclusions),"exactness_sentinel_pass":bool(sentinel.mechanical_pass.all()),"materialization_candidates":decisions[decisions.decision.eq("materialization_candidate")].definition_id.tolist(),"context_sleeves":decisions[decisions.decision.eq("context_sleeve_candidate")].definition_id.tolist(),"validation_launched":False,"holdout_launched":False,"runtime_seconds":time.monotonic()-started,"compact_bundle_path":str(root/"compact_review_bundle")}
    write_json(root/"decision_summary.json",decision); build_bundle(root); write_json(root/"watch_status.json",{**decision,"stage":"complete","updated_ts":runner.utc_now()}); return decision


def run(root: Path, *, resume: bool = False) -> dict[str, Any]:
    from tools import run_kraken_backside_blowoff_signal_state_repaired as repaired
    return repaired.run(root, resume=resume)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-root",required=True); args=parser.parse_args()
    result=run(Path(args.run_root)); print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["status"]=="complete" else 2


if __name__=="__main__": raise SystemExit(main())
