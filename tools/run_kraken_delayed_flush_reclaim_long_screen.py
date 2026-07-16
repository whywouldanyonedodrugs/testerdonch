#!/usr/bin/env python3
"""Train-only Kraken delayed-flush reclaim long screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tools import qlmg_evidence_contracts as evidence
from tools import run_kraken_backside_blowoff_short_screen as reports
from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools import run_kraken_liquid_failed_breakout_short_screen as lfbs
from tools import run_kraken_riskoff_failed_bounce_short_screen as rfbs


START = pd.Timestamp("2023-01-01", tz="UTC")
PROTECTED = pd.Timestamp("2026-01-01", tz="UTC")
END = PROTECTED-pd.Timedelta(minutes=5)
CONTRACT_VERSION = "kraken_delayed_flush_reclaim_long_v1_20260715"
EVALUATION_WINDOWS = rfbs.EVALUATION_WINDOWS
RISK_MATCH_TOLERANCE_ATR = 0.25
CONTROL_CLASSES = (
    "same_symbol_same_parent_state_random_long",
    "large_flush_without_completed_reclaim",
    "completed_reclaim_without_qualifying_flush",
    "generic_20d_breakdown_failure_reclaim_long",
    "pit_vol_liquidity_matched_random_long",
)
CONTEXTUAL_CONTROLS = {"same_symbol_same_parent_state_random_long", "pit_vol_liquidity_matched_random_long"}
STRUCTURAL_CONTROLS = {
    "large_flush_without_completed_reclaim", "completed_reclaim_without_qualifying_flush",
    "generic_20d_breakdown_failure_reclaim_long",
}


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def parameter_hash(row: Mapping[str, Any], *, selected_key: bool) -> str:
    fields = ["flush_profile", "stabilization_bars", "parent_policy"]
    if not selected_key: fields.append("exit_policy")
    vector = {field: row[field] for field in fields}
    vector.update({
        "side": "long", "signal_timeframe": "4h_completed", "execution_timeframe": "5m_next_open",
        "universe_policy": "pit_kraken_tier_ab", "minimum_live_days": 30,
        "flush_reference": "completed_daily_bar_exactly_3_or_5_bars_before_flush_close_high_and_atr14",
        "stabilization": "exactly_1_or_3_completed_4h_bars_without_new_sequence_low_lower_low_resets",
        "reclaim": "window_completion_close_above_previous_completed_4h_high_and_flush_anchored_vwap",
        "risk_band_completed_daily_atr": [0.25, 1.5], "protected_boundary": PROTECTED.isoformat(),
        "contract_version": CONTRACT_VERSION,
    })
    return stable_hash(vector)


def frozen_manifest() -> pd.DataFrame:
    rows = []
    for flush in ("moderate_12pct_3d_1.5atr", "strong_20pct_5d_2.5atr"):
        for stabilization in (1, 3):
            for parent in ("stress_both_down", "all_regime_comparator"):
                for exit_policy in ("fixed_72h", "daily_ema10_close", "swing_low_trail_7d"):
                    row = {"definition_id": f"dfrl_v1_{len(rows)+1:03d}", "flush_profile": flush, "stabilization_bars": stabilization, "parent_policy": parent, "exit_policy": exit_policy}
                    row["selected_key_policy_hash"] = parameter_hash(row, selected_key=True); row["parameter_vector_hash"] = parameter_hash(row, selected_key=False); rows.append(row)
    return pd.DataFrame(rows)


def context(root: Path) -> SimpleNamespace:
    return SimpleNamespace(run_root=root, start=START, end=END, args=SimpleNamespace(kraken_data_root=str(runner.DEFAULT_KRAKEN_DATA_ROOT), smoke=False))


def pit_allowed(ctx: SimpleNamespace, panel: pd.DataFrame, decision_ts: pd.Timestamp, symbol: str) -> bool:
    return lfbs.pit_allowed(ctx, panel, decision_ts, symbol)


def evaluation_window(ts: pd.Timestamp) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    return rfbs.evaluation_window(ts)


def feature_frames(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = bars.copy().sort_values("ts")
    work = work[
        work[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & work.high.ge(work[["open", "close"]].max(axis=1))
        & work.low.le(work[["open", "close"]].min(axis=1))
    ].copy()
    work["known_ts"] = pd.to_datetime(work.ts, utc=True)+pd.Timedelta(minutes=5)
    work["typical"] = (work.high+work.low+work.close)/3; work["vwap_num"] = work.typical*work.volume.fillna(0)
    four = work.set_index("known_ts").resample("4h", label="right", closed="right").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), execution_bar_count=("close", "count"),
    )
    four = four[(four.execution_bar_count >= 36) & four.close.notna()].reset_index().rename(columns={"known_ts": "decision_ts"})
    four["typical"] = (four.high+four.low+four.close)/3; four["vwap_num"] = four.typical*four.volume
    four["trailing_24h_vwap"] = four.vwap_num.rolling(6, min_periods=6).sum()/four.volume.rolling(6, min_periods=6).sum()
    daily = work.set_index("known_ts").resample("1D", label="right", closed="right").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"), bar_count=("close", "count"),
    )
    daily = daily[(daily.bar_count >= 250) & daily.close.notna()].reset_index().rename(columns={"known_ts": "daily_source_ts"})
    previous = daily.close.shift(1)
    tr = pd.concat([(daily.high-daily.low), (daily.high-previous).abs(), (daily.low-previous).abs()], axis=1).max(axis=1)
    daily["atr_14d"] = tr.rolling(14, min_periods=14).mean(); daily["ema_10"] = daily.close.ewm(span=10, adjust=False, min_periods=10).mean()
    daily["prior_low_20d"] = daily.low.rolling(20, min_periods=20).min().shift(1)
    for name, days, threshold, atr_multiple in (("moderate", 3, .12, 1.5), ("strong", 5, .20, 2.5)):
        daily[f"{name}_pre_close"] = daily.close.shift(days); daily[f"{name}_pre_high"] = daily.high.shift(days)
        daily[f"{name}_pre_atr"] = daily.atr_14d.shift(days); daily[f"{name}_pre_source_ts"] = daily.daily_source_ts.shift(days)
        daily[f"{name}_flush"] = (
            (daily.close/daily[f"{name}_pre_close"]-1 <= -threshold)
            & (daily[f"{name}_pre_high"]-daily.close >= atr_multiple*daily[f"{name}_pre_atr"])
        )
    keep = ["daily_source_ts", "close", "atr_14d", "ema_10", "prior_low_20d", "moderate_flush", "moderate_pre_high", "moderate_pre_atr", "moderate_pre_source_ts", "strong_flush", "strong_pre_high", "strong_pre_atr", "strong_pre_source_ts"]
    merged = daily[keep].rename(columns={"close": "daily_close"})
    frame = pd.merge_asof(four.sort_values("decision_ts"), merged.sort_values("daily_source_ts"), left_on="decision_ts", right_on="daily_source_ts", direction="backward", allow_exact_matches=True)
    for flag in ("moderate_flush", "strong_flush"):
        frame[flag] = frame[flag].eq(True)
    frame["bullish_reclaim_no_flush"] = (frame.close > frame.high.shift(1)) & (frame.close > frame.trailing_24h_vwap) & ~frame.moderate_flush & ~frame.strong_flush
    frame["generic_20d_breakdown_reclaim"] = (frame.low < frame.prior_low_20d) & (frame.close > frame.prior_low_20d)
    frame["feature_available_ts"] = frame[["decision_ts", "daily_source_ts"]].max(axis=1)
    return frame, work, daily


def flush_columns(profile: str) -> tuple[str, str, str, str]:
    prefix = "moderate" if profile.startswith("moderate") else "strong"
    return f"{prefix}_flush", f"{prefix}_pre_source_ts", f"{prefix}_pre_atr", f"{prefix}_pre_high"


def anchored_vwap(work: pd.DataFrame, anchor_ts: pd.Timestamp, decision_ts: pd.Timestamp) -> float:
    rows = work[(work.known_ts > anchor_ts) & (work.known_ts <= decision_ts)]; denominator = float(rows.volume.fillna(0).sum())
    return float(rows.vwap_num.sum()/denominator) if denominator > 0 else np.nan


def reclaim_sequences(frame: pd.DataFrame, work: pd.DataFrame, flush_profile: str, stabilization_bars: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    flush_col, anchor_col, atr_col, high_col = flush_columns(flush_profile)
    confirmed = []; expired = []; seen: set[pd.Timestamp] = set(); index = 0
    while index < len(frame):
        source = pd.Timestamp(frame.iloc[index].daily_source_ts) if pd.notna(frame.iloc[index].daily_source_ts) else pd.NaT
        if pd.isna(source) or source in seen or not bool(frame.iloc[index].get(flush_col, False)):
            index += 1; continue
        seen.add(source); anchor_ts = pd.Timestamp(frame.iloc[index][anchor_col]); low_index = index; no_new_low = 0; cursor = index+1
        while cursor < len(frame):
            current = frame.iloc[cursor]
            if float(current.low) < float(frame.iloc[low_index].low):
                low_index = cursor; no_new_low = 0; cursor += 1; continue
            no_new_low += 1
            if no_new_low < stabilization_bars:
                cursor += 1; continue
            previous = frame.iloc[cursor-1]; vwap = anchored_vwap(work, anchor_ts, pd.Timestamp(current.decision_ts))
            item = {
                "start_index": index, "low_index": low_index, "decision_index": cursor, "sequence_low": float(frame.iloc[index:cursor+1].low.min()),
                "anchor_ts": anchor_ts, "flush_anchored_vwap": vwap, "daily_atr": float(frame.iloc[index][atr_col]), "pre_flush_high": float(frame.iloc[index][high_col]),
            }
            if float(current.close) > float(previous.high) and pd.notna(vwap) and float(current.close) > vwap: confirmed.append(item)
            else: expired.append(item)
            index = cursor+1; break
        else:
            index += 1
    return confirmed, expired


def prepare_symbol(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    frame, work, daily = feature_frames(bars); frame = rfbs.attach_parent_state(frame); frame["feature_available_ts"] = frame[["feature_available_ts", "parent_source_ts"]].max(axis=1)
    sequences = {}
    for profile in ("moderate_12pct_3d_1.5atr", "strong_20pct_5d_2.5atr"):
        for stabilization in (1, 3):
            confirmed, expired = reclaim_sequences(frame, work, profile, stabilization); sequences[(profile, stabilization)] = {"confirmed": confirmed, "expired": expired}
    return frame, work, daily, sequences


def parent_allowed(policy: str, state: str) -> bool:
    return policy == "all_regime_comparator" and state != "unknown" or policy == "stress_both_down" and state == "both_down"


def enumerate_candidates(ctx: SimpleNamespace, panel: pd.DataFrame, symbol: str, bars: pd.DataFrame, specs: list[dict[str, Any]], prepared: tuple | None = None) -> tuple[list[dict[str, Any]], pd.DataFrame, dict]:
    frame, _, _, sequences = prepared or prepare_symbol(bars); rows = []; panel_row = panel[panel.symbol.eq(symbol)]
    if panel_row.empty or str(panel_row.iloc[0].status) != "available": return rows, frame, sequences
    listed = pd.Timestamp(panel_row.iloc[0].start_ts)
    for spec in specs:
        blocked_until = pd.Timestamp.min.tz_localize("UTC")
        for sequence in sequences[(spec["flush_profile"], int(spec["stabilization_bars"]))]["confirmed"]:
            decision = frame.iloc[sequence["decision_index"]]; decision_ts = pd.Timestamp(decision.decision_ts)
            if decision_ts < START or decision_ts >= PROTECTED or decision_ts < listed+pd.Timedelta(days=30) or not pit_allowed(ctx, panel, decision_ts, symbol): continue
            if not parent_allowed(spec["parent_policy"], str(decision.parent_state)) or pd.isna(decision.feature_available_ts) or pd.Timestamp(decision.feature_available_ts) > decision_ts: continue
            entries = bars[bars.ts >= decision_ts]
            if entries.empty or not np.isfinite(sequence["daily_atr"]) or sequence["daily_atr"] <= 0: continue
            entry = entries.iloc[0]
            if pd.Timestamp(entry.ts) < blocked_until: continue
            stop = float(sequence["sequence_low"]); risk = float(entry.open)-stop; risk_atr = risk/float(sequence["daily_atr"])
            if risk_atr < .25 or risk_atr > 1.5: continue
            period, wstart, wend = evaluation_window(pd.Timestamp(entry.ts)); vector = {"policy": spec["selected_key_policy_hash"], "symbol": symbol, "decision_ts": decision_ts, "entry_ts": entry.ts}
            rows.append({
                "candidate_key": "DFRLK_"+stable_hash(vector)[:24], "selected_key_policy_hash": spec["selected_key_policy_hash"], "symbol": symbol,
                "flush_profile": spec["flush_profile"], "stabilization_bars": spec["stabilization_bars"], "parent_policy": spec["parent_policy"], "parent_state": decision.parent_state,
                "decision_ts": decision_ts, "feature_available_ts": decision.feature_available_ts, "entry_ts": entry.ts, "entry_price": float(entry.open), "initial_stop": stop,
                "risk_denominator": risk, "risk_to_daily_atr": risk_atr, "daily_atr": float(sequence["daily_atr"]), "sequence_low": stop,
                "flush_anchor_ts": sequence["anchor_ts"], "flush_anchored_vwap": sequence["flush_anchored_vwap"], "pre_flush_high": sequence["pre_flush_high"],
                "evaluation_period": period, "evaluation_window_start": wstart, "evaluation_window_end": wend, "selected_key_frozen": True,
                "imputed_funding_gate_activated": False, "bar_based_forced_flow_proxy_only": True,
            }); blocked_until = pd.Timestamp(entry.ts)+pd.Timedelta(days=7)
    return rows, frame, sequences


def stop_fill_long(bar: pd.Series, stop: float) -> float | None:
    if float(bar.open) <= stop: return float(bar.open)
    if float(bar.low) <= stop: return stop
    return None


def execute_event(key: Mapping[str, Any], exit_policy: str, bars: pd.DataFrame, frame: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    entry_ts = pd.Timestamp(key["entry_ts"]); boundary = pd.Timestamp(key["evaluation_window_end"])
    natural_limit = entry_ts+(pd.Timedelta(hours=72) if exit_policy == "fixed_72h" else pd.Timedelta(days=7))
    if exit_policy == "daily_ema10_close": natural_limit = boundary-pd.Timedelta(minutes=5)
    elif natural_limit >= boundary: return None, {"candidate_key": key.get("candidate_key", key.get("control_key")), "exit_policy": exit_policy, "reason": "maximum_hold_crosses_evaluation_boundary", "entry_ts": entry_ts, "evaluation_window_end": boundary}
    path = bars[(bars.ts >= entry_ts) & (bars.ts <= natural_limit)].copy()
    if path.empty or (exit_policy != "daily_ema10_close" and pd.Timestamp(path.iloc[-1].ts) < natural_limit): return None, {"candidate_key": key.get("candidate_key", key.get("control_key")), "exit_policy": exit_policy, "reason": "insufficient_bars_for_natural_exit", "entry_ts": entry_ts, "evaluation_window_end": boundary}
    relevant = frame[(frame.decision_ts > entry_ts) & (frame.decision_ts <= natural_limit)]; stop = float(key["initial_stop"]); exit_ts = pd.NaT; exit_price = np.nan; reason = ""; processed = 0
    for _, bar in path.iterrows():
        completed = relevant[relevant.decision_ts <= bar.ts]
        if exit_policy == "swing_low_trail_7d" and len(completed) >= 3 and len(completed) > processed:
            values = completed.reset_index(drop=True); middle = values.iloc[-2]
            if float(middle.low) < float(values.iloc[-3].low) and float(middle.low) <= float(values.iloc[-1].low): stop = max(stop, float(middle.low))
            processed = len(completed)
        fill = stop_fill_long(bar, stop)
        if fill is not None: exit_ts, exit_price, reason = bar.ts, fill, "flush_or_swing_low_stop"; break
        if exit_policy == "daily_ema10_close" and len(completed):
            updates = completed[completed.decision_ts.eq(completed.daily_source_ts)]
            if len(updates):
                latest = updates.iloc[-1]
                if pd.notna(latest.ema_10) and float(latest.daily_close) <= float(latest.ema_10): exit_ts, exit_price, reason = bar.ts, float(bar.open), "completed_daily_close_below_ema10"; break
    if pd.isna(exit_ts):
        if exit_policy == "daily_ema10_close": return None, {"candidate_key": key.get("candidate_key", key.get("control_key")), "exit_policy": exit_policy, "reason": "no_natural_ema_exit_before_evaluation_boundary", "entry_ts": entry_ts, "evaluation_window_end": boundary}
        final = path[path.ts >= natural_limit]
        if final.empty: return None, {"candidate_key": key.get("candidate_key", key.get("control_key")), "exit_policy": exit_policy, "reason": "natural_exit_bar_unavailable", "entry_ts": entry_ts, "evaluation_window_end": boundary}
        final = final.iloc[0]; exit_ts, exit_price = final.ts, float(final.open); reason = "fixed_72h_time_exit" if exit_policy == "fixed_72h" else "maximum_7d_time_exit"
    risk = float(key["risk_denominator"]); used = path[path.ts <= exit_ts]
    return {**dict(key), "exit_policy": exit_policy, "exit_ts": exit_ts, "exit_price": exit_price, "exit_reason": reason, "stop_price": stop, "maximum_exit_ts": natural_limit,
            "gross_R": (exit_price-float(key["entry_price"]))/risk, "mae_R": min(0.0, (float(used.low.min())-float(key["entry_price"]))/risk), "mfe_R": max(0.0, (float(used.high.max())-float(key["entry_price"]))/risk),
            "side": "long", "protected_violation": exit_ts >= PROTECTED, "artificial_horizon_exit": False, "ohlcv_stop_approximation_cap": True}, None


def economic_address(row: Mapping[str, Any]) -> str:
    return stable_hash({field: row[field] for field in ("symbol", "decision_ts", "entry_ts", "initial_stop", "risk_denominator", "exit_policy", "maximum_exit_ts")})


def signal_address(row: Mapping[str, Any]) -> str:
    """Identity of an executable signal independent of policy and exit labels."""
    return stable_hash({field: row[field] for field in ("symbol", "decision_ts", "entry_ts", "entry_price", "initial_stop", "risk_denominator")})


def sequence_pool(frame: pd.DataFrame, items: list[dict[str, Any]], kind: str) -> pd.DataFrame:
    rows = []
    for item in items:
        source = frame.iloc[min(int(item["decision_index"]), len(frame)-1)]
        rows.append({**source.to_dict(), "sequence_low": item["sequence_low"], "sequence_daily_atr": item["daily_atr"], "sequence_kind": kind})
    return pd.DataFrame(rows)


def proposed_control_rows(frame: pd.DataFrame, eligible: pd.DataFrame, candidate_risk_atr: float) -> pd.DataFrame:
    rows = []
    for _, proposal in eligible.iterrows():
        history = frame[frame.decision_ts <= proposal.decision_ts].tail(3)
        stop = float(proposal.get("sequence_low", history.low.min()))
        entry_price = float(proposal.close); atr = float(proposal.get("sequence_daily_atr", proposal.atr_14d))
        risk = entry_price-stop; risk_atr = risk/atr if atr > 0 else np.nan
        if .25 <= risk_atr <= 1.5 and abs(risk_atr-candidate_risk_atr) <= RISK_MATCH_TOLERANCE_ATR:
            rows.append({**proposal.to_dict(), "proposed_stop": stop, "proposed_risk_atr": risk_atr, "risk_match_distance": abs(risk_atr-candidate_risk_atr)})
    return pd.DataFrame(rows)


def volatility_matched(eligible: pd.DataFrame, candidate_volatility: float) -> pd.DataFrame:
    """Apply the frozen +/-25% relative completed-daily ATR matching rule."""
    if eligible.empty or not np.isfinite(candidate_volatility) or candidate_volatility <= 0:
        return eligible.iloc[0:0].copy()
    work = eligible.copy()
    work["relative_daily_atr"] = work.atr_14d/work.close
    work["volatility_match_distance"] = (work.relative_daily_atr/candidate_volatility-1).abs()
    return work[work.volatility_match_distance <= .25].copy()


def build_control_keys(candidates: pd.DataFrame, outcomes: pd.DataFrame, feature_cache: dict[str, pd.DataFrame], sequence_cache: dict[str, dict], bars_cache: dict[str, pd.DataFrame], panel: pd.DataFrame, ctx: SimpleNamespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []; unavailable = []; policies = {key: group[["definition_id", "exit_policy"]].drop_duplicates() for key, group in outcomes.groupby("candidate_key")}
    for key in candidates.itertuples(index=False):
        frame = feature_cache[key.symbol]; bars = bars_cache[key.symbol]; historical = frame[(frame.decision_ts < key.decision_ts) & (frame.decision_ts >= START) & frame.atr_14d.notna()].copy()
        sequence_set = sequence_cache[key.symbol][(key.flush_profile, int(key.stabilization_bars))]; expired = sequence_pool(frame, sequence_set["expired"], "flush_without_reclaim")
        same = historical[historical.parent_state.eq(key.parent_state)]
        choices = {
            "same_symbol_same_parent_state_random_long": same,
            "large_flush_without_completed_reclaim": expired[expired.decision_ts < key.decision_ts] if len(expired) else expired,
            "completed_reclaim_without_qualifying_flush": historical[historical.bullish_reclaim_no_flush],
            "generic_20d_breakdown_failure_reclaim_long": historical[historical.generic_20d_breakdown_reclaim],
            # Same-symbol sampling fixes the liquidity identity; the frozen
            # relative-ATR rule below additionally matches volatility state.
            "pit_vol_liquidity_matched_random_long": volatility_matched(
                historical, float(key.daily_atr)/float(key.entry_price)
            ),
        }
        for control_class, eligible in choices.items():
            if eligible.empty: unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "zero_decision_time_eligible_controls"}); continue
            eligible = proposed_control_rows(frame, eligible, float(key.risk_to_daily_atr))
            if eligible.empty: unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "zero_controls_in_0.25_1.5_atr_band_and_0.25_atr_match_tolerance"}); continue
            eligible["_tie"] = [stable_hash({"candidate": key.candidate_key, "class": control_class, "decision_ts": ts}) for ts in eligible.decision_ts]
            ordered = eligible.sort_values(["risk_match_distance", "_tie"]); match = None
            for _, proposal in ordered.iterrows():
                if pit_allowed(ctx, panel, pd.Timestamp(proposal.decision_ts), key.symbol): match = proposal; break
            if match is None: unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "zero_pit_universe_eligible_risk_matched_controls"}); continue
            entries = bars[bars.ts >= match.decision_ts]
            if entries.empty: unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "next_executable_bar_unavailable"}); continue
            entry = entries.iloc[0]; stop = float(match.proposed_stop); atr = float(match.get("sequence_daily_atr", match.atr_14d)); risk = float(entry.open)-stop; risk_atr = risk/atr
            if not (.25 <= risk_atr <= 1.5 and abs(risk_atr-float(key.risk_to_daily_atr)) <= RISK_MATCH_TOLERANCE_ATR): unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "next_open_changed_risk_outside_frozen_match_band"}); continue
            period, wstart, wend = evaluation_window(pd.Timestamp(entry.ts))
            for definition in policies.get(key.candidate_key, pd.DataFrame()).itertuples(index=False):
                maximum = pd.Timestamp(entry.ts)+(pd.Timedelta(hours=72) if definition.exit_policy == "fixed_72h" else pd.Timedelta(days=7))
                if definition.exit_policy == "daily_ema10_close": maximum = wend-pd.Timedelta(minutes=5)
                elif maximum >= wend: unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "control_maximum_hold_crosses_evaluation_boundary"}); continue
                address = {"symbol": key.symbol, "decision_ts": match.decision_ts, "entry_ts": entry.ts, "initial_stop": stop, "risk_denominator": risk, "exit_policy": definition.exit_policy, "maximum_exit_ts": maximum}
                rows.append({"control_key": "DFRLC_"+stable_hash({"candidate": key.candidate_key, "class": control_class, "definition": definition.definition_id})[:24], "candidate_key": key.candidate_key, "definition_id": definition.definition_id,
                             "control_class": control_class, "symbol": key.symbol, "decision_ts": match.decision_ts, "feature_available_ts": match.feature_available_ts, "entry_ts": entry.ts, "entry_price": float(entry.open),
                             "initial_stop": stop, "risk_denominator": risk, "risk_to_daily_atr": risk_atr, "candidate_risk_to_daily_atr": key.risk_to_daily_atr, "risk_match_distance": abs(risk_atr-key.risk_to_daily_atr),
                             "daily_atr": atr, "exit_policy": definition.exit_policy, "evaluation_period": period, "evaluation_window_start": wstart, "evaluation_window_end": wend, "maximum_exit_ts": maximum,
                             "control_economic_address_hash": economic_address(address), "placeholder_control": False, "outcome_accessed_before_freeze": False})
    return pd.DataFrame(rows).drop_duplicates("control_key"), pd.DataFrame(unavailable)


def attach_long_costs(events: pd.DataFrame, panel: pd.DataFrame, key_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rate_columns = ["funding_rate_central", "funding_rate_conservative", "funding_rate_severe"]; panel_idx = panel.set_index(["symbol", "timestamp"])
    symbol_locations = panel.groupby("symbol")[rate_columns].median(); global_location = panel[rate_columns].median(); rows = []
    for event in events.itertuples(index=False):
        ratio = float(event.entry_price)/float(event.risk_denominator)
        for ts in pd.date_range(pd.Timestamp(event.entry_ts).ceil("h"), pd.Timestamp(event.exit_ts).floor("h"), freq="h"):
            try:
                source = panel_idx.loc[(event.symbol, ts)]; source = source.iloc[0] if isinstance(source, pd.DataFrame) else source; extension = False
            except KeyError:
                source = symbol_locations.loc[event.symbol] if event.symbol in symbol_locations.index else global_location; extension = True
            rows.append({key_col: getattr(event, key_col), "boundary_ts": ts, "missing": False, "funding_exact": bool(source.get("funding_exact", False)), "funding_imputed": bool(source.get("funding_imputed", True)),
                         "central_R": -float(source.funding_rate_central)*ratio, "conservative_R": -float(source.funding_rate_conservative)*ratio, "severe_R": -float(source.funding_rate_severe)*ratio,
                         "funding_gate_activated": False, "panel_extension": extension})
    boundary = pd.DataFrame(rows)
    sums = boundary.groupby(key_col).agg(funding_central_R=("central_R", "sum"), funding_conservative_R=("conservative_R", "sum"), funding_severe_R=("severe_R", "sum"), exact_funding_boundaries=("funding_exact", "sum"), imputed_funding_boundaries=("funding_imputed", "sum"), funding_boundary_count=("boundary_ts", "size")).reset_index() if len(boundary) else pd.DataFrame(columns=[key_col])
    out = events.merge(sums, on=key_col, how="left", validate="one_to_one").fillna({"funding_central_R": 0, "funding_conservative_R": 0, "funding_severe_R": 0, "exact_funding_boundaries": 0, "imputed_funding_boundaries": 0, "funding_boundary_count": 0})
    for mode, fee, slip, funding in (("base", 5, 4, "central"), ("conservative", 5, 8, "conservative"), ("severe", 10, 12, "severe")):
        out[f"fee_{mode}_R"] = -((out.entry_price+out.exit_price)/out.risk_denominator)*fee/10000; out[f"slippage_{mode}_R"] = -(out.entry_price/out.risk_denominator)*slip/10000
        out[f"net_{mode}_R"] = out.gross_R+out[f"fee_{mode}_R"]+out[f"slippage_{mode}_R"]+out[f"funding_{funding}_R"]
    out["funding_imputed_train_screen_cap"] = out.imputed_funding_boundaries.gt(0); out["net_zero_funding_base_R"] = out.gross_R+out.fee_base_R+out.slippage_base_R; out["net_zero_fee_base_R"] = out.gross_R+out.slippage_base_R+out.funding_central_R
    return out, boundary


def control_report(events: pd.DataFrame, controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    addresses = controls.groupby(["definition_id", "control_economic_address_hash"]).agg(control_classes=("control_class", lambda x: "|".join(sorted(set(x)))), rows=("control_event_id", "size")).reset_index(); addresses["duplicated_addresses_counted_independently"] = 0
    rows = []
    for definition in sorted(events.definition_id.unique()):
        candidate = events[events.definition_id.eq(definition)]
        for control_class in CONTROL_CLASSES:
            group = controls[(controls.definition_id == definition) & controls.control_class.eq(control_class)].sort_values(["control_economic_address_hash", "candidate_key"]).drop_duplicates("control_economic_address_hash")
            matched = candidate[candidate.candidate_key.isin(group.candidate_key)]; unmatched = candidate[~candidate.candidate_key.isin(group.candidate_key)]; coverage = matched.candidate_key.nunique()/max(1, candidate.candidate_key.nunique()); adequate = group.control_economic_address_hash.nunique() >= 15 and coverage >= .70
            for mode in ("base", "conservative", "severe"):
                rows.append({"definition_id": definition, "control_class": control_class, "cost_mode": mode, "matched_count": len(matched), "unmatched_count": len(unmatched), "full_count": len(candidate),
                             "unique_control_addresses": group.control_economic_address_hash.nunique(), "coverage": coverage, "adequate_control": adequate, "matched_candidate_mean_R": matched[f"net_{mode}_R"].mean(),
                             "unmatched_only_candidate_mean_R": unmatched[f"net_{mode}_R"].mean(), "full_candidate_mean_R": candidate[f"net_{mode}_R"].mean(), "control_mean_R": group[f"net_{mode}_R"].mean(),
                             "candidate_minus_control_mean_R": matched[f"net_{mode}_R"].mean()-group[f"net_{mode}_R"].mean(), "matched_minus_unmatched_mean_R": matched[f"net_{mode}_R"].mean()-unmatched[f"net_{mode}_R"].mean() if len(unmatched) else np.nan})
    return addresses, pd.DataFrame(rows)


def risk_stable_control_diagnostics(controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (definition, control_class), group in controls.groupby(["definition_id", "control_class"]):
        for field in ("risk_denominator", "risk_to_daily_atr", "risk_match_distance", "net_conservative_R"):
            values = group[field].dropna(); rows.append({"definition_id": definition, "control_class": control_class, "field": field, "rows": len(values), "minimum": values.min(), "q05": values.quantile(.05), "median": values.median(), "q95": values.quantile(.95), "maximum": values.max()})
    return pd.DataFrame(rows)


def path_report(events: pd.DataFrame, bars_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for event in events.itertuples(index=False):
        bars = bars_cache[event.symbol]; entry = pd.Timestamp(event.entry_ts)
        for hours in (24, 48, 72):
            horizon = entry+pd.Timedelta(hours=hours); used = bars[(bars.ts >= entry) & (bars.ts <= horizon)]; stops = used[(used.open <= event.initial_stop) | (used.low <= event.initial_stop)]
            if len(stops): bar = stops.iloc[0]; price = float(bar.open) if bar.open <= event.initial_stop else float(event.initial_stop); exit_ts = bar.ts; status = "stopped_before_horizon"
            else:
                target = bars[bars.ts >= horizon].head(1)
                if target.empty: continue
                bar = target.iloc[0]; price = float(bar.open); exit_ts = bar.ts; status = "marked_at_horizon_next_5m_open"
            path = bars[(bars.ts >= entry) & (bars.ts <= exit_ts)]
            rows.append({"definition_id": event.definition_id, "event_id": event.event_id, "horizon_hours": hours, "path_status": status, "path_exit_ts": exit_ts, "gross_R": (price-event.entry_price)/event.risk_denominator,
                         "mae_R": min(0.0, (path.low.min()-event.entry_price)/event.risk_denominator), "mfe_R": max(0.0, (path.high.max()-event.entry_price)/event.risk_denominator)})
    return pd.DataFrame(rows)


def overlap_audits(candidates: pd.DataFrame, outcomes: pd.DataFrame, definitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report policy-independent signal overlap and exit-aware trade overlap."""
    candidate = candidates.copy()
    candidate["canonical_signal_address_hash"] = [signal_address(row) for row in candidate.to_dict("records")]
    event = outcomes.copy()
    event["canonical_signal_address_hash"] = [signal_address(row) for row in event.to_dict("records")]
    event["trade_economic_address_hash"] = [economic_address(row) for row in event.to_dict("records")]
    pairwise = []
    for left_index, left in definitions.iterrows():
        left_rows = event[event.definition_id.eq(left.definition_id)]
        signal_left = set(left_rows.canonical_signal_address_hash)
        trade_left = set(left_rows.trade_economic_address_hash)
        for _, right in definitions.iloc[left_index+1:].iterrows():
            right_rows = event[event.definition_id.eq(right.definition_id)]
            signal_right = set(right_rows.canonical_signal_address_hash)
            trade_right = set(right_rows.trade_economic_address_hash)
            signal_union = signal_left | signal_right
            trade_union = trade_left | trade_right
            pairwise.append({
                "left_definition_id": left.definition_id,
                "right_definition_id": right.definition_id,
                "shared_signal_addresses": len(signal_left & signal_right),
                "signal_jaccard": len(signal_left & signal_right)/len(signal_union) if signal_union else np.nan,
                "shared_trade_economic_addresses": len(trade_left & trade_right),
                "trade_jaccard": len(trade_left & trade_right)/len(trade_union) if trade_union else np.nan,
            })
    nesting = []
    policy = definitions.drop_duplicates("selected_key_policy_hash")
    for (flush, stabilization), group in policy.groupby(["flush_profile", "stabilization_bars"]):
        strict = group[group.parent_policy.eq("stress_both_down")]
        broad = group[group.parent_policy.eq("all_regime_comparator")]
        if len(strict) and len(broad):
            strict_set = set(candidate[candidate.selected_key_policy_hash.eq(strict.iloc[0].selected_key_policy_hash)].canonical_signal_address_hash)
            broad_set = set(candidate[candidate.selected_key_policy_hash.eq(broad.iloc[0].selected_key_policy_hash)].canonical_signal_address_hash)
            nesting.append({"comparison": "strict_parent_within_all_regime", "flush_profile": flush, "left_stabilization_bars": stabilization, "right_stabilization_bars": stabilization,
                            "left_rows": len(strict_set), "right_rows": len(broad_set), "left_not_in_right": len(strict_set-broad_set), "pass": not (strict_set-broad_set), "explanation": "stress_both_down is a PIT subset of known all-regime states"})
    for (flush, parent), group in policy.groupby(["flush_profile", "parent_policy"]):
        one = group[group.stabilization_bars.eq(1)]
        three = group[group.stabilization_bars.eq(3)]
        if len(one) and len(three):
            one_set = set(candidate[candidate.selected_key_policy_hash.eq(one.iloc[0].selected_key_policy_hash)].canonical_signal_address_hash)
            three_set = set(candidate[candidate.selected_key_policy_hash.eq(three.iloc[0].selected_key_policy_hash)].canonical_signal_address_hash)
            nesting.append({"comparison": "one_vs_three_bar_exact_stabilization", "flush_profile": flush, "parent_policy": parent,
                            "left_stabilization_bars": 1, "right_stabilization_bars": 3, "left_rows": len(one_set), "right_rows": len(three_set), "left_not_in_right": len(one_set-three_set),
                            "pass": True, "explanation": "nesting_not_required: exact completion bars create different decisions and lower-low resets alter episode state"})
    return pd.DataFrame(pairwise), pd.DataFrame(nesting)


def decision_table(summary: pd.DataFrame, concentration: pd.DataFrame, controls: pd.DataFrame, period: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition in definitions.itertuples(index=False):
        stats = summary[summary.definition_id.eq(definition.definition_id)].set_index("cost_mode"); base, cons, severe = stats.loc["base"], stats.loc["conservative"], stats.loc["severe"]
        forensic_rows = concentration[(concentration.definition_id == definition.definition_id) & concentration.cost_mode.eq("conservative")]; robust = False if forensic_rows.empty else bool(forensic_rows.iloc[0].mean_after_top3 > 0 and forensic_rows.iloc[0].worst_leave_one_symbol_mean_R > 0 and forensic_rows.iloc[0].worst_leave_one_month_mean_R > 0)
        positive = controls[(controls.definition_id == definition.definition_id) & controls.cost_mode.eq("conservative") & controls.adequate_control & controls.candidate_minus_control_mean_R.gt(0)]; classes = set(positive.control_class); stable = int(period[(period.definition_id == definition.definition_id) & period.cost_mode.eq("conservative")].mean_R.gt(0).sum())
        if base.events >= 30 and base.symbols >= 10 and base.mean_R > 0 and cons.mean_R > 0 and robust and stable >= 3 and len(classes) >= 2 and classes & CONTEXTUAL_CONTROLS and classes & STRUCTURAL_CONTROLS: decision = "materialization_candidate"
        elif definition.parent_policy == "stress_both_down" and base.events >= 15 and base.mean_R > 0 and cons.mean_R > 0 and robust and classes & CONTEXTUAL_CONTROLS: decision = "fragile_context_sleeve"
        else: decision = "current_translation_weak"
        rows.append({"definition_id": definition.definition_id, "decision": decision, "events": int(base.events), "symbols": int(base.symbols), "base_mean_R": base.mean_R, "conservative_mean_R": cons.mean_R, "severe_mean_R": severe.mean_R, "positive_adequate_control_classes": len(classes), "positive_periods": stable, "evidence_cap": "train_only_bar_based_forced_flow_proxy_shared_funding_ohlcv_no_depth"})
    return pd.DataFrame(rows)


def build_bundle(root: Path) -> Path:
    files = ("decision_summary.json", "contract/delayed_flush_reclaim_contract.md", "manifest/delayed_flush_reclaim_definitions.csv", "audit/exactness_sentinel.csv", "audit/hard_gate_audit.csv", "audit/boundary_censor_audit.csv", "audit/pairwise_definition_overlap.csv", "audit/policy_nesting_audit.csv", "economics/definition_summary.csv", "economics/period_summary.csv", "forensics/concentration_and_removal.csv", "forensics/parameter_neighborhood.csv", "forensics/exact_vs_imputed_funding.csv", "forensics/horizon_path_behavior.csv", "controls/control_summary.csv", "controls/risk_stable_control_diagnostics.csv", "decision/candidate_decisions.csv", "candidate_library/delayed_flush_reclaim_candidate_library_update.csv", "reproducibility/run_manifest.json")
    temp = root/".compact_review_bundle.tmp"; temp.mkdir(); inventory = []
    for relative in files:
        source = root/relative; target = temp/relative.replace("/", "__"); shutil.copy2(source, target); inventory.append({"bundle_file": target.name, "source_relative_path": relative, "bytes": source.stat().st_size, "sha256": file_hash(source)})
    write_csv(temp/"bundle_manifest.csv", inventory); os.replace(temp, root/"compact_review_bundle"); return root/"compact_review_bundle"


def run(root: Path) -> dict[str, Any]:
    if root.exists(): raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True); started = time.monotonic(); definitions = frozen_manifest(); write_csv(root/"manifest/delayed_flush_reclaim_definitions.csv", definitions)
    contract = """# Delayed Flush Reclaim Long Contract

Train-only 2023-2025 bar-based forced-flow proxy; it does not claim historical liquidation or OI confirmation. Moderate/strong flush compares a completed daily close with the completed close exactly three/five daily bars earlier. The pre-flush reference high and ATR14 are the high and completed ATR14 from that same shifted bar. Required decline/displacement is 12%/1.5 ATR or 20%/2.5 ATR. One pending sequence tracks its complete low; lower completed 4h lows reset the stabilization count. Reclaim is tested only on the completed 4h bar that completes exactly one/three consecutive no-new-low bars and requires close above the previous completed 4h high and VWAP anchored immediately after the pre-flush daily close. This exact-window rule prevents indefinite stale flush signals. Entry is next 5m open and cannot occur on the flush bar.

Stress parent requires completed PIT BTC and ETH both-down; comparator admits every known parent state. Stop is complete sequence low. Entry risk must be 0.25-1.5 pre-flush completed-daily ATR. Fixed 72h, completed daily close below EMA10, and completed 4h swing-low trail with seven-day maximum are executable. Evaluation-window crossings are dropped, never force-exited. Controls use the same risk band and absolute risk-to-ATR matching tolerance of 0.25 before key freeze. The volatility/liquidity random control is same-symbol and requires completed relative daily ATR within 25% of the candidate state. Shared exact/imputed funding is long-signed outcome cost only. No OI or liquidation field activates a signal.
"""
    path = root/"contract/delayed_flush_reclaim_contract.md"; path.parent.mkdir(); path.write_text(contract, encoding="utf-8")
    ctx = context(root); panel = runner.full_panel_for_launch_gate(ctx); write_csv(root/"manifest/pit_panel.csv", panel); paths = runner.data_paths(ctx); specs = definitions.drop_duplicates("selected_key_policy_hash").to_dict("records")
    candidates = []; feature_cache = {}; sequence_cache = {}; bars_cache = {}; peak_rss = 0
    for number, symbol in enumerate(panel.symbol.astype(str), 1):
        bars = runner.load_symbol_bars(paths, symbol, START-pd.Timedelta(days=120), END)
        if bars.empty: continue
        bars = bars[["ts", "open", "high", "low", "close", "volume"]].copy(); prepared = prepare_symbol(bars); rows, frame, sequences = enumerate_candidates(ctx, panel, symbol, bars, specs, prepared); candidates.extend(rows)
        if rows: feature_cache[symbol] = frame; sequence_cache[symbol] = sequences; bars_cache[symbol] = bars
        peak_rss = max(peak_rss, runner.current_rss_bytes()); write_json(root/"watch_status.json", {"status": "running", "stage": "candidate_key_build", "symbols_completed": number, "symbols_planned": len(panel), "selected_keys": len(candidates), "rss_bytes": peak_rss, "elapsed_seconds": time.monotonic()-started, "updated_ts": runner.utc_now()})
    candidates = pd.DataFrame(candidates).drop_duplicates("candidate_key")
    if candidates.empty: raise RuntimeError("no delayed-flush reclaim candidate keys")
    freeze = stable_hash(sorted(candidates.candidate_key)); candidates["selected_key_freeze_hash"] = freeze; write_csv(root/"keys/candidate_key_manifest.csv", candidates)
    sentinel_rows = []
    for definition in definitions.sort_values("definition_id").groupby("selected_key_policy_hash", as_index=False).first().itertuples(index=False):
        selected = candidates[candidates.selected_key_policy_hash.eq(definition.selected_key_policy_hash)]; first = []; second = []
        for key in selected.to_dict("records"):
            a, _ = execute_event(key, definition.exit_policy, bars_cache[key["symbol"]], feature_cache[key["symbol"]]); b, _ = execute_event(key, definition.exit_policy, bars_cache[key["symbol"]], feature_cache[key["symbol"]])
            if a: first.append(stable_hash({field: a[field] for field in ("candidate_key", "exit_ts", "exit_price", "exit_reason", "gross_R")}))
            if b: second.append(stable_hash({field: b[field] for field in ("candidate_key", "exit_ts", "exit_price", "exit_reason", "gross_R")}))
        sentinel_rows.append({"definition_id": definition.definition_id, "selected_key_policy_hash": definition.selected_key_policy_hash, "first_outcomes": len(first), "second_outcomes": len(second), "mismatch_count": len(set(first).symmetric_difference(second)), "mechanical_pass": first == second, "profitability_used_for_continuation": False})
    sentinel = pd.DataFrame(sentinel_rows); write_csv(root/"audit/exactness_sentinel.csv", sentinel)
    if len(sentinel) != 8 or sentinel.selected_key_policy_hash.nunique() != 8 or not sentinel.mechanical_pass.all(): raise RuntimeError("selected-key exactness sentinel failed")
    outcomes = []; exclusions = []
    for definition in definitions.itertuples(index=False):
        selected = candidates[candidates.selected_key_policy_hash.eq(definition.selected_key_policy_hash)].sort_values(["symbol", "entry_ts"]); blocked = {}
        for key in selected.to_dict("records"):
            if pd.Timestamp(key["entry_ts"]) < blocked.get(key["symbol"], pd.Timestamp.min.tz_localize("UTC")): exclusions.append({"candidate_key": key["candidate_key"], "definition_id": definition.definition_id, "reason": "overlapping_position_same_symbol_definition"}); continue
            event, excluded = execute_event(key, definition.exit_policy, bars_cache[key["symbol"]], feature_cache[key["symbol"]])
            if excluded: exclusions.append({**excluded, "definition_id": definition.definition_id}); continue
            event["definition_id"] = definition.definition_id; event["parameter_vector_hash"] = definition.parameter_vector_hash; event["event_id"] = "DFRLE_"+stable_hash({"candidate": key["candidate_key"], "definition": definition.definition_id})[:24]; event["candidate_economic_address_hash"] = economic_address(event); outcomes.append(event); blocked[key["symbol"]] = pd.Timestamp(event["exit_ts"])
    outcomes = pd.DataFrame(outcomes); write_csv(root/"audit/boundary_censor_audit.csv", exclusions)
    if outcomes.empty: raise RuntimeError("no economic outcomes")
    funding = lfbs.funding_panel(); outcomes, boundaries = attach_long_costs(outcomes, funding, "event_id"); write_csv(root/"materialized/event_ledger.csv", outcomes)
    control_keys, unavailable = build_control_keys(candidates, outcomes, feature_cache, sequence_cache, bars_cache, panel, ctx); control_freeze = stable_hash(sorted(control_keys.control_key)) if len(control_keys) else stable_hash([])
    if len(control_keys): control_keys["control_key_freeze_hash"] = control_freeze
    write_csv(root/"controls/control_key_manifest.csv", control_keys); write_csv(root/"controls/control_unavailable_reasons.csv", unavailable)
    control_outcomes = []; control_exclusions = []
    for control in control_keys.to_dict("records"):
        event, excluded = execute_event(control, control["exit_policy"], bars_cache[control["symbol"]], feature_cache[control["symbol"]])
        if excluded: control_exclusions.append({**excluded, "control_key": control["control_key"]}); continue
        event.update({"control_event_id": control["control_key"], "candidate_key": control["candidate_key"], "definition_id": control["definition_id"], "control_class": control["control_class"], "control_economic_address_hash": control["control_economic_address_hash"]}); control_outcomes.append(event)
    control_outcomes = pd.DataFrame(control_outcomes)
    if len(control_outcomes): control_outcomes, control_boundaries = attach_long_costs(control_outcomes, funding, "control_event_id")
    else: control_boundaries = pd.DataFrame()
    write_csv(root/"controls/control_event_ledger.csv", control_outcomes); write_csv(root/"audit/control_boundary_censor_audit.csv", control_exclusions)
    address_audit, control_summary = control_report(outcomes, control_outcomes); write_csv(root/"controls/control_address_audit.csv", address_audit); write_csv(root/"controls/control_summary.csv", control_summary); write_csv(root/"controls/risk_stable_control_diagnostics.csv", risk_stable_control_diagnostics(control_outcomes))
    summary, attribution, period = reports.summarize_economics(outcomes, definitions); write_csv(root/"economics/definition_summary.csv", summary); write_csv(root/"economics/cost_funding_attribution.csv", attribution); write_csv(root/"economics/period_summary.csv", period)
    concentration = lfbs.concentration_forensics(outcomes); write_csv(root/"forensics/concentration_and_removal.csv", concentration)
    neighborhood = outcomes.groupby(["flush_profile", "stabilization_bars", "parent_policy", "exit_policy"]).agg(events=("event_id", "size"), symbols=("symbol", "nunique"), months=("entry_ts", lambda x: pd.to_datetime(x, utc=True).dt.strftime("%Y-%m").nunique()), base_mean_R=("net_base_R", "mean"), conservative_mean_R=("net_conservative_R", "mean"), severe_mean_R=("net_severe_R", "mean")).reset_index(); write_csv(root/"forensics/parameter_neighborhood.csv", neighborhood)
    write_csv(root/"forensics/exact_vs_imputed_funding.csv", reports.funding_partition_report(outcomes)); write_csv(root/"forensics/horizon_path_behavior.csv", path_report(outcomes, bars_cache))
    decisions = decision_table(summary, concentration, control_summary, period, definitions); write_csv(root/"decision/candidate_decisions.csv", decisions)
    overlap, nesting = overlap_audits(candidates, outcomes, definitions)
    write_csv(root/"audit/pairwise_definition_overlap.csv", overlap)
    write_csv(root/"audit/policy_nesting_audit.csv", nesting)
    interval_violations = []
    for label, (wstart, wend) in EVALUATION_WINDOWS.items(): interval_violations.extend(evidence.validate_evaluation_window_intervals(outcomes[outcomes.evaluation_period.eq(label)], window_start=wstart, window_end=wend).violations)
    hard = {"definitions_evaluated": int(summary.definition_id.nunique()), "sentinel_policy_hashes_covered": int(sentinel.selected_key_policy_hash.nunique()), "candidate_duplicate_addresses": int(outcomes.duplicated(["definition_id", "candidate_economic_address_hash"]).sum()),
            "artificial_boundary_exits": int(outcomes.artificial_horizon_exit.sum()), "funding_join_missing": int(boundaries.missing.sum()) if len(boundaries) else 0, "funding_join_duplicates": int(boundaries.duplicated(["event_id", "boundary_ts"]).sum()) if len(boundaries) else 0,
            "protected_period_violations": int(outcomes.protected_violation.sum()), "decision_input_leaks": int((candidates.feature_available_ts > candidates.decision_ts).sum()) + int((control_keys.feature_available_ts > control_keys.decision_ts).sum()) if len(control_keys) else int((candidates.feature_available_ts > candidates.decision_ts).sum()), "placeholder_controls": int(control_keys.placeholder_control.sum()) if len(control_keys) else 0,
            "duplicate_controls_counted_independently": int(address_audit.duplicated_addresses_counted_independently.sum()) if len(address_audit) else 0, "risk_band_control_violations": int(((control_keys.risk_to_daily_atr < .25) | (control_keys.risk_to_daily_atr > 1.5) | (control_keys.risk_match_distance > RISK_MATCH_TOLERANCE_ATR+1e-12)).sum()) if len(control_keys) else 0,
            "actual_complement_accounting_failures": int(((control_summary.matched_count+control_summary.unmatched_count) != control_summary.full_count).sum()), "evaluation_interval_contract_violations": len(interval_violations),
            "control_funding_join_missing": int(control_boundaries.missing.sum()) if len(control_boundaries) else 0,
            "control_funding_join_duplicates": int(control_boundaries.duplicated(["control_event_id", "boundary_ts"]).sum()) if len(control_boundaries) else 0,
            "control_outcomes_accessed_before_freeze": int(control_keys.outcome_accessed_before_freeze.sum()) if len(control_keys) else 0,
            "imputed_funding_signal_activations": int(candidates.imputed_funding_gate_activated.sum()),
            "strict_parent_nesting_failures": int((~nesting[nesting.comparison.eq("strict_parent_within_all_regime")]["pass"]).sum()) if len(nesting) else 0}
    write_csv(root/"audit/hard_gate_audit.csv", [{"gate": key, "value": value, "pass": value == (24 if key == "definitions_evaluated" else 8 if key == "sentinel_policy_hashes_covered" else 0)} for key, value in hard.items()])
    gate_pass = hard["definitions_evaluated"] == 24 and hard["sentinel_policy_hashes_covered"] == 8 and not any(v for k, v in hard.items() if k not in {"definitions_evaluated", "sentinel_policy_hashes_covered"})
    final = "focused_mechanical_repair_required" if not gate_pass else "materialization_candidate" if (decisions.decision == "materialization_candidate").any() else "fragile_context_sleeve" if (decisions.decision == "fragile_context_sleeve").any() else "current_translation_weak"
    library = []
    for definition in definitions.itertuples(index=False):
        row = decisions[decisions.definition_id.eq(definition.definition_id)].iloc[0]; library.append({"candidate_id": definition.definition_id, "candidate_definition_id": definition.definition_id, "definition_id": definition.definition_id, "hypothesis_id": "delayed_flush_reclaim_long", "family_engine_id": "kraken_dfrl_v1", "parameter_vector_hash": definition.parameter_vector_hash, "selected_key_policy_hash": definition.selected_key_policy_hash,
            "candidate_library_state": row.decision, "candidate_decision": row.decision, "evidence_level": "level_2_train_only_bounded_screen_capped", "evidence_level_contract": "train_only_not_validation_not_holdout_not_live", "clean_evidence_allowed": False, "mechanics_cap_active": True,
            "evidence_cap_reason": "bar_based_forced_flow_proxy_no_liquidation_or_oi_confirmation_shared_funding_imputation_ohlcv_no_depth", "family_rejected": False, "train_only": True, "validation_run": False, "holdout_touched": False, "live_ready": False, "can_support_strategy_claim": False,
            "event_rows": row.events, "symbols": row.symbols, "base_mean_R": row.base_mean_R, "conservative_mean_R": row.conservative_mean_R, "severe_mean_R": row.severe_mean_R, "source_run_root": str(root), "contract_version": CONTRACT_VERSION})
    write_csv(root/"candidate_library/delayed_flush_reclaim_candidate_library_update.csv", library)
    data_manifest = Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv"); funding_manifest = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1/funding/shared_funding_panel_manifest.csv")
    repro = {"commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "code_path": str(Path(__file__)), "code_hash": file_hash(Path(__file__)), "config_hash": file_hash(root/"manifest/delayed_flush_reclaim_definitions.csv"), "contract_hash": file_hash(path), "data_snapshot_manifest_hash": file_hash(data_manifest), "pit_universe_manifest_hash": file_hash(root/"manifest/pit_panel.csv"), "funding_manifest_hash": file_hash(funding_manifest), "protected_boundary": PROTECTED.isoformat(), "seed_values": [], "contract_type": "Kraken PF perpetual instruments; payoff unit type not inferred by this R-normalized OHLCV screen"}; write_json(root/"reproducibility/run_manifest.json", repro)
    decision = {"run_root": str(root), "status": "complete" if gate_pass else "blocked_by_protocol_issue", "final_decision": final, **hard, "selected_keys": len(candidates), "canonical_event_rows": len(outcomes), "control_event_rows": len(control_outcomes), "boundary_exclusions": len(exclusions), "peak_rss_bytes": peak_rss,
                "materialization_candidates": decisions[decisions.decision.eq("materialization_candidate")].definition_id.tolist(), "context_sleeves": decisions[decisions.decision.eq("fragile_context_sleeve")].definition_id.tolist(), "validation_launched": False, "cpcv_launched": False, "holdout_launched": False, "portfolio_construction_launched": False, "live_work_launched": False, "runtime_seconds": time.monotonic()-started, "compact_bundle_path": str(root/"compact_review_bundle")}
    write_json(root/"decision_summary.json", decision); build_bundle(root); write_json(root/"watch_status.json", {**decision, "stage": "complete", "updated_ts": runner.utc_now()}); return decision


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True); args = parser.parse_args(); result = run(Path(args.run_root)); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"] == "complete" else 2


if __name__ == "__main__": raise SystemExit(main())
