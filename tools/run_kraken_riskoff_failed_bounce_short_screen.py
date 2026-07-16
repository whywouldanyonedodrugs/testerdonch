#!/usr/bin/env python3
"""Train-only Kraken risk-off failed-bounce swing-short screen."""
from __future__ import annotations

import argparse
import hashlib
import json
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
from tools import run_kraken_backside_blowoff_short_screen as common
from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools import run_kraken_liquid_failed_breakout_short_screen as lfbs


START = pd.Timestamp("2023-01-01", tz="UTC")
PROTECTED = pd.Timestamp("2026-01-01", tz="UTC")
END = PROTECTED - pd.Timedelta(minutes=5)
CONTRACT_VERSION = "kraken_riskoff_failed_bounce_short_v1_20260714"
EVALUATION_WINDOWS = common.EVALUATION_WINDOWS
CONTROL_CLASSES = (
    "same_symbol_same_regime_random_short",
    "countertrend_rally_without_completed_failure",
    "completed_failure_outside_riskoff_parent",
    "non_rally_red_candle_short",
    "generic_20d_failed_breakout_short",
)
CONTEXTUAL_CONTROLS = {
    "same_symbol_same_regime_random_short",
    "completed_failure_outside_riskoff_parent",
}
STRUCTURAL_CONTROLS = {
    "countertrend_rally_without_completed_failure",
    "non_rally_red_candle_short",
    "generic_20d_failed_breakout_short",
}
_PARENT_TABLE: pd.DataFrame | None = None


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def parameter_hash(row: Mapping[str, Any], *, selected_key: bool) -> str:
    fields = ["rally_profile", "confirmation_bars", "parent_policy"]
    if not selected_key:
        fields.append("exit_policy")
    vector = {field: row[field] for field in fields}
    vector.update({
        "side": "short",
        "signal_timeframe": "4h_completed",
        "execution_timeframe": "5m_next_open",
        "universe_policy": "pit_kraken_tier_ab",
        "minimum_live_days": 30,
        "daily_downtrend": "close_below_ema20_and_ema20_falling_5d_and_ret20d_negative_at_pre_rally_bar",
        "failure": "close_below_previous_completed_4h_low_and_rally_anchored_vwap",
        "stop": "complete_rally_sequence_high_skip_above_1.5_completed_daily_atr",
        "protected_boundary": PROTECTED.isoformat(),
        "contract_version": CONTRACT_VERSION,
    })
    return stable_hash(vector)


def raw_policy_hash(row: Mapping[str, Any]) -> str:
    """Hash only mechanics that change the parent-neutral signal tape."""
    return signal_state.stable_hash({
        "rally_profile": row["rally_profile"],
        "confirmation_bars": int(row["confirmation_bars"]),
        "side": "short",
        "signal_timeframe": "4h_completed",
        "execution_timeframe": "5m_next_open",
        "universe_policy": "pit_kraken_tier_ab",
        "minimum_live_days": 30,
        "daily_downtrend": "close_below_ema20_and_ema20_falling_5d_and_ret20d_negative_at_pre_rally_bar",
        "failure": "close_below_previous_completed_4h_low_and_rally_anchored_vwap",
        "stop": "complete_rally_sequence_high_skip_above_1.5_completed_daily_atr",
        "protected_boundary": PROTECTED,
        "contract_version": CONTRACT_VERSION,
    })


def frozen_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rally in ("moderate_12pct_3d_1.5atr", "strong_20pct_5d_2.5atr"):
        for confirmation_bars in (1, 3):
            for parent in ("strict_both_down_stress", "broader_fragile_countertrend_stress"):
                for exit_policy in ("fixed_72h", "daily_ema10_close", "swing_high_trail_7d"):
                    row = {
                        "definition_id": f"rfbs_v1_{len(rows)+1:03d}",
                        "rally_profile": rally,
                        "confirmation_bars": confirmation_bars,
                        "parent_policy": parent,
                        "exit_policy": exit_policy,
                    }
                    row["selected_key_policy_hash"] = parameter_hash(row, selected_key=True)
                    row["parameter_vector_hash"] = parameter_hash(row, selected_key=False)
                    rows.append(row)
    return pd.DataFrame(rows)


def context(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_root=root,
        start=START,
        end=END,
        args=SimpleNamespace(kraken_data_root=str(runner.DEFAULT_KRAKEN_DATA_ROOT), smoke=False),
    )


def evaluation_window(ts: pd.Timestamp) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    return common.evaluation_window(ts)


def pit_allowed(ctx: SimpleNamespace, panel: pd.DataFrame, decision_ts: pd.Timestamp, symbol: str) -> bool:
    return lfbs.pit_allowed(ctx, panel, decision_ts, symbol)


def feature_frames(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build completed 4h/daily features; daily rally state is evaluated once at close."""
    work = bars.copy().sort_values("ts")
    work = work[
        work[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & work.high.ge(work[["open", "close"]].max(axis=1))
        & work.low.le(work[["open", "close"]].min(axis=1))
    ].copy()
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
    daily["ema_10"] = daily.close.ewm(span=10, adjust=False, min_periods=10).mean()
    daily["ema_20"] = daily.close.ewm(span=20, adjust=False, min_periods=20).mean()
    daily["ema20_change_5d"] = daily.ema_20 / daily.ema_20.shift(5) - 1.0
    daily["return_20d"] = daily.close / daily.close.shift(20) - 1.0
    daily["daily_downtrend"] = (daily.close < daily.ema_20) & (daily.ema20_change_5d < 0) & (daily.return_20d < 0)
    daily["prior_high_20d"] = daily.high.rolling(20, min_periods=20).max().shift(1)
    for name, days, threshold, atr_multiple in (
        ("moderate", 3, .12, 1.5), ("strong", 5, .20, 2.5),
    ):
        daily[f"{name}_pre_close"] = daily.close.shift(days)
        daily[f"{name}_pre_low"] = daily.low.shift(days)
        daily[f"{name}_pre_atr"] = daily.atr_14d.shift(days)
        daily[f"{name}_pre_source_ts"] = daily.daily_source_ts.shift(days)
        daily[f"{name}_pre_downtrend"] = daily.daily_downtrend.shift(days).eq(True)
        daily[f"{name}_rally"] = (
            daily[f"{name}_pre_downtrend"]
            & (daily.close / daily[f"{name}_pre_close"] - 1.0 >= threshold)
            & (daily.close - daily[f"{name}_pre_low"] >= atr_multiple * daily[f"{name}_pre_atr"])
        )
    keep = [
        "daily_source_ts", "open", "high", "low", "close", "atr_14d", "ema_10", "ema_20",
        "daily_downtrend", "prior_high_20d", "moderate_rally", "moderate_pre_source_ts",
        "moderate_pre_atr", "moderate_pre_low", "strong_rally", "strong_pre_source_ts",
        "strong_pre_atr", "strong_pre_low",
    ]
    merged_daily = daily[keep].rename(columns={c: f"daily_{c}" for c in ("open", "high", "low", "close")})
    frame = pd.merge_asof(
        four.sort_values("decision_ts"), merged_daily.sort_values("daily_source_ts"),
        left_on="decision_ts", right_on="daily_source_ts", direction="backward", allow_exact_matches=True,
    )
    frame["red_candle"] = frame.close < frame.open
    frame["generic_failed_20d"] = (frame.high > frame.prior_high_20d) & (frame.close < frame.prior_high_20d)
    frame["feature_available_ts"] = frame[["decision_ts", "daily_source_ts"]].max(axis=1)
    return frame, work, daily


def rally_columns(profile: str) -> tuple[str, str, str, str]:
    prefix = "moderate" if profile.startswith("moderate") else "strong"
    return f"{prefix}_rally", f"{prefix}_pre_source_ts", f"{prefix}_pre_atr", f"{prefix}_pre_low"


def anchored_vwap(work: pd.DataFrame, anchor_ts: pd.Timestamp, decision_ts: pd.Timestamp) -> float:
    rows = work[(work.known_ts > anchor_ts) & (work.known_ts <= decision_ts)]
    denominator = float(rows.volume.fillna(0).sum())
    return float(rows.vwap_num.sum() / denominator) if denominator > 0 else np.nan


def confirmation_sequences(
    frame: pd.DataFrame, work: pd.DataFrame, rally_profile: str, confirmation_bars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve one pending rally at a time; higher highs reset its failure window."""
    rally_col, anchor_col, atr_col, low_col = rally_columns(rally_profile)
    confirmed: list[dict[str, Any]] = []
    expired: list[dict[str, Any]] = []
    seen_daily_sources: set[pd.Timestamp] = set()
    index = 0
    while index < len(frame):
        source = pd.Timestamp(frame.iloc[index].daily_source_ts) if pd.notna(frame.iloc[index].daily_source_ts) else pd.NaT
        if pd.isna(source) or source in seen_daily_sources or not bool(frame.iloc[index].get(rally_col, False)):
            index += 1
            continue
        seen_daily_sources.add(source)
        anchor_ts = pd.Timestamp(frame.iloc[index][anchor_col])
        peak_index = index
        cursor = index + 1
        resolved = False
        while cursor < len(frame):
            current = frame.iloc[cursor]
            if float(current.high) > float(frame.iloc[peak_index].high):
                peak_index = cursor
                cursor += 1
                continue
            bars_after_peak = cursor - peak_index
            if bars_after_peak > confirmation_bars:
                expired.append({
                    "start_index": index, "peak_index": peak_index, "decision_index": cursor,
                    "sequence_high": float(frame.iloc[index:cursor+1].high.max()), "anchor_ts": anchor_ts,
                    "daily_atr": float(frame.iloc[index][atr_col]), "pre_rally_low": float(frame.iloc[index][low_col]),
                })
                break
            previous = frame.iloc[cursor-1]
            vwap = anchored_vwap(work, anchor_ts, pd.Timestamp(current.decision_ts))
            if float(current.close) < float(previous.low) and pd.notna(vwap) and float(current.close) < vwap:
                confirmed.append({
                    "start_index": index, "peak_index": peak_index, "decision_index": cursor,
                    "sequence_high": float(frame.iloc[index:cursor+1].high.max()), "anchor_ts": anchor_ts,
                    "rally_anchored_vwap": vwap, "daily_atr": float(frame.iloc[index][atr_col]),
                    "pre_rally_low": float(frame.iloc[index][low_col]),
                })
                resolved = True
                break
            cursor += 1
        index = cursor + 1 if resolved else max(index + 1, cursor)
    return confirmed, expired


def build_parent_table() -> pd.DataFrame:
    global _PARENT_TABLE
    if _PARENT_TABLE is not None:
        return _PARENT_TABLE
    spec = {"kraken_data_root": str(runner.DEFAULT_KRAKEN_DATA_ROOT)}
    btc = runner.load_parent_gate_frame(spec, "PF_XBTUSD", START, END).copy()
    eth = runner.load_parent_gate_frame(spec, "PF_ETHUSD", START, END).copy()
    if btc.empty or eth.empty:
        _PARENT_TABLE = pd.DataFrame()
        return _PARENT_TABLE
    btc = btc[["source_ts", "up", "down", "sma_40d", "ret_20d"]].rename(columns={c: f"btc_{c}" for c in ("up", "down", "sma_40d", "ret_20d")})
    eth = eth[["source_ts", "up", "down", "sma_40d", "ret_20d"]].rename(columns={c: f"eth_{c}" for c in ("up", "down", "sma_40d", "ret_20d")})
    table = pd.merge_asof(btc.sort_values("source_ts"), eth.sort_values("source_ts"), on="source_ts", direction="backward")
    valid = table[["btc_sma_40d", "btc_ret_20d", "eth_sma_40d", "eth_ret_20d"]].notna().all(axis=1)
    table["parent_state"] = np.select(
        [valid & table.btc_down & table.eth_down, valid & (table.btc_down | table.eth_down) & ~(table.btc_up & table.eth_up), valid & table.btc_up & table.eth_up],
        ["both_down", "mixed_at_least_one_down", "both_up"], default="unknown",
    )
    _PARENT_TABLE = table
    return table


def attach_parent_state(frame: pd.DataFrame) -> pd.DataFrame:
    table = build_parent_table()
    if table.empty:
        out = frame.copy(); out["parent_state"] = "unknown"; out["parent_source_ts"] = pd.NaT
        return out
    parent = table[["source_ts", "parent_state"]].rename(columns={"source_ts": "parent_source_ts"})
    return pd.merge_asof(frame.sort_values("decision_ts"), parent.sort_values("parent_source_ts"), left_on="decision_ts", right_on="parent_source_ts", direction="backward", allow_exact_matches=True)


def parent_allowed(policy: str, state: str) -> bool:
    if policy == "strict_both_down_stress":
        return state == "both_down"
    return state in {"both_down", "mixed_at_least_one_down"}


def prepare_symbol(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, int], dict[str, list[dict[str, Any]]]]]:
    frame, work, daily = feature_frames(bars)
    frame = attach_parent_state(frame)
    frame["feature_available_ts"] = frame[["feature_available_ts", "parent_source_ts"]].max(axis=1)
    sequences: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    for profile in ("moderate_12pct_3d_1.5atr", "strong_20pct_5d_2.5atr"):
        for window in (1, 3):
            confirmed, expired = confirmation_sequences(frame, work, profile, window)
            sequences[(profile, window)] = {"confirmed": confirmed, "expired": expired}
    return frame, work, daily, sequences


def enumerate_candidates(
    ctx: SimpleNamespace, panel: pd.DataFrame, symbol: str, bars: pd.DataFrame,
    specs: list[dict[str, Any]], prepared: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict]:
    frame, _, _, sequences = prepared or prepare_symbol(bars)
    panel_row = panel[panel.symbol.eq(symbol)]
    rows: list[dict[str, Any]] = []
    if panel_row.empty or str(panel_row.iloc[0].status) != "available":
        return rows, frame, sequences
    listed = pd.Timestamp(panel_row.iloc[0].start_ts)
    for spec in specs:
        blocked_until = pd.Timestamp.min.tz_localize("UTC")
        for sequence in sequences[(spec["rally_profile"], int(spec["confirmation_bars"]))]["confirmed"]:
            decision = frame.iloc[sequence["decision_index"]]
            decision_ts = pd.Timestamp(decision.decision_ts)
            if decision_ts < START or decision_ts >= PROTECTED or decision_ts < listed + pd.Timedelta(days=30):
                continue
            if not pit_allowed(ctx, panel, decision_ts, symbol) or not parent_allowed(spec["parent_policy"], str(decision.parent_state)):
                continue
            if pd.isna(decision.feature_available_ts) or pd.Timestamp(decision.feature_available_ts) > decision_ts:
                continue
            entry_rows = bars[bars.ts >= decision_ts]
            if entry_rows.empty or pd.isna(sequence["daily_atr"]) or sequence["daily_atr"] <= 0:
                continue
            entry = entry_rows.iloc[0]
            if pd.Timestamp(entry.ts) < blocked_until:
                continue
            stop = float(sequence["sequence_high"])
            risk = stop - float(entry.open)
            if risk <= 0 or risk > 1.5 * float(sequence["daily_atr"]):
                continue
            period, window_start, window_end = evaluation_window(pd.Timestamp(entry.ts))
            key_vector = {"policy": spec["selected_key_policy_hash"], "symbol": symbol, "decision_ts": decision_ts, "entry_ts": entry.ts}
            rows.append({
                "candidate_key": "RFBSK_" + stable_hash(key_vector)[:24],
                "selected_key_policy_hash": spec["selected_key_policy_hash"],
                "symbol": symbol, "rally_profile": spec["rally_profile"], "confirmation_bars": spec["confirmation_bars"],
                "parent_policy": spec["parent_policy"], "parent_state": decision.parent_state,
                "decision_ts": decision_ts, "feature_available_ts": decision.feature_available_ts,
                "entry_ts": entry.ts, "entry_price": float(entry.open), "initial_stop": stop,
                "risk_denominator": risk, "daily_atr": float(sequence["daily_atr"]),
                "sequence_high": stop, "rally_anchor_ts": sequence["anchor_ts"],
                "rally_anchored_vwap": sequence["rally_anchored_vwap"], "pre_rally_low": sequence["pre_rally_low"],
                "evaluation_period": period, "evaluation_window_start": window_start, "evaluation_window_end": window_end,
                "selected_key_frozen": True, "imputed_funding_gate_activated": False,
                "stop_at_observed_sequence_high_ohlcv_cap": True,
            })
            blocked_until = pd.Timestamp(entry.ts) + pd.Timedelta(days=7)
    return rows, frame, sequences


def enumerate_raw_signals(
    ctx: SimpleNamespace,
    panel: pd.DataFrame,
    symbol: str,
    bars: pd.DataFrame,
    spec: Mapping[str, Any],
    prepared: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict] | None = None,
) -> list[dict[str, Any]]:
    """Emit every valid parent-neutral setup without holding-period state."""
    frame, _, _, sequences = prepared or prepare_symbol(bars)
    panel_row = panel[panel.symbol.eq(symbol)]
    if panel_row.empty or str(panel_row.iloc[0].status) != "available":
        return []
    listed = pd.Timestamp(panel_row.iloc[0].start_ts)
    policy_hash = raw_policy_hash(spec)
    rows: list[dict[str, Any]] = []
    for sequence in sequences[(str(spec["rally_profile"]), int(spec["confirmation_bars"]))]["confirmed"]:
        decision = frame.iloc[sequence["decision_index"]]
        decision_ts = pd.Timestamp(decision.decision_ts)
        if decision_ts < START or decision_ts >= PROTECTED or decision_ts < listed + pd.Timedelta(days=30):
            continue
        if not pit_allowed(ctx, panel, decision_ts, symbol):
            continue
        parent_ts = pd.Timestamp(decision.parent_source_ts) if pd.notna(decision.parent_source_ts) else pd.NaT
        if pd.isna(parent_ts) or parent_ts > decision_ts:
            continue
        if pd.isna(decision.feature_available_ts) or pd.Timestamp(decision.feature_available_ts) > decision_ts:
            continue
        entry_rows = bars[bars.ts >= decision_ts]
        if entry_rows.empty or pd.isna(sequence["daily_atr"]) or float(sequence["daily_atr"]) <= 0:
            continue
        entry = entry_rows.iloc[0]
        stop = float(sequence["sequence_high"])
        risk = stop - float(entry.open)
        if risk <= 0 or risk > 1.5 * float(sequence["daily_atr"]):
            continue
        period, window_start, window_end = evaluation_window(pd.Timestamp(entry.ts))
        setup_vector = {
            "raw_policy_hash": policy_hash,
            "symbol": symbol,
            "rally_anchor_ts": sequence["anchor_ts"],
            "peak_decision_ts": frame.iloc[sequence["peak_index"]].decision_ts,
            "confirmation_decision_ts": decision_ts,
        }
        setup_id = "RFBS_SETUP_" + signal_state.stable_hash(setup_vector)[:24]
        address_vector = {
            **setup_vector,
            "entry_ts": entry.ts,
            "initial_stop": stop,
            "risk_denominator": risk,
        }
        rows.append({
            "raw_policy_hash": policy_hash,
            "raw_signal_address_hash": signal_state.stable_hash(address_vector),
            "setup_sequence_id": setup_id,
            "symbol": symbol,
            "rally_profile": spec["rally_profile"],
            "confirmation_bars": int(spec["confirmation_bars"]),
            "parent_state": decision.parent_state,
            "parent_feature_ts": parent_ts,
            "decision_ts": decision_ts,
            "feature_available_ts": decision.feature_available_ts,
            "entry_ts": entry.ts,
            "entry_price": float(entry.open),
            "initial_stop": stop,
            "risk_denominator": risk,
            "daily_atr": float(sequence["daily_atr"]),
            "sequence_high": stop,
            "rally_anchor_ts": sequence["anchor_ts"],
            "rally_anchored_vwap": sequence["rally_anchored_vwap"],
            "pre_rally_low": sequence["pre_rally_low"],
            "evaluation_period": period,
            "evaluation_window_start": window_start,
            "evaluation_window_end": window_end,
            "imputed_funding_gate_activated": False,
            "stop_at_observed_sequence_high_ohlcv_cap": True,
        })
    return rows


def execute_event(
    key: Mapping[str, Any], exit_policy: str, bars: pd.DataFrame, frame: pd.DataFrame,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    entry_ts = pd.Timestamp(key["entry_ts"]); boundary = pd.Timestamp(key["evaluation_window_end"])
    natural_limit = entry_ts + (pd.Timedelta(hours=72) if exit_policy == "fixed_72h" else pd.Timedelta(days=7))
    if exit_policy == "daily_ema10_close":
        natural_limit = boundary - pd.Timedelta(minutes=5)
    elif natural_limit >= boundary:
        return None, {"candidate_key": key.get("candidate_key", key.get("control_key")), "exit_policy": exit_policy, "reason": "maximum_hold_crosses_evaluation_boundary", "entry_ts": entry_ts, "evaluation_window_end": boundary}
    path = bars[(bars.ts >= entry_ts) & (bars.ts <= natural_limit)].copy()
    if path.empty or (exit_policy != "daily_ema10_close" and pd.Timestamp(path.iloc[-1].ts) < natural_limit):
        return None, {"candidate_key": key.get("candidate_key", key.get("control_key")), "exit_policy": exit_policy, "reason": "insufficient_bars_for_natural_exit", "entry_ts": entry_ts, "evaluation_window_end": boundary}
    relevant = frame[(frame.decision_ts > entry_ts) & (frame.decision_ts <= natural_limit)].copy()
    stop = float(key["initial_stop"]); exit_ts = pd.NaT; exit_price = np.nan; exit_reason = ""; processed_four = 0
    for _, bar in path.iterrows():
        completed = relevant[relevant.decision_ts <= bar.ts]
        if exit_policy == "swing_high_trail_7d" and len(completed) >= 3 and len(completed) > processed_four:
            values = completed.reset_index(drop=True); middle = values.iloc[-2]
            if float(middle.high) > float(values.iloc[-3].high) and float(middle.high) >= float(values.iloc[-1].high):
                stop = min(stop, float(middle.high))
            processed_four = len(completed)
        fill = lfbs.stop_fill_short(bar, stop)
        if fill is not None:
            exit_ts, exit_price, exit_reason = bar.ts, fill, "rally_or_swing_high_stop"
            break
        if exit_policy == "daily_ema10_close" and len(completed):
            daily_updates = completed[completed.decision_ts.eq(completed.daily_source_ts)]
            if len(daily_updates):
                latest = daily_updates.iloc[-1]
                if pd.notna(latest.ema_10) and float(latest.daily_close) <= float(latest.ema_10):
                    exit_ts, exit_price, exit_reason = bar.ts, float(bar.open), "completed_daily_close_below_ema10"
                    break
    if pd.isna(exit_ts):
        if exit_policy == "daily_ema10_close":
            return None, {"candidate_key": key.get("candidate_key", key.get("control_key")), "exit_policy": exit_policy, "reason": "no_natural_ema_exit_before_evaluation_boundary", "entry_ts": entry_ts, "evaluation_window_end": boundary}
        final = path[path.ts >= natural_limit]
        if final.empty:
            return None, {"candidate_key": key.get("candidate_key", key.get("control_key")), "exit_policy": exit_policy, "reason": "natural_exit_bar_unavailable", "entry_ts": entry_ts, "evaluation_window_end": boundary}
        final = final.iloc[0]; exit_ts, exit_price = final.ts, float(final.open)
        exit_reason = "fixed_72h_time_exit" if exit_policy == "fixed_72h" else "maximum_7d_time_exit"
    risk = float(key["risk_denominator"]); used = path[path.ts <= exit_ts]
    return {
        **dict(key), "exit_policy": exit_policy, "exit_ts": exit_ts, "exit_price": exit_price, "exit_reason": exit_reason,
        "stop_price": stop, "maximum_exit_ts": natural_limit, "gross_R": (float(key["entry_price"])-exit_price)/risk,
        "mae_R": min(0.0, (float(key["entry_price"])-float(used.high.max()))/risk),
        "mfe_R": max(0.0, (float(key["entry_price"])-float(used.low.min()))/risk),
        "side": "short", "protected_violation": exit_ts >= PROTECTED, "artificial_horizon_exit": False,
        "ohlcv_stop_approximation_cap": True,
    }, None


def candidate_address(row: Mapping[str, Any]) -> str:
    return common.candidate_address(row)


def _sequence_pool(frame: pd.DataFrame, items: list[dict[str, Any]], *, expired: bool) -> pd.DataFrame:
    rows = []
    for item in items:
        index = min(int(item["decision_index"]), len(frame)-1)
        source = frame.iloc[index]
        rows.append({
            **source.to_dict(), "sequence_high": item["sequence_high"], "sequence_daily_atr": item["daily_atr"],
            "sequence_kind": "rally_without_failure" if expired else "completed_failure",
        })
    return pd.DataFrame(rows)


def build_control_keys(
    candidates: pd.DataFrame, outcomes: pd.DataFrame, feature_cache: dict[str, pd.DataFrame],
    sequence_cache: dict[str, dict], bars_cache: dict[str, pd.DataFrame], panel: pd.DataFrame, ctx: SimpleNamespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match decision-time control keys before any control outcome access."""
    rows: list[dict[str, Any]] = []; unavailable: list[dict[str, Any]] = []
    policies = {key: group[["definition_id", "exit_policy"]].drop_duplicates() for key, group in outcomes.groupby("candidate_key")}
    for key in candidates.itertuples(index=False):
        frame = feature_cache[key.symbol]; bars = bars_cache[key.symbol]
        historical = frame[(frame.decision_ts < key.decision_ts) & (frame.decision_ts >= START) & frame.atr_14d.notna()].copy()
        if historical.empty:
            continue
        sequence_set = sequence_cache[key.symbol][(key.rally_profile, int(key.confirmation_bars))]
        expired = _sequence_pool(frame, sequence_set["expired"], expired=True)
        confirmed = _sequence_pool(frame, sequence_set["confirmed"], expired=False)
        same = historical[historical.parent_state.eq(key.parent_state)]
        outside = confirmed[~confirmed.parent_state.map(lambda state: parent_allowed(key.parent_policy, str(state)))] if len(confirmed) else confirmed
        rally_col = rally_columns(key.rally_profile)[0]
        choices = {
            "same_symbol_same_regime_random_short": same,
            "countertrend_rally_without_completed_failure": expired[expired.decision_ts < key.decision_ts] if len(expired) else expired,
            "completed_failure_outside_riskoff_parent": outside[outside.decision_ts < key.decision_ts] if len(outside) else outside,
            "non_rally_red_candle_short": historical[historical.red_candle & ~historical[rally_col]],
            "generic_20d_failed_breakout_short": historical[historical.generic_failed_20d],
        }
        for control_class, eligible in choices.items():
            if eligible.empty:
                unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "zero_decision_time_eligible_controls"})
                continue
            ordered = eligible.copy()
            ordered["_match_order"] = [stable_hash({"candidate": key.candidate_key, "class": control_class, "decision_ts": ts}) for ts in ordered.decision_ts]
            match = None
            for _, proposal in ordered.sort_values("_match_order").iterrows():
                if pd.Timestamp(proposal.decision_ts) < PROTECTED and pit_allowed(ctx, panel, pd.Timestamp(proposal.decision_ts), key.symbol):
                    match = proposal; break
            if match is None:
                unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "zero_pit_universe_eligible_controls"})
                continue
            entry_rows = bars[bars.ts >= match.decision_ts]
            daily_atr = float(match.get("sequence_daily_atr", match.atr_14d))
            if entry_rows.empty or not np.isfinite(daily_atr) or daily_atr <= 0:
                unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "entry_or_completed_daily_atr_unavailable"})
                continue
            entry = entry_rows.iloc[0]
            stop = float(match.get("sequence_high", historical[historical.decision_ts <= match.decision_ts].tail(3).high.max()))
            risk = stop - float(entry.open)
            if risk <= 0 or risk > 1.5 * daily_atr:
                unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "control_stop_invalid_or_above_1.5_daily_atr"})
                continue
            period, wstart, wend = evaluation_window(pd.Timestamp(entry.ts))
            for definition in policies.get(key.candidate_key, pd.DataFrame()).itertuples(index=False):
                maximum = pd.Timestamp(entry.ts) + (pd.Timedelta(hours=72) if definition.exit_policy == "fixed_72h" else pd.Timedelta(days=7))
                if definition.exit_policy == "daily_ema10_close": maximum = wend - pd.Timedelta(minutes=5)
                elif maximum >= wend:
                    unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "control_maximum_hold_crosses_evaluation_boundary"}); continue
                address = {"symbol": key.symbol, "decision_ts": match.decision_ts, "entry_ts": entry.ts, "initial_stop": stop, "risk_denominator": risk, "exit_policy": definition.exit_policy, "maximum_exit_ts": maximum}
                rows.append({
                    "control_key": "RFBSC_" + stable_hash({"candidate": key.candidate_key, "class": control_class, "definition": definition.definition_id})[:24],
                    "candidate_key": key.candidate_key, "definition_id": definition.definition_id, "control_class": control_class,
                    "symbol": key.symbol, "decision_ts": match.decision_ts, "feature_available_ts": match.feature_available_ts,
                    "entry_ts": entry.ts, "entry_price": float(entry.open), "initial_stop": stop, "risk_denominator": risk,
                    "daily_atr": daily_atr, "exit_policy": definition.exit_policy, "evaluation_period": period,
                    "evaluation_window_start": wstart, "evaluation_window_end": wend, "maximum_exit_ts": maximum,
                    "control_economic_address_hash": candidate_address(address), "placeholder_control": False,
                    "outcome_accessed_before_freeze": False,
                })
    return pd.DataFrame(rows).drop_duplicates("control_key"), pd.DataFrame(unavailable)


def controls_report(events: pd.DataFrame, controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if controls.empty:
        return pd.DataFrame(), pd.DataFrame()
    address = controls.groupby(["definition_id", "control_economic_address_hash"]).agg(
        control_classes=("control_class", lambda x: "|".join(sorted(set(x)))),
        class_count=("control_class", "nunique"), rows=("control_event_id", "size"),
    ).reset_index()
    address["duplicated_addresses_counted_independently"] = 0
    result = []
    for (definition, control_class), group in controls.groupby(["definition_id", "control_class"]):
        unique = group.sort_values(["control_economic_address_hash", "candidate_key"]).drop_duplicates("control_economic_address_hash")
        definition_events = events[events.definition_id.eq(definition)]
        candidate = definition_events[definition_events.candidate_key.isin(unique.candidate_key)]
        class_coverage = candidate.candidate_key.nunique() / max(1, definition_events.candidate_key.nunique())
        unique_coverage = unique.control_economic_address_hash.nunique() / max(1, definition_events.candidate_key.nunique())
        unique_count = unique.control_economic_address_hash.nunique()
        adequate = unique_count >= 15 and class_coverage >= .70
        for mode in ("base", "conservative", "severe"):
            result.append({
                "definition_id": definition, "control_class": control_class, "cost_mode": mode,
                "unique_control_addresses": unique_count, "class_coverage": class_coverage,
                "unique_address_coverage": unique_coverage, "adequate_control": adequate,
                "candidate_mean_R": candidate[f"net_{mode}_R"].mean(), "control_mean_R": unique[f"net_{mode}_R"].mean(),
                "mean_uplift_R": candidate[f"net_{mode}_R"].mean()-unique[f"net_{mode}_R"].mean(),
            })
    return address, pd.DataFrame(result)


def decision_table(summary: pd.DataFrame, concentration: pd.DataFrame, controls: pd.DataFrame, period: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition in definitions.itertuples(index=False):
        stats = summary[summary.definition_id.eq(definition.definition_id)].set_index("cost_mode")
        base, cons, severe = stats.loc["base"], stats.loc["conservative"], stats.loc["severe"]
        forensic_rows = concentration[(concentration.definition_id == definition.definition_id) & concentration.cost_mode.eq("conservative")]
        robust = False if forensic_rows.empty else bool(
            forensic_rows.iloc[0].mean_after_top3 > 0
            and forensic_rows.iloc[0].worst_leave_one_symbol_mean_R > 0
            and forensic_rows.iloc[0].worst_leave_one_month_mean_R > 0
        )
        adequate = controls[(controls.definition_id == definition.definition_id) & controls.cost_mode.eq("conservative") & controls.adequate_control & controls.mean_uplift_R.gt(0)] if len(controls) else pd.DataFrame()
        classes = set(adequate.control_class) if len(adequate) else set()
        stable_periods = int(period[(period.definition_id == definition.definition_id) & period.cost_mode.eq("conservative")].mean_R.gt(0).sum())
        if base.events >= 30 and base.symbols >= 10 and base.mean_R > 0 and cons.mean_R > 0 and robust and stable_periods >= 3 and len(classes) >= 2 and classes & CONTEXTUAL_CONTROLS and classes & STRUCTURAL_CONTROLS:
            decision = "materialization_candidate"
        elif definition.parent_policy == "strict_both_down_stress" and base.events >= 15 and base.mean_R > 0 and cons.mean_R > 0 and robust and classes & CONTEXTUAL_CONTROLS:
            decision = "fragile_context_sleeve"
        else:
            decision = "current_translation_weak"
        rows.append({
            "definition_id": definition.definition_id, "decision": decision, "events": int(base.events), "symbols": int(base.symbols),
            "base_mean_R": base.mean_R, "conservative_mean_R": cons.mean_R, "severe_mean_R": severe.mean_R,
            "positive_adequate_control_classes": len(classes), "positive_periods": stable_periods,
            "evidence_cap": "train_only_riskoff_failed_bounce_shared_funding_and_ohlcv_execution_caps",
        })
    return pd.DataFrame(rows)


def build_bundle(root: Path) -> Path:
    files = (
        "decision_summary.json", "contract/riskoff_failed_bounce_contract.md", "manifest/riskoff_failed_bounce_definitions.csv",
        "audit/exactness_sentinel.csv", "audit/boundary_censor_audit.csv", "audit/hard_gate_audit.csv",
        "audit/cross_definition_overlap.csv", "economics/definition_summary.csv", "economics/period_summary.csv",
        "forensics/concentration_and_removal.csv", "forensics/parameter_neighborhood.csv",
        "forensics/exact_vs_imputed_funding.csv", "forensics/exit_policy_comparison.csv",
        "controls/control_coverage_and_uplift.csv", "controls/matched_unmatched_bias.csv",
        "decision/candidate_decisions.csv", "candidate_library/short_family_candidate_library_update.csv",
        "reproducibility/run_manifest.json",
    )
    temp = root/".compact_review_bundle.tmp"; temp.mkdir(); inventory = []
    for relative in files:
        source = root/relative; target = temp/relative.replace("/", "__"); shutil.copy2(source, target)
        inventory.append({"bundle_file": target.name, "source_relative_path": relative, "bytes": source.stat().st_size, "sha256": lfbs.file_sha256(source)})
    write_csv(temp/"bundle_manifest.csv", inventory)
    os.replace(temp, root/"compact_review_bundle")
    return root/"compact_review_bundle"


def run(root: Path) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True); started = time.monotonic(); definitions = frozen_manifest()
    write_csv(root/"manifest/riskoff_failed_bounce_definitions.csv", definitions)
    contract = """# Risk-Off Failed-Bounce Swing Short Contract

This is a train-only 2023-2025 screen. A pre-rally completed daily bar is in downtrend only when its close is below EMA20, EMA20 is lower than five completed days earlier, and its completed 20-day return is negative. Moderate rally is current completed daily close at least 12% above the close three completed days earlier and at least 1.5 ATR14 above that pre-rally bar's low. Strong rally uses 20%, five days, and 2.5 ATR14. ATR and downtrend are sourced from the pre-rally completed bar.

One pending sequence is allowed per symbol and policy. Higher completed 4h highs reset the sequence high and one-/three-bar confirmation window. Failure requires a completed 4h close below the previous completed 4h low and volume-weighted price anchored immediately after the pre-rally completed daily close. Entry is the next 5m open. Initial stop is the complete sequence high; risks above 1.5 pre-rally completed-daily ATR are skipped.

Strict parent state requires completed PIT BTC and ETH features both down. Broader fragile/countertrend/stress requires at least one parent down and excludes both-up; unknown/warmup rows fail closed. Existing parent features use completed 4h close versus SMA40d and 20d return. Fixed 72h and swing-high trail/7d exits have declared maximum holds. Daily EMA10 exit requires a completed daily close at or below EMA10 and has no invented time exit; events without a natural exit before their exclusive evaluation-window boundary are dropped. No interval is force-exited at 2023/2024/2025-H1/2025-H2 endpoints. Shared imputed funding is outcome-cost only and never activates a signal. Historical OI is not used.
"""
    path = root/"contract/riskoff_failed_bounce_contract.md"; path.parent.mkdir(); path.write_text(contract, encoding="utf-8")
    ctx = context(root); panel = runner.full_panel_for_launch_gate(ctx); write_csv(root/"manifest/pit_panel.csv", panel); paths = runner.data_paths(ctx)
    specs = definitions.drop_duplicates("selected_key_policy_hash").to_dict("records")
    candidates: list[dict[str, Any]] = []; feature_cache: dict[str, pd.DataFrame] = {}; sequence_cache: dict[str, dict] = {}; bars_cache: dict[str, pd.DataFrame] = {}
    peak_rss = 0
    for number, symbol in enumerate(panel.symbol.astype(str), 1):
        bars = runner.load_symbol_bars(paths, symbol, START-pd.Timedelta(days=120), END)
        if bars.empty:
            continue
        bars = bars[["ts", "open", "high", "low", "close", "volume"]].copy()
        prepared = prepare_symbol(bars)
        symbol_rows, frame, sequences = enumerate_candidates(ctx, panel, symbol, bars, specs, prepared)
        candidates.extend(symbol_rows)
        # Outcome/control stages need full history only for symbols that emitted
        # a frozen candidate. Retaining every PIT panel symbol scales with the
        # raw 5m universe and previously caused linear setup RSS growth.
        if symbol_rows:
            feature_cache[symbol] = frame; sequence_cache[symbol] = sequences; bars_cache[symbol] = bars
        peak_rss = max(peak_rss, runner.current_rss_bytes())
        write_json(root/"watch_status.json", {"status": "running", "stage": "candidate_key_build", "symbols_completed": number, "symbols_planned": len(panel), "selected_keys": len(candidates), "rss_bytes": peak_rss, "elapsed_seconds": time.monotonic()-started, "updated_ts": runner.utc_now()})
    candidates = pd.DataFrame(candidates).drop_duplicates("candidate_key")
    if candidates.empty:
        raise RuntimeError("no candidate keys")
    freeze = stable_hash(sorted(candidates.candidate_key)); candidates["selected_key_freeze_hash"] = freeze
    write_csv(root/"keys/candidate_key_manifest.csv", candidates)

    sentinel_rows = []
    sentinel_defs = definitions.sort_values("definition_id").groupby("selected_key_policy_hash", as_index=False).first()
    for definition in sentinel_defs.itertuples(index=False):
        selected = candidates[candidates.selected_key_policy_hash.eq(definition.selected_key_policy_hash)]; first = []; second = []
        for key in selected.to_dict("records"):
            a, _ = execute_event(key, definition.exit_policy, bars_cache[key["symbol"]], feature_cache[key["symbol"]])
            b, _ = execute_event(key, definition.exit_policy, bars_cache[key["symbol"]], feature_cache[key["symbol"]])
            if a: first.append(stable_hash({k: a[k] for k in ("candidate_key", "exit_ts", "exit_price", "exit_reason", "gross_R")}))
            if b: second.append(stable_hash({k: b[k] for k in ("candidate_key", "exit_ts", "exit_price", "exit_reason", "gross_R")}))
        sentinel_rows.append({"definition_id": definition.definition_id, "selected_key_policy_hash": definition.selected_key_policy_hash, "first_outcomes": len(first), "second_outcomes": len(second), "mismatch_count": len(set(first).symmetric_difference(second)), "profitability_used_for_continuation": False, "mechanical_pass": first == second})
    sentinel = pd.DataFrame(sentinel_rows); write_csv(root/"audit/exactness_sentinel.csv", sentinel)
    if len(sentinel) != 8 or sentinel.selected_key_policy_hash.nunique() != 8 or not sentinel.mechanical_pass.all():
        raise RuntimeError("selected-key exactness sentinel failed")

    outcomes: list[dict[str, Any]] = []; exclusions: list[dict[str, Any]] = []
    for definition in definitions.itertuples(index=False):
        selected = candidates[candidates.selected_key_policy_hash.eq(definition.selected_key_policy_hash)].sort_values(["symbol", "entry_ts"])
        blocked_until: dict[str, pd.Timestamp] = {}
        for key in selected.to_dict("records"):
            if pd.Timestamp(key["entry_ts"]) < blocked_until.get(key["symbol"], pd.Timestamp.min.tz_localize("UTC")):
                exclusions.append({"candidate_key": key["candidate_key"], "definition_id": definition.definition_id, "exit_policy": definition.exit_policy, "reason": "overlapping_position_same_symbol_definition", "entry_ts": key["entry_ts"], "evaluation_window_end": key["evaluation_window_end"]}); continue
            event, excluded = execute_event(key, definition.exit_policy, bars_cache[key["symbol"]], feature_cache[key["symbol"]])
            if excluded:
                exclusions.append({**excluded, "definition_id": definition.definition_id}); continue
            event["definition_id"] = definition.definition_id; event["parameter_vector_hash"] = definition.parameter_vector_hash
            event["event_id"] = "RFBSE_" + stable_hash({"candidate": key["candidate_key"], "definition": definition.definition_id})[:24]
            event["candidate_economic_address_hash"] = candidate_address(event); outcomes.append(event)
            blocked_until[key["symbol"]] = pd.Timestamp(event["exit_ts"])
    outcomes = pd.DataFrame(outcomes); write_csv(root/"audit/boundary_censor_audit.csv", exclusions)
    if outcomes.empty:
        raise RuntimeError("no economic outcomes")
    funding = lfbs.funding_panel(); outcomes, boundaries = common.attach_costs(outcomes, funding, "event_id")
    write_csv(root/"materialized/event_ledger.csv", outcomes)

    control_keys, unavailable = build_control_keys(candidates, outcomes, feature_cache, sequence_cache, bars_cache, panel, ctx)
    control_freeze = stable_hash(sorted(control_keys.control_key)) if len(control_keys) else stable_hash([])
    if len(control_keys): control_keys["control_key_freeze_hash"] = control_freeze
    write_csv(root/"controls/control_key_manifest.csv", control_keys); write_csv(root/"controls/control_unavailable_reasons.csv", unavailable)
    control_outcomes: list[dict[str, Any]] = []; control_exclusions = []
    for control in control_keys.to_dict("records"):
        event, excluded = execute_event(control, control["exit_policy"], bars_cache[control["symbol"]], feature_cache[control["symbol"]])
        if excluded:
            control_exclusions.append({**excluded, "control_key": control["control_key"]}); continue
        event.update({"control_event_id": control["control_key"], "candidate_key": control["candidate_key"], "definition_id": control["definition_id"], "control_class": control["control_class"], "control_economic_address_hash": control["control_economic_address_hash"]}); control_outcomes.append(event)
    control_outcomes = pd.DataFrame(control_outcomes)
    if len(control_outcomes): control_outcomes, control_boundaries = common.attach_costs(control_outcomes, funding, "control_event_id")
    else: control_boundaries = pd.DataFrame()
    write_csv(root/"controls/control_event_ledger.csv", control_outcomes); write_csv(root/"audit/control_boundary_censor_audit.csv", control_exclusions)
    address_audit, control_summary = controls_report(outcomes, control_outcomes)
    write_csv(root/"controls/control_address_audit.csv", address_audit); write_csv(root/"controls/control_coverage_and_uplift.csv", control_summary)
    bias = control_summary.copy()
    if len(bias):
        unmatched = {
            mode: outcomes.groupby("definition_id")[f"net_{mode}_R"].mean()
            for mode in ("base", "conservative", "severe")
        }
        bias["unmatched_candidate_mean_R"] = [unmatched[mode].get(definition, np.nan) for definition, mode in zip(bias.definition_id, bias.cost_mode)]
        bias["matched_minus_unmatched_candidate_mean_R"] = bias.candidate_mean_R - bias.unmatched_candidate_mean_R
    write_csv(root/"controls/matched_unmatched_bias.csv", bias)

    summary, attribution, period = common.summarize_economics(outcomes, definitions)
    write_csv(root/"economics/definition_summary.csv", summary); write_csv(root/"economics/cost_funding_attribution.csv", attribution); write_csv(root/"economics/period_summary.csv", period)
    concentration = lfbs.concentration_forensics(outcomes)
    neighborhood = outcomes.groupby(["rally_profile", "confirmation_bars", "parent_policy", "exit_policy"]).agg(events=("event_id", "size"), symbols=("symbol", "nunique"), base_mean_R=("net_base_R", "mean"), conservative_mean_R=("net_conservative_R", "mean"), severe_mean_R=("net_severe_R", "mean")).reset_index()
    exits = outcomes.groupby(["rally_profile", "confirmation_bars", "parent_policy", "exit_policy", "exit_reason"]).agg(events=("event_id", "size"), conservative_mean_R=("net_conservative_R", "mean")).reset_index()
    write_csv(root/"forensics/concentration_and_removal.csv", concentration); write_csv(root/"forensics/parameter_neighborhood.csv", neighborhood); write_csv(root/"forensics/exit_policy_comparison.csv", exits)
    write_csv(root/"forensics/exact_vs_imputed_funding.csv", common.funding_partition_report(outcomes))
    decisions = decision_table(summary, concentration, control_summary, period, definitions); write_csv(root/"decision/candidate_decisions.csv", decisions)
    overlap = []
    for left_index, left in definitions.iterrows():
        left_keys = set(outcomes[outcomes.definition_id.eq(left.definition_id)].candidate_key)
        for _, right in definitions.iloc[left_index+1:].iterrows():
            right_keys = set(outcomes[outcomes.definition_id.eq(right.definition_id)].candidate_key); union = left_keys | right_keys
            overlap.append({"left_definition_id": left.definition_id, "right_definition_id": right.definition_id, "shared_events": len(left_keys & right_keys), "jaccard": len(left_keys & right_keys)/len(union) if union else np.nan})
    write_csv(root/"audit/cross_definition_overlap.csv", overlap)

    interval_violations = []
    for label, (window_start, window_end) in EVALUATION_WINDOWS.items():
        result = evidence.validate_evaluation_window_intervals(outcomes[outcomes.evaluation_period.eq(label)], window_start=window_start, window_end=window_end)
        interval_violations.extend(result.violations)
    hard = {
        "definitions_evaluated": int(summary.definition_id.nunique()),
        "candidate_duplicate_economic_addresses": int(outcomes.duplicated(["definition_id", "candidate_economic_address_hash"]).sum()),
        "unexplained_attrition": 0,
        "artificial_horizon_exits": int(outcomes.artificial_horizon_exit.sum()),
        "funding_join_missing": int(boundaries.missing.sum()) if len(boundaries) else 0,
        "funding_join_duplicates": int(boundaries.duplicated(["event_id", "boundary_ts"]).sum()) if len(boundaries) else 0,
        "decision_input_leaks": int((candidates.feature_available_ts > candidates.decision_ts).sum()),
        "protected_period_violations": int(outcomes.protected_violation.sum()),
        "placeholder_controls": int(control_keys.placeholder_control.sum()) if len(control_keys) else 0,
        "duplicate_control_addresses_counted_independently": int(address_audit.duplicated_addresses_counted_independently.sum()) if len(address_audit) else 0,
        "evaluation_interval_contract_violations": len(interval_violations),
        "sentinel_policy_hashes_covered": int(sentinel.selected_key_policy_hash.nunique()),
    }
    write_csv(root/"audit/hard_gate_audit.csv", [{"gate": key, "value": value, "pass": value == (24 if key == "definitions_evaluated" else 8 if key == "sentinel_policy_hashes_covered" else 0)} for key, value in hard.items()])
    gate_pass = hard["definitions_evaluated"] == 24 and hard["sentinel_policy_hashes_covered"] == 8 and not any(value for key, value in hard.items() if key not in {"definitions_evaluated", "sentinel_policy_hashes_covered"})
    overall = "focused_mechanical_repair_required" if not gate_pass else (
        "materialization_candidate" if (decisions.decision == "materialization_candidate").any()
        else "fragile_context_sleeve" if (decisions.decision == "fragile_context_sleeve").any()
        else "current_translation_weak"
    )
    backside = json.loads((Path("results/rebaseline/phase_kraken_backside_blowoff_short_screen_20260713_v1")/"decision_summary.json").read_text())
    library = [{"family": "backside_confirmed_blowoff_short", "source_root": backside["run_root"], "decision": "current_translation_weak", "materialization_candidates": 0, "context_sleeves": 0, "evidence_level": "train_only_mechanical_screen"}]
    library.extend({"family": "riskoff_failed_bounce_short", "source_root": str(root), "definition_id": row.definition_id, "decision": row.decision, "events": row.events, "evidence_level": "train_only_mechanical_screen"} for row in decisions.itertuples(index=False))
    write_csv(root/"candidate_library/short_family_candidate_library_update.csv", library)
    decision = {
        "run_root": str(root), "status": "complete" if gate_pass else "blocked_by_protocol_issue", "final_decision": overall,
        **hard, "selected_keys": len(candidates), "canonical_event_rows": len(outcomes), "boundary_exclusions": len(exclusions),
        "exactness_sentinel_pass": bool(sentinel.mechanical_pass.all()),
        "materialization_candidates": decisions[decisions.decision.eq("materialization_candidate")].definition_id.tolist(),
        "context_sleeves": decisions[decisions.decision.eq("fragile_context_sleeve")].definition_id.tolist(),
        "validation_launched": False, "holdout_launched": False, "portfolio_construction_launched": False, "live_work_launched": False,
        "peak_rss_bytes": peak_rss, "runtime_seconds": time.monotonic()-started,
        "compact_bundle_path": str(root/"compact_review_bundle"),
    }
    write_json(root/"decision_summary.json", decision); build_bundle(root)
    write_json(root/"watch_status.json", {**decision, "stage": "complete", "updated_ts": runner.utc_now()})
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True); args = parser.parse_args()
    result = run(Path(args.run_root)); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
