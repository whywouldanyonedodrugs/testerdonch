#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from tools.qlmg_screening_core import ReplayConfig, replay_trade
from tools.run_qlmg_engine_and_first_screen import add_features, cost_bps_for_tier, funding_events_from_df

FINAL_HOLDOUT_START = pd.Timestamp("2026-01-01T00:00:00Z")
SCREENING_END = pd.Timestamp("2025-12-31T23:59:59Z")
HORIZON_MINUTES = {
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "24h": 1440,
    "48h": 2880,
    "72h": 4320,
}


def stable_hash(obj: Any, n: int = 12) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:n]


def validate_no_protected(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        vals = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
        if not vals.empty and bool((vals >= FINAL_HOLDOUT_START).any()):
            raise RuntimeError(f"protected timestamp detected in {col}")


def _rolling_prior_high(s: pd.Series, bars: int) -> pd.Series:
    return s.rolling(bars, min_periods=max(24, min(bars, 288))).max().shift(1)


def _rolling_prior_low(s: pd.Series, bars: int) -> pd.Series:
    return s.rolling(bars, min_periods=max(24, min(bars, 288))).min().shift(1)


def add_short_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_features(df).copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "turnover", "volume", "open_interest", "funding_rate"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for hours in (12, 24, 48, 72, 120, 168):
        bars = hours * 12
        out[f"ret_{hours}h"] = out["close"] / out["close"].shift(bars) - 1.0
        out[f"prior_high_{hours}h"] = _rolling_prior_high(out["high"], bars)
        out[f"prior_low_{hours}h"] = _rolling_prior_low(out["low"], bars)
    out["ema50"] = out["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    atr = pd.to_numeric(out.get("atr_proxy"), errors="coerce").replace(0, np.nan)
    out["dist_ema10_atr"] = (out["close"] - out["ema10"]) / atr
    out["dist_ema20_atr"] = (out["close"] - out["ema20"]) / atr
    out["dist_ema50_atr"] = (out["close"] - out["ema50"]) / atr
    out["atr_bps"] = atr / out["close"].replace(0, np.nan) * 10000.0
    out["range_bps"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan) * 10000.0
    out["turnover_pct_30d"] = out["turnover"].rolling(288 * 30, min_periods=288).rank(pct=True)
    out["range_pct_30d"] = out["range_bps"].rolling(288 * 30, min_periods=288).rank(pct=True)
    if "open_interest" in out.columns:
        oi = out["open_interest"]
        out["oi_chg_72h"] = oi / oi.shift(864) - 1.0
        out["oi_chg_4h"] = oi / oi.shift(48) - 1.0
    else:
        out["oi_chg_72h"] = np.nan
        out["oi_chg_4h"] = np.nan
    if "funding_rate" in out.columns:
        f = out["funding_rate"]
        out["funding_pct_30d"] = f.rolling(288 * 30, min_periods=288).rank(pct=True)
    else:
        out["funding_pct_30d"] = np.nan
    price_leg = pd.Series(np.where(out["ret_24h"] >= 0, "price_up", "price_down"), index=out.index)
    oi_leg = pd.Series(np.where(out["oi_chg_24h"] >= 0, "oi_up", "oi_down"), index=out.index)
    out["price_oi_matrix_24h"] = price_leg.str.cat(oi_leg, sep="_")
    out.loc[out["ret_24h"].isna() | out["oi_chg_24h"].isna(), "price_oi_matrix_24h"] = "unknown"
    if "mark_close" in out.columns:
        out["mark_gap_bps"] = (pd.to_numeric(out["mark_close"], errors="coerce") / out["close"].replace(0, np.nan) - 1.0) * 10000.0
    else:
        out["mark_gap_bps"] = np.nan
    return out


def f1_variants(max_variants: int = 90) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for lookback_h in (24, 72, 120):
        for extension in (0.20, 0.35, 0.55):
            for dist_atr in (2.5, 4.0, 6.0):
                for trigger in ("lower_low", "ema10_fail", "vwap_fail"):
                    for crowding in ("any", "funding_high_or_oi_up"):
                        v = {
                            "family": "F1",
                            "side": "short",
                            "lookback_h": lookback_h,
                            "extension_return": extension,
                            "dist_ema20_atr_min": dist_atr,
                            "trigger": trigger,
                            "crowding_gate": crowding,
                            "max_hold_h": 4.0,
                        }
                        v["variant_id"] = f"F1_lb{lookback_h}_ext{int(extension*100)}_d{str(dist_atr).replace('.', 'p')}_{trigger}_{crowding}"
                        v["parameter_hash"] = stable_hash(v)
                        variants.append(v)
    return variants[:max_variants]


def g1_parent_variants() -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for base_h in (24, 72, 168):
        for ret_h in (24, 72):
            for prior_ret in (0.08, 0.15, 0.30):
                v = {"family": "A1_parent_for_G1", "side": "long", "base_h": base_h, "ret_h": ret_h, "prior_ret_min": prior_ret, "breakout_buffer_bps": 20}
                v["variant_id"] = f"A1p_base{base_h}_ret{ret_h}_{int(prior_ret*100)}"
                v["parameter_hash"] = stable_hash(v)
                variants.append(v)
    return variants


def g1_variants(max_variants: int = 90) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for parent in g1_parent_variants():
        for fail_h in (4, 12, 24, 48):
            for trigger in ("close_back_inside", "ema10_fail", "retest_failure"):
                for buffer_bps in (0, 25):
                    v = {
                        "family": "G1",
                        "side": "short",
                        "parent_variant_id": parent["variant_id"],
                        "fail_window_h": fail_h,
                        "trigger": trigger,
                        "failure_buffer_bps": buffer_bps,
                        "max_hold_h": 4.0,
                    }
                    v["variant_id"] = f"G1_{parent['variant_id']}_fw{fail_h}_{trigger}_buf{buffer_bps}"
                    v["parameter_hash"] = stable_hash(v)
                    variants.append(v)
    return variants[:max_variants]


def _selected_indices(mask: pd.Series, min_gap_bars: int) -> list[int]:
    idx = np.flatnonzero(mask.fillna(False).to_numpy())
    out: list[int] = []
    last = -10**12
    for i in idx:
        if int(i) - last >= min_gap_bars:
            out.append(int(i))
            last = int(i)
    return out


def _tier_at(tiers: pd.DataFrame, symbol: str, date: str) -> str:
    if tiers is None or tiers.empty:
        return "UNKNOWN"
    sub = tiers[(tiers["symbol"] == symbol) & (tiers["date"] <= date)]
    if sub.empty:
        return "UNKNOWN"
    return str(sub.iloc[-1].get("liquidity_tier", "UNKNOWN"))


def _base_event(symbol: str, row: pd.Series, entry_row: pd.Series, tier: str, variant: Mapping[str, Any], *, family: str, stop_price: float, trigger_fields: Mapping[str, Any]) -> dict[str, Any] | None:
    entry = float(entry_row.get("open", np.nan))
    atr = float(row.get("atr_proxy", np.nan))
    decision_ts = pd.Timestamp(row["timestamp"])
    entry_ts = pd.Timestamp(entry_row["timestamp"])
    if decision_ts >= FINAL_HOLDOUT_START or entry_ts >= FINAL_HOLDOUT_START:
        raise RuntimeError("protected timestamp generated")
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr) or atr <= 0:
        return None
    risk_bps = abs(stop_price - entry) / entry * 10000.0
    if not np.isfinite(risk_bps) or risk_bps <= 0:
        return None
    event = {
        "family": family,
        "variant_id": variant["variant_id"],
        "parameter_hash": variant["parameter_hash"],
        "symbol": symbol,
        "side": "short",
        "liquidity_tier": tier,
        "decision_ts": decision_ts,
        "entry_ts": entry_ts,
        "entry_ref_price": entry,
        "reference_stop_price": float(stop_price),
        "reference_risk_bps": float(risk_bps),
        "atr_proxy": atr,
        "atr_bps": float(atr / entry * 10000.0),
        "ret_24h": row.get("ret_24h", np.nan),
        "ret_72h": row.get("ret_72h", np.nan),
        "oi_chg_24h": row.get("oi_chg_24h", np.nan),
        "oi_chg_72h": row.get("oi_chg_72h", np.nan),
        "funding_rate": row.get("funding_rate", np.nan),
        "funding_pct_30d": row.get("funding_pct_30d", np.nan),
        "turnover": row.get("turnover", np.nan),
        "turnover_med_24h": row.get("turnover_med_24h", np.nan),
        "range_pct": row.get("range_pct", np.nan),
        "range_bps": row.get("range_bps", np.nan),
        "price_oi_matrix_24h": row.get("price_oi_matrix_24h", "unknown"),
        "mark_gap_bps": row.get("mark_gap_bps", np.nan),
        "mark_path_status": "mark_available" if pd.notna(row.get("mark_gap_bps", np.nan)) else "mark_missing_last_proxy_only",
        "data_quality_flags": "mark_path_proxy" if pd.isna(row.get("mark_gap_bps", np.nan)) else "",
    }
    event.update(trigger_fields)
    event["event_id"] = stable_hash({"family": family, "variant_id": variant["variant_id"], "symbol": symbol, "decision_ts": str(decision_ts)}, 16)
    return event


def generate_f1_events(symbol: str, raw_df: pd.DataFrame, tiers: pd.DataFrame, variants: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_short_features(raw_df)
    rows: list[dict[str, Any]] = []
    cov: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    tier = _tier_at(tiers, symbol, str(pd.Timestamp(df["timestamp"].max()).date()))
    if tier not in {"A", "B", "C"}:
        for v in variants:
            cov.append({"family": "F1", "variant_id": v["variant_id"], "symbol": symbol, "liquidity_tier": tier, "raw_triggers": 0, "retained_events": 0, "skip_reason": "tier_not_shortable_in_phase1"})
        return pd.DataFrame(), pd.DataFrame(cov)
    for v in variants:
        look = int(v["lookback_h"])
        ret_col = f"ret_{look}h"
        prior_high_col = f"prior_high_{look}h"
        ext_prev = (df[ret_col].shift(1) >= float(v["extension_return"])) & (df["dist_ema20_atr"].shift(1) >= float(v["dist_ema20_atr_min"])) & (df["high"].shift(1) >= df[prior_high_col] * 0.995)
        if v["crowding_gate"] == "funding_high_or_oi_up":
            crowd = (df["funding_pct_30d"].shift(1) >= 0.75) | (df["oi_chg_24h"].shift(1) > 0)
        else:
            crowd = pd.Series(True, index=df.index)
        if v["trigger"] == "lower_low":
            trigger = df["close"] < df["low"].shift(1)
        elif v["trigger"] == "ema10_fail":
            trigger = (df["close"] < df["ema10"]) & (df["close"].shift(1) >= df["ema10"].shift(1))
        else:
            trigger = (df["close"] < df["vwap_proxy_24h"]) & (df["close"].shift(1) >= df["vwap_proxy_24h"].shift(1))
        raw_mask = ext_prev & crowd & trigger
        idxs = _selected_indices(raw_mask, min_gap_bars=48)
        retained = 0
        for i in idxs:
            if i + 1 >= len(df):
                continue
            row = df.iloc[i]
            entry_row = df.iloc[i + 1]
            local_stop = max(float(row.get("high", np.nan)), float(df["high"].iloc[max(0, i - 12): i + 1].max()))
            if not np.isfinite(local_stop):
                continue
            event = _base_event(symbol, row, entry_row, tier, v, family="F1", stop_price=local_stop, trigger_fields={
                "trigger_type": v["trigger"],
                "extension_return_threshold": v["extension_return"],
                "extension_lookback_h": look,
                "crowding_gate": v["crowding_gate"],
                "chart_only_extension_flag": bool(v["crowding_gate"] == "any"),
                "backside_confirmation": True,
            })
            if event is not None:
                rows.append(event)
                retained += 1
        cov.append({"family": "F1", "variant_id": v["variant_id"], "symbol": symbol, "liquidity_tier": tier, "raw_triggers": int(raw_mask.fillna(False).sum()), "retained_events": retained, "skip_reason": ""})
    events = pd.DataFrame(rows)
    if not events.empty:
        validate_no_protected(events, ["decision_ts", "entry_ts"])
    return events, pd.DataFrame(cov)


def generate_a1_breakout_parents(symbol: str, raw_df: pd.DataFrame, tiers: pd.DataFrame, variants: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_short_features(raw_df)
    rows: list[dict[str, Any]] = []
    cov: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    tier = _tier_at(tiers, symbol, str(pd.Timestamp(df["timestamp"].max()).date()))
    if tier not in {"A", "B", "C"}:
        for v in variants:
            cov.append({"family": "A1_parent_for_G1", "variant_id": v["variant_id"], "symbol": symbol, "liquidity_tier": tier, "raw_triggers": 0, "retained_events": 0, "skip_reason": "tier_not_allowed"})
        return pd.DataFrame(), pd.DataFrame(cov)
    for v in variants:
        base_bars = int(v["base_h"]) * 12
        ret_col = f"ret_{int(v['ret_h'])}h"
        level = df["high"].rolling(base_bars, min_periods=max(24, min(base_bars, 288))).max().shift(1)
        breakout = (df["close"] > level * (1.0 + float(v["breakout_buffer_bps"]) / 10000.0)) & (df["close"].shift(1) <= level.shift(1) * (1.0 + float(v["breakout_buffer_bps"]) / 10000.0)) & (df[ret_col].shift(1) >= float(v["prior_ret_min"]))
        idxs = _selected_indices(breakout, min_gap_bars=288)
        retained = 0
        for i in idxs:
            row = df.iloc[i]
            if i + 1 >= len(df):
                continue
            parent = {
                "parent_id": stable_hash({"parent": v["variant_id"], "symbol": symbol, "decision_ts": str(row["timestamp"])}, 16),
                "parent_variant_id": v["variant_id"],
                "symbol": symbol,
                "liquidity_tier": tier,
                "parent_decision_ts": pd.Timestamp(row["timestamp"]),
                "parent_entry_ts": pd.Timestamp(df.iloc[i + 1]["timestamp"]),
                "breakout_level": float(level.iloc[i]),
                "breakout_high": float(row["high"]),
                "parent_ret": float(row.get(ret_col, np.nan)),
                "base_h": v["base_h"],
                "parent_index": int(i),
            }
            if parent["parent_decision_ts"] >= FINAL_HOLDOUT_START or parent["parent_entry_ts"] >= FINAL_HOLDOUT_START:
                raise RuntimeError("protected timestamp generated in parent ledger")
            rows.append(parent)
            retained += 1
        cov.append({"family": "A1_parent_for_G1", "variant_id": v["variant_id"], "symbol": symbol, "liquidity_tier": tier, "raw_triggers": int(breakout.fillna(False).sum()), "retained_events": retained, "skip_reason": ""})
    parents = pd.DataFrame(rows)
    if not parents.empty:
        validate_no_protected(parents, ["parent_decision_ts", "parent_entry_ts"])
    return parents, pd.DataFrame(cov)


def generate_g1_events(symbol: str, raw_df: pd.DataFrame, tiers: pd.DataFrame, parent_events: pd.DataFrame, variants: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_short_features(raw_df)
    rows: list[dict[str, Any]] = []
    cov: list[dict[str, Any]] = []
    if df.empty or parent_events.empty:
        for v in variants:
            cov.append({"family": "G1", "variant_id": v["variant_id"], "symbol": symbol, "liquidity_tier": "UNKNOWN", "raw_triggers": 0, "retained_events": 0, "skip_reason": "no_parent_breakouts"})
        return pd.DataFrame(), pd.DataFrame(cov)
    tier = _tier_at(tiers, symbol, str(pd.Timestamp(df["timestamp"].max()).date()))
    ts_to_idx = {pd.Timestamp(ts): i for i, ts in enumerate(df["timestamp"])}
    for v in variants:
        parents = parent_events[parent_events["parent_variant_id"] == v["parent_variant_id"]]
        raw = 0
        retained = 0
        for _, parent in parents.iterrows():
            parent_ts = pd.Timestamp(parent["parent_decision_ts"])
            if parent_ts not in ts_to_idx:
                continue
            start_i = ts_to_idx[parent_ts] + 1
            end_i = min(len(df) - 2, start_i + int(float(v["fail_window_h"]) * 12))
            if start_i >= end_i:
                continue
            level = float(parent["breakout_level"])
            sub = df.iloc[start_i : end_i + 1]
            buffer = float(v["failure_buffer_bps"]) / 10000.0
            if v["trigger"] == "close_back_inside":
                mask = sub["close"] < level * (1.0 - buffer)
            elif v["trigger"] == "ema10_fail":
                mask = (sub["close"] < sub["ema10"]) & (sub["close"].shift(1) >= sub["ema10"].shift(1))
            else:
                mask = (sub["high"] >= level * 0.995) & (sub["close"] < level * (1.0 - buffer))
            raw += int(mask.fillna(False).sum())
            hit_idx = np.flatnonzero(mask.fillna(False).to_numpy())
            if len(hit_idx) == 0:
                continue
            i = int(sub.index[hit_idx[0]])
            if i + 1 >= len(df):
                continue
            row = df.iloc[i]
            entry_row = df.iloc[i + 1]
            stop = max(float(row.get("high", np.nan)), float(parent.get("breakout_high", np.nan)), level * 1.005)
            event = _base_event(symbol, row, entry_row, tier, v, family="G1", stop_price=stop, trigger_fields={
                "trigger_type": v["trigger"],
                "parent_id": parent["parent_id"],
                "parent_variant_id": parent["parent_variant_id"],
                "parent_decision_ts": parent_ts,
                "breakout_level": level,
                "failure_buffer_bps": v["failure_buffer_bps"],
                "backside_confirmation": True,
            })
            if event is not None:
                rows.append(event)
                retained += 1
        cov.append({"family": "G1", "variant_id": v["variant_id"], "symbol": symbol, "liquidity_tier": tier, "raw_triggers": raw, "retained_events": retained, "skip_reason": ""})
    events = pd.DataFrame(rows)
    if not events.empty:
        validate_no_protected(events, ["decision_ts", "entry_ts", "parent_decision_ts"])
    return events, pd.DataFrame(cov)


def compute_short_path_row(event: Mapping[str, Any], df_indexed: pd.DataFrame) -> dict[str, Any]:
    entry_ts = pd.Timestamp(event["entry_ts"])
    entry = float(event["entry_ref_price"])
    risk_bps = float(event.get("reference_risk_bps", np.nan))
    if not np.isfinite(risk_bps) or risk_bps <= 0:
        risk_bps = 100.0
    out = {k: event.get(k) for k in ["event_id", "family", "variant_id", "symbol", "side", "liquidity_tier", "decision_ts", "entry_ts", "entry_ref_price", "reference_risk_bps", "atr_bps", "mark_path_status", "data_quality_flags"]}
    bars_all = df_indexed[df_indexed.index > entry_ts]
    for label, minutes in HORIZON_MINUTES.items():
        bars = bars_all[bars_all.index <= entry_ts + pd.Timedelta(minutes=minutes)]
        if bars.empty:
            out[f"{label}_path_available"] = False
            out[f"{label}_mfe_bps"] = np.nan
            out[f"{label}_mae_bps"] = np.nan
            out[f"{label}_close_return_bps"] = np.nan
            out[f"{label}_pos1R_before_neg1R"] = np.nan
            out[f"{label}_liquidation_10x"] = np.nan
            continue
        lows = pd.to_numeric(bars["low"], errors="coerce")
        highs = pd.to_numeric(bars["high"], errors="coerce")
        closes = pd.to_numeric(bars["close"], errors="coerce")
        mfe = (entry - float(lows.min())) / entry * 10000.0
        mae = (float(highs.max()) - entry) / entry * 10000.0
        close_ret = (entry - float(closes.iloc[-1])) / entry * 10000.0
        tp_ts = None
        sl_ts = None
        tp_level = entry * (1.0 - risk_bps / 10000.0)
        sl_level = entry * (1.0 + risk_bps / 10000.0)
        tp_hits = bars.index[lows <= tp_level]
        sl_hits = bars.index[highs >= sl_level]
        if len(tp_hits):
            tp_ts = tp_hits[0]
        if len(sl_hits):
            sl_ts = sl_hits[0]
        pos_before = np.nan
        if tp_ts is not None or sl_ts is not None:
            pos_before = bool(tp_ts is not None and (sl_ts is None or tp_ts < sl_ts))
            if tp_ts is not None and sl_ts is not None and tp_ts == sl_ts:
                pos_before = False
        mark_high = bars["mark_high"] if "mark_high" in bars.columns else highs
        liq_level_10x = entry * (1.0 + 0.095)
        out[f"{label}_path_available"] = True
        out[f"{label}_mfe_bps"] = float(mfe)
        out[f"{label}_mae_bps"] = float(mae)
        out[f"{label}_close_return_bps"] = float(close_ret)
        out[f"{label}_pos1R_before_neg1R"] = pos_before
        out[f"{label}_liquidation_10x"] = bool(pd.to_numeric(mark_high, errors="coerce").max() >= liq_level_10x)
    return out


def replay_short_event(event: Mapping[str, Any], df_indexed: pd.DataFrame, *, stop_atr_mult: float, target_r: float, hold_h: float, tie_breaker: str = "sl_wins") -> dict[str, Any] | None:
    entry_ts = pd.Timestamp(event["entry_ts"])
    entry = float(event["entry_ref_price"])
    atr = float(event.get("atr_proxy", np.nan))
    if not np.isfinite(atr) or atr <= 0:
        atr = entry * float(event.get("atr_bps", 100.0)) / 10000.0
    stop = entry + stop_atr_mult * atr
    risk = stop - entry
    if risk <= 0:
        return None
    target = entry - target_r * risk
    fee_bps, slip_bps = cost_bps_for_tier(str(event.get("liquidity_tier", "UNKNOWN")))
    try:
        res = replay_trade(
            df_indexed[df_indexed.index >= entry_ts],
            ReplayConfig(
                side="short",
                decision_ts=pd.Timestamp(event["decision_ts"]),
                entry_ts=entry_ts,
                entry_price=entry,
                stop_price=stop,
                target_price=target,
                max_holding_hours=hold_h,
                qty=1.0,
                fee_bps_round_trip=fee_bps,
                slippage_bps_round_trip=slip_bps,
                leverage=10.0,
                tie_breaker=tie_breaker,
            ),
            funding_events_from_df(df_indexed.reset_index(drop=True)),
        )
    except Exception:
        return None
    row = res.as_dict()
    row.update({
        "event_id": event.get("event_id"),
        "family": event.get("family"),
        "variant_id": event.get("variant_id"),
        "symbol": event.get("symbol"),
        "liquidity_tier": event.get("liquidity_tier"),
        "decision_ts": str(event.get("decision_ts")),
        "surface_id": f"stopATR{stop_atr_mult:g}_target{target_r:g}_hold{hold_h:g}_{tie_breaker}",
        "stop_atr_mult": stop_atr_mult,
        "target_r": target_r,
        "hold_h": hold_h,
    })
    return row


def revised_short_score(summary: Mapping[str, Any]) -> float:
    try:
        net_r = float(summary.get("net_R", 0.0))
        pf = float(summary.get("PF", 0.0))
        trades = float(summary.get("trades", summary.get("events", 0.0)))
        liq = float(summary.get("liquidation_count", 0.0))
        max_dd = abs(float(summary.get("max_dd_R", summary.get("max_dd_R_proxy", 0.0))))
        conc = max(float(summary.get("max_symbol_positive_share", 0.0) or 0.0), float(summary.get("max_month_positive_share", 0.0) or 0.0))
    except Exception:
        return -1e9
    if net_r <= 0 or pf <= 1.0 or liq > 0:
        return -1000000.0 + net_r - 1000.0 * liq
    score = net_r + 10.0 * np.log1p(max(trades, 0.0)) + 25.0 * (pf - 1.0) - 0.25 * max_dd - 50.0 * conc
    return float(score)
