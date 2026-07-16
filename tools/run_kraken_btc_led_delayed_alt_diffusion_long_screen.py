#!/usr/bin/env python3
"""Train-only btc-led delayed alt diffusion screen v1."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from tools import qlmg_evidence_contracts as evidence
from tools import qlmg_signal_state_contract as signal_state
from tools import run_kraken_backside_blowoff_short_screen as reports
from tools import run_kraken_delayed_flush_reclaim_long_screen as execution
from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools import run_kraken_liquid_failed_breakout_short_screen as lfbs
from tools import run_kraken_riskoff_failed_bounce_short_screen as parent


RUN_ROOT = Path("results/rebaseline/phase_kraken_btc_led_delayed_alt_diffusion_long_screen_20260716_v1")
REFERENCE_ROOT = Path("results/rebaseline/phase_kraken_relative_strength_breakout_vs_btc_screen_20260716_v1")
VALID_ARCHITECTURE_ROOT = Path("results/rebaseline/phase_kraken_delayed_flush_reclaim_signal_state_repair_20260715_v1")
CAMPAIGN_ROOT = Path("results/rebaseline/phase_kraken_cross_family_repair_campaign_closure_20260715_v1")
START = pd.Timestamp("2023-01-01", tz="UTC")
PROTECTED = pd.Timestamp("2026-01-01", tz="UTC")
END = PROTECTED - pd.Timedelta(minutes=5)
CONTRACT_VERSION = "kraken_btc_led_delayed_alt_diffusion_v1_20260716"
SIGNAL_STATE_CONTRACT_VERSION = signal_state.SIGNAL_STATE_CONTRACT_VERSION
RISK_MATCH_TOLERANCE_ATR = 0.25
CONTROL_CLASSES = (
    "btc_impulse_top_half_residual",
    "lag_without_btc_impulse",
    "alt_own_impulse_without_btc_lead",
    "unconditional_alt_after_btc_impulse",
    "same_symbol_same_parent_random_long",
)
CONTEXTUAL_CONTROLS = {"same_symbol_same_parent_random_long"}
STRUCTURAL_CONTROLS = set(CONTROL_CLASSES) - CONTEXTUAL_CONTROLS


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_hash(root: Path) -> str:
    return stable_hash([(str(path.relative_to(root)), file_hash(path)) for path in sorted(root.rglob("*")) if path.is_file()])


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def policy_hash(row: Mapping[str, Any], *, include_exit: bool) -> str:
    fields = ["lag_profile", "parent_policy"]
    if include_exit:
        fields.append("exit_policy")
    vector = {field: row[field] for field in fields}
    vector.update({
        "side": "long", "signal_timeframe": "completed_aligned_2h",
        "btc_impulse": "ret_2h_gte_prior60_mean_plus_2std_and_eth_ret_nonnegative",
        "beta": "prior60_aligned_returns_min30_clip_0_3",
        "residual": "alt_return_minus_clipped_beta_times_btc_return",
        "execution": "next_executable_5m_open", "stop": "completed_alt_2h_impulse_bar_low",
        "risk_band_daily_atr": [0.25, 1.5],
        "parent_projection": "pit_both_up_or_all_regime", "protected_boundary": PROTECTED.isoformat(),
        "signal_state_contract_version": SIGNAL_STATE_CONTRACT_VERSION, "strategy_contract_version": CONTRACT_VERSION,
    })
    return stable_hash(vector)


def frozen_manifest() -> pd.DataFrame:
    rows = []
    for lag_profile in ("moderate_lag", "deep_lag"):
        for parent_policy in ("both_up", "all_regime_comparator"):
            for exit_policy in ("fixed_4h", "fixed_8h", "fixed_12h"):
                row = {"definition_id": f"h43_v1_{len(rows)+1:03d}", "lag_profile": lag_profile,
                       "parent_policy": parent_policy, "exit_policy": exit_policy}
                row["selected_key_policy_hash"] = policy_hash(row, include_exit=False)
                row["parameter_vector_hash"] = policy_hash(row, include_exit=True)
                rows.append(row)
    return pd.DataFrame(rows)


def raw_policy_hash(lag_profile: str) -> str:
    return stable_hash({
        "lag_profile": lag_profile, "signal_semantics": "btc_impulse_then_cross_sectional_beta_residual_lag",
        "parent_neutral": True, "protected_boundary": PROTECTED.isoformat(),
        "strategy_contract_version": CONTRACT_VERSION, "signal_state_contract_version": SIGNAL_STATE_CONTRACT_VERSION,
    })


def instrument_universe_audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Freeze structural product exclusions from the official instrument snapshot."""
    paths = sorted(Path("/opt/parquet/kraken_derivatives/parquet/instruments").glob("*.parquet"))
    if len(paths) != 1:
        raise RuntimeError(f"one authoritative instrument snapshot required, found {len(paths)}")
    instruments = pd.read_parquet(paths[0])
    instruments = instruments[instruments.symbol.astype(str).str.startswith("PF_")].copy()
    fields = instruments[["symbol", "base", "quote", "category", "tradfi", "type", "openingDate"]].drop_duplicates("symbol")
    out = panel[["symbol"]].merge(fields, on="symbol", how="left", validate="one_to_one")
    excluded_categories = {"Stablecoin", "Forex", "xStocks", "Pre-IPO", "Commodities", "DTF"}
    out["structural_exclusion_reason"] = np.select(
        [out.symbol.eq("PF_XBTUSD"), out.base.astype(str).str.upper().isin({"BTC", "XBT"}), out.tradfi.fillna(False).astype(bool), out.category.astype(str).isin(excluded_categories), out.base.isna()],
        ["btc_reference_not_alt", "btc_base_not_alt", "tradfi_instrument", "non_crypto_or_stable_synthetic_category", "instrument_metadata_missing"],
        default="",
    )
    out["rankable_alt_perpetual"] = out.structural_exclusion_reason.eq("")
    out["instrument_snapshot_path"] = str(paths[0])
    out["instrument_snapshot_hash"] = file_hash(paths[0])
    return out


def completed_two_hour(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, work, daily = execution.feature_frames(bars)
    two = work.set_index("known_ts").resample("2h", label="right", closed="right").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), execution_bar_count=("close", "count"),
    )
    two = two[two.execution_bar_count.eq(24) & two.close.notna()].reset_index().rename(columns={"known_ts": "decision_ts"})
    two["return_2h"] = two.close.pct_change(fill_method=None).where(two.decision_ts.diff().eq(pd.Timedelta(hours=2)))
    daily_fields = daily[["daily_source_ts", "atr_14d", "ema_10", "close"]].rename(columns={"close": "daily_close"})
    two = pd.merge_asof(two.sort_values("decision_ts"), daily_fields.sort_values("daily_source_ts"),
                        left_on="decision_ts", right_on="daily_source_ts", direction="backward", allow_exact_matches=True)
    return two, work, daily


def btc_eth_reference_frames(btc_bars: pd.DataFrame, eth_bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    btc, _, _ = completed_two_hour(btc_bars)
    eth, _, _ = completed_two_hour(eth_bars)
    btc = btc[["decision_ts", "close", "return_2h"]].rename(columns={"close": "btc_close", "return_2h": "btc_return_2h", "decision_ts": "btc_source_ts"})
    btc["btc_prior60_return_mean"] = btc.btc_return_2h.rolling(60, min_periods=60).mean().shift(1)
    btc["btc_prior60_return_std"] = btc.btc_return_2h.rolling(60, min_periods=60).std().shift(1)
    btc["btc_impulse"] = btc.btc_return_2h.ge(btc.btc_prior60_return_mean + 2.0 * btc.btc_prior60_return_std)
    eth = eth[["decision_ts", "return_2h"]].rename(columns={"decision_ts": "eth_source_ts", "return_2h": "eth_return_2h"})
    refs = btc.merge(eth, left_on="btc_source_ts", right_on="eth_source_ts", how="inner", validate="one_to_one")
    refs["qualified_btc_impulse"] = refs.btc_impulse & refs.eth_return_2h.ge(0)
    return refs.sort_values("btc_source_ts"), btc, eth


def feature_frames(bars: pd.DataFrame, references: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alt, work, daily = completed_two_hour(bars)
    frame = alt.merge(references, left_on="decision_ts", right_on="btc_source_ts", how="inner", validate="one_to_one")
    valid_pair = frame.return_2h.notna() & frame.btc_return_2h.notna()
    pair_count = valid_pair.astype(int).rolling(60, min_periods=1).sum().shift(1)
    covariance = frame.return_2h.rolling(60, min_periods=30).cov(frame.btc_return_2h).shift(1)
    variance = frame.btc_return_2h.rolling(60, min_periods=30).var().shift(1)
    frame["beta_observations"] = pair_count
    frame["beta_raw"] = covariance / variance.replace(0, np.nan)
    frame["beta_clipped"] = frame.beta_raw.clip(0, 3)
    frame["residual_return"] = frame.return_2h - frame.beta_clipped * frame.btc_return_2h
    frame["trailing_residual_std"] = frame.residual_return.rolling(60, min_periods=30).std().shift(1)
    frame["alt_own_impulse"] = frame.return_2h.ge(frame.return_2h.rolling(60, min_periods=60).mean().shift(1) + 2 * frame.return_2h.rolling(60, min_periods=60).std().shift(1))
    entries = bars[["ts", "open"]].sort_values("ts").rename(columns={"ts":"entry_ts", "open":"next_5m_open"})
    frame = pd.merge_asof(frame.sort_values("decision_ts"), entries.sort_values("entry_ts"),
                          left_on="decision_ts", right_on="entry_ts", direction="forward", allow_exact_matches=True)
    frame = parent.attach_parent_state(frame)
    frame["feature_available_ts"] = frame[["decision_ts", "btc_source_ts", "eth_source_ts", "daily_source_ts", "parent_source_ts"]].max(axis=1)
    frame.attrs["alt_completed_2h_rows"] = len(alt)
    return frame, work, daily


def pit_universe_symbols(ctx: Any, panel: pd.DataFrame, decision_ts: pd.Timestamp, cache: dict[pd.Timestamp, set[str]]) -> set[str]:
    timestamp = pd.Timestamp(decision_ts)
    if timestamp not in cache:
        ranked = runner.pit_liquidity_ranking_by_checkpoint(ctx, panel, timestamp)
        eligible = ranked[ranked.eligible_at_checkpoint.astype(bool)].copy()
        eligible["tie_hash"] = [stable_hash({"symbol": value, "decision_ts": timestamp}) for value in eligible.symbol]
        selected = eligible.sort_values(["pit_liquidity_proxy_score", "tie_hash"], ascending=[False, True]).head(runner.TSMOM_TIER_AB_UNIVERSE_LIMIT)
        cache[timestamp] = set(selected.symbol.astype(str))
    return cache[timestamp]


def lag_condition(row: Mapping[str, Any], profile: str) -> bool:
    if profile == "moderate_lag":
        return float(row["residual_return"]) <= 0 and float(row["residual_rank_pct"]) <= 0.50
    return float(row["residual_return"]) <= -0.5 * float(row["trailing_residual_std"]) and float(row["residual_rank_pct"]) <= (1.0 / 3.0)


def raw_signal_address(row: Mapping[str, Any]) -> str:
    return stable_hash({field: row[field] for field in (
        "raw_policy_hash", "symbol", "decision_ts", "entry_ts", "entry_price", "initial_stop", "risk_denominator",
    )})


def economic_address(row: Mapping[str, Any]) -> str:
    """Canonical trade identity includes direction and executable economics."""
    return stable_hash({field: str(row[field]) for field in (
        "side", "symbol", "decision_ts", "entry_ts", "entry_price", "initial_stop",
        "risk_denominator", "exit_policy", "maximum_exit_ts",
    )})


def cross_section_at_timestamp(
    ctx: Any, panel: pd.DataFrame, feature_cache: dict[str, pd.DataFrame], decision_ts: pd.Timestamp,
    universe_cache: dict[pd.Timestamp, set[str]], rank_cache: dict[pd.Timestamp, pd.DataFrame],
) -> pd.DataFrame:
    timestamp = pd.Timestamp(decision_ts)
    if timestamp in rank_cache:
        return rank_cache[timestamp].copy()
    eligible = pit_universe_symbols(ctx, panel, timestamp, universe_cache)
    rows = []
    for symbol in sorted(eligible):
        frame = feature_cache.get(symbol)
        if frame is None:
            continue
        match = frame[frame.decision_ts.eq(timestamp)]
        if match.empty:
            continue
        row = match.iloc[0]
        if row[["residual_return", "trailing_residual_std", "beta_clipped", "atr_14d"]].isna().any() or int(row.beta_observations) < 30:
            continue
        rows.append({**row.to_dict(), "symbol": symbol})
    result = pd.DataFrame(rows)
    if len(result):
        result["residual_rank_pct"] = result.residual_return.rank(method="average", pct=True, ascending=True)
        result["residual_rank_population"] = len(result)
        result = result.sort_values(["residual_rank_pct", "symbol"]).reset_index(drop=True)
    rank_cache[timestamp] = result.copy()
    return result


def enumerate_raw_signals(
    ctx: Any, panel: pd.DataFrame, feature_cache: dict[str, pd.DataFrame], references: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[pd.Timestamp, set[str]], dict[pd.Timestamp, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    universe_cache: dict[pd.Timestamp, set[str]] = {}
    rank_cache: dict[pd.Timestamp, pd.DataFrame] = {}
    listing = panel.set_index("symbol").start_ts.map(pd.Timestamp).to_dict()
    impulses = references[
        references.qualified_btc_impulse & references.btc_source_ts.ge(START) & references.btc_source_ts.lt(PROTECTED)
    ].sort_values("btc_source_ts")
    for impulse in impulses.itertuples(index=False):
        decision_ts = pd.Timestamp(impulse.btc_source_ts)
        cross = cross_section_at_timestamp(ctx, panel, feature_cache, decision_ts, universe_cache, rank_cache)
        for lag_profile in ("moderate_lag", "deep_lag"):
            raw_hash = raw_policy_hash(lag_profile)
            selected = cross[cross.apply(lambda row: lag_condition(row, lag_profile), axis=1)] if len(cross) else cross
            for decision in selected.itertuples(index=False):
                symbol = str(decision.symbol)
                if decision_ts < pd.Timestamp(listing[symbol]) + pd.Timedelta(days=30):
                    continue
                if pd.isna(decision.feature_available_ts) or pd.Timestamp(decision.feature_available_ts) > decision_ts:
                    continue
                if pd.isna(decision.entry_ts) or pd.isna(decision.next_5m_open):
                    drops.append({"symbol": symbol, "decision_ts": decision_ts, "reason": "next_5m_open_unavailable", "raw_policy_hash": raw_hash})
                    continue
                entry_ts = pd.Timestamp(decision.entry_ts)
                if entry_ts >= PROTECTED:
                    drops.append({"symbol": symbol, "decision_ts": decision_ts, "entry_ts": entry_ts, "reason": "entry_crosses_protected_boundary", "raw_policy_hash": raw_hash})
                    continue
                risk = float(decision.next_5m_open) - float(decision.low)
                risk_atr = risk / float(decision.atr_14d)
                if not 0.25 <= risk_atr <= 1.5:
                    continue
                period, window_start, window_end = execution.evaluation_window(entry_ts)
                row = {
                    "raw_policy_hash": raw_hash, "lag_profile": lag_profile, "symbol": symbol, "side": "long",
                    "decision_ts": decision_ts, "feature_available_ts": decision.feature_available_ts,
                    "btc_source_ts": decision.btc_source_ts, "eth_source_ts": decision.eth_source_ts,
                    "btc_return_2h": float(decision.btc_return_2h), "eth_return_2h": float(decision.eth_return_2h),
                    "parent_state": decision.parent_state, "parent_source_ts": decision.parent_source_ts,
                    "entry_ts": entry_ts, "entry_price": float(decision.next_5m_open), "initial_stop": float(decision.low),
                    "risk_denominator": risk, "risk_to_daily_atr": risk_atr, "daily_atr": float(decision.atr_14d),
                    "alt_return_2h": float(decision.return_2h), "beta_clipped": float(decision.beta_clipped),
                    "beta_observations": int(decision.beta_observations), "residual_return": float(decision.residual_return),
                    "trailing_residual_std": float(decision.trailing_residual_std), "residual_rank_pct": float(decision.residual_rank_pct),
                    "residual_rank_population": int(decision.residual_rank_population),
                    "evaluation_period": period, "evaluation_window_start": window_start, "evaluation_window_end": window_end,
                    "imputed_funding_gate_activated": False,
                }
                row["raw_signal_address_hash"] = raw_signal_address(row)
                row["raw_signal_id"] = "H43RAW_" + row["raw_signal_address_hash"][:24]
                rows.append(row)
    return rows, drops, universe_cache, rank_cache


def parent_allowed(policy: str, state: str) -> bool:
    return state == "both_up" if policy == "both_up" else state in {"both_up", "both_down", "mixed_at_least_one_down"}


def project_parent_policies(raw: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in definitions.drop_duplicates("selected_key_policy_hash").to_dict("records"):
        selected = raw[raw.lag_profile.eq(policy["lag_profile"])]
        selected = selected[selected.parent_state.map(lambda value: parent_allowed(policy["parent_policy"], str(value)))]
        for source in selected.to_dict("records"):
            row = {**source, "parent_policy": policy["parent_policy"], "selected_key_policy_hash": policy["selected_key_policy_hash"]}
            row["candidate_key"] = "H43K_" + stable_hash({"policy": policy["selected_key_policy_hash"], "raw": source["raw_signal_address_hash"]})[:24]
            rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["selected_key_policy_hash", "symbol", "entry_ts", "candidate_key"]).drop_duplicates("candidate_key") if len(result) else result


def simulate_definition(candidates: pd.DataFrame, definition: Mapping[str, Any], execute_fn: Callable) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    open_by_symbol: dict[str, dict[str, Any]] = {}
    for key in candidates.sort_values(["entry_ts", "symbol", "candidate_key"]).to_dict("records"):
        prior = open_by_symbol.get(key["symbol"])
        if prior is not None and pd.Timestamp(key["entry_ts"]) < pd.Timestamp(prior["exit_ts"]):
            skips.append({
                "definition_id": definition["definition_id"], "candidate_key": key["candidate_key"], "symbol": key["symbol"],
                "entry_ts": key["entry_ts"], "prior_trade_id": prior["event_id"], "prior_entry_ts": prior["entry_ts"],
                "prior_actual_exit_ts": prior["exit_ts"], "skip_reason": "same_symbol_same_definition_position_actually_open",
            })
            continue
        event, exclusion = execute_fn(key, definition["exit_policy"])
        if exclusion is not None:
            exclusions.append({**exclusion, "definition_id": definition["definition_id"]})
            continue
        assert event is not None
        event["definition_id"] = definition["definition_id"]
        event["parameter_vector_hash"] = definition["parameter_vector_hash"]
        event["event_id"] = "H43E_" + stable_hash({"candidate": key["candidate_key"], "definition": definition["definition_id"]})[:24]
        event["candidate_economic_address_hash"] = economic_address(event)
        accepted.append(event)
        open_by_symbol[key["symbol"]] = event
    return accepted, skips, exclusions


def indexed_execution_data(bars: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    ordered_bars = bars.sort_values("ts").reset_index(drop=True)
    ordered_frame = frame.sort_values("decision_ts").reset_index(drop=True)
    return {
        "bars": ordered_bars,
        "bar_ts": ordered_bars.ts.to_numpy(dtype="datetime64[ns]"),
        "bar_open": ordered_bars.open.to_numpy(dtype=float),
        "bar_high": ordered_bars.high.to_numpy(dtype=float),
        "bar_low": ordered_bars.low.to_numpy(dtype=float),
        "frame": ordered_frame,
        "frame_ts": ordered_frame.decision_ts.to_numpy(dtype="datetime64[ns]"),
    }


def _first_stop_fill(data: dict[str, Any], start: int, end: int, stop: float, side: str) -> tuple[int, float] | None:
    if end <= start:
        return None
    opens = data["bar_open"][start:end]
    extremes = data["bar_low"][start:end] if side == "long" else data["bar_high"][start:end]
    hit = np.flatnonzero((opens <= stop) | (extremes <= stop)) if side == "long" else np.flatnonzero((opens >= stop) | (extremes >= stop))
    if not len(hit):
        return None
    index = start + int(hit[0])
    gap = data["bar_open"][index] <= stop if side == "long" else data["bar_open"][index] >= stop
    price = float(data["bar_open"][index]) if gap else float(stop)
    return index, price


def execute_event_indexed(key: Mapping[str, Any], exit_policy: str, data: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    entry_ts = pd.Timestamp(key["entry_ts"]); boundary = pd.Timestamp(key["evaluation_window_end"])
    hours = {"fixed_4h": 4, "fixed_8h": 8, "fixed_12h": 12}[exit_policy]
    natural_limit = entry_ts + pd.Timedelta(hours=hours)
    if natural_limit >= boundary:
        return None, {"candidate_key": key.get("candidate_key", key.get("control_key")), "exit_policy": exit_policy, "reason": "maximum_hold_crosses_evaluation_boundary", "entry_ts": entry_ts, "evaluation_window_end": boundary}
    start = int(np.searchsorted(data["bar_ts"], entry_ts.to_datetime64(), side="left"))
    limit_index = int(np.searchsorted(data["bar_ts"], natural_limit.to_datetime64(), side="left"))
    if start >= limit_index or limit_index > len(data["bar_ts"]):
        return None, {"candidate_key": key.get("candidate_key", key.get("control_key")), "exit_policy": exit_policy, "reason": "insufficient_bars_for_natural_exit", "entry_ts": entry_ts, "evaluation_window_end": boundary}
    stop_fill = _first_stop_fill(data, start, limit_index, float(key["initial_stop"]), "long")
    if stop_fill is not None:
        exit_index, exit_price = stop_fill; reason = "completed_alt_2h_impulse_low_stop"
    else:
        if limit_index >= len(data["bar_ts"]) or data["bar_ts"][limit_index] != natural_limit.to_datetime64():
            return None, {"candidate_key": key.get("candidate_key", key.get("control_key")), "exit_policy": exit_policy, "reason": "natural_exit_bar_unavailable", "entry_ts": entry_ts, "evaluation_window_end": boundary}
        exit_index = limit_index; exit_price = float(data["bar_open"][exit_index]); reason = f"fixed_{hours}h_time_exit"
    exit_ts = pd.Timestamp(data["bar_ts"][exit_index], tz="UTC")
    # Time-exit bars are outcome bars and stay excluded; a stop-fill bar is part of the realized path.
    used_end = exit_index if reason.endswith("time_exit") else exit_index + 1
    risk = float(key["risk_denominator"]); entry_price = float(key["entry_price"])
    result = {**dict(key), "exit_policy": exit_policy, "exit_ts": exit_ts, "exit_price": exit_price, "exit_reason": reason,
              "stop_price": float(key["initial_stop"]), "maximum_exit_ts": natural_limit,
              "gross_R": (exit_price-entry_price)/risk,
              "mae_R": min(0.0, (float(data["bar_low"][start:used_end].min())-entry_price)/risk),
              "mfe_R": max(0.0, (float(data["bar_high"][start:used_end].max())-entry_price)/risk),
              "side": "long", "protected_violation": exit_ts >= PROTECTED, "artificial_horizon_exit": False,
              "ohlcv_stop_approximation_cap": True}
    return result, None


def execute_event_scalar(key: Mapping[str, Any], exit_policy: str, bars: pd.DataFrame, frame: pd.DataFrame) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    entry_ts = pd.Timestamp(key["entry_ts"]); boundary = pd.Timestamp(key["evaluation_window_end"])
    hours = {"fixed_4h": 4, "fixed_8h": 8, "fixed_12h": 12}[exit_policy]
    natural_limit = entry_ts + pd.Timedelta(hours=hours)
    if natural_limit >= boundary:
        return None, {"reason": "maximum_hold_crosses_evaluation_boundary"}
    path = bars[(bars.ts >= entry_ts) & (bars.ts < natural_limit)].sort_values("ts")
    if path.empty:
        return None, {"reason": "insufficient_bars_for_natural_exit"}
    stop_path = path
    hit = stop_path[(stop_path.open <= float(key["initial_stop"])) | (stop_path.low <= float(key["initial_stop"]))]
    if len(hit):
        row = hit.iloc[0]; exit_ts = pd.Timestamp(row.ts); exit_price = float(row.open) if float(row.open) <= float(key["initial_stop"]) else float(key["initial_stop"]); reason = "completed_alt_2h_impulse_low_stop"
    else:
        fill = bars[bars.ts.eq(natural_limit)]
        if fill.empty: return None, {"reason": "natural_exit_bar_unavailable"}
        exit_ts = natural_limit; exit_price = float(fill.iloc[0].open); reason = f"fixed_{hours}h_time_exit"
    used = bars[(bars.ts >= entry_ts) & ((bars.ts < exit_ts) if reason.endswith("time_exit") else (bars.ts <= exit_ts))]
    risk=float(key["risk_denominator"]); entry=float(key["entry_price"])
    return {**dict(key), "exit_policy":exit_policy, "exit_ts":exit_ts, "exit_price":exit_price, "exit_reason":reason,
            "stop_price":float(key["initial_stop"]), "maximum_exit_ts":natural_limit, "gross_R":(exit_price-entry)/risk,
            "mae_R":min(0.0,(float(used.low.min())-entry)/risk), "mfe_R":max(0.0,(float(used.high.max())-entry)/risk),
            "side":"long", "protected_violation":exit_ts>=PROTECTED, "artificial_horizon_exit":False, "ohlcv_stop_approximation_cap":True}, None


def simulate_all(candidates: pd.DataFrame, definitions: pd.DataFrame, indexed_cache: dict[str, dict[str, Any]], progress_fn: Callable[[int, int, int, int], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes, skips, exclusions = [], [], []
    for number, definition in enumerate(definitions.to_dict("records"), 1):
        selected = candidates[candidates.selected_key_policy_hash.eq(definition["selected_key_policy_hash"])]
        def execute_fn(key: Mapping[str, Any], exit_policy: str):
            return execute_event_indexed(key, exit_policy, indexed_cache[key["symbol"]])
        accepted, omitted, excluded = simulate_definition(selected, definition, execute_fn)
        outcomes.extend(accepted); skips.extend(omitted); exclusions.extend(excluded)
        if progress_fn is not None:
            progress_fn(number, len(outcomes), len(skips), len(exclusions))
    return pd.DataFrame(outcomes), pd.DataFrame(skips), pd.DataFrame(exclusions)


def deterministic_hash(frame: pd.DataFrame, fields: list[str], sort_fields: list[str]) -> str:
    if frame.empty:
        return stable_hash([])
    return stable_hash(frame.sort_values(sort_fields)[fields].astype(str).to_dict("records"))


def raw_nesting_audit(raw: pd.DataFrame, projected: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lag_profile, group in definitions.drop_duplicates("selected_key_policy_hash").groupby("lag_profile"):
        strict_hash = group[group.parent_policy.eq("both_up")].selected_key_policy_hash.iloc[0]
        broad_hash = group[group.parent_policy.eq("all_regime_comparator")].selected_key_policy_hash.iloc[0]
        strict = set(projected[projected.selected_key_policy_hash.eq(strict_hash)].raw_signal_address_hash)
        broad = set(projected[projected.selected_key_policy_hash.eq(broad_hash)].raw_signal_address_hash)
        raw_set = set(raw[raw.lag_profile.eq(lag_profile)].raw_signal_address_hash)
        known_raw = set(raw[raw.lag_profile.eq(lag_profile) & raw.parent_state.isin({"both_up","both_down","mixed_at_least_one_down"})].raw_signal_address_hash)
        rows.append({"lag_profile": lag_profile, "raw_rows": len(raw_set), "known_parent_raw_rows":len(known_raw), "strict_rows": len(strict), "all_regime_rows": len(broad), "strict_not_in_all": len(strict-broad), "all_missing_known_raw": len(known_raw-broad), "pass": strict <= broad and broad == known_raw})
    return pd.DataFrame(rows)


def exactness_sentinel(raw: pd.DataFrame, projected: pd.DataFrame, definitions: pd.DataFrame, bars_cache: dict[str, pd.DataFrame], four_cache: dict[str, pd.DataFrame], indexed_cache: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for raw_hash, group in raw.groupby("raw_policy_hash"):
        ordered = group.sort_values(["symbol", "entry_ts"]).head(5)
        first = deterministic_hash(ordered, ["raw_signal_address_hash"], ["raw_signal_address_hash"])
        second = deterministic_hash(ordered.copy(), ["raw_signal_address_hash"], ["raw_signal_address_hash"])
        rows.append({"scope": "raw_policy_hash", "policy_hash": raw_hash, "sample_rows": len(ordered), "first_hash": first, "second_hash": second, "mismatch_count": int(first != second), "pass": first == second})
    representatives = definitions.sort_values("definition_id")
    for definition in representatives.to_dict("records"):
        selected = projected[projected.selected_key_policy_hash.eq(definition["selected_key_policy_hash"])].sort_values(["symbol", "entry_ts"]).head(3)
        scalar_rows, indexed_rows = [], []
        for key in selected.to_dict("records"):
            scalar, scalar_exclusion = execute_event_scalar(key, definition["exit_policy"], bars_cache[key["symbol"]], four_cache[key["symbol"]])
            indexed, indexed_exclusion = execute_event_indexed(key, definition["exit_policy"], indexed_cache[key["symbol"]])
            if scalar is not None:
                scalar_value = {field: scalar[field] for field in ("candidate_key","exit_ts","exit_price","exit_reason","stop_price","gross_R","mae_R","mfe_R")}
            else:
                scalar_value = {"excluded": scalar_exclusion.get("reason") if scalar_exclusion else "missing"}
            if indexed is not None:
                indexed_value = {field: indexed[field] for field in ("candidate_key","exit_ts","exit_price","exit_reason","stop_price","gross_R","mae_R","mfe_R")}
            else:
                indexed_value = {"excluded": indexed_exclusion.get("reason") if indexed_exclusion else "missing"}
            scalar_rows.append(stable_hash(scalar_value)); indexed_rows.append(stable_hash(indexed_value))
        scalar_hash = stable_hash(scalar_rows); indexed_hash = stable_hash(indexed_rows)
        rows.append({"scope": "selected_key_policy_hash_exit_policy", "policy_hash": definition["selected_key_policy_hash"], "exit_policy":definition["exit_policy"], "sample_rows": len(selected), "first_hash": scalar_hash, "second_hash": indexed_hash, "mismatch_count": int(scalar_hash != indexed_hash), "pass": scalar_hash == indexed_hash})
    return pd.DataFrame(rows)


def attach_directional_costs(events: pd.DataFrame, funding: pd.DataFrame, key_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply venue funding with the correct long/short sign without changing exact rows."""
    if events.empty:
        return events, pd.DataFrame()
    outputs, boundary_outputs = [], []
    for side, group in events.groupby("side", sort=False):
        if side == "long":
            scored, boundaries = execution.attach_long_costs(group.copy(), funding, key_col)
        elif side == "short":
            scored, boundaries = reports.attach_costs(group.copy(), funding, key_col)
        else:
            raise RuntimeError(f"unsupported direction: {side}")
        outputs.append(scored)
        if len(boundaries):
            boundaries = boundaries.copy(); boundaries["side"] = side; boundary_outputs.append(boundaries)
    scored = pd.concat(outputs, ignore_index=True).sort_values(key_col).reset_index(drop=True)
    boundaries = pd.concat(boundary_outputs, ignore_index=True) if boundary_outputs else pd.DataFrame()
    return scored, boundaries


def build_control_pool(frame: pd.DataFrame, data: dict[str, Any]) -> pd.DataFrame:
    """Construct outcome-blind completed-4h controls with the candidate stop rule."""
    pool = frame.copy().sort_values("decision_ts").reset_index(drop=True)
    pool["initial_stop"] = pool.low
    indices = np.searchsorted(data["bar_ts"], pool.decision_ts.to_numpy(dtype="datetime64[ns]"), side="left")
    valid = indices < len(data["bar_ts"])
    pool["entry_ts"] = pd.Series(pd.NaT, index=pool.index, dtype="datetime64[ns, UTC]")
    pool["entry_price"] = np.nan
    pool.loc[valid,"entry_ts"] = pd.to_datetime(data["bar_ts"][indices[valid]],utc=True)
    pool.loc[valid,"entry_price"] = data["bar_open"][indices[valid]]
    pool["risk_denominator"] = pool.entry_price - pool.initial_stop
    pool["risk_to_daily_atr"] = pool.risk_denominator/pool.atr_14d
    pool["daily_atr"] = pool.atr_14d
    pool["side"] = "long"
    pool = pool[
        pool.feature_available_ts.le(pool.decision_ts) & pool.risk_to_daily_atr.between(.25,1.5)
        & pool.entry_ts.notna() & pool.entry_price.notna()
        & pool.decision_ts.ge(START) & pool.entry_ts.lt(PROTECTED)
    ].copy()
    return pool


def _pool_choice(
    pool: pd.DataFrame, mask: pd.Series, decision_ts: pd.Timestamp, candidate_risk: float,
    pit_eligible: Callable[[pd.Timestamp], bool], row_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any] | None:
    eligible = pool[mask & pool.decision_ts.lt(decision_ts)].copy()
    if eligible.empty:
        return None
    eligible["risk_match_distance"] = (eligible.risk_to_daily_atr-candidate_risk).abs()
    eligible = eligible[eligible.risk_match_distance.le(RISK_MATCH_TOLERANCE_ATR)]
    if eligible.empty:
        return None
    eligible["stable_tie"] = eligible.apply(
        lambda row: stable_hash({"decision_ts": row.decision_ts, "entry_price": row.entry_price}), axis=1
    )
    eligible = eligible.sort_values(["risk_match_distance", "decision_ts", "stable_tie"])
    row = None
    for candidate in eligible.to_dict("records"):
        if pit_eligible(pd.Timestamp(candidate["decision_ts"])) and (row_predicate is None or row_predicate(candidate)):
            row = candidate
            break
    if row is None:
        return None
    fields = (
        "decision_ts", "feature_available_ts", "entry_ts", "entry_price", "initial_stop",
        "risk_denominator", "risk_to_daily_atr", "daily_atr", "parent_state", "side",
        "btc_source_ts", "eth_source_ts", "btc_return_2h", "eth_return_2h", "return_2h",
        "beta_clipped", "beta_observations", "residual_return", "trailing_residual_std",
    )
    return {field: row[field] for field in fields}


def deduplicate_control_addresses(controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign each definition-level economic address to one evidence class."""
    if controls.empty:
        return controls, pd.DataFrame()
    priority = {name: index for index, name in enumerate(CONTROL_CLASSES)}
    ordered = controls.assign(control_class_priority=controls.control_class.map(priority)).sort_values([
        "definition_id", "control_economic_address_hash", "control_class_priority", "candidate_key"
    ])
    duplicated = ordered.duplicated(["definition_id", "control_economic_address_hash"], keep="first")
    owners = ordered.loc[~duplicated, ["definition_id", "control_economic_address_hash", "control_class"]].rename(
        columns={"control_class": "retained_control_class"}
    )
    rejected = ordered.loc[duplicated].merge(
        owners, on=["definition_id", "control_economic_address_hash"], how="left", validate="many_to_one"
    )
    return ordered.loc[~duplicated].drop(columns="control_class_priority"), rejected


def build_control_keys(
    ctx: Any, panel: pd.DataFrame, candidates: pd.DataFrame, outcomes: pd.DataFrame,
    feature_cache: dict[str, pd.DataFrame], indexed_cache: dict[str, dict[str, Any]],
    universe_cache: dict[pd.Timestamp, set[str]], rank_cache: dict[pd.Timestamp, pd.DataFrame],
    progress_fn: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, unavailable = [], []
    policies = {key: group[["definition_id", "exit_policy"]].drop_duplicates() for key, group in outcomes.groupby("candidate_key")}
    pools = {}
    for symbol in indexed_cache:
        pools[symbol] = build_control_pool(feature_cache[symbol], indexed_cache[symbol])
    def pit_control_allowed(symbol: str, decision_ts: pd.Timestamp) -> bool:
        return symbol in pit_universe_symbols(ctx, panel, pd.Timestamp(decision_ts), universe_cache)
    def rank_row(symbol: str, decision_ts: pd.Timestamp) -> Mapping[str, Any] | None:
        cross = cross_section_at_timestamp(ctx, panel, feature_cache, pd.Timestamp(decision_ts), universe_cache, rank_cache)
        match = cross[cross.symbol.eq(symbol)] if len(cross) else cross
        return match.iloc[0].to_dict() if len(match) else None
    for number,key in enumerate(candidates.itertuples(index=False),1):
        if key.candidate_key not in policies:
            continue
        pool = pools[key.symbol].copy()
        candidate_risk = float(key.risk_to_daily_atr)
        proposals: dict[str, list[dict[str, Any]]] = {name: [] for name in CONTROL_CLASSES}
        masks = {
            "btc_impulse_top_half_residual": pool.qualified_btc_impulse,
            "lag_without_btc_impulse": ~pool.qualified_btc_impulse,
            "alt_own_impulse_without_btc_lead": pool.alt_own_impulse & ~pool.qualified_btc_impulse,
            "unconditional_alt_after_btc_impulse": pool.qualified_btc_impulse,
            "same_symbol_same_parent_random_long": pool.parent_state.eq(key.parent_state),
        }
        for control_class, mask in masks.items():
            predicate = None
            if control_class == "btc_impulse_top_half_residual":
                predicate = lambda row, symbol=key.symbol: (ranked := rank_row(symbol, pd.Timestamp(row["decision_ts"]))) is not None and float(ranked["residual_rank_pct"]) > 0.5
            elif control_class == "lag_without_btc_impulse":
                predicate = lambda row, symbol=key.symbol, profile=key.lag_profile: (ranked := rank_row(symbol, pd.Timestamp(row["decision_ts"]))) is not None and lag_condition(ranked, profile)
            choice = _pool_choice(
                pool, mask, pd.Timestamp(key.decision_ts), candidate_risk,
                lambda ts, symbol=key.symbol: pit_control_allowed(symbol, ts),
                predicate,
            )
            if choice:
                proposals[control_class].append(choice)
        for control_class, options in proposals.items():
            eligible = [item for item in options if 0.25 <= item["risk_to_daily_atr"] <= 1.5]
            if not eligible:
                unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "zero_risk_band_eligible_decision_time_controls"})
                continue
            eligible.sort(key=lambda item: (abs(item["risk_to_daily_atr"]-candidate_risk), pd.Timestamp(item["decision_ts"]), float(item["entry_price"])))
            chosen = eligible[0]
            risk_distance = abs(chosen["risk_to_daily_atr"]-candidate_risk)
            if risk_distance > RISK_MATCH_TOLERANCE_ATR:
                unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "no_control_within_frozen_risk_distance"})
                continue
            for definition in policies[key.candidate_key].itertuples(index=False):
                period, window_start, window_end = execution.evaluation_window(pd.Timestamp(chosen["entry_ts"]))
                maximum = pd.Timestamp(chosen["entry_ts"]) + pd.Timedelta(hours={"fixed_4h":4,"fixed_8h":8,"fixed_12h":12}[definition.exit_policy])
                if maximum >= window_end:
                    unavailable.append({"candidate_key": key.candidate_key, "control_class": control_class, "reason": "control_natural_maximum_crosses_evaluation_boundary"})
                    continue
                vector = {"definition_id": definition.definition_id, "candidate_key": key.candidate_key, "control_class": control_class, "symbol": key.symbol, "entry_ts": chosen["entry_ts"], "stop": chosen["initial_stop"]}
                control_key = "H43C_" + stable_hash(vector)[:24]
                address_row = {**chosen, "symbol": key.symbol, "exit_policy": definition.exit_policy, "maximum_exit_ts": maximum}
                rows.append({
                    **chosen, "control_key": control_key, "candidate_key": key.candidate_key, "definition_id": definition.definition_id,
                    "control_class": control_class, "symbol": key.symbol, "lag_profile": key.lag_profile, "exit_policy": definition.exit_policy,
                    "evaluation_period": period, "evaluation_window_start": window_start, "evaluation_window_end": window_end,
                    "maximum_exit_ts": maximum, "risk_match_distance": risk_distance,
                    "control_economic_address_hash": economic_address(address_row), "placeholder_control": False,
                    "outcome_accessed_before_freeze": False,
                })
        if progress_fn is not None and (number % 100 == 0 or number == len(candidates)):
            progress_fn(number,len(rows))
    controls = pd.DataFrame(rows)
    if len(controls):
        controls, rejected = deduplicate_control_addresses(controls)
        unavailable.extend({
            "candidate_key": row.candidate_key,
            "control_class": row.control_class,
            "reason": f"economic_address_duplicate_retained_as_{row.retained_control_class}",
        } for row in rejected.itertuples(index=False))
    return controls, pd.DataFrame(unavailable)


def indexed_path_report(events: pd.DataFrame, indexed_cache: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows=[]
    for event in events.itertuples(index=False):
        data=indexed_cache[event.symbol]; entry=pd.Timestamp(event.entry_ts); start=int(np.searchsorted(data["bar_ts"],entry.to_datetime64(),side="left"))
        for hours in (2,4,8,12):
            horizon=entry+pd.Timedelta(hours=hours); horizon_index=int(np.searchsorted(data["bar_ts"],horizon.to_datetime64(),side="left"))
            end=min(horizon_index,len(data["bar_ts"])); fill=_first_stop_fill(data,start,end,float(event.initial_stop),event.side)
            if fill is not None:
                exit_index,price=fill; status="stopped_before_horizon"
            elif horizon_index < len(data["bar_ts"]):
                exit_index=horizon_index; price=float(data["bar_open"][exit_index]); status="marked_at_horizon_next_5m_open"
            else: continue
            used_end=exit_index+1 if status == "stopped_before_horizon" else exit_index
            risk=float(event.risk_denominator); entry_price=float(event.entry_price)
            if event.side == "long":
                gross=(price-entry_price)/risk
                mae=min(0.0,(float(data["bar_low"][start:used_end].min())-entry_price)/risk)
                mfe=max(0.0,(float(data["bar_high"][start:used_end].max())-entry_price)/risk)
            else:
                gross=(entry_price-price)/risk
                mae=min(0.0,(entry_price-float(data["bar_high"][start:used_end].max()))/risk)
                mfe=max(0.0,(entry_price-float(data["bar_low"][start:used_end].min()))/risk)
            frame_match=data["frame"][data["frame"].decision_ts.eq(horizon)]
            horizon_residual=float(frame_match.iloc[0].residual_return) if len(frame_match) else np.nan
            rows.append({"definition_id":event.definition_id,"event_id":event.event_id,"horizon_hours":hours,"path_status":status,"path_exit_ts":pd.Timestamp(data["bar_ts"][exit_index],tz="UTC"),"gross_R":gross,"mae_R":mae,"mfe_R":mfe,"entry_residual_return":event.residual_return,"horizon_residual_return":horizon_residual,"residual_convergence":horizon_residual-event.residual_return if pd.notna(horizon_residual) else np.nan})
    return pd.DataFrame(rows)


def control_reports(events: pd.DataFrame, controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary, bias, paired = [], [], []
    for definition_id in sorted(events.definition_id.unique()):
        candidate = events[events.definition_id.eq(definition_id)]
        for control_class in CONTROL_CLASSES:
            group = controls[(controls.definition_id == definition_id) & controls.control_class.eq(control_class)]
            matched = candidate[candidate.candidate_key.isin(set(group.candidate_key))]
            unmatched = candidate[~candidate.candidate_key.isin(set(group.candidate_key))]
            coverage = len(matched) / max(1, len(candidate))
            adequate = len(group) >= 15 and coverage >= 0.70
            for mode in ("base", "conservative", "severe"):
                candidate_col = f"net_{mode}_R"
                summary.append({"definition_id": definition_id, "control_class": control_class, "cost_mode": mode, "matched_count": len(matched), "unmatched_count": len(unmatched), "full_count": len(candidate), "unique_control_addresses": group.control_economic_address_hash.nunique(), "coverage": coverage, "adequate_control": adequate, "risk_comparable": True, "matched_candidate_mean_R": matched[candidate_col].mean(), "control_mean_R": group[candidate_col].mean(), "candidate_minus_control_mean_R": matched[candidate_col].mean()-group[candidate_col].mean()})
                bias.append({"definition_id": definition_id, "control_class": control_class, "cost_mode": mode, "full_count": len(candidate), "matched_count": len(matched), "unmatched_count": len(unmatched), "matched_candidate_mean_R": matched[candidate_col].mean(), "unmatched_only_candidate_mean_R": unmatched[candidate_col].mean(), "full_candidate_mean_R": candidate[candidate_col].mean(), "matched_minus_unmatched_mean_R": matched[candidate_col].mean()-unmatched[candidate_col].mean() if len(unmatched) else np.nan})
                if control_class != "same_symbol_same_parent_random_long":
                    merged = matched[["candidate_key", candidate_col]].merge(group[["candidate_key", candidate_col]], on="candidate_key", suffixes=("_candidate", "_control"))
                    component = {
                        "btc_impulse_top_half_residual": "lag_selection_given_btc_impulse",
                        "lag_without_btc_impulse": "btc_impulse_given_lag_condition",
                        "alt_own_impulse_without_btc_lead": "btc_lead_over_alt_own_impulse",
                        "unconditional_alt_after_btc_impulse": "lag_selection_over_unconditional_impulse_entry",
                    }[control_class]
                    paired.append({"definition_id": definition_id, "control_class": control_class, "incremental_component": component, "cost_mode": mode, "pairs": len(merged), "mean_incremental_R": (merged[f"{candidate_col}_candidate"]-merged[f"{candidate_col}_control"]).mean(), "median_incremental_R": (merged[f"{candidate_col}_candidate"]-merged[f"{candidate_col}_control"]).median(), "positive_incremental_fraction": (merged[f"{candidate_col}_candidate"]>merged[f"{candidate_col}_control"]).mean()})
    return pd.DataFrame(summary), pd.DataFrame(bias), pd.DataFrame(paired)


def leave_one(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol_rows, month_rows = [], []
    work = events.copy(); work["month"] = pd.to_datetime(work.entry_ts, utc=True).dt.strftime("%Y-%m")
    for definition_id, group in work.groupby("definition_id"):
        for mode in ("base", "conservative", "severe"):
            column = f"net_{mode}_R"
            for symbol in group.symbol.unique():
                remaining = group[group.symbol.ne(symbol)]; symbol_rows.append({"definition_id": definition_id, "cost_mode": mode, "omitted_symbol": symbol, "events": len(remaining), "mean_R": remaining[column].mean(), "total_R": remaining[column].sum()})
            for month in group.month.unique():
                remaining = group[group.month.ne(month)]; month_rows.append({"definition_id": definition_id, "cost_mode": mode, "omitted_month": month, "events": len(remaining), "mean_R": remaining[column].mean(), "total_R": remaining[column].sum()})
    return pd.DataFrame(symbol_rows), pd.DataFrame(month_rows)


def pairwise_definition_overlap(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Parent projection and exit fanout are policy metadata, not signal identity.
    signals = {definition: set(group.raw_signal_address_hash) for definition, group in events.groupby("definition_id")}
    economics = {definition: set(group.candidate_economic_address_hash) for definition, group in events.groupby("definition_id")}
    for left, right in itertools.combinations(sorted(signals), 2):
        intersection = signals[left] & signals[right]
        union = signals[left] | signals[right]
        economic_intersection = economics[left] & economics[right]
        rows.append({
            "left_definition_id": left, "right_definition_id": right,
            "left_rows": len(signals[left]), "right_rows": len(signals[right]),
            "shared_signal_addresses": len(intersection),
            "shared_economic_addresses": len(economic_intersection),
            "signal_jaccard_overlap": len(intersection) / len(union) if union else np.nan,
        })
    return pd.DataFrame(rows)


def decision_table(summary: pd.DataFrame, concentration: pd.DataFrame, controls: pd.DataFrame, periods: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition in definitions.itertuples(index=False):
        stats = summary[summary.definition_id.eq(definition.definition_id)].set_index("cost_mode")
        base_row, conservative, severe = stats.loc["base"], stats.loc["conservative"], stats.loc["severe"]
        forensic = concentration[(concentration.definition_id == definition.definition_id) & concentration.cost_mode.eq("conservative")]
        robust = bool(len(forensic) and forensic.iloc[0].mean_after_top3 > 0 and forensic.iloc[0].worst_leave_one_symbol_mean_R > 0 and forensic.iloc[0].worst_leave_one_month_mean_R > 0)
        positive = controls[(controls.definition_id == definition.definition_id) & controls.cost_mode.eq("conservative") & controls.adequate_control & controls.candidate_minus_control_mean_R.gt(0)]
        classes = set(positive.control_class)
        positive_periods = int(periods[(periods.definition_id == definition.definition_id) & periods.cost_mode.eq("conservative")].mean_R.gt(0).sum())
        if base_row.events >= 30 and base_row.symbols >= 10 and base_row.mean_R > 0 and conservative.mean_R > 0 and severe.mean_R > 0 and robust and positive_periods >= 3 and len(classes) >= 2 and classes & CONTEXTUAL_CONTROLS and classes & STRUCTURAL_CONTROLS:
            decision = "materialization_candidate"
        elif definition.parent_policy == "both_up" and base_row.events >= 15 and conservative.mean_R > 0 and robust and classes & STRUCTURAL_CONTROLS:
            decision = "fragile_context_sleeve"
        else:
            decision = "current_translation_weak"
        rows.append({"definition_id": definition.definition_id, "decision": decision, "events": int(base_row.events), "symbols": int(base_row.symbols), "base_mean_R": base_row.mean_R, "conservative_mean_R": conservative.mean_R, "severe_mean_R": severe.mean_R, "positive_adequate_control_classes": len(classes), "positive_periods": positive_periods, "evidence_level": "level_4_event_ledger_plus_real_controls"})
    return pd.DataFrame(rows)


def update_central_artifacts(root: Path, library: pd.DataFrame, final_decision: str) -> None:
    """Create a current snapshot without mutating any historical source root."""
    sources = [
        CAMPAIGN_ROOT / "candidate_library/central_full_schema_candidate_library.csv",
        REFERENCE_ROOT / "candidate_library/candidate_library_update.csv",
    ]
    frames = [pd.read_csv(path) for path in sources]
    central = pd.concat([*frames, library], ignore_index=True, sort=False)
    central = central.drop_duplicates("candidate_id", keep="last").sort_values("candidate_id")
    fbsr = central.get("family_engine_id", pd.Series(index=central.index, dtype=object)).eq("kraken_fbsr_v1")
    central.loc[fbsr, "evidence_level"] = "level_4_event_ledger_plus_real_controls"
    rsbb = central.get("family_engine_id", pd.Series(index=central.index, dtype=object)).eq("kraken_rsbb_v1")
    rsbb_sleeves = rsbb & central.candidate_id.astype(str).isin({"rsbb_v1_013","rsbb_v1_014"})
    central.loc[rsbb, ["candidate_decision","candidate_library_state"]] = "current_translation_weak"
    central.loc[rsbb_sleeves, ["candidate_decision","candidate_library_state"]] = "fragile_context_sleeve"
    write_csv(root / "candidate_library/central_full_schema_candidate_library.csv", central)
    write_json(root / "continuity/continuity_state_snapshot.json", {
        "as_of_run_root": str(root),
        "family": "btc_led_delayed_alt_diffusion",
        "final_decision": final_decision,
        "signal_state_contract_version": SIGNAL_STATE_CONTRACT_VERSION,
        "train_data_end": END.isoformat(),
        "protected_period_start": PROTECTED.isoformat(),
        "final_holdout_sealed": True,
        "automatic_follow_on_authorized": False,
        "materialization_launched": False,
        "validation_launched": False,
        "cpcv_launched": False,
        "portfolio_construction_launched": False,
        "live_work_launched": False,
        "next_phase_requires_human_authorization": True,
        "preserved_rsbb_context_sleeves": ["rsbb_v1_013","rsbb_v1_014"],
        "rsbb_broader_family_decision": "current_translation_weak",
        "rsbb_materialization_or_stability_scheduled": False,
        "candidate_library_rows": int(len(central)),
        "candidate_library_hash": file_hash(root / "candidate_library/central_full_schema_candidate_library.csv"),
    })


def write_artifact_hash_inventory(root: Path) -> Path:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "compact_review_bundle" in path.parts or path.name == "artifact_hash_manifest.csv":
            continue
        rows.append({
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        })
    target = root / "reproducibility/artifact_hash_manifest.csv"
    write_csv(target, rows)
    return target


def compact_bundle(root: Path) -> Path:
    files = (
        "SCREEN_REPORT.md", "decision_summary.json", "contract/btc_led_delayed_alt_diffusion_contract.md",
        "manifest/definitions.csv", "manifest/instrument_universe_audit.csv",
        "audit/rankable_signal_state_contract.json", "audit/exactness_sentinel.csv", "audit/raw_parent_nesting.csv", "audit/btc_alignment_audit.csv",
        "audit/deterministic_replay_parity.csv", "audit/non_overlap_skip_ledger.csv", "audit/boundary_reconciliation.csv", "audit/pairwise_definition_overlap.csv",
        "audit/hard_gate_audit.csv", "audit/reference_root_immutability.csv", "economics/definition_summary.csv", "economics/period_summary.csv", "economics/dimension_summary.csv",
        "forensics/concentration_and_removal.csv", "forensics/parameter_neighborhood.csv", "forensics/exact_vs_imputed_funding.csv",
        "forensics/horizon_path_summary.csv",
        "controls/control_summary.csv", "controls/risk_stable_control_diagnostics.csv", "controls/paired_incremental_value.csv",
        "controls/matched_unmatched_bias.csv", "decision/candidate_decisions.csv", "candidate_library/candidate_library_update.csv",
        "candidate_library/central_full_schema_candidate_library.csv", "continuity/continuity_state_snapshot.json",
        "reproducibility/run_manifest.json", "reproducibility/executed_runner.py", "reproducibility/artifact_hash_manifest.csv",
    )
    temp = root / ".compact_review_bundle.tmp"
    if temp.exists(): shutil.rmtree(temp)
    temp.mkdir()
    inventory = []
    for relative in files:
        source = root / relative; target = temp / relative.replace("/", "__")
        shutil.copy2(source, target)
        inventory.append({"source_relative_path": relative, "bundle_file": target.name, "bytes": source.stat().st_size, "sha256": file_hash(source)})
    write_csv(temp / "bundle_manifest.csv", inventory)
    final = root / "compact_review_bundle"
    if final.exists(): shutil.rmtree(final)
    os.replace(temp, final)
    return final


def run(root: Path) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError(f"fresh root required: {root}")
    authority = json.loads((REFERENCE_ROOT / "decision_summary.json").read_text())
    if authority.get("status") != "complete" or authority.get("final_decision") not in {
        "current_translation_weak", "fragile_context_sleeve", "materialization_candidate",
    }:
        raise RuntimeError("reference independent-family closure is not complete")
    architecture = json.loads((VALID_ARCHITECTURE_ROOT / "decision_summary.json").read_text())
    campaign = json.loads((CAMPAIGN_ROOT / "decision_summary.json").read_text())
    if architecture.get("status") != "complete" or campaign.get("status") != "complete":
        raise RuntimeError("signal-state architecture or campaign authority is not closed")
    source_hashes_before = {
        str(path): directory_hash(path)
        for path in (REFERENCE_ROOT, VALID_ARCHITECTURE_ROOT, CAMPAIGN_ROOT)
    }
    root.mkdir(parents=True); started = time.monotonic(); peak_rss = runner.current_rss_bytes()
    definitions = frozen_manifest(); write_csv(root / "manifest/definitions.csv", definitions)
    contract_text = """# BTC-Led Delayed Alt Diffusion v1 Contract

Train-only 2023-2025. Completed ALT, BTC and ETH two-hour bars require exactly 24 valid five-minute source bars and are inner-aligned at identical decision timestamps. A BTC impulse is current BTC two-hour return at least prior-60 mean plus two prior-60 standard deviations, with aligned ETH return nonnegative. Alt beta uses only the previous 60 aligned return pairs, requires 30 observations, and is clipped to [0,3]. Residual is current alt return minus frozen beta times current BTC return. Moderate/deep lag ranks are computed only inside the exact PIT Tier A/B panel at each impulse timestamp.

Entry is the next executable five-minute open, stop is the completed alt two-hour impulse-bar low, and initial risk is 0.25-1.5 completed-daily ATR. BTC and non-crypto/stable synthetic products are structurally excluded using the frozen official instrument snapshot. Parent-neutral raw tapes are frozen before PIT both-up/all-regime projection. Non-overlap is definition-local and uses actual 4h/8h/12h exits. Funding and OI never activate a signal. Controls are frozen before outcomes with the same risk band and +/-0.25 ATR risk-distance rule.
"""
    contract_path = root / "contract/btc_led_delayed_alt_diffusion_contract.md"; contract_path.parent.mkdir(); contract_path.write_text(contract_text, encoding="utf-8")
    ctx = execution.context(root); panel = runner.full_panel_for_launch_gate(ctx); write_csv(root / "manifest/pit_panel.csv", panel); paths = runner.data_paths(ctx)
    universe_audit = instrument_universe_audit(panel); write_csv(root / "manifest/instrument_universe_audit.csv", universe_audit)
    rankable_symbols = set(universe_audit[universe_audit.rankable_alt_perpetual].symbol.astype(str))
    rankable_panel = panel[panel.symbol.astype(str).isin(rankable_symbols)].copy()
    btc_bars = runner.load_symbol_bars(paths, "PF_XBTUSD", START-pd.Timedelta(days=120), END)
    eth_bars = runner.load_symbol_bars(paths, "PF_ETHUSD", START-pd.Timedelta(days=120), END)
    if btc_bars.empty or eth_bars.empty:
        raise RuntimeError("complete PF_XBTUSD and PF_ETHUSD reference bars required")
    fields = ["ts", "open", "high", "low", "close", "volume"]
    btc_bars = btc_bars[fields].copy(); eth_bars = eth_bars[fields].copy()
    references, btc_two, eth_two = btc_eth_reference_frames(btc_bars, eth_bars)
    feature_cache: dict[str, pd.DataFrame] = {}; alignment_rows = []
    symbols = [symbol for symbol in panel.symbol.astype(str) if symbol in rankable_symbols]
    for number, symbol in enumerate(symbols, 1):
        bars = runner.load_symbol_bars(paths, symbol, START-pd.Timedelta(days=120), END)
        if bars.empty: continue
        frame, _, _ = feature_frames(bars[fields].copy(), references)
        feature_cache[symbol] = frame
        alt_rows = int(frame.attrs["alt_completed_2h_rows"])
        alignment_rows.append({
            "symbol": symbol, "alt_completed_2h_rows": alt_rows, "exact_btc_eth_aligned_rows": len(frame),
            "rows_excluded_for_missing_exact_reference_bar": alt_rows-len(frame),
            "btc_timestamp_mismatches": int((pd.to_datetime(frame.btc_source_ts,utc=True)!=pd.to_datetime(frame.decision_ts,utc=True)).sum()),
            "eth_timestamp_mismatches": int((pd.to_datetime(frame.eth_source_ts,utc=True)!=pd.to_datetime(frame.decision_ts,utc=True)).sum()),
        })
        peak_rss = max(peak_rss, runner.current_rss_bytes())
        write_json(root / "watch_status.json", {"status":"running","stage":"aligned_2h_beta_residual_feature_build","symbols_completed":number,"symbols_planned":len(symbols),"rss_bytes":peak_rss,"elapsed_seconds":time.monotonic()-started,"updated_ts":runner.utc_now()})
    raw_rows, raw_drops, universe_cache, rank_cache = enumerate_raw_signals(ctx, rankable_panel, feature_cache, references)
    raw = pd.DataFrame(raw_rows).sort_values(["raw_policy_hash", "symbol", "entry_ts", "raw_signal_address_hash"])
    if raw.empty:
        raise RuntimeError("no PIT-valid parent-neutral delayed-diffusion signals")
    signal_symbols=set(raw.symbol.astype(str))
    bars_cache={symbol:runner.load_symbol_bars(paths,symbol,START-pd.Timedelta(days=1),END)[fields].copy() for symbol in signal_symbols}
    four_cache={symbol:feature_cache[symbol] for symbol in signal_symbols}
    write_csv(root / "audit/btc_alignment_audit.csv", alignment_rows)
    raw_freeze_hash = stable_hash(raw.raw_signal_address_hash.tolist()); raw["raw_signal_freeze_hash"] = raw_freeze_hash
    write_csv(root / "signals/raw_signal_manifest.csv", raw); write_csv(root / "audit/raw_boundary_drop_audit.csv", raw_drops)
    projected = project_parent_policies(raw, definitions)
    projection_hash = stable_hash(projected.candidate_key.tolist()); projected["projection_freeze_hash"] = projection_hash
    write_csv(root / "signals/parent_projected_manifest.csv", projected)
    nesting = raw_nesting_audit(raw, projected, definitions); write_csv(root / "audit/raw_parent_nesting.csv", nesting)
    indexed_cache = {symbol:indexed_execution_data(bars_cache[symbol],four_cache[symbol]) for symbol in bars_cache}
    sentinel = exactness_sentinel(raw, projected, definitions, bars_cache, four_cache, indexed_cache); write_csv(root / "audit/exactness_sentinel.csv", sentinel)
    if len(sentinel) != 14 or not sentinel["pass"].all():
        raise RuntimeError("scalar/indexed exactness sentinel failed")
    write_json(root / "watch_status.json", {"status":"running","stage":"definition_local_actual_exit_non_overlap","raw_signals":len(raw),"projected_signals":len(projected),"rss_bytes":runner.current_rss_bytes(),"elapsed_seconds":time.monotonic()-started,"updated_ts":runner.utc_now()})
    def progress(number: int, accepted: int, skipped: int, excluded: int) -> None:
        nonlocal peak_rss
        peak_rss=max(peak_rss,runner.current_rss_bytes())
        write_json(root / "watch_status.json", {"status":"running","stage":"definition_local_actual_exit_non_overlap","definitions_completed":number,"definitions_planned":12,"accepted_trades":accepted,"overlap_skips":skipped,"outcome_exclusions":excluded,"rss_bytes":peak_rss,"elapsed_seconds":time.monotonic()-started,"updated_ts":runner.utc_now()})
    outcomes, skips, exclusions = simulate_all(projected, definitions, indexed_cache, progress)
    write_json(root / "watch_status.json", {"status":"running","stage":"deterministic_full_replay_parity","accepted_trades":len(outcomes),"rss_bytes":runner.current_rss_bytes(),"elapsed_seconds":time.monotonic()-started,"updated_ts":runner.utc_now()})
    parity_outcomes, parity_skips, parity_exclusions = simulate_all(projected, definitions, indexed_cache)
    outcome_hash = deterministic_hash(outcomes, ["definition_id","candidate_key","entry_ts","exit_ts","exit_price","exit_reason","gross_R"], ["definition_id","candidate_key"])
    parity_hash = deterministic_hash(parity_outcomes, ["definition_id","candidate_key","entry_ts","exit_ts","exit_price","exit_reason","gross_R"], ["definition_id","candidate_key"])
    parity = {"outcome_hash_first":outcome_hash,"outcome_hash_second":parity_hash,"skip_hash_first":deterministic_hash(skips,sorted(skips.columns),["definition_id","candidate_key"]),"skip_hash_second":deterministic_hash(parity_skips,sorted(parity_skips.columns),["definition_id","candidate_key"]),"exclusion_hash_first":deterministic_hash(exclusions,sorted(exclusions.columns),["definition_id","candidate_key"]),"exclusion_hash_second":deterministic_hash(parity_exclusions,sorted(parity_exclusions.columns),["definition_id","candidate_key"])}
    parity["mismatch_count"] = int(parity["outcome_hash_first"] != parity["outcome_hash_second"] or parity["skip_hash_first"] != parity["skip_hash_second"] or parity["exclusion_hash_first"] != parity["exclusion_hash_second"])
    write_csv(root / "audit/deterministic_replay_parity.csv", [parity]); write_csv(root / "audit/non_overlap_skip_ledger.csv", skips); write_csv(root / "audit/outcome_exclusion_ledger.csv", exclusions)
    accepted_hash = stable_hash(outcomes.sort_values(["definition_id","candidate_key"])[["definition_id","candidate_key"]].astype(str).to_dict("records"))
    contract_manifest = {"signal_state_contract_version":SIGNAL_STATE_CONTRACT_VERSION,"raw_signal_hash":raw_freeze_hash,"projection_hash":projection_hash,"accepted_trade_hash":accepted_hash,"raw_signal_count":len(raw),"eligible_definition_rows":sum(len(projected[projected.selected_key_policy_hash.eq(row.selected_key_policy_hash)]) for row in definitions.itertuples()),"accepted_trade_count":len(outcomes),"non_overlap_skip_count":len(skips),"outcome_exclusion_count":len(exclusions),"raw_tape_frozen_before_outcomes":True,"projection_frozen_before_outcomes":True,"non_overlap_reconciled":True,"no_mutable_state_shared_across_definitions":True}
    evidence.assert_rankable_signal_state_contract(contract_manifest); write_json(root / "audit/rankable_signal_state_contract.json", contract_manifest)
    funding = lfbs.funding_panel(); outcomes, boundaries = attach_directional_costs(outcomes, funding, "event_id"); write_csv(root / "materialized/event_ledger.csv", outcomes)
    accepted_candidates = projected[projected.candidate_key.isin(set(outcomes.candidate_key))]
    write_json(root / "watch_status.json", {"status":"running","stage":"control_key_build","accepted_trades":len(outcomes),"rss_bytes":runner.current_rss_bytes(),"elapsed_seconds":time.monotonic()-started,"updated_ts":runner.utc_now()})
    def control_progress(completed: int, rows: int) -> None:
        nonlocal peak_rss
        peak_rss=max(peak_rss,runner.current_rss_bytes())
        write_json(root / "watch_status.json", {"status":"running","stage":"control_key_build","candidate_keys_completed":completed,"candidate_keys_planned":len(accepted_candidates),"control_keys":rows,"rss_bytes":peak_rss,"elapsed_seconds":time.monotonic()-started,"updated_ts":runner.utc_now()})
    control_keys, unavailable = build_control_keys(ctx, rankable_panel, accepted_candidates, outcomes, feature_cache, indexed_cache, universe_cache, rank_cache, control_progress)
    control_freeze_hash = stable_hash(control_keys.control_key.tolist()) if len(control_keys) else stable_hash([])
    if len(control_keys): control_keys["control_key_freeze_hash"] = control_freeze_hash
    write_csv(root / "controls/control_key_manifest.csv", control_keys); write_csv(root / "controls/control_unavailable_reasons.csv", unavailable)
    control_rows, control_exclusions = [], []
    for key in control_keys.to_dict("records"):
        event, exclusion = execute_event_indexed(key, key["exit_policy"], indexed_cache[key["symbol"]])
        if exclusion:
            control_exclusions.append({**exclusion,"control_key":key["control_key"]}); continue
        assert event is not None
        event.update({"control_event_id":key["control_key"],"candidate_key":key["candidate_key"],"definition_id":key["definition_id"],"control_class":key["control_class"],"control_economic_address_hash":key["control_economic_address_hash"],"risk_match_distance":key["risk_match_distance"]})
        control_rows.append(event)
    controls = pd.DataFrame(control_rows)
    controls, control_boundaries = attach_directional_costs(controls, funding, "control_event_id") if len(controls) else (controls, pd.DataFrame())
    write_csv(root / "controls/control_event_ledger.csv", controls); write_csv(root / "audit/control_outcome_exclusions.csv", control_exclusions)
    control_summary, matched_bias, paired = control_reports(outcomes, controls)
    write_csv(root / "controls/control_summary.csv", control_summary); write_csv(root / "controls/matched_unmatched_bias.csv", matched_bias); write_csv(root / "controls/paired_incremental_value.csv", paired)
    write_csv(root / "controls/risk_stable_control_diagnostics.csv", execution.risk_stable_control_diagnostics(controls))
    summary, attribution, periods = reports.summarize_economics(outcomes, definitions); write_csv(root / "economics/definition_summary.csv", summary); write_csv(root / "economics/cost_funding_attribution.csv", attribution); write_csv(root / "economics/period_summary.csv", periods)
    dimension_rows = []
    for fields in (("lag_profile",), ("parent_policy",), ("exit_policy",)):
        for key, group in outcomes.groupby(list(fields)):
            values = key if isinstance(key, tuple) else (key,)
            dimensions = dict(zip(fields, values))
            for mode in ("base", "conservative", "severe"):
                series = group[f"net_{mode}_R"]
                dimension_rows.append({"dimension": "|".join(fields), **dimensions, "cost_mode": mode, "events": len(group), "symbols": group.symbol.nunique(), "mean_R": series.mean(), "median_R": series.median(), "total_R": series.sum(), "positive_fraction": series.gt(0).mean()})
    write_csv(root / "economics/dimension_summary.csv", dimension_rows)
    concentration = lfbs.concentration_forensics(outcomes); write_csv(root / "forensics/concentration_and_removal.csv", concentration)
    neighborhood = outcomes.groupby(["lag_profile","parent_policy","exit_policy"]).agg(events=("event_id","size"),symbols=("symbol","nunique"),months=("entry_ts",lambda x:pd.to_datetime(x,utc=True).dt.strftime("%Y-%m").nunique()),base_mean_R=("net_base_R","mean"),conservative_mean_R=("net_conservative_R","mean"),severe_mean_R=("net_severe_R","mean")).reset_index(); write_csv(root / "forensics/parameter_neighborhood.csv", neighborhood)
    write_csv(root / "forensics/exact_vs_imputed_funding.csv", reports.funding_partition_report(outcomes))
    path_behavior = indexed_path_report(outcomes,indexed_cache)
    write_csv(root / "forensics/horizon_path_behavior.csv", path_behavior)
    path_summary = path_behavior.groupby("horizon_hours").agg(
        event_rows=("event_id", "size"),
        mean_gross_R=("gross_R", "mean"),
        median_gross_R=("gross_R", "median"),
        mean_mae_R=("mae_R", "mean"),
        mean_mfe_R=("mfe_R", "mean"),
        mean_residual_convergence=("residual_convergence", "mean"),
    ).reset_index()
    write_csv(root / "forensics/horizon_path_summary.csv", path_summary)
    leave_symbol, leave_month = leave_one(outcomes); write_csv(root / "forensics/leave_one_symbol.csv",leave_symbol); write_csv(root / "forensics/leave_one_month.csv",leave_month)
    write_csv(root / "audit/pairwise_definition_overlap.csv", pairwise_definition_overlap(outcomes))
    decisions = decision_table(summary, concentration, control_summary, periods, definitions); write_csv(root / "decision/candidate_decisions.csv", decisions)
    interval_violations=[]
    for label,(window_start,window_end) in execution.EVALUATION_WINDOWS.items(): interval_violations.extend(evidence.validate_evaluation_window_intervals(outcomes[outcomes.evaluation_period.eq(label)],window_start=window_start,window_end=window_end).violations)
    artificial = int(outcomes.artificial_horizon_exit.sum())
    control_artificial = int(controls.artificial_horizon_exit.sum()) if len(controls) else 0
    boundary_reconciliation = [
        {"scope": "raw_boundary_drops", "count": len(raw_drops)},
        {"scope": "definition_outcome_exclusions", "count": len(exclusions)},
        {"scope": "control_outcome_exclusions", "count": len(control_exclusions)},
        {"scope": "accepted_candidate_interval_violations", "count": len(interval_violations)},
        {"scope": "candidate_artificial_endpoint_exits", "count": artificial},
        {"scope": "control_artificial_endpoint_exits", "count": control_artificial},
    ]
    write_csv(root / "audit/boundary_reconciliation.csv", boundary_reconciliation)
    eligible_rows=contract_manifest["eligible_definition_rows"]
    source_immutability = [
        {
            "reference_root": path,
            "hash_before": before,
            "hash_after": directory_hash(Path(path)),
        }
        for path, before in source_hashes_before.items()
    ]
    for row in source_immutability:
        row["mutated"] = row["hash_before"] != row["hash_after"]
    write_csv(root / "audit/reference_root_immutability.csv", source_immutability)
    hard = {
        "definitions_evaluated": int(summary.definition_id.nunique()),
        "raw_policy_hashes": raw.raw_policy_hash.nunique(),
        "selected_key_policy_hashes": projected.selected_key_policy_hash.nunique(),
        "exactness_sentinel_rows": len(sentinel),
        "exactness_sentinel_failures": int((~sentinel["pass"]).sum()),
        "raw_duplicates": int(raw.duplicated(["raw_policy_hash", "raw_signal_address_hash"]).sum()),
        "candidate_duplicate_addresses": int(outcomes.duplicated(["definition_id", "candidate_economic_address_hash"]).sum()),
        "control_duplicate_addresses": int(control_keys.duplicated(["definition_id", "control_economic_address_hash"]).sum()) if len(control_keys) else 0,
        "strict_parent_nesting_failures": int((~nesting["pass"]).sum()),
        "deterministic_replay_mismatches": parity["mismatch_count"],
        "unexplained_attrition": int(eligible_rows != len(outcomes) + len(skips) + len(exclusions)),
        "candidate_artificial_endpoint_exits": artificial,
        "control_artificial_endpoint_exits": control_artificial,
        "evaluation_interval_violations": len(interval_violations),
        "candidate_funding_join_missing": int(boundaries.missing.sum()) if len(boundaries) else 0,
        "candidate_funding_join_duplicates": int(boundaries.duplicated(["event_id", "boundary_ts"]).sum()) if len(boundaries) else 0,
        "control_funding_join_missing": int(control_boundaries.missing.sum()) if len(control_boundaries) else 0,
        "control_funding_join_duplicates": int(control_boundaries.duplicated(["control_event_id", "boundary_ts"]).sum()) if len(control_boundaries) else 0,
        "decision_input_leaks": int((raw.feature_available_ts > raw.decision_ts).sum()) + (int((control_keys.feature_available_ts > control_keys.decision_ts).sum()) if len(control_keys) else 0),
        "candidate_protected_period_violations": int(outcomes.protected_violation.sum()),
        "control_protected_period_violations": int(controls.protected_violation.sum()) if len(controls) else 0,
        "placeholder_controls": int(control_keys.placeholder_control.sum()) if len(control_keys) else 0,
        "matched_unmatched_failures": int(((matched_bias.matched_count + matched_bias.unmatched_count) != matched_bias.full_count).sum()) if len(matched_bias) else 0,
        "control_outcomes_before_freeze": int(control_keys.outcome_accessed_before_freeze.sum()) if len(control_keys) else 0,
        "imputed_funding_gate_activations": int(raw.imputed_funding_gate_activated.sum()),
        "btc_alignment_mismatches": int((pd.to_datetime(raw.btc_source_ts, utc=True) != pd.to_datetime(raw.decision_ts, utc=True)).sum()),
        "btc_feature_panel_alignment_mismatches": int(sum(row["btc_timestamp_mismatches"]+row["eth_timestamp_mismatches"] for row in alignment_rows)),
        "excluded_instrument_signal_violations": int(raw.symbol.isin(set(universe_audit[~universe_audit.rankable_alt_perpetual].symbol.astype(str))).sum()),
        "reference_root_mutations": int(sum(row["mutated"] for row in source_immutability)),
    }
    expected={"definitions_evaluated":12,"raw_policy_hashes":2,"selected_key_policy_hashes":4,"exactness_sentinel_rows":14}; gate_rows=[{"gate":key,"value":int(value),"expected":expected.get(key,0),"pass":int(value)==expected.get(key,0)} for key,value in hard.items()]; write_csv(root / "audit/hard_gate_audit.csv",gate_rows); mechanics=all(row["pass"] for row in gate_rows)
    final="focused_mechanical_repair_required" if not mechanics else "materialization_candidate" if decisions.decision.eq("materialization_candidate").any() else "fragile_context_sleeve" if decisions.decision.eq("fragile_context_sleeve").any() else "current_translation_weak"
    library=[]
    for definition in definitions.itertuples(index=False):
        decision_row=decisions[decisions.definition_id.eq(definition.definition_id)].iloc[0]
        library.append({"candidate_id":definition.definition_id,"candidate_definition_id":definition.definition_id,"definition_id":definition.definition_id,"hypothesis_id":"btc_led_delayed_alt_diffusion","family_engine_id":"kraken_h43_v1","parameter_vector_hash":definition.parameter_vector_hash,"selected_key_policy_hash":definition.selected_key_policy_hash,"candidate_library_state":decision_row.decision if mechanics else final,"candidate_decision":decision_row.decision if mechanics else final,"evidence_level":"level_4_event_ledger_plus_real_controls","evidence_level_contract":"train_only_not_validation_not_holdout_not_live","clean_evidence_allowed":False,"evidence_cap_reason":"trailing_beta_proxy|shared_funding_imputation|ohlcv_stop|no_depth|train_only","family_rejected":False,"train_only":True,"validation_run":False,"holdout_touched":False,"live_ready":False,"event_rows":decision_row.events,"symbols":decision_row.symbols,"base_mean_R":decision_row.base_mean_R,"conservative_mean_R":decision_row.conservative_mean_R,"severe_mean_R":decision_row.severe_mean_R,"source_run_root":str(root),"contract_version":CONTRACT_VERSION,"signal_state_contract_version":SIGNAL_STATE_CONTRACT_VERSION})
    library_frame = pd.DataFrame(library)
    write_csv(root / "candidate_library/candidate_library_update.csv", library_frame)
    report=f"""# BTC-Led Delayed Alt Diffusion Screen v1\n\nStatus: `{'complete' if mechanics else 'blocked_by_protocol_issue'}`. Final decision: `{final}`. The 12 frozen definitions used two parent-neutral PIT cross-sectional lag tapes and four PIT parent projections under `{SIGNAL_STATE_CONTRACT_VERSION}`. ALT, BTC and ETH completed two-hour bars were inner-aligned at identical decision timestamps; beta and BTC thresholds used shifted prior-60 windows. Raw/projected keys were frozen before outcomes; controls were frozen before control outcomes. No simultaneous breakout, momentum basket, prior-high, breakout-retest, session, sector, funding, or OI signal gate was used. No validation, CPCV, holdout, portfolio, or live phase was launched.\n"""; (root / "SCREEN_REPORT.md").write_text(report,encoding="utf-8")
    data_manifest=Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv"); funding_manifest=Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1/funding/shared_funding_panel_manifest.csv")
    write_json(root / "reproducibility/run_manifest.json",{"commit_hash":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"code_path":str(Path(__file__)),"code_hash":file_hash(Path(__file__)),"config_hash":file_hash(root/"manifest/definitions.csv"),"contract_hash":file_hash(contract_path),"data_snapshot_manifest_hash":file_hash(data_manifest),"funding_manifest_hash":file_hash(funding_manifest),"pit_universe_manifest_hash":file_hash(root/"manifest/pit_panel.csv"),"reference_root_hashes":source_hashes_before,"protected_boundary":PROTECTED.isoformat(),"signal_state_contract_version":SIGNAL_STATE_CONTRACT_VERSION,"seed_values":[],"contract_type":"Kraken PF perpetual instruments; linear perpetual cost outcomes in R units with OHLCV execution approximation"})
    peak_rss=max(peak_rss,runner.current_rss_bytes())
    decision={"run_root":str(root),"status":"complete" if mechanics else "blocked_by_protocol_issue","final_decision":final,**hard,"raw_signals":len(raw),"parent_projected_signals":len(projected),"accepted_trade_rows":len(outcomes),"non_overlap_skips":len(skips),"definition_outcome_exclusions":len(exclusions),"control_event_rows":len(controls),"raw_signal_freeze_hash":raw_freeze_hash,"projection_freeze_hash":projection_hash,"accepted_trade_freeze_hash":accepted_hash,"control_key_freeze_hash":control_freeze_hash,"peak_rss_bytes":peak_rss,"runtime_seconds":time.monotonic()-started,"materialization_candidates":decisions[decisions.decision.eq("materialization_candidate")].definition_id.tolist() if mechanics else [],"context_sleeves":decisions[decisions.decision.eq("fragile_context_sleeve")].definition_id.tolist() if mechanics else [],"validation_launched":False,"cpcv_launched":False,"holdout_launched":False,"portfolio_construction_launched":False,"live_work_launched":False,"compact_bundle_path":str(root/"compact_review_bundle")}
    write_json(root / "decision_summary.json", decision)
    update_central_artifacts(root, library_frame, final)
    shutil.copy2(Path(__file__), root / "reproducibility/executed_runner.py")
    write_artifact_hash_inventory(root)
    compact_bundle(root)
    write_json(root / "watch_status.json",{**decision,"stage":"complete","updated_ts":runner.utc_now()})
    return decision


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-root",type=Path,default=RUN_ROOT); args=parser.parse_args(); result=run(args.run_root); print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["status"]=="complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
