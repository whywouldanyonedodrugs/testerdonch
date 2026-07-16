#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

FINAL_HOLDOUT_START = pd.Timestamp("2026-01-01T00:00:00Z")
SCREENING_END = pd.Timestamp("2025-12-31T23:59:59Z")

REGIME_LAYERS = [
    {
        "layer": "parent_market_participation",
        "features": ["btc_trend_label", "eth_trend_label", "btc_drawdown_bucket", "breadth_label", "alt_rs_label", "breakout_participation_label"],
        "status": "implemented_from_available_btc_eth_event_context_with_breadth_placeholders",
    },
    {
        "layer": "volatility_liquidity_quality",
        "features": ["realized_vol_bucket", "atr_bucket", "vol_of_vol_bucket", "liquidity_quality_label", "zero_turnover_label", "bad_wick_proxy_label"],
        "status": "implemented_from_trailing_event_range_atr_turnover_proxy",
    },
    {
        "layer": "derivatives_state",
        "features": ["price_oi_matrix_24h", "funding_percentile_bucket", "funding_sign_persistence_label", "premium_mark_stretch_label", "crowding_label"],
        "status": "implemented_where_oi_funding_or_context_proxy_exists",
    },
    {
        "layer": "deleveraging_liquidation_reset",
        "features": ["deleveraged_2of4", "deleveraged_3of4", "oi_collapse_label", "funding_reset_label", "post_flush_reclaim_proxy", "liquidation_proxy_label"],
        "status": "implemented_from_decision_time_price_oi_funding_proxy_only_no_future_path_reclaim",
    },
    {
        "layer": "session_stress_lifecycle",
        "features": ["weekend_flag", "utc_0000_window", "funding_window", "session_bucket", "listing_age_bucket", "data_integrity_label"],
        "status": "implemented_from_timestamp_and_proxy_listing_age",
    },
    {
        "layer": "sector_catalyst_lifecycle",
        "features": ["sector_label", "catalyst_label", "sector_catalyst_status"],
        "status": "schema_only_blocked_no_point_in_time_sector_catalyst_store_found",
    },
]


def stable_hash(obj: Any, n: int = 12) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:n]


def to_utc_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def validate_no_protected(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        vals = to_utc_series(df[col]).dropna()
        if not vals.empty and bool((vals >= FINAL_HOLDOUT_START).any()):
            raise RuntimeError(f"protected timestamp detected in {col}")


def trailing_percentile(values: pd.Series, window: int = 252, min_periods: int = 20) -> pd.Series:
    """Trailing percentile of each value using only rows up to and including that row."""
    s = pd.to_numeric(values, errors="coerce")

    def pct(arr: np.ndarray) -> float:
        cur = arr[-1]
        hist = arr[np.isfinite(arr)]
        if not np.isfinite(cur) or len(hist) < min_periods:
            return np.nan
        return float((hist <= cur).sum() / len(hist))

    return s.rolling(window=window, min_periods=min_periods).apply(pct, raw=True)


def bucket_quantile(x: Any, low: float = 0.33, high: float = 0.66) -> str:
    try:
        v = float(x)
    except Exception:
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    if v < low:
        return "low"
    if v > high:
        return "high"
    return "mid"


def signed_bucket(x: Any, pos: float = 0.0, neg: float = 0.0) -> str:
    try:
        v = float(x)
    except Exception:
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    if v > pos:
        return "positive"
    if v < neg:
        return "negative"
    return "flat"


def parent_trend_label(btc4: Any, btc24: Any, eth4: Any, eth24: Any) -> str:
    vals = []
    for v in (btc4, btc24, eth4, eth24):
        try:
            fv = float(v)
            if np.isfinite(fv):
                vals.append(fv)
        except Exception:
            pass
    if len(vals) < 2:
        return "unknown"
    strong = sum(v > 0.005 for v in vals)
    weak = sum(v < -0.005 for v in vals)
    if strong >= 3:
        return "strong_up"
    if weak >= 3:
        return "down"
    if sum(v >= 0 for v in vals) >= 3:
        return "neutral_up"
    return "neutral_down"


def btc_eth_regime_label(btc4: Any, btc24: Any, eth4: Any, eth24: Any) -> str:
    btc_bad = pd.notna(btc4) and float(btc4) < -0.005 or pd.notna(btc24) and float(btc24) < -0.015
    eth_bad = pd.notna(eth4) and float(eth4) < -0.005 or pd.notna(eth24) and float(eth24) < -0.015
    if btc_bad and eth_bad:
        return "both_deteriorating"
    if btc_bad or eth_bad:
        return "one_deteriorating"
    return "non_deteriorating"


def session_bucket(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "unknown"
    h = int(pd.Timestamp(ts).hour)
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 13:
        return "eu"
    if 13 <= h < 21:
        return "us"
    return "late_us_asia_overlap"


def listing_age_bucket(age: Any) -> str:
    try:
        v = float(age)
    except Exception:
        return "unknown"
    if not np.isfinite(v):
        return "unknown"
    if v <= 3:
        return "0_3d"
    if v <= 14:
        return "4_14d"
    if v <= 30:
        return "15_30d"
    return "gt30d"


def price_oi_matrix(ret: Any, oi: Any) -> str:
    r = np.nan
    o = np.nan
    try:
        r = float(ret)
    except Exception:
        pass
    try:
        o = float(oi)
    except Exception:
        pass
    if not np.isfinite(r) or not np.isfinite(o):
        return "unknown"
    return ("price_up" if r >= 0 else "price_down") + "_" + ("oi_up" if o >= 0 else "oi_down")


def build_regime_panel(source: pd.DataFrame, *, min_history: int = 10) -> pd.DataFrame:
    """Build point-in-time regime labels from existing event/path rows.

    The function intentionally uses only columns already present at decision time in the
    existing QLMG event/path ledgers. It does not infer sector/catalyst data or true
    order-book/liquidation fields that are unavailable locally.
    """
    if source.empty:
        return pd.DataFrame()
    df = source.copy()
    if "decision_ts" not in df.columns:
        raise ValueError("decision_ts required")
    df["decision_ts"] = to_utc_series(df["decision_ts"])
    validate_no_protected(df, ["decision_ts", "entry_ts"])
    sort_cols = [c for c in ["symbol", "decision_ts", "event_id"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    out = pd.DataFrame()
    for col in ["event_id", "family", "variant_id", "symbol", "side", "liquidity_tier", "decision_ts"]:
        out[col] = df[col] if col in df.columns else np.nan
    out["feature_ts"] = df["decision_ts"]

    missing_series = pd.Series(np.nan, index=df.index)
    btc4 = df.get("btc_ret_4h", missing_series)
    btc24 = df.get("btc_ret_24h", missing_series)
    eth4 = df.get("eth_ret_4h", missing_series)
    eth24 = df.get("eth_ret_24h", missing_series)
    out["parent_trend_label"] = [parent_trend_label(a, b, c, d) for a, b, c, d in zip(btc4, btc24, eth4, eth24)]
    out["btc_eth_non_deteriorating"] = [btc_eth_regime_label(a, b, c, d) == "non_deteriorating" for a, b, c, d in zip(btc4, btc24, eth4, eth24)]
    out["btc_eth_regime_label"] = [btc_eth_regime_label(a, b, c, d) for a, b, c, d in zip(btc4, btc24, eth4, eth24)]
    if "btc_eth_regime" in df.columns:
        raw_regime = df["btc_eth_regime"].fillna("unknown").astype(str)
        out.loc[out["btc_eth_regime_label"].eq("unknown"), "btc_eth_regime_label"] = raw_regime
        out.loc[out["parent_trend_label"].eq("unknown") & raw_regime.eq("both_positive"), "parent_trend_label"] = "strong_up"
        out.loc[out["parent_trend_label"].eq("unknown") & raw_regime.eq("both_negative"), "parent_trend_label"] = "down"
        out.loc[out["parent_trend_label"].eq("unknown") & raw_regime.ne("unknown"), "parent_trend_label"] = "neutral_up"
        out.loc[raw_regime.eq("both_negative"), "btc_eth_non_deteriorating"] = False
    out["btc_drawdown_bucket"] = "unknown_no_63d_btc_drawdown_in_event_ledger"
    out["breadth_label"] = "unknown_no_full_breadth_panel"
    out["alt_rs_label"] = "unknown_no_alt_basket_panel"
    out["breakout_participation_label"] = "unknown_no_breakout_participation_panel"

    # Event-local volatility/liquidity proxies, with trailing percentile per symbol.
    for src, dst, window in [("range_pct", "realized_vol_pct", 252), ("atr_bps", "atr_pct", 252), ("turnover", "turnover_pct", 252)]:
        if src in df.columns:
            out[dst] = df.groupby("symbol", group_keys=False)[src].apply(lambda s: trailing_percentile(s, window=window, min_periods=min_history))
        else:
            out[dst] = np.nan
    out["realized_vol_bucket"] = out["realized_vol_pct"].map(bucket_quantile)
    out["atr_bucket"] = out["atr_pct"].map(bucket_quantile)
    out["vol_of_vol_bucket"] = "unknown_no_full_vol_of_vol_panel"
    out["turnover_bucket"] = out["turnover_pct"].map(bucket_quantile)
    out["liquidity_quality_label"] = np.where(out["turnover_bucket"].eq("low"), "thin_proxy", "normal_or_unknown")
    turnover = pd.to_numeric(df.get("turnover", np.nan), errors="coerce")
    out["zero_turnover_label"] = np.where(turnover <= 0, "zero_turnover", "nonzero_or_unknown")
    range_pct = pd.to_numeric(df.get("range_pct", np.nan), errors="coerce")
    if "range_pct" in df.columns:
        wick_pct = df.groupby("symbol", group_keys=False)["range_pct"].apply(lambda s: trailing_percentile(s, window=252, min_periods=min_history))
        out["bad_wick_proxy_label"] = np.where(wick_pct > 0.99, "trailing_top_1pct_range_proxy", np.where(wick_pct.isna(), "unknown_insufficient_trailing_history", "normal_or_unknown"))
    else:
        out["bad_wick_proxy_label"] = "unknown_no_range_pct"

    ret24 = df.get("24h_close_return_bps", missing_series)
    if "ret_24h" in df.columns:
        ret24 = pd.to_numeric(df["ret_24h"], errors="coerce")
    out["price_oi_matrix_24h"] = [price_oi_matrix(r, o) for r, o in zip(ret24, df.get("oi_chg_24h", missing_series))]
    if "funding_rate" in df.columns:
        out["funding_pct"] = df.groupby("symbol", group_keys=False)["funding_rate"].apply(lambda s: trailing_percentile(s, window=252, min_periods=min_history))
    else:
        out["funding_pct"] = np.nan
    out["funding_percentile_bucket"] = out["funding_pct"].map(bucket_quantile)
    funding_rate = pd.to_numeric(df.get("funding_rate", np.nan), errors="coerce")
    out["funding_sign_label"] = funding_rate.map(lambda v: signed_bucket(v, 0.0, 0.0))
    out["funding_sign_persistence_label"] = "unknown_no_funding_window_history"
    gap = pd.to_numeric(df.get("mark_last_gap", missing_series), errors="coerce")
    out["premium_mark_stretch_label"] = pd.cut(gap.abs(), bins=[-np.inf, 0.001, 0.005, np.inf], labels=["low", "mid", "high"]).astype(str).replace("nan", "unknown")
    out["crowding_label"] = np.select(
        [out["funding_percentile_bucket"].eq("high") & out["price_oi_matrix_24h"].str.contains("oi_up", na=False), out["funding_percentile_bucket"].eq("low") & out["price_oi_matrix_24h"].str.contains("oi_up", na=False)],
        ["crowded_long_proxy", "crowded_short_proxy"],
        default="not_crowded_or_unknown",
    )

    oi = pd.to_numeric(df.get("oi_chg_24h", np.nan), errors="coerce")
    price_down = pd.to_numeric(df.get("ret_24h", np.nan), errors="coerce") < -0.05 if "ret_24h" in df.columns else pd.Series(False, index=df.index)
    oi_down = oi < -0.03
    funding_reset = out["funding_percentile_bucket"].isin(["low", "mid"])
    # Do not use future path fields such as 24h_mfe_bps/24h_mae_bps in a
    # decision-time regime label. Older artifacts used a post-flush reclaim
    # proxy from those fields and must be quarantined for ranking.
    reclaim = pd.Series(False, index=df.index)
    components = pd.concat([price_down.rename("price_down"), oi_down.rename("oi_down"), funding_reset.rename("funding_reset")], axis=1).fillna(False)
    out["deleveraged_component_count"] = components.astype(int).sum(axis=1)
    out["deleveraged_2of4"] = out["deleveraged_component_count"] >= 2
    out["deleveraged_3of4"] = out["deleveraged_component_count"] >= 3
    out["deleveraged_component_contract"] = "safe_3_components_price_oi_funding_no_future_reclaim"
    out["oi_collapse_label"] = np.where(oi < -0.08, "large_oi_down", np.where(oi < -0.03, "moderate_oi_down", "not_oi_down_or_unknown"))
    out["funding_reset_label"] = np.where(funding_reset, "funding_normalized_or_low", "funding_high_or_unknown")
    out["post_flush_reclaim_proxy"] = reclaim.fillna(False)
    out["post_flush_reclaim_proxy_status"] = "disabled_future_path_label_not_decision_time_safe"
    liq_cols = [c for c in df.columns if c.endswith("liquidation_10x")]
    if liq_cols:
        liq_any = df[liq_cols].fillna(False).astype(bool).any(axis=1)
        out["liquidation_proxy_label"] = np.where(liq_any, "proxy_liquidation_seen", "no_proxy_liquidation")
    else:
        out["liquidation_proxy_label"] = "unknown_no_liquidation_columns"

    ts = out["decision_ts"]
    out["weekend_flag"] = ts.dt.dayofweek >= 5
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    out["utc_0000_window"] = np.select([minute_of_day <= 30, minute_of_day >= 1410, minute_of_day <= 60, minute_of_day >= 1380], ["within_30m", "within_30m", "within_60m", "within_60m"], default="outside_60m")
    out["funding_window"] = np.where(ts.dt.hour.isin([0, 8, 16]) & (ts.dt.minute <= 30), "funding_window_proxy", "outside_funding_window")
    out["session_bucket"] = ts.map(session_bucket)
    out["listing_age_bucket"] = df.get("listing_age_proxy_days", pd.Series(np.nan, index=df.index)).map(listing_age_bucket)
    dq = df.get("data_quality_flags", pd.Series("", index=df.index)).fillna("").astype(str)
    out["data_integrity_label"] = np.where(dq.str.len() > 0, "flagged", "clean_or_unflagged")
    out["sector_label"] = "unknown_no_pit_sector_store"
    out["catalyst_label"] = "unknown_no_pit_catalyst_store"
    out["sector_catalyst_status"] = "blocked_no_local_point_in_time_store"

    out["regime_row_hash"] = [stable_hash(r) for r in out[["event_id", "symbol", "decision_ts", "parent_trend_label", "price_oi_matrix_24h", "deleveraged_component_count"]].to_dict("records")]
    validate_no_protected(out, ["decision_ts", "feature_ts"])
    return out


def regime_feature_dictionary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for layer in REGIME_LAYERS:
        for feat in layer["features"]:
            rows.append({"layer": layer["layer"], "feature": feat, "status": layer["status"], "pit_rule": "feature_ts <= decision_ts; trailing-only where computed"})
    return pd.DataFrame(rows)


def join_regime(events: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    if regime.empty:
        out = events.copy()
        out["regime_join_status"] = "missing_regime_panel"
        return out
    keys = ["event_id"] if "event_id" in events.columns and "event_id" in regime.columns else ["symbol", "decision_ts"]
    reg_cols = [c for c in regime.columns if c not in set(events.columns) or c in keys]
    out = events.merge(regime[reg_cols], on=keys, how="left", suffixes=("", "_regime"))
    out["regime_join_status"] = np.where(out.get("parent_trend_label").isna(), "missing_regime", "joined")
    return out


def label_overlap_matrix(regime: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in regime.columns]
    rows: list[dict[str, Any]] = []
    for a in cols:
        for b in cols:
            if a == b:
                score = 1.0
            else:
                sa = regime[a].astype(str).fillna("unknown")
                sb = regime[b].astype(str).fillna("unknown")
                score = float((sa == sb).mean()) if len(regime) else np.nan
            rows.append({"label_a": a, "label_b": b, "same_label_share": score})
    return pd.DataFrame(rows)
