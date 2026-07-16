#!/usr/bin/env python3
"""Build point-in-time match features for real QLMG controls.

Features are derived from local 5m bars with fixed buckets and as-of joins using
source timestamps <= decision_ts. Missing data is explicit and never imputed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROTECTED_TS = pd.Timestamp("2026-01-01T00:00:00Z")
DEFAULT_BAR_ROOT = Path("/opt/parquet/5m")


def to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def fixed_vol_bucket(x: Any) -> str:
    if pd.isna(x):
        return "vol_missing"
    x = float(x)
    if x < 2.0:
        return "vol_low_lt2pct"
    if x < 5.0:
        return "vol_mid_2_5pct"
    if x < 10.0:
        return "vol_high_5_10pct"
    return "vol_extreme_gte10pct"


def fixed_liquidity_tier(turnover_24h: Any) -> str:
    if pd.isna(turnover_24h):
        return "liq_missing"
    x = float(turnover_24h)
    if x < 1_000_000:
        return "liq_low_lt1m_24h"
    if x < 10_000_000:
        return "liq_mid_1m_10m_24h"
    if x < 100_000_000:
        return "liq_high_10m_100m_24h"
    return "liq_ultra_gte100m_24h"


def fixed_funding_bucket(x: Any) -> str:
    if pd.isna(x):
        return "funding_missing"
    x = float(x)
    if x < -0.0001:
        return "funding_negative"
    if x <= 0.0001:
        return "funding_neutral"
    if x < 0.0005:
        return "funding_positive"
    return "funding_high_positive"


def fixed_oi_bucket(x: Any) -> str:
    if pd.isna(x):
        return "oi_missing"
    x = float(x)
    if x < -0.05:
        return "oi_down_gt5pct"
    if x <= 0.05:
        return "oi_flat_pm5pct"
    if x < 0.25:
        return "oi_up_5_25pct"
    return "oi_surge_gte25pct"


def load_symbol_feature_frame(symbol: str, bar_root: Path = DEFAULT_BAR_ROOT) -> pd.DataFrame:
    path = bar_root / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "feature_source_ts", "realized_vol_24h_pct", "turnover_24h", "funding_rate_pt", "oi_chg_24h"])
    cols = ["timestamp", "close", "turnover", "open_interest", "funding_rate"]
    try:
        raw = pd.read_parquet(path, columns=cols)
    except Exception:
        raw = pd.read_parquet(path)
        raw = raw[[c for c in cols if c in raw.columns]].copy()
    if raw.empty or "timestamp" not in raw.columns:
        return pd.DataFrame(columns=["symbol", "feature_source_ts", "realized_vol_24h_pct", "turnover_24h", "funding_rate_pt", "oi_chg_24h"])
    df = raw.copy()
    df["timestamp"] = to_utc(df["timestamp"])
    df = df[df["timestamp"].notna() & (df["timestamp"] < PROTECTED_TS)].sort_values("timestamp", kind="mergesort")
    if df.empty:
        return pd.DataFrame(columns=["symbol", "feature_source_ts", "realized_vol_24h_pct", "turnover_24h", "funding_rate_pt", "oi_chg_24h"])
    close = pd.to_numeric(df.get("close"), errors="coerce")
    ret = close.pct_change()
    # 288 five-minute bars = 24h. Shift by one bar so the current decision row is not used to define its own trailing state.
    df["realized_vol_24h_pct"] = (ret.rolling(288, min_periods=48).std() * np.sqrt(288) * 100.0).shift(1)
    df["turnover_24h"] = pd.to_numeric(df.get("turnover"), errors="coerce").rolling(288, min_periods=48).sum().shift(1)
    oi = pd.to_numeric(df.get("open_interest"), errors="coerce")
    df["oi_chg_24h"] = (oi / oi.shift(288) - 1.0).shift(1)
    df["funding_rate_pt"] = pd.to_numeric(df.get("funding_rate"), errors="coerce").shift(1)
    df["symbol"] = symbol
    df["feature_source_ts"] = df["timestamp"]
    keep = ["symbol", "feature_source_ts", "realized_vol_24h_pct", "turnover_24h", "funding_rate_pt", "oi_chg_24h"]
    return df[keep].dropna(subset=["feature_source_ts"]).copy()


def enrich_event_pool_with_match_features(pool: pd.DataFrame, bar_root: Path = DEFAULT_BAR_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pool.empty:
        return pool.copy(), pd.DataFrame()
    out_parts = []
    coverage_rows = []
    base = pool.copy()
    base["decision_ts"] = to_utc(base["decision_ts"])
    if bool((base["decision_ts"] >= PROTECTED_TS).any()):
        raise ValueError("protected decision_ts in event pool")
    for symbol, events in base.groupby("symbol", sort=False):
        events = events.sort_values("decision_ts", kind="mergesort").copy()
        feat = load_symbol_feature_frame(str(symbol), bar_root=bar_root)
        if feat.empty:
            enriched = events.copy()
            for c in ["feature_source_ts", "realized_vol_24h_pct", "turnover_24h", "funding_rate_pt", "oi_chg_24h"]:
                enriched[c] = pd.NaT if c == "feature_source_ts" else np.nan
        else:
            # As-of join ensures feature_source_ts <= decision_ts. allow_exact_matches=True follows repository bar timestamp convention.
            enriched = pd.merge_asof(
                events.sort_values("decision_ts", kind="mergesort"),
                feat.sort_values("feature_source_ts", kind="mergesort"),
                left_on="decision_ts",
                right_on="feature_source_ts",
                by="symbol",
                direction="backward",
                allow_exact_matches=True,
            )
        out_parts.append(enriched)
        n = len(enriched)
        coverage_rows.append({
            "symbol": symbol,
            "events": n,
            "bar_file_exists": (bar_root / f"{symbol}.parquet").exists(),
            "feature_source_coverage": float(enriched["feature_source_ts"].notna().mean()) if n else 0.0,
            "vol_coverage": float(pd.to_numeric(enriched.get("realized_vol_24h_pct"), errors="coerce").notna().mean()) if n else 0.0,
            "liquidity_coverage": float(pd.to_numeric(enriched.get("turnover_24h"), errors="coerce").notna().mean()) if n else 0.0,
            "funding_coverage": float(pd.to_numeric(enriched.get("funding_rate_pt"), errors="coerce").notna().mean()) if n else 0.0,
            "oi_coverage": float(pd.to_numeric(enriched.get("oi_chg_24h"), errors="coerce").notna().mean()) if n else 0.0,
        })
    out = pd.concat(out_parts, ignore_index=True) if out_parts else base
    out["volatility_bucket"] = out["realized_vol_24h_pct"].map(fixed_vol_bucket)
    out["liquidity_tier"] = out["turnover_24h"].map(fixed_liquidity_tier)
    out["funding_bucket"] = out["funding_rate_pt"].map(fixed_funding_bucket)
    out["oi_bucket"] = out["oi_chg_24h"].map(fixed_oi_bucket)
    out["match_feature_source"] = "local_5m_ohlcv_oi_funding_asof"
    out["match_feature_pit_ok"] = pd.to_datetime(out["feature_source_ts"], utc=True, errors="coerce") <= out["decision_ts"]
    out.loc[out["feature_source_ts"].isna(), "match_feature_pit_ok"] = False
    if bool((pd.to_datetime(out["feature_source_ts"], utc=True, errors="coerce") > out["decision_ts"]).fillna(False).any()):
        raise ValueError("feature source timestamp after decision_ts")
    return out, pd.DataFrame(coverage_rows)
