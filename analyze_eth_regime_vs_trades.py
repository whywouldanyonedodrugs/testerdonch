#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


# ----------------- Helpers ----------------- #

def _load_trades(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path, parse_dates=["entry_ts", "exit_ts"])
    if not df.empty:
        # ensure UTC (backtester usually writes UTC-like timestamps)
        df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    return df


def _load_eth_ohlcv() -> pd.DataFrame:
    parq_dir = getattr(cfg, "PARQUET_DIR", Path("/opt/parquet/5m"))
    regime_asset = getattr(cfg, "REGIME_ASSET", "ETHUSDT")
    path = Path(parq_dir) / f"{regime_asset}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"ETH parquet not found at {path}")

    eth = pd.read_parquet(path)

    # Case 1: has a 'timestamp' column
    if "timestamp" in eth.columns:
        eth = eth.copy()
        eth["timestamp"] = pd.to_datetime(eth["timestamp"], utc=True, errors="coerce")
        eth = eth.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
        return eth

    # Case 2: no 'timestamp' column, but index is datetime (your current case)
    if isinstance(eth.index, pd.DatetimeIndex):
        eth = eth.copy()
        eth.index = pd.to_datetime(eth.index, utc=True, errors="coerce")
        eth = eth[~eth.index.isna()]
        eth.index.name = "timestamp"
        return eth

    # Otherwise we really don't know how to interpret this parquet
    raise ValueError(
        "ETH parquet must have either a 'timestamp' column or a DatetimeIndex."
    )



def _compute_eth_macd_4h(eth: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 4h MACD line, signal, histogram for ETH, using cfg.REGIME_* params.
    Returns a DataFrame indexed by 4h timestamps with columns:
      ['close_4h', 'macd_line', 'macd_signal', 'macd_hist', 'macd_hist_prev']
    """
    fast = int(getattr(cfg, "REGIME_MACD_FAST", 12))
    slow = int(getattr(cfg, "REGIME_MACD_SLOW", 26))
    signal = int(getattr(cfg, "REGIME_MACD_SIGNAL", 9))

    close_4h = (
        eth["close"].astype(float)
        .resample("4h", label="right", closed="right")
        .last()
        .dropna()
    )

    if close_4h.empty:
        raise ValueError("No 4h close data for ETH; cannot compute MACD")

    ema_fast = close_4h.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_4h.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - macd_signal

    out = pd.DataFrame(
        {
            "close_4h": close_4h,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
        }
    )

    out["macd_hist_prev"] = out["macd_hist"].shift(1)
    out = out.dropna(subset=["macd_hist"])  # first few will be NaN
    return out


@dataclass
class BucketParams:
    H_weak: float
    H_strong: float


def _choose_hist_thresholds(macd_df: pd.DataFrame) -> BucketParams:
    """
    Choose thresholds for hist magnitude from ETH MACD history itself, ex-ante style.
    We use simple quantiles of |hist|:
      H_weak  ~ 60th percentile of |hist|
      H_strong ~ 85th percentile of |hist|
    """
    hist_abs = macd_df["macd_hist"].abs().dropna()
    if hist_abs.empty:
        return BucketParams(H_weak=0.0, H_strong=0.0)
    H_weak = float(hist_abs.quantile(0.60))
    H_strong = float(hist_abs.quantile(0.85))
    return BucketParams(H_weak=H_weak, H_strong=H_strong)


def _bucket_hist(h: float, params: BucketParams) -> str:
    if not np.isfinite(h):
        return "nan"
    if h <= -params.H_strong:
        return "strong_neg"
    if -params.H_strong < h < 0:
        return "weak_neg"
    if 0 <= h < params.H_weak:
        return "weak_pos"
    if h >= params.H_weak:
        return "strong_pos"
    # fallback
    return "other"


def _bucket_slope(h_now: float, h_prev: float) -> str:
    if not (np.isfinite(h_now) and np.isfinite(h_prev)):
        return "nan"
    delta = h_now - h_prev
    if delta > 0:
        return "slope_up"
    if delta < 0:
        return "slope_down"
    return "slope_flat"


def _merge_trades_with_macd(trades: pd.DataFrame, macd_4h: pd.DataFrame) -> pd.DataFrame:
    """
    As-of merge 4h MACD context onto trades using entry_ts.
    trade.entry_ts (UTC) -> latest 4h bar <= entry_ts
    """
    if trades.empty:
        return trades.copy()

    macd = macd_4h.reset_index().rename(columns={"index": "ts_4h"})
    macd = macd.rename(columns={"timestamp": "ts_4h"}) if "timestamp" in macd.columns else macd
    macd["ts_4h"] = pd.to_datetime(macd["ts_4h"], utc=True, errors="coerce")
    macd = macd.dropna(subset=["ts_4h"]).sort_values("ts_4h")

    tr = trades.copy()
    tr["entry_ts"] = pd.to_datetime(tr["entry_ts"], utc=True, errors="coerce")
    tr = tr.sort_values("entry_ts")

    merged = pd.merge_asof(
        tr,
        macd,
        left_on="entry_ts",
        right_on="ts_4h",
        direction="backward",
    )
    return merged


def _summarize_by_bucket(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    """
    Produce a summary per bucket:
      trades, avg_pnl_R, med_pnl_R, win_rate, total_pnl, avg_pnl, big_loss_frac
    """
    if df.empty:
        return pd.DataFrame()

    def _row(g: pd.DataFrame) -> pd.Series:
        g = g.copy()
        g["win"] = (g["pnl"] > 0).astype(int)
        trades = len(g)
        avg_R = g["pnl_R"].mean() if trades else np.nan
        med_R = g["pnl_R"].median() if trades else np.nan
        win_rate = g["win"].mean() if trades else np.nan
        total_pnl = g["pnl"].sum() if trades else np.nan
        avg_pnl = g["pnl"].mean() if trades else np.nan
        big_loss_frac = (g["pnl_R"] <= -2.0).mean() if trades else np.nan
        return pd.Series(
            {
                "trades": trades,
                "avg_pnl_R": avg_R,
                "med_pnl_R": med_R,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl": avg_pnl,
                "big_loss_frac_R<=-2": big_loss_frac,
            }
        )

    out = (
        df.groupby(bucket_col, dropna=False)
        .apply(_row, include_groups=False)
        .reset_index()
        .sort_values(bucket_col)
    )
    return out


# ----------------- Main analysis ----------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Analyze ETH 4h MACD histogram regime vs trade outcomes."
    )
    ap.add_argument(
        "--results-dir",
        default="results",
        help="Directory with trades.csv (default: results)",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    print(f"[info] Loading trades from {results_dir}/trades.csv ...")
    trades = _load_trades(results_dir)
    print(f"[info] Loaded {len(trades)} trades")

    if trades.empty:
        print("[warn] No trades; nothing to analyze.")
        return

    print("[info] Loading ETH OHLCV and computing 4h MACD...")
    eth = _load_eth_ohlcv()
    macd_4h = _compute_eth_macd_4h(eth)

    print("[info] Basic ETH 4h MACD stats:")
    print(macd_4h["macd_hist"].describe())

    params = _choose_hist_thresholds(macd_4h)
    print("\n[info] Histogram bucket thresholds (absolute values):")
    print(f"  H_weak   (|hist| 60th percentile) ≈ {params.H_weak:.4f}")
    print(f"  H_strong (|hist| 85th percentile) ≈ {params.H_strong:.4f}")

    merged = _merge_trades_with_macd(trades, macd_4h)

    # Drop trades where we failed to attach MACD context
    attached = merged[merged["macd_hist"].notna()].copy()
    n_attached = len(attached)
    print(f"\n[info] Attached MACD context to {n_attached} / {len(trades)} trades")

    if attached.empty:
        print("[warn] No trades with MACD context; aborting.")
        return

    # Buckets at entry
    attached["hist_bucket"] = [
        _bucket_hist(h, params) for h in attached["macd_hist"].values
    ]
    attached["slope_bucket"] = [
        _bucket_slope(h_now, h_prev)
        for h_now, h_prev in zip(attached["macd_hist"].values, attached["macd_hist_prev"].values)
    ]

    # --- Summary 1: by hist_bucket ---
    print("\n=== Trade stats by ETH 4h MACD histogram bucket at entry ===")
    summary_hist = _summarize_by_bucket(attached, "hist_bucket")
    if summary_hist.empty:
        print("No data.")
    else:
        print(summary_hist.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    # --- Summary 2: by slope_bucket ---
    print("\n=== Trade stats by ETH 4h MACD histogram slope bucket at entry ===")
    summary_slope = _summarize_by_bucket(attached, "slope_bucket")
    if summary_slope.empty:
        print("No data.")
    else:
        print(summary_slope.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    # --- Summary 3: by (hist_bucket, slope_bucket) combo ---
    print("\n=== Trade stats by (hist_bucket, slope_bucket) combo at entry ===")
    if not attached.empty:
        combo = attached.copy()
        combo["combo"] = combo["hist_bucket"].astype(str) + "|" + combo["slope_bucket"].astype(str)
        summary_combo = _summarize_by_bucket(combo, "combo")
        if summary_combo.empty:
            print("No data.")
        else:
            print(summary_combo.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

    print("\n[done] Regime vs trades analysis complete.")


if __name__ == "__main__":
    main()
