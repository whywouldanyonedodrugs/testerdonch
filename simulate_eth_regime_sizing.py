#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


def _load_trades(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"trades.csv not found in {results_dir}")
    df = pd.read_csv(path, parse_dates=["entry_ts", "exit_ts"])
    if df.empty:
        raise ValueError("trades.csv is empty")
    return df


def _load_eth_ohlcv() -> pd.DataFrame:
    eth_path = cfg.PARQUET_DIR / f"{getattr(cfg, 'REGIME_ASSET', 'ETHUSDT')}.parquet"
    if not eth_path.exists():
        raise FileNotFoundError(f"ETH parquet not found at {eth_path}")

    eth = pd.read_parquet(eth_path)

    # Case 1: there is an explicit 'timestamp' column
    if "timestamp" in eth.columns:
        eth = eth.copy()
        eth["timestamp"] = pd.to_datetime(eth["timestamp"], utc=True, errors="coerce")
        eth = eth.dropna(subset=["timestamp"]).sort_values("timestamp")
        return eth

    # Case 2: 'timestamp' is actually the index (your current case)
    if isinstance(eth.index, pd.DatetimeIndex):
        eth = eth.copy()
        eth.index = pd.to_datetime(eth.index, utc=True, errors="coerce")
        eth = eth[~eth.index.isna()]
        eth.index.name = "timestamp"
        eth = eth.reset_index()
        return eth

    # Otherwise we really don't know how to interpret this parquet
    raise ValueError(
        "ETH parquet must have either a 'timestamp' column or a DatetimeIndex."
    )



def _compute_macd_hist_4h(eth: pd.DataFrame) -> pd.DataFrame:
    eth = eth.copy()
    eth = eth.set_index("timestamp")

    close_4h = (
        eth["close"].astype(float)
        .resample("4h", label="right", closed="right")
        .last()
        .dropna()
    )
    if close_4h.empty:
        raise ValueError("No 4H closes available for ETH")

    # Standard MACD(12,26,9) using config where available
    fast = int(getattr(cfg, "REGIME_MACD_FAST", 12))
    slow = int(getattr(cfg, "REGIME_MACD_SLOW", 26))
    sig  = int(getattr(cfg, "REGIME_MACD_SIGNAL", 9))

    ema_fast = close_4h.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_4h.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=sig, adjust=False, min_periods=sig).mean()
    macd_hist = macd_line - macd_signal

    df = pd.DataFrame({"macd_hist": macd_hist})
    df["macd_hist_slope"] = df["macd_hist"].diff()
    return df.dropna()


def _attach_macd_to_trades(tr: pd.DataFrame, macd4h: pd.DataFrame) -> pd.DataFrame:
    """As-of merge: each trade gets the latest 4h MACD bar known at entry_ts (no lookahead)."""
    tr = tr.copy().sort_values("entry_ts")
    macd4h = macd4h.sort_index()

    merged = pd.merge_asof(
        tr,
        macd4h,
        left_on="entry_ts",
        right_index=True,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged


def _bucket_hist_and_slope(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df["macd_hist"].isna().all():
        raise ValueError("All macd_hist values are NaN on trades; check ETH data / merge logic.")

    # Absolute-value buckets for histogram magnitude
    abs_hist = df["macd_hist"].abs().dropna()
    h_weak = abs_hist.quantile(0.60)
    h_strong = abs_hist.quantile(0.85)

    print("[info] Histogram bucket thresholds (absolute values):")
    print(f"  H_weak   (|hist| 60th percentile) ≈ {h_weak:.4f}")
    print(f"  H_strong (|hist| 85th percentile) ≈ {h_strong:.4f}")

    def hist_bucket(x: float) -> str:
        if not np.isfinite(x):
            return "nan"
        ax = abs(x)
        if ax >= h_strong:
            return "strong_pos" if x > 0 else "strong_neg"
        if ax >= h_weak:
            return "weak_pos" if x > 0 else "weak_neg"
        # below 60th percentile, still treat as "weak_*"
        return "weak_pos" if x > 0 else "weak_neg"

    df["hist_bucket"] = df["macd_hist"].map(hist_bucket)

    # Slope buckets
    def slope_bucket(v: float) -> str:
        if not np.isfinite(v):
            return "nan"
        return "slope_up" if v >= 0 else "slope_down"

    df["slope_bucket"] = df["macd_hist_slope"].map(slope_bucket)
    return df


# === Scenario definitions: (hist_bucket, slope_bucket) -> size_mult ===

def _scenario_baseline(hist_bucket: str, slope_bucket: str) -> float:
    return 1.0


def _scenario_no_strong(hist_bucket: str, slope_bucket: str) -> float:
    # Skip extreme momentum
    return 0.0 if hist_bucket in ("strong_pos", "strong_neg") else 1.0


def _scenario_only_slope_up(hist_bucket: str, slope_bucket: str) -> float:
    # Trade only when MACD histogram slope ≥ 0
    return 1.0 if slope_bucket == "slope_up" else 0.0


def _scenario_weak_and_slope_up(hist_bucket: str, slope_bucket: str) -> float:
    # "Sweet spot" only: moderate magnitude + improving slope
    return 1.0 if (hist_bucket in ("weak_pos", "weak_neg") and slope_bucket == "slope_up") else 0.0


def _scenario_gradual(hist_bucket: str, slope_bucket: str) -> float:
    """
    Smooth dynamic sizing:
      - strong_pos: heavy downsize (0.25x)
      - strong_neg: mild downsize, especially if slope_up
      - weak_* & slope_up: size_up (1.5x)
      - weak_* & slope_down: slight downsize (0.75x)
    """
    if hist_bucket == "strong_pos":
        return 0.25
    if hist_bucket == "strong_neg":
        return 0.75 if slope_bucket == "slope_up" else 0.5
    if hist_bucket in ("weak_pos", "weak_neg"):
        if slope_bucket == "slope_up":
            return 1.5
        else:
            return 0.75
    return 1.0


SCENARIOS = {
    "baseline": _scenario_baseline,
    "no_strong": _scenario_no_strong,
    "only_slope_up": _scenario_only_slope_up,
    "weak_and_slope_up": _scenario_weak_and_slope_up,
    "gradual_sizing": _scenario_gradual,
}


def _run_scenario(df: pd.DataFrame, scenario_name: str, fn) -> None:
    df = df.copy()

    size = df.apply(lambda row: fn(row["hist_bucket"], row["slope_bucket"]), axis=1)
    df["size_mult"] = size.astype(float)

    df["adj_pnl"] = df["pnl"] * df["size_mult"]
    df["adj_pnl_R"] = df["pnl_R"] * df["size_mult"]

    active = df[df["size_mult"] > 0]

    trades = len(active)
    if trades == 0:
        print(f"\n=== Scenario: {scenario_name} ===")
        print("No trades survive this filter.")
        return

    gp = active.loc[active["adj_pnl"] > 0, "adj_pnl"].sum()
    gl = -active.loc[active["adj_pnl"] < 0, "adj_pnl"].sum()
    pf = float(gp / gl) if gl > 0 else np.inf

    win_rate = float((active["adj_pnl"] > 0).mean())
    avg_R = float(active["adj_pnl_R"].mean())
    avg_pnl = float(active["adj_pnl"].mean())
    total_pnl = float(active["adj_pnl"].sum())

    print(f"\n=== Scenario: {scenario_name} ===")
    print(f"Trades kept      : {trades} / {len(df)}")
    print(f"Total adj PnL    : {total_pnl:8.2f}")
    print(f"Avg PnL / trade  : {avg_pnl:8.3f}")
    print(f"Avg R / trade    : {avg_R:8.3f}")
    print(f"Win rate         : {win_rate:6.1%}")
    print(f"Profit factor    : {pf:8.2f}")


def main():
    ap = argparse.ArgumentParser(
        description="Simulate ETH 4h MACD-based regime filters/sizing on existing trades.csv"
    )
    ap.add_argument(
        "--results-dir",
        type=str,
        default=str(cfg.RESULTS_DIR),
        help="Directory with trades.csv (default: config.RESULTS_DIR)",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    print("[info] Loading trades from", results_dir / "trades.csv", "...")
    trades = _load_trades(results_dir)
    print(f"[info] Loaded {len(trades)} trades")

    print("[info] Loading ETH OHLCV and computing 4h MACD...")
    eth = _load_eth_ohlcv()
    macd4h = _compute_macd_hist_4h(eth)

    print("[info] Attaching MACD context to trades...")
    trades_ctx = _attach_macd_to_trades(trades, macd4h)
    trades_ctx = _bucket_hist_and_slope(trades_ctx)

    # sanity: show small table
    print("\n[info] Example trades with MACD context:")
    print(
        trades_ctx[
            ["entry_ts", "symbol", "pnl", "pnl_R",
             "macd_hist", "macd_hist_slope", "hist_bucket", "slope_bucket"]
        ].head()
    )

    # Run scenarios
    for name, fn in SCENARIOS.items():
        _run_scenario(trades_ctx, name, fn)


if __name__ == "__main__":
    main()
