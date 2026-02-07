import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from shared_utils import load_parquet_data
from indicators import resample_ohlcv, atr


def compute_entry_quality_panel(df5: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by 5m timestamps with the 4 entry-quality columns.
    Uses past-only computations (no look-ahead).
    """
    df5 = df5.sort_index()
    idx = df5.index

    # Ensure tz-aware UTC index for consistent alignment
    if idx.tz is None:
        df5 = df5.copy()
        df5.index = pd.to_datetime(df5.index, utc=True, errors="coerce")
    else:
        df5 = df5.copy()
        df5.index = df5.index.tz_convert("UTC")
    df5 = df5[~df5.index.isna()]
    df5 = df5.sort_index()
    idx = df5.index


    # infer bars/day from spacing, but default to 288 for 5m data
    if len(idx) >= 2:
        step_min = int(round((idx[1] - idx[0]).total_seconds() / 60.0))
        bars_per_day = max(1, int(round(24 * 60 / max(1, step_min))))
    else:
        bars_per_day = 288

    # ATR(1h) mapped to 5m
    df1h = resample_ohlcv(df5, "1h")
    atr1h = atr(df1h, int(getattr(cfg, "ATR_LEN", 14)))
    atr1h_5m = atr1h.reindex(idx, method="ffill")

    # days_since_prev_break vs daily Donch upper
    N = int(getattr(cfg, "DON_N_DAYS", 20))
    daily_high = df5["high"].resample("1D").max().dropna()
    don_daily = daily_high.rolling(N, min_periods=N).max().shift(1)
    don_5m = don_daily.reindex(idx, method="ffill")

    touch = (df5["high"] >= don_5m)
    touch_time = pd.Series(idx.where(touch), index=idx)
    last_touch = touch_time.ffill()

    days_since_prev_break = (idx.to_series() - last_touch).dt.total_seconds() / 86400.0
    days_since_prev_break = days_since_prev_break.replace([np.inf, -np.inf], np.nan).astype(float)

    # consolidation_range_atr over pullback window
    win_bars = int(getattr(cfg, "PULLBACK_WINDOW_BARS", 12) or 0)
    if win_bars <= 0:
        hours = float(getattr(cfg, "PULLBACK_WINDOW_HOURS", 24) or 24)
        win_bars = max(1, int(round(hours * 60.0 / 5.0)))

    cons_range = (df5["high"].rolling(win_bars).max() - df5["low"].rolling(win_bars).min())
    consolidation_range_atr = (cons_range / atr1h_5m.replace(0, np.nan)).astype(float)

    # prior_1d_ret
    prior_1d_ret = (df5["close"] / df5["close"].shift(bars_per_day) - 1.0).astype(float)

    # rv_3d
    logret = np.log(df5["close"]).diff()
    rv_3d = logret.rolling(3 * bars_per_day).std().astype(float)

    out = pd.DataFrame(
        {
            "days_since_prev_break": days_since_prev_break.values,
            "consolidation_range_atr": consolidation_range_atr.values,
            "prior_1d_ret": prior_1d_ret.values,
            "rv_3d": rv_3d.values,
        },
        index=idx,
    )
    out.index.name = "timestamp"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="results/trades.enriched.csv")
    ap.add_argument("--outfile", default="results/trades.enriched.filled.csv")
    args = ap.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)

    df = pd.read_csv(infile)
    if df.empty:
        print(f"[fill] input is empty: {infile}")
        df.to_csv(outfile, index=False)
        return

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_ts", "symbol"]).copy()
    df["symbol"] = df["symbol"].astype(str)

    # Ensure columns exist
    for c in ["days_since_prev_break", "consolidation_range_atr", "prior_1d_ret", "rv_3d"]:
        if c not in df.columns:
            df[c] = np.nan

    # Fill only missing values in these columns
    syms = sorted(df["symbol"].unique().tolist())
    print(f"[fill] symbols: {len(syms)}")

    for i, sym in enumerate(syms, 1):
        sub_idx = df.index[df["symbol"] == sym]
        ts = df.loc[sub_idx, "entry_ts"].sort_values()
        ts = pd.to_datetime(df.loc[sub_idx, "entry_ts"], utc=True, errors="coerce")
        ts = ts.dropna().sort_values()
        ts = pd.DatetimeIndex(ts.unique()).sort_values()

        if ts.empty:
            continue

        # Load enough history for accurate days_since_prev_break (use full project window)
        df5 = load_parquet_data(
            sym,
            start_date=cfg.START_DATE,
            end_date=cfg.END_DATE,
            drop_last_partial=True,
            columns=["open", "high", "low", "close", "volume"],
        )
        if df5 is None or df5.empty:
            continue

        df5 = df5.sort_index()
        panel = compute_entry_quality_panel(df5).reset_index()
        panel = panel.sort_values("timestamp")

        sub = df.loc[sub_idx, ["entry_ts", "symbol"]].copy()
        sub["_idx"] = sub.index
        sub["entry_ts"] = pd.to_datetime(sub["entry_ts"], utc=True, errors="coerce")
        sub = sub.dropna(subset=["entry_ts"]).sort_values("entry_ts")

        joined = pd.merge_asof(
            sub,
            panel,
            left_on="entry_ts",
            right_on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("_idx")

        for c in ["days_since_prev_break", "consolidation_range_atr", "prior_1d_ret", "rv_3d"]:
            m = df.loc[sub_idx, c].isna()
            df.loc[sub_idx[m], c] = pd.to_numeric(joined.loc[m.values, c], errors="coerce").values


        # free memory
        del df5, panel, joined, sub
        gc.collect()


        if i % 25 == 0:
            print(f"[fill] done {i}/{len(syms)}")

    df.to_csv(outfile, index=False)
    print(f"[fill] wrote: {outfile} rows={len(df)}")


if __name__ == "__main__":
    main()
