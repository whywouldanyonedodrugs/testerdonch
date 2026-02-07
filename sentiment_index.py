# sentiment_index.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, List

import numpy as np
import pandas as pd

import config as cfg
from shared_utils import get_symbols_from_file, load_parquet_data


def _build_canonical_index(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Build a canonical 5m UTC grid from start_date to end_date inclusive.

    We assume underlying perps data is on a 5m grid; this gives us a stable
    index to aggregate onto without ever concatenating all per-symbol frames.
    """
    start = pd.to_datetime(start_date, utc=True)
    # end at end_date 23:55 so we cover the full day on a 5m grid
    end = pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
    idx = pd.date_range(start, end, freq="5min", tz="UTC")
    idx.name = "timestamp"
    return idx


def _normalize_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DatetimeIndex named 'timestamp', UTC, sorted, no duplicate timestamps.
    """
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.set_index("timestamp", drop=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index.name = "timestamp"
    df = df.sort_index()
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df


def _zscore_rolling(series: pd.Series, window_bars: int, min_periods: int) -> pd.Series:
    """
    Rolling z-score with a window specified in bars.
    """
    roll_mean = series.rolling(window_bars, min_periods=min_periods).mean()
    roll_std = series.rolling(window_bars, min_periods=min_periods).std()
    return (series - roll_mean) / (roll_std + 1e-12)


def build_perps_sentiment_index(
    symbols: Optional[Iterable[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    save_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Build cross-sectional perps sentiment indices on a 5m grid WITHOUT
    concatenating all per-symbol frames (streaming aggregation).

    For each symbol on the canonical 5m grid we compute:

      ret_1h       : 1h close-to-close return
      ret_1d       : 1d close-to-close return
      oi_chg_1h    : 1h OI change normalised by 1h volume
      oi_chg_1d    : 1d OI change normalised by 1d volume
      funding_rate : funding_rate on the 5m grid

    Then aggregate cross-sectionally and derive:

      sent_rets_1h_z      : z-score of cross-sectional mean 1h return
      sent_rets_1d_z      : z-score of cross-sectional mean 1d return
      sent_oi_chg_1h_z    : z-score of cross-sectional mean OI change (1h)
      sent_oi_chg_1d_z    : z-score of cross-sectional mean OI change (1d)
      sent_funding_mean_1d: 1d rolling avg of cross-sectional funding
      sent_funding_z_1d   : z-score of that 1d funding average
      sent_beta_risk_on   : tanh of a weighted combo of the above
    """
    if symbols is None:
        symbols = list(get_symbols_from_file())
    else:
        symbols = list(symbols)

    if not symbols:
        print("[sentiment_index] No symbols provided; nothing to do.")
        return pd.DataFrame()

    start_date = start_date or cfg.START_DATE
    end_date = end_date or cfg.END_DATE

    # 5m constants
    WIN_1H = 12          # 12 * 5m = 60m
    WIN_1D = 288         # 288 * 5m = 1 day
    WIN_Z = 7 * WIN_1D   # 7 days rolling window for z-scores
    MIN_Z = WIN_1D       # require at least 1 day of history

    # Canonical 5m grid
    idx = _build_canonical_index(start_date, end_date)
    n = len(idx)

    # Aggregator: for each metric we keep sum and count per timestamp
    base_metrics = ["ret_1h", "ret_1d", "oi_chg_1h", "oi_chg_1d", "funding_rate"]
    agg_data = {}
    for base in base_metrics:
        agg_data[f"{base}_sum"] = np.zeros(n, dtype="float64")
        agg_data[f"{base}_count"] = np.zeros(n, dtype="int32")

    agg = pd.DataFrame(agg_data, index=idx)

    # --- streaming aggregation over symbols ------------------------------------
    n_used = 0

    for sym in symbols:
        df5 = load_parquet_data(
            sym,
            start_date=start_date,
            end_date=end_date,
            drop_last_partial=True,
            columns=["close", "open_interest", "funding_rate", "volume"],
        )
        if df5 is None or df5.empty:
            continue

        df5 = _normalize_ts_index(df5)
        # Reindex each symbol to the canonical grid; missing bars become NaN
        df5 = df5.reindex(idx)

        # Ensure numeric
        for col in ["close", "open_interest", "funding_rate", "volume"]:
            if col in df5.columns:
                df5[col] = pd.to_numeric(df5[col], errors="coerce")

        if "close" not in df5.columns or "volume" not in df5.columns:
            continue

        close = df5["close"]
        vol = df5["volume"]

        # 1h & 1d returns
        ret_1h = close.pct_change(WIN_1H, fill_method=None)
        ret_1d = close.pct_change(WIN_1D, fill_method=None)

        # OI-normalised changes
        if "open_interest" in df5.columns:
            oi = df5["open_interest"]
            vol_1h = vol.rolling(WIN_1H).sum()
            vol_1d = vol.rolling(WIN_1D).sum()
            oi_chg_1h = oi.diff(WIN_1H) / (vol_1h + 1e-9)
            oi_chg_1d = oi.diff(WIN_1D) / (vol_1d + 1e-9)
        else:
            oi_chg_1h = pd.Series(index=idx, dtype="float64")
            oi_chg_1d = pd.Series(index=idx, dtype="float64")

        funding = (
            df5["funding_rate"] if "funding_rate" in df5.columns
            else pd.Series(index=idx, dtype="float64")
        )

        series_by_name = {
            "ret_1h": ret_1h,
            "ret_1d": ret_1d,
            "oi_chg_1h": oi_chg_1h,
            "oi_chg_1d": oi_chg_1d,
            "funding_rate": funding,
        }

        # Update running sums & counts
        for base, ser in series_by_name.items():
            arr = ser.to_numpy(dtype="float64", copy=False)
            mask = np.isfinite(arr)
            if not mask.any():
                continue
            agg[f"{base}_sum"].values[mask] += arr[mask]
            agg[f"{base}_count"].values[mask] += 1

        n_used += 1

    if n_used == 0:
        print("[sentiment_index] No symbols had usable data; exiting.")
        return pd.DataFrame()

    print(f"[sentiment_index] Aggregated sentiment across {n_used} symbols on {n} timestamps.")

    # --- compute cross-sectional means from sums & counts ----------------------
    cs_data = {}
    for base in base_metrics:
        sum_arr = agg[f"{base}_sum"].values
        cnt_arr = agg[f"{base}_count"].values.astype("float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_arr = np.where(cnt_arr > 0, sum_arr / cnt_arr, np.nan)
        cs_data[base] = mean_arr

    cs = pd.DataFrame(cs_data, index=idx)

    # --- derive sentiment features --------------------------------------------
    sent = pd.DataFrame(index=idx)
    sent.index.name = "timestamp"

    sent["sent_rets_1h_z"] = _zscore_rolling(cs["ret_1h"], WIN_Z, MIN_Z)
    sent["sent_rets_1d_z"] = _zscore_rolling(cs["ret_1d"], WIN_Z, MIN_Z)
    sent["sent_oi_chg_1h_z"] = _zscore_rolling(cs["oi_chg_1h"], WIN_Z, MIN_Z)
    sent["sent_oi_chg_1d_z"] = _zscore_rolling(cs["oi_chg_1d"], WIN_Z, MIN_Z)

    # funding: 1d rolling avg of cross-sectional funding, then z-score
    fund_1d = cs["funding_rate"].rolling(WIN_1D, min_periods=WIN_1D // 2).mean()
    sent["sent_funding_mean_1d"] = fund_1d
    sent["sent_funding_z_1d"] = _zscore_rolling(fund_1d, WIN_Z, MIN_Z)

    # Composite sentiment / risk-on beta (bounded with tanh)
    beta_raw = (
        0.5 * sent["sent_rets_1d_z"].fillna(0.0)
        + 0.25 * sent["sent_rets_1h_z"].fillna(0.0)
        + 0.15 * sent["sent_oi_chg_1d_z"].fillna(0.0)
        + 0.10 * sent["sent_funding_z_1d"].fillna(0.0)
    )
    sent["sent_beta_risk_on"] = np.tanh(beta_raw)

    sent.replace([np.inf, -np.inf], np.nan, inplace=True)
    sent = sent.sort_index()

    # --- persist ---------------------------------------------------------------
    target_path = Path(save_path or (cfg.RESULTS_DIR / "sentiment_index.parquet"))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    sent.to_parquet(target_path)

    print(f"[sentiment_index] built sentiment index with {len(sent)} rows, saved to {target_path}")
    return sent


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build perps-wide sentiment index from 5m perps data.")
    ap.add_argument("--symbols", nargs="*", default=None, help="Optional subset of symbols to use.")
    ap.add_argument("--start-date", default=None, help="Override cfg.START_DATE")
    ap.add_argument("--end-date", default=None, help="Override cfg.END_DATE")
    ap.add_argument("--out", default=None, help="Output parquet path (default: results/sentiment_index.parquet)")

    args = ap.parse_args()

    build_perps_sentiment_index(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        save_path=args.out,
    )


if __name__ == "__main__":
    main()
