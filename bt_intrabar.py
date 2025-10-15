# bt_intrabar.py
from __future__ import annotations

from typing import Tuple, Optional
import pandas as pd

def resolve_first_touch_1m(
    side: str,
    df_1m: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    stop_price: float,
    take_price: float,
    tie_breaker: str = "sl_wins",
) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
    """
    Resolve which level (SL/TP) is touched first using 1-minute OHLCV.
    Assumes df_1m has UTC DatetimeIndex and columns high/low.

    Parameters
    ----------
    side : "long" | "short"
    df_1m : 1-minute OHLCV DataFrame
    start_ts : entry timestamp (exclusive; we check minutes AFTER this point)
    end_ts : stop checking at or before this time (inclusive)
    stop_price : SL level
    take_price : TP level
    tie_breaker : "sl_wins" or "tp_wins" when both hit same minute

    Returns
    -------
    (hit, hit_ts) where hit ∈ {"sl", "tp", None}
    """
    if df_1m.index.tz is None:
        df_1m.index = df_1m.index.tz_localize("UTC")
    else:
        df_1m.index = df_1m.index.tz_convert("UTC")
    df_1m = df_1m.sort_index()

    # minute bars strictly AFTER start_ts up to and including end_ts
    mask = (df_1m.index > start_ts) & (df_1m.index <= end_ts)
    scan = df_1m.loc[mask]
    if scan.empty:
        return None, None

    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")

    for ts, row in scan.iterrows():
        hi = float(row["high"]); lo = float(row["low"])

        if side == "long":
            tp_hit = (hi >= take_price)
            sl_hit = (lo <= stop_price)
        else:  # short
            tp_hit = (lo <= take_price)  # take price is below for shorts
            sl_hit = (hi >= stop_price)

        if tp_hit and sl_hit:
            # Ambiguity within 1-minute. Use tie-breaker.
            if tie_breaker == "tp_wins":
                return "tp", ts
            else:
                return "sl", ts
        elif tp_hit:
            return "tp", ts
        elif sl_hit:
            return "sl", ts

    return None, None
