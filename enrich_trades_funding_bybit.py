from __future__ import annotations

import os
import time
import argparse
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import requests

BYBIT_BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
SESSION = requests.Session()

# Match your 5m-bar rolling logic in scout.py
WIN_1H = 12
WIN_1D = 288
WIN_3D = 3 * WIN_1D
WIN_7D = 7 * WIN_1D


def _pick_entry_ts_col(df: pd.DataFrame) -> str:
    for c in ("entry_ts", "entry_time", "entry_timestamp", "entry_datetime"):
        if c in df.columns:
            return c
    raise ValueError("Could not find entry timestamp column (expected entry_ts or similar).")


def _bybit_get(path: str, params: dict, max_retries: int = 6) -> dict:
    url = BYBIT_BASE_URL.rstrip("/") + path
    last_err = None
    for attempt in range(max_retries):
        try:
            r = SESSION.get(url, params=params, timeout=25)

            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                time.sleep(min(12.0, float(ra) if ra else (1.0 + 0.5 * attempt)))
                continue

            r.raise_for_status()
            data = r.json()

            if str(data.get("retCode", "0")) != "0":
                raise RuntimeError(f"retCode={data.get('retCode')} retMsg={data.get('retMsg')} params={params}")

            return data.get("result", {}) or {}

        except Exception as e:
            last_err = e
            time.sleep(0.6 + 0.4 * attempt)

    raise RuntimeError(f"Bybit request failed after {max_retries} tries: {last_err}")


def get_instrument(symbol: str) -> Optional[Tuple[str, int, Optional[int]]]:
    """
    Returns (category, launchTime_ms, fundingInterval_minutes) if found.
    Uses /v5/market/instruments-info. :contentReference[oaicite:2]{index=2}
    """
    sym = symbol.upper()
    for category in ("linear", "inverse"):
        res = _bybit_get("/v5/market/instruments-info", {"category": category, "symbol": sym, "limit": 1})
        lst = res.get("list", []) or []
        if not lst:
            continue
        info = lst[0]
        launch_ms = int(info.get("launchTime")) if info.get("launchTime") is not None else 0
        # Normalize launchTime to milliseconds (some feeds/clients return s/us/ns)
        # ms around 2025 is ~1.7e12
        if 0 < launch_ms < 10**11:          # seconds
            launch_ms *= 1000
        elif launch_ms > 10**16:            # nanoseconds
            launch_ms //= 1_000_000
        elif launch_ms > 10**13:            # microseconds
            launch_ms //= 1000

        fint = int(info.get("fundingInterval")) if info.get("fundingInterval") is not None else None
        return category, launch_ms, fint
    return None


def fetch_funding_history(symbol: str, category: str, start_ms: int, end_ms: int, warmup_days: int = 8) -> pd.DataFrame:
    """
    Page backwards using endTime. Endpoint:
      GET /v5/market/funding/history  (limit max 200, timestamp is ms). :contentReference[oaicite:3]{index=3}
    Returns DataFrame indexed by UTC timestamp with column 'funding_rate' at funding settlement times.
    """
    sym = symbol.upper()

    warmup_ms = int(warmup_days * 24 * 3600 * 1000)
    target_min_ms = max(0, int(start_ms) - warmup_ms)
    cur_end = int(end_ms)

    rows_all: List[dict] = []
    detected_unit = None  # "ms" or "s"

    while True:
        res = _bybit_get(
            "/v5/market/funding/history",
            {"category": category, "symbol": sym, "endTime": cur_end, "limit": 200},
        )
        rows = res.get("list", []) or []
        if not rows:
            break

        # Detect seconds-vs-ms defensively (should be ms per docs, but be robust)
        ts_raw = []
        for x in rows:
            try:
                ts_raw.append(int(x.get("fundingRateTimestamp")))
            except Exception:
                pass

        if not ts_raw:
            break

        med = int(np.median(ts_raw))
        if detected_unit is None:
            detected_unit = "s" if med < 10**11 else "ms"

        # normalize timestamps to ms
        if detected_unit == "s":
            for x in rows:
                try:
                    x["fundingRateTimestamp"] = str(int(x["fundingRateTimestamp"]) * 1000)
                except Exception:
                    pass
            ts_vals = [v * 1000 for v in ts_raw]
        else:
            ts_vals = ts_raw

        rows_all.extend(rows)

        oldest = int(min(ts_vals))
        if oldest <= target_min_ms:
            break

        cur_end = oldest - 1
        time.sleep(0.05)

    if not rows_all:
        return pd.DataFrame()

    df = pd.DataFrame(rows_all)
    df["fundingRateTimestamp"] = pd.to_numeric(df["fundingRateTimestamp"], errors="coerce")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df = df.dropna(subset=["fundingRateTimestamp", "fundingRate"])

    df["ts"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True)
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")

    out = df[["ts", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"}).set_index("ts")
    out.index.name = "ts"
    return out


def build_funding_5m_features(f_settle: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Convert funding settlement series -> 5m forward-filled series, then compute rolling features
    using the same bar-count windows your scout uses.
    """
    if f_settle.empty:
        return pd.DataFrame()

    start_ts = start_ts.floor("5min")
    end_ts = end_ts.ceil("5min")

    idx5 = pd.date_range(start_ts, end_ts, freq="5min", tz="UTC")
    f5 = pd.DataFrame(index=idx5)
    f5["funding_rate"] = f_settle["funding_rate"].reindex(idx5, method="ffill")

    # Derived columns
    f5["funding_abs"] = f5["funding_rate"].abs()

    mean_7d = f5["funding_rate"].rolling(WIN_7D, min_periods=WIN_1D).mean()
    std_7d = f5["funding_rate"].rolling(WIN_7D, min_periods=WIN_1D).std(ddof=0)
    f5["funding_z_7d"] = (f5["funding_rate"] - mean_7d) / (std_7d + 1e-12)

    f5["funding_rollsum_3d"] = f5["funding_rate"].rolling(WIN_3D, min_periods=WIN_1D).sum()
    return f5


def map_at_entries(f5: pd.DataFrame, entry_ts: pd.Series) -> pd.DataFrame:
    """
    Map 5m features to entries using backward fill (asof on 5m grid).
    IMPORTANT: keep tz-aware timestamps; do NOT use .values (drops tz).
    """
    if f5.empty:
        return pd.DataFrame(index=entry_ts.index)

    e = pd.to_datetime(entry_ts, utc=True, errors="coerce").dt.floor("5min")

    # Keep tz-aware dtype by using DatetimeIndex (NOT e.values)
    e_idx = pd.DatetimeIndex(e)

    # Handle NaT entries safely
    out = pd.DataFrame(index=entry_ts.index, columns=f5.columns, dtype="float64")
    valid = ~e.isna()
    if valid.any():
        mapped = f5.reindex(e_idx[valid.to_numpy()], method="ffill")
        out.loc[valid.to_numpy(), :] = mapped.to_numpy()

    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="results/trades.csv")
    ap.add_argument("--out", default="results/trades.enriched.csv")
    ap.add_argument("--throttle", type=float, default=0.0, help="Extra sleep seconds between symbols.")
    args = ap.parse_args()

    df = pd.read_csv(args.trades)

    entry_col = _pick_entry_ts_col(df)
    df[entry_col] = pd.to_datetime(df[entry_col], utc=True, errors="coerce")
    df = df.dropna(subset=[entry_col])

    if "symbol" not in df.columns:
        raise ValueError("trades.csv must contain 'symbol' column.")

    # Ensure target columns exist (NO renames)
    for c in [
        "funding_rate", "funding_abs", "funding_z_7d", "funding_rollsum_3d", "funding_oi_div",
        "btc_funding_rate", "eth_funding_rate",
    ]:
        if c not in df.columns:
            df[c] = np.nan

    # Coerce numeric targets to real NaNs
    for c in ["funding_rate", "funding_abs", "funding_z_7d", "funding_rollsum_3d", "funding_oi_div", "btc_funding_rate", "eth_funding_rate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Cache funding per symbol
    cache: Dict[str, pd.DataFrame] = {}

    def enrich_symbol(sym: str, cols_to_fill: List[str]) -> Optional[pd.DataFrame]:
        sym_u = sym.upper()
        inst = get_instrument(sym_u)
        if inst is None:
            print(f"[enrich] {sym_u}: not found on instruments-info -> skipping")
            return None

        category, launch_ms, fint = inst

        # compute window from trades of this symbol
        idx = df.index[df["symbol"].astype(str).str.upper().eq(sym_u)]
        if len(idx) == 0:
            return None

        tmin = df.loc[idx, entry_col].min()
        tmax = df.loc[idx, entry_col].max()
        start_ms = int(tmin.value // 10**6)
        end_ms = int(tmax.value // 10**6)

        # clamp to launch time
        if launch_ms and start_ms < launch_ms:
            start_ms = launch_ms

        if start_ms >= end_ms:
            # Before skipping, probe whether Bybit returns ANY funding history up to end_ms.
            # This protects against incorrect launchTime units / metadata.
            print(f"[enrich] {sym_u}: trades appear pre-launch; probing funding history anyway...")
            probe = fetch_funding_history(sym_u, category, max(0, end_ms - 10 * 24 * 3600 * 1000), end_ms)
            if probe.empty:
                print(f"[enrich] {sym_u}: probe returned 0 points -> cannot enrich (likely truly pre-launch)")
                return None
            else:
                print(f"[enrich] {sym_u}: probe returned {len(probe)} points -> launchTime likely misleading; continuing")
                start_ms = max(0, end_ms - 30 * 24 * 3600 * 1000)  # widen a bit so features warm up


        if sym_u not in cache:
            f_settle = fetch_funding_history(sym_u, category, start_ms, end_ms)
            print(f"[enrich] {sym_u}: fetched {len(f_settle)} funding points (category={category}, intervalMin={fint})")
            cache[sym_u] = f_settle
        else:
            f_settle = cache[sym_u]

        if f_settle.empty:
            print(f"[enrich] {sym_u}: funding history empty in requested window")
            return None

        # Build 5m features with warmup (8 days already fetched)
        f5 = build_funding_5m_features(f_settle, tmin - pd.Timedelta(days=8), tmax)
        if f5.empty:
            print(f"[enrich] {sym_u}: failed to build 5m funding series")
            return None

        # Debug ranges when things don't fill
        e_min, e_max = tmin, tmax
        f_min, f_max = f5.index.min(), f5.index.max()
        if e_max < f_min or e_min > f_max:
            print(f"[enrich] {sym_u}: entry range {e_min}..{e_max} outside funding range {f_min}..{f_max}")
            return None

        mapped = map_at_entries(f5[cols_to_fill], df.loc[idx, entry_col])
        return idx, mapped

    # (A) BTC/ETH snapshot columns (apply to ALL rows)
    for ref_sym, out_col in [("BTCUSDT", "btc_funding_rate"), ("ETHUSDT", "eth_funding_rate")]:
        need_idx = df.index[df[out_col].isna()]
        if len(need_idx) == 0:
            continue

        result = enrich_symbol(ref_sym, ["funding_rate"])
        if result is None:
            continue
        idx, mapped = result
        # mapped here is only for ref_sym trades; but btc/eth snapshot is needed for all trades.
        # We'll instead build from full window once and map for all rows:
        inst = get_instrument(ref_sym)
        category, launch_ms, fint = inst
        tmin_all = df[entry_col].min()
        tmax_all = df[entry_col].max()
        start_ms_all = int(tmin_all.value // 10**6)
        end_ms_all = int(tmax_all.value // 10**6)
        if launch_ms and start_ms_all < launch_ms:
            start_ms_all = launch_ms

        f_settle = fetch_funding_history(ref_sym, category, start_ms_all, end_ms_all)
        print(f"[enrich] {ref_sym}: fetched {len(f_settle)} funding points (category={category}, intervalMin={fint})")
        f5 = build_funding_5m_features(f_settle, tmin_all - pd.Timedelta(days=8), tmax_all)
        mapped_all = map_at_entries(f5[["funding_rate"]], df[entry_col])
        df[out_col] = df[out_col].combine_first(mapped_all["funding_rate"])

    # (B) per-trade symbol columns
    symbols = df["symbol"].astype(str).str.upper().unique().tolist()
    total_cells_filled = 0

    for sym in symbols:
        sym_idx = df.index[df["symbol"].astype(str).str.upper().eq(sym)]
        if len(sym_idx) == 0:
            continue

        need_any = df.loc[sym_idx, ["funding_rate", "funding_abs", "funding_z_7d", "funding_rollsum_3d"]].isna().any(axis=1)
        need_idx = sym_idx[need_any.to_numpy()]
        if len(need_idx) == 0:
            continue

        result = enrich_symbol(sym, ["funding_rate", "funding_abs", "funding_z_7d", "funding_rollsum_3d"])
        if result is None:
            continue

        idx, mapped = result

        # Fill only missing values, count per-cell fills
        for c in ["funding_rate", "funding_abs", "funding_z_7d", "funding_rollsum_3d"]:
            before = df.loc[idx, c].isna().sum()
            df.loc[idx, c] = df.loc[idx, c].combine_first(mapped[c])
            after = df.loc[idx, c].isna().sum()
            total_cells_filled += int(before - after)

        # funding_oi_div matches your scout: funding_z_7d * oi_z_7d
        if "oi_z_7d" in df.columns:
            df["oi_z_7d"] = pd.to_numeric(df["oi_z_7d"], errors="coerce")
            need_div = df.loc[idx, "funding_oi_div"].isna()
            if need_div.any():
                fill_div = df.loc[idx, "funding_z_7d"] * df.loc[idx, "oi_z_7d"]
                df.loc[idx, "funding_oi_div"] = df.loc[idx, "funding_oi_div"].combine_first(fill_div)

        still_missing = df.loc[idx, "funding_rate"].isna().sum()
        print(f"[enrich] {sym}: remaining missing funding_rate rows for this symbol: {still_missing} / {len(idx)}")

        if args.throttle > 0:
            time.sleep(args.throttle)

    df.to_csv(args.out, index=False)
    print(f"[enrich] wrote {args.out} rows={len(df)}")
    print(f"[enrich] total newly filled cells (funding_rate/abs/z/rollsum): {total_cells_filled}")


if __name__ == "__main__":
    main()
