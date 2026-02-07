#!/usr/bin/env python3
# bybit_pull_kline_v12.py
# Pull 5m (or 1m) klines for symbols in perplist.txt AND join open_interest + funding_rate.
# - UPDATE (file exists): page forward with `start`
# - FULL (no file yet):   page backward with `end`
# Adds:
#   * OI via /v5/market/open-interest (intervalTime=5min, cursor paging)
#   * Funding via /v5/market/funding/history (limit=200 paging by time)
# Output CSV columns remain: open_time, open, high, low, close, volume, turnover[, open_interest][, funding_rate]

from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time, threading, requests, pandas as pd
import sys

# ---------------- Config ----------------
SYMBOL_LIST_FILE = "symbols.txt"
INTERVALS_TO_DOWNLOAD = ["5"]         # allowed: "1", "5"
OUTPUT_ROOT = Path(".")               # creates data5/, data1/
MAX_WORKERS = 20
CHUNK_LIMIT = 1000                    # kline limit (Bybit supports up to 1000)
REQUEST_TIMEOUT = 20
USER_AGENT = "bybit-kline-puller-v12"
APPEND_DEDUP_AFTER_RUN = True
CATEGORY = "linear"                   # USDT perps
# --- New knobs ---
FETCH_OI = True
FETCH_FUNDING = True
OI_INTERVAL = "5min"                  # aligns to our 5m bars
OI_PAGE_LIMIT = 200                   # max per docs
FUNDING_PAGE_LIMIT = 200              # max per docs
GLOBAL_MIN_SLEEP = 0.02               # gentle throttle between calls
# ---------------------------------------

REST_API_URL_KLINE = "https://api.bybit.com/v5/market/kline"
REST_API_URL_OI    = "https://api.bybit.com/v5/market/open-interest"
REST_API_URL_FUND  = "https://api.bybit.com/v5/market/funding/history"

# Common RL/system busy codes (public market) per Bybit docs
RATE_LIMIT_RET_CODES = {10006, 10018, 10016}

INTERVAL_TO_MIN = {"1": 1, "5": 5}
NUMERIC_COLS = ["open", "high", "low", "close", "volume", "turnover"]
ALL_COLS = ["open_time"] + NUMERIC_COLS  # base kline schema

# ---------- tqdm (optional prettiness) ----------
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *a, **k): self.total = k.get("total", 0)
        def update(self, n=1): pass
        def close(self): pass
        @staticmethod
        def write(s): print(s)
def safe_print(msg: str): tqdm.write(msg)
# -----------------------------------------------

def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")

def to_ms(ts: pd.Timestamp) -> int:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)

def read_last_saved_ts(csv_path: Path) -> pd.Timestamp | None:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    df = pd.read_csv(csv_path, usecols=["open_time"])
    last = str(df["open_time"].iloc[-1]).strip()
    if last.isdigit():
        return pd.to_datetime(int(last), unit="ms", utc=True)
    if "/" in last and "-" not in last:
        ts = pd.to_datetime(last, format="%d/%m/%Y %H:%M", utc=True, errors="coerce")
        if not pd.isna(ts): return ts
    ts = pd.to_datetime(last, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Cannot parse last open_time: {last}")
    return ts

def normalize_kline_chunk(chunk_list: list[list|tuple]) -> pd.DataFrame:
    df = pd.DataFrame(chunk_list, columns=["open_time", *NUMERIC_COLS])
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

def api_get(session: requests.Session, url: str, params: dict) -> dict:
    time.sleep(GLOBAL_MIN_SLEEP)
    r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()

def pull_symbol(symbol: str, interval: str, out_dir: Path) -> str:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    thread_id = threading.get_ident()
    log_prefix = f"[T{thread_id:05d} {symbol:>12s} {interval:>2}m]"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{symbol}.csv"
    needs_header = not out_csv.exists() or out_csv.stat().st_size == 0

    last_saved_ts = read_last_saved_ts(out_csv)
    mode = "UPDATE" if last_saved_ts is not None else "FULL"
    safe_print(f"{log_prefix} Start {mode}. last_saved_ts={last_saved_ts}")

    start_time = time.time()
    total_new = 0
    backoff = 1.0
    retries = 0
    step = pd.Timedelta(minutes=INTERVAL_TO_MIN.get(interval, 5))
    first_new_ts = None
    latest_ts = last_saved_ts

    try:
        # -------------- KLINE --------------
        if mode == "UPDATE":
            cursor = (last_saved_ts + step) if last_saved_ts is not None else None
            no_progress_hits = 0
            while True:
                try:
                    params = dict(category=CATEGORY, symbol=symbol, interval=interval, limit=CHUNK_LIMIT)
                    if cursor: params["start"] = to_ms(cursor)
                    data = api_get(session, REST_API_URL_KLINE, params)
                except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                    retries += 1
                    if retries > 10: return f"{log_prefix} HARDFAIL NET| {e}"
                    safe_print(f"{log_prefix} NETERR → sleep {backoff:.1f}s")
                    time.sleep(backoff); backoff = min(backoff*2, 60)
                    continue

                ret = int(data.get("retCode", -1))
                if ret != 0:
                    if ret in RATE_LIMIT_RET_CODES:
                        retries += 1
                        if retries > 10: return f"{log_prefix} HARDFAIL RL | retCode={ret} {data.get('retMsg')}"
                        safe_print(f"{log_prefix} RL      → sleep {backoff:.1f}s")
                        time.sleep(backoff); backoff = min(backoff*2, 60)
                        continue
                    return f"{log_prefix} APIERROR | retCode={ret} {data.get('retMsg')}"

                chunk = data.get("result", {}).get("list", []) or []
                if not chunk: break

                df_chunk = normalize_kline_chunk(chunk)
                if last_saved_ts is not None:
                    df_new = df_chunk[df_chunk["open_time"] > last_saved_ts]
                else:
                    df_new = df_chunk

                if df_new.empty:
                    no_progress_hits += 1
                    cursor = (cursor + step*CHUNK_LIMIT) if cursor else now_utc()
                    if no_progress_hits >= 3: break
                    continue

                df_new.to_csv(out_csv, mode="a", header=needs_header, index=False, columns=ALL_COLS)
                if first_new_ts is None: first_new_ts = df_new["open_time"].iloc[0]
                needs_header = False
                total_new += len(df_new)
                latest_ts = df_new["open_time"].iloc[-1]
                safe_print(f"{log_prefix} +{len(df_new):5d} up to {latest_ts.strftime('%Y-%m-%d %H:%M:%S%z')}")
                last_saved_ts = latest_ts
                cursor = last_saved_ts + step
                retries, backoff, no_progress_hits = 0, 1.0, 0

        else:
            end_ts = now_utc()
            last_boundary = None
            while True:
                try:
                    params = dict(category=CATEGORY, symbol=symbol, interval=interval, limit=CHUNK_LIMIT, end=to_ms(end_ts))
                    data = api_get(session, REST_API_URL_KLINE, params)
                except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                    retries += 1
                    if retries > 10: return f"{log_prefix} HARDFAIL NET| {e}"
                    safe_print(f"{log_prefix} NETERR   → sleep {backoff:.1f}s")
                    time.sleep(backoff); backoff = min(backoff*2, 60)
                    continue

                ret = int(data.get("retCode", -1))
                if ret != 0:
                    if ret in RATE_LIMIT_RET_CODES:
                        retries += 1
                        if retries > 10: return f"{log_prefix} HARDFAIL RL | retCode={ret} {data.get('retMsg')}"
                        safe_print(f"{log_prefix} RL       → sleep {backoff:.1f}s")
                        time.sleep(backoff); backoff = min(backoff*2, 60)
                        continue
                    return f"{log_prefix} APIERROR  | retCode={ret} {data.get('retMsg')}"

                chunk = data.get("result", {}).get("list", []) or []
                if not chunk: break
                df_chunk = normalize_kline_chunk(chunk)

                df_new = df_chunk  # full backfill path appends all pages
                if not df_new.empty:
                    df_new.to_csv(out_csv, mode="a", header=needs_header, index=False, columns=ALL_COLS)
                    needs_header = False
                    total_new += len(df_new)
                    latest_ts = df_new["open_time"].iloc[-1]
                    if first_new_ts is None: first_new_ts = df_new["open_time"].iloc[0]
                    safe_print(f"{log_prefix} +{len(df_new):5d} thru {latest_ts.strftime('%Y-%m-%d %H:%M:%S%z')}")

                earliest_raw = df_chunk["open_time"].iloc[0]
                if last_boundary is not None and earliest_raw >= last_boundary:
                    end_ts = earliest_raw - step*CHUNK_LIMIT
                else:
                    end_ts = earliest_raw - pd.Timedelta(milliseconds=1)
                last_boundary = earliest_raw
                retries, backoff = 0, 1.0

    finally:
        if APPEND_DEDUP_AFTER_RUN and out_csv.exists() and out_csv.stat().st_size > 0:
            try:
                df = pd.read_csv(out_csv)
                df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
                df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
                df.to_csv(out_csv, index=False)
            except Exception as e:
                safe_print(f"{log_prefix} WARN dedupe: {e}")

    # -------------- OI + FUNDING ENRICHMENT --------------
    # We merge onto the *full* CSV to keep it simple & robust.
    if (FETCH_OI or FETCH_FUNDING) and out_csv.exists() and out_csv.stat().st_size > 0:
        df = pd.read_csv(out_csv)
        if df.empty:
            return f"{log_prefix} DONE {mode:<6} | +{total_new:6d} rows | no data"
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        t_min = df["open_time"].min()
        t_max = df["open_time"].max()

        if FETCH_OI:
            oi_rows = []
            cursor = None
            retries, backoff = 0, 1.0
            # First page: time-bounded, then cursor pages
            while True:
                try:
                    params = {
                        "category": CATEGORY,
                        "symbol": symbol,
                        "intervalTime": OI_INTERVAL,
                        "limit": OI_PAGE_LIMIT,
                    }
                    if cursor:
                        params["cursor"] = cursor
                    else:
                        params["startTime"] = to_ms(t_min)
                        params["endTime"] = to_ms(t_max)
                    data = api_get(session, REST_API_URL_OI, params)
                except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                    retries += 1
                    if retries > 10:
                        safe_print(f"{log_prefix} OI  HARDFAIL| {e}"); break
                    safe_print(f"{log_prefix} OI  NETERR → sleep {backoff:.1f}s")
                    time.sleep(backoff); backoff = min(backoff*2, 60)
                    continue

                ret = int(data.get("retCode", -1))
                if ret != 0:
                    if ret in RATE_LIMIT_RET_CODES:
                        retries += 1
                        if retries > 10: 
                            safe_print(f"{log_prefix} OI  HARDFAIL RL | {data.get('retMsg')}"); break
                        safe_print(f"{log_prefix} OI  RL     → sleep {backoff:.1f}s")
                        time.sleep(backoff); backoff = min(backoff*2, 60)
                        continue
                    safe_print(f"{log_prefix} OI  APIERR | retCode={ret} {data.get('retMsg')}"); break

                lst = data.get("result", {}).get("list", []) or []
                for row in lst:
                    oi_rows.append([pd.to_datetime(int(row["timestamp"]), unit="ms", utc=True),
                                    float(row["openInterest"]) if row.get("openInterest") is not None else None])
                cursor = (data.get("result", {}) or {}).get("nextPageCursor") or None
                if not cursor:
                    break

            if oi_rows:
                oi_df = (pd.DataFrame(oi_rows, columns=["open_time", "open_interest"])
                           .dropna(subset=["open_time"])
                           .drop_duplicates(subset=["open_time"])
                           .sort_values("open_time")
                           .reset_index(drop=True))
                df = df.merge(oi_df, how="left", on="open_time")

        if FETCH_FUNDING:
            fund_rows = []
            cursor_start = to_ms(t_min)
            retries, backoff = 0, 1.0
            while True:
                try:
                    params = {
                        "category": CATEGORY,
                        "symbol": symbol,
                        "startTime": cursor_start,
                        "endTime": to_ms(t_max),
                        "limit": FUNDING_PAGE_LIMIT,
                    }
                    data = api_get(session, REST_API_URL_FUND, params)
                except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                    retries += 1
                    if retries > 10:
                        safe_print(f"{log_prefix} FUND HARDFAIL| {e}"); break
                    safe_print(f"{log_prefix} FUND NETERR → sleep {backoff:.1f}s")
                    time.sleep(backoff); backoff = min(backoff*2, 60)
                    continue

                ret = int(data.get("retCode", -1))
                if ret != 0:
                    if ret in RATE_LIMIT_RET_CODES:
                        retries += 1
                        if retries > 10:
                            safe_print(f"{log_prefix} FUND HARDFAIL RL | {data.get('retMsg')}"); break
                        safe_print(f"{log_prefix} FUND RL    → sleep {backoff:.1f}s")
                        time.sleep(backoff); backoff = min(backoff*2, 60)
                        continue
                    safe_print(f"{log_prefix} FUND APIERR| retCode={ret} {data.get('retMsg')}"); break

                lst = (data.get("result", {}) or {}).get("list", []) or []
                if not lst:
                    break

                # Funding history may arrive newest-first; we normalize later
                for row in lst:
                    fund_rows.append([pd.to_datetime(int(row["fundingRateTimestamp"]), unit="ms", utc=True),
                                      float(row["fundingRate"]) if row.get("fundingRate") is not None else None])

                # Advance start beyond last fetched point to paginate by time
                last_ts = min(pd.to_datetime(int(row["fundingRateTimestamp"]), unit="ms", utc=True)
                              for row in lst)
                # Move window forward to the next millisecond after last oldest point fetched
                cursor_start = int(last_ts.value // 10**6) + 1
                if len(lst) < FUNDING_PAGE_LIMIT:
                    break

            if fund_rows:
                fdf = (pd.DataFrame(fund_rows, columns=["funding_ts", "funding_rate"])
                         .dropna(subset=["funding_ts"])
                         .drop_duplicates(subset=["funding_ts"])
                         .sort_values("funding_ts")
                         .reset_index(drop=True))
                # As-of merge: each 5m bar gets last known funding rate (settlement) up to that time
                df = pd.merge_asof(df.sort_values("open_time"),
                                   fdf.sort_values("funding_ts"),
                                   left_on="open_time", right_on="funding_ts",
                                   direction="backward")
                df.drop(columns=["funding_ts"], inplace=True)

        # Re-write CSV with enriched columns (preserve base order)
        cols = [c for c in ["open_time", "open", "high", "low", "close", "volume", "turnover",
                            "open_interest", "funding_rate"] if c in df.columns]
        df = df[cols].sort_values("open_time").reset_index(drop=True)
        df.to_csv(out_csv, index=False)

    dur = time.time() - start_time
    return f"{log_prefix} DONE {mode:<6} | +{total_new:6d} rows | {dur:.1f}s"

def main():
    p = Path(SYMBOL_LIST_FILE)
    if not p.is_file():
        print(f"ERROR: {SYMBOL_LIST_FILE} not found."); sys.exit(1)
    raw = [s.strip() for s in p.read_text().splitlines() if s.strip()]
    symbols = sorted(set(raw))
    print(f"Symbols: {len(symbols)} (removed {len(raw)-len(symbols)} duplicates)")

    tasks = []
    for interval in INTERVALS_TO_DOWNLOAD:
        out_dir = OUTPUT_ROOT / f"data{interval}"
        for sym in symbols:
            tasks.append((sym, interval, out_dir))

    print(f"Tasks: {len(tasks)} | Threads: {MAX_WORKERS}")
    print("="*80)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(pull_symbol, *t): t for t in tasks}
        try:
            bar = tqdm(total=len(futures), desc="Overall")
            for fut in as_completed(futures):
                try:
                    msg = fut.result()
                except Exception as e:
                    msg = f"[WORKER ERROR] {e!r}"
                if msg: safe_print(msg)
                bar.update(1)
        finally:
            try: bar.close()
            except Exception: pass
    print("="*80)
    print("All downloads completed.")

if __name__ == "__main__":
    main()
