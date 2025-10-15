#!/usr/bin/env python3
# bybit_pull_kline_v11.py
# Pull 5m (or 1m) klines for symbols in perplist.txt.
# - UPDATE (file exists): page forward with `start`
# - FULL (no file yet):   page backward with `end`
# Robust against boundary echoes; UTC-aware; no deprecated pandas usage.
# Hardened: uses pd.Timestamp.now(tz="UTC") (fixes tz_localize crash) and
# thread error isolation so one failing symbol doesn't stop the run.

from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time, threading, requests, pandas as pd
import sys

# ---------------- Config ----------------
SYMBOL_LIST_FILE = "perplist.txt"
INTERVALS_TO_DOWNLOAD = ["5"]         # allowed: "1", "5"
OUTPUT_ROOT = Path(".")               # creates data5/, data1/
MAX_WORKERS = 24
CHUNK_LIMIT = 1000
REQUEST_TIMEOUT = 20
USER_AGENT = "bybit-kline-puller-v11"
APPEND_DEDUP_AFTER_RUN = True         # drop duplicate open_time and sort at end
CATEGORY = "linear"                   # USDT perps
# ---------------------------------------

REST_API_URL = "https://api.bybit.com/v5/market/kline"
RATE_LIMIT_RET_CODES = {10006, 10018, 10016}  # RL/system busy

INTERVAL_TO_MIN = {"1": 1, "5": 5}
NUMERIC_COLS = ["open", "high", "low", "close", "volume", "turnover"]
ALL_COLS = ["open_time"] + NUMERIC_COLS

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
    """Return a tz-aware UTC Timestamp (no tz_localize needed)."""
    return pd.Timestamp.now(tz="UTC")

def to_ms(ts: pd.Timestamp) -> int:
    """UTC Timestamp -> epoch milliseconds."""
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return int(ts.timestamp() * 1000)

def read_last_saved_ts(csv_path: Path) -> pd.Timestamp | None:
    """Return last open_time as UTC-aware Timestamp, or None if empty/missing."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None

    df = pd.read_csv(csv_path, usecols=["open_time"])
    last = str(df["open_time"].iloc[-1]).strip()

    # epoch ms?
    if last.isdigit():
        return pd.to_datetime(int(last), unit="ms", utc=True)

    # legacy DD/MM/YYYY HH:MM?
    if "/" in last and "-" not in last:
        ts = pd.to_datetime(last, format="%d/%m/%Y %H:%M", utc=True, errors="coerce")
        if not pd.isna(ts):
            return ts

    # general ISO fallback
    ts = pd.to_datetime(last, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Cannot parse last open_time: {last}")
    return ts

def normalize_chunk(chunk_list: list[list|tuple]) -> pd.DataFrame:
    """
    Convert raw Bybit chunk -> DataFrame:
    - open_time ms -> UTC-aware datetime
    - cast price/volume to numeric
    - ascending sorted by open_time
    """
    df = pd.DataFrame(chunk_list, columns=["open_time", *NUMERIC_COLS])
    df["open_time"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df

def api_fetch(session: requests.Session, symbol: str, interval: str, **params) -> dict:
    q = dict(category=CATEGORY, symbol=symbol, interval=interval, limit=CHUNK_LIMIT)
    q.update(params)
    r = session.get(REST_API_URL, params=q, timeout=REQUEST_TIMEOUT)
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

    last_saved_ts = read_last_saved_ts(out_csv)  # None => FULL
    mode = "UPDATE" if last_saved_ts is not None else "FULL"
    safe_print(f"{log_prefix} Start {mode}. last_saved_ts={last_saved_ts}")

    start_time = time.time()
    total_new = 0
    backoff = 1.0
    retries = 0
    step = pd.Timedelta(minutes=INTERVAL_TO_MIN.get(interval, 5))

    try:
        if mode == "UPDATE":
            # Forward paging with `start` (inclusive on Bybit)
            cursor = (last_saved_ts + step) if last_saved_ts is not None else None
            no_progress_hits = 0
            while True:
                try:
                    params = {"start": to_ms(cursor)} if cursor else {"start": 0}
                    data = api_fetch(session, symbol, interval, **params)
                except requests.exceptions.RequestException as e:
                    retries += 1
                    if retries > 10:
                        return f"{log_prefix} HARDFAIL NET| {e}"
                    safe_print(f"{log_prefix} NETERR → sleep {backoff:.1f}s")
                    time.sleep(backoff); backoff = min(backoff * 2, 60)
                    continue

                ret = int(data.get("retCode", -1))
                if ret != 0:
                    if ret in RATE_LIMIT_RET_CODES:
                        retries += 1
                        if retries > 10:
                            return f"{log_prefix} HARDFAIL RL | retCode={ret} {data.get('retMsg')}"
                        safe_print(f"{log_prefix} RL      → sleep {backoff:.1f}s")
                        time.sleep(backoff); backoff = min(backoff * 2, 60)
                        continue
                    return f"{log_prefix} APIERROR | retCode={ret} {data.get('retMsg')}"

                chunk = data.get("result", {}).get("list", []) or []
                if not chunk:
                    break

                df_chunk = normalize_chunk(chunk)
                # strictly newer than last_saved_ts
                if last_saved_ts is not None:
                    df_new = df_chunk[df_chunk["open_time"] > last_saved_ts]
                else:
                    df_new = df_chunk

                if df_new.empty:
                    # No progress; bump cursor by a page to avoid echo loops
                    no_progress_hits += 1
                    cursor = (cursor + step * CHUNK_LIMIT) if cursor else now_utc()
                    if no_progress_hits >= 3:
                        # Assume we are up to date
                        break
                    continue

                # Append data
                df_new.to_csv(out_csv, mode="a", header=needs_header, index=False, columns=ALL_COLS)
                needs_header = False
                total_new += len(df_new)
                new_last = df_new["open_time"].iloc[-1]
                safe_print(f"{log_prefix} +{len(df_new):5d} up to {new_last.strftime('%Y-%m-%d %H:%M:%S%z')}")

                # Advance cursors & reset guards
                last_saved_ts = new_last
                cursor = last_saved_ts + step
                retries, backoff, no_progress_hits = 0, 1.0, 0

        else:
            # FULL backfill: backward paging with `end`
            end_ts = now_utc()
            last_boundary = None
            while True:
                try:
                    data = api_fetch(session, symbol, interval, end=to_ms(end_ts))
                except requests.exceptions.RequestException as e:
                    retries += 1
                    if retries > 10:
                        return f"{log_prefix} HARDFAIL NET| {e}"
                    safe_print(f"{log_prefix} NETERR   → sleep {backoff:.1f}s")
                    time.sleep(backoff); backoff = min(backoff * 2, 60)
                    continue

                ret = int(data.get("retCode", -1))
                if ret != 0:
                    if ret in RATE_LIMIT_RET_CODES:
                        retries += 1
                        if retries > 10:
                            return f"{log_prefix} HARDFAIL RL | retCode={ret} {data.get('retMsg')}"
                        safe_print(f"{log_prefix} RL       → sleep {backoff:.1f}s")
                        time.sleep(backoff); backoff = min(backoff * 2, 60)
                        continue
                    return f"{log_prefix} APIERROR  | retCode={ret} {data.get('retMsg')}"

                chunk = data.get("result", {}).get("list", []) or []
                if not chunk:
                    break

                df_chunk = normalize_chunk(chunk)

                # Filter out anything we may already have
                if last_saved_ts is not None:
                    df_new = df_chunk[df_chunk["open_time"] > last_saved_ts]
                else:
                    df_new = df_chunk

                if not df_new.empty:
                    df_new.to_csv(out_csv, mode="a", header=needs_header, index=False, columns=ALL_COLS)
                    needs_header = False
                    total_new += len(df_new)
                    last_saved_ts = df_new["open_time"].iloc[-1]
                    safe_print(f"{log_prefix} +{len(df_new):5d} thru {last_saved_ts.strftime('%Y-%m-%d %H:%M:%S%z')}")

                # Page backward using earliest time from chunk (ascending)
                earliest_raw = df_chunk["open_time"].iloc[0]
                if last_boundary is not None and earliest_raw >= last_boundary:
                    # No backward movement -> nudge back one full page window to prevent loops
                    end_ts = earliest_raw - step * CHUNK_LIMIT
                else:
                    end_ts = earliest_raw - pd.Timedelta(milliseconds=1)
                last_boundary = earliest_raw
                retries, backoff = 0, 1.0

    finally:
        # Optional dedupe + sort to clean minor overlaps
        if APPEND_DEDUP_AFTER_RUN and out_csv.exists() and out_csv.stat().st_size > 0:
            try:
                df = pd.read_csv(out_csv)
                df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
                df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time")
                df.to_csv(out_csv, index=False)
            except Exception as e:
                safe_print(f"{log_prefix} WARN dedupe: {e}")

    dur = time.time() - start_time
    return f"{log_prefix} DONE {mode:<6} | +{total_new:6d} rows | {dur:.1f}s"

def main():
    # symbols
    p = Path(SYMBOL_LIST_FILE)
    if not p.is_file():
        print(f"ERROR: {SYMBOL_LIST_FILE} not found.")
        sys.exit(1)
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
                    # Don't let one symbol kill the entire run
                    msg = f"[WORKER ERROR] {e!r}"
                if msg: safe_print(msg)
                bar.update(1)
        finally:
            try:
                bar.close()
            except Exception:
                pass
    print("="*80)
    print("All downloads completed.")

if __name__ == "__main__":
    main()
