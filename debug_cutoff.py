# debug_cutoff.py
from __future__ import annotations
import glob
from collections import defaultdict
from pathlib import Path

import pandas as pd


UTC_TS_MAX = pd.Timestamp.max.tz_localize("UTC")
UTC_TS_MIN = pd.Timestamp.min.tz_localize("UTC")
CUTOFF_TS = pd.Timestamp("2025-10-01", tz="UTC")

def tscol(x):
    return pd.to_datetime(x, utc=True, errors="coerce")

def _infer_symbol(path: Path) -> str:
    """Best-effort extraction of a symbol from a partitioned parquet path."""

    for part in reversed(path.parts):
        if "=" in part:
            key, _, value = part.partition("=")
            if key.lower() in {"symbol", "ticker", "asset", "pair"} and value:
                return value
    stem = path.stem
    return stem.split(".")[0] if stem else stem


def parquet_max_timestamp(parquet_dir: str = "/parquet"):
    root = Path(parquet_dir)
    if not root.exists():
        return pd.DataFrame(columns=["symbol", "ts_min", "ts_max", "n", "files", "failed_files"])

    parquet_files = [
        p
        for p in root.rglob("*.parquet")
        if p.is_file() and not p.name.startswith("_")
    ]
    if not parquet_files:
        return pd.DataFrame(columns=["symbol", "ts_min", "ts_max", "n", "files", "failed_files"])

    stats = defaultdict(
        lambda: {
            "ts_min": UTC_TS_MAX,
            "ts_max": UTC_TS_MIN,
            "n": 0,
            "files": 0,
            "failed_files": 0,
        }
    )

    for f in sorted(parquet_files):
        symbol = _infer_symbol(f)
        rec = stats[symbol]
        rec["files"] += 1
        try:
            df = pd.read_parquet(f, columns=["timestamp"])
        except Exception:
            rec["failed_files"] += 1
            continue

        if df.empty:
            rec["failed_files"] += 1
            continue

        s = tscol(df["timestamp"])
        rec["ts_min"] = min(rec["ts_min"], s.min())
        rec["ts_max"] = max(rec["ts_max"], s.max())
        rec["n"] += len(df)

    rows = []
    for symbol, rec in stats.items():
        ts_min = rec["ts_min"]
        ts_max = rec["ts_max"]
        if rec["n"] == 0:
            ts_min = pd.NaT
            ts_max = pd.NaT
        rows.append(
            (
                symbol,
                ts_min,
                ts_max,
                rec["n"],
                rec["files"],
                rec["failed_files"],
            )
        )

    out = pd.DataFrame(
        rows,
        columns=["symbol", "ts_min", "ts_max", "n", "files", "failed_files"],
    ).sort_values("ts_max")
    return out

def signals_max_timestamp(sig_dir="signals"):
    parts = sorted(glob.glob(f"{sig_dir}/symbol=*/part-*.parquet"))
    if not parts: return None
    mx = pd.Timestamp("1970-01-01", tz="UTC"); mn = pd.Timestamp("2100-01-01", tz="UTC")
    sample = parts
    if len(parts) > 400:
        sample = parts[:200] + parts[-200:]
    for p in sample:
        df = pd.read_parquet(p, columns=["timestamp"])
        s = tscol(df["timestamp"])
        mn = min(mn, s.min()); mx = max(mx, s.max())
    return mn, mx, len(parts)

def trades_max_timestamp(trades_csv="results/trades.csv"):
    if not Path(trades_csv).exists(): return None
    t = pd.read_csv(trades_csv, parse_dates=["entry_ts","exit_ts"])
    return t["entry_ts"].min(), t["entry_ts"].max(), t["exit_ts"].max(), len(t)

def main():
    print("=== RAW PARQUET WINDOW ===")
    pq = parquet_max_timestamp("parquet")
    if not pq.empty:
        print("raw:", pq["ts_min"].min(), "→", pq["ts_max"].max(), " symbols:", len(pq))
        late = pq[pd.to_datetime(pq["ts_max"], utc=True) >= CUTOFF_TS]
        print("symbols with data >= 2025-10-01:", len(late))
    else:
        print("no parquet files found")

    print("\n=== SIGNALS WINDOW (hive partitions) ===")
    sm = signals_max_timestamp("signals")
    if sm:
        mn, mx, nparts = sm
        print(f"signals: {mn} → {mx}  (parts={nparts})")
    else:
        print("no signals found")

    print("\n=== TRADES WINDOW ===")
    tm = trades_max_timestamp("results/trades.csv")
    if tm:
        emin, emax, xmax, n = tm
        print(f"trades: entries {emin} → {emax}; exits ≤ {xmax}; n={n}")
    else:
        print("trades.csv not found")

    # Suggest likely bottleneck
    if not pq.empty and sm and tm:
        raw_max = pd.to_datetime(pq["ts_max"].max(), utc=True)
        if pd.isna(raw_max) or raw_max < CUTOFF_TS:
            print("\nLikely bottleneck: RAW 5m parquet coverage stops before Oct.")
        elif pd.isna(sm[1]) or sm[1] < CUTOFF_TS:
            print("\nLikely bottleneck: SIGNALS end before Oct (filters/windows).")
        elif pd.isna(tm[1]) or tm[1] < CUTOFF_TS:
            print("\nLikely bottleneck: BACKTEST skipped/locked after Aug; inspect dedup/cooldown/daycap/variant guards.")
        else:
            print("\nAll three extend into Oct; if trades still end early, inspect time stops and regime filters on late months.")

if __name__ == "__main__":
    main()
