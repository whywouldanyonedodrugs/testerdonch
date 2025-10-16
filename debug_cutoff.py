# debug_cutoff.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
import glob

def tscol(x):
    return pd.to_datetime(x, utc=True, errors="coerce")

def parquet_max_timestamp(parquet_dir="/parquet"):
    rows=[]
    for f in sorted(Path(parquet_dir).glob("*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["timestamp"])
            if df.empty: continue
            s = tscol(df["timestamp"])
            rows.append((f.stem, s.min(), s.max(), len(df)))
        except Exception:
            rows.append((f.stem, None, None, 0))
    out = pd.DataFrame(rows, columns=["symbol","ts_min","ts_max","n"]).sort_values("ts_max")
    return out

def signals_max_timestamp(sig_dir="signals"):
    parts = sorted(glob.glob(f"{sig_dir}/symbol=*/part-*.parquet"))
    if not parts: return None
    mx = pd.Timestamp("1970-01-01", tz="UTC"); mn = pd.Timestamp("2100-01-01", tz="UTC")
    for p in parts[:200]:  # sample to keep it fast
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
        late = pq.query("ts_max >= '2025-09-01'")
        print("symbols with data >= 2025-09-01:", len(late))
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
        if str(pq["ts_max"].max()) < "2025-09-01":
            print("\nLikely bottleneck: RAW 5m parquet coverage stops before Sept.")
        elif str(sm[1]) < "2025-09-01":
            print("\nLikely bottleneck: SIGNALS end before Sept (filters/windows).")
        elif str(tm[1]) < "2025-09-01":
            print("\nLikely bottleneck: BACKTEST skipped/locked after Aug; inspect dedup/cooldown/daycap/variant guards.")
        else:
            print("\nAll three extend into Sept; if trades still end early, inspect time stops and regime filters on late months.")

if __name__ == "__main__":
    main()
