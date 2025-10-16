# debug_cutoff.py
from __future__ import annotations
import glob
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

def tscol(x):
    return pd.to_datetime(x, utc=True, errors="coerce")

def _infer_symbol_from_path(f: Path, base: Path) -> str:
    try:
        relative_parts = f.relative_to(base).parts
    except ValueError:
        relative_parts = f.parts
    for part in reversed(relative_parts):
        if part.startswith("symbol="):
            return part.split("=", 1)[1]
    return f.stem


def _convert_stat(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    result = pd.to_datetime(value, utc=True, errors="coerce")
    if isinstance(result, (pd.Series, pd.Index)):
        return result.iloc[0]
    return result


def parquet_max_timestamp(parquet_dir="parquet"):
    base = Path(parquet_dir)
    rows = []
    columns = ["symbol", "ts_min", "ts_max", "n"]
    if not base.exists():
        return pd.DataFrame([], columns=columns)

    direct_files = sorted(base.glob("*.parquet"))
    all_files = direct_files if direct_files else sorted(base.glob("**/*.parquet"))
    if not all_files:
        return pd.DataFrame([], columns=columns)

    stats = defaultdict(lambda: {"ts_min": None, "ts_max": None, "n": 0})

    for f in all_files:
        symbol = _infer_symbol_from_path(f, base)
        try:
            pf = pq.ParquetFile(f)
        except Exception:
            rows.append((symbol, None, None, 0))
            continue

        try:
            ts_idx = pf.schema_arrow.get_field_index("timestamp")
        except AttributeError:  # fallback for older pyarrow
            ts_idx = pf.schema.get_field_index("timestamp")
        if ts_idx == -1:
            continue

        entry = stats[symbol]
        entry["n"] += pf.metadata.num_rows

        file_min = None
        file_max = None
        for rg in range(pf.metadata.num_row_groups):
            column = pf.metadata.row_group(rg).column(ts_idx)
            statistics = column.statistics
            if statistics and statistics.has_min_max:
                rg_min = _convert_stat(statistics.min)
                rg_max = _convert_stat(statistics.max)
            else:
                data = pf.read_row_group(rg, columns=["timestamp"]).to_pandas()
                if data.empty:
                    continue
                s = tscol(data["timestamp"])
                rg_min = s.min()
                rg_max = s.max()
            if file_min is None or (rg_min is not None and rg_min < file_min):
                file_min = rg_min
            if file_max is None or (rg_max is not None and rg_max > file_max):
                file_max = rg_max

        if file_min is not None:
            entry["ts_min"] = file_min if entry["ts_min"] is None else min(entry["ts_min"], file_min)
        if file_max is not None:
            entry["ts_max"] = file_max if entry["ts_max"] is None else max(entry["ts_max"], file_max)

    for symbol, entry in stats.items():
        rows.append((symbol, entry["ts_min"], entry["ts_max"], entry["n"]))

    return pd.DataFrame(rows, columns=columns).dropna(subset=["ts_min", "ts_max"], how="all").sort_values("ts_max")

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
    pq_dir = Path("parquet")
    pq = parquet_max_timestamp(pq_dir)
    if not pq.empty:
        print("raw:", pq["ts_min"].min(), "→", pq["ts_max"].max(), " symbols:", len(pq))
        late = pq.query("ts_max >= '2025-09-01'")
        print("symbols with data >= 2025-09-01:", len(late))
    else:
        if pq_dir.exists():
            print("parquet directory present but no timestamp data found")
        else:
            print("parquet directory not found")

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
