import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path("/opt/testerdonch")

core_path = ROOT / "results" / "core_unmatched_live_analysis.csv"
core = pd.read_csv(core_path, parse_dates=["exit_ts_live"])

# Focus on cases where scout/backtest had no usable signal:
mask = core["classification"].isin(["no_signals_for_symbol", "no_signal_in_window"])
subset = core[mask].copy()

symbols = sorted(subset["symbol"].unique())
print(f"Total symbols in these buckets: {len(symbols)}")
print()

for sym in symbols:
    rows = subset[subset["symbol"] == sym]
    live_min = rows["exit_ts_live"].min()
    live_max = rows["exit_ts_live"].max()

    print("=" * 80)
    print(f"Symbol: {sym}")
    print(f"  Live exit range : {live_min}  ->  {live_max}")

    pq_path = ROOT / "parquet" / f"{sym}.parquet"
    if not pq_path.exists():
        print(f"  OHLC parquet    : MISSING at {pq_path}")
        continue

    try:
        ohlc = pq.read_table(pq_path).to_pandas()
    except Exception as e:
        print(f"  OHLC parquet    : ERROR reading ({e})")
        continue

    if "timestamp" not in ohlc.columns:
        print(f"  OHLC parquet    : has no 'timestamp' column; columns = {list(ohlc.columns)}")
        continue

    ohlc["timestamp"] = pd.to_datetime(ohlc["timestamp"], utc=True, errors="coerce")
    ts_min = ohlc["timestamp"].min()
    ts_max = ohlc["timestamp"].max()

    print(f"  OHLC coverage   : {ts_min}  ->  {ts_max}")

    # Quick sanity: does OHLC cover the live exits?
    covers_all = (ts_min <= live_min) and (ts_max >= live_max)
    print(f"  Covers live window? {'YES' if covers_all else 'NO'}")

print()