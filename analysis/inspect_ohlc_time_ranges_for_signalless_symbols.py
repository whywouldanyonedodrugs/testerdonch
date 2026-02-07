import pandas as pd
from pathlib import Path

ROOT = Path("/opt/testerdonch")

core_path = ROOT / "results" / "core_unmatched_live_analysis.csv"
core = pd.read_csv(core_path)

# Focus only on these two problem classes
mask = core["classification"].isin(["no_signals_for_symbol", "no_signal_in_window"])
subset = core[mask].copy()

symbols = sorted(subset["symbol"].unique())
print(f"Total unique symbols in these buckets: {len(symbols)}")
print()

def load_ohlc(sym: str) -> pd.DataFrame:
    path = ROOT / "parquet" / f"{sym}.parquet"
    if not path.exists():
        print(f"  [OHLC] parquet missing for {sym}: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # Try to standardize to a DatetimeIndex named "timestamp"
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.drop(columns=["timestamp"])
        df.index = ts
        df.index.name = "timestamp"
    else:
        # Maybe timestamp is in the index (typical for older files)
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to coerce the index to datetime just in case
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.index.name = "timestamp"

    # Drop rows where timestamp could not be parsed
    df = df[df.index.notna()]
    return df.sort_index()

for sym in symbols:
    print("=" * 80)
    print(f"Symbol: {sym}")

    sym_rows = subset[subset["symbol"] == sym]
    live_min = pd.to_datetime(sym_rows["exit_ts_live"].min(), utc=True)
    live_max = pd.to_datetime(sym_rows["exit_ts_live"].max(), utc=True)
    print(f"  Live exit range: {live_min}  ->  {live_max}")

    df = load_ohlc(sym)
    if df.empty:
        print("  OHLC: MISSING or empty after timestamp parsing")
        continue

    ohlc_min = df.index.min()
    ohlc_max = df.index.max()
    print(f"  OHLC range      : {ohlc_min}  ->  {ohlc_max}")

    # Simple diagnostics: are live exits inside OHLC range?
    before = live_min < ohlc_min
    after  = live_max > ohlc_max

    if before and after:
        status = "live exits START BEFORE and END AFTER OHLC range (OHLC is a subset)."
    elif before:
        status = "live exits START BEFORE OHLC range."
    elif after:
        status = "live exits END AFTER OHLC range."
    else:
        status = "live exits are FULLY INSIDE OHLC range."

    print(f"  Coverage vs live: {status}")
