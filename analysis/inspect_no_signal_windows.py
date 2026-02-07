import pandas as pd
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path("/opt/testerdonch")

core_path = ROOT / "results" / "core_unmatched_live_analysis.csv"
core = pd.read_csv(core_path)

# Make sure exit_ts_live is a proper datetime
core["exit_ts_live"] = pd.to_datetime(core["exit_ts_live"], utc=True, errors="coerce")

# Focus only on 'no_signal_in_window'
nsw = core[core["classification"] == "no_signal_in_window"].copy()
print("no_signal_in_window count:", len(nsw))
print()

# How many symbols?
print("Symbols:")
print(nsw["symbol"].value_counts())
print()

# We'll inspect up to this many examples per symbol
MAX_EXAMPLES_PER_SYMBOL = 2

def load_signals(sym: str) -> pd.DataFrame:
    """Load all signals for a symbol (scout output)."""
    sig_dir = ROOT / "signals" / f"symbol={sym}"
    if not sig_dir.exists():
        print(f"[signals] directory not found for {sym}: {sig_dir}")
        return pd.DataFrame()

    files = list(sig_dir.glob("*.parquet"))
    if not files:
        print(f"[signals] no parquet files for {sym} in {sig_dir}")
        return pd.DataFrame()

    dfs = [pq.read_table(f).to_pandas() for f in files]
    df = pd.concat(dfs, ignore_index=True)
    # ensure timestamp is datetime with timezone if present
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)

# Pick a few examples (we'll just take head() by symbol)
examples = (
    nsw.sort_values(["symbol", "exit_ts_live"])
       .groupby("symbol")
       .head(MAX_EXAMPLES_PER_SYMBOL)
       .copy()
)

print(
    "Inspecting",
    len(examples),
    "example trades across",
    examples["symbol"].nunique(),
    "symbols",
)
print()

for _, row in examples.iterrows():
    sym = row["symbol"]
    exit_ts_live = row["exit_ts_live"]

    print("=" * 80)
    print(f"Symbol: {sym}")
    print(f"Live exit_ts: {exit_ts_live}")
    print()

    sig_df = load_signals(sym)
    if sig_df.empty:
        print(f"[WARN] No signals loaded for {sym}")
        print()
        continue

    # Look at signals in a ±48h window around live exit
    win_start = exit_ts_live - pd.Timedelta(hours=48)
    win_end   = exit_ts_live + pd.Timedelta(hours=48)

    local = sig_df[
        (sig_df["timestamp"] >= win_start) &
        (sig_df["timestamp"] <= win_end)
    ]

    print(f"Signals in ±48h window: {len(local)}")

    if local.empty:
        # Show nearest 3 signals before and after for context
        before = sig_df[sig_df["timestamp"] < win_start].tail(3)
        after  = sig_df[sig_df["timestamp"] > win_end].head(3)

        print("\nNearest 3 signals BEFORE window:")
        if not before.empty:
            print(before[["timestamp", "entry", "don_break_level", "rs_pct"]]
                  .to_string(index=False))
        else:
            print("  (none)")

        print("\nNearest 3 signals AFTER window:")
        if not after.empty:
            print(after[["timestamp", "entry", "don_break_level", "rs_pct"]]
                  .to_string(index=False))
        else:
            print("  (none)")
    else:
        # Show up to 10 signals in the window
        print(local[["timestamp", "entry", "don_break_level", "rs_pct"]]
              .head(10)
              .to_string(index=False))

    print()