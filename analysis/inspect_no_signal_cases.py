import pandas as pd
from pathlib import Path

root = Path("/opt/testerdonch")

core_path = root / "results" / "core_unmatched_live_analysis.csv"
core = pd.read_csv(core_path, parse_dates=["exit_ts_live", "signal_ts"])

print("=== Classification counts (core unmatched LIVE) ===")
print(core["classification"].value_counts())
print()

no_sym = core[core["classification"] == "no_signals_for_symbol"].copy()
no_win = core[core["classification"] == "no_signal_in_window"].copy()

print("no_signals_for_symbol:", len(no_sym))
print("no_signal_in_window:", len(no_win))
print()

print("=== no_signals_for_symbol: top 20 symbols ===")
print(no_sym["symbol"].value_counts().head(20))
print()

print("=== no_signal_in_window: top 20 symbols ===")
print(no_win["symbol"].value_counts().head(20))
print()

# For a few examples, show the live trade time window and whether signals directory exists
for label, df_sub in [("no_signals_for_symbol", no_sym), ("no_signal_in_window", no_win)]:
    if df_sub.empty:
        continue
    print(f"\n=== Sample rows for {label} ===")
    sample = df_sub.head(10)
    for _, row in sample.iterrows():
        sym = row["symbol"]
        sig_ts = row.get("signal_ts", pd.NaT)
        exit_ts = row.get("exit_ts_live", pd.NaT)
        sig_dir = root / "signals" / f"symbol={sym}"
        has_signals_dir = sig_dir.exists()
        print(
            f"symbol={sym:>12}  signal_ts={sig_ts}  exit_ts_live={exit_ts}  "
            f"signals_dir_exists={has_signals_dir}"
        )
