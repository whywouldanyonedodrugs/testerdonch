import pandas as pd

# 1. Load core unmatched live with BT decisions
df = pd.read_csv("results/core_unmatched_live_with_bt_decisions.csv")

print("=== df shape ===")
print(df.shape)
print()

print("=== bt_reason value counts ===")
print(df["bt_reason"].value_counts())
print()

# 2. Focus on cooldown-only cases
cd = df[df["bt_reason"] == "cooldown"].copy()

print("=== cooldown subset shape ===")
print(cd.shape)
print()

# 3. Top symbols by cooldown count
print("=== Top symbols by cooldown-unmatched count ===")
print(cd["symbol"].value_counts().head(20))
print()

# 4. Basic time overview (days with most cooldown misses)
if "signal_ts" in cd.columns:
    cd["signal_ts"] = pd.to_datetime(cd["signal_ts"], utc=True, errors="coerce")
    cd["date"] = cd["signal_ts"].dt.date
    print("=== Top dates by cooldown-unmatched count ===")
    print(cd["date"].value_counts().head(20))
    print()
else:
    print("No 'signal_ts' column in dataframe; cannot compute per-day stats.")
    print()

# 5. Show a few example rows with lock info
cols_to_show = [
    "symbol",
    "signal_ts",
    "exit_ts_live",
    "classification",
    "n_signals_in_window",
    "n_bt_trades_near_signal",
    "bt_ts_effective",
    "bt_decision",
    "bt_reason",
    "bt_lock_until",
    "bt_active_count",
]

print("=== Sample cooldown-unmatched rows (first 10) ===")
print(cd[cols_to_show].head(10))
print()

print("Hint: For deeper diagnosis, we should correlate these with results/lock_timeline.csv")
