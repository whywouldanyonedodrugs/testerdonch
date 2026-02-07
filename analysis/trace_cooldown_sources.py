import pandas as pd

# 1. Load core unmatched LIVE trades with BT decisions
core_path = "results/core_unmatched_live_with_bt_decisions.csv"
core = pd.read_csv(core_path)

# Parse timestamps
core["signal_ts"] = pd.to_datetime(core["signal_ts"], utc=True, errors="coerce")

# Focus on cooldown-only cases
cd = core[core["bt_reason"] == "cooldown"].copy()
print("Cooldown-unmatched LIVE trades:", len(cd))
print()

# 2. Load lock timeline
locks_path = "results/lock_timeline.csv"
locks = pd.read_csv(locks_path)

print("Lock timeline shape:", locks.shape)
print("Lock timeline columns:", list(locks.columns))
print()

# Parse timestamps
locks["ts"] = pd.to_datetime(locks["ts"], utc=True, errors="coerce")
locks["lock_until"] = pd.to_datetime(locks["lock_until"], utc=True, errors="coerce")

# Only keep lock *updates* (when a new lock is set, typically on trade exit)
locks_upd = locks[locks["event"] == "update"].copy()

# 3. For each cooldown-unmatched trade, find last lock update before its signal_ts
rows = []

for idx, row in cd.iterrows():
    sym = row["symbol"]
    sig_ts = row["signal_ts"]

    # All lock updates for this symbol before or at the signal
    sub = locks_upd[locks_upd["symbol"] == sym]
    sub = sub[sub["ts"] <= sig_ts].sort_values("ts")

    if sub.empty:
        # No prior lock update found for this symbol
        source_ts = pd.NaT
        source_lock_until = pd.NaT
        reason = None
        minutes_since_lock_update = None
    else:
        last = sub.iloc[-1]
        source_ts = last["ts"]
        source_lock_until = last["lock_until"]
        reason = last.get("reason", None)
        delta = (sig_ts - source_ts).total_seconds() / 60.0
        minutes_since_lock_update = delta

    rows.append(
        {
            "symbol": sym,
            "signal_ts": sig_ts,
            "bt_lock_until_at_decision": row.get("bt_lock_until", None),
            "source_lock_ts": source_ts,
            "source_lock_until": source_lock_until,
            "source_lock_reason": reason,
            "minutes_since_lock_update": minutes_since_lock_update,
        }
    )

result = pd.DataFrame(rows)

print("=== First 20 cooldown-unmatched LIVE trades with their lock sources ===")
print(
    result.sort_values(["symbol", "signal_ts"])
          .head(20)
          .to_string(index=False)
)
print()

print("=== Summary: delay between lock update and skipped signal (minutes) ===")
print(result["minutes_since_lock_update"].describe())
print()

print("=== Top symbols by number of distinct lock updates causing cooldown-unmatched trades ===")
print(
    result.groupby("symbol")["source_lock_ts"]
          .nunique()
          .sort_values(ascending=False)
          .head(20)
)
print()

print("Hint:")
print("- Each distinct 'source_lock_ts' per symbol is a trade exit that seeded a cooldown.")
print("- If many cooldown-unmatched trades share the same 'source_lock_ts',")
print("  that one prior trade (possibly BT-only) is responsible for multiple parity misses.")
