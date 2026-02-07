# analyze_throughput_causes.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
import config as cfg

def main():
    results_dir = Path(getattr(cfg, "RESULTS_DIR", "results"))
    
    # 1. Load Core Unmatched Analysis
    core_path = results_dir / "parity_unmatched_live_core.csv"
    if not core_path.exists():
        print(f"Error: {core_path} not found. Run analyze_core_unmatched_live.py first.")
        return
    core = pd.read_csv(core_path)
    
    # Filter for signal_no_bt_trade
    targets = core[core["classification"] == "signal_no_bt_trade"].copy()
    if targets.empty:
        print("No 'signal_no_bt_trade' cases found in core analysis.")
        return
        
    print(f"Analyzing {len(targets)} core unmatched trades (signal_no_bt_trade)...")
    
    # 2. Load Decision Log
    dec_path = results_dir / "signal_decisions.csv"
    if not dec_path.exists():
        print(f"Error: {dec_path} not found.")
        return
    dec = pd.read_csv(dec_path)
    # Ensure timestamps match format
    dec["signal_ts"] = pd.to_datetime(dec["signal_ts"], utc=True)
    targets["signal_ts"] = pd.to_datetime(targets["signal_ts"], utc=True)
    
    # Join to get the reason
    merged = pd.merge(targets, dec, on=["symbol", "signal_ts"], how="left", suffixes=("", "_bt"))
    
    # 3. Load Lock Timeline
    lock_path = results_dir / "lock_timeline.csv"
    has_locks = lock_path.exists()
    locks = pd.DataFrame()
    if has_locks:
        locks = pd.read_csv(lock_path)
        locks["ts"] = pd.to_datetime(locks["ts"], utc=True)
        locks["lock_until"] = pd.to_datetime(locks["lock_until"], utc=True)
        locks = locks.sort_values("ts")
    
    # 4. Analyze each skipped trade
    explanations = []
    
    for idx, row in merged.iterrows():
        reason = row.get("reason", "unknown")
        expl = {"symbol": row["symbol"], "signal_ts": row["signal_ts"], "reason": reason}
        
        if reason == "cooldown" and has_locks:
            # Find the lock that caused this
            sym_locks = locks[locks["symbol"] == row["symbol"]]
            # Find the most recent update BEFORE this signal
            relevant_update = sym_locks[
                (sym_locks["event"] == "update") & 
                (sym_locks["ts"] <= row["signal_ts"])
            ].last_valid_index()
            
            if relevant_update is not None:
                upd = sym_locks.loc[relevant_update]
                expl["lock_set_at"] = upd["ts"]
                expl["lock_until"] = upd["lock_until"]
                expl["minutes_since_lock_set"] = (row["signal_ts"] - upd["ts"]).total_seconds() / 60.0
                expl["lock_duration_applied"] = (upd["lock_until"] - upd["ts"]).total_seconds() / 60.0
            else:
                expl["lock_error"] = "No prior lock update found"
                
        explanations.append(expl)
        
    # 5. Summarize
    df_expl = pd.DataFrame(explanations)
    print("\n=== Skip Reasons ===")
    print(df_expl["reason"].value_counts())
    
    cooldowns = df_expl[df_expl["reason"] == "cooldown"]
    if not cooldowns.empty and "minutes_since_lock_set" in cooldowns.columns:
        print("\n=== Cooldown Analysis (for 'cooldown' skips) ===")
        print(cooldowns[["minutes_since_lock_set", "lock_duration_applied"]].describe())
        
        # Check if any were skipped despite being outside the window (bug check)
        # Note: lock_until is the deadline. signal_ts <= lock_until means skipped.
        # We want to see how deep into the lock they were.
        
        print("\nSample of lock explanations:")
        print(cooldowns.head(5).to_string())
        
    out_path = results_dir / "throughput_analysis.csv"
    df_expl.to_csv(out_path, index=False)
    print(f"\nDetailed analysis saved to {out_path}")

if __name__ == "__main__":
    main()