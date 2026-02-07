# diagnose_missed_trades.py
import pandas as pd
from pathlib import Path

def main():
    # 1. Load the list of trades Live took but BT missed
    unmatched_path = Path("results/core_unmatched_live_analysis.csv")
    if not unmatched_path.exists():
        print("results/core_unmatched_live_analysis.csv not found. Run analyze_core_unmatched_live.py first.")
        return

    unmatched = pd.read_csv(unmatched_path)
    # We only care about cases where a signal existed but BT didn't trade
    target = unmatched[unmatched["classification"] == "signal_no_bt_trade"].copy()
    
    if target.empty:
        print("No 'signal_no_bt_trade' cases found to diagnose.")
        return

    # 2. Load the Backtester's decision log
    decisions_path = Path("results/signal_decisions.csv")
    if not decisions_path.exists():
        print("results/signal_decisions.csv not found. Run backtester.py first.")
        return

    print(f"Loading decision log...")
    decisions = pd.read_csv(decisions_path)
    decisions["signal_ts"] = pd.to_datetime(decisions["signal_ts"], utc=True)
    decisions["symbol"] = decisions["symbol"].astype(str).str.upper()

    print(f"Analyzing {len(target)} trades where Live traded but BT skipped...")

    reasons = []
    decision_timestamps = []

    for idx, row in target.iterrows():
        sym = row["symbol"]
        # Live trade exit time
        exit_ts = pd.to_datetime(row["exit_ts_live"], utc=True)
        
        # We need to find the ENTRY signal that corresponds to this exit.
        # We look for the last signal decision for this symbol BEFORE the exit.
        # (Live trades can last up to 72h, so look back 4 days to be safe)
        
        candidates = decisions[
            (decisions["symbol"] == sym) &
            (decisions["signal_ts"] < exit_ts) &
            (decisions["signal_ts"] > exit_ts - pd.Timedelta(days=4))
        ].sort_values("signal_ts")
        
        if candidates.empty:
            reasons.append("NO_SIGNAL_FOUND_IN_LOG")
            decision_timestamps.append(None)
            continue
            
        # The most likely signal is the one closest to the actual trade execution.
        # However, if there are multiple, the last one before exit is the best guess 
        # for the one that *should* have triggered the trade.
        last_decision = candidates.iloc[-1]
        
        status = f"{last_decision['decision']}:{last_decision['reason']}"
        reasons.append(status)
        decision_timestamps.append(last_decision["signal_ts"])

    target["bt_reason"] = reasons
    target["bt_signal_ts"] = decision_timestamps
    
    # 3. Summary
    print("\n=== WHY BACKTESTER SKIPPED THESE TRADES ===")
    print(target["bt_reason"].value_counts())
    
    # 4. Save detailed report
    out_path = Path("results/missed_trades_diagnosis.csv")
    target.to_csv(out_path, index=False)
    print(f"\nDetailed report saved to {out_path}")

if __name__ == "__main__":
    main()