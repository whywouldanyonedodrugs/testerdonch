# analyze_blocking_trade_fate.py
import pandas as pd
from pathlib import Path
import config as cfg

def main():
    # 1. Load the blocking trades identified in the previous step
    blockers_path = cfg.RESULTS_DIR / "blocking_trades_report.csv"
    if not blockers_path.exists():
        print("blocking_trades_report.csv not found. Run find_blocking_trades.py first.")
        return
    blockers = pd.read_csv(blockers_path)
    
    # 2. Load Live Trades
    live_path = cfg.RESULTS_DIR / "livetrading.csv"
    if not live_path.exists():
        print("livetrading.csv not found.")
        return
        
    live = pd.read_csv(live_path)
    live["symbol"] = live["Market"].astype(str).str.upper()
    # Parse live timestamps (handling potential format variations)
    live["exit_ts"] = pd.to_datetime(live["Trade time"], utc=True, errors="coerce")
    # Filter for valid trades
    live = live[live["Trade Type"] == "Trade"].dropna(subset=["exit_ts"]).copy()
    
    print(f"Loaded {len(blockers)} blocking trades and {len(live)} live trades.")
    
    results = []
    
    for _, b in blockers.iterrows():
        if pd.isna(b["blocking_trade_entry"]):
            continue
            
        sym = b["symbol"]
        bt_entry = pd.to_datetime(b["blocking_trade_entry"], utc=True)
        bt_exit = pd.to_datetime(b["blocking_trade_exit"], utc=True)
        
        # Find a live trade for this symbol that exited AFTER the backtest entry
        # and BEFORE the backtest exit.
        # This implies Live finished the trade while Backtest was still holding it.
        
        # Look for live exits in the window [BT Entry, BT Exit]
        candidates = live[
            (live["symbol"] == sym) & 
            (live["exit_ts"] >= bt_entry) & 
            (live["exit_ts"] <= bt_exit)
        ].copy()
        
        if candidates.empty:
            # Maybe Live didn't take it? Or exited slightly after?
            # Let's look for the closest exit to BT Entry
            candidates = live[live["symbol"] == sym].copy()
            if not candidates.empty:
                candidates["time_diff"] = (candidates["exit_ts"] - bt_entry).abs()
                closest = candidates.sort_values("time_diff").iloc[0]
                
                # If closest exit is way off (e.g. > 72h), likely unrelated
                if closest["time_diff"] > pd.Timedelta(hours=96):
                    match_status = "LIVE_DID_NOT_TRADE"
                    live_exit = None
                    live_pnl = None
                else:
                    match_status = "LIVE_EXITED_NEARBY"
                    live_exit = closest["exit_ts"]
                    live_pnl = closest["Realized P&L"]
            else:
                match_status = "NO_LIVE_HISTORY"
                live_exit = None
                live_pnl = None
        else:
            # We found live exits WITHIN the backtest holding period
            # This confirms Live exited faster.
            # Pick the earliest one after entry
            candidates = candidates.sort_values("exit_ts")
            match = candidates.iloc[0]
            match_status = "LIVE_EXITED_EARLIER"
            live_exit = match["exit_ts"]
            live_pnl = match["Realized P&L"]

        # Calculate duration difference
        bt_duration = (bt_exit - bt_entry).total_seconds() / 3600
        live_duration = (live_exit - bt_entry).total_seconds() / 3600 if live_exit else 0
        
        results.append({
            "symbol": sym,
            "bt_entry": bt_entry,
            "bt_exit": bt_exit,
            "bt_reason": b["blocking_trade_reason"],
            "bt_duration_h": round(bt_duration, 1),
            "live_status": match_status,
            "live_exit": live_exit,
            "live_duration_h": round(live_duration, 1) if live_exit else None,
            "duration_diff_h": round(bt_duration - live_duration, 1) if live_exit else None,
            "bt_pnl": b["blocking_trade_pnl"],
            "live_pnl": live_pnl
        })

    df = pd.DataFrame(results)
    
    # Filter for cases where Live exited earlier
    earlier = df[df["live_status"] == "LIVE_EXITED_EARLIER"].sort_values("duration_diff_h", ascending=False)
    
    print("\n=== Trades where Live exited faster than Backtest ===")
    print(earlier[["symbol", "bt_reason", "bt_duration_h", "live_duration_h", "duration_diff_h", "bt_pnl", "live_pnl"]].head(20).to_string())
    
    # Save full report
    out_path = cfg.RESULTS_DIR / "blocking_fate_analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull analysis saved to {out_path}")

if __name__ == "__main__":
    main()