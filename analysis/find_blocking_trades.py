# find_blocking_trades.py
import pandas as pd
from pathlib import Path
import config as cfg

def main():
    # Load skipped signals
    skipped = pd.read_csv(cfg.RESULTS_DIR / "throughput_analysis.csv")
    skipped = skipped[skipped["reason"] == "cooldown"].copy()
    skipped["signal_ts"] = pd.to_datetime(skipped["signal_ts"], utc=True)

    # Load backtest trades
    trades = pd.read_csv(cfg.RESULTS_DIR / "trades.csv")
    trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], utc=True)
    
    # Cooldown setting used in run
    COOLDOWN_MIN = 240 
    
    print(f"Checking {len(skipped)} skipped signals against {len(trades)} backtest trades...")
    
    results = []
    
    for _, row in skipped.iterrows():
        sym = row["symbol"]
        ts = row["signal_ts"]
        
        # Find trade for this symbol that covers this timestamp
        # Condition: Trade Entry <= Signal < (Trade Exit + Cooldown)
        # Note: The lock starts at Exit, but the "busy" state effectively starts at Entry
        
        blockers = trades[
            (trades["symbol"] == sym) &
            (trades["entry_ts"] <= ts) &
            (trades["exit_ts"] + pd.Timedelta(minutes=COOLDOWN_MIN) >= ts)
        ]
        
        if not blockers.empty:
            b = blockers.iloc[0]
            results.append({
                "symbol": sym,
                "skipped_signal_ts": ts,
                "blocking_trade_entry": b["entry_ts"],
                "blocking_trade_exit": b["exit_ts"],
                "blocking_trade_pnl": b["pnl"],
                "blocking_trade_reason": b["exit_reason"],
                "holding_hours": (b["exit_ts"] - b["entry_ts"]).total_seconds() / 3600
            })
        else:
            # Should not happen if logic holds, unless lock persisted from a previous run (unlikely)
            results.append({
                "symbol": sym,
                "skipped_signal_ts": ts,
                "blocking_trade_entry": "NOT FOUND",
                "note": "Ghost lock?"
            })

    df = pd.DataFrame(results)
    print(df.head(20).to_string())
    df.to_csv(cfg.RESULTS_DIR / "blocking_trades_report.csv", index=False)
    print(f"\nReport saved to {cfg.RESULTS_DIR / 'blocking_trades_report.csv'}")

if __name__ == "__main__":
    main()