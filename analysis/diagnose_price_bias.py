# diagnose_price_bias.py
import pandas as pd
import numpy as np
from pathlib import Path

# --- Config ---
BT_TRADES_PATH = Path("results/trades.csv")
LIVE_TRADES_PATH = Path("results/livetrading.csv")

def main():
    if not BT_TRADES_PATH.exists() or not LIVE_TRADES_PATH.exists():
        print("Missing trade files.")
        return

    # Load
    bt = pd.read_csv(BT_TRADES_PATH)
    live = pd.read_csv(LIVE_TRADES_PATH)
    
    # Normalize
    bt["symbol"] = bt["symbol"].astype(str).str.upper()
    live["symbol"] = live["Market"].astype(str).str.upper()
    
    # Parse Live TS
    live["ts"] = pd.to_datetime(live["Trade time"], utc=True, errors="coerce")
    if live["ts"].isna().any():
        live.loc[live["ts"].isna(), "ts"] = pd.to_datetime(
            live.loc[live["ts"].isna(), "Trade time"], 
            format="%d/%m/%Y %H:%M", utc=True, errors="coerce"
        )
    
    # Match on Symbol + Exit Time (approximate)
    # We use Exit Time because Live "Trade time" is exit time
    bt["exit_ts"] = pd.to_datetime(bt["exit_ts"], utc=True)
    
    matched = []
    for idx, l_row in live.iterrows():
        sym = l_row["symbol"]
        l_exit = l_row["ts"]
        l_entry_px = float(l_row["Entry Price"])
        
        # Find BT trade for same symbol exiting within 2 hours
        candidates = bt[
            (bt["symbol"] == sym) &
            (bt["exit_ts"] >= l_exit - pd.Timedelta(hours=2)) &
            (bt["exit_ts"] <= l_exit + pd.Timedelta(hours=2))
        ]
        
        if not candidates.empty:
            # Pick closest
            candidates = candidates.copy()
            candidates["diff"] = (candidates["exit_ts"] - l_exit).abs()
            best = candidates.sort_values("diff").iloc[0]
            
            b_entry_px = float(best["entry"])
            
            # Calculate Bias: (Backtest - Live) / Live
            bias_pct = (b_entry_px - l_entry_px) / l_entry_px * 100
            
            matched.append({
                "symbol": sym,
                "live_entry": l_entry_px,
                "bt_entry": b_entry_px,
                "bias_pct": bias_pct
            })

    df = pd.DataFrame(matched)
    
    if df.empty:
        print("No matched trades found to analyze bias.")
        return

    print("\n=== DATA BIAS DIAGNOSIS ===")
    print(f"Matched Trades: {len(df)}")
    print(f"Mean Bias (BT - Live): {df['bias_pct'].mean():.4f}%")
    print(f"Median Bias: {df['bias_pct'].median():.4f}%")
    print(f"Std Dev: {df['bias_pct'].std():.4f}%")
    
    print("\nDistribution:")
    print(f"BT > Live (> 0.1%): {len(df[df['bias_pct'] > 0.1])}")
    print(f"BT < Live (< -0.1%): {len(df[df['bias_pct'] < -0.1])}")
    print(f"Close (~0%): {len(df[(df['bias_pct'] >= -0.1) & (df['bias_pct'] <= 0.1)])}")
    
    print("\nSample Biases:")
    print(df.head(10).to_string())
    
    df.to_csv("results/price_bias.csv", index=False)

if __name__ == "__main__":
    main()