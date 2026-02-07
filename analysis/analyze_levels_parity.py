# analyze_levels_parity.py
import pandas as pd
from pathlib import Path
import config as cfg
import numpy as np

def main():
    # 1. Load BT Trades
    bt_path = cfg.RESULTS_DIR / "trades.csv"
    if not bt_path.exists():
        print("trades.csv not found.")
        return
    bt = pd.read_csv(bt_path)
    bt["symbol"] = bt["symbol"].astype(str).str.upper()
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True)
    
    # 2. Load Live Trades
    live_path = cfg.RESULTS_DIR / "livetrading.csv"
    if not live_path.exists():
        print("livetrading.csv not found.")
        return
    live = pd.read_csv(live_path)
    live["symbol"] = live["Market"].astype(str).str.upper()
    # Live CSV "Trade time" is exit time
    live["exit_ts"] = pd.to_datetime(live["Trade time"], utc=True, errors="coerce")
    live = live.dropna(subset=["exit_ts"])
    
    # 3. Load Blocking Trades Report (to focus on the problem set)
    blockers_path = cfg.RESULTS_DIR / "blocking_trades_report.csv"
    if not blockers_path.exists():
        print("blocking_trades_report.csv not found.")
        return
    blockers = pd.read_csv(blockers_path)
    
    print(f"Analyzing levels for {len(blockers)} blocking trades...")
    
    results = []
    
    for _, row in blockers.iterrows():
        if pd.isna(row["blocking_trade_entry"]):
            continue
            
        sym = row["symbol"]
        bt_entry_ts = pd.to_datetime(row["blocking_trade_entry"], utc=True)
        
        # Find the specific BT trade row
        bt_trades_sym = bt[
            (bt["symbol"] == sym) & 
            (bt["entry_ts"] == bt_entry_ts)
        ]
        if bt_trades_sym.empty:
            continue
        bt_trade = bt_trades_sym.iloc[0]
        
        # Find matching Live trade (heuristic: exit after BT entry, closest in time)
        live_cands = live[live["symbol"] == sym].copy()
        live_cands = live_cands[live_cands["exit_ts"] > bt_entry_ts]
        
        if live_cands.empty:
            continue
            
        # Pick closest exit to BT Entry (assuming trade duration isn't infinite)
        # If BT held for 72h, Live might have exited in 2h.
        live_cands["diff"] = (live_cands["exit_ts"] - bt_entry_ts).abs()
        live_trade = live_cands.sort_values("diff").iloc[0]
        
        # Extract Data
        bt_entry_px = float(bt_trade["entry"])
        bt_atr = float(bt_trade["atr_at_entry"])
        bt_sl = float(bt_trade["sl"])
        bt_tp = float(bt_trade["tp"])
        side = bt_trade["side"]
        
        live_entry_px = float(live_trade["Entry Price"])
        live_exit_px = float(live_trade["Exit Price"])
        live_pnl = float(live_trade["Realized P&L"])
        
        # 1. Entry Price Discrepancy
        entry_diff_pct = (live_entry_px - bt_entry_px) / bt_entry_px * 100
        
        # 2. Did Live hit BT levels?
        hit_bt_sl = False
        hit_bt_tp = False
        
        if side == "long":
            if live_exit_px <= bt_sl: hit_bt_sl = True
            if live_exit_px >= bt_tp: hit_bt_tp = True
        else: # short
            if live_exit_px >= bt_sl: hit_bt_sl = True
            if live_exit_px <= bt_tp: hit_bt_tp = True
            
        results.append({
            "symbol": sym,
            "bt_entry_px": bt_entry_px,
            "live_entry_px": live_entry_px,
            "entry_diff_pct": entry_diff_pct,
            "bt_atr": bt_atr,
            "bt_sl": bt_sl,
            "bt_tp": bt_tp,
            "live_exit_px": live_exit_px,
            "live_pnl": live_pnl,
            "did_live_hit_bt_sl": hit_bt_sl,
            "did_live_hit_bt_tp": hit_bt_tp,
            "bt_reason": bt_trade["exit_reason"]
        })

    df = pd.DataFrame(results)
    
    print("\n=== Entry Price Discrepancies (Live vs BT) ===")
    print(df["entry_diff_pct"].describe())
    
    # Focus on trades where Live took a loss (SL) but BT held (Time)
    # This implies Live SL was tighter than BT SL
    tighter_live_sl = df[
        (df["live_pnl"] < 0) & 
        (df["did_live_hit_bt_sl"] == False) &
        (df["bt_reason"] == "time")
    ].copy()
    
    print(f"\n=== Trades where Live Stopped Out but BT Held (Count: {len(tighter_live_sl)}) ===")
    if not tighter_live_sl.empty:
        # Implied Live ATR: Assuming Live SL = Entry +/- 2.0 * ATR
        # ATR = |Entry - Exit| / 2.0
        tighter_live_sl["implied_live_atr"] = (tighter_live_sl["live_entry_px"] - tighter_live_sl["live_exit_px"]).abs() / 2.0
        tighter_live_sl["atr_ratio"] = tighter_live_sl["implied_live_atr"] / tighter_live_sl["bt_atr"]
        
        print(tighter_live_sl[["symbol", "bt_atr", "implied_live_atr", "atr_ratio", "bt_sl", "live_exit_px"]].head(10).to_string())
        
        print("\n=== Implied Live ATR vs BT ATR Ratio (Live/BT) ===")
        print(tighter_live_sl["atr_ratio"].describe())

    out_path = cfg.RESULTS_DIR / "levels_parity.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDetailed levels report saved to {out_path}")

if __name__ == "__main__":
    main()