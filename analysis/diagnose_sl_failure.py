# diagnose_sl_failure.py
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

# --- Config ---
BT_TRADES_PATH = Path("results/trades.csv")
PARQUET_DIR = Path("/opt/parquet/5m")

def load_data():
    if not BT_TRADES_PATH.exists(): return None
    bt = pd.read_csv(BT_TRADES_PATH)
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True)
    bt["exit_ts"] = pd.to_datetime(bt["exit_ts"], utc=True)
    bt["symbol"] = bt["symbol"].astype(str).str.upper()
    return bt

def analyze_failure(row):
    sym = row["symbol"]
    entry_ts = row["entry_ts"]
    exit_ts = row["exit_ts"]
    sl_recorded = float(row["sl"])
    entry_px = float(row["entry"])
    
    # Recalculate Initial SL to see if it moved
    # SL = Entry - (Mult * ATR)
    # We need ATR. It's in the CSV as 'atr_at_entry'
    atr = float(row["atr_at_entry"])
    sl_mult = float(cfg.SL_ATR_MULT)
    
    # Adjust for sensitivity if it was used in the run
    # We can't know for sure if SL_SENSITIVITY was 1.0 or 0.9 in the run, 
    # but we can infer from the recorded SL.
    # Expected SL (Standard)
    sl_standard_long = entry_px - (sl_mult * atr)
    sl_standard_short = entry_px + (sl_mult * atr) # Assuming long for now based on strategy
    
    is_moved = abs(sl_recorded - sl_standard_long) > (entry_px * 0.001)
    
    # Load OHLCV
    pq_path = PARQUET_DIR / f"{sym}.parquet"
    if not pq_path.exists(): return None
    
    df = pd.read_parquet(pq_path)
    if "timestamp" in df.columns: df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    
    # Check Entry Bar
    if entry_ts in df.index:
        entry_bar = df.loc[entry_ts]
        entry_low = float(entry_bar["low"])
        hit_on_entry = entry_low <= sl_recorded
    else:
        hit_on_entry = None

    # Check Subsequent Bars
    mask = (df.index > entry_ts) & (df.index <= exit_ts)
    walk = df.loc[mask]
    
    first_breach_ts = None
    min_low_val = 999999.0
    
    if not walk.empty:
        breaches = walk[walk["low"] <= sl_recorded]
        if not breaches.empty:
            first_breach_ts = breaches.index[0]
            min_low_val = float(breaches["low"].min())
        else:
            min_low_val = float(walk["low"].min())

    return {
        "symbol": sym,
        "entry_ts": entry_ts,
        "exit_reason": row["exit_reason"],
        "sl_recorded": sl_recorded,
        "sl_initial_calc": sl_standard_long,
        "sl_moved": is_moved,
        "hit_on_entry": hit_on_entry,
        "first_breach_ts": first_breach_ts,
        "min_low": min_low_val,
        "breach_gap_pct": (sl_recorded - min_low_val) / sl_recorded * 100 if min_low_val < 999999 else None
    }

def main():
    print("Loading trades...")
    bt = load_data()
    if bt is None: return

    # Filter for the 13 "Already Hit SL" candidates from previous step
    # We don't have the list, so we re-scan all trades that exited by TIME (Zombies)
    # or where we suspect a miss.
    # Let's just scan ALL trades where PnL < 0 and exit_reason != 'sl'
    
    suspects = bt[
        (bt["pnl"] < 0) & 
        (bt["exit_reason"] != "sl") & 
        (bt["exit_reason"] != "immediate_sl")
    ].copy()
    
    print(f"Analyzing {len(suspects)} suspect trades (Losses not exited via SL)...")
    
    results = []
    for idx, row in suspects.iterrows():
        res = analyze_failure(row)
        if res and (res["hit_on_entry"] or res["first_breach_ts"] is not None):
            results.append(res)

    df = pd.DataFrame(results)
    
    if df.empty:
        print("No SL failures found. The previous diagnosis might have used different parameters.")
        return

    print("\n=== SL FAILURE DIAGNOSIS ===")
    print(f"Total Failures Found: {len(df)}")
    
    entry_hits = df[df["hit_on_entry"] == True]
    print(f"Hit on Entry Bar: {len(entry_hits)}")
    
    later_hits = df[(df["hit_on_entry"] == False) & (df["first_breach_ts"].notna())]
    print(f"Hit on Later Bar: {len(later_hits)}")
    
    print("\nSample Entry Bar Hits:")
    if not entry_hits.empty:
        print(entry_hits[["symbol", "entry_ts", "sl_recorded", "exit_reason"]].head().to_string())
        
    print("\nSample Later Hits:")
    if not later_hits.empty:
        print(later_hits[["symbol", "entry_ts", "first_breach_ts", "sl_recorded", "exit_reason"]].head().to_string())

    df.to_csv("results/sl_failure_details.csv", index=False)

if __name__ == "__main__":
    main()