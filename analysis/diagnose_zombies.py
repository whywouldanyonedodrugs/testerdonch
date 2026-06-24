# diagnose_zombies.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import config as cfg

# --- Configuration ---
PARQUET_DIR = Path("/opt/parquet/5m")
BT_TRADES_PATH = Path("results/trades.csv")
LIVE_TRADES_PATH = Path("results/livetrading.csv")

# Thresholds to test against
MIN_AGE_DAYS = 34
MIN_LIQ_USD = 500_000
SPREAD_SIM = 0.002  # 0.2% spread simulation
SL_MULT = 2.0       # Standard SL multiplier

def load_data():
    # 1. Load Backtest Trades
    if not BT_TRADES_PATH.exists():
        print("No backtest trades found.")
        return None, None
    bt = pd.read_csv(BT_TRADES_PATH)
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True)
    bt["symbol"] = bt["symbol"].astype(str).str.upper()
    
    # 2. Load Live Trades
    if not LIVE_TRADES_PATH.exists():
        print("No live trades found.")
        live = pd.DataFrame()
    else:
        live = pd.read_csv(LIVE_TRADES_PATH)
        # Try parsing live timestamps
        live["ts"] = pd.to_datetime(live["Trade time"], utc=True, errors="coerce")
        # Fallback for day-first format if needed
        if live["ts"].isna().any():
            live.loc[live["ts"].isna(), "ts"] = pd.to_datetime(
                live.loc[live["ts"].isna(), "Trade time"], 
                format="%d/%m/%Y %H:%M", utc=True, errors="coerce"
            )
        live["symbol"] = live["Market"].astype(str).str.upper()

    return bt, live

def identify_zombies(bt, live):
    # A Zombie is a BT trade with no matching Live trade within 2 hours
    zombies = []
    
    for idx, row in bt.iterrows():
        sym = row["symbol"]
        entry_time = row["entry_ts"]
        
        # Look for match in Live
        if not live.empty:
            # Live "Trade time" is usually Exit time, but we check if ANY trade 
            # for this symbol happened roughly around the BT duration
            # This is a heuristic: if Live didn't trade this symbol +/- 12h of BT entry, it's a Zombie
            window_start = entry_time - timedelta(hours=12)
            window_end = entry_time + timedelta(hours=72)
            
            match = live[
                (live["symbol"] == sym) & 
                (live["ts"] >= window_start) & 
                (live["ts"] <= window_end)
            ]
            
            if not match.empty:
                continue # Found a match, not a zombie

        zombies.append(row)
        
    return pd.DataFrame(zombies)

def analyze_zombie(row):
    sym = row["symbol"]
    entry_ts = row["entry_ts"]
    entry_px = float(row["entry"])
    atr = float(row["atr_at_entry"]) if "atr_at_entry" in row else 0.0
    
    # Load OHLCV
    pq_path = PARQUET_DIR / f"{sym}.parquet"
    if not pq_path.exists():
        return {"reason": "MISSING_DATA"}
    
    try:
        df = pd.read_parquet(pq_path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
    except Exception:
        return {"reason": "CORRUPT_DATA"}

    # 1. Check Age (Implicit Warmup)
    first_ts = df.index[0]
    age_days = (entry_ts - first_ts).days
    if age_days < MIN_AGE_DAYS:
        return {"reason": "TOO_YOUNG", "details": f"Age {age_days}d < {MIN_AGE_DAYS}d"}

    # 2. Check Daily Liquidity (Universe Lag)
    # Sum volume for the 24h prior to entry
    start_24h = entry_ts - timedelta(hours=24)
    slice_24h = df.loc[start_24h:entry_ts]
    if slice_24h.empty:
        return {"reason": "NO_RECENT_DATA"}
    
    # Approx USD Vol = Sum(Vol) * Avg(Close)
    vol_sum = slice_24h["volume"].sum()
    avg_close = slice_24h["close"].mean()
    usd_vol = vol_sum * avg_close
    
    if usd_vol < MIN_LIQ_USD:
        return {"reason": "LOW_LIQUIDITY", "details": f"${usd_vol:,.0f} < ${MIN_LIQ_USD:,.0f}"}

    # 3. Check Immediate Stop (Spread/Wick)
    # Did the entry candle wick down enough to hit SL if we account for spread?
    if entry_ts in df.index:
        bar = df.loc[entry_ts]
        low = float(bar["low"])
        
        # Standard SL
        sl_price = entry_px - (SL_MULT * atr)
        
        # Bid Price at Low (Simulated)
        bid_low = low * (1 - SPREAD_SIM/2)
        
        if bid_low <= sl_price:
             return {"reason": "IMMEDIATE_STOP", "details": f"BidLow {bid_low:.4f} <= SL {sl_price:.4f}"}

    return {"reason": "UNKNOWN_DIVERGENCE", "details": "Data matches, logic mismatch?"}

def main():
    print("Loading trades...")
    bt, live = load_data()
    if bt is None: return

    print(f"Backtest Trades: {len(bt)}")
    print(f"Live Trades: {len(live)}")

    print("Identifying Zombies (BT trades with no Live counterpart)...")
    zombies = identify_zombies(bt, live)
    print(f"Found {len(zombies)} Zombies.")
    
    if zombies.empty:
        return

    print("Diagnosing Zombies (checking Age, Liquidity, and Wicks)...")
    results = []
    for idx, row in zombies.iterrows():
        res = analyze_zombie(row)
        res["symbol"] = row["symbol"]
        res["entry_ts"] = row["entry_ts"]
        results.append(res)

    df_res = pd.DataFrame(results)
    
    print("\n=== DIAGNOSIS REPORT ===")
    print(df_res["reason"].value_counts())
    
    print("\n=== SAMPLE DETAILS ===")
    print(df_res.head(10).to_string())
    
    df_res.to_csv("results/zombie_diagnosis.csv", index=False)
    print("\nFull report saved to results/zombie_diagnosis.csv")

if __name__ == "__main__":
    main()