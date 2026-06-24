# diagnose_zombie_levels.py
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

# --- Config ---
BT_TRADES_PATH = Path("results/trades.csv")
LIVE_TRADES_PATH = Path("results/livetrading.csv")
PARQUET_DIR = Path("/opt/parquet/5m")
SIGNALS_DIR = Path("signals")

def load_data():
    # 1. Load Backtest Trades
    if not BT_TRADES_PATH.exists(): return None, None
    bt = pd.read_csv(BT_TRADES_PATH)
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True)
    bt["symbol"] = bt["symbol"].astype(str).str.upper()
    
    # 2. Identify Zombies (BT trades not in Live)
    if LIVE_TRADES_PATH.exists():
        live = pd.read_csv(LIVE_TRADES_PATH)
        live["symbol"] = live["Market"].astype(str).str.upper()
        # Simple filter: symbols Live never traded are definitely Zombies
        # (A more precise time-match was done in previous scripts, 
        # here we just want a sample of definitely-missed trades)
        traded_syms = set(live["symbol"].unique())
        zombies = bt[~bt["symbol"].isin(traded_syms)].copy()
    else:
        zombies = bt.copy()
        
    return zombies

def calculate_levels(symbol, entry_ts):
    # Load OHLCV
    pq_path = PARQUET_DIR / f"{symbol}.parquet"
    if not pq_path.exists(): return None, None
    
    df = pd.read_parquet(pq_path)
    if "timestamp" in df.columns: df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    # Resample to Daily Highs
    daily_highs = df["high"].resample("1D").max()
    
    entry_day = entry_ts.floor("D")
    
    # T-1 Level: Max of 20 days ending Yesterday
    t1_slice = daily_highs[entry_day - pd.Timedelta(days=20) : entry_day - pd.Timedelta(days=1)]
    t1_level = float(t1_slice.max()) if len(t1_slice) > 0 else None
    
    # T-2 Level: Max of 20 days ending Day Before Yesterday
    t2_slice = daily_highs[entry_day - pd.Timedelta(days=21) : entry_day - pd.Timedelta(days=2)]
    t2_level = float(t2_slice.max()) if len(t2_slice) > 0 else None
    
    return t1_level, t2_level

def main():
    print("Loading Zombie trades...")
    zombies = load_data()
    if zombies is None or zombies.empty:
        print("No zombies found.")
        return

    print(f"Analyzing {len(zombies)} Zombie trades for Level Logic...")
    
    results = []
    
    for idx, row in zombies.iterrows():
        sym = row["symbol"]
        entry_px = float(row["entry"])
        ts = row["entry_ts"]
        
        t1, t2 = calculate_levels(sym, ts)
        
        if t1 is None or t2 is None: continue
        
        # Classification
        # Did we beat T-2? (Scout Logic)
        beat_t2 = entry_px > t2
        # Did we beat T-1? (Hypothetical Live Logic)
        beat_t1 = entry_px > t1
        
        category = "UNKNOWN"
        if beat_t2 and not beat_t1:
            category = "LAG_TRAP (Beat T-2, Failed T-1)"
        elif beat_t2 and beat_t1:
            category = "VALID_BREAKOUT (Beat Both)"
        elif not beat_t2:
            category = "NOISE (Failed Both)"
            
        results.append({
            "symbol": sym,
            "entry_ts": ts,
            "entry_px": entry_px,
            "t2_level": t2,
            "t1_level": t1,
            "category": category
        })

    df = pd.DataFrame(results)
    
    print("\n=== ZOMBIE LEVEL DIAGNOSIS ===")
    print(df["category"].value_counts())
    
    print("\n=== SAMPLE: LAG TRAPS (Live Bot likely used T-1) ===")
    traps = df[df["category"].str.contains("LAG_TRAP")]
    if not traps.empty:
        print(traps[["symbol", "entry_px", "t2_level", "t1_level"]].head(10).to_string())
        
    print("\n=== SAMPLE: VALID BREAKOUTS (Data/Spread Issue) ===")
    valid = df[df["category"].str.contains("VALID")]
    if not valid.empty:
        print(valid[["symbol", "entry_px", "t2_level", "t1_level"]].head(10).to_string())

    df.to_csv("results/zombie_level_diagnosis.csv", index=False)

if __name__ == "__main__":
    main()