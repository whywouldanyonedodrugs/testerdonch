# diagnose_live_entry_levels.py
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

# --- Config ---
LIVE_TRADES_PATH = Path("results/livetrading.csv")
PARQUET_DIR = Path("parquet")

def load_live_trades():
    if not LIVE_TRADES_PATH.exists():
        print("No live trades found.")
        return pd.DataFrame()
    
    df = pd.read_csv(LIVE_TRADES_PATH)
    # Parse timestamps
    df["ts"] = pd.to_datetime(df["Trade time"], utc=True, errors="coerce")
    if df["ts"].isna().any():
        df.loc[df["ts"].isna(), "ts"] = pd.to_datetime(
            df.loc[df["ts"].isna(), "Trade time"], 
            format="%d/%m/%Y %H:%M", utc=True, errors="coerce"
        )
    
    df["symbol"] = df["Market"].astype(str).str.upper()
    df["entry_price"] = pd.to_numeric(df["Entry Price"], errors="coerce")
    
    # Filter for valid entries
    df = df.dropna(subset=["ts", "entry_price"])
    return df

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
    print("Loading Live trades...")
    live = load_live_trades()
    if live.empty: return

    print(f"Analyzing {len(live)} Live trades for Level Logic...")
    
    results = []
    
    for idx, row in live.iterrows():
        sym = row["symbol"]
        entry_px = row["entry_price"]
        ts = row["ts"]
        
        t1, t2 = calculate_levels(sym, ts)
        
        if t1 is None or t2 is None: continue
        
        # Classification
        beat_t1 = entry_px > t1
        beat_t2 = entry_px > t2
        
        category = "UNKNOWN"
        if beat_t1:
            category = "VALID_T1 (Strict)"
        elif beat_t2:
            category = "VALID_T2_ONLY (Loose)"
        else:
            category = "BELOW_BOTH (Data Mismatch)"
            
        results.append({
            "symbol": sym,
            "entry_ts": ts,
            "entry_px": entry_px,
            "t1_level": t1,
            "t2_level": t2,
            "category": category
        })

    df = pd.DataFrame(results)
    
    print("\n=== LIVE TRADE LEVEL ANALYSIS ===")
    print(df["category"].value_counts())
    
    print("\n=== SAMPLE: VALID_T2_ONLY (Trades that fail T-1 check) ===")
    t2_only = df[df["category"] == "VALID_T2_ONLY (Loose)"]
    if not t2_only.empty:
        print(t2_only[["symbol", "entry_px", "t1_level", "t2_level"]].head(10).to_string())
        
    df.to_csv("results/live_level_diagnosis.csv", index=False)

if __name__ == "__main__":
    main()