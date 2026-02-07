# diagnose_lag_structure.py
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

# --- Config ---
BT_TRADES_PATH = Path("results/trades.csv")
SIGNALS_DIR = Path("signals")
PARQUET_DIR = Path("parquet")

def load_zombies():
    # Load trades and filter for those identified as Zombies in previous steps
    # (For simplicity, we'll just check ALL backtest trades to see the pattern)
    if not BT_TRADES_PATH.exists():
        print("No trades.csv found.")
        return pd.DataFrame()
    
    df = pd.read_csv(BT_TRADES_PATH)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df

def get_signal_level(symbol, entry_ts):
    # Retrieve the actual level used by the backtester from the signals file
    try:
        p = list(SIGNALS_DIR.glob(f"symbol={symbol}/*.parquet"))
        if not p: return None
        sig_df = pd.read_parquet(p[0])
        
        ts_col = "timestamp" if "timestamp" in sig_df.columns else "entry_ts"
        sig_df[ts_col] = pd.to_datetime(sig_df[ts_col], utc=True)
        
        row = sig_df[sig_df[ts_col] == entry_ts]
        if not row.empty:
            return float(row.iloc[0]["don_break_level"])
    except Exception:
        pass
    return None

def calculate_lags(symbol, entry_ts):
    # Load OHLCV
    pq_path = PARQUET_DIR / f"{symbol}.parquet"
    if not pq_path.exists(): return None, None
    
    df = pd.read_parquet(pq_path)
    if "timestamp" in df.columns: df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    # Resample to Daily Highs
    daily_highs = df["high"].resample("1D").max()
    
    # Entry Day
    entry_day = entry_ts.floor("D")
    
    # T-1 Level: Max of 20 days ending Yesterday (entry_day - 1d)
    # Range: [entry_day - 20d, entry_day - 1d]
    t1_end = entry_day - pd.Timedelta(days=1)
    t1_start = entry_day - pd.Timedelta(days=20)
    t1_slice = daily_highs[t1_start:t1_end]
    t1_level = float(t1_slice.max()) if len(t1_slice) > 0 else None
    
    # T-2 Level: Max of 20 days ending Day Before Yesterday (entry_day - 2d)
    # Range: [entry_day - 21d, entry_day - 2d]
    t2_end = entry_day - pd.Timedelta(days=2)
    t2_start = entry_day - pd.Timedelta(days=21)
    t2_slice = daily_highs[t2_start:t2_end]
    t2_level = float(t2_slice.max()) if len(t2_slice) > 0 else None
    
    return t1_level, t2_level

def main():
    print("Loading trades...")
    trades = load_zombies()
    if trades.empty: return

    print(f"Analyzing Lag Structure for {len(trades)} trades...")
    
    results = []
    
    for idx, row in trades.iterrows():
        sym = row["symbol"]
        ts = row["entry_ts"]
        
        used_level = get_signal_level(sym, ts)
        if used_level is None: continue
        
        t1, t2 = calculate_lags(sym, ts)
        
        if t1 is None or t2 is None: continue
        
        # Check matches (within floating point tolerance)
        is_t1 = abs(used_level - t1) < (t1 * 0.0001)
        is_t2 = abs(used_level - t2) < (t2 * 0.0001)
        
        match_type = "UNKNOWN"
        if is_t1 and is_t2: match_type = "AMBIGUOUS (T1=T2)"
        elif is_t1: match_type = "T-1 (Correct)"
        elif is_t2: match_type = "T-2 (Lagged)"
        
        results.append({
            "symbol": sym,
            "entry_ts": ts,
            "used_level": used_level,
            "t1_level": t1,
            "t2_level": t2,
            "match_type": match_type
        })

    df = pd.DataFrame(results)
    
    print("\n=== LAG STRUCTURE DIAGNOSIS ===")
    print(df["match_type"].value_counts())
    
    t2_cases = df[df["match_type"] == "T-2 (Lagged)"]
    if not t2_cases.empty:
        print("\nSample T-2 Cases (Scout used older level):")
        print(t2_cases[["symbol", "entry_ts", "used_level", "t1_level"]].head().to_string())
        
    df.to_csv("results/lag_diagnosis.csv", index=False)
    print("\nSaved to results/lag_diagnosis.csv")

if __name__ == "__main__":
    main()