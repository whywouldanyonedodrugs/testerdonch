# diagnose_daily_breakout.py
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

# --- Config ---
BT_TRADES_PATH = Path("results/trades.csv")
LIVE_TRADES_PATH = Path("results/livetrading.csv")
PARQUET_DIR = Path("parquet")

def load_data():
    if not BT_TRADES_PATH.exists(): return None, None
    bt = pd.read_csv(BT_TRADES_PATH)
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True)
    bt["symbol"] = bt["symbol"].astype(str).str.upper()
    
    # Identify Zombies (BT trades not in Live)
    if LIVE_TRADES_PATH.exists():
        live = pd.read_csv(LIVE_TRADES_PATH)
        live["symbol"] = live["Market"].astype(str).str.upper()
        traded_syms = set(live["symbol"].unique())
        zombies = bt[~bt["symbol"].isin(traded_syms)].copy()
    else:
        zombies = bt.copy()
        
    return zombies

def check_breakout_strength(symbol, entry_ts):
    # Load OHLCV
    pq_path = PARQUET_DIR / f"{symbol}.parquet"
    if not pq_path.exists(): return None
    
    df = pd.read_parquet(pq_path)
    if "timestamp" in df.columns: df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    # Resample to Daily
    daily = df.resample("1D").agg({"high": "max", "close": "last"}).dropna()
    
    # The breakout must have happened Yesterday (T-1) relative to entry
    entry_day = entry_ts.floor("D")
    breakout_day = entry_day - pd.Timedelta(days=1)
    
    if breakout_day not in daily.index: return None
    
    # Calculate Level (Max of 20 days ending T-2)
    # Range: [breakout_day - 20d, breakout_day - 1d]
    lookback_end = breakout_day - pd.Timedelta(days=1)
    lookback_start = breakout_day - pd.Timedelta(days=20)
    
    window = daily.loc[lookback_start:lookback_end]
    if window.empty: return None
    
    level = float(window["high"].max())
    close = float(daily.loc[breakout_day, "close"])
    
    # Margin: (Close - Level) / Level
    margin_pct = (close - level) / level * 100
    
    return {
        "symbol": symbol,
        "entry_ts": entry_ts,
        "breakout_day": breakout_day,
        "level": level,
        "close": close,
        "margin_pct": margin_pct
    }

def main():
    print("Loading Zombie trades...")
    zombies = load_data()
    if zombies is None or zombies.empty:
        print("No zombies found.")
        return

    print(f"Analyzing Daily Breakout Strength for {len(zombies)} Zombies...")
    
    results = []
    for idx, row in zombies.iterrows():
        res = check_breakout_strength(row["symbol"], row["entry_ts"])
        if res: results.append(res)

    df = pd.DataFrame(results)
    
    print("\n=== DAILY BREAKOUT MARGIN DIAGNOSIS ===")
    
    # Define "Weak" as < 0.3% (The margin we proposed earlier)
    weak = df[df["margin_pct"] < 0.3]
    negative = df[df["margin_pct"] <= 0] # Should not happen if Scout logic holds
    
    print(f"Total Analyzed: {len(df)}")
    print(f"Weak Breakouts (< 0.3%): {len(weak)} ({len(weak)/len(df)*100:.1f}%)")
    print(f"Negative Breakouts (Error?): {len(negative)}")
    
    if not weak.empty:
        print("\nSample Weak Breakouts:")
        print(weak[["symbol", "breakout_day", "level", "close", "margin_pct"]].head(10).to_string())

    df.to_csv("results/zombie_breakout_strength.csv", index=False)

if __name__ == "__main__":
    main()