# diagnose_required_spread.py
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

# --- Config ---
BT_TRADES_PATH = Path("results/trades.csv")
LIVE_TRADES_PATH = Path("results/livetrading.csv")
PARQUET_DIR = Path("/opt/parquet/5m")

def load_data():
    if not BT_TRADES_PATH.exists(): return None, None
    bt = pd.read_csv(BT_TRADES_PATH)
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True)
    bt["exit_ts"] = pd.to_datetime(bt["exit_ts"], utc=True)
    bt["symbol"] = bt["symbol"].astype(str).str.upper()
    
    if LIVE_TRADES_PATH.exists():
        live = pd.read_csv(LIVE_TRADES_PATH)
        live["symbol"] = live["Market"].astype(str).str.upper()
        traded_syms = set(live["symbol"].unique())
        zombies = bt[~bt["symbol"].isin(traded_syms)].copy()
    else:
        zombies = bt.copy()
    return zombies

def calculate_required_spread(row):
    sym = row["symbol"]
    entry_ts = row["entry_ts"]
    exit_ts = row["exit_ts"]
    sl_price = float(row["sl"])
    
    pq_path = PARQUET_DIR / f"{sym}.parquet"
    if not pq_path.exists(): return None
    
    df = pd.read_parquet(pq_path)
    if "timestamp" in df.columns: df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    
    # Get trade window
    mask = (df.index >= entry_ts) & (df.index <= exit_ts)
    trade_bars = df.loc[mask]
    
    if trade_bars.empty: return None
    
    # Find the lowest Low during the trade
    min_low = float(trade_bars["low"].min())
    
    # If Low <= SL, it should have stopped out even with 0 spread (Data Mismatch)
    if min_low <= sl_price:
        return 0.0
    
    # Calculate spread needed to bridge the gap
    # Bid_Low = Min_Low * (1 - Spread/2)
    # We want Bid_Low <= SL
    # Min_Low * (1 - Spread/2) <= SL
    # 1 - Spread/2 <= SL / Min_Low
    # Spread/2 >= 1 - (SL / Min_Low)
    # Spread >= 2 * (1 - SL / Min_Low)
    
    required_spread = 2 * (1 - (sl_price / min_low))
    return required_spread * 100 # In Percent

def main():
    print("Loading Zombie trades...")
    zombies = load_data()
    if zombies.empty:
        print("No zombies found.")
        return

    print(f"Calculating Required Spread for {len(zombies)} Zombies...")
    results = []
    
    for idx, row in zombies.iterrows():
        req = calculate_required_spread(row)
        if req is not None:
            results.append({
                "symbol": row["symbol"],
                "entry_ts": row["entry_ts"],
                "required_spread_pct": req
            })

    df = pd.DataFrame(results)
    
    print("\n=== REQUIRED SPREAD DIAGNOSIS ===")
    print(f"Total Analyzed: {len(df)}")
    
    # Buckets
    zero = df[df["required_spread_pct"] <= 0]
    tiny = df[(df["required_spread_pct"] > 0) & (df["required_spread_pct"] <= 0.2)]
    small = df[(df["required_spread_pct"] > 0.2) & (df["required_spread_pct"] <= 0.5)]
    medium = df[(df["required_spread_pct"] > 0.5) & (df["required_spread_pct"] <= 1.0)]
    large = df[df["required_spread_pct"] > 1.0]
    
    print(f"Already Hit SL (Data Mismatch): {len(zero)}")
    print(f"Needs <= 0.2% Spread: {len(tiny)}")
    print(f"Needs 0.2% - 0.5% Spread: {len(small)}")
    print(f"Needs 0.5% - 1.0% Spread: {len(medium)}")
    print(f"Needs > 1.0% Spread: {len(large)}")
    
    print("\nSample 'Small Spread' Candidates (0.2% - 0.5%):")
    print(small.head(10).to_string())

    df.to_csv("results/zombie_spread_analysis.csv", index=False)

if __name__ == "__main__":
    main()