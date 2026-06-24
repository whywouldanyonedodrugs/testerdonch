# diagnose_zombie_lifecycle.py
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

# --- Config ---
BT_TRADES_PATH = Path("results/trades.csv")
LIVE_TRADES_PATH = Path("results/livetrading.csv")
SIGNALS_DIR = Path("signals")
PARQUET_DIR = Path("/opt/parquet/5m")

def load_data():
    # Load Trades
    if not BT_TRADES_PATH.exists(): return None, None, None
    bt = pd.read_csv(BT_TRADES_PATH)
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True)
    bt["exit_ts"] = pd.to_datetime(bt["exit_ts"], utc=True)
    bt["symbol"] = bt["symbol"].astype(str).str.upper()

    # Load Live to identify Zombies
    if LIVE_TRADES_PATH.exists():
        live = pd.read_csv(LIVE_TRADES_PATH)
        live["symbol"] = live["Market"].astype(str).str.upper()
        # Simple heuristic: if live didn't trade this symbol, it's a zombie candidate
        traded_syms = set(live["symbol"].unique())
        zombies = bt[~bt["symbol"].isin(traded_syms)].copy()
    else:
        zombies = bt.copy()

    return bt, zombies

def get_signal_level(symbol, entry_ts):
    # Find the signal row to get the Donchian Level
    try:
        # Lazy load signal file for symbol
        p = list(SIGNALS_DIR.glob(f"symbol={symbol}/*.parquet"))
        if not p: return None
        df = pd.read_parquet(p[0])
        
        # Normalize cols
        ts_col = "timestamp" if "timestamp" in df.columns else "entry_ts"
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        
        # Find exact match
        row = df[df[ts_col] == entry_ts]
        if not row.empty:
            return float(row.iloc[0]["don_break_level"])
    except Exception:
        pass
    return None

def analyze_lifecycle(row):
    sym = row["symbol"]
    entry_ts = row["entry_ts"]
    exit_ts = row["exit_ts"]
    entry_px = float(row["entry"])
    sl_px = float(row["sl"])
    
    # 1. Get Breakout Level
    level = get_signal_level(sym, entry_ts)
    
    # 2. Load OHLCV for MAE
    mae_dist = 9999.0
    min_low = 9999.0
    
    pq_path = PARQUET_DIR / f"{sym}.parquet"
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
        if "timestamp" in df.columns: df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        
        # Slice trade duration
        mask = (df.index >= entry_ts) & (df.index <= exit_ts)
        trade_bars = df.loc[mask]
        
        if not trade_bars.empty:
            min_low = float(trade_bars["low"].min())
            # How close did we get to SL? (Positive = survived, Negative = hit)
            # For Longs: Low - SL
            mae_dist = min_low - sl_px

    return {
        "symbol": sym,
        "entry_ts": entry_ts,
        "entry_px": entry_px,
        "breakout_level": level,
        "breakout_margin_pct": ((entry_px - level) / level * 100) if level else None,
        "sl_price": sl_px,
        "min_low_during_trade": min_low,
        "dist_to_sl_pct": (mae_dist / entry_px * 100) if min_low != 9999.0 else None,
        "duration_h": (exit_ts - entry_ts).total_seconds() / 3600
    }

def main():
    print("Loading data...")
    bt, zombies = load_data()
    if zombies.empty:
        print("No zombies found.")
        return

    print(f"Analyzing {len(zombies)} Zombie trades...")
    results = []
    
    for idx, row in zombies.iterrows():
        res = analyze_lifecycle(row)
        results.append(res)

    df = pd.DataFrame(results)
    
    # Filter for valid analysis
    df = df.dropna(subset=["breakout_margin_pct", "dist_to_sl_pct"])
    
    print("\n=== ZOMBIE DIAGNOSIS ===")
    
    # 1. Entry Quality
    weak_entries = df[df["breakout_margin_pct"] < 0.1]
    print(f"\n[Hypothesis A] Weak Entries (< 0.1% above level):")
    print(f"Count: {len(weak_entries)} / {len(df)} ({len(weak_entries)/len(df)*100:.1f}%)")
    if not weak_entries.empty:
        print(weak_entries[["symbol", "breakout_margin_pct", "duration_h"]].head().to_string())

    # 2. Near-Death Experiences
    near_death = df[(df["dist_to_sl_pct"] > 0) & (df["dist_to_sl_pct"] < 0.2)]
    print(f"\n[Hypothesis B] Near-Miss Stop Loss (Survived by < 0.2%):")
    print(f"Count: {len(near_death)} / {len(df)} ({len(near_death)/len(df)*100:.1f}%)")
    if not near_death.empty:
        print(near_death[["symbol", "dist_to_sl_pct", "duration_h"]].head().to_string())

    df.to_csv("results/zombie_lifecycle.csv", index=False)
    print("\nSaved to results/zombie_lifecycle.csv")

if __name__ == "__main__":
    main()