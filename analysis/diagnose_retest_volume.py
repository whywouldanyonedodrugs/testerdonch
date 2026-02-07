# diagnose_retest_volume.py
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

# --- Config ---
BT_TRADES_PATH = Path("results/trades.csv")
LIVE_TRADES_PATH = Path("results/livetrading.csv")
PARQUET_DIR = Path("parquet")
SIGNALS_DIR = Path("signals")

def load_data():
    if not BT_TRADES_PATH.exists(): return None, None
    bt = pd.read_csv(BT_TRADES_PATH)
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True)
    bt["symbol"] = bt["symbol"].astype(str).str.upper()
    
    if LIVE_TRADES_PATH.exists():
        live = pd.read_csv(LIVE_TRADES_PATH)
        live["symbol"] = live["Market"].astype(str).str.upper()
        traded_syms = set(live["symbol"].unique())
        zombies = bt[~bt["symbol"].isin(traded_syms)].copy()
    else:
        zombies = bt.copy()
    return zombies

def get_signal_details(symbol, entry_ts):
    try:
        p = list(SIGNALS_DIR.glob(f"symbol={symbol}/*.parquet"))
        if not p: return None
        df = pd.read_parquet(p[0])
        ts_col = "timestamp" if "timestamp" in df.columns else "entry_ts"
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        
        row = df[df[ts_col] == entry_ts]
        if not row.empty:
            return row.iloc[0]
    except Exception:
        pass
    return None

def analyze_conditions(row):
    sym = row["symbol"]
    entry_ts = row["entry_ts"]
    
    # Get signal data (contains pre-calculated vol_mult, level)
    sig = get_signal_details(sym, entry_ts)
    if sig is None: return None
    
    level = float(sig["don_break_level"])
    
    # Load OHLCV for Retest Check
    pq_path = PARQUET_DIR / f"{sym}.parquet"
    if not pq_path.exists(): return None
    
    df = pd.read_parquet(pq_path)
    if "timestamp" in df.columns: df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    
    # Retest Logic (from scout.py)
    lb = int(cfg.RETEST_LOOKBACK_BARS)
    eps = float(cfg.RETEST_EPS_PCT)
    
    # Get window ending at entry
    if entry_ts not in df.index: return None
    idx = df.index.get_loc(entry_ts)
    start = max(0, idx - lb + 1)
    sub = df.iloc[start : idx + 1]
    
    band_hi = level * (1.0 + eps)
    band_lo = level * (1.0 - eps)
    
    # Calculate "Retest Margin"
    # How deep did we penetrate the band?
    # We want to know if we *barely* touched it.
    # Touched if Low <= Band_Hi AND High >= Band_Lo
    
    # Find the bar that satisfied the condition
    touches = sub[ (sub["low"] <= band_hi) & (sub["high"] >= band_lo) ]
    
    retest_margin = 0.0
    if not touches.empty:
        # Calculate how far inside the band we went (normalized by band width)
        # A value close to 0 means we barely touched the edge.
        # A value > 0 means we went deeper.
        
        # Distance from Band_Hi (for Lows)
        dist_hi = (band_hi - touches["low"].min()) / (band_hi - band_lo)
        # Distance from Band_Lo (for Highs)
        dist_lo = (touches["high"].max() - band_lo) / (band_hi - band_lo)
        
        retest_margin = max(dist_hi, dist_lo)

    # Volume Multiple
    # Recalculate to be sure, or use signal value if available
    # Scout doesn't save vol_mult in the final parquet usually, unless we added it.
    # Let's recalculate.
    bars_per_day = 288
    vol_days = int(cfg.VOL_LOOKBACK_DAYS)
    lookback_vol = min(9000, vol_days * bars_per_day)
    
    vol_sub = df.iloc[max(0, idx - lookback_vol) : idx + 1]["volume"]
    vol_med = vol_sub.median()
    cur_vol = df.iloc[idx]["volume"]
    vol_mult = cur_vol / vol_med if vol_med > 0 else 0.0
    
    return {
        "symbol": sym,
        "entry_ts": entry_ts,
        "vol_mult": vol_mult,
        "retest_margin": retest_margin,
        "is_borderline_vol": 2.0 <= vol_mult < 2.2,
        "is_borderline_retest": 0.0 < retest_margin < 0.1 # Bottom 10% of the band
    }

def main():
    print("Loading Zombie trades...")
    zombies = load_data()
    if zombies.empty:
        print("No zombies found.")
        return

    print(f"Analyzing Retest/Volume for {len(zombies)} Zombies...")
    results = []
    for idx, row in zombies.iterrows():
        res = analyze_conditions(row)
        if res: results.append(res)

    df = pd.DataFrame(results)
    
    print("\n=== BORDERLINE DIAGNOSIS ===")
    print(f"Total Analyzed: {len(df)}")
    
    borderline_vol = df[df["is_borderline_vol"]]
    print(f"Borderline Volume (2.0 <= x < 2.2): {len(borderline_vol)} ({len(borderline_vol)/len(df)*100:.1f}%)")
    
    borderline_retest = df[df["is_borderline_retest"]]
    print(f"Borderline Retest (Barely touched): {len(borderline_retest)} ({len(borderline_retest)/len(df)*100:.1f}%)")
    
    if not borderline_vol.empty:
        print("\nSample Borderline Volume:")
        print(borderline_vol[["symbol", "vol_mult"]].head().to_string())

    if not borderline_retest.empty:
        print("\nSample Borderline Retest:")
        print(borderline_retest[["symbol", "retest_margin"]].head().to_string())

    df.to_csv("results/zombie_retest_volume.csv", index=False)

if __name__ == "__main__":
    main()