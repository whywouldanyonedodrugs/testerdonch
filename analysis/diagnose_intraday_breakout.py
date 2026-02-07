# diagnose_intraday_breakout.py
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

# --- Config ---
BT_TRADES_PATH = Path("results/trades.csv")
LIVE_TRADES_PATH = Path("results/livetrading.csv")
SIGNALS_DIR = Path("signals")

def load_data():
    if not BT_TRADES_PATH.exists(): return None, None
    bt = pd.read_csv(BT_TRADES_PATH)
    bt["entry_ts"] = pd.to_datetime(bt["entry_ts"], utc=True)
    bt["symbol"] = bt["symbol"].astype(str).str.upper()
    
    if LIVE_TRADES_PATH.exists():
        live = pd.read_csv(LIVE_TRADES_PATH)
        live["symbol"] = live["Market"].astype(str).str.upper()
        
        # Match trades to find unmatched BT trades (Zombies)
        # We use a loose time window to exclude matched trades
        bt["is_matched"] = False
        for idx, row in bt.iterrows():
            sym = row["symbol"]
            ts = row["entry_ts"]
            # Check if Live traded this symbol within +/- 4 hours of entry
            match = live[
                (live["symbol"] == sym) & 
                (live["Trade time"] >= (ts - pd.Timedelta(hours=4)).isoformat()) &
                (live["Trade time"] <= (ts + pd.Timedelta(hours=72)).isoformat()) # Live trade time is exit
            ]
            if not match.empty:
                bt.at[idx, "is_matched"] = True
                
        zombies = bt[~bt["is_matched"]].copy()
    else:
        zombies = bt.copy()
        
    return zombies

def get_breakout_level(symbol, entry_ts):
    try:
        p = list(SIGNALS_DIR.glob(f"symbol={symbol}/*.parquet"))
        if not p: return None
        df = pd.read_parquet(p[0])
        ts_col = "timestamp" if "timestamp" in df.columns else "entry_ts"
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
        
        row = df[df[ts_col] == entry_ts]
        if not row.empty:
            return float(row.iloc[0]["don_break_level"])
    except Exception:
        pass
    return None

def main():
    print("Loading data...")
    zombies = load_data()
    if zombies is None or zombies.empty:
        print("No zombies found.")
        return

    print(f"Analyzing Intraday Breakout Strength for {len(zombies)} Zombies...")
    
    results = []
    for idx, row in zombies.iterrows():
        level = get_breakout_level(row["symbol"], row["entry_ts"])
        if level:
            entry_px = float(row["entry"])
            # Margin: (Entry - Level) / Level
            margin_pct = (entry_px - level) / level * 100
            results.append({
                "symbol": row["symbol"],
                "entry_ts": row["entry_ts"],
                "entry_px": entry_px,
                "level": level,
                "margin_pct": margin_pct
            })

    df = pd.DataFrame(results)
    
    print("\n=== INTRADAY BREAKOUT MARGIN DIAGNOSIS ===")
    print(f"Total Analyzed: {len(df)}")
    
    # Buckets
    tiny = df[df["margin_pct"] < 0.1]
    small = df[(df["margin_pct"] >= 0.1) & (df["margin_pct"] < 0.3)]
    solid = df[df["margin_pct"] >= 0.3]
    
    print(f"Tiny Breakout (< 0.1%): {len(tiny)} ({len(tiny)/len(df)*100:.1f}%)")
    print(f"Small Breakout (0.1% - 0.3%): {len(small)} ({len(small)/len(df)*100:.1f}%)")
    print(f"Solid Breakout (>= 0.3%): {len(solid)} ({len(solid)/len(df)*100:.1f}%)")
    
    if not tiny.empty:
        print("\nSample Tiny Breakouts (Noise Candidates):")
        print(tiny[["symbol", "entry_px", "level", "margin_pct"]].head(10).to_string())

    df.to_csv("results/zombie_intraday_margin.csv", index=False)

if __name__ == "__main__":
    main()