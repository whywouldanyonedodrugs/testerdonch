# debug_highs.py
import pandas as pd
from pathlib import Path
import config as cfg

def main():
    sym = "1000TOSHIUSDT"
    print(f"Checking Highs for {sym} around Sept 18 2025...")
    
    pq_path = cfg.PARQUET_DIR / f"{sym}.parquet"
    if not pq_path.exists():
        print("Parquet not found.")
        return

    df = pd.read_parquet(pq_path)
    df.index = pd.to_datetime(df.index, utc=True)
    
    # Get Daily Highs
    daily = df["high"].resample("1D").max()
    
    # Show Sept 1 - Sept 20
    subset = daily["2025-09-01":"2025-09-20"]
    print(subset)
    
    # Calc 20-day max for Sept 18
    # Standard: Max of Aug 29 - Sept 17
    # Lagged: Max of Aug 28 - Sept 16
    
    t_minus_1 = subset[:"2025-09-17"].tail(20).max()
    t_minus_2 = subset[:"2025-09-16"].tail(20).max()
    
    print(f"\nMax (ending Sept 17): {t_minus_1}")
    print(f"Max (ending Sept 16): {t_minus_2}")
    
    # Check Signal Level
    sig_path = cfg.SIGNALS_DIR / f"symbol={sym}"
    if sig_path.exists():
        sigs = pd.read_parquet(sig_path)
        ts_col = "timestamp" if "timestamp" in sigs.columns else "entry_ts"
        sigs[ts_col] = pd.to_datetime(sigs[ts_col], utc=True)
        
        target = sigs[sigs[ts_col].dt.date == pd.to_datetime("2025-09-18").date()]
        if not target.empty:
            print(f"\nSignal Level on Sept 18: {target.iloc[0]['don_break_level']}")

if __name__ == "__main__":
    main()