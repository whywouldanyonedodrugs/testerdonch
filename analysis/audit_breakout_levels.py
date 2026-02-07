# audit_breakout_levels.py
import pandas as pd
from pathlib import Path
import config as cfg

def main():
    # 1. Load Blocking Trades (The Zombies)
    blockers_path = cfg.RESULTS_DIR / "blocking_trades_report.csv"
    if not blockers_path.exists():
        print("blocking_trades_report.csv not found.")
        return
    blockers = pd.read_csv(blockers_path)
    
    print(f"Auditing breakout levels for {len(blockers)} zombie trades...")
    
    results = []
    
    # Group by symbol to optimize loading
    for sym, group in blockers.groupby("symbol"):
        if pd.isna(group.iloc[0]["blocking_trade_entry"]):
            continue
            
        # Load OHLC Data
        df_ohlc = None
        try:
            # Try partitioned path first
            pq_path = cfg.PARQUET_DIR / f"{sym}.parquet"
            if pq_path.exists():
                df_ohlc = pd.read_parquet(pq_path)
            else:
                # Fallback to raw csv if needed, or skip
                continue
                
            # Ensure datetime index
            if "timestamp" in df_ohlc.columns:
                df_ohlc = df_ohlc.set_index("timestamp")
            df_ohlc.index = pd.to_datetime(df_ohlc.index, utc=True)
            df_ohlc = df_ohlc.sort_index()
            
            # Resample to Daily to calc true Donchian
            # Note: Live bot uses "closed" days. 
            # At time T (intraday), the level is max(High) of [T-21d, T-1d]
            df_daily = df_ohlc["high"].resample("1D").max()
            
            # Load Signals to get the used level
            sig_path = cfg.SIGNALS_DIR / f"symbol={sym}"
            if sig_path.exists():
                df_sig = pd.read_parquet(sig_path)
            else:
                df_sig = pd.read_parquet(cfg.SIGNALS_DIR)
                df_sig = df_sig[df_sig["symbol"] == sym]
            
            ts_col = "timestamp" if "timestamp" in df_sig.columns else "entry_ts"
            df_sig[ts_col] = pd.to_datetime(df_sig[ts_col], utc=True)
            
            for _, row in group.iterrows():
                entry_ts = pd.to_datetime(row["blocking_trade_entry"], utc=True)
                
                # 1. Get Signal's Level
                sig = df_sig[df_sig[ts_col] == entry_ts]
                if sig.empty:
                    continue
                used_level = float(sig.iloc[0]["don_break_level"])
                
                # 2. Calc True Level
                # Get 20 days PRIOR to the entry day
                entry_day = entry_ts.floor("D")
                # Shift: We want the max of the 20 days strictly before entry_day
                # Slice: [entry_day - 20d, entry_day - 1d]
                start_lookback = entry_day - pd.Timedelta(days=20)
                end_lookback = entry_day - pd.Timedelta(days=1)
                
                relevant_highs = df_daily[start_lookback:end_lookback]
                
                if len(relevant_highs) < 20:
                    true_level = -1 # Insufficient data
                else:
                    true_level = float(relevant_highs.max())
                
                diff_pct = (used_level - true_level) / true_level * 100 if true_level > 0 else 0
                
                results.append({
                    "symbol": sym,
                    "entry_ts": entry_ts,
                    "used_level": used_level,
                    "true_level": true_level,
                    "diff_pct": diff_pct,
                    "data_days": len(relevant_highs)
                })
                
        except Exception as e:
            print(f"Error processing {sym}: {e}")

    df = pd.DataFrame(results)
    
    # Filter for mismatches
    mismatches = df[df["diff_pct"].abs() > 0.1].copy()
    
    print("\n=== Breakout Level Audit ===")
    print(f"Total Checked: {len(df)}")
    print(f"Mismatches (>0.1%): {len(mismatches)}")
    
    if not mismatches.empty:
        print("\nTop Mismatches:")
        print(mismatches.sort_values("diff_pct", ascending=False).head(20).to_string())
    else:
        print("\nLevels match perfectly. The issue is not the Donchian calculation.")

    out_path = cfg.RESULTS_DIR / "breakout_audit.csv"
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()