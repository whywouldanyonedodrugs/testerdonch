# audit_rs_dynamic.py
import pandas as pd
from pathlib import Path
import config as cfg
import numpy as np

def main():
    # 1. Load Blocking Trades
    blockers_path = cfg.RESULTS_DIR / "blocking_trades_report.csv"
    if not blockers_path.exists():
        print("blocking_trades_report.csv not found.")
        return
    blockers = pd.read_csv(blockers_path)
    
    # 2. Load RS Weekly (Scout's view)
    rs_path = cfg.RESULTS_DIR / "rs_weekly.parquet"
    if not rs_path.exists():
        print("rs_weekly.parquet not found.")
        return
    rs_weekly = pd.read_parquet(rs_path)
    # Ensure week_start is timezone-aware UTC
    rs_weekly["week_start"] = pd.to_datetime(rs_weekly["week_start"], utc=True)
    
    print(f"Auditing RS for {len(blockers)} zombie trades...")
    
    results = []
    
    # Group by symbol to optimize parquet loading
    for sym, group in blockers.groupby("symbol"):
        if pd.isna(group.iloc[0]["blocking_trade_entry"]):
            continue
            
        pq_path = cfg.PARQUET_DIR / f"{sym}.parquet"
        if not pq_path.exists():
            continue
        
        df = pd.read_parquet(pq_path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        
        # Resample to Daily Close to mimic Live Bot's data source
        daily = df["close"].resample("1D").last()
        
        for _, row in group.iterrows():
            entry_ts = pd.to_datetime(row["blocking_trade_entry"], utc=True)
            
            # --- 1. Scout RS (Weekly) ---
            # Find the relevant weekly row (week_start <= entry_ts)
            scout_rs_row = rs_weekly[
                (rs_weekly["symbol"] == sym) & 
                (rs_weekly["week_start"] <= entry_ts)
            ].sort_values("week_start").iloc[-1] if not rs_weekly.empty else None
            
            scout_rs = scout_rs_row["rs_pct"] if scout_rs_row is not None else -1
            
            # --- 2. Live RS (Rolling 7D) ---
            # Live bot uses: last = close[T-1], prev = close[T-8]
            # (Assuming entry is on Day T, it looks at completed candles)
            t_minus_1 = entry_ts.floor("D") - pd.Timedelta(days=1)
            t_minus_8 = t_minus_1 - pd.Timedelta(days=7)
            
            try:
                # Use asof to get the closest available close if exact match missing
                # (Live bot logic handles missing data by looking back, we simplify here)
                c1 = float(daily.asof(t_minus_1))
                c8 = float(daily.asof(t_minus_8))
                
                if pd.isna(c1) or pd.isna(c8) or c8 == 0:
                    live_ret = np.nan
                else:
                    live_ret = (c1 / c8) - 1.0
            except Exception:
                live_ret = np.nan
                
            results.append({
                "symbol": sym,
                "entry_ts": entry_ts,
                "scout_rs_pct": scout_rs,
                "live_rolling_ret_pct": round(live_ret * 100, 2) if pd.notna(live_ret) else None
            })

    df = pd.DataFrame(results)
    
    # Filter for potential mismatches:
    # High Scout RS (>70) but Low/Negative Rolling Return
    # (Note: We don't have the full universe to calc exact Rolling Percentile, 
    # but a negative return is almost certainly < 70th percentile in a bull market)
    
    mismatches = df[
        (df["scout_rs_pct"] >= 70) & 
        (df["live_rolling_ret_pct"] < 5.0) # Arbitrary low threshold for "weak"
    ].copy()
    
    print("\n=== RS Audit Results ===")
    print(f"Total Trades Checked: {len(df)}")
    print(f"Potential Mismatches (High Weekly RS vs Low Rolling Ret): {len(mismatches)}")
    
    if not mismatches.empty:
        print("\nTop Mismatches:")
        print(mismatches.sort_values("live_rolling_ret_pct").head(20).to_string())
        
    out_path = cfg.RESULTS_DIR / "rs_audit.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()