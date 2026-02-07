# audit_listing_age.py
import pandas as pd
from pathlib import Path
import config as cfg

def main():
    # 1. Load Blocking Trades
    blockers_path = cfg.RESULTS_DIR / "blocking_trades_report.csv"
    if not blockers_path.exists():
        print("blocking_trades_report.csv not found.")
        return
    blockers = pd.read_csv(blockers_path)
    
    print(f"Auditing Listing Age for {len(blockers)} zombie trades...")
    
    results = []
    
    for sym, group in blockers.groupby("symbol"):
        if pd.isna(group.iloc[0]["blocking_trade_entry"]):
            continue
            
        # Load Parquet to find first timestamp
        pq_path = cfg.PARQUET_DIR / f"{sym}.parquet"
        if not pq_path.exists():
            continue
            
        df = pd.read_parquet(pq_path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        
        first_ts = df.index[0]
        
        for _, row in group.iterrows():
            entry_ts = pd.to_datetime(row["blocking_trade_entry"], utc=True)
            
            # Calc age in days
            age_days = (entry_ts - first_ts).total_seconds() / 86400.0
            
            results.append({
                "symbol": sym,
                "entry_ts": entry_ts,
                "first_data_ts": first_ts,
                "age_days": round(age_days, 1)
            })

    df = pd.DataFrame(results)
    
    # Check for young coins (< 20 days)
    young = df[df["age_days"] < 20].copy()
    
    print("\n=== Listing Age Audit ===")
    print(f"Total Checked: {len(df)}")
    print(f"Young Trades (< 20 days): {len(young)}")
    
    if not young.empty:
        print("\nTop Young Trades:")
        print(young.sort_values("age_days").head(20).to_string())
        
    out_path = cfg.RESULTS_DIR / "listing_age_audit.csv"
    df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()