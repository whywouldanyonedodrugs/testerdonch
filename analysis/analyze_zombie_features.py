# analyze_zombie_features.py
import pandas as pd
from pathlib import Path
import config as cfg

def main():
    # 1. Load Blocking Trades (Zombies)
    blockers_path = cfg.RESULTS_DIR / "blocking_trades_report.csv"
    if not blockers_path.exists():
        print("blocking_trades_report.csv not found.")
        return
    blockers = pd.read_csv(blockers_path)
    
    # 2. Load Signals (to get features)
    # We need to scan the parquet files for the specific entry timestamps
    print(f"Scanning signals for {len(blockers)} zombie trades...")
    
    zombie_feats = []
    
    # Optimization: Group by symbol to load parquet once
    for sym, group in blockers.groupby("symbol"):
        if pd.isna(group.iloc[0]["blocking_trade_entry"]):
            continue
            
        pq_path = cfg.SIGNALS_DIR / f"symbol={sym}"
        if not pq_path.exists():
            # Try file
            pq_path = cfg.SIGNALS_DIR / "signals.parquet"
            if not pq_path.exists():
                continue
                
        try:
            # Load signals for this symbol
            # Note: This assumes partitioned structure or single file. 
            # Using pandas read_parquet with filters if possible, or just loading.
            # Given the structure, we'll try loading the specific partition.
            
            # Construct path manually if partitioned
            part_path = cfg.SIGNALS_DIR / f"symbol={sym}"
            if part_path.exists():
                df = pd.read_parquet(part_path)
            else:
                # Fallback to full load (slow, but robust)
                df = pd.read_parquet(cfg.SIGNALS_DIR)
                df = df[df["symbol"] == sym]
                
            # Normalize timestamp
            ts_col = "timestamp" if "timestamp" in df.columns else "entry_ts"
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
            
            for _, row in group.iterrows():
                entry_ts = pd.to_datetime(row["blocking_trade_entry"], utc=True)
                
                # Find signal at this time
                sig = df[df[ts_col] == entry_ts]
                
                if not sig.empty:
                    s = sig.iloc[0]
                    zombie_feats.append({
                        "symbol": sym,
                        "entry_ts": entry_ts,
                        "rs_pct": s.get("rs_pct"),
                        "vol_mult": s.get("vol_mult") if "vol_mult" in s else s.get("vol_spike"), # Handle boolean or float
                        "atr": s.get("atr"),
                        "entry_price": s.get("entry"),
                        "regime": s.get("regime_up") if "regime_up" in s else "N/A"
                    })
                else:
                    # Signal might be slightly offset?
                    pass
                    
        except Exception as e:
            print(f"Error reading signals for {sym}: {e}")

    # 3. Analyze
    df = pd.DataFrame(zombie_feats)
    
    # Config thresholds
    RS_MIN = getattr(cfg, "RS_MIN_PERCENTILE", 70)
    VOL_MIN = getattr(cfg, "VOL_MULTIPLE", 2.0)
    
    print(f"\n=== Zombie Trade Feature Analysis (Thresholds: RS={RS_MIN}, Vol={VOL_MIN}) ===")
    
    # Check for borderline RS
    borderline_rs = df[
        (df["rs_pct"] >= RS_MIN) & 
        (df["rs_pct"] < RS_MIN + 5)
    ]
    print(f"\nBorderline RS ({RS_MIN}-{RS_MIN+5}): {len(borderline_rs)} / {len(df)}")
    if not borderline_rs.empty:
        print(borderline_rs[["symbol", "entry_ts", "rs_pct"]].head(10).to_string())

    # Check for borderline Volume (if numeric)
    # Note: vol_spike might be boolean in some versions
    if "vol_mult" in df.columns and pd.api.types.is_numeric_dtype(df["vol_mult"]):
        borderline_vol = df[
            (df["vol_mult"] >= VOL_MIN) & 
            (df["vol_mult"] < VOL_MIN + 0.5)
        ]
        print(f"\nBorderline Volume ({VOL_MIN}-{VOL_MIN+0.5}): {len(borderline_vol)} / {len(df)}")
        if not borderline_vol.empty:
            print(borderline_vol[["symbol", "entry_ts", "vol_mult"]].head(10).to_string())
            
    # Save
    out_path = cfg.RESULTS_DIR / "zombie_features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()