# diagnose_schema.py
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config as cfg
from winprob_loader import WinProbScorer
from backtester import Backtester

def main():
    print("=== Donch Meta-Model Schema Diagnostic ===\n")

    # 1. Load Model Manifest
    print("--- 1. Loading Model Manifest ---")
    model_dir = Path(cfg.META_MODEL_DIR).resolve()
    try:
        scorer = WinProbScorer(model_dir)
        raw_cols = set(scorer.raw_cols)
        print(f"Model loaded from: {model_dir}")
        print(f"Model expects {len(raw_cols)} features.")
    except Exception as e:
        print(f"FATAL: Failed to load scorer: {e}")
        return

    # 2. Find a Signal
    print("\n--- 2. Finding a Signal ---")
    signals_dir = Path(cfg.SIGNALS_DIR)
    parquet_files = list(signals_dir.glob("**/*.parquet"))
    if not parquet_files:
        print("FATAL: No signal files found in signals/.")
        return
    
    # Pick the first file
    pfile = parquet_files[0]
    print(f"Loading signal from: {pfile}")
    
    # Infer symbol from path (partition structure)
    sym_from_path = None
    for part in pfile.parts:
        if part.startswith("symbol="):
            sym_from_path = part.split("=")[1]
            break
            
    df_sig = pd.read_parquet(pfile)
    if df_sig.empty:
        print("FATAL: Signal file is empty.")
        return
        
    row = df_sig.iloc[0].to_dict()
    
    # Fix symbol if missing in row
    if "symbol" not in row or not row["symbol"]:
        if sym_from_path:
            row["symbol"] = sym_from_path
        else:
            print("FATAL: No symbol in row and could not infer from path.")
            return
            
    sym = row["symbol"]
    ts = pd.to_datetime(row["timestamp"], utc=True)
    entry_price = float(row["entry"])
    
    print(f"Selected Signal: {sym} at {ts} (Entry: {entry_price})")

    # 3. Initialize Backtester (Mocking run state)
    print("\n--- 3. Initializing Backtester State ---")
    bt = Backtester(1000, 0.01, 10.0)
    
    # Load 5m data
    print(f"Loading 5m data for {sym}...")
    try:
        df5 = bt._get_5m(sym)
        print(f"Loaded 5m data: {len(df5)} rows.")
        print(f"Index type: {type(df5.index)}")
        if "timestamp" in df5.columns:
            print("WARNING: 'timestamp' column exists in df5 (Ambiguity Risk!)")
        else:
            print("OK: 'timestamp' column correctly dropped from df5.")
    except Exception as e:
        print(f"FATAL: Failed to load 5m data: {e}")
        return

    # 4. Replay Store Lookup
    print("\n--- 4. Replay Store Lookup ---")
    bt._meta_store_load()
    if getattr(bt, "_meta_store_df", None) is None:
        print("WARNING: Meta store DF is None (trades.clean.csv not found?). Replay will be empty.")
    
    replay_feats = bt._meta_store_lookup(sym, ts, entry_price)
    print(f"Replay features found: {len(replay_feats)}")
    if not replay_feats:
        print("WARNING: No replay features found for this signal (key mismatch?).")
    
    # 5. Compute Extra Features
    print("\n--- 5. Computing Extra Features ---")
    
    # Mock the fix: merge replay into signal row for the 's' argument
    s_combined = {**row, **replay_feats}
    
    # Get ATR (needed for extra features)
    try:
        if ts in df5.index:
            atr_now = float(df5.loc[ts, "atr_pre"])
        else:
            idx = df5.index[df5.index >= ts]
            if len(idx) > 0:
                atr_now = float(df5.loc[idx[0], "atr_pre"])
            else:
                atr_now = np.nan
    except Exception:
        atr_now = np.nan

    # Get regime_up
    regime_up = bt.regime.is_up(ts)

    try:
        # Call the backtester's feature generation logic
        # We pass replay_values explicitly to test the fix
        extra = bt._meta_extra_features(
            sym=sym,
            s=s_combined, 
            ts=ts,
            entry_price=entry_price,
            atr_now=atr_now,
            df5=df5,
            regime_up=regime_up,
            replay_values=replay_feats 
        )
    except Exception as e:
        print(f"FATAL: Error in _meta_extra_features: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Merge and Diff
    print("\n--- 6. Schema Validation ---")
    # The final dict passed to scorer is: row + extra (which includes replay if we did it right)
    final_row = {**row, **extra}
    
    present_cols = set(final_row.keys())
    missing = raw_cols - present_cols
    
    print(f"Total features provided: {len(present_cols)}")
    print(f"Missing features: {len(missing)}")
    
    if missing:
        print("\n!!! MISSING COLUMNS !!!")
        print(sorted(list(missing)))
        print("\nFAILURE: Schema mismatch detected.")
    else:
        print("\nSUCCESS: All training columns are present.")
        print("The backtester should run without schema errors.")

if __name__ == "__main__":
    main()