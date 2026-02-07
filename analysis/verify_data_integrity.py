# verify_data_integrity.py
import pandas as pd
import requests
import time
from pathlib import Path
import config as cfg

# Config matching pull.py
REST_API_URL_KLINE = "https://api.bybit.com/v5/market/kline"
CATEGORY = "linear"

def fetch_bybit_candle(symbol, ts_start_ms):
    """Fetch a single 5m candle from Bybit Linear API."""
    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "interval": "5",
        "start": ts_start_ms,
        "limit": 1
    }
    try:
        r = requests.get(REST_API_URL_KLINE, params=params, timeout=10)
        data = r.json()
        if data["retCode"] == 0 and data["result"]["list"]:
            # Bybit returns [startTime, open, high, low, close, volume, turnover]
            # We want Close
            kline = data["result"]["list"][0]
            return float(kline[4]) 
    except Exception as e:
        print(f"API Error for {symbol}: {e}")
    return None

def main():
    # 1. Load the discrepancies identified previously
    parity_path = cfg.RESULTS_DIR / "levels_parity.csv"
    if not parity_path.exists():
        print("levels_parity.csv not found. Run analyze_levels_parity.py first.")
        return
    
    df = pd.read_csv(parity_path)
    
    # Filter for significant deviations (> 2%) to save time
    targets = df[df["entry_diff_pct"].abs() > 2.0].copy()
    
    # Also include one 'good' match for control
    control = df[df["entry_diff_pct"].abs() < 0.5].head(1)
    targets = pd.concat([targets.head(5), control])
    
    print(f"Verifying {len(targets)} data points against Bybit API (Category: {CATEGORY})...\n")
    
    results = []
    
    for _, row in targets.iterrows():
        sym = row["symbol"]
        # Find the timestamp from the blocking report (we need to reload it to get the TS)
        # Or we can infer it from the backtest trades file using the price.
        # Better: Load trades.csv and find the timestamp for this specific entry price/symbol.
        
        # Re-loading trades to get exact timestamp for this entry price
        trades = pd.read_csv(cfg.RESULTS_DIR / "trades.csv")
        trades["entry_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
        
        # Match by symbol and close-enough entry price
        match = trades[
            (trades["symbol"] == sym) & 
            (np.isclose(trades["entry"], row["bt_entry_px"], rtol=1e-5))
        ]
        
        if match.empty:
            print(f"Skipping {sym}: Could not locate trade in trades.csv")
            continue
            
        ts = match.iloc[0]["entry_ts"]
        ts_ms = int(ts.timestamp() * 1000)
        
        # 1. Get Local Parquet Value
        # We assume the value in trades.csv came from parquet, but let's verify the file directly
        pq_path = cfg.PARQUET_DIR / f"{sym}.parquet"
        local_close = "N/A"
        if pq_path.exists():
            try:
                pq_df = pd.read_parquet(pq_path)
                # Ensure index is datetime
                if not isinstance(pq_df.index, pd.DatetimeIndex):
                    pq_df.index = pd.to_datetime(pq_df["timestamp"] if "timestamp" in pq_df.columns else pq_df.index, utc=True)
                
                if ts in pq_df.index:
                    local_close = float(pq_df.loc[ts]["close"])
                else:
                    local_close = "MISSING"
            except Exception as e:
                local_close = f"ERR: {e}"
        else:
            local_close = "NO_FILE"

        # 2. Get Remote API Value
        remote_close = fetch_bybit_candle(sym, ts_ms)
        
        # 3. Compare
        diff_pct = 0.0
        if isinstance(local_close, float) and remote_close:
            diff_pct = (remote_close - local_close) / local_close * 100
            
        results.append({
            "symbol": sym,
            "timestamp": ts,
            "local_parquet_close": local_close,
            "remote_api_close": remote_close,
            "diff_pct": round(diff_pct, 2),
            "bt_entry_px": row["bt_entry_px"],
            "live_entry_px": row["live_entry_px"]
        })
        
        time.sleep(0.1) # Rate limit niceness

    print(pd.DataFrame(results).to_string())

import numpy as np
if __name__ == "__main__":
    main()