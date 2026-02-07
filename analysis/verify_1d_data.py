# verify_1d_data.py
import pandas as pd
import requests
import time
from pathlib import Path
import config as cfg

REST_API_URL_KLINE = "https://api.bybit.com/v5/market/kline"
CATEGORY = "linear"

def fetch_bybit_daily_high(symbol, ts_start_ms):
    """Fetch a single 1D candle from Bybit Linear API."""
    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "interval": "D",
        "start": ts_start_ms,
        "limit": 1
    }
    try:
        r = requests.get(REST_API_URL_KLINE, params=params, timeout=10)
        data = r.json()
        if data["retCode"] == 0 and data["result"]["list"]:
            # Bybit returns [startTime, open, high, low, close, volume, turnover]
            kline = data["result"]["list"][0]
            return float(kline[2]) # High
    except Exception as e:
        print(f"API Error for {symbol}: {e}")
    return None

def main():
    # Load blocking trades to get symbols and dates
    blockers_path = cfg.RESULTS_DIR / "blocking_trades_report.csv"
    if not blockers_path.exists():
        print("blocking_trades_report.csv not found.")
        return
    blockers = pd.read_csv(blockers_path)
    
    # Pick top 5 symbols with biggest breakout level mismatch
    # (We don't have the mismatch file loaded here, so just pick first 5 unique)
    symbols = blockers["symbol"].unique()[:5]
    
    print(f"Verifying 1D Highs for {len(symbols)} symbols...")
    
    results = []
    
    for sym in symbols:
        # Load local parquet
        pq_path = cfg.PARQUET_DIR / f"{sym}.parquet"
        if not pq_path.exists():
            continue
            
        df = pd.read_parquet(pq_path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        
        # Resample to 1D
        daily = df["high"].resample("1D").max()
        
        # Check last 5 days
        recent_days = daily.index[-10:-5] # Pick a few days from history
        
        for day in recent_days:
            ts_ms = int(day.timestamp() * 1000)
            local_high = float(daily.loc[day])
            remote_high = fetch_bybit_daily_high(sym, ts_ms)
            
            diff_pct = 0.0
            if remote_high:
                diff_pct = (remote_high - local_high) / local_high * 100
                
            results.append({
                "symbol": sym,
                "date": day.date(),
                "local_high": local_high,
                "remote_high": remote_high,
                "diff_pct": round(diff_pct, 2)
            })
            time.sleep(0.1)
            
    print(pd.DataFrame(results).to_string())

if __name__ == "__main__":
    main()