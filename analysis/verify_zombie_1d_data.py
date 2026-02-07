# verify_zombie_1d_data.py
import pandas as pd
import requests
import time
from pathlib import Path
import config as cfg

REST_API_URL_KLINE = "https://api.bybit.com/v5/market/kline"
CATEGORY = "linear"

def fetch_bybit_daily(symbol, ts_start_ms):
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
            # [startTime, open, high, low, close, volume, turnover]
            kline = data["result"]["list"][0]
            return {
                "high": float(kline[2]),
                "close": float(kline[4])
            }
    except Exception as e:
        print(f"API Error for {symbol}: {e}")
    return None

def main():
    # 1. Load Blocking Trades
    blockers_path = cfg.RESULTS_DIR / "blocking_trades_report.csv"
    if not blockers_path.exists():
        print("blocking_trades_report.csv not found.")
        return
    blockers = pd.read_csv(blockers_path)
    
    print(f"Verifying 1D Data for {len(blockers)} zombie trades...")
    
    results = []
    
    # Group by symbol
    for sym, group in blockers.groupby("symbol"):
        if pd.isna(group.iloc[0]["blocking_trade_entry"]):
            continue
            
        # Load Local Parquet
        pq_path = cfg.PARQUET_DIR / f"{sym}.parquet"
        if not pq_path.exists():
            continue
            
        df = pd.read_parquet(pq_path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)
        
        # Resample to 1D (mimicking scout.py)
        daily_local = df.resample("1D").agg({"high": "max", "close": "last"})
        
        for _, row in group.iterrows():
            entry_ts = pd.to_datetime(row["blocking_trade_entry"], utc=True)
            
            # The "Breakout Day" is the day BEFORE the entry (T-1)
            # Because we check if Yesterday Closed > Level
            breakout_day = entry_ts.floor("D") - pd.Timedelta(days=1)
            
            if breakout_day not in daily_local.index:
                continue
                
            local_data = daily_local.loc[breakout_day]
            
            # Fetch Remote
            ts_ms = int(breakout_day.timestamp() * 1000)
            remote_data = fetch_bybit_daily(sym, ts_ms)
            
            if remote_data:
                # Check Close Price (Critical for donch_break_ok)
                close_diff = (remote_data["close"] - local_data["close"]) / local_data["close"] * 100
                high_diff = (remote_data["high"] - local_data["high"]) / local_data["high"] * 100
                
                results.append({
                    "symbol": sym,
                    "breakout_day": breakout_day.date(),
                    "local_close": local_data["close"],
                    "remote_close": remote_data["close"],
                    "close_diff_pct": round(close_diff, 3),
                    "local_high": local_data["high"],
                    "remote_high": remote_data["high"],
                    "high_diff_pct": round(high_diff, 3)
                })
            
            time.sleep(0.05)

    df_res = pd.DataFrame(results)
    
    # Filter for mismatches
    mismatches = df_res[df_res["close_diff_pct"].abs() > 0.1].copy()
    
    print("\n=== 1D Data Mismatches (Local vs Bybit) ===")
    print(f"Total Checked: {len(df_res)}")
    print(f"Significant Mismatches (>0.1%): {len(mismatches)}")
    
    if not mismatches.empty:
        print("\nTop Mismatches (Close Price):")
        print(mismatches.sort_values("close_diff_pct", ascending=False).head(20).to_string())
        
    out_path = cfg.RESULTS_DIR / "zombie_data_audit.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()