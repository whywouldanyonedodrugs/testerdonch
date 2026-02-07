# prove_double_shift.py
import pandas as pd
import numpy as np

def prove_logic():
    print("=== STRUCTURAL LOGIC PROOF: SCOUT.PY ===")
    
    # 1. Create Synthetic Data (Strictly increasing highs to make lags obvious)
    # Day 1: High 10
    # Day 2: High 20
    # Day 3: High 30
    # Day 4: High 40
    # Day 5: High 50
    dates = pd.date_range(start="2024-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({
        "high": [10.0, 20.0, 30.0, 40.0, 50.0],
        "close": [10.0, 20.0, 30.0, 40.0, 50.0] # Close = High for simplicity
    }, index=dates)
    
    print("\n[Input Data]")
    print(df)
    
    # 2. Apply Scout Logic (Verbatim from scout.py)
    don_n_days = 2 # Use 2-day window for easy mental math
    
    # "Prior N-day rolling high, shifted by 1 day to avoid look-ahead"
    df["donch_upper"] = df["high"].rolling(don_n_days, min_periods=don_n_days).max().shift(1)
    
    print("\n[After .shift(1) inside Daily DataFrame]")
    print(df[["high", "donch_upper"]])
    
    # "We mirror this by shifting the daily breakout information forward by 1 day"
    daily_effect = df[["donch_upper"]].copy()
    daily_effect.index = daily_effect.index + pd.Timedelta(days=1)
    daily_effect = daily_effect.rename(columns={"donch_upper": "donch_break_level"})
    
    print("\n[After Index Shift (+1 Day) for Mapping]")
    print(daily_effect)
    
    # 3. Analyze Result for Target Day (Day 5: 2024-01-05)
    target_date = pd.Timestamp("2024-01-05", tz="UTC")
    
    # Expected Level (T-1): Max of [Day 3, Day 4] = Max(30, 40) = 40.0
    # Lagged Level (T-2):   Max of [Day 2, Day 3] = Max(20, 30) = 30.0
    
    actual_level = daily_effect.loc[target_date, "donch_break_level"]
    
    print(f"\n[Verification for {target_date.date()}]")
    print(f"Target Day High: {50.0}")
    print(f"True T-1 Level (Max of Jan 3, Jan 4): 40.0")
    print(f"True T-2 Level (Max of Jan 2, Jan 3): 30.0")
    print(f"Scout Logic Output: {actual_level}")
    
    if actual_level == 30.0:
        print("\n>>> PROOF CONFIRMED: Logic produces T-2 Level (Double Lag). <<<")
    elif actual_level == 40.0:
        print("\n>>> PROOF FAILED: Logic produces T-1 Level (Correct). <<<")
    else:
        print(f"\n>>> UNEXPECTED RESULT: {actual_level} <<<")

if __name__ == "__main__":
    prove_logic()