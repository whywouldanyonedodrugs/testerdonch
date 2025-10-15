# check_header.py
"""
Small utility to check data headers; optional.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import config as cfg

def check(symbol: str, timeframe: str = "5m"):
    dirp = cfg.PARQUET_DIR if timeframe == "5m" else cfg.PARQUET_1M_DIR
    p = dirp / f"{symbol}.parquet"
    if not p.exists():
        print(f"Missing: {p}")
        return
    df = pd.read_parquet(p)
    print(df.head())
    print(df.dtypes)

if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "ETHUSDT"
    tf = sys.argv[2] if len(sys.argv) > 2 else "5m"
    check(sym, tf)
