# ======================================
# etl.py  –  CSV ➜ Parquet converter
# ======================================
"""Convert raw Bybit 5-minute CSV files to compressed Parquet.
Run once before any back-test phase.

$ python etl.py
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config as cfg

# ----------------------------------------------------------------------
# Definitions remain the same
COLS_IN_CSV_BASE = [
    "open_time", "open", "high", "low", "close", "volume", "turnover",
]
COL_RENAME = {
    "open_time": "timestamp", "open": "open", "high": "high", "low": "low",
    "close": "close", "volume": "volume", "turnover": "turnover",
    "open_interest": "open_interest",
}
DTYPES = {
    "open": "float32", "high": "float32", "low": "float32", "close": "float32",
    "volume": "float64", "turnover": "float64", "open_interest": "float64",
}

# ----------------------------------------------------------------------------------
# _parse_datetime function remains the same
def _parse_datetime(series: pd.Series) -> pd.Series:
    """
    Parse many timestamp formats and ALWAYS return UTC-aware datetimes
    (dtype: datetime64[ns, UTC]).
    Supported:
      - epoch millis (e.g. 1723197600000)
      - 'DD/MM/YYYY HH:MM'
      - 'YYYY-MM-DD HH:MM:SS'
      - ISO strings with offsets (e.g. '2025-08-09 11:05:00+00:00')
    """
    s = series.astype(str)

    # 1) Pure digits → epoch ms
    if s.str.fullmatch(r"\d+").all():
        return pd.to_datetime(s.astype("int64"), unit="ms", utc=True)

    # 2) Try explicit formats (UTC-aware)
    dt = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce", utc=True)
    mask = dt.isna()
    if mask.any():
        dt_iso = pd.to_datetime(s[mask], format="%Y-%m-%d %H:%M:%S", errors="coerce", utc=True)
        dt.loc[mask] = dt_iso

    # 3) Fallback: free-form parse (will respect offsets like +00:00)
    mask = dt.isna()
    if mask.any():
        dt_fb = pd.to_datetime(s[mask], errors="coerce", utc=True)
        dt.loc[mask] = dt_fb

    # Final sanity
    if dt.isna().any():
        bad = s[dt.isna()].head()
        raise ValueError(f"Unable to parse some timestamps. Examples: {', '.join(map(str, bad))}")
    return dt


# ----------------------------------------------------------------------

def _single_file(csv_path: Path) -> Tuple[str, str]:
    # Extract symbol correctly from either "SYMBOL_5m_merged.csv" or "SYMBOL.csv"
    filename = csv_path.name
    if "_5m_merged.csv" in filename:
        symbol = filename.split("_5m_merged.csv")[0].upper()
    else:
        symbol = csv_path.stem.upper()

    pq_path = cfg.PARQUET_DIR / f"{symbol}.parquet"

    if pq_path.exists():
        return symbol, "skipped"

    # --- MODIFICATION: Simplified and more robust CSV loading ---
    # Let pandas infer the columns from the header directly.
    df = pd.read_csv(csv_path, header=0, dtype="str")

    # Check if 'open_interest' column exists after loading
    has_oi = 'open_interest' in df.columns
    # --- END MODIFICATION ---

    # Parse dates with our flexible helper
    df["open_time"] = _parse_datetime(df["open_time"])

    # Rename & cast numeric columns
    df.rename(columns=COL_RENAME, inplace=True)

    # Build the list of dtypes to apply based on columns that actually exist in the DataFrame
    # This is now robust to missing columns.
    dtypes_to_apply = {k: v for k, v in DTYPES.items() if k in df.columns}

    for col, dtype in dtypes_to_apply.items():
        df[col] = df[col].astype(dtype)

    # Trim by date window if requested
    if cfg.START_DATE:
        start_utc = pd.to_datetime(cfg.START_DATE, utc=True)
        df = df[df["timestamp"] >= start_utc]
    if cfg.END_DATE:
        end_utc = pd.to_datetime(cfg.END_DATE, utc=True)
        df = df[df["timestamp"] <= end_utc]

    # Index & sort
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Sanity checks
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Timestamps out of order in {symbol}")
    if not (df["close"] > 0).all():
        raise ValueError(f"Non-positive prices detected in {symbol}")

    # Write compressed Parquet
    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, pq_path, compression="zstd", flavor="spark")

    status = "converted with OI" if has_oi else "converted without OI"
    return symbol, status

# ----------------------------------------------------------------------

def _symbols_from_file() -> list[str]:
    if not cfg.SYMBOLS_FILE.exists():
        raise FileNotFoundError("symbols.txt not found – place it in project root")
    with cfg.SYMBOLS_FILE.open() as fh:
        symbols = [
            line.split("#", 1)[0].split()[0].upper()
            for line in fh
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return symbols

# ----------------------------------------------------------------------

def main() -> None:
    symbols = _symbols_from_file()
    
    csv_files_to_process = []
    missing_symbols = []

    for s in symbols:
        new_path = cfg.MERGED_CSV_DIR / f"{s}_5m_merged.csv"
        if new_path.exists():
            csv_files_to_process.append(new_path)
            continue
        old_path = cfg.RAW_CSV_DIR / f"{s}.csv"
        if old_path.exists():
            csv_files_to_process.append(old_path)
            continue
        missing_symbols.append(s)

    if missing_symbols:
        print(f"Warning: No CSV data found for {len(missing_symbols)} symbols.")
    
    if not csv_files_to_process:
        print("No new data to process. Exiting.")
        return

    # --- MODIFICATION: Switched to a sequential loop for debugging performance ---
    print("\nRunning in sequential mode to identify slow files...")
    results = []
    total_files = len(csv_files_to_process)

    for i, file_path in enumerate(csv_files_to_process):
        print(f"Processing file {i+1}/{total_files}: {file_path.name}...")
        try:
            result = _single_file(file_path)
            results.append(result)
        except Exception as e:
            print(f"  ERROR processing {file_path.name}: {e}")
            results.append((file_path.stem, f"error: {e}"))
    # --- END MODIFICATION ---

    converted_with_oi = sum(1 for _, status in results if status == "converted with OI")
    converted_without_oi = sum(1 for _, status in results if status == "converted without OI")
    skipped = sum(1 for _, status in results if status == "skipped")
    errors = len(results) - converted_with_oi - converted_without_oi - skipped
    
    print(f"\nDone. Processed {len(results)} files.")
    print(f"Converted with OI: {converted_with_oi}, Converted without OI: {converted_without_oi}, Skipped: {skipped}, Errors: {errors}.")

# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()