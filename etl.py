# ======================================
# etl.py – CSV → Parquet converter
# ======================================
"""Convert raw Bybit 5-minute CSV files to compressed Parquet.

Rules (updated):
- ONLY read input CSVs from the project-local directory: <repo>/raw_csv
- File naming: SYMBOL.csv (e.g., ETHUSDT.csv or ETHUSDT.CSV).
- No support for merged or other directories.

$ python etl.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config as cfg  # used for PARQUET_DIR, START/END_DATE, SYMBOLS_FILE

# ----------------------------------------------------------------------
# Hardcoded raw CSV directory: <repo>/raw_csv (project-local)
RAW_DIR: Path = (Path(__file__).resolve().parent / "raw_csv").resolve()

# Column definitions / types
COLS_IN_CSV_BASE = [
    "open_time", "open", "high", "low", "close", "volume", "turnover",
    # Optional extras if present:
    # "open_interest", "funding_rate",
]
COL_RENAME = {
    "open_time": "timestamp", "open": "open", "high": "high", "low": "low",
    "close": "close", "volume": "volume", "turnover": "turnover",
    "open_interest": "open_interest",
    "funding_rate": "funding_rate",
}
DTYPES = {
    "open": "float32", "high": "float32", "low": "float32", "close": "float32",
    "volume": "float64", "turnover": "float64",
    "open_interest": "float64",
    "funding_rate": "float32",
}


def _parse_datetime(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    # 1) Unix ms epoch
    if s.str.fullmatch(r"\d+").all():
        return pd.to_datetime(s.astype("int64"), unit="ms", utc=True)
    # 2) dd/mm/YYYY HH:MM
    dt = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce", utc=True)
    mask = dt.isna()
    # 3) ISO-like "YYYY-mm-dd HH:MM:SS"
    if mask.any():
        dt_iso = pd.to_datetime(s[mask], format="%Y-%m-%d %H:%M:%S", errors="coerce", utc=True)
        dt.loc[mask] = dt_iso
    mask = dt.isna()
    # 4) Fallback: let pandas try
    if mask.any():
        dt_fb = pd.to_datetime(s[mask], errors="coerce", utc=True)
        dt.loc[mask] = dt_fb
    if dt.isna().any():
        bad = s[dt.isna()].head()
        raise ValueError(f"Unable to parse some timestamps. Examples: {', '.join(map(str, bad))}")
    return dt


def _single_file(csv_path: Path) -> Tuple[str, str]:
    """Convert a single CSV from RAW_DIR → Parquet."""
    csv_path = csv_path.resolve()

    # Hard guard: only allow files from RAW_DIR
    if csv_path.parent != RAW_DIR:
        raise ValueError(f"Refusing to read from '{csv_path.parent}'. Only '{RAW_DIR}' is allowed.")

    symbol = csv_path.stem.upper()

    pq_path = cfg.PARQUET_DIR / f"{symbol}.parquet"
    if pq_path.exists():
        return symbol, "skipped"

    df = pd.read_csv(csv_path, header=0, dtype="str")

    has_oi = "open_interest" in df.columns
    has_fr = "funding_rate" in df.columns

    # Required columns presence check (helps fail fast)
    missing = [c for c in ("open_time", "open", "high", "low", "close", "volume", "turnover") if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    df["open_time"] = _parse_datetime(df["open_time"])
    df.rename(columns=COL_RENAME, inplace=True)

    # Enforce numeric types where present
    dtypes_to_apply = {k: v for k, v in DTYPES.items() if k in df.columns}
    for col, dtype in dtypes_to_apply.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)

    # Date windowing if configured in config.py
    if getattr(cfg, "START_DATE", None):
        start_utc = pd.to_datetime(cfg.START_DATE, utc=True)
        df = df[df["timestamp"] >= start_utc]
    if getattr(cfg, "END_DATE", None):
        end_utc = pd.to_datetime(cfg.END_DATE, utc=True)
        df = df[df["timestamp"] <= end_utc]

    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"Timestamps out of order in {symbol}")
    if not (df["close"] > 0).all():
        raise ValueError(f"Non-positive prices detected in {symbol}")

    table = pa.Table.from_pandas(df, preserve_index=True)
    pq.write_table(table, pq_path, compression="zstd", flavor="spark")

    if has_oi and has_fr:
        status = "converted with OI+FR"
    elif has_oi:
        status = "converted with OI"
    elif has_fr:
        status = "converted with FR"
    else:
        status = "converted"
    return symbol, status


def _symbols_from_file() -> List[str]:
    if not cfg.SYMBOLS_FILE.exists():
        raise FileNotFoundError("symbols.txt not found – place it in project root")
    with cfg.SYMBOLS_FILE.open() as fh:
        symbols = [
            line.split("#", 1)[0].split()[0].upper()
            for line in fh
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return symbols


def _index_raw_csvs(raw_dir: Path) -> Dict[str, Path]:
    """Build a case-insensitive index of SYMBOL → Path for *.csv files in RAW_DIR."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"RAW_DIR does not exist: {raw_dir}")
    index: Dict[str, Path] = {}
    for p in raw_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".csv":
            index[p.stem.upper()] = p.resolve()
    return index


def main() -> None:
    symbols = _symbols_from_file()
    csv_index = _index_raw_csvs(RAW_DIR)

    csv_files_to_process = []
    missing_symbols = []

    for s in symbols:
        p = csv_index.get(s.upper())
        if p is not None:
            csv_files_to_process.append(p)
        else:
            missing_symbols.append(s)

    if missing_symbols:
        print(f"Warning: No CSV data found in {RAW_DIR} for {len(missing_symbols)} symbols.")

    if not csv_files_to_process:
        print("No new data to process. Exiting.")
        return

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
            results.append((file_path.stem.upper(), f"error: {e}"))

    with_oi = sum(1 for _, status in results if "OI" in status)
    with_fr = sum(1 for _, status in results if "FR" in status)
    skipped = sum(1 for _, status in results if status == "skipped")
    converted = sum(1 for _, status in results if status.startswith("converted"))
    errors = len(results) - skipped - converted

    print(f"\nDone. Processed {len(results)} files from {RAW_DIR}.")
    print(f"Converted with OI: {with_oi}, with FR: {with_fr}, Skipped: {skipped}, Errors: {errors}.")


if __name__ == "__main__":
    main()
