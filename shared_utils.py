# shared_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Set, Dict, List

import numpy as np
import pandas as pd
import config as cfg

# ---------------- Arrow Parquet helpers ----------------

def _read_parquet_arrow(path: Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fast Parquet read via Arrow (memory-mapped by default).
    """
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path), memory_map=bool(getattr(cfg, "IO_MEMORY_MAP", True)))
    tbl = pf.read(columns=columns)
    return tbl.to_pandas()

def _parquet_schema_columns(path: Path) -> set[str]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path), memory_map=bool(getattr(cfg, "IO_MEMORY_MAP", True)))
    return set(pf.schema.names)

# ---------------- Time column handling ----------------

# Common time-like column names we support
TIME_CANDIDATES: List[str] = [
    "timestamp", "time", "datetime", "date",
    "open_time", "openTime", "open_time_ms",
    "close_time", "t"
]

def _columns_with_time(schema_cols: set[str], requested: Optional[List[str]]) -> Optional[List[str]]:
    """
    If `requested` is provided, ensure at least one time-like column is also read.
    We pick the first candidate that exists in the file schema.
    """
    if requested is None:
        return None
    cols = list(dict.fromkeys(requested))  # preserve order, drop dupes
    for c in TIME_CANDIDATES:
        if c in schema_cols and c not in cols:
            cols.append(c)
            break  # one time column is enough
    return cols

def _normalize_timestamp_column(df: pd.DataFrame) -> Optional[pd.DatetimeIndex]:
    """
    Return a tz-aware UTC DatetimeIndex from df, if possible.
    Accepts:
      - Existing DatetimeIndex (naive or tz-aware)
      - Any single column among TIME_CANDIDATES
        * integer epoch in ns/ms/s
        * strings / datetime-like values, with or without timezone
    """
    # Already a DatetimeIndex?
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    # Prefer explicit 'timestamp', else try the rest
    for c in TIME_CANDIDATES:
        if c not in df.columns:
            continue
        s = df[c]

        # datetime-like dtypes first (handles tz-aware Series cleanly)
        if pd.api.types.is_datetime64_any_dtype(s):
            # Pandas DatetimeTZDtype is an extension dtype (not numpy) but we can convert/ensure UTC.
            # If tz-aware -> convert; if naive -> localize.
            if pd.api.types.is_datetime64tz_dtype(s):
                idx = s.dt.tz_convert("UTC")
            else:
                idx = s.dt.tz_localize("UTC")
            return pd.DatetimeIndex(idx)

        # integer epochs (guess unit)
        if pd.api.types.is_integer_dtype(s):
            vmax = int(s.dropna().max()) if len(s) else 0
            if vmax >= 10**14:   # nanoseconds
                idx = pd.to_datetime(s, unit="ns", utc=True, errors="coerce")
            elif vmax >= 10**12: # milliseconds (Binance style)
                idx = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
            else:                # seconds (or small ms)
                # try ms first; if it fails completely, fall back to seconds
                idx = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
                if idx.isna().all():
                    idx = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
            if not idx.isna().all():
                return pd.DatetimeIndex(idx)

        # strings / object → parse with UTC
        idx = pd.to_datetime(s, utc=True, errors="coerce")
        if not idx.isna().all():
            return pd.DatetimeIndex(idx)

    return None

# ---------------- Symbol discovery ----------------

def get_symbols_from_file() -> list[str]:
    """
    Universe:
      1) If symbols.txt exists → use listed tickers that have a parquet *and* a time-like column
      2) Else → infer all parquets with a valid schema

    We only keep symbols whose parquet has:
      - OHLCV columns
      - At least one TIME_CANDIDATES column
    """
    pq_dir = Path(cfg.PARQUET_DIR)
    sym_path = Path(getattr(cfg, "SYMBOLS_FILE", "symbols.txt"))

    all_parquets = {p.stem.upper(): p for p in pq_dir.rglob("*.parquet")}

    def _valid(sym: str) -> bool:
        p = all_parquets.get(sym)
        if not p:
            return False
        try:
            cols = _parquet_schema_columns(p)
        except Exception:
            return False
        has_ohlcv = {"open","high","low","close","volume"}.issubset(cols)
        has_time = any(c in cols for c in TIME_CANDIDATES)
        return has_ohlcv and has_time

    if sym_path.is_file():
        raw = [ln.strip().upper() for ln in sym_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        uniq = sorted(set(raw))
        missing = [s for s in uniq if s not in all_parquets]
        if missing:
            print(f"[get_symbols_from_file] Missing parquet for {len(missing)} listed symbols; first few: {missing[:10]}")
        symbols = [s for s in uniq if s in all_parquets and _valid(s)]
        dropped = [s for s in uniq if s in all_parquets and not _valid(s)]
        if dropped:
            print(f"[get_symbols_from_file] Dropped {len(dropped)} malformed parquets (no timestamp/ohlcv). First few: {dropped[:8]}")
    else:
        symbols = [s for s in sorted(all_parquets) if _valid(s)]
        print(f"[get_symbols_from_file] Using {len(symbols)} symbols inferred from parquet (schema-valid).")

    if not symbols:
        raise FileNotFoundError(
            f"No valid symbols. Fix malformed parquets under {pq_dir} or provide a curated symbols.txt."
        )
    return symbols

# ---------------- Robust parquet loader ----------------

def load_parquet_data(symbol: str,
                      start_date=None,
                      end_date=None,
                      drop_last_partial: bool = True,
                      columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load one symbol with robust UTC handling:
      - Ensures a time-like column is read even if `columns` is passed
      - Accepts epoch ns/ms/s and tz-aware strings (e.g., '2021-12-06 09:25:00+00:00')
      - Returns a DataFrame indexed by tz-aware UTC 'timestamp'
    """
    # Locate file
    p = Path(cfg.PARQUET_DIR) / f"{symbol}.parquet"
    if not p.exists():
        alts = list(Path(cfg.PARQUET_DIR).rglob(f"{symbol}.parquet"))
        if not alts:
            print(f"[load_parquet_data] {symbol}: parquet not found under {cfg.PARQUET_DIR}")
            return pd.DataFrame()
        p = alts[0]

    # Compute columns to read (add one time-like column if needed)
    try:
        schema_cols = _parquet_schema_columns(p)
    except Exception:
        schema_cols = set()
    cols_to_read = _columns_with_time(schema_cols, columns)

    # Read
    try:
        df = _read_parquet_arrow(p, columns=cols_to_read)
    except Exception as e:
        print(f"[load_parquet_data] {symbol}: failed to read parquet: {e}")
        return pd.DataFrame()

    # Build UTC DatetimeIndex
    ts = _normalize_timestamp_column(df)
    if ts is None:
        print(f"[load_parquet_data] {symbol}: could not derive timestamp column; skipping.")
        return pd.DataFrame()

    # Install index and drop helper columns
    df.index = ts
    df.index.name = "timestamp"
    # Drop any time-like columns we might have read
    for c in TIME_CANDIDATES:
        if c in df.columns:
            df = df.drop(columns=[c], errors="ignore")

    # Keep only requested OHLCV if `columns` was provided
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        if keep:
            df = df[keep]

    # Clean index
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]

    # Window
    if start_date is not None:
        df = df[df.index >= pd.to_datetime(start_date, utc=True, errors="coerce")]
    if end_date is not None:
        df = df[df.index <= pd.to_datetime(end_date, utc=True, errors="coerce")]

    # Optionally drop a still-forming last bar
    if drop_last_partial and not df.empty:
        now_utc = pd.Timestamp.now(tz="UTC")
        if (now_utc - df.index[-1]) < pd.Timedelta(minutes=5):
            df = df.iloc[:-1]

    return df

# ---------------- (optional) blacklist / category caches ----------------
_blacklist_cache: Optional[Set[str]] = None
_symbol_map_cache: Optional[Dict[str, str]] = None
_cg_details_cache: Optional[Dict[str, dict]] = None

def load_blacklist_data() -> None:
    """Leave as-is or implement if you use category filters."""
    pass

def is_blacklisted(symbol: str) -> bool:
    """No-op unless you wire CoinGecko categories; always returns False."""
    return False
