# normalize_existing_csvs.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path("data5")  # change if needed
NUMERIC_COLS = ("open","high","low","close","volume","turnover")

def parse_any_to_utc(series: pd.Series) -> pd.Series:
    """
    Parse a column to tz-aware UTC datetimes.
    Supports:
      - epoch ms (e.g. '1676620800000')
      - 'DD/MM/YYYY HH:MM'
      - ISO-like strings (with or without offset)
    """
    s = series.astype(str).str.strip()

    # Start with all-NaT tz-aware
    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")

    # 1) epoch ms
    m = s.str.fullmatch(r"\d+")
    if m.any():
        dt.loc[m] = pd.to_datetime(s.loc[m].astype("int64"), unit="ms", utc=True, errors="coerce")

    # 2) legacy DD/MM/YYYY HH:MM (no '-')
    m = dt.isna() & s.str.contains("/", na=False) & ~s.str.contains("-", na=False)
    if m.any():
        dt.loc[m] = pd.to_datetime(s.loc[m], format="%d/%m/%Y %H:%M", utc=True, errors="coerce")

    # 3) general ISO fallback
    m = dt.isna()
    if m.any():
        dt.loc[m] = pd.to_datetime(s.loc[m], utc=True, errors="coerce")

    if dt.isna().any():
        bad = s[dt.isna()].head().tolist()
        raise ValueError(f"Unparseable timestamps, e.g. {bad}")
    return dt

def normalize_csv(path: Path):
    df = pd.read_csv(path)
    if "open_time" not in df.columns:
        print(f"SKIP (no open_time): {path.name}")
        return

    # Parse timestamps (into tz-aware UTC)
    df["open_time"] = parse_any_to_utc(df["open_time"])

    # Ensure numeric
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean + order
    df = (
        df.dropna(subset=["open_time"])
          .drop_duplicates(subset=["open_time"])
          .sort_values("open_time")
    )

    # Write back (ISO UTC)
    df.to_csv(path, index=False)
    print(
        f"Normalized: {path.name}  rows={len(df)}  "
        f"first={df['open_time'].iloc[0]}  last={df['open_time'].iloc[-1]}"
    )

def main():
    paths = sorted(ROOT.glob("*.csv"))
    print(f"Found {len(paths)} files under {ROOT}")
    for p in paths:
        try:
            normalize_csv(p)
        except Exception as e:
            print(f"ERROR {p.name}: {e}")

if __name__ == "__main__":
    main()
