#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _load_5m_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Accept either:
    #  - a 'timestamp' column, OR
    #  - a DatetimeIndex named 'timestamp'
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp", drop=True)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{path} has no 'timestamp' column and index is not DatetimeIndex")
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.copy()
        df.index = idx
        df = df.dropna().sort_index()
        df.index.name = "timestamp"

    need = {"open", "high", "low", "close", "volume"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise ValueError(f"{path} missing required OHLCV columns: {missing}")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # Dedup and sort
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df


def _parse_symbol_paths(items: List[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--input must be like SYMBOL=/path/to/file.parquet, got: {it}")
        sym, p = it.split("=", 1)
        sym = sym.strip().upper()
        path = Path(p).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        out[sym] = path
    return out


def _tf_to_timedelta(tf: str) -> pd.Timedelta:
    t = tf.strip()
    t = t.replace("MIN", "min").replace("Min", "min")
    t = t.replace("H", "h").replace("D", "D")
    try:
        return pd.Timedelta(t)
    except Exception as e:
        raise ValueError(f"Cannot parse tf='{tf}' into Timedelta: {e!r}")


def _infer_base_td(idx: pd.DatetimeIndex) -> pd.Timedelta:
    d = idx.to_series().diff().dropna()
    if d.empty:
        return pd.Timedelta(minutes=5)
    # most common diff (mode); fallback to median; fallback 5m
    try:
        mode = d.value_counts().index[0]
        if isinstance(mode, pd.Timedelta) and mode > pd.Timedelta(0):
            return mode
    except Exception:
        pass
    try:
        med = d.median()
        if isinstance(med, pd.Timedelta) and med > pd.Timedelta(0):
            return med
    except Exception:
        pass
    return pd.Timedelta(minutes=5)


def resample_ohlcv_right_right(df: pd.DataFrame, tf: str, base_ts: str = "open") -> pd.DataFrame:
    """
    Right/right OHLCV resample with bar timestamps at bar CLOSE.

    If base_ts == "open": assumes df index is bar OPEN time and shifts by one base bar so that
    values are aligned to bar CLOSE time before resampling with label='right', closed='right'.
    If base_ts == "close": assumes df index already represents bar CLOSE time (no shift).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have DatetimeIndex")
    df = df.copy()
    df.index = df.index.tz_convert("UTC")

    base_td = _infer_base_td(df.index)

    if base_ts.lower() == "open":
        # Convert open-stamped bars into close-stamped bars.
        df.index = (df.index + base_td)
    elif base_ts.lower() == "close":
        pass
    else:
        raise ValueError("--base-ts must be 'open' or 'close'")

    r = df.resample(tf, label="right", closed="right")

    out = pd.DataFrame(
        {
            "open": r["open"].first(),
            "high": r["high"].max(),
            "low": r["low"].min(),
            "close": r["close"].last(),
            "volume": r["volume"].sum(),
        }
    ).dropna(how="any")

    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out.dropna().sort_index()
    out.index.name = "timestamp"
    return out


def _clip_df_for_tf_padding(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp, tfs: List[str]) -> pd.DataFrame:
    max_td = max([_tf_to_timedelta(tf) for tf in tfs]) if tfs else pd.Timedelta(0)
    in_start = start_ts - max_td
    df = df.sort_index()
    return df.loc[(df.index >= in_start) & (df.index <= end_ts)].copy()


def _export_tf(df5: pd.DataFrame, tf: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, out_path: Path, base_ts: str) -> None:
    out = resample_ohlcv_right_right(df5, tf, base_ts=base_ts)
    if out.empty:
        raise ValueError(f"Resample produced empty TF={tf} output for {out_path.name}")

    # Clip OUTPUT bars to requested close-timestamp window.
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)].copy()

    # Write with explicit timestamp column (safer than index-only).
    out = out.reset_index()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="raise")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="SYMBOL=/abs/path/to/5m.parquet (repeatable)")
    ap.add_argument("--start", required=True, help="UTC ISO timestamp")
    ap.add_argument("--end", required=True, help="UTC ISO timestamp")
    ap.add_argument("--out-dir", required=True, help="Output directory for fixtures")
    ap.add_argument("--tfs", default="1D,4h", help="Comma-separated timeframes to export (default: 1D,4h)")
    ap.add_argument("--base-ts", default="open", choices=["open", "close"], help="5m parquet timestamps semantics")
    args = ap.parse_args()

    start_ts = pd.to_datetime(args.start, utc=True, errors="raise")
    end_ts = pd.to_datetime(args.end, utc=True, errors="raise")
    out_dir = Path(args.out_dir).expanduser().resolve()
    tfs = [x.strip() for x in str(args.tfs).split(",") if x.strip()]

    sym_paths = _parse_symbol_paths(args.input)

    for sym, path in sym_paths.items():
        df5 = _load_5m_parquet(path)
        df5 = _clip_df_for_tf_padding(df5, start_ts, end_ts, tfs)
        if df5.empty:
            raise ValueError(f"{sym} 5m slice is empty after clipping to [{start_ts}, {end_ts}] (with TF padding)")

        for tf in tfs:
            tf_tag = tf.strip().upper().replace("MIN", "M")
            out_path = out_dir / f"{sym}_{tf_tag}.parquet"
            _export_tf(df5, tf, start_ts, end_ts, out_path, base_ts=args.base_ts)
            print(f"WROTE {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
