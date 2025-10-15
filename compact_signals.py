#!/usr/bin/env python3
"""
Compact partitioned signals (signals/symbol=*) into a single signals.parquet,
unifying types across partitions and streaming row-groups to keep memory flat.

- Text-like cols (symbol, pullback_type, entry_rule) -> string
- Numeric features we model on (vol_mult, rs_pct, etc.) -> float64
- Small ints kept as int8/uint8 as configured
- timestamp normalized to timestamp[ns, tz=UTC]
- Missing columns are added as nulls of correct type

Refs:
- Pandas can read a directory of partitioned Parquet files with engine='pyarrow'.  [pandas.read_parquet] :contentReference[oaicite:2]{index=2}
- ParquetWriter requires identical schema for all row groups.                         :contentReference[oaicite:3]{index=3}
- Arrow casting is done with pyarrow.compute.cast.                                   :contentReference[oaicite:4]{index=4}
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.types as pat

# ---------- configuration of target types you care about ----------
FORCE_STRING = {"symbol", "pullback_type", "entry_rule"}
FORCE_FLOAT64 = {"vol_mult", "rs_pct", "atr", "atr_1h", "rsi_1h", "adx_1h",
                 "entry", "don_break_level", "atr_pct", "close", "don_upper", "don_dist_atr"}
FORCE_INT8 = {"don_break_len"}
FORCE_UINT8 = {"vol_spike"}
TIMESTAMP_COL = "timestamp"  # expected to be UTC

# ---------- helpers ----------

def _resolve_in_path(p: str | Path) -> Path:
    p = Path(p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    if p.is_file():
        return p
    if not any(p.rglob("*.parquet")):
        raise FileNotFoundError(f"{p} contains no parquet files")
    return p

def _list_parquet_files(in_path: Path, out_file: Path | None) -> List[Path]:
    files = [in_path] if in_path.is_file() else sorted(f for f in in_path.rglob("*.parquet"))
    if out_file is not None:
        out_file = out_file.resolve()
        files = [f for f in files if f.resolve() != out_file]
    if not files:
        raise FileNotFoundError("No parquet files found to compact.")
    return files

def _arrow_type_for(name: str, seen: Set[pa.DataType]) -> pa.DataType:
    # Hard overrides first
    if name in FORCE_STRING:
        return pa.string()
    if name in FORCE_FLOAT64:
        return pa.float64()
    if name in FORCE_INT8:
        return pa.int8()
    if name in FORCE_UINT8:
        return pa.uint8()
    if name == TIMESTAMP_COL:
        # normalize to timestamp[ns, tz=UTC]
        return pa.timestamp("ns", tz="UTC")

    # Otherwise promote based on types we've seen
    if any(pat.is_dictionary(t) for t in seen):
        return pa.string()
    if any(pat.is_floating(t) for t in seen):
        return pa.float64()
    if any(pat.is_integer(t) or pat.is_unsigned_integer(t) for t in seen):
        return pa.int64()
    if any(pat.is_boolean(t) for t in seen):
        return pa.bool_()
    if any(pat.is_timestamp(t) for t in seen):
        # prefer UTC ns if any timestamp; assume/force UTC
        return pa.timestamp("ns", tz="UTC")
    # Fallback to string for weird/unknowns
    return pa.string()

def _collect_type_sets(files: List[Path]) -> Dict[str, Set[pa.DataType]]:
    type_sets: Dict[str, Set[pa.DataType]] = {}
    for path in files:
        pf = pq.ParquetFile(path)
        sch = pf.schema_arrow  # Arrow schema is easier to reason about here
        for field in sch:
            type_sets.setdefault(field.name, set()).add(field.type)
    return type_sets

def _build_target_schema(type_sets: Dict[str, Set[pa.DataType]]) -> pa.Schema:
    fields = []
    # Keep a stable-ish order: timestamp, symbol, others sorted
    cols = list(type_sets.keys())
    cols.sort()
    if TIMESTAMP_COL in cols:
        cols.remove(TIMESTAMP_COL)
        cols = [TIMESTAMP_COL] + cols
    if "symbol" in cols:
        cols.remove("symbol")
        cols = ["symbol"] + cols

    for name in cols:
        target = _arrow_type_for(name, type_sets[name])
        fields.append(pa.field(name, target))
    return pa.schema(fields)

def _cast_or_add(table: pa.Table, target_schema: pa.Schema) -> pa.Table:
    """Return a table with exactly target_schema columns and types."""
    columns = []
    n = table.num_rows
    present = {f.name for f in table.schema}
    for field in target_schema:
        if field.name in present:
            col = table[field.name]
            src_type = col.type
            dst_type = field.type
            # dictionary or mismatched → cast
            if src_type != dst_type or pat.is_dictionary(src_type):
                try:
                    col = pc.cast(col, dst_type)  # handles NullType -> float64, etc.  :contentReference[oaicite:5]{index=5}
                except Exception:
                    # fallback: build a null array of dst_type
                    col = pa.nulls(n)
                    col = pc.cast(col, dst_type)
        else:
            col = pc.cast(pa.nulls(n), field.type)
        columns.append(col)
    return pa.Table.from_arrays(columns, schema=target_schema)

def _fast_compact(in_path: Path, out_file: Path, sort: bool) -> int:
    # Pandas supports reading a partitioned directory with engine='pyarrow'. :contentReference[oaicite:6]{index=6}
    df = pd.read_parquet(in_path, engine="pyarrow")
    if sort and TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True, errors="coerce")
        if "symbol" in df.columns:
            df["symbol"] = df["symbol"].astype(str)
        df = df.sort_values([TIMESTAMP_COL, "symbol"], kind="mergesort", ignore_index=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_file, engine="pyarrow", compression="snappy", index=False)
    return len(df)

def _stream_compact(files: List[Path], out_file: Path, row_group_size: int = 100_000) -> int:
    # Build a unified target schema across all inputs
    type_sets = _collect_type_sets(files)
    target_schema = _build_target_schema(type_sets)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_file, schema=target_schema, compression="snappy", version="2.6")
    total_rows = 0
    try:
        for path in files:
            pf = pq.ParquetFile(path)
            for rg in range(pf.metadata.num_row_groups):
                tbl = pf.read_row_group(rg)
                tbl = _cast_or_add(tbl, target_schema)  # ensure exact schema match before writing
                writer.write_table(tbl, row_group_size=row_group_size)  # supports row_group_size  :contentReference[oaicite:7]{index=7}
                total_rows += tbl.num_rows
    finally:
        writer.close()
    return total_rows

def _print_metadata(out_file: Path) -> None:
    meta = pq.read_metadata(out_file)  # FileMetaData with num_columns/num_row_groups  :contentReference[oaicite:8]{index=8}
    num_cols = getattr(meta, "num_columns", None)
    if num_cols is None:
        num_cols = len(pq.ParquetFile(out_file).schema_arrow.names)
    print(f"[ok] wrote → {out_file}")
    print(f"[info] row groups: {meta.num_row_groups}, columns: {num_cols}, rows: {meta.num_rows}")

def main():
    ap = argparse.ArgumentParser(description="Compact partitioned signals into one Parquet file with unified schema.")
    ap.add_argument("--in", dest="in_path", type=str, default="/opt/testerdonch/signals",
                    help="Input directory (partitioned signals) or single parquet file.")
    ap.add_argument("--out", dest="out_file", type=str, default=None,
                    help="Output Parquet file (default: <IN>/signals.parquet if IN is a directory).")
    ap.add_argument("--mode", choices=["fast", "streamed"], default="fast",
                    help="fast=pandas (global sort optional); streamed=low-memory with schema unification.")
    ap.add_argument("--no-sort", action="store_true", help="Disable sorting (fast mode only).")
    ap.add_argument("--row-group-size", type=int, default=100_000, help="Row group size (streamed mode).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output file if present.")
    args = ap.parse_args()

    in_path = _resolve_in_path(args.in_path)
    out_file = Path(args.out_file) if args.out_file else (in_path / "signals.parquet" if in_path.is_dir() else in_path)
    out_file = out_file.resolve()

    if out_file.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {out_file} (use --overwrite)")

    if args.mode == "fast":
        try:
            n = _fast_compact(in_path, out_file, sort=not args.no_sort)
            print(f"[fast] wrote {n:,} rows")
            _print_metadata(out_file)
            return
        except Exception as e:
            msg = str(e)
            if "Unable to merge: Field" in msg and "incompatible types" in msg:
                print("[fast] schema conflict detected; falling back to streamed compaction…")
            else:
                raise

    files = _list_parquet_files(in_path, out_file)
    n = _stream_compact(files, out_file, row_group_size=args.row_group_size)
    print(f"[streamed] wrote {n:,} rows")
    _print_metadata(out_file)

if __name__ == "__main__":
    main()
