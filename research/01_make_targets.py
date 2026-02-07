#!/usr/bin/env python3
"""
research/01_make_targets.py

Step 1: Create target labels from trades.clean.csv (chunked).

Targets created (where possible):
  - y_win        : binary "win" (prefers WIN col; fallback pnl_R>0; fallback pnl>0)
  - y_good_05    : pnl_R >= 0.5 (if pnl_R exists)
  - y_good_10    : pnl_R >= 1.0 (if pnl_R exists)
  - y_time       : TIME exit (EXIT_FINAL==2 OR exit_reason indicates time)
  - y_tp         : TP exit (EXIT_FINAL==1 OR exit_reason indicates tp/trail)
  - y_sl         : SL exit (EXIT_FINAL==0 OR exit_reason indicates sl/immediate_sl)
  - y_exit_class : {0:SL, 1:TP, 2:TIME, -1:UNKNOWN}

Outputs (in --outdir):
  - <outfile>.parquet  (default targets.parquet)
  - targets_report.json
  - (optional) targets.csv.gz  if --also-csv

Primary key for merging later: trade_id (required).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TP_REASON_TOKENS = (
    "tp", "take", "profit", "tp_final", "trail", "trailing", "partial_tp", "tp_partial"
)
SL_REASON_TOKENS = (
    "sl", "stop", "stoploss", "stop_loss", "immediate_sl", "liquid", "liq"
)
TIME_REASON_TOKENS = (
    "time", "timeout", "time_limit", "timelimit", "max_hold", "maxhold"
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_int01(x: pd.Series) -> pd.Series:
    """Convert boolean-ish series to Int8 with NA->0 only if already boolean; otherwise preserve NA."""
    if pd.api.types.is_bool_dtype(x):
        return x.astype("int8")
    return pd.to_numeric(x, errors="coerce").astype("Int64")


def _infer_win(chunk: pd.DataFrame) -> Tuple[pd.Series, str]:
    """
    Returns (y_win_int8, source_str)
    Preference:
      1) WIN if present and valid 0/1
      2) pnl_R > 0 if pnl_R present
      3) pnl > 0 if pnl present
    """
    if "WIN" in chunk.columns:
        win = pd.to_numeric(chunk["WIN"], errors="coerce")
        y = (win == 1)
        # If WIN is missing for some rows, they become False; we'll set those to NA later
        y = y.where(win.notna(), other=pd.NA)
        return y.astype("Int8"), "WIN"

    if "pnl_R" in chunk.columns:
        pr = pd.to_numeric(chunk["pnl_R"], errors="coerce")
        y = (pr > 0).where(pr.notna(), other=pd.NA)
        return y.astype("Int8"), "pnl_R>0"

    if "pnl" in chunk.columns:
        pnl = pd.to_numeric(chunk["pnl"], errors="coerce")
        y = (pnl > 0).where(pnl.notna(), other=pd.NA)
        return y.astype("Int8"), "pnl>0"

    # No info: all NA
    return pd.Series(pd.array([pd.NA] * len(chunk), dtype="Int8"), index=chunk.index), "NA"


def _lower_str(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower()


def _reason_has_any(reason_lc: pd.Series, tokens: Tuple[str, ...]) -> pd.Series:
    """
    Vectorized token search:
      - Treat NaN as empty
      - True if any token appears as substring
    """
    r = reason_lc.fillna("")
    mask = pd.Series(False, index=r.index)
    for t in tokens:
        mask = mask | r.str.contains(t, regex=False)
    return mask


def _infer_exit_targets(chunk: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Build y_time/y_tp/y_sl/y_exit_class using:
      - EXIT_FINAL if present (0/1/2)
      - exit_reason string tokens as fallback/augmentation
    """
    out: Dict[str, pd.Series] = {}

    ef = None
    if "EXIT_FINAL" in chunk.columns:
        ef = pd.to_numeric(chunk["EXIT_FINAL"], errors="coerce")

    reason_lc = None
    if "exit_reason" in chunk.columns:
        reason_lc = _lower_str(chunk["exit_reason"])

    # Start with NA masks
    idx = chunk.index
    y_time = pd.Series(pd.array([pd.NA] * len(chunk), dtype="Int8"), index=idx)
    y_tp = pd.Series(pd.array([pd.NA] * len(chunk), dtype="Int8"), index=idx)
    y_sl = pd.Series(pd.array([pd.NA] * len(chunk), dtype="Int8"), index=idx)

    if ef is not None:
        y_time = (ef == 2).where(ef.notna(), other=pd.NA).astype("Int8")
        y_tp = (ef == 1).where(ef.notna(), other=pd.NA).astype("Int8")
        y_sl = (ef == 0).where(ef.notna(), other=pd.NA).astype("Int8")

    # If exit_reason exists, fill missing where ef is NA, and also "augment" UNKNOWN cases
    if reason_lc is not None:
        time_r = _reason_has_any(reason_lc, TIME_REASON_TOKENS)
        tp_r = _reason_has_any(reason_lc, TP_REASON_TOKENS)
        sl_r = _reason_has_any(reason_lc, SL_REASON_TOKENS)

        # Where ef is missing, use reason-based
        if ef is None:
            y_time = time_r.astype("Int8")
            y_tp = tp_r.astype("Int8")
            y_sl = sl_r.astype("Int8")
        else:
            y_time = y_time.where(y_time.notna(), other=time_r.astype("Int8"))
            y_tp = y_tp.where(y_tp.notna(), other=tp_r.astype("Int8"))
            y_sl = y_sl.where(y_sl.notna(), other=sl_r.astype("Int8"))

    # Construct class with priority:
    # TIME > TP > SL (TIME is distinct, TP covers trail/tp_final, SL covers sl/immediate_sl)
    # Unknown if none matched.
    y_exit = pd.Series(np.full(len(chunk), -1, dtype=np.int16), index=idx)
    # Only set where we have any non-NA info
    # Use boolean comparisons carefully with NA
    time_mask = (y_time == 1)
    tp_mask = (y_tp == 1)
    sl_mask = (y_sl == 1)

    y_exit = y_exit.mask(time_mask.fillna(False), 2)
    y_exit = y_exit.mask((~time_mask.fillna(False)) & tp_mask.fillna(False), 1)
    y_exit = y_exit.mask((~time_mask.fillna(False)) & (~tp_mask.fillna(False)) & sl_mask.fillna(False), 0)

    out["y_time"] = y_time
    out["y_tp"] = y_tp
    out["y_sl"] = y_sl
    out["y_exit_class"] = y_exit.astype("int16")

    return out


def _good_thresholds(chunk: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Create y_good_05 / y_good_10 if pnl_R exists, else return empty.
    """
    if "pnl_R" not in chunk.columns:
        return {}
    pr = pd.to_numeric(chunk["pnl_R"], errors="coerce")
    y05 = (pr >= 0.5).where(pr.notna(), other=pd.NA).astype("Int8")
    y10 = (pr >= 1.0).where(pr.notna(), other=pd.NA).astype("Int8")
    return {"y_good_05": y05, "y_good_10": y10}


def _summarize_counts(arr: pd.Series) -> Dict[str, int]:
    """
    Summarize Int-like series into counts.
    """
    s = pd.to_numeric(arr, errors="coerce")
    return {
        "n": int(s.notna().sum()),
        "n_1": int((s == 1).sum()),
        "n_0": int((s == 0).sum()),
        "n_na": int(s.isna().sum()),
    }


def _rate(n1: int, n: int) -> Optional[float]:
    if n <= 0:
        return None
    return float(n1) / float(n)


def process(
    infile: str,
    outdir: str,
    outfile: str,
    chunksize: int,
    also_csv: bool,
) -> None:
    _ensure_dir(outdir)
    out_parquet_path = os.path.join(outdir, outfile)
    out_csv_path = os.path.join(outdir, "targets.csv.gz")
    report_path = os.path.join(outdir, "targets_report.json")

    if not os.path.exists(infile):
        raise FileNotFoundError(f"Input file not found: {infile}")

    # Read header
    header = pd.read_csv(infile, nrows=0)
    cols = list(header.columns)
    if "trade_id" not in cols:
        raise RuntimeError("trade_id is required to create mergeable targets.")
    if "symbol" not in cols:
        raise RuntimeError("symbol is required (for reporting sanity).")

    # Column subset needed
    need = ["trade_id", "symbol"]
    for c in ["entry_ts", "exit_ts", "pnl", "pnl_R", "WIN", "EXIT_FINAL", "exit_reason"]:
        if c in cols:
            need.append(c)

    # For Parquet, easiest is collect fragments and concat at end.
    # For huge files, this is still typically manageable because we output only a few columns.
    out_parts: List[pd.DataFrame] = []

    # Report accumulators
    report: Dict[str, object] = {
        "infile": infile,
        "outfile": outfile,
        "win_source": None,
        "total_rows": 0,
        "targets": {},
        "consistency": {},
    }

    # For consistency: WIN vs pnl_R sign (if both exist)
    n_mismatch = 0
    n_both_valid = 0

    reader = pd.read_csv(infile, chunksize=chunksize, low_memory=False, usecols=need)

    win_source_global: Optional[str] = None

    for i, chunk in enumerate(reader):
        n = len(chunk)
        if n == 0:
            continue
        report["total_rows"] = int(report["total_rows"]) + n

        # Ensure trade_id numeric
        trade_id = pd.to_numeric(chunk["trade_id"], errors="coerce")
        valid_tid = trade_id.notna()
        if valid_tid.sum() < n:
            chunk = chunk.loc[valid_tid].copy()
            trade_id = trade_id.loc[valid_tid]
            n = len(chunk)
            if n == 0:
                continue
        chunk["trade_id"] = trade_id.astype("int64")

        # y_win
        y_win, win_source = _infer_win(chunk)
        if win_source_global is None:
            win_source_global = win_source
            report["win_source"] = win_source_global

        # Exit targets
        exit_targets = _infer_exit_targets(chunk)

        # Good thresholds
        good_targets = _good_thresholds(chunk)

        # Optional: carry entry_ts/exit_ts for later purging/splitting (kept minimal)
        keep_cols = ["trade_id"]
        if "symbol" in chunk.columns:
            keep_cols.append("symbol")
        if "entry_ts" in chunk.columns:
            keep_cols.append("entry_ts")
        if "exit_ts" in chunk.columns:
            keep_cols.append("exit_ts")

        out_df = chunk[keep_cols].copy()
        out_df["y_win"] = y_win

        for k, v in good_targets.items():
            out_df[k] = v
        for k, v in exit_targets.items():
            out_df[k] = v

        out_parts.append(out_df)

        # Consistency: if WIN and pnl_R both exist in this chunk
        if ("WIN" in chunk.columns) and ("pnl_R" in chunk.columns):
            win_num = pd.to_numeric(chunk["WIN"], errors="coerce")
            pr = pd.to_numeric(chunk["pnl_R"], errors="coerce")
            valid = win_num.isin([0, 1]) & pr.notna()
            mismatch = ((win_num == 1) & (pr <= 0)) | ((win_num == 0) & (pr > 0))
            n_both_valid += int(valid.sum())
            n_mismatch += int((valid & mismatch).sum())

        if (i + 1) % 10 == 0:
            print(f"[01_make_targets] processed chunks={i+1} rows={int(report['total_rows']):,}", flush=True)

    if not out_parts:
        raise RuntimeError("No output produced (empty input or all trade_id invalid).")

    targets = pd.concat(out_parts, ignore_index=True)

    # Write outputs
    targets.to_parquet(out_parquet_path, index=False)

    if also_csv:
        targets.to_csv(out_csv_path, index=False, compression="gzip")

    # Build report stats
    def add_target_stats(name: str) -> None:
        if name not in targets.columns:
            return
        counts = _summarize_counts(targets[name])
        rate = _rate(counts["n_1"], counts["n"])
        report["targets"][name] = {
            **counts,
            "rate_1": rate,
        }

    for col in ["y_win", "y_good_05", "y_good_10", "y_time", "y_tp", "y_sl"]:
        add_target_stats(col)

    # Exit class distribution
    if "y_exit_class" in targets.columns:
        vc = targets["y_exit_class"].value_counts(dropna=False).to_dict()
        # Convert numpy ints to python ints
        vc2 = {str(int(k)): int(v) for k, v in vc.items()}
        report["targets"]["y_exit_class_counts"] = vc2

    # Consistency report
    if n_both_valid > 0:
        report["consistency"]["WIN_vs_pnl_R"] = {
            "n_valid_both": int(n_both_valid),
            "n_mismatch": int(n_mismatch),
            "mismatch_rate": float(n_mismatch) / float(n_both_valid),
        }
    else:
        report["consistency"]["WIN_vs_pnl_R"] = None

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"[01_make_targets] DONE. Wrote: {out_parquet_path}")
    if also_csv:
        print(f"[01_make_targets] Also wrote: {out_csv_path}")
    print(f"[01_make_targets] Report: {report_path}")


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 1: create target labels from trades CSV.")
    p.add_argument("--infile", type=str, default="results/trades.clean.csv", help="Input trades CSV.")
    p.add_argument("--outdir", type=str, default="research_outputs/01_targets", help="Output directory.")
    p.add_argument("--outfile", type=str, default="targets.parquet", help="Parquet filename (inside outdir).")
    p.add_argument("--chunksize", type=int, default=250_000, help="CSV chunk size.")
    p.add_argument("--also-csv", action="store_true", help="Also write targets.csv.gz.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if int(args.chunksize) < 10_000:
        raise ValueError("--chunksize too small; use at least 10,000.")
    process(
        infile=args.infile,
        outdir=args.outdir,
        outfile=args.outfile,
        chunksize=int(args.chunksize),
        also_csv=bool(args.also_csv),
    )


if __name__ == "__main__":
    main()
