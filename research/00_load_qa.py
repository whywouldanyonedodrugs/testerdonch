#!/usr/bin/env python3
"""
research/00_load_qa.py

Step 0: Data QA for results/trades.clean.csv (or any similar trade-level export).

Outputs (in --outdir):
  - feature_coverage.csv
  - missingness_by_symbol_long.csv
  - missingness_by_month_long.csv
  - missingness_by_symbol_month_key.csv
  - sanity_checks.csv
  - label_sanity.csv
  - flags_by_trade_id.csv.gz           (trade_id + has_oi/has_funding/has_btc_context/has_eth_context + entry_month)
  - qa_report.html
  - data_contract_columns.csv         (column names + inferred dtypes from sample)

Design goals:
  - Chunked processing for very large CSVs
  - Coverage/missingness overall + by symbol + by month
  - Key crypto-context missingness by (symbol, month)
  - Basic time/price/quantity sanity checks
  - Label consistency checks for WIN / pnl_R / EXIT_FINAL
  - Generate "missingness flags" as live-eligible predictors (stored separately for merge)

Usage:
  python research/00_load_qa.py \
      --infile results/trades.clean.csv \
      --outdir research_outputs/00_qa \
      --chunksize 250000
"""

from __future__ import annotations

import argparse
import gzip
import html
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Column groups (robust to missing columns)
# ----------------------------

OI_COLS = [
    "oi_level",
    "oi_notional_est",
    "est_leverage",
    "oi_pct_1h",
    "oi_pct_4h",
    "oi_pct_1d",
    "oi_z_7d",
    "oi_chg_norm_vol_1h",
    "oi_price_div_1h",
    "crowded_long",
    "crowded_short",
    "crowd_side",
]

FUNDING_COLS = [
    "funding_rate",
    "funding_abs",
    "funding_z_7d",
    "funding_rollsum_3d",
    "funding_oi_div",
]

BTC_CONTEXT_COLS = [
    "btc_funding_rate",
    "btc_oi_z_7d",
    "btc_vol_regime_level",
    "btc_trend_slope",
]

# Note: "eth_*" can mean either cross-asset context fields OR ETH MACD context.
ETH_CONTEXT_COLS = [
    "eth_funding_rate",
    "eth_oi_z_7d",
    "eth_vol_regime_level",
    "eth_trend_slope",
    "eth_macd_line_4h",
    "eth_macd_signal_4h",
    "eth_macd_hist_4h",
    "eth_macd_both_pos_4h",
    "eth_macd_hist_slope_4h",
    "eth_macd_hist_slope_1h",
]

# For symbol-month missingness matrix, focus on the columns that are known to be patchy / operationally critical
KEY_SYMBOL_MONTH_COLS = sorted(set(OI_COLS + FUNDING_COLS + BTC_CONTEXT_COLS + ETH_CONTEXT_COLS))


# ----------------------------
# Utilities
# ----------------------------

def _safe_cols(all_cols: List[str], candidates: List[str]) -> List[str]:
    s = set(all_cols)
    return [c for c in candidates if c in s]


def _infer_ts_unit(ts_max: float) -> str:
    """
    Heuristic:
      - seconds since epoch ~ 1.6e9 to 2.0e9 for modern dates
      - milliseconds since epoch ~ 1.6e12+
    """
    if not np.isfinite(ts_max):
        return "unknown"
    if ts_max > 1e12:
        return "ms"
    return "s"


def _to_datetime_from_ts(series: pd.Series) -> pd.Series:
    """
    Robust conversion:
      - if numeric: infer s vs ms and convert
      - if already datetime: return
      - if string: attempt pd.to_datetime
    Returns UTC-aware timestamps (DatetimeTZDtype) where possible.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        # If tz-naive, localize to UTC; if tz-aware keep
        try:
            if getattr(series.dt, "tz", None) is None:
                return series.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
            return series
        except Exception:
            return pd.to_datetime(series, errors="coerce", utc=True)

    if pd.api.types.is_numeric_dtype(series):
        s = pd.to_numeric(series, errors="coerce")
        ts_max = s.max()
        unit = _infer_ts_unit(float(ts_max) if ts_max is not None else np.nan)
        if unit == "ms":
            return pd.to_datetime(s, unit="ms", errors="coerce", utc=True)
        if unit == "s":
            return pd.to_datetime(s, unit="s", errors="coerce", utc=True)
        return pd.to_datetime(s, errors="coerce", utc=True)

    return pd.to_datetime(series, errors="coerce", utc=True)


def _month_key(dt_utc: pd.Series) -> pd.Series:
    """
    Month key as YYYY-MM string, tz-safe (no Period conversion warning).
    NaT becomes NaN.
    """
    return dt_utc.dt.strftime("%Y-%m")


def _reset_index_named(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Pandas can drop/lose index.name after align/add operations.
    Also, df can already contain a column with the same name as the desired index column
    (e.g. the dataset has a 'symbol' column, and we also want the index to be 'symbol').

    This helper:
      - Renames any existing column == name to a safe alternative (name__col, name__col2, ...)
      - Sets index.name = name
      - Resets index and ensures the index column is actually called `name`
    """
    df2 = df.copy()

    # If there's already a column called `name`, rename it to avoid reset_index() collision.
    if name in df2.columns:
        new_name = f"{name}__col"
        k = 2
        while new_name in df2.columns:
            new_name = f"{name}__col{k}"
            k += 1
        df2 = df2.rename(columns={name: new_name})

    df2.index.name = name
    out = df2.reset_index()

    # Defensive: some pandas paths still call it "index"
    if name not in out.columns and "index" in out.columns:
        out = out.rename(columns={"index": name})

    return out



def _html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "<p><em>(empty)</em></p>"
    if len(df) > max_rows:
        df2 = df.head(max_rows).copy()
        return df2.to_html(index=False, escape=True) + f"<p><em>Showing first {max_rows} rows of {len(df)}.</em></p>"
    return df.to_html(index=False, escape=True)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _gzip_append_csv(path: str, df: pd.DataFrame, header: bool) -> None:
    """
    Append a CSV fragment to a gzip file in TEXT mode.
    Appending creates multiple gzip members, which is valid and readable by gzip tools/pandas.
    """
    mode = "wt" if header else "at"
    with gzip.open(path, mode, encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False, header=header)


# ----------------------------
# Accumulators
# ----------------------------

@dataclass
class QAAccumulators:
    total_rows: int
    null_counts: pd.Series  # per column
    sym_totals: pd.Series
    sym_nulls: pd.DataFrame  # index: symbol, columns: all cols
    month_totals: pd.Series
    month_nulls: pd.DataFrame  # index: entry_month, columns: all cols
    sym_month_totals: pd.Series  # MultiIndex (symbol, entry_month)
    sym_month_nulls: pd.DataFrame  # MultiIndex (symbol, entry_month), columns: key cols
    sanity_counts: Dict[str, int]
    label_counts: Dict[str, int]


def _init_accumulators(all_cols: List[str], key_sym_month_cols: List[str]) -> QAAccumulators:
    null_counts = pd.Series(0, index=all_cols, dtype="int64")

    sym_totals = pd.Series(dtype="int64")
    sym_nulls = pd.DataFrame(columns=all_cols, dtype="int64")

    month_totals = pd.Series(dtype="int64")
    month_nulls = pd.DataFrame(columns=all_cols, dtype="int64")

    sym_month_totals = pd.Series(dtype="int64")
    sym_month_nulls = pd.DataFrame(columns=key_sym_month_cols, dtype="int64")

    sanity_counts = {
        "rows_with_bad_entry_ts": 0,
        "rows_with_bad_exit_ts": 0,
        "rows_with_entry_ge_exit": 0,
        "rows_with_nonpositive_entry_price": 0,
        "rows_with_nonpositive_exit_price": 0,
        "rows_with_nonpositive_qty": 0,
        "rows_with_nan_pnl": 0,
        "rows_with_nan_pnl_R": 0,
        "rows_with_nan_WIN": 0,
        "rows_with_bad_EXIT_FINAL": 0,
    }

    label_counts = {
        "rows_with_WIN_1": 0,
        "rows_with_WIN_0": 0,
        "rows_with_pnl_R_pos": 0,
        "rows_with_pnl_R_nonpos": 0,
        "rows_with_EXIT_FINAL_TP": 0,
        "rows_with_EXIT_FINAL_SL": 0,
        "rows_with_EXIT_FINAL_TIME": 0,
        "rows_with_WIN_mismatch_pnl_R": 0,
    }

    return QAAccumulators(
        total_rows=0,
        null_counts=null_counts,
        sym_totals=sym_totals,
        sym_nulls=sym_nulls,
        month_totals=month_totals,
        month_nulls=month_nulls,
        sym_month_totals=sym_month_totals,
        sym_month_nulls=sym_month_nulls,
        sanity_counts=sanity_counts,
        label_counts=label_counts,
    )


def _add_series(acc: pd.Series, other: pd.Series) -> pd.Series:
    if acc.empty:
        return other.copy().astype("int64")
    return acc.add(other, fill_value=0).astype("int64")


def _add_df(acc: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    if acc.empty:
        return other.copy().astype("int64")
    acc2, other2 = acc.align(other, join="outer", axis=1, fill_value=0)
    out = acc2.add(other2, fill_value=0)
    return out.fillna(0).astype("int64")


# ----------------------------
# Core QA logic
# ----------------------------

def process_csv(
    infile: str,
    outdir: str,
    chunksize: int,
    write_flags: bool,
    flags_out_name: str,
    sample_n: int = 5000,
) -> None:
    _ensure_dir(outdir)

    # Read header-only to get column list
    header_df = pd.read_csv(infile, nrows=0)
    all_cols = list(header_df.columns)
    if not all_cols:
        raise RuntimeError("No columns detected in input CSV.")

    if "symbol" not in all_cols:
        raise RuntimeError("Required column 'symbol' missing from input.")

    key_sym_month_cols = _safe_cols(all_cols, KEY_SYMBOL_MONTH_COLS)

    # Data contract snapshot from sample
    sample_df = pd.read_csv(infile, nrows=sample_n, low_memory=False)
    contract = pd.DataFrame(
        {
            "column": sample_df.columns,
            "inferred_dtype": [str(sample_df[c].dtype) for c in sample_df.columns],
            "sample_nonnull_rate": [float(sample_df[c].notna().mean()) for c in sample_df.columns],
        }
    )
    contract.to_csv(os.path.join(outdir, "data_contract_columns.csv"), index=False)

    acc = _init_accumulators(all_cols=all_cols, key_sym_month_cols=key_sym_month_cols)

    # Prepare flags output
    flags_out_path = os.path.join(outdir, flags_out_name)
    if write_flags and os.path.exists(flags_out_path):
        os.remove(flags_out_path)

    first_flags_chunk = True

    reader = pd.read_csv(infile, chunksize=chunksize, low_memory=False)

    for i, chunk in enumerate(reader):
        n = len(chunk)
        if n == 0:
            continue
        acc.total_rows += n

        # Overall null counts
        chunk_nulls = chunk.isna().sum(numeric_only=False).reindex(all_cols, fill_value=0).astype("int64")
        acc.null_counts = acc.null_counts.add(chunk_nulls, fill_value=0).astype("int64")

        # Symbol series
        sym = chunk["symbol"].astype(str)

        # Entry month (UTC) for groupings (prefer entry_ts, fallback to exit_ts)
        entry_month = None
        if "entry_ts" in chunk.columns:
            entry_dt = _to_datetime_from_ts(chunk["entry_ts"])
            entry_month = _month_key(entry_dt)
        elif "exit_ts" in chunk.columns:
            exit_dt = _to_datetime_from_ts(chunk["exit_ts"])
            entry_month = _month_key(exit_dt)

        # Missingness by symbol (all columns)
        nulls_by_sym = chunk.isna().groupby(sym, sort=False).sum().reindex(columns=all_cols, fill_value=0).astype("int64")
        totals_by_sym = sym.value_counts(dropna=False).astype("int64")
        acc.sym_nulls = _add_df(acc.sym_nulls, nulls_by_sym)
        acc.sym_totals = _add_series(acc.sym_totals, totals_by_sym)

        # Missingness by month (all columns)
        if entry_month is not None:
            m = entry_month.astype(str)
            nulls_by_month = chunk.isna().groupby(m, sort=False).sum().reindex(columns=all_cols, fill_value=0).astype("int64")
            totals_by_month = m.value_counts(dropna=False).astype("int64")
            acc.month_nulls = _add_df(acc.month_nulls, nulls_by_month)
            acc.month_totals = _add_series(acc.month_totals, totals_by_month)

            # Missingness by (symbol, month) for key cols
            if key_sym_month_cols:
                key_null = chunk[key_sym_month_cols].isna().copy()
                key_null["symbol"] = sym.values
                key_null["entry_month"] = m.values
                sm_nulls = key_null.groupby(["symbol", "entry_month"], sort=False).sum(numeric_only=True).astype("int64")

                sm_totals = chunk.groupby([sym.values, m.values], sort=False).size().astype("int64")
                sm_totals.index = pd.MultiIndex.from_arrays(
                    [sm_totals.index.get_level_values(0), sm_totals.index.get_level_values(1)],
                    names=["symbol", "entry_month"],
                )

                acc.sym_month_nulls = _add_df(acc.sym_month_nulls, sm_nulls)
                acc.sym_month_totals = _add_series(acc.sym_month_totals, sm_totals)

        # ----------------------------
        # Sanity checks (counts only)
        # ----------------------------
        if "entry_ts" in chunk.columns:
            e_dt = _to_datetime_from_ts(chunk["entry_ts"])
            acc.sanity_counts["rows_with_bad_entry_ts"] += int(e_dt.isna().sum())
        else:
            e_dt = None

        if "exit_ts" in chunk.columns:
            x_dt = _to_datetime_from_ts(chunk["exit_ts"])
            acc.sanity_counts["rows_with_bad_exit_ts"] += int(x_dt.isna().sum())
        else:
            x_dt = None

        if e_dt is not None and x_dt is not None:
            acc.sanity_counts["rows_with_entry_ge_exit"] += int((e_dt >= x_dt).sum())

        if "entry" in chunk.columns:
            entry_px = pd.to_numeric(chunk["entry"], errors="coerce")
            acc.sanity_counts["rows_with_nonpositive_entry_price"] += int((entry_px <= 0).sum())

        if "exit" in chunk.columns:
            exit_px = pd.to_numeric(chunk["exit"], errors="coerce")
            acc.sanity_counts["rows_with_nonpositive_exit_price"] += int((exit_px <= 0).sum())

        if "qty" in chunk.columns:
            qty = pd.to_numeric(chunk["qty"], errors="coerce")
            acc.sanity_counts["rows_with_nonpositive_qty"] += int((qty <= 0).sum())

        if "pnl" in chunk.columns:
            pnl = pd.to_numeric(chunk["pnl"], errors="coerce")
            acc.sanity_counts["rows_with_nan_pnl"] += int(pnl.isna().sum())

        if "pnl_R" in chunk.columns:
            pr = pd.to_numeric(chunk["pnl_R"], errors="coerce")
            acc.sanity_counts["rows_with_nan_pnl_R"] += int(pr.isna().sum())
        else:
            pr = None

        if "WIN" in chunk.columns:
            win_num = pd.to_numeric(chunk["WIN"], errors="coerce")
            acc.sanity_counts["rows_with_nan_WIN"] += int(win_num.isna().sum())
        else:
            win_num = None

        if "EXIT_FINAL" in chunk.columns:
            ef = pd.to_numeric(chunk["EXIT_FINAL"], errors="coerce")
            acc.sanity_counts["rows_with_bad_EXIT_FINAL"] += int((~ef.isin([0, 1, 2])).sum())
        else:
            ef = None

        # ----------------------------
        # Label sanity (WIN vs pnl_R, EXIT_FINAL distribution)
        # ----------------------------
        if win_num is not None:
            acc.label_counts["rows_with_WIN_1"] += int((win_num == 1).sum())
            acc.label_counts["rows_with_WIN_0"] += int((win_num == 0).sum())

        if pr is not None:
            acc.label_counts["rows_with_pnl_R_pos"] += int((pr > 0).sum())
            acc.label_counts["rows_with_pnl_R_nonpos"] += int((pr <= 0).sum())

        if ef is not None:
            acc.label_counts["rows_with_EXIT_FINAL_TP"] += int((ef == 1).sum())
            acc.label_counts["rows_with_EXIT_FINAL_SL"] += int((ef == 0).sum())
            acc.label_counts["rows_with_EXIT_FINAL_TIME"] += int((ef == 2).sum())

        if win_num is not None and pr is not None:
            valid = win_num.isin([0, 1]) & pr.notna()
            mismatch = ((win_num == 1) & (pr <= 0)) | ((win_num == 0) & (pr > 0))
            acc.label_counts["rows_with_WIN_mismatch_pnl_R"] += int((valid & mismatch).sum())

        # ----------------------------
        # Write flags file (trade_id + flags + entry_month) for later merge
        # ----------------------------
        if write_flags and ("trade_id" in chunk.columns):
            oi_cols = _safe_cols(all_cols, OI_COLS)
            fu_cols = _safe_cols(all_cols, FUNDING_COLS)
            btc_cols = _safe_cols(all_cols, BTC_CONTEXT_COLS)
            eth_cols = _safe_cols(all_cols, ETH_CONTEXT_COLS)

            has_oi = chunk[oi_cols].notna().any(axis=1) if oi_cols else pd.Series(False, index=chunk.index)
            has_funding = chunk[fu_cols].notna().any(axis=1) if fu_cols else pd.Series(False, index=chunk.index)
            has_btc_context = chunk[btc_cols].notna().any(axis=1) if btc_cols else pd.Series(False, index=chunk.index)
            has_eth_context = chunk[eth_cols].notna().any(axis=1) if eth_cols else pd.Series(False, index=chunk.index)

            out_flags = pd.DataFrame(
                {
                    "trade_id": pd.to_numeric(chunk["trade_id"], errors="coerce").astype("Int64"),
                    "entry_ts": chunk["entry_ts"] if "entry_ts" in chunk.columns else np.nan,
                    "entry_month": entry_month if entry_month is not None else "",
                    "has_oi": has_oi.astype("int8"),
                    "has_funding": has_funding.astype("int8"),
                    "has_btc_context": has_btc_context.astype("int8"),
                    "has_eth_context": has_eth_context.astype("int8"),
                }
            )
            # Drop rows where trade_id failed to parse
            out_flags = out_flags[out_flags["trade_id"].notna()].copy()
            out_flags["trade_id"] = out_flags["trade_id"].astype("int64")

            _gzip_append_csv(flags_out_path, out_flags, header=first_flags_chunk)
            first_flags_chunk = False

        if (i + 1) % 10 == 0:
            print(f"[00_load_qa] processed chunks={i+1} rows={acc.total_rows:,}", flush=True)

    # ----------------------------
    # Build outputs
    # ----------------------------
    total_rows = acc.total_rows
    if total_rows <= 0:
        raise RuntimeError("No rows processed. Check input file path and format.")

    # Overall coverage
    coverage = pd.DataFrame(
        {
            "column": acc.null_counts.index,
            "total_rows": total_rows,
            "null_count": acc.null_counts.values.astype("int64"),
        }
    )
    coverage["nonnull_count"] = coverage["total_rows"] - coverage["null_count"]
    coverage["null_rate"] = coverage["null_count"] / coverage["total_rows"]
    coverage = coverage.sort_values("null_rate", ascending=False)
    coverage_path = os.path.join(outdir, "feature_coverage.csv")
    coverage.to_csv(coverage_path, index=False)

    # Missingness by symbol (long)
    sym_totals = acc.sym_totals.rename("total_rows").astype("int64")
    sym_nulls = acc.sym_nulls.copy().astype("int64")
    sym_nulls.index.name = "symbol"

    sym_rates = sym_nulls.div(sym_totals, axis=0)
    sym_rates.index.name = "symbol"
    sym_rates_reset = _reset_index_named(sym_rates, "symbol")
    sym_long = sym_rates_reset.melt(id_vars=["symbol"], var_name="column", value_name="null_rate")
    sym_long["total_rows_symbol"] = sym_long["symbol"].map(sym_totals.to_dict()).fillna(0).astype("int64")
    sym_long = sym_long.sort_values(["column", "null_rate"], ascending=[True, False])
    sym_path = os.path.join(outdir, "missingness_by_symbol_long.csv")
    sym_long.to_csv(sym_path, index=False)

    # Missingness by month (long)
    month_path = os.path.join(outdir, "missingness_by_month_long.csv")
    if not acc.month_totals.empty and not acc.month_nulls.empty:
        month_totals = acc.month_totals.rename("total_rows").astype("int64")
        month_nulls = acc.month_nulls.copy().astype("int64")
        month_nulls.index.name = "entry_month"

        month_rates = month_nulls.div(month_totals, axis=0)
        month_rates.index.name = "entry_month"
        month_rates_reset = _reset_index_named(month_rates, "entry_month")
        month_long = month_rates_reset.melt(id_vars=["entry_month"], var_name="column", value_name="null_rate")
        month_long["total_rows_month"] = month_long["entry_month"].map(month_totals.to_dict()).fillna(0).astype("int64")
        month_long = month_long.sort_values(["entry_month", "column"], ascending=[True, True])
        month_long.to_csv(month_path, index=False)
    else:
        pd.DataFrame(columns=["entry_month", "column", "null_rate", "total_rows_month"]).to_csv(month_path, index=False)

    # Missingness by (symbol, month) for key cols (long)
    sm_path = os.path.join(outdir, "missingness_by_symbol_month_key.csv")
    if not acc.sym_month_totals.empty and not acc.sym_month_nulls.empty:
        sm_totals = acc.sym_month_totals.rename("total_rows").astype("int64")
        sm_nulls = acc.sym_month_nulls.copy().astype("int64")
        sm_nulls.index.names = ["symbol", "entry_month"]

        sm_rates = sm_nulls.div(sm_totals, axis=0)
        sm_rates.index.names = ["symbol", "entry_month"]
        sm_rates_reset = sm_rates.reset_index()

        sm_long = sm_rates_reset.melt(
            id_vars=["symbol", "entry_month"],
            var_name="column",
            value_name="null_rate",
        )

        # Merge totals (more reliable than mapping on tuple index)
        sm_totals_df = sm_totals.reset_index()
        sm_totals_df.columns = ["symbol", "entry_month", "total_rows_symbol_month"]
        sm_long = sm_long.merge(sm_totals_df, on=["symbol", "entry_month"], how="left")
        sm_long["total_rows_symbol_month"] = sm_long["total_rows_symbol_month"].fillna(0).astype("int64")
        sm_long = sm_long.sort_values(["column", "null_rate"], ascending=[True, False])

        sm_long.to_csv(sm_path, index=False)
    else:
        pd.DataFrame(columns=["symbol", "entry_month", "column", "null_rate", "total_rows_symbol_month"]).to_csv(sm_path, index=False)

    # Sanity checks summary
    sanity_df = pd.DataFrame(
        [{"check": k, "count": int(v), "rate": (int(v) / total_rows)} for k, v in acc.sanity_counts.items()]
    ).sort_values("rate", ascending=False)
    sanity_path = os.path.join(outdir, "sanity_checks.csv")
    sanity_df.to_csv(sanity_path, index=False)

    # Label sanity summary
    label_df = pd.DataFrame([{"metric": k, "count": int(v)} for k, v in acc.label_counts.items()])
    label_path = os.path.join(outdir, "label_sanity.csv")
    label_df.to_csv(label_path, index=False)

    # ----------------------------
    # QA HTML report
    # ----------------------------
    top_missing = coverage.head(30)[["column", "null_rate", "null_count", "nonnull_count"]].copy()
    key_cols_present = _safe_cols(all_cols, KEY_SYMBOL_MONTH_COLS)
    key_overall = coverage[coverage["column"].isin(key_cols_present)].sort_values("null_rate", ascending=False)

    # Worst symbols for key columns by mean null_rate
    worst_symbols = pd.DataFrame()
    if key_cols_present:
        cols = [c for c in key_cols_present if c in sym_rates.columns]
        if cols:
            worst_symbols = (
                sym_rates[cols]
                .mean(axis=1)
                .rename("mean_null_rate_key_cols")
                .to_frame()
                .join(sym_totals.rename("total_rows"))
                .sort_values(["mean_null_rate_key_cols", "total_rows"], ascending=[False, False])
                .head(50)
                .reset_index()
            )

    # Monthly mean missingness for key columns
    worst_months = pd.DataFrame()
    try:
        month_long_df = pd.read_csv(month_path)
        if key_cols_present and not month_long_df.empty:
            ml = month_long_df[month_long_df["column"].isin(key_cols_present)].copy()
            if not ml.empty:
                worst_months = (
                    ml.groupby("entry_month", as_index=False)["null_rate"].mean()
                    .rename(columns={"null_rate": "mean_null_rate_key_cols"})
                    .sort_values("entry_month")
                )
    except Exception:
        worst_months = pd.DataFrame()

    html_report = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>QA Report - trades.clean.csv</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1,h2,h3 {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; margin-top: 10px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 12px; }}
    th {{ background: #f4f4f4; }}
    code {{ background: #f7f7f7; padding: 2px 4px; }}
  </style>
</head>
<body>
<h1>QA Report: {html.escape(os.path.basename(infile))}</h1>

<h2>Summary</h2>
<ul>
  <li><strong>Total rows processed:</strong> {total_rows:,}</li>
  <li><strong>Unique symbols:</strong> {int(sym_totals.shape[0]) if not sym_totals.empty else 0:,}</li>
  <li><strong>Total columns:</strong> {len(all_cols)}</li>
  <li><strong>Flags file written:</strong> {"Yes" if (write_flags and os.path.exists(flags_out_path)) else "No"}</li>
</ul>

<h2>Top missing columns (overall)</h2>
{_html_table(top_missing, max_rows=30)}

<h2>Key context columns missingness (overall)</h2>
{_html_table(key_overall[["column","null_rate","null_count","nonnull_count"]].head(60), max_rows=60)}

<h2>Sanity checks</h2>
{_html_table(sanity_df, max_rows=50)}

<h2>Label sanity</h2>
{_html_table(label_df, max_rows=50)}

<h2>Worst symbols by mean missingness (key columns)</h2>
{_html_table(worst_symbols, max_rows=50)}

<h2>Monthly mean missingness (key columns)</h2>
{_html_table(worst_months, max_rows=120)}

<h2>Artifacts</h2>
<ul>
  <li><code>feature_coverage.csv</code></li>
  <li><code>missingness_by_symbol_long.csv</code></li>
  <li><code>missingness_by_month_long.csv</code></li>
  <li><code>missingness_by_symbol_month_key.csv</code></li>
  <li><code>sanity_checks.csv</code></li>
  <li><code>label_sanity.csv</code></li>
  <li><code>data_contract_columns.csv</code></li>
  <li><code>{html.escape(flags_out_name)}</code></li>
</ul>

</body>
</html>
"""
    report_path = os.path.join(outdir, "qa_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_report)

    print(f"[00_load_qa] DONE. Outputs in: {outdir}")
    print(f"[00_load_qa] - feature_coverage.csv")
    print(f"[00_load_qa] - missingness_by_symbol_long.csv")
    print(f"[00_load_qa] - missingness_by_month_long.csv")
    print(f"[00_load_qa] - missingness_by_symbol_month_key.csv")
    print(f"[00_load_qa] - sanity_checks.csv")
    print(f"[00_load_qa] - label_sanity.csv")
    print(f"[00_load_qa] - qa_report.html")
    if write_flags and os.path.exists(flags_out_path):
        print(f"[00_load_qa] - {os.path.basename(flags_out_path)}")


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 0 QA for trades.clean.csv (chunked).")
    p.add_argument("--infile", type=str, default="results/trades.clean.csv", help="Input CSV path.")
    p.add_argument("--outdir", type=str, default="research_outputs/00_qa", help="Output directory.")
    p.add_argument("--chunksize", type=int, default=250_000, help="CSV chunk size.")
    p.add_argument(
        "--no-flags",
        action="store_true",
        help="Disable writing flags_by_trade_id.csv.gz (missingness flags for merge).",
    )
    p.add_argument(
        "--flags-out",
        type=str,
        default="flags_by_trade_id.csv.gz",
        help="Filename for flags output (inside outdir).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    infile = args.infile
    outdir = args.outdir
    chunksize = int(args.chunksize)

    if chunksize < 10_000:
        raise ValueError("--chunksize too small; use at least 10,000 for reasonable performance.")
    if not os.path.exists(infile):
        raise FileNotFoundError(f"Input file not found: {infile}")

    process_csv(
        infile=infile,
        outdir=outdir,
        chunksize=chunksize,
        write_flags=(not args.no_flags),
        flags_out_name=args.flags_out,
    )


if __name__ == "__main__":
    main()
