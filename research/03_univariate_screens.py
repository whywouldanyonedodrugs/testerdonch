#!/usr/bin/env python3
"""
research/03_univariate_screens.py

Univariate screens inside regimes.

Key change vs prior version:
  - Robust cutpoint estimation with diagnostics.
  - If cutpoints == 0, writes cutpoints_diagnostics.csv and raises RuntimeError
    instead of producing empty downstream artifacts.

Run example:
  python research/03_univariate_screens.py \
    --trades results/trades.clean.csv \
    --targets research_outputs/01_targets/targets.parquet \
    --regimes research_outputs/02_regimes/regimes.parquet \
    --outdir research_outputs/03_univariate \
    --chunksize 250000 \
    --sample-per-feature 10000 \
    --sample-rows-per-chunk 20000 \
    --min-samples-for-cutpoints 2000 \
    --min-n-per-bin 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError(f"pyarrow is required. Import error: {e}")


DEFAULT_SLICES = [
    "ALL",
    "risk_on",
    "S1_regime_code_1d",
    "S5_btcRisk_x_regimeUp",
    "S3_funding_x_oi",
]

DEFAULT_EXCLUDE = {
    "trade_id",
    "symbol",
    "entry_ts",
    "exit_ts",
    "entry",
    "exit",
    "qty",
    "fees",
    "pnl",
    "pnl_R",
    "mae_over_atr",
    "mfe_over_atr",
    "WIN",
    "EXIT_FINAL",
    "exit_reason",
    "sl",
    "tp",
    "side",
}

EXCLUDE_PREFIXES = ("exit",)

AUX_COLS_TO_EXCLUDE_IF_PRESENT = {
    # targets
    "y_win", "y_time", "y_tp", "y_sl", "y_exit_class", "y_good_05", "y_good_10",
    # regimes + sets
    "regime_code_1d", "trend_regime_code_1d", "vol_regime_code_1d", "markov_state_4h", "regime_up",
    "funding_regime_code", "oi_regime_code", "crowd_side",
    "btc_trend_up", "btc_vol_high", "btc_risk_regime_code", "freshness_code", "compression_code", "volimpulse_code",
    "risk_on",
    "S1_regime_code_1d", "S2_markov_x_vol1d", "S3_funding_x_oi", "S4_crowd_x_trend1d", "S5_btcRisk_x_regimeUp", "S6_fresh_x_compress",
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_cols(all_cols: Sequence[str], candidates: Sequence[str]) -> List[str]:
    s = set(all_cols)
    return [c for c in candidates if c in s]


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _infer_ts_unit(ts_max: float) -> str:
    if not np.isfinite(ts_max):
        return "unknown"
    return "ms" if ts_max > 1e12 else "s"


def _to_datetime_utc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        try:
            if getattr(series.dt, "tz", None) is None:
                return series.dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")
            return series
        except Exception:
            return pd.to_datetime(series, errors="coerce", utc=True)

    if pd.api.types.is_numeric_dtype(series):
        s = _to_num(series)
        ts_max = s.max()
        unit = _infer_ts_unit(float(ts_max) if ts_max is not None else np.nan)
        return pd.to_datetime(s, unit=unit if unit in ("s", "ms") else None, errors="coerce", utc=True)

    return pd.to_datetime(series, errors="coerce", utc=True)


def _month_key(dt_utc: pd.Series) -> pd.Series:
    return dt_utc.dt.strftime("%Y-%m")


def _is_excluded(col: str, exclude_set: set[str]) -> bool:
    if col in exclude_set:
        return True
    lc = col.lower()
    for p in EXCLUDE_PREFIXES:
        if lc.startswith(p):
            return True
    return False


def _sanitize_filename(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return s[:180] if len(s) > 180 else s


def _spearman_from_bins(bin_means: pd.Series) -> Optional[float]:
    if bin_means is None or bin_means.empty:
        return None
    x = pd.to_numeric(bin_means.index, errors="coerce").to_numpy()
    y = pd.to_numeric(bin_means.values, errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    try:
        return float(pd.Series(y[mask]).corr(pd.Series(x[mask]), method="spearman"))
    except Exception:
        return None


def _bin_edges_from_thresholds(thr: List[float]) -> List[Tuple[float, float]]:
    if len(thr) != 9:
        raise ValueError("Expected 9 thresholds for deciles.")
    edges: List[Tuple[float, float]] = []
    edges.append((float("-inf"), float(thr[0])))
    for i in range(1, 9):
        edges.append((float(thr[i - 1]), float(thr[i])))
    edges.append((float(thr[8]), float("inf")))
    return edges


@dataclass
class BinAgg:
    n_total: int = 0
    sum_pnl_R: float = 0.0
    cnt_pnl_R: int = 0
    sum_pnl_R_not_time: float = 0.0
    cnt_pnl_R_not_time: int = 0
    sum_y_win: float = 0.0
    cnt_y_win: int = 0
    sum_y_time: float = 0.0
    cnt_y_time: int = 0
    sum_y_tp: float = 0.0
    cnt_y_tp: int = 0
    sum_y_sl: float = 0.0
    cnt_y_sl: int = 0
    sum_y_good05: float = 0.0
    cnt_y_good05: int = 0
    sum_y_good10: float = 0.0
    cnt_y_good10: int = 0

    def update_from_row(self, r: pd.Series) -> None:
        self.n_total += int(r.get("n_total", 0))

        self.sum_pnl_R += float(r.get("sum_pnl_R", 0.0))
        self.cnt_pnl_R += int(r.get("cnt_pnl_R", 0))

        self.sum_pnl_R_not_time += float(r.get("sum_pnl_R_not_time", 0.0))
        self.cnt_pnl_R_not_time += int(r.get("cnt_pnl_R_not_time", 0))

        self.sum_y_win += float(r.get("sum_y_win", 0.0))
        self.cnt_y_win += int(r.get("cnt_y_win", 0))

        self.sum_y_time += float(r.get("sum_y_time", 0.0))
        self.cnt_y_time += int(r.get("cnt_y_time", 0))

        self.sum_y_tp += float(r.get("sum_y_tp", 0.0))
        self.cnt_y_tp += int(r.get("cnt_y_tp", 0))

        self.sum_y_sl += float(r.get("sum_y_sl", 0.0))
        self.cnt_y_sl += int(r.get("cnt_y_sl", 0))

        self.sum_y_good05 += float(r.get("sum_y_good05", 0.0))
        self.cnt_y_good05 += int(r.get("cnt_y_good05", 0))

        self.sum_y_good10 += float(r.get("sum_y_good10", 0.0))
        self.cnt_y_good10 += int(r.get("cnt_y_good10", 0))



@dataclass
class MonthAgg:
    sum_pnl_R: float = 0.0
    cnt_pnl_R: int = 0

    def update(self, sum_p: float, cnt_p: int) -> None:
        self.sum_pnl_R += float(sum_p)
        self.cnt_pnl_R += int(cnt_p)


def select_features(
    trades_csv: str,
    explicit_features: Optional[List[str]],
    features_file: Optional[str],
    max_features: Optional[int],
) -> Tuple[List[str], List[str], List[str]]:
    header = pd.read_csv(trades_csv, nrows=0)
    all_cols = list(header.columns)

    if features_file:
        with open(features_file, "r", encoding="utf-8") as f:
            explicit = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
        explicit_features = (explicit_features or []) + explicit
        explicit_features = list(dict.fromkeys(explicit_features))

    if explicit_features:
        feats = [c for c in explicit_features if c in all_cols]
        sample = pd.read_csv(trades_csv, nrows=20000, low_memory=False, usecols=_safe_cols(all_cols, feats))
        numeric, categorical = [], []
        for c in feats:
            if c not in sample.columns:
                continue
            if pd.api.types.is_numeric_dtype(sample[c]):
                numeric.append(c)
            else:
                categorical.append(c)
        return numeric, categorical, all_cols

    sample = pd.read_csv(trades_csv, nrows=20000, low_memory=False)
    exclude = set(DEFAULT_EXCLUDE) | set(AUX_COLS_TO_EXCLUDE_IF_PRESENT)

    numeric, categorical = [], []
    for c in all_cols:
        if _is_excluded(c, exclude):
            continue
        if c in AUX_COLS_TO_EXCLUDE_IF_PRESENT:
            continue
        if c not in sample.columns:
            continue
        s = sample[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
        else:
            nun = s.dropna().astype(str).nunique()
            if 2 <= nun <= 50:
                categorical.append(c)

    if max_features is not None and max_features > 0:
        numeric = numeric[:max_features]

    return numeric, categorical, all_cols


def estimate_decile_cutpoints(
    trades_csv: str,
    numeric_features: List[str],
    chunksize: int,
    sample_per_feature: int,
    sample_rows_per_chunk: int,
    min_samples_for_cutpoints: int,
    seed: int,
    outdir: str,
) -> Dict[str, Dict[str, object]]:
    """
    Returns dict:
      cutpoints[feature] = {"thresholds": [q10..q90], "n_sample": int}

    Writes:
      - cutpoints_diagnostics.csv
    """
    rng = np.random.default_rng(seed)

    samples: Dict[str, List[np.ndarray]] = {f: [] for f in numeric_features}
    sample_counts: Dict[str, int] = {f: 0 for f in numeric_features}

    header = pd.read_csv(trades_csv, nrows=0)
    all_cols = list(header.columns)
    usecols = _safe_cols(all_cols, numeric_features)
    missing = sorted(set(numeric_features) - set(usecols))
    if missing:
        print(f"[03_univariate] WARNING: {len(missing)} numeric_features missing from CSV header (skipped).", flush=True)

    reader = pd.read_csv(trades_csv, chunksize=chunksize, low_memory=False, usecols=usecols)

    for ci, chunk in enumerate(reader):
        n = len(chunk)
        if n == 0:
            continue

        k = min(sample_rows_per_chunk, n)
        idx = rng.choice(n, size=k, replace=False)

        for f in usecols:
            if sample_counts.get(f, 0) >= sample_per_feature:
                continue

            v = _to_num(chunk[f].iloc[idx])
            v = v[np.isfinite(v)]
            if v.empty:
                continue

            remaining = sample_per_feature - sample_counts[f]
            arr = v.to_numpy(dtype=np.float64, copy=False)
            if arr.size > remaining:
                # randomize before truncation so we don't bias toward earlier rows
                if arr.size > 1:
                    rng.shuffle(arr)
                arr = arr[:remaining]

            samples[f].append(arr)
            sample_counts[f] += int(arr.size)

        if (ci + 1) % 20 == 0:
            done = sum(int(sample_counts[f] >= min_samples_for_cutpoints) for f in usecols)
            print(f"[03_univariate pass1] chunks={ci+1} features_with_enough_samples={done}/{len(usecols)}", flush=True)

        if all(sample_counts[f] >= sample_per_feature for f in usecols):
            break

    # Build cutpoints + diagnostics
    diag_rows = []
    cutpoints: Dict[str, Dict[str, object]] = {}

    for f in usecols:
        n_samp = int(sample_counts.get(f, 0))
        arr = np.concatenate(samples[f], axis=0) if samples[f] else np.array([], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        nunique = int(pd.Series(arr).nunique()) if arr.size else 0
        vmin = float(np.min(arr)) if arr.size else np.nan
        vmax = float(np.max(arr)) if arr.size else np.nan

        ok = (arr.size >= int(min_samples_for_cutpoints)) and (nunique >= 20) and np.isfinite(vmin) and np.isfinite(vmax) and (vmax > vmin)
        reason = ""
        if not ok:
            if arr.size < int(min_samples_for_cutpoints):
                reason = f"too_few_samples(<{min_samples_for_cutpoints})"
            elif nunique < 20:
                reason = "too_few_unique_values(<20)"
            elif not (np.isfinite(vmin) and np.isfinite(vmax)):
                reason = "nonfinite_minmax"
            elif not (vmax > vmin):
                reason = "constant_or_inverted"
            else:
                reason = "unknown_skip"

            diag_rows.append(
                {
                    "feature": f,
                    "n_sample": int(arr.size),
                    "n_unique": nunique,
                    "min": vmin,
                    "max": vmax,
                    "status": "SKIP",
                    "reason": reason,
                }
            )
            continue

        qs = [i / 10.0 for i in range(1, 10)]
        thr = [float(np.quantile(arr, q)) for q in qs]

        # If fully constant, skip
        if np.allclose(thr, thr[0], rtol=0, atol=0):
            diag_rows.append(
                {
                    "feature": f,
                    "n_sample": int(arr.size),
                    "n_unique": nunique,
                    "min": vmin,
                    "max": vmax,
                    "status": "SKIP",
                    "reason": "all_thresholds_equal",
                }
            )
            continue

        # Allow duplicates (sparse/tied features), but record how many distinct thresholds
        n_thr_unique = int(pd.Series(thr).nunique())
        diag_rows.append(
            {
                "feature": f,
                "n_sample": int(arr.size),
                "n_unique": nunique,
                "min": vmin,
                "max": vmax,
                "status": "OK",
                "reason": "" if n_thr_unique == 9 else f"duplicate_thresholds(n_unique_thr={n_thr_unique})",
            }
        )
        cutpoints[f] = {"thresholds": thr, "n_sample": int(arr.size)}

    diag_df = pd.DataFrame(diag_rows).sort_values(["status", "n_sample"], ascending=[True, False])
    diag_path = os.path.join(outdir, "cutpoints_diagnostics.csv")
    diag_df.to_csv(diag_path, index=False)

    return cutpoints


def _prepare_aux_tables(targets_path: str, regimes_path: str, slices: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load targets/regimes, index by trade_id.

    Important: drop entry_ts/exit_ts columns from aux tables to avoid overlap with trades CSV.
    """
    slice_cols_needed: List[str] = [s for s in slices if s != "ALL"]

    # --- targets ---
    tdf = pd.read_parquet(targets_path)
    if "trade_id" not in tdf.columns:
        raise RuntimeError("targets parquet must contain trade_id.")

    # keep only target columns + trade_id
    keep_t = ["trade_id", "y_win", "y_time", "y_tp", "y_sl", "y_good_05", "y_good_10"]
    keep_t = [c for c in keep_t if c in tdf.columns]
    tdf = tdf[keep_t].copy()

    tdf["trade_id"] = pd.to_numeric(tdf["trade_id"], errors="coerce").astype("int64")
    tdf = tdf.dropna(subset=["trade_id"]).set_index("trade_id", drop=True)

    for c in ["y_win", "y_time", "y_tp", "y_sl", "y_good_05", "y_good_10"]:
        if c not in tdf.columns:
            tdf[c] = np.nan
        tdf[c] = _to_num(tdf[c])

    # --- regimes ---
    rdf = pd.read_parquet(regimes_path)
    if "trade_id" not in rdf.columns:
        raise RuntimeError("regimes parquet must contain trade_id.")

    # Always include risk_on if present (used by stability scope RISK_ON_1)
    rkeep = ["trade_id"]
    for s in slice_cols_needed:
        if s in rdf.columns and s not in rkeep:
            rkeep.append(s)
    if "risk_on" in rdf.columns and "risk_on" not in rkeep:
        rkeep.append("risk_on")

    rdf = rdf[rkeep].copy()
    rdf["trade_id"] = pd.to_numeric(rdf["trade_id"], errors="coerce").astype("int64")
    rdf = rdf.dropna(subset=["trade_id"]).set_index("trade_id", drop=True)

    for s in slice_cols_needed:
        if s not in rdf.columns:
            rdf[s] = np.nan

    if "risk_on" not in rdf.columns:
        rdf["risk_on"] = np.nan
    else:
        rdf["risk_on"] = _to_num(rdf["risk_on"])

    return tdf, rdf, slice_cols_needed



def accumulate_univariate(
    trades_csv: str,
    numeric_features: List[str],
    categorical_features: List[str],
    cutpoints: Dict[str, Dict[str, object]],
    targets_df: pd.DataFrame,
    regimes_df: pd.DataFrame,
    slices: List[str],
    chunksize: int,
    min_n_per_bin: int,
    stability: bool,
) -> Tuple[
    Dict[Tuple[str, str, object, int], BinAgg],
    Dict[Tuple[str, str, str, int], MonthAgg],
    Dict[str, Dict[str, int]],
]:
    bin_aggs: Dict[Tuple[str, str, object, int], BinAgg] = {}
    month_aggs: Dict[Tuple[str, str, str, int], MonthAgg] = {}
    cat_counts: Dict[str, Dict[str, int]] = {f: {} for f in categorical_features}

    need_cols = ["trade_id", "pnl_R"]
    if stability:
        need_cols.append("entry_ts")
    need_cols += numeric_features
    need_cols += categorical_features

    header = pd.read_csv(trades_csv, nrows=0)
    all_cols = list(header.columns)
    for req in ["trade_id", "pnl_R"]:
        if req not in all_cols:
            raise RuntimeError(f"trades CSV missing required column: {req}")

    usecols = _safe_cols(all_cols, need_cols)
    reader = pd.read_csv(trades_csv, chunksize=chunksize, low_memory=False, usecols=usecols)

    for ci, chunk in enumerate(reader):
        if len(chunk) == 0:
            continue

        tid = pd.to_numeric(chunk["trade_id"], errors="coerce")
        chunk = chunk.loc[tid.notna()].copy()
        if len(chunk) == 0:
            continue
        chunk["trade_id"] = pd.to_numeric(chunk["trade_id"], errors="coerce").astype("int64")
        chunk = chunk.set_index("trade_id", drop=True)

        chunk = (
            chunk.join(targets_df, how="left", rsuffix="__t")
                 .join(regimes_df, how="left", rsuffix="__r")
        )

        chunk["pnl_R"] = _to_num(chunk["pnl_R"])

        for c in ["y_win", "y_time", "y_tp", "y_sl", "y_good_05", "y_good_10"]:
            if c not in chunk.columns:
                chunk[c] = np.nan
            chunk[c] = _to_num(chunk[c])

        if stability and ("entry_ts" in chunk.columns):
            dt = _to_datetime_utc(chunk["entry_ts"])
            chunk["_entry_month"] = _month_key(dt)
        elif stability:
            chunk["_entry_month"] = np.nan

        for f in categorical_features:
            if f not in chunk.columns:
                continue
            s = chunk[f].astype(str)
            vc = s.value_counts(dropna=True).head(200)
            d = cat_counts[f]
            for k, v in vc.items():
                d[k] = d.get(k, 0) + int(v)

        pnl_R_not_time = chunk["pnl_R"].where(chunk["y_time"] == 0)

        for f in numeric_features:
            if f not in cutpoints:
                continue
            if f not in chunk.columns:
                continue

            thr = cutpoints[f].get("thresholds", None)
            if not isinstance(thr, list) or len(thr) != 9:
                continue

            v = _to_num(chunk[f]).to_numpy(dtype=np.float64, copy=False)
            b = np.full(v.shape[0], -1, dtype=np.int16)

            mask = np.isfinite(v)
            if mask.any():
                b[mask] = np.digitize(v[mask], thr, right=True).astype(np.int16)

            df = pd.DataFrame(
                {
                    "_bin": b,
                    "pnl_R": chunk["pnl_R"].to_numpy(dtype=np.float64, copy=False),
                    "pnl_R_not_time": pnl_R_not_time.to_numpy(dtype=np.float64, copy=False),
                    "y_win": chunk["y_win"].to_numpy(dtype=np.float64, copy=False),
                    "y_time": chunk["y_time"].to_numpy(dtype=np.float64, copy=False),
                    "y_tp": chunk["y_tp"].to_numpy(dtype=np.float64, copy=False),
                    "y_sl": chunk["y_sl"].to_numpy(dtype=np.float64, copy=False),
                    "y_good_05": chunk["y_good_05"].to_numpy(dtype=np.float64, copy=False),
                    "y_good_10": chunk["y_good_10"].to_numpy(dtype=np.float64, copy=False),
                },
                index=chunk.index,
            )

            df = df[df["_bin"] >= 0]
            if df.empty:
                continue

            for slice_col in slices:
                if slice_col == "ALL":
                    state = pd.Series(0, index=df.index, dtype="int16")
                    slice_name = "ALL"
                else:
                    if slice_col not in chunk.columns:
                        continue
                    state = pd.to_numeric(chunk.loc[df.index, slice_col], errors="coerce")
                    slice_name = slice_col

                gdf = df.copy()
                gdf["_state"] = state.values

                agg = gdf.groupby(["_state", "_bin"], dropna=False).agg(
                    n_total=("pnl_R", "size"),
                    sum_pnl_R=("pnl_R", "sum"),
                    cnt_pnl_R=("pnl_R", "count"),
                    sum_pnl_R_not_time=("pnl_R_not_time", "sum"),
                    cnt_pnl_R_not_time=("pnl_R_not_time", "count"),
                    sum_y_win=("y_win", "sum"),
                    cnt_y_win=("y_win", "count"),
                    sum_y_time=("y_time", "sum"),
                    cnt_y_time=("y_time", "count"),
                    sum_y_tp=("y_tp", "sum"),
                    cnt_y_tp=("y_tp", "count"),
                    sum_y_sl=("y_sl", "sum"),
                    cnt_y_sl=("y_sl", "count"),
                    sum_y_good05=("y_good_05", "sum"),
                    cnt_y_good05=("y_good_05", "count"),
                    sum_y_good10=("y_good_10", "sum"),
                    cnt_y_good10=("y_good_10", "count"),
                )

                for (st, bn), row in agg.iterrows():
                    if pd.isna(st):
                        state_key = None
                    else:
                        try:
                            state_key = int(st)
                        except Exception:
                            state_key = str(st)

                    key = (f, slice_name, state_key, int(bn))
                    cur = bin_aggs.get(key)
                    if cur is None:
                        cur = BinAgg()
                        bin_aggs[key] = cur
                    cur.update_from_row(row)

            if stability and ("_entry_month" in chunk.columns):
                months = chunk.loc[df.index, "_entry_month"]
                df_m = df.copy()
                df_m["_entry_month"] = months.values
                df_m = df_m[df_m["_bin"].isin([0, 9])]
                df_m = df_m[df_m["_entry_month"].notna()]
                if not df_m.empty:
                    m_agg = df_m.groupby(["_entry_month", "_bin"], dropna=False).agg(
                        sum_pnl_R=("pnl_R", "sum"),
                        cnt_pnl_R=("pnl_R", "count"),
                    )
                    for (mth, bn), row in m_agg.iterrows():
                        keym = (f, "ALL", str(mth), int(bn))
                        curm = month_aggs.get(keym)
                        if curm is None:
                            curm = MonthAgg()
                            month_aggs[keym] = curm
                        curm.update(float(row["sum_pnl_R"]), int(row["cnt_pnl_R"]))

                    if "risk_on" in chunk.columns:
                        risk_on = pd.to_numeric(chunk.loc[df_m.index, "risk_on"], errors="coerce")
                        df_r = df_m[risk_on == 1]
                        if not df_r.empty:
                            r_agg = df_r.groupby(["_entry_month", "_bin"], dropna=False).agg(
                                sum_pnl_R=("pnl_R", "sum"),
                                cnt_pnl_R=("pnl_R", "count"),
                            )
                            for (mth, bn), row in r_agg.iterrows():
                                keym = (f, "RISK_ON_1", str(mth), int(bn))
                                curm = month_aggs.get(keym)
                                if curm is None:
                                    curm = MonthAgg()
                                    month_aggs[keym] = curm
                                curm.update(float(row["sum_pnl_R"]), int(row["cnt_pnl_R"]))

        if (ci + 1) % 10 == 0:
            print(f"[03_univariate pass2] chunks={ci+1} bin_aggs_keys={len(bin_aggs):,}", flush=True)

    return bin_aggs, month_aggs, cat_counts


def build_bin_summary_df(
    bin_aggs: Dict[Tuple[str, str, object, int], BinAgg],
    cutpoints: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    rows = []
    for (feature, slice_col, state, bn), agg in bin_aggs.items():
        thr = cutpoints.get(feature, {}).get("thresholds", None)
        bin_low = bin_high = np.nan
        if isinstance(thr, list) and len(thr) == 9 and 0 <= bn <= 9:
            edges = _bin_edges_from_thresholds([float(x) for x in thr])
            bin_low, bin_high = edges[bn]

        mean_pnl_R = (agg.sum_pnl_R / agg.cnt_pnl_R) if agg.cnt_pnl_R > 0 else np.nan
        mean_pnl_R_not_time = (agg.sum_pnl_R_not_time / agg.cnt_pnl_R_not_time) if agg.cnt_pnl_R_not_time > 0 else np.nan

        win_rate = (agg.sum_y_win / agg.cnt_y_win) if agg.cnt_y_win > 0 else np.nan
        time_rate = (agg.sum_y_time / agg.cnt_y_time) if agg.cnt_y_time > 0 else np.nan
        tp_rate = (agg.sum_y_tp / agg.cnt_y_tp) if agg.cnt_y_tp > 0 else np.nan
        sl_rate = (agg.sum_y_sl / agg.cnt_y_sl) if agg.cnt_y_sl > 0 else np.nan
        good05_rate = (agg.sum_y_good05 / agg.cnt_y_good05) if agg.cnt_y_good05 > 0 else np.nan
        good10_rate = (agg.sum_y_good10 / agg.cnt_y_good10) if agg.cnt_y_good10 > 0 else np.nan

        # Force state to STRING to avoid mixed-type Parquet issues
        if state is None or (isinstance(state, float) and not np.isfinite(state)):
            state_str = "<NA>"
        else:
            state_str = str(state)

        rows.append(
            {
                "feature": str(feature),
                "slice": str(slice_col),
                "state": state_str,
                "bin": int(bn),
                "bin_low": float(bin_low) if np.isfinite(bin_low) else bin_low,
                "bin_high": float(bin_high) if np.isfinite(bin_high) else bin_high,

                "n_total": int(agg.n_total),
                "cnt_pnl_R": int(agg.cnt_pnl_R),
                "cnt_pnl_R_not_time": int(agg.cnt_pnl_R_not_time),

                "mean_pnl_R": float(mean_pnl_R) if np.isfinite(mean_pnl_R) else mean_pnl_R,
                "mean_pnl_R_not_time": float(mean_pnl_R_not_time) if np.isfinite(mean_pnl_R_not_time) else mean_pnl_R_not_time,

                "win_rate": float(win_rate) if np.isfinite(win_rate) else win_rate,
                "time_rate": float(time_rate) if np.isfinite(time_rate) else time_rate,
                "tp_rate": float(tp_rate) if np.isfinite(tp_rate) else tp_rate,
                "sl_rate": float(sl_rate) if np.isfinite(sl_rate) else sl_rate,
                "good05_rate": float(good05_rate) if np.isfinite(good05_rate) else good05_rate,
                "good10_rate": float(good10_rate) if np.isfinite(good10_rate) else good10_rate,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ensure stable dtypes for parquet
    df["feature"] = df["feature"].astype("string")
    df["slice"] = df["slice"].astype("string")
    df["state"] = df["state"].astype("string")

    return df.sort_values(["feature", "slice", "state", "bin"])



def build_rankings(bin_df: pd.DataFrame, min_n_per_bin: int) -> pd.DataFrame:
    """
    Build effect-size summary per (feature, slice, state).

    Robust behavior:
      - If no groups pass the bin0/bin9 + min_n filter, returns an empty DF with expected columns
        (no KeyError).
    """
    cols_needed = {"feature", "slice", "state", "bin", "cnt_pnl_R", "mean_pnl_R", "time_rate", "mean_pnl_R_not_time"}
    if bin_df is None or bin_df.empty:
        return pd.DataFrame(columns=[
            "feature", "slice", "state",
            "n_bin0", "n_bin9",
            "mean_pnl_R_bin0", "mean_pnl_R_bin9", "delta_mean_pnl_R",
            "time_rate_bin0", "time_rate_bin9", "delta_time_rate",
            "mean_pnl_R_not_time_bin0", "mean_pnl_R_not_time_bin9", "delta_mean_pnl_R_not_time",
            "spearman_pnl_R", "spearman_time_rate",
            "rank_abs_delta_pnl", "rank_pnl_minus_time",
        ])

    missing = sorted(list(cols_needed - set(bin_df.columns)))
    if missing:
        raise RuntimeError(f"build_rankings: bin_df missing required columns: {missing}")

    rows = []

    gby = bin_df.groupby(["feature", "slice", "state"], sort=False)
    for (feature, slice_col, state), g in gby:
        g0 = g[g["bin"] == 0]
        g9 = g[g["bin"] == 9]
        if g0.empty or g9.empty:
            continue

        n0 = int(pd.to_numeric(g0["cnt_pnl_R"].iloc[0], errors="coerce") or 0)
        n9 = int(pd.to_numeric(g9["cnt_pnl_R"].iloc[0], errors="coerce") or 0)
        if (n0 < min_n_per_bin) or (n9 < min_n_per_bin):
            continue

        m0 = pd.to_numeric(g0["mean_pnl_R"].iloc[0], errors="coerce")
        m9 = pd.to_numeric(g9["mean_pnl_R"].iloc[0], errors="coerce")
        t0 = pd.to_numeric(g0["time_rate"].iloc[0], errors="coerce")
        t9 = pd.to_numeric(g9["time_rate"].iloc[0], errors="coerce")

        m0_nt = pd.to_numeric(g0["mean_pnl_R_not_time"].iloc[0], errors="coerce")
        m9_nt = pd.to_numeric(g9["mean_pnl_R_not_time"].iloc[0], errors="coerce")

        delta_pnl = (m9 - m0) if (np.isfinite(m0) and np.isfinite(m9)) else np.nan
        delta_time = (t9 - t0) if (np.isfinite(t0) and np.isfinite(t9)) else np.nan
        delta_pnl_nt = (m9_nt - m0_nt) if (np.isfinite(m0_nt) and np.isfinite(m9_nt)) else np.nan

        # Spearman monotonicity over bins with enough count
        g_ok = g[pd.to_numeric(g["cnt_pnl_R"], errors="coerce") >= min_n_per_bin].copy()
        spearman_pnl = None
        spearman_time = None
        if not g_ok.empty:
            g_ok = g_ok.sort_values("bin")
            spearman_pnl = _spearman_from_bins(g_ok.set_index("bin")["mean_pnl_R"])
            spearman_time = _spearman_from_bins(g_ok.set_index("bin")["time_rate"])

        rows.append(
            {
                "feature": feature,
                "slice": slice_col,
                "state": state,
                "n_bin0": n0,
                "n_bin9": n9,
                "mean_pnl_R_bin0": float(m0) if np.isfinite(m0) else np.nan,
                "mean_pnl_R_bin9": float(m9) if np.isfinite(m9) else np.nan,
                "delta_mean_pnl_R": float(delta_pnl) if np.isfinite(delta_pnl) else np.nan,
                "time_rate_bin0": float(t0) if np.isfinite(t0) else np.nan,
                "time_rate_bin9": float(t9) if np.isfinite(t9) else np.nan,
                "delta_time_rate": float(delta_time) if np.isfinite(delta_time) else np.nan,
                "mean_pnl_R_not_time_bin0": float(m0_nt) if np.isfinite(m0_nt) else np.nan,
                "mean_pnl_R_not_time_bin9": float(m9_nt) if np.isfinite(m9_nt) else np.nan,
                "delta_mean_pnl_R_not_time": float(delta_pnl_nt) if np.isfinite(delta_pnl_nt) else np.nan,
                "spearman_pnl_R": spearman_pnl,
                "spearman_time_rate": spearman_time,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        # Return empty but well-formed DF; upstream writes empty CSV and continues.
        return pd.DataFrame(columns=[
            "feature", "slice", "state",
            "n_bin0", "n_bin9",
            "mean_pnl_R_bin0", "mean_pnl_R_bin9", "delta_mean_pnl_R",
            "time_rate_bin0", "time_rate_bin9", "delta_time_rate",
            "mean_pnl_R_not_time_bin0", "mean_pnl_R_not_time_bin9", "delta_mean_pnl_R_not_time",
            "spearman_pnl_R", "spearman_time_rate",
            "rank_abs_delta_pnl", "rank_pnl_minus_time",
        ])

    out["rank_abs_delta_pnl"] = out["delta_mean_pnl_R"].abs()
    out["rank_pnl_minus_time"] = out["delta_mean_pnl_R"] - 0.5 * out["delta_time_rate"].fillna(0.0)
    return out.sort_values(["rank_abs_delta_pnl"], ascending=False)



def build_stability_tables(month_aggs: Dict[Tuple[str, str, str, int], MonthAgg]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (feature, scope, month, bn), agg in month_aggs.items():
        mean = (agg.sum_pnl_R / agg.cnt_pnl_R) if agg.cnt_pnl_R > 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "scope": scope,
                "month": month,
                "bin": int(bn),
                "cnt_pnl_R": int(agg.cnt_pnl_R),
                "mean_pnl_R": float(mean) if np.isfinite(mean) else mean,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df

    piv = df.pivot_table(index=["feature", "scope", "month"], columns="bin", values="mean_pnl_R", aggfunc="first").reset_index()
    piv = piv.rename(columns={0: "mean_bin0", 9: "mean_bin9"})
    piv["delta_mean_pnl_R"] = piv["mean_bin9"] - piv["mean_bin0"]

    summ = piv.groupby(["feature", "scope"], as_index=False).agg(
        n_months=("delta_mean_pnl_R", "count"),
        mean_delta=("delta_mean_pnl_R", "mean"),
        std_delta=("delta_mean_pnl_R", "std"),
        frac_pos=("delta_mean_pnl_R", lambda x: float((x > 0).mean()) if len(x) else np.nan),
    ).sort_values(["scope", "mean_delta"], ascending=[True, False])

    return piv.sort_values(["scope", "feature", "month"]), summ


def write_feature_cards(
    outdir: str,
    cutpoints: Dict[str, Dict[str, object]],
    rankings_df: pd.DataFrame,
    top_k_slices: int = 12,
) -> None:
    cards_dir = os.path.join(outdir, "feature_cards")
    _ensure_dir(cards_dir)
    if rankings_df.empty:
        return

    for feature, g in rankings_df.groupby("feature", sort=False):
        thr = cutpoints.get(feature, {}).get("thresholds", None)
        thr_txt = ""
        if isinstance(thr, list) and len(thr) == 9:
            thr_txt = ", ".join([f"{x:.6g}" for x in thr])

        g2 = g.sort_values("rank_abs_delta_pnl", ascending=False).head(top_k_slices)

        lines = [f"# Feature: `{feature}`"]
        if thr_txt:
            lines += ["", "## Decile thresholds (q10..q90)", "", f"`{thr_txt}`"]
        lines += ["", "## Top slice/state effects (bin9 - bin0)", ""]
        lines += ["| slice | state | n_bin0 | n_bin9 | delta_mean_pnl_R | delta_time_rate | spearman_pnl_R | spearman_time |"]
        lines += ["|---|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in g2.iterrows():
            lines.append(
                f"| {r['slice']} | {r['state']} | "
                f"{int(r['n_bin0']) if pd.notna(r.get('n_bin0', np.nan)) else ''} | "
                f"{int(r['n_bin9']) if pd.notna(r.get('n_bin9', np.nan)) else ''} | "
                f"{r['delta_mean_pnl_R']:.6g} | "
                f"{r['delta_time_rate']:.6g} | "
                f"{(r['spearman_pnl_R'] if pd.notna(r.get('spearman_pnl_R', np.nan)) else '')} | "
                f"{(r['spearman_time_rate'] if pd.notna(r.get('spearman_time_rate', np.nan)) else '')} |"
            )

        path = os.path.join(cards_dir, f"{_sanitize_filename(feature)}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def process(
    trades_csv: str,
    targets_path: str,
    regimes_path: str,
    outdir: str,
    chunksize: int,
    slices: List[str],
    sample_per_feature: int,
    sample_rows_per_chunk: int,
    min_samples_for_cutpoints: int,
    seed: int,
    min_n_per_bin: int,
    max_features: Optional[int],
    features: Optional[List[str]],
    features_file: Optional[str],
    stability: bool,
) -> None:
    _ensure_dir(outdir)

    numeric_features, categorical_features, all_cols = select_features(
        trades_csv=trades_csv,
        explicit_features=features,
        features_file=features_file,
        max_features=max_features,
    )
    if "pnl_R" not in all_cols:
        raise RuntimeError("trades CSV must contain pnl_R for univariate screens.")

    targets_df, regimes_df, _ = _prepare_aux_tables(targets_path, regimes_path, slices)

    print(f"[03_univariate] numeric_features={len(numeric_features)} categorical_features={len(categorical_features)}", flush=True)

    cutpoints = estimate_decile_cutpoints(
        trades_csv=trades_csv,
        numeric_features=numeric_features,
        chunksize=chunksize,
        sample_per_feature=sample_per_feature,
        sample_rows_per_chunk=sample_rows_per_chunk,
        min_samples_for_cutpoints=min_samples_for_cutpoints,
        seed=seed,
        outdir=outdir,
    )
    kept_numeric = sorted(cutpoints.keys())

    print(f"[03_univariate] cutpoints computed for {len(kept_numeric)}/{len(numeric_features)} numeric features", flush=True)

    if len(kept_numeric) == 0:
        diag_path = os.path.join(outdir, "cutpoints_diagnostics.csv")
        raise RuntimeError(
            "No cutpoints were computed for any numeric features. "
            "This means features are too sparse / non-numeric / constant under current thresholds. "
            f"Inspect: {diag_path}. "
            "Then rerun with smaller --min-samples-for-cutpoints and/or reduce feature list."
        )

    bin_aggs, month_aggs, _ = accumulate_univariate(
        trades_csv=trades_csv,
        numeric_features=kept_numeric,
        categorical_features=categorical_features,
        cutpoints=cutpoints,
        targets_df=targets_df,
        regimes_df=regimes_df,
        slices=slices,
        chunksize=chunksize,
        min_n_per_bin=min_n_per_bin,
        stability=stability,
    )

    bin_df = build_bin_summary_df(bin_aggs, cutpoints)
    bins_parquet_path = os.path.join(outdir, "univariate_bins.parquet")
    bins_csv_path = os.path.join(outdir, "univariate_bins.csv.gz")
    rankings_path = os.path.join(outdir, "ranked_features_by_regime.csv")
    stab_month_path = os.path.join(outdir, "stability_by_month.csv")
    stab_summary_path = os.path.join(outdir, "stability_summary.csv")

    if not bin_df.empty:
        bin_df.to_parquet(bins_parquet_path, index=False)
        bin_df.to_csv(bins_csv_path, index=False, compression="gzip")
    else:
        # still write empty files explicitly
        pd.DataFrame().to_csv(bins_csv_path, index=False, compression="gzip")

    rankings_df = build_rankings(bin_df, min_n_per_bin=min_n_per_bin)
    rankings_df.to_csv(rankings_path, index=False)

    stab_month_df, stab_summary_df = build_stability_tables(month_aggs)
    stab_month_df.to_csv(stab_month_path, index=False)
    stab_summary_df.to_csv(stab_summary_path, index=False)

    write_feature_cards(outdir, cutpoints, rankings_df)

    report = {
        "trades_csv": trades_csv,
        "targets_path": targets_path,
        "regimes_path": regimes_path,
        "slices": slices,
        "chunksize": chunksize,
        "sample_per_feature": sample_per_feature,
        "sample_rows_per_chunk": sample_rows_per_chunk,
        "min_samples_for_cutpoints": min_samples_for_cutpoints,
        "seed": seed,
        "min_n_per_bin": min_n_per_bin,
        "numeric_features_requested": len(numeric_features),
        "numeric_features_with_cutpoints": len(kept_numeric),
        "categorical_features_detected": len(categorical_features),
        "outputs": {
            "univariate_bins_parquet": "univariate_bins.parquet",
            "univariate_bins_csv": "univariate_bins.csv.gz",
            "ranked_features_by_regime": "ranked_features_by_regime.csv",
            "stability_by_month": "stability_by_month.csv",
            "stability_summary": "stability_summary.csv",
            "feature_cards_dir": "feature_cards/",
            "cutpoints_diagnostics": "cutpoints_diagnostics.csv",
        },
    }
    with open(os.path.join(outdir, "univariate_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"[03_univariate] DONE. Outputs in: {outdir}", flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3: univariate screens by regime slices (robust cutpoints).")
    p.add_argument("--trades", type=str, default="results/trades.clean.csv", help="Trades CSV.")
    p.add_argument("--targets", type=str, default="research_outputs/01_targets/targets.parquet", help="Targets parquet.")
    p.add_argument("--regimes", type=str, default="research_outputs/02_regimes/regimes.parquet", help="Regimes parquet.")
    p.add_argument("--outdir", type=str, default="research_outputs/03_univariate", help="Output directory.")
    p.add_argument("--chunksize", type=int, default=250_000, help="CSV chunk size.")
    p.add_argument("--slices", type=str, default=",".join(DEFAULT_SLICES), help="Comma-separated slice columns. Include ALL.")
    p.add_argument("--sample-per-feature", type=int, default=10_000, help="Max sample kept per feature.")
    p.add_argument("--sample-rows-per-chunk", type=int, default=20_000, help="Random rows sampled per chunk for cutpoints.")
    p.add_argument("--min-samples-for-cutpoints", type=int, default=2_000, help="Min finite samples to compute deciles.")
    p.add_argument("--seed", type=int, default=1337, help="RNG seed.")
    p.add_argument("--min-n-per-bin", type=int, default=200, help="Min cnt_pnl_R in bin0 and bin9 for ranking rows.")
    p.add_argument("--max-features", type=int, default=0, help="If >0, cap autodetected numeric features to this count.")
    p.add_argument("--features", type=str, default="", help="Optional comma-separated explicit feature list.")
    p.add_argument("--features-file", type=str, default="", help="Optional file listing features (one per line).")
    p.add_argument("--no-stability", action="store_true", help="Disable monthly stability computations.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if int(args.chunksize) < 10_000:
        raise ValueError("--chunksize too small; use at least 10,000.")
    for path, name in [(args.trades, "Trades CSV"), (args.targets, "Targets parquet"), (args.regimes, "Regimes parquet")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    slices = [s.strip() for s in args.slices.split(",") if s.strip()]
    max_features = int(args.max_features) if int(args.max_features) > 0 else None

    features = None
    if args.features.strip():
        features = [s.strip() for s in args.features.split(",") if s.strip()]
    features_file = args.features_file.strip() or None

    process(
        trades_csv=args.trades,
        targets_path=args.targets,
        regimes_path=args.regimes,
        outdir=args.outdir,
        chunksize=int(args.chunksize),
        slices=slices,
        sample_per_feature=int(args.sample_per_feature),
        sample_rows_per_chunk=int(args.sample_rows_per_chunk),
        min_samples_for_cutpoints=int(args.min_samples_for_cutpoints),
        seed=int(args.seed),
        min_n_per_bin=int(args.min_n_per_bin),
        max_features=max_features,
        features=features,
        features_file=features_file,
        stability=(not args.no_stability),
    )


if __name__ == "__main__":
    main()
