#!/usr/bin/env python3
"""
research/04_models_cv.py

Step 4: multivariate models with Purged K-Fold CV + embargo.

Robustness fixes:
  - Drops duplicate column names in inputs and after joins.
  - Loads only y_* from targets.parquet, and only regime/slice fields from regimes.parquet.
  - If trades already contains regime columns, drops overlapping columns from trades before attaching regimes.
  - Converts categorical features to object dtype with np.nan (no pd.NA) to avoid sklearn SimpleImputer crash.
  - Uses Purged K-Fold CV to ensure 100% OOF coverage while maintaining strict embargoes.
  - Handles folds where y_train has a single class by using a constant probability baseline for that fold.
  - Fails loudly if any eligible rows remain unscored.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# -----------------------------
# Generic helpers
# -----------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def _safe_cols(all_cols: Sequence[str], candidates: Sequence[str]) -> List[str]:
    s = set(all_cols)
    return [c for c in candidates if c in s]


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _drop_duplicate_columns(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.columns.duplicated().any():
        dup = df.columns[df.columns.duplicated()].tolist()
        print(
            f"[04_models_cv] WARNING: dropping {len(dup)} duplicate column names in {label}: {sorted(set(dup))[:30]}",
            flush=True,
        )
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _as_object_with_nan(s: pd.Series) -> pd.Series:
    """
    Convert a series to object dtype and ensure missing values are np.nan, not pd.NA.
    """
    # First force to object so we don't keep pandas extension NA semantics
    obj = s.astype("object")
    # Replace pandas NA/NaT with np.nan
    obj = obj.where(pd.notna(obj), np.nan)
    return obj


# -----------------------------
# I/O: trades / targets / regimes
# -----------------------------
def _read_trades_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = _drop_duplicate_columns(df, "trades_csv")

    if "trade_id" not in df.columns:
        raise RuntimeError("Trades CSV must contain trade_id.")
    df["trade_id"] = pd.to_numeric(df["trade_id"], errors="coerce")
    df = df.dropna(subset=["trade_id"]).copy()
    df["trade_id"] = df["trade_id"].astype("int64")
    df = df.drop_duplicates(subset=["trade_id"]).set_index("trade_id", drop=True)

    if "entry_ts" not in df.columns:
        raise RuntimeError("Trades CSV must contain entry_ts.")
    df["entry_ts"] = _to_datetime_utc(df["entry_ts"])
    return df


def _read_targets_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = _drop_duplicate_columns(df, "targets_parquet")

    if "trade_id" not in df.columns:
        raise RuntimeError("Targets parquet must contain trade_id.")
    df["trade_id"] = pd.to_numeric(df["trade_id"], errors="coerce")
    df = df.dropna(subset=["trade_id"]).copy()
    df["trade_id"] = df["trade_id"].astype("int64")

    y_cols = [c for c in df.columns if c.startswith("y_")]
    keep = ["trade_id"] + y_cols
    df = df[keep].drop_duplicates(subset=["trade_id"]).set_index("trade_id", drop=True)

    for c in y_cols:
        df[c] = _to_num(df[c])

    return df


def _read_regimes_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = _drop_duplicate_columns(df, "regimes_parquet")

    if "trade_id" not in df.columns:
        raise RuntimeError("Regimes parquet must contain trade_id.")
    df["trade_id"] = pd.to_numeric(df["trade_id"], errors="coerce")
    df = df.dropna(subset=["trade_id"]).copy()
    df["trade_id"] = df["trade_id"].astype("int64")

    keep_cols = ["trade_id"]
    for c in df.columns:
        if c == "trade_id":
            continue
        if c == "risk_on" or c == "risk_on_1":
            keep_cols.append(c)
            continue
        if c.startswith("S") and "_" in c:
            keep_cols.append(c)
            continue
        if c.startswith("regime_"):
            keep_cols.append(c)
            continue
        if c.endswith("_regime_code"):
            keep_cols.append(c)
            continue
        if c in ("markov_state_4h", "regime_up", "trend_regime_1d", "vol_regime_1d", "regime_code_1d"):
            keep_cols.append(c)
            continue

    keep_cols = list(dict.fromkeys(keep_cols))
    df = df[keep_cols].drop_duplicates(subset=["trade_id"]).set_index("trade_id", drop=True)

    # normalize obvious numeric regimes
    if "risk_on" in df.columns:
        df["risk_on"] = _to_num(df["risk_on"])
    if "risk_on_1" in df.columns:
        df["risk_on_1"] = _to_num(df["risk_on_1"])
    for c in df.columns:
        if c.startswith("S") and "_" in c:
            # keep as object/np.nan-safe rather than pandas string
            df[c] = _as_object_with_nan(df[c])

    return df


# -----------------------------
# Feature selection
# -----------------------------
DEFAULT_EXCLUDE = {
    # identifiers / execution
    "symbol",
    "entry_ts",
    "exit_ts",
    "entry",
    "exit",
    "qty",
    "fees",
    "side",
    "sl",
    "tp",
    "exit_reason",
    # outcomes/diagnostics (not predictors)
    "pnl",
    "pnl_R",
    "WIN",
    "EXIT_FINAL",
    "mae_over_atr",
    "mfe_over_atr",
}


def autodetect_features(df: pd.DataFrame, max_cat_card: int = 60) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    cat_cols: List[str] = []

    for c in df.columns:
        if c in DEFAULT_EXCLUDE:
            continue
        if c.startswith("y_"):
            continue

        s = df[c]

        # regime/slice fields: categorical
        if c == "risk_on" or c == "risk_on_1" or (c.startswith("S") and "_" in c) or c.endswith("_regime_code") or c.startswith("regime_") or c in ("markov_state_4h", "regime_up", "regime_code_1d"):
            cat_cols.append(c)
            continue

        if pd.api.types.is_bool_dtype(s):
            cat_cols.append(c)
            continue

        if pd.api.types.is_numeric_dtype(s):
            numeric_cols.append(c)
            continue

        coerced = pd.to_numeric(s, errors="coerce")
        frac_num = float(coerced.notna().mean()) if len(coerced) else 0.0
        if frac_num >= 0.95:
            numeric_cols.append(c)
            continue

        nun = s.dropna().astype(str).nunique()
        if 2 <= nun <= max_cat_card:
            cat_cols.append(c)

    numeric_cols = list(dict.fromkeys(numeric_cols))
    cat_cols = list(dict.fromkeys(cat_cols))
    cat_cols = [c for c in cat_cols if c not in set(numeric_cols)]
    return numeric_cols, cat_cols


def filter_feature_columns(
    df: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    max_missing: float,
    min_unique_numeric: int,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    keep_num = []
    for c in numeric_cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        miss = float(s.isna().mean())
        nun = int(s.nunique(dropna=True))
        if miss <= max_missing and nun >= min_unique_numeric:
            keep_num.append(c)

    keep_cat = []
    for c in cat_cols:
        if c not in df.columns:
            continue
        s = df[c]
        miss = float(pd.isna(s).mean())
        nun = int(pd.Series(s).dropna().astype(str).nunique())
        if miss <= max_missing and nun >= 2:
            keep_cat.append(c)

    out = df.copy()
    for c in keep_num:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    for c in keep_cat:
        out[c] = _as_object_with_nan(out[c])

    return keep_num, keep_cat, out


def load_top_features_from_univariate(path: str, top_k: int) -> List[str]:
    df = pd.read_csv(path)
    if df.empty or "feature" not in df.columns:
        return []
    score_col = "rank_abs_delta_pnl" if "rank_abs_delta_pnl" in df.columns else None
    if score_col is None:
        return []
    agg = df.groupby("feature", as_index=False)[score_col].max().sort_values(score_col, ascending=False)
    feats = agg["feature"].astype(str).tolist()
    return feats[:top_k] if top_k and top_k > 0 else feats


# -----------------------------
# CV splitting (Purged K-Fold)
# -----------------------------
@dataclass
class Fold:
    fold: int
    train_ids: np.ndarray
    test_ids: np.ndarray
    t_test_start: pd.Timestamp
    t_test_end: pd.Timestamp


def make_purged_kfold(
    df: pd.DataFrame,
    n_splits: int,
    embargo_days: float,
) -> List[Fold]:
    """
    Purged K-Fold CV to ensure 100% OOF coverage while preventing leakage.
    
    Logic:
      1. Sort data by time.
      2. Split into K contiguous blocks (standard KFold on sorted index).
      3. For each block k (Test):
         - Train = All indices NOT in block k AND NOT in embargo zone.
         - Embargo zone: any train sample whose time is within `embargo_days` 
           AFTER the test block ends (if train is future) or BEFORE test block starts (if train is past).
           Actually, standard purging usually just removes the overlap. 
           Here we implement a strict embargo:
             - If Train is BEFORE Test: Train.end < Test.start - embargo (optional, usually 0 is fine if no overlap)
             - If Train is AFTER Test: Train.start > Test.end + embargo
    """
    if "entry_ts" not in df.columns:
        raise RuntimeError("entry_ts missing.")
    
    # Ensure sorted by time
    df_sorted = df.sort_values("entry_ts").copy()
    times = pd.to_datetime(df_sorted["entry_ts"], utc=True)
    indices = df_sorted.index.to_numpy()
    
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    embargo = pd.Timedelta(days=float(embargo_days))
    folds: List[Fold] = []

    for fold_idx, (train_iloc, test_iloc) in enumerate(kf.split(indices)):
        test_ids = indices[test_iloc]
        
        # Determine test time bounds
        t_test_start = times.iloc[test_iloc].min()
        t_test_end = times.iloc[test_iloc].max()
        
        # Filter train_iloc to enforce embargo
        # We need to map iloc back to times
        train_times = times.iloc[train_iloc]
        
        # Keep train samples that are:
        # 1. Before test start (minus optional embargo, usually 0 is fine for past data, but let's be safe)
        # 2. After test end + embargo
        
        # Note: Standard "Purged K-Fold" often implies removing the "embargo" period *after* the test set 
        # from the *subsequent* training set. 
        # Here we apply it symmetrically or just strictly based on time.
        
        # Strict non-overlap condition:
        # Train must not be in [t_test_start, t_test_end] (guaranteed by KFold)
        # AND if Train > Test, Train must be > t_test_end + embargo
        
        mask_before = train_times < t_test_start
        mask_after = train_times > (t_test_end + embargo)
        
        keep_mask = mask_before | mask_after
        final_train_iloc = train_iloc[keep_mask]
        
        train_ids = indices[final_train_iloc]
        
        if len(train_ids) == 0:
            print(f"[04_models_cv] WARNING: Fold {fold_idx} has empty training set after purging/embargo!", flush=True)
            continue
            
        folds.append(
            Fold(
                fold=fold_idx,
                train_ids=train_ids,
                test_ids=test_ids,
                t_test_start=t_test_start,
                t_test_end=t_test_end,
            )
        )

    return folds


# -----------------------------
# Modeling / Metrics
# -----------------------------
def time_decay_weights(entry_ts: pd.Series, halflife_days: Optional[float]) -> np.ndarray:
    if halflife_days is None or halflife_days <= 0:
        return np.ones(len(entry_ts), dtype=np.float64)
    ts = pd.to_datetime(entry_ts, utc=True, errors="coerce")
    tmax = ts.max()
    age_days = (tmax - ts).dt.total_seconds() / 86400.0
    w = np.power(0.5, age_days / float(halflife_days))
    w = w.fillna(0.0).to_numpy(dtype=np.float64)
    return w


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def safe_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        eps = 1e-12
        p = np.clip(y_prob, eps, 1 - eps)
        return float(log_loss(y_true, p))
    except Exception:
        return float("nan")


def safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return float("nan")


def topk_stats(
    pnl_R: np.ndarray,
    y_win: Optional[np.ndarray],
    y_time: Optional[np.ndarray],
    y_prob: np.ndarray,
    frac: float,
) -> Dict[str, float]:
    n = len(y_prob)
    k = max(1, int(np.floor(frac * n)))
    order = np.argsort(-y_prob)[:k]
    out: Dict[str, float] = {}
    out[f"top{int(frac*100)}_n"] = float(k)
    out[f"top{int(frac*100)}_mean_pnl_R"] = float(np.nanmean(pnl_R[order])) if k > 0 else float("nan")
    if y_win is not None:
        out[f"top{int(frac*100)}_winrate"] = float(np.nanmean(y_win[order])) if k > 0 else float("nan")
    if y_time is not None:
        out[f"top{int(frac*100)}_timerate"] = float(np.nanmean(y_time[order])) if k > 0 else float("nan")
    return out


def build_preprocess(numeric_cols: List[str], cat_cols: List[str], scale_numeric_for_linear: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric_for_linear:
        num_steps.append(("scale", StandardScaler(with_mean=False)))
    num_pipe = Pipeline(steps=num_steps)

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),  # expects np.nan, not pd.NA
            ("ohe", _make_ohe()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def build_models(seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {}

    models["logreg"] = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=4000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=seed,
    )

    if HAS_LGBM:
        models["lgbm"] = LGBMClassifier(
            objective="binary",
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=80,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )

    return models


def eval_scope_masks(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {"ALL": np.ones(len(df), dtype=bool)}
    # Check for risk_on_1 first, then risk_on
    if "risk_on_1" in df.columns:
        r = pd.to_numeric(df["risk_on_1"], errors="coerce").fillna(0).to_numpy()
        masks["RISK_ON_1"] = (r == 1)
    elif "risk_on" in df.columns:
        r = pd.to_numeric(df["risk_on"], errors="coerce").fillna(0).to_numpy()
        masks["RISK_ON_1"] = (r == 1)
    return masks


def compute_metrics_block(
    *,
    df_test: pd.DataFrame,
    y_col: str,
    y_prob: np.ndarray,
    pnl_col: str = "pnl_R",
    y_win_col: str = "y_win",
    y_time_col: str = "y_time",
    threshold: float = 0.5,
    top_frac: float = 0.1,
) -> Dict[str, float]:
    y_true = pd.to_numeric(df_test[y_col], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(y_true)
    y_true = y_true[ok].astype(int)
    p = y_prob[ok].astype(np.float64)

    pnl = pd.to_numeric(df_test.loc[df_test.index[ok], pnl_col], errors="coerce").to_numpy(dtype=np.float64)

    y_win = None
    if y_win_col in df_test.columns:
        y_win = pd.to_numeric(df_test.loc[df_test.index[ok], y_win_col], errors="coerce").to_numpy(dtype=np.float64)

    y_time = None
    if y_time_col in df_test.columns:
        y_time = pd.to_numeric(df_test.loc[df_test.index[ok], y_time_col], errors="coerce").to_numpy(dtype=np.float64)

    pred = (p >= threshold).astype(int)

    out: Dict[str, float] = {}
    out["n"] = float(len(y_true))
    out["pos_rate"] = float(np.mean(y_true)) if len(y_true) else float("nan")
    out["auc"] = safe_auc(y_true, p)
    out["avg_precision"] = safe_ap(y_true, p)
    out["logloss"] = safe_logloss(y_true, p)
    out["brier"] = safe_brier(y_true, p)
    out["accuracy@0.5"] = float(accuracy_score(y_true, pred)) if len(y_true) else float("nan")
    out["precision@0.5"] = float(precision_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan")
    out["recall@0.5"] = float(recall_score(y_true, pred, zero_division=0)) if len(y_true) else float("nan")
    out["mean_pnl_R"] = float(np.nanmean(pnl)) if len(pnl) else float("nan")
    out.update(topk_stats(pnl, y_win, y_time, p, frac=top_frac))
    return out


# -----------------------------
# Main workflow
# -----------------------------
def process(
    trades_csv: str,
    targets_parquet: str,
    regimes_parquet: str,
    outdir: str,
    target: str,
    n_splits: int,
    embargo_days: float,
    min_eval_n: int,
    seed: int,
    max_missing: float,
    min_unique_numeric: int,
    include_regimes_as_features: bool,
    train_scope: str,
    top_features_csv: Optional[str],
    top_k_features: int,
    fit_final: bool,
    time_decay_halflife_days: Optional[float],
    allow_missing_oof: bool,
) -> None:
    _ensure_dir(outdir)

    trades = _read_trades_csv(trades_csv)
    targets = _read_targets_parquet(targets_parquet)
    regimes = _read_regimes_parquet(regimes_parquet)

    # If trades already contains regime columns, drop them and re-attach from regimes.parquet (authoritative).
    overlap = [c for c in regimes.columns if c in trades.columns]
    if overlap:
        print(f"[04_models_cv] INFO: trades/regimes overlap -> dropping from trades: {sorted(overlap)}", flush=True)
        trades = trades.drop(columns=overlap)

    df = trades.join(targets, how="left").join(regimes, how="left")
    df = _drop_duplicate_columns(df, "joined_frame")

    if target not in df.columns:
        raise RuntimeError(
            f"Target '{target}' not found after join. "
            f"Available y_*: {[c for c in df.columns if c.startswith('y_')]}"
        )
    if "pnl_R" not in df.columns:
        raise RuntimeError("Trades CSV must contain pnl_R for evaluation.")

    df = df[df["entry_ts"].notna()].copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df[df[target].notna()].copy()

    # Optional scope restriction for training dataset
    if train_scope.upper() == "RISK_ON_1":
        risk_col = "risk_on_1" if "risk_on_1" in df.columns else ("risk_on" if "risk_on" in df.columns else None)
        if risk_col is None:
            raise RuntimeError("train_scope=RISK_ON_1 requested, but risk_on_1/risk_on is missing.")
        r = pd.to_numeric(df[risk_col], errors="coerce").fillna(0).to_numpy()
        df = df.loc[r == 1].copy()

    numeric_cols, cat_cols = autodetect_features(df)

    if include_regimes_as_features:
        for c in df.columns:
            if c == "risk_on" or c == "risk_on_1" or (c.startswith("S") and "_" in c) or c.endswith("_regime_code") or c.startswith("regime_") or c in ("markov_state_4h", "regime_up", "regime_code_1d"):
                if c not in cat_cols and c not in DEFAULT_EXCLUDE and not c.startswith("y_"):
                    cat_cols.append(c)
        cat_cols = list(dict.fromkeys(cat_cols))
        numeric_cols = [c for c in numeric_cols if c not in set(cat_cols)]

    if top_features_csv:
        top_feats = load_top_features_from_univariate(top_features_csv, top_k_features)
        if top_feats:
            keep = set(top_feats)
            numeric_cols = [c for c in numeric_cols if c in keep]

    numeric_cols, cat_cols, df = filter_feature_columns(
        df=df,
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        max_missing=max_missing,
        min_unique_numeric=min_unique_numeric,
    )

    feat_cols = list(dict.fromkeys(numeric_cols + cat_cols))
    if not feat_cols:
        raise RuntimeError("No usable features after filtering.")

    # Build X with sklearn-safe missing values:
    # - numerics: float with np.nan
    # - categoricals: object with np.nan (NO pd.NA)
    X = df[feat_cols].copy()
    X = _drop_duplicate_columns(X, "X_features")

    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    for c in cat_cols:
        if c in X.columns:
            X[c] = _as_object_with_nan(X[c])

    # last safety sweep for any remaining pd.NA
    X = X.where(pd.notna(X), np.nan)

    if not X.columns.is_unique:
        raise RuntimeError("X_features has non-unique columns after dedup; inspect input files.")

    y = pd.to_numeric(df[target], errors="coerce").astype(int)

    # Use Purged K-Fold to ensure 100% OOF coverage
    folds = make_purged_kfold(df, n_splits=n_splits, embargo_days=embargo_days)
    
    pd.DataFrame(
        [{
            "fold": f.fold,
            "train_n": int(len(f.train_ids)),
            "test_n": int(len(f.test_ids)),
            "t_test_start": str(f.t_test_start),
            "t_test_end": str(f.t_test_end),
        } for f in folds]
    ).to_csv(os.path.join(outdir, "cv_folds.csv"), index=False)

    models = build_models(seed=seed)
    if not models:
        raise RuntimeError("No models available.")

    preprocess_scaled = build_preprocess(numeric_cols, cat_cols, scale_numeric_for_linear=True)
    preprocess_unscaled = build_preprocess(numeric_cols, cat_cols, scale_numeric_for_linear=False)

    # OOF container - pre-filled with NaN
    oof = pd.DataFrame(index=df.index)
    oof["entry_ts"] = df["entry_ts"]
    oof["pnl_R"] = pd.to_numeric(df["pnl_R"], errors="coerce")
    oof[target] = pd.to_numeric(df[target], errors="coerce")
    if "y_win" in df.columns:
        oof["y_win"] = pd.to_numeric(df["y_win"], errors="coerce")
    if "y_time" in df.columns:
        oof["y_time"] = pd.to_numeric(df["y_time"], errors="coerce")
    if "risk_on" in df.columns:
        oof["risk_on"] = pd.to_numeric(df["risk_on"], errors="coerce")
    if "risk_on_1" in df.columns:
        oof["risk_on_1"] = pd.to_numeric(df["risk_on_1"], errors="coerce")
    for c in df.columns:
        if c.startswith("S") and "_" in c:
            oof[c] = _as_object_with_nan(df[c])

    metrics_rows: List[Dict[str, object]] = []
    metrics_by_slice_rows: List[Dict[str, object]] = []

    for model_name, model in models.items():
        pre = preprocess_scaled if model_name == "logreg" else preprocess_unscaled
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])

        pred_col = f"p_{model_name}"
        oof[pred_col] = np.nan

        for f in folds:
            tr_mask = df.index.isin(f.train_ids)
            te_mask = df.index.isin(f.test_ids)

            X_train = X.loc[tr_mask]
            y_train = y.loc[tr_mask]
            X_test = X.loc[te_mask]
            
            if len(X_test) == 0:
                continue

            # If a fold's training labels have only one class, the classifier fit will fail.
            # Use a constant probability baseline for that fold.
            uniq = np.unique(y_train.to_numpy())
            if len(uniq) < 2:
                p_const = float(np.mean(y_train)) if len(y_train) else 0.5
                p = np.full(len(X_test), p_const, dtype=np.float64)
                oof.loc[X_test.index, pred_col] = p

                df_test = df.loc[X_test.index].copy()
                base = {
                    "model": model_name,
                    "fold": int(f.fold),
                    "target": target,
                    "train_n": int(len(X_train)),
                    "test_n": int(len(X_test)),
                    "t_test_start": str(f.t_test_start),
                    "t_test_end": str(f.t_test_end),
                    "note": "single_class_train_baseline",
                }
                m_all = compute_metrics_block(df_test=df_test, y_col=target, y_prob=p, top_frac=0.1)
                metrics_rows.append({**base, **{"scope": "ALL"}, **m_all})
                continue

            w = time_decay_weights(df.loc[tr_mask, "entry_ts"], time_decay_halflife_days)

            try:
                pipe.fit(X_train, y_train, model__sample_weight=w)
            except TypeError:
                pipe.fit(X_train, y_train)

            if hasattr(pipe, "predict_proba"):
                p = pipe.predict_proba(X_test)[:, 1]
            else:
                z = pipe.decision_function(X_test)
                p = 1.0 / (1.0 + np.exp(-z))

            oof.loc[X_test.index, pred_col] = p

            df_test = df.loc[X_test.index].copy()
            base = {
                "model": model_name,
                "fold": int(f.fold),
                "target": target,
                "train_n": int(len(X_train)),
                "test_n": int(len(X_test)),
                "t_test_start": str(f.t_test_start),
                "t_test_end": str(f.t_test_end),
            }

            m_all = compute_metrics_block(df_test=df_test, y_col=target, y_prob=p, top_frac=0.1)
            metrics_rows.append({**base, **{"scope": "ALL"}, **m_all})

            scopes = eval_scope_masks(df_test)
            for scope_name, mask in scopes.items():
                if scope_name == "ALL":
                    continue
                if int(mask.sum()) < int(min_eval_n):
                    continue
                m_sc = compute_metrics_block(df_test=df_test.loc[mask], y_col=target, y_prob=p[mask], top_frac=0.1)
                metrics_by_slice_rows.append({**base, **{"slice": "SCOPE", "state": scope_name}, **m_sc})

            slice_cols = [c for c in df_test.columns if (c.startswith("S") and "_" in c)]
            for sc in slice_cols:
                st = _as_object_with_nan(df_test[sc])
                for state_val in pd.Series(st).dropna().unique().tolist():
                    sel = (st == state_val)
                    if int(sel.sum()) < int(min_eval_n):
                        continue
                    m_sub = compute_metrics_block(df_test=df_test.loc[sel], y_col=target, y_prob=p[sel.to_numpy()], top_frac=0.1)
                    metrics_by_slice_rows.append({**base, **{"slice": sc, "state": str(state_val)}, **m_sub})

        # Check OOF coverage
        missing_mask = oof[pred_col].isna()
        n_missing = missing_mask.sum()
        if n_missing > 0:
            msg = f"Model {model_name} has {n_missing} unscored OOF rows ({n_missing/len(oof):.1%})."
            print(f"[04_models_cv] ERROR: {msg}", flush=True)
            
            # Dump missing rows for debug
            missing_rows = oof.loc[missing_mask].reset_index()
            missing_out = os.path.join(outdir, "missing_oof_rows.csv")
            missing_rows.to_csv(missing_out, index=False)
            
            if not allow_missing_oof:
                raise RuntimeError(f"{msg} See {missing_out}. Use --allow-missing-oof to bypass.")

        if fit_final:
            w_full = time_decay_weights(df["entry_ts"], time_decay_halflife_days)
            try:
                pipe.fit(X, y, model__sample_weight=w_full)
            except TypeError:
                pipe.fit(X, y)
            joblib.dump(pipe, os.path.join(outdir, f"final_model_{model_name}.joblib"))

    pd.DataFrame(metrics_rows).to_csv(os.path.join(outdir, "cv_metrics.csv"), index=False)
    pd.DataFrame(metrics_by_slice_rows).to_csv(os.path.join(outdir, "cv_metrics_by_slice.csv"), index=False)

    oof_out = oof.reset_index().rename(columns={"index": "trade_id"})
    oof_out.to_parquet(os.path.join(outdir, "oof_predictions.parquet"), index=False)
    oof_out.to_csv(os.path.join(outdir, "oof_predictions.csv.gz"), index=False, compression="gzip")

    manifest = {
        "trades_csv": trades_csv,
        "targets_parquet": targets_parquet,
        "regimes_parquet": regimes_parquet,
        "target": target,
        "train_scope": train_scope,
        "n_rows_used": int(len(df)),
        "n_splits_requested": int(n_splits),
        "n_folds_built": int(len(folds)),
        "embargo_days": float(embargo_days),
        "min_eval_n": int(min_eval_n),
        "max_missing": float(max_missing),
        "min_unique_numeric": int(min_unique_numeric),
        "include_regimes_as_features": bool(include_regimes_as_features),
        "top_features_csv": top_features_csv,
        "top_k_features": int(top_k_features),
        "time_decay_halflife_days": (None if time_decay_halflife_days is None else float(time_decay_halflife_days)),
        "features": {"numeric_cols": numeric_cols, "cat_cols": cat_cols},
        "models": list(models.keys()),
        "outputs": {
            "cv_folds": "cv_folds.csv",
            "cv_metrics": "cv_metrics.csv",
            "cv_metrics_by_slice": "cv_metrics_by_slice.csv",
            "oof_predictions_parquet": "oof_predictions.parquet",
            "oof_predictions_csv": "oof_predictions.csv.gz",
            "final_models": [f"final_model_{m}.joblib" for m in models.keys()] if fit_final else [],
        },
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"[04_models_cv] DONE. Outputs in: {outdir}", flush=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4: multivariate models with Purged K-Fold CV + embargo.")
    p.add_argument("--trades", type=str, default="results/trades.clean.csv")
    p.add_argument("--targets", type=str, default="research_outputs/01_targets/targets.parquet")
    p.add_argument("--regimes", type=str, default="research_outputs/02_regimes/regimes.parquet")
    p.add_argument("--outdir", type=str, default="research_outputs/04_models_cv")

    p.add_argument("--target", type=str, default="y_good_05")

    p.add_argument("--n-splits", type=int, default=6)
    p.add_argument("--embargo-days", type=float, default=1.0)
    p.add_argument("--min-eval-n", type=int, default=200)

    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-missing", type=float, default=0.95)
    p.add_argument("--min-unique-numeric", type=int, default=5)

    p.add_argument("--include-regimes-as-features", action="store_true")
    p.add_argument("--train-scope", type=str, default="ALL", choices=["ALL", "RISK_ON_1"])

    p.add_argument("--top-features-csv", type=str, default="")
    p.add_argument("--top-k-features", type=int, default=0)

    p.add_argument("--no-fit-final", action="store_true")
    p.add_argument("--time-decay-halflife-days", type=float, default=0.0)
    
    p.add_argument("--allow-missing-oof", action="store_true", help="Do not fail if some OOF rows are unscored.")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    for path, name in [(args.trades, "Trades CSV"), (args.targets, "Targets parquet"), (args.regimes, "Regimes parquet")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    top_csv = args.top_features_csv.strip() or None
    if top_csv and not os.path.exists(top_csv):
        raise FileNotFoundError(f"--top-features-csv not found: {top_csv}")

    time_decay = None
    if float(args.time_decay_halflife_days) and float(args.time_decay_halflife_days) > 0:
        time_decay = float(args.time_decay_halflife_days)

    process(
        trades_csv=args.trades,
        targets_parquet=args.targets,
        regimes_parquet=args.regimes,
        outdir=args.outdir,
        target=args.target,
        n_splits=int(args.n_splits),
        embargo_days=float(args.embargo_days),
        min_eval_n=int(args.min_eval_n),
        seed=int(args.seed),
        max_missing=float(args.max_missing),
        min_unique_numeric=int(args.min_unique_numeric),
        include_regimes_as_features=bool(args.include_regimes_as_features),
        train_scope=args.train_scope,
        top_features_csv=top_csv,
        top_k_features=int(args.top_k_features),
        fit_final=(not args.no_fit_final),
        time_decay_halflife_days=time_decay,
        allow_missing_oof=bool(args.allow_missing_oof),
    )


if __name__ == "__main__":
    main()