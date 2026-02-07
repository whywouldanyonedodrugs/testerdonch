#!/usr/bin/env python3
"""
export_golden_features.py

Deterministic "golden features" exporter for parity testing between:
- offline/backtest feature pipeline (training/backtest store), and
- live feature-building + meta scoring.

Guarantees:
- Output includes: timestamp (UTC tz-aware), symbol, and all 73 raw feature columns
  exactly as in bundle feature_manifest.json (features.numeric_cols + features.cat_cols).
- If the trade store is missing some *derivable* regime-set columns
  (funding_regime_code, oi_regime_code, btc_risk_regime_code, risk_on, S1..S6),
  they are derived using regimes_report.json thresholds and the same fallback logic
  as backtester._meta_build_regime_sets / _meta_trend_vol_codes.

Optional:
- p_raw and p_cal computed via WinProbScorer from the bundle artifacts.

Usage:
  python export_golden_features.py \
    --bundle-id cdf34052888fead8 \
    --trade-store results/trades.clean.csv \
    --regimes-report results/meta_export/regimes_report.json \
    --out golden_features.parquet \
    --n-rows 800
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from winprob_loader import WinProbScorer


# --------------------------
# Utilities
# --------------------------

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def infer_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(12):
        if (p / ".git").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start.resolve()


def git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def utc_now_iso() -> str:
    """
    Robust across pandas versions:
    - If utcnow() is tz-naive, localize.
    - If it is already tz-aware, convert.
    """
    ts = pd.Timestamp.utcnow()
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def list_store_columns(path: Path) -> List[str]:
    suf = path.suffix.lower()
    if suf == ".parquet":
        # reads metadata; does not load full dataset
        return list(pd.read_parquet(path).head(0).columns)
    return list(pd.read_csv(path, nrows=0).columns)


# --------------------------
# Artifacts discovery
# --------------------------

def find_artifacts_dir(bundle_id: str, override: Optional[str] = None) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"--artifacts-dir not found: {p}")

    candidates = [
        Path(f"results/bundles/{bundle_id}"),
        Path(f"results/meta_export/{bundle_id}"),
        Path("results/meta_export"),
        Path(f"research_outputs/07_deployment_artifacts/{bundle_id}"),
        Path("research_outputs/07_deployment_artifacts"),
        Path(f"deploy/bundles/{bundle_id}"),
        Path(f"bundles/{bundle_id}"),
    ]

    for c in candidates:
        p = c.expanduser().resolve()
        if not p.exists():
            continue
        if (p / "model.joblib").exists() and (p / "feature_manifest.json").exists():
            return p

    msg = "Could not find artifacts dir for bundle_id.\nSearched:\n" + "\n".join([f"  - {c}" for c in candidates])
    msg += "\nProvide --artifacts-dir explicitly."
    raise FileNotFoundError(msg)


# --------------------------
# Trade store loading
# --------------------------

def load_trade_store(path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trade store not found: {path}")

    suf = path.suffix.lower()
    if suf == ".parquet":
        df = pd.read_parquet(path, columns=usecols) if usecols else pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False, usecols=usecols)

    if df.empty:
        raise RuntimeError(f"Trade store is empty: {path}")
    return df


def find_ts_sym_cols(columns: List[str]) -> Tuple[str, str]:
    ts_col = None
    for c in ("entry_ts", "timestamp", "ts", "decision_ts", "entry_time", "entry_datetime", "entry_timestamp"):
        if c in columns:
            ts_col = c
            break
    if ts_col is None:
        raise KeyError("Could not find entry timestamp column (expected one of entry_ts/timestamp/ts/decision_ts).")

    sym_col = None
    for c in ("symbol", "sym"):
        if c in columns:
            sym_col = c
            break
    if sym_col is None:
        raise KeyError("Could not find symbol column (expected 'symbol' or 'sym').")

    return ts_col, sym_col


# --------------------------
# Sampling (deterministic; boundary-biased)
# --------------------------

def boundary_mask(ts: pd.Series) -> pd.Series:
    t = ts.dt.tz_convert("UTC")
    hr = t.dt.hour
    mn = t.dt.minute
    is_1h = (mn == 0) | (mn == 55)
    is_4h = ((hr % 4 == 0) & (mn == 0)) | ((hr % 4 == 3) & (mn == 55))
    is_day = ((hr == 0) & (mn == 0)) | ((hr == 23) & (mn == 55))
    return is_1h | is_4h | is_day


def deterministic_sample(
    df: pd.DataFrame,
    ts: pd.Series,
    sym: pd.Series,
    n_rows: int,
    min_syms: int,
    max_syms: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    syms_all = sym.dropna().unique().tolist()
    syms_all = sorted([str(s).upper().strip() for s in syms_all if str(s).strip() != ""])
    if not syms_all:
        raise RuntimeError("No symbols found after parsing.")
    k = min(max_syms, len(syms_all))
    k = max(min_syms, min(k, len(syms_all)))
    syms_keep = list(rng.choice(np.array(syms_all, dtype=object), size=k, replace=False))
    syms_keep = sorted([str(s) for s in syms_keep])

    d = df.copy()
    d["_ts"] = ts
    d["_sym"] = sym
    d = d.dropna(subset=["_ts", "_sym"]).copy()
    d["_sym"] = d["_sym"].astype(str).str.upper().str.strip()
    d = d[d["_sym"].isin(syms_keep)].copy()
    if d.empty:
        raise RuntimeError("No rows left after filtering to selected symbols.")

    d["_ts"] = pd.to_datetime(d["_ts"], utc=True, errors="coerce").dt.tz_convert("UTC")
    d = d.dropna(subset=["_ts"]).copy()
    d["_month"] = d["_ts"].dt.strftime("%Y-%m")

    bm = boundary_mask(d["_ts"])
    d_b = d[bm].copy()
    n_b = min(max(int(round(n_rows * 0.25)), min(200, n_rows)), len(d_b))
    picked: List[int] = []

    if n_b > 0 and not d_b.empty:
        per_sym = max(2, int(np.ceil(n_b / max(1, d_b["_sym"].nunique()))))
        for _, g in d_b.groupby("_sym"):
            if len(picked) >= n_b:
                break
            g2 = g.sort_values("_ts")
            for idx in (g2.head(1).index.tolist() + g2.tail(1).index.tolist()):
                ii = int(idx)
                if ii not in picked:
                    picked.append(ii)
                    if len(picked) >= n_b:
                        break
            if len(picked) >= n_b:
                break
            remaining = [int(i) for i in g2.index.tolist() if int(i) not in picked]
            need = min(per_sym, n_b - len(picked))
            if need > 0 and remaining:
                take = rng.choice(np.array(remaining, dtype=int), size=min(need, len(remaining)), replace=False).tolist()
                picked.extend([int(x) for x in take])

        if len(picked) < n_b:
            remaining = [int(i) for i in d_b.index.tolist() if int(i) not in picked]
            if remaining:
                take = rng.choice(np.array(remaining, dtype=int), size=min(n_b - len(picked), len(remaining)), replace=False).tolist()
                picked.extend([int(x) for x in take])

    picked = picked[:n_b]

    d_r = d.drop(index=picked, errors="ignore").copy()
    need_r = max(0, n_rows - len(picked))
    if need_r > 0 and not d_r.empty:
        groups = list(d_r.groupby(["_sym", "_month"]).groups.items())
        rng.shuffle(groups)
        per_group = max(1, int(np.ceil(need_r / max(1, len(groups)))))
        for (_, _), idxs in groups:
            if need_r <= 0:
                break
            idxs = [int(i) for i in list(idxs)]
            take_n = min(per_group, need_r, len(idxs))
            take = rng.choice(np.array(idxs, dtype=int), size=take_n, replace=False).tolist()
            picked.extend([int(x) for x in take])
            need_r -= take_n

        if need_r > 0:
            remaining = [int(i) for i in d_r.index.tolist() if int(i) not in picked]
            if remaining:
                take = rng.choice(np.array(remaining, dtype=int), size=min(need_r, len(remaining)), replace=False).tolist()
                picked.extend([int(x) for x in take])

    out = d.loc[picked].copy()
    out = out.sort_values(["_sym", "_ts"]).reset_index(drop=True)
    return out


# --------------------------
# Regime-set derivation (matches backtester fallback semantics)
# --------------------------

DERIVABLE_REGIME_SET_COLS = {
    "funding_regime_code",
    "oi_regime_code",
    "btc_risk_regime_code",
    "risk_on",
    "risk_on_1",
    "S1_regime_code_1d",
    "S2_markov_x_vol1d",
    "S3_funding_x_oi",
    "S4_crowd_x_trend1d",
    "S5_btcRisk_x_regimeUp",
    "S6_fresh_x_compress",
}

ALT_COLS = {
    "btc_trend_slope": ["btc_trend_slope", "btcusdt_trend_slope"],
    "btc_vol_regime_level": ["btc_vol_regime_level", "btcusdt_vol_regime_level"],
}

def first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def require_col(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

def require_any(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    c = first_present(df, candidates)
    if c is None:
        raise KeyError(f"Missing required column ({label}). Acceptable: {candidates}")
    return c

def _bucket_terciles(x: float, q33: float, q66: float) -> float:
    try:
        xv = float(x)
        q33v = float(q33)
        q66v = float(q66)
    except Exception:
        return np.nan
    if not (np.isfinite(xv) and np.isfinite(q33v) and np.isfinite(q66v)):
        return np.nan
    if xv <= q33v:
        return 0.0
    if xv >= q66v:
        return 2.0
    return 1.0

def _trend_vol_codes(
    trend_regime_1d: Optional[Any],
    vol_regime_1d: Optional[Any],
    vol_prob_low_1d: Optional[Any],
    regime_code_1d: Optional[Any],
) -> Tuple[float, float]:
    tcode = np.nan
    if trend_regime_1d is not None and not (isinstance(trend_regime_1d, float) and np.isnan(trend_regime_1d)):
        s = str(trend_regime_1d).upper()
        if "BULL" in s:
            tcode = 1.0
        elif "BEAR" in s:
            tcode = 0.0
    if not np.isfinite(tcode) and regime_code_1d is not None:
        try:
            rc = int(float(regime_code_1d))
            tcode = 1.0 if rc in (2, 3) else 0.0
        except Exception:
            pass

    vcode = np.nan
    if vol_regime_1d is not None and not (isinstance(vol_regime_1d, float) and np.isnan(vol_regime_1d)):
        s = str(vol_regime_1d).upper()
        if "LOW" in s:
            vcode = 0.0
        elif "HIGH" in s:
            vcode = 1.0
    if not np.isfinite(vcode) and (vol_prob_low_1d is not None):
        try:
            vpl = float(vol_prob_low_1d)
            if np.isfinite(vpl):
                vcode = 1.0 if vpl < 0.5 else 0.0
        except Exception:
            pass
    if not np.isfinite(vcode) and regime_code_1d is not None:
        try:
            rc = int(float(regime_code_1d))
            vcode = 1.0 if rc in (0, 2) else 0.0
        except Exception:
            pass

    return tcode, vcode

def load_regimes_thresholds(path: Path) -> dict:
    rep = read_json(path)
    thr = rep.get("thresholds") or {}
    if not isinstance(thr, dict) or not thr:
        raise RuntimeError(f"regimes_report.json has no thresholds: {path}")
    thr["_regimes_report_path"] = str(path.resolve())
    return thr

def derive_regime_set_cols(sampled: pd.DataFrame, raw_cols: List[str], thresholds: dict) -> pd.DataFrame:
    missing = [c for c in DERIVABLE_REGIME_SET_COLS if (c in raw_cols and c not in sampled.columns)]
    if not missing:
        return sampled

    need_funding = any(c in missing for c in ("funding_regime_code", "S3_funding_x_oi"))
    need_oi = any(c in missing for c in ("oi_regime_code", "S3_funding_x_oi"))
    need_btc = any(c in missing for c in ("btc_risk_regime_code", "risk_on", "risk_on_1", "S5_btcRisk_x_regimeUp"))
    need_ru = any(c in missing for c in ("risk_on", "risk_on_1", "S5_btcRisk_x_regimeUp"))
    need_rc = any(c in missing for c in ("S1_regime_code_1d", "S2_markov_x_vol1d", "S4_crowd_x_trend1d"))
    need_m4 = "S2_markov_x_vol1d" in missing
    need_crowd = "S4_crowd_x_trend1d" in missing
    need_fresh_comp = "S6_fresh_x_compress" in missing

    if need_funding:
        require_col(sampled, "funding_rate")

    oi_source = str(thresholds.get("oi_source") or "oi_z_7d")
    if need_oi:
        if oi_source == "oi_z_7d":
            require_col(sampled, "oi_z_7d")
        else:
            require_col(sampled, "oi_pct_1d")

    btc_trend_col = None
    btc_vol_col = None
    if need_btc:
        btc_trend_col = require_any(sampled, ALT_COLS["btc_trend_slope"], "btc_trend_slope")
        btc_vol_col = require_any(sampled, ALT_COLS["btc_vol_regime_level"], "btc_vol_regime_level")

    if need_ru:
        require_col(sampled, "regime_up")

    if need_rc:
        require_col(sampled, "regime_code_1d")

    if need_m4:
        require_col(sampled, "markov_state_4h")

    if need_crowd:
        require_col(sampled, "crowd_side")

    if need_fresh_comp:
        require_col(sampled, "days_since_prev_break")
        require_col(sampled, "consolidation_range_atr")

    eps = float(thresholds.get("funding_neutral_eps"))
    oi_q33 = thresholds.get("oi_q33")
    oi_q66 = thresholds.get("oi_q66")
    btc_vol_hi = float(thresholds.get("btc_vol_hi"))
    fresh_q33 = thresholds.get("fresh_q33")
    fresh_q66 = thresholds.get("fresh_q66")
    comp_q33 = thresholds.get("compression_q33")
    comp_q66 = thresholds.get("compression_q66")

    for c in missing:
        sampled[c] = np.nan

    for i in range(len(sampled)):
        r = sampled.iloc[i]

        funding_regime_code = np.nan
        if need_funding:
            fr = pd.to_numeric(r.get("funding_rate", np.nan), errors="coerce")
            if np.isfinite(fr) and np.isfinite(eps) and eps > 0:
                if fr <= -eps:
                    funding_regime_code = -1.0
                elif fr >= eps:
                    funding_regime_code = 1.0
                else:
                    funding_regime_code = 0.0

        oi_regime_code = np.nan
        if need_oi and (oi_q33 is not None) and (oi_q66 is not None):
            try:
                q33v = float(oi_q33)
                q66v = float(oi_q66)
                oi_val = pd.to_numeric(r.get(oi_source, np.nan), errors="coerce")
                if np.isfinite(oi_val):
                    if oi_val <= q33v:
                        oi_regime_code = -1.0
                    elif oi_val >= q66v:
                        oi_regime_code = 1.0
                    else:
                        oi_regime_code = 0.0
            except Exception:
                oi_regime_code = np.nan

        btc_trend_up = np.nan
        btc_vol_high = np.nan
        btc_risk_regime_code = np.nan
        if need_btc and btc_trend_col and btc_vol_col:
            bts = pd.to_numeric(r.get(btc_trend_col, np.nan), errors="coerce")
            bvl = pd.to_numeric(r.get(btc_vol_col, np.nan), errors="coerce")
            if np.isfinite(bts) and np.isfinite(bvl) and np.isfinite(btc_vol_hi):
                btc_trend_up = 1.0 if bts > 0.0 else 0.0
                btc_vol_high = 1.0 if bvl >= btc_vol_hi else 0.0
                btc_risk_regime_code = btc_trend_up * 2.0 + btc_vol_high

        ru = 0.0
        if need_ru:
            x = pd.to_numeric(r.get("regime_up", np.nan), errors="coerce")
            ru = 1.0 if (np.isfinite(x) and float(x) >= 0.5) else 0.0

        risk_on = np.nan
        if ("risk_on" in missing or "risk_on_1" in missing) and np.isfinite(btc_trend_up) and np.isfinite(btc_vol_high):
            risk_on = 1.0 if (ru == 1.0 and btc_trend_up == 1.0 and btc_vol_high == 0.0) else 0.0

        freshness_code = np.nan
        compression_code = np.nan
        if need_fresh_comp and (fresh_q33 is not None and fresh_q66 is not None and comp_q33 is not None and comp_q66 is not None):
            ds = pd.to_numeric(r.get("days_since_prev_break", np.nan), errors="coerce")
            cr = pd.to_numeric(r.get("consolidation_range_atr", np.nan), errors="coerce")
            if np.isfinite(ds):
                freshness_code = _bucket_terciles(ds, float(fresh_q33), float(fresh_q66))
            if np.isfinite(cr):
                compression_code = _bucket_terciles(cr, float(comp_q33), float(comp_q66))

        trend_regime_1d = r.get("trend_regime_1d", None) if "trend_regime_1d" in sampled.columns else None
        vol_regime_1d = r.get("vol_regime_1d", None) if "vol_regime_1d" in sampled.columns else None
        vol_prob_low_1d = r.get("vol_prob_low_1d", None) if "vol_prob_low_1d" in sampled.columns else None
        regime_code_1d = r.get("regime_code_1d", None) if "regime_code_1d" in sampled.columns else None

        trend_code, vol_code = _trend_vol_codes(trend_regime_1d, vol_regime_1d, vol_prob_low_1d, regime_code_1d)

        S1 = np.nan
        if "S1_regime_code_1d" in missing:
            try:
                S1 = float(regime_code_1d) if regime_code_1d is not None else np.nan
            except Exception:
                S1 = np.nan

        S2 = np.nan
        if "S2_markov_x_vol1d" in missing:
            ms = r.get("markov_state_4h", None)
            try:
                ms_i = int(float(ms)) if ms is not None and not (isinstance(ms, float) and np.isnan(ms)) else None
            except Exception:
                ms_i = None
            if (ms_i is not None) and np.isfinite(vol_code):
                S2 = float(ms_i * 2 + int(vol_code))

        S3 = np.nan
        if "S3_funding_x_oi" in missing and np.isfinite(funding_regime_code) and np.isfinite(oi_regime_code):
            S3 = float((int(funding_regime_code) + 1) * 3 + (int(oi_regime_code) + 1))

        S4 = np.nan
        if "S4_crowd_x_trend1d" in missing:
            crowd_side = r.get("crowd_side", np.nan)
            try:
                cs = float(crowd_side)
            except Exception:
                cs = np.nan
            if np.isfinite(cs) and np.isfinite(trend_code):
                S4 = float((int(cs) + 1) * 2 + int(trend_code))

        S5 = np.nan
        if "S5_btcRisk_x_regimeUp" in missing and np.isfinite(btc_risk_regime_code):
            S5 = float(int(btc_risk_regime_code) * 2 + int(ru))

        S6 = np.nan
        if "S6_fresh_x_compress" in missing and np.isfinite(freshness_code) and np.isfinite(compression_code):
            S6 = float(int(freshness_code) * 3 + int(compression_code))

        updates = {
            "funding_regime_code": funding_regime_code,
            "oi_regime_code": oi_regime_code,
            "btc_risk_regime_code": btc_risk_regime_code,
            "risk_on": risk_on,
            "risk_on_1": risk_on,
            "S1_regime_code_1d": S1,
            "S2_markov_x_vol1d": S2,
            "S3_funding_x_oi": S3,
            "S4_crowd_x_trend1d": S4,
            "S5_btcRisk_x_regimeUp": S5,
            "S6_fresh_x_compress": S6,
        }

        for k, v in updates.items():
            if k in missing:
                sampled.at[i, k] = v

    return sampled


# --------------------------
# Scoring (optional)
# --------------------------

def model_feature_names(model: Any) -> Optional[List[str]]:
    names = None
    if hasattr(model, "feature_name_"):
        try:
            names = list(model.feature_name_)
        except Exception:
            names = None
    if not names and hasattr(model, "feature_names_in_"):
        try:
            names = list(model.feature_names_in_)
        except Exception:
            names = None
    if not names and hasattr(model, "booster_") and hasattr(model.booster_, "feature_name"):
        try:
            names = list(model.booster_.feature_name())
        except Exception:
            names = None
    if names and len(names) > 0:
        return names
    return None


def ensure_named_matrix(X: Any, model: Any) -> Any:
    """
    If model expects named features and X is not a DataFrame, wrap into DataFrame
    with model feature names (only if shape matches).
    """
    if isinstance(X, pd.DataFrame):
        return X
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    names = model_feature_names(model)
    if names and len(names) == arr.shape[1]:
        return pd.DataFrame(arr, columns=names)
    return arr


def to_row_dict(df_row: pd.Series, raw_cols: List[str]) -> Dict[str, Any]:
    return {c: df_row.get(c, np.nan) for c in raw_cols}


def score_rows(df: pd.DataFrame, scorer: WinProbScorer, raw_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    p_raw = np.full(len(df), np.nan, dtype=float)
    p_cal = np.full(len(df), np.nan, dtype=float)

    # With DataFrame wrapping, the LightGBM "valid feature names" warning should stop.
    # If it still appears (e.g., shape mismatch), this filter prevents log spam.
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names*",
        category=UserWarning,
        module="sklearn.utils.validation",
    )

    for i in range(len(df)):
        row = to_row_dict(df.loc[i], raw_cols)
        X = scorer._build_X_raw(row)
        X = ensure_named_matrix(X, scorer.model)
        pr = float(scorer.model.predict_proba(X)[:, 1][0])
        pc = float(scorer._calibrate(pr))
        p_raw[i] = pr
        p_cal[i] = pc

    return p_raw, p_cal


# --------------------------
# CLI / main
# --------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-id", required=True)
    ap.add_argument("--artifacts-dir", default="")
    ap.add_argument("--trade-store", default="results/trades.clean.csv")
    ap.add_argument("--regimes-report", default="", help="Path to regimes_report.json")
    ap.add_argument("--out", default="golden_features.parquet")
    ap.add_argument("--n-rows", type=int, default=800)
    ap.add_argument("--min-symbols", type=int, default=10)
    ap.add_argument("--max-symbols", type=int, default=20)
    ap.add_argument("--no-score", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    bundle_id = str(args.bundle_id).strip()
    seed = int(bundle_id[:8], 16) if len(bundle_id) >= 8 else abs(hash(bundle_id)) % (2**32 - 1)

    artifacts_dir = find_artifacts_dir(bundle_id, override=(args.artifacts_dir or None))
    scorer = WinProbScorer(artifacts_dir)

    man = read_json(artifacts_dir / "feature_manifest.json")
    feats = man.get("features") or {}
    numeric_cols = list(feats.get("numeric_cols") or [])
    cat_cols = list(feats.get("cat_cols") or [])
    raw_cols = numeric_cols + cat_cols
    if len(raw_cols) != 73:
        raise RuntimeError(f"Expected 73 raw feature cols; got {len(raw_cols)}")

    store_path = Path(args.trade_store).expanduser().resolve()
    store_cols = list_store_columns(store_path)
    ts_col, sym_col = find_ts_sym_cols(store_cols)

    # Load minimal first: ts/sym + raw cols that exist
    base_usecols = [ts_col, sym_col] + raw_cols
    base_usecols = [c for c in base_usecols if c in store_cols]
    df0 = load_trade_store(store_path, usecols=base_usecols)

    ts = pd.to_datetime(df0[ts_col], utc=True, errors="coerce")
    sym = df0[sym_col].astype(str).str.upper().str.strip()

    # Which raw cols are missing from store?
    missing_raw = [c for c in raw_cols if c not in df0.columns]
    missing_unfillable = [c for c in missing_raw if c not in DERIVABLE_REGIME_SET_COLS]
    if missing_unfillable:
        raise KeyError(
            "Trade store missing required raw feature columns that are not derivable:\n"
            + "\n".join([f"  - {c}" for c in missing_unfillable])
            + f"\nTrade store: {store_path}"
        )

    # If we must derive, ensure inputs are present
    thresholds: Optional[dict] = None
    rr_path: Optional[Path] = None
    if missing_raw:
        if not args.regimes_report:
            raise FileNotFoundError(
                "Trade store is missing derivable regime-set columns, but --regimes-report was not provided."
            )
        rr_path = Path(args.regimes_report).expanduser().resolve()
        thresholds = load_regimes_thresholds(rr_path)

        needed_inputs = {
            "funding_rate",
            "oi_z_7d",
            "oi_pct_1d",
            "btc_trend_slope",
            "btcusdt_trend_slope",
            "btc_vol_regime_level",
            "btcusdt_vol_regime_level",
            "crowd_side",
            "markov_state_4h",
            "vol_prob_low_1d",
            "regime_code_1d",
            "regime_up",
            "days_since_prev_break",
            "consolidation_range_atr",
        }
        add_cols = [c for c in needed_inputs if (c not in df0.columns and c in store_cols)]
        if add_cols:
            df_extra = load_trade_store(store_path, usecols=[ts_col, sym_col] + add_cols)
            # Align by index (same row ordering as original file)
            for c in add_cols:
                df0[c] = df_extra[c]

    sampled = deterministic_sample(
        df=df0,
        ts=ts,
        sym=sym,
        n_rows=max(1, int(args.n_rows)),
        min_syms=max(1, int(args.min_symbols)),
        max_syms=max(1, int(args.max_symbols)),
        seed=seed,
    )

    if missing_raw:
        assert thresholds is not None
        sampled = derive_regime_set_cols(sampled, raw_cols=raw_cols, thresholds=thresholds)

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(sampled["_ts"], utc=True, errors="coerce").dt.tz_convert("UTC")
    out["symbol"] = sampled["_sym"].astype(str).str.upper().str.strip()

    for c in numeric_cols:
        out[c] = pd.to_numeric(sampled[c], errors="coerce").astype(float)

    for c in cat_cols:
        v = sampled[c]
        out[c] = v.where(pd.notna(v), np.nan).astype("object")
        out.loc[out[c].notna(), c] = out.loc[out[c].notna(), c].astype(str)

    if not args.no_score:
        pr, pc = score_rows(out[["timestamp", "symbol"] + raw_cols].reset_index(drop=True), scorer, raw_cols)
        out["p_raw"] = pr
        out["p_cal"] = pc

    repo_root = infer_repo_root(Path.cwd())
    out["bundle_id"] = bundle_id
    out["artifacts_dir"] = str(artifacts_dir)
    out["trade_store_path"] = str(store_path)
    out["trade_store_ts_col"] = ts_col
    out["trade_store_sym_col"] = sym_col
    out["model_sha256"] = sha256_file(artifacts_dir / "model.joblib")

    if rr_path is not None:
        out["regimes_report_path"] = str(rr_path)
        out["regimes_report_sha256"] = sha256_file(rr_path)

    cal_path = artifacts_dir / "calibration.json"
    out["calibration_method"] = (
        str(read_json(cal_path).get("chosen_method")).strip().lower() if cal_path.exists() else "none"
    )

    out["exporter_git_commit"] = git_commit(repo_root)
    out["exported_at_utc"] = utc_now_iso()

    out["python_version"] = sys.version.split()[0]
    out["platform"] = platform.platform()
    out["numpy_version"] = getattr(np, "__version__", "")
    out["pandas_version"] = getattr(pd, "__version__", "")
    try:
        import sklearn  # type: ignore
        out["sklearn_version"] = getattr(sklearn, "__version__", "")
    except Exception:
        out["sklearn_version"] = ""
    try:
        import joblib  # type: ignore
        out["joblib_version"] = getattr(joblib, "__version__", "")
    except Exception:
        out["joblib_version"] = ""

    required = ["timestamp", "symbol"] + raw_cols
    extras = [c for c in out.columns if c not in required]
    out = out[required + extras]

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = False
    if out_path.suffix.lower() == ".parquet":
        try:
            out.to_parquet(out_path, index=False)
            wrote = True
        except Exception as e:
            print(f"[export] WARNING: parquet write failed ({e!r}); falling back to CSV", file=sys.stderr)

    if not wrote:
        if out_path.suffix.lower() != ".csv":
            out_path = out_path.with_suffix(".csv")
        out.to_csv(out_path, index=False)
        wrote = True

    print(f"[export] wrote {len(out)} rows to {out_path}")
    print(f"[export] symbols={out['symbol'].nunique()} ts_range={out['timestamp'].min()}..{out['timestamp'].max()}")
    if "p_cal" in out.columns:
        arr = out["p_cal"].to_numpy(dtype=float)
        if np.isfinite(arr).any():
            print(f"[export] p_cal range: {np.nanmin(arr):.6f}..{np.nanmax(arr):.6f}")


if __name__ == "__main__":
    main()
