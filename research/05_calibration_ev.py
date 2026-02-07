#!/usr/bin/env python3
"""
research/05_calibration_ev.py

Step 05: calibrate OOF probabilities + compute EV / threshold / sizing tables.

Inputs:
  - OOF predictions from step 04: research_outputs/04_models_cv/oof_predictions.parquet
    Expected columns:
      - trade_id (optional; if missing, index will be used)
      - entry_ts
      - pnl_R
      - y_* targets (at least the chosen --target)
      - p_* probability columns (e.g. p_lgbm, p_logreg)
      - risk_on_1 (optional; preferred)
      - risk_on (optional; legacy alias)
      - S*_... slice columns (optional)

Outputs (per model column):
  - calibration_bins_<model>.csv
  - calibration_metrics_<model>.json
  - calibrator_<model>_<method>.joblib
  - oof_with_calibrated_<model>.parquet
  - ev_thresholds_<model>.csv
  - sizing_curve_<model>.csv
  - summary.json

Key ideas:
  - OOF predictions are already out-of-sample; we fit calibrators on OOF and validate on a time holdout.
  - Two calibrators supported: sigmoid (Platt) and isotonic. Also "none" (raw).
  - EV tables computed from pnl_R and target outcomes; supports ALL and RISK_ON_1 scopes.
  - Sizing curve maps calibrated probability to a risk multiplier based on mean pnl_R by probability bins.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

# matplotlib is optional; we only save PNGs if available and not disabled
try:
    import matplotlib.pyplot as plt  # type: ignore
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _to_datetime_utc(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            return pd.to_datetime(s, utc=True, errors="coerce")
        except Exception:
            return pd.to_datetime(s.astype(str), utc=True, errors="coerce")
    # numeric ts? try s/ms
    if pd.api.types.is_numeric_dtype(s):
        x = _to_num(s)
        mx = x.max()
        unit = "ms" if (mx is not None and np.isfinite(mx) and mx > 1e12) else "s"
        return pd.to_datetime(x, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


def _clip01(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(p.astype(np.float64), eps, 1.0 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip01(p)
    return np.log(p / (1.0 - p))


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def _safe_logloss(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(log_loss(y, _clip01(p)))
    except Exception:
        return float("nan")


def _safe_brier(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y, _clip01(p)))
    except Exception:
        return float("nan")


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> float:
    """
    Expected Calibration Error with equal-width bins.
    """
    y = y.astype(int)
    p = _clip01(p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            continue
        w = float(np.sum(m)) / float(n)
        acc = float(np.mean(y[m]))
        conf = float(np.mean(p[m]))
        ece += w * abs(acc - conf)
    return float(ece)


def _calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    p = _clip01(p)
    y = y.astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.digitize(p, bins) - 1
    b = np.clip(b, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        m = (b == i)
        if not np.any(m):
            rows.append(
                {
                    "bin": i,
                    "bin_low": float(bins[i]),
                    "bin_high": float(bins[i + 1]),
                    "n": 0,
                    "mean_p": np.nan,
                    "obs_rate": np.nan,
                }
            )
            continue
        rows.append(
            {
                "bin": i,
                "bin_low": float(bins[i]),
                "bin_high": float(bins[i + 1]),
                "n": int(np.sum(m)),
                "mean_p": float(np.mean(p[m])),
                "obs_rate": float(np.mean(y[m])),
            }
        )
    return pd.DataFrame(rows)


def _maybe_plot_reliability(df_bins: pd.DataFrame, out_png: str, title: str) -> None:
    if not HAS_PLT:
        return
    x = df_bins["mean_p"].to_numpy()
    y = df_bins["obs_rate"].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.scatter(x[m], y[m])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title(title)
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close()


# -----------------------------
# Calibrator wrappers
# -----------------------------
@dataclass
class Calibrator:
    method: str  # "none" | "sigmoid" | "isotonic"
    model_col: str
    fitted: object

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = _clip01(p_raw)
        if self.method == "none":
            return p_raw
        if self.method == "isotonic":
            iso: IsotonicRegression = self.fitted  # type: ignore
            return _clip01(iso.transform(p_raw))
        if self.method == "sigmoid":
            lr: LogisticRegression = self.fitted  # type: ignore
            z = _logit(p_raw).reshape(-1, 1)
            return _clip01(lr.predict_proba(z)[:, 1])
        raise ValueError(f"Unknown method: {self.method}")


def fit_calibrator(method: str, y: np.ndarray, p_raw: np.ndarray) -> Calibrator:
    y = y.astype(int)
    p_raw = _clip01(p_raw)

    if method == "none":
        return Calibrator(method="none", model_col="", fitted=None)

    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_raw, y)
        return Calibrator(method="isotonic", model_col="", fitted=iso)

    if method == "sigmoid":
        # Platt scaling on logit(p_raw)
        z = _logit(p_raw).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", max_iter=2000)
        lr.fit(z, y)
        return Calibrator(method="sigmoid", model_col="", fitted=lr)

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# EV / threshold / sizing
# -----------------------------
def threshold_table(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    thresholds: np.ndarray,
    min_trades: int,
    scope_name: str,
) -> pd.DataFrame:
    """
    For each threshold: keep trades where score>=thr and report stats.
    """
    s = _to_num(df[score_col]).to_numpy(dtype=np.float64)
    y = _to_num(df[target_col]).to_numpy(dtype=np.float64)
    pnl = _to_num(df["pnl_R"]).to_numpy(dtype=np.float64)

    y_win = _to_num(df["y_win"]).to_numpy(dtype=np.float64) if "y_win" in df.columns else None
    y_time = _to_num(df["y_time"]).to_numpy(dtype=np.float64) if "y_time" in df.columns else None

    rows = []
    for thr in thresholds:
        m = np.isfinite(s) & (s >= float(thr)) & np.isfinite(pnl) & np.isfinite(y)
        n = int(np.sum(m))
        if n < min_trades:
            continue

        pnl_m = pnl[m]
        y_m = y[m]
        row = {
            "scope": scope_name,
            "threshold": float(thr),
            "n": n,
            "pos_rate": float(np.mean(y_m)),
            "mean_pnl_R": float(np.mean(pnl_m)),
            "sum_pnl_R": float(np.sum(pnl_m)),
        }

        if y_win is not None:
            row["win_rate"] = float(np.nanmean(y_win[m]))
        if y_time is not None:
            row["time_rate"] = float(np.nanmean(y_time[m]))
            pnl_not_time = pnl_m[(y_time[m] == 0)]
            row["mean_pnl_R_not_time"] = float(np.mean(pnl_not_time)) if len(pnl_not_time) else float("nan")

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # convenience ranks
    out["rank_by_mean_pnl"] = out["mean_pnl_R"].rank(ascending=False, method="dense")
    out["rank_by_sum_pnl"] = out["sum_pnl_R"].rank(ascending=False, method="dense")
    return out.sort_values(["scope", "threshold"])


def sizing_curve(
    df: pd.DataFrame,
    score_col: str,
    n_bins: int,
    min_bin_n: int,
    min_mult: float,
    max_mult: float,
) -> pd.DataFrame:
    """
    Map score -> mean pnl_R in bins and suggest a risk multiplier.
    Simple, robust mapping:
      - compute mean pnl_R per probability bin
      - convert to multiplier:
          if mean_pnl_R <= 0 => min_mult
          else scale linearly between min_mult..max_mult using mean_pnl_R / p95_positive_mean
    """
    s = _to_num(df[score_col]).to_numpy(dtype=np.float64)
    pnl = _to_num(df["pnl_R"]).to_numpy(dtype=np.float64)

    ok = np.isfinite(s) & np.isfinite(pnl)
    s = s[ok]
    pnl = pnl[ok]
    if len(s) == 0:
        return pd.DataFrame()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.digitize(s, bins) - 1
    b = np.clip(b, 0, n_bins - 1)

    rows = []
    means_pos = []
    for i in range(n_bins):
        m = (b == i)
        n = int(np.sum(m))
        if n == 0:
            rows.append(
                {"bin": i, "bin_low": float(bins[i]), "bin_high": float(bins[i + 1]), "n": 0,
                 "mean_score": np.nan, "mean_pnl_R": np.nan}
            )
            continue
        mp = float(np.mean(pnl[m]))
        ms = float(np.mean(s[m]))
        rows.append(
            {"bin": i, "bin_low": float(bins[i]), "bin_high": float(bins[i + 1]), "n": n,
             "mean_score": ms, "mean_pnl_R": mp}
        )
        if n >= min_bin_n and np.isfinite(mp) and mp > 0:
            means_pos.append(mp)

    dfc = pd.DataFrame(rows)
    if dfc.empty:
        return dfc

    p95 = float(np.percentile(means_pos, 95)) if len(means_pos) else float("nan")

    mult = []
    for _, r in dfc.iterrows():
        mp = r["mean_pnl_R"]
        n = int(r["n"])

        # sparse/invalid => mark NaN, we will fill using previous stable bin
        if (not np.isfinite(mp)) or (n < min_bin_n):
            mult.append(np.nan)
            continue

        if mp <= 0:
            mult.append(float(min_mult))
            continue
        if (not np.isfinite(p95)) or (p95 <= 0):
            mult.append(float(min_mult))
            continue

        x = float(np.clip(mp / p95, 0.0, 1.0))
        mult.append(float(min_mult + (max_mult - min_mult) * x))

    # carry-forward fill: sparse bin inherits previous stable multiplier
    last = float(min_mult)
    for i in range(len(mult)):
        if np.isfinite(mult[i]):
            last = float(mult[i])
        else:
            mult[i] = float(last) if i > 0 else float(min_mult)

    dfc["risk_multiplier"] = mult
    return dfc

# -----------------------------
# Main
# -----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 05: calibrate OOF probabilities and compute EV/threshold tables.")
    p.add_argument("--oof", type=str, default="research_outputs/04_models_cv/oof_predictions.parquet")
    p.add_argument("--outdir", type=str, default="research_outputs/05_calibration_ev")

    p.add_argument("--target", type=str, default="y_good_05", help="Binary target column to calibrate against.")
    p.add_argument("--model-col", type=str, default="", help="Which p_* column to use (default: all p_* columns found).")

    p.add_argument("--methods", type=str, default="none,sigmoid,isotonic", help="Comma list: none,sigmoid,isotonic.")
    p.add_argument("--fit-scope", type=str, default="ALL", choices=["ALL", "RISK_ON_1"],
                   help="Fit calibrator on ALL or only risk_on_1==1 subset (fallback: risk_on).")

    p.add_argument("--holdout-frac", type=float, default=0.30,
                   help="Time holdout fraction (last part of time-ordered data) for calibrator selection.")
    p.add_argument("--n-calib-bins", type=int, default=20)

    p.add_argument("--min-trades", type=int, default=200, help="Min trades required for threshold rows.")
    p.add_argument("--n-thresholds", type=int, default=51, help="Threshold grid size over [0,1].")

    p.add_argument("--sizing-bins", type=int, default=8)
    p.add_argument("--sizing-min-bin-n", type=int, default=25)
    p.add_argument("--min-mult", type=float, default=0.01, help="Min risk multiplier (e.g. 0.01 for tiny probing size).")
    p.add_argument("--max-mult", type=float, default=1.00, help="Max risk multiplier (e.g. 1.0 = full risk budget).")

    p.add_argument("--no-plots", action="store_true", help="Disable PNG reliability plots.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    _ensure_dir(args.outdir)

    if not os.path.exists(args.oof):
        raise FileNotFoundError(f"OOF file not found: {args.oof}")

    df = pd.read_parquet(args.oof)
    if df.empty:
        raise RuntimeError("OOF parquet is empty.")

    # Ensure trade_id
    if "trade_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "trade_id"})

    # Required columns
    if "entry_ts" not in df.columns:
        raise RuntimeError("OOF file must contain entry_ts.")
    if "pnl_R" not in df.columns:
        raise RuntimeError("OOF file must contain pnl_R.")
    if args.target not in df.columns:
        raise RuntimeError(f"OOF file missing target column: {args.target}")

    df["entry_ts"] = _to_datetime_utc(df["entry_ts"])
    df = df[df["entry_ts"].notna()].copy()

    # Identify model probability columns
    p_cols = [c for c in df.columns if c.startswith("p_")]
    if args.model_col.strip():
        if args.model_col not in df.columns:
            raise RuntimeError(f"--model-col not found: {args.model_col}")
        p_cols = [args.model_col]
    if not p_cols:
        raise RuntimeError("No p_* probability columns found in OOF file.")

    # Prepare y and base filters
    df[args.target] = _to_num(df[args.target])
    df = df[df[args.target].notna()].copy()
    df["pnl_R"] = _to_num(df["pnl_R"])

    # Optional fit scope
    fit_df = df
    if args.fit_scope == "RISK_ON_1":
        risk_col = "risk_on_1" if "risk_on_1" in df.columns else ("risk_on" if "risk_on" in df.columns else None)
        if risk_col is None:
            raise RuntimeError("fit-scope=RISK_ON_1 requested but neither risk_on_1 nor risk_on is present in OOF.")
        r = _to_num(df[risk_col]).fillna(0).to_numpy()
        fit_df = df.loc[r == 1].copy()

    # Time split for calibrator selection
    fit_df = fit_df.sort_values("entry_ts")
    n = len(fit_df)
    n_hold = int(np.floor(float(args.holdout_frac) * n))
    n_hold = max(1, min(n_hold, n - 1))
    split_idx = n - n_hold
    df_cal_train = fit_df.iloc[:split_idx].copy()
    df_cal_test = fit_df.iloc[split_idx:].copy()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"none", "sigmoid", "isotonic"}
    for m in methods:
        if m not in allowed:
            raise RuntimeError(f"Unknown method in --methods: {m}")

    thresholds = np.linspace(0.0, 1.0, int(args.n_thresholds))

    summary: Dict[str, object] = {
        "oof": args.oof,
        "outdir": args.outdir,
        "target": args.target,
        "fit_scope": args.fit_scope,
        "holdout_frac": float(args.holdout_frac),
        "n_rows_total": int(len(df)),
        "n_rows_fit_scope": int(len(fit_df)),
        "n_rows_cal_train": int(len(df_cal_train)),
        "n_rows_cal_test": int(len(df_cal_test)),
        "models": {},
    }

    for model_col in p_cols:
        p_raw_full = _to_num(df[model_col]).to_numpy(dtype=np.float64)
        y_full = _to_num(df[args.target]).to_numpy(dtype=np.float64)
        ok_full = np.isfinite(p_raw_full) & np.isfinite(y_full)
        df_model = df.loc[ok_full].copy()
        if df_model.empty:
            print(f"[05_calibration_ev] WARNING: model {model_col} has no valid rows after numeric filtering.", flush=True)
            continue

        # For selection, use fit_df split (scope-selected)
        p_train = _to_num(df_cal_train[model_col]).to_numpy(dtype=np.float64)
        y_train = _to_num(df_cal_train[args.target]).to_numpy(dtype=np.float64)
        ok_tr = np.isfinite(p_train) & np.isfinite(y_train)
        p_train = p_train[ok_tr]
        y_train = y_train[ok_tr].astype(int)

        p_test = _to_num(df_cal_test[model_col]).to_numpy(dtype=np.float64)
        y_test = _to_num(df_cal_test[args.target]).to_numpy(dtype=np.float64)
        ok_te = np.isfinite(p_test) & np.isfinite(y_test)
        p_test = p_test[ok_te]
        y_test = y_test[ok_te].astype(int)

        if len(p_train) < 200 or len(p_test) < 50:
            print(f"[05_calibration_ev] WARNING: small calibrator split for {model_col}: train={len(p_train)} test={len(p_test)}", flush=True)

        # Evaluate methods on time-holdout
        method_metrics: Dict[str, Dict[str, float]] = {}
        for m in methods:
            cal = fit_calibrator(m, y_train, p_train)
            p_te = cal.predict(p_test)

            method_metrics[m] = {
                "auc": _safe_auc(y_test, p_te),
                "avg_precision": _safe_ap(y_test, p_te),
                "logloss": _safe_logloss(y_test, p_te),
                "brier": _safe_brier(y_test, p_te),
                "ece": _ece(y_test, p_te, n_bins=int(args.n_calib_bins)),
            }

        # Choose best by logloss then brier
        best_method = sorted(
            methods,
            key=lambda m: (
                np.inf if not np.isfinite(method_metrics[m]["logloss"]) else method_metrics[m]["logloss"],
                np.inf if not np.isfinite(method_metrics[m]["brier"]) else method_metrics[m]["brier"],
            ),
        )[0]

        # Fit chosen calibrator on full fit_df scope (for deployment/export)
        p_fit = _to_num(fit_df[model_col]).to_numpy(dtype=np.float64)
        y_fit = _to_num(fit_df[args.target]).to_numpy(dtype=np.float64)
        ok_fit = np.isfinite(p_fit) & np.isfinite(y_fit)
        cal_final = fit_calibrator(best_method, y_fit[ok_fit].astype(int), p_fit[ok_fit])
        cal_final.model_col = model_col

        # Apply calibrator to all rows (regardless of fit_scope)
        p_all = _to_num(df_model[model_col]).to_numpy(dtype=np.float64)
        p_cal = cal_final.predict(p_all)

        out_df = df_model.copy()
        out_df[f"{model_col}_cal"] = p_cal

        # Calibration bins (raw and calibrated)
        y_all = _to_num(out_df[args.target]).to_numpy(dtype=np.float64).astype(int)
        bins_raw = _calibration_bins(y_all, _clip01(p_all), n_bins=int(args.n_calib_bins))
        bins_cal = _calibration_bins(y_all, p_cal, n_bins=int(args.n_calib_bins))

        bins_raw["variant"] = "raw"
        bins_cal["variant"] = f"cal_{best_method}"
        bins = pd.concat([bins_raw, bins_cal], ignore_index=True)

        bins_path = os.path.join(args.outdir, f"calibration_bins_{model_col}.csv")
        bins.to_csv(bins_path, index=False)

        # Reliability plots
        if not args.no_plots:
            _maybe_plot_reliability(
                bins_raw,
                os.path.join(args.outdir, f"reliability_{model_col}_raw.png"),
                title=f"{model_col} raw reliability ({args.target})",
            )
            _maybe_plot_reliability(
                bins_cal,
                os.path.join(args.outdir, f"reliability_{model_col}_cal_{best_method}.png"),
                title=f"{model_col} calibrated ({best_method}) reliability ({args.target})",
            )

        # Save calibrator artifact
        cal_path = os.path.join(args.outdir, f"calibrator_{model_col}_{best_method}.joblib")
        joblib.dump(cal_final, cal_path)

        # Save calibrated OOF
        oof_path = os.path.join(args.outdir, f"oof_with_calibrated_{model_col}.parquet")
        out_df.to_parquet(oof_path, index=False)

        # EV threshold tables for ALL and (if available) RISK_ON_1
        ev_rows = []
        ev_all = threshold_table(out_df, score_col=f"{model_col}_cal", target_col=args.target,
                                 thresholds=thresholds, min_trades=int(args.min_trades), scope_name="ALL")
        if not ev_all.empty:
            ev_rows.append(ev_all)

        risk_col = "risk_on_1" if "risk_on_1" in out_df.columns else ("risk_on" if "risk_on" in out_df.columns else None)
        if risk_col is not None:
            r = _to_num(out_df[risk_col]).fillna(0).to_numpy()
            out_risk = out_df.loc[r == 1].copy()
            ev_r = threshold_table(out_risk, score_col=f"{model_col}_cal", target_col=args.target,
                                   thresholds=thresholds, min_trades=int(args.min_trades), scope_name="RISK_ON_1")
            if not ev_r.empty:
                ev_rows.append(ev_r)

        ev = pd.concat(ev_rows, ignore_index=True) if ev_rows else pd.DataFrame()
        ev_path = os.path.join(args.outdir, f"ev_thresholds_{model_col}.csv")
        ev.to_csv(ev_path, index=False)

        # Best thresholds (by mean pnl_R and by sum pnl_R), per scope
        best: Dict[str, Dict[str, object]] = {}
        if not ev.empty:
            for scope in ev["scope"].unique().tolist():
                sub = ev[ev["scope"] == scope].copy()
                if sub.empty:
                    continue
                best_mean = sub.sort_values("mean_pnl_R", ascending=False).head(1)
                best_sum = sub.sort_values("sum_pnl_R", ascending=False).head(1)
                best[scope] = {
                    "best_by_mean_pnl_R": best_mean.to_dict(orient="records")[0],
                    "best_by_sum_pnl_R": best_sum.to_dict(orient="records")[0],
                }

        # Sizing curve (ALL scope, based on calibrated score)
        sc = sizing_curve(
            out_df,
            score_col=f"{model_col}_cal",
            n_bins=int(args.sizing_bins),
            min_bin_n=int(args.sizing_min_bin_n),
            min_mult=float(args.min_mult),
            max_mult=float(args.max_mult),
        )
        sc_path = os.path.join(args.outdir, f"sizing_curve_{model_col}.csv")
        sc.to_csv(sc_path, index=False)

        # Store metrics json
        metrics = {
            "model_col": model_col,
            "methods_test_metrics": method_metrics,
            "chosen_method": best_method,
            "calibrator_path": os.path.basename(cal_path),
            "oof_calibrated_path": os.path.basename(oof_path),
            "calibration_bins_path": os.path.basename(bins_path),
            "ev_thresholds_path": os.path.basename(ev_path),
            "sizing_curve_path": os.path.basename(sc_path),
            "best_thresholds": best,
            # Global (ALL rows) metrics for raw vs calibrated (informational)
            "global_raw": {
                "auc": _safe_auc(y_all, _clip01(p_all)),
                "avg_precision": _safe_ap(y_all, _clip01(p_all)),
                "logloss": _safe_logloss(y_all, _clip01(p_all)),
                "brier": _safe_brier(y_all, _clip01(p_all)),
                "ece": _ece(y_all, _clip01(p_all), n_bins=int(args.n_calib_bins)),
            },
            "global_calibrated": {
                "auc": _safe_auc(y_all, p_cal),
                "avg_precision": _safe_ap(y_all, p_cal),
                "logloss": _safe_logloss(y_all, p_cal),
                "brier": _safe_brier(y_all, p_cal),
                "ece": _ece(y_all, p_cal, n_bins=int(args.n_calib_bins)),
            },
        }

        with open(os.path.join(args.outdir, f"calibration_metrics_{model_col}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)

        summary["models"][model_col] = metrics

        print(
            f"[05_calibration_ev] {model_col}: chosen={best_method} "
            f"holdout_logloss={method_metrics[best_method]['logloss']:.6f} "
            f"holdout_brier={method_metrics[best_method]['brier']:.6f}",
            flush=True,
        )

    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[05_calibration_ev] DONE. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
