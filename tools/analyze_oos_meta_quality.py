#!/usr/bin/env python3
"""
Analyze meta-model quality on an OOS trades export.

Inputs:
  - trades parquet/csv with at least: meta_p, pnl_R, entry_ts

Outputs:
  - JSON summary metrics
  - CSV bin table (probability bins)
  - CSV monthly threshold table
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score

        if len(np.unique(y)) < 2:
            return None
        return float(roc_auc_score(y, p))
    except Exception:
        return None


def _safe_ap(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import average_precision_score

        if len(np.unique(y)) < 2:
            return None
        return float(average_precision_score(y, p))
    except Exception:
        return None


def _safe_brier(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import brier_score_loss

        return float(brier_score_loss(y, p))
    except Exception:
        return None


def _safe_logloss(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import log_loss

        return float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))
    except Exception:
        return None


def _ece_10(y: np.ndarray, p: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.digitize(p, bins, right=True) - 1
    idx = np.clip(idx, 0, 9)
    n = len(y)
    if n == 0:
        return float("nan")
    ece = 0.0
    for b in range(10):
        m = idx == b
        if not m.any():
            continue
        conf = float(np.mean(p[m]))
        acc = float(np.mean(y[m]))
        ece += abs(conf - acc) * (float(np.sum(m)) / float(n))
    return float(ece)


def _metrics_for_target(
    y: np.ndarray, p: np.ndarray, base_rate: float
) -> Dict[str, Optional[float]]:
    p_base = np.full_like(p, base_rate, dtype=float)
    return {
        "base_rate": float(base_rate),
        "auc": _safe_auc(y, p),
        "avg_precision": _safe_ap(y, p),
        "brier_model": _safe_brier(y, p),
        "brier_baseline": _safe_brier(y, p_base),
        "logloss_model": _safe_logloss(y, p),
        "logloss_baseline": _safe_logloss(y, p_base),
        "ece_10": _ece_10(y, p),
    }


def _bin_table(df: pd.DataFrame) -> pd.DataFrame:
    q = min(10, int(df["meta_p"].nunique()))
    if q < 2:
        return pd.DataFrame(
            columns=["bin", "n", "p_mean", "win_rate", "good05_rate", "mean_pnl_R"]
        )
    d = df.copy()
    d["bin"] = pd.qcut(d["meta_p"], q=q, duplicates="drop")
    out = (
        d.groupby("bin", observed=True)
        .agg(
            n=("meta_p", "size"),
            p_mean=("meta_p", "mean"),
            win_rate=("pnl_R", lambda s: float((s > 0).mean())),
            good05_rate=("pnl_R", lambda s: float((s >= 0.5).mean())),
            mean_pnl_R=("pnl_R", "mean"),
        )
        .reset_index()
    )
    out["bin"] = out["bin"].astype(str)
    return out


def _monthly_threshold_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    d = df.copy()
    d["entry_ts"] = pd.to_datetime(d["entry_ts"], utc=True, errors="coerce")
    d = d[d["entry_ts"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["month", "is_above_threshold", "n", "win_rate", "mean_pnl_R"])
    d["month"] = d["entry_ts"].dt.to_period("M").astype(str)
    d["is_above_threshold"] = d["meta_p"] >= float(threshold)
    out = (
        d.groupby(["month", "is_above_threshold"], observed=True)
        .agg(
            n=("meta_p", "size"),
            win_rate=("pnl_R", lambda s: float((s > 0).mean())),
            mean_pnl_R=("pnl_R", "mean"),
        )
        .reset_index()
    )
    return out


def _bootstrap_auc_ci(
    y: np.ndarray, p: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> Dict[str, Optional[float]]:
    auc = _safe_auc(y, p)
    if auc is None:
        return {"auc": None, "auc_ci95_lo": None, "auc_ci95_hi": None}
    rng = np.random.default_rng(seed)
    vals: List[float] = []
    n = len(y)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if len(np.unique(ys)) < 2:
            continue
        ps = p[idx]
        a = _safe_auc(ys, ps)
        if a is not None:
            vals.append(float(a))
    if not vals:
        return {"auc": float(auc), "auc_ci95_lo": None, "auc_ci95_hi": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "auc": float(auc),
        "auc_ci95_lo": float(np.quantile(arr, 0.025)),
        "auc_ci95_hi": float(np.quantile(arr, 0.975)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze OOS meta probability quality.")
    ap.add_argument("--trades", required=True, help="Path to trades parquet/csv.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--threshold", type=float, default=0.42, help="Decision threshold to audit.")
    ap.add_argument("--boots", type=int, default=2000, help="Bootstrap samples for AUC CI.")
    args = ap.parse_args()

    trades_path = Path(args.trades).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = _read_table(trades_path)
    for c in ("meta_p", "pnl_R"):
        if c not in df.columns:
            raise SystemExit(f"Missing required column: {c}")
    if "entry_ts" not in df.columns:
        raise SystemExit("Missing required column: entry_ts")

    d = df.copy()
    d["meta_p"] = pd.to_numeric(d["meta_p"], errors="coerce")
    d["pnl_R"] = pd.to_numeric(d["pnl_R"], errors="coerce")
    if "risk_on" in d.columns:
        d["risk_on"] = pd.to_numeric(d["risk_on"], errors="coerce")
    if "size_mult" in d.columns:
        d["size_mult"] = pd.to_numeric(d["size_mult"], errors="coerce")

    d = d[np.isfinite(d["meta_p"]) & np.isfinite(d["pnl_R"])].copy()
    if d.empty:
        raise SystemExit("No valid rows after filtering finite meta_p and pnl_R.")

    p = np.clip(d["meta_p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    y_win = (d["pnl_R"].to_numpy(dtype=float) > 0.0).astype(int)
    y_good05 = (d["pnl_R"].to_numpy(dtype=float) >= 0.5).astype(int)
    y_good10 = (d["pnl_R"].to_numpy(dtype=float) >= 1.0).astype(int)

    summary: Dict[str, object] = {
        "trades_path": str(trades_path),
        "n": int(len(d)),
        "meta_p_min": float(np.min(p)),
        "meta_p_median": float(np.median(p)),
        "meta_p_mean": float(np.mean(p)),
        "meta_p_max": float(np.max(p)),
        "meta_p_unique": int(d["meta_p"].nunique()),
        "threshold_audit": float(args.threshold),
    }

    summary["y_win"] = _metrics_for_target(y_win, p, float(np.mean(y_win)))
    summary["y_good_05"] = _metrics_for_target(y_good05, p, float(np.mean(y_good05)))
    summary["y_good_10"] = _metrics_for_target(y_good10, p, float(np.mean(y_good10)))
    summary["y_win_auc_bootstrap"] = _bootstrap_auc_ci(y_win, p, n_boot=int(args.boots))

    hi = d[d["meta_p"] >= float(args.threshold)]
    lo = d[d["meta_p"] < float(args.threshold)]
    summary["threshold_split"] = {
        "n_high": int(len(hi)),
        "n_low": int(len(lo)),
        "win_rate_high": float((hi["pnl_R"] > 0).mean()) if len(hi) else None,
        "win_rate_low": float((lo["pnl_R"] > 0).mean()) if len(lo) else None,
        "mean_pnl_R_high": float(hi["pnl_R"].mean()) if len(hi) else None,
        "mean_pnl_R_low": float(lo["pnl_R"].mean()) if len(lo) else None,
    }

    if "size_mult" in d.columns:
        summary["size_mult_distribution"] = (
            d["size_mult"].value_counts(dropna=False).sort_index().to_dict()
        )
    if "risk_on" in d.columns:
        summary["risk_on_distribution"] = (
            d["risk_on"].value_counts(dropna=False).sort_index().to_dict()
        )

    bins_df = _bin_table(d)
    bins_csv = outdir / "probability_bins.csv"
    bins_df.to_csv(bins_csv, index=False)

    monthly_df = _monthly_threshold_table(d, threshold=float(args.threshold))
    monthly_csv = outdir / "monthly_threshold_split.csv"
    monthly_df.to_csv(monthly_csv, index=False)

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[oos-meta-quality] wrote {summary_path}")
    print(f"[oos-meta-quality] wrote {bins_csv}")
    print(f"[oos-meta-quality] wrote {monthly_csv}")


if __name__ == "__main__":
    main()

