#!/usr/bin/env python3
"""
research/07_export_deployment_artifacts.py

Step 07: export deployment artifacts:
  - final trained model pipeline (joblib) from research_outputs/04_models_cv
  - deployment-safe calibration artifact (JSON for sigmoid; joblib for isotonic)
  - best threshold(s) and sizing curve from research_outputs/05_calibration_ev
  - deployment_config.json + sha256 checksums

Default behavior:
  - chooses best model_col among p_* using Step 05 summary.json chosen-method holdout logloss (lower is better)
  - uses Step 05 chosen calibration method for that model
  - threshold selection from ev_thresholds_<model_col>.csv, by --criterion (mean or sum pnl_R)

Notes:
  - This avoids pickling a custom Calibrator class for deployment compatibility.
  - Sigmoid calibration is exported as (a,b) for p_cal = sigmoid(a*logit(p_raw) + b).
  - Isotonic calibration is exported as sklearn IsotonicRegression joblib.

Outputs in outdir:
  - model.joblib
  - calibration.json  (+ isotonic.joblib if needed)
  - sizing_curve.csv
  - ev_thresholds.csv
  - thresholds.json
  - feature_manifest.json
  - deployment_config.json
  - checksums_sha256.json
  - optional bundle.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


EPS = 1e-12


# -----------------------------
# Helpers
# -----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clip01(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    return np.clip(p.astype(np.float64), eps, 1.0 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip01(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _copy(src: str, dst: str) -> None:
    shutil.copy2(src, dst)


def _model_name_from_prob_col(model_col: str) -> str:
    # Step 04 uses final_model_<name>.joblib; we used model keys "logreg" and "lgbm".
    if model_col == "p_lgbm":
        return "lgbm"
    if model_col == "p_logreg":
        return "logreg"
    # fallback: p_xxx -> xxx
    if model_col.startswith("p_") and len(model_col) > 2:
        return model_col[2:]
    return model_col


# -----------------------------
# Calibration export (deployment-safe)
# -----------------------------
@dataclass
class SigmoidParams:
    a: float
    b: float

    def apply(self, p_raw: np.ndarray) -> np.ndarray:
        z = _logit(_clip01(p_raw))
        return _clip01(_sigmoid(self.a * z + self.b))


def fit_sigmoid_params(y: np.ndarray, p_raw: np.ndarray) -> SigmoidParams:
    y = np.asarray(y, dtype=int)
    p_raw = _clip01(np.asarray(p_raw, dtype=np.float64))
    z = _logit(p_raw).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=4000)
    lr.fit(z, y)
    a = float(lr.coef_.ravel()[0])
    b = float(lr.intercept_.ravel()[0])
    return SigmoidParams(a=a, b=b)


def fit_isotonic(y: np.ndarray, p_raw: np.ndarray) -> IsotonicRegression:
    y = np.asarray(y, dtype=int)
    p_raw = _clip01(np.asarray(p_raw, dtype=np.float64))
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_raw, y)
    return iso


# -----------------------------
# Threshold selection
# -----------------------------
def select_thresholds(
    ev_csv_path: str,
    criterion: str,
) -> Dict[str, dict]:
    """
    Returns best thresholds per scope and also top-5 table per scope.
    """
    ev = pd.read_csv(ev_csv_path)
    if ev.empty:
        return {}

    if "scope" not in ev.columns or "threshold" not in ev.columns:
        raise RuntimeError(f"EV thresholds CSV missing required columns: {ev_csv_path}")

    if criterion not in ("mean", "sum"):
        raise ValueError("--criterion must be 'mean' or 'sum'")

    score_col = "mean_pnl_R" if criterion == "mean" else "sum_pnl_R"
    if score_col not in ev.columns:
        raise RuntimeError(f"EV thresholds CSV missing column '{score_col}': {ev_csv_path}")

    out: Dict[str, dict] = {}
    for scope, sub in ev.groupby("scope"):
        sub2 = sub.copy()
        sub2 = sub2[np.isfinite(pd.to_numeric(sub2[score_col], errors="coerce"))]
        if sub2.empty:
            continue
        sub2[score_col] = pd.to_numeric(sub2[score_col], errors="coerce")
        best = sub2.sort_values(score_col, ascending=False).head(1)
        top5 = sub2.sort_values(score_col, ascending=False).head(5)
        out[str(scope)] = {
            "best": best.to_dict(orient="records")[0],
            "top5": top5.to_dict(orient="records"),
        }
    return out


# -----------------------------
# Main export
# -----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 07: export deployment artifacts.")
    p.add_argument("--cvdir", type=str, default="research_outputs/04_models_cv")
    p.add_argument("--caldir", type=str, default="research_outputs/05_calibration_ev")
    p.add_argument("--oof", type=str, default="research_outputs/04_models_cv/oof_predictions.parquet")

    p.add_argument("--outdir", type=str, default="research_outputs/07_deployment_artifacts")

    p.add_argument("--target", type=str, default="y_good_05")
    p.add_argument("--model-col", type=str, default="", help="Probability column to export (e.g. p_lgbm). Default: auto-select best from caldir/summary.json")
    p.add_argument("--fit-scope", type=str, default="ALL", choices=["ALL", "RISK_ON_1"], help="Scope used to fit calibration params.")
    p.add_argument("--decision-scope", type=str, default="", help="Which scope to take threshold from (ALL or RISK_ON_1). Default: prefer RISK_ON_1 if present else ALL.")
    p.add_argument("--criterion", type=str, default="mean", choices=["mean", "sum"], help="Pick threshold by mean_pnl_R or sum_pnl_R.")

    p.add_argument("--bundle-tar", action="store_true", help="Also write bundle.tar.gz in outdir.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    _ensure_dir(args.outdir)

    # Required inputs
    summary_path = os.path.join(args.caldir, "summary.json")
    manifest_path = os.path.join(args.cvdir, "manifest.json")

    for path, name in [
        (summary_path, "Step05 summary.json"),
        (manifest_path, "Step04 manifest.json"),
        (args.oof, "OOF parquet"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    summary = _read_json(summary_path)
    manifest = _read_json(manifest_path)

    # Determine model_col
    model_col = args.model_col.strip()
    if not model_col:
        # pick best by chosen-method holdout logloss
        best = None
        best_key = None
        models = summary.get("models", {})
        if not isinstance(models, dict) or not models:
            raise RuntimeError("summary.json missing 'models' block; cannot auto-select model-col.")
        for k, v in models.items():
            # k is model_col like "p_lgbm"
            chosen = v.get("chosen_method")
            mt = v.get("methods_test_metrics", {}).get(chosen, {})
            ll = mt.get("logloss")
            if ll is None or not np.isfinite(float(ll)):
                continue
            if best is None or float(ll) < best:
                best = float(ll)
                best_key = k
        if best_key is None:
            raise RuntimeError("Could not auto-select a model (no finite holdout logloss). Specify --model-col.")
        model_col = str(best_key)

    if "models" not in summary or model_col not in summary["models"]:
        raise RuntimeError(f"Model column not found in Step05 summary.json: {model_col}")

    model_info = summary["models"][model_col]
    chosen_method = str(model_info.get("chosen_method"))
    if chosen_method not in ("none", "sigmoid", "isotonic"):
        raise RuntimeError(f"Unsupported chosen_method in summary for {model_col}: {chosen_method}")

    model_name = _model_name_from_prob_col(model_col)
    final_model_path = os.path.join(args.cvdir, f"final_model_{model_name}.joblib")
    if not os.path.exists(final_model_path):
        raise FileNotFoundError(f"Final model not found: {final_model_path}")

    # Copy model pipeline
    out_model_path = os.path.join(args.outdir, "model.joblib")
    _copy(final_model_path, out_model_path)

    # Copy sizing curve and EV tables from Step 05
    sizing_src = os.path.join(args.caldir, f"sizing_curve_{model_col}.csv")
    ev_src = os.path.join(args.caldir, f"ev_thresholds_{model_col}.csv")
    if not os.path.exists(sizing_src):
        raise FileNotFoundError(f"Sizing curve not found: {sizing_src}")
    if not os.path.exists(ev_src):
        raise FileNotFoundError(f"EV thresholds not found: {ev_src}")

    sizing_dst = os.path.join(args.outdir, "sizing_curve.csv")
    ev_dst = os.path.join(args.outdir, "ev_thresholds.csv")
    _copy(sizing_src, sizing_dst)
    _copy(ev_src, ev_dst)

    # Export feature manifest
    feat_manifest = {
        "target": args.target,
        "features": manifest.get("features", {}),
        "include_regimes_as_features": manifest.get("include_regimes_as_features", None),
    }
    feat_manifest_path = os.path.join(args.outdir, "feature_manifest.json")
    _write_json(feat_manifest_path, feat_manifest)

    # Fit deployment-safe calibrator using OOF + chosen method, on args.fit_scope
    df_oof = pd.read_parquet(args.oof)
    if "trade_id" not in df_oof.columns:
        df_oof = df_oof.reset_index().rename(columns={"index": "trade_id"})

    if args.target not in df_oof.columns:
        raise RuntimeError(f"OOF missing target: {args.target}")
    if model_col not in df_oof.columns:
        raise RuntimeError(f"OOF missing model column: {model_col}")

    df_oof[args.target] = _to_num(df_oof[args.target])
    df_oof[model_col] = _to_num(df_oof[model_col])
    for _rc in ("risk_on_1", "risk_on"):
        if _rc in df_oof.columns:
            df_oof[_rc] = _to_num(df_oof[_rc])
    df_oof = df_oof[df_oof[args.target].notna() & df_oof[model_col].notna()].copy()
    if df_oof.empty:
        raise RuntimeError("No valid rows in OOF after filtering numeric target and model_col.")

    fit_df = df_oof
    if args.fit_scope == "RISK_ON_1":
        # Prioritize risk_on_1, fallback to risk_on
        risk_col = "risk_on_1" if "risk_on_1" in df_oof.columns else "risk_on"
        if risk_col not in df_oof.columns:
            raise RuntimeError("fit-scope=RISK_ON_1 requested but OOF has no risk_on_1 or risk_on.")
        fit_df = df_oof.loc[df_oof[risk_col].fillna(0) == 1].copy()
        if fit_df.empty:
            raise RuntimeError("fit-scope=RISK_ON_1 produced 0 rows.")

    y_fit = fit_df[args.target].astype(int).to_numpy()
    p_fit = _clip01(fit_df[model_col].to_numpy(dtype=np.float64))

    calibration_json: Dict[str, object] = {
        "model_col": model_col,
        "chosen_method": chosen_method,
        "fit_scope": args.fit_scope,
        "formula": None,
        "params": {},
        "artifact": None,
    }

    iso_joblib_path = None
    if chosen_method == "none":
        calibration_json["formula"] = "p_cal = clip(p_raw)"
        calibration_json["params"] = {"eps": EPS}

    elif chosen_method == "sigmoid":
        sp = fit_sigmoid_params(y_fit, p_fit)
        calibration_json["formula"] = "p_cal = sigmoid(a*logit(clip(p_raw))+b)"
        calibration_json["params"] = {"a": sp.a, "b": sp.b, "eps": EPS}

    elif chosen_method == "isotonic":
        iso = fit_isotonic(y_fit, p_fit)
        iso_joblib_path = os.path.join(args.outdir, "isotonic.joblib")
        joblib.dump(iso, iso_joblib_path)
        calibration_json["formula"] = "p_cal = isotonic(clip(p_raw))"
        calibration_json["artifact"] = "isotonic.joblib"
        calibration_json["params"] = {"eps": EPS}

    calib_path = os.path.join(args.outdir, "calibration.json")
    _write_json(calib_path, calibration_json)

    # Threshold selection export
    thresholds = select_thresholds(ev_dst, criterion=args.criterion)
    if not thresholds:
        raise RuntimeError("No thresholds found/selected from EV table; check ev_thresholds.csv.")

    decision_scope = args.decision_scope.strip()
    if not decision_scope:
        decision_scope = "RISK_ON_1" if "RISK_ON_1" in thresholds else "ALL"
    if decision_scope not in thresholds:
        # fallback
        decision_scope = "ALL" if "ALL" in thresholds else list(thresholds.keys())[0]

    thresholds_out = {
        "criterion": args.criterion,
        "selected_scope": decision_scope,
        "thresholds_by_scope": thresholds,
    }
    thresholds_path = os.path.join(args.outdir, "thresholds.json")
    _write_json(thresholds_path, thresholds_out)

    # Deployment config
    config = {
        "version": "deploy_v1",
        "target": args.target,
        "probability_column": model_col,
        "model": {
            "name": model_name,
            "path": "model.joblib",
            "source": final_model_path,
        },
        "calibration": {
            "method": chosen_method,
            "path": "calibration.json",
            "isotonic_artifact": ("isotonic.joblib" if iso_joblib_path else None),
        },
        "decision": {
            "type": "threshold",
            "scope": decision_scope,
            "criterion": args.criterion,
            "threshold": float(thresholds[decision_scope]["best"]["threshold"]),
            "expected": thresholds[decision_scope]["best"],
        },
        "sizing": {
            "curve_csv": "sizing_curve.csv",
            "note": "risk_multiplier is looked up by calibrated probability bin; bins are in sizing_curve.csv",
        },
        "files": {
            "ev_thresholds_csv": "ev_thresholds.csv",
            "thresholds_json": "thresholds.json",
            "feature_manifest_json": "feature_manifest.json",
        },
    }
    config_path = os.path.join(args.outdir, "deployment_config.json")
    _write_json(config_path, config)

    # Checksums
    checksums: Dict[str, str] = {}
    for fn in os.listdir(args.outdir):
        p = os.path.join(args.outdir, fn)
        if os.path.isfile(p) and not fn.endswith(".tar.gz"):
            checksums[fn] = _sha256_file(p)
    checksums_path = os.path.join(args.outdir, "checksums_sha256.json")
    _write_json(checksums_path, checksums)

    # Optional tarball
    if args.bundle_tar:
        tar_path = os.path.join(args.outdir, "bundle.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for fn in sorted(os.listdir(args.outdir)):
                if fn == "bundle.tar.gz":
                    continue
                tar.add(os.path.join(args.outdir, fn), arcname=fn)

    print(
        f"[07_export] DONE. Exported model_col={model_col} model_name={model_name} "
        f"calib={chosen_method} threshold_scope={decision_scope} threshold={config['decision']['threshold']}",
        flush=True,
    )
    print(f"[07_export] Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()