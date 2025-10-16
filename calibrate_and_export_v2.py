# calibrate_and_export_v2.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from calibration_helpers import (
    temperature_scale, platt_scale, isotonic_scale
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--pred-ts", default="entry_ts")
    ap.add_argument("--pred-sym", default="symbol")
    ap.add_argument("--trades-ts", default="entry_ts")
    ap.add_argument("--trades-sym", default="symbol")
    ap.add_argument("--min-trades", type=int, default=150)

    # Guardrails: keep calibrator only if it improves proper scores AND preserves sharpness
    ap.add_argument("--min-ece-gain", type=float, default=0.005)
    ap.add_argument("--min-brier-gain", type=float, default=0.002)
    ap.add_argument("--min-std-ratio", type=float, default=0.50)
    ap.add_argument("--min-unique-ratio", type=float, default=0.50)
    ap.add_argument("--n-bins-ece", type=int, default=15)
    args = ap.parse_args()

    oos = pd.read_parquet(args.oos)
    assert {"y_true","y_proba"}.issubset(oos.columns), "Expected y_true,y_proba in OOS file"

    y = oos["y_true"].astype(int).to_numpy()
    p = oos["y_proba"].to_numpy().astype(float)

    # Baseline (raw)
    def metrics(p_cal):
        from calibration_helpers import brier_score, ece_score
        ece_raw  = ece_score(y, p, args.n_bins_ece)
        ece_cal  = ece_score(y, p_cal, args.n_bins_ece)
        brier_raw = brier_score(y, p)
        brier_cal = brier_score(y, p_cal)
        std_ratio = float(np.std(p_cal) / (np.std(p) + 1e-12))
        unique_ratio = float(len(np.unique(np.round(p_cal, 10))) / (len(np.unique(np.round(p, 10))) + 1e-12))
        return dict(ece_raw=ece_raw, ece_cal=ece_cal, ece_gain=ece_raw-ece_cal,
                    brier_raw=brier_raw, brier_cal=brier_cal, brier_gain=brier_raw-brier_cal,
                    std_ratio=std_ratio, unique_ratio=unique_ratio)

    results = {}
    results["raw"] = dict(kind="raw", ok=True, p_cal=p, metrics=metrics(p))

    # Try TS → Platt → Isotonic (prefer gentle methods first)
    for kind, fn in [("temperature", temperature_scale), ("platt", platt_scale), ("isotonic", isotonic_scale)]:
        r = fn(y, p)
        if not r.ok:
            results[kind] = dict(kind=kind, ok=False, p_cal=p, metrics={}, error=r.error)
        else:
            results[kind] = dict(kind=kind, ok=True, p_cal=r.p_cal, metrics=r.metrics, error=r.error)

    def ok(m):  # pass guardrails
        return (
            (m["ece_gain"] >= args.min_ece_gain) or (m["brier_gain"] >= args.min_brier_gain)
        ) and (m["std_ratio"] >= args.min_std_ratio) and (m["unique_ratio"] >= args.min_unique_ratio)

    picked = "raw"; best = 0.0
    for k in ("temperature","platt","isotonic"):
        r = results[k]
        if r["ok"] and r["metrics"]:
            if ok(r["metrics"]):
                gain = r["metrics"]["ece_gain"] + r["metrics"]["brier_gain"]
                if gain > best:
                    best = gain; picked = k

    # Persist calibrated preds
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    oos2 = oos.copy(); oos2["y_proba_cal"] = results[picked]["p_cal"]
    oos2.to_parquet(out / "oos_predictions_calibrated.parquet", index=False)

    # Save report
    report = {
        "picked": picked,
        "rule": dict(
            min_ece_gain=args.min_ece_gain, min_brier_gain=args.min_brier_gain,
            min_std_ratio=args.min_std_ratio, min_unique_ratio=args.min_unique_ratio,
            n_bins_ece=args.n_bins_ece
        ),
        "metrics": {k: dict(**v["metrics"], error=v.get("error")) for k,v in results.items()}
    }
    (out/"calibration_report.json").write_text(json.dumps(report, indent=2))
    print("[calibration] picked:", picked)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
