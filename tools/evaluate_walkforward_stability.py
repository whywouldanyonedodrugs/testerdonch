#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GateSpec:
    name: str
    metric: str
    op: str
    threshold: float
    severity: str  # hard | soft
    description: str


def _resolve(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p.resolve()


def _latest_run(root: Path) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No walkforward runs under {root}")
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[0]


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _gate_pass(op: str, value: float, threshold: float) -> bool:
    if not np.isfinite(value):
        return False
    if op == ">=":
        return bool(value >= threshold)
    if op == "<=":
        return bool(value <= threshold)
    raise ValueError(f"Unsupported op: {op}")


def _default_gates() -> List[GateSpec]:
    return [
        GateSpec("min_windows_ok", "windows_ok", ">=", 12.0, "hard", "Require enough independent validation windows."),
        GateSpec("min_months_covered", "months_covered", ">=", 12.0, "hard", "Require at least 12 monthly OOS points."),
        GateSpec("median_calibration_gap", "monthly_cal_gap_median", "<=", 0.08, "hard", "Median |p-win_rate| should stay controlled."),
        GateSpec("p90_calibration_gap", "monthly_cal_gap_p90", "<=", 0.15, "hard", "Tail calibration error must remain bounded."),
        GateSpec("median_top_decile_lift", "monthly_top_decile_lift_median", ">=", 1.20, "hard", "Ranking edge should be material."),
        GateSpec("p25_top_decile_lift", "monthly_top_decile_lift_p25", ">=", 1.00, "hard", "Bottom quartile month should not invert ranking edge."),
        GateSpec("positive_month_ratio", "monthly_positive_pnl_ratio", ">=", 0.45, "soft", "Avoid too many negative months."),
        GateSpec("worst_month_mean_pnl_R", "monthly_mean_pnl_min", ">=", -0.75, "soft", "Worst month should not be catastrophically negative."),
        GateSpec("worst_window_mean_pnl_R", "window_mean_pnl_min", ">=", -0.50, "soft", "Worst window should stay within risk tolerance."),
        GateSpec("median_prob_vs_pnl_spearman", "window_spearman_median", ">=", 0.05, "soft", "Predictions should remain directionally informative."),
    ]


def _compute_metrics(window_df: pd.DataFrame, monthly_df: pd.DataFrame) -> Dict[str, float]:
    m: Dict[str, float] = {}
    w_ok = window_df[window_df.get("status", "").astype(str) == "ok"].copy() if not window_df.empty else window_df
    months = monthly_df.copy()

    m["windows_total"] = float(len(window_df))
    m["windows_ok"] = float(len(w_ok))
    m["months_covered"] = float(len(months))

    if not months.empty:
        cal = pd.to_numeric(months.get("calibration_abs_gap"), errors="coerce")
        lift = pd.to_numeric(months.get("top_decile_win_lift"), errors="coerce")
        pnl = pd.to_numeric(months.get("mean_pnl_R"), errors="coerce")

        m["monthly_cal_gap_median"] = _safe_float(np.nanmedian(cal))
        m["monthly_cal_gap_p90"] = _safe_float(np.nanquantile(cal, 0.9))
        m["monthly_top_decile_lift_median"] = _safe_float(np.nanmedian(lift))
        m["monthly_top_decile_lift_p25"] = _safe_float(np.nanquantile(lift, 0.25))
        m["monthly_mean_pnl_median"] = _safe_float(np.nanmedian(pnl))
        m["monthly_mean_pnl_min"] = _safe_float(np.nanmin(pnl))
        m["monthly_positive_pnl_ratio"] = _safe_float(np.nanmean(pnl > 0.0))
    else:
        for k in [
            "monthly_cal_gap_median",
            "monthly_cal_gap_p90",
            "monthly_top_decile_lift_median",
            "monthly_top_decile_lift_p25",
            "monthly_mean_pnl_median",
            "monthly_mean_pnl_min",
            "monthly_positive_pnl_ratio",
        ]:
            m[k] = float("nan")

    if not w_ok.empty:
        wpnl = pd.to_numeric(w_ok.get("mean_pnl_R_val"), errors="coerce")
        wlft = pd.to_numeric(w_ok.get("top_decile_win_lift_val"), errors="coerce")
        wsp = pd.to_numeric(w_ok.get("spearman_prob_vs_pnl_val"), errors="coerce")
        m["window_mean_pnl_median"] = _safe_float(np.nanmedian(wpnl))
        m["window_mean_pnl_min"] = _safe_float(np.nanmin(wpnl))
        m["window_top_decile_lift_median"] = _safe_float(np.nanmedian(wlft))
        m["window_spearman_median"] = _safe_float(np.nanmedian(wsp))
    else:
        m["window_mean_pnl_median"] = float("nan")
        m["window_mean_pnl_min"] = float("nan")
        m["window_top_decile_lift_median"] = float("nan")
        m["window_spearman_median"] = float("nan")

    return m


def _evaluate_gates(metrics: Dict[str, float], gates: List[GateSpec]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for g in gates:
        v = _safe_float(metrics.get(g.metric, float("nan")))
        ok = _gate_pass(g.op, v, g.threshold)
        out.append(
            {
                "name": g.name,
                "metric": g.metric,
                "op": g.op,
                "threshold": g.threshold,
                "value": None if not np.isfinite(v) else float(v),
                "pass": bool(ok),
                "severity": g.severity,
                "description": g.description,
            }
        )
    return out


def _verdict(gates_eval: List[Dict[str, object]]) -> Tuple[str, str, List[str]]:
    hard_fail = [g for g in gates_eval if (g["severity"] == "hard" and not g["pass"])]
    soft_fail = [g for g in gates_eval if (g["severity"] == "soft" and not g["pass"])]
    failed_names = [str(g["name"]) for g in hard_fail + soft_fail]

    if hard_fail:
        return "fail", "probe_only_or_no_go", failed_names
    if len(soft_fail) >= 2:
        return "fail", "probe_only_or_no_go", failed_names
    if len(soft_fail) == 1:
        return "caution", "reduced_size_canary", failed_names
    return "pass", "full_size_phased", []


def _fmt(v: object, nd: int = 4) -> str:
    try:
        f = float(v)
    except Exception:
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.{nd}f}"


def _write_exec_md(
    out_path: Path,
    run_id: str,
    verdict: str,
    risk_mode: str,
    failed: List[str],
    metrics: Dict[str, float],
    gates_eval: List[Dict[str, object]],
) -> None:
    lines: List[str] = []
    lines.append("# Stability Executive Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Run ID: `{run_id}`")
    lines.append("")
    lines.append(f"- Verdict: `{verdict.upper()}`")
    lines.append(f"- Recommended live-risk mode: `{risk_mode}`")
    lines.append(f"- Failed gates: `{', '.join(failed) if failed else 'none'}`")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    key_order = [
        "windows_ok",
        "months_covered",
        "monthly_cal_gap_median",
        "monthly_cal_gap_p90",
        "monthly_top_decile_lift_median",
        "monthly_top_decile_lift_p25",
        "monthly_positive_pnl_ratio",
        "monthly_mean_pnl_min",
        "window_mean_pnl_min",
        "window_spearman_median",
    ]
    for k in key_order:
        lines.append(f"- `{k}`: `{_fmt(metrics.get(k), 6)}`")
    lines.append("")
    lines.append("## Gate Results")
    lines.append("")
    gdf = pd.DataFrame(gates_eval)
    if not gdf.empty:
        lines.append(gdf.to_markdown(index=False))
    else:
        lines.append("No gates.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_exec_html(
    out_path: Path,
    run_id: str,
    verdict: str,
    risk_mode: str,
    failed: List[str],
    metrics: Dict[str, float],
    gates_eval: List[Dict[str, object]],
) -> None:
    gdf = pd.DataFrame(gates_eval)
    g_html = gdf.to_html(index=False, border=0) if not gdf.empty else "<p><em>No gates</em></p>"

    key_rows = []
    for k, v in metrics.items():
        key_rows.append(f"<tr><th>{k}</th><td>{_fmt(v, 6)}</td></tr>")
    key_tbl = "\n".join(key_rows)
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stability Executive Summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; text-align:left; }}
th {{ background: #f4f4f4; }}
.tag {{ display:inline-block; padding:4px 8px; border-radius:4px; font-weight:bold; }}
.pass {{ background:#dff0d8; }}
.caution {{ background:#fcf8e3; }}
.fail {{ background:#f2dede; }}
</style>
</head><body>
<h1>Stability Executive Summary</h1>
<p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
<p>Run ID: <code>{run_id}</code></p>
<p>Verdict: <span class="tag {verdict}">{verdict.upper()}</span></p>
<p>Recommended live-risk mode: <code>{risk_mode}</code></p>
<p>Failed gates: <code>{', '.join(failed) if failed else 'none'}</code></p>

<h2>Key Metrics</h2>
<table>{key_tbl}</table>

<h2>Gate Results</h2>
{g_html}
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JT-011: fixed-gate overfitting/stability evaluator for walk-forward outputs.")
    p.add_argument("--root", default="results/walkforward_oos")
    p.add_argument("--run-id", default="", help="If empty, evaluates latest run.")
    p.add_argument("--out-name", default="stability_verdict.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = _resolve(args.root)
    run_dir = (root / args.run_id).resolve() if args.run_id.strip() else _latest_run(root)
    agg = run_dir / "aggregate"

    w_path = agg / "window_metrics.csv"
    m_path = agg / "monthly_oos_stability.csv"
    if not w_path.exists() or not m_path.exists():
        raise SystemExit(f"Missing aggregate inputs under {agg}")

    wdf = pd.read_csv(w_path)
    mdf = pd.read_csv(m_path)
    metrics = _compute_metrics(wdf, mdf)
    gates = _default_gates()
    gates_eval = _evaluate_gates(metrics, gates)
    verdict, risk_mode, failed = _verdict(gates_eval)

    report = {
        "run_id": run_dir.name,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdict": verdict,
        "recommended_live_risk_mode": risk_mode,
        "failed_gates": failed,
        "metrics": metrics,
        "gates": gates_eval,
        "inputs": {
            "window_metrics_csv": str(w_path),
            "monthly_stability_csv": str(m_path),
        },
    }

    out_json = agg / args.out_name
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    _write_exec_md(
        out_path=agg / "stability_executive_summary.md",
        run_id=run_dir.name,
        verdict=verdict,
        risk_mode=risk_mode,
        failed=failed,
        metrics=metrics,
        gates_eval=gates_eval,
    )
    _write_exec_html(
        out_path=agg / "stability_executive_summary.html",
        run_id=run_dir.name,
        verdict=verdict,
        risk_mode=risk_mode,
        failed=failed,
        metrics=metrics,
        gates_eval=gates_eval,
    )

    print(f"[jt011] run_id={run_dir.name} verdict={verdict} risk_mode={risk_mode} failed={len(failed)}", flush=True)
    print(f"[jt011] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

