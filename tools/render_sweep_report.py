#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row.get(c, "")
            if pd.isna(v):
                v = ""
            vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--topn", type=int, default=15)
    ap.add_argument("--lam", type=float, default=2.0)
    ap.add_argument("--mu", type=float, default=1.0)
    a = ap.parse_args()

    root = Path(a.results_dir) / "policy_sweeps" / a.run_id
    root = root.resolve()
    if not root.exists():
        raise SystemExit(f"Run folder not found: {root}")

    ignore = {"_scoped_signals", "_smoke_run", "_smoke_signals"}
    rows: List[Dict[str, Any]] = []

    for d in sorted([p for p in root.iterdir() if p.is_dir() and p.name not in ignore]):
        met = d / "metrics.json"
        done = d / "_DONE.json"
        if met.exists():
            try:
                obj = json.loads(met.read_text(encoding="utf-8"))
            except Exception:
                obj = {"status": "bad_json"}
        else:
            obj = {"status": "missing_metrics"}

        obj.setdefault("setting", d.name)
        obj.setdefault("_has_done", done.exists())
        rows.append(obj)

    df = pd.DataFrame(rows)
    df.to_csv(root / "summary.csv", index=False)

    # compute utility if possible
    if all(c in df.columns for c in ["pnl_risk_on", "pnl_risk_off", "max_drawdown"]):
        pon = pd.to_numeric(df["pnl_risk_on"], errors="coerce")
        poff = pd.to_numeric(df["pnl_risk_off"], errors="coerce")
        mdd = pd.to_numeric(df["max_drawdown"], errors="coerce")
        df["utility"] = pon - a.lam * (poff.clip(upper=0).abs()) - a.mu * (mdd.clip(upper=0).abs())

    df_ok = df.copy()
    df_ok["utility"] = pd.to_numeric(df_ok.get("utility", np.nan), errors="coerce")
    df_ok = df_ok.sort_values("utility", ascending=False)

    top3 = df_ok.head(3)

    cols = [c for c in ["setting","utility","pnl_risk_on","pnl_risk_off","bad_regime_damage","max_drawdown","number_of_trades","win_rate","status","_has_done"] if c in df_ok.columns]
    show = df_ok[cols].head(a.topn) if cols else df_ok.head(a.topn)

    lines = []
    lines.append(f"# Policy sweep report: {a.run_id}")
    lines.append("")
    lines.append(f"Utility = PnL_risk_on - {a.lam:g}*abs(min(PnL_risk_off,0)) - {a.mu:g}*max_drawdown")
    lines.append("")
    lines.append("## Top 3")
    if len(top3) == 0:
        lines.append("- No metrics found.")
    else:
        for _, r in top3.iterrows():
            lines.append(
                f"- {r.get('setting')}: utility={r.get('utility')}, "
                f"pnl_on={r.get('pnl_risk_on')}, bad_damage={r.get('bad_regime_damage')}, "
                f"max_dd={r.get('max_drawdown')}"
            )

    lines.append("")
    lines.append(f"## Ranked (top {a.topn})")
    lines.append("")
    lines.append(markdown_table(show))

    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(str(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
