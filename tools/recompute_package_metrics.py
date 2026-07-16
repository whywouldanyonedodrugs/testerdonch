#!/usr/bin/env python3
"""Recompute principal QLMG package metrics from standardized Parquet ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COST_COLUMNS = {
    "base": "net_base_R",
    "conservative": "net_conservative_R",
    "severe": "net_severe_R",
    "zero_funding": "net_zero_funding_base_R",
}


def profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else float("nan"))


def recompute(ledger: Path) -> pd.DataFrame:
    frame = pd.read_parquet(ledger)
    if "definition_id" not in frame:
        frame["definition_id"] = "unknown_definition"
    rows: list[dict[str, object]] = []
    for mode, column in COST_COLUMNS.items():
        if column not in frame:
            continue
        group_columns = ["definition_id"] + (["source_root"] if "source_root" in frame else [])
        for group_key, part in frame.groupby(group_columns, dropna=False, sort=True):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            definition_id = group_key[0]
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "definition_id": definition_id,
                    "source_root": group_key[1] if len(group_key) > 1 else "",
                    "cost_mode": mode,
                    "events": int(len(values)),
                    "mean_R": float(values.mean()),
                    "median_R": float(values.median()),
                    "total_R": float(values.sum()),
                    "profit_factor": profit_factor(values),
                    "hit_rate": float((values > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    outputs: list[pd.DataFrame] = []
    for ledger in sorted(package_root.glob("families/*/candidate_event_ledger.parquet")):
        result = recompute(ledger)
        if not result.empty:
            result.insert(0, "family_id", ledger.parent.name)
            outputs.append(result)
    combined = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    output = args.output or package_root / "engineering" / "independent_recomputed_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(json.dumps({"families": int(combined.family_id.nunique()) if not combined.empty else 0, "rows": len(combined), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
