#!/usr/bin/env python
"""
Join core unmatched LIVE analysis with backtester decision log.

Inputs (in cfg.RESULTS_DIR):
    - core_unmatched_live_analysis.csv  (from analyze_core_unmatched_live.py)
    - signal_decisions.csv              (from backtester decision logging)

Output:
    - core_unmatched_live_with_bt_decisions.csv

The output keeps all columns from the core analysis and adds:

    bt_decision, bt_reason, bt_ts_effective, bt_exit_ts, bt_pnl, bt_pnl_R, ...

for each core unmatched trade in the "signal_no_bt_trade" bucket.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

import config as cfg


def _results_path(name: str) -> Path:
    """Helper to build paths under cfg.RESULTS_DIR."""
    root = Path(getattr(cfg, "RESULTS_DIR", "results"))
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def main() -> None:
    core_path = _results_path("parity_unmatched_live_core.csv")
    dec_path = _results_path("signal_decisions.csv")

    if not core_path.exists():
        raise FileNotFoundError(
            f"{core_path} not found. Run analyze_core_unmatched_live.py first."
        )
    if not dec_path.exists():
        raise FileNotFoundError(
            f"{dec_path} not found. Make sure the backtester is run with decision logging enabled."
        )

    print(f"[info] loading core unmatched analysis from {core_path}")
    core = pd.read_csv(core_path)

    # Normalise symbol & parse timestamps
    if "symbol" not in core.columns:
        raise ValueError("Expected 'symbol' column in core_unmatched_live_analysis.csv")
    core["symbol"] = core["symbol"].astype(str).str.upper()

    for col in ["signal_ts", "exit_ts_live"]:
        if col in core.columns:
            core[col] = pd.to_datetime(core[col], utc=True, errors="coerce")

    # Focus on the bucket we care about
    if "classification" not in core.columns:
        raise ValueError("Expected 'classification' column in core_unmatched_live_analysis.csv")

    core_sig = core[core["classification"] == "signal_no_bt_trade"].copy()
    core_sig = core_sig.dropna(subset=["signal_ts"])
    core_sig = core_sig.reset_index(drop=True)

    print(f"[info] core rows total: {len(core)}")
    print(f"[info] core rows in 'signal_no_bt_trade' bucket: {len(core_sig)}")

    print(f"[info] loading backtester decisions from {dec_path}")
    dec = pd.read_csv(dec_path)

    # Expect at least these columns
    required_cols: List[str] = ["symbol", "signal_ts", "decision", "reason", "ts_effective"]
    missing = [c for c in required_cols if c not in dec.columns]
    if missing:
        raise ValueError(
            "signal_decisions.csv is missing expected columns: "
            + ", ".join(missing)
        )

    dec["symbol"] = dec["symbol"].astype(str).str.upper()

    # Parse datetime columns if present
    for col in ["signal_ts", "ts_effective", "exit_ts", "lock_until"]:
        if col in dec.columns:
            dec[col] = pd.to_datetime(dec[col], utc=True, errors="coerce")

    # We will keep symbol/signal_ts as join keys and prefix the rest with "bt_"
    join_keys = ["symbol", "signal_ts"]
    rename_map = {}
    for c in dec.columns:
        if c in join_keys:
            continue
        rename_map[c] = f"bt_{c}"

    dec_renamed = dec.rename(columns=rename_map)

    print("[info] joining on ['symbol', 'signal_ts'] ...")
    merged = core_sig.merge(dec_renamed, on=join_keys, how="left", validate="m:1")

    # Simple summary of reasons
    if "bt_reason" in merged.columns:
        print("\n=== Backtester reason counts for 'signal_no_bt_trade' ===")
        print(merged["bt_reason"].value_counts(dropna=False))
    else:
        print("[warn] 'bt_reason' column missing after merge; check signal_decisions.csv format.")

    out_path = _results_path("core_unmatched_live_with_bt_decisions.csv")
    merged.to_csv(out_path, index=False)
    print(f"\n[done] wrote joined file to: {out_path}")


if __name__ == "__main__":
    main()
