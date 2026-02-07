#!/usr/bin/env python
"""
Quantify how much of the LIVE-vs-BT mismatch is caused by the 8h dedup window.

Inputs (in cfg.RESULTS_DIR):
    - core_unmatched_live_with_bt_decisions.csv

Assumptions:
    - Created by join_core_with_bt_decisions.py
    - Dedup decisions have bt_reason == "dedup_entry"
    - join_core... prefixed all decision-log columns with "bt_",
      so 'hours_since' is now 'bt_hours_since'.
"""

from pathlib import Path
import pandas as pd
import config as cfg


def main():
    results_dir = cfg.RESULTS_DIR
    path = results_dir / "core_unmatched_live_with_bt_decisions.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run join_core_with_bt_decisions.py first."
        )

    print(f"[info] loading {path}")
    df = pd.read_csv(path)

    # Sanity: we only care about signal_no_bt_trade bucket
    if "classification" in df.columns:
        df = df[df["classification"] == "signal_no_bt_trade"].copy()

    total_unmatched = len(df)
    print(f"[info] total unmatched LIVE trades (signal_no_bt_trade): {total_unmatched}")

    if "bt_reason" not in df.columns:
        raise ValueError("bt_reason column missing – check join_core_with_bt_decisions.py output.")

    reason_counts = df["bt_reason"].value_counts(dropna=False)
    print("\n=== bt_reason counts for unmatched LIVE trades ===")
    print(reason_counts)

    # Focus on dedup_entry bucket
    dd = df[df["bt_reason"] == "dedup_entry"].copy()
    n_dd = len(dd)
    print(f"\n[info] unmatched LIVE trades blocked by dedup_entry (8h): {n_dd} "
          f"({n_dd / total_unmatched:.2%} of unmatched, if total>0)" if total_unmatched else "")

    if n_dd == 0:
        print("[info] no dedup_entry in unmatched bucket; dedup mismatch is not the main driver.")
        return

    # We need bt_hours_since from signal_decisions extra payload
    if "bt_hours_since" not in dd.columns:
        print("[warn] bt_hours_since not present; "
              "backtester decision logging may not include 'hours_since' in extra.")
        print("       Re-run backtest with current backtester.py and join_core_with_bt_decisions.py.")
        return

    # Approximate effect of switching to 2h dedup_window
    dd["skip_8h"] = dd["bt_hours_since"] < 8.0
    dd["skip_2h"] = dd["bt_hours_since"] < 2.0

    # These are trades that are skipped under 8h but would NOT be skipped under 2h
    unlocked = dd[(dd["skip_8h"]) & (~dd["skip_2h"])]

    n_unlocked = len(unlocked)
    print("\n=== Counterfactual: DEDUP_WINDOW_HOURS = 2 instead of 8 ===")
    print(f"Trades currently blocked by dedup_entry that would be ALLOWED with 2h window: {n_unlocked}")
    print(f"Share of dedup_entry-unmatched: {n_unlocked / n_dd:.2%}")
    print(f"Share of all unmatched LIVE trades: {n_unlocked / total_unmatched:.2%}")

    # Simple distribution summary
    print("\n[info] bt_hours_since distribution for dedup_entry in unmatched LIVE trades:")
    print(dd["bt_hours_since"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))

    print("\n[done] dedup impact analysis complete.")


if __name__ == "__main__":
    main()
