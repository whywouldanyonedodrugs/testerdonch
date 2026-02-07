#!/usr/bin/env python
"""
analyze_throughput_parity.py

Quick-and-dirty diagnostics for backtester gating:

- Summarises how many signals were:
  - taken
  - skipped, and why (reason)
- Splits `lock` into:
  - dedup_entry
  - cooldown
  - other lock-related reasons (if any)
- Summarises lock timeline: how long locks last, and how often we skip inside them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

import config as cfg


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[warn] File not found: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        print(f"[warn] File is empty: {path}")
        return None
    return df


def analyze_signal_decisions(results_dir: Path) -> None:
    dec_path = results_dir / "signal_decisions.csv"
    df = _load_csv(dec_path)
    if df is None:
        return

    # Basic sanity
    expected_cols = {"symbol", "signal_ts", "ts_effective", "decision", "reason"}
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"[warn] signal_decisions.csv missing columns: {missing}")

    print("\n=== Signal decisions: high-level ===")
    print(f"Total rows: {len(df)}")
    print("\nCount by decision:")
    print(df["decision"].value_counts(dropna=False))

    if "reason" in df.columns:
        print("\nCount by (decision, reason):")
        print(df.groupby(["decision", "reason"]).size().sort_values(ascending=False))

    # Focus on skipped by lock-related reasons
    if "reason" in df.columns:
        lock_reasons = ["dedup_entry", "cooldown", "max_open_positions", "daycap"]
        print("\n=== Lock-related skip reasons ===")
        mask = df["reason"].isin(lock_reasons)
        if mask.any():
            print(df[mask]["reason"].value_counts())
        else:
            print("(no lock-related reasons found)")

        # Dedup stats
        if "dedup_entry" in df["reason"].values:
            ddf = df[df["reason"] == "dedup_entry"].copy()
            # hours_since is logged in extra if backtester is unmodified
            if "hours_since" in ddf.columns:
                print("\nDedup: distribution of hours_since last_entry (in hours):")
                print(ddf["hours_since"].describe())
            print(f"\nNumber of dedup_entry skips: {len(ddf)}")

        # Cooldown stats
        if "cooldown" in df["reason"].values:
            cdf = df[df["reason"] == "cooldown"].copy()
            print(f"\nNumber of cooldown skips: {len(cdf)}")
            # If lock_until is in extra, we can compute remaining lock length at skip time
            if {"signal_ts", "ts_effective", "lock_until"}.issubset(cdf.columns):
                # Parse datetimes
                for col in ["signal_ts", "ts_effective", "lock_until"]:
                    cdf[col] = pd.to_datetime(cdf[col], errors="coerce", utc=True)
                # Effective ts is what backtester actually used for decision
                dt = (cdf["lock_until"] - cdf["ts_effective"]).dt.total_seconds() / 3600.0
                dt = dt[dt.notna()]
                if not dt.empty:
                    print("\nCooldown: hours remaining in lock at time of skipped signal:")
                    print(dt.describe())


def analyze_lock_timeline(results_dir: Path) -> None:
    lock_path = results_dir / "lock_timeline.csv"
    lf = _load_csv(lock_path)
    if lf is None:
        return

    expected_cols = {"symbol", "event", "ts", "lock_until", "reason"}
    missing = expected_cols - set(lf.columns)
    if missing:
        print(f"[warn] lock_timeline.csv missing columns: {missing}")

    print("\n=== Lock timeline ===")
    print(f"Total rows: {len(lf)}")

    print("\nEvents by type:")
    print(lf["event"].value_counts())

    if "reason" in lf.columns:
        print("\nEvents by (event, reason):")
        print(lf.groupby(["event", "reason"]).size().sort_values(ascending=False))

    # Rough lock duration stats per update
    if {"event", "ts", "lock_until"}.issubset(lf.columns):
        upd = lf[lf["event"] == "update"].copy()
        if not upd.empty:
            for col in ["ts", "lock_until"]:
                upd[col] = pd.to_datetime(upd[col], errors="coerce", utc=True)

            dur_h = (upd["lock_until"] - upd["ts"]).dt.total_seconds() / 3600.0
            dur_h = dur_h[dur_h.notna()]
            if not dur_h.empty:
                print("\nCooldown duration per update (hours):")
                print(dur_h.describe())


def main() -> None:
    results_dir = Path(getattr(cfg, "RESULTS_DIR", "results"))
    print(f"[info] Using RESULTS_DIR={results_dir}")

    analyze_signal_decisions(results_dir)
    analyze_lock_timeline(results_dir)


if __name__ == "__main__":
    main()
