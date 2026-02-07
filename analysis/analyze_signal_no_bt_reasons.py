#!/usr/bin/env python
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")

def main():
    core_path = RESULTS_DIR / "core_unmatched_live_analysis.csv"
    dec_path  = RESULTS_DIR / "signal_decisions.csv"

    print(f"[info] Loading core unmatched LIVE from: {core_path}")
    print(f"[info] Loading signal decisions from: {dec_path}")

    core = pd.read_csv(core_path)
    dec  = pd.read_csv(dec_path)

    # ---- Basic sanity ----
    if "classification" not in core.columns:
        raise RuntimeError("core_unmatched_live_analysis.csv is missing 'classification' column")

    if "symbol" not in core.columns or "symbol" not in dec.columns:
        raise RuntimeError("Both core and decisions must have a 'symbol' column")

    # Detect live trade timestamp column
    time_candidates = [
        "entry_ts_live",
        "entry_live_ts",
        "exit_ts_live",
        "exit_live_ts",
    ]
    live_ts_col = None
    for c in time_candidates:
        if c in core.columns:
            live_ts_col = c
            break
    if live_ts_col is None:
        raise RuntimeError(
            "Could not find a live timestamp column in core file. "
            "Expected one of: " + ", ".join(time_candidates)
        )

    print(f"[info] Using '{live_ts_col}' as live trade timestamp column")

    # Parse timestamps
    core[live_ts_col] = pd.to_datetime(core[live_ts_col], utc=True, errors="coerce")
    if "signal_ts" not in dec.columns:
        raise RuntimeError("signal_decisions.csv is missing 'signal_ts' column")
    dec["signal_ts"] = pd.to_datetime(dec["signal_ts"], utc=True, errors="coerce")

    # Filter to the bucket we care about
    core_sig_no_trade = core[core["classification"] == "signal_no_bt_trade"].copy()
    print(f"[info] Core unmatched LIVE with classification='signal_no_bt_trade': {len(core_sig_no_trade)}")

    # Drop rows with missing timestamps on the left
    before = len(core_sig_no_trade)
    core_sig_no_trade = core_sig_no_trade.dropna(subset=[live_ts_col])
    after = len(core_sig_no_trade)
    if after < before:
        print(f"[warn] Dropped {before - after} rows with NaN {live_ts_col}")

    # Drop rows with missing signal_ts on the right
    before_dec = len(dec)
    dec = dec.dropna(subset=["signal_ts"])
    after_dec = len(dec)
    if after_dec < before_dec:
        print(f"[warn] Dropped {before_dec - after_dec} rows with NaN signal_ts in decisions")

    # --- CRITICAL: sort by time (primary) then symbol for merge_asof ---
    core_sig_no_trade = (
        core_sig_no_trade
        .sort_values([live_ts_col, "symbol"])
        .reset_index(drop=True)
    )

    dec_sorted = (
        dec
        .sort_values(["signal_ts", "symbol"])
        .reset_index(drop=True)
    )

    # (Optional) sanity assertions – will raise if something is still off
    if not core_sig_no_trade[live_ts_col].is_monotonic_increasing:
        raise RuntimeError(f"Left {live_ts_col} is not globally sorted after sort_values")
    if not dec_sorted["signal_ts"].is_monotonic_increasing:
        raise RuntimeError("Right signal_ts is not globally sorted after sort_values")

    # As-of join: nearest earlier signal per symbol within 12h
    tolerance = pd.Timedelta("12h")
    print(f"[info] Performing as-of join with tolerance {tolerance}")

    merged = pd.merge_asof(
        core_sig_no_trade,
        dec_sorted[["symbol", "signal_ts", "decision", "reason"]],
        left_on=live_ts_col,
        right_on="signal_ts",
        by="symbol",
        direction="backward",
        tolerance=tolerance,
    )

    # How many did we successfully attach a decision to?
    attached = merged["reason"].notna().sum()
    print(f"[info] Attached backtest decision to {attached} / {len(merged)} rows")

    # Basic reason breakdown
    print("\n=== Reason counts for signal_no_bt_trade (nearest earlier decision) ===")
    print(merged["reason"].value_counts(dropna=False))

    # Extra: distribution of time difference between live trade and attached signal
    if "signal_ts" in merged.columns:
        delta = (merged[live_ts_col] - merged["signal_ts"]).dt.total_seconds() / 3600.0
        merged["hours_since_signal"] = delta

        print("\n=== Time delta (live trade - signal_ts) in hours ===")
        print(delta.describe())

    # Save for manual inspection
    out_path = RESULTS_DIR / "signal_no_bt_trade_with_decisions.csv"
    merged.to_csv(out_path, index=False)
    print(f"[info] Saved merged details to: {out_path}")

if __name__ == "__main__":
    main()
