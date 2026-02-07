#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

CORE_PATH = RESULTS_DIR / "core_unmatched_live_analysis.csv"
DECISIONS_PATH = RESULTS_DIR / "signal_decisions.csv"
UNMATCHED_LIVE_PATH = RESULTS_DIR / "parity_unmatched_live.csv"

TOLERANCE = pd.Timedelta("2h")  # adjust if needed


def to_dt_utc_series(s: pd.Series) -> pd.Series:
    # Always return tz-aware UTC, coercing bad values to NaT
    return pd.to_datetime(s, utc=True, errors="coerce")


def require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns {missing}. Have={list(df.columns)}")


def merge_asof_per_symbol(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    out = []

    for sym, lgrp in left.groupby("symbol", sort=False):
        lgrp = lgrp.copy()

        # Ensure left key is datetime
        lgrp["bar_ts"] = to_dt_utc_series(lgrp["bar_ts"])
        lgrp = lgrp.sort_values("bar_ts").reset_index(drop=True)

        rgrp = right[right["symbol"] == sym].copy()
        if rgrp.empty:
            # Make sure these exist with correct "time-ish" NaT placeholders
            if "signal_ts" not in lgrp.columns:
                lgrp["signal_ts"] = pd.NaT
            else:
                lgrp["signal_ts"] = to_dt_utc_series(lgrp["signal_ts"])

            for c in ["decision", "reason", "prob_val", "thr"]:
                if c not in lgrp.columns:
                    lgrp[c] = pd.NA

            if "lock_until" not in lgrp.columns:
                lgrp["lock_until"] = pd.NaT
            else:
                lgrp["lock_until"] = to_dt_utc_series(lgrp["lock_until"])

            out.append(lgrp)
            continue

        # Ensure right key is datetime
        rgrp["signal_ts"] = to_dt_utc_series(rgrp["signal_ts"])
        rgrp = rgrp.sort_values("signal_ts").reset_index(drop=True)

        merged = pd.merge_asof(
            lgrp,
            rgrp,
            left_on="bar_ts",
            right_on="signal_ts",
            direction="nearest",
            tolerance=TOLERANCE,
            suffixes=("", "_bt"),
        )
        out.append(merged)

    return pd.concat(out, ignore_index=True) if out else left.copy()


def main():
    print(f"[info] Loading core: {CORE_PATH}")
    core = pd.read_csv(CORE_PATH)

    print(f"[info] Loading decisions: {DECISIONS_PATH}")
    dec = pd.read_csv(DECISIONS_PATH)

    print(f"[info] Loading unmatched live: {UNMATCHED_LIVE_PATH}")
    ul = pd.read_csv(UNMATCHED_LIVE_PATH)

    # Normalize column naming
    if "exit_ts_live" not in ul.columns and "exit_ts" in ul.columns:
        print("[info] Renaming parity_unmatched_live 'exit_ts' -> 'exit_ts_live'")
        ul = ul.rename(columns={"exit_ts": "exit_ts_live"})

    if "exit_ts_live" not in core.columns and "exit_ts" in core.columns:
        print("[info] Renaming core 'exit_ts' -> 'exit_ts_live'")
        core = core.rename(columns={"exit_ts": "exit_ts_live"})

    require_cols(core, ["symbol", "exit_ts_live", "classification"], "core_unmatched_live_analysis")
    require_cols(ul, ["symbol", "exit_ts_live", "bar_ts"], "parity_unmatched_live")
    require_cols(dec, ["symbol", "signal_ts", "decision", "reason"], "signal_decisions")

    # Parse datetimes
    core["exit_ts_live"] = to_dt_utc_series(core["exit_ts_live"])
    ul["exit_ts_live"] = to_dt_utc_series(ul["exit_ts_live"])
    ul["bar_ts"] = to_dt_utc_series(ul["bar_ts"])
    dec["signal_ts"] = to_dt_utc_series(dec["signal_ts"])
    if "lock_until" in dec.columns:
        dec["lock_until"] = to_dt_utc_series(dec["lock_until"])

    # Attach bar_ts to core via (symbol, exit_ts_live)
    core = core.merge(
        ul[["symbol", "exit_ts_live", "bar_ts"]],
        on=["symbol", "exit_ts_live"],
        how="left",
        validate="many_to_one",
    )

    subset = core[core["classification"] == "signal_no_bt_trade"].copy()
    print(f"[info] signal_no_bt_trade rows: {len(subset)}")
    print(f"[info] Missing bar_ts for these rows: {int(subset['bar_ts'].isna().sum())}")

    subset = subset.dropna(subset=["bar_ts"]).copy()

    # Keep only needed columns from decisions
    keep = ["symbol", "signal_ts", "decision", "reason"]
    for extra in ["prob_val", "thr", "lock_until", "ts_effective", "exit_ts", "exit_reason", "pnl", "pnl_R"]:
        if extra in dec.columns:
            keep.append(extra)
    dec2 = dec[keep].copy()

    print(f"[info] Performing per-symbol as-of join with tol={TOLERANCE}")
    merged = merge_asof_per_symbol(subset, dec2)

    # HARDEN: enforce datetime dtypes after concat (prevents Timestamp - str)
    merged["bar_ts"] = to_dt_utc_series(merged["bar_ts"])
    merged["signal_ts"] = to_dt_utc_series(merged["signal_ts"])
    if "lock_until" in merged.columns:
        merged["lock_until"] = to_dt_utc_series(merged["lock_until"])

    attached = int(merged["decision"].notna().sum())
    print(f"[info] Attached decisions to {attached} / {len(merged)} rows")

    # Compute abs delta only where both timestamps are present
    mask = merged["bar_ts"].notna() & merged["signal_ts"].notna()
    merged["abs_dt_min"] = pd.NA
    merged.loc[mask, "abs_dt_min"] = (
        (merged.loc[mask, "bar_ts"] - merged.loc[mask, "signal_ts"])
        .abs()
        .dt.total_seconds()
        / 60.0
    )

    ok_mask = merged["abs_dt_min"].notna()
    if ok_mask.any():
        print("\nabs(bar_ts - signal_ts) minutes (attached+timed rows):")
        print(merged.loc[ok_mask, "abs_dt_min"].astype(float).describe().to_string())

    print("\n=== Reason counts (bar_ts-aligned) ===")
    print(merged["reason"].value_counts(dropna=False))

    out = RESULTS_DIR / "signal_no_bt_trade_with_decisions_bar_ts.csv"
    merged.to_csv(out, index=False)
    print(f"[info] Saved: {out}")


if __name__ == "__main__":
    main()
