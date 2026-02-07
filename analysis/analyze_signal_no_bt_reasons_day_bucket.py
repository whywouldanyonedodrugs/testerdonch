#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")

CORE_PATH = RESULTS_DIR / "core_unmatched_live_analysis.csv"
DECISIONS_PATH = RESULTS_DIR / "signal_decisions.csv"
UNMATCHED_LIVE_PATH = RESULTS_DIR / "parity_unmatched_live.csv"

TOL = pd.Timedelta("1D")  # daily batch; backward join within 24h


def to_dt_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing columns {missing}. Have={list(df.columns)}")


def merge_asof_per_symbol_backward(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, lgrp in left.groupby("symbol", sort=False):
        lgrp = lgrp.copy()
        lgrp["bar_ts"] = to_dt_utc(lgrp["bar_ts"])
        lgrp = lgrp.sort_values("bar_ts").reset_index(drop=True)

        rgrp = right[right["symbol"] == sym].copy()
        if rgrp.empty:
            # No decisions for this symbol at all
            for c in ["decision_signal_ts", "decision", "reason", "prob_val", "thr", "lock_until"]:
                if c not in lgrp.columns:
                    lgrp[c] = pd.NA
            out.append(lgrp)
            continue

        rgrp["signal_ts"] = to_dt_utc(rgrp["signal_ts"])
        if "lock_until" in rgrp.columns:
            rgrp["lock_until"] = to_dt_utc(rgrp["lock_until"])

        rgrp = rgrp.sort_values("signal_ts").reset_index(drop=True)

        merged = pd.merge_asof(
            lgrp,
            rgrp,
            left_on="bar_ts",
            right_on="signal_ts",
            direction="backward",
            tolerance=TOL,
            suffixes=("", "_dec"),
        )

        # Make the decision timestamp explicit (and stable even if left had a signal_ts column)
        merged["decision_signal_ts"] = merged["signal_ts"]
        out.append(merged)

    return pd.concat(out, ignore_index=True) if out else left.copy()


def main():
    print(f"[info] Loading core: {CORE_PATH}")
    core = pd.read_csv(CORE_PATH)

    print(f"[info] Loading decisions: {DECISIONS_PATH}")
    dec = pd.read_csv(DECISIONS_PATH)

    print(f"[info] Loading unmatched live: {UNMATCHED_LIVE_PATH}")
    ul = pd.read_csv(UNMATCHED_LIVE_PATH)

    # Normalize naming
    if "exit_ts_live" not in ul.columns and "exit_ts" in ul.columns:
        print("[info] Renaming parity_unmatched_live 'exit_ts' -> 'exit_ts_live'")
        ul = ul.rename(columns={"exit_ts": "exit_ts_live"})
    if "exit_ts_live" not in core.columns and "exit_ts" in core.columns:
        core = core.rename(columns={"exit_ts": "exit_ts_live"})

    require_cols(core, ["symbol", "classification", "exit_ts_live"], "core_unmatched_live_analysis")
    require_cols(ul, ["symbol", "exit_ts_live", "bar_ts"], "parity_unmatched_live")
    require_cols(dec, ["symbol", "signal_ts", "decision", "reason"], "signal_decisions")

    # Parse timestamps
    core["exit_ts_live"] = to_dt_utc(core["exit_ts_live"])
    ul["exit_ts_live"] = to_dt_utc(ul["exit_ts_live"])
    ul["bar_ts"] = to_dt_utc(ul["bar_ts"])
    dec["signal_ts"] = to_dt_utc(dec["signal_ts"])
    if "lock_until" in dec.columns:
        dec["lock_until"] = to_dt_utc(dec["lock_until"])

    # Attach bar_ts to core
    core = core.merge(
        ul[["symbol", "exit_ts_live", "bar_ts"]],
        on=["symbol", "exit_ts_live"],
        how="left",
        validate="many_to_one",
    )

    subset = core[core["classification"] == "signal_no_bt_trade"].copy()
    print(f"[info] signal_no_bt_trade rows: {len(subset)}")
    missing_bt = int(subset["bar_ts"].isna().sum())
    print(f"[info] Missing bar_ts: {missing_bt}")
    subset = subset.dropna(subset=["bar_ts"]).copy()

    # Keep decision columns we care about (others may exist)
    keep = ["symbol", "signal_ts", "decision", "reason"]
    for c in ["prob_val", "thr", "lock_until", "ts_effective", "exit_ts", "exit_reason", "pnl", "pnl_R"]:
        if c in dec.columns:
            keep.append(c)
    dec2 = dec[keep].copy()

    print(f"[info] Backward as-of join within {TOL} per symbol (bar_ts -> latest signal_ts)")
    merged = merge_asof_per_symbol_backward(subset, dec2)

    # Compute lag from daily signal to intraday bar (should be 0..1440 mins typically)
    merged["bar_ts"] = to_dt_utc(merged["bar_ts"])
    merged["decision_signal_ts"] = to_dt_utc(merged["decision_signal_ts"])
    mask = merged["bar_ts"].notna() & merged["decision_signal_ts"].notna()
    merged["mins_since_signal"] = pd.NA
    merged.loc[mask, "mins_since_signal"] = (
        (merged.loc[mask, "bar_ts"] - merged.loc[mask, "decision_signal_ts"])
        .dt.total_seconds()
        / 60.0
    )

    attached = int(merged["decision"].notna().sum())
    print(f"[info] Attached decisions to {attached} / {len(merged)} rows")

    print("\n=== Reason counts (day-bucket backward join) ===")
    print(merged["reason"].value_counts(dropna=False))

    # Cooldown diagnostics: did live trade while BT said still locked?
    if "lock_until" in merged.columns:
        merged["lock_until"] = to_dt_utc(merged["lock_until"])
        cd = merged[merged["reason"] == "cooldown"].copy()
        cd = cd.dropna(subset=["bar_ts", "lock_until"])
        cd["lock_remaining_hr_at_bar"] = (cd["lock_until"] - cd["bar_ts"]).dt.total_seconds() / 3600.0
        cd["live_traded_while_locked"] = cd["lock_remaining_hr_at_bar"] > 0

        print("\n=== Cooldown: did live trade while locked? ===")
        if len(cd):
            print(cd["live_traded_while_locked"].value_counts().to_string())
            print("\nlock_remaining_hr_at_bar (summary):")
            print(cd["lock_remaining_hr_at_bar"].describe().to_string())
        else:
            print("[info] No rows with both bar_ts and lock_until for cooldown cases.")

    # Meta prob diagnostics
    if "prob_val" in merged.columns and "thr" in merged.columns:
        mp = merged[merged["reason"] == "meta_prob"].copy()
        mp["prob_val"] = pd.to_numeric(mp["prob_val"], errors="coerce")
        mp["thr"] = pd.to_numeric(mp["thr"], errors="coerce")
        mp["prob_margin"] = mp["prob_val"] - mp["thr"]
        mp = mp.dropna(subset=["prob_margin"])

        print("\n=== Meta_prob: prob_val - thr (summary) ===")
        if len(mp):
            print(mp["prob_margin"].describe().to_string())
        else:
            print("[info] No numeric prob_val/thr margins available in meta_prob rows.")

    # Lag sanity
    lag = merged.dropna(subset=["mins_since_signal"]).copy()
    if len(lag):
        print("\nmins_since_signal (summary):")
        print(lag["mins_since_signal"].astype(float).describe().to_string())
        weird = (lag["mins_since_signal"] < -1) | (lag["mins_since_signal"] > 1440 + 1)
        print(f"[info] mins_since_signal outside [0,1440] (±1min): {int(weird.sum())} / {len(lag)}")

    out = RESULTS_DIR / "signal_no_bt_trade_with_decisions_day_bucket.csv"
    merged.to_csv(out, index=False)
    print(f"\n[info] Saved: {out}")


if __name__ == "__main__":
    main()
