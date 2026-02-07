#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RESULTS = Path("results")
DECISIONS = RESULTS / "signal_decisions.csv"

def main():
    df = pd.read_csv(DECISIONS)
    for c in ["signal_ts", "ts_effective", "lock_until", "exit_ts"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

    need = {"signal_ts", "ts_effective", "lock_until", "decision", "reason"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in signal_decisions.csv: {missing}. Have: {list(df.columns)}")

    cd = df[(df["decision"] == "skipped") & (df["reason"] == "cooldown")].copy()
    print(f"[info] cooldown rows: {len(cd)}")

    cd["rem_hr_signal_ts"] = (cd["lock_until"] - cd["signal_ts"]).dt.total_seconds() / 3600.0
    cd["rem_hr_effective"] = (cd["lock_until"] - cd["ts_effective"]).dt.total_seconds() / 3600.0

    def summarize(s, name):
        s2 = s.dropna()
        print(f"\n=== {name} ===")
        if len(s2) == 0:
            print("all NaN")
            return
        print(s2.describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]).to_string())
        for h in [4, 8, 12, 24, 48, 72]:
            print(f"[info] fraction > {h}h: {(s2 > h).mean():.3f}")

    summarize(cd["rem_hr_signal_ts"], "lock_remaining computed vs signal_ts")
    summarize(cd["rem_hr_effective"], "lock_remaining computed vs ts_effective")

    # show examples where they diverge massively
    cd["delta_hr"] = cd["rem_hr_signal_ts"] - cd["rem_hr_effective"]
    interesting = cd.sort_values("delta_hr", ascending=False).head(30)[
        ["symbol", "signal_ts", "ts_effective", "lock_until", "rem_hr_signal_ts", "rem_hr_effective", "delta_hr"]
    ]
    out = RESULTS / "debug_lock_remaining_signal_vs_effective.csv"
    interesting.to_csv(out, index=False)
    print(f"\n[info] wrote: {out}")

    # sanity: how “daily” is signal_ts?
    if "signal_ts" in df.columns:
        t = df["signal_ts"].dropna()
        daily_midnight_frac = (t.dt.hour.eq(0) & t.dt.minute.eq(0) & t.dt.second.eq(0)).mean()
        print(f"\n[info] signal_ts midnight fraction: {daily_midnight_frac:.3f}")

if __name__ == "__main__":
    main()
