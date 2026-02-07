#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
DECISIONS_PATH = RESULTS_DIR / "signal_decisions.csv"

TIME_CANDIDATES = [
    "signal_ts", "bar_ts", "ts", "timestamp", "time", "decision_ts", "event_ts",
    "dt", "datetime", "bar_time"
]

def to_dt_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def pick_time_col(df: pd.DataFrame) -> str:
    for c in TIME_CANDIDATES:
        if c in df.columns:
            return c
    # try heuristic: any column that looks like time
    for c in df.columns:
        lc = c.lower()
        if "ts" in lc or "time" in lc or "date" in lc:
            return c
    raise RuntimeError(f"No obvious time column found. Columns={list(df.columns)}")

def pick_cooldown_remaining_col(df: pd.DataFrame):
    # common patterns: cooldown_remaining_hours, cooldown_hours_remaining, etc.
    cols = []
    for c in df.columns:
        lc = c.lower()
        if "cooldown" in lc and ("remain" in lc or "remaining" in lc or "left" in lc):
            cols.append(c)
    return cols[0] if cols else None

def main():
    print(f"[info] Loading: {DECISIONS_PATH}")
    df = pd.read_csv(DECISIONS_PATH)
    print(f"[info] Rows: {len(df)}")
    print(f"[info] Columns: {list(df.columns)}")

    tcol = pick_time_col(df)
    print(f"[info] Using time column: {tcol}")
    df[tcol] = to_dt_utc(df[tcol])

    bad_ts = df[tcol].isna().sum()
    if bad_ts:
        print(f"[warn] NaT in {tcol}: {bad_ts} rows")

    # Check monotonicity in FILE ORDER (this approximates processing order if you appended rows as you processed)
    t = df[tcol]
    diffs = t.diff()
    neg = diffs < pd.Timedelta(0)
    n_neg = int(neg.sum())

    print("\n=== Time order in file ===")
    print(f"Monotonic increasing: {t.is_monotonic_increasing}")
    print(f"Negative time steps (time went backwards): {n_neg} / {len(df)-1}")

    if n_neg > 0:
        idx = df.index[neg.fillna(False)]
        sample = idx[:25]
        print("\nFirst inversions (showing prev and current):")
        out = pd.DataFrame({
            "prev_idx": sample - 1,
            "prev_time": df.loc[sample - 1, tcol].values,
            "idx": sample,
            "time": df.loc[sample, tcol].values,
            "symbol": df.loc[sample, "symbol"].values if "symbol" in df.columns else None,
            "decision": df.loc[sample, "decision"].values if "decision" in df.columns else None,
            "reason": df.loc[sample, "reason"].values if "reason" in df.columns else None,
        })
        print(out.to_string(index=False))

    # Cooldown remaining sanity
    rem_col = pick_cooldown_remaining_col(df)
    if rem_col:
        print(f"\n[info] Found cooldown remaining column: {rem_col}")
        if pd.api.types.is_numeric_dtype(df[rem_col]):
            cd = df.loc[df.get("reason").eq("cooldown") if "reason" in df.columns else slice(None), rem_col]
            print("\n=== Cooldown remaining sanity (cooldown rows) ===")
            print(cd.describe())
            print(f">4h count: {(cd > 4).sum()} / {cd.notna().sum()}")
            print(f">10h count: {(cd > 10).sum()} / {cd.notna().sum()}")
        else:
            print(f"[warn] {rem_col} is not numeric (dtype={df[rem_col].dtype})")
    else:
        print("\n[warn] Could not find a cooldown-remaining column (that’s ok).")

    # Also check per-symbol monotonicity to catch symbol-partition processing artifacts
    if "symbol" in df.columns:
        print("\n=== Per-symbol time monotonicity (file order) ===")
        grp = df.dropna(subset=[tcol]).groupby("symbol", sort=False)
        nonmono = []
        for sym, g in grp:
            if not g[tcol].is_monotonic_increasing:
                nonmono.append(sym)
        print(f"Symbols with non-monotonic times: {len(nonmono)}")
        if nonmono:
            print("First 30:", nonmono[:30])

if __name__ == "__main__":
    main()
