#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
DECISIONS = RESULTS_DIR / "signal_decisions.csv"

def to_dt_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def hours(td: pd.Series) -> pd.Series:
    return td.dt.total_seconds() / 3600.0

def main():
    print(f"[info] Loading: {DECISIONS}")
    df = pd.read_csv(DECISIONS)

    # Parse timestamps
    for c in ["signal_ts", "ts_effective", "exit_ts", "open_until", "cooldown_until", "lock_until"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")


    # Basic sanity
    need = ["symbol", "signal_ts", "decision", "reason", "lock_until"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Have={list(df.columns)}")

    df = df.sort_values(["signal_ts", "symbol"]).reset_index(drop=True)

    # ----- Scope test: do multiple symbols share the same lock_until at the same signal_ts? -----
    # Group by exact signal_ts (scout tends to emit many symbols per bar -> perfect for this).
    g = df.groupby("signal_ts", dropna=False)
    summary = g.agg(
        n_rows=("symbol", "size"),
        n_symbols=("symbol", "nunique"),
        n_unique_lock_until=("lock_until", "nunique"),
        cooldown_rows=("reason", lambda s: int((s == "cooldown").sum())),
    ).reset_index()

    # Focus on timestamps where we clearly have many symbols (so scope test is meaningful)
    big = summary[summary["n_symbols"] >= 25].copy()
    if len(big) == 0:
        print("[warn] Not enough multi-symbol timestamps (n_symbols>=25) to strongly infer scope.")
    else:
        print("\n=== Lock scope inference (timestamps with >=25 symbols) ===")
        print("If n_unique_lock_until is usually 1, lock is GLOBAL; if it's large, lock is PER-SYMBOL.")
        print(big["n_unique_lock_until"].describe().to_string())
        top = big.sort_values(["n_symbols", "n_unique_lock_until"], ascending=False).head(20)
        print("\nTop 20 timestamps by (n_symbols, n_unique_lock_until):")
        print(top.to_string(index=False))

    # ----- Semantics test: for taken trades, is lock_until ~= exit_ts + 4h? -----
    taken = df[df["decision"] == "taken"].copy()
    have_exit = ("exit_ts" in taken.columns) and taken["exit_ts"].notna().any()

    print("\n=== Taken trades ===")
    print(f"taken rows: {len(taken)}")

    if have_exit:
        taken = taken.dropna(subset=["exit_ts", "lock_until", "signal_ts"])
        post_exit = hours(taken["lock_until"] - taken["exit_ts"])
        hold = hours(taken["exit_ts"] - taken["signal_ts"])
        lock_from_signal = hours(taken["lock_until"] - taken["signal_ts"])

        print("\nLock minus exit (hours) — should be ~4h if cooldown is post-exit only:")
        print(post_exit.describe().to_string())

        print("\nHold time (exit - signal_ts) hours — trade duration:")
        print(hold.describe().to_string())

        print("\nLock from signal (lock_until - signal_ts) hours — duration + cooldown:")
        print(lock_from_signal.describe().to_string())
    else:
        print("[warn] No usable exit_ts for taken rows; cannot compute hold/cooldown decomposition.")

    # ----- Cooldown rows: how big is lock_remaining at signal time? -----
    cd = df[df["reason"] == "cooldown"].dropna(subset=["signal_ts", "lock_until"]).copy()
    if len(cd):
        rem = hours(cd["lock_until"] - cd["signal_ts"])
        print("\n=== Cooldown-skip rows ===")
        print(rem.describe().to_string())
        print(f"\n> 4h remaining fraction: {(rem > 4).mean():.3f}")
        print(f"> 12h remaining fraction: {(rem > 12).mean():.3f}")
        print(f"> 24h remaining fraction: {(rem > 24).mean():.3f}")
    else:
        print("\n[warn] No cooldown rows found.")

if __name__ == "__main__":
    main()
