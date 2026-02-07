#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
UNMATCHED = RESULTS_DIR / "parity_unmatched_live.csv"
MATCHED   = RESULTS_DIR / "parity_matched_trades.csv"

def to_dt_utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")

def main():
    u = pd.read_csv(UNMATCHED)
    m = pd.read_csv(MATCHED)

    # ---- normalize unmatched ----
    if "exit_ts_live" not in u.columns and "exit_ts" in u.columns:
        u = u.rename(columns={"exit_ts": "exit_ts_live"})
    # candidate bar column
    bar_u = "bar_ts" if "bar_ts" in u.columns else None

    u_out = pd.DataFrame({
        "source": "unmatched",
        "symbol": u.get("symbol"),
        "bar_ts": u.get(bar_u) if bar_u else None,
        "exit_ts_live": u.get("exit_ts_live"),
        "trade_time": u.get("Trade time")  # Bybit column name in your CSV
    })

    # ---- normalize matched ----
    m_out = pd.DataFrame({
        "source": "matched",
        "symbol": m.get("live_symbol"),
        "bar_ts": m.get("live_bar_ts"),
        "exit_ts_live": m.get("live_exit_ts"),
        "trade_time": m.get("live_Trade time"),
    })

    live = pd.concat([u_out, m_out], ignore_index=True)

    # parse datetimes
    for c in ["bar_ts", "exit_ts_live", "trade_time"]:
        live[c] = to_dt_utc(live[c])

    # report missingness
    print(f"[info] Rows total: {len(live)}")
    print("[info] Missing counts:")
    print(live[["symbol","bar_ts","exit_ts_live","trade_time"]].isna().sum().to_string())

    # keep rows where we have at least exit and bar
    x = live.dropna(subset=["symbol","bar_ts","exit_ts_live"]).copy()
    x["exit_minus_bar_hr"] = (x["exit_ts_live"] - x["bar_ts"]).dt.total_seconds() / 3600.0

    print("\n=== exit_ts_live - bar_ts (hours) ===")
    print(x["exit_minus_bar_hr"].describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]).to_string())

    # how often is bar close to exit?
    for h in [0.5, 1, 2, 4, 12, 24]:
        frac = (x["exit_minus_bar_hr"].abs() <= h).mean()
        print(f"[info] Fraction |exit-bar| <= {h}h: {frac:.3f}")

    # trade_time sanity vs exit_ts_live
    y = live.dropna(subset=["exit_ts_live","trade_time"]).copy()
    y["trade_minus_exit_min"] = (y["trade_time"] - y["exit_ts_live"]).dt.total_seconds() / 60.0
    print("\n=== trade_time - exit_ts_live (minutes) ===")
    print(y["trade_minus_exit_min"].describe(percentiles=[.01,.05,.1,.25,.5,.75,.9,.95,.99]).to_string())

    # show some biggest-duration candidates if bar is entry-like
    # (if exit_minus_bar_hr is often huge and positive, bar_ts might be entry-ish)
    top = x.sort_values("exit_minus_bar_hr", ascending=False).head(25)
    out = RESULTS_DIR / "debug_exit_minus_bar_top.csv"
    top[["source","symbol","bar_ts","exit_ts_live","exit_minus_bar_hr"]].to_csv(out, index=False)
    print(f"\n[info] Wrote: {out}")

if __name__ == "__main__":
    main()
