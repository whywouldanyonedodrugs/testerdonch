#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
UNMATCHED = RESULTS_DIR / "parity_unmatched_live.csv"
MATCHED = RESULTS_DIR / "parity_matched_trades.csv"

COOLDOWN_HOURS = 4.0


def to_dt_utc(s):
    return pd.to_datetime(s, utc=True, errors="coerce")


def load_live_trades():
    u = pd.read_csv(UNMATCHED)
    m = pd.read_csv(MATCHED)

    # Unmatched live schema
    if "exit_ts_live" not in u.columns and "exit_ts" in u.columns:
        u = u.rename(columns={"exit_ts": "exit_ts_live"})
    if "entry_ts_live" not in u.columns:
        # Bybit export commonly has "Trade time" as entry/open time in your files
        if "Trade time" in u.columns:
            u = u.rename(columns={"Trade time": "entry_ts_live"})
        elif "trade_time" in u.columns:
            u = u.rename(columns={"trade_time": "entry_ts_live"})

    u_need = ["symbol", "entry_ts_live", "exit_ts_live"]
    for c in u_need:
        if c not in u.columns:
            raise RuntimeError(f"parity_unmatched_live missing {c}. Have={list(u.columns)}")

    u_out = u[u_need].copy()
    u_out["source"] = "unmatched"

    # Matched live schema (prefixed)
    sym_col = "live_symbol" if "live_symbol" in m.columns else None
    ent_col = "live_Trade time" if "live_Trade time" in m.columns else None
    ex_col = "live_exit_ts" if "live_exit_ts" in m.columns else None
    if not (sym_col and ent_col and ex_col):
        raise RuntimeError(
            "parity_matched_trades missing one of required columns live_symbol/live_Trade time/live_exit_ts. "
            f"Have={list(m.columns)}"
        )

    m_out = m[[sym_col, ent_col, ex_col]].copy()
    m_out = m_out.rename(
        columns={sym_col: "symbol", ent_col: "entry_ts_live", ex_col: "exit_ts_live"}
    )
    m_out["source"] = "matched"

    live = pd.concat([u_out, m_out], ignore_index=True)

    live["entry_ts_live"] = to_dt_utc(live["entry_ts_live"])
    live["exit_ts_live"] = to_dt_utc(live["exit_ts_live"])

    # Basic cleaning
    live = live.dropna(subset=["symbol", "entry_ts_live", "exit_ts_live"]).copy()
    live = live[live["exit_ts_live"] >= live["entry_ts_live"]].copy()

    return live


def max_concurrency(live: pd.DataFrame):
    ev = []
    for _, r in live.iterrows():
        ev.append((r["entry_ts_live"], +1))
        ev.append((r["exit_ts_live"], -1))

    ev = pd.DataFrame(ev, columns=["ts", "delta"])
    # process exits before entries at same timestamp
    ev = ev.sort_values(["ts", "delta"]).reset_index(drop=True)

    cur = 0
    max_cur = 0
    max_ts = None
    for _, r in ev.iterrows():
        cur += int(r["delta"])
        if cur > max_cur:
            max_cur = cur
            max_ts = r["ts"]

    return max_cur, max_ts


def per_symbol_cooldown_violations(live: pd.DataFrame):
    s = live.sort_values(["symbol", "entry_ts_live"]).copy()
    s["prev_exit"] = s.groupby("symbol")["exit_ts_live"].shift(1)
    s["gap_hr"] = (s["entry_ts_live"] - s["prev_exit"]).dt.total_seconds() / 3600.0
    # Only meaningful if prev_exit exists
    v = s.dropna(subset=["gap_hr"]).copy()
    v["viol"] = v["gap_hr"] < COOLDOWN_HOURS

    viols = v[v["viol"]].copy()
    summary = viols.groupby("symbol").size().sort_values(ascending=False)

    return v, viols, summary


def global_cooldown_violations(live: pd.DataFrame):
    # For each entry, compare to most recent exit BEFORE that entry across all symbols
    s = live.sort_values("entry_ts_live").copy()
    exits = live[["exit_ts_live"]].sort_values("exit_ts_live").copy()

    merged = pd.merge_asof(
        s,
        exits.rename(columns={"exit_ts_live": "prev_exit_global"}),
        left_on="entry_ts_live",
        right_on="prev_exit_global",
        direction="backward",
        tolerance=pd.Timedelta("365D"),
    )

    merged["gap_hr_global"] = (
        (merged["entry_ts_live"] - merged["prev_exit_global"]).dt.total_seconds() / 3600.0
    )
    merged = merged.dropna(subset=["gap_hr_global"])
    merged["viol_global"] = merged["gap_hr_global"] < COOLDOWN_HOURS

    return merged


def main():
    live = load_live_trades()
    print(f"[info] Loaded LIVE trades: {len(live)} (matched+unmatched), symbols={live['symbol'].nunique()}")

    mc, ts = max_concurrency(live)
    print("\n=== LIVE concurrency ===")
    print(f"Max concurrent open trades: {mc}")
    print(f"First timestamp reaching max: {ts}")

    print("\n=== Per-symbol cooldown (4h) ===")
    v, viols, summary = per_symbol_cooldown_violations(live)
    print(f"Rows with a previous trade in same symbol: {len(v)}")
    print(f"Violations (entry within <{COOLDOWN_HOURS}h after prev exit): {len(viols)}")
    if len(viols):
        print("\nTop symbols by violations:")
        print(summary.head(20).to_string())

        out = RESULTS_DIR / "live_per_symbol_cooldown_violations.csv"
        viols.sort_values(["symbol", "entry_ts_live"]).to_csv(out, index=False)
        print(f"[info] Saved details: {out}")

    print("\n=== Global cooldown (4h) diagnostic ===")
    g = global_cooldown_violations(live)
    viol_g = int(g["viol_global"].sum())
    print(f"Entries checked: {len(g)}")
    print(f"Violations vs global last-exit (<{COOLDOWN_HOURS}h): {viol_g}")
    print("\ngap_hr_global summary:")
    print(g["gap_hr_global"].describe().to_string())

    outg = RESULTS_DIR / "live_global_cooldown_diagnostic.csv"
    g.to_csv(outg, index=False)
    print(f"[info] Saved: {outg}")


if __name__ == "__main__":
    main()
