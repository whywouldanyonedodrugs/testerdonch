#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
UNMATCHED = RESULTS_DIR / "parity_unmatched_live.csv"
MATCHED = RESULTS_DIR / "parity_matched_trades.csv"

COOLDOWN_HOURS = 4.0


def to_dt_utc(x):
    return pd.to_datetime(x, utc=True, errors="coerce")


def load_live_trades_bar_entry():
    u = pd.read_csv(UNMATCHED)
    m = pd.read_csv(MATCHED)

    # --- unmatched live ---
    # has: symbol, bar_ts, exit_ts
    if "exit_ts_live" not in u.columns and "exit_ts" in u.columns:
        u = u.rename(columns={"exit_ts": "exit_ts_live"})
    if "entry_ts_live" not in u.columns:
        if "bar_ts" in u.columns:
            u = u.rename(columns={"bar_ts": "entry_ts_live"})
        else:
            raise RuntimeError(f"parity_unmatched_live missing bar_ts; cols={list(u.columns)}")

    need_u = ["symbol", "entry_ts_live", "exit_ts_live"]
    for c in need_u:
        if c not in u.columns:
            raise RuntimeError(f"parity_unmatched_live missing {c}; cols={list(u.columns)}")

    u_out = u[need_u].copy()
    u_out["source"] = "unmatched"

    # --- matched live ---
    # has: live_symbol, live_bar_ts, live_exit_ts
    if "live_symbol" not in m.columns:
        raise RuntimeError(f"parity_matched_trades missing live_symbol; cols={list(m.columns)}")
    if "live_bar_ts" not in m.columns:
        raise RuntimeError(f"parity_matched_trades missing live_bar_ts; cols={list(m.columns)}")
    if "live_exit_ts" not in m.columns:
        raise RuntimeError(f"parity_matched_trades missing live_exit_ts; cols={list(m.columns)}")

    m_out = m[["live_symbol", "live_bar_ts", "live_exit_ts"]].copy()
    m_out = m_out.rename(
        columns={"live_symbol": "symbol", "live_bar_ts": "entry_ts_live", "live_exit_ts": "exit_ts_live"}
    )
    m_out["source"] = "matched"

    live = pd.concat([u_out, m_out], ignore_index=True)

    live["entry_ts_live"] = to_dt_utc(live["entry_ts_live"])
    live["exit_ts_live"] = to_dt_utc(live["exit_ts_live"])

    live = live.dropna(subset=["symbol", "entry_ts_live", "exit_ts_live"]).copy()
    live = live[live["exit_ts_live"] >= live["entry_ts_live"]].copy()

    return live


def max_concurrency(live: pd.DataFrame):
    # open interval [entry, exit) => process entries before exits at same timestamp
    ev = []
    for _, r in live.iterrows():
        ev.append((r["entry_ts_live"], +1))
        ev.append((r["exit_ts_live"], -1))

    ev = pd.DataFrame(ev, columns=["ts", "delta"])
    ev = ev.sort_values(["ts", "delta"], ascending=[True, False]).reset_index(drop=True)

    cur = 0
    max_cur = 0
    max_ts = None
    min_cur = 0

    for _, r in ev.iterrows():
        cur += int(r["delta"])
        if cur > max_cur:
            max_cur = cur
            max_ts = r["ts"]
        if cur < min_cur:
            min_cur = cur

    return max_cur, max_ts, min_cur


def per_symbol_cooldown_violations(live: pd.DataFrame):
    s = live.sort_values(["symbol", "entry_ts_live"]).copy()
    s["prev_exit"] = s.groupby("symbol")["exit_ts_live"].shift(1)
    s["gap_hr"] = (s["entry_ts_live"] - s["prev_exit"]).dt.total_seconds() / 3600.0

    v = s.dropna(subset=["gap_hr"]).copy()
    v["viol"] = v["gap_hr"] < COOLDOWN_HOURS

    viols = v[v["viol"]].copy()
    summary = viols.groupby("symbol").size().sort_values(ascending=False)

    return v, viols, summary


def global_cooldown_violations(live: pd.DataFrame):
    s = live.sort_values("entry_ts_live").copy()
    exits = live[["exit_ts_live"]].sort_values("exit_ts_live").copy()

    merged = pd.merge_asof(
        s,
        exits.rename(columns={"exit_ts_live": "prev_exit_global"}),
        left_on="entry_ts_live",
        right_on="prev_exit_global",
        direction="backward",
        tolerance=pd.Timedelta("365D"),
        allow_exact_matches=False,  # key fix: avoid gap==0 due to same-timestamp matches
    )

    merged["gap_hr_global"] = (
        (merged["entry_ts_live"] - merged["prev_exit_global"]).dt.total_seconds() / 3600.0
    )
    merged = merged.dropna(subset=["gap_hr_global"])
    merged["viol_global"] = merged["gap_hr_global"] < COOLDOWN_HOURS

    return merged


def main():
    live = load_live_trades_bar_entry()
    print(f"[info] Loaded LIVE trades: {len(live)} (matched+unmatched), symbols={live['symbol'].nunique()}")

    mc, ts, minc = max_concurrency(live)
    print("\n=== LIVE concurrency (entry=bar_ts proxy) ===")
    print(f"Max concurrent open trades: {mc}")
    print(f"First timestamp reaching max: {ts}")
    if minc < 0:
        print(f"[warn] Concurrency dipped negative at some point (min={minc}). This usually means bad timestamps.")

    print("\n=== Per-symbol cooldown (4h) using bar_ts->prev exit ===")
    v, viols, summary = per_symbol_cooldown_violations(live)
    print(f"Rows with a previous trade in same symbol: {len(v)}")
    print(f"Violations (entry within <{COOLDOWN_HOURS}h after prev exit): {len(viols)}")
    if len(viols):
        print("\nTop symbols by violations:")
        print(summary.head(25).to_string())

        out = RESULTS_DIR / "live_per_symbol_cooldown_violations_bar_ts.csv"
        viols.sort_values(["symbol", "entry_ts_live"]).to_csv(out, index=False)
        print(f"[info] Saved details: {out}")

    print("\n=== Global cooldown (4h) using bar_ts->prev exit (portfolio-level) ===")
    g = global_cooldown_violations(live)
    viol_g = int(g["viol_global"].sum())
    print(f"Entries checked (with a prior exit found): {len(g)}")
    print(f"Violations vs global last-exit (<{COOLDOWN_HOURS}h): {viol_g}")
    print("\ngap_hr_global summary:")
    print(g["gap_hr_global"].describe().to_string())

    outg = RESULTS_DIR / "live_global_cooldown_diagnostic_bar_ts.csv"
    g.to_csv(outg, index=False)
    print(f"[info] Saved: {outg}")


if __name__ == "__main__":
    main()
