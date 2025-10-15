# offline_eth_probe.py
import argparse
import pandas as pd
import numpy as np
import sys
import re
from pathlib import Path


def pct(a, b):
    return (a / b - 1.0)


def load_eth(eth_path: Path):
    df = pd.read_parquet(eth_path)

    # Ensure UTC datetime index
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            df.index = idx.tz_localize("UTC")
        else:
            df.index = idx.tz_convert("UTC")
    else:
        tscol = (
            "open_time"
            if "open_time" in df.columns
            else ("timestamp" if "timestamp" in df.columns else None)
        )
        if tscol is None:
            raise ValueError(
                "ETH parquet must have a DatetimeIndex or an 'open_time'/'timestamp' column"
            )
        df[tscol] = pd.to_datetime(df[tscol], utc=True, errors="coerce")
        df = df.set_index(tscol).sort_index()

    # Normalize columns
    df = df.rename(columns={c: c.lower() for c in df.columns})
    need = {"open", "high", "low", "close"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"ETH file missing columns: {miss}")
    return df[["open", "high", "low", "close"]].sort_index()


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def build_eth_features(eth, fast_tf="4h", fast_n=20, slow_tf="1D", slow_n=200):
    out = pd.DataFrame(index=eth.index)
    out["close"] = eth["close"]

    # Resample to fast/slow grids
    fast = eth["close"].resample(fast_tf, label="right", closed="right").last().dropna()
    slow = eth["close"].resample(slow_tf, label="right", closed="right").last().dropna()

    out["ema_fast"] = ema(fast, fast_n).reindex(out.index, method="ffill")
    out["ema_slow"] = ema(slow, slow_n).reindex(out.index, method="ffill")

    out["pct_dev_fast"] = pct(out["close"], out["ema_fast"])
    out["pct_dev_slow"] = pct(out["close"], out["ema_slow"])

    # Z of fast deviation (compute on fast grid to avoid leakage)
    fast_dev = pct(fast, ema(fast, fast_n))
    z_mu = fast_dev.rolling(200, min_periods=50).mean()
    z_sd = fast_dev.rolling(200, min_periods=50).std().replace(0, np.nan)
    z_fast = (fast_dev - z_mu) / z_sd
    out["z_fast"] = z_fast.reindex(out.index, method="ffill")

    return out.dropna(subset=["ema_fast", "ema_slow", "pct_dev_fast"])


def load_trades(csv_path: Path):
    print(f"[probe] reading trades CSV: {csv_path}")
    if not csv_path.exists():
        print(f"[probe][ERR] CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Parse datetimes if present
    for col in ("timestamp", "entry_ts", "exit_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Choose entry timestamp: prefer entry_ts, else timestamp
    base_col = "entry_ts" if "entry_ts" in df.columns else "timestamp"
    if base_col not in df.columns:
        print("[probe][ERR] Neither 'entry_ts' nor 'timestamp' present.")
        sys.exit(1)

    df = df.dropna(subset=[base_col]).copy()
    df["entry_t"] = df[base_col]

    # Normalise PnL column name
    if "final_pnl" not in df.columns and "pnl" in df.columns:
        df = df.rename(columns={"pnl": "final_pnl"})

    print(f"[probe] trades loaded: {len(df)} rows (using '{base_col}' as entry)")
    print(df.head(3).to_string(index=False))
    return df


def wr_ci(wins, n):
    if n == 0:
        return (np.nan, np.nan)
    p = wins / n
    se = np.sqrt(p * (1 - p) / n)
    return (max(0, p - 1.96 * se), min(1, p + 1.96 * se))


def pf_avg(df):
    gp = df["final_pnl"][df["final_pnl"] > 0].sum()
    gl = -df["final_pnl"][df["final_pnl"] < 0].sum()
    pf = gp / gl if gl > 0 else np.nan
    return pf, df["final_pnl"].mean(), len(df), gp, gl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--eth_pq", required=True)
    ap.add_argument("--fast_tf", default="4h")
    ap.add_argument("--fast_n", type=int, default=20)
    ap.add_argument("--slow_tf", default="1D")
    ap.add_argument("--slow_n", type=int, default=200)

    # ETH-based actions
    ap.add_argument("--gate_when_up", type=float, default=None,
                    help="If set, drop trades where pct_dev_fast >= this threshold")
    ap.add_argument("--boost_band", default=None,
                    help='Optional boost band like "-0.02:-0.01" on pct_dev_fast')
    ap.add_argument("--boost_factor", type=float, default=1.0)
    ap.add_argument("--size_when_up", type=float, default=0.6)
    ap.add_argument("--up_thr", type=float, default=0.02)

    # VWAP stacking (from your aggregated CSV)
    ap.add_argument("--min_abs_vwap", type=float, default=None,
                    help="If set, keep only trades with abs(pct_to_vwap) >= this")
    ap.add_argument("--require_consolidated", action="store_true",
                    help="If set, keep only vwap_consolidated_at_entry == True")

    # Exploration knobs
    ap.add_argument("--print_eth_regimes", action="store_true",
                    help="Print WR/PF by ETH regime (Up/Down/Chop)")
    ap.add_argument("--sweep_up_thr", default=None,
                    help="Comma list like 0.005,0.01,0.015,0.02 to sweep")
    ap.add_argument("--sweep_size", default=None,
                    help="Comma list like 0.4,0.5,0.6,0.7 to sweep")

    # Accept EITHER: space-separated list OR a single comma-separated string
    ap.add_argument(
        "--bands",
        nargs="+",
        default=["-0.03,-0.02,-0.01,0,0.01,0.02,0.03"],
        help=(
            "ETH fast-EMA pct deviation band edges. Example:\n"
            "--bands -0.03 -0.02 -0.01 0 0.01 0.02 0.03\n"
            "OR --bands \"-0.03,-0.02,-0.01,0,0.01,0.02,0.03\""
        ),
    )

    args = ap.parse_args()

    # Robust parse for bands (works with both styles)
    tokens = []
    for item in args.bands:
        tokens += [t for t in re.split(r"[,\s]+", str(item).strip()) if t != ""]
    try:
        cuts = [float(t) for t in tokens]
        cuts = sorted(set(cuts))  # de-dup & sort
    except Exception as e:
        print(f"[probe][ERR] could not parse --bands {args.bands!r}: {e}")
        sys.exit(2)

    print("[probe] script start")
    print(f"[probe] bands parsed → {cuts}")

    trades = load_trades(Path(args.csv))

    print(f"[probe] reading ETH parquet: {args.eth_pq}")
    eth = load_eth(Path(args.eth_pq))

    feats = build_eth_features(eth, args.fast_tf, args.fast_n, args.slow_tf, args.slow_n)
    print(f"[probe] ETH bars: {len(eth)} | feature window: {feats.index.min()} → {feats.index.max()}")

    # Align and merge features to trade entries
    trades = trades.sort_values("entry_t")
    feats = feats.sort_index()
    joined = pd.merge_asof(
        trades, feats,
        left_on="entry_t", right_index=True,
        direction="backward"
    ).dropna(subset=["pct_dev_fast", "final_pnl"])

    # Optional VWAP stacking (purely offline)
    if args.min_abs_vwap is not None and "pct_to_vwap" in joined.columns:
        joined = joined[joined["pct_to_vwap"].abs() >= float(args.min_abs_vwap)]
    if args.require_consolidated and "vwap_consolidated_at_entry" in joined.columns:
        joined = joined[joined["vwap_consolidated_at_entry"] == True]

    print(f"[probe] merged rows: {len(joined)}")
    if joined.empty:
        print("[probe][WARN] merge produced 0 rows. Likely a time alignment issue. "
              "Check that ETH parquet is 5m candles with UTC timestamps.")
        sys.exit(0)

    # === ETH pct_dev_fast buckets ===
    print("\n=== ETH pct_dev_fast buckets ===")
    bins = [-np.inf] + cuts + [np.inf]
    labels = [f"({bins[i]:+.3f},{bins[i+1]:+.3f}]" for i in range(len(bins) - 1)]
    joined["band"] = pd.cut(joined["pct_dev_fast"], bins=bins, labels=labels, include_lowest=True)

    rows = []
    for lab in labels:
        sub = joined[joined["band"] == lab]
        n = len(sub)
        wins = (sub["final_pnl"] > 0).sum()
        wr_lwr, wr_upr = wr_ci(wins, n)
        gp = sub.loc[sub["final_pnl"] > 0, "final_pnl"].sum()
        gl = -sub.loc[sub["final_pnl"] < 0, "final_pnl"].sum()
        pf = gp / gl if gl > 0 else np.nan
        avg = sub["final_pnl"].mean() if n else np.nan
        rows.append([n, wins, wins / n if n else np.nan, wr_lwr, wr_upr, pf, avg, gp, gl, lab])

    out = pd.DataFrame(
        rows,
        columns=["n", "wins", "win_rate", "wr_lwr", "wr_upr", "pf", "avg_pnl", "gp", "gl", "band"],
    ).sort_values("band")
    print(out.to_string(index=False))

    # === Hard gate (optional) ===
    print("\n=== Hard gate (if configured) ===")
    if args.gate_when_up is not None:
        gated = joined[joined["pct_dev_fast"] < float(args.gate_when_up)]
        g_pf, g_avg, g_n, g_gp, g_gl = pf_avg(gated)
        print(
            f"Gate when pct_dev_fast >= {args.gate_when_up:+.3f} → "
            f"N={g_n}  PF={g_pf:.3f}  Avg={g_avg:.4f}"
        )
    else:
        print("(not enabled)")

    # === Two-sided sizing: shrink when up, optional boost in a weak band ===
    scaled = joined.copy(deep=True)
    scaled.loc[scaled["pct_dev_fast"] >= args.up_thr, "final_pnl"] *= float(args.size_when_up)

    if args.boost_band:
        lo, hi = [float(x) for x in args.boost_band.split(":")]
        mask = (scaled["pct_dev_fast"] >= lo) & (scaled["pct_dev_fast"] < hi)
        scaled.loc[mask, "final_pnl"] *= float(args.boost_factor)

    base_pf, base_avg, base_n, base_gp, base_gl = pf_avg(joined)
    new_pf, new_avg, new_n, new_gp, new_gl = pf_avg(scaled)

    print("\n=== Two-sided sizing ===")
    print(f"Base: N={base_n} PF={base_pf:.3f} Avg={base_avg:.4f}")
    print(f"Shrink to {args.size_when_up}x when pct_dev_fast ≥ {args.up_thr:+.3f}")
    if args.boost_band:
        print(f"Boost {args.boost_factor}x when {lo:+.3f} ≤ pct_dev_fast < {hi:+.3f}")
    print(f"Result: N={new_n} PF={new_pf:.3f} Avg={new_avg:.4f}")

    # === What-if summary (simple shrink only) ===
    base_pf2, base_avg2, _, _, _ = pf_avg(joined)
    new_pf2, new_avg2, _, _, _ = pf_avg(scaled)
    print("\n=== What-if sizing ===")
    print(f"Base: PF={base_pf2:.3f}  Avg={base_avg2:.4f}  N={len(joined)}")
    print(
        f"Size {args.size_when_up}x when pct_dev_fast ≥ {args.up_thr:+.3f}: "
        f"PF={new_pf2:.3f}  Avg={new_avg2:.4f}"
    )

    # === Optional: ETH regime snapshot using merged features ===
    if args.print_eth_regimes:
        reg = np.where(
            (scaled["ema_fast"] > scaled["ema_slow"]) & (scaled["z_fast"] > 0.5),
            "Up",
            np.where(
                (scaled["ema_fast"] < scaled["ema_slow"]) & (scaled["z_fast"] < -0.5),
                "Down",
                "Chop",
            ),
        )
        scaled = scaled.assign(eth_regime=reg)
        print("\n=== ETH regime snapshot (scaled sizing applied) ===")
        rows = []
        for name, sub in scaled.groupby("eth_regime"):
            pf, avg, n, gp, gl = pf_avg(sub)
            wr = (sub["final_pnl"] > 0).mean()
            rows.append([name, n, wr, pf, avg])
        out = pd.DataFrame(rows, columns=["regime", "n", "wr", "pf", "avg"]).sort_values("regime")
        print(out.to_string(index=False))

    # === Optional: sweep up-threshold × size ===
    if args.sweep_up_thr and args.sweep_size:
        print("\n=== Sweep (up_thr × size_when_up) ===")
        thrs = [float(x) for x in args.sweep_up_thr.split(",")]
        sizes = [float(x) for x in args.sweep_size.split(",")]
        grid = []
        for t in thrs:
            for s in sizes:
                tmp = joined.copy()
                tmp.loc[tmp["pct_dev_fast"] >= t, "final_pnl"] *= s
                pf, avg, n, gp, gl = pf_avg(tmp)
                wr = (tmp["final_pnl"] > 0).mean()
                grid.append([t, s, n, wr, pf, avg])
        gdf = pd.DataFrame(grid, columns=["up_thr", "size", "n", "wr", "pf", "avg"])
        print(gdf.sort_values(["up_thr", "size"]).to_string(index=False))


if __name__ == "__main__":
    main()
