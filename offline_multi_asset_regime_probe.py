# offline_multi_asset_regime_probe.py
import argparse, sys, re
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- small utils ----------
def pct(a, b): 
    return a / b - 1.0

def ensure_utc_index(df, tscol_candidates=("open_time", "timestamp")):
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        df.index = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC"))
    else:
        tscol = None
        for c in tscol_candidates:
            if c in df.columns:
                tscol = c; break
        if tscol is None:
            raise ValueError("No DatetimeIndex and no open_time/timestamp column found.")
        df[tscol] = pd.to_datetime(df[tscol], utc=True, errors="coerce")
        df = df.set_index(tscol)
    return df.sort_index()

def load_parquet_ohlc(pq_path: Path):
    df = pd.read_parquet(pq_path)
    df = ensure_utc_index(df)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    need = {"open","high","low","close"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{pq_path} missing columns: {miss}")
    return df[["open","high","low","close"]]

def resample_to(df, tf="4h"):
    # right-closed bars to be conservative for merge_asof
    return df.resample(tf, label="right", closed="right").agg({
        "open":"first","high":"max","low":"min","close":"last"
    }).dropna(how="any")

def ema(x, n): 
    return x.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(close, n=14):
    r = close.diff()
    up = r.clip(lower=0.0)
    dn = (-r).clip(lower=0.0)
    rs = ema(up, n) / ema(dn, n)
    return 100.0 - (100.0 / (1.0 + rs))

def macd_sig_hist(close, fast=12, slow=26, sig=9):
    macd = ema(close, fast) - ema(close, slow)
    signal = ema(macd, sig)
    hist = macd - signal
    return macd, signal, hist

def heikin_ashi(ohlc):
    o = ohlc["open"].copy()
    h = ohlc["high"].copy()
    l = ohlc["low"].copy()
    c = ohlc["close"].copy()

    ha_c = (o + h + l + c) / 4.0
    ha_o = ha_c.copy()
    # seed first HA open = real open (common convention for first bar)
    ha_o.iloc[0] = o.iloc[0]
    for i in range(1, len(o)):
        ha_o.iat[i] = 0.5 * (ha_o.iat[i-1] + ha_c.iat[i-1])
    ha_h = pd.concat([h, ha_o, ha_c], axis=1).max(axis=1)
    ha_l = pd.concat([l, ha_o, ha_c], axis=1).min(axis=1)
    return ha_o, ha_h, ha_l, ha_c

def psar(high, low, step=0.02, step_max=0.2):
    hi = high.values
    lo = low.values
    n = len(hi)
    sar = np.zeros(n)
    bull = True
    af = step
    ep = hi[0]
    sar[0] = lo[0]
    for i in range(1, n):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if bull:
            sar[i] = min(sar[i], lo[i-1], lo[i-2] if i > 1 else lo[i-1])
            if hi[i] > ep:
                ep = hi[i]; af = min(af + step, step_max)
            if lo[i] < sar[i]:
                bull = False
                sar[i] = ep
                ep = lo[i]
                af = step
        else:
            sar[i] = max(sar[i], hi[i-1], hi[i-2] if i > 1 else hi[i-1])
            if lo[i] < ep:
                ep = lo[i]; af = min(af + step, step_max)
            if hi[i] > sar[i]:
                bull = True
                sar[i] = ep
                ep = hi[i]
                af = step
    return pd.Series(sar, index=high.index)

def build_features_4h(ohlc_5m, fast_ema=20, psar_step=0.02, psar_max=0.2, rsi_len=14):
    ohlc = resample_to(ohlc_5m, "4h")
    ha_o, ha_h, ha_l, ha_c = heikin_ashi(ohlc)
    ha_up = (ha_c > ha_o)

    sar = psar(ohlc["high"], ohlc["low"], step=psar_step, step_max=psar_max)
    psar_up = (sar < ohlc["close"])

    r = rsi(ohlc["close"], rsi_len)
    rsi_up = (r >= 55)  # tweakable band

    macd, sig, hist = macd_sig_hist(ohlc["close"])  # 12/26/9
    macd_up = (macd > sig) & (hist > 0)

    ema_fast = ema(ohlc["close"], fast_ema)

    feats = pd.DataFrame({
        "ha_up": ha_up,
        "psar_up": psar_up,
        "rsi": r,
        "rsi_up": rsi_up,
        "macd": macd,
        "macd_sig": sig,
        "macd_hist": hist,
        "macd_up": macd_up,
        "ema_fast": ema_fast,
        "close_4h": ohlc["close"]
    }, index=ohlc.index).dropna(how="any")

    return feats

def merge_asof_features(trades, feats, prefix, debug=False):
    # forward-fill to 5m granularity so merge_asof can find a match
    feats_ff = feats.reindex(trades["entry_t"].sort_values().unique(), method="ffill")
    feats_ff.index.name = "t"

    merged = pd.merge_asof(
        trades.sort_values("entry_t"),
        feats_ff,
        left_on="entry_t", right_index=True,
        direction="backward"
    )
    # prefix indicators to avoid collisions
    rename = {c: f"{prefix}_{c}" for c in feats.columns}
    merged = merged.rename(columns=rename)

    if debug:
        print(f"[probe] merge '{prefix}': rows={len(merged)}  "
              f"NA counts for {prefix}_ha_up={merged[f'{prefix}_ha_up'].isna().sum()}")

    return merged

def pf_avg(df):
    gp = df.loc[df["final_pnl"] > 0, "final_pnl"].sum()
    gl = -df.loc[df["final_pnl"] < 0, "final_pnl"].sum()
    pf = gp / gl if gl > 0 else np.nan
    return pf, df["final_pnl"].mean(), len(df), gp, gl

def bucket_print(df, col, title):
    print(f"\n=== {title} ({col}) ===")
    if col not in df.columns:
        print(f"(missing column: {col})")
        return
    if df[col].dropna().nunique() == 0:
        print("(all NaN)")
        return
    rows = []
    for val, sub in df.groupby(col, dropna=False):
        n = len(sub)
        wr = (sub["final_pnl"] > 0).mean() if n else np.nan
        pf, avg, _, gp, gl = pf_avg(sub)
        rows.append([val, n, wr, pf, avg, gp, gl])
    out = pd.DataFrame(rows, columns=["state","n","wr","pf","avg","gp","gl"]).sort_values("state")
    print(out.to_string(index=False))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="trades_aggregated_*.csv")
    ap.add_argument("--btc_pq", default=None)
    ap.add_argument("--eth_pq", default=None)
    ap.add_argument("--sol_pq", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not any([args.btc_pq, args.eth_pq, args.sol_pq]):
        print("[probe][ERR] Provide at least one of --btc_pq / --eth_pq / --sol_pq")
        sys.exit(2)

    print("[probe] start")
    trades = pd.read_csv(args.csv)
    for c in ("timestamp","entry_ts","exit_ts"):
        if c in trades.columns:
            trades[c] = pd.to_datetime(trades[c], utc=True, errors="coerce")
    base = "entry_ts" if "entry_ts" in trades.columns else ("timestamp" if "timestamp" in trades.columns else None)
    if base is None:
        print("[probe][ERR] neither entry_ts nor timestamp in trades CSV"); sys.exit(2)
    trades = trades.dropna(subset=[base]).copy()
    trades["entry_t"] = trades[base]
    if "final_pnl" not in trades and "pnl" in trades:
        trades = trades.rename(columns={"pnl":"final_pnl"})

    print(f"[probe] trades: N={len(trades)}  time: {trades['entry_t'].min()} → {trades['entry_t'].max()}")

    merged = trades.copy()

    # ETH
    if args.eth_pq:
        eth = load_parquet_ohlc(Path(args.eth_pq))
        if args.debug:
            print(f"[probe] ETH bars: {len(eth)}  {eth.index.min()} → {eth.index.max()}")
        f = build_features_4h(eth)
        merged = merge_asof_features(merged, f, "eth", debug=args.debug)

    # BTC
    if args.btc_pq:
        btc = load_parquet_ohlc(Path(args.btc_pq))
        if args.debug:
            print(f"[probe] BTC bars: {len(btc)}  {btc.index.min()} → {btc.index.max()}")
        f = build_features_4h(btc)
        merged = merge_asof_features(merged, f, "btc", debug=args.debug)

    # SOL
    if args.sol_pq:
        sol = load_parquet_ohlc(Path(args.sol_pq))
        if args.debug:
            print(f"[probe] SOL bars: {len(sol)}  {sol.index.min()} → {sol.index.max()}")
        f = build_features_4h(sol)
        merged = merge_asof_features(merged, f, "sol", debug=args.debug)

    # Print quick health
    print(f"[probe] merged rows: {len(merged)}")
    print("[probe] columns (sample):", [c for c in merged.columns if re.search(r"(eth|btc|sol)_(ha_up|psar_up|rsi_up|macd_up)$", c)][:12])

    # Buckets per asset
    for asset in ("eth","btc","sol"):
        for key in ("ha_up","psar_up","rsi_up","macd_up"):
            col = f"{asset}_{key}"
            if col in merged.columns:
                bucket_print(merged, col, f"{asset.upper()} – {key}")

    # Composite “UP” snapshot per asset (HA + PSAR agree and MACD_hist>0 and RSI>=55)
    for asset in ("eth","btc","sol"):
        fields = [f"{asset}_ha_up", f"{asset}_psar_up", f"{asset}_macd_hist", f"{asset}_rsi"]
        if all(f in merged.columns for f in fields):
            comp_col = f"{asset}_UP"
            merged[comp_col] = (
                (merged[f"{asset}_ha_up"] == True) &
                (merged[f"{asset}_psar_up"] == True) &
                (merged[f"{asset}_macd_hist"] > 0) &
                (merged[f"{asset}_rsi"] >= 55)
            )
            bucket_print(merged, comp_col, f"{asset.upper()} – composite UP")

    # What-if: drop trades when asset composite UP is True (inverse exposure test)
    for asset in ("eth","btc","sol"):
        comp = f"{asset}_UP"
        if comp in merged.columns:
            base_pf, base_avg, base_n, *_ = pf_avg(merged)
            gated = merged[merged[comp] != True]
            g_pf, g_avg, g_n, *_ = pf_avg(gated)
            print(f"\n=== What-if: gate when {asset.upper()}_UP ===")
            print(f"Base: N={base_n} PF={base_pf:.3f} Avg={base_avg:.4f}")
            print(f"Gate: N={g_n} PF={g_pf:.3f} Avg={g_avg:.4f}")

if __name__ == "__main__":
    main()
