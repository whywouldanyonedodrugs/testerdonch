# apply_meta_filter_and_rebacktest.py
from __future__ import annotations
import argparse, pandas as pd
from pathlib import Path

_TS_CANDS = ["timestamp","entry_ts","entry_time","ts","time","dt"]
_SYM_CANDS = ["symbol","asset","pair","ticker"]
_PROBA_CANDS = ["meta_p","y_proba","proba","prob","p","y_pred","pred_proba"]

import numpy as np
import pandas as pd

def _print_bt_summary(results_dir: Path = Path("results")) -> None:
    """Print PF, win-rate, avg R, max DD, CAGR, Sharpe from results/trades.csv & equity.csv."""
    tpath = results_dir / "trades.csv"
    epath = results_dir / "equity.csv"
    if not tpath.exists():
        print("[bt] no trades.csv found; nothing to summarize")
        return

    tr = pd.read_csv(tpath, parse_dates=["entry_ts","exit_ts"])
    n = len(tr)
    if n == 0:
        print("[bt] 0 trades")
        return

    # Profit factor (use cash PnL); protect against divide-by-zero
    gross_profit = tr.loc[tr["pnl"] > 0, "pnl"].sum()
    gross_loss   = -tr.loc[tr["pnl"] < 0, "pnl"].sum()
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    win = float((tr["pnl"] > 0).mean())  # win-rate
    avg_R = float(tr["pnl_R"].mean())

    # Max drawdown, CAGR, Sharpe from equity curve if available
    max_dd = np.nan
    cagr   = np.nan
    sharpe = np.nan
    if epath.exists():
        eq = pd.read_csv(epath, parse_dates=["timestamp"]).sort_values("timestamp")
        if len(eq) >= 2:
            equity = eq["equity"].astype(float)
            dd = equity / equity.cummax() - 1.0
            max_dd = float(dd.min())  # negative number
            # CAGR
            yrs = (eq["timestamp"].iloc[-1] - eq["timestamp"].iloc[0]).total_seconds() / (365.25*24*3600)
            if yrs > 0:
                cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1/yrs) - 1.0)
            # daily Sharpe on resampled equity
            daily = eq.set_index("timestamp")["equity"].resample("1D").last().pct_change().dropna()
            if daily.std(ddof=1) > 0:
                sharpe = float(np.sqrt(252.0) * daily.mean() / daily.std(ddof=1))

    print(
        "[bt] trades={n}, PF={pf:.2f}, win={win:.1%}, avg_R={avg_R:.3f}, "
        "maxDD={dd:.1%}, CAGR={cagr:.1%}, Sharpe={sh:.2f}"
        .format(n=n, pf=pf, win=win, avg_R=avg_R,
                dd=abs(max_dd) if np.isfinite(max_dd) else float("nan"),
                cagr=cagr if np.isfinite(cagr) else float("nan"),
                sh=sharpe if np.isfinite(sharpe) else float("nan"))
    )

def _to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def _guess(cols: list[str], candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for k in candidates:
        if k in low:
            return low[k]
    return None

def _read_signals(path: Path) -> pd.DataFrame:
    sig = pd.read_parquet(path)
    ts_col = _guess(sig.columns.tolist(), _TS_CANDS)
    sym_col = _guess(sig.columns.tolist(), _SYM_CANDS)
    if not (ts_col and sym_col):
        raise ValueError("Signals must contain a timestamp and symbol column "
                         f"(looked for { _TS_CANDS } and { _SYM_CANDS }).")
    sig = sig.copy()
    sig["ts"]  = _to_utc(sig[ts_col])
    sig["sym"] = sig[sym_col].astype(str)
    # normalize house-keeping
    if "timestamp" not in sig.columns:
        sig["timestamp"] = sig["ts"]
    if "symbol" not in sig.columns:
        sig["symbol"] = sig["sym"]
    return sig

def _read_preds(path: Path, proba_col: str | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if proba_col and proba_col in df.columns:
        df = df.rename(columns={proba_col: "meta_p"})
    else:
        for c in _PROBA_CANDS:
            if c in df.columns:
                df = df.rename(columns={c: "meta_p"})
                break
        else:
            raise ValueError("Predictions must contain a probability column. "
                             f"Pass --proba-col or include one of: { _PROBA_CANDS }")
    return df

def _dedup_preds(pred: pd.DataFrame, ts_col: str, sym_col: str, how: str) -> pd.DataFrame:
    pred = pred.dropna(subset=[ts_col, sym_col, "meta_p"]).copy()
    pred["ts"]  = _to_utc(pred[ts_col])
    pred["sym"] = pred[sym_col].astype(str)
    if how in ("mean","median","max","min"):
        agg = {"mean":"mean","median":"median","max":"max","min":"min"}[how]
        out = pred.groupby(["ts","sym"], as_index=False)["meta_p"].agg(agg)
    elif how in ("first","last"):
        ascending = (how == "first")
        sort_cols = ["sym","ts"] + (["fold"] if "fold" in pred.columns else [])
        p2 = pred.sort_values(sort_cols, ascending=ascending)
        out = p2.drop_duplicates(subset=["ts","sym"], keep="first")[["ts","sym","meta_p"]]
    else:
        raise ValueError(f"Unknown dedup method: {how}")
    return out

def _merge_meta(sig: pd.DataFrame, pred_agg: pd.DataFrame,
                round_to: str | None, asof_tol: str | None) -> pd.DataFrame:
    s = sig.copy()
    p = pred_agg.copy()
    if round_to:
        s["ts_round"] = s["ts"].dt.round(round_to)
        p["ts_round"] = p["ts"].dt.round(round_to)
        left_on = ["ts_round","sym"]; right_on = ["ts_round","sym"]
    else:
        left_on = ["ts","sym"]; right_on = ["ts","sym"]

    m = s.merge(p, left_on=left_on, right_on=right_on, how="left")
    matched = m["meta_p"].notna().sum()
    print(f"[merge] exact matches: {matched} / {len(m)} ({matched/max(len(m),1):.1%})")

    if matched < int(0.75 * len(m)) and asof_tol:
        print(f"[merge] trying per-symbol asof with tolerance={asof_tol}")
        # prepare for asof: sort by ts
        parts = []
        for sym, gsig in s.sort_values("ts").groupby("sym"):
            gpred = p[p["sym"] == sym].sort_values("ts")
            if gpred.empty:
                gsig["meta_p"] = pd.NA
                parts.append(gsig)
                continue
            mm = pd.merge_asof(gsig.sort_values("ts"), gpred[["ts","meta_p"]].sort_values("ts"),
                               on="ts", direction="nearest",
                               tolerance=pd.to_timedelta(asof_tol))
            parts.append(mm.assign(sym=sym))
        m = pd.concat(parts, ignore_index=True)
        matched = m["meta_p"].notna().sum()
        print(f"[merge] asof matches: {matched} / {len(m)} ({matched/max(len(m),1):.1%})")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", required=True)
    ap.add_argument("--pred",    required=True)
    ap.add_argument("--out",     required=True)
    ap.add_argument("--pstar",   type=float, required=True)

    # optional column overrides
    ap.add_argument("--proba-col", default=None)
    ap.add_argument("--pred-ts",   default=None)
    ap.add_argument("--pred-sym",  default=None)

    # timestamp handling
    ap.add_argument("--round", dest="round_to", default="5min",
                    help="round both timestamps to this freq before merge (e.g., 5min). '' disables")
    ap.add_argument("--tol",   dest="asof_tol", default="10min",
                    help="asof tolerance if exact merge is weak (e.g., 10min). '' disables")

    # dedup behavior
    ap.add_argument("--dedup", default="mean",
                    help="collapse duplicate preds per (ts,sym): mean|median|max|min|first|last")

    # optional re-backtest
    ap.add_argument("--rebt", action="store_true", help="run backtest immediately on the filtered signals")

    args = ap.parse_args()
    round_to = None if args.round_to in ("", None) else args.round_to
    asof_tol = None  if args.asof_tol in ("", "0", None) else args.asof_tol

    sig  = _read_signals(Path(args.signals))
    pred = _read_preds(Path(args.pred), args.proba_col)

    pred_ts  = args.pred_ts  or _guess(pred.columns.tolist(), _TS_CANDS)
    pred_sym = args.pred_sym or _guess(pred.columns.tolist(), _SYM_CANDS)
    if not (pred_ts and pred_sym):
        raise ValueError("Predictions must have a timestamp and symbol column "
                         f"(looked for { _TS_CANDS } / { _SYM_CANDS }).")

    pred_agg = _dedup_preds(pred, pred_ts, pred_sym, args.dedup)
    print(f"[dedup] preds unique (ts,sym): {len(pred_agg)}")

    merged = _merge_meta(sig, pred_agg, round_to=round_to, asof_tol=asof_tol)
    keep = merged[merged["meta_p"] >= float(args.pstar)].copy()
    n_before, n_after = len(merged), len(keep)
    print(f"[filter] p >= {args.pstar:.2f} → {n_after}/{n_before} signals kept ({n_after/max(n_before,1):.1%})")

    # save in the same schema as input signals + meta_p column
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep.drop(columns=[c for c in ["ts_round"] if c in keep.columns]).to_parquet(out_path, index=False)
    print(f"Saved filtered signals → {out_path}")


    if args.rebt:
        try:
            from backtester import run_backtest
            print("[rebt] launching backtest on filtered signals…")
            run_backtest(signals_path=str(out_path))
            # NEW: print a compact performance summary from results/*
            _print_bt_summary(Path("results"))
        except Exception as e:
            print(f"[rebt] failed to run backtest automatically: {e}\n"
                  f"Run manually with: python manager.py (or call run_backtest(signals_path='{out_path}') )")


if __name__ == "__main__":
    main()
