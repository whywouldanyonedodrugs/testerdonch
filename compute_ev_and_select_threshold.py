# compute_ev_and_select_threshold.py
from __future__ import annotations
import argparse, numpy as np, pandas as pd
from pathlib import Path

_TS_CANDIDATES = ["timestamp", "entry_ts", "entry_time", "ts", "time", "dt"]
_SYM_CANDIDATES = ["symbol", "asset", "pair", "ticker"]
_ID_CANDIDATES  = ["trade_id", "id"]
_PROBA_CANDIDATES = ["meta_p","y_proba","proba","prob","p","y_pred","pred_proba"]

def _read_preds(pred_path: Path, proba_col: str | None) -> pd.DataFrame:
    df = pd.read_parquet(pred_path)
    if proba_col and proba_col in df.columns:
        return df.rename(columns={proba_col: "meta_p"})
    for c in _PROBA_CANDIDATES:
        if c in df.columns:
            return df.rename(columns={c: "meta_p"})
    # fallback: pick a single [0,1] float-like column if present
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() > 0.9 and (s.dropna().between(0,1)).all():
            return df.rename(columns={c: "meta_p"})
    raise ValueError("Predictions must contain a probability column. "
                     "Pass --proba-col or include one of: " + ",".join(_PROBA_CANDIDATES))

def _to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")

def _guess_column(cols: list[str], candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for k in candidates:
        if k in low:
            return low[k]
    return None

def _diagnose(df: pd.DataFrame, name: str):
    print(f"[diag] {name} columns: {list(df.columns)[:12]}{' ...' if len(df.columns)>12 else ''}")
    print(df.head(3).to_string(index=False), "\n")

def _dedup_preds(pred: pd.DataFrame, ts_col: str, sym_col: str, how: str) -> pd.DataFrame:
    pred = pred.dropna(subset=[ts_col, sym_col, "meta_p"]).copy()
    if how in ("mean","median","max","min"):
        agg = {"mean": "mean", "median": "median", "max": "max", "min": "min"}[how]
        out = (pred
               .groupby([ts_col, sym_col], as_index=False)["meta_p"]
               .agg(agg))
    elif how in ("first","last"):
        # keep first/last occurrence per (ts,sym)
        ascending = (how == "first")
        # if there's a 'fold' column, use it to define order; otherwise original order
        sort_cols = [sym_col, ts_col] + (["fold"] if "fold" in pred.columns else [])
        pred2 = pred.sort_values(sort_cols, ascending=ascending)
        out = pred2.drop_duplicates(subset=[ts_col, sym_col], keep="first")
        out = out[[ts_col, sym_col, "meta_p"]]
    else:
        raise ValueError(f"Unknown dedup method: {how}")
    return out

def _merge_preds_trades(
    pred_path: Path, trades_path: Path,
    proba_col: str | None,
    pred_id: str | None, trades_id: str | None,
    pred_ts: str | None, trades_ts: str | None,
    pred_sym: str | None, trades_sym: str | None,
    round_to: str | None, asof_tolerance: str | None,
    dedup: str = "mean",
) -> pd.DataFrame:

    pred = _read_preds(pred_path, proba_col).copy()
    tr   = pd.read_csv(trades_path).copy()

    _diagnose(pred, "pred")
    _diagnose(tr, "trades")

    # --- ID join if possible ---
    pred_id   = pred_id   or _guess_column(pred.columns.tolist(), _ID_CANDIDATES)
    trades_id = trades_id or _guess_column(tr.columns.tolist(),   _ID_CANDIDATES)
    if pred_id and trades_id and pred_id in pred and trades_id in tr:
        print(f"[merge] ID join: pred.{pred_id} == trades.{trades_id}")
        m = pd.merge(tr, pred[[pred_id,"meta_p"]], left_on=trades_id, right_on=pred_id,
                     how="inner")  # don't validate yet
        if "pnl_R" not in m.columns:
            raise ValueError("trades.csv must contain 'pnl_R'.")
        return m[["meta_p","pnl_R"]].dropna()

    # --- timestamp+symbol path ---
    pred_ts   = pred_ts   or _guess_column(pred.columns.tolist(), _TS_CANDIDATES)
    trades_ts = trades_ts or _guess_column(tr.columns.tolist(),   _TS_CANDIDATES)
    pred_sym  = pred_sym  or _guess_column(pred.columns.tolist(), _SYM_CANDIDATES)
    trades_sym= trades_sym or _guess_column(tr.columns.tolist(),  _SYM_CANDIDATES)
    if not (pred_ts and trades_ts and pred_sym and trades_sym):
        raise ValueError("Cannot find merge keys. Provide --pred-id/--trades-id or "
                         "--pred-ts/--pred-sym/--trades-ts/--trades-sym.")

    # normalize & round both sides
    pred[pred_ts] = _to_utc(pred[pred_ts])
    tr[trades_ts] = _to_utc(tr[trades_ts])
    if round_to:
        pred[pred_ts] = pred[pred_ts].dt.round(round_to)
        tr[trades_ts] = tr[trades_ts].dt.round(round_to)

    # --- deduplicate predictions on (ts,sym) ---
    before = len(pred)
    pred_agg = _dedup_preds(pred, pred_ts, pred_sym, dedup)
    after  = len(pred_agg)
    if after < before:
        print(f"[dedup] predictions: {before} → {after} unique (by {dedup} over (ts,sym))")

    # exact merge
    m = pd.merge(
        tr,
        pred_agg.rename(columns={pred_ts: "ts", pred_sym: "sym"}),
        left_on=[trades_ts, trades_sym], right_on=["ts","sym"],
        how="inner"  # relax validate; we deduped RHS already
    )
    ratio = len(m) / max(len(tr), 1)
    print(f"[merge] exact (ts,sym) matches: {len(m)} / {len(tr)} ({ratio:.1%})")

    # fallback: asof per symbol
    if ratio < 0.75 and asof_tolerance:
        print(f"[merge] trying merge_asof with tolerance={asof_tolerance}")
        pred2 = pred_agg.rename(columns={pred_ts:"ts", pred_sym:"sym"}).dropna().sort_values(["sym","ts"])
        tr2   = tr[[trades_ts,trades_sym]].rename(columns={trades_ts:"ts",trades_sym:"sym"}).dropna().sort_values(["sym","ts"])
        parts = []
        for sym, gtr in tr2.groupby("sym"):
            gpd = pred2[pred2["sym"]==sym]
            if gpd.empty:
                continue
            mm = pd.merge_asof(gtr, gpd, on="ts", direction="nearest",
                               tolerance=pd.to_timedelta(asof_tolerance))
            parts.append(mm)
        if parts:
            mm = pd.concat(parts, ignore_index=True).dropna(subset=["meta_p"])
            m2 = pd.merge(tr, mm[["ts","sym","meta_p"]],
                          left_on=[trades_ts,trades_sym], right_on=["ts","sym"],
                          how="inner")
            if len(m2) > len(m):
                m = m2
            print(f"[merge] asof matches: {len(m)} / {len(tr)} ({len(m)/max(len(tr),1):.1%})")

    if "pnl_R" not in tr.columns:
        raise ValueError("trades.csv must contain 'pnl_R'.")
    return m[["meta_p","pnl_R"]].dropna()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-trades", type=int, default=150)
    ap.add_argument("--grid", type=str, default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")

    # overrides
    ap.add_argument("--proba-col", default=None)
    ap.add_argument("--pred-id", default=None);   ap.add_argument("--trades-id", default=None)
    ap.add_argument("--pred-ts", default=None);   ap.add_argument("--trades-ts", default=None)
    ap.add_argument("--pred-sym", default=None);  ap.add_argument("--trades-sym", default=None)

    # timestamp handling
    ap.add_argument("--round", dest="round_to", default="5min")
    ap.add_argument("--tol",   dest="asof_tol",  default="10min")

    # dedup
    ap.add_argument("--dedup", default="mean",
                    help="how to collapse duplicate predictions per (ts,sym): mean|median|max|min|first|last")

    args = ap.parse_args()
    round_to = None if args.round_to in ("", None) else args.round_to
    asof_tol = None  if args.asof_tol in ("", "0", None) else args.asof_tol

    df = _merge_preds_trades(
        Path(args.pred), Path(args.trades),
        proba_col=args.proba_col,
        pred_id=args.pred_id, trades_id=args.trades_id,
        pred_ts=args.pred_ts, trades_ts=args.trades_ts,
        pred_sym=args.pred_sym, trades_sym=args.trades_sym,
        round_to=round_to, asof_tolerance=asof_tol,
        dedup=args.dedup,
    )

    thrs = [float(x) for x in args.grid.split(",")]
    rows = []
    for t in thrs:
        sub = df[df["meta_p"] >= t]
        rows.append({"threshold": t, "n": len(sub), "ev_R": float(sub["pnl_R"].mean()) if len(sub) else np.nan})
    evc = pd.DataFrame(rows).dropna().sort_values("threshold")

    cand = evc[evc["n"] >= args.min_trades]
    best = cand.loc[cand["ev_R"].idxmax()] if not cand.empty else evc.loc[evc["ev_R"].idxmax()]
    print("\n=== EV by threshold (R) ===")
    print(evc.to_string(index=False))
    print(f"\nSuggested p* = {best.threshold:.2f}  (EV={best.ev_R:.4f} R, trades={int(best.n)})")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        evc.to_csv(args.out, index=False)
        print(f"Saved EV curve → {args.out}")

if __name__ == "__main__":
    main()
