# bt_bt_analytics.py
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from typing import Iterable, Tuple
from math import sqrt
from scipy.stats import binomtest

def wr_ci(k: int, n: int, alpha=0.05) -> Tuple[float, float]:
    if n == 0: return (0.0, 1.0)
    bt = binomtest(k, n)
    return bt.proportion_ci(confidence_level=1-alpha, method="wilson").low, bt.proportion_ci(confidence_level=1-alpha, method="wilson").high

def profit_factor(p: pd.Series) -> float:
    g, l = p.clip(lower=0).sum(), -p.clip(upper=0).sum()
    return (g / l) if l > 0 else np.inf if g > 0 else 0.0

def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect columns from reporting.create_aggregated_report()
    # plus the extra entry features once you add them to backtester legs and aggregation.
    return df

def vwap_abs_distance_grid(df: pd.DataFrame, thresholds=(0.025, 0.02, 0.015, 0.01, 0.005)):
    print("\n" + "="*72 + "\n VWAP ABS DISTANCE GRID  (keep |pct_to_vwap| ≥ t)\n" + "="*72)
    rows = []
    for t in thresholds:
        sub = df[np.abs(df["pct_to_vwap"]) >= t]
        n = len(sub); wins = (sub["final_pnl"] > 0).sum()
        wr = wins / n if n else 0
        lwr, upr = wr_ci(wins, n)
        pf = profit_factor(sub["final_pnl"])
        rows.append([n, wins, wr, lwr, upr, pf, sub["final_pnl"].mean(), sub["final_pnl"].clip(lower=0).sum(), -sub["final_pnl"].clip(upper=0).sum(), f"|pct_to_vwap| ≥ {t}", n, len(df)-n])
    print(pd.DataFrame(rows, columns=["n","wins","win_rate","wr_lwr","wr_upr","pf","avg_pnl","gp","gl","rule","kept","dropped"]).to_string(index=False, float_format=lambda x: f"{x: .6f}"))

def vwap_z_veto_grid(df: pd.DataFrame, thresholds=(1.0,1.5,2.0,2.5)):
    print("\n" + "="*72 + "\n VWAP Z EXTREME VETO GRID  (keep |z| ≤ t)\n" + "="*72)
    rows=[]
    for t in thresholds:
        sub = df[np.abs(df["vwap_z_at_entry"]) <= t]
        n=len(sub); wins=(sub["final_pnl"]>0).sum(); wr=wins/n if n else 0
        lwr,upr=wr_ci(wins,n); pf=profit_factor(sub["final_pnl"])
        rows.append([n,wins,wr,lwr,upr,pf,sub["final_pnl"].mean(),sub["final_pnl"].clip(lower=0).sum(),-sub["final_pnl"].clip(upper=0).sum(), f"|vwap_z_at_entry| ≤ {t}", n, len(df)-n])
    print(pd.DataFrame(rows, columns=["n","wins","win_rate","wr_lwr","wr_upr","pf","avg_pnl","gp","gl","rule","kept","dropped"]).to_string(index=False, float_format=lambda x: f"{x: .6f}"))

def consolidation_gate(df: pd.DataFrame):
    print("\n" + "="*72 + "\n VWAP CONSOLIDATION GATE  (in-gate = consolidated=True)\n" + "="*72)
    in_g = df[df["vwap_consolidated_at_entry"]==True]
    out_g= df[df["vwap_consolidated_at_entry"]==False]
    def stats(x):
        n=len(x); wr=(x["final_pnl"]>0).mean() if n else 0; pf=profit_factor(x["final_pnl"])
        return n, wr, pf
    in_n,in_wr,in_pf = stats(in_g)
    out_n,out_wr,out_pf = stats(out_g)
    mde = abs(out_wr - in_wr)
    # simple z on WR difference
    p = (in_wr*in_n + out_wr*out_n) / max(1,(in_n+out_n))
    z = (out_wr - in_wr) / np.sqrt(max(1e-9, p*(1-p)*(1/in_n + 1/out_n)))
    print(pd.DataFrame([["VWAP_CONSOLIDATED", in_n, in_wr, in_pf, out_n, out_wr, out_pf, mde, z]],
                       columns=["gate","in_n","in_wr","in_pf","out_n","out_wr","out_pf","wr_mde_abs","z"]).to_string(index=False, float_format=lambda x: f"{x: .6f}"))

def hour_pooling(df: pd.DataFrame):
    print("\n" + "="*72 + "\n HOUR POOLING  (candidate blocked hours: WR<45%, n≥6)\n" + "="*72)
    tbl = df.groupby("hour_of_day_at_entry").agg(n=("final_pnl","size"), wins=("final_pnl", lambda s: (s>0).sum()))
    tbl["wr"]=tbl["wins"]/tbl["n"]
    print(tbl.to_string())
    weak = tbl[(tbl["n"]>=6)&(tbl["wr"]<0.45)].index.tolist()
    kept = df[~df["hour_of_day_at_entry"].isin(weak)]
    print(f"\nCandidate blocked hours: {weak}")
    print(f"What-if keep non-weak hours → N={len(kept)}  WR={ (kept['final_pnl']>0).mean(): .4f}  PF={ profit_factor(kept['final_pnl']):.3f}  Avg={ kept['final_pnl'].mean():.4f}")

def tp_sl_whatif(df: pd.DataFrame, tps=(1.0,2.0), sls=(1.2,1.5,1.8,2.0,2.5), tol=0.05):
    """
    Uses (a) exit reasons containing TP/TP1/TP_FINAL as TP hits at 1x and
    (b) MAE/MFE columns if present with ±5% tolerance on ATR multiples.
    Assumes columns: 'final_reason' (string), 'atr' (float at entry), plus optional 'mfe_over_atr','mae_over_atr'.
    avg_R and PF are computed in R units if mae/mfe provided; else counts from reasons only.
    """
    print("\n" + "="*72 + "\n PAYOFF WHAT-IF (TP/SL in ATR units; P&L in R)\n" + "="*72)
    cols = set(c.lower() for c in df.columns)
    has_mfe = "mfe_over_atr" in cols; has_mae = "mae_over_atr" in cols

    def row_hits_tp(r, tp):
        if isinstance(r.get("final_reason",""), str) and any(tag in r["final_reason"] for tag in ["TP","TP1","TP_FINAL","TP2"]):
            return tp <= 1.05  # treat 1x TP as valid within 5% tol
        if has_mfe and not pd.isna(r.get("mfe_over_atr", np.nan)):
            return r["mfe_over_atr"] >= (1 - tol) * tp
        return False

    def row_hits_sl(r, sl):
        if isinstance(r.get("final_reason",""), str) and "SL" in r["final_reason"]:
            return True if sl >= 2.0 else False  # crude mapping if only reason is known
        if has_mae and not pd.isna(r.get("mae_over_atr", np.nan)):
            return r["mae_over_atr"] >= (1 - tol) * sl
        return False

    records=[]
    for tp in tps:
        for sl in sls:
            wins=losses=unres=0
            R=[]
            for _, r in df.iterrows():
                hit_tp = row_hits_tp(r, tp)
                hit_sl = row_hits_sl(r, sl)
                if hit_tp and not hit_sl:
                    wins += 1; R.append(+tp/sl)  # reward in R units
                elif hit_sl and not hit_tp:
                    losses += 1; R.append(-1.0)  # -1R at SL
                elif hit_tp and hit_sl:
                    # unresolved ordering => midpoint assumption
                    wins += 0.5; losses += 0.5; R.append(0.0)
                else:
                    unres += 1
            n = len(df)
            wr = wins / max(1,n)
            avg_R = np.mean(R) if R else 0.0
            PF = (sum([x for x in R if x>0]) / -sum([x for x in R if x<0])) if any(x<0 for x in R) else (np.inf if any(x>0 for x in R) else 0.0)
            records.append(["midpoint", tp, sl, tp/sl, n, int(wins), int(losses), unres, wr, avg_R, PF])
    out = pd.DataFrame(records, columns=["policy","tp","sl","reward_R","n","wins","losses","unresolved","win_rate","avg_R","pf"])
    print(out.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    df = load_trades(args.csv)

    # Expect the following columns after you add features & aggregate legs:
    # pct_to_vwap, vwap_z_at_entry, vwap_consolidated_at_entry, hour_of_day_at_entry, final_pnl, final_reason, atr, mae_over_atr, mfe_over_atr
    for col in ["pct_to_vwap","vwap_z_at_entry","vwap_consolidated_at_entry","hour_of_day_at_entry","final_pnl"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column in aggregated trades: {col}")

    print("\n" + "="*72 + "\n OVERALL METRICS\n" + "="*72)
    N=len(df); wins=(df["final_pnl"]>0).sum()
    wr = wins / N if N else 0
    lwr,upr = wr_ci(wins,N)
    print(f"N: {N}   Wins: {wins}   WR:{wr:7.2%}  (95% CI {lwr*100:6.2f}%–{upr*100:6.2f}%)")
    print(f"PF: {profit_factor(df['final_pnl']):.3f}   Avg PnL: {df['final_pnl'].mean():.4f}")

    vwap_abs_distance_grid(df)
    vwap_z_veto_grid(df)
    consolidation_gate(df)
    hour_pooling(df)
    tp_sl_whatif(df)

if __name__ == "__main__":
    main()
