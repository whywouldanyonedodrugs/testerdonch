# summary_report.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity/peak - 1.0).min())

def sharpe_annualized(equity_df: pd.DataFrame, rf: float = 0.0, periods_per_year: int = 365) -> float:
    e = equity_df.sort_values("timestamp").copy()
    e["ret"] = e["equity"].pct_change().fillna(0.0)
    excess = e["ret"] - rf/periods_per_year
    mu, sd = excess.mean(), excess.std(ddof=0)
    return float(mu / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")

def profit_factor(r: pd.Series) -> float:
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    return float(gp / gl) if gl > 0 else float("inf")

def main():
    tfile = Path("results/trades.csv"); efile = Path("results/equity.csv")
    if not tfile.exists(): raise SystemExit("results/trades.csv not found")
    t = pd.read_csv(tfile, parse_dates=["entry_ts","exit_ts"])
    n = len(t)
    winrate = float((t["pnl_R"] > 0).mean()) if n else float("nan")
    ev_R = float(t["pnl_R"].mean()) if n else float("nan")
    pf = profit_factor(t["pnl_R"])
    avg_win = float(t.loc[t["pnl_R"]>0,"pnl_R"].mean()) if (t["pnl_R"]>0).any() else float("nan")
    avg_loss = float(t.loc[t["pnl_R"]<0,"pnl_R"].mean()) if (t["pnl_R"]<0).any() else float("nan")
    sharpe = mdd = float("nan")
    if efile.exists():
        e = pd.read_csv(efile, parse_dates=["timestamp"])
        sharpe = sharpe_annualized(e)
        mdd = max_drawdown(e["equity"])
    # optional prob bins
    bin_md = ""
    if "y_proba_cal" in t.columns:
        bins = [0.0,0.60,0.65,0.70,0.75,0.80,0.85,0.90,1.01]
        gb = t.groupby(pd.cut(t["y_proba_cal"].clip(0,1), bins=bins, right=False))["pnl_R"].agg(["count","mean"]).reset_index()
        gb = gb.rename(columns={"p_bin":"p_bin","count":"count","mean":"mean_R"}); gb["p_bin"] = gb["p_bin"].astype(str)
        bin_md = gb.rename(columns={"p_bin":"p_bin","count":"count","mean":"mean_R"}).to_markdown(index=False)

    md = f"""# Run Summary

**Window:** {t['entry_ts'].min()} → {t['entry_ts'].max()}  
**Trades:** {n:,}  
**Winrate:** {winrate:.2%}  
**EV (R):** {ev_R:.4f}  
**Profit Factor:** {pf:.3f}  
**Avg Win (R):** {avg_win:.3f}  
**Avg Loss (R):** {avg_loss:.3f}  
**Sharpe (ann.):** {sharpe:.3f}  
**Max Drawdown:** {mdd:.2%}

## Notes
- EV(R) = average R-multiple (PnL normalized by initial risk).
- Sharpe annualized from daily equity changes (crypto 24/7).
"""
    if bin_md:
        md += "\n## Performance by probability bins\n" + bin_md + "\n"
    Path("results/run_summary.md").write_text(md, encoding="utf-8")
    print(md)

if __name__ == "__main__":
    main()
