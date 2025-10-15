import pandas as pd, numpy as np

tr = pd.read_csv("results/trades.csv", parse_dates=["entry_ts","exit_ts"])
eq = pd.read_csv("results/equity.csv", parse_dates=["timestamp"])
init = 1000.0  # your INITIAL_CAPITAL

# 1) Reconcile final equity
pnl_sum = tr.get("pnl_cash", tr.get("pnl", pd.Series([np.nan]))).sum()
eq_end = float(eq["equity"].iloc[-1])
print("reconcile:", eq_end, "vs", init + pnl_sum, "diff=", eq_end - (init + pnl_sum))

# 2) Estimate per-trade unit risk and portfolio risk at entry
# Assumes you stored sl_initial; if not, recompute from size math you use.
if {"entry","sl_initial","qty"}.issubset(tr.columns):
    tr["unit_risk"] = (tr["entry"] - tr["sl_initial"]).clip(lower=0)
    tr["trade_risk_cash"] = tr["unit_risk"] * tr["qty"]
    # rough portfolio risk snapshot: sum of overlapping trade_risk_cash / equity_at_that_time
    events = []
    for i, r in tr.iterrows():
        events.append((r["entry_ts"], +r["trade_risk_cash"]))
        events.append((r["exit_ts"],  -r["trade_risk_cash"]))
    ev = (pd.DataFrame(events, columns=["ts","delta"])
            .sort_values("ts").dropna())
    ev["open_risk"] = ev["delta"].cumsum()
    # map equity (daily) to event times for ratio
    eqd = eq.set_index("timestamp")["equity"].resample("1H").last().ffill()
    ev["equity"] = ev["ts"].dt.floor("1H").map(eqd)
    ev["risk_pct_of_equity"] = ev["open_risk"] / ev["equity"]
    print("portfolio risk % (mean, p95, max):",
          ev["risk_pct_of_equity"].mean(),
          ev["risk_pct_of_equity"].quantile(0.95),
          ev["risk_pct_of_equity"].max())
