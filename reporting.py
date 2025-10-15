# reporting.py  — CPCV + PBO + PSR/DSR console reporting
from __future__ import annotations

import argparse
import itertools
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

from pathlib import Path
import numpy as np
import pandas as pd



import config as cfg

# =============================================================================
# I/O
# =============================================================================

def _results_dir() -> pd.Path:
    return cfg.RESULTS_DIR  # expected in your config.py

def load_trades(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load trades from CSV or Parquet.
    - Accepts str/Path; defaults to results/trades.csv if None.
    - Coerces datetime columns to tz-aware UTC.
    - Reconstructs pnl_R if it's missing but we can infer risk-per-unit.
    """
    if path is None:
        # fall back to the standard output location
        path = Path("results") / "trades.csv"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Trades file not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported trades format: {path.suffix}")

    # normalize datetimes (tz-aware UTC)
    for col in ("entry_ts", "exit_ts", "timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # ensure pnl_R exists; infer when possible
    if "pnl_R" not in df.columns:
        needed = {"entry", "sl", "qty", "pnl"}
        if needed.issubset(df.columns):
            risk_per_unit = (df["entry"] - df["sl"]).astype(float)
            denom = (risk_per_unit * df["qty"]).replace(0, np.nan)
            df["pnl_R"] = df["pnl"].astype(float) / denom
        else:
            # create present but empty to keep downstream code consistent
            df["pnl_R"] = np.nan

    return df

# =============================================================================
# Small stats helpers
# =============================================================================

def sharpe_unannualized(x: np.ndarray) -> float:
    """Sharpe on the provided frequency (e.g., daily matrix rows)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 2:
        return np.nan
    mu = x.mean()
    sd = x.std(ddof=1)
    return mu / sd if sd > 0 else np.nan

def skew_kurt(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size < 3:
        return np.nan, np.nan
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return np.nan, np.nan
    z = (x - m) / s
    skew = (z**3).mean()
    kurt = (z**4).mean()
    return float(skew), float(kurt)  # kurt (not excess)

def psr(sr_hat: float, sr_thresh: float, T: int, gamma3: float, gamma4: float) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado).
    PSR = Φ( (SR̂ - SR0)*sqrt(T-1) / sqrt(1 - γ3*SR̂ + ((γ4-1)/4)*SR̂^2) )
    """
    if any(np.isnan([sr_hat, sr_thresh, T, gamma3, gamma4])) or T <= 1:
        return np.nan
    num = (sr_hat - sr_thresh) * math.sqrt(T - 1.0)
    den = math.sqrt(max(1e-12, 1.0 - gamma3 * sr_hat + ((gamma4 - 1.0) / 4.0) * (sr_hat ** 2)))
    z = num / den
    # normal CDF
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def effective_num_trials_from_corr(corr: np.ndarray) -> float:
    """
    Very conservative N_eff using average off-diagonal correlation (ρ̄),
    consistent with the interpolation discussed in DSR materials:
      ρ→1 ⇒ N_eff→1 ;  ρ→0 ⇒ N_eff→M .
    We use N_eff = 1 + (M-1)*(1-ρ̄_offdiag), bounded to [1, M].
    """
    M = corr.shape[0]
    if M <= 1:
        return float(M)
    off = corr.copy()
    np.fill_diagonal(off, np.nan)
    rho = np.nanmean(off)
    if np.isnan(rho):
        rho = 0.0
    neff = 1.0 + (M - 1.0) * max(0.0, min(1.0, 1.0 - rho))
    return float(max(1.0, min(M, neff)))

def dsr(
    ret_mat_daily: pd.DataFrame,
    selected_col: str,
) -> Dict[str, float]:
    """
    Deflated Sharpe Ratio for the selected variant given the family of trials.
    - ret_mat_daily: rows=time (e.g., daily), cols=variants, values=daily returns (R-units or pct)
    - selected_col: which variant we are adjudicating

    Returns dict with SR_hat, SR0 (noise-max threshold), PSR0, DSR (PSR vs SR0), N_eff, etc.
    """
    # 1) SR across variants on full sample (unannualized)
    sr_by_col = ret_mat_daily.apply(lambda s: sharpe_unannualized(s.values), axis=0)
    sr_hat = float(sr_by_col.get(selected_col, np.nan))
    if np.isnan(sr_hat):
        return dict(sr_hat=np.nan, sr0=np.nan, dsr=np.nan, psr0=np.nan, neff=np.nan, t=np.nan)

    # 2) Effective number of trials from cross-variant correlations
    corr = ret_mat_daily.corr().values
    neff = effective_num_trials_from_corr(corr)

    # 3) Cross-sectional SR variance (used in SR0)
    sr_var = float(np.nanvar(sr_by_col.values, ddof=1)) if sr_by_col.notna().sum() > 1 else 0.0
    sr_sd = math.sqrt(max(sr_var, 1e-12))

    # Expected max of N standard normal ≈ (1-γ)Φ^-1(1-1/N) + γΦ^-1(1-1/(Ne)) (False Strategy theorem)
    # Then scale by SR dispersion.
    gamma = 0.5772156649015329  # Euler–Mascheroni
    from scipy.stats import norm  # local import; used only here
    term = (1.0 - gamma) * norm.ppf(1.0 - 1.0 / neff) + gamma * norm.ppf(1.0 - 1.0 / (neff * math.e))
    sr0 = sr_sd * term

    # 4) PSR and DSR for selected variant (with non-normal correction)
    x = ret_mat_daily[selected_col].values
    T = int(np.isfinite(x).sum())
    gamma3, gamma4 = skew_kurt(x)
    psr0 = psr(sr_hat, 0.0, T, gamma3, gamma4)         # vs 0 threshold
    dsr_val = psr(sr_hat, sr0, T, gamma3, gamma4)      # deflated vs SR0 threshold

    return dict(sr_hat=sr_hat, sr0=sr0, dsr=dsr_val, psr0=psr0, neff=neff, t=T)


# =============================================================================
# Build a “variant × daily returns” matrix from trades
# =============================================================================

def _pick_returns_col(df: pd.DataFrame, preferred: str | None) -> str:
    cands = [c for c in [preferred, "pnl_R", "r", "ret", "pnl"] if c and c in df.columns]
    if not cands:
        raise ValueError(
            "No returns column found. Provide one via --returns-col or include one of "
            "['pnl_R','r','ret','pnl'] in trades.csv"
        )
    return cands[0]

def _ensure_variant(df: pd.DataFrame, variant_cols: Optional[List[str]]) -> pd.DataFrame:
    df = df.copy()
    if "variant" in df.columns and not variant_cols:
        return df
    cols = variant_cols or []
    cols = [c for c in cols if c in df.columns]
    if not cols:
        # try a sensible default
        guess = [c for c in ["symbol", "entry_rule", "pullback_type", "don_break_len", "regime_up"] if c in df.columns]
        if not guess:
            # fallback to single-variant
            df["variant"] = "default"
            return df
        cols = guess
    df["variant"] = df[cols].astype(str).agg("|".join, axis=1)
    return df

def trades_to_daily_matrix(
    df: pd.DataFrame,
    returns_col: str,
    variant_cols: Optional[List[str]] = None,
    use_exit_ts: bool = True,
    fill_missing_days_with_zero: bool = True,
) -> pd.DataFrame:
    """
    Map trades → daily returns per variant.
    - If you used R-units for PnL (recommended), use that column as returns_col.
    - When there is no trade on a day, zero return is assumed (typical for trade-sum R).
    """
    df = _ensure_variant(df, variant_cols)
    ts_col = "exit_ts" if use_exit_ts and "exit_ts" in df.columns else ("entry_ts" if "entry_ts" in df.columns else None)
    if ts_col is None:
        raise ValueError("trades.csv needs entry_ts or exit_ts column.")
    if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    ret_col = returns_col
    df = df[[ts_col, "variant", ret_col]].copy()
    df["date"] = df[ts_col].dt.tz_convert("UTC").dt.floor("D")
    daily = df.groupby(["date", "variant"], observed=True)[ret_col].sum().unstack("variant")

    if fill_missing_days_with_zero:
        # fill holes per column with 0 (no trades that day)
        daily = daily.asfreq("D", fill_value=0.0)

    return daily.sort_index()


# =============================================================================
# CPCV / CSCV with purging + embargo
# =============================================================================

@dataclass
class CPCVConfig:
    blocks: int = 12          # number of contiguous time blocks
    k_test: int = 3           # how many blocks in test on each fold
    embargo: int = 1          # number of adjacent blocks to embargo on each side
    metric: str = "sharpe"    # 'sharpe' only for now (unannualized)

def _block_slices(n_rows: int, n_blocks: int) -> List[slice]:
    sizes = [n_rows // n_blocks] * n_blocks
    for i in range(n_rows % n_blocks):
        sizes[i] += 1
    idx = 0
    blocks = []
    for s in sizes:
        blocks.append(slice(idx, idx + s))
        idx += s
    return blocks

def _index_mask_from_blocks(n_rows: int, blocks: List[slice]) -> List[np.ndarray]:
    masks = []
    for sl in blocks:
        m = np.zeros(n_rows, dtype=bool)
        m[sl] = True
        masks.append(m)
    return masks

def _purged_train_mask(n_rows: int, test_ids: List[int], block_masks: List[np.ndarray], embargo: int) -> np.ndarray:
    test_mask = np.zeros(n_rows, dtype=bool)
    for j in test_ids:
        test_mask |= block_masks[j]
    train_mask = ~test_mask

    # embargo blocks adjacent to any test block
    for j in test_ids:
        for e in range(1, embargo + 1):
            if j - e >= 0:
                train_mask &= ~block_masks[j - e]
            if j + e < len(block_masks):
                train_mask &= ~block_masks[j + e]
    return train_mask

def _metric_series(ret_mat: pd.DataFrame, rows_mask: np.ndarray) -> pd.Series:
    sub = ret_mat.iloc[rows_mask, :]
    if sub.shape[0] < 2:
        return pd.Series(np.nan, index=sub.columns)
    if True:  # only 'sharpe' for now
        return sub.apply(lambda s: sharpe_unannualized(s.values), axis=0)

def cscv_pbo(
    ret_mat_daily: pd.DataFrame,
    cfg: CPCVConfig,
) -> Dict[str, object]:
    """
    CSCV/PBO per Bailey et al.:
      - Partition days into contiguous blocks.
      - For each combination of k_test blocks as OOS, select IS-optimal variant,
        compute its OOS rank percentile u among all variants; logit(u).
      - PBO = fraction of logits < 0 (i.e., u<0.5 → below-median OOS).
    """
    nT = ret_mat_daily.shape[0]
    if nT < cfg.blocks:
        raise ValueError(f"Not enough rows ({nT}) for blocks={cfg.blocks}. Reduce --blocks.")
    blocks = _block_slices(nT, cfg.blocks)
    block_masks = _index_mask_from_blocks(nT, blocks)

    cols = ret_mat_daily.columns.tolist()
    logits = []
    records = []

    for test_ids in itertools.combinations(range(cfg.blocks), cfg.k_test):
        test_ids = list(test_ids)
        train_mask = _purged_train_mask(nT, test_ids, block_masks, cfg.embargo)
        test_mask = np.zeros(nT, dtype=bool)
        for j in test_ids:
            test_mask |= block_masks[j]

        # skip if we purged away too much
        if train_mask.sum() < 5 or test_mask.sum() < 2:
            continue

        perf_is = _metric_series(ret_mat_daily, train_mask)
        best_col = perf_is.idxmax()
        perf_oos_all = _metric_series(ret_mat_daily, test_mask)

        # rank percentile of the chosen IS-optimal, within OOS distribution
        target = perf_oos_all[best_col]
        ranks = perf_oos_all.rank(method="average")
        u = float(ranks[best_col] / (ranks.notna().sum() + 1e-12))  # in (0,1]
        u = min(max(u, 1e-6), 1 - 1e-6)
        logit_u = math.log(u / (1.0 - u))
        logits.append(logit_u)
        records.append(dict(best=best_col, is_perf=float(perf_is[best_col]), oos_perf=float(target), u=u, logit=logit_u))

    if not logits:
        return dict(pbo=np.nan, logits=[], table=pd.DataFrame(), nfolds=0)

    logits = np.array(logits, float)
    pbo = float((logits < 0.0).mean())  # probability IS-optimal ranks below OOS median

    return dict(
        pbo=pbo,
        logits=logits,
        table=pd.DataFrame.from_records(records),
        nfolds=len(logits),
    )


# =============================================================================
# Console Reporter
# =============================================================================

def overall_kpis(df: pd.DataFrame, returns_col: str = "pnl") -> Dict[str, float]:
    x = df[returns_col].astype(float).values
    n = x.size
    wins = int((x > 0).sum())
    wr = wins / n if n else np.nan
    gp = x[x > 0].sum()
    gl = -x[x <= 0].sum()
    pf = (gp / gl) if gl > 0 else np.nan
    avg = float(np.nanmean(x)) if n else np.nan
    return dict(n=n, wins=wins, wr=wr, pf=pf, avg=avg)

def print_overall(df: pd.DataFrame, returns_col: str):
    m = overall_kpis(df, returns_col)
    wr_ci = wilson_ci(m["wins"], m["n"])
    print("\n===================== OVERALL =====================")
    print(f"N={m['n']}  Wins={m['wins']}  WR={m['wr']*100:5.2f}%  (95% CI {wr_ci[0]*100:5.2f}–{wr_ci[1]*100:5.2f}%)")
    print(f"PF={m['pf']:.3f}  Avg={m['avg']:.5f}")

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = 1.959963984540054
    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2*n)
    half = z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n)
    lower = (centre - half) / denom
    upper = (centre + half) / denom
    return (lower, upper)

def run_cpcv_and_dsr(
    trades_csv: Optional[str],
    returns_col: Optional[str],
    variant_cols: Optional[List[str]],
    blocks: int,
    k_test: int,
    embargo: int,
) -> None:
    df = load_trades(trades_csv)
    ret_col = _pick_returns_col(df, returns_col)
    print_overall(df, ret_col)

    # Build daily matrix
    ret_mat = trades_to_daily_matrix(df, ret_col, variant_cols=variant_cols)
    print(f"\nMatrix shape: days={ret_mat.shape[0]}  variants={ret_mat.shape[1]}")

    # Choose selected variant (best SR on full sample, transparently)
    sr_full = ret_mat.apply(lambda s: sharpe_unannualized(s.values), axis=0)
    sel = sr_full.idxmax()
    print(f"Selected variant by full-sample SR: {sel}  (SR={sr_full[sel]:.4f})")

    # DSR / PSR
    dsr_res = dsr(ret_mat, sel)
    print("\n===================== PSR / DSR =====================")
    print(f"PSR vs 0 Sharpe: {dsr_res['psr0']:.3f}  |  DSR (vs noise-max SR0): {dsr_res['dsr']:.3f}")
    print(f"SR_hat={dsr_res['sr_hat']:.4f}  SR0={dsr_res['sr0']:.4f}  N_eff≈{dsr_res['neff']:.1f}  T={int(dsr_res['t'])}")

    # CPCV / PBO
    cfg_cpcv = CPCVConfig(blocks=blocks, k_test=k_test, embargo=embargo)
    pbo_res = cscv_pbo(ret_mat, cfg_cpcv)
    print("\n===================== CSCV / PBO =====================")
    if pbo_res["nfolds"] == 0:
        print("Not enough data for CPCV with the requested settings.")
        return
    print(f"Splits={pbo_res['nfolds']}  PBO={pbo_res['pbo']:.3f}  (fraction of logits < 0)")
    q = np.quantile(pbo_res["logits"], [0.1, 0.5, 0.9])
    print(f"logit(u) quantiles: 10%={q[0]:.3f}  50%={q[1]:.3f}  90%={q[2]:.3f}")
    # Show top-5 worst folds by OOS performance of the IS-optimal
    tbl = pbo_res["table"].sort_values("oos_perf").head(5)
    if not tbl.empty:
        print("\nWorst folds (IS-optimal’s OOS metric):")
        print(tbl.to_string(index=False))

# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Reporting: CPCV/CSCV + PBO + PSR/DSR")
    ap.add_argument("--trades-csv", default=None, help="Override path to trades.csv")
    ap.add_argument("--returns-col", default=None, help="Which column to treat as returns (default: pnl_R>r>ret>pnl)")
    ap.add_argument("--variant-cols", nargs="*", default=None, help="Columns to concatenate into 'variant' if absent")
    ap.add_argument("--cpcv", action="store_true", help="Run CPCV/CSCV + PBO + PSR/DSR")
    ap.add_argument("--run-all", action="store_true", help="Alias to run CPCV + PSR/DSR")
    ap.add_argument("--blocks", type=int, default=12, help="Number of contiguous time blocks")
    ap.add_argument("--k-test", type=int, default=3, help="How many blocks in test for each split")
    ap.add_argument("--embargo", type=int, default=1, help="Embargo (in blocks) around each test block")
    args = ap.parse_args()

    if args.cpcv or args.run_all:
        run_cpcv_and_dsr(
            trades_csv=args.trades_csv,
            returns_col=args.returns_col,
            variant_cols=args.variant_cols,
            blocks=args.blocks,
            k_test=args.k_test,
            embargo=args.embargo,
        )
    else:
        # default: just show overall KPIs to avoid surprises
        df = load_trades(args.trades_csv)
        ret_col = _pick_returns_col(df, args.returns_col)
        print_overall(df, ret_col)
        print("\nTip: run with --cpcv (or --run-all) to compute CSCV/PBO and PSR/DSR.")

if __name__ == "__main__":
    main()
