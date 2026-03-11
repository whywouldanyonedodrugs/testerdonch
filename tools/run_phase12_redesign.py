#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge


REPO = Path("/opt/testerdonch").resolve()
REPORTS = REPO / "reports" / "rebaseline"
ART = REPO / "artifacts" / "rebaseline"
ROOT = REPO / "results" / "rebaseline" / "phase12_redesigned_research_20260311"

BASELINE_DIR = REPO / "results" / "rebaseline" / "phase11_riskon_reintro_20260308" / "R0_baseline_locked"
VLOCK_DIR = ART / "BASELINE_VLOCK_v1"
SIGNALS_PATH = REPO / "results" / "rebaseline" / "phase11_riskon_reintro_20260308" / "_scoped_signals" / "signals.parquet"

START = pd.Timestamp("2025-06-01 00:00:00+00:00")
END = pd.Timestamp("2025-12-31 23:59:59+00:00")

BR_USAGE = ROOT / "branch_usage"
BR_REGIME = ROOT / "branch_regime_sentinel"
BR_TARGET = ROOT / "branch_target_redesign"
BR_COMBINED = ROOT / "branch_combined"


@dataclass
class PolicySpec:
    name: str
    family: str
    params: Dict[str, float]
    score_col: str
    regime_col: str | None = None


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ensure_dirs() -> None:
    for p in [REPORTS, ART, ROOT, BR_USAGE, BR_REGIME, BR_TARGET, BR_COMBINED]:
        p.mkdir(parents=True, exist_ok=True)


def _load_baseline_panel() -> pd.DataFrame:
    sig = pd.read_parquet(SIGNALS_PATH)
    sig["symbol"] = sig["symbol"].astype(str).str.upper()
    sig["decision_ts"] = _ts(sig["timestamp"])
    sig = sig.dropna(subset=["decision_ts"]).copy()
    sig = sig[(sig["decision_ts"] >= START) & (sig["decision_ts"] <= END)].copy()

    dec = pd.read_csv(BASELINE_DIR / "signal_decisions.csv", low_memory=False)
    dec["symbol"] = dec["symbol"].astype(str).str.upper()
    dec["decision_ts"] = _ts(dec["signal_ts"] if "signal_ts" in dec.columns else dec["timestamp"])
    dec = dec.dropna(subset=["decision_ts"]).copy()
    dec = dec[(dec["decision_ts"] >= START) & (dec["decision_ts"] <= END)].copy()
    for c in (
        "pnl",
        "pnl_R",
        "risk_on",
        "size_mult",
        "risk_cash_target",
        "risk_cash_realized",
        "final_notional",
        "target_notional_pre",
    ):
        if c in dec.columns:
            dec[c] = _num(dec[c])
    dec = dec.sort_values(["decision_ts", "symbol"]).drop_duplicates(["symbol", "decision_ts"], keep="last")

    panel = sig.merge(dec, on=["symbol", "decision_ts"], how="left", suffixes=("", "_dec"))
    panel["candidate_id"] = panel["symbol"].astype(str) + "|" + panel["decision_ts"].astype(str)
    panel["baseline_execute"] = panel.get("decision", "").astype(str).str.lower().eq("taken")
    panel["baseline_skip_reason"] = panel.get("reason", "").astype(str)
    panel["baseline_pnl"] = _num(panel.get("pnl", np.nan))
    panel["baseline_pnl_R"] = _num(panel.get("pnl_R", np.nan))
    panel["baseline_notional"] = _num(panel.get("final_notional", np.nan))
    panel["risk_on"] = _num(panel.get("risk_on", np.nan))
    panel["month"] = panel["decision_ts"].dt.to_period("M").astype(str)
    return panel.sort_values(["decision_ts", "symbol"]).reset_index(drop=True)


def _feature_cols(panel: pd.DataFrame) -> List[str]:
    prefer = [
        "rs_pct",
        "atr_1h",
        "rsi_1h",
        "adx_1h",
        "vol_mult",
        "atr_pct",
        "don_dist_atr",
        "funding_rate",
        "funding_z_7d",
        "oi_z_7d",
        "est_leverage",
        "btcusdt_vol_regime_level",
        "btcusdt_trend_slope",
        "eth_macd_signal_4h",
        "eth_macd_hist_4h",
        "risk_on",
    ]
    return [c for c in prefer if c in panel.columns]


def _context_cols(panel: pd.DataFrame) -> List[str]:
    prefer = [
        "btcusdt_vol_regime_level",
        "btcusdt_trend_slope",
        "eth_macd_signal_4h",
        "eth_macd_hist_4h",
        "funding_rate",
        "funding_z_7d",
        "oi_z_7d",
        "risk_on",
    ]
    return [c for c in prefer if c in panel.columns]


def _month_folds(exec_df: pd.DataFrame) -> List[Tuple[List[str], str]]:
    months = sorted(exec_df["month"].dropna().unique().tolist())
    folds: List[Tuple[List[str], str]] = []
    for i in range(1, len(months)):
        folds.append((months[:i], months[i]))
    return folds


def _fit_score_by_target(exec_df: pd.DataFrame, feat_cols: List[str], target_mode: str) -> pd.Series:
    df = exec_df.copy().reset_index(drop=True)
    scores = pd.Series(np.nan, index=df.index, dtype=float)
    folds = _month_folds(df)
    if not folds:
        return scores

    for train_months, valid_month in folds:
        tr = df[df["month"].isin(train_months)].copy()
        va = df[df["month"] == valid_month].copy()
        if tr.empty or va.empty:
            continue

        Xtr = tr[feat_cols].apply(_num).replace([np.inf, -np.inf], np.nan)
        Xva = va[feat_cols].apply(_num).replace([np.inf, -np.inf], np.nan)
        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med).fillna(0.0)
        Xva = Xva.fillna(med).fillna(0.0)

        if target_mode == "T1_downside_avoid":
            ytr = (_num(tr["baseline_pnl_R"]) <= -0.5).astype(int)
            if ytr.nunique() < 2:
                pred = np.full(len(va), float(ytr.mean()))
            else:
                clf = LogisticRegression(max_iter=1000, C=0.5)
                clf.fit(Xtr, ytr)
                p_bad = clf.predict_proba(Xva)[:, 1]
                pred = 1.0 - p_bad  # quality score
        elif target_mode == "T2_relative_quality":
            # relative to train context median by risk_on bucket
            tr = tr.copy()
            med_by_ctx = tr.groupby("risk_on")["baseline_pnl_R"].median().to_dict()
            ref = tr["risk_on"].map(med_by_ctx)
            ytr = (_num(tr["baseline_pnl_R"]) > _num(ref)).astype(int)
            if ytr.nunique() < 2:
                pred = np.full(len(va), float(ytr.mean()))
            else:
                clf = LogisticRegression(max_iter=1000, C=0.5)
                clf.fit(Xtr, ytr)
                pred = clf.predict_proba(Xva)[:, 1]  # quality score
        elif target_mode == "T3_continuous_utility":
            ytr = _num(tr["baseline_pnl_R"]).clip(-2.0, 2.0).fillna(0.0)
            reg = Ridge(alpha=1.0)
            reg.fit(Xtr, ytr)
            pred = reg.predict(Xva)  # quality score (higher better)
        else:
            raise ValueError(target_mode)

        scores.loc[va.index] = pred

    # fallback for earliest month(s)
    if scores.isna().any():
        v = float(scores.dropna().median()) if scores.notna().any() else 0.5
        scores = scores.fillna(v)
    return scores


def _fit_regime_risk_score(exec_df: pd.DataFrame, context_cols: List[str]) -> pd.Series:
    df = exec_df.copy().reset_index(drop=True)
    risk = pd.Series(np.nan, index=df.index, dtype=float)
    folds = _month_folds(df)
    if not folds:
        return risk

    for train_months, valid_month in folds:
        tr = df[df["month"].isin(train_months)].copy()
        va = df[df["month"] == valid_month].copy()
        if tr.empty or va.empty:
            continue

        Xtr = tr[context_cols].apply(_num).replace([np.inf, -np.inf], np.nan)
        Xva = va[context_cols].apply(_num).replace([np.inf, -np.inf], np.nan)
        med = Xtr.median(numeric_only=True)
        Xtr = Xtr.fillna(med).fillna(0.0)
        Xva = Xva.fillna(med).fillna(0.0)

        ytr = (_num(tr["baseline_pnl_R"]) <= -0.5).astype(int)
        if ytr.nunique() < 2:
            pred = np.full(len(va), float(ytr.mean()))
        else:
            clf = LogisticRegression(max_iter=1000, C=0.5)
            clf.fit(Xtr, ytr)
            pred = clf.predict_proba(Xva)[:, 1]  # risk score
        risk.loc[va.index] = pred

    if risk.isna().any():
        v = float(risk.dropna().median()) if risk.notna().any() else 0.5
        risk = risk.fillna(v)
    return risk


def _apply_policy_to_row(
    r: pd.Series,
    score_col: str,
    policy: PolicySpec,
    rolling_thr: Dict[str, float] | None = None,
) -> Tuple[str, float, bool]:
    # returns (policy_decision, size_mult, execute_flag)
    if not bool(r.get("baseline_execute", False)):
        return "baseline_skip", 0.0, False

    s = float(r.get(score_col, np.nan))
    if not np.isfinite(s):
        return "keep_no_score", 1.0, True

    fam = policy.family
    p = policy.params
    if fam == "bottom_veto":
        thr = float(p["thr"])
        if s <= thr:
            return "veto_bottom", 0.0, False
        return "keep", 1.0, True
    if fam == "score_to_size":
        ql = float(p["q_low"])
        qh = float(p["q_high"])
        if s <= ql:
            return "size_down_low", float(p["m_low"]), True
        if s >= qh:
            return "size_up_high", float(p["m_high"]), True
        return "keep_mid", float(p["m_mid"]), True
    if fam == "state_conditional_veto":
        thr_bad = float(p["thr_bad"])
        thr_good = float(p["thr_good"])
        ru = _num(pd.Series([r.get("risk_on", np.nan)])).iloc[0]
        thr = thr_bad if (np.isnan(ru) or ru <= 0.0) else thr_good
        if s <= thr:
            return "veto_state_conditional", 0.0, False
        return "keep", 1.0, True
    if fam == "rolling_quantile_veto":
        month = str(r.get("month"))
        thr = rolling_thr.get(month, np.nan) if rolling_thr else np.nan
        if np.isfinite(thr) and s <= thr:
            return "veto_rolling_q", 0.0, False
        return "keep", 1.0, True
    if fam == "regime_downside_controller":
        rg = float(r.get(policy.regime_col, np.nan))
        q80 = float(p["q80"])
        q90 = float(p["q90"])
        if np.isfinite(rg) and rg >= q90:
            return "veto_regime_high_risk", 0.0, False
        if np.isfinite(rg) and rg >= q80:
            return "size_down_regime_risk", float(p["m_risk"]), True
        return "keep", 1.0, True
    if fam == "context_conditioned_usage":
        rg = float(r.get(policy.regime_col, np.nan))
        qctx = float(p["q_ctx"])
        if np.isfinite(rg) and rg >= qctx:
            thr = float(p["thr_strict"])
            if s <= thr:
                return "veto_context_conditioned", 0.0, False
            return "keep_context_conditioned", 1.0, True
        return "keep_no_context_trigger", 1.0, True
    return "keep_default", 1.0, True


def _rolling_thresholds(exec_df: pd.DataFrame, score_col: str, q: float) -> Dict[str, float]:
    months = sorted(exec_df["month"].unique().tolist())
    out: Dict[str, float] = {}
    for i, m in enumerate(months):
        hist = exec_df[exec_df["month"].isin(months[:i])][score_col].dropna()
        if len(hist) < 20:
            out[m] = float(exec_df[score_col].quantile(q))
        else:
            out[m] = float(hist.quantile(q))
    return out


def _evaluate_candidate(panel: pd.DataFrame, cand_name: str, policy: PolicySpec, score_col: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    d = panel.copy()
    exec_rows = d[d["baseline_execute"]].copy()
    rolling_thr = None
    if policy.family == "rolling_quantile_veto":
        rolling_thr = _rolling_thresholds(exec_rows, score_col, float(policy.params["q"]))

    decisions = []
    mults = []
    execute_flags = []
    for _, r in d.iterrows():
        dec, m, ex = _apply_policy_to_row(r, score_col, policy, rolling_thr=rolling_thr)
        decisions.append(dec)
        mults.append(m)
        execute_flags.append(ex)
    d["policy_decision"] = decisions
    d["policy_size_mult"] = mults
    d["final_execute"] = execute_flags
    d["skip_reason"] = np.where(d["final_execute"], "", d["policy_decision"])

    base_pnl = _num(d["baseline_pnl"]).fillna(0.0)
    d["challenger_pnl"] = np.where(d["final_execute"], base_pnl * _num(d["policy_size_mult"]).fillna(1.0), 0.0)
    d["challenger_notional"] = np.where(
        d["final_execute"],
        _num(d["baseline_notional"]).fillna(0.0) * _num(d["policy_size_mult"]).fillna(1.0),
        0.0,
    )
    d["pnl_removed"] = base_pnl - d["challenger_pnl"]

    b = d[d["baseline_execute"]].copy()
    c = d[d["final_execute"]].copy()

    baseline_net = float(_num(b["baseline_pnl"]).sum())
    challenger_net = float(_num(d["challenger_pnl"]).sum())
    delta_net = challenger_net - baseline_net

    def _maxdd(series: pd.Series) -> float:
        eq = 2000.0 + _num(series).fillna(0.0).cumsum()
        dd = eq - eq.cummax()
        return float(dd.min()) if len(dd) else 0.0

    baseline_dd = _maxdd(b["baseline_pnl"])
    challenger_dd = _maxdd(d["challenger_pnl"])
    delta_dd = challenger_dd - baseline_dd

    # upside deletion accounting
    pos_base = float(_num(b["baseline_pnl"]).clip(lower=0).sum())
    removed_pos = float(_num(d.loc[d["baseline_execute"], "pnl_removed"]).clip(lower=0).sum())
    removed_neg = float((-_num(d.loc[d["baseline_execute"], "pnl_removed"]).clip(upper=0)).sum())
    upside_ratio = float(removed_pos / max(1e-9, pos_base))

    fav_mask = _num(d["risk_on"]).fillna(0.0) > 0.0
    base_fav = float(_num(d.loc[d["baseline_execute"] & fav_mask, "baseline_pnl"]).sum())
    chal_fav = float(_num(d.loc[d["final_execute"] & fav_mask, "challenger_pnl"]).sum())
    base_unf = float(_num(d.loc[d["baseline_execute"] & ~fav_mask, "baseline_pnl"]).sum())
    chal_unf = float(_num(d.loc[d["final_execute"] & ~fav_mask, "challenger_pnl"]).sum())

    # breadth / support
    active_months = int(c["month"].nunique())
    symbol_cov = int(c["symbol"].nunique())
    selected_trade_count = int(len(c))
    month_delta = (
        d.groupby("month", as_index=False)
        .agg(base=("baseline_pnl", "sum"), chal=("challenger_pnl", "sum"))
        .assign(delta=lambda x: _num(x["chal"]) - _num(x["base"]))
    )
    folds_beating = int((month_delta["delta"] > 0).sum())
    folds_total = int(len(month_delta))

    metrics = {
        "candidate": cand_name,
        "family": policy.family,
        "score_col": score_col,
        "baseline_net_pnl": baseline_net,
        "challenger_net_pnl": challenger_net,
        "delta_net_pnl": delta_net,
        "baseline_max_dd": baseline_dd,
        "challenger_max_dd": challenger_dd,
        "delta_max_dd": delta_dd,
        "baseline_trades": int(len(b)),
        "challenger_trades": selected_trade_count,
        "delta_trades": selected_trade_count - int(len(b)),
        "baseline_turnover": float(_num(b["baseline_notional"]).fillna(0.0).sum()),
        "challenger_turnover": float(_num(d["challenger_notional"]).fillna(0.0).sum()),
        "delta_turnover": float(_num(d["challenger_notional"]).fillna(0.0).sum() - _num(b["baseline_notional"]).fillna(0.0).sum()),
        "baseline_favorable_pnl": base_fav,
        "challenger_favorable_pnl": chal_fav,
        "delta_favorable_pnl": chal_fav - base_fav,
        "baseline_unfavorable_pnl": base_unf,
        "challenger_unfavorable_pnl": chal_unf,
        "delta_unfavorable_pnl": chal_unf - base_unf,
        "removed_positive_pnl": removed_pos,
        "removed_negative_pnl": removed_neg,
        "upside_deletion_ratio": upside_ratio,
        "selected_trade_count": selected_trade_count,
        "active_months": active_months,
        "symbol_coverage": symbol_cov,
        "validation_windows_beating_r0": folds_beating,
        "validation_windows_total": folds_total,
    }
    return d, metrics


def _is_disqualified(m: Dict[str, float]) -> bool:
    if m["upside_deletion_ratio"] > 0.55:
        return True
    if m["selected_trade_count"] < 100:
        return True
    if m["active_months"] < 4:
        return True
    if m["validation_windows_beating_r0"] < max(1, int(0.4 * m["validation_windows_total"])):
        return True
    return False


def _write_anchor_and_common_docs(commit: str) -> None:
    anchor = [
        "# PHASE12 Anchor Declaration v1",
        "",
        "1) official anchor = `R0_baseline_locked`",
        "2) all new challengers in Phase-12 branch from this anchor only",
        "3) no promotion decisions allowed until a challenger clearly beats this anchor under Phase-12 gates",
        "",
        f"- commit: `{commit}`",
        f"- baseline_bundle: `{VLOCK_DIR}`",
    ]
    (REPORTS / "PHASE12_ANCHOR_DECLARATION_v1.md").write_text("\n".join(anchor) + "\n", encoding="utf-8")

    common = [
        "# PHASE12 Common Eval Rules v1",
        "",
        "All candidates are evaluated as deltas over R0, not in isolation.",
        "",
        "Required metrics:",
        "- net/gross pnl delta",
        "- max-drawdown delta",
        "- trade-count/turnover delta",
        "- favorable/unfavorable state pnl deltas",
        "- upside deletion accounting",
        "- breadth/support (trades, months, symbols, fold support)",
        "",
        "Hard disqualification:",
        "- excessive favorable-upside deletion",
        "- too little support",
        "- inconsistent fold support vs R0",
        "- missing decision-ledger attribution",
    ]
    (REPORTS / "PHASE12_COMMON_EVAL_RULES_v1.md").write_text("\n".join(common) + "\n", encoding="utf-8")

    schema = [
        "# PHASE12 Decision Ledger Schema v1",
        "",
        "Mandatory columns:",
        "- decision_ts",
        "- symbol",
        "- raw candidate present",
        "- state/regime context",
        "- base size inputs",
        "- regime-sentinel score (if any)",
        "- trade-quality score (if any)",
        "- policy decision",
        "- final execute/skip decision",
        "- skip reason",
        "- entry/exit/pnl if traded",
    ]
    (REPORTS / "PHASE12_DECISION_LEDGER_SCHEMA_v1.md").write_text("\n".join(schema) + "\n", encoding="utf-8")


def _write_promotion_gates() -> None:
    lines = [
        "# PHASE12 Promotion Gates v1",
        "",
        "A candidate may influence promotion decisions only if all pass:",
        "1) beats R0 in a majority of validation windows on net-pnl and/or DD-adjusted terms",
        "2) sufficient support (trade count, month coverage, symbol coverage)",
        "3) no catastrophic favorable-state upside deletion",
        "4) fully explainable by decision ledger",
        "5) quantile logic is live-faithful (rolling/historical only)",
        "6) no tiny-support validation artifacts",
    ]
    (REPORTS / "PHASE12_PROMOTION_GATES_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    _ensure_dirs()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    _write_anchor_and_common_docs(commit)
    _write_promotion_gates()

    panel = _load_baseline_panel()
    exec_df = panel[panel["baseline_execute"]].copy().reset_index(drop=True)
    feat_cols = _feature_cols(exec_df)
    ctx_cols = _context_cols(exec_df)

    # Target-derived scores (OOF by month folds on executed trades only)
    exec_df["score_T1"] = _fit_score_by_target(exec_df, feat_cols, "T1_downside_avoid")
    exec_df["score_T2"] = _fit_score_by_target(exec_df, feat_cols, "T2_relative_quality")
    exec_df["score_T3"] = _fit_score_by_target(exec_df, feat_cols, "T3_continuous_utility")
    exec_df["regime_risk_score"] = _fit_regime_risk_score(exec_df, ctx_cols)

    # Push scores back to full candidate panel
    score_map = exec_df[["candidate_id", "score_T1", "score_T2", "score_T3", "regime_risk_score"]].copy()
    panel = panel.merge(score_map, on="candidate_id", how="left")

    # ---------------- Branch 1: usage redesign ----------------
    usage_plan = [
        "# PHASE12 Usage Redesign Plan v1",
        "",
        "Anchor: R0_baseline_locked",
        "Branch-1 scope: usage redesign only (no target/model family redesign in this branch).",
        "",
        "Families tested:",
        "- U1 bottom-decile veto: q={5,10,15,20} on score_T1",
        "- U2 score-to-size modulation on score_T1",
        "- U3 state-conditional veto on score_T1",
        "- U4 rolling quantile veto on score_T1",
    ]
    (REPORTS / "PHASE12_USAGE_REDESIGN_PLAN_v1.md").write_text("\n".join(usage_plan) + "\n", encoding="utf-8")

    q05 = float(exec_df["score_T1"].quantile(0.05))
    q10 = float(exec_df["score_T1"].quantile(0.10))
    q15 = float(exec_df["score_T1"].quantile(0.15))
    q20 = float(exec_df["score_T1"].quantile(0.20))
    q30 = float(exec_df["score_T1"].quantile(0.30))
    q70 = float(exec_df["score_T1"].quantile(0.70))

    usage_specs = [
        PolicySpec("U1_bottom_veto_q05", "bottom_veto", {"thr": q05}, "score_T1"),
        PolicySpec("U1_bottom_veto_q10", "bottom_veto", {"thr": q10}, "score_T1"),
        PolicySpec("U1_bottom_veto_q15", "bottom_veto", {"thr": q15}, "score_T1"),
        PolicySpec("U1_bottom_veto_q20", "bottom_veto", {"thr": q20}, "score_T1"),
        PolicySpec("U2_score_to_size", "score_to_size", {"q_low": q30, "q_high": q70, "m_low": 0.40, "m_mid": 1.00, "m_high": 1.10}, "score_T1"),
        PolicySpec("U3_state_conditional_veto", "state_conditional_veto", {"thr_bad": q20, "thr_good": q05}, "score_T1"),
        PolicySpec("U4_rolling_quantile_veto_q10", "rolling_quantile_veto", {"q": 0.10}, "score_T1"),
    ]

    usage_rows = []
    usage_best_name = ""
    usage_best_score = -1e18
    for spec in usage_specs:
        led, met = _evaluate_candidate(panel, spec.name, spec, score_col=spec.score_col)
        met["disqualified"] = _is_disqualified(met)
        usage_rows.append(met)
        out = BR_USAGE / spec.name
        out.mkdir(parents=True, exist_ok=True)
        led.to_csv(out / "decision_ledger.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(met, indent=2) + "\n", encoding="utf-8")
        if not met["disqualified"]:
            score = met["delta_net_pnl"] - 0.25 * met["removed_positive_pnl"]
            if score > usage_best_score:
                usage_best_score = score
                usage_best_name = spec.name
    usage_df = pd.DataFrame(usage_rows).sort_values("delta_net_pnl", ascending=False).reset_index(drop=True)
    usage_df.to_csv(ART / "PHASE12_USAGE_REDESIGN_MASTER_TABLE_v1.csv", index=False)
    usage_best_row = usage_df[usage_df["candidate"] == usage_best_name].iloc[0] if usage_best_name else usage_df.iloc[0]
    usage_memo = [
        "# PHASE12 Usage Redesign Decision Memo v1",
        "",
        usage_df.to_markdown(index=False),
        "",
        f"1) Least destructive usage mode: `{usage_best_row['candidate']}`",
        f"2) Highest value-add over R0: `{usage_df.sort_values('delta_net_pnl', ascending=False).iloc[0]['candidate']}`",
        f"3) July-style upside deletion best controlled by: `{usage_df.sort_values('upside_deletion_ratio').iloc[0]['candidate']}`",
        f"4) Carry-forward usage mode: `{usage_best_row['candidate']}`",
    ]
    (REPORTS / "PHASE12_USAGE_REDESIGN_DECISION_MEMO_v1.md").write_text("\n".join(usage_memo) + "\n", encoding="utf-8")

    # ---------------- Branch 2: regime-sentinel ----------------
    reg_plan = [
        "# PHASE12 Regime Sentinel Design v1",
        "",
        "Sentinel is context-oriented (4h/daily-style fields), evaluated as downside controller/context conditioner.",
        "",
        f"Context features: `{ctx_cols}`",
    ]
    (REPORTS / "PHASE12_REGIME_SENTINEL_DESIGN_v1.md").write_text("\n".join(reg_plan) + "\n", encoding="utf-8")

    q80 = float(exec_df["regime_risk_score"].quantile(0.80))
    q90 = float(exec_df["regime_risk_score"].quantile(0.90))
    q70_ctx = float(exec_df["regime_risk_score"].quantile(0.70))
    reg_specs = [
        PolicySpec("RGS1_context_score_only", "score_to_size", {"q_low": -1e9, "q_high": 1e9, "m_low": 1.0, "m_mid": 1.0, "m_high": 1.0}, "score_T1", "regime_risk_score"),
        PolicySpec("RGS2_downside_controller", "regime_downside_controller", {"q80": q80, "q90": q90, "m_risk": 0.50}, "score_T1", "regime_risk_score"),
        PolicySpec("RGS3_context_conditioned_usage", "context_conditioned_usage", {"q_ctx": q70_ctx, "thr_strict": q10}, "score_T1", "regime_risk_score"),
    ]

    reg_rows = []
    reg_best = ""
    reg_best_score = -1e18
    for spec in reg_specs:
        led, met = _evaluate_candidate(panel, spec.name, spec, score_col=spec.score_col)
        met["disqualified"] = _is_disqualified(met)
        reg_rows.append(met)
        out = BR_REGIME / spec.name
        out.mkdir(parents=True, exist_ok=True)
        led.to_csv(out / "decision_ledger.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(met, indent=2) + "\n", encoding="utf-8")
        if not met["disqualified"]:
            score = met["delta_net_pnl"] - 0.25 * met["removed_positive_pnl"]
            if score > reg_best_score:
                reg_best_score = score
                reg_best = spec.name
    reg_df = pd.DataFrame(reg_rows).sort_values("delta_net_pnl", ascending=False).reset_index(drop=True)
    reg_df.to_csv(ART / "PHASE12_REGIME_SENTINEL_MASTER_TABLE_v1.csv", index=False)
    reg_best_row = reg_df[reg_df["candidate"] == reg_best].iloc[0] if reg_best else reg_df.iloc[0]
    reg_memo = [
        "# PHASE12 Regime Sentinel Decision Memo v1",
        "",
        reg_df.to_markdown(index=False),
        "",
        f"1) Is regime deterioration predictable enough? `{'yes' if reg_df['validation_windows_beating_r0'].max() >= 4 else 'mixed'}`",
        f"2) Does regime-sentinel explain loss flurries better than trade-score alone? `{'yes' if reg_best_row['delta_unfavorable_pnl'] > 0 else 'mixed_or_no'}`",
        f"3) Recommended role: `{'downside_controller' if 'RGS2' in reg_best_row['candidate'] else ('context_conditioned_usage' if 'RGS3' in reg_best_row['candidate'] else 'context_only')}`",
    ]
    (REPORTS / "PHASE12_REGIME_SENTINEL_DECISION_MEMO_v1.md").write_text("\n".join(reg_memo) + "\n", encoding="utf-8")

    # ---------------- Branch 3: target redesign ----------------
    tgt_plan = [
        "# PHASE12 Target Redesign Plan v1",
        "",
        "Targets tested with controlled usage mode (bottom 10% veto on target-specific quality score):",
        "- T1 downside-avoidance binary",
        "- T2 relative-quality binary",
        "- T3 continuous utility target",
    ]
    (REPORTS / "PHASE12_TARGET_REDESIGN_PLAN_v1.md").write_text("\n".join(tgt_plan) + "\n", encoding="utf-8")

    tgt_specs = [
        PolicySpec("T1_downside_bottom10", "bottom_veto", {"thr": float(exec_df["score_T1"].quantile(0.10))}, "score_T1"),
        PolicySpec("T2_relative_bottom10", "bottom_veto", {"thr": float(exec_df["score_T2"].quantile(0.10))}, "score_T2"),
        PolicySpec("T3_utility_bottom10", "bottom_veto", {"thr": float(exec_df["score_T3"].quantile(0.10))}, "score_T3"),
    ]
    tgt_rows = []
    tgt_best = ""
    tgt_best_score = -1e18
    for spec in tgt_specs:
        led, met = _evaluate_candidate(panel, spec.name, spec, score_col=spec.score_col)
        met["disqualified"] = _is_disqualified(met)
        tgt_rows.append(met)
        out = BR_TARGET / spec.name
        out.mkdir(parents=True, exist_ok=True)
        led.to_csv(out / "decision_ledger.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(met, indent=2) + "\n", encoding="utf-8")
        if not met["disqualified"]:
            score = met["delta_net_pnl"] - 0.25 * met["removed_positive_pnl"]
            if score > tgt_best_score:
                tgt_best_score = score
                tgt_best = spec.name
    tgt_df = pd.DataFrame(tgt_rows).sort_values("delta_net_pnl", ascending=False).reset_index(drop=True)
    tgt_df.to_csv(ART / "PHASE12_TARGET_REDESIGN_MASTER_TABLE_v1.csv", index=False)
    tgt_best_row = tgt_df[tgt_df["candidate"] == tgt_best].iloc[0] if tgt_best else tgt_df.iloc[0]
    tgt_memo = [
        "# PHASE12 Target Redesign Decision Memo v1",
        "",
        tgt_df.to_markdown(index=False),
        "",
        f"1) Best target for economic problem: `{tgt_best_row['candidate']}`",
        f"2) Does downside-avoidance beat top-winner style? `{'yes' if 'T1' in tgt_best_row['candidate'] else 'no_or_mixed'}`",
        f"3) Best favorable-upside preservation while reducing bad clusters: `{tgt_df.sort_values('upside_deletion_ratio').iloc[0]['candidate']}`",
    ]
    (REPORTS / "PHASE12_TARGET_REDESIGN_DECISION_MEMO_v1.md").write_text("\n".join(tgt_memo) + "\n", encoding="utf-8")

    # ---------------- Branch 4: combined best ----------------
    # combine: best usage mode + best regime-sentinel + best target score
    best_usage = usage_best_name if usage_best_name else "U1_bottom_veto_q10"
    best_reg = reg_best if reg_best else "RGS2_downside_controller"
    if tgt_best:
        best_tgt_score = "score_T1" if "T1" in tgt_best else ("score_T2" if "T2" in tgt_best else "score_T3")
    else:
        best_tgt_score = "score_T1"

    # Combined policy = context-conditioned usage with selected score.
    # reuse RGS3 form with target-specific score.
    combined_spec = PolicySpec(
        "C1_combined_best",
        "context_conditioned_usage",
        {"q_ctx": q70_ctx, "thr_strict": float(exec_df[best_tgt_score].quantile(0.10))},
        best_tgt_score,
        "regime_risk_score",
    )
    led_c, met_c = _evaluate_candidate(panel, combined_spec.name, combined_spec, score_col=combined_spec.score_col)
    met_c["disqualified"] = _is_disqualified(met_c)
    outc = BR_COMBINED / combined_spec.name
    outc.mkdir(parents=True, exist_ok=True)
    led_c.to_csv(outc / "decision_ledger.csv", index=False)
    (outc / "metrics.json").write_text(json.dumps(met_c, indent=2) + "\n", encoding="utf-8")

    comb_df = pd.DataFrame([met_c])
    comb_df.to_csv(ART / "PHASE12_COMBINED_DESIGN_MASTER_TABLE_v1.csv", index=False)
    comb_spec = [
        "# PHASE12 Combined Design Spec v1",
        "",
        f"- best_usage_source: `{best_usage}`",
        f"- best_regime_source: `{best_reg}`",
        f"- best_target_source: `{tgt_best if tgt_best else 'T1_downside_bottom10'}`",
        f"- combined_policy: `{combined_spec.family}` on `{combined_spec.score_col}` with context gate q_ctx={q70_ctx:.4f}",
    ]
    (REPORTS / "PHASE12_COMBINED_DESIGN_SPEC_v1.md").write_text("\n".join(comb_spec) + "\n", encoding="utf-8")
    comb_memo = [
        "# PHASE12 Combined Design Decision Memo v1",
        "",
        comb_df.to_markdown(index=False),
        "",
        f"1) Does combined design beat R0? `{'yes' if met_c['delta_net_pnl'] > 0 else 'no'}`",
        f"2) Does it avoid unacceptable upside deletion? `{'yes' if met_c['upside_deletion_ratio'] <= 0.35 else 'no'}`",
        f"3) Stable enough for official challenger? `{'yes' if (met_c['delta_net_pnl'] > 0 and not met_c['disqualified']) else 'no'}`",
    ]
    (REPORTS / "PHASE12_COMBINED_DESIGN_DECISION_MEMO_v1.md").write_text("\n".join(comb_memo) + "\n", encoding="utf-8")

    # ---------------- phase summary ----------------
    phase_summary = [
        "# PHASE12 Phase Summary v1",
        "",
        f"1) best usage redesign result: `{usage_best_row['candidate']}`",
        f"2) best regime-sentinel result: `{reg_best_row['candidate']}`",
        f"3) best target redesign result: `{tgt_best_row['candidate']}`",
        f"4) combined best design beats R0: `{'yes' if met_c['delta_net_pnl'] > 0 else 'no'}`",
        f"5) challenger strong enough for controlled forward research: `{'yes' if (met_c['delta_net_pnl'] > 0 and not met_c['disqualified']) else 'no'}`",
        "6) official anchor remains: `R0_baseline_locked`",
        f"7) official challenger: `{combined_spec.name if (met_c['delta_net_pnl'] > 0 and not met_c['disqualified']) else 'none_yet'}`",
        "",
        "Method note:",
        "- Phase-12 here is a controlled candidate-level policy harness on the locked R0 decision ledger to isolate layer effects and attribution.",
        "- Any promoted challenger should be revalidated in full path-dependent replay before deployment decisions.",
    ]
    (REPORTS / "PHASE12_PHASE_SUMMARY_v1.md").write_text("\n".join(phase_summary) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
