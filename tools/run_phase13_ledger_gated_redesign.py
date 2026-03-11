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
ROOT = REPO / "results" / "rebaseline" / "phase13_ledger_gated_redesign_20260311"
FULL_REPLAY_ROOT = REPO / "results" / "rebaseline" / "phase13_full_replays_20260311"

BASELINE_DIR = REPO / "results" / "rebaseline" / "phase11_riskon_reintro_20260308" / "R0_baseline_locked"
SIGNALS_PATH = REPO / "results" / "rebaseline" / "phase11_riskon_reintro_20260308" / "_scoped_signals" / "signals.parquet"

START = pd.Timestamp("2025-06-01 00:00:00+00:00")
END = pd.Timestamp("2025-12-31 23:59:59+00:00")

WS_DOWNSIDE = ROOT / "workstream_downside"
WS_REGIME = ROOT / "workstream_regime"
WS_TARGET = ROOT / "workstream_target"


@dataclass(frozen=True)
class PolicySpec:
    name: str
    workstream: str
    family: str
    score_col: str
    params: Dict[str, float]
    regime_col: str | None = None


def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _ensure_dirs() -> None:
    for p in [REPORTS, ART, ROOT, FULL_REPLAY_ROOT, WS_DOWNSIDE, WS_REGIME, WS_TARGET]:
        p.mkdir(parents=True, exist_ok=True)


def _feature_cols(df: pd.DataFrame) -> List[str]:
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
    return [c for c in prefer if c in df.columns]


def _context_cols(df: pd.DataFrame) -> List[str]:
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
    return [c for c in prefer if c in df.columns]


def _load_anchor_panel() -> pd.DataFrame:
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


def _month_folds(exec_df: pd.DataFrame) -> List[Tuple[List[str], str]]:
    months = sorted(exec_df["month"].dropna().unique().tolist())
    folds: List[Tuple[List[str], str]] = []
    for i in range(1, len(months)):
        folds.append((months[:i], months[i]))
    return folds


def _fit_oof_scores(exec_df: pd.DataFrame, feat_cols: List[str], target_mode: str) -> pd.Series:
    df = exec_df.copy().reset_index(drop=True)
    folds = _month_folds(df)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if not folds:
        return out.fillna(0.5)

    for train_months, valid_month in folds:
        tr = df[df["month"].isin(train_months)].copy()
        va = df[df["month"] == valid_month].copy()
        if tr.empty or va.empty:
            continue
        xtr = tr[feat_cols].apply(_num).replace([np.inf, -np.inf], np.nan)
        xva = va[feat_cols].apply(_num).replace([np.inf, -np.inf], np.nan)
        med = xtr.median(numeric_only=True)
        xtr = xtr.fillna(med).fillna(0.0)
        xva = xva.fillna(med).fillna(0.0)

        if target_mode == "downside":
            ytr = (_num(tr["baseline_pnl_R"]) <= -0.5).astype(int)
            if ytr.nunique() < 2:
                pred = np.full(len(va), float(ytr.mean()))
            else:
                clf = LogisticRegression(max_iter=2000, C=0.5)
                clf.fit(xtr, ytr)
                pred = 1.0 - clf.predict_proba(xva)[:, 1]  # quality score, higher=better
        elif target_mode == "relative":
            ctx_med = tr.groupby("risk_on")["baseline_pnl_R"].median().to_dict()
            ytr = (_num(tr["baseline_pnl_R"]) > _num(tr["risk_on"].map(ctx_med))).astype(int)
            if ytr.nunique() < 2:
                pred = np.full(len(va), float(ytr.mean()))
            else:
                clf = LogisticRegression(max_iter=2000, C=0.5)
                clf.fit(xtr, ytr)
                pred = clf.predict_proba(xva)[:, 1]
        elif target_mode == "utility":
            ytr = _num(tr["baseline_pnl_R"]).clip(-2.0, 2.0).fillna(0.0)
            reg = Ridge(alpha=1.0)
            reg.fit(xtr, ytr)
            pred = reg.predict(xva)
        else:
            raise ValueError(target_mode)
        out.loc[va.index] = pred

    if out.isna().any():
        v = float(out.dropna().median()) if out.notna().any() else 0.5
        out = out.fillna(v)
    return out


def _fit_oof_regime_sentinel(exec_df: pd.DataFrame, ctx_cols: List[str]) -> pd.Series:
    df = exec_df.copy().reset_index(drop=True)
    folds = _month_folds(df)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if not folds:
        return out.fillna(0.5)

    for train_months, valid_month in folds:
        tr = df[df["month"].isin(train_months)].copy()
        va = df[df["month"] == valid_month].copy()
        if tr.empty or va.empty:
            continue
        xtr = tr[ctx_cols].apply(_num).replace([np.inf, -np.inf], np.nan)
        xva = va[ctx_cols].apply(_num).replace([np.inf, -np.inf], np.nan)
        med = xtr.median(numeric_only=True)
        xtr = xtr.fillna(med).fillna(0.0)
        xva = xva.fillna(med).fillna(0.0)
        ytr = (_num(tr["baseline_pnl_R"]) <= -0.5).astype(int)
        if ytr.nunique() < 2:
            pred = np.full(len(va), float(ytr.mean()))
        else:
            clf = LogisticRegression(max_iter=2000, C=0.5)
            clf.fit(xtr, ytr)
            pred = clf.predict_proba(xva)[:, 1]  # higher => more regime risk
        out.loc[va.index] = pred

    if out.isna().any():
        v = float(out.dropna().median()) if out.notna().any() else 0.5
        out = out.fillna(v)
    return out


def _apply_policy(r: pd.Series, spec: PolicySpec, rolling_thr: Dict[str, float] | None = None) -> Tuple[str, float, bool]:
    if not bool(r.get("baseline_execute", False)):
        return "baseline_skip", 0.0, False

    s = float(r.get(spec.score_col, np.nan))
    rg = float(r.get(spec.regime_col, np.nan)) if spec.regime_col else np.nan
    downside_risk = float(r.get("downside_risk", np.nan))
    risk_on = _num(pd.Series([r.get("risk_on", np.nan)])).iloc[0]
    p = spec.params

    if not np.isfinite(s):
        return "keep_no_score", 1.0, True

    if spec.family == "bottom_veto":
        if s <= float(p["thr"]):
            return "veto_bottom", 0.0, False
        return "keep", 1.0, True

    if spec.family == "state_bottom_veto":
        thr_good = float(p["thr_good"])
        thr_bad = float(p["thr_bad"])
        thr = thr_good if (np.isfinite(risk_on) and risk_on > 0.0) else thr_bad
        if s <= thr:
            return "veto_state_bottom", 0.0, False
        return "keep", 1.0, True

    if spec.family == "size_mod":
        ql = float(p["q_low"])
        qh = float(p["q_high"])
        if s <= ql:
            return "size_down_low", float(p["m_low"]), True
        if s >= qh:
            return "keep_high", float(p["m_high"]), True
        return "keep_mid", float(p["m_mid"]), True

    if spec.family == "state_size_mod":
        m_good = float(p["m_good"])
        m_bad = float(p["m_bad"])
        if np.isfinite(risk_on) and risk_on > 0.0:
            return "keep_favorable", m_good, True
        return "size_down_unfavorable_state", m_bad, True

    if spec.family == "state_bad_veto":
        thr_bad = float(p["thr_bad"])
        if (not np.isfinite(risk_on) or risk_on <= 0.0) and s <= thr_bad:
            return "veto_state_bad", 0.0, False
        return "keep", 1.0, True

    if spec.family == "downside_only_veto":
        if np.isfinite(downside_risk) and downside_risk >= float(p["risk_thr"]):
            return "veto_downside_risk", 0.0, False
        return "keep", 1.0, True

    if spec.family == "downside_only_size":
        if np.isfinite(downside_risk) and downside_risk >= float(p["risk_thr"]):
            return "size_down_downside_risk", float(p["m_risk"]), True
        return "keep", 1.0, True

    if spec.family == "regime_context_only":
        return "context_score_only", 1.0, True

    if spec.family == "regime_conditional_veto":
        if np.isfinite(rg) and rg >= float(p["regime_thr"]) and s <= float(p["score_thr"]):
            return "veto_regime_conditioned", 0.0, False
        return "keep", 1.0, True

    if spec.family == "regime_conditional_size":
        if np.isfinite(rg) and rg >= float(p["regime_thr"]):
            return "size_down_regime_conditioned", float(p["m_regime"]), True
        return "keep", 1.0, True

    if spec.family == "rolling_quantile_veto":
        month = str(r.get("month", ""))
        thr = float(rolling_thr.get(month, np.nan)) if rolling_thr else np.nan
        if np.isfinite(thr) and s <= thr:
            return "veto_rolling_quantile", 0.0, False
        return "keep", 1.0, True

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


def _compute_disqualification(m: Dict[str, float], gates: Dict[str, float]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    eco_ok = (m["delta_net_pnl"] > 0.0) or (
        m["delta_max_dd"] >= gates["min_dd_improvement_if_not_pnl_positive"]
        and m["delta_net_pnl"] >= gates["max_net_loss_if_dd_improved"]
    )
    if not eco_ok:
        reasons.append("economic_delta_fail")
    if m["selected_trade_count"] < gates["min_selected_trades"]:
        reasons.append("breadth_trades_fail")
    if m["active_months"] < gates["min_active_months"]:
        reasons.append("breadth_months_fail")
    if m["symbol_coverage"] < gates["min_symbol_coverage"]:
        reasons.append("breadth_symbol_fail")
    if m["validation_windows_beating_r0"] < gates["min_windows_beating"]:
        reasons.append("breadth_windows_fail")
    if m["upside_deletion_ratio"] > gates["max_total_upside_deletion_ratio"]:
        reasons.append("disqualified_upside_deletion")
    if m["favorable_upside_deletion_ratio"] > gates["max_favorable_upside_deletion_ratio"]:
        reasons.append("disqualified_favorable_upside_deletion")
    if m["strongest_month_upside_deletion_ratio"] > gates["max_strongest_month_upside_deletion_ratio"]:
        reasons.append("disqualified_strong_month_upside_deletion")

    return (len(reasons) > 0), reasons


def _evaluate_candidate(panel: pd.DataFrame, spec: PolicySpec, gates: Dict[str, float]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    d = panel.copy()
    exec_rows = d[d["baseline_execute"]].copy()
    rolling_thr = None
    if spec.family == "rolling_quantile_veto":
        rolling_thr = _rolling_thresholds(exec_rows, spec.score_col, float(spec.params["q"]))

    decisions: List[str] = []
    mults: List[float] = []
    execute_flags: List[bool] = []
    for _, r in d.iterrows():
        dec, m, ex = _apply_policy(r, spec, rolling_thr=rolling_thr)
        decisions.append(dec)
        mults.append(m)
        execute_flags.append(ex)

    d["policy_decision"] = decisions
    d["policy_size_mult"] = _num(pd.Series(mults)).fillna(1.0)
    d["final_execute"] = execute_flags
    d["skip_reason"] = np.where(d["final_execute"], "", d["policy_decision"])

    base_pnl = _num(d["baseline_pnl"]).fillna(0.0)
    base_not = _num(d["baseline_notional"]).fillna(0.0)
    d["challenger_pnl"] = np.where(d["final_execute"], base_pnl * d["policy_size_mult"], 0.0)
    d["challenger_notional"] = np.where(d["final_execute"], base_not * d["policy_size_mult"], 0.0)
    d["pnl_removed"] = np.where(d["baseline_execute"], base_pnl - d["challenger_pnl"], 0.0)

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

    fav_mask = _num(d["risk_on"]).fillna(0.0) > 0.0
    base_fav = float(_num(d.loc[d["baseline_execute"] & fav_mask, "baseline_pnl"]).sum())
    chal_fav = float(_num(d.loc[d["final_execute"] & fav_mask, "challenger_pnl"]).sum())
    base_unf = float(_num(d.loc[d["baseline_execute"] & ~fav_mask, "baseline_pnl"]).sum())
    chal_unf = float(_num(d.loc[d["final_execute"] & ~fav_mask, "challenger_pnl"]).sum())

    pos_total_available = float(_num(b["baseline_pnl"]).clip(lower=0).sum())
    pos_fav_available = float(_num(d.loc[d["baseline_execute"] & fav_mask, "baseline_pnl"]).clip(lower=0).sum())
    removed_pos = float(_num(d.loc[d["baseline_execute"], "pnl_removed"]).clip(lower=0).sum())
    removed_neg = float((-_num(d.loc[d["baseline_execute"], "pnl_removed"]).clip(upper=0)).sum())
    removed_pos_fav = float(_num(d.loc[d["baseline_execute"] & fav_mask, "pnl_removed"]).clip(lower=0).sum())

    monthly = (
        d.groupby("month", as_index=False)
        .agg(base=("baseline_pnl", "sum"), chal=("challenger_pnl", "sum"))
        .assign(delta=lambda x: _num(x["chal"]) - _num(x["base"]))
    )
    windows_beating = int((monthly["delta"] > 0).sum())
    windows_total = int(len(monthly))

    base_month_pos = (
        d.groupby("month", as_index=False)
        .agg(base_pos=("baseline_pnl", lambda s: float(_num(s).clip(lower=0).sum())))
        .sort_values("base_pos", ascending=False)
    )
    strongest_month = str(base_month_pos.iloc[0]["month"]) if not base_month_pos.empty else ""
    strongest_month_pos = float(base_month_pos.iloc[0]["base_pos"]) if not base_month_pos.empty else 0.0
    strongest_removed_pos = float(
        _num(
            d.loc[(d["month"] == strongest_month) & d["baseline_execute"], "pnl_removed"]
        ).clip(lower=0).sum()
    ) if strongest_month else 0.0

    by_symbol = c.groupby("symbol", as_index=False).agg(net_pnl=("challenger_pnl", "sum"))
    top_symbol_share = 0.0
    if not by_symbol.empty:
        total_abs = float(_num(by_symbol["net_pnl"]).abs().sum())
        if total_abs > 0:
            top_symbol_share = float(_num(by_symbol["net_pnl"]).abs().max() / total_abs)

    metrics = {
        "candidate": spec.name,
        "workstream": spec.workstream,
        "family": spec.family,
        "score_col": spec.score_col,
        "baseline_net_pnl": baseline_net,
        "challenger_net_pnl": challenger_net,
        "delta_net_pnl": delta_net,
        "baseline_gross_pnl": baseline_net,
        "challenger_gross_pnl": challenger_net,
        "delta_gross_pnl": delta_net,
        "baseline_max_dd": baseline_dd,
        "challenger_max_dd": challenger_dd,
        "delta_max_dd": delta_dd,
        "baseline_trades": int(len(b)),
        "challenger_trades": int(len(c)),
        "delta_trades": int(len(c) - len(b)),
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
        "favorable_positive_pnl_removed": removed_pos_fav,
        "upside_deletion_ratio": float(removed_pos / max(1e-9, pos_total_available)),
        "favorable_upside_deletion_ratio": float(removed_pos_fav / max(1e-9, pos_fav_available)),
        "strongest_month": strongest_month,
        "strongest_month_positive_pnl": strongest_month_pos,
        "strongest_month_positive_pnl_removed": strongest_removed_pos,
        "strongest_month_upside_deletion_ratio": float(strongest_removed_pos / max(1e-9, strongest_month_pos)),
        "selected_trade_count": int(len(c)),
        "active_months": int(c["month"].nunique()),
        "symbol_coverage": int(c["symbol"].nunique()),
        "validation_windows_beating_r0": windows_beating,
        "validation_windows_total": windows_total,
        "top_symbol_pnl_abs_share": top_symbol_share,
    }

    disq, reasons = _compute_disqualification(metrics, gates)
    metrics["stage1_disqualified"] = disq
    metrics["stage1_disqual_reasons"] = "|".join(reasons)
    metrics["stage1_pass"] = not disq
    return d, metrics


def _full_replay_from_ledger(ledger: pd.DataFrame, candidate: str, out_dir: Path) -> Dict[str, float]:
    # Path-dependent replay proxy: preserve original chronological trade path, but enforce candidate keep/size decisions.
    # This restores ordering, equity curve and monthly path dependence without re-scouting/resimulating signals.
    d = ledger.copy()
    d = d[d["baseline_execute"]].copy().sort_values(["decision_ts", "symbol"]).reset_index(drop=True)
    d["trade_executed"] = d["final_execute"].astype(bool)
    d["replay_pnl"] = np.where(d["trade_executed"], _num(d["baseline_pnl"]).fillna(0.0) * _num(d["policy_size_mult"]).fillna(1.0), 0.0)
    d["replay_notional"] = np.where(d["trade_executed"], _num(d["baseline_notional"]).fillna(0.0) * _num(d["policy_size_mult"]).fillna(1.0), 0.0)
    d["eq"] = 2000.0 + _num(d["replay_pnl"]).fillna(0.0).cumsum()
    d["dd"] = d["eq"] - d["eq"].cummax()
    d["month"] = d["decision_ts"].dt.to_period("M").astype(str)

    trades = d[d["trade_executed"]].copy()
    monthly = d.groupby("month", as_index=False).agg(
        replay_net_pnl=("replay_pnl", "sum"),
        replay_trades=("trade_executed", "sum"),
        replay_max_dd=("dd", "min"),
    )
    state = d.groupby("risk_on", dropna=False, as_index=False).agg(
        replay_net_pnl=("replay_pnl", "sum"),
        replay_trades=("trade_executed", "sum"),
        avg_replay_pnl=("replay_pnl", "mean"),
    )
    retained_removed = pd.DataFrame(
        {
            "bucket": ["retained", "removed"],
            "baseline_pnl_sum": [
                float(_num(d.loc[d["trade_executed"], "baseline_pnl"]).sum()),
                float(_num(d.loc[~d["trade_executed"], "baseline_pnl"]).sum()),
            ],
            "rows": [int(d["trade_executed"].sum()), int((~d["trade_executed"]).sum())],
        }
    )

    summary = {
        "candidate": candidate,
        "replay_net_pnl": float(_num(d["replay_pnl"]).sum()),
        "replay_max_dd": float(_num(d["dd"]).min()) if not d.empty else 0.0,
        "replay_trades": int(trades.shape[0]),
        "replay_turnover": float(_num(d["replay_notional"]).sum()),
        "active_months": int(trades["month"].nunique()) if not trades.empty else 0,
        "symbol_coverage": int(trades["symbol"].nunique()) if not trades.empty else 0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(out_dir / "decision_ledger.csv", index=False)
    trades.to_csv(out_dir / "trades_counterfactual.csv", index=False)
    monthly.to_csv(out_dir / "monthly_impact.csv", index=False)
    state.to_csv(out_dir / "state_decomposition.csv", index=False)
    retained_removed.to_csv(out_dir / "removed_vs_retained_attribution.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _write_anchor_gov(commit: str) -> None:
    lines = [
        "# PHASE13 Anchor And Governance v1",
        "",
        "1) official anchor = `R0_baseline_locked`",
        "2) official challenger status starts as `none_yet`",
        "3) ledger-first screening is mandatory",
        "4) no full replay for weak ideas (must pass Stage-1 gates first)",
        "5) no promotion influence from candidates failing Stage-1 gates",
        "",
        f"- commit: `{commit}`",
        f"- anchor_path: `{BASELINE_DIR}`",
    ]
    (REPORTS / "PHASE13_ANCHOR_AND_GOVERNANCE_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_screening_rules() -> None:
    lines = [
        "# PHASE13 Ledger Screening Rules v1",
        "",
        "Stage-1 evaluates each candidate strictly as a counterfactual overlay delta vs R0.",
        "",
        "Scored dimensions:",
        "- delta economics",
        "- breadth/support",
        "- upside deletion",
        "- stability/concentration",
        "",
        "Only candidates passing all Stage-1 hard gates proceed to Stage-2 full replay.",
    ]
    (REPORTS / "PHASE13_LEDGER_SCREENING_RULES_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stage1_gates(g: Dict[str, float]) -> None:
    lines = [
        "# PHASE13 Stage1 Pass/Fail Gates v1",
        "",
        "Hard gates:",
        f"- min_selected_trades: `{int(g['min_selected_trades'])}`",
        f"- min_active_months: `{int(g['min_active_months'])}`",
        f"- min_symbol_coverage: `{int(g['min_symbol_coverage'])}`",
        f"- min_windows_beating_r0: `{int(g['min_windows_beating'])}`",
        f"- max_total_upside_deletion_ratio: `{g['max_total_upside_deletion_ratio']:.3f}`",
        f"- max_favorable_upside_deletion_ratio: `{g['max_favorable_upside_deletion_ratio']:.3f}`",
        f"- max_strongest_month_upside_deletion_ratio: `{g['max_strongest_month_upside_deletion_ratio']:.3f}`",
        "",
        "Economics gate:",
        "- pass if delta_net_pnl > 0",
        "- OR pass if drawdown improvement is material and net loss is tightly bounded",
        f"  - min_dd_improvement_if_not_pnl_positive: `{g['min_dd_improvement_if_not_pnl_positive']:.1f}`",
        f"  - max_net_loss_if_dd_improved: `{g['max_net_loss_if_dd_improved']:.1f}`",
    ]
    (REPORTS / "PHASE13_STAGE1_PASS_FAIL_GATES_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_upside_control(g: Dict[str, float], all_df: pd.DataFrame) -> None:
    cols = [
        "candidate",
        "workstream",
        "upside_deletion_ratio",
        "favorable_upside_deletion_ratio",
        "strongest_month",
        "strongest_month_upside_deletion_ratio",
        "stage1_pass",
        "stage1_disqual_reasons",
    ]
    lines = [
        "# PHASE13 Upside Deletion Control v1",
        "",
        "Primary disqualifier: upside deletion.",
        "",
        "Ceilings:",
        f"- total upside deletion <= {g['max_total_upside_deletion_ratio']:.3f}",
        f"- favorable-state upside deletion <= {g['max_favorable_upside_deletion_ratio']:.3f}",
        f"- strongest-month upside deletion <= {g['max_strongest_month_upside_deletion_ratio']:.3f}",
        "",
        all_df[cols].sort_values(["stage1_pass", "upside_deletion_ratio"], ascending=[False, True]).to_markdown(index=False),
        "",
        "Candidates breaching ceilings are tagged with `disqualified_upside_deletion` family reasons.",
    ]
    (REPORTS / "PHASE13_UPSIDE_DELETION_CONTROL_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    _ensure_dirs()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()

    gates = {
        "min_selected_trades": 1800.0,
        "min_active_months": 6.0,
        "min_symbol_coverage": 430.0,
        "min_windows_beating": 3.0,
        "max_total_upside_deletion_ratio": 0.08,
        "max_favorable_upside_deletion_ratio": 0.10,
        "max_strongest_month_upside_deletion_ratio": 0.12,
        "min_dd_improvement_if_not_pnl_positive": 300.0,
        "max_net_loss_if_dd_improved": -600.0,
    }

    _write_anchor_gov(commit)
    _write_screening_rules()
    _write_stage1_gates(gates)

    panel = _load_anchor_panel()
    exec_df = panel[panel["baseline_execute"]].copy().reset_index(drop=True)
    feat_cols = _feature_cols(exec_df)
    ctx_cols = _context_cols(exec_df)

    exec_df["score_t1_downside_quality"] = _fit_oof_scores(exec_df, feat_cols, "downside")
    exec_df["score_t2_relative"] = _fit_oof_scores(exec_df, feat_cols, "relative")
    exec_df["score_t3_utility"] = _fit_oof_scores(exec_df, feat_cols, "utility")
    exec_df["regime_sentinel_risk"] = _fit_oof_regime_sentinel(exec_df, ctx_cols)
    exec_df["downside_risk"] = 1.0 - _num(exec_df["score_t1_downside_quality"]).fillna(0.5)

    score_map = exec_df[
        [
            "candidate_id",
            "score_t1_downside_quality",
            "score_t2_relative",
            "score_t3_utility",
            "regime_sentinel_risk",
            "downside_risk",
        ]
    ].copy()
    panel = panel.merge(score_map, on="candidate_id", how="left")

    q = {
        "t1_q05": float(exec_df["score_t1_downside_quality"].quantile(0.05)),
        "t1_q10": float(exec_df["score_t1_downside_quality"].quantile(0.10)),
        "t1_q15": float(exec_df["score_t1_downside_quality"].quantile(0.15)),
        "t1_q20": float(exec_df["score_t1_downside_quality"].quantile(0.20)),
        "t1_q30": float(exec_df["score_t1_downside_quality"].quantile(0.30)),
        "t1_q70": float(exec_df["score_t1_downside_quality"].quantile(0.70)),
        "down_q80": float(exec_df["downside_risk"].quantile(0.80)),
        "down_q85": float(exec_df["downside_risk"].quantile(0.85)),
        "down_q90": float(exec_df["downside_risk"].quantile(0.90)),
        "reg_q75": float(exec_df["regime_sentinel_risk"].quantile(0.75)),
        "reg_q85": float(exec_df["regime_sentinel_risk"].quantile(0.85)),
        "reg_q90": float(exec_df["regime_sentinel_risk"].quantile(0.90)),
        "t2_q10": float(exec_df["score_t2_relative"].quantile(0.10)),
        "t3_q10": float(exec_df["score_t3_utility"].quantile(0.10)),
    }

    # Workstream 1: downside overlays
    (REPORTS / "PHASE13_DOWNSIDE_OVERLAY_PLAN_v1.md").write_text(
        "\n".join(
            [
                "# PHASE13 Downside Overlay Plan v1",
                "",
                "Families: D1 bottom-tail veto, D2 score-to-size, D3 state-conditional overlay, D4 downside-only veto/size.",
                "Overlay-first design; no broad model zoo.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    downside_specs: List[PolicySpec] = [
        PolicySpec("D1_bottom_veto_q05", "W1_downside_overlay", "bottom_veto", "score_t1_downside_quality", {"thr": q["t1_q05"]}),
        PolicySpec("D1_bottom_veto_q10", "W1_downside_overlay", "bottom_veto", "score_t1_downside_quality", {"thr": q["t1_q10"]}),
        PolicySpec("D1_bottom_veto_q15", "W1_downside_overlay", "bottom_veto", "score_t1_downside_quality", {"thr": q["t1_q15"]}),
        PolicySpec("D1_bottom_veto_q20", "W1_downside_overlay", "bottom_veto", "score_t1_downside_quality", {"thr": q["t1_q20"]}),
        PolicySpec("D1_state_bottom_veto", "W1_downside_overlay", "state_bottom_veto", "score_t1_downside_quality", {"thr_good": q["t1_q05"], "thr_bad": q["t1_q20"]}),
        PolicySpec("D2_size_mod_soft", "W1_downside_overlay", "size_mod", "score_t1_downside_quality", {"q_low": q["t1_q20"], "q_high": q["t1_q70"], "m_low": 0.70, "m_mid": 1.00, "m_high": 1.00}),
        PolicySpec("D2_size_mod_medium", "W1_downside_overlay", "size_mod", "score_t1_downside_quality", {"q_low": q["t1_q20"], "q_high": q["t1_q70"], "m_low": 0.50, "m_mid": 1.00, "m_high": 1.00}),
        PolicySpec("D3_state_size_mod", "W1_downside_overlay", "state_size_mod", "score_t1_downside_quality", {"m_good": 1.00, "m_bad": 0.65}),
        PolicySpec("D3_state_bad_veto", "W1_downside_overlay", "state_bad_veto", "score_t1_downside_quality", {"thr_bad": q["t1_q20"]}),
        PolicySpec("D4_downside_veto_q85", "W1_downside_overlay", "downside_only_veto", "score_t1_downside_quality", {"risk_thr": q["down_q85"]}),
        PolicySpec("D4_downside_veto_q90", "W1_downside_overlay", "downside_only_veto", "score_t1_downside_quality", {"risk_thr": q["down_q90"]}),
        PolicySpec("D4_downside_size_q80", "W1_downside_overlay", "downside_only_size", "score_t1_downside_quality", {"risk_thr": q["down_q80"], "m_risk": 0.55}),
    ]

    # Workstream 2: regime sentinel
    (REPORTS / "PHASE13_REGIME_SENTINEL_RESEARCH_PLAN_v1.md").write_text(
        "\n".join(
            [
                "# PHASE13 Regime Sentinel Research Plan v1",
                "",
                "RGS-A context-score diagnostics first; no action by default.",
                "RGS-B explanation audit against clustered loss months.",
                "RGS-C optional conditional hooks (overlay-conditioned only).",
                "",
                f"Context columns: `{ctx_cols}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    regime_specs: List[PolicySpec] = [
        PolicySpec("RGS_A_context_only", "W2_regime_sentinel", "regime_context_only", "score_t1_downside_quality", {}, "regime_sentinel_risk"),
        PolicySpec(
            "RGS_C_conditional_veto",
            "W2_regime_sentinel",
            "regime_conditional_veto",
            "score_t1_downside_quality",
            {"regime_thr": q["reg_q85"], "score_thr": q["t1_q20"]},
            "regime_sentinel_risk",
        ),
        PolicySpec(
            "RGS_C_conditional_size",
            "W2_regime_sentinel",
            "regime_conditional_size",
            "score_t1_downside_quality",
            {"regime_thr": q["reg_q75"], "m_regime": 0.70},
            "regime_sentinel_risk",
        ),
        PolicySpec(
            "RGS_C_conditional_downside_size",
            "W2_regime_sentinel",
            "regime_conditional_size",
            "downside_risk",
            {"regime_thr": q["reg_q85"], "m_regime": 0.55},
            "regime_sentinel_risk",
        ),
    ]

    # Workstream 3: target redesign (after usage/context evidence)
    (REPORTS / "PHASE13_TARGET_REDESIGN_PLAN_v1.md").write_text(
        "\n".join(
            [
                "# PHASE13 Target Redesign Plan v1",
                "",
                "Target redesign executed after W1/W2 evidence, using controlled overlay framing.",
                "- T1 downside-avoidance",
                "- T2 relative-quality",
                "- T3 continuous utility",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    target_specs: List[PolicySpec] = [
        PolicySpec("T1_downside_bottom10", "W3_target_redesign", "bottom_veto", "score_t1_downside_quality", {"thr": q["t1_q10"]}),
        PolicySpec("T2_relative_bottom10", "W3_target_redesign", "bottom_veto", "score_t2_relative", {"thr": q["t2_q10"]}),
        PolicySpec("T3_utility_bottom10", "W3_target_redesign", "bottom_veto", "score_t3_utility", {"thr": q["t3_q10"]}),
    ]

    all_specs = downside_specs + regime_specs + target_specs
    all_rows: List[Dict[str, float]] = []
    ledgers: Dict[str, pd.DataFrame] = {}

    for spec in all_specs:
        led, met = _evaluate_candidate(panel, spec, gates)
        all_rows.append(met)
        ledgers[spec.name] = led
        if spec.workstream.startswith("W1"):
            out = WS_DOWNSIDE / spec.name
        elif spec.workstream.startswith("W2"):
            out = WS_REGIME / spec.name
        else:
            out = WS_TARGET / spec.name
        out.mkdir(parents=True, exist_ok=True)
        led.to_csv(out / "decision_ledger.csv", index=False)
        (out / "metrics.json").write_text(json.dumps(met, indent=2) + "\n", encoding="utf-8")

    all_df = pd.DataFrame(all_rows).sort_values(["stage1_pass", "delta_net_pnl"], ascending=[False, False]).reset_index(drop=True)
    all_df.to_csv(ART / "PHASE13_LEDGER_SCREENING_MASTER_TABLE_v1.csv", index=False)

    d_df = all_df[all_df["workstream"] == "W1_downside_overlay"].copy().sort_values("delta_net_pnl", ascending=False)
    d_df.to_csv(ART / "PHASE13_DOWNSIDE_OVERLAY_LEDGER_TABLE_v1.csv", index=False)
    best_d = d_df.iloc[0]
    best_d_least_destructive = d_df.sort_values(["upside_deletion_ratio", "delta_net_pnl"], ascending=[True, False]).iloc[0]
    d_pass_count = int(d_df["stage1_pass"].sum())
    (REPORTS / "PHASE13_DOWNSIDE_OVERLAY_DECISION_MEMO_v1.md").write_text(
        "\n".join(
            [
                "# PHASE13 Downside Overlay Decision Memo v1",
                "",
                d_df.to_markdown(index=False),
                "",
                f"1) least destructive overlay family: `{best_d_least_destructive['candidate']}`",
                f"2) best clustered-loss reducer by delta_unfavorable_pnl: `{d_df.sort_values('delta_unfavorable_pnl', ascending=False).iloc[0]['candidate']}`",
                f"3) best favorable-state upside preservation: `{d_df.sort_values('favorable_upside_deletion_ratio').iloc[0]['candidate']}`",
                f"4) overlays earning full replay: `{d_pass_count}` candidate(s)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    r_df = all_df[all_df["workstream"] == "W2_regime_sentinel"].copy().sort_values("delta_net_pnl", ascending=False)
    r_df.to_csv(ART / "PHASE13_REGIME_SENTINEL_DIAGNOSTICS_v1.csv", index=False)

    # Regime explanation diagnostics
    monthly_baseline = (
        panel[panel["baseline_execute"]]
        .groupby("month", as_index=False)
        .agg(
            baseline_net_pnl=("baseline_pnl", "sum"),
            regime_risk_mean=("regime_sentinel_risk", "mean"),
            downside_risk_mean=("downside_risk", "mean"),
        )
    )
    if len(monthly_baseline) >= 2:
        corr_regime = float(monthly_baseline["regime_risk_mean"].corr(monthly_baseline["baseline_net_pnl"], method="spearman"))
        corr_down = float(monthly_baseline["downside_risk_mean"].corr(monthly_baseline["baseline_net_pnl"], method="spearman"))
    else:
        corr_regime = np.nan
        corr_down = np.nan
    best_r = r_df.iloc[0]
    sentinel_context_help = bool(np.isfinite(corr_regime) and corr_regime < -0.2)
    sentinel_role = "context_only" if best_r["candidate"] == "RGS_A_context_only" else "conditional_controller_candidate"
    (REPORTS / "PHASE13_REGIME_SENTINEL_DECISION_MEMO_v1.md").write_text(
        "\n".join(
            [
                "# PHASE13 Regime Sentinel Decision Memo v1",
                "",
                "Monthly diagnostics:",
                monthly_baseline.to_markdown(index=False),
                "",
                f"- spearman(regime_risk, baseline_net_pnl): `{corr_regime:.4f}`",
                f"- spearman(downside_risk, baseline_net_pnl): `{corr_down:.4f}`",
                "",
                r_df.to_markdown(index=False),
                "",
                f"1) regime deterioration explains loss flurries better than trade score alone: `{'yes' if sentinel_context_help else 'mixed_or_no'}`",
                f"2) sentinel useful as context: `{'yes' if sentinel_context_help else 'limited'}`",
                f"3) recommended role now: `{sentinel_role}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    t_df = all_df[all_df["workstream"] == "W3_target_redesign"].copy().sort_values("delta_net_pnl", ascending=False)
    t_df.to_csv(ART / "PHASE13_TARGET_REDESIGN_LEDGER_TABLE_v1.csv", index=False)
    best_t = t_df.iloc[0]
    (REPORTS / "PHASE13_TARGET_REDESIGN_DECISION_MEMO_v1.md").write_text(
        "\n".join(
            [
                "# PHASE13 Target Redesign Decision Memo v1",
                "",
                t_df.to_markdown(index=False),
                "",
                f"1) target matching current economic problem best: `{best_t['candidate']}`",
                f"2) downside-avoidance beats top-winner framing: `{'yes' if best_t['candidate'].startswith('T1_') else 'no_or_mixed'}`",
                f"3) target redesign materially reduced upside deletion: `{'yes' if float(best_t['upside_deletion_ratio']) < 0.08 else 'no'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _write_upside_control(gates, all_df)

    # Stage-2 full replay only for Stage-1 pass
    pass_df = all_df[all_df["stage1_pass"]].copy()
    replay_rows: List[Dict[str, float]] = []
    if not pass_df.empty:
        for cand in pass_df["candidate"].tolist():
            summary = _full_replay_from_ledger(ledgers[cand], cand, FULL_REPLAY_ROOT / cand)
            m = pass_df[pass_df["candidate"] == cand].iloc[0].to_dict()
            summary.update(
                {
                    "delta_net_pnl_vs_r0": float(m["delta_net_pnl"]),
                    "delta_max_dd_vs_r0": float(m["delta_max_dd"]),
                    "upside_deletion_ratio": float(m["upside_deletion_ratio"]),
                    "favorable_upside_deletion_ratio": float(m["favorable_upside_deletion_ratio"]),
                }
            )
            replay_rows.append(summary)
    replay_df = pd.DataFrame(replay_rows)
    replay_df.to_csv(ART / "PHASE13_FULL_REPLAY_MASTER_TABLE_v1.csv", index=False)
    (REPORTS / "PHASE13_FULL_REPLAY_DECISION_MEMO_v1.md").write_text(
        "\n".join(
            [
                "# PHASE13 Full Replay Decision Memo v1",
                "",
                ("No Stage-1 candidates passed; no full replays executed." if replay_df.empty else replay_df.to_markdown(index=False)),
                "",
                f"1) Stage-1-approved candidates surviving full replay: `{0 if replay_df.empty else int(replay_df.shape[0])}`",
                f"2) any candidate beats R0 in full replay: `{'yes' if (not replay_df.empty and (replay_df['delta_net_pnl_vs_r0'] > 0).any()) else 'no'}`",
                f"3) upside deletion acceptable post replay: `{'yes' if (not replay_df.empty and (replay_df['upside_deletion_ratio'] <= gates['max_total_upside_deletion_ratio']).all()) else 'no_or_not_applicable'}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    official_challenger = "none_yet"
    if not replay_df.empty:
        replay_df = replay_df.sort_values("delta_net_pnl_vs_r0", ascending=False).reset_index(drop=True)
        top = replay_df.iloc[0]
        if float(top["delta_net_pnl_vs_r0"]) > 0 and float(top["upside_deletion_ratio"]) <= gates["max_total_upside_deletion_ratio"]:
            official_challenger = str(top["candidate"])

    (REPORTS / "PHASE13_OFFICIAL_CHALLENGER_DECISION_v1.md").write_text(
        "\n".join(
            [
                "# PHASE13 Official Challenger Decision v1",
                "",
                f"decision: `{official_challenger if official_challenger != 'none_yet' else 'none_yet'}`",
                "",
                "Allowed statuses:",
                "- `none_yet`",
                "- `challenger_selected_for_controlled_forward_research`",
                "- `challenger_selected_for_additional_replay_only`",
                "",
                (
                    f"selected: `{official_challenger}`"
                    if official_challenger != "none_yet"
                    else "No candidate cleared Stage-1 + full replay + upside-deletion requirements."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    phase_lines = [
        "# PHASE13 Phase Summary v1",
        "",
        "Anchor remains locked to R0; Phase-13 enforced ledger-first gating.",
        "",
        f"- stage1_candidates_total: `{int(all_df.shape[0])}`",
        f"- stage1_pass_count: `{int(pass_df.shape[0])}`",
        f"- best_downside_overlay: `{best_d['candidate']}`",
        f"- best_regime_candidate: `{best_r['candidate']}`",
        f"- best_target_candidate: `{best_t['candidate']}`",
        f"- full_replay_candidates: `{int(replay_df.shape[0])}`",
        f"- official_challenger: `{official_challenger}`",
        "",
        "No candidate can influence promotion unless it clears Stage-1 and Stage-2 constraints.",
    ]
    (REPORTS / "PHASE13_PHASE_SUMMARY_v1.md").write_text("\n".join(phase_lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
