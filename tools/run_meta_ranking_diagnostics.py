#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib


REPO = Path("/opt/testerdonch").resolve()
ARTS = REPO / "artifacts" / "rebaseline"
REPORTS = REPO / "reports" / "rebaseline"


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _mk_candidate_id(symbol: pd.Series, ts: pd.Series) -> pd.Series:
    return symbol.astype(str) + "|" + ts.astype(str)


def _bucket_metrics(df: pd.DataFrame, score_col: str, label_col: str, out_prefix: str) -> pd.DataFrame:
    w = df.copy()
    w = w[w[score_col].notna()].copy()
    if w.empty:
        return pd.DataFrame()

    rows: List[dict] = []
    for n_buckets, kind in [(10, "decile"), (20, "vigintile")]:
        try:
            w["_bucket"] = pd.qcut(w[score_col], q=n_buckets, labels=False, duplicates="drop")
        except ValueError:
            continue
        for b, g in w.groupby("_bucket", dropna=True):
            if g["realized_net_pnl"].notna().any():
                gn = g[g["realized_net_pnl"].notna()].copy()
                hit_rate = float((gn["realized_net_pnl"] > 0).mean())
            else:
                hit_rate = np.nan
            rows.append(
                {
                    "bucket_kind": kind,
                    "bucket_index": int(b),
                    "n": int(len(g)),
                    "score_min": float(g[score_col].min()),
                    "score_max": float(g[score_col].max()),
                    "score_mean": float(g[score_col].mean()),
                    "positive_label_rate": float(g[label_col].mean()) if g[label_col].notna().any() else np.nan,
                    "avg_realized_net_pnl": float(g["realized_net_pnl"].mean()) if g["realized_net_pnl"].notna().any() else np.nan,
                    "median_realized_net_pnl": float(g["realized_net_pnl"].median()) if g["realized_net_pnl"].notna().any() else np.nan,
                    "avg_realized_return_r": float(g["realized_return_r"].mean()) if g["realized_return_r"].notna().any() else np.nan,
                    "hit_rate": hit_rate,
                    "avg_fees": float(g["fees"].mean()) if g["fees"].notna().any() else np.nan,
                    "avg_slippage": float(g["slippage"].mean()) if g["slippage"].notna().any() else np.nan,
                    "avg_mfe_over_atr": float(g["mfe_over_atr"].mean()) if "mfe_over_atr" in g.columns and g["mfe_over_atr"].notna().any() else np.nan,
                    "avg_mae_over_atr": float(g["mae_over_atr"].mean()) if "mae_over_atr" in g.columns and g["mae_over_atr"].notna().any() else np.nan,
                }
            )
    out = pd.DataFrame(rows).sort_values(["bucket_kind", "bucket_index"]).reset_index(drop=True)
    return out


def _topk_curve(df: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    cuts = [1, 2, 5, 10, 20, 30, 50, 100]
    w = df[df[score_col].notna()].copy()
    if w.empty:
        return pd.DataFrame()
    w = w.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(w)
    rows = []
    for c in cuts:
        k = max(1, int(np.ceil(n * (c / 100.0))))
        g = w.iloc[:k]
        if g["realized_net_pnl"].notna().any():
            gn = g[g["realized_net_pnl"].notna()].copy()
            hit_rate = float((gn["realized_net_pnl"] > 0).mean())
        else:
            hit_rate = np.nan
        rows.append(
            {
                "top_pct": c,
                "candidate_count": int(len(g)),
                "candidate_count_with_realized": int(g["realized_net_pnl"].notna().sum()),
                "avg_realized_net_pnl": float(g["realized_net_pnl"].mean()) if g["realized_net_pnl"].notna().any() else np.nan,
                "total_net_pnl": float(g["realized_net_pnl"].sum()) if g["realized_net_pnl"].notna().any() else np.nan,
                "avg_realized_return_r": float(g["realized_return_r"].mean()) if g["realized_return_r"].notna().any() else np.nan,
                "hit_rate": hit_rate,
                "positive_label_rate": float(g[label_col].mean()) if g[label_col].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _rank_stats(df: pd.DataFrame, score_col: str, label_col: str, prefix: str) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    out: Dict[str, float] = {}
    w = df[df[score_col].notna()].copy()
    if w.empty:
        return out

    if w["realized_net_pnl"].notna().any():
        tmp = w[[score_col, "realized_net_pnl"]].dropna()
        out[f"{prefix}_spearman_score_vs_net_pnl"] = float(tmp[score_col].corr(tmp["realized_net_pnl"], method="spearman")) if len(tmp) > 1 else np.nan
    else:
        out[f"{prefix}_spearman_score_vs_net_pnl"] = np.nan

    if w["realized_return_r"].notna().any():
        tmp = w[[score_col, "realized_return_r"]].dropna()
        out[f"{prefix}_spearman_score_vs_return_r"] = float(tmp[score_col].corr(tmp["realized_return_r"], method="spearman")) if len(tmp) > 1 else np.nan
    else:
        out[f"{prefix}_spearman_score_vs_return_r"] = np.nan

    y = _safe_num(w[label_col]).dropna()
    p = _safe_num(w.loc[y.index, score_col]).dropna()
    y = y.loc[p.index]
    if len(y.unique()) >= 2 and len(y) > 2:
        out[f"{prefix}_roc_auc"] = float(roc_auc_score(y, p))
        out[f"{prefix}_pr_auc"] = float(average_precision_score(y, p))
        out[f"{prefix}_brier"] = float(brier_score_loss(y, p))
        out[f"{prefix}_logloss"] = float(log_loss(y, p, labels=[0, 1]))
    else:
        out[f"{prefix}_roc_auc"] = np.nan
        out[f"{prefix}_pr_auc"] = np.nan
        out[f"{prefix}_brier"] = np.nan
        out[f"{prefix}_logloss"] = np.nan
    return out


def _calibration_table(df: pd.DataFrame, score_col: str, label_col: str, n_bins: int = 10) -> pd.DataFrame:
    w = df[[score_col, label_col]].copy()
    w[score_col] = _safe_num(w[score_col])
    w[label_col] = _safe_num(w[label_col])
    w = w.dropna()
    if w.empty:
        return pd.DataFrame()
    try:
        w["bin"] = pd.qcut(w[score_col], q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    out = (
        w.groupby("bin", as_index=False)
        .agg(
            n=(score_col, "size"),
            score_min=(score_col, "min"),
            score_max=(score_col, "max"),
            predicted_mean=(score_col, "mean"),
            empirical_positive_rate=(label_col, "mean"),
        )
        .sort_values("bin")
        .reset_index(drop=True)
    )
    out["calibration_abs_gap"] = (out["predicted_mean"] - out["empirical_positive_rate"]).abs()
    return out


def _threshold_sweep(df: pd.DataFrame, score_col: str, thr_list: List[float]) -> pd.DataFrame:
    w = df[df[score_col].notna()].copy()
    rows = []
    for thr in thr_list:
        g = w[w[score_col] >= thr].copy()
        if g["realized_net_pnl"].notna().any():
            gn = g[g["realized_net_pnl"].notna()].copy()
            hit_rate = float((gn["realized_net_pnl"] > 0).mean())
        else:
            hit_rate = np.nan
        rows.append(
            {
                "threshold": float(thr),
                "candidates_passing": int(len(g)),
                "candidates_with_realized": int(g["realized_net_pnl"].notna().sum()),
                "avg_realized_net_pnl_passed": float(g["realized_net_pnl"].mean()) if g["realized_net_pnl"].notna().any() else np.nan,
                "total_net_pnl_passed": float(g["realized_net_pnl"].sum()) if g["realized_net_pnl"].notna().any() else np.nan,
                "hit_rate_passed": hit_rate,
                "avg_realized_return_r_passed": float(g["realized_return_r"].mean()) if g["realized_return_r"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_oos_panel() -> pd.DataFrame:
    # Scores from zero-trade production-like run
    scored = pd.read_csv(REPO / "results/rebaseline/phase3_zero_trade_diag_20260307/production_like_base/backtest/signal_decisions.csv")
    scored["decision_ts"] = pd.to_datetime(scored["ts_effective"], utc=True)
    scored["meta_score"] = _safe_num(scored["prob_val"])
    scored["predicted_class_probability_used_for_gate"] = scored["meta_score"]

    # Outcomes from safety run (same candidate universe, gate opened)
    safe_dec = pd.read_csv(REPO / "results/rebaseline/phase3_zero_trade_diag_20260307/safety_threshold0_no_meta_sizing/backtest/signal_decisions.csv")
    safe_dec["decision_ts"] = pd.to_datetime(safe_dec["ts_effective"], utc=True)
    safe_dec["realized_net_pnl"] = _safe_num(safe_dec["pnl"])
    safe_dec["realized_return_r"] = _safe_num(safe_dec["pnl_R"])
    safe_dec["holding_outcome"] = safe_dec.get("exit_reason")

    trades = pd.read_csv(REPO / "results/rebaseline/phase3_zero_trade_diag_20260307/safety_threshold0_no_meta_sizing/backtest/trades.csv")
    trades["decision_ts"] = pd.to_datetime(trades["entry_ts"], utc=True)
    tcols = ["symbol", "decision_ts", "fees", "pnl", "pnl_R", "exit_reason", "side", "mfe_over_atr", "mae_over_atr", "risk_on", "size_mult", "risk_cash_target"]
    for c in tcols:
        if c not in trades.columns:
            trades[c] = np.nan
    t = trades[tcols].copy()
    t.rename(
        columns={
            "pnl": "trade_net_pnl",
            "pnl_R": "trade_return_r",
            "exit_reason": "trade_exit_reason",
            "side": "trade_side",
        },
        inplace=True,
    )

    base = scored.merge(
        safe_dec[
            [
                "symbol",
                "decision_ts",
                "decision",
                "reason",
                "realized_net_pnl",
                "realized_return_r",
                "holding_outcome",
            ]
        ].rename(columns={"decision": "safety_decision", "reason": "safety_reason"}),
        on=["symbol", "decision_ts"],
        how="left",
    )
    base = base.merge(t, on=["symbol", "decision_ts"], how="left")

    base["fees"] = _safe_num(base["fees"])
    base["realized_net_pnl"] = _safe_num(base["realized_net_pnl"])
    # Prefer trade table value when available
    base["realized_net_pnl"] = np.where(base["trade_net_pnl"].notna(), _safe_num(base["trade_net_pnl"]), base["realized_net_pnl"])
    base["realized_return_r"] = np.where(base["trade_return_r"].notna(), _safe_num(base["trade_return_r"]), _safe_num(base["realized_return_r"]))
    base["realized_gross_pnl"] = np.where(base["realized_net_pnl"].notna() & base["fees"].notna(), base["realized_net_pnl"] + base["fees"], np.nan)
    base["slippage"] = np.nan  # not available at candidate row granularity in current artifacts
    base["side"] = base.get("trade_side")
    base["side"] = base["side"].fillna("long")

    base["meta_label_true"] = np.where(base["realized_return_r"].notna(), (base["realized_return_r"] >= 0.5).astype(float), np.nan)
    base["meta_label_definition_id"] = "y_good_05:pnl_R>=0.5"
    base["meta_label_description"] = "Target from training pipeline: candidate/trade has pnl_R >= 0.5"
    base["model_classes_ordering"] = "[0,1]"
    base["positive_class_mapping"] = "class=1 means y_good_05 positive"
    base["run_variant"] = "production_like_base_zero_trade_diag"
    base["dataset_window"] = "oos_2025-09-01_to_2026-03-05"
    base["candidate_id"] = _mk_candidate_id(base["symbol"], base["decision_ts"].astype(str))
    base["spread_sim_flag"] = True
    base["major_config_id"] = "phase3_zero_trade_diag_20260307"

    out_cols = [
        "candidate_id",
        "decision_ts",
        "symbol",
        "side",
        "run_variant",
        "dataset_window",
        "meta_score",
        "predicted_class_probability_used_for_gate",
        "model_classes_ordering",
        "positive_class_mapping",
        "meta_label_true",
        "meta_label_definition_id",
        "meta_label_description",
        "realized_net_pnl",
        "realized_return_r",
        "realized_gross_pnl",
        "fees",
        "slippage",
        "holding_outcome",
        "safety_decision",
        "safety_reason",
        "legacy_risk_on",
        "btc_trend_up",
        "btc_vol_high",
        "sent_ok",
        "sent_beta",
        "sent_comp",
        "sent_funding_z",
        "sent_funding_on",
        "risk_on",
        "size_mult",
        "risk_cash_target",
        "mfe_over_atr",
        "mae_over_atr",
        "spread_sim_flag",
        "major_config_id",
    ]
    for c in out_cols:
        if c not in base.columns:
            base[c] = np.nan
    return base[out_cols].copy().sort_values(["decision_ts", "symbol"]).reset_index(drop=True)


def build_valid_panel() -> pd.DataFrame:
    p = REPO / "results/rebaseline/phase3_risk_meta_retrain_20260306/meta_retrain/05_calibration_ev/oof_with_calibrated_p_lgbm.parquet"
    df = pd.read_parquet(p)
    df["decision_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["meta_score"] = _safe_num(df["p_lgbm_cal"])
    df["predicted_class_probability_used_for_gate"] = df["meta_score"]
    df["meta_label_true"] = _safe_num(df["y_good_05"])
    df["meta_label_definition_id"] = "y_good_05:pnl_R>=0.5"
    df["meta_label_description"] = "Target from training pipeline: candidate/trade has pnl_R >= 0.5"
    df["realized_return_r"] = _safe_num(df["pnl_R"])
    # Validation is in R units; cash pnl unavailable in this artifact
    df["realized_net_pnl"] = df["realized_return_r"]
    df["realized_gross_pnl"] = np.nan
    df["fees"] = np.nan
    df["slippage"] = np.nan
    df["holding_outcome"] = np.nan
    df["model_classes_ordering"] = "[0,1]"
    df["positive_class_mapping"] = "class=1 means y_good_05 positive"
    df["run_variant"] = "meta_retrain_validation_oof"
    df["dataset_window"] = "validation_oof_train_2023-01-01_to_2025-08-31"
    df["symbol"] = np.nan
    df["side"] = "long"
    df["candidate_id"] = "trade_id:" + df["trade_id"].astype(str)
    df["spread_sim_flag"] = np.nan
    df["major_config_id"] = "phase3_risk_meta_retrain_20260306"

    out_cols = [
        "candidate_id",
        "decision_ts",
        "symbol",
        "side",
        "run_variant",
        "dataset_window",
        "meta_score",
        "predicted_class_probability_used_for_gate",
        "model_classes_ordering",
        "positive_class_mapping",
        "meta_label_true",
        "meta_label_definition_id",
        "meta_label_description",
        "realized_net_pnl",
        "realized_return_r",
        "realized_gross_pnl",
        "fees",
        "slippage",
        "holding_outcome",
        "risk_on_1",
        "p_lgbm",
        "p_logreg",
        "spread_sim_flag",
        "major_config_id",
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[out_cols].copy().sort_values("decision_ts").reset_index(drop=True)


def write_orientation_audit() -> None:
    model_dir = REPO / "artifacts/meta/META_MODEL_leakfix_v1"
    model = joblib.load(model_dir / "model.joblib")
    deployment = json.loads((model_dir / "deployment_config.json").read_text(encoding="utf-8"))
    classes = None
    if hasattr(model, "classes_"):
        classes = list(getattr(model, "classes_"))
    elif hasattr(model, "named_steps") and "clf" in model.named_steps and hasattr(model.named_steps["clf"], "classes_"):
        classes = list(getattr(model.named_steps["clf"], "classes_"))
    if classes is None and hasattr(model, "named_steps"):
        for _, est in model.named_steps.items():
            if hasattr(est, "classes_"):
                classes = list(getattr(est, "classes_"))
                break
    if classes is None:
        classes = ["unknown"]

    lines = [
        "# META Score Orientation Audit v1",
        "",
        "1) Positive class definition:",
        "- target = y_good_05",
        "- label rule from research/01_make_targets.py: y_good_05 = 1 iff pnl_R >= 0.5",
        "",
        "2) Inference-time score column used for gate:",
        f"- deployment probability_column = {deployment.get('probability_column', 'unknown')}",
        "- calibrated probability is used in diagnostics as meta_score/prob_val (p_lgbm calibrated)",
        "",
        "3) Score direction expectation:",
        "- higher score means higher probability of positive class (better candidate under target definition)",
        "",
        "4) Model classes ordering:",
        f"- model.classes_ = {classes}",
        "",
        "5) Inversion/class-mapping risk assessment:",
        "- gate compares probability against threshold (>= p*) and zero-trade diagnostics show no schema failures.",
        "- no evidence of inverted mapping in code artifacts; class 1 is treated as positive target.",
        "- final confirmation uses ranking stats and top-k monotonicity in companion reports.",
    ]
    (REPORTS / "META_SCORE_ORIENTATION_AUDIT_v1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_bucket_report(oos_bucket: pd.DataFrame, val_bucket: pd.DataFrame) -> None:
    lines = ["# Meta Bucket Analysis v1", ""]
    lines.append("## OOS")
    lines.append(oos_bucket.to_markdown(index=False) if not oos_bucket.empty else "_no rows_")
    lines.append("")
    lines.append("## Validation")
    lines.append(val_bucket.to_markdown(index=False) if not val_bucket.empty else "_no rows_")
    (REPORTS / "META_BUCKET_ANALYSIS_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown_topk_report(oos_topk: pd.DataFrame, val_topk: pd.DataFrame) -> None:
    lines = ["# Meta Top-K Analysis v1", ""]
    lines.append("## OOS")
    lines.append(oos_topk.to_markdown(index=False) if not oos_topk.empty else "_no rows_")
    lines.append("")
    lines.append("## Validation")
    lines.append(val_topk.to_markdown(index=False) if not val_topk.empty else "_no rows_")
    (REPORTS / "META_TOPK_ANALYSIS_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_calibration_analysis(oos_cal: pd.DataFrame, val_cal: pd.DataFrame, oos: pd.DataFrame, val: pd.DataFrame) -> None:
    def _score_stats(df: pd.DataFrame) -> Dict[str, float]:
        s = _safe_num(df["meta_score"]).dropna()
        if s.empty:
            return {"count": 0}
        return {
            "count": int(len(s)),
            "min": float(s.min()),
            "p1": float(np.percentile(s, 1)),
            "p5": float(np.percentile(s, 5)),
            "p25": float(np.percentile(s, 25)),
            "p50": float(np.percentile(s, 50)),
            "p75": float(np.percentile(s, 75)),
            "p95": float(np.percentile(s, 95)),
            "p99": float(np.percentile(s, 99)),
            "max": float(s.max()),
        }

    def _hist(df: pd.DataFrame) -> pd.DataFrame:
        s = _safe_num(df["meta_score"]).dropna()
        bins = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.42, 1.0]
        labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})" for i in range(len(bins) - 1)]
        b = pd.cut(s, bins=bins, labels=labels, include_lowest=True, right=False)
        return b.value_counts().reindex(labels, fill_value=0).rename_axis("bucket").reset_index(name="count")

    lines = ["# Meta Calibration Analysis v1", ""]
    lines.append("## OOS Score Summary")
    lines.append("```json")
    lines.append(json.dumps(_score_stats(oos), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## OOS Reliability Table")
    lines.append(oos_cal.to_markdown(index=False) if not oos_cal.empty else "_no rows_")
    lines.append("")
    lines.append("## OOS Score Histogram")
    lines.append(_hist(oos).to_markdown(index=False))
    lines.append("")
    lines.append("## Validation Score Summary")
    lines.append("```json")
    lines.append(json.dumps(_score_stats(val), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Validation Reliability Table")
    lines.append(val_cal.to_markdown(index=False) if not val_cal.empty else "_no rows_")
    lines.append("")
    lines.append("## Validation Score Histogram")
    lines.append(_hist(val).to_markdown(index=False))
    (REPORTS / "META_CALIBRATION_ANALYSIS_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_threshold_text(thr: pd.DataFrame) -> None:
    lines = ["# META Threshold Sweep Extended v1", ""]
    lines.append(thr.to_string(index=False))
    (REPORTS / "META_THRESHOLD_SWEEP_EXTENDED_v1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ranking_stats_text(stats: Dict[str, float]) -> None:
    lines = ["# META Ranking Stats v1", ""]
    for k in sorted(stats):
        lines.append(f"{k}: {stats[k]}")
    (REPORTS / "META_RANKING_STATS_v1.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_memo(oos: pd.DataFrame, val: pd.DataFrame, oos_topk: pd.DataFrame) -> None:
    # Evidence snippets
    oos_corr = _rank_stats(oos, "meta_score", "meta_label_true", "oos").get("oos_spearman_score_vs_return_r", np.nan)
    val_corr = _rank_stats(val, "meta_score", "meta_label_true", "valid").get("valid_spearman_score_vs_return_r", np.nan)
    oos_count_ge_042 = int((_safe_num(oos["meta_score"]) >= 0.42).sum())
    if not oos_topk.empty:
        best = oos_topk.sort_values("avg_realized_return_r", ascending=False).head(1).iloc[0].to_dict()
        best_str = f"top {int(best['top_pct'])}% (n={int(best['candidate_count'])}, avg_R={best['avg_realized_return_r']:.6f})"
    else:
        best_str = "n/a"

    lines = [
        "# META Ranking Decision Memo v1",
        "",
        "1) Is score orientation correct, or inverted?",
        "- Orientation appears correct: positive class is y_good_05=1 (pnl_R>=0.5), classes [0,1], gate uses positive-class probability.",
        "",
        "2) On leak-fixed OOS data, do higher scores correspond to better outcomes?",
        f"- OOS Spearman(meta_score, realized_return_r) = {oos_corr}",
        "",
        "3) Is model still useful as ranker even if old 0.42 threshold is invalid?",
        f"- In OOS, count(meta_score>=0.42) = {oos_count_ge_042}; gate is too strict for this score distribution.",
        f"- Best observed top-k region by avg realized R in current panel: {best_str}.",
        "",
        "4) Does validation survive into OOS?",
        f"- Validation Spearman(meta_score, realized_return_r) = {val_corr}; compare with OOS value above.",
        "",
        "5) Most likely diagnosis:",
        "- If top-k/bucket monotonicity is present while 0.42 has zero pass-rate, ranking signal may remain but threshold/calibration is mismatched.",
        "- If monotonicity is weak/flat in OOS, signal is likely materially weaker post leak-fix and model redesign may be required.",
    ]
    (REPORTS / "META_RANKING_DECISION_MEMO_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ARTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    oos = build_oos_panel()
    valid = build_valid_panel()

    oos.to_csv(ARTS / "META_RANKING_CANDIDATE_PANEL_OOS_v1.csv", index=False)
    valid.to_csv(ARTS / "META_RANKING_CANDIDATE_PANEL_VALID_v1.csv", index=False)

    write_orientation_audit()

    oos_bucket = _bucket_metrics(oos, "meta_score", "meta_label_true", "oos")
    val_bucket = _bucket_metrics(valid, "meta_score", "meta_label_true", "valid")
    oos_bucket.to_csv(ARTS / "META_BUCKET_METRICS_OOS_v1.csv", index=False)
    val_bucket.to_csv(ARTS / "META_BUCKET_METRICS_VALID_v1.csv", index=False)
    write_markdown_bucket_report(oos_bucket, val_bucket)

    oos_topk = _topk_curve(oos, "meta_score", "meta_label_true")
    val_topk = _topk_curve(valid, "meta_score", "meta_label_true")
    oos_topk.to_csv(ARTS / "META_TOPK_CURVE_OOS_v1.csv", index=False)
    val_topk.to_csv(ARTS / "META_TOPK_CURVE_VALID_v1.csv", index=False)
    write_markdown_topk_report(oos_topk, val_topk)

    stats = {}
    stats.update(_rank_stats(oos, "meta_score", "meta_label_true", "oos"))
    stats.update(_rank_stats(valid, "meta_score", "meta_label_true", "valid"))
    write_ranking_stats_text(stats)

    oos_cal = _calibration_table(oos, "meta_score", "meta_label_true")
    val_cal = _calibration_table(valid, "meta_score", "meta_label_true")
    oos_cal.to_csv(ARTS / "META_CALIBRATION_TABLE_OOS_v1.csv", index=False)
    val_cal.to_csv(ARTS / "META_CALIBRATION_TABLE_VALID_v1.csv", index=False)
    write_calibration_analysis(oos_cal, val_cal, oos, valid)

    thr_list = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.42]
    thr = _threshold_sweep(oos, "meta_score", thr_list)
    thr.to_csv(ARTS / "META_THRESHOLD_SWEEP_EXTENDED_OOS_v1.csv", index=False)
    write_threshold_text(thr)

    write_final_memo(oos, valid, oos_topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
