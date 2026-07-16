#!/usr/bin/env python3
"""Global QLMG result invalidation audit.

Scans active QLMG runners and result artifacts for placeholder controls, metric
lineage risks, proxy mark/funding caps, promotion-label misuse, and protected
slice leaks. This is an audit/reporting tool only; it does not score strategies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "rebaseline"
BASE_RUN = RESULTS / "phase_qlmg_global_result_invalidation_audit_20260629_v1"
PROTECTED_TS = pd.Timestamp("2026-01-01T00:00:00Z")
PROMO_TERMS = [
    "prelead",
    "confirmed",
    "validation",
    "promote",
    "survives",
    "tier1_prelead",
    "targeted_execution_data_prelead",
    "candidate_survives",
    "family_specific_validation",
]
CONTROL_TYPE_VALUES = [
    "same_symbol",
    "same_regime",
    "nearest_neighbor",
    "nearest_neighbor_vol_liq_funding_oi",
    "generic_momentum",
    "A2_A3_overlap",
    "matched",
    "same_time",
    "shifted_time",
]
SUSPICIOUS_CODE_PATTERNS = {
    "hardcoded_negative_control_r_per_event": re.compile(r"[-=]\s*0\.(?:03|05|08|10|12)\s*\*\s*(?:control_)?events"),
    "placeholder_or_synthetic_text": re.compile(r"placeholder|synthetic|fabricated|dummy", re.I),
    "projection_text": re.compile(r"project(?:ed|ion)|summary_projection|internal_validation_projection", re.I),
    "control_type_loop": re.compile(r"same_symbol|same_regime|nearest_neighbor|generic_momentum|A2_A3_overlap"),
    "copied_control_metric_names": re.compile(r"normalized_control_net_R|control_uplift_R|raw_control_net_R|beats_control"),
}
FUTURE_LEAKAGE_CODE_PATTERNS = {
    "future_path_mfe_mae_used": re.compile(r"24h_mfe_bps|24h_mae_bps|6h_mfe_bps|6h_mae_bps"),
    "future_reclaim_regime_proxy": re.compile(r"post_flush_reclaim_proxy|deleveraged_2of4|deleveraged_3of4"),
    "full_sample_quantile": re.compile(r"\.quantile\(\s*0\.(?:9|99|95)"),
    "bad_wick_proxy_gate": re.compile(r"bad_wick_proxy_label|bad_wick_gate"),
    "future_outcome_filter_text": re.compile(r"future|realized bad|liquidation[-_ ]flagged", re.I),
}
CONTROL_SOURCE_COL_HINTS = [
    "control_event_id",
    "control_window_id",
    "matched_event_id",
    "matched_window_id",
    "control_decision_ts",
    "matched_decision_ts",
    "control_symbol",
    "matched_symbol",
    "control_month",
    "match_distance",
    "target_window_id",
    "source_event_id",
    "control_source",
    "window_id",
]
ID_COLS = [
    "candidate_id",
    "variant_id",
    "definition_id",
    "family",
    "subfamily",
    "mechanism",
    "mechanism_group",
    "mode",
]
TYPE_COLS = ["control_type", "null_type", "baseline_type", "matched_type", "comparison_type"]
CONTROL_METRIC_COLS = [
    "normalized_control_net_R",
    "control_uplift_R",
    "raw_control_net_R",
    "candidate_net_R",
    "candidate_pf",
    "control_pf",
    "beats_control",
    "candidate_events",
    "control_events",
    "replayed_control_event_count",
    "control_event_count",
]
PROXY_COLS = [
    "funding_exact",
    "funding_proxy_used",
    "mark_proxy_used",
    "mark_available",
    "mark_price_available",
    "funding_available",
    "label_cap_reason",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_run_root(base: Path) -> Path:
    if not base.exists():
        return base
    return base.with_name(base.name + "_" + utc_now())


def sha256_file(path: Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv_sample(path: Path, max_rows: int = 250_000) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, nrows=max_rows, low_memory=False)
    except Exception:
        return None


def read_parquet_sample(path: Path, max_rows: int = 250_000) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(path)
        if len(df) > max_rows:
            return df.head(max_rows).copy()
        return df
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys or ["empty"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def discover_files() -> list[Path]:
    if not RESULTS.exists():
        return []
    wanted_suffixes = {".csv", ".json", ".md", ".parquet"}
    files: list[Path] = []
    for p in RESULTS.rglob("*"):
        if p.is_file() and p.suffix.lower() in wanted_suffixes and "phase_qlmg" in str(p):
            try:
                if p.stat().st_size <= 512 * 1024 * 1024:
                    files.append(p)
            except OSError:
                continue
    return sorted(files)


def inventory(files: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for p in files:
        try:
            st = p.stat()
        except OSError:
            continue
        rows.append({
            "path": safe_rel(p),
            "suffix": p.suffix.lower(),
            "size_bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "root": next((part for part in p.parts if part.startswith("phase_qlmg")), ""),
            "sha256_head_1mb": sha256_file(p, max_bytes=1024 * 1024),
        })
    return rows


def scan_code() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(set((ROOT / "tools").glob("run_qlmg*.py")).union((ROOT / "tools").glob("qlmg*.py"))):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            for name, pat in SUSPICIOUS_CODE_PATTERNS.items():
                if pat.search(line):
                    rows.append({
                        "path": safe_rel(path),
                        "line": i,
                        "pattern": name,
                        "text": line.strip()[:500],
                        "risk": "review_required" if name != "hardcoded_negative_control_r_per_event" else "high_placeholder_control_risk",
                    })
    return rows


def scan_future_leakage_code() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(set((ROOT / "tools").glob("run_qlmg*.py")).union((ROOT / "tools").glob("qlmg*.py"))):
        try:
            lines = path.read_text(errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for name, pat in FUTURE_LEAKAGE_CODE_PATTERNS.items():
                if pat.search(line):
                    risk = "review_required"
                    if name in {"future_reclaim_regime_proxy", "full_sample_quantile"}:
                        risk = "high_temporal_leakage_risk"
                    rows.append({
                        "path": safe_rel(path),
                        "line": i,
                        "pattern": name,
                        "text": stripped[:500],
                        "risk": risk,
                    })
    return rows


def canonical_root(path: Path) -> str:
    for part in path.parts:
        if part.startswith("phase_qlmg"):
            return part
    return ""


def control_file_candidate(path: Path) -> bool:
    name = path.name.lower()
    s = str(path).lower()
    return any(tok in name or tok in s for tok in ["control", "null", "baseline"])


def unique_non_null(series: pd.Series) -> int:
    try:
        return int(series.dropna().astype(str).nunique())
    except Exception:
        return int(series.dropna().nunique())


def scan_control_files(files: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identical_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for p in files:
        if p.suffix.lower() != ".csv" or not control_file_candidate(p):
            continue
        df = read_csv_sample(p)
        if df is None or df.empty:
            continue
        cols = set(df.columns)
        type_col = next((c for c in TYPE_COLS if c in cols), None)
        source_cols = [c for c in CONTROL_SOURCE_COL_HINTS if c in cols]
        metric_cols = [c for c in CONTROL_METRIC_COLS if c in cols]
        has_control_type_values = False
        if type_col:
            vals = set(df[type_col].dropna().astype(str).unique().tolist())
            has_control_type_values = bool(vals.intersection(CONTROL_TYPE_VALUES)) or df[type_col].dropna().nunique() > 1
        source_rows.append({
            "path": safe_rel(p),
            "root": canonical_root(p),
            "rows_sampled": len(df),
            "type_col": type_col or "",
            "type_value_count": int(df[type_col].dropna().nunique()) if type_col else 0,
            "metric_cols_present": ";".join(metric_cols),
            "source_cols_present": ";".join(source_cols),
            "missing_source_window_columns": bool(has_control_type_values and not source_cols),
            "risk": "high_missing_control_source_windows" if has_control_type_values and not source_cols else "review",
        })
        if not type_col or not metric_cols:
            continue
        id_cols = [c for c in ID_COLS if c in cols]
        if not id_cols:
            id_cols = ["__all__"]
            df = df.copy()
            df["__all__"] = "all"
        group_cols = id_cols[:4]
        for key, g in df.groupby(group_cols, dropna=False):
            if g[type_col].dropna().nunique() < 2:
                continue
            numeric_metrics = []
            identical_metric_count = 0
            for c in metric_cols:
                if c == type_col:
                    continue
                if c in g.columns:
                    numeric_metrics.append(c)
                    if unique_non_null(g[c]) <= 1:
                        identical_metric_count += 1
            focus_cols = [c for c in ["normalized_control_net_R", "control_uplift_R", "raw_control_net_R", "beats_control"] if c in g.columns]
            focus_identical = [c for c in focus_cols if unique_non_null(g[c]) <= 1]
            if focus_identical or (numeric_metrics and identical_metric_count >= max(2, math.ceil(len(numeric_metrics) * 0.7))):
                identical_rows.append({
                    "path": safe_rel(p),
                    "root": canonical_root(p),
                    "group_key": repr(key),
                    "type_col": type_col,
                    "control_types": ";".join(map(str, sorted(g[type_col].dropna().astype(str).unique().tolist()))),
                    "rows_in_group": len(g),
                    "metric_cols_checked": ";".join(numeric_metrics),
                    "identical_metric_count": identical_metric_count,
                    "focus_identical_cols": ";".join(focus_identical),
                    "risk": "high_identical_control_metrics_across_types",
                })
    return identical_rows, source_rows


def scan_proxy_caps(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in files:
        if p.suffix.lower() not in {".csv", ".parquet"}:
            continue
        name = p.name.lower()
        if not any(tok in name for tok in ["summary", "replay", "ledger", "candidate", "audit"]):
            continue
        df = read_csv_sample(p, 100_000) if p.suffix.lower() == ".csv" else read_parquet_sample(p, 100_000)
        if df is None or df.empty:
            continue
        cols = set(df.columns)
        present = [c for c in PROXY_COLS if c in cols]
        if not present:
            continue
        row: dict[str, Any] = {
            "path": safe_rel(p),
            "root": canonical_root(p),
            "rows_sampled": len(df),
            "proxy_cols_present": ";".join(present),
        }
        for c in present:
            s = df[c]
            if s.dtype == bool or str(s.dtype) == "boolean":
                row[f"{c}_true"] = int(s.fillna(False).astype(bool).sum())
                row[f"{c}_false"] = int((~s.fillna(False).astype(bool)).sum())
            else:
                vals = s.dropna().astype(str).str.lower()
                row[f"{c}_true_like"] = int(vals.isin(["true", "1", "yes"]).sum())
                row[f"{c}_false_like"] = int(vals.isin(["false", "0", "no"]).sum())
        risk_parts = []
        if "funding_proxy_used" in present:
            vals = df["funding_proxy_used"].dropna().astype(str).str.lower()
            if vals.isin(["true", "1", "yes"]).any():
                risk_parts.append("funding_proxy_used")
        if "mark_proxy_used" in present:
            vals = df["mark_proxy_used"].dropna().astype(str).str.lower()
            if vals.isin(["true", "1", "yes"]).any():
                risk_parts.append("mark_proxy_used")
        if "funding_exact" in present:
            vals = df["funding_exact"].dropna().astype(str).str.lower()
            if vals.isin(["false", "0", "no"]).any():
                risk_parts.append("funding_not_exact")
        row["risk"] = ";".join(risk_parts) or "proxy_columns_present_review"
        rows.append(row)
    return rows


def scan_promotions(files: list[Path], roots_with_control_flags: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text_files = [p for p in files if p.suffix.lower() in {".json", ".md", ".csv"}]
    promo_re = re.compile("|".join(re.escape(t) for t in PROMO_TERMS), re.I)
    for p in text_files:
        try:
            text = p.read_text(errors="replace")
        except Exception:
            continue
        if not promo_re.search(text):
            continue
        root = canonical_root(p)
        snippets = []
        for m in promo_re.finditer(text):
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 90)
            snippets.append(text[start:end].replace("\n", " ")[:220])
            if len(snippets) >= 5:
                break
        rows.append({
            "path": safe_rel(p),
            "root": root,
            "promotion_terms_found": ";".join(sorted(set(t for t in PROMO_TERMS if re.search(re.escape(t), text, re.I)))),
            "root_has_control_integrity_flags": root in roots_with_control_flags,
            "risk": "high_demote_promotion_labels_pending_recheck" if root in roots_with_control_flags else "review_promotion_label_evidence_level",
            "snippets": " || ".join(snippets),
        })
    return rows


def scan_metric_lineage(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_cols = ["net_R", "PF", "pf", "profit_factor", "sharpe", "Sharpe", "max_dd", "drawdown", "CAGR", "cagr"]
    basis_cols = ["metric_basis", "evidence_level", "lineage", "source_type", "replay_scope", "window_scope"]
    for p in files:
        if p.suffix.lower() != ".csv":
            continue
        name = p.name.lower()
        if not any(tok in name for tok in ["summary", "decision", "triage", "sweep", "validation", "stress", "portfolio"]):
            continue
        df = read_csv_sample(p, 100_000)
        if df is None or df.empty:
            continue
        present_metrics = [c for c in metric_cols if c in df.columns]
        if not present_metrics:
            continue
        basis_present = [c for c in basis_cols if c in df.columns]
        evidence_bad = False
        bad_basis_values: list[str] = []
        for c in basis_present:
            vals = df[c].dropna().astype(str).str.lower().unique().tolist()[:20]
            for v in vals:
                if any(tok in v for tok in ["projection", "summary", "mae", "mfe", "support_only", "seed", "path_only", "sampled"]):
                    evidence_bad = True
                    bad_basis_values.append(f"{c}={v}")
        has_event_ledger_hint = any(c in df.columns for c in ["event_id", "trade_id", "decision_ts", "entry_ts", "exit_ts"])
        if evidence_bad or not has_event_ledger_hint:
            rows.append({
                "path": safe_rel(p),
                "root": canonical_root(p),
                "rows_sampled": len(df),
                "metric_cols_present": ";".join(present_metrics),
                "basis_cols_present": ";".join(basis_present),
                "bad_basis_values": ";".join(sorted(set(bad_basis_values)))[:1000],
                "has_event_ledger_hint_columns": has_event_ledger_hint,
                "risk": "metrics_without_event_ledger_or_bad_basis_needs_recheck",
            })
    return rows


def build_invalidated_inventory(payload: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reasons: dict[str, set[str]] = {}
    severity: dict[str, str] = {}

    def add(root: str, reason: str, sev: str) -> None:
        if not root:
            return
        reasons.setdefault(root, set()).add(reason)
        if sev == "invalidated" or severity.get(root) != "invalidated":
            severity[root] = sev

    for r in payload.get("identical_controls", []):
        add(str(r.get("root", "")), "identical_control_metrics_across_control_types", "invalidated")
    for r in payload.get("control_sources", []):
        if r.get("missing_source_window_columns"):
            add(str(r.get("root", "")), "control_source_window_columns_missing", "invalidated")
    for r in payload.get("metric_lineage", []):
        add(str(r.get("root", "")), "metrics_without_event_ledger_or_bad_basis", "invalidated")
    for r in payload.get("proxy_caps", []):
        add(str(r.get("root", "")), f"proxy_cap:{r.get('risk', '')}", "capped")
    for r in payload.get("promotions", []):
        add(str(r.get("root", "")), "promotion_label_requires_recheck", "capped")
    for r in payload.get("protected", []):
        add(str(r.get("root", "")), "protected_timestamp_present_review_context", "invalidated")

    # Roots known to consume leaky regime features must be quarantined even if
    # their result artifacts do not expose the leakage directly.
    leaky_regime_roots = {
        "phase_qlmg_regime_stack_and_smart_sweep_20260625_v1",
        "phase_qlmg_alpha_discovery_marathon_20260625_v1",
        "phase_qlmg_simple_alpha_plus_d4_20260626_v1",
        "phase_qlmg_simple_alpha_plus_d4_20260626_v1_amended_full_20260626_162555",
    }
    discovered_roots = set(reasons)
    for p in RESULTS.glob("phase_qlmg*"):
        if p.is_dir() and any(p.name.startswith(root) for root in leaky_regime_roots):
            discovered_roots.add(p.name)
            add(p.name, "uses_or_descends_from_future_path_regime_features_or_full_sample_wick_gate", "invalidated")

    rows: list[dict[str, Any]] = []
    reruns: list[dict[str, Any]] = []
    for root in sorted(discovered_roots):
        reason_text = ";".join(sorted(reasons.get(root, set())))
        sev = severity.get(root, "capped")
        if "identical_control" in reason_text:
            next_action = "supersede_with_real_control_rebuild_then_recompute_from_event_rows"
        elif "future_path_regime" in reason_text:
            next_action = "quarantine_old_sweep_and_rebuild_only_with_pit_safe_regime_labels"
        elif "metrics_without_event" in reason_text:
            next_action = "recompute_metrics_from_event_level_trade_ledger_or_demote_to_support_only"
        elif "proxy_cap" in reason_text:
            next_action = "cap_labels_until_exact_mark_and_funding_rebuild"
        else:
            next_action = "manual_recheck_before_ranking"
        rows.append({
            "run_root": root,
            "classification": sev,
            "invalidated_or_capped_reasons": reason_text,
            "hypothesis_status": "preserved_not_rejected",
            "ranking_allowed": False,
            "next_action": next_action,
        })
        reruns.append({
            "run_root": root,
            "rerun_action": next_action,
            "auto_rerun_safe_now": next_action in {
                "supersede_with_real_control_rebuild_then_recompute_from_event_rows",
                "recompute_metrics_from_event_level_trade_ledger_or_demote_to_support_only",
                "cap_labels_until_exact_mark_and_funding_rebuild",
            },
            "auto_rerun_command_or_stage": "tools/run_qlmg_real_control_rebuild.py; tools/run_qlmg_evidence_remediation_family_repair.py; tools/run_qlmg_corrected_event_level_development_sweep.py if gate passes",
            "do_not_reject_hypothesis": True,
        })
    return rows, reruns


def _looks_like_timestamp_col(name: str) -> bool:
    n = name.lower()
    exact = {
        "ts", "timestamp", "decision_ts", "entry_ts", "exit_ts", "open_ts", "close_ts",
        "source_close_ts", "event_ts", "event_anchor_ts", "start_ts", "end_ts",
        "start", "end", "date", "month", "open_time", "close_time",
    }
    if n in exact:
        return True
    suffixes = ("_ts", "_ts_utc", "_time", "_time_utc", "_date", "_date_utc", "_dt")
    if not n.endswith(suffixes):
        return False
    # Avoid parsing descriptive/status columns that often contain free text.
    bad_tokens = ("status", "reason", "precision", "source", "report", "label", "verdict", "path")
    return not any(tok in n for tok in bad_tokens)


def _read_timestamp_columns(path: Path, max_rows: int = 50_000) -> pd.DataFrame | None:
    if path.suffix.lower() == ".csv":
        df = read_csv_sample(path, max_rows)
        if df is None:
            return None
        cols = [c for c in df.columns if _looks_like_timestamp_col(str(c))]
        return df[cols].copy() if cols else None
    if path.suffix.lower() == ".parquet" and pq is not None:
        try:
            pf = pq.ParquetFile(path)
            cols = [c for c in pf.schema_arrow.names if _looks_like_timestamp_col(str(c))]
            if not cols:
                return None
            table = pf.read(columns=cols)
            df = table.to_pandas()
            return df.head(max_rows).copy() if len(df) > max_rows else df
        except Exception:
            return None
    return None


def scan_protected_timestamps(files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in files:
        if p.suffix.lower() not in {".csv", ".parquet"}:
            continue
        df = _read_timestamp_columns(p, 50_000)
        if df is None or df.empty:
            continue
        for c in list(df.columns)[:30]:
            try:
                vals = df[c].dropna().astype(str)
                if vals.empty:
                    continue
                sample_vals = vals.head(1000)
                # Avoid dateutil on long text. Only parse ISO-ish timestamps/dates or unix epochs.
                looks_ts = sample_vals.str.match(r"^(\d{4}-\d{2}-\d{2}|\d{10}(?:\.\d+)?|\d{13})")
                if not bool(looks_ts.any()):
                    continue
                vals = vals[vals.str.match(r"^(\d{4}-\d{2}-\d{2}|\d{10}(?:\.\d+)?|\d{13})")].head(50_000)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s = pd.to_datetime(vals, utc=True, errors="coerce", format="mixed")
            except Exception:
                continue
            if s.notna().sum() == 0:
                continue
            n = int((s >= PROTECTED_TS).sum())
            if n:
                rows.append({
                    "path": safe_rel(p),
                    "root": canonical_root(p),
                    "column": c,
                    "protected_rows_in_sample": n,
                    "max_ts_in_sample": str(s.max()),
                    "risk": "protected_timestamp_present_review_context",
                })
    return rows


def write_report(run_root: Path, payload: dict[str, Any]) -> None:
    controls = payload["identical_controls"]
    source = payload["control_sources"]
    code = payload["code_patterns"]
    future_code = payload.get("future_leakage_code_patterns", [])
    proxy = payload["proxy_caps"]
    promo = payload["promotions"]
    lineage = payload["metric_lineage"]
    protected = payload["protected"]

    high_roots = sorted({r["root"] for r in controls if r.get("root")})
    missing_source_count = sum(1 for r in source if r.get("missing_source_window_columns"))
    report = f"""# QLMG Global Result Invalidation Audit

Run root: `{safe_rel(run_root)}`  
Generated UTC: `{datetime.now(timezone.utc).isoformat()}`

## Direct Answer

The placeholder-control issue was caused by runner implementations that generated shape-compatible control/null artifacts from deterministic formulas or copied/shared distributions instead of constructing independent matched windows. That was an implementation shortcut and should have been fail-closed or explicitly marked `support_only`. It is not acceptable evidence for control uplift, prelead confirmation, or validation-style decisions.

## High-Impact Findings

- Suspicious control-code patterns found: `{len(code)}` rows.
- Future-leakage code patterns requiring review/quarantine found: `{len(future_code)}` rows.
- Identical control metrics across multiple control types found: `{len(controls)}` grouped flags.
- Control/null files with missing source-window columns: `{missing_source_count}` files.
- Metric/proxy cap files found: `{len(proxy)}` files.
- Promotion-label files requiring demotion/recheck: `{len(promo)}` files.
- Metric-lineage risk files: `{len(lineage)}` files.
- Protected timestamp scan flags: `{len(protected)}` sample flags.

## Affected Roots With Identical Control-Type Metrics

{os.linesep.join(f'- `{r}`' for r in high_roots) if high_roots else '- None detected by this scan.'}

## What Is Invalidated

- Any conclusion depending on `same_symbol`, `same_regime`, `nearest_neighbor_vol_liq_funding_oi`, `generic_momentum`, or `A2_A3_overlap` uplift from flagged files.
- Any `prelead`, `confirmed`, `survives`, or validation-style label derived from flagged controls/nulls.
- Any PF/DD/Sharpe/CAGR computed from summary projections rather than event-level trade ledgers.
- Any Branch X promotion-style conclusion that relies on 5m/OHLCV or proxy execution instead of 1m mark/depth/trade/liquidation evidence.

## What May Still Be Usable

- Raw event ledgers and replay rows where `net_R` is computed per event and can be independently summed.
- Path/MAE/MFE diagnostics as path diagnostics only, not trade performance.
- Seed/idea preservation outputs, provided they are not treated as ranked trade evidence.
- Branch X status/capture requirements, not Branch X PnL promotion.

## Required Remediation

1. Quarantine flagged control/null artifacts and deprecated labels.
2. Rebuild controls from real event windows with explicit source columns: `control_event_id`, `control_symbol`, `control_decision_ts`, matching basis, and window identifiers.
3. Recompute PF/DD/Sharpe/CAGR only from event-level ledgers.
4. Cap every candidate with `funding_proxy_used=True`, `mark_proxy_used=True`, or `funding_exact=False`.
5. Do not run C2/B1 validation until real controls and broader event ledgers exist.

## Key Output Tables

- `inventory/scanned_files.csv`
- `code/suspicious_control_code_patterns.csv`
- `code/future_leakage_code_patterns.csv`
- `controls/identical_control_metric_flags.csv`
- `controls/control_source_columns_audit.csv`
- `metrics/proxy_mark_funding_caps.csv`
- `metrics/promotion_label_risk.csv`
- `metrics/metric_lineage_risk.csv`
- `seal/protected_timestamp_scan.csv`
- `inventory/invalidated_or_capped_run_inventory.csv`
- `inventory/clean_rerun_manifest.csv`

## Current Operator Decision

`repair_controls_before_any_validation`

No output from this audit authorizes live trading, sealed validation, final validation, or production deployment.
"""
    (run_root / "QLMG_GLOBAL_RESULT_INVALIDATION_AUDIT_REPORT.md").write_text(report)

    decision = {
        "audit_verdict": "material_control_integrity_issue_found" if controls or any(r.get("risk") == "high_placeholder_control_risk" for r in code) else "no_material_placeholder_control_issue_detected",
        "operator_decision": "repair_controls_before_any_validation",
        "control_uplift_trustworthy": False if controls or missing_source_count else None,
        "promotion_labels_require_recheck": bool(promo),
        "flag_counts": {
            "code_patterns": len(code),
            "future_leakage_code_patterns": len(future_code),
            "identical_controls": len(controls),
            "missing_control_source_files": missing_source_count,
            "proxy_caps": len(proxy),
            "promotion_label_risks": len(promo),
            "metric_lineage_risks": len(lineage),
            "protected_timestamp_flags": len(protected),
        },
        "final_holdout_untouched_by_audit": True,
        "no_validation_or_live_claims": True,
    }
    (run_root / "decision_summary.json").write_text(json.dumps(decision, indent=2, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(BASE_RUN))
    ap.add_argument("--max-output-gb", type=float, default=40.0)
    args = ap.parse_args()

    run_root = make_run_root(Path(args.run_root))
    run_root.mkdir(parents=True, exist_ok=True)
    for d in ["inventory", "code", "controls", "metrics", "seal", "logs"]:
        (run_root / d).mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if free_gb < 5:
        raise SystemExit(f"free disk below hard stop: {free_gb:.2f}GB")

    files = discover_files()
    inv = inventory(files)
    write_csv(run_root / "inventory" / "scanned_files.csv", inv)

    code_rows = scan_code()
    write_csv(run_root / "code" / "suspicious_control_code_patterns.csv", code_rows)

    future_leakage_rows = scan_future_leakage_code()
    write_csv(run_root / "code" / "future_leakage_code_patterns.csv", future_leakage_rows)

    identical, source_rows = scan_control_files(files)
    write_csv(run_root / "controls" / "identical_control_metric_flags.csv", identical)
    write_csv(run_root / "controls" / "control_source_columns_audit.csv", source_rows)

    proxy_rows = scan_proxy_caps(files)
    write_csv(run_root / "metrics" / "proxy_mark_funding_caps.csv", proxy_rows)

    roots_with_control_flags = {r["root"] for r in identical if r.get("root")}
    promo_rows = scan_promotions(files, roots_with_control_flags)
    write_csv(run_root / "metrics" / "promotion_label_risk.csv", promo_rows)

    lineage_rows = scan_metric_lineage(files)
    write_csv(run_root / "metrics" / "metric_lineage_risk.csv", lineage_rows)

    protected_rows = scan_protected_timestamps(files)
    write_csv(run_root / "seal" / "protected_timestamp_scan.csv", protected_rows)

    payload = {
        "code_patterns": code_rows,
        "future_leakage_code_patterns": future_leakage_rows,
        "identical_controls": identical,
        "control_sources": source_rows,
        "proxy_caps": proxy_rows,
        "promotions": promo_rows,
        "metric_lineage": lineage_rows,
        "protected": protected_rows,
    }
    invalidated, reruns = build_invalidated_inventory(payload)
    write_csv(run_root / "inventory" / "invalidated_or_capped_run_inventory.csv", invalidated)
    write_csv(run_root / "inventory" / "clean_rerun_manifest.csv", reruns)
    write_report(run_root, payload)
    print(json.dumps({"run_root": str(run_root), "status": "complete", "flags": {k: len(v) for k, v in payload.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
