#!/usr/bin/env python3
"""Reusable evidence-contract validators for QLMG research runners.

These checks are intentionally conservative. They do not prove absence of all
leakage; they block known unsafe evidence shapes from becoming rankable.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PROTECTED_TS = pd.Timestamp("2026-01-01T00:00:00Z")
PROMOTION_LABEL_RE = re.compile(
    r"prelead|confirmed|validated|validation|beats[_ -]?controls|stress[_ -]?survives|promote",
    re.IGNORECASE,
)
FUTURE_FIELD_RE = re.compile(r"(^|[^a-z0-9])(\d+h_|future|mfe|mae|realized_|forward_)", re.IGNORECASE)
SYNTHETIC_CONTROL_RE = re.compile(r"placeholder|synthetic|fabricated|dummy|copied", re.IGNORECASE)
CURRENT_ONLY_RE = re.compile(r"current[_ -]?only|taxonomy[_ -]?proxy|backfill", re.IGNORECASE)

EVENT_TRADE_REQUIRED_FIELDS = [
    "candidate_id",
    "family",
    "branch_id",
    "symbol",
    "decision_ts",
    "side",
    "entry_ts",
    "entry_price",
    "entry_price_source",
    "stop_price",
    "exit_rule",
    "exit_ts",
    "exit_price",
    "exit_reason",
    "gross_R",
    "fees_R",
    "slippage_R",
    "funding_R",
    "net_R",
    "mark_liquidation_flag",
    "same_bar_ambiguity_flag",
    "funding_timestamps_crossed",
    "mark_available",
    "funding_exact",
    "lifecycle_status",
    "data_tier",
    "control_group_id",
    "source_data_hash",
]

CONTROL_REQUIRED_FIELDS = [
    "control_event_id",
    "control_symbol",
    "control_decision_ts",
    "matched_candidate_id",
    "matching_basis",
    "source_window_id",
    "feature_source_ts",
]

METRIC_COLUMNS = {"PF", "pf", "max_dd_R", "drawdown", "sharpe", "Sharpe", "CAGR", "cagr", "win_rate"}


class ContractViolation(ValueError):
    """Raised when an evidence contract fails closed."""


@dataclass(frozen=True)
class ContractCheckResult:
    status: str
    violations: list[str]
    warnings: list[str]
    rows_checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.violations and self.status == "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "violations": self.violations,
            "warnings": self.warnings,
            "rows_checked": self.rows_checked,
            "passed": self.passed,
        }


def _to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _bool_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default).astype(bool)
    return s.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def require_no_protected_timestamps(df: pd.DataFrame, ts_cols: Iterable[str] | None = None, *, label: str = "dataframe") -> ContractCheckResult:
    violations: list[str] = []
    warnings: list[str] = []
    cols = list(ts_cols) if ts_cols is not None else [c for c in df.columns if c.endswith("_ts") or "timestamp" in c.lower() or c in {"decision_ts", "entry_ts", "exit_ts"}]
    for col in cols:
        if col not in df.columns:
            continue
        ts = _to_utc(df[col])
        bad = ts >= PROTECTED_TS
        if bool(bad.any()):
            violations.append(f"{label}:{col}:protected_rows={int(bad.sum())}")
    return ContractCheckResult("fail" if violations else "pass", violations, warnings, len(df))


def validate_event_trade_schema(df: pd.DataFrame, *, require_all_fields: bool = True, allow_empty: bool = False) -> ContractCheckResult:
    violations: list[str] = []
    warnings: list[str] = []
    if df.empty:
        if allow_empty:
            return ContractCheckResult("pass", [], ["empty_event_ledger"], 0)
        return ContractCheckResult("fail", ["empty_event_ledger"], [], 0)
    missing = [c for c in EVENT_TRADE_REQUIRED_FIELDS if c not in df.columns]
    if require_all_fields and missing:
        violations.append("missing_event_trade_fields:" + ",".join(missing))
    elif missing:
        warnings.append("missing_optional_event_trade_fields:" + ",".join(missing))
    if "net_R" not in df.columns and "net_R_variant" not in df.columns and "source_net_R" not in df.columns:
        violations.append("missing_net_R_like_column")
    if "decision_ts" in df.columns:
        r = require_no_protected_timestamps(df, ["decision_ts", "entry_ts", "exit_ts"], label="event_trade_schema")
        violations.extend(r.violations)
    if "side" in df.columns:
        bad_side = ~df["side"].astype(str).str.lower().isin({"long", "short"})
        if bool(bad_side.any()):
            violations.append(f"invalid_side_rows={int(bad_side.sum())}")
    return ContractCheckResult("fail" if violations else "pass", violations, warnings, len(df))


def validate_control_rows(df: pd.DataFrame, *, allow_empty: bool = False) -> ContractCheckResult:
    violations: list[str] = []
    warnings: list[str] = []
    if df.empty:
        if allow_empty:
            return ContractCheckResult("pass", [], ["empty_control_rows"], 0)
        return ContractCheckResult("fail", ["empty_control_rows"], [], 0)
    missing = [c for c in CONTROL_REQUIRED_FIELDS if c not in df.columns]
    if missing:
        violations.append("missing_control_fields:" + ",".join(missing))
    for col in ["control_event_id", "source_window_id", "matching_basis"]:
        if col in df.columns:
            bad = df[col].isna() | df[col].astype(str).str.strip().isin({"", "nan", "None"})
            if bool(bad.any()):
                violations.append(f"blank_{col}_rows={int(bad.sum())}")
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ["basis", "type", "source", "label", "status"])]
    if text_cols:
        joined = df[text_cols].astype(str).agg(" ".join, axis=1)
        bad = joined.str.contains(SYNTHETIC_CONTROL_RE, na=False)
        if bool(bad.any()):
            violations.append(f"synthetic_or_placeholder_control_rows={int(bad.sum())}")
    if {"matched_candidate_id", "control_event_id", "source_window_id"}.issubset(df.columns):
        dup = df.duplicated(["matched_candidate_id", "control_event_id", "source_window_id"])
        if bool(dup.any()):
            violations.append(f"duplicate_control_source_rows={int(dup.sum())}")
    r = require_no_protected_timestamps(df, ["control_decision_ts", "feature_source_ts"], label="control_rows")
    violations.extend(r.violations)
    if {"feature_source_ts", "control_decision_ts"}.issubset(df.columns):
        fs = _to_utc(df["feature_source_ts"])
        ds = _to_utc(df["control_decision_ts"])
        bad = fs.notna() & ds.notna() & (fs > ds)
        if bool(bad.any()):
            violations.append(f"control_feature_after_decision_rows={int(bad.sum())}")
    return ContractCheckResult("fail" if violations else "pass", violations, warnings, len(df))


def validate_funding_mark_flags(df: pd.DataFrame, *, liquidation_possible: bool = True) -> ContractCheckResult:
    violations: list[str] = []
    warnings: list[str] = []
    if df.empty:
        return ContractCheckResult("pass", [], ["empty_funding_mark_frame"], 0)
    funding_exact = _bool_series(df, "funding_exact", False)
    funding_proxy = _bool_series(df, "funding_proxy_used", False)
    mark_available = _bool_series(df, "mark_available", False)
    mark_proxy = _bool_series(df, "mark_proxy_used", False)
    if bool((funding_exact & funding_proxy).any()):
        violations.append(f"funding_proxy_treated_exact_rows={int((funding_exact & funding_proxy).sum())}")
    if bool((mark_available & mark_proxy).any()):
        violations.append(f"mark_proxy_treated_available_rows={int((mark_available & mark_proxy).sum())}")
    if liquidation_possible and "mark_available" in df.columns and bool((~mark_available & ~mark_proxy).any()):
        warnings.append(f"mark_missing_without_proxy_flag_rows={int((~mark_available & ~mark_proxy).sum())}")
    crossed = pd.to_numeric(df.get("funding_timestamps_crossed", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    if bool(((crossed > 0) & ~funding_exact & ~funding_proxy).any()):
        warnings.append(f"funding_crossed_missing_without_proxy_flag_rows={int(((crossed > 0) & ~funding_exact & ~funding_proxy).sum())}")
    return ContractCheckResult("fail" if violations else "pass", violations, warnings, len(df))


def validate_pit_feature_timestamps(df: pd.DataFrame, *, decision_col: str = "decision_ts", feature_ts_cols: Sequence[str] = ("feature_source_ts", "source_close_ts")) -> ContractCheckResult:
    violations: list[str] = []
    warnings: list[str] = []
    if df.empty:
        return ContractCheckResult("pass", [], ["empty_pit_frame"], 0)
    if decision_col not in df.columns:
        return ContractCheckResult("fail", [f"missing_decision_col:{decision_col}"], [], len(df))
    decision = _to_utc(df[decision_col])
    for col in feature_ts_cols:
        if col not in df.columns:
            continue
        src = _to_utc(df[col])
        bad = src.notna() & decision.notna() & (src > decision)
        if bool(bad.any()):
            violations.append(f"{col}_after_{decision_col}_rows={int(bad.sum())}")
    future_cols = [c for c in df.columns if FUTURE_FIELD_RE.search(str(c))]
    if future_cols:
        violations.append("future_path_fields_present:" + ",".join(sorted(future_cols)[:20]))
    r = require_no_protected_timestamps(df, [decision_col, *feature_ts_cols], label="pit_features")
    violations.extend(r.violations)
    return ContractCheckResult("fail" if violations else "pass", violations, warnings, len(df))


def validate_no_projected_metric_promotion(df: pd.DataFrame) -> ContractCheckResult:
    violations: list[str] = []
    warnings: list[str] = []
    if df.empty:
        return ContractCheckResult("pass", [], ["empty_metric_frame"], 0)
    label_cols = [c for c in df.columns if any(k in c.lower() for k in ["label", "verdict", "status", "decision"])]
    labels = df[label_cols].astype(str).agg(" ".join, axis=1) if label_cols else pd.Series("", index=df.index)
    promoted = labels.str.contains(PROMOTION_LABEL_RE, na=False)
    lineage = df.get("metric_lineage", df.get("metric_basis", pd.Series("", index=df.index))).astype(str).str.lower()
    ledger_ok = lineage.str.contains("event_level_trade_ledger|event_rows|event_level", na=False)
    if bool((promoted & ~ledger_ok).any()):
        violations.append(f"promotion_without_event_trade_ledger_rows={int((promoted & ~ledger_ok).sum())}")
    if METRIC_COLUMNS.intersection(df.columns) and "event_level_trade_ledger_exists" in df.columns:
        exists = _bool_series(df, "event_level_trade_ledger_exists", False)
        if bool((~exists).any()):
            violations.append(f"metrics_without_event_ledger_rows={int((~exists).sum())}")
    return ContractCheckResult("fail" if violations else "pass", violations, warnings, len(df))


def validate_no_synthetic_controls(df: pd.DataFrame) -> ContractCheckResult:
    if df.empty:
        return ContractCheckResult("pass", [], ["empty_frame"], 0)
    text = df.astype(str).agg(" ".join, axis=1)
    bad = text.str.contains(SYNTHETIC_CONTROL_RE, na=False)
    violations = [f"synthetic_control_text_rows={int(bad.sum())}"] if bool(bad.any()) else []
    return ContractCheckResult("fail" if violations else "pass", violations, [], len(df))


def validate_no_current_only_taxonomy_rankable(df: pd.DataFrame) -> ContractCheckResult:
    violations: list[str] = []
    if df.empty:
        return ContractCheckResult("pass", [], ["empty_taxonomy_frame"], 0)
    rankable = _bool_series(df, "rankable", False) | _bool_series(df, "rankable_pit_sector_seed", False)
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ["taxonomy", "source", "label", "status", "mode"])]
    if text_cols:
        text = df[text_cols].astype(str).agg(" ".join, axis=1)
        bad = rankable & text.str.contains(CURRENT_ONLY_RE, na=False)
        if bool(bad.any()):
            violations.append(f"current_only_taxonomy_rankable_rows={int(bad.sum())}")
    return ContractCheckResult("fail" if violations else "pass", violations, [], len(df))


def artifact_risk_scan(df: pd.DataFrame, *, path: str = "") -> list[dict[str, Any]]:
    risks: list[dict[str, Any]] = []
    if df.empty:
        return risks
    present = set(df.columns)
    if {"control_type", "candidate_id"}.issubset(present):
        metric = next((c for c in ["control_net_R", "normalized_control_net_R", "control_uplift_R", "raw_control_net_R"] if c in present), None)
        if metric is not None:
            for cid, g in df.groupby("candidate_id", dropna=False):
                vals = g.groupby("control_type")[metric].first().dropna()
                if len(vals) >= 2 and vals.nunique(dropna=True) == 1:
                    risks.append({"path": path, "risk": "identical_control_values_across_types", "candidate_id": cid, "rows": int(len(g))})
    if any(c in present for c in ["control_type", "matched_candidate_id"]):
        missing = [c for c in CONTROL_REQUIRED_FIELDS if c not in present]
        if missing:
            risks.append({"path": path, "risk": "missing_control_ids_or_window_ids", "missing": ",".join(missing), "rows": int(len(df))})
    if METRIC_COLUMNS.intersection(present):
        if "event_level_trade_ledger_exists" not in present and "metric_lineage" not in present and "metric_basis" not in present:
            risks.append({"path": path, "risk": "metrics_without_event_ledger_lineage", "metric_columns": ",".join(sorted(METRIC_COLUMNS.intersection(present)))})
    fm = validate_funding_mark_flags(df)
    for v in fm.violations:
        risks.append({"path": path, "risk": v})
    pit = validate_pit_feature_timestamps(df) if "decision_ts" in present else ContractCheckResult("pass", [], [])
    for v in pit.violations:
        risks.append({"path": path, "risk": v})
    tax = validate_no_current_only_taxonomy_rankable(df)
    for v in tax.violations:
        risks.append({"path": path, "risk": v})
    protected = require_no_protected_timestamps(df)
    for v in protected.violations:
        risks.append({"path": path, "risk": v})
    return risks


def scan_output_tree_for_protected(root: Path) -> ContractCheckResult:
    violations: list[str] = []
    warnings: list[str] = []
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".jsonl", ".md", ".parquet"}]
    for p in files:
        rel_path = p.relative_to(root)
        rel_str = str(rel_path)
        if rel_str in {"watch_status.json", "run_context.json"} or rel_str.startswith("notifications/") or rel_str.startswith("live_capture/"):
            # Operational timestamps and forward live-capture inventory are not candidate-selection rows.
            # The live-capture subtree is separately provenance-capped and inventory-only.
            continue
        try:
            if p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
                r = require_no_protected_timestamps(df, label=rel_str)
                violations.extend(r.violations)
            else:
                txt = p.read_text(errors="ignore")
                if "2026-01-01" in txt or re.search(r"2026-(0[1-9]|1[0-2])-", txt):
                    # Allow contract text that names the protected boundary, but flag generated rows/reports for review.
                    if "protected" not in txt[:2000].lower() and "holdout" not in txt[:2000].lower():
                        violations.append(f"{rel_str}:possible_protected_timestamp_text")
        except Exception as exc:
            warnings.append(f"scan_failed:{rel_str}:{type(exc).__name__}:{exc}")
    status = "fail" if violations else "pass"
    return ContractCheckResult(status, violations, warnings, len(files))


def assert_pass(result: ContractCheckResult) -> None:
    if not result.passed:
        raise ContractViolation(";".join(result.violations or result.warnings))


def result_to_jsonable(result: ContractCheckResult) -> dict[str, Any]:
    return result.as_dict()
