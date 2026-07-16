#!/usr/bin/env python3
"""Real event-ledger control construction for QLMG research.

This module intentionally does not synthesize control R values. Controls are
selected from event-level source rows and every selected control carries source
row/window identifiers so summaries can be audited back to underlying events.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROTECTED_TS = pd.Timestamp("2026-01-01T00:00:00Z")
CONTROL_TYPES = [
    "same_symbol",
    "same_regime",
    "nearest_neighbor_vol_liq_funding_oi",
    "generic_momentum",
    "A2_A3_overlap",
]


def stable_hash(*parts: object, n: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:n]


def to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def validate_no_protected_rows(df: pd.DataFrame, ts_cols: Iterable[str]) -> None:
    for col in ts_cols:
        if col not in df.columns:
            continue
        ts = to_utc(df[col])
        bad = ts >= PROTECTED_TS
        if bool(bad.any()):
            raise ValueError(f"protected timestamp rows found in {col}: {int(bad.sum())}")


def pf(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    wins = float(vals[vals > 0].sum())
    losses = float(vals[vals < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return wins / abs(losses)


def max_dd(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if vals.size == 0:
        return 0.0
    curve = np.cumsum(vals)
    peak = np.maximum.accumulate(curve)
    return float(np.min(curve - peak))


def normalize_control_net(candidate_events: int, raw_control_net: float, control_events: int) -> float:
    if candidate_events <= 0 or control_events <= 0 or not np.isfinite(raw_control_net):
        return float("nan")
    return float(raw_control_net) * float(candidate_events) / float(control_events)


def _first_present(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def standardize_event_ledger(df: pd.DataFrame, source_path: str, family_hint: str | None = None) -> pd.DataFrame:
    """Return canonical event rows with a candidate key and source identifiers."""
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "family" not in out.columns:
        out["family"] = family_hint or "unknown"
    out["family"] = out["family"].fillna(family_hint or "unknown").astype(str)
    key_col = _first_present(out, ["variant_id", "candidate_id", "definition_id"])
    if key_col is None:
        raise ValueError(f"ledger has no candidate identifier: {source_path}")
    out["candidate_key"] = out[key_col].astype(str)
    # Mixed-family ledgers often have variant_id for A2/A3 but null variant_id for B1/C2.
    # Fall back row-wise so sidecars do not collapse into a literal "nan" candidate.
    if "candidate_id" in out.columns:
        missing_key = out["candidate_key"].isin(["", "nan", "None", "NaT"]) | out["candidate_key"].isna()
        out.loc[missing_key, "candidate_key"] = out.loc[missing_key, "candidate_id"].astype(str)
    if "candidate_id" not in out.columns:
        out["candidate_id"] = out["candidate_key"]
    if "event_id" not in out.columns:
        base_cols = [c for c in ["family", "candidate_key", "symbol", "decision_ts", "entry_ts"] if c in out.columns]
        out["event_id"] = [stable_hash(*(row[c] for c in base_cols)) for _, row in out[base_cols].iterrows()]
    net_col = _first_present(out, ["net_R_variant", "net_R_repair", "net_R"])
    if net_col is None:
        raise ValueError(f"ledger has no net_R-like column: {source_path}")
    out["source_net_R"] = pd.to_numeric(out[net_col], errors="coerce")
    for col in ["decision_ts", "entry_ts", "exit_ts"]:
        if col in out.columns:
            out[col] = to_utc(out[col])
    validate_no_protected_rows(out, ["decision_ts", "entry_ts", "exit_ts"])
    if "symbol" not in out.columns:
        out["symbol"] = "unknown"
    if "parent_regime" not in out.columns:
        out["parent_regime"] = "unknown"
    if "risk_bps_used" not in out.columns:
        out["risk_bps_used"] = np.nan
    if "mark_price_available" in out.columns and "mark_available" not in out.columns:
        out["mark_available"] = out["mark_price_available"]
    if "mark_available" not in out.columns:
        out["mark_available"] = False
    if "funding_exact" not in out.columns:
        out["funding_exact"] = False
    if "mark_proxy_used" not in out.columns:
        out["mark_proxy_used"] = ~out["mark_available"].fillna(False).astype(bool)
    if "funding_proxy_used" not in out.columns:
        out["funding_proxy_used"] = ~out["funding_exact"].fillna(False).astype(bool)
    out["decision_month"] = out["decision_ts"].dt.strftime("%Y-%m") if "decision_ts" in out.columns else "unknown"
    out["source_path"] = source_path
    out["source_row_id"] = [
        stable_hash(source_path, r.get("family"), r.get("candidate_key"), r.get("event_id"), r.get("symbol"), r.get("decision_ts"))
        for _, r in out.iterrows()
    ]
    out["source_window_id"] = [
        stable_hash(r.get("symbol"), r.get("decision_ts"), r.get("entry_ts"), r.get("exit_ts"), r.get("event_id"))
        for _, r in out.iterrows()
    ]
    keep = [
        "source_row_id", "source_window_id", "source_path", "event_id", "candidate_id", "candidate_key", "family", "symbol",
        "decision_ts", "entry_ts", "exit_ts", "side", "source_net_R", "risk_bps_used", "parent_regime", "decision_month",
        "mark_available", "funding_exact", "mark_proxy_used", "funding_proxy_used", "label_cap_reason", "metric_basis",
    ]
    return out[[c for c in keep if c in out.columns]].copy()


def metric_summary(df: pd.DataFrame, candidate_key: str, family: str) -> dict[str, Any]:
    vals = pd.to_numeric(df["source_net_R"], errors="coerce")
    events = int(vals.notna().sum())
    wins = int((vals > 0).sum())
    symbols = int(df["symbol"].nunique()) if "symbol" in df.columns else 0
    months = int(df["decision_month"].nunique()) if "decision_month" in df.columns else 0
    return {
        "candidate_key": candidate_key,
        "candidate_id": str(df["candidate_id"].iloc[0]) if "candidate_id" in df.columns and len(df) else candidate_key,
        "family": family,
        "events": events,
        "net_R": float(vals.sum()) if events else float("nan"),
        "PF": pf(vals),
        "win_rate": wins / events if events else float("nan"),
        "avg_R": float(vals.mean()) if events else float("nan"),
        "median_R": float(vals.median()) if events else float("nan"),
        "max_dd_R": max_dd(vals.reset_index(drop=True)),
        "symbols": symbols,
        "months": months,
        "mark_available_all": bool(df.get("mark_available", pd.Series(False, index=df.index)).fillna(False).astype(bool).all()) if len(df) else False,
        "funding_exact_all": bool(df.get("funding_exact", pd.Series(False, index=df.index)).fillna(False).astype(bool).all()) if len(df) else False,
        "mark_proxy_used_any": bool(df.get("mark_proxy_used", pd.Series(False, index=df.index)).fillna(False).astype(bool).any()) if len(df) else False,
        "funding_proxy_used_any": bool(df.get("funding_proxy_used", pd.Series(False, index=df.index)).fillna(False).astype(bool).any()) if len(df) else False,
    }


def _candidate_subset(pool: pd.DataFrame, candidate_key: str) -> pd.DataFrame:
    return pool[pool["candidate_key"].astype(str).eq(str(candidate_key))].copy()


def _exclude_self(pool: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    own_rows = set(cand["source_row_id"].astype(str))
    own_events = set(cand["event_id"].astype(str))
    own_key = set(cand["candidate_key"].astype(str))
    m = ~pool["source_row_id"].astype(str).isin(own_rows)
    m &= ~((pool["event_id"].astype(str).isin(own_events)) & (pool["candidate_key"].astype(str).isin(own_key)))
    return pool[m].copy()


def _cap_index_part(df: pd.DataFrame, key: str, cap: int = 50_000) -> pd.DataFrame:
    if len(df) <= cap:
        return df
    start = int(stable_hash("index-cap", key, n=12), 16) % len(df)
    idx = (start + np.arange(cap)) % len(df)
    return df.iloc[idx]


def _concat_index_parts(index: dict[str, pd.DataFrame], keys: set[str]) -> pd.DataFrame:
    parts = [_cap_index_part(index[k], k) for k in keys if k in index]
    return pd.concat(parts, ignore_index=False) if parts else pd.DataFrame(columns=next(iter(index.values())).columns if index else [])


def _drop_self_rows(df: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    own_rows = set(cand["source_row_id"].astype(str))
    own_events = set(cand["event_id"].astype(str))
    own_key = set(cand["candidate_key"].astype(str))
    m = ~df["source_row_id"].astype(str).isin(own_rows)
    m &= ~((df["event_id"].astype(str).isin(own_events)) & (df["candidate_key"].astype(str).isin(own_key)))
    return df.loc[m]


def _eligible_pool(control_type: str, cand: pd.DataFrame, pool: pd.DataFrame, indexes: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
    fam = str(cand["family"].iloc[0]) if len(cand) else "unknown"
    symbols = set(cand["symbol"].astype(str))
    regimes = set(cand["parent_regime"].astype(str)) - {"unknown", "nan", "None", ""}
    months = set(cand["decision_month"].astype(str))
    empty = pool.iloc[0:0]
    if control_type == "same_symbol":
        b = _concat_index_parts(indexes["by_symbol"], symbols)
        return _drop_self_rows(b, cand), "same symbol, excluding same candidate/source rows", "exact_symbol"
    if control_type == "same_regime":
        if not regimes:
            return empty, "parent_regime missing for candidate rows", "missing_parent_regime"
        b = _concat_index_parts(indexes["by_regime"], regimes)
        return _drop_self_rows(b, cand), "same parent_regime, excluding same candidate/source rows", "same_parent_regime"
    if control_type == "nearest_neighbor_vol_liq_funding_oi":
        required = {"volatility_bucket", "liquidity_tier", "funding_bucket", "oi_bucket"}
        if not required.issubset(set(pool.columns)):
            return empty, "true vol/liquidity/funding/OI match features absent; control not built", "missing_match_features"
        # Conservative exact bucket matching if future ledgers contain these features.
        b = pool.copy()
        for col in required:
            vals = set(cand[col].dropna().astype(str))
            b = b[b[col].astype(str).isin(vals)]
        return _drop_self_rows(b, cand), "nearest-neighbor exact bucket match on vol/liquidity/funding/OI", "true_feature_bucket_match"
    if control_type == "generic_momentum":
        if fam in {"B1", "C2"}:
            b = indexes["a2a3"]
        elif fam == "A3":
            b = indexes["by_family"].get("A2_redesign_only", empty)
        elif fam == "A2_redesign_only":
            b = indexes["by_family"].get("A3", empty)
        else:
            b = indexes["a2a3"]
        return _drop_self_rows(b, cand), "generic liquid momentum event rows from A2/A3 pools", "generic_momentum_event_pool"
    if control_type == "A2_A3_overlap":
        b = indexes["a2a3"]
        if symbols:
            b = b[b["symbol"].astype(str).isin(symbols)]
        if not b.empty and months:
            same_month = b[b["decision_month"].astype(str).isin(months)]
            if not same_month.empty:
                b = same_month
        return _drop_self_rows(b, cand), "A2/A3 event rows sharing symbol and preferably month", "a2_a3_symbol_month_overlap"
    return empty, "unknown control type", "unknown_control_type"

def _deterministic_select(df: pd.DataFrame, n: int, salt: str) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.iloc[0:0].copy()
    tmp = df.reset_index(drop=True)
    if n >= len(tmp):
        # If there are fewer real controls than requested, do not duplicate. Coverage is reported explicitly.
        return tmp.copy()
    start = int(stable_hash(salt, n=12), 16) % len(tmp)
    idx = (start + np.arange(n)) % len(tmp)
    return tmp.iloc[idx].copy()


def build_real_controls(
    pool: pd.DataFrame,
    candidate_keys: Iterable[str] | None = None,
    control_types: Iterable[str] = CONTROL_TYPES,
    nulls_per_event: int = 3,
    seed: int = 20260629,
    max_control_rows_per_candidate: int = 20000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a source-row control ledger and normalized summary.

    Returns: candidate_summary, control_event_ledger, control_summary.
    """
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    validate_no_protected_rows(pool, ["decision_ts", "entry_ts", "exit_ts"])
    pool = pool.sort_values("source_row_id", kind="mergesort").reset_index(drop=True)
    indexes = {
        "by_symbol": {str(k): g for k, g in pool.groupby("symbol", sort=False)},
        "by_regime": {str(k): g for k, g in pool.groupby("parent_regime", sort=False)},
        "by_family": {str(k): g for k, g in pool.groupby("family", sort=False)},
    }
    indexes["a2a3"] = pool[pool["family"].isin(["A2_redesign_only", "A3"])]
    keys = list(candidate_keys) if candidate_keys is not None else sorted(pool["candidate_key"].dropna().astype(str).unique())
    cand_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for key in keys:
        cand = _candidate_subset(pool, key)
        if cand.empty:
            continue
        family = str(cand["family"].iloc[0])
        cand_sum = metric_summary(cand, key, family)
        cand_rows.append(cand_sum)
        candidate_events = int(cand_sum["events"])
        candidate_net = float(cand_sum["net_R"]) if candidate_events else float("nan")
        target_controls = min(max_control_rows_per_candidate, candidate_events * max(1, int(nulls_per_event)))
        for ctype in control_types:
            eligible, basis, status = _eligible_pool(ctype, cand, pool, indexes)
            selected = _deterministic_select(eligible, target_controls, f"{seed}:{key}:{ctype}")
            selected_net = pd.to_numeric(selected.get("source_net_R", pd.Series(dtype=float)), errors="coerce")
            raw_control = float(selected_net.sum()) if len(selected) else float("nan")
            control_events = int(selected_net.notna().sum())
            norm = normalize_control_net(candidate_events, raw_control, control_events)
            control_source_set_hash = stable_hash(*selected["source_row_id"].astype(str).tolist()) if len(selected) else ""
            for rank, (_, r) in enumerate(selected.iterrows(), 1):
                control_rows.append({
                    "candidate_key": key,
                    "candidate_id": cand_sum["candidate_id"],
                    "family": family,
                    "control_type": ctype,
                    "candidate_event_count": candidate_events,
                    "candidate_net_R": candidate_net,
                    "control_event_id": r.get("event_id"),
                    "control_symbol": r.get("symbol"),
                    "control_decision_ts": r.get("decision_ts"),
                    "control_entry_ts": r.get("entry_ts"),
                    "control_exit_ts": r.get("exit_ts"),
                    "control_source_candidate_key": r.get("candidate_key"),
                    "control_source_family": r.get("family"),
                    "control_source_row_id": r.get("source_row_id"),
                    "control_window_id": r.get("source_window_id"),
                    "source_window_id": r.get("source_window_id"),
                    "matched_candidate_id": cand_sum["candidate_id"],
                    "matching_basis": basis,
                    "feature_source_ts": r.get("feature_source_ts"),
                    "control_net_R": r.get("source_net_R"),
                    "match_basis": basis,
                    "match_status": status,
                    "selection_rank": rank,
                    "source_path": r.get("source_path"),
                })
            summary_rows.append({
                "candidate_key": key,
                "candidate_id": cand_sum["candidate_id"],
                "family": family,
                "control_type": ctype,
                "candidate_event_count": candidate_events,
                "control_event_count": control_events,
                "target_control_event_count": target_controls,
                "control_coverage_ratio": control_events / target_controls if target_controls else float("nan"),
                "candidate_net_R": candidate_net,
                "raw_control_net_R": raw_control,
                "normalized_control_net_R": norm,
                "control_uplift_R": candidate_net - norm if np.isfinite(norm) and np.isfinite(candidate_net) else float("nan"),
                "beats_control": bool(np.isfinite(norm) and np.isfinite(candidate_net) and candidate_net > norm),
                "controls_normalized_to_candidate_count": True,
                "control_source_set_hash": control_source_set_hash,
                "match_basis": basis,
                "match_status": status,
                "all_control_rows_have_source_ids": bool(len(selected) and selected["source_row_id"].notna().all() and selected["source_window_id"].notna().all()),
            })
    return pd.DataFrame(cand_rows), pd.DataFrame(control_rows), pd.DataFrame(summary_rows)


def apply_real_control_labels(candidate_summary: pd.DataFrame, control_summary: pd.DataFrame) -> pd.DataFrame:
    if candidate_summary.empty:
        return candidate_summary.copy()
    out = candidate_summary.copy()
    if control_summary.empty:
        out["real_controls_available"] = False
        out["beats_all_real_controls"] = False
        out["min_real_control_uplift_R"] = np.nan
        out["real_control_label"] = "not_fairly_tested_missing_real_controls"
        return out
    grouped = control_summary.groupby("candidate_key")
    pass_map = grouped["beats_control"].all().to_dict()
    min_uplift = grouped["control_uplift_R"].min().to_dict()
    coverage = grouped["control_coverage_ratio"].min().to_dict()
    built_types = grouped["control_type"].nunique().to_dict()
    out["real_controls_available"] = out["candidate_key"].map(lambda x: x in pass_map)
    out["beats_all_real_controls"] = out["candidate_key"].map(lambda x: bool(pass_map.get(x, False)))
    out["min_real_control_uplift_R"] = out["candidate_key"].map(lambda x: min_uplift.get(x, np.nan))
    out["min_real_control_coverage_ratio"] = out["candidate_key"].map(lambda x: coverage.get(x, np.nan))
    out["real_control_type_count"] = out["candidate_key"].map(lambda x: built_types.get(x, 0))
    labels = []
    for _, r in out.iterrows():
        fam = str(r.get("family"))
        net = float(r.get("net_R", np.nan)) if pd.notna(r.get("net_R", np.nan)) else np.nan
        pfx = float(r.get("PF", np.nan)) if pd.notna(r.get("PF", np.nan)) else np.nan
        proxy_cap = bool(r.get("mark_proxy_used_any", False) or r.get("funding_proxy_used_any", False) or not r.get("funding_exact_all", False))
        controls_ok = bool(r.get("beats_all_real_controls", False))
        coverage_ok = float(r.get("min_real_control_coverage_ratio", 0) or 0) >= 0.5
        if fam in {"B1", "C2"}:
            labels.append("seed_limited_support_only_real_controls_built" if controls_ok else "seed_limited_support_only_controls_not_passed")
        elif not np.isfinite(net) or net <= 0 or not np.isfinite(pfx) or pfx <= 1:
            labels.append("reject_current_translation_only")
        elif controls_ok and coverage_ok and not proxy_cap:
            labels.append("event_level_candidate_with_real_controls")
        elif controls_ok and coverage_ok and proxy_cap:
            labels.append("research_candidate_capped_by_mark_or_funding_proxy")
        elif controls_ok:
            labels.append("research_candidate_control_coverage_limited")
        else:
            labels.append("path_edge_or_positive_isolation_only_controls_not_passed")
    out["real_control_label"] = labels
    return out
