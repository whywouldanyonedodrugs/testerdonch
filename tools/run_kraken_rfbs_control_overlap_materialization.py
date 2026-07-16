#!/usr/bin/env python3
"""RFBS frozen-ledger overlap/control closure and targeted adjudication."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools import run_kraken_liquid_failed_breakout_short_screen as lfbs
from tools import run_kraken_riskoff_failed_bounce_short_screen as rfbs


SOURCE_ROOT = Path("results/rebaseline/phase_kraken_riskoff_failed_bounce_short_screen_20260714_v1")
TARGET_IDS = ("rfbs_v1_001", "rfbs_v1_004", "rfbs_v1_007", "rfbs_v1_010")
FORMAL_CANDIDATE = "rfbs_v1_004"
CONTEXTUAL = {"same_symbol_same_regime_random_short", "completed_failure_outside_riskoff_parent"}
STRUCTURAL = {"countertrend_rally_without_completed_failure", "non_rally_red_candle_short", "generic_20d_failed_breakout_short"}
CONTRACT_VERSION = "rfbs_control_overlap_materialization_v1_20260714"


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def canonical_float(value: Any) -> str:
    if pd.isna(value):
        return "null"
    return format(float(value), ".17g")


def canonical_ts(value: Any) -> str:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    return ts.isoformat().replace("+00:00", "Z")


def signal_address_vector(row: Mapping[str, Any]) -> dict[str, str]:
    return {
        "symbol": str(row["symbol"]),
        "decision_ts": canonical_ts(row["decision_ts"]),
        "entry_ts": canonical_ts(row["entry_ts"]),
        "entry_price": canonical_float(row["entry_price"]),
        "initial_stop": canonical_float(row["initial_stop"]),
        "risk_denominator": canonical_float(row["risk_denominator"]),
    }


def signal_address_hash(row: Mapping[str, Any]) -> str:
    return stable_hash(signal_address_vector(row))


def trade_address_hash(row: Mapping[str, Any]) -> str:
    vector = signal_address_vector(row)
    vector.update({"exit_policy": str(row["exit_policy"]), "maximum_exit_ts": canonical_ts(row["maximum_exit_ts"])})
    return stable_hash(vector)


def pairwise_overlap(frame: pd.DataFrame, group_col: str, address_col: str) -> pd.DataFrame:
    groups = {str(key): set(group[address_col]) for key, group in frame.groupby(group_col)}
    rows = []
    keys = sorted(groups)
    for index, left in enumerate(keys):
        for right in keys[index+1:]:
            intersection = groups[left] & groups[right]; union = groups[left] | groups[right]
            rows.append({
                "left_id": left, "right_id": right, "left_count": len(groups[left]), "right_count": len(groups[right]),
                "shared_count": len(intersection), "jaccard": len(intersection)/len(union) if union else np.nan,
                "left_is_subset": groups[left].issubset(groups[right]), "right_is_subset": groups[right].issubset(groups[left]),
            })
    return pd.DataFrame(rows)


def nesting_audit(keys: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    policy = manifest.drop_duplicates("selected_key_policy_hash").set_index(["rally_profile", "confirmation_bars", "parent_policy"])
    sets = {key: set(group.signal_address_hash) for key, group in keys.groupby("selected_key_policy_hash")}
    rows = []
    for rally in manifest.rally_profile.unique():
        for window in (1, 3):
            strict = policy.loc[(rally, window, "strict_both_down_stress")].selected_key_policy_hash
            broader = policy.loc[(rally, window, "broader_fragile_countertrend_stress")].selected_key_policy_hash
            missing = sets[strict] - sets[broader]
            sequencing_details = []
            unexplained = []
            broader_rows = keys[keys.selected_key_policy_hash.eq(broader)]
            for address in sorted(missing):
                strict_row = keys[(keys.selected_key_policy_hash == strict) & keys.signal_address_hash.eq(address)].iloc[0]
                entry_ts = pd.Timestamp(strict_row.entry_ts)
                blockers = broader_rows[
                    broader_rows.symbol.eq(strict_row.symbol)
                    & (broader_rows.entry_ts < entry_ts)
                    & (broader_rows.entry_ts+pd.Timedelta(days=7) > entry_ts)
                ]
                if blockers.empty:
                    unexplained.append(address)
                else:
                    blocker = blockers.sort_values("entry_ts").iloc[-1]
                    sequencing_details.append(f"{address}:{blocker.signal_address_hash}:{canonical_ts(blocker.entry_ts)}")
            rows.append({
                "comparison": "strict_vs_broader", "rally_profile": rally, "confirmation_bars": window,
                "left_policy_hash": strict, "right_policy_hash": broader, "expected_relation": "strict_subset_broader",
                "left_only_count": len(missing), "sequencing_explained_count": len(sequencing_details), "unexplained_count": len(unexplained),
                "relation_pass": not unexplained, "sequencing_blocker_details": "|".join(sequencing_details),
                "explanation": "Broader admits both-down plus mixed-at-least-one-down. A strict-only selected row is explained only when an earlier broader-only entry on the same symbol activates the frozen seven-day per-policy non-overlap block.",
            })
        for parent in manifest.parent_policy.unique():
            one = policy.loc[(rally, 1, parent)].selected_key_policy_hash
            three = policy.loc[(rally, 3, parent)].selected_key_policy_hash
            missing = sets[one] - sets[three]
            rows.append({
                "comparison": "one_bar_vs_three_bar", "rally_profile": rally, "parent_policy": parent,
                "left_policy_hash": one, "right_policy_hash": three, "expected_relation": "reported_not_assumed",
                "left_only_count": len(missing), "sequencing_explained_count": len(missing), "unexplained_count": 0, "relation_pass": True, "sequencing_blocker_details": "state_machine_window_difference",
                "explanation": "Nesting is not required: the pending-sequence state machine expires at different bars, and higher-high resets plus per-policy non-overlap can change subsequent sequence starts.",
            })
    return pd.DataFrame(rows)


def repaired_bias(events: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition in sorted(events.definition_id.unique()):
        candidate = events[events.definition_id.eq(definition)]
        for control_class in rfbs.CONTROL_CLASSES:
            group = controls[(controls.definition_id == definition) & controls.control_class.eq(control_class)]
            unique = group.sort_values(["control_economic_address_hash", "candidate_key"]).drop_duplicates("control_economic_address_hash")
            matched_keys = set(unique.candidate_key)
            matched = candidate[candidate.candidate_key.isin(matched_keys)]
            unmatched = candidate[~candidate.candidate_key.isin(matched_keys)]
            for mode in ("base", "conservative", "severe"):
                rows.append({
                    "definition_id": definition, "control_class": control_class, "cost_mode": mode,
                    "matched_count": len(matched), "unmatched_count": len(unmatched), "full_count": len(candidate),
                    "unique_control_address_count": unique.control_economic_address_hash.nunique(),
                    "matched_candidate_mean_R": matched[f"net_{mode}_R"].mean(),
                    "unmatched_only_candidate_mean_R": unmatched[f"net_{mode}_R"].mean(),
                    "full_candidate_mean_R": candidate[f"net_{mode}_R"].mean(),
                    "control_mean_R": unique[f"net_{mode}_R"].mean(),
                    "matched_minus_unmatched_mean_R": matched[f"net_{mode}_R"].mean()-unmatched[f"net_{mode}_R"].mean() if len(unmatched) else np.nan,
                    "candidate_minus_control_mean_R": matched[f"net_{mode}_R"].mean()-unique[f"net_{mode}_R"].mean(),
                    "class_coverage": matched.candidate_key.nunique()/max(1, candidate.candidate_key.nunique()),
                    "adequate_control": unique.control_economic_address_hash.nunique() >= 15 and matched.candidate_key.nunique()/max(1, candidate.candidate_key.nunique()) >= .70,
                })
    return pd.DataFrame(rows)


def add_symmetric_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["stop_distance"] = out.initial_stop-out.entry_price
    out["risk_to_daily_atr"] = out.risk_denominator/out.daily_atr
    out["raw_short_price_return"] = (out.entry_price-out.exit_price)/out.entry_price
    out["atr_normalized_short_pnl"] = (out.entry_price-out.exit_price)/out.daily_atr
    out["diagnostic_rule"] = "raw_price_return_and_completed_daily_atr_normalization_applied_symmetrically_no_floor_no_winsorization"
    return out


def control_risk_outputs(controls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = add_symmetric_diagnostics(controls[controls.definition_id.isin(TARGET_IDS)])
    quantile_rows = []
    for (definition, control_class), group in work.groupby(["definition_id", "control_class"]):
        for field in ("risk_denominator", "stop_distance", "risk_to_daily_atr", "net_conservative_R", "raw_short_price_return", "atr_normalized_short_pnl"):
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            quantile_rows.append({
                "definition_id": definition, "control_class": control_class, "field": field, "rows": len(values),
                "minimum": values.min(), "q01": values.quantile(.01), "q05": values.quantile(.05), "median": values.median(),
                "q95": values.quantile(.95), "q99": values.quantile(.99), "maximum": values.max(),
            })
    extremes = []
    for (definition, control_class), group in work.groupby(["definition_id", "control_class"]):
        ordered = group.sort_values("net_conservative_R")
        for tail, subset in (("negative", ordered.head(10)), ("positive", ordered.tail(10).sort_values("net_conservative_R", ascending=False))):
            for rank, (_, row) in enumerate(subset.iterrows(), 1):
                values = row.to_dict(); values.update({"tail": tail, "tail_rank": rank}); extremes.append(values)
    focus = work[
        ((work.definition_id.isin(["rfbs_v1_007", "rfbs_v1_010"])) & work.control_class.eq("same_symbol_same_regime_random_short"))
        | (work.definition_id.eq("rfbs_v1_004") & work.control_class.eq("countertrend_rally_without_completed_failure"))
    ].copy()
    return pd.DataFrame(quantile_rows), pd.DataFrame(extremes), focus


def build_context(root: Path) -> SimpleNamespace:
    return SimpleNamespace(run_root=root, start=rfbs.START, end=rfbs.END, args=SimpleNamespace(kraken_data_root=str(runner.DEFAULT_KRAKEN_DATA_ROOT), smoke=False))


def replay_and_path_audits(events: pd.DataFrame, panel: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = runner.local_data_paths_from_root(str(runner.DEFAULT_KRAKEN_DATA_ROOT)); ctx = build_context(root)
    parity = []; integrity = []; horizon_rows = []; adjudication = []
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for symbol in sorted(events.symbol.unique()):
        bars = runner.load_symbol_bars(paths, symbol, rfbs.START-pd.Timedelta(days=120), rfbs.END)
        frame, _, _ = rfbs.feature_frames(bars[["ts", "open", "high", "low", "close", "volume"]])
        cache[symbol] = (bars, frame)
    for row in events.to_dict("records"):
        bars, frame = cache[row["symbol"]]
        replay, excluded = rfbs.execute_event(row, row["exit_policy"], bars, frame)
        mismatches = []
        if excluded:
            mismatches.append(f"unexpected_replay_exclusion:{excluded['reason']}")
        else:
            for field in ("exit_ts", "exit_reason"):
                left = canonical_ts(row[field]) if field.endswith("ts") else str(row[field]); right = canonical_ts(replay[field]) if field.endswith("ts") else str(replay[field])
                if left != right: mismatches.append(field)
            for field in ("exit_price", "gross_R", "mae_R", "mfe_R"):
                if not np.isclose(float(row[field]), float(replay[field]), rtol=1e-12, atol=1e-12): mismatches.append(field)
        parity.append({"definition_id": row["definition_id"], "event_id": row["event_id"], "mismatch_count": len(mismatches), "mismatch_fields": "|".join(mismatches), "parity_pass": not mismatches})

        entry_ts = pd.Timestamp(row["entry_ts"]); exit_ts = pd.Timestamp(row["exit_ts"])
        path = bars[(bars.ts >= entry_ts) & (bars.ts <= exit_ts)].copy()
        valid_ohlc = path[["open", "high", "low", "close"]].gt(0).all(axis=1) & path.high.ge(path[["open", "close"]].max(axis=1)) & path.low.le(path[["open", "close"]].min(axis=1))
        stop_hits = path[(path.open >= row["initial_stop"]) | (path.high >= row["initial_stop"])]
        first_stop_ts = pd.NaT if stop_hits.empty else stop_hits.iloc[0].ts
        entry_bar = bars[bars.ts.eq(entry_ts)].head(1); exit_bar = bars[bars.ts.eq(exit_ts)].head(1)
        entry_mark = entry_bar.mark_close.iloc[0] if len(entry_bar) and "mark_close" in entry_bar else np.nan
        exit_mark = exit_bar.mark_close.iloc[0] if len(exit_bar) and "mark_close" in exit_bar else np.nan
        max_range_atr = ((path.high-path.low)/float(row["daily_atr"])).max() if len(path) and row["daily_atr"] > 0 else np.nan
        integrity.append({
            "definition_id": row["definition_id"], "event_id": row["event_id"], "symbol": row["symbol"],
            "ohlcv_invalid_rows": int((~valid_ohlc).sum()), "mark_entry_available": pd.notna(entry_mark), "mark_exit_available": pd.notna(exit_mark),
            "entry_trade_mark_gap_bps": (float(row["entry_price"])/entry_mark-1)*1e4 if pd.notna(entry_mark) and entry_mark else np.nan,
            "exit_trade_mark_gap_bps": (float(row["exit_price"])/exit_mark-1)*1e4 if pd.notna(exit_mark) and exit_mark else np.nan,
            "first_stop_breach_ts": first_stop_ts, "stop_before_exit_consistent": (pd.isna(first_stop_ts) or first_stop_ts >= exit_ts) if "time_exit" in row["exit_reason"] else first_stop_ts == exit_ts,
            "lifecycle_pit_eligible": rfbs.pit_allowed(ctx, panel, pd.Timestamp(row["decision_ts"]), row["symbol"]),
            "max_5m_range_over_daily_atr": max_range_atr, "pathological_wick_diagnostic": bool(pd.notna(max_range_atr) and max_range_atr > 1.5),
            "pathological_wick_rule": "diagnostic_only_5m_high_low_range_above_1.5_completed_daily_atr_no_outcome_filter",
        })
        for hours in (24, 48, 72):
            horizon = entry_ts+pd.Timedelta(hours=hours); used = bars[(bars.ts >= entry_ts) & (bars.ts <= horizon)]
            stop_rows = used[(used.open >= row["initial_stop"]) | (used.high >= row["initial_stop"])]
            if not stop_rows.empty:
                stop_bar = stop_rows.iloc[0]; price = float(stop_bar.open) if stop_bar.open >= row["initial_stop"] else float(row["initial_stop"]); path_exit = stop_bar.ts; status = "stopped_before_horizon"
            else:
                horizon_bar = bars[bars.ts >= horizon].head(1)
                if horizon_bar.empty: continue
                price = float(horizon_bar.iloc[0].open); path_exit = horizon_bar.iloc[0].ts; status = "marked_at_horizon_next_5m_open"
            used = bars[(bars.ts >= entry_ts) & (bars.ts <= path_exit)]
            horizon_rows.append({
                "definition_id": row["definition_id"], "event_id": row["event_id"], "evaluation_period": row["evaluation_period"], "funding_partition": "exact_or_mixed" if row["exact_funding_boundaries"] > 0 else "imputed_or_zero",
                "horizon_hours": hours, "path_status": status, "path_exit_ts": path_exit, "path_exit_price": price,
                "gross_R": (float(row["entry_price"])-price)/float(row["risk_denominator"]),
                "mae_R": min(0.0, (float(row["entry_price"])-float(used.high.max()))/float(row["risk_denominator"])),
                "mfe_R": max(0.0, (float(row["entry_price"])-float(used.low.min()))/float(row["risk_denominator"])),
            })
        if row["definition_id"] == FORMAL_CANDIDATE:
            tags = []
            if row["evaluation_period"] == "2023": tags.append("all_2023")
            if row["exact_funding_boundaries"] > 0: tags.append("exact_or_mixed_2025h2")
            adjudication.append({**row, "adjudication_tags": "|".join(tags)})
    adj = pd.DataFrame(adjudication)
    top = adj.nlargest(3, "net_conservative_R").event_id if len(adj) else []
    if len(adj): adj.loc[adj.event_id.isin(top), "adjudication_tags"] = adj.loc[adj.event_id.isin(top), "adjudication_tags"].map(lambda x: "|".join(filter(None, [x, "top_three_contributor"])))
    return pd.DataFrame(parity), pd.DataFrame(integrity), pd.DataFrame(horizon_rows), adj


def arithmetic_audit(frame: pd.DataFrame, key_col: str) -> pd.DataFrame:
    rows = []
    funding = {"base": "funding_central_R", "conservative": "funding_conservative_R", "severe": "funding_severe_R"}
    for mode in ("base", "conservative", "severe"):
        expected = frame.gross_R+frame[f"fee_{mode}_R"]+frame[f"slippage_{mode}_R"]+frame[funding[mode]]
        error = frame[f"net_{mode}_R"]-expected
        rows.append({"ledger": key_col, "cost_mode": mode, "rows": len(frame), "maximum_absolute_error": error.abs().max(), "mismatch_count": int((error.abs() > 1e-12).sum()), "status": "pass" if (error.abs() <= 1e-12).all() else "fail"})
    return pd.DataFrame(rows)


def top_winners_losers(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for definition, group in events.groupby("definition_id"):
        ordered = group.sort_values("net_conservative_R")
        for tail, subset in (("loser", ordered.head(5)), ("winner", ordered.tail(5).sort_values("net_conservative_R", ascending=False))):
            for rank, (_, row) in enumerate(subset.iterrows(), 1): rows.append({**row.to_dict(), "tail": tail, "tail_rank": rank})
    return pd.DataFrame(rows)


def original_vs_repaired_gate(
    source_decisions: pd.DataFrame, source_summary: pd.DataFrame, source_concentration: pd.DataFrame,
    source_period: pd.DataFrame, repaired: pd.DataFrame, manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for definition_id in TARGET_IDS:
        definition = manifest[manifest.definition_id.eq(definition_id)].iloc[0]
        stats = source_summary[source_summary.definition_id.eq(definition_id)].set_index("cost_mode")
        forensic = source_concentration[(source_concentration.definition_id == definition_id) & source_concentration.cost_mode.eq("conservative")].iloc[0]
        positive = repaired[(repaired.definition_id == definition_id) & repaired.cost_mode.eq("conservative") & repaired.adequate_control & repaired.candidate_minus_control_mean_R.gt(0)]
        classes = set(positive.control_class)
        stable_periods = int(source_period[(source_period.definition_id == definition_id) & source_period.cost_mode.eq("conservative")].mean_R.gt(0).sum())
        robust = forensic.mean_after_top3 > 0 and forensic.worst_leave_one_symbol_mean_R > 0 and forensic.worst_leave_one_month_mean_R > 0
        base, conservative = stats.loc["base"], stats.loc["conservative"]
        if base.events >= 30 and base.symbols >= 10 and base.mean_R > 0 and conservative.mean_R > 0 and robust and stable_periods >= 3 and len(classes) >= 2 and classes & CONTEXTUAL and classes & STRUCTURAL:
            repaired_decision = "materialization_candidate"
        elif definition.parent_policy == "strict_both_down_stress" and base.events >= 15 and base.mean_R > 0 and conservative.mean_R > 0 and robust and classes & CONTEXTUAL:
            repaired_decision = "fragile_context_sleeve"
        else:
            repaired_decision = "current_translation_weak"
        original = source_decisions[source_decisions.definition_id.eq(definition_id)].iloc[0].decision
        rows.append({
            "definition_id": definition_id, "original_frozen_gate_decision": original, "repaired_reporting_gate_decision": repaired_decision,
            "decision_changed": original != repaired_decision, "mechanically_defective_control_calculation_found": False,
            "adequate_positive_control_classes": "|".join(sorted(classes)), "adequate_positive_control_class_count": len(classes),
            "note": "Matched/unmatched reporting was corrected; raw control outcomes and candidate-minus-control uplift were unchanged.",
        })
    return pd.DataFrame(rows)


def build_bundle(root: Path, files: Iterable[str]) -> Path:
    temp = root/".compact_review_bundle.tmp"; temp.mkdir(); inventory = []
    for relative in files:
        source = root/relative; target = temp/relative.replace("/", "__"); shutil.copy2(source, target)
        inventory.append({"bundle_file": target.name, "source_relative_path": relative, "bytes": source.stat().st_size, "sha256": file_hash(source)})
    write_csv(temp/"bundle_manifest.csv", inventory); os.replace(temp, root/"compact_review_bundle"); return root/"compact_review_bundle"


def run(root: Path, source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    if root.exists(): raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True); started = time.monotonic()
    manifest = pd.read_csv(source_root/"manifest/riskoff_failed_bounce_definitions.csv")
    keys = pd.read_csv(source_root/"keys/candidate_key_manifest.csv")
    events = pd.read_csv(source_root/"materialized/event_ledger.csv")
    controls = pd.read_csv(source_root/"controls/control_event_ledger.csv")
    panel = pd.read_csv(source_root/"manifest/pit_panel.csv")
    for frame in (keys, events, controls):
        for column in [c for c in frame.columns if c.endswith("_ts")]: frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    keys["signal_address_hash"] = [signal_address_hash(row) for row in keys.to_dict("records")]
    events["signal_address_hash"] = [signal_address_hash(row) for row in events.to_dict("records")]
    events["trade_economic_address_hash_v2"] = [trade_address_hash(row) for row in events.to_dict("records")]
    identity_manifest = events[["definition_id", "selected_key_policy_hash", "candidate_key", "event_id", "signal_address_hash", "trade_economic_address_hash_v2", "symbol", "decision_ts", "entry_ts", "entry_price", "initial_stop", "risk_denominator", "exit_policy", "maximum_exit_ts"]]
    write_csv(root/"identity/canonical_address_manifest.csv", identity_manifest)
    policy_overlap = pairwise_overlap(keys, "selected_key_policy_hash", "signal_address_hash"); write_csv(root/"identity/selected_policy_pairwise_overlap.csv", policy_overlap)
    definition_overlap = pairwise_overlap(events, "definition_id", "trade_economic_address_hash_v2"); write_csv(root/"identity/definition_pairwise_overlap.csv", definition_overlap)
    nesting = nesting_audit(keys, manifest); write_csv(root/"identity/nesting_audit.csv", nesting)

    bias = repaired_bias(events, controls); write_csv(root/"controls/matched_unmatched_bias_repaired.csv", bias)
    target_events = add_symmetric_diagnostics(events[events.definition_id.isin(TARGET_IDS)]); target_controls = add_symmetric_diagnostics(controls[controls.definition_id.isin(TARGET_IDS)])
    write_csv(root/"materialized/candidate_event_ledger.csv", target_events); write_csv(root/"controls/materialized_control_ledger.csv", target_controls)
    quantiles, extremes, focus = control_risk_outputs(controls)
    write_csv(root/"controls/control_risk_quantiles.csv", quantiles); write_csv(root/"controls/control_extreme_rows.csv", extremes); write_csv(root/"controls/control_risk_focus_audit.csv", focus)

    parity, integrity, horizons, adjudication = replay_and_path_audits(target_events, panel, root)
    write_csv(root/"audit/candidate_replay_parity.csv", parity); write_csv(root/"audit/event_path_integrity.csv", integrity)
    write_csv(root/"forensics/rfbs_004_horizon_paths.csv", horizons[horizons.definition_id.eq(FORMAL_CANDIDATE)])
    write_csv(root/"forensics/rfbs_004_special_event_adjudication.csv", adjudication[adjudication.adjudication_tags.ne("")])
    arithmetic = pd.concat([arithmetic_audit(target_events, "candidate"), arithmetic_audit(target_controls, "control")], ignore_index=True)
    write_csv(root/"audit/cost_funding_arithmetic.csv", arithmetic); write_csv(root/"forensics/top_five_winners_losers.csv", top_winners_losers(target_events))

    source_summary = pd.read_csv(source_root/"economics/definition_summary.csv")
    source_concentration = pd.read_csv(source_root/"forensics/concentration_and_removal.csv")
    source_period = pd.read_csv(source_root/"economics/period_summary.csv")
    source_decisions = pd.read_csv(source_root/"decision/candidate_decisions.csv")
    neighborhood = source_summary[source_summary.definition_id.isin(TARGET_IDS)].merge(manifest, on="definition_id").merge(
        source_concentration[source_concentration.cost_mode.eq("conservative")][["definition_id", "mean_after_top1", "mean_after_top3", "worst_leave_one_symbol_mean_R", "worst_leave_one_month_mean_R"]], on="definition_id", how="left"
    )
    write_csv(root/"forensics/frozen_neighborhood_robustness.csv", neighborhood)
    gate_comparison = original_vs_repaired_gate(source_decisions, source_summary, source_concentration, source_period, bias, manifest)
    write_csv(root/"decision/original_vs_repaired_gate.csv", gate_comparison)

    parity_pass = bool(len(parity) and parity.parity_pass.all())
    identity_pass = bool(keys.signal_address_hash.notna().all() and events.trade_economic_address_hash_v2.notna().all() and nesting[nesting.comparison.eq("strict_vs_broader")].relation_pass.all())
    bias_pass = bool(len(bias) == 24*len(rfbs.CONTROL_CLASSES)*3 and (bias.matched_count+bias.unmatched_count == bias.full_count).all())
    mechanics_pass = bool(parity_pass and not arithmetic.mismatch_count.sum() and not integrity.ohlcv_invalid_rows.sum() and integrity.stop_before_exit_consistent.all() and integrity.lifecycle_pit_eligible.all())
    formal = source_summary[(source_summary.definition_id == FORMAL_CANDIDATE)].set_index("cost_mode")
    formal_conc = source_concentration[(source_concentration.definition_id == FORMAL_CANDIDATE) & source_concentration.cost_mode.eq("conservative")].iloc[0]
    formal_bias = bias[(bias.definition_id == FORMAL_CANDIDATE) & bias.cost_mode.eq("conservative") & bias.adequate_control & bias.candidate_minus_control_mean_R.gt(0)]
    supporting_classes = set(formal_bias.control_class); control_support = bool(supporting_classes & CONTEXTUAL and supporting_classes & STRUCTURAL and len(supporting_classes) >= 2)
    period_2023 = source_period[(source_period.definition_id == FORMAL_CANDIDATE) & source_period.period.eq("2023") & source_period.cost_mode.eq("conservative")]
    top_event_pass = bool(formal_conc.mean_after_top3 > 0)
    raw_controls_interpretable = bool((target_controls.risk_denominator > 0).all() and (target_controls.risk_to_daily_atr <= 1.5+1e-12).all() and not arithmetic[arithmetic.ledger.eq("control")].mismatch_count.sum())
    weakness_explained_without_filter = bool(len(period_2023) and period_2023.iloc[0].mean_R >= 0)
    if not (identity_pass and bias_pass and mechanics_pass): final = "focused_mechanical_repair_required"
    elif formal.loc["base"].mean_R <= 0 and formal.loc["conservative"].mean_R <= 0: final = "current_translation_weak"
    elif top_event_pass and control_support and raw_controls_interpretable and weakness_explained_without_filter: final = "train_only_stability_review_candidate"
    else: final = "fragile_context_sleeve"

    decisions = []
    for definition in manifest.itertuples(index=False):
        stats = source_summary[source_summary.definition_id.eq(definition.definition_id)].set_index("cost_mode")
        conc_rows = source_concentration[(source_concentration.definition_id == definition.definition_id) & source_concentration.cost_mode.eq("conservative")]
        positive_controls = bias[(bias.definition_id == definition.definition_id) & bias.cost_mode.eq("conservative") & bias.adequate_control & bias.candidate_minus_control_mean_R.gt(0)]
        state = final if definition.definition_id == FORMAL_CANDIDATE else "frozen_neighborhood_comparator_not_promotable" if definition.definition_id in TARGET_IDS else "screen_definition_not_materialized"
        decisions.append({
            "candidate_id": definition.definition_id, "candidate_definition_id": definition.definition_id, "definition_id": definition.definition_id,
            "hypothesis_id": "riskoff_failed_bounce_short", "family_engine_id": "kraken_rfbs_v1", "parameter_vector_hash": definition.parameter_vector_hash,
            "selected_key_policy_hash": definition.selected_key_policy_hash, "candidate_library_state": state, "candidate_decision": state,
            "evidence_level": "level_3_train_only_materialized_controls_forensics_capped" if definition.definition_id in TARGET_IDS else "level_2_train_only_aggregate_screen_capped",
            "evidence_level_contract": "train_only_not_validation_not_holdout_not_live", "clean_evidence_allowed": False, "mechanics_cap_active": True,
            "evidence_cap_reason": "shared_funding_imputation_ohlcv_stop_no_depth_and_train_selection_caps", "family_rejected": False,
            "train_only": True, "validation_run": False, "holdout_touched": False, "live_ready": False, "can_support_strategy_claim": False,
            "event_rows": int(stats.loc["base"].events), "symbols": int(stats.loc["base"].symbols), "months": int(stats.loc["base"].months),
            "base_mean_R": stats.loc["base"].mean_R, "conservative_mean_R": stats.loc["conservative"].mean_R, "severe_mean_R": stats.loc["severe"].mean_R,
            "base_profit_factor": stats.loc["base"].profit_factor, "conservative_profit_factor": stats.loc["conservative"].profit_factor, "severe_profit_factor": stats.loc["severe"].profit_factor,
            "mean_after_top1": conc_rows.iloc[0].mean_after_top1 if len(conc_rows) else np.nan, "mean_after_top3": conc_rows.iloc[0].mean_after_top3 if len(conc_rows) else np.nan,
            "adequate_positive_control_classes": "|".join(sorted(set(positive_controls.control_class))), "adequate_positive_control_class_count": positive_controls.control_class.nunique(),
            "has_contextual_control": bool(set(positive_controls.control_class) & CONTEXTUAL), "has_structural_control": bool(set(positive_controls.control_class) & STRUCTURAL),
            "identity_repair_pass": identity_pass, "bias_repair_pass": bias_pass, "parity_pass": parity_pass,
            "source_run_root": str(source_root), "closure_run_root": str(root), "contract_version": CONTRACT_VERSION,
        })
    library = pd.DataFrame(decisions); write_csv(root/"candidate_library/rfbs_candidate_library_update.csv", library)
    write_csv(root/"decision/candidate_adjudication.csv", library[library.candidate_definition_id.isin(TARGET_IDS)])

    gate = {
        "identity_repair_pass": identity_pass, "bias_repair_pass": bias_pass, "candidate_replay_parity_pass": parity_pass,
        "mechanics_pass": mechanics_pass, "top_event_robustness_pass": top_event_pass, "raw_controls_interpretable": raw_controls_interpretable,
        "adequate_contextual_and_structural_control_support": control_support, "weak_2023_explained_without_new_filter": weakness_explained_without_filter,
        "pathological_wick_diagnostic_rows": int(integrity.pathological_wick_diagnostic.sum()), "protected_period_violations": int((target_events.exit_ts >= rfbs.PROTECTED).sum()),
    }
    write_json(root/"decision/final_decision.json", {"final_decision": final, **gate})
    source_manifest = json.loads((source_root/"reproducibility/run_manifest.json").read_text())
    code_path = Path(__file__)
    repro = {
        "commit_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "code_path": str(code_path), "code_hash": file_hash(code_path),
        "source_run_root": str(source_root), "source_decision_summary_hash": file_hash(source_root/"decision_summary.json"),
        "source_event_ledger_hash": file_hash(source_root/"materialized/event_ledger.csv"), "source_control_ledger_hash": file_hash(source_root/"controls/control_event_ledger.csv"),
        "source_code_hash": source_manifest["code_hash"], "source_config_hash": source_manifest["config_hash"], "source_data_snapshot_manifest_hash": source_manifest["data_snapshot_manifest_hash"],
        "source_funding_manifest_hash": source_manifest["funding_manifest_hash"], "protected_boundary": rfbs.PROTECTED.isoformat(), "contract_version": CONTRACT_VERSION,
    }
    write_json(root/"reproducibility/run_manifest.json", repro)
    summary = {
        "run_root": str(root), "status": "complete" if final != "focused_mechanical_repair_required" else "blocked_by_protocol_issue", "final_decision": final,
        "formal_candidate": FORMAL_CANDIDATE, "frozen_comparators": list(TARGET_IDS[0:1]+TARGET_IDS[2:]), "definitions_materialized": len(TARGET_IDS),
        "candidate_event_rows": len(target_events), "control_event_rows": len(target_controls), **gate,
        "validation_launched": False, "cpcv_launched": False, "holdout_launched": False, "portfolio_construction_launched": False, "live_work_launched": False,
        "runtime_seconds": time.monotonic()-started, "compact_bundle_path": str(root/"compact_review_bundle"),
    }
    write_json(root/"decision_summary.json", summary)
    bundle_files = (
        "decision_summary.json", "decision/final_decision.json", "identity/selected_policy_pairwise_overlap.csv", "identity/definition_pairwise_overlap.csv", "identity/nesting_audit.csv",
        "controls/matched_unmatched_bias_repaired.csv", "controls/control_risk_quantiles.csv", "controls/control_extreme_rows.csv", "controls/control_risk_focus_audit.csv",
        "materialized/candidate_event_ledger.csv", "controls/materialized_control_ledger.csv", "audit/candidate_replay_parity.csv", "audit/event_path_integrity.csv", "audit/cost_funding_arithmetic.csv",
        "forensics/top_five_winners_losers.csv", "forensics/rfbs_004_special_event_adjudication.csv", "forensics/rfbs_004_horizon_paths.csv", "forensics/frozen_neighborhood_robustness.csv",
        "decision/original_vs_repaired_gate.csv", "decision/candidate_adjudication.csv", "candidate_library/rfbs_candidate_library_update.csv", "reproducibility/run_manifest.json",
    )
    build_bundle(root, bundle_files)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True); parser.add_argument("--source-root", default=str(SOURCE_ROOT)); args = parser.parse_args()
    result = run(Path(args.run_root), Path(args.source_root)); print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
