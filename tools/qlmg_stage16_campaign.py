#!/usr/bin/env python3
"""Outcome-free semantic contract and synthetic utilities for Stage 16.

This module deliberately has no real-data reader.  It operates on JSON-like
contract objects and caller-supplied synthetic rows only.
"""

from __future__ import annotations

import calendar
import hashlib
import itertools
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROTECTED_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
REQUIRED_TRANSLATION_FIELDS = {
    "cell_id", "family", "search_axes", "feature_contract", "rule_model_id",
    "payoff_archetype", "instrument_mapping", "side_mapping", "decision",
    "entry", "exit", "stop_target", "adds_partials", "cost_funding",
    "boundary_handling", "fold_role", "complexity", "multiplicity_lineage",
    "canonical_economic_address_template",
}


class Stage16ContractError(ValueError):
    """Fail-closed Stage 16 semantic contract violation."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Stage16ContractError("timestamp must use UTC Z format")
    result = datetime.fromisoformat(value[:-1] + "+00:00")
    if result.tzinfo != timezone.utc:
        raise Stage16ContractError("timestamp must be UTC")
    return result


class OutcomeReadSpy:
    """Records synthetic reads and rejects any forbidden outcome field/path."""

    FORBIDDEN_TOKENS = {
        "forward", "future", "return", "pnl", "profit", "entry_fill",
        "exit_fill", "outer_label", "protected", "capitalcom", "bootstrap_result",
    }

    def __init__(self) -> None:
        self.reads: list[dict[str, Any]] = []

    def read(self, path: str, columns: Iterable[str], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tokens = {str(path).lower(), *(str(column).lower() for column in columns)}
        if any(forbidden in token for token in tokens for forbidden in self.FORBIDDEN_TOKENS):
            raise Stage16ContractError("outcome firewall rejected read")
        selected = list(columns)
        self.reads.append({"path": path, "columns": selected, "row_count": len(rows)})
        return [{column: row[column] for column in selected} for row in rows]


def quantile_type7(values: Iterable[float], probability: float) -> float:
    """Hyndman-Fan type 7 / NumPy linear quantile, finite values only."""
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not 0.0 <= probability <= 1.0 or not finite:
        raise Stage16ContractError("quantile requires finite observations and p in [0,1]")
    if len(finite) == 1:
        return finite[0]
    index = (len(finite) - 1) * probability
    lower = math.floor(index)
    fraction = index - lower
    return finite[lower] + fraction * (finite[min(lower + 1, len(finite) - 1)] - finite[lower])


def collapse_edges(edges: Iterable[float], minimum_unique: int = 3) -> list[float]:
    result: list[float] = []
    for value in edges:
        number = float(value)
        if not math.isfinite(number):
            raise Stage16ContractError("bin edges must be finite")
        if not result or number > result[-1]:
            result.append(number)
        elif number < result[-1]:
            raise Stage16ContractError("bin edges must be sorted")
    if len(result) < minimum_unique:
        raise Stage16ContractError("insufficient unique quantile edges")
    return result


def assign_right_closed_bin(value: float, edges: list[float]) -> int:
    if not math.isfinite(value):
        raise Stage16ContractError("non-finite feature value")
    if value < edges[0] or value > edges[-1]:
        raise Stage16ContractError("feature value outside registered bin range")
    for index, edge in enumerate(edges[1:], start=0):
        if value <= edge:
            return index
    return len(edges) - 2


def response_surface_contract() -> dict[str, Any]:
    common = {
        "causal_availability": "source_close_ts<=decision_ts and feature_available_ts<=decision_ts",
        "fold_local_normalization": "fit on inner-training observations only; apply frozen edges to validation",
        "quantile_method": "Hyndman-Fan_type_7_linear",
        "interval_convention": "[first_edge,second_edge], then (left,right]",
        "tie_handling": "stable source order by (feature_available_ts,symbol,event_id); equal values share a bin",
        "duplicate_edge_handling": "collapse adjacent equal edges; reject cell-fold when fewer than 3 unique edges",
        "minimum_unique_values": 3,
        "out_of_range": "reject observation; never clip",
        "missingness_rule": "reject observation before any outcome read",
    }
    axes = [
        ("oi_contraction_raw", "oi_log_change_1h", "log_fraction", "identity", "lower_is_stronger", [-0.12, -0.03, -0.02, -0.01, 0.0], ["trade_mark_displacement", "liquidation_context"]),
        ("oi_contraction_rank", "oi_log_change_1h", "fold_local_rank", "empirical_cdf", "lower_is_stronger", [0.0, .2, .4, .6, .8, 1.0], ["trade_mark_displacement", "liquidation_context"]),
        ("trade_mark_displacement_raw", "signed_trade_and_mark_displacement_bps_1h", "bps", "absolute_magnitude_plus_agreed_sign", "separate_positive_negative", [-500.0, -64.0, -32.0, -14.0, 0.0, 14.0, 32.0, 64.0, 500.0], ["oi_contraction", "liquidation_context"]),
        ("trade_mark_displacement_rank", "abs_signed_trade_and_mark_displacement_bps_1h", "fold_local_rank", "empirical_cdf", "higher_is_stronger", [0.0, .2, .4, .6, .8, 1.0], ["oi_contraction", "liquidation_context"]),
        ("liquidation_intensity", "causal_liquidation_volume_intensity", "fold_local_rank", "empirical_cdf", "higher_is_stronger", [0.0, .2, .4, .6, .8, 1.0], ["oi_contraction", "trade_mark_displacement"]),
        ("purge_breadth_raw", "directional_completed_purge_count_over_PIT_eligible_denominator", "share", "identity", "direction_kept_separate", [0.0, .01, .02, .05, .10, 1.0], ["purge_identity", "diagnostic_window"]),
        ("purge_breadth_rank", "directional_completed_purge_breadth_share", "fold_local_rank", "empirical_cdf", "higher_is_broader", [0.0, .2, .4, .6, .8, 1.0], ["purge_identity", "diagnostic_window"]),
        ("basis_change_raw", "signed_basis_change_bps_1h", "bps", "identity", "negative_is_downside_confirmation", [-500.0, -64.0, -32.0, -14.0, 0.0], ["trade_mark_structural", "oi_contraction"]),
        ("basis_or_component_rank", "registered_component_intensity", "fold_local_rank", "empirical_cdf", "higher_is_stronger", [0.0, .2, .4, .6, .8, 1.0], ["trade_mark_structural", "oi_contraction"]),
    ]
    rows = []
    for axis_id, feature, unit, transform, sign, edges, partners in axes:
        row = dict(common)
        row.update({"axis_id": axis_id, "source_feature": feature, "native_unit": unit,
                    "transformation": transform, "sign_direction_treatment": sign,
                    "edges_or_probabilities": edges, "allowed_interaction_partners": partners,
                    "reason_for_inclusion": "Stage-14 Phase-1 admitted causal mechanism primitive; thresholds are outcome-free cost-scale or bounded rank bins"})
        rows.append(row)
    return {"version": "stage16_v1", "axes": rows, "all_bin_formulas_serialized": True,
            "threshold_derivation": {"raw": "fixed pre-outcome native-unit edges above",
                                     "rank": "type-7 edges from inner-training only at registered probabilities"}}


def estimator_rule_inventory() -> dict[str, Any]:
    grammar = {
        "KDA02B": {"primitive_predicates": ["oi_contraction", "trade_mark_sign_agreement", "price_magnitude", "liquidation_context"], "maximum_conditions": 4, "maximum_interaction_depth": 3, "boolean_operators": ["AND"], "episode_identity": "symbol-local first false-to-true OI-vacuum onset; reset only after all mandatory predicates false on a completed bar", "confirmation_sequence": ["OI vacuum and agreed trade/mark impulse complete at decision"], "direction_branches": ["continuation", "reversal"], "horizon_choices": ["1h", "3h", "6h"], "complexity_score": "count(mandatory predicates)+interaction depth-1", "duplicate_rule_canonicalization": "sorted predicate IDs + branch + horizon; SHA-256 canonical JSON"},
        "KDA02C": {"primitive_predicates": ["completed_purge_reversal", "directional_PIT_breadth", "purge_identity", "window"], "maximum_conditions": 4, "maximum_interaction_depth": 3, "boolean_operators": ["AND"], "episode_identity": "underlying base-event ID; primary and robustness purge identities are separate inherited attempts", "confirmation_sequence": ["purge completes", "symbol reclaim/failure completes", "breadth is sampled causally at that completion"], "direction_branches": ["negative_reclaim_long", "positive_failure_short"], "horizon_choices": ["1h"], "complexity_score": "count(mandatory predicates)+interaction depth-1", "duplicate_rule_canonicalization": "base identity + breadth form + window + direction; SHA-256 canonical JSON"},
        "KDX01": {"primitive_predicates": ["downside_trade_mark_displacement", "oi_contraction", "liquidation", "basis_level", "basis_change", "breadth", "completed_trade_mark_reclaim"], "maximum_conditions": 7, "maximum_interaction_depth": 6, "boolean_operators": ["AND"], "episode_identity": "symbol-local downside state onset through completed trade-and-mark reclaim; one candidate per episode", "confirmation_sequence": ["downside state", "registered derivative primitives", "completed trade-and-mark structural reclaim"], "direction_branches": ["reversal_long"], "horizon_choices": ["1h", "3h", "6h"], "complexity_score": "count(mandatory predicates)+interaction depth-1", "duplicate_rule_canonicalization": "sorted component IDs + scaling + horizon; SHA-256 canonical JSON"},
    }
    return {"model_inventory": [], "maximum_model_count": 0,
            "model_inventory_interpretation": "explicitly empty: response bins plus constrained deterministic rules only",
            "rule_grammar": grammar}


OUTER_QUARTERS = [
    ("2023Q4", "2023-09-30T12:00:00Z", "2023-10-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    ("2024Q1", "2023-12-31T12:00:00Z", "2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z"),
    ("2024Q2", "2024-03-31T12:00:00Z", "2024-04-01T00:00:00Z", "2024-07-01T00:00:00Z"),
    ("2024Q3", "2024-06-30T12:00:00Z", "2024-07-01T00:00:00Z", "2024-10-01T00:00:00Z"),
    ("2024Q4", "2024-09-30T12:00:00Z", "2024-10-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    ("2025Q1", "2024-12-31T12:00:00Z", "2025-01-01T00:00:00Z", "2025-04-01T00:00:00Z"),
    ("2025Q2", "2025-03-31T12:00:00Z", "2025-04-01T00:00:00Z", "2025-07-01T00:00:00Z"),
    ("2025Q3", "2025-06-30T12:00:00Z", "2025-07-01T00:00:00Z", "2025-10-01T00:00:00Z"),
    ("2025Q4", "2025-09-30T12:00:00Z", "2025-10-01T00:00:00Z", "2026-01-01T00:00:00Z"),
]


def month_start(value: datetime) -> datetime:
    return datetime(value.year, value.month, 1, tzinfo=timezone.utc)


def next_month(value: datetime) -> datetime:
    year, month = (value.year + 1, 1) if value.month == 12 else (value.year, value.month + 1)
    return datetime(year, month, 1, tzinfo=timezone.utc)


def build_inner_folds(development_start: str, development_end: str) -> list[dict[str, Any]]:
    start, end = parse_utc(development_start), parse_utc(development_end)
    cursor = next_month(month_start(start))  # guarantees nonempty expanding training history
    candidates = []
    while next_month(cursor) <= end:
        validation_end = next_month(cursor)
        candidates.append({
            "inner_fold_id": f"M_{cursor:%Y%m}",
            "training_start": development_start,
            "training_latest_exit_exclusive": (cursor - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "purge_start": (cursor - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "validation_start": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "validation_end_exclusive": validation_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "embargo_end": (validation_end + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "maximum_horizon_hours": 6,
            "maximum_episode_overlap_hours": 6,
            "purge_embargo_hours": 12,
            "event_boundary": "decision, entry, and actual exit all inside validation block; otherwise reject",
        })
        cursor = validation_end
    selected = candidates[-6:]
    if len(selected) < 3:
        raise Stage16ContractError("fewer than three valid complete monthly inner folds")
    return selected


def inner_fold_contract() -> dict[str, Any]:
    rows = []
    for lane in ("KDA02B_v2_oi_vacuum_redevelopment", "KDA02C_v1_purge_breadth_context", "KDX01_v1_downside_completed_derivatives_state_rejection"):
        for quarter, dev_end, eval_start, eval_end in OUTER_QUARTERS:
            rows.append({"outer_fold_id": f"{lane}:{quarter}", "hypothesis_id": lane,
                         "development_start": "2023-04-01T00:00:00Z", "development_end_exclusive": dev_end,
                         "outer_evaluation_start": eval_start, "outer_evaluation_end_exclusive": eval_end,
                         "purge_interval_hours": 12, "embargo_interval_hours": 12,
                         "maximum_label_horizon_hours": 6, "maximum_episode_duration_for_overlap_hours": 6,
                         "maximum_episode_duration_basis": "pre-outcome registered cap: an episode still active more than 6h after onset is rejected before outcome; 12h purge equals 6h maximum label plus 6h maximum episode overlap",
                         "boundary_treatment": "decision, entry, and actual exit must all be in the same outer block; crossing event rejected, never force-closed",
                         "programme_exposure_class": "program_exposed_historical" if not lane.startswith("KDX01") else "cross_family_program_exposed_redevelopment",
                         "inner_folds": build_inner_folds("2023-04-01T00:00:00Z", dev_end),
                         "no_backward_transfer": True})
    return {"construction": "latest six complete UTC calendar months with nonempty expanding prior history; all when fewer; minimum three",
            "Stage14_fold_change_reason": "authority defect closed: Stage14 six-hour gap did not cover both six-hour maximum horizon and a bounded episode overlap; Stage16 uses a pre-outcome six-hour episode cap and twelve-hour purge/embargo",
            "outer_folds": rows}


def metric_contract() -> dict[str, Any]:
    metrics = {
        "accepted_trade_count": ["count", "count accepted economic addresses", "maximize diagnostic"],
        "independent_market_day_clusters": ["count", "unique UTC entry dates", "hard minimum"],
        "independent_utc_hour_clusters": ["count", "unique UTC entry-hour buckets", "diagnostic"],
        "aggregate_base_net_mean_bps": ["bps", "mean of market-day means after 14bps and primary funding", "maximize"],
        "aggregate_stress_net_mean_bps": ["bps", "mean of market-day means after 32bps and conservative funding", "maximize"],
        "base_net_median_bps": ["bps", "median accepted-trade base net", "maximize diagnostic"],
        "median_inner_fold_base_net_mean_bps": ["bps", "type-7 p=.5 of inner-fold aggregate base means", "maximize"],
        "p20_inner_fold_base_net_mean_bps": ["bps", "type-7 p=.2 of inner-fold aggregate base means", "maximize"],
        "cluster_bootstrap_lower_bound_bps": ["bps", "5th percentile of 2000 market-day cluster resamples with replacement; seed 20260720", "maximize"],
        "left_tail_utility_bps": ["bps", "negative maximum peak-to-trough drawdown of chronological equal-day base-net cumulative bps", "maximize"],
        "opportunity_frequency_per_30d": ["trades/30d", "30*accepted trades/eligible calendar days", "maximize subject to independence"],
        "capital_occupancy": ["fraction", "sum holding seconds/(eligible symbols*eligible interval seconds)", "diagnostic; reject outside [0,1]"],
        "execution_margin_bps": ["bps", "aggregate stress-net mean bps", "maximize"],
        "symbol_day_year_contribution": ["share", "absolute base-net contribution by symbol/day/year divided by total absolute contribution", "diagnostic limitation; [0,1]"],
        "complexity": ["integer", "registered grammar score", "minimize"],
        "candidate_return_correlation": ["absolute Pearson", "aligned equal-market-day base-net series; fewer than 3 overlaps is missing/worst=1", "minimize"],
    }
    return {"primary_weighting": "equal weight trades inside UTC market day; equal weight market days",
            "diagnostic_weighting": ["trade_weighted", "equal UTC-hour clusters"],
            "metrics": {key: {"unit": value[0], "formula": value[1], "direction": value[2]} for key, value in metrics.items()},
            "eligibility": {"integrity_required": True, "minimum_accepted_trades": 30,
                            "minimum_market_day_clusters": 20, "minimum_utc_hour_clusters": 20,
                            "aggregate_development_base_net_mean_bps": ">0",
                            "nonpositive_median_or_stability_diagnostics": "route_or_limitation_tag_not_universal_family_kill"}}


OBJECTIVES = [
    ("aggregate_base_net_mean_bps", "maximize"),
    ("median_inner_fold_base_net_mean_bps", "maximize"),
    ("p20_inner_fold_base_net_mean_bps", "maximize"),
    ("cluster_bootstrap_lower_bound_bps", "maximize"),
    ("left_tail_utility_bps", "maximize"),
    ("opportunity_frequency_per_30d", "maximize"),
    ("execution_margin_bps", "maximize"),
    ("complexity", "minimize"),
    ("max_abs_correlation_to_selected", "minimize"),
]


def utility_pareto_contract() -> dict[str, Any]:
    return {"objectives": [{"objective": name, "direction": direction, "normalization": "none; native finite scale",
                            "missing_or_nonfinite": "worst: -infinity for maximize, +infinity for minimize"}
                           for name, direction in OBJECTIVES],
            "dominance": "A dominates B iff A is no worse on every active objective and strictly better on at least one",
            "finite_values_required_for_selection": True,
            "correlation_first_candidate": "0.0 because selected set is empty; subsequent values use maximum absolute aligned-day correlation",
            "bootstrap": {"resamples": 2000, "seed": 20260720, "unit": "whole UTC market-day cluster", "quantile": "type7 p=0.05"}}


def _worst(value: Any, direction: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -math.inf if direction == "maximize" else math.inf
    if not math.isfinite(number):
        return -math.inf if direction == "maximize" else math.inf
    return number


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    no_worse, strict = True, False
    for name, direction in OBJECTIVES:
        av, bv = _worst(a.get(name), direction), _worst(b.get(name), direction)
        if (direction == "maximize" and av < bv) or (direction == "minimize" and av > bv):
            no_worse = False
        if (direction == "maximize" and av > bv) or (direction == "minimize" and av < bv):
            strict = True
    return no_worse and strict


def pareto_set(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if not any(dominates(other, candidate) for other in candidates if other is not candidate)]


def aligned_day_correlation(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return 1.0
    av, bv = [float(a[key]) for key in keys], [float(b[key]) for key in keys]
    if any(not math.isfinite(value) for value in av + bv):
        return 1.0
    am, bm = sum(av) / len(av), sum(bv) / len(bv)
    numerator = sum((x-am)*(y-bm) for x, y in zip(av, bv))
    ad = math.sqrt(sum((x-am)**2 for x in av)); bd = math.sqrt(sum((y-bm)**2 for y in bv))
    return 1.0 if ad == 0 or bd == 0 else abs(numerator / (ad * bd))


def beam_contract() -> dict[str, Any]:
    return {"maximum_candidates_per_family_fold": 5,
            "source": "nondominated eligible cells only",
            "selection": "deterministic greedy",
            "correlation_preference": "after first, prefer abs development-return correlation <0.85; if none tag candidate_beam_high_correlation",
            "lexicographic_tie_break": ["higher median_inner_fold_base_net_mean_bps", "higher p20_inner_fold_base_net_mean_bps", "higher aggregate_base_net_mean_bps", "higher cluster_bootstrap_lower_bound_bps", "higher left_tail_utility_bps", "higher execution_margin_bps", "lower complexity", "lower max_abs_correlation_to_selected", "lexicographically smaller canonical_translation_id"],
            "manual_choice_allowed": False}


def deterministic_beam(candidates: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    static = [name for name, _ in OBJECTIVES if name != "max_abs_correlation_to_selected"]
    remaining = []
    for row in candidates:
        if not isinstance(row.get("market_day_returns"), dict):
            continue
        eligible = (row.get("integrity_pass") is True and row.get("accepted_trade_count", 0) >= 30 and
                    row.get("independent_market_day_clusters", 0) >= 20 and
                    row.get("independent_utc_hour_clusters", 0) >= 20 and
                    _finite(row.get("aggregate_base_net_mean_bps")) and float(row["aggregate_base_net_mean_bps"]) > 0)
        if eligible and all(_finite(row.get(name)) for name in static):
            remaining.append(dict(row))
    selected = []
    while remaining and len(selected) < limit:
        annotated = []
        for row in remaining:
            item = dict(row)
            item["max_abs_correlation_to_selected"] = 0.0 if not selected else max(
                aligned_day_correlation(item["market_day_returns"], chosen["market_day_returns"])
                for chosen in selected)
            annotated.append(item)
        nondominated = pareto_set(annotated)
        low_corr = [row for row in nondominated if not selected or row["max_abs_correlation_to_selected"] < .85]
        pool = low_corr or nondominated
        def key(row: dict[str, Any]) -> tuple[Any, ...]:
            return tuple(-_worst(row.get(name), "maximize") for name in
                         ("median_inner_fold_base_net_mean_bps", "p20_inner_fold_base_net_mean_bps", "aggregate_base_net_mean_bps", "cluster_bootstrap_lower_bound_bps", "left_tail_utility_bps", "execution_margin_bps")) + (
                         _worst(row.get("complexity"), "minimize"),
                         _worst(row.get("max_abs_correlation_to_selected"), "minimize"),
                         str(row["canonical_translation_id"]))
        chosen = min(pool, key=key)
        if selected and not low_corr:
            chosen.setdefault("tags", []).append("candidate_beam_high_correlation")
        selected.append(chosen)
        remaining = [row for row in remaining if row["canonical_translation_id"] != chosen["canonical_translation_id"]]
    return selected


def common_execution(horizon: str) -> dict[str, Any]:
    return {
        "decision": {"timestamp_rule": "causal completed event-bar close availability", "requires": ["source_close_ts<=decision_ts", "feature_available_ts<=decision_ts"]},
        "entry": {"timestamp_rule": "first authorized native PF 5m trade-bar open >= decision_ts", "price_field": "trade_open", "maximum_delay_minutes": 10},
        "exit": {"type": "fixed_horizon", "horizon": horizon, "target_rule": "decision_ts + registered horizon", "timestamp_rule": "first authorized native PF 5m trade-bar open >= target", "price_field": "trade_open", "maximum_delay_minutes": 10},
        "stop_target": {"stop": "none", "target": "none"},
        "adds_partials": {"adds": "none", "partial_exits": "none"},
        "boundary_handling": {"non_overlap": "definition-local and symbol-local using actual executable exit_ts", "fold": "decision, entry, exit inside same authorized outer block", "protected": "all timestamps strictly before 2026-01-01T00:00:00Z", "lifecycle": "PIT tradable and valid for full interval; else reject before outcome", "missing_executable_price": "reject before outcome", "simultaneous_collision": "canonical translation ID lexical order only within identical definition-symbol address; duplicates rejected, distinct definitions remain separate attempts"},
    }


def _address_template(cell: dict[str, Any]) -> str:
    payload = {key: cell[key] for key in ("cell_id", "family", "search_axes", "feature_contract", "rule_model_id", "payoff_archetype", "instrument_mapping", "side_mapping", "decision", "entry", "exit", "stop_target", "adds_partials", "cost_funding", "boundary_handling", "fold_role", "multiplicity_lineage")}
    if "source_stage14_cell_id" in cell:
        payload["source_stage14_cell_id"] = cell["source_stage14_cell_id"]
    return canonical_sha256(payload)


def economic_address(template_hash: str, base_event_id: str, symbol: str, side: str,
                     decision_ts: str, outer_fold_id: str) -> str:
    if not base_event_id or side not in {"long", "short"} or parse_utc(decision_ts) >= PROTECTED_START:
        raise Stage16ContractError("invalid economic address identity")
    return canonical_sha256({"template_hash": template_hash, "base_event_id": base_event_id,
                             "native_pf_symbol": symbol,
                             "side": side, "decision_ts": decision_ts, "outer_fold_id": outer_fold_id})


def resolve_fixed_execution(decision_ts: str, horizon: str, trade_bar_opens: list[str],
                            fold_start: str, fold_end: str) -> dict[str, str]:
    """Resolve first eligible synthetic 5m opens with a strict ten-minute cap."""
    decision, start, end = map(parse_utc, (decision_ts, fold_start, fold_end))
    if not start <= decision < end or decision >= PROTECTED_START:
        raise Stage16ContractError("decision outside authorized fold")
    try:
        hours = int(horizon.removesuffix("h"))
    except (ValueError, AttributeError) as exc:
        raise Stage16ContractError("fixed horizon must be integer hours") from exc
    target = decision + timedelta(hours=hours)
    opens = sorted(parse_utc(value) for value in trade_bar_opens)
    entry_options = [value for value in opens if decision <= value <= decision + timedelta(minutes=10)]
    exit_options = [value for value in opens if target <= value <= target + timedelta(minutes=10)]
    if not entry_options or not exit_options:
        raise Stage16ContractError("missing executable bar within delay cap")
    entry, exit_ts = entry_options[0], exit_options[0]
    if not (start <= entry <= exit_ts < end) or exit_ts >= PROTECTED_START:
        raise Stage16ContractError("entry or exit crosses fold/protected boundary")
    return {"decision_ts": decision_ts, "entry_ts": entry.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "exit_target_ts": target.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "exit_ts": exit_ts.strftime("%Y-%m-%dT%H:%M:%SZ")}


def validate_episode_duration(onset_ts: str, episode_end_ts: str, cap_hours: int = 6) -> None:
    onset, end = parse_utc(onset_ts), parse_utc(episode_end_ts)
    if end < onset or end - onset > timedelta(hours=cap_hours):
        raise Stage16ContractError("episode duration exceeds registered pre-outcome cap")


def side_for_kda02b(branch: str, trade_displacement: float, mark_displacement: float) -> str | None:
    if trade_displacement == 0 or mark_displacement == 0 or math.copysign(1, trade_displacement) != math.copysign(1, mark_displacement):
        return None
    impulse_positive = trade_displacement > 0
    if branch == "continuation":
        return "long" if impulse_positive else "short"
    if branch == "reversal":
        return "short" if impulse_positive else "long"
    raise Stage16ContractError("unknown KDA02B branch")


def side_for_kda02c(direction: str) -> str:
    if direction == "negative": return "long"
    if direction == "positive": return "short"
    raise Stage16ContractError("unknown KDA02C direction")


def directional_breadth(events: list[dict[str, Any]], decision_source_ts: str, window_minutes: int,
                        direction: str, identity: str, eligible_denominator: int) -> dict[str, float]:
    decision = parse_utc(decision_source_ts)
    if window_minutes not in {5, 15, 30, 60} or direction not in {"negative", "positive"} or identity not in {"primary_z2", "robust_pct95"} or eligible_denominator <= 0:
        raise Stage16ContractError("invalid directional breadth request")
    expected_parent = -1 if direction == "negative" else 1
    lower = decision - timedelta(minutes=window_minutes)
    identities = {str(row["base_event_id"]) for row in events
                  if lower < parse_utc(row["decision_source_ts"]) <= decision
                  and int(row["parent_direction"]) == expected_parent
                  and row["purge_identity"] == identity}
    return {"directional_onset_count": len(identities), "directional_onset_share": len(identities)/eligible_denominator}


def kdx_episode_trace(rows: list[dict[str, Any]], mandatory_predicates: list[str]) -> list[dict[str, str]]:
    """Synthetic state-machine trace for the exact KDX onset/reset contract."""
    ordered = sorted(rows, key=lambda row: parse_utc(row["source_close_ts"]))
    open_onset: datetime | None = None; rearmed = True; trace = []
    for row in ordered:
        ts = parse_utc(row["source_close_ts"])
        state = all(row.get(predicate) is True for predicate in mandatory_predicates)
        if open_onset is not None and ts - open_onset > timedelta(hours=6):
            trace.append({"event": "expired", "source_close_ts": row["source_close_ts"]}); open_onset = None; rearmed = False
        if open_onset is not None and row.get("completed_trade_mark_reclaim") is True:
            trace.append({"event": "completed_reclaim", "source_close_ts": row["source_close_ts"]}); open_onset = None; rearmed = False
        if open_onset is None and not state:
            rearmed = True
        if open_onset is None and rearmed and state:
            open_onset = ts; rearmed = False
            trace.append({"event": "onset", "source_close_ts": row["source_close_ts"]})
    return trace


def build_translation_registry() -> dict[str, Any]:
    cost = {"base_pre_funding_round_trip_bps": 14, "stress_pre_funding_round_trip_bps": 32,
            "funding_partitions": ["exact", "mixed", "imputed", "zero_boundary"],
            "missing_funding": "reject before outcome; never zero-fill"}
    cells: list[dict[str, Any]] = []

    def add(family: str, index: int, axes: dict[str, str], archetype: str,
            instrument: dict[str, Any], side: dict[str, Any], feature: dict[str, Any], horizon: str,
            complexity: int, lineage: list[str]) -> None:
        execution = common_execution(horizon)
        cell = {"cell_id": f"{family}_{index:03d}", "family": family, "search_axes": axes,
                "feature_contract": feature, "rule_model_id": f"RULE_{family}_{index:03d}",
                "payoff_archetype": archetype, "instrument_mapping": instrument,
                "side_mapping": side, "decision": execution["decision"], "entry": execution["entry"],
                "exit": execution["exit"], "stop_target": execution["stop_target"],
                "adds_partials": execution["adds_partials"], "cost_funding": cost,
                "boundary_handling": execution["boundary_handling"], "fold_role": "inner_development_then_deterministic_freeze_for_next_outer_block",
                "complexity": complexity, "multiplicity_lineage": lineage}
        cell["canonical_economic_address_template"] = _address_template(cell)
        cell["canonical_translation_id"] = f"TR_{cell['canonical_economic_address_template'][:24]}"
        cells.append(cell)

    axes_b = {"oi_axis": ["raw_oi_log_change", "fold_local_percentile"], "price_axis": ["raw_bps", "fold_local_rank"], "price_state": ["negative", "positive"], "branch": ["continuation", "reversal"], "horizon": ["1h", "3h", "6h"], "liquidation_context": ["continuous_intensity", "present_absent"]}
    for index, values in enumerate(itertools.product(*axes_b.values()), 1):
        axes = dict(zip(axes_b, values)); branch = axes["branch"]
        thresholds = {"oi": "oi_log_change_1h<=-0.01" if axes["oi_axis"] == "raw_oi_log_change" else "inner-training type7 rank<=0.20",
                      "price": "abs(trade_displacement_bps_1h)>=14 and abs(mark_displacement_bps_1h)>=14" if axes["price_axis"] == "raw_bps" else "both inner-training type7 absolute-displacement ranks>=0.80",
                      "liquidation": "causal_liquidation_volume_intensity inner-training type7 rank>=0.80" if axes["liquidation_context"] == "continuous_intensity" else "causal_liquidation_volume>0"}
        add("KDA02B", index, axes, "symmetric_directional" if branch == "continuation" else "mean_reversion",
            {"instrument_type": "Kraken linear PF", "rule": "native PF symbol of causal OI-vacuum event; current-roster substitution forbidden"},
            {"rule": "agreed contemporaneous trade/mark impulse sign; continuation follows, reversal opposes", "zero_or_conflicting_sign": "no candidate"},
            {"price_window": "completed 1h displacement ending at decision", "sign_agreement_required": True, "required_impulse_sign": axes["price_state"], "exact_thresholds": thresholds, "liquidation_state": axes["liquidation_context"], "episode_reset": "after all mandatory predicates false on completed bar"}, axes["horizon"], 4, ["KDA02", "KDA02B_redevelopment", "programme_exposed_historical"])

    axes_c = {"purge_identity": ["primary_z2", "robust_pct95"], "diagnostic_window": ["5m", "15m", "30m", "60m"], "breadth_form": ["raw_share", "fold_local_rank", "isolated_vs_nonisolated"], "direction": ["negative", "positive"]}
    for index, values in enumerate(itertools.product(*axes_c.values()), 1):
        axes = dict(zip(axes_c, values))
        breadth_threshold = {"raw_share": "directional breadth share>=0.02",
                             "fold_local_rank": "inner-training type7 directional breadth rank>=0.80",
                             "isolated_vs_nonisolated": "directional completed-purge count>=2; isolated count==1 is rejected"}[axes["breadth_form"]]
        window_minutes = int(axes["diagnostic_window"].removesuffix("m"))
        breadth_formula = {"source": "Stage14 causal completed-purge base-event tape plus KDA02C PIT eligible denominator", "source_builder_sha256": "9f1bdd3426e97944dbd66ad56bf3bd17b91ab7de926adaf692b9d77aa422de7c", "window_minutes": window_minutes, "interval": "(decision_source_ts-window, decision_source_ts]", "direction_filter": axes["direction"], "direction_field": "parent_direction; negative=-1, positive=+1; trade_direction is not used because completed reversal negates it", "identity_filter": axes["purge_identity"], "breadth_form_mapping": {"raw_share": "directional_onset_count/eligible_at_window_end", "fold_local_rank": "type7 rank of directional_onset_count/eligible_at_window_end fit on inner-training only", "isolated_vs_nonisolated": "directional_onset_count==1 isolated; >=2 nonisolated; cell requires nonisolated"}, "numerator": "count distinct completed-purge base_event_id onsets passing identity and parent_direction filters in interval", "denominator": "eligible PF count at decision_source_ts after lifecycle/trade/mark/analytics masks", "raw_share": "numerator/denominator", "decision_source_ts": "underlying base-event decision_ts minus five minutes; selects the completed source-bar timestamp in the Stage14 panel", "feature_available_ts": "decision_source_ts plus five minutes, exactly equal to underlying base-event decision_ts", "denominator_missing_or_zero": "reject before outcome"}
        add("KDA02C", index, axes, "mean_reversion",
            {"instrument_type": "Kraken linear PF", "rule": "native PF symbol of underlying frozen completed-purge reversal base event", "not_allowed": ["BTC proxy", "ETH proxy", "equal-weight portfolio"]},
            {"negative": "completed downside purge/reclaim -> long", "positive": "completed upside purge/failure -> short", "selected": side_for_kda02c(axes["direction"])},
            {"base_event": "exact frozen completed-purge reversal identity", "base_event_authority": {"generator_path": "tools/qlmg_kda02_v2.py", "generator_file_sha256": "2588bbe362488c7d8a753718ea47b7c573176047bc98a37689e5b396d13b1873", "event_id_prefix": "kda02v2_event_", "event_id_payload_keys": ["branch_id", "attempt", "symbol", "parent_direction", "trade_direction", "parent_episode_id", "decision_ts", "feature_extension_hash", "generator_hash"], "event_type_required": "completed_purge_reversal", "contract_file_sha256": "2833fbab498ebdf3bf3d86801e442779ef1f3396fd5f32d5c8f3658402eb671d"}, "decision": "completed reclaim/failure availability for same symbol", "breadth_formula": breadth_formula, "exact_breadth_threshold": breadth_threshold, "purge_attempt": axes["purge_identity"], "primary_and_robustness_are_separate_attempts": True}, "1h", 4, ["KDA02", "KDA02C_context", axes["purge_identity"], "programme_exposed_historical"])

    ladders = ["trade_mark_structural", "trade_mark_structural_oi", "trade_mark_structural_oi_liquidation", "trade_mark_structural_oi_basis_level", "trade_mark_structural_oi_basis_change", "trade_mark_structural_oi_breadth", "trade_mark_structural_oi_liquidation_basis_change"]
    ladder_primitives = {
        "trade_mark_structural": ["downside_trade_displacement", "downside_mark_displacement", "completed_trade_mark_reclaim"],
        "trade_mark_structural_oi": ["downside_trade_displacement", "downside_mark_displacement", "oi_contraction", "completed_trade_mark_reclaim"],
        "trade_mark_structural_oi_liquidation": ["downside_trade_displacement", "downside_mark_displacement", "oi_contraction", "liquidation_intensity", "completed_trade_mark_reclaim"],
        "trade_mark_structural_oi_basis_level": ["downside_trade_displacement", "downside_mark_displacement", "oi_contraction", "negative_basis_level", "completed_trade_mark_reclaim"],
        "trade_mark_structural_oi_basis_change": ["downside_trade_displacement", "downside_mark_displacement", "oi_contraction", "negative_basis_change", "completed_trade_mark_reclaim"],
        "trade_mark_structural_oi_breadth": ["downside_trade_displacement", "downside_mark_displacement", "oi_contraction", "directional_PIT_breadth", "completed_trade_mark_reclaim"],
        "trade_mark_structural_oi_liquidation_basis_change": ["downside_trade_displacement", "downside_mark_displacement", "oi_contraction", "liquidation_intensity", "negative_basis_change", "completed_trade_mark_reclaim"],
    }
    axes_x = {"component_ladder": ladders, "component_scaling": ["raw_unit", "fold_local_rank"], "horizon": ["1h", "3h", "6h"]}
    for index, values in enumerate(itertools.product(*axes_x.values()), 1):
        axes = dict(zip(axes_x, values))
        stage14_block = (index - 1) // 3
        source_stage14_cell_id = f"KDX01_{stage14_block * 6 + 4 + (index - 1) % 3:03d}"
        raw_thresholds = {"trade_and_mark": "both completed 1h displacements<=-14bps", "oi": "oi_log_change_1h<=-0.01", "liquidation": "causal_liquidation_volume>0", "basis_level": "basis_bps<=-14", "basis_change": "basis_change_bps_1h<=-14", "breadth": "directional breadth share>=0.02"}
        rank_thresholds = {"trade_and_mark": "both inner-training type7 downside-magnitude ranks>=0.80", "oi": "inner-training type7 OI rank<=0.20", "liquidation": "inner-training type7 intensity rank>=0.80", "basis_level": "inner-training type7 downside basis rank<=0.20", "basis_change": "inner-training type7 downside-change rank<=0.20", "breadth": "inner-training type7 directional breadth rank>=0.80"}
        before = len(cells)
        add("KDX01", index, axes, "mean_reversion",
            {"instrument_type": "Kraken linear PF", "rule": "native PF symbol of downside completed-rejection event"},
            {"rule": "long after completed downside trade-and-mark structural rejection/reclaim"},
            {"required_components": ladder_primitives[axes["component_ladder"]], "component_scaling": axes["component_scaling"], "exact_thresholds": raw_thresholds if axes["component_scaling"] == "raw_unit" else rank_thresholds, "episode_identity": {"onset": "first completed 5m bar on which every cell-required pre-reclaim primitive is true after a rearm bar", "mandatory_component_timing": "all required primitives must be true on the same completed source bar; delayed OI/liquidation/basis activation delays onset", "flicker": "once open, predicate flicker neither closes nor creates a new episode", "close": "first completed causal reclaim or expiry strictly after onset+6h", "reset": "after reclaim/expiry, require one completed bar with at least one mandatory pre-reclaim primitive false before another all-true false-to-true onset", "cap_hours": 6, "reference_binding": "reference prices are taken immediately before this exact onset"}, "causal_reclaim": {"reference_timestamp": "last completed contiguous 5m bar strictly before downside episode onset", "trade_reference_field": "trade_close", "mark_reference_field": "mark_close", "lookback": "one completed 5m bar; missing or noncontiguous reference rejects", "confirmation_bar": "first completed 5m bar after onset on which trade_close>=trade_reference and mark_close>=mark_reference", "comparison_operator": ">= on both fields", "decision_ts": "confirmation source_close_ts plus five minutes availability", "availability": "source_close_ts plus five minutes equals feature_available_ts equals decision_ts"}, "breadth_context": ({"required": True, "purge_identity": "primary_z2", "direction": "negative", "direction_field": "parent_direction=-1", "window_minutes": 5, "interval": "(decision_source_ts-5m,decision_source_ts]", "timestamp": "KDX decision_ts minus five minutes; selects completed source-bar timestamp", "feature_available_ts": "decision_source_ts plus five minutes equals KDX decision_ts", "numerator": "count distinct primary_z2 completed-purge base_event_id with parent_direction=-1 in interval", "denominator": "PIT eligible PF count at decision_source_ts", "formula": "directional_onset_count/eligible", "raw_threshold": ">=0.02", "rank_threshold": "inner-training type7 rank>=0.80", "source_builder_sha256": "9f1bdd3426e97944dbd66ad56bf3bd17b91ab7de926adaf692b9d77aa422de7c"} if "breadth" in axes["component_ladder"] else {"required": False})}, axes["horizon"], 2 + len(ladder_primitives[axes["component_ladder"]]), ["KDA01", "KDA02", "KDA03", "KDX01", "cross_family_program_exposed_redevelopment", "Stage14_continuation_null_attempts_retained_in_inherited_multiplicity_not_executable"])
        cells[before]["source_stage14_cell_id"] = source_stage14_cell_id
        cells[before]["canonical_economic_address_template"] = _address_template(cells[before])
        cells[before]["canonical_translation_id"] = f"TR_{cells[before]['canonical_economic_address_template'][:24]}"

    if len(cells) != 186 or len({row["cell_id"] for row in cells}) != 186:
        raise Stage16ContractError("translation registry must contain 186 unique executable cells")
    retired = retired_stage14_kdx_attempts()
    return {"version": "stage16_v1", "cells_by_family": {"KDA02B": 96, "KDA02C": 48, "KDX01": 42},
            "total_cells": 186, "Stage14_registered_cells": 228,
            "removed_from_executable_registry": {"KDX01_continuation_null_cells": 42, "attempts": retired, "multiplicity_treatment": "retained as inherited explored attempts; never eligible executable translations because frozen KDX01 archetype is long mean-reversion"},
            "every_cell_is_potential_frozen_translation": True, "cells": cells}


def retired_stage14_kdx_attempts() -> list[dict[str, Any]]:
    ladders = ["trade_mark_structural", "trade_mark_structural_oi", "trade_mark_structural_oi_liquidation", "trade_mark_structural_oi_basis_level", "trade_mark_structural_oi_basis_change", "trade_mark_structural_oi_breadth", "trade_mark_structural_oi_liquidation_basis_change"]
    rows = []
    for ladder, scaling, direction, horizon in itertools.product(ladders, ["raw_unit_continuous", "fold_local_rank_continuous"], ["continuation", "reversal"], ["1h", "3h", "6h"]):
        index = len(rows) + 1
        if direction == "continuation":
            rows.append({"stage14_cell_id": f"KDX01_{index:03d}", "parameters": {"component_ladder": ladder, "component_scaling": scaling, "direction": direction, "horizon": horizon}, "status": "non_executable_inherited_attempt", "exclusion_reason": "contradicts Stage16 frozen KDX01 long mean-reversion payoff archetype", "can_enter_translation_or_beam": False})
        else:
            # Keep enumeration aligned to the original 84-cell Cartesian order.
            rows.append(None)
    return [row for row in rows if row is not None]


def boundary_contract() -> dict[str, Any]:
    return {"feature_lookback_crossing_source_start": "reject event before outcome",
            "OI_retention_start": "reject until full registered OI lookback is observed; never backfill",
            "missing_basis_or_liquidation_normalization": "reject cells requiring missing primitive; never zero/impute",
            "changing_PIT_denominator": "compute from lifecycle-authorized eligible symbols at decision; denominator<=0 rejects",
            "duplicate_quantile_edges": "collapse adjacent equal edges; fewer than 3 unique rejects cell-fold",
            "event_spanning_fold_boundary": "reject; no artificial close",
            "entry_or_exit_after_fold_end": "reject before outcome",
            "entry_or_exit_after_protected_start": "reject before outcome and global stop on attempted protected read",
            "lifecycle_invalid_interval": "reject before outcome",
            "missing_funding_boundary": "reject event before outcome; never relabel zero-boundary",
            "missing_executable_bar": "reject when no authorized open within 10 minutes",
            "simultaneous_candidate_collisions": "deduplicate exact economic-address collisions; distinct definition addresses remain separate and use definition/symbol actual-exit nonoverlap",
            "timestamp_order": "decision_ts<=entry_ts<=exit_ts; all timezone-aware UTC",
            "price_roles": "trade_open only for execution; mark remains a distinct signal/reference object; index and funding remain distinct"}


def funding_contract() -> dict[str, Any]:
    return {"base_pre_funding_round_trip_bps": 14, "stress_pre_funding_round_trip_bps": 32,
            "included_cost_components": "aggregate pre-funding allowance for fee, spread crossing, and slippage; not claimed as realized exchange execution",
            "funding_provenance_partitions": ["exact", "mixed", "imputed", "zero_boundary"],
            "minimum_funding_coverage_per_hypothesis_fold": .90, "minimum_campaign_weighted_coverage": .95,
            "missing_funding": "reject event before outcome and report; never relabel as zero",
            "coverage_formula": "covered required settlement boundaries / all required settlement boundaries; zero-boundary events count covered only when the complete venue boundary calendar proves no boundary in (entry_ts,exit_ts]",
            "provenance_precedence": ["exact signed venue cashflow", "frozen pre-outcome model value tagged mixed/imputed", "otherwise missing and reject"],
            "boundary_interval": "entry_ts < settlement_ts <= exit_ts",
            "signed_exact_formula_bps": "sum(-position_sign * funding_rate * boundary_notional / entry_notional * 10000); position_sign=+1 long,-1 short; positive rate means longs pay",
            "primary_imputed_boundary_bps": "min(model_signed_cashflow_bps,-32) per non-exact required boundary",
            "severe_imputed_boundary_bps": "min(model_signed_cashflow_bps,-64) per non-exact required boundary",
            "primary_selection_metric": "side_sign*10000*(exit_trade_open-entry_trade_open)/entry_trade_open - 14 + exact_signed_funding_bps + sum(primary_imputed_boundary_bps)",
            "stress_selection_sensitivity": "side_sign*10000*(exit_trade_open-entry_trade_open)/entry_trade_open - 32 + exact_signed_funding_bps + sum(severe_imputed_boundary_bps)",
            "exact_boundary_only_sensitivity": "same gross return minus 14bps using exact signed funding only; events with any non-exact required boundary are excluded",
            "zero_funding_diagnostic": "same gross return minus 14bps with funding set to zero; diagnostic only and cannot select",
            "favourable_imputation": "cannot activate, rescue, rank, or select a candidate",
            "funding_sign": "signed venue cashflow on position notional at actual settlement boundaries after entry and through exit; no boundary means zero_boundary only when complete evidence proves none occurred"}


def funding_net_bps(gross_bps: float, position_side: str, exact_boundaries: list[dict[str, float]],
                    imputed_signed_bps: list[float], mode: str = "primary") -> float:
    if position_side not in {"long", "short"} or not _finite(gross_bps):
        raise Stage16ContractError("invalid funding fixture")
    position_sign = 1.0 if position_side == "long" else -1.0
    exact = 0.0
    for row in exact_boundaries:
        rate, ratio = float(row["funding_rate"]), float(row["boundary_notional_over_entry_notional"])
        if not _finite(rate) or not _finite(ratio) or ratio <= 0:
            raise Stage16ContractError("invalid exact funding boundary")
        exact += -position_sign * rate * ratio * 10000.0
    if mode == "primary":
        cost, floor = 14.0, -32.0
    elif mode == "stress":
        cost, floor = 32.0, -64.0
    elif mode == "exact_only":
        if imputed_signed_bps:
            raise Stage16ContractError("exact-only excludes imputed boundary")
        cost, floor = 14.0, 0.0
    else:
        raise Stage16ContractError("unknown funding mode")
    if any(not _finite(value) for value in imputed_signed_bps):
        raise Stage16ContractError("missing funding rejects event")
    adverse = sum(min(float(value), floor) for value in imputed_signed_bps)
    return float(gross_bps) - cost + exact + adverse


def telegram_supervision_contract() -> dict[str, Any]:
    return {"stage16_real_message_sent": False, "later_launch": {"notifier": "existing secure notifier only", "secret_values_in_logs_archives_responses": False, "preflight_dry_run_required": True, "synthetic_heartbeat_and_stop_alert_required": True, "heartbeat_minutes": 30, "notifications": ["start", "phase", "fold", "family_stop", "global_warning", "global_stop", "completion"], "unavailable_status": "blocked_telegram_notifier_unavailable before outcomes"},
            "supervision": {"persistent_supervisor_required": True, "maximum_workers": 4, "wall_cap_seconds": 14400, "output_cap_bytes": 5368709120, "idempotent_restart": True, "atomic_state_and_artifacts": True, "family_global_stop_isolation": True, "outcome_conditioned_manual_intervention": False}}


def validate_translation_registry(registry: dict[str, Any]) -> None:
    cells = registry.get("cells")
    if not isinstance(cells, list) or len(cells) != registry.get("total_cells"):
        raise Stage16ContractError("translation cell count mismatch")
    for cell in cells:
        missing = REQUIRED_TRANSLATION_FIELDS - set(cell)
        if missing:
            raise Stage16ContractError(f"translation missing fields: {sorted(missing)}")
        expected = _address_template(cell)
        if cell["canonical_economic_address_template"] != expected:
            raise Stage16ContractError("economic-address template drift")
        if not isinstance(cell["instrument_mapping"], dict) or not isinstance(cell["side_mapping"], dict) or not cell["exit"].get("horizon"):
            raise Stage16ContractError("ambiguous instrument, side, or exit")
        allowed_horizons = {"KDA02B": {"1h", "3h", "6h"}, "KDA02C": {"1h"}, "KDX01": {"1h", "3h", "6h"}}
        if cell["exit"]["horizon"] not in allowed_horizons.get(cell["family"], set()):
            raise Stage16ContractError("unregistered horizon")
        if "exact_thresholds" not in cell["feature_contract"] and "exact_breadth_threshold" not in cell["feature_contract"]:
            raise Stage16ContractError("translation lacks exact threshold")
        if cell["family"] == "KDA02C":
            authority = cell["feature_contract"].get("base_event_authority", {})
            breadth = cell["feature_contract"].get("breadth_formula", {})
            if (authority.get("event_id_prefix") != "kda02v2_event_" or len(authority.get("event_id_payload_keys", [])) != 9 or
                    breadth.get("window_minutes") not in {5, 15, 30, 60} or
                    breadth.get("direction_filter") not in {"negative", "positive"} or
                    not str(breadth.get("direction_field", "")).startswith("parent_direction") or
                    breadth.get("identity_filter") not in {"primary_z2", "robust_pct95"} or
                    set(breadth.get("breadth_form_mapping", {})) != {"raw_share", "fold_local_rank", "isolated_vs_nonisolated"} or
                    "numerator/denominator" != breadth.get("raw_share")):
                raise Stage16ContractError("KDA02C base-event authority incomplete")
        if cell["family"] == "KDX01":
            allowed = {"downside_trade_displacement", "downside_mark_displacement", "oi_contraction", "liquidation_intensity", "negative_basis_level", "negative_basis_change", "directional_PIT_breadth", "completed_trade_mark_reclaim"}
            components = cell["feature_contract"].get("required_components", [])
            reclaim = cell["feature_contract"].get("causal_reclaim", {})
            episode = cell["feature_contract"].get("episode_identity", {})
            reclaim_fields = {"reference_timestamp", "trade_reference_field", "mark_reference_field", "lookback", "confirmation_bar", "comparison_operator", "decision_ts", "availability"}
            episode_fields = {"onset", "mandatory_component_timing", "flicker", "close", "reset", "cap_hours", "reference_binding"}
            if not components or not set(components) <= allowed or reclaim_fields - set(reclaim) or episode_fields - set(episode) or episode.get("cap_hours") != 6:
                raise Stage16ContractError("KDX primitive or reclaim contract incomplete")
            breadth = cell["feature_contract"].get("breadth_context", {})
            needs_breadth = "directional_PIT_breadth" in components
            if needs_breadth and (breadth.get("required") is not True or breadth.get("purge_identity") != "primary_z2" or breadth.get("window_minutes") != 5 or breadth.get("direction_field") != "parent_direction=-1" or breadth.get("formula") != "directional_onset_count/eligible"):
                raise Stage16ContractError("KDX breadth contract incomplete")
            if not needs_breadth and breadth.get("required") is not False:
                raise Stage16ContractError("KDX breadth enabled outside registered ladder")
    retired = registry.get("removed_from_executable_registry", {}).get("attempts", [])
    if len(retired) != 42 or len({row.get("stage14_cell_id") for row in retired}) != 42:
        raise Stage16ContractError("retired KDX attempt identities incomplete")
    if any(row.get("can_enter_translation_or_beam") is not False or row.get("status") != "non_executable_inherited_attempt" for row in retired):
        raise Stage16ContractError("retired KDX attempt can enter executable path")
    if set(row["stage14_cell_id"] for row in retired) & set(cell.get("source_stage14_cell_id") for cell in cells):
        raise Stage16ContractError("retired and executable Stage14 KDX identities overlap")


def validate_packet(packet: dict[str, Any], dependencies: dict[str, Any]) -> dict[str, bool]:
    required = {"packet_id", "status", "campaign_manifest_file_sha256", "campaign_manifest_canonical_sha256",
                "dependency_file_sha256", "dependency_canonical_sha256", "packet_payload_canonical_sha256",
                "economic_run_authorized", "external_human_approval_required"}
    if required - set(packet):
        raise Stage16ContractError("approval packet incomplete")
    if packet["economic_run_authorized"] is not False or packet["external_human_approval_required"] is not True:
        raise Stage16ContractError("self-authorization rejected")
    payload = {key: value for key, value in packet.items() if key != "packet_payload_canonical_sha256"}
    if canonical_sha256(payload) != packet["packet_payload_canonical_sha256"]:
        raise Stage16ContractError("packet canonical payload hash drift")
    if dependencies.get("semantics_complete") is not True:
        raise Stage16ContractError("packet semantics incomplete")
    return {"packet_semantics_complete": True, "campaign_engine_can_execute_without_discretion": True,
            "external_human_approval_still_required": True}


def synthetic_metrics(rows: list[dict[str, Any]], *, eligible_calendar_days: int,
                      eligible_symbols: int, eligible_interval_seconds: float,
                      complexity: int, comparison_day_returns: dict[str, float] | None = None) -> dict[str, Any]:
    required = {"event_id", "day", "utc_hour", "symbol", "year", "inner_fold", "base_net_bps", "stress_net_bps", "holding_seconds"}
    if not rows or any(required - set(row) for row in rows) or eligible_calendar_days <= 0 or eligible_symbols <= 0 or eligible_interval_seconds <= 0:
        raise Stage16ContractError("synthetic metrics require complete rows and positive denominators")
    if any(not all(_finite(row[key]) for key in ("base_net_bps", "stress_net_bps", "holding_seconds")) for row in rows):
        raise Stage16ContractError("synthetic metrics require finite rows")
    days: dict[str, list[float]] = {}; stress_days: dict[str, list[float]] = {}; hours = set(); inner_days: dict[str, dict[str, list[float]]] = {}
    contributions = {"symbol": {}, "day": {}, "year": {}}
    for row in sorted(rows, key=lambda item: (item["day"], item["event_id"])):
        base = float(row["base_net_bps"]); stress = float(row["stress_net_bps"])
        days.setdefault(row["day"], []).append(base); stress_days.setdefault(row["day"], []).append(stress)
        hours.add(row["utc_hour"]); inner_days.setdefault(row["inner_fold"], {}).setdefault(row["day"], []).append(base)
        for dimension in contributions:
            key = str(row[dimension]); contributions[dimension][key] = contributions[dimension].get(key, 0.0) + base
    day_series = {key: sum(values)/len(values) for key, values in sorted(days.items())}
    day_means = list(day_series.values())
    stress_day_means = [sum(values)/len(values) for _, values in sorted(stress_days.items())]
    inner_means = []
    for _, fold_days in sorted(inner_days.items()):
        fold_day_means = [sum(values)/len(values) for _, values in sorted(fold_days.items())]
        inner_means.append(sum(fold_day_means)/len(fold_day_means))
    aggregate = sum(day_means) / len(day_means)
    rng = random.Random(20260720)
    replicates = [sum(rng.choice(day_means) for _ in day_means) / len(day_means) for _ in range(2000)]
    cumulative = peak = max_drawdown = 0.0
    for value in day_means:
        cumulative += value; peak = max(peak, cumulative); max_drawdown = max(max_drawdown, peak - cumulative)
    shares = {}
    for dimension, values in contributions.items():
        denominator = sum(abs(value) for value in values.values())
        shares[dimension] = {key: (abs(value)/denominator if denominator else 0.0) for key, value in sorted(values.items())}
    occupancy = sum(float(row["holding_seconds"]) for row in rows) / (eligible_symbols * eligible_interval_seconds)
    if not 0 <= occupancy <= 1:
        raise Stage16ContractError("capital occupancy outside [0,1]")
    return {"accepted_trade_count": len(rows), "independent_market_day_clusters": len(days),
            "independent_utc_hour_clusters": len(hours), "aggregate_base_net_mean_bps": aggregate,
            "aggregate_stress_net_mean_bps": sum(stress_day_means)/len(stress_day_means),
            "base_net_median_bps": quantile_type7([r["base_net_bps"] for r in rows], .5),
            "median_inner_fold_base_net_mean_bps": quantile_type7(inner_means, .5),
            "p20_inner_fold_base_net_mean_bps": quantile_type7(inner_means, .2),
            "cluster_bootstrap_lower_bound_bps": quantile_type7(replicates, .05),
            "left_tail_utility_bps": -max_drawdown,
            "opportunity_frequency_per_30d": 30.0*len(rows)/eligible_calendar_days,
            "capital_occupancy": occupancy, "execution_margin_bps": sum(stress_day_means)/len(stress_day_means),
            "symbol_day_year_contribution": shares, "complexity": complexity,
            "candidate_return_correlation": 0.0 if comparison_day_returns is None else aligned_day_correlation(day_series, comparison_day_returns),
            "market_day_returns": day_series}


def synthetic_canary() -> dict[str, Any]:
    spy = OutcomeReadSpy()
    safe = spy.read("synthetic/causal_features", ["event_id", "feature_available_ts", "oi_log_change_1h"],
                    [{"event_id": "e1", "feature_available_ts": "2024-01-01T00:00:00Z", "oi_log_change_1h": -.02}])
    edges = collapse_edges([quantile_type7(range(10), p) for p in (0, .2, .4, .6, .8, 1)])
    bins = [assign_right_closed_bin(value, edges) for value in (0, 2, 4, 6, 8, 9)]
    inner = build_inner_folds("2023-04-01T00:00:00Z", "2023-09-30T18:00:00Z")
    registry = build_translation_registry(); validate_translation_registry(registry)
    metric_rows = [{"event_id": f"e{i}", "day": f"2024-01-{i + 1:02d}", "utc_hour": f"2024-01-{i + 1:02d}T{i%24:02d}",
                    "symbol": ("PF_XBTUSD", "PF_ETHUSD", "PF_SOLUSD")[i % 3], "year": "2024",
                    "inner_fold": f"M_{1+i%3}", "base_net_bps": float((i % 7)-2),
                    "stress_net_bps": float((i % 7)-4), "holding_seconds": 3600.0} for i in range(30)]
    metrics = synthetic_metrics(metric_rows, eligible_calendar_days=30, eligible_symbols=3,
                                eligible_interval_seconds=30*86400, complexity=4)
    candidates = []
    series = [{"d1": 1, "d2": 2, "d3": 3, "d4": 4}, {"d1": -1, "d2": 1, "d3": -1, "d4": 1}, {"d1": 2, "d2": 4, "d3": 6, "d4": 8}]
    for index, cell in enumerate(registry["cells"][:3]):
        candidates.append({"canonical_translation_id": cell["canonical_translation_id"], "aggregate_base_net_mean_bps": 3-index,
                           "median_inner_fold_base_net_mean_bps": 3-index, "p20_inner_fold_base_net_mean_bps": 3-index,
                           "cluster_bootstrap_lower_bound_bps": 3-index, "left_tail_utility_bps": -index,
                           "opportunity_frequency_per_30d": 1+index, "execution_margin_bps": 3-index,
                           "complexity": 3-index, "market_day_returns": series[index], "integrity_pass": True,
                           "accepted_trade_count": 30, "independent_market_day_clusters": 20,
                           "independent_utc_hour_clusters": 20})
    beam = deterministic_beam(candidates)
    template = registry["cells"][0]["canonical_economic_address_template"]
    address = economic_address(template, "SYNTHETIC_EVENT_1", "PF_XBTUSD", "long", "2024-01-01T00:00:00Z", "F1")
    execution = resolve_fixed_execution("2024-01-01T00:00:00Z", "1h",
                                        ["2024-01-01T00:05:00Z", "2024-01-01T01:05:00Z"],
                                        "2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z")
    validate_episode_duration("2023-12-31T12:00:00Z", "2023-12-31T18:00:00Z")
    funding = {"long_exact": funding_net_bps(20, "long", [{"funding_rate": .001, "boundary_notional_over_entry_notional": 1}], []),
               "short_exact": funding_net_bps(20, "short", [{"funding_rate": .001, "boundary_notional_over_entry_notional": 1}], []),
               "primary_favourable_imputation_capped_adverse": funding_net_bps(20, "long", [], [10]),
               "stress_imputation": funding_net_bps(20, "long", [], [-10], mode="stress"),
               "zero_boundary": funding_net_bps(20, "long", [], [])}
    breadth_fixture = directional_breadth([
        {"base_event_id": "b1", "decision_source_ts": "2024-01-01T00:25:00Z", "parent_direction": -1, "purge_identity": "primary_z2"}],
        "2024-01-01T00:30:00Z", 15, "negative", "primary_z2", 100)
    kdx_trace = kdx_episode_trace([
        {"source_close_ts": "2024-01-01T00:00:00Z", "price": True, "oi": False, "completed_trade_mark_reclaim": False},
        {"source_close_ts": "2024-01-01T00:05:00Z", "price": True, "oi": True, "completed_trade_mark_reclaim": False},
        {"source_close_ts": "2024-01-01T00:10:00Z", "price": False, "oi": True, "completed_trade_mark_reclaim": False},
        {"source_close_ts": "2024-01-01T00:15:00Z", "price": True, "oi": True, "completed_trade_mark_reclaim": True}], ["price", "oi"])
    try:
        spy.read("synthetic/forward_returns", ["forward_return"], [])
        firewall = False
    except Stage16ContractError:
        firewall = True
    return {"safe_rows": len(safe), "read_spy_forward_rejected": firewall, "read_count": len(spy.reads),
            "bins": bins, "inner_fold_count": len(inner), "registered_cells": registry["total_cells"],
            "metric_keys": sorted(metrics), "metrics_complete": set(metric_contract()["metrics"]) <= set(metrics),
            "metric_scalar_values_finite": all(_finite(metrics[key]) for key in metric_contract()["metrics"] if key != "symbol_day_year_contribution"),
            "contribution_shares_sum_to_one": all(abs(sum(values.values())-1.0)<1e-12 for values in metrics["symbol_day_year_contribution"].values()),
            "beam_ids": [x["canonical_translation_id"] for x in beam], "beam_tags": [x.get("tags", []) for x in beam],
            "KDA02B_sides": [side_for_kda02b("continuation", 1, 1), side_for_kda02b("reversal", 1, 1), side_for_kda02b("continuation", 1, -1)],
            "KDA02C_identity": {"symbol": "PF_SOLUSD", "side": side_for_kda02c("negative")},
            "KDX01_identity": {"symbol": "PF_ETHUSD", "side": "long", "completion": "trade_and_mark_reclaim"},
            "execution": execution, "funding_fixture": funding, "breadth_fixture": breadth_fixture,
            "kdx_episode_trace": kdx_trace, "economic_address": address, "external_approval_required": True,
            "economic_outputs_computed": False}
