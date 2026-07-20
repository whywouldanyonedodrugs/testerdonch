#!/usr/bin/env python3
"""Fail-closed launch-readiness validation for the Stage 16 packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.qlmg_stage16_campaign import (
    OBJECTIVES, Stage16ContractError, canonical_sha256, file_sha256, parse_utc, synthetic_canary,
    validate_packet, validate_translation_registry,
)


REQUIRED_FILES = [
    "RESPONSE_SURFACE_AND_BIN_SPEC.json", "ESTIMATOR_AND_RULE_INVENTORY.json",
    "INNER_FOLD_MAP.json", "DEVELOPMENT_METRIC_CONTRACT.json",
    "UTILITY_AND_PARETO_CONTRACT.json", "CANDIDATE_BEAM_CONTRACT.json",
    "ECONOMIC_TRANSLATION_REGISTRY.json", "BOUNDARY_AND_MISSINGNESS_CONTRACT.json",
    "FUNDING_COST_AND_COVERAGE_CONTRACT.json", "TELEGRAM_AND_SUPERVISION_CONTRACT.json",
    "SEARCH_SPACE_REGISTRY.json", "RESOURCE_PROJECTION.json",
]


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage16ContractError(f"cannot load {path.name}") from exc


def validate(root: Path) -> dict:
    for name in REQUIRED_FILES + ["CAMPAIGN_MANIFEST.json", "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json"]:
        if not (root / name).is_file():
            raise Stage16ContractError(f"missing required file: {name}")
    manifest = load(root / "CAMPAIGN_MANIFEST.json")
    packet = load(root / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json")
    bins = load(root / "RESPONSE_SURFACE_AND_BIN_SPEC.json")
    if not bins.get("all_bin_formulas_serialized") or not bins.get("axes"):
        raise Stage16ContractError("absent response bins")
    axis_fields = {"axis_id", "source_feature", "native_unit", "causal_availability", "transformation",
                   "sign_direction_treatment", "fold_local_normalization", "edges_or_probabilities",
                   "tie_handling", "duplicate_edge_handling", "missingness_rule", "allowed_interaction_partners"}
    if any(axis_fields - set(axis) for axis in bins["axes"]):
        raise Stage16ContractError("incomplete feature axis")
    inventory = load(root / "ESTIMATOR_AND_RULE_INVENTORY.json")
    if "model_inventory" not in inventory or inventory.get("maximum_model_count") != len(inventory["model_inventory"]):
        raise Stage16ContractError("unspecified estimator inventory")
    if set(inventory.get("rule_grammar", {})) != {"KDA02B", "KDA02C", "KDX01"}:
        raise Stage16ContractError("rule inventory incomplete")
    folds = load(root / "INNER_FOLD_MAP.json")
    if len(folds.get("outer_folds", [])) != 27 or any(len(row.get("inner_folds", [])) < 3 for row in folds["outer_folds"]):
        raise Stage16ContractError("ambiguous inner folds")
    for outer in folds["outer_folds"]:
        dev_end, evaluation_start, evaluation_end = map(parse_utc, (outer["development_end_exclusive"], outer["outer_evaluation_start"], outer["outer_evaluation_end_exclusive"]))
        if not dev_end < evaluation_start < evaluation_end or outer.get("purge_interval_hours") < outer.get("maximum_label_horizon_hours") + outer.get("maximum_episode_duration_for_overlap_hours") or outer.get("embargo_interval_hours") < 12:
            raise Stage16ContractError("outer fold chronology or purge arithmetic invalid")
        if (evaluation_start - dev_end).total_seconds() < outer["purge_interval_hours"] * 3600:
            raise Stage16ContractError("outer development gap shorter than purge")
        for inner in outer["inner_folds"]:
            train_exit, validation_start, validation_end, embargo_end = map(parse_utc, (inner["training_latest_exit_exclusive"], inner["validation_start"], inner["validation_end_exclusive"], inner["embargo_end"]))
            if not train_exit < validation_start < validation_end < embargo_end:
                raise Stage16ContractError("inner fold chronology invalid")
            if inner.get("purge_embargo_hours") < inner.get("maximum_horizon_hours") + inner.get("maximum_episode_overlap_hours"):
                raise Stage16ContractError("inner purge shorter than horizon plus overlap")
    metrics = load(root / "DEVELOPMENT_METRIC_CONTRACT.json")
    expected_metrics = {"accepted_trade_count", "independent_market_day_clusters", "independent_utc_hour_clusters", "aggregate_base_net_mean_bps", "aggregate_stress_net_mean_bps", "base_net_median_bps", "median_inner_fold_base_net_mean_bps", "p20_inner_fold_base_net_mean_bps", "cluster_bootstrap_lower_bound_bps", "left_tail_utility_bps", "opportunity_frequency_per_30d", "capital_occupancy", "execution_margin_bps", "symbol_day_year_contribution", "complexity", "candidate_return_correlation"}
    if set(metrics.get("metrics", {})) != expected_metrics or any(not row.get("unit") or not row.get("formula") or not row.get("direction") for row in metrics["metrics"].values()):
        raise Stage16ContractError("development metric inventory/formula incomplete")
    eligibility = metrics.get("eligibility", {})
    if eligibility.get("integrity_required") is not True or eligibility.get("minimum_accepted_trades") != 30 or eligibility.get("minimum_market_day_clusters") != 20 or eligibility.get("minimum_utc_hour_clusters") != 20 or eligibility.get("aggregate_development_base_net_mean_bps") != ">0":
        raise Stage16ContractError("candidate eligibility incomplete")
    pareto = load(root / "UTILITY_AND_PARETO_CONTRACT.json")
    if [(row.get("objective"), row.get("direction")) for row in pareto.get("objectives", [])] != OBJECTIVES:
        raise Stage16ContractError("unspecified objective direction")
    if "no worse" not in pareto.get("dominance", "") or not pareto.get("finite_values_required_for_selection"):
        raise Stage16ContractError("ambiguous Pareto handling")
    beam = load(root / "CANDIDATE_BEAM_CONTRACT.json")
    if beam.get("manual_choice_allowed") is not False or len(beam.get("lexicographic_tie_break", [])) != 9:
        raise Stage16ContractError("ambiguous deterministic beam")
    registry = load(root / "ECONOMIC_TRANSLATION_REGISTRY.json")
    validate_translation_registry(registry)
    if not any(cell["family"] == "KDA02C" and "underlying frozen completed-purge" in cell["instrument_mapping"]["rule"] for cell in registry["cells"]):
        raise Stage16ContractError("undefined KDA02C identity")
    funding = load(root / "FUNDING_COST_AND_COVERAGE_CONTRACT.json")
    funding_required = {"coverage_formula", "provenance_precedence", "boundary_interval", "signed_exact_formula_bps", "primary_imputed_boundary_bps", "severe_imputed_boundary_bps", "primary_selection_metric", "stress_selection_sensitivity", "exact_boundary_only_sensitivity", "zero_funding_diagnostic"}
    if funding_required - set(funding) or funding.get("base_pre_funding_round_trip_bps") != 14 or funding.get("stress_pre_funding_round_trip_bps") != 32 or funding.get("minimum_funding_coverage_per_hypothesis_fold") != .90 or funding.get("minimum_campaign_weighted_coverage") != .95:
        raise Stage16ContractError("funding arithmetic or coverage incomplete")
    telegram = load(root / "TELEGRAM_AND_SUPERVISION_CONTRACT.json")
    later, supervision = telegram.get("later_launch", {}), telegram.get("supervision", {})
    if telegram.get("stage16_real_message_sent") is not False or later.get("heartbeat_minutes") != 30 or later.get("unavailable_status") != "blocked_telegram_notifier_unavailable before outcomes" or later.get("preflight_dry_run_required") is not True or supervision.get("maximum_workers") != 4 or supervision.get("wall_cap_seconds") != 14400 or supervision.get("output_cap_bytes") != 5368709120 or supervision.get("persistent_supervisor_required") is not True:
        raise Stage16ContractError("Telegram or supervision contract incomplete")
    resource = load(root / "RESOURCE_PROJECTION.json")
    if any(not isinstance(resource.get(key), int) or resource[key] <= 0 for key in ("worker_count", "wall_seconds", "max_disk_bytes", "max_memory_bytes")) or resource["worker_count"] > 4:
        raise Stage16ContractError("resource budget incomplete")
    search = load(root / "SEARCH_SPACE_REGISTRY.json")
    retired = search.get("non_executable_inherited_attempts", [])
    if len(retired) != 42 or search.get("maximum_total_cells") != 186 or len(search.get("registered_cell_ids", [])) != 186 or 186 + len(retired) != 228:
        raise Stage16ContractError("search/multiplicity reconciliation invalid")
    for name in REQUIRED_FILES:
        value = load(root / name)
        if manifest["dependency_file_sha256"].get(name) != file_sha256(root / name):
            raise Stage16ContractError(f"manifest raw hash drift: {name}")
        if manifest["dependency_canonical_sha256"].get(name) != canonical_sha256(value):
            raise Stage16ContractError(f"manifest canonical hash drift: {name}")
        if packet["dependency_file_sha256"].get(name) != file_sha256(root / name):
            raise Stage16ContractError(f"packet raw hash drift: {name}")
        if packet["dependency_canonical_sha256"].get(name) != canonical_sha256(value):
            raise Stage16ContractError(f"packet canonical hash drift: {name}")
    if packet["campaign_manifest_file_sha256"] != file_sha256(root / "CAMPAIGN_MANIFEST.json") or packet["campaign_manifest_canonical_sha256"] != canonical_sha256(manifest):
        raise Stage16ContractError("packet/manifest hash drift")
    canary = synthetic_canary()
    if (not canary["read_spy_forward_rejected"] or canary["registered_cells"] != 186 or
            not canary["external_approval_required"] or not canary["metrics_complete"] or
            not canary["metric_scalar_values_finite"] or not canary["contribution_shares_sum_to_one"] or
            len(canary["beam_ids"]) < 3 or "candidate_beam_high_correlation" not in sum(canary["beam_tags"], [])):
        raise Stage16ContractError("synthetic canary failed")
    semantics_complete = True  # reached only after every independent semantic check above passes
    readiness = validate_packet(packet, {"semantics_complete": semantics_complete})
    return {**readiness, "synthetic_canary": "pass", "registered_cells": 186,
            "economic_outputs_computed": False, "protected_rows_opened": 0,
            "Capitalcom_payload_opened": False, "real_Telegram_messages": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), sort_keys=True))
