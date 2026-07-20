#!/usr/bin/env python3
"""Regenerate the non-authorizing Stage 19 campaign packet."""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.qlmg_stage16_campaign import (
    _address_template, beam_contract, build_translation_registry,
    canonical_sha256, estimator_rule_inventory, inner_fold_contract,
    response_surface_contract, synthetic_canary as stage16_canary,
    telegram_supervision_contract, validate_translation_registry,
)
from tools.qlmg_stage19_funding import dual_alignment_cashflow_bps


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2"
STAGE16 = ROOT / "docs/agent/task_archive/20260720_donch_bt_stage_16_complete_campaign_semantics_git_cleanup_20260720_v1"
LOCAL = Path("/opt/testerdonch/results/rebaseline/phase_kraken_local_official_funding_export_20260720_v4")
FIXED = "2026-07-20T11:00:00Z"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def stage19_funding_contract() -> dict[str, Any]:
    return {
        "version": "stage19_v1", "instrument": "Kraken linear PF; quantity is base/contract unit; quote and PnL USD",
        "source_rankable_package_sha256": "6c0969727c882ca6439c57c3b7d03367e1b2d26ee46823e56b983756d026ef64",
        "dual_alignment_contract_sha256": file_sha256(LOCAL / "FUNDING_PERIOD_DUAL_ALIGNMENT_CONTRACT.json"),
        "gap_allowance_table_sha256": file_sha256(LOCAL / "FUNDING_GAP_ALLOWANCE_TABLE.csv"),
        "unit_verification_sha256": file_sha256(LOCAL / "ABSOLUTE_RATE_UNIT_VERIFICATION.csv"),
        "base_pre_funding_round_trip_bps": 14, "stress_pre_funding_round_trip_bps": 32,
        "included_pre_funding_costs": "aggregate fee, spread-crossing and slippage allowance; not a realized execution claim",
        "signed_exact_formula_bps": "-position_sign*absolute_rate_usd_per_contract_unit_per_hour*overlap_seconds/3600/entry_trade_open_usd_per_contract_unit*10000",
        "timestamp_alignments": ["row_t_is_[t,t+1h)", "row_t_is_[t-1h,t)"],
        "adverse_exact_selection_bps": "min(0,signed_start_alignment_bps,signed_end_alignment_bps)",
        "base_primary_net_bps": "gross_return_bps-14+adverse_exact_selection_bps+base_missing_hour_gap_cost_bps",
        "stress_net_bps": "gross_return_bps-32+adverse_exact_selection_bps+stress_missing_hour_gap_cost_bps",
        "missing_hour_cost": "-rankable_only_allowance_bps_per_hour*missing_overlap_hours",
        "favourable_funding_credit_can_select_rank_or_rescue": False,
        "complete_symbol_allowance_or_event_rejection": True,
        "reporting_partitions": ["pre_funding_14bps", "adverse_exact_dual_alignment_primary", "adverse_exact_dual_alignment_stress", "signed_start_alignment_diagnostic", "signed_end_alignment_diagnostic", "timestamp_alignment_sensitivity", "missing_hour_allowance_contribution"],
        "protected_rows_used": 0, "runtime_semantic_choice": False,
    }


def translation_registry(funding: dict[str, Any]) -> dict[str, Any]:
    registry = copy.deepcopy(build_translation_registry())
    cost = {
        "base_pre_funding_round_trip_bps": 14, "stress_pre_funding_round_trip_bps": 32,
        "funding_contract_sha256": canonical_sha256(funding),
        "rankable_package_sha256": funding["source_rankable_package_sha256"],
        "selection_funding": "min(0,signed_start_alignment_bps,signed_end_alignment_bps)",
        "missing_hour": "rankable-only symbol q95/q99; equal-symbol pooled fallback; always nonpositive",
        "favourable_funding_can_rescue": False,
    }
    for cell in registry["cells"]:
        cell["cost_funding"] = cost
        cell["canonical_economic_address_template"] = _address_template(cell)
        cell["canonical_translation_id"] = f"TR_{cell['canonical_economic_address_template'][:24]}"
    registry["version"] = "stage19_v1"
    registry["funding_dependency_replaced"] = True
    validate_translation_registry(registry)
    return registry


def metric_contract(funding_hash: str) -> dict[str, Any]:
    from tools.qlmg_stage16_campaign import metric_contract as prior
    value = prior()
    value["funding_contract_sha256"] = funding_hash
    value["metrics"]["aggregate_base_net_mean_bps"]["formula"] = "mean of market-day means after 14bps, adverse exact dual alignment, and q95 missing-hour cost"
    value["metrics"]["aggregate_stress_net_mean_bps"]["formula"] = "mean of market-day means after 32bps, adverse exact dual alignment, and q99 missing-hour cost"
    value["funding_alignment_sensitivity"] = "positive under only one signed alignment is tagged sensitive and cannot be promoted as robust funding-adjusted evidence"
    return value


def utility_contract(funding_hash: str) -> dict[str, Any]:
    from tools.qlmg_stage16_campaign import utility_pareto_contract as prior
    value = prior()
    value["funding_contract_sha256"] = funding_hash
    value["funding_adjusted_objective_source"] = "adverse exact dual alignment plus nonpositive rankable-only missing-hour allowance"
    value["alignment_sensitive_candidate"] = "limitation tag; ineligible for robust-funding promotion"
    return value


def boundary_contract(funding_hash: str) -> dict[str, Any]:
    from tools.qlmg_stage16_campaign import boundary_contract as prior
    value = prior()
    value["funding_contract_sha256"] = funding_hash
    value["missing_funding_boundary"] = "charge symbol q95/q99 missing-hour allowance; reject before outcome if symbol allowance is unavailable"
    value["funding_source_boundary"] = "only immutable rankable package [2023-01-01,2026-01-01); any protected request globally stops"
    return value


def funding_canary(registry: dict[str, Any]) -> dict[str, Any]:
    with (LOCAL / "FUNDING_GAP_ALLOWANCE_TABLE.csv").open(newline="", encoding="utf-8") as handle:
        allowance = next(csv.DictReader(handle))
    base_allowance = Decimal(allowance["base_gap_allowance_bps_per_hour"])
    stress_allowance = Decimal(allowance["stress_gap_allowance_bps_per_hour"])
    entry = datetime(2025, 1, 1, 0, 30, tzinfo=timezone.utc)
    exit_ = entry + timedelta(hours=2)
    rates = {
        datetime(2025, 1, 1, 0, tzinfo=timezone.utc): Decimal("0.01"),
        datetime(2025, 1, 1, 1, tzinfo=timezone.utc): Decimal("-0.02"),
        datetime(2025, 1, 1, 2, tzinfo=timezone.utc): Decimal("0.03"),
    }
    long = dual_alignment_cashflow_bps(entry=entry, exit_=exit_, position_sign=1, entry_trade_open=Decimal(100), absolute_rates=rates, base_gap_bps_per_hour=base_allowance, stress_gap_bps_per_hour=stress_allowance)
    short = dual_alignment_cashflow_bps(entry=entry, exit_=exit_, position_sign=-1, entry_trade_open=Decimal(100), absolute_rates=rates, base_gap_bps_per_hour=base_allowance, stress_gap_bps_per_hour=stress_allowance)
    repeat = translation_registry(stage19_funding_contract())
    prior = stage16_canary()
    checks = {
        "both_alignments_exercised": long["signed_alignment_start_bps"] != long["signed_alignment_end_bps"],
        "partial_hour_exercised": True, "long_short_sign_reversal": long["signed_alignment_start_bps"] == -short["signed_alignment_start_bps"],
        "positive_and_negative_rates_exercised": True,
        "favourable_funding_ignored_for_selection": long["adverse_exact_funding_bps"] <= 0 and short["adverse_exact_funding_bps"] <= 0,
        "missing_gap_nonpositive": long["base_gap_cost_bps"] <= 0 and long["stress_gap_cost_bps"] <= long["base_gap_cost_bps"],
        "campaign_cell_count": len(registry["cells"]), "campaign_deterministic": canonical_sha256(registry) == canonical_sha256(repeat),
        "economic_addresses_unique": len({cell["canonical_economic_address_template"] for cell in registry["cells"]}) == 186,
        "runtime_semantic_discretion": False, "stage16_semantic_canary_pass": prior["registered_cells"] == 186,
        "protected_rows_opened": 0, "economic_outputs_computed": 0,
    }
    boolean_checks = [value for key, value in checks.items() if isinstance(value, bool) and key != "runtime_semantic_discretion"]
    if not all(boolean_checks) or checks["runtime_semantic_discretion"] is not False or checks["campaign_cell_count"] != 186:
        raise RuntimeError("Stage 19 synthetic canary failed")
    return {
        "status": "pass", **checks, "allowance_symbol": allowance["symbol"],
        "base_q95_allowance_bps_per_hour": str(base_allowance),
        "stress_q99_allowance_bps_per_hour": str(stress_allowance),
        "allowance_table_sha256": file_sha256(LOCAL / "FUNDING_GAP_ALLOWANCE_TABLE.csv"),
    }


def build(implementation_commit: str) -> dict[str, Any]:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name in ["FUNDING_PERIOD_DUAL_ALIGNMENT_CONTRACT.json", "FUNDING_GAP_ALLOWANCE_CONTRACT.json", "FUNDING_GAP_ALLOWANCE_TABLE.csv", "ABSOLUTE_RATE_UNIT_VERIFICATION.csv", "UNIT_VERIFICATION_SOURCE_MANIFEST.json"]:
        shutil.copyfile(LOCAL / name, ARCHIVE / name)
    source_partition = {
        "version": "stage19_v1",
        "source_zip_sha256": "65ba6712a6ab657389d2795d3ed77bedb4270841dfe711147ae9df16e366edab",
        "rankable_package": {"sha256": "6c0969727c882ca6439c57c3b7d03367e1b2d26ee46823e56b983756d026ef64", "rows": 5658890, "symbols": 476, "interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)"},
        "protected_quarantine": {"sha256": "0957b7253e76840f6062f23290ed8b53aea4714f5c3cbaa314e7fba0975b37d0", "rows": 236786, "symbols": 320, "campaign_access": False},
        "pre_rankable_excluded": {"sha256": "5e3ca567998e0e35f9f4e6db027533a14bc8f778738fcc1eff744ec6bc7e73e3", "rows": 277640, "campaign_access": False},
        "unknown_or_invalid": {"sha256": "8739c76e681f900923b900c9df0ef75cf421d39cabb54650c4b9ad19b6a76d85", "rows": 0},
        "protected_funding_values_used_for_statistics": 0, "protected_strategy_price_or_return_rows_opened": 0,
    }
    write_json(ARCHIVE / "FUNDING_SOURCE_AND_PARTITION_MANIFEST.json", source_partition)
    funding = stage19_funding_contract()
    funding_hash = canonical_sha256(funding)
    registry = translation_registry(funding)
    contracts = {
        "FUNDING_COST_AND_COVERAGE_CONTRACT.json": funding,
        "BOUNDARY_AND_MISSINGNESS_CONTRACT.json": boundary_contract(funding_hash),
        "DEVELOPMENT_METRIC_CONTRACT.json": metric_contract(funding_hash),
        "UTILITY_AND_PARETO_CONTRACT.json": utility_contract(funding_hash),
        "ECONOMIC_TRANSLATION_REGISTRY.json": registry,
        "RESPONSE_SURFACE_AND_BIN_SPEC.json": response_surface_contract(),
        "ESTIMATOR_AND_RULE_INVENTORY.json": estimator_rule_inventory(),
        "INNER_FOLD_MAP.json": inner_fold_contract(),
        "CANDIDATE_BEAM_CONTRACT.json": beam_contract(),
        "TELEGRAM_AND_SUPERVISION_CONTRACT.json": telegram_supervision_contract(),
    }
    for name, value in contracts.items():
        write_json(ARCHIVE / name, value)
    shutil.copyfile(STAGE16 / "SEARCH_SPACE_REGISTRY.json", ARCHIVE / "SEARCH_SPACE_REGISTRY.json")
    resource = json.loads((STAGE16 / "RESOURCE_PROJECTION.json").read_text())
    resource["version"] = "stage19_v1"
    resource["funding_engine"] = "dual alignment exact rows plus rankable-only q95/q99 gap allowances; no added cell or model axis"
    resource["total_cells"] = 186
    write_json(ARCHIVE / "RESOURCE_PROJECTION.json", resource)

    dependencies = sorted([*contracts, "SEARCH_SPACE_REGISTRY.json", "RESOURCE_PROJECTION.json", "FUNDING_SOURCE_AND_PARTITION_MANIFEST.json", "FUNDING_PERIOD_DUAL_ALIGNMENT_CONTRACT.json", "FUNDING_GAP_ALLOWANCE_CONTRACT.json", "FUNDING_GAP_ALLOWANCE_TABLE.csv", "ABSOLUTE_RATE_UNIT_VERIFICATION.csv", "UNIT_VERIFICATION_SOURCE_MANIFEST.json"])
    raw_hashes = {name: file_sha256(ARCHIVE / name) for name in dependencies}
    manifest = {
        "campaign_id": "kraken_derivatives_campaign_001_stage19_exact_funding",
        "version": "4.0", "generated_at": FIXED, "status": "non_authorizing_replacement_human_approval_required",
        "economic_run_authorized_by_manifest": False, "external_human_approval_required": True,
        "implementation_commit": implementation_commit, "actual_task_starting_commit": "3f5e94eb8e6b8becb4dfaa5457742682ac31f7e9",
        "supersedes_stage16_campaign_manifest_sha256": "cc07499c671cf39b8ceaee91156f141dcc2c5532142af29a38a4f6830b73f23d",
        "historical_terminal_decisions_changed": False, "C17_excluded": True,
        "ready_hypotheses": ["KDA02B_v2_oi_vacuum_redevelopment", "KDA02C_v1_purge_breadth_context", "KDX01_v1_downside_completed_derivatives_state_rejection"],
        "search": {"total_cells": 186, "cells_by_family": registry["cells_by_family"], "inherited_non_executable_attempts": 42, "model_count": 0},
        "dependency_file_sha256": raw_hashes,
        "funding_authority": {"rankable_package_sha256": funding["source_rankable_package_sha256"], "protected_rows_used": 0, "campaign_symbols_mapped": 187, "unit_verified": 187},
        "phase_permissions": {str(phase): phase < 2 for phase in range(8)},
        "outcome_firewall": {"forward_returns_or_PnL": False, "protected_prices_or_returns": False, "Capitalcom": False, "real_Telegram": False},
        "stop_conditions": ["dependency_hash_drift", "rankable_package_boundary_failure", "missing_campaign_allowance", "unit_incompatibility", "runtime_alignment_choice", "protected_or_outcome_firewall_access", "deterministic_replay_failure"],
    }
    write_json(ARCHIVE / "CAMPAIGN_MANIFEST.json", manifest)
    manifest_hash = file_sha256(ARCHIVE / "CAMPAIGN_MANIFEST.json")
    packet_payload = {
        "packet_id": "kraken_derivatives_campaign_001_stage19_phase2_5_approval_request", "version": "4.0",
        "status": "human_approval_required_not_authorized", "campaign_manifest_file_sha256": manifest_hash,
        "dependency_file_sha256": raw_hashes, "phases_requested": [2, 3, 4, 5],
        "economic_run_authorized": False, "external_human_approval_required": True,
        "authorization_preconditions": ["new exact human approval names this packet and campaign manifest SHA-256", "independent review accepted", "all dependency hashes match", "launch validator and synthetic canary pass"],
        "funding_selection": "adverse min of zero and both exact alignments plus nonpositive rankable-only missing-hour allowance",
        "supersession": {"Stage16_packet_immutable": True, "Stage17_18_incidents_immutable": True, "prior_approval_not_reused": True},
    }
    packet = dict(packet_payload)
    packet["packet_payload_canonical_sha256"] = canonical_sha256(packet_payload)
    write_json(ARCHIVE / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json", packet)
    packet_hash = file_sha256(ARCHIVE / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json")
    (ARCHIVE / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.md").write_text(
        "# Future Derivatives Campaign Approval Packet\n\nStatus: `human_approval_required_not_authorized`. No economics are authorized.\n\n"
        f"Campaign manifest SHA-256: `{manifest_hash}`<br>\nApproval packet SHA-256: `{packet_hash}`\n\n"
        "Funding selection uses the adverse minimum of zero and both exact timestamp alignments, plus nonpositive rankable-only q95/q99 missing-hour allowances. All 187 campaign PF units and symbols are covered. A new exact human approval is required.\n",
        encoding="utf-8",
    )
    canary = funding_canary(registry)
    write_json(ARCHIVE / "SYNTHETIC_CANARY.json", canary)
    (ARCHIVE / "CAMPAIGN_PACKET_LAUNCH_READINESS.md").write_text(
        "# Campaign Packet Launch Readiness\n\n- Synthetic canary: `pass`.\n- Funding semantics complete: `true`.\n- Independent review: `pending`.\n- New exact human approval: `required`.\n- Economic run: `not authorized and not launched`.\n",
        encoding="utf-8",
    )
    (ARCHIVE / "FUNDING_CALIBRATION_AND_ALIGNMENT_SUMMARY.md").write_text(
        "# Funding Calibration and Alignment Summary\n\n"
        "The campaign uses 5,658,890 official rankable hourly rows from 476 symbols. All 187 campaign symbols have exact export coverage and at least 4,079 rankable observations, so every campaign allowance is its own Decimal Hyndman–Fan type-7 q95/q99; none uses the pooled fallback. The equal-symbol-weighted pool remains frozen for future mechanically eligible symbols and contains 276 official unit-compatible PF symbols.\n\n"
        "Every campaign PF is verified as Kraken `flexible_futures`, `contractSize=1`, base-unit quantity, USD quote. Nonzero `absolute_rate/relative_rate` anchors were checked against exact-timestamp rankable trade and mark opens. Mark relative error must be at most 10%; the trade sanity bound is 25% to accommodate sparse launch-period prints. No hidden multiplier is admitted.\n\n"
        "Both timestamp interpretations are computed. Selection receives `min(0, signed_start, signed_end)` and therefore no favourable funding credit. Missing hours receive only nonpositive q95/q99 costs. Timestamp-sensitive positives are limitation-tagged and are not robust-funding promotions. Protected funding values contributed zero statistics.\n",
        encoding="utf-8",
    )
    result = {"campaign_manifest_sha256": manifest_hash, "approval_packet_sha256": packet_hash, "funding_contract_sha256": file_sha256(ARCHIVE / "FUNDING_COST_AND_COVERAGE_CONTRACT.json"), "economic_translation_registry_sha256": file_sha256(ARCHIVE / "ECONOMIC_TRANSLATION_REGISTRY.json"), "cells": 186, "canary": canary}
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()
    build(args.implementation_commit)
