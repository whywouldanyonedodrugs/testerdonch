#!/usr/bin/env python3
"""Independent fail-closed technical launch validator for Stage 19."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.qlmg_stage16_campaign import (
    _address_template, build_translation_registry, canonical_sha256,
    validate_translation_registry,
)
from tools.qlmg_stage19_funding import Stage19FundingEngine


STAGE16 = Path(__file__).resolve().parents[1] / "docs/agent/task_archive/20260720_donch_bt_stage_16_complete_campaign_semantics_git_cleanup_20260720_v1"
UNCHANGED = [
    "RESPONSE_SURFACE_AND_BIN_SPEC.json", "ESTIMATOR_AND_RULE_INVENTORY.json",
    "INNER_FOLD_MAP.json", "CANDIDATE_BEAM_CONTRACT.json",
    "TELEGRAM_AND_SUPERVISION_CONTRACT.json", "SEARCH_SPACE_REGISTRY.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"cannot load JSON authority: {path.name}") from exc


def validate(root: Path) -> dict:
    manifest = load(root / "CAMPAIGN_MANIFEST.json")
    packet = load(root / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json")
    dependencies = manifest.get("dependency_file_sha256", {})
    if not dependencies or dependencies != packet.get("dependency_file_sha256"):
        raise RuntimeError("manifest/packet dependency inventory mismatch")
    for name, digest in dependencies.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"dependency hash drift: {name}")
    if packet.get("campaign_manifest_file_sha256") != sha256_file(root / "CAMPAIGN_MANIFEST.json"):
        raise RuntimeError("packet campaign-manifest hash drift")
    payload = {key: value for key, value in packet.items() if key != "packet_payload_canonical_sha256"}
    if canonical_sha256(payload) != packet.get("packet_payload_canonical_sha256"):
        raise RuntimeError("packet canonical payload drift")
    if packet.get("economic_run_authorized") is not False or packet.get("external_human_approval_required") is not True:
        raise RuntimeError("packet self-authorization")
    if (manifest.get("economic_run_authorized_by_manifest") is not False
            or manifest.get("external_human_approval_required") is not True
            or manifest.get("status") != "non_authorizing_replacement_human_approval_required"
            or packet.get("status") != "human_approval_required_not_authorized"
            or packet.get("phases_requested") != [2, 3, 4, 5]):
        raise RuntimeError("manifest or packet authorization state invalid")
    permissions = manifest.get("phase_permissions", {})
    if permissions != {str(phase): phase < 2 for phase in range(8)}:
        raise RuntimeError("campaign phase permissions invalid")
    implementation_commit = manifest.get("implementation_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise RuntimeError("implementation commit is not a full SHA-1")
    resolved = subprocess.run(
        ["git", "cat-file", "-e", f"{implementation_commit}^{{commit}}"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True,
    )
    if resolved.returncode:
        raise RuntimeError("implementation commit is not locally resolvable")

    for name in UNCHANGED:
        if sha256_file(root / name) != sha256_file(STAGE16 / name):
            raise RuntimeError(f"non-funding campaign dependency changed: {name}")
    registry = load(root / "ECONOMIC_TRANSLATION_REGISTRY.json")
    validate_translation_registry(registry)
    prior = build_translation_registry()
    if len(registry.get("cells", [])) != 186:
        raise RuntimeError("Stage 19 registry is not 186 cells")
    for old, new in zip(prior["cells"], registry["cells"]):
        if old["cell_id"] != new["cell_id"] or old["canonical_economic_address_template"] == new["canonical_economic_address_template"]:
            raise RuntimeError("economic-address funding replacement incomplete")
        for key in old:
            if key not in {"cost_funding", "canonical_economic_address_template", "canonical_translation_id"} and old[key] != new[key]:
                raise RuntimeError(f"non-funding cell semantics changed: {new['cell_id']}:{key}")
        if new["canonical_economic_address_template"] != _address_template(new):
            raise RuntimeError("economic address is not canonical")

    funding = load(root / "FUNDING_COST_AND_COVERAGE_CONTRACT.json")
    expected_funding_semantics = {
        "timestamp_alignments": ["row_t_is_[t,t+1h)", "row_t_is_[t-1h,t)"],
        "signed_exact_formula_bps": "-position_sign*absolute_rate_usd_per_contract_unit_per_hour*overlap_seconds/3600/entry_trade_open_usd_per_contract_unit*10000",
        "adverse_exact_selection_bps": "min(0,signed_start_alignment_bps,signed_end_alignment_bps)",
        "base_primary_net_bps": "gross_return_bps-14+adverse_exact_selection_bps+base_missing_hour_gap_cost_bps",
        "stress_net_bps": "gross_return_bps-32+adverse_exact_selection_bps+stress_missing_hour_gap_cost_bps",
        "missing_hour_cost": "-rankable_only_allowance_bps_per_hour*missing_overlap_hours",
        "complete_symbol_allowance_or_event_rejection": True,
        "protected_rows_used": 0,
        "reporting_partitions": [
            "pre_funding_14bps", "adverse_exact_dual_alignment_primary",
            "adverse_exact_dual_alignment_stress", "signed_start_alignment_diagnostic",
            "signed_end_alignment_diagnostic", "timestamp_alignment_sensitivity",
            "missing_hour_allowance_contribution",
        ],
    }
    required_funding = {
        "source_rankable_package_sha256", "dual_alignment_contract_sha256", "gap_allowance_table_sha256",
        "unit_verification_sha256", *expected_funding_semantics,
    }
    if (required_funding - set(funding)
            or funding.get("base_pre_funding_round_trip_bps") != 14
            or funding.get("stress_pre_funding_round_trip_bps") != 32
            or any(funding.get(key) != value for key, value in expected_funding_semantics.items())):
        raise RuntimeError("Stage 19 funding contract incomplete or semantically inconsistent")
    if funding.get("favourable_funding_credit_can_select_rank_or_rescue") is not False or funding.get("runtime_semantic_choice") is not False:
        raise RuntimeError("funding credit or runtime semantic choice remains")
    if funding["dual_alignment_contract_sha256"] != sha256_file(root / "FUNDING_PERIOD_DUAL_ALIGNMENT_CONTRACT.json"):
        raise RuntimeError("dual-alignment contract hash drift")
    if funding["gap_allowance_table_sha256"] != sha256_file(root / "FUNDING_GAP_ALLOWANCE_TABLE.csv"):
        raise RuntimeError("allowance-table hash drift")
    if funding["unit_verification_sha256"] != sha256_file(root / "ABSOLUTE_RATE_UNIT_VERIFICATION.csv"):
        raise RuntimeError("unit-verification hash drift")
    dual = load(root / "FUNDING_PERIOD_DUAL_ALIGNMENT_CONTRACT.json")
    if (dual.get("unresolved_timestamp_semantics") is not True
            or dual.get("alignments") != {"alignment_start": "row t applies to [t,t+1h)", "alignment_end": "row t applies to [t-1h,t)"}
            or dual.get("cashflow_bps") != "-position_sign * absolute_rate_usd_per_contract_unit_per_hour * overlap_seconds/3600 / entry_trade_open_usd_per_contract_unit * 10000"
            or dual.get("selection_funding_bps") != "min(0,signed_alignment_start_bps,signed_alignment_end_bps)"
            or dual.get("favourable_credit_for_selection") is not False
            or dual.get("partial_hour_accrual") is not True
            or dual.get("runtime_alignment_choice") is not False):
        raise RuntimeError("dual-alignment semantic contract invalid")
    gap = load(root / "FUNDING_GAP_ALLOWANCE_CONTRACT.json")
    if (gap.get("rankable_interval") != "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)"
            or gap.get("measure") != "abs(relative_rate)*10000 bps/hour"
            or gap.get("quantile") != "Hyndman-Fan type 7 Decimal linear interpolation"
            or gap.get("minimum_symbol_observations") != 720
            or not str(gap.get("fallback", "")).startswith("equal-symbol-weighted empirical mixture")
            or gap.get("candidate_fold_context_or_return_conditioning") is not False
            or gap.get("protected_rows") != 0
            or gap.get("table_sha256") != funding["gap_allowance_table_sha256"]
            or gap.get("source_package_sha256") != funding["source_rankable_package_sha256"]):
        raise RuntimeError("gap allowance semantic contract invalid")

    source = load(root / "FUNDING_SOURCE_AND_PARTITION_MANIFEST.json")
    rankable = source["rankable_package"]
    package_path = Path("/opt/testerdonch/results/rebaseline/phase_kraken_local_official_funding_export_20260720_v4/kraken_funding_rankable_2023_2025.zip")
    if rankable["sha256"] != funding["source_rankable_package_sha256"] or sha256_file(package_path) != rankable["sha256"]:
        raise RuntimeError("rankable package authority mismatch")
    with zipfile.ZipFile(package_path) as archive:
        members = archive.namelist()
        if len(members) != rankable["symbols"] or any(not name.startswith("rankable_2023_2025/") for name in members):
            raise RuntimeError("rankable package member inventory mismatch")
    if source["protected_funding_values_used_for_statistics"] != 0 or source["protected_strategy_price_or_return_rows_opened"] != 0:
        raise RuntimeError("protected access firewall failed")

    with (root / "FUNDING_GAP_ALLOWANCE_TABLE.csv").open(newline="", encoding="utf-8") as handle:
        allowances = list(csv.DictReader(handle))
    if len(allowances) != 187 or len({row["symbol"] for row in allowances}) != 187:
        raise RuntimeError("campaign allowance coverage incomplete")
    for row in allowances:
        base, stress = Decimal(row["base_gap_allowance_bps_per_hour"]), Decimal(row["stress_gap_allowance_bps_per_hour"])
        if not base.is_finite() or not stress.is_finite() or base < 0 or stress < base:
            raise RuntimeError("invalid funding allowance")
    with (root / "CAMPAIGN_SYMBOL_FUNDING_MAPPING.csv").open(newline="", encoding="utf-8") as handle:
        mappings = list(csv.DictReader(handle))
    if len(mappings) != 187 or any(row["mapping_status"] != "mapped_exact_filename_and_tradeable" for row in mappings):
        raise RuntimeError("campaign symbol mapping incomplete")

    with (root / "ABSOLUTE_RATE_UNIT_VERIFICATION.csv").open(newline="", encoding="utf-8") as handle:
        units = list(csv.DictReader(handle))
    if len(units) != 187 or any(row["unit_status"] != "verified_one_contract_unit_equals_one_base_unit_no_hidden_multiplier" for row in units):
        raise RuntimeError("campaign unit verification incomplete")
    unit_sources = load(root / "UNIT_VERIFICATION_SOURCE_MANIFEST.json")
    if unit_sources["unit_audit_sha256"] != sha256_file(root / "ABSOLUTE_RATE_UNIT_VERIFICATION.csv") or len(unit_sources["anchor_source_files"]) != 187:
        raise RuntimeError("unit source manifest incomplete")
    mapping_symbols = {row["campaign_symbol"] for row in mappings}
    allowance_symbols = {row["symbol"] for row in allowances}
    unit_symbols = {row["symbol"] for row in units}
    anchor_symbols = {row["symbol"] for row in unit_sources["anchor_source_files"]}
    if not (mapping_symbols == allowance_symbols == unit_symbols == anchor_symbols) or len(mapping_symbols) != 187:
        raise RuntimeError("campaign symbol sets differ across funding authorities")
    for item in unit_sources["anchor_source_files"]:
        if sha256_file(Path(item["trade_path"])) != item["trade_sha256"] or sha256_file(Path(item["mark_path"])) != item["mark_sha256"]:
            raise RuntimeError(f"unit source hash drift: {item['symbol']}")
    for item in (unit_sources["instrument_snapshot"], unit_sources["acquisition_manifest"]):
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise RuntimeError("unit authority source hash drift")

    engine = Stage19FundingEngine(
        package_path, funding["source_rankable_package_sha256"], root / "FUNDING_GAP_ALLOWANCE_TABLE.csv",
        funding["gap_allowance_table_sha256"],
    )
    for symbol in sorted(row["symbol"] for row in allowances):
        if not engine.load_symbol(symbol):
            raise RuntimeError(f"campaign funding rows absent: {symbol}")

    canary = load(root / "SYNTHETIC_CANARY.json")
    required_canary = [
        "both_alignments_exercised", "campaign_deterministic", "economic_addresses_unique",
        "favourable_funding_ignored_for_selection", "long_short_sign_reversal", "missing_gap_nonpositive",
        "partial_hour_exercised", "positive_and_negative_rates_exercised", "stage16_semantic_canary_pass",
    ]
    if canary.get("status") != "pass" or not all(canary.get(key) is True for key in required_canary):
        raise RuntimeError("Stage 19 synthetic canary failed")
    if canary.get("allowance_table_sha256") != funding["gap_allowance_table_sha256"] or canary.get("protected_rows_opened") != 0 or canary.get("economic_outputs_computed") != 0:
        raise RuntimeError("canary source or firewall mismatch")

    return {
        "status": "pass", "registered_cells": 186, "campaign_symbols": 187,
        "rankable_package_sha256": rankable["sha256"], "dual_alignment": "pass",
        "gap_allowance": "pass", "unit_verification": "pass", "runtime_adapter": "pass",
        "synthetic_canary": "pass", "economic_outputs_computed": False,
        "protected_strategy_price_or_return_rows_opened": 0,
        "external_human_approval_required": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), sort_keys=True))
