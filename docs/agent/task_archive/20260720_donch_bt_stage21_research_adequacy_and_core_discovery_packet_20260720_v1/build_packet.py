#!/usr/bin/env python3
"""Build the no-outcome Stage 21 audit and frozen campaign contracts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scipy.stats import qmc


HERE = Path(__file__).resolve().parent
CAMPAIGN = "kraken_core_liquid_discovery_campaign_002"
GENERATED_AT = "2026-07-20T17:04:20Z"  # exact received-task archive timestamp, truncated to seconds
BASE_COMMIT = "405878887780f7e85fe90a601a643bee502d7d5c"
STAGE20_RUN = Path("/opt/testerdonch/results/rebaseline/phase_kraken_derivatives_campaign_stage20_20260720_v01")
STAGE19 = Path("/opt/testerdonch/docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name: str, obj: object) -> None:
    (HERE / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


stage20_hashes = {
    "terminal_handoff_zip": "3a3fd639f9e9bb0cb2b7d87828310aa27a82aef689ae085f1e6624e8d020526c",
    "CAMPAIGN_STATE.json": "03fe54d3c8dd30a838b7ff2d688cb2adc2c058cf7bea1ff1be55ff8d54a810f8",
    "PHASE2_DEVELOPMENT_RESPONSE_SURFACE.csv": "e31ba061afba2346c6a1ec3574d793b5d5c2dcf0afa273fbfa31676602774fba",
    "PHASE3_FROZEN_BEAM_REGISTRY.json": "1c84fffea8daabdcd5aca8e8170ef42bfdefcb42ad6945874b1e8255e0de0943",
    "PHASE4_5_OUTER_ROLLING_RESULTS.csv": "a4af28424acacb18559fe9928fd77d3afd3b8ae1a2358c6ba588b111c556f7b3",
    "FOLD_AND_FAMILY_DECISIONS.json": "6d82f6cafde6b65a36070369d7f552ca2a21632eaa9e1c3333f5d7bbdcc973a1",
    "FAMILY_ROUTES.json": "b3a8fb3caba4320f21059bf3bcf5265a138eda9a0ee8970a528a056eaa102670",
    "FULL_ATTEMPT_REGISTRY.json": "1bba412d30dd64f556868d2cdeb526a06206e43c45f2de44db7fbe720e0478bf",
    "MULTIPLICITY_RECONCILIATION.json": "72bc0d6c7fa48f264fb7e5472c25e75ed816fb03f27d53b1ae12cde38639bdfe",
    "SYMBOL_SCORING_MANIFEST.json": "b52886b0cd7429dc7f00880d6b717d725d43904ffea99b4ffa301db4ebb68885",
}

economic_dependencies = {
    "repository_base_commit": BASE_COMMIT,
    "kraken_k0_download_manifest": "f598cc1fb5714386923272399b98fa560c119c96fd5af33f5b30735f40cea420",
    "stage8a_feature_cache_manifest": "463e108569469309e064fd8168235064e8b242eb11668a1e186ec4ceb1cf5538",
    "shared_funding_panel_manifest": "3c3512dd96f5dda02d03cc3b40005a3c0700c8fda443db48d383d7d20c244ad7",
    "rankable_funding_package": "6c0969727c882ca6439c57c3b7d03367e1b2d26ee46823e56b983756d026ef64",
    "stage19_funding_contract": "c0d5bf091080a681055a792a8a7cf4a1084dcd60deb17c60381339df7297c3e7",
    "stage19_gap_allowance_table": "4ff2c6bcbe125e6f966348a029cd2cc63f55deb752b89c454a28642c41f119b5",
    "tier1_prior_input_manifest": "6837d15e501d1d65db8b3a604023cab0d790fbe09eaa30fdb21f91639ab936ea",
    "tier1_prior_frozen_artifact_hashes": "55cb4b21d5800f60a17d44046bcd0b19ac1ec4d0c4194243fcaeb79a962f925a",
    "tsmom_v6_definition_manifest": "6d895e6dc60b3ad6177bbb26c40c3b901a8e8c7792d8449801d808c19914f003",
    "first_wave_program_decision_register": "53e6946261c5f27bf7096b162ca4d8c80910fed44da61c10f3839cf585af04d8",
    "first_wave_authority_and_scope": "fd99524fdf9a479038e48325efe7aa01a40306b5d2ef012fa05e2d457b5a2618",
    "first_wave_lineage_map": "48ae91a37aeb5c18adeffd5e8ee60395904fb6d3220ca4d676d71ac097e87bfc",
    "stage20_external_human_approval": "57d521488e88373afd557eb457ffd119089aac0d863a0c319e73d70cf9f7690c",
    "stage20_campaign_manifest": "e7d618a2a24c574c9ba83d323df605a6434c2e816fe24c465d183ba5d6256990",
    "stage20_approval_packet": "3fde09f16efff2479ee847fe26c859a67ef1d516b0f0acbcbaec141154a87bd6",
    **{f"stage20_{key}": value for key, value in stage20_hashes.items()},
}


coverage = []


def add(family: str, axis: str, tested: str, untested: str, provenance: str, conclusion: str) -> None:
    coverage.append({"family": family, "axis": axis, "tested_values": tested,
                     "untested_values_or_scope": untested, "provenance": provenance,
                     "adequacy_conclusion": conclusion})


add("KDA02B", "feature_window", "completed 1h trade/mark/OI displacement", "other causal windows", "registered_design_choice", "narrow")
add("KDA02B", "price_threshold", "raw abs 14..500 bps; inner-train Type-7 q80..q100", "continuous magnitude and other bounded tails", "registered_design_choice", "partial")
add("KDA02B", "OI_threshold", "raw -0.12..-0.01 log change; inner-train Type-7 q0..q20", "continuous OI response and other tails", "registered_design_choice", "partial")
add("KDA02B", "liquidation", "present/absent; causal intensity q80..q100", "continuous interactions outside two forms", "outcome_free_measurement_derived", "partial")
add("KDA02B", "directions", "positive/negative impulse x continuation/reversal", "none within symmetric mapping", "mechanism_derived", "covered")
add("KDA02B", "contexts", "liquidation only", "BTC/ETH trend/vol/drawdown, breadth, liquidity, listing age, funding veto", "registered_design_choice", "incomplete")
add("KDA02B", "entry", "next native PF 5m trade open, <=10m delay", "confirmation and bounded delay variants", "registered_design_choice", "narrow")
add("KDA02B", "exit", "fixed 1h/3h/6h", "structural, ATR, trail, stop/target", "registered_design_choice", "incomplete")
add("KDA02B", "overlap", "definition- and symbol-local chronological actual-exit non-overlap", "portfolio collision/capacity", "registered_design_choice", "mechanically covered")
add("KDA02B", "costs", "14 bps base; 32 bps stress; exact adverse funding plus gap allowance", "impact/capacity curve", "registered_design_choice", "execution-sensitive")
add("KDA02C", "base_identity", "completed-purge reversal, primary_z2 and robust_pct95", "other event identities", "source_prior", "frozen narrow translation")
add("KDA02C", "breadth", "raw share, fold-local rank, isolated/nonisolated; 5/15/30/60m", "continuous sizing/permission overlays outside these forms", "registered_design_choice", "sparse/sample-limited")
add("KDA02C", "directions", "negative reclaim long; positive failure short", "none within frozen payoff map", "mechanism_derived", "covered but sparse")
add("KDA02C", "entry_exit", "next 5m open; fixed 1h", "other entries/exits/stops/targets/trails", "registered_design_choice", "incomplete")
add("KDA02C", "contexts", "PIT directional breadth only", "BTC/ETH, dispersion, liquidity, listing age, funding state", "registered_design_choice", "incomplete; no more 2023-2025 tuning")
add("KDA02C", "costs", "14/32 bps and exact adverse funding/gap allowance", "impact/capacity curve", "registered_design_choice", "not reached outer evaluation")
add("KDX01", "component_ladder", "7 nested trade/mark/OI/liquidation/basis/breadth ladders", "non-nested sparse combinations and materially different mechanisms", "mechanism_derived", "current translation tested, family not exhausted")
add("KDX01", "transform", "raw units; fold-local ranks", "continuous smooth scores", "outcome_free_measurement_derived", "partial")
add("KDX01", "direction", "completed downside rejection/reclaim long", "continuation excluded as inherited contradictory attempts", "mechanism_derived", "exact translation covered")
add("KDX01", "entry_exit", "next 5m open; fixed 1h/3h/6h", "structural/ATR/trail/stop/target", "registered_design_choice", "incomplete")
add("KDX01", "contexts", "optional breadth in one ladder", "BTC/ETH, liquidity, listing age, funding veto", "registered_design_choice", "incomplete")
add("KDX01", "costs", "14/32 bps and exact adverse funding/gap allowance", "impact/capacity curve", "registered_design_choice", "translation rejected before outer")

with (HERE / "STAGE20_SEARCH_COVERAGE_MATRIX.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(coverage[0]))
    writer.writeheader(); writer.writerows(coverage)


audit = {
    "schema": "stage20_research_adequacy_audit_v1",
    "authority_status": "verified",
    "repository_commit": BASE_COMMIT,
    "terminal_artifact_hashes": stage20_hashes,
    "row_decomposition": {
        "development": {"KDA02B": {"cells": 96, "folds": 9, "rows": 864},
                        "KDA02C": {"cells": 48, "folds_reached": 1, "rows": 48},
                        "KDX01": {"cells": 42, "folds_reached": 1, "rows": 42}, "total": 954},
        "outer": {"KDA02B": {"folds": 9, "beam_per_fold": 5, "rows": 45},
                  "KDA02C": 0, "KDX01": 0, "total": 45},
    },
    "empty_inner_fold_observations": {"available": 5094, "empty_unavailable": 258,
                                      "KDA02C_empty": 186, "KDX01_empty": 72,
                                      "silent_omission": False},
    "family_stops": {
        "KDA02B": {"continued_through": "2025Q4", "reason": "five deterministic development candidates existed in every fold"},
        "KDA02C": {"stopped_at": "2023Q4", "reason": "no beam-eligible positive development candidate; later eight folds recorded family_stopped_earlier"},
        "KDX01": {"stopped_at": "2023Q4", "reason": "no positive development candidate; later eight folds recorded family_stopped_earlier"},
    },
    "routes_supported": {
        "KDA02B": "execution_sensitive_candidate",
        "KDA02C": "sample_limited_prospective_candidate",
        "KDX01": "translation_rejected",
        "defect_proven": False,
    },
    "scope_conclusions": {
        "execution_correctness": "pass for the frozen Stage20 packet: hash authority, deterministic replay, 954/45 exact metric recomputation, 8415 job markers, 4556 artifact claims, temporal boundaries, costs and funding reconciled",
        "search_completeness": "not established; each family used a deliberately bounded and often narrow grammar",
        "family_exhaustion": "not established; only the exact KDX01 translation is rejected, KDA02C is frozen prospective, and KDA02B requires controls/forensics rather than retuning",
        "validation_or_live_readiness": "none; all evidence is program_exposed_historical, controls were not run, and no independent validation or deployment evidence exists",
        "independent_review_interpretation": "zero metric mismatch verifies arithmetic and replay, not search completeness",
    },
    "provenance_classes_used": ["source_prior", "mechanism_derived", "outcome_free_measurement_derived", "registered_design_choice"],
    "unsupported_guess_count": 0,
    "unsupported_guess_note": "Registration alone is not empirical support. Exact choices lacking a source or measurement are classified registered_design_choice, not silently promoted to a prior.",
}
quarters_stage20 = ["2023Q4", "2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"]
audit["family_quarter_counts"] = [
    {"family": family, "quarter": quarter,
     "development_rows": (96 if family == "KDA02B" else (48 if family == "KDA02C" and quarter == "2023Q4" else (42 if family == "KDX01" and quarter == "2023Q4" else 0))),
     "outer_rows": 5 if family == "KDA02B" else 0,
     "status": ("outer_evaluated" if family == "KDA02B" else ("family_stop_no_positive_development_candidate" if quarter == "2023Q4" else "family_stopped_earlier"))}
    for quarter in quarters_stage20 for family in ("KDA02B", "KDA02C", "KDX01")
]


def choice_provenance(family: str, path: str, value: object) -> str:
    text = f"{path}={json.dumps(value, sort_keys=True)}".lower()
    if "purge_identity" in text or "base_event" in text or "source_stage14" in text:
        return "source_prior"
    if any(token in text for token in ("branch", "direction", "price_state", "component_ladder", "payoff_archetype", "side_mapping")):
        return "mechanism_derived"
    if any(token in text for token in ("fold_local", "type7", "rank", "continuous_intensity")):
        return "outcome_free_measurement_derived"
    return "registered_design_choice"


translation_registry = json.loads((STAGE19 / "ECONOMIC_TRANSLATION_REGISTRY.json").read_text(encoding="utf-8"))
choice_records = []
cell_provenance_records = []
for family in ("KDA02B", "KDA02C", "KDX01"):
    family_cells = [row for row in translation_registry["cells"] if row["family"] == family]
    distinct = set()
    for row in family_cells:
        for section in ("search_axes", "entry", "exit", "stop_target", "side_mapping", "payoff_archetype"):
            payload = row.get(section)
            if isinstance(payload, dict):
                for key, value in payload.items():
                    distinct.add((f"{section}.{key}", json.dumps(value, sort_keys=True)))
            else:
                distinct.add((section, json.dumps(payload, sort_keys=True)))
        exact = row.get("feature_contract", {}).get("exact_thresholds", {})
        for key, value in exact.items():
            distinct.add((f"feature_contract.exact_thresholds.{key}", json.dumps(value, sort_keys=True)))
        for key, value in row.get("cost_funding", {}).items():
            distinct.add((f"cost_funding.{key}", json.dumps(value, sort_keys=True)))
    for path, encoded in sorted(distinct):
        value = json.loads(encoded)
        choice_records.append({"family": family, "choice_path": path, "exact_value": value,
                               "provenance": choice_provenance(family, path, value)})
    for row in family_cells:
        cell_choices = []
        for section in ("search_axes", "entry", "exit", "stop_target", "side_mapping", "payoff_archetype"):
            payload = row.get(section)
            items = payload.items() if isinstance(payload, dict) else [(section, payload)]
            for key, value in items:
                path = f"{section}.{key}" if isinstance(payload, dict) else section
                cell_choices.append({"choice_path": path, "exact_value": value, "provenance": choice_provenance(family, path, value)})
        for key, value in row.get("feature_contract", {}).get("exact_thresholds", {}).items():
            path = f"feature_contract.exact_thresholds.{key}"
            cell_choices.append({"choice_path": path, "exact_value": value, "provenance": choice_provenance(family, path, value)})
        for key, value in row.get("cost_funding", {}).items():
            path = f"cost_funding.{key}"
            cell_choices.append({"choice_path": path, "exact_value": value, "provenance": choice_provenance(family, path, value)})
        cell_provenance_records.append({"family": family, "cell_id": row["cell_id"], "canonical_translation_id": row["canonical_translation_id"], "choices": sorted(cell_choices, key=lambda item: item["choice_path"])})
audit["translation_choice_provenance"] = choice_records
audit["translation_choice_provenance_count"] = len(choice_records)
audit["translation_cell_provenance_ledger"] = {"path": "STAGE20_TRANSLATION_PROVENANCE.jsonl", "rows": len(cell_provenance_records)}
provenance_path = HERE / "STAGE20_TRANSLATION_PROVENANCE.jsonl"
provenance_path.write_bytes(b"".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n" for row in cell_provenance_records))
audit["translation_cell_provenance_ledger"]["sha256"] = sha(provenance_path)
write_json("STAGE20_RESEARCH_ADEQUACY_AUDIT.json", audit)


mechanism_common = {
    "PIT_universe_and_liquidity_scope": "Kraken native linear PF instruments in the existing current-roster/lifecycle-capped authority; eligible at decision, known lifecycle, >=90 calendar days since first authorized bar, ranked daily using only lagged 30-day median quote notional; top-N is part of the registered address; no historical-completeness claim",
    "feature_availability": "all inputs require source_close_ts<=decision_ts and feature_available_ts<=decision_ts; completed-bar fields become usable at the next native 5m decision boundary",
    "entry_semantics": "first authorized native PF 5m trade open at or after decision_ts, maximum lookup delay 10 minutes; missing price or lifecycle rejects before outcome",
    "boundary": "decision, entry, and actual executable exit must remain in the authorized fold; crossing events reject and are never force-closed",
    "cost_funding": "14 bps base and 32 bps stress pre-funding round trip; exact signed funding at both start/end boundary alignments with adverse=min(0,start,end); rankable-only nonpositive gap allowance; favourable funding cannot rescue selection",
    "programme_exposure": "all 2023-2025 results are program_exposed_historical",
}
cards = {
    "schema": "core_campaign_mechanism_cards_v1", "campaign_id": CAMPAIGN,
    "common": mechanism_common,
    "families": [
        {"family_id": "A4_TSMOM_V7", "lineage": ["A4", "H02", "tsmom_v6"],
         "mechanism_and_compelled_actor": "Slow trend followers, hedgers and risk-targeting allocators adjust exposures gradually; persistent liquid-perpetual moves may outlast the first rebalance.",
         "observable_Kraken_state": "causal multi-horizon signed return, EMA slope, breakout-distance rank, realized volatility, path smoothness, lagged liquidity and exact funding",
         "payoff_archetype": "symmetric directional persistence with volatility-scaled exposure",
         "material_difference": "continuous multi-horizon ensemble plus explicit path-quality coverage and MAE/MFE repair; not a relabel of v6 fixed curated definitions",
         "failure_mode": "turnover/cost dominance, crash reversal, single-symbol concentration, or no stable neighborhood",
         "main_null_control": "same rebalance/universe/vol scaling with lagged direction signs permuted within symbol-year before returns are opened",
         "component_ablations": ["single-horizon", "no-vol-scaling", "no-path-smoothness", "no-context", "no-funding-veto"],
         "falsification_rule": "stop family if no predeclared plateau passes integrity, base-cost positivity and replication requirements or if generic price-only TSMOM is not improved by the claimed component",
         "multiplicity_lineage": "inherit all v1-v6 attempts and unresolved historical count; do not reset or claim a complete numeric prior total"},
        {"family_id": "A1_COMPRESSION_V2", "lineage": ["A1", "H01", "H06", "H12", "H13"],
         "mechanism_and_compelled_actor": "Liquid leaders with a causal impulse and orderly low-volatility base can retain under-positioned follow-through as slower participants enter after confirmation.",
         "observable_Kraken_state": "episode-level impulse magnitude, base duration, contraction, path smoothness, close confirmation, relative strength and lagged liquidity",
         "payoff_archetype": "right-skewed continuation from a uniquely identified impulse/base episode",
         "material_difference": "one episode contributes once per definition and portfolio claims are equal-episode/equal-day, replacing pooled row-weighted evidence",
         "failure_mode": "duplicate episodes, false compression, broad-market beta, cost dominance or unstable thresholds",
         "main_null_control": "liquidity/lifecycle/month-matched completed-bar pseudo-episodes with the same hold grammar",
         "component_ablations": ["price-only impulse", "no-compression", "no-smoothness", "no-relative-strength", "no-context", "generic breakout"],
         "falsification_rule": "stop if episode identity is non-unique, controls explain the effect, or no stable development plateau replicates forward",
         "multiplicity_lineage": "inherit the full A1/compression programme including prior 180-shard and targeted-materialization attempts; prior total is not fully recoverable"},
        {"family_id": "A2_PRIOR_HIGH_RS_CONTEXT_V1", "lineage": ["A2_filter", "H03", "H04", "prior_high_reclaim_v2", "rs_breakout_btc"],
         "mechanism_and_compelled_actor": "Prior highs and cross-sectional strength proxy latent supply and leadership; they may condition, size or permit another causal continuation entry rather than create an opaque standalone regime.",
         "observable_Kraken_state": "continuous prior-high proximity/reclaim rank, BTC-relative strength rank, PIT breadth/dispersion and BTC/ETH trend/volatility/drawdown",
         "payoff_archetype": "contextual uplift to A1/A3 continuation; never a standalone candidate in this campaign",
         "material_difference": "continuous ranks with explicit in/out and feature-removal ablations, not retuned binary gates",
         "failure_mode": "no matched uplift, unstable direction, low coverage or current-roster concentration",
         "main_null_control": "parent A1/A3 signal with context score permuted within decision month and liquidity decile",
         "component_ablations": ["prior-high only", "relative-strength only", "BTC/ETH only", "breadth only", "parent signal without context"],
         "falsification_rule": "reject overlay if matched uplift is nonpositive or context removal does not degrade the complete address",
         "multiplicity_lineage": "inherit prior-high v1/v2, reclaim and RS-vs-BTC attempts; no same-sample binary-gate rescue"},
        {"family_id": "A3_STARTER_RETEST_V3", "lineage": ["A3", "H05", "breakout_retest_v2"],
         "mechanism_and_compelled_actor": "Close-confirmed breakout can start continuation immediately while a later orderly retest may provide separate confirmation and sizing capacity.",
         "observable_Kraken_state": "completed breakout close, causal range/ATR, immediate starter, separately timestamped retest and reclaim, and context ranks",
         "payoff_archetype": "convex continuation with separately attributed starter and optional add legs",
         "material_difference": "mandatory-retest-only entry remains closed; every address has an immediate starter and the retest is a separately accounted optional add that never substitutes for the starter",
         "failure_mode": "failed breakouts, delayed untradeable adds, double-counting legs, or no incremental retest contribution",
         "main_null_control": "same breakout and starter with retest/add timestamps permuted among eligible episodes before return access",
         "component_ablations": ["starter-only", "starter-plus-add", "confirmation-to-one-close", "no-context", "matched pseudo-event"],
         "falsification_rule": "reject retest mechanism if the separately attributed add is nonpositive or the starter/retest identity cannot be reconciled without duplication",
         "multiplicity_lineage": "inherit all A3 and breakout-retest attempts; mandatory-retest-only route remains terminally closed"},
        {"family_id": "KDA02B_SURVIVOR_ADJUDICATION_V1", "lineage": ["KDA02", "KDA02B", "Stage20"],
         "mechanism_and_compelled_actor": "OI-vacuum with agreed trade/mark impulse and liquidation may reflect forced inventory clearing whose continuation/reversal depends on execution and derivative-state components.",
         "observable_Kraken_state": "the exact 19 unique Stage20 selected cell identities and their frozen event/execution semantics",
         "payoff_archetype": "unchanged Stage20 symmetric directional addresses",
         "material_difference": "none; this is adjudication only, never threshold rescue or redevelopment",
         "failure_mode": "price-only/null equivalence, funding or cost sensitivity, contributor concentration or execution delay fragility",
         "main_null_control": "exact price-only and generic-structure controls frozen before outcomes",
         "component_ablations": ["OI removed", "liquidation removed", "price-only", "time-matched generic structure"],
         "falsification_rule": "do not advance an address unless components add value and all predeclared execution/forensic checks are reconciled",
         "multiplicity_lineage": "inherits all 228 Stage20 programme attempts; 19 unique selected cells are adjudicated without retuning"},
    ],
    "excluded_routes": {"KDA02C": "freeze sample_limited_prospective_candidate; no 2023-2025 tuning",
                        "KDX01": "preserve translation_rejected; only a future materially different translation ID may reopen",
                        "C17": "excluded", "Capitalcom": "excluded", "protected_period": "excluded"},
}
write_json("CORE_CAMPAIGN_MECHANISM_CARDS.json", cards)


measurement = {
    "schema": "outcome_free_measurement_plan_v1", "campaign_id": CAMPAIGN,
    "outcome_firewall": {"forward_returns_or_PnL": False, "protected_rows": 0, "Capitalcom": False,
                         "allowed": "timestamps, lifecycle, coverage, units, feature distributions, event counts, metadata, cost/funding availability and cache hashes"},
    "required_checks": ["file and row-group time bounds before payload read", "source_close_ts and feature_available_ts", "units and finite ranges", "PIT lifecycle and daily lagged-liquidity membership", "event identity uniqueness", "coverage by symbol/month/family", "feature quantiles", "event frequency", "exact funding coverage and missing-hour partitions", "base/stress cost scale", "cache deterministic replay", "TSMOM path MAE/MFE coverage"],
    "measurement_quantiles_type7": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    "domain_binding": "Only symbolic quantile labels and bounded source-prior anchors listed in REGISTERED_DISCOVERY_SPACE may be used. For an inner validation month, numeric values are Type-7 quantiles fit on [2023-01-01, validation_start-10d). For an outer quarter, they are refit on [2023-01-01, outer_start-10d), then hash-frozen before the outer reader opens. Missing values are excluded only from fitting and remain unavailable in evaluation; ties receive the weak percentile (# strictly less + # equal)/N; no return information enters fitting.",
    "hard_gates": {"required_feature_coverage": 0.95, "required_lifecycle_coverage": 1.0,
                   "minimum_unique_values_for_rank": 20, "minimum_events_per_family_development_fold": 20,
                   "minimum_exact_funding_mapping_symbols": 187, "protected_rows_opened": 0,
                   "on_failure": "family stop; no threshold substitution or scope broadening"},
    "outputs": ["MEASUREMENT_MANIFEST.json", "FEATURE_COVERAGE.csv", "EVENT_FREQUENCY.csv", "FEATURE_DISTRIBUTIONS.csv", "UNIT_AND_TIMESTAMP_AUDIT.json", "CACHE_REPLAY.json"],
    "benchmark_summary": {"performed_without_new_outcomes": True, "basis": "existing terminal Stage20 runtime plus prior-family runtime/memory records and current host capacity", "stage20_elapsed_seconds": 712, "stage20_jobs": 8415, "stage20_output_bytes": 1111965563, "prior_a1_runtime_seconds": 62772.83027601242, "prior_a1_peak_rss_bytes": 5110345728, "current_logical_cpus": 4, "current_memory_total_bytes": 16369111040, "current_disk_free_bytes": 36596535296},
}
write_json("OUTCOME_FREE_MEASUREMENT_PLAN.json", measurement)


family_spaces = [
    {"family_id": "A4_TSMOM_V7", "screening": 640, "anchors_ablations": 64, "refinement_reserve": 160,
     "axes": {"signal_estimator": ["signed_return", "ema_slope", "breakout_distance_rank", "equal_weight_ensemble"], "lookback_days": [5, 10, 20, 40, 60, 90, 120, 180], "vol_window_days": [10, 20, 40, 60], "annualized_vol_target": ["none", 0.10, 0.20, 0.30], "rebalance": ["8h", "1d"], "direction": ["long_short", "long_flat", "short_flat"], "path_smoothness_quantile_min": ["none", "q20", "q40", "q60"], "PIT_liquidity_top_n": [10, 20, 40], "exit": ["time_1d", "time_3d", "time_5d", "time_10d", "signal_reversal", "ATR_trail_2", "ATR_trail_3", "ATR_trail_4"], "context_overlay": ["none", "BTC_ETH", "breadth_dispersion", "funding_veto"]}},
    {"family_id": "A1_COMPRESSION_V2", "screening": 768, "anchors_ablations": 80, "refinement_reserve": 192,
     "axes": {"impulse_window": ["6h", "12h", "1d", "3d", "7d"], "impulse_rank_min": ["q60", "q70", "q80", "q90"], "base_duration": ["2h", "6h", "12h", "1d", "3d"], "contraction_rank_max": ["none", "q20", "q30", "q40", "q50"], "smoothness_rank_min": ["none", "q40", "q60", "q80"], "confirmation": ["one_close", "two_closes", "close_plus_bounded_15m_delay"], "direction": ["long", "short", "symmetric"], "PIT_liquidity_top_n": [10, 20, 40], "exit": ["time_1d", "time_3d", "time_5d", "time_10d", "base_failure", "ATR_stop_1.5", "ATR_stop_2.5", "ATR_trail_2", "ATR_trail_3"], "fixed_target_R": ["none", 1.5, 3.0], "context_overlay": ["none", "prior_high_RS", "BTC_ETH", "breadth_dispersion", "funding_veto"]}},
    {"family_id": "A2_PRIOR_HIGH_RS_CONTEXT_V1", "screening": 512, "anchors_ablations": 64, "refinement_reserve": 128,
     "axes": {"parent_family": ["A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"], "prior_high_lookback_days": [20, 60, 120, 250], "proximity_rank": ["none", "continuous", "q40", "q60", "q80"], "reclaim_state": ["continuous_distance", "first_close_above", "none"], "RS_lookback_days": [5, 10, 20, 60], "RS_rank": ["none", "continuous", "q40", "q60", "q80"], "overlay_action": ["permission", "linear_size_0_to_1", "tercile_size"], "BTC_ETH_context": ["none", "trend", "volatility", "drawdown"], "breadth_dispersion": ["none", "breadth", "dispersion", "both"]}},
    {"family_id": "A3_STARTER_RETEST_V3", "screening": 512, "anchors_ablations": 64, "refinement_reserve": 128,
     "axes": {"breakout_lookback_days": [5, 10, 20, 60], "breakout_rank_min": ["q60", "q70", "q80", "q90"], "confirmation": ["one_close", "two_closes", "close_plus_15m_delay"], "starter_fraction": [0.25, 0.5, 0.75, 1.0], "retest_window": ["6h", "1d", "3d"], "retest_depth_ATR": [0.25, 0.5, 1.0, 1.5], "add_fraction": [0.0, 0.25, 0.5, 0.75], "add_requires_reclaim": [True], "PIT_liquidity_top_n": [10, 20, 40], "exit": ["time_1d", "time_3d", "time_5d", "time_10d", "breakout_failure", "ATR_stop_1.5", "ATR_stop_2.5", "ATR_trail_2", "ATR_trail_3"], "fixed_target_R": ["none", 1.5, 3.0], "context_overlay": ["none", "prior_high_RS", "BTC_ETH", "breadth_dispersion", "funding_veto"]}},
    {"family_id": "KDA02B_SURVIVOR_ADJUDICATION_V1", "screening": 0, "anchors_ablations": 209, "refinement_reserve": 0,
     "exact_addresses": {"Stage20_unique_cells": ["KDA02B_009", "KDA02B_011", "KDA02B_031", "KDA02B_032", "KDA02B_033", "KDA02B_034", "KDA02B_035", "KDA02B_036", "KDA02B_045", "KDA02B_047", "KDA02B_048", "KDA02B_058", "KDA02B_060", "KDA02B_079", "KDA02B_081", "KDA02B_082", "KDA02B_083", "KDA02B_084", "KDA02B_093"], "evaluations_per_cell": 11, "allowed": ["identity_replay", "price_only", "OI_removed", "liquidation_removed", "generic_structure_control", "stress_cost_32bps", "funding_start_alignment", "funding_end_alignment", "funding_zero", "entry_delay_15m", "entry_delay_60m"]}},
]
discovery = {
    "schema": "registered_discovery_space_v1", "campaign_id": CAMPAIGN,
    "design": {"method": "deterministic scrambled Sobol for mixed domains", "seed": 20260721,
               "categorical_mapping": "floor(u*k), values in listed order", "numeric_mapping": "nearest listed value, ties to lower list index",
               "deduplication": "canonical JSON economic address SHA-256; duplicates retained in attempt ledger as unavailable_duplicate and not rerun",
               "interactions": "main effects and mechanism-listed pairwise interactions only; no three-way interaction",
               "refinement": "reserve IDs are assigned now; after a stable region is identified, generate +/- one adjacent registered axis level around its deterministic medoid, lexically truncate to reserve, atomically hash-register before refinement outcomes; unused reserve IDs are recorded unavailable_no_plateau",
               "registration_boundary": "every cell address and seed must be persisted before its own outcome reader is opened"},
    "feature_and_execution_semantics": {
        "base_clock": "native PF 5m trade candles in UTC, explicitly sorted; every W-window is (decision_ts-W,decision_ts] over fully completed contiguous bars",
        "price_fields": "trade close for features, trade open for execution; mark remains separate and is never substituted for trade execution",
        "signed_return": "log(trade_close_t/trade_close_at_or_before_t_minus_W), a single signed scalar independent of any future trade side; missing or noncontiguous endpoints reject",
        "ema_slope": "log(EMA_span_W_t/EMA_span_W_at_or_before_t_minus_1d), a single signed scalar with span=W*288 completed 5m bars, min_periods=span, exponentially weighted adjust=false",
        "breakout_distance_rank": "a single signed scalar: max((close-prior_high)/ATR20,0)+min((close-prior_low)/ATR20,0), where the prior W high/low excludes the decision bar; values inside the prior range equal zero",
        "realized_volatility": "sample standard deviation of 5m log returns over W, annualized by sqrt(365*288), requiring >=95% expected contiguous observations",
        "equal_weight_ensemble": "compute signed_return/realized_vol_W, ema_slope/realized_vol_W and breakout_distance_rank, winsorize each at its symbol-local training q05/q95, scale each by its symbol-local training MAD (zero MAD unavailable), and take their arithmetic mean; strict sign is the direction",
        "TSMOM_direction": "long_short enters long for estimator>0 and short for estimator<0; long_flat enters only estimator>0; short_flat enters only estimator<0; exact zero is flat; side is derived only after the scalar estimator is complete",
        "volatility_scaling": "if annualized_vol_target=none, absolute exposure=1.0; otherwise at entry exposure=min(2.0,max(0.25,target_vol/realized_vol_window)); equal weight across simultaneous eligible symbols; exposure stays frozen for the episode; no leverage or portfolio-capacity claim",
        "A4_weighted_aggregation": "at each registered rebalance t, form accepted set I_t only after PIT universe, signal, context, lifecycle and actual-exit non-overlap gates. For i in I_t: u_i=side_i*(exit_open_i/entry_open_i-1)*10000; e_i=1 when target is none else clip(target_vol/realized_vol_i,0.25,2); c_i is frozen context multiplier in [0,1]; w_i=e_i*c_i/|I_t| with no post-scaling renormalization; n_i=u_i-14+adverse_exact_funding_i+base_gap_allowance_i; s_i=u_i-32+adverse_exact_funding_i+stress_gap_allowance_i. Base/stress contributions are w_i*n_i and w_i*s_i, so gross, cost, funding and allowance share the same notional weight. Cohort return is the sum of contributions. A UTC entry day's return is the arithmetic mean of its nonempty cohort returns, and primary/stress are arithmetic means of day returns. Empty cohorts are no-opportunities, not zeros. Event count is accepted symbol episodes; market-day/bootstrap metrics use cohort-derived day returns.",
        "TSMOM_rebalance_and_turnover": "8h boundaries are 00:00/08:00/16:00 UTC and 1d is 00:00 UTC; generate a new entry opportunity only when the definition-symbol is flat; definition-local actual-exit non-overlap rejects intervening opportunities; each completed episode incurs the registered round-trip cost",
        "signal_reversal": "at a registered rebalance, the complete side-aware estimator or equal-weight ensemble changes strict sign; exact zero means flat; exit/reversal occurs at the next authorized 5m open",
        "path_smoothness": "abs(log(close_t/close_t_minus_W))/sum(abs(5m log returns)) over the same complete window, in [0,1]",
        "ATR20": "Wilder 20-day ATR on completed UTC daily trade OHLC, available only after the day closes; missing day makes ATR-dependent cell unavailable",
        "rank_population": "unless explicitly cross-sectional, qXX is a symbol-local empirical percentile fit only on the registered training interval; RS, PIT liquidity, breadth and dispersion ranks are cross-sectional among same-timestamp PIT-eligible members with average ranks for ties divided by member count",
        "A1_episode": "first side-signed impulse-return symbol-local percentile crossing after rearm, followed immediately by a contiguous base of the registered duration; base contraction is base 5m realized-vol/prior immediately preceding equal-duration realized-vol and smoothness uses the base; confirmation is close beyond the frozen impulse extreme after the base; rearm requires impulse percentile below q50",
        "A1_identity": "SHA-256(family,symbol,side,impulse_start,impulse_end,base_start,base_end,confirmation_ts); one event per definition/episode; equal-event then equal-market-day aggregation",
        "prior_high": "previous maximum completed daily high over the registered lookback excluding the decision day; proximity=(close-prior_high)/ATR20; reclaim is first completed 5m close back beyond the frozen prior-high level",
        "relative_strength": "symbol signed return minus BTC signed return over the same registered lookback, ranked cross-sectionally among PIT-eligible members at decision",
        "breadth_dispersion": "share of PIT-eligible members with positive 20d return and cross-sectional MAD of 20d returns, respectively; both use the same decision snapshot",
        "BTC_ETH_context": "separate continuous BTC and ETH 20d/60d trend ranks, 20d realized-vol ranks, and 60d close-to-rolling-maximum drawdowns; no composite opaque label",
        "A3_breakout": "first completed 5m close beyond the prior registered-lookback high (long) or low (short), excluding the decision bar, after a completed bar back inside the level; breakout magnitude is abs(close-level)/ATR20 and breakout_rank_min is its symbol-local training percentile",
        "A3_starter": "registered starter fraction enters at the first authorized 5m open after confirmation; starter exists for every A3 address",
        "A3_retest_add": "within the registered retest window, price close may approach/cross the frozen breakout level by no more than retest_depth_ATR, then must complete a close back beyond it; add enters next authorized 5m open and is a separate leg; add_fraction<=1-starter_fraction",
        "confirmation": "one_close is first qualifying completed close; two_closes requires two consecutive qualifying closes; bounded delay waits exactly 15m after first qualifying close and requires the condition still true",
        "rank_ties_and_missingness": "Type-7 numeric thresholds; weak percentile (# observations strictly below + # equal)/N; exact threshold equality passes >= or <= as stated; missing, infinite, collapsed (<20 unique), noncontiguous or unavailable feature observations fail closed for that cell/event",
        "named_parent_contexts": {"none": "multiplier 1", "prior_high_RS": "permission requiring both side-aligned prior-high proximity percentile and side-aligned BTC-relative-strength percentile >=0.50, using the parent's registered lookbacks", "BTC_ETH": "permission requiring arithmetic mean of side-aligned BTC and ETH 20d trend percentiles >=0.50", "breadth_dispersion": "linear multiplier=clip(2*(0.5*side-aligned 20d breadth percentile+0.5*(1-20d dispersion percentile))-1,0,1)", "funding_veto": "multiplier 0 when side-signed trailing-24h payer-burden percentile>=0.90, otherwise 1"},
        "A2_component_aggregation": "build an ordered component list: proximity and RS use their percentile when value=continuous, binary 1/0 for p>=the listed qXX, and are omitted for none; reclaim is binary 1/0 first-close or its side-aligned continuous-distance percentile and is omitted for none; BTC/ETH contributes the selected side-aligned trend, inverse-volatility or inverse-drawdown percentile; breadth contributes side-aligned breadth percentile and dispersion contributes inverse dispersion percentile. permission requires every enabled component>=0.50; linear_size_0_to_1 is the product of clip(2*p-1,0,1); tercile_size applies 0/0.5/1 to the minimum enabled component at <1/3, [1/3,2/3), >=2/3. No enabled component makes the overlay an explicit parent-only component-null.",
        "time_exit": "first authorized 5m open at or after entry_ts plus registered duration, maximum lookup delay 10m; every structural/stop/target/trail address also has a hard 10d time cap",
        "structural_exit": "A1 base_failure or A3 breakout_failure triggers on first completed 5m close through the frozen structure against the position; exit at next authorized open",
        "ATR_stop_and_trail": "initial stop is registered ATR20 multiple from fill; trailing stop updates only at completed 5m closes from best completed close; close-triggered exit occurs at next authorized open",
        "fixed_target": "target is registered R multiple of the initial ATR stop distance; target is invalid when no ATR stop exists; trigger requires a completed 5m close beyond target and exits at next authorized open",
        "simultaneous_exit_priority": ["structural_or_ATR_stop", "trail", "fixed_target", "signal_reversal", "time"],
        "entry_exit_missingness": "missing executable open, lookup delay >10m, lifecycle invalidity or fold crossing rejects the event before outcome; no forward fill or artificial close",
        "gap_and_fill": "all triggers use completed 5m trade closes, never intrabar high/low; execution is the next authorized trade open, so any gap is represented by that open and no stop/target level-price fill is fabricated",
        "non_overlap": "definition-local and symbol-local chronological acceptance by actual executable exit_ts; separate definitions remain separate attempts and no combined portfolio claim is made",
        "costs_and_funding": "apply 14bps base or 32bps stress once per completed round trip per leg, plus exact funding accrued over each leg's actual holding interval under both boundary alignments and nonpositive registered gap allowance",
        "A3_leg_accounting": "one accepted breakout episode is one parent event and one metric observation. The starter is always present and its entry, actual exit, gross, 14/32bps cost, exact funding and allowances are computed independently. An add may exist only while the starter remains open; a retest/reclaim after starter_exit_ts is unavailable_add_after_parent_exit. A present add has its own entry, actual exit, cost and funding under the same exit type and frozen structural levels, with any time horizon starting at add_entry_ts. base_parent=starter_fraction*base_starter+add_fraction*base_add and stress_parent analogously; unavailable add is zero exposure/PnL. parent_exit_ts=max leg exits when add exists, else starter_exit_ts. Definition-symbol non-overlap accepts the next parent only when next_parent_entry_ts>=prior_parent_exit_ts and never runs independently by leg. Trade count counts parents. Median, concentration, bootstrap and fold metrics assign the one weighted parent net to parent symbol, starter-entry UTC day/year and parent ID. Occupancy is sum(starter_fraction*starter holding seconds+add_fraction*add holding seconds)/(eligible symbols*interval seconds). Opportunity frequency is 30*accepted parents/eligible days. Separate leg ledgers are reconciliation and incremental-add evidence only, never extra trades.",
        "funding_context": "use only signed funding observations whose publication/payment timestamp is <=decision_ts; state is side-signed cumulative payer burden over trailing 24h, transformed to symbol-local training percentile; veto/cap never uses the next unpublished funding value",
        "invalid_cell_rules": ["add_fraction>1-starter_fraction", "fixed_target_R!=none without ATR_stop", "lookback or ATR coverage gate fails", "three-way interaction", "duplicate canonical address"],
    },
    "families": family_spaces,
    "new_attempt_budget": {"A4_TSMOM_V7": 864, "A1_COMPRESSION_V2": 1040, "A2_PRIOR_HIGH_RS_CONTEXT_V1": 704, "A3_STARTER_RETEST_V3": 704, "KDA02B_SURVIVOR_ADJUDICATION_V1": 209, "total": 3521},
    "conditional_survivor_control_address_budget": 800,
    "maximum_programme_economic_or_control_addresses": 4321,
    "historical_multiplicity": "all prior family attempts inherited; incomplete legacy counts remain explicitly unknown and are not reset",
    "forbidden": ["event sampling", "event caps", "alphabetical/static symbol selection", "manual winner selection", "hidden variants", "outcome-derived thresholds", "same-sample rescue", "KDA02C retuning", "KDX01 exact-translation reopen", "C17", "Capitalcom", "protected outcomes"],
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def valid_config(family: str, config: dict[str, object]) -> bool:
    if family == "A3_STARTER_RETEST_V3" and float(config["add_fraction"]) > 1.0 - float(config["starter_fraction"]):
        return False
    if family in {"A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"}:
        target = config.get("fixed_target_R")
        if target != "none" and not str(config.get("exit", "")).startswith("ATR_stop_"):
            return False
    return True


def map_point(axes: dict[str, list[object]], point: list[float]) -> dict[str, object]:
    result = {}
    for (key, values), number in zip(axes.items(), point):
        result[key] = values[min(len(values) - 1, int(number * len(values)))]
    return result


def config_hash(family: str, config: dict[str, object]) -> str:
    return hashlib.sha256(canonical_bytes({"family_id": family, "config": config})).hexdigest()


def enumerate_anchor(axes: dict[str, list[object]], index: int) -> dict[str, object]:
    # Mixed-radix enumeration with an odd per-axis phase produces exact, reviewable anchors.
    config = {}
    quotient = index
    for axis_index, (key, values) in enumerate(axes.items()):
        config[key] = values[(quotient + 2 * axis_index) % len(values)]
        quotient //= len(values)
    return config


def neighbours(axes: dict[str, list[object]], center: dict[str, object]):
    keys = list(axes)
    for i, key in enumerate(keys):
        values = axes[key]; position = values.index(center[key])
        for delta in (-1, 1):
            other = position + delta
            if 0 <= other < len(values):
                candidate = dict(center); candidate[key] = values[other]
                yield candidate, [key]
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            lvals, rvals = axes[left], axes[right]
            lp, rp = lvals.index(center[left]), rvals.index(center[right])
            for ld, rd in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                if 0 <= lp + ld < len(lvals) and 0 <= rp + rd < len(rvals):
                    candidate = dict(center); candidate[left] = lvals[lp + ld]; candidate[right] = rvals[rp + rd]
                    yield candidate, [left, right]


registry = []
family_seed_offsets = {"A4_TSMOM_V7": 0, "A1_COMPRESSION_V2": 10000,
                       "A2_PRIOR_HIGH_RS_CONTEXT_V1": 20000, "A3_STARTER_RETEST_V3": 30000}
explicit_anchors = {
    "A4_TSMOM_V7": [
        ({"signal_estimator": "signed_return", "lookback_days": 20, "vol_window_days": 20, "annualized_vol_target": 0.20, "rebalance": "1d", "direction": "long_flat", "path_smoothness_quantile_min": "none", "PIT_liquidity_top_n": 20, "exit": "time_1d", "context_overlay": "none"}, "source_prior_partial_anchor_and_single_signal_null", "tsmom_v6_002 supplies lookback=20d, vol_window=20d, target=.2, daily rebalance, long_flat, hold=1d; v6 4h bar clock, source universe, BTC/ETH regime, funding cap and 150bps stop are deliberately not inherited and are replaced by this campaign's native-5m clock, PIT top20, no context, exact funding contract and time exit"),
        ({"signal_estimator": "signed_return", "lookback_days": 60, "vol_window_days": 20, "annualized_vol_target": 0.20, "rebalance": "1d", "direction": "long_flat", "path_smoothness_quantile_min": "none", "PIT_liquidity_top_n": 20, "exit": "time_1d", "context_overlay": "none"}, "source_prior_partial_anchor_and_single_signal_null", "tsmom_v6_018 supplies lookback=60d, vol_window=20d, target=.2, daily rebalance, long_flat, hold=1d; v6 4h bar clock, source universe, BTC/ETH regime, funding cap and 150bps stop are deliberately not inherited and are replaced by this campaign's native-5m clock, PIT top20, no context, exact funding contract and time exit"),
        ({"signal_estimator": "signed_return", "lookback_days": 20, "vol_window_days": 20, "annualized_vol_target": "none", "rebalance": "1d", "direction": "long_flat", "path_smoothness_quantile_min": "none", "PIT_liquidity_top_n": 20, "exit": "time_1d", "context_overlay": "none"}, "component_null_anchor", "A4 anchor 0001 with volatility scaling removed"),
        ({"signal_estimator": "equal_weight_ensemble", "lookback_days": 20, "vol_window_days": 20, "annualized_vol_target": 0.20, "rebalance": "1d", "direction": "long_flat", "path_smoothness_quantile_min": "none", "PIT_liquidity_top_n": 20, "exit": "time_1d", "context_overlay": "none"}, "mechanism_anchor", "V7 ensemble paired exactly against A4 anchor 0001"),
    ],
    "A1_COMPRESSION_V2": [
        ({"impulse_window": "1d", "impulse_rank_min": "q80", "base_duration": "12h", "contraction_rank_max": "q40", "smoothness_rank_min": "q60", "confirmation": "one_close", "direction": "long", "PIT_liquidity_top_n": 20, "exit": "time_5d", "fixed_target_R": "none", "context_overlay": "none"}, "mechanism_anchor", "impulse_base_compression"),
        ({"impulse_window": "1d", "impulse_rank_min": "q80", "base_duration": "12h", "contraction_rank_max": "none", "smoothness_rank_min": "q60", "confirmation": "one_close", "direction": "long", "PIT_liquidity_top_n": 20, "exit": "time_5d", "fixed_target_R": "none", "context_overlay": "none"}, "component_null_anchor", "contraction_removed"),
        ({"impulse_window": "1d", "impulse_rank_min": "q80", "base_duration": "12h", "contraction_rank_max": "q40", "smoothness_rank_min": "none", "confirmation": "one_close", "direction": "long", "PIT_liquidity_top_n": 20, "exit": "time_5d", "fixed_target_R": "none", "context_overlay": "none"}, "component_null_anchor", "smoothness_removed"),
        ({"impulse_window": "1d", "impulse_rank_min": "q80", "base_duration": "12h", "contraction_rank_max": "none", "smoothness_rank_min": "none", "confirmation": "one_close", "direction": "long", "PIT_liquidity_top_n": 20, "exit": "time_5d", "fixed_target_R": "none", "context_overlay": "none"}, "component_null_anchor", "price_only_impulse_base"),
    ],
    "A2_PRIOR_HIGH_RS_CONTEXT_V1": [
        ({"parent_family": "A1_COMPRESSION_V2", "prior_high_lookback_days": 60, "proximity_rank": "continuous", "reclaim_state": "continuous_distance", "RS_lookback_days": 20, "RS_rank": "none", "overlay_action": "permission", "BTC_ETH_context": "none", "breadth_dispersion": "none", "parent_binding": "A1_COMPRESSION_V2:anchor_ablation:0001"}, "component_anchor", "prior_high_only"),
        ({"parent_family": "A1_COMPRESSION_V2", "prior_high_lookback_days": 60, "proximity_rank": "none", "reclaim_state": "none", "RS_lookback_days": 20, "RS_rank": "continuous", "overlay_action": "permission", "BTC_ETH_context": "none", "breadth_dispersion": "none", "parent_binding": "A1_COMPRESSION_V2:anchor_ablation:0001"}, "component_anchor", "relative_strength_only"),
        ({"parent_family": "A1_COMPRESSION_V2", "prior_high_lookback_days": 60, "proximity_rank": "none", "reclaim_state": "none", "RS_lookback_days": 20, "RS_rank": "none", "overlay_action": "permission", "BTC_ETH_context": "none", "breadth_dispersion": "none", "parent_binding": "A1_COMPRESSION_V2:anchor_ablation:0001"}, "component_null_anchor", "parent_only"),
        ({"parent_family": "A3_STARTER_RETEST_V3", "prior_high_lookback_days": 60, "proximity_rank": "continuous", "reclaim_state": "continuous_distance", "RS_lookback_days": 20, "RS_rank": "continuous", "overlay_action": "linear_size_0_to_1", "BTC_ETH_context": "none", "breadth_dispersion": "none", "parent_binding": "A3_STARTER_RETEST_V3:anchor_ablation:0001"}, "mechanism_anchor", "prior_high_plus_RS"),
    ],
    "A3_STARTER_RETEST_V3": [
        ({"breakout_lookback_days": 20, "breakout_rank_min": "q80", "confirmation": "one_close", "starter_fraction": 0.5, "retest_window": "1d", "retest_depth_ATR": 0.5, "add_fraction": 0.5, "add_requires_reclaim": True, "PIT_liquidity_top_n": 20, "exit": "time_5d", "fixed_target_R": "none", "context_overlay": "none"}, "mechanism_anchor", "starter_plus_separate_retest_add"),
        ({"breakout_lookback_days": 20, "breakout_rank_min": "q80", "confirmation": "one_close", "starter_fraction": 1.0, "retest_window": "1d", "retest_depth_ATR": 0.5, "add_fraction": 0.0, "add_requires_reclaim": True, "PIT_liquidity_top_n": 20, "exit": "time_5d", "fixed_target_R": "none", "context_overlay": "none"}, "component_null_anchor", "starter_only"),
        ({"breakout_lookback_days": 20, "breakout_rank_min": "q80", "confirmation": "two_closes", "starter_fraction": 0.5, "retest_window": "1d", "retest_depth_ATR": 0.5, "add_fraction": 0.5, "add_requires_reclaim": True, "PIT_liquidity_top_n": 20, "exit": "time_5d", "fixed_target_R": "none", "context_overlay": "none"}, "mechanism_anchor", "two_close_confirmation"),
        ({"breakout_lookback_days": 20, "breakout_rank_min": "q80", "confirmation": "one_close", "starter_fraction": 0.5, "retest_window": "1d", "retest_depth_ATR": 0.5, "add_fraction": 0.5, "add_requires_reclaim": True, "PIT_liquidity_top_n": 20, "exit": "breakout_failure", "fixed_target_R": "none", "context_overlay": "none"}, "mechanism_anchor", "structural_exit"),
    ],
}
for family in family_spaces[:4]:
    family_id = family["family_id"]; axes = family["axes"]
    reserved_anchors = explicit_anchors[family_id]
    seen: set[str] = {config_hash(family_id, row[0]) for row in reserved_anchors}
    lane_rows: dict[str, list[dict[str, object]]] = {"screening": [], "anchor_ablation": [], "refinement": []}
    engine = qmc.Sobol(d=len(axes), scramble=True, seed=20260721 + family_seed_offsets[family_id])
    screening_index = 0
    while len(lane_rows["screening"]) < family["screening"]:
        for point in engine.random(256).tolist():
            config = map_point(axes, point)
            if family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                config["parent_binding"] = ("A1_COMPRESSION_V2:anchor_ablation:0001" if config["parent_family"] == "A1_COMPRESSION_V2" else "A3_STARTER_RETEST_V3:anchor_ablation:0001")
            address = config_hash(family_id, config)
            if valid_config(family_id, config) and address not in seen:
                seen.add(address); screening_index += 1
                lane_rows["screening"].append({"family_id": family_id, "lane": "screening", "lane_index": screening_index, "config": config, "canonical_economic_address_sha256": address, "status": "registered_executable"})
                if len(lane_rows["screening"]) == family["screening"]: break
    anchor_index = candidate_index = 0
    anchors = []
    for config, provenance, source_reference in reserved_anchors:
        if not valid_config(family_id, config): raise RuntimeError(f"invalid explicit anchor: {family_id}")
        anchor_index += 1
        row = {"family_id": family_id, "lane": "anchor_ablation", "lane_index": anchor_index, "config": config, "canonical_economic_address_sha256": config_hash(family_id, config), "status": "registered_executable", "provenance": provenance, "source_or_null_reference": source_reference}
        lane_rows["anchor_ablation"].append(row); anchors.append(row)
    while len(lane_rows["anchor_ablation"]) < family["anchors_ablations"]:
        config = enumerate_anchor(axes, candidate_index); candidate_index += 1
        if family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
            config["parent_binding"] = ("A1_COMPRESSION_V2:anchor_ablation:0001" if config["parent_family"] == "A1_COMPRESSION_V2" else "A3_STARTER_RETEST_V3:anchor_ablation:0001")
        address = config_hash(family_id, config)
        if valid_config(family_id, config) and address not in seen:
            seen.add(address); anchor_index += 1
            row = {"family_id": family_id, "lane": "anchor_ablation", "lane_index": anchor_index, "config": config, "canonical_economic_address_sha256": address, "status": "registered_executable", "provenance": "registered_design_anchor_or_ablation"}
            lane_rows["anchor_ablation"].append(row); anchors.append(row)
    refinement_index = 0
    for center in anchors:
        for config, changed_axes in neighbours(axes, center["config"]):
            if family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                config["parent_binding"] = center["config"]["parent_binding"]
            address = config_hash(family_id, config)
            if valid_config(family_id, config) and address not in seen:
                seen.add(address); refinement_index += 1
                lane_rows["refinement"].append({"family_id": family_id, "lane": "refinement", "lane_index": refinement_index, "config": config, "canonical_economic_address_sha256": address, "status": "registered_conditional", "execute_only_if_center_plateau_passes": f"{family_id}:anchor_ablation:{int(center['lane_index']):04d}", "changed_axes": changed_axes})
                if refinement_index == family["refinement_reserve"]: break
        if refinement_index == family["refinement_reserve"]: break
    if refinement_index != family["refinement_reserve"]:
        raise RuntimeError(f"insufficient pre-registered refinement neighbours: {family_id}")
    for lane in ("screening", "anchor_ablation", "refinement"):
        for row in lane_rows[lane]:
            row["attempt_id"] = f"{family_id}:{lane}:{int(row['lane_index']):04d}"
            registry.append(row)

kda_variants = ["identity_replay", "price_only", "OI_removed", "liquidation_removed", "generic_structure_control", "stress_cost_32bps", "funding_start_alignment", "funding_end_alignment", "funding_zero", "entry_delay_15m", "entry_delay_60m"]
for cell in family_spaces[4]["exact_addresses"]["Stage20_unique_cells"]:
    for variant_index, variant in enumerate(kda_variants, 1):
        config = {"stage20_cell_id": cell, "adjudication_variant": variant}
        registry.append({"family_id": "KDA02B_SURVIVOR_ADJUDICATION_V1", "lane": "fixed_adjudication", "lane_index": variant_index, "attempt_id": f"KDA02B_SURVIVOR_ADJUDICATION_V1:{cell}:{variant}", "config": config, "canonical_economic_address_sha256": config_hash("KDA02B_SURVIVOR_ADJUDICATION_V1", config), "status": "registered_executable"})

if len(registry) != 3521 or len({row["canonical_economic_address_sha256"] for row in registry}) != 3521:
    raise RuntimeError("full registered configuration identity reconciliation failed")
registry_path = HERE / "FULL_REGISTERED_CONFIGURATION_REGISTRY.jsonl"
registry_path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in registry))
discovery["full_configuration_registry"] = {"path": registry_path.name, "sha256": sha(registry_path), "rows": len(registry), "unique_economic_addresses": len(registry), "generator": {"engine": "scipy.stats.qmc.Sobol", "scipy_version": "1.16.1", "scramble": True, "seed": 20260721, "family_seed_offsets": family_seed_offsets, "batch_size": 256, "canonical_serialization": "UTF-8 JSON sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False", "registration_time": "before any campaign outcome access"}}
discovery["KDA02B_exact_adjudication_variants"] = kda_variants
write_json("REGISTERED_DISCOVERY_SPACE.json", discovery)


context = {
    "schema": "context_and_ablation_registry_v1", "campaign_id": CAMPAIGN,
    "context_definition": "Each context is a separately addressed component overlay or permission/sizing rule. No opaque regime label is allowed.",
    "contexts": [
        {"id": "BTC_ETH_TREND", "inputs": "completed BTC and ETH 20d/60d continuous trend ranks", "actions": ["none", "permission", "linear_size"], "ablations": ["BTC_only", "ETH_only", "both_removed"]},
        {"id": "BTC_ETH_VOL_DRAWDOWN", "inputs": "completed 20d realized-vol ranks and causal rolling drawdowns", "actions": ["none", "permission", "linear_size"], "ablations": ["vol_removed", "drawdown_removed"]},
        {"id": "PIT_BREADTH_DISPERSION", "inputs": "eligible current-roster/lifecycle-capped PIT members using same decision timestamp", "actions": ["none", "permission", "linear_size"], "ablations": ["breadth_removed", "dispersion_removed"]},
        {"id": "LAGGED_LIQUIDITY_LISTING_AGE", "inputs": "one-day-lagged 30d median quote notional rank and causal listing age", "actions": ["top10", "top20", "top40"], "ablations": ["topN_constant", "listing_age_removed"]},
        {"id": "EXACT_FUNDING", "inputs": "signed exact start/end alignment and rankable-only missing-hour allowance", "actions": ["no_veto", "veto_positive_payer_q90", "linear_cap_q90"], "ablations": ["zero_funding", "no_veto"]},
    ],
    "matched_comparisons": "same parent events, fold, symbol, month, liquidity bin and executable timing; context on versus component removed",
    "outcome_use": "contexts cannot redefine a parent family after outer results and cannot be used to form a post-hoc subgroup claim",
}
write_json("CONTEXT_AND_ABLATION_REGISTRY.json", context)


def month_starts(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m == 13: y, m = y + 1, 1


def iso(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}-01T00:00:00Z"


quarters = [(2024, 1), (2024, 4), (2024, 7), (2024, 10), (2025, 1), (2025, 4), (2025, 7), (2025, 10)]
outer_folds = []
for y, m in quarters:
    ey, em = (y + 1, 1) if m == 10 else (y, m + 3)
    prior_month = m - 2
    py = y
    if prior_month <= 0: prior_month += 12; py -= 1
    inners = []
    for iy, im in month_starts(2023, 7, py, prior_month):
        ny, nm = (iy + 1, 1) if im == 12 else (iy, im + 1)
        inners.append({"inner_fold_id": f"M_{iy:04d}{im:02d}", "training_start": "2023-01-01T00:00:00Z",
                       "training_latest_exit_exclusive": (datetime(iy, im, 1, tzinfo=timezone.utc) - timedelta(days=10)).isoformat().replace("+00:00", "Z"),
                       "validation_start": iso(iy, im), "validation_end_exclusive": iso(ny, nm),
                       "purge_before_validation_days": 10, "event_rule": "decision, entry and actual exit inside validation month; otherwise reject"})
    q = f"{y}Q{((m - 1)//3)+1}"
    outer_folds.append({"outer_fold_id": q, "development_start": "2023-01-01T00:00:00Z",
                        "inner_folds": inners, "outer_evaluation_start": iso(y, m),
                        "outer_evaluation_end_exclusive": iso(ey, em), "purge_days": 10, "embargo_days": 10,
                        "boundary": "decision, entry and actual exit inside quarter; crossing events reject; no artificial close",
                        "exposure_class": "program_exposed_historical", "no_backward_transfer": True})
fold_map = {"schema": "core_campaign_inner_outer_fold_map_v1", "campaign_id": CAMPAIGN,
            "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)", "maximum_horizon_days": 10,
            "warmup": "January-June 2023 is feature warmup/training only; first inner validation is July 2023",
            "outer_folds": outer_folds, "families_using_map": ["A4_TSMOM_V7", "A1_COMPRESSION_V2", "A2_PRIOR_HIGH_RS_CONTEXT_V1", "A3_STARTER_RETEST_V3"],
            "KDA02B_adjudication": "reuse exact Stage20 nine-fold identities and frozen outputs; controls/forensics only",
            "empty_fold_rule": "persist empty/unavailable observation and include it in all stability denominators; never silently omit"}
write_json("INNER_OUTER_FOLD_MAP.json", fold_map)


selection = {
    "schema": "selection_and_plateau_contract_v1", "campaign_id": CAMPAIGN,
    "development_metrics": {"primary": "mean of UTC-market-day means of base_net_bps", "trade_count": "accepted actual-exit-nonoverlapping legs/events", "market_day_clusters": "unique UTC entry dates", "UTC_hour_clusters": "unique UTC entry YYYY-MM-DDTHH", "median_trade_bps": "Type-7 q50 of event/weighted-address base_net_bps", "inner_fold_values": "each inner fold's equal-day primary metric; empty/unavailable is persisted and treated as negative infinity for ordering and quantiles, never omitted", "median_inner_fold": "Type-7 q50 over the complete registered inner-fold vector under empty=-infinity", "p20_inner_fold": "Type-7 q20 over the same complete vector", "stress_net": "mean of UTC-day means under 32bps and stress gap funding", "concentration": "for symbol, UTC day and UTC year separately: max absolute grouped sum / sum absolute event base_net_bps; report all three and their maximum", "opportunity_frequency": "30*accepted_event_count/eligible_calendar_days", "occupancy": "sum actual leg holding seconds/(eligible_symbols*evaluation_interval_seconds)", "left_tail_utility": "negative maximum drawdown of chronologically ordered UTC-day mean base-net cumulative sum", "required": ["trade_count", "market_day_clusters", "UTC_hour_clusters", "median_trade_bps", "median_inner_fold", "p20_inner_fold", "5pct day-cluster bootstrap lower", "stress_net", "symbol/day/year concentration", "opportunity_frequency", "occupancy", "left_tail_utility"], "bootstrap": "IID resample complete UTC-day means with replacement, same length, numpy PCG64 seed=first unsigned 64 bits SHA256(20260721,canonical_address,fold_id), 5000 replicates; lower bound is Type-7 q05", "quantiles": "Hyndman-Fan Type 7 / numpy method=linear", "bootstrap_seed": 20260721, "bootstrap_replicates": 5000},
    "hard_integrity": ["all PIT/timestamp/lifecycle checks pass", "all attempted folds represented including empty", "actual-exit non-overlap", "cost/funding reconciliation", "no protected rows", "no hidden attempt"],
    "Gower_distance": "equal weight per registered config field. Listed numeric/ordinal fields map to their zero-based listed-level index/(levels-1); categorical fields contribute 0 if exact equal else 1; singleton fields contribute 0. Distance is the arithmetic mean over every family axis, including inactive none values; no outcome-derived weights.",
    "complexity": "base 1 for the primary signal; +1 for equal_weight_ensemble; +1 for each active non-none context; +1 for each active secondary feature (smoothness, contraction, reclaim, RS, BTC/ETH, breadth or dispersion); +1 for confirmation other than one_close; +1 for non-time structural/ATR/trail exit; +1 for fixed_target_R; +1 for A3 add_fraction>0; +1 for sizing other than fixed unit/permission. KDA02B inherits the exact Stage20 complexity. Ties use lower complexity.",
    "stable_region": {"distance": "exact Gower_distance above", "radius": 0.15, "minimum_distinct_cells": 5, "minimum_axes_varied": 2, "positive_base_fraction": 0.60, "region_median_base_net_gt_bps": 0, "region_median_stress_net_gt_bps": -18, "minimum_nonempty_inner_fold_fraction": 0.75, "medians": "Type-7 q50; empty cells are not positive and remain in denominators"},
    "beam": {"maximum_per_family_outer_fold": 5, "eligibility": ["stable_region", "accepted_trades>=30", "market_days>=20", "finite base metric", "base_net>0"], "lexicographic_order": ["integrity", "plateau_support_count_desc", "cluster_bootstrap_lower_desc", "p20_inner_fold_desc", "stress_net_desc", "opportunity_frequency_desc", "complexity_asc", "canonical_address_asc"], "manual_selection": False},
    "freeze": "complete economic addresses, transformations, directions, context, entry, exit, costs, funding, seed and multiplicity are atomically frozen before each outer reader opens",
    "outer": "forward-only quarterly; no outer result transfers backward; rolling replication reported by family and complete address",
    "route_caps": {"program_exposed_historical": True, "independent_validation_claim": False, "deployment_claim": False},
}
write_json("SELECTION_AND_PLATEAU_CONTRACT.json", selection)


controls = {
    "schema": "control_and_forensic_contract_v1", "campaign_id": CAMPAIGN,
    "control_address_template": "SHA256(canonical JSON {campaign_id,parent_economic_address,outer_fold_id,control_id,seed}); generate for all 4 families x 8 folds x 5 possible beam slots x 5 family-specific controls before any development outcome, mark unselected parents unavailable_parent_not_selected",
    "conditional_control_address_count": 800,
    "matched_pseudo_event_allocator": "sort parent events by canonical parent event ID ascending; maintain a used-decision_ts set per control address; for each parent in that order, enumerate eligible completed 5m decisions matched on symbol, UTC month, side, hour-of-week and one-day-lagged 30d median quote-notional decile, exclude decisions within plus or minus the parent's maximum registered holding interval and every decision already used, then choose the minimum tuple (SHA256(seed,symbol,month,side,hour_of_week,liquidity_decile,decision_ts),decision_ts). Selection is without replacement. No eligible decision is persisted as explicit unavailable; no parent is silently omitted.",
    "family_control_slots": {
        "A4_TSMOM_V7": [
            {"slot": 1, "control_id": "A4_SIGN_PERMUTED_MAIN_NULL", "definition": "compute the complete scalar-estimator sign vector before return access; within symbol x UTC-year, sort decisions ascending and permute signs including zeros without replacement with numpy PCG64 seeded by first unsigned big-endian 64 bits SHA256(20260721,control_id,symbol,year); preserve timestamp magnitude, volatility, universe, context, entry and exit"},
            {"slot": 2, "control_id": "A4_GENERIC_SIGNED_RETURN", "definition": "set signal_estimator=signed_return, annualized_vol_target=none, path_smoothness_quantile_min=none and context_overlay=none; preserve lookback, direction, rebalance, universe and exit"},
            {"slot": 3, "control_id": "A4_VOL_SCALING_REMOVED", "definition": "set annualized_vol_target=none only"},
            {"slot": 4, "control_id": "A4_PATH_COMPONENT_REMOVED", "definition": "set path_smoothness_quantile_min=none; if estimator is equal_weight_ensemble, remove breakout_distance_rank and average the remaining signed_return/vol and ema_slope/vol under unchanged winsorization and MAD rules"},
            {"slot": 5, "control_id": "A4_CONTEXT_REMOVED", "definition": "set context_overlay=none only"}],
        "A1_COMPRESSION_V2": [
            {"slot": 1, "control_id": "A1_MATCHED_PSEUDO_EVENT_MAIN_NULL", "definition": "use matched_pseudo_event_allocator"},
            {"slot": 2, "control_id": "A1_PRICE_ONLY_IMPULSE", "definition": "set contraction_rank_max=none, smoothness_rank_min=none and context_overlay=none; preserve impulse, base clock, direction, confirmation and execution"},
            {"slot": 3, "control_id": "A1_CONTRACTION_REMOVED", "definition": "set contraction_rank_max=none only"},
            {"slot": 4, "control_id": "A1_SMOOTHNESS_REMOVED", "definition": "set smoothness_rank_min=none only"},
            {"slot": 5, "control_id": "A1_CONTEXT_REMOVED", "definition": "set context_overlay=none only"}],
        "A2_PRIOR_HIGH_RS_CONTEXT_V1": [
            {"slot": 1, "control_id": "A2_CONTEXT_PERMUTED_MAIN_NULL", "definition": "permute the complete ordered context-component vector within symbol x UTC-month x lagged-liquidity-decile using registered PCG64 hash seed; preserve parent events"},
            {"slot": 2, "control_id": "A2_PARENT_ONLY", "definition": "set proximity_rank=none, reclaim_state=none, RS_rank=none, BTC_ETH_context=none and breadth_dispersion=none"},
            {"slot": 3, "control_id": "A2_PRIOR_HIGH_REMOVED", "definition": "set proximity_rank=none and reclaim_state=none only"},
            {"slot": 4, "control_id": "A2_RS_REMOVED", "definition": "set RS_rank=none only"},
            {"slot": 5, "control_id": "A2_EXTERNAL_CONTEXT_REMOVED", "definition": "set BTC_ETH_context=none and breadth_dispersion=none only"}],
        "A3_STARTER_RETEST_V3": [
            {"slot": 1, "control_id": "A3_RETEST_TIME_PERMUTED_MAIN_NULL", "definition": "within symbol x UTC-quarter x side, encode each parent as add_lag_5m_bars or NO_ADD; sort parent IDs lexically and permute without replacement with PCG64 seeded by first unsigned big-endian 64 bits SHA256(20260721,control_id,symbol,quarter,side); a numeric lag is usable only when the original frozen retest/reclaim predicate holds at that lag, otherwise persist unavailable_control"},
            {"slot": 2, "control_id": "A3_STARTER_ONLY", "definition": "set starter_fraction=1.0 and add_fraction=0.0; preserve breakout and exit"},
            {"slot": 3, "control_id": "A3_MATCHED_PSEUDO_EVENT", "definition": "use matched_pseudo_event_allocator with exact starter/add execution grammar"},
            {"slot": 4, "control_id": "A3_CONFIRMATION_REMOVED", "definition": "set confirmation=one_close only; duplicate addresses are unavailable_duplicate_address"},
            {"slot": 5, "control_id": "A3_CONTEXT_REMOVED", "definition": "set context_overlay=none only"}]
    },
    "duplicate_control_rule": "if a transformed control is economically identical to its parent or another control, persist unavailable_duplicate_address and do not count it as independent control evidence",
    "canonical_serialization": "UTF-8 JSON sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False",
    "control_seed": 20260721,
    "forensics": ["leave-one-symbol-out", "leave-one-month-out", "leave-one-symbol-month-out", "remove top absolute contributor", "base 14bps and stress 32bps", "funding start/end/zero/gap partitions", "entry delays 5m/15m/60m", "parameter-neighborhood stability"],
    "forensic_address_rule": "forensics are deterministic recomputations of a frozen survivor ledger, not new candidate definitions: SHA256(parent_address,forensic_id,excluded_key_or_scenario). All symbols/months/symbol-months present in the ledger are enumerated lexically; none is sampled or omitted.",
    "forensic_recomputation_semantics": {
        "leave_one_group_out": "for each symbol, UTC month, and symbol-month key present in the frozen ledger, sort keys lexically, remove exactly that key, and recompute every development metric named in SELECTION_AND_PLATEAU_CONTRACT.json over the remaining complete ledger; persist empty/unavailable rather than dropping a result",
        "top_absolute_contributor": "group frozen event base_net_bps by symbol, UTC month, and symbol-month separately; within each grouping choose max absolute grouped sum with lexical key as tie-break, remove it, and recompute every registered development metric; report all three removals",
        "cost_scenarios": "recompute gross, exactly 14bps or 32bps pre-funding round-trip cost, registered gap allowance, funding and resulting net over the unchanged event ledger",
        "funding_scenarios": "start_alignment means min(0,signed_start_alignment)+registered base gap allowance; end_alignment means min(0,signed_end_alignment)+registered base gap allowance; zero means exactly zero funding and zero gap allowance; gap_stress means adverse min(0,start,end)+the registered stress gap allowance. Recompute net and every registered metric for each scenario",
        "entry_delays": "for each of 5m, 15m and 60m, entry is the first authorized open at or after original decision_ts plus the delay; reject if no open, lookup exceeds 10m, lifecycle is invalid, or actual exit would cross the fold. The full registered horizon begins at the delayed fill, and costs plus exact funding are recomputed over that stressed actual interval",
        "parameter_neighborhood": "use only the pre-registered refinement rows whose execute_only_if_center_plateau_passes resolves to the survivor's exact attempt_id; do not synthesize or inspect any unregistered neighbor"
    },
    "KDA02B": {"scope": "19 exact Stage20 selected cell IDs x exactly 11 economically distinct registered variants", "variants": ["identity_replay", "price_only", "OI_removed", "liquidation_removed", "generic_structure_control", "stress_cost_32bps", "funding_start_alignment", "funding_end_alignment", "funding_zero", "entry_delay_15m", "entry_delay_60m"], "variant_semantics": {"identity_replay": "exact Stage20 economic address, 14bps cost and adverse min(0,start,end) funding", "price_only": "generate onsets from the cell's exact price sign/magnitude/axis predicate only; preserve branch, horizon, entry, exit, non-overlap and costs", "OI_removed": "remove only the OI predicate from the exact cell; preserve price and liquidation predicates", "liquidation_removed": "remove only the liquidation predicate; preserve exact price and OI predicates", "generic_structure_control": "use matched_pseudo_event_allocator with exact cell side, horizon, entry, exit and cost address", "stress_cost_32bps": "identity event ledger with exactly 32bps pre-funding round-trip", "funding_start_alignment": "identity ledger with funding=min(0,signed_start_alignment)+the same registered base gap allowance; end alignment excluded", "funding_end_alignment": "identity ledger with funding=min(0,signed_end_alignment)+the same registered base gap allowance; start alignment excluded", "funding_zero": "unchanged identity event ledger with exact funding cashflow=0, gap allowance=0, and 14bps pre-funding round-trip cost", "entry_delay_15m": "entry at first authorized open >=original decision+15m", "entry_delay_60m": "entry at first authorized open >=original decision+60m"}, "entry_delay_semantics": "exit horizon runs from stressed fill; missing open, >10m lookup delay, lifecycle invalidity or fold crossing rejects; costs/funding recompute over the stressed actual interval", "allowed": "component controls, execution forensics, cost/funding sensitivity", "prohibited": "threshold rescue, new cell, new direction, new horizon, new context, candidate reselection"},
    "route_rules": {"control_failure": "reject claimed component or cap route; do not retune", "concentration_failure": "cap as concentrated historical observation; no subset rescue", "execution_failure": "execution_sensitive or reject", "no_survivor": "record negative family decision"},
}
control_templates = []
for family_id in ("A4_TSMOM_V7", "A1_COMPRESSION_V2", "A2_PRIOR_HIGH_RS_CONTEXT_V1", "A3_STARTER_RETEST_V3"):
    for outer_fold in [row["outer_fold_id"] for row in outer_folds]:
        for beam_slot in range(1, 6):
            for control_row in controls["family_control_slots"][family_id]:
                selector = {"family_id": family_id, "outer_fold_id": outer_fold, "deterministic_beam_slot": beam_slot, "control_id": control_row["control_id"], "seed": controls["control_seed"]}
                control_templates.append({**selector, "control_template_address_sha256": hashlib.sha256(canonical_bytes(selector)).hexdigest(), "status": "registered_conditional", "parent_resolution": "resolve to the complete address atomically frozen in this deterministic beam slot; if slot absent, unavailable_parent_not_selected"})
control_template_path = HERE / "FULL_CONTROL_TEMPLATE_REGISTRY.jsonl"
control_template_path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in control_templates))
if len(control_templates) != 800 or len({row["control_template_address_sha256"] for row in control_templates}) != 800:
    raise RuntimeError("control template registry reconciliation failed")
controls["control_template_registry"] = {"path": control_template_path.name, "sha256": sha(control_template_path), "rows": 800}
write_json("CONTROL_AND_FORENSIC_CONTRACT.json", controls)


runtime = {
    "schema": "unattended_runtime_and_recovery_contract_v1", "campaign_id": CAMPAIGN,
    "launch": {"one_exact_external_human_approval_required": True, "persistent_detached_supervisor": True, "secure_Telegram_preflight": True, "synthetic_end_to_end_canary": True, "independent_preoutcome_review": True, "atomic_final_authority_hash_audit": True},
    "state": {"transactional_generation_counter": True, "atomic_sharded_artifacts": True, "job_markers_hash_bind_inputs_and_outputs": True, "idempotent_resume": True, "valid_artifacts_never_deleted_or_recomputed": True, "complete_attempt_reconciliation": True},
    "scheduler": {"bounded_lazy": True, "workers": 4, "max_in_flight": 4, "checks_before_each_submission": ["authority hashes", "elapsed wall", "process-tree RSS including root", "output bytes", "free disk", "global/family stop", "Telegram health"]},
    "health_release": ["persistent process verified", "one completed and reconciled real registered partition/cell", "campaign state round-trip verified", "Telegram delivery verified", "first scheduled 30-minute heartbeat delivered"],
    "notifications": {"heartbeat_seconds": 1800, "include": ["phase", "fold", "family", "attempt counts", "resource envelope", "stop/completion"], "exclude": ["partial rankings", "partial candidate outcomes"]},
    "limits": {"wall_seconds": 604800, "aggregate_process_tree_RSS_bytes": 10737418240, "campaign_output_bytes": 25769803776, "minimum_free_disk_bytes": 8589934592, "graceful_stop_seconds": 300},
    "family_isolation": "a registered family stop prevents new submissions for that family; common authority/economic/runtime failures stop globally",
    "mechanical_repair_authority_after_final_approval": {"allowed": "narrow runtime-only patch, tests, independent review, hash refresh and automatic idempotent resume", "conditions": ["no economic/selection/data/funding/execution/protected semantic change", "no partial outcome-guided repair", "packet authority still matches"], "hard_stop": ["semantic change", "protected access", "new acquisition", "unsafe Git", "unresolved common economic defect"]},
    "terminal_administration": ["full attempts/cells/folds reconciliation", "independent postrun review", "non-force Git publication", "approved Drive handoff", "dynamic continuity publication"],
}
write_json("UNATTENDED_RUNTIME_AND_RECOVERY_CONTRACT.json", runtime)


cold_benchmark = json.loads((HERE / "NO_OUTCOME_RUNTIME_BENCHMARK.json").read_text(encoding="utf-8"))
resume_benchmark = json.loads((HERE / "NO_OUTCOME_RUNTIME_RESUME_BENCHMARK.json").read_text(encoding="utf-8"))
bound_stop_benchmark = json.loads((HERE / "NO_OUTCOME_BOUND_STOP_CANARY.json").read_text(encoding="utf-8"))
bound_recovery_benchmark = json.loads((HERE / "NO_OUTCOME_BOUND_STOP_RECOVERY.json").read_text(encoding="utf-8"))
resource = {
    "schema": "resource_projection_v1", "campaign_id": CAMPAIGN,
    "benchmark_type": "full-registry outcome-free transactional supervisor plus inherited runtime evidence", "benchmark_inputs": measurement["benchmark_summary"],
    "synthetic_supervisor_benchmark": {"source_sha256": sha(HERE / "synthetic_campaign_supervisor.py"),
                                       "cold_file_sha256": sha(HERE / "NO_OUTCOME_RUNTIME_BENCHMARK.json"), "cold": cold_benchmark,
                                       "resume_file_sha256": sha(HERE / "NO_OUTCOME_RUNTIME_RESUME_BENCHMARK.json"), "resume": resume_benchmark,
                                       "bound_stop_file_sha256": sha(HERE / "NO_OUTCOME_BOUND_STOP_CANARY.json"), "bound_stop": bound_stop_benchmark,
                                       "bound_recovery_file_sha256": sha(HERE / "NO_OUTCOME_BOUND_STOP_RECOVERY.json"), "bound_recovery": bound_recovery_benchmark},
    "attempts": discovery["new_attempt_budget"], "outer_folds": 8,
    "projection": {"cold_p50_hours": 48, "cold_p90_hours": 120, "hard_wall_hours": 168,
                   "reason": "prior A1 alone used 17.4h/5.11GB; Stage20 provided measured 4-worker sharded throughput; broader 3,521-attempt mixed-family design needs multi-day headroom and cache reuse"},
    "limits": runtime["limits"], "workers": 4,
    "launch_capacity_gate": "available memory >= RSS cap+1GiB and free disk >= output cap+free-disk floor; otherwise stop before outcomes",
    "reuse": "reuse valid K0, Tier1, v6, feature, funding and Stage20 caches by hash; do not duplicate large artifacts without demonstrated invalidity",
    "uncertainty": "runtime is a conservative projection because the new executor is not run in this no-economics task; limits are safety bounds, not evidence quality criteria",
}
write_json("RESOURCE_PROJECTION.json", resource)

execution_spec = {
    "schema": "campaign_execution_spec_v1", "campaign_id": CAMPAIGN,
    "semantic_authority_order": ["CORE_CAMPAIGN_MECHANISM_CARDS.json", "REGISTERED_DISCOVERY_SPACE.json", "FULL_REGISTERED_CONFIGURATION_REGISTRY.jsonl", "CONTEXT_AND_ABLATION_REGISTRY.json", "INNER_OUTER_FOLD_MAP.json", "SELECTION_AND_PLATEAU_CONTRACT.json", "CONTROL_AND_FORENSIC_CONTRACT.json", "FULL_CONTROL_TEMPLATE_REGISTRY.jsonl", "OUTCOME_FREE_MEASUREMENT_PLAN.json"],
    "semantic_authority_sha256": {name: sha(HERE / name) for name in ["CORE_CAMPAIGN_MECHANISM_CARDS.json", "REGISTERED_DISCOVERY_SPACE.json", "FULL_REGISTERED_CONFIGURATION_REGISTRY.jsonl", "CONTEXT_AND_ABLATION_REGISTRY.json", "INNER_OUTER_FOLD_MAP.json", "SELECTION_AND_PLATEAU_CONTRACT.json", "CONTROL_AND_FORENSIC_CONTRACT.json", "FULL_CONTROL_TEMPLATE_REGISTRY.jsonl", "OUTCOME_FREE_MEASUREMENT_PLAN.json"]},
    "runtime_reference_source": {"path": "synthetic_campaign_supervisor.py", "sha256": sha(HERE / "synthetic_campaign_supervisor.py"), "scope": "transactional state, bounded lazy scheduling, process-tree RSS, atomic markers, heartbeat, bound stop and idempotent recovery; contains no economic outcome reader"},
    "economic_interpreter_rule": "The future executor is a mechanical interpreter of the complete registry and ordered semantic authorities. It may not add a field, default, threshold, tie rule, missingness rule, address, context, control, selection rule or execution convention.",
    "post_approval_implementation_gate": {"authorized_without_second_human_cycle": True, "requirements": ["implementation changes are task-scoped mechanical transcription only", "focused deterministic unit tests cover every formula and boundary", "synthetic full-registry supervisor/canary passes", "independent pre-outcome semantic-equivalence review passes", "source files and tests are SHA-256 bound in a final launch-boundary record", "all packet and economic dependency hashes remain exact", "no outcome reader opens before the gate"], "on_any_semantic_gap": "global stop; a new human approval and packet version are required"},
    "phase_order": ["outcome-free measurement/cache validation", "registered development", "component ablations/sparse discovery", "pre-registered conditional refinement", "plateau/deduplication", "deterministic beam and numeric-threshold freeze", "forward quarterly outer", "registered survivor controls", "forensics", "route decision"],
    "outcome_firewall": {"implementation_and_tests": "synthetic or timestamp/coverage/unit data only", "protected_rows": 0, "Capitalcom": False, "partial_outcome_guided_patch": False},
}
write_json("CAMPAIGN_EXECUTION_SPEC.json", execution_spec)


audit_md = f"""# Stage 20 research-adequacy audit

## Authority and execution correctness

The sequence-3 continuity pointer, immutable Stage 20 terminal ZIP (`{stage20_hashes['terminal_handoff_zip']}`), repository commit `{BASE_COMMIT}`, and terminal run-root artifacts reconcile. The terminal review independently replayed 954 development rows and 45 outer rows with zero metric mismatch, reconciled 8,415 job markers and 4,556 artifact claims, and found no protected or Capital.com access. The candidate CSV in this packet independently recomputes separate symbol/day/year concentrations and gross, fixed-cost, adverse-funding, gap-allowance and net means from the terminal scored shards; its stored-versus-recomputed mean deltas are below 8e-15.

This verifies execution of the frozen packet. It does not verify search completeness.

## Why 954 and 45

KDA02B: 96 cells x 9 development folds = 864 rows. KDA02C: 48 cells x only 2023Q4 = 48 rows. KDX01: 42 cells x only 2023Q4 = 42 rows. Thus 864+48+42=954. KDA02C and KDX01 stopped after that first surface; their remaining eight potential folds were explicit `family_stopped_earlier` decisions.

Only KDA02B reached outer evaluation. Its deterministic beam held five candidates in each of nine quarters, so 5x9=45 outer rows. There are 19 unique selected KDA02B cell IDs. `STAGE20_FOLD_AND_CANDIDATE_SUMMARY.csv` lists every fold selection, requested metrics, separate concentration shares, funding/cost decomposition and matched outer result.

Empty inner folds were preserved: 5,094 available observations and 258 explicit `empty_unavailable` observations (186 KDA02C, 72 KDX01). They were not silently omitted.

## Family routes

- KDA02B continued through 2025Q4. Twenty-five of 45 outer candidates had positive base mean and median, but alignment/stress sensitivity supports only `execution_sensitive_candidate`. No controls were run.
- KDA02C stopped in 2023Q4 because no beam-eligible positive development candidate existed. Sparse availability supports the preserved `sample_limited_prospective_candidate` route and no more 2023-2025 tuning.
- KDX01 stopped in 2023Q4 because no positive development candidate existed. The exact translation remains `translation_rejected`; this does not exhaust materially different mechanisms.

No calculation or authority defect was found, so the terminal routes are preserved.

## Search adequacy

The search covered the exact registered axes summarized in `STAGE20_SEARCH_COVERAGE_MATRIX.csv`, but it was narrow in windows, entries, exits, context overlays and execution structures. KDA02B tested a useful factorial grammar but no structural/ATR/trailing exit or broader contexts. KDA02C tested breadth forms/windows around a fixed sparse base identity and one-hour exit. KDX01 tested seven nested component ladders, two scalings and three fixed horizons, but not a broad continuous or sparse interaction design.

Every exact choice is classified in the coverage matrix as source prior, mechanism derived, outcome-free measurement derived, or registered design choice. No unsupported guess is silently asserted. A registered design choice is not treated as empirical support.

## Direct conclusion

- Frozen-packet execution correctness: pass.
- Search completeness within each family: not established.
- Family exhaustion: not established. Preserve the exact KDX01 rejection and KDA02C prospective freeze; adjudicate KDA02B without retuning.
- Independent validation or live readiness: none. All 2023-2025 evidence is `program_exposed_historical`; no Phase 6, deployment, or protected-period claim is authorized.
"""
(HERE / "STAGE20_RESEARCH_ADEQUACY_AUDIT.md").write_text(audit_md, encoding="utf-8")


# The campaign manifest binds every contract and economic dependency, but not the approval request (avoids a hash cycle).
contract_names = ["CORE_CAMPAIGN_MECHANISM_CARDS.json", "OUTCOME_FREE_MEASUREMENT_PLAN.json", "REGISTERED_DISCOVERY_SPACE.json", "FULL_REGISTERED_CONFIGURATION_REGISTRY.jsonl", "CONTEXT_AND_ABLATION_REGISTRY.json", "INNER_OUTER_FOLD_MAP.json", "SELECTION_AND_PLATEAU_CONTRACT.json", "CONTROL_AND_FORENSIC_CONTRACT.json", "FULL_CONTROL_TEMPLATE_REGISTRY.jsonl", "UNATTENDED_RUNTIME_AND_RECOVERY_CONTRACT.json", "RESOURCE_PROJECTION.json", "CAMPAIGN_EXECUTION_SPEC.json", "synthetic_campaign_supervisor.py", "recompute_stage20_candidate_summary.py", "NO_OUTCOME_RUNTIME_BENCHMARK.json", "NO_OUTCOME_RUNTIME_RESUME_BENCHMARK.json", "NO_OUTCOME_BOUND_STOP_CANARY.json", "NO_OUTCOME_BOUND_STOP_RECOVERY.json"]
manifest = {
    "schema": "core_campaign_manifest_v1", "campaign_id": CAMPAIGN, "status": "frozen_not_authorized",
    "created_at_utc": GENERATED_AT, "repository_base_commit": BASE_COMMIT,
    "economic_run_authorized_by_manifest": False, "one_external_human_approval_required": True,
    "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
    "platform": "Kraken native linear PF only", "programme_exposure_class": "program_exposed_historical",
    "new_attempt_budget": discovery["new_attempt_budget"], "economic_dependency_sha256": economic_dependencies,
    "contract_file_sha256": {name: sha(HERE / name) for name in contract_names},
    "permissions_after_exact_approval": ["mechanical executor transcription and independent semantic-equivalence gate under CAMPAIGN_EXECUTION_SPEC", "outcome-free measurement/cache validation", "registered development surfaces", "ablations/sparse discovery", "plateau/refinement", "deterministic freeze", "forward quarterly outer evaluation", "survivor controls", "forensics", "route decisions", "routine mechanical remediation under runtime contract", "terminal review/publication/handoff/continuity"],
    "prohibited": ["Phase 6", "independent validation claim", "protected outcomes", "Capital.com", "new acquisition", "capture restart", "C17", "account action", "order", "deployment", "live trading", "force push"],
    "stop_conditions": ["authority/hash mismatch", "outcome-firewall failure", "measurement hard-gate failure", "funding/cost/unit failure", "fold or attempt registry drift", "Telegram/canary/review failure", "resource gate failure", "economic semantic change required"],
}
write_json("CAMPAIGN_MANIFEST.json", manifest)

approval = {
    "schema": "human_approval_request_v1", "request_id": f"{CAMPAIGN}_single_launch_approval",
    "status": "exact_human_approval_required", "campaign_id": CAMPAIGN,
    "campaign_manifest_sha256": sha(HERE / "CAMPAIGN_MANIFEST.json"),
    "economic_dependency_sha256": economic_dependencies,
    "requested_exact_authorization": "Mechanically compile and independently verify an executor for the hash-bound spec without semantic discretion, then execute the registered sequence: measurement, 3,521 strategy/adjudication addresses plus at most 800 conditional survivor-control addresses (4,321 maximum), eight forward outer folds, deterministic forensics, unattended runtime, allowed narrow mechanical remediation/resume, terminal reconciliation/review, non-force Git publication, approved Drive handoff and continuity publication.",
    "authorization_preconditions": ["repository and every dependency hash match", "independent preoutcome review passes", "outcome-free measurement and cache gates pass", "synthetic end-to-end canary passes", "secure Telegram preflight passes", "resource gate passes", "protected and Capital.com firewalls remain closed"],
    "not_requested": manifest["prohibited"],
    "approval_semantics": "Approval must name this file's SHA-256 and the campaign manifest SHA-256. No post-approval economic or selection meaning may be invented; deterministic measurement-derived numeric quantiles and predeclared refinement addresses are generated and registered by the bound algorithms before their own outcome access.",
}
write_json("HUMAN_APPROVAL_REQUEST.json", approval)

# Closed archive inventory. The manifest excludes itself and transient compiled
# bytecode so it can be reproduced without a recursive hash.
artifact_rows = []
for artifact_path in sorted(HERE.rglob("*")):
    relative = artifact_path.relative_to(HERE).as_posix()
    if (not artifact_path.is_file() or relative == "ARTIFACT_MANIFEST.json"
            or "__pycache__" in artifact_path.parts or artifact_path.suffix == ".pyc"
            or artifact_path.suffix == ".zip"):
        continue
    artifact_rows.append({"path": relative, "bytes": artifact_path.stat().st_size,
                          "sha256": sha(artifact_path)})
write_json("ARTIFACT_MANIFEST.json", {
    "schema": "task_artifact_manifest_v1",
    "task_id": "donch_bt_stage21_research_adequacy_and_core_discovery_packet_20260720_v1",
    "campaign_id": CAMPAIGN, "artifact_count": len(artifact_rows),
    "artifacts": artifact_rows,
})
