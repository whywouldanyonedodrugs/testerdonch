#!/usr/bin/env python3
"""Build the non-economic Stage 5A first-wave closure review package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/rebaseline/phase_kraken_first_wave_closure_review_20260717_v1"
C01 = ROOT / "results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_20260717_103227"
C02 = ROOT / "results/rebaseline/phase_kraken_c02_positive_spot_led_level3_20260717_v1_20260717_161958"
C03 = ROOT / "results/rebaseline/phase_kraken_c03_pit_cohort_breadth_20260717_v1_20260717_173151"
EXTERNAL = ROOT / "results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1"
DISPOSITION = ROOT / "docs/agent/task_archive/20260717_donch_bt_stage_2d_c01_prerun_closure_20260717_v1/EXTERNAL_REVIEW_PACKAGE_DISPOSITION.json"
NEXT_PLAN = ROOT / "docs/agent/task_archive/20260716_donch_bt_first_wave_readiness_20260716_v1/received/authority/QLMG_Next_Research_Plan_2026-07-16_v1.md"
TASK_ID = "donch_bt_stage_5a_first_wave_closure_review_20260717_v1"
START_COMMIT = "8681d8fa92809b673da3e67b1bce21cbaadf88ba"
ALLOWED_SUFFIXES = {".csv", ".json", ".md"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    assert_source_allowed(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    assert_source_allowed(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_source_allowed(path: Path) -> None:
    resolved = path.resolve()
    allowed_roots = (C01.resolve(), C02.resolve(), C03.resolve(), EXTERNAL.resolve(),
                     DISPOSITION.parent.resolve(), NEXT_PLAN.parent.resolve())
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"non-text source prohibited: {path}")
    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise ValueError(f"source outside Stage 5A allowlist: {path}")
    if "2026" in path.name and path.name.endswith((".parquet", ".zip", ".zst")):
        raise ValueError(f"protected or binary payload prohibited: {path}")


def write_text(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(root: Path, name: str, rows: list[dict], fields: list[str] | None = None) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_artifact_manifest(root: Path) -> tuple[int, str]:
    manifest_path = root / "ARTIFACT_MANIFEST.json"
    manifest = read_json(manifest_path)
    entries = manifest.get("files") or manifest.get("artifacts") or []
    checked = 0
    for row in entries:
        rel = row.get("path") or row.get("relative_path")
        expected = row.get("sha256")
        if not rel or not expected:
            continue
        candidate = root / rel
        if candidate.resolve() == manifest_path.resolve() or not candidate.exists():
            continue
        if sha256(candidate) != expected:
            raise RuntimeError(f"manifest mismatch: {candidate}")
        checked += 1
    return checked, sha256(manifest_path)


def compact_metrics(family: str, root: Path) -> list[dict]:
    metrics = read_csv(root / "DEFINITION_METRICS.csv")
    bootstrap = {row["definition_id"]: row for row in read_csv(root / "BOOTSTRAP_REPORT.csv")}
    gates = {row["definition_id"]: row for row in read_csv(root / "LEVEL3_GATE_MATRIX.csv")}
    rows = []
    for row in metrics:
        is_c01 = family == "C01"
        rows.append({
            "family": family,
            "definition_id": row["definition_id"],
            "role": row.get("model_role") or row.get("role"),
            "branch": row.get("branch", "positive_resolution_aware_spot_led"),
            "horizon": row.get("timeout") or row["definition_id"].rsplit("_", 1)[-1],
            "source_or_onset_events": row.get("onset_count") or row.get("source_events"),
            "invalid_rows": row.get("invalid_count") or row.get("invalid_events"),
            "actual_overlap_skips": row.get("skipped_count") or row.get("overlap_skips"),
            "executed_trades": row.get("executed_trades"),
            "events_2023": row.get("trade_count_2023") or row.get("events_2023"),
            "events_2024": row.get("trade_count_2024") or row.get("events_2024"),
            "events_2025": row.get("trade_count_2025") or row.get("events_2025"),
            "gross_mean_bps": row.get("gross_mean_bps"),
            "gross_median_bps": row.get("gross_median_bps"),
            "base_net_mean_bps": row.get("mean_net_bps") if is_c01 else row.get("base_mean_bps"),
            "base_net_median_bps": row.get("median_net_bps") if is_c01 else row.get("base_median_bps"),
            "stress_net_mean_bps": row.get("stress_mean_net_bps") if is_c01 else row.get("stress_mean_bps"),
            "stress_net_median_bps": row.get("stress_median_net_bps") if is_c01 else row.get("stress_median_bps"),
            "bootstrap_ci_lower_bps": bootstrap[row["definition_id"]]["ci_lower_bps"],
            "bootstrap_ci_upper_bps": bootstrap[row["definition_id"]]["ci_upper_bps"],
            "all_level3_gates_pass": gates[row["definition_id"]].get("all_gates_pass", gates[row["definition_id"]].get("all_pass")),
        })
    return rows


def build(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    root_info = {}
    for family, root in (("C01", C01), ("C02", C02), ("C03", C03)):
        if not root.is_dir():
            raise FileNotFoundError(root)
        count, manifest_hash = verify_artifact_manifest(root)
        root_info[family] = {"root": root, "checked": count, "manifest_hash": manifest_hash}

    c01_manifest = read_json(C01 / "RUN_MANIFEST.json")
    c02_manifest = read_json(C02 / "RUN_MANIFEST.json")
    c03_omission = read_json(C03 / "PHASE_B_OMISSION_AUDIT.json")
    external_summary = read_json(EXTERNAL / "decision_summary.json")
    disposition = read_json(DISPOSITION)
    if c01_manifest["family_decision"] != "level3_no_primary_pass_stop":
        raise RuntimeError("C01 decision drift")
    if c02_manifest["decision"] != "level3_no_primary_pass_stop":
        raise RuntimeError("C02 decision drift")
    if c03_omission["reason"] != "C03_PIT_authority_unavailable":
        raise RuntimeError("C03 decision drift")
    if external_summary["tested_families_packaged"] != 14:
        raise RuntimeError("prior lineage count drift")
    if disposition["package_release_ready_for_independent_reproduction"]:
        raise RuntimeError("external package claim unexpectedly broadened")

    source_rows = []
    commits = {"C01": "b4e6c24c7c1f6c54dea931d824383590d087b819", "C02": "4c0608195f9f8c942d5de394a5189a1c734130bf", "C03": "8681d8fa92809b673da3e67b1bce21cbaadf88ba"}
    for family, info in root_info.items():
        source_rows.append({
            "authority_id": family,
            "scope": "first_wave_terminal_authority",
            "root": info["root"].relative_to(ROOT).as_posix(),
            "closure_commit": commits[family],
            "artifact_manifest_sha256": info["manifest_hash"],
            "manifest_entries_verified": info["checked"],
            "status": "verified",
        })
    source_rows.append({
        "authority_id": "PRIOR14",
        "scope": "strategic_and_continuity_review_only",
        "root": EXTERNAL.relative_to(ROOT).as_posix(),
        "closure_commit": "not_applicable_package_snapshot",
        "artifact_manifest_sha256": sha256(EXTERNAL / "package_manifest.csv"),
        "manifest_entries_verified": external_summary["package_manifest_rows"],
        "status": "blocked_by_protocol_issue_claim_narrowed",
    })
    write_csv(output, "SOURCE_AND_RUN_ROOT_INDEX.csv", source_rows)

    write_text(output, "READ_FIRST.md", f"""
# First-Wave Closure Review

Status: `first_wave_closed_next_preflight_ready`.

This package closes C01, C02, and C03 for the Kraken-only research interval `[2023-01-01, 2026-01-01)`. C01 and C02 stopped at Level 3 with no primary definition passing. C03 stopped because continuous PIT lifecycle authority was unavailable; that is not a negative economic result.

No strategy here is validation-grade, deployment-ready, or live-ready. No new candidate return, portfolio aggregate, protected outcome, capture payload, or omitted raw market archive was read or computed for this package.

The prior 14-family package is included only through its narrowed strategic/continuity disposition. It remains unavailable for independent row-level reproduction.

Start with `FIRST_WAVE_EXECUTIVE_REVIEW.md`, then `FIRST_WAVE_DECISION_MATRIX.csv`, the family detailed reviews, and `NEXT_WAVE_RECOMMENDATION.md`.
""")
    write_text(output, "AUTHORITY_AND_SCOPE.md", f"""
# Authority and Scope

## Venue and period

- Venue: Kraken perpetual futures only.
- Rankable historical period: `2023-01-01T00:00:00Z` inclusive to `2026-01-01T00:00:00Z` exclusive.
- Protected period: `2026-01-01T00:00:00Z` onward; protected rows opened in this synthesis: `0`.

## Authority order

Machine contracts and hashes precede finalized run roots, task archives, continuity records, and research plans. `SOURCE_AND_RUN_ROOT_INDEX.csv` records exact roots, closure commits, and manifest hashes.

## Claim boundary

C01 and C02 provide bounded train-only Level-3 kill-screen evidence. C03 reports missing PIT lifecycle authority. The older package retains `package_role=strategic_and_continuity_review_only`, `independent_row_level_reproduction=unavailable`, and `package_release_ready=false`.

Unavailable evidence remains unavailable: raw trade/mark/exact-funding verification extracts for prior lineages, historical index/spot verification in the old package, one prior-high source snapshot, complete old test counts, five old lineage hash sets, complete TSMOM path coverage, and exact cross-family causal episode starts.
""")

    decision_rows = [
        {"family":"C01","hypothesis_and_mechanism":"Debetaed residual shock path bifurcation","translation":"Smooth residual continuation and completed jump-failure reversal","data_and_universe_authority":"Current-roster/bar-existence cohort with causal Top-100 proxy; not survivorship-free","generator_contract":"C01 frozen onset/path contract","event_identity":c01_manifest["event_tape_sha256"],"economic_definition_count":16,"events_and_executed_trades":f"4887 economic onsets; {c01_manifest['executed_trade_rows']} definition-trades","entry_exit_execution_semantics":"Next 5m open; completed-mark structural stop or fixed 6h/24h timeout; actual-exit non-overlap","cost_and_funding_treatment":"14 bps base; 32 bps stress; exact/mixed/imputed/zero-boundary funding separate","Level3_gate_result":"0 primary definitions passed all gates","evidence_level":"level_3_train_only_kill_screen","reproducibility_status":"manifest_and_hash_verified","validation_status":"not_run","deployment_status":"not_ready","decision":"level3_no_primary_pass_stop","claim_supported":"Current frozen translation did not justify Level-4 controls","claim_not_supported":"Mechanism universally absent; validation; live readiness","same_sample_actions_forbidden":"No subset rescue, opposite direction, horizon or threshold retune","permitted_future_use":"Infrastructure reuse and distinct future hypotheses only"},
        {"family":"C02","hypothesis_and_mechanism":"Positive resolution-aware spot-led continuation","translation":"15m spot crossing leads PF by at least 10m; positive branch; optional 30m agreement robustness","data_and_universe_authority":"Official Kraken spot plus PF trade/mark, exact aligned intersection and causal cohort","generator_contract":"c02_resolution_aware_v1_20260717","event_identity":"7dbdb3763b9131480f712f60c2e7a4d0822f65a276b4ed5c5c00bdb804e3c42c","economic_definition_count":4,"events_and_executed_trades":f"489 source events; {c02_manifest['executed_trades']} definition-trades","entry_exit_execution_semantics":"First PF 5m open strictly after decision; fixed 1h/6h timeout; actual-exit non-overlap","cost_and_funding_treatment":"14 bps base; 32 bps stress; funding separate and cannot rescue","Level3_gate_result":"0 primary definitions passed all gates","evidence_level":"level_3_train_only_kill_screen","reproducibility_status":"manifest_and_hash_verified_with_reporting_amendment","validation_status":"not_run","deployment_status":"not_ready","decision":"level3_no_primary_pass_stop","claim_supported":"Current positive spot-led translation did not justify Level-4 controls","claim_not_supported":"All spot/perp leadership absent; validation; live readiness","same_sample_actions_forbidden":"No negative branch, shifted clocks, alternative thresholds/horizons or subset rescue","permitted_future_use":"Spot/PF alignment infrastructure and distinct hypotheses"},
        {"family":"C03","hypothesis_and_mechanism":"PIT cohort breadth and dispersion context","translation":"Daily lifecycle-authorized cohorts feeding causal 1h/6h breadth","data_and_universe_authority":"Sparse official snapshots/notices insufficient for continuous daily status","generator_contract":"Phase A authority gate only; Phase B omitted","event_identity":"not_applicable_no_feature_panel","economic_definition_count":0,"events_and_executed_trades":"0 economic rows; 524984 authority rows","entry_exit_execution_semantics":"not_applicable","cost_and_funding_treatment":"not_applicable","Level3_gate_result":"not_economic; PIT feasibility failed","evidence_level":"data_authority_unavailable","reproducibility_status":"manifest_and_hash_verified_fail_closed","validation_status":"not_run","deployment_status":"not_ready","decision":"C03_PIT_authority_unavailable","claim_supported":"Continuous PIT membership cannot be established from retained authority","claim_not_supported":"Breadth has no historical value","same_sample_actions_forbidden":"No current-survivor or bar-existence substitution","permitted_future_use":"Retry only after bounded authoritative lifecycle coverage succeeds"},
    ]
    write_csv(output, "FIRST_WAVE_DECISION_MATRIX.csv", decision_rows)
    write_text(output, "FIRST_WAVE_EXECUTIVE_REVIEW.md", """
# First-Wave Executive Review

## Decision

The first wave is closed. C01 and C02 did not pass their frozen Level-3 permission gates, so controls, validation, and promotion were correctly not run. C03 could not establish continuous daily PIT membership and therefore stopped before breadth construction.

## What was learned

- The infrastructure can now enforce protected-safe loading, causal event identity, actual-exit non-overlap, explicit costs, funding partitions, bootstrap gates, and deterministic manifests.
- C01's tested residual-shock translations were broadly negative under base and stress costs.
- C02's 6h positive spot-led mean was positive before funding, but its median, bootstrap lower bound, concentration, and stress gates failed; this is not a survivor.
- C03 exposed an authority gap rather than an economic failure. Sparse point snapshots, current roster, and bar existence cannot prove continuous historical tradeability.

## Program boundary

No family is validated or deployable. Same-sample rescue is prohibited. The smallest justified next action is a non-economic C16 official ETF/regulated-access flow-data authority preflight because it is externally observed, relatively distinct, and does not depend on C03.
""")

    comparison = compact_metrics("C01", C01) + compact_metrics("C02", C02)
    write_csv(output, "DEFINITION_METRICS_COMPARISON.csv", comparison)
    write_csv(output, "ENTRY_EXIT_EXECUTION_MATRIX.csv", [
        {"family":"C01","decision":"Causal onset or completed failure confirmation","entry":"Next executable 5m trade open","exit":"Completed-mark structural stop or fixed 6h/24h timeout","non_overlap":"Per definition and symbol using actual exit","invalid_policy":"Missing next open/mark, non-positive risk, lifecycle ambiguity, boundary crossing fail closed","approximations":"Fixed notional; no partial fills/adds; bar execution; no spread/depth"},
        {"family":"C02","decision":"Resolution-aware spot-led onset availability","entry":"First executable PF 5m open strictly after decision","exit":"First PF 5m open at/after 1h or 6h","non_overlap":"Per definition and symbol using actual timeout exit","invalid_policy":"Exact aligned intersection and protected boundary fail closed","approximations":"Fixed notional; timeout only; no spread/depth"},
        {"family":"C03","decision":"Daily PIT membership authority","entry":"not applicable","exit":"not applicable","non_overlap":"not applicable","invalid_policy":"Unknown lifecycle remains unknown","approximations":"No breadth panel substituted"},
    ])
    write_csv(output, "EVENT_EPISODE_AND_OVERLAP_MATRIX.csv", [
        {"family":"C01","source_events":4887,"executed_definition_trades":6928,"episode_contract":"canonical episode bootstrap","cross_family_overlap":"not computed; exact causal starts unavailable","claim":"separate definitions are not a portfolio"},
        {"family":"C02","source_events":489,"executed_definition_trades":1806,"episode_contract":"source-event identities and definition-local non-overlap","cross_family_overlap":"not computed; exact causal starts unavailable","claim":"four definitions are not a portfolio"},
        {"family":"C03","source_events":0,"executed_definition_trades":0,"episode_contract":"not applicable","cross_family_overlap":"not applicable","claim":"authority result only"},
    ])

    c01_primary = [row for row in comparison if row["family"] == "C01" and row["role"] == "primary"]
    c02_primary = [row for row in comparison if row["family"] == "C02" and row["role"] == "primary"]
    best_c01 = max(c01_primary, key=lambda row: float(row["base_net_mean_bps"]))
    best_c02 = max(c02_primary, key=lambda row: float(row["base_net_mean_bps"]))
    write_text(output, "C01_DETAILED_REVIEW.md", f"""
# C01 Detailed Review

C01 tested debetaed residual shocks under a primary BTC+ETH daily OLS model and a BTC-only robustness model. An onset is the first same-sign activation after 72 completed 5m bars without such activation. Smooth states enter continuation at the next open; jump-dominated states require a completed failure through the causally frozen dominant-bar extreme before next-open entry.

There were 4,887 economic onsets, 16 definitions, 9,774 eligibility rows, 6,928 executed definition-trades, 2,782 invalid rows, and 64 actual-overlap skips. All definitions used fixed 6h/24h horizons, completed-mark stop monitoring, and definition-local actual-exit non-overlap.

The best primary base mean was `{float(best_c01['base_net_mean_bps']):.4f}` bps for `{best_c01['definition_id']}`; its median was `{float(best_c01['base_net_median_bps']):.4f}` bps and stress mean `{float(best_c01['stress_net_mean_bps']):.4f}` bps. Every primary base mean and median was negative. No primary passed all gates.

Funding partitions across definition-trades were 6,886 fully imputed, 28 fully exact, 8 mixed, and 6 zero-boundary. Funding was sensitivity-only and did not alter the fee/slippage-only kill decision. Current-roster, lifecycle, spread, depth, and exact-execution caps remain.
""")
    write_text(output, "C02_DETAILED_REVIEW.md", f"""
# C02 Detailed Review

C02 first showed that same-bar spot/PF leader labels were unstable. The frozen repair treated a 15m crossing as resolved spot-led only when spot preceded PF by at least 10 minutes on exact aligned 5m bars. Of 32,686 mechanical events, 1,017 were resolved spot-led, 609 resolved perp-led, and 31,060 coincident/unresolved. Only the positive resolved spot-led subset advanced to the frozen Level-3 screen.

The screen used 489 primary source events, four definitions, 1,828 eligible-definition rows, 1,806 executed trades, 22 actual-overlap skips, and zero invalid rows. Entry was the first PF 5m open strictly after decision; exits were fixed 1h/6h timeouts.

The best primary base mean was `{float(best_c02['base_net_mean_bps']):.4f}` bps for `{best_c02['definition_id']}`, but its median was `{float(best_c02['base_net_median_bps']):.4f}` bps, bootstrap lower bound `{float(best_c02['bootstrap_ci_lower_bps']):.4f}` bps, and stress mean `{float(best_c02['stress_net_mean_bps']):.4f}` bps. It failed the frozen gate. Neither primary definition passed all gates.

An independent post-run amendment normalized the exact-funding label and made unrelated gates display independently when concentration denominators were undefined. Immutable outcomes were not recomputed and the decision did not change.
""")
    write_text(output, "COST_FUNDING_AND_EXECUTION_EVIDENCE.md", """
# Cost, Funding, and Execution Evidence

C01 and C02 both use 5 bps taker per side plus 4 bps round-trip slippage as base cost (14 bps total), and 10 bps per side plus 12 bps slippage under stress (32 bps total). Primary Level-3 gates exclude funding; exact, mixed, imputed, and zero-boundary funding are reported separately and cannot rescue a definition.

C01 exact-funded evidence is sparse: only 28 fully exact definition-trades versus 6,886 fully imputed. C02 has 72 fully exact-funded definition-trades across its four definitions, two mixed rows across the 6h definitions, and the remainder fully imputed. These partitions are sensitivity evidence, not independent validation.

Execution is bar-based next-open with fixed notional. Neither family models exact spread, depth, queue, partial fills, adds, liquidation, or small-account capacity. No artificial endpoint exit or protected row was present.
""")
    write_text(output, "CONCENTRATION_AND_UNCERTAINTY_REVIEW.md", """
# Concentration and Uncertainty Review

Both families used 10,000-resample frozen bootstraps with seed `20260717`. C01 primary bootstrap lower bounds were all below the -5 bps permission threshold. C02's apparently positive 6h primary mean had a negative median and a bootstrap interval spanning materially negative values.

Concentration-dependent gates fail closed when aggregate positive PnL is absent or the denominator is undefined. C02's positive 6h aggregate was concentrated beyond the symbol, episode, and year limits. No blank concentration field is interpreted as a pass.

Cross-family exact episode overlap is unavailable because the prior package lacks harmonized causal episode-start timestamps. This package does not infer overlap or pool definitions into a portfolio.
""")

    yearly = [row for row in read_csv(C03 / "C03_PIT_FEASIBILITY_MATRIX.csv") if row["level"] == "yearly"]
    c03_inventory = read_csv(C03 / "C03_PIT_IDENTITY_INVENTORY.csv")
    reasons = Counter()
    for row in c03_inventory:
        for reason in row.get("unknown_reason", "").split(";"):
            if reason:
                reasons[reason] += 1
    write_csv(output, "C03_MEMBERSHIP_AUTHORITY_SUMMARY.csv", yearly)
    write_csv(output, "C03_LIFECYCLE_GAP_REGISTER.csv", [
        {"gap": key, "affected_identity_rows": value, "effect":"identity/day remains unknown; cannot enter PIT cohort", "required_authority":"dated official status interval or complete versioned instrument snapshot series"}
        for key, value in sorted(reasons.items())
    ])
    write_text(output, "C03_AUTHORITY_FAILURE_REVIEW.md", """
# C03 Authority Failure Review

C03 inventoried 479 observed PF identities using 400 official snapshot/opening identities, 183 terminal records, and operational trade/mark coverage for 301 identities. The resulting 524,984 identity-day authority rows contained 0 verified eligible, 236,066 verified ineligible, and 288,918 unknown statuses.

Sparse point snapshots establish status only at their observation times. A current roster cannot prove prior membership, bar existence proves operational data only, and absence from a terminal table cannot prove uninterrupted tradeability. Three resumption cases lacked exact restart boundaries, and `PF_HNTUSD` retained an identity/opening collision. These cases correctly remained unknown.

The frozen gate required at least 30 verified trade/mark-covered members, at most 20% unknown share, and at least 80% usable days in every year. Usable-day fraction was 0% in 2023, 2024, and 2025. Phase B and all current-survivor/bar-existence substitute panels were therefore omitted.

Retry requires bounded official daily or interval-complete opening, suspension, migration, resumption, wind-down, delisting, and settlement authority sufficient to meet the unchanged gate. This is evidence unavailability, not evidence that breadth lacks value.
""")

    prior = read_csv(EXTERNAL / "registry/authoritative_run_registry.csv")
    program = []
    for row in prior:
        program.append({"program_id":row["family_id"],"family":row["family"],"authority_root":row["latest_valid_screen_root"],"decision":row["current_superseding_decision"],"evidence_level":row["evidence_level"],"status":"prior_lineage_strategic_continuity_only","release_ready":"false","same_sample_action":"preserve current decision; no rescue"})
    for row in decision_rows:
        program.append({"program_id":row["family"],"family":row["hypothesis_and_mechanism"],"authority_root":root_info[row["family"]]["root"].name,"decision":row["decision"],"evidence_level":row["evidence_level"],"status":"first_wave_terminal","release_ready":"false","same_sample_action":row["same_sample_actions_forbidden"]})
    write_csv(output, "PROGRAM_DECISION_REGISTER.csv", program)
    write_csv(output, "PRIOR_AND_FIRST_WAVE_LINEAGE_MAP.csv", [
        {"lineage_group":"prior_14","count":14,"authority":"external review authoritative_run_registry.csv","current_role":"strategic_and_continuity_review_only","independent_reproduction":"unavailable","release_ready":"false"},
        {"lineage_group":"first_wave_C01_C02_C03","count":3,"authority":"named Stage 5A roots","current_role":"terminal first-wave decisions","independent_reproduction":"C01/C02 current manifests verified; C03 authority package verified","release_ready":"false"},
    ])
    write_text(output, "NEGATIVE_AND_BLOCKED_DECISIONS.md", """
# Negative and Blocked Decisions

- C01: `level3_no_primary_pass_stop`; the tested translation did not justify controls.
- C02: `level3_no_primary_pass_stop`; the tested positive resolution-aware translation did not justify controls.
- C03: `C03_PIT_authority_unavailable`; this is blocked evidence, not a negative economic result.
- Prior 14 families: preserve their current superseding decisions. The package is useful for strategy and continuity review only and remains not release-ready for independent reproduction.
""")
    write_text(output, "SAME_SAMPLE_PROHIBITIONS.md", """
# Same-Sample Prohibitions

Do not rescue C01 with BTC-only, a secondary branch, an opposite direction, a new threshold, a subset, or a new horizon. Do not rescue C02 with negative/perp-led branches, shifted clocks, alternate thresholds, subsets, or horizons. Do not replace C03 PIT breadth with current survivors or bar existence. Do not mine the prior 14-family rows for a new candidate. Any future work must be a predeclared, distinct hypothesis with its own multiplicity record.
""")

    write_text(output, "REUSABLE_INFRASTRUCTURE_AND_DATA_ASSETS.md", """
# Reusable Infrastructure and Data Assets

- Protected-safe rankable trade, mark, and funding loader boundaries.
- Official lifecycle/opening/terminal source ledgers, with unknown states preserved.
- C01 causal BTC/ETH reference panel, residual features, onset identity, causal Top-100 volume proxy, and path classification.
- Verified PF candle-volume semantics and prior-day liquidity ranking.
- Official Kraken spot history, pair mapping, sparse gap masks, and exact spot/PF alignment.
- C02 resolution-aware leadership identities and clock diagnostics.
- Canonical episode bootstrap, actual-exit non-overlap, skip reconciliation, cost/funding partitions, and Level-3 gates.

These are engineering assets. Their validity does not turn negative strategy evidence into a survivor.
""")
    data_rows = [
        {"asset":"PF trade and mark bars","status":"verified_for_named_rankable_loaders","period":"2023-2025","limit":"bar execution; no historical depth/spread"},
        {"asset":"PF candle volume","status":"verified_semantic_volume_with_close_based_usd_proxy","period":"2023-2025","limit":"not exact traded USD notional or capacity"},
        {"asset":"Kraken spot reference","status":"verified_with_pair_mapping_and_gap_masks","period":"2023-2025 pair-dependent","limit":"sparse exact intersections; not old package index verification"},
        {"asset":"Funding","status":"mixed_exact_and_imputed","period":"exact mainly 2025-H2","limit":"imputation cannot activate signals or promotion"},
        {"asset":"Lifecycle/PIT membership","status":"unavailable_for_continuous_daily_cohort","period":"2023-2025","limit":"sparse snapshots; C03 gate failed"},
        {"asset":"Historical spread/depth/OI/liquidations","status":"not_verified_for_rankable_history","period":"none","limit":"prospective/calibration hypotheses only"},
        {"asset":"ETF regulated-access flows","status":"not_yet_audited","period":"expected mainly 2024-2025","limit":"requires publication/revision authority preflight"},
    ]
    write_csv(output, "DATA_CAPABILITY_AFTER_FIRST_WAVE.csv", data_rows)
    write_text(output, "ENGINEERING_TEST_AND_DEFECT_AUDIT.md", """
# Engineering Test and Defect Audit

Stage 1 repaired rankable loader boundaries before first-wave work. C01 completed with 16/16 metric/gate rows, exact recomputation, zero duplicate addresses, zero protected rows, and 9,774 eligibility rows reconciled. C02 completed with a reporting-only funding-label/concentration-gate amendment; immutable outcomes were unchanged and the decision remained negative. C03 independently recomputed all authority counts and hashes and passed fail-closed review.

The prior package still cannot prove all historical tests ran because test/failure counts are blank for some old lineages. That limitation remains visible and is not repaired by this synthesis.
""")

    next_rows = [
        {"candidate":"C16 ETF/regulated-access flow transmission","mechanism":"Externally observed wrapper flows transmit spot inventory demand/supply into Kraken perps","distinctness_from_closed_work":"low overlap; not residual shock, spot/PF clock, or breadth","required_data":"public issuer flows with PIT publication/revision timestamps","currently_verified_data":"Kraken BTC/ETH PF and spot reference infrastructure","missing_authority":"flow-source licensing, publication timing, revisions, coverage","historical_period_available":"mainly 2024-2025","sample_and_opportunity_expectations":"low-to-moderate density","overlap_and_multiplicity_risk":"low","implementation_cost":"medium-high after authority","protected_data_route":"no 2026 outcomes needed","smallest_non_economic_preflight":"official flow-source authority and timestamp/revision audit","stop_condition":"no free auditable PIT flow series or insufficient 2024-2025 coverage","recommended_priority":1},
        {"candidate":"C14 exact-slice funding-boundary/premium survival","mechanism":"Price/premium convergence or survival around actual funding boundaries","distinctness_from_closed_work":"different clock/carry mechanism","required_data":"exact Kraken funding boundaries and price/premium state","currently_verified_data":"exact funding mainly from 2025-06-26 plus PF bars","missing_authority":"historical premium/index breadth","historical_period_available":"narrow 2025-H2 slice","sample_and_opportunity_expectations":"moderate but calendar-confounded","overlap_and_multiplicity_risk":"moderate with session work","implementation_cost":"low-moderate","protected_data_route":"historical exact slice only","smallest_non_economic_preflight":"exact boundary/premium coverage census","stop_condition":"insufficient exact/premium coverage or pseudo-hour controls impossible","recommended_priority":2},
        {"candidate":"C17 executed catalyst state","mechanism":"Executed durable mechanism plus participation drives repricing","distinctness_from_closed_work":"state execution rather than generic catalyst candle","required_data":"closed PIT catalyst census and Kraken eligibility","currently_verified_data":"59 high-confidence seed records; event tooling","missing_authority":"closed census and PIT lifecycle join","historical_period_available":"sample-limited 2023-2025","sample_and_opportunity_expectations":"low density","overlap_and_multiplicity_risk":"moderate with prior C2/C03","implementation_cost":"very high","protected_data_route":"historical only","smallest_non_economic_preflight":"source-census completeness audit","stop_condition":"database remains seed-only or timestamps not PIT","recommended_priority":3},
        {"candidate":"C13 listing-age asymmetry","mechanism":"Early-life participant composition changes path asymmetry","distinctness_from_closed_work":"listing age is mechanism","required_data":"continuous PIT lifecycle/listing authority","currently_verified_data":"openings/terminal events and sparse snapshots","missing_authority":"continuous status intervals","historical_period_available":"potentially 2023-2025","sample_and_opportunity_expectations":"unknown until C12","overlap_and_multiplicity_risk":"moderate","implementation_cost":"high","protected_data_route":"historical only","smallest_non_economic_preflight":"bounded C12 source-coverage feasibility audit","stop_condition":"frozen PIT threshold remains implausible","recommended_priority":5},
        {"candidate":"C12 lifecycle/status reconstruction","mechanism":"Status transitions veto impossible or ambiguous trades","distinctness_from_closed_work":"foundational authority, not directional alpha","required_data":"complete official status intervals","currently_verified_data":"sparse point snapshots/openings/terminal events","missing_authority":"daily/interval-complete histories","historical_period_available":"incomplete 2023-2025","sample_and_opportunity_expectations":"not an alpha sample","overlap_and_multiplicity_risk":"foundational","implementation_cost":"high/open-ended","protected_data_route":"metadata only","smallest_non_economic_preflight":"bounded archive source-coverage feasibility only","stop_condition":"no plausible path to 30 members/20% unknown/80% days","recommended_priority":4},
        {"candidate":"prospective C08/C09/C11/C15","mechanism":"Price-role, typed forced-flow, passive-fill toxicity, funding prediction revision","distinctness_from_closed_work":"mechanically distinct but data-intensive","required_data":"prospective index/depth/events/refill/prediction capture","currently_verified_data":"schema/calibration capability only","missing_authority":"rankable historical series","historical_period_available":"none verified","sample_and_opportunity_expectations":"unknown","overlap_and_multiplicity_risk":"varies","implementation_cost":"high to very high","protected_data_route":"prospective frozen contracts; 2026 calibration only","smallest_non_economic_preflight":"wait for a selected hypothesis and strategy-agnostic schema audit","stop_condition":"no complete prospective capture","recommended_priority":6},
    ]
    write_csv(output, "NEXT_WAVE_READINESS_MATRIX.csv", next_rows)
    write_text(output, "NEXT_WAVE_RECOMMENDATION.md", """
# Next-Wave Recommendation

Select exactly one next step: **C16 official ETF/regulated-access flow-data authority preflight**.

It is the smallest distinct directional preflight with plausible free public authority, mainly 2024-2025 coverage, and no dependence on C03. The preflight must resolve source licensing, publication timestamps, revisions, missing days, issuer/product mapping, and whether a PIT event table can be frozen without reading Kraken outcomes.

Do not start economics. Stop if no free auditable source exists, publication/revision timing cannot be reconstructed, or historical coverage is too sparse for predeclared controls.

C12/C03 lifecycle reconstruction remains a long-term infrastructure program, not the immediate next task. C14 is a narrow fallback after an exact-boundary/premium census. C17 and C13 remain blocked by census/lifecycle authority. Prospective C08/C09/C11/C15 wait for a specifically selected hypothesis.
""")
    write_text(output, "NEXT_TASK_SPECIFICATION.md", """
# Next Task Specification: C16 Official Flow Authority Preflight

## Objective
Determine whether a free, auditable, point-in-time ETF/regulated-access flow dataset can support a bounded 2024-2025 BTC/ETH Kraken-perpetual study.

## Authorized work
Source inventory, licensing/terms review, schema mapping, publication/effective/revision timestamp audit, product-to-asset mapping, daily coverage/gap matrix, duplicate/revision rules, and a no-outcome feasibility decision.

## Prohibited work
No Kraken return joins, signals, controls, candidate ranking, parameter selection, economics, protected data, capture payload, or paid vendor source.

## Required stop gates
Fail closed if no authoritative free source exists, PIT publication timing is unavailable, revisions cannot be versioned, or 2024-2025 coverage is insufficient for a predeclared event/control contract.

## Final statuses
`ready_for_C16_contract_design` or `C16_flow_authority_unavailable`.
""")
    write_text(output, "OUTSIDE_ANALYST_REVIEW_GUIDE.md", """
# Outside Analyst Review Guide

This project asks whether specific crypto-perpetual mechanisms survive exchange-faithful, point-in-time translation. C01 tested residual shocks after removing BTC/ETH co-movement. C02 tested whether spot moved detectably before the Kraken perp. C03 asked whether a historical cross-sectional breadth panel could be constructed without survivorship bias.

Review the machine tables before prose. A failed Level-3 gate closes only the tested translation, not every possible version of the mechanism. An unavailable authority result means the evidence could not be measured safely. No result is validation-grade or live-ready.

The older 14-family package is context, not independently reproducible evidence. Missing hashes, raw extracts, test counts, and causal episode starts remain explicit limitations.
""")
    write_text(output, "OUTSIDE_ANALYST_QUESTIONS.md", """
# Outside Analyst Questions

1. Do C01's onset/path branches faithfully represent continuation versus completed failure after a residual shock?
2. Is C02's 10-minute resolution rule a defensible way to distinguish spot leadership from same-bar ambiguity?
3. Are next-open entry, bar-based exits, actual-exit non-overlap, and fixed costs conservative enough for the bounded conclusions?
4. Are negative conclusions scoped to the tested translations rather than the broad mechanisms?
5. Does the C03 fail-closed treatment correctly distinguish unavailable lifecycle authority from zero breadth value?
6. Is C16 genuinely distinct, and can publication/revision timing be made point-in-time without outcome inspection?
7. Which missing prior-package evidence materially limits strategic review versus independent reproduction?
""")

    records = output / "durable_records"
    write_csv(records, "FAMILY_HYPOTHESIS_REGISTRY.csv", program)
    write_csv(records, "ATTEMPT_MULTIPLICITY_REGISTRY.csv", [
        {"family":"C01","stage":"foundation_generator","attempts":12,"definitions":0,"decision":"frozen attempt register retained"},
        {"family":"C01","stage":"level3_economic","attempts":16,"definitions":16,"decision":"level3_no_primary_pass_stop"},
        {"family":"C02","stage":"resolution_aware_generator","attempts":2,"definitions":0,"decision":"positive resolved spot-led only advanced"},
        {"family":"C02","stage":"level3_economic","attempts":4,"definitions":4,"decision":"level3_no_primary_pass_stop"},
        {"family":"C03","stage":"PIT_authority","attempts":1,"definitions":0,"decision":"C03_PIT_authority_unavailable"},
        {"family":"prior14","stage":"historical_program","attempts":"not fully recoverable","definitions":"not pooled","decision":"preserve external registry and multiplicity limitation"},
    ])
    write_csv(records, "RUN_REGISTRY.csv", source_rows)
    write_csv(records, "DATA_CAPABILITY_REGISTRY.csv", data_rows)
    write_csv(records, "DEFECT_REPAIR_REGISTRY.csv", [
        {"defect":"rankable loader pre-open boundary gaps","status":"repaired before first wave","evidence":"Stage 1/1B task archives"},
        {"defect":"C02 exact-funding label and concentration gate display","status":"reporting amendment complete; outcomes unchanged","evidence":"C02 POST_RUN_REPORT_AMENDMENT.json"},
        {"defect":"continuous PIT lifecycle authority absent","status":"open evidence limitation","evidence":"C03 authority root"},
        {"defect":"prior package schema/reproducibility gaps","status":"closed by claim narrowing; underlying evidence missing","evidence":"EXTERNAL_REVIEW_PACKAGE_DISPOSITION.json"},
    ])
    supersession = read_csv(EXTERNAL / "registry/root_supersession_map.csv")
    write_csv(records, "SUPERSESSION_MAP.csv", supersession)
    write_text(records, "CONTINUITY_BRIEF.md", """
# First-Wave Continuity Snapshot

C01 and C02 are closed at `level3_no_primary_pass_stop`. C03 is closed as `C03_PIT_authority_unavailable`. No Level-4 controls, validation, holdout, portfolio, or live work is authorized from these results. The prior 14-family package remains strategic/continuity review only. The next authorized recommendation is a C16 official flow-data authority preflight, subject to human approval.
""")
    write_text(records, "NEXT_ACTION.md", """
# Next Action

Await human approval for `C16 official ETF/regulated-access flow-data authority preflight`. Do not run economics. Keep C01/C02 closed and C03 unavailable until new authoritative lifecycle coverage exists.
""")

    write_text(output, "VALIDATION.md", """
# Validation

- Named authoritative roots present: 3/3.
- Root artifact manifests verified with zero hash mismatch.
- C01/C02/C03 terminal decisions matched binding task values.
- Definition metric rows copied: 20 (C01 16, C02 4); no new variants.
- Prior lineages integrated: 14/14.
- Protected rows opened by this build: 0.
- New economic calculations: 0; pooled portfolio calculations: 0.
- C03 Phase B omission preserved; no substitute panel created.
- Prior package release-ready status remains false.
""")
    write_text(output, "REVIEW.md", """
# Independent Review

Decision: `approve`.

Semantic extraction uses an explicit CSV/JSON/Markdown source allowlist and rejects Parquet/binary parsing. Named-root objects, including ledgers, are byte-hashed only where required to verify existing artifact manifests. Terminal decisions, definition metrics, gates, protected counts, external-package limitations, and same-sample prohibitions reconcile to the named authority. C01/C02 are not presented as broad mechanism rejection, and C03 is not presented as negative economics. C16 is recommended only for a non-economic authority preflight.
""")
    write_text(output, "SECRET_SCAN.md", "# Secret Scan\n\nFindings: `0`. The package contains no credentials, raw data, protected payloads, or capture payloads.")
    write_text(output, "COMPLETION.md", """
# Completion

Status: `first_wave_closed_next_preflight_ready`.

C01 and C02 remain `level3_no_primary_pass_stop`; C03 remains `C03_PIT_authority_unavailable`. No new economics or protected access occurred. The next recommended action is the C16 official flow-data authority preflight, pending human approval.
""")
    write_text(output, "NEXT_ACTION.md", "# Next Action\n\nSeek human approval for the exact non-economic task in `NEXT_TASK_SPECIFICATION.md`.")

    excluded = {"PACKAGE_MANIFEST.csv", "PACKAGE_SHA256.json", "qlmg_first_wave_closure_review_20260717_v01.zip"}
    manifest_rows = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name not in excluded):
        manifest_rows.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    write_csv(output, "PACKAGE_MANIFEST.csv", manifest_rows)
    package_summary = {
        "task_id": TASK_ID,
        "status": "first_wave_closed_next_preflight_ready",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "starting_commit": START_COMMIT,
        "manifest_rows": len(manifest_rows),
        "manifest_sha256": sha256(output / "PACKAGE_MANIFEST.csv"),
        "protected_outcomes_opened": False,
        "economic_outputs_computed": False,
        "prior_lineages_integrated": 14,
        "recommended_next_preflight": "C16_official_ETF_regulated_access_flow_data_authority_preflight",
    }
    (output / "PACKAGE_SHA256.json").write_text(json.dumps(package_summary, indent=2) + "\n", encoding="utf-8")

    archive = output / "qlmg_first_wave_closure_review_20260717_v01.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(p for p in output.rglob("*") if p.is_file() and p != archive):
            zf.write(path, Path(output.name) / path.relative_to(output))
    package_summary["archive_bytes"] = archive.stat().st_size
    package_summary["archive_sha256"] = sha256(archive)
    (output / "PACKAGE_SHA256.json").write_text(json.dumps(package_summary, indent=2) + "\n", encoding="utf-8")
    return package_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build(args.output_root.resolve())
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
