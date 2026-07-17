#!/usr/bin/env python3
"""Build the outcome-free C01 Level-3 approval contract and package disposition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from tools.kraken_c01_prerun_contract import (
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        LEVEL3_GATE_NAMES,
        PRIMARY_MODEL,
        ROBUSTNESS_MODEL,
        definition_register,
        validate_package_disposition,
    )
except ModuleNotFoundError:  # Direct script execution from tools/.
    from kraken_c01_prerun_contract import (
        BOOTSTRAP_RESAMPLES,
        BOOTSTRAP_SEED,
        LEVEL3_GATE_NAMES,
        PRIMARY_MODEL,
        ROBUSTNESS_MODEL,
        definition_register,
        validate_package_disposition,
    )


TASK_ID = "donch_bt_stage_2d_c01_prerun_closure_20260717_v1"
START_COMMIT = "45d92488a41fb97a9a30936075c19581f358357d"
GENERATOR_HASH = "3464e79a79956c881c7418840068a61e3f3a47776a5a4d3a669e98df124fd970"
ECONOMIC_DRAFT_HASH = "f1c8c612ea9f7ffcc2abad3f2efde36b5dfb68fde20d2769fdc5ce40ab306c13"
FEATURE_HASH = "c0d2955d6447f360beff528fc0985d328c2853cf9a45bf03958d2a61216470bb"
COHORT_HASH = "768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15"
REFERENCE_PANEL_HASH = "2c0cae827c6f83361ea347796d0323b20d83c3acd222c506aac40c0e97b73763"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_register(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_authority(stage2c_root: Path, package_root: Path) -> dict[str, Any]:
    summary = json.loads((stage2c_root / "C01_STAGE2C_SUMMARY.json").read_text(encoding="utf-8"))
    expected = {
        "generator_contract_hash": GENERATOR_HASH,
        "economic_contract_draft_hash": ECONOMIC_DRAFT_HASH,
        "feature_contract_hash": FEATURE_HASH,
        "cohort_hash": COHORT_HASH,
        "reference_panel_hash": REFERENCE_PANEL_HASH,
        "onset_event_count": 38_347,
        "canonical_episode_count": 11_343,
        "protected_outcomes_opened": False,
        "economic_outputs_computed": False,
    }
    mismatches = {key: (summary.get(key), value) for key, value in expected.items() if summary.get(key) != value}
    if mismatches:
        raise ValueError(f"Stage 2C1 authority mismatch: {mismatches}")
    if sha256_file(stage2c_root / "C01_FROZEN_GENERATOR_CONTRACT.md") != GENERATOR_HASH:
        raise ValueError("generator contract bytes do not match accepted hash")
    if sha256_file(stage2c_root / "C01_ECONOMIC_CONTRACT_DRAFT.md") != ECONOMIC_DRAFT_HASH:
        raise ValueError("economic draft bytes do not match accepted hash")
    package = json.loads((package_root / "decision_summary.json").read_text(encoding="utf-8"))
    if package.get("status") != "blocked_by_protocol_issue" or package.get("package_release_ready") is not False:
        raise ValueError("external package source status differs from accepted blocked state")
    for archive_name in ("qlmg_external_review_core_20260716_v1.zip", "qlmg_external_review_full_20260716_v1.tar.zst"):
        if not (package_root / archive_name).is_file():
            raise ValueError(f"external package archive missing: {archive_name}")
    return package


def package_disposition(package_root: Path) -> dict[str, Any]:
    gaps = [
        {
            "item": "raw_trade_mark_and_exact_funding_verification_extracts",
            "status": "unavailable",
            "effect": "no independent row-level reproduction",
        },
        {
            "item": "historical_index_or_spot_source",
            "status": "unavailable",
            "effect": "no exact historical index/spot verification",
        },
        {
            "item": "prior_high_v2_runner_source_snapshot",
            "status": "deferred_with_exact_task",
            "exact_task": "donch_bt_prior_high_v2_source_snapshot_recovery_v1",
            "effect": "one prior lineage lacks its packaged source snapshot",
        },
        {
            "item": "complete_prior_lineage_test_and_failure_counts",
            "status": "unavailable",
            "effect": "package cannot prove all old tests ran",
        },
        {
            "item": "five_lineage_reproducibility_hash_sets",
            "status": "deferred_with_exact_task",
            "exact_task": "donch_bt_external_review_five_lineage_hash_recovery_v1",
            "effect": "five old lineages remain non-reproducible from package alone",
        },
        {
            "item": "tsmom_complete_path_and_mae_mfe_coverage",
            "status": "deferred_with_exact_task",
            "exact_task": "donch_bt_tsmom_path_coverage_recovery_v1",
            "effect": "TSMOM path verification remains incomplete",
        },
        {
            "item": "cross_family_causal_episode_start_timestamps",
            "status": "deferred_with_exact_task",
            "exact_task": "donch_bt_cross_family_causal_episode_start_export_v1",
            "effect": "exact cross-family episode overlap is unavailable",
        },
    ]
    disposition = {
        "source_package_root": str(package_root),
        "source_package_status_preserved": "blocked_by_protocol_issue",
        "supersedes_open_workflow_interpretations": [
            "docs/agent/task_archive/20260716_donch_bt_first_wave_readiness_20260716_v1/PACKAGE_PROTOCOL_STATUS.md",
            "docs/agent/task_archive/20260716_donch_bt_stage_1b_remaining_loader_boundary_20260716_v1/readiness_rerun/PACKAGE_PROTOCOL_STATUS.md",
        ],
        "protocol_disposition": "closed_by_claim_narrowing",
        "package_role": "strategic_and_continuity_review_only",
        "package_release_ready_for_independent_reproduction": False,
        "allowed_uses": [
            "research_history",
            "supersession_and_negative_decision_continuity",
            "broad_multiplicity_context",
            "identification_of_nearest_prior_families",
        ],
        "prohibited_uses": [
            "independent_row_level_reproduction",
            "validation_grade_claims",
            "proof_all_old_tests_ran",
            "exact_cross_family_episode_overlap",
            "promotion_of_C01_or_any_prior_family",
        ],
        "missing_items": gaps,
        "economic_outputs_recomputed": False,
        "protected_outcomes_opened": False,
    }
    validate_package_disposition(disposition)
    return disposition


def final_contract() -> str:
    return f"""# C01 Final Level-3 Economic Contract

Status: frozen pre-run contract; no economic run is authorized by this file.

## Authority and immutable inputs

- Family: `C01_debetaed_residual_shock_path_bifurcation`
- Repository lineage: `{START_COMMIT}`
- Generator contract: `{GENERATOR_HASH}`
- Stage 2C economic draft: `{ECONOMIC_DRAFT_HASH}`
- Feature contract: `{FEATURE_HASH}`
- Cohort: `{COHORT_HASH}`
- Reference panel: `{REFERENCE_PANEL_HASH}`
- Candidate cohort: `current_roster_bar_existence_cohort`; it is not survivorship-free and does not prove continuous tradeability.
- Train interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`; later rows are protected.

No generator, feature, cohort, reference-panel, onset, or path threshold is changed.

## Definition and multiplicity freeze

The primary model is `{PRIMARY_MODEL}`. The BTC-only `{ROBUSTNESS_MODEL}` model is robustness-only and cannot rescue a failed primary definition. The four branches are `positive_smooth_long`, `negative_smooth_short`, `positive_jump_completed_failure_short`, and `negative_jump_completed_failure_long`. Intermediate states are diagnostics only.

Each model has all four branches at fixed 6h and 24h timeouts: 8 primary definitions plus 8 separately registered robustness definitions, 16 total. Zero-trade definitions remain registered. Every later control and ablation is an additional multiplicity attempt.

## Entry and confirmation

Positive smooth enters long and negative smooth enters short at the next executable five-minute trade-bar open after onset. For positive jump-dominated onsets, confirmation is the first completed trade bar within 24h closing below the causally frozen dominant residual bar low, followed by short entry at the next executable open. The negative jump branch is symmetric above the dominant bar high and enters long. Dominant-bar identity and confirmation are deterministic and frozen before outcomes. No confirmation means no trade.

## Exits and non-overlap

Timeouts are fixed at 6h and 24h. Smooth stops require a completed mark close through the opposite six-hour shock-window extreme, then execute at the next trade-bar open. Jump-failure stops require a completed mark close beyond the dominant jump-bar extreme in the original shock direction, then execute at the next trade-bar open.

Each definition runs independently. Within symbol and definition, onsets are chronological; a later onset is skipped only while an actual position remains open. Eligibility resumes after the actual executed exit. No nominal maximum-hold preblocking and no combined portfolio are allowed. Accepted plus skipped rows must reconcile to eligible rows in a complete skip ledger.

## Boundary and invalid-row policy

Confirmation, entry, every stop-monitoring bar, selected timeout, funding accounting, and next-open execution must lie wholly inside the train interval. Mixed 2025/2026 payloads must not be opened. Boundary-crossing intervals are excluded; no artificial endpoint close is allowed.

Fail closed on missing next-open trade bars, missing mark bars during monitoring, non-positive structural stop distance, a stop already breached before entry, non-finite price or funding, known lifecycle-invalid intervals, duplicate economic addresses, or same-bar ambiguity. There are no touch fills, passive fills, partial fills, adds, leverage optimization, or inferred intrabar ordering.

## Outcome units, costs, and funding

Exposure is fixed notional; structural stop distance does not size trades. Primary units are fixed-notional net return and basis points. Structural R is diagnostic only, with full denominator distributions and extremes, and cannot be the sole permission metric. No post-outcome denominator floor is permitted.

- Base costs: 5 bps taker per side plus 4 bps round-trip slippage.
- Stress costs: 10 bps taker per side plus 12 bps round-trip slippage.
- Full-period gross and fee/slippage-net results are kill-screen evidence only.
- Fully exact-funded rows are the primary funding-valid subset and are reported by calendar period.
- Mixed and imputed full-period funding results are sensitivity only and never promotion evidence.
- Zero-funding-boundary rows are separate.
- No pooled funding partition can rescue a definition.

## Level-3 permission-to-test-controls rule

For a primary definition under base costs, all must hold: at least 100 executed trades; at least 20 trades in each of 2023, 2024, and 2025; positive fixed-notional mean and median net return; canonical-episode bootstrap 95% CI lower bound at least -5 bps; no symbol above 25% of aggregate net PnL; no canonical episode above 10%; no year above 70% of aggregate positive net PnL; and stress mean at least -10 bps. Bootstrap seed is `{BOOTSTRAP_SEED}` with exactly `{BOOTSTRAP_RESAMPLES:,}` resamples.

These gates permit Level-4 controls only. They do not establish incremental value, validation, robustness, promotion, or live readiness. If no primary definition passes, C01 stops at Level 3. BTC-only and secondary branches cannot rescue it.

## Claim boundary

A passing result may be described only as: "train-period event-ledger economics sufficient to justify predeclared controls under current-roster, funding, execution, multiplicity, and lifecycle caps."
"""


def level4_contract() -> str:
    return """# C01 Level-4 Control Contract

This contract is pre-registered only. Controls must not run unless a primary Level-3 definition passes every frozen gate.

Control attempts are: (1) raw 6h USD shock without residualization; (2) residual shock without path separation; (3) raw-return path classification at the same causal timestamps; (4) matched non-events; and (5) BTC-only residual robustness. Each is an additional registered attempt. BTC-only cannot rescue primary failure.

Matched non-events use exactly one control per event with the same symbol, calendar year, and direction. Frozen calipers are lagged 24h volatility within 20%, absolute BTC 6h return within 50 bps, and absolute ETH 6h return within 50 bps. A control cannot lie inside any same-symbol C01 canonical episode and must be at least 48h from event onset. Select the nearest deterministic match, breaking ties by timestamp. If no match exists, mark unavailable. Calipers may not widen after outcomes.

Candidate and control identities must be frozen before any control outcome read. Control adequacy cannot be inferred from a Level-3 result.
"""


def build(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage2c = Path(args.stage2c_root).resolve()
    package = Path(args.external_package_root).resolve()
    verify_authority(stage2c, package)

    disposition = package_disposition(package)
    write_json(output / "EXTERNAL_REVIEW_PACKAGE_DISPOSITION.json", disposition)
    gap_lines = "\n".join(
        f"- `{gap['item']}`: `{gap['status']}`; {gap['effect']}"
        + (f"; exact deferred task `{gap['exact_task']}`" if gap.get("exact_task") else "")
        for gap in disposition["missing_items"]
    )
    (output / "EXTERNAL_REVIEW_PACKAGE_DISPOSITION.md").write_text(f"""# External Review Package Disposition

Protocol disposition: `closed_by_claim_narrowing`.
Package role: `strategic_and_continuity_review_only`.
Release-ready for independent reproduction: `false`.

The immutable source package remains recorded as `blocked_by_protocol_issue`; this superseding disposition closes only the ambiguity that it is an open prerequisite workflow. Its hashes and archives remain useful for history, supersession/negative-decision continuity, broad multiplicity context, and nearest-family identification. It cannot support independent row-level reproduction, validation-grade claims, proof all old tests ran, exact cross-family episode overlap, or promotion of C01 or any prior family.

## Missing evidence retained

{gap_lines}

No economics were regenerated and no protected outcome was opened.
""", encoding="utf-8")

    definitions = definition_register()
    write_register(output / "C01_LEVEL3_DEFINITION_REGISTER.csv", definitions)
    contract_path = output / "C01_FINAL_LEVEL3_ECONOMIC_CONTRACT.md"
    contract_path.write_text(final_contract(), encoding="utf-8")
    contract_hash = sha256_file(contract_path)
    (output / "C01_LEVEL4_CONTROL_CONTRACT.md").write_text(level4_contract(), encoding="utf-8")

    rules = {
        "contract_hash": contract_hash,
        "bootstrap": {"cluster": "canonical_episode_id", "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
        "definition_counts": {"total": 16, "primary": 8, "robustness_only": 8},
        "gate_names": list(LEVEL3_GATE_NAMES),
        "all_gates_required": True,
        "permission_only": "Level4_controls",
        "primary_model": PRIMARY_MODEL,
        "robustness_model_cannot_rescue": True,
        "funding_evidence_hierarchy": [
            "full_period_gross_and_fee_slippage_net_kill_screen_only",
            "fully_exact_funded_primary_subset_by_calendar_period",
            "mixed_and_imputed_full_period_sensitivity_only",
            "zero_boundary_rows_separate",
        ],
        "no_pooled_funding_rescue": True,
        "protected_outcomes_opened": False,
        "economic_outputs_computed": False,
    }
    write_json(output / "C01_LEVEL3_DECISION_RULES.json", rules)

    expected_root = "results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_<UTC_SUFFIX>"
    repository_root = Path.cwd().resolve()
    output_arg = output.relative_to(repository_root)
    stage2c_arg = stage2c.relative_to(repository_root)
    command = (
        "./.venv/bin/python tools/run_kraken_c01_level3_economic.py "
        f"--contract {output_arg / 'C01_FINAL_LEVEL3_ECONOMIC_CONTRACT.md'} "
        f"--definition-register {output_arg / 'C01_LEVEL3_DEFINITION_REGISTER.csv'} "
        f"--event-tape {stage2c_arg / 'C01_ONSET_EVENT_TAPE.parquet'} "
        f"--run-root {expected_root} --execute-economic-run"
    )
    (output / "C01_PRERUN_APPROVAL_PACKET.md").write_text(f"""# C01 Level-3 Pre-Run Approval Packet

Final contract SHA-256: `{contract_hash}`.

Input hashes: generator `{GENERATOR_HASH}`; draft `{ECONOMIC_DRAFT_HASH}`; feature `{FEATURE_HASH}`; cohort `{COHORT_HASH}`; reference panel `{REFERENCE_PANEL_HASH}`.

Expected fresh run root: `{expected_root}`.

Frozen later command interface (do not execute without a separate economic-run authorization):

```bash
{command}
```

The later authorized execution task must first implement or verify the named runner at its approved commit, substitute an actual UTC suffix, confirm all input hashes, and refuse an existing root. This task does not create or execute the economic runner.

Rollback: stop before outcome access, preserve any incomplete fresh root as failed provenance, and do not mutate Stage 2B, Stage 2C1, the external package, or this contract root.

Prohibited: protected or mixed 2025/2026 reads; threshold or definition changes; event caps/sampling; artificial boundary closes; pooled funding rescue; nominal-hold preblocking; controls before a primary Level-3 pass; validation/promotion/live claims; overwriting prior roots.
""", encoding="utf-8")

    (output / "COMPLETION.md").write_text(
        "# Completion\n\nStatus: `ready_for_human_C01_Level3_run_approval`. "
        "The package blocker is closed only by claim narrowing and remains release-ready false. "
        "The 16-definition C01 contract is frozen. No economics or protected outcomes were read.\n",
        encoding="utf-8",
    )
    (output / "NEXT_ACTION.md").write_text(
        "# Next Action\n\nHuman review may authorize one separate C01 Level-3 implementation/execution task against the exact final contract hash. "
        "Do not execute the frozen command from this archive without that authorization.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ready_for_human_C01_Level3_run_approval", "contract_hash": contract_hash}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage2c-root", required=True)
    parser.add_argument("--external-package-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
