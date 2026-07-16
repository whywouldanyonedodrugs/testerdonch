#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd

from tools import kraken_a1_balanced_50 as funding_helpers
from tools import kraken_shared_funding_consumer as funding_consumer
from tools import run_kraken_a1_funding_corrected_balanced_50shard as funding_run
from tools import run_kraken_family_engine_aggregate_first_sweep as runner


RUN_ROOT = Path("results/rebaseline/phase_kraken_a1_selected_key_hash_canonicalization_repair_20260711_v1")
STOPPED_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_funding_corrected_balanced_50shard_20260711_v1")
FIRST_PACK_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_funding_policy_universe_repair_20260709_v1")
FEATURE_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_feature_mask_compiler_repair_20260709_v1_20260709_112535")


def write_csv(path: Path, frame: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_key_set(frame: pd.DataFrame) -> set[tuple[str, str, str]]:
    if frame.empty:
        return set()
    return set(zip(
        frame["candidate_definition_id"].astype(str),
        frame.get("symbol", frame.get("symbol_id", pd.Series(dtype=str))).astype(str),
        pd.to_datetime(frame["decision_ts"], utc=True, errors="coerce").astype(str),
    ))


def source_selected_keys(root: Path) -> pd.DataFrame:
    paths = sorted((root / "cache").glob("*_selected_event_key_shards/*.csv"))
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def assess_prior_roots(canonical_by_candidate: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    impact_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    stopped_compiler = source_selected_keys(STOPPED_ROOT)
    for manifest_path in sorted((STOPPED_ROOT / "aggregate_shards").glob("*/shard_manifest.json")):
        shard_dir = manifest_path.parent
        selected = pd.read_csv(shard_dir / "selected_keys.csv")
        candidate_ids = set(selected["candidate_definition_id"].astype(str))
        expected_hashes = {canonical_by_candidate.get(cid, "") for cid in candidate_ids} - {""}
        baseline = stopped_compiler[stopped_compiler["candidate_definition_id"].astype(str).isin(candidate_ids)].copy()
        old_keys = event_key_set(selected)
        compiler_keys = event_key_set(baseline)
        comparison_rows.append({
            "root": str(STOPPED_ROOT), "shard_id": shard_dir.name,
            "selected_event_rows": len(selected), "compiler_event_rows": len(baseline),
            "missing_from_shard": len(compiler_keys - old_keys), "extra_in_shard": len(old_keys - compiler_keys),
            "canonical_hash_count": len(expected_hashes),
            "selected_events_changed": old_keys != compiler_keys,
            "status": "pass" if old_keys == compiler_keys and len(expected_hashes) == 1 else "fail",
        })
    stopped_changed = any(bool(row["selected_events_changed"]) for row in comparison_rows)
    impact_rows.append({
        "root": str(STOPPED_ROOT), "sample_scope": "all_completed_shards",
        "rows_checked": sum(int(row["selected_event_rows"]) for row in comparison_rows),
        "selected_events_changed": stopped_changed,
        "classification": "selected_events_changed_results_quarantined" if stopped_changed else "cache_lineage_invalid_rerun_required",
        "rerun_required": True,
    })
    for root, label in [(FIRST_PACK_ROOT, "deterministic_all_available"), (FEATURE_ROOT, "deterministic_first_50000_rows")]:
        selected = source_selected_keys(root)
        if len(selected) > 50000:
            selected = selected.assign(_sample=selected["candidate_definition_id"].astype(str).map(lambda x: hashlib.sha256(x.encode()).hexdigest())).sort_values("_sample", kind="mergesort").head(50000).drop(columns="_sample")
        hashes = selected["candidate_definition_id"].astype(str).map(canonical_by_candidate)
        missing = int(hashes.isna().sum())
        # Exit variants sharing a canonical policy must have the same symbol/decision address set.
        normalized = selected.assign(canonical_hash=hashes).dropna(subset=["canonical_hash"])
        inconsistent_groups = 0
        for _, group in normalized.groupby("canonical_hash", sort=True):
            sets = []
            for _, candidate in group.groupby("candidate_definition_id", sort=True):
                sets.append(set(zip(candidate.get("symbol", candidate.get("symbol_id")).astype(str), pd.to_datetime(candidate["decision_ts"], utc=True).astype(str))))
            inconsistent_groups += int(bool(sets) and any(value != sets[0] for value in sets[1:]))
        classification = "hash_identity_only_results_semantically_unchanged" if missing == 0 and inconsistent_groups == 0 else "selected_events_changed_results_quarantined"
        impact_rows.append({
            "root": str(root), "sample_scope": label, "rows_checked": len(selected),
            "missing_candidate_mapping": missing, "inconsistent_exit_fanout_groups": inconsistent_groups,
            "selected_events_changed": inconsistent_groups > 0,
            "classification": classification,
            "rerun_required": classification != "hash_identity_only_results_semantically_unchanged",
        })
    return pd.DataFrame(impact_rows), pd.DataFrame(comparison_rows)


def main() -> int:
    if RUN_ROOT.exists() and any(RUN_ROOT.iterdir()):
        raise RuntimeError(f"repair root exists and is nonempty: {RUN_ROOT}")
    started = time.monotonic()
    for rel in ["diagnostic", "contract", "audit", "smoke", "compact_review_bundle"]:
        (RUN_ROOT / rel).mkdir(parents=True, exist_ok=True)
    args = runner.parse_args(["--phase-profile", runner.A1_COMPRESSION_PRODUCTION_SHARDED_AGGREGATE_PHASE_PROFILE, "--run-root", str(RUN_ROOT), "--disable-telegram", "--start", "2024-01-01", "--end", "2025-12-31"])
    ctx = runner.Context(args=args, run_root=RUN_ROOT, start=pd.Timestamp("2024-01-01", tz="UTC"), end=pd.Timestamp("2025-12-31", tz="UTC"), notifier=None)
    manifest = runner.load_a1_compression_manifest()
    first = runner.a1_definitions_with_selected_key_hash(manifest, ctx)
    second = runner.a1_definitions_with_selected_key_hash(first, ctx)
    payloads = first.apply(lambda row: json.dumps(runner.a1_selected_key_policy_payload(row.to_dict()), sort_keys=True, separators=(",", ":"), allow_nan=False), axis=1)
    audit = pd.DataFrame({
        "candidate_definition_id": first["candidate_definition_id"],
        "entry_spec_id": first["entry_spec_id"],
        "exit_policy_id": first["exit_policy_id"],
        "canonical_policy_hash": first["selected_key_policy_hash"],
        "recomputed_policy_hash": second["selected_key_policy_hash"],
        "canonical_policy_vector_hash": payloads.map(lambda x: hashlib.sha256(x.encode()).hexdigest()),
    })
    audit["idempotence_failure"] = audit["canonical_policy_hash"] != audit["recomputed_policy_hash"]
    audit["status"] = audit["idempotence_failure"].map({False: "pass", True: "fail"})
    write_csv(RUN_ROOT / "audit/full_manifest_hash_idempotence.csv", audit)
    collision = audit.groupby("canonical_policy_hash", sort=True).agg(
        definition_count=("candidate_definition_id", "size"),
        exit_policy_count=("exit_policy_id", "nunique"),
        canonical_vector_count=("canonical_policy_vector_hash", "nunique"),
    ).reset_index()
    collision["collision"] = collision["canonical_vector_count"].gt(1)
    collision["status"] = ((collision["definition_count"] == 8) & (collision["exit_policy_count"] == 8) & ~collision["collision"]).map({True: "pass", False: "fail"})
    write_csv(RUN_ROOT / "audit/hash_collision_audit.csv", collision)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "manifest.csv"
        first.to_csv(path, index=False)
        roundtrip = pd.read_csv(path)
        roundtrip_hashed = runner.a1_definitions_with_selected_key_hash(roundtrip, ctx)
    shuffled_failures = 0
    roundtrip_rows = []
    for position, row in first.iterrows():
        original = str(row["selected_key_policy_hash"])
        csv_hash = str(roundtrip_hashed.loc[position, "selected_key_policy_hash"])
        shuffled_hash = runner.a1_selected_key_policy_hash(dict(reversed(list(row.to_dict().items()))))
        shuffled_failures += int(shuffled_hash != original)
        roundtrip_rows.append({"candidate_definition_id": row["candidate_definition_id"], "canonical_hash": original, "csv_roundtrip_hash": csv_hash, "shuffled_order_hash": shuffled_hash, "csv_roundtrip_match": csv_hash == original, "shuffled_order_match": shuffled_hash == original, "status": "pass" if csv_hash == original and shuffled_hash == original else "fail"})
    write_csv(RUN_ROOT / "audit/csv_roundtrip_hash_audit.csv", roundtrip_rows)
    canonical_by_candidate = dict(zip(first["candidate_definition_id"].astype(str), first["selected_key_policy_hash"].astype(str)))
    impact, comparison = assess_prior_roots(canonical_by_candidate)
    write_csv(RUN_ROOT / "audit/prior_root_impact_assessment.csv", impact)
    write_csv(RUN_ROOT / "audit/completed_shard_selected_event_comparison.csv", comparison)
    (RUN_ROOT / "diagnostic/hash_instability_root_cause.md").write_text(
        "# Hash Instability Root Cause\n\nThe v1 selected-key hash used a blacklist over a runtime-enriched definition dictionary. It therefore included `selected_key_policy_hash` itself on recomputation, plus mutable derived/report fields not covered by the blacklist. All 1,440 rows changed hash when hashed twice. The repair uses an explicit selection-semantic allowlist, typed null/boolean/integer/float normalization, sorted deterministic JSON, a protected-boundary field, and a versioned signal-semantics field.\n",
        encoding="utf-8",
    )
    (RUN_ROOT / "contract/canonical_selected_key_policy_contract.md").write_text(
        "# Canonical Selected-Key Policy Contract\n\nOnly entry and selection semantics listed in `A1_SELECTED_KEY_POLICY_FIELDS` participate. Exit, fee, slippage, outcome funding mode, runtime, reporting, cap labels, candidate identity, parameter hashes, and existing selected-key hashes are excluded. Values are typed and null-normalized; JSON keys are sorted; NaN is forbidden. Contract version: `a1_selected_key_policy_v2_20260711`.\n",
        encoding="utf-8",
    )
    # Fresh two-shard integration smoke on bounded real-data mechanical windows.
    smoke_root = RUN_ROOT / "smoke/two_shard_run"
    smoke_root.mkdir(parents=True, exist_ok=True)
    smoke_ctx = runner.Context(args=args, run_root=smoke_root, start=ctx.start, end=ctx.end, notifier=None)
    plan = runner.a1_build_full_shard_plan(smoke_ctx, manifest, runner.a1_feature_mask_repair_root())
    first_pack_status = []
    for path in sorted((FIRST_PACK_ROOT / "aggregate_shards").glob("*/shard_manifest.json")):
        first_pack_status.append(json.loads(path.read_text(encoding="utf-8")))
    first_pack_status = pd.DataFrame(first_pack_status)
    selected_entries = []
    for lane in ["a1_impulse_base_breakout", "h12_rv_compression_breakout"]:
        lane_status = first_pack_status[first_pack_status["definition_lane"].eq(lane)].sort_values(["outcome_event_count", "entry_spec_id"], kind="mergesort")
        if lane_status.empty:
            raise RuntimeError(f"funding-covered prior first-pack smoke source missing for {lane}")
        selected_entries.append(str(lane_status.iloc[0]["entry_spec_id"]))
    selected_plan = plan[plan["entry_spec_id"].astype(str).isin(selected_entries)].sort_values("definition_lane", kind="mergesort").reset_index(drop=True)
    definitions = runner.a1_definitions_for_selected_key_specs(manifest, smoke_ctx, selected_plan["selected_key_policy_hash"])
    windows = runner.a1_full_train_window_manifest(smoke_ctx, manifest)
    status_rows = []
    consistency_rows = []
    for _, shard in selected_plan.iterrows():
        spec_hash = str(shard["selected_key_policy_hash"])
        shard_defs = definitions[definitions["selected_key_policy_hash"].eq(spec_hash)].copy()
        result = runner.a1_execute_economic_shard(smoke_ctx, shard_row=shard.to_dict(), definitions=shard_defs, feature_root=smoke_root, windows=windows)
        status_rows.append(result)
        shard_dir = smoke_root / "aggregate_shards" / str(shard["shard_id"])
        selected = pd.read_csv(shard_dir / "selected_keys.csv")
        outcomes = pd.read_parquet(shard_dir / "outcome_events.parquet")
        selected_manifest = pd.read_csv(shard_dir / "selected_key_manifest.csv").iloc[0]
        outcome_manifest = pd.read_csv(shard_dir / "outcome_cache_manifest.csv").iloc[0]
        policy_fields = json.loads(selected_manifest["policy_fields"])
        sources = {
            "plan": spec_hash,
            "definition_rows": ";".join(sorted(set(shard_defs["selected_key_policy_hash"].astype(str)))),
            "selected_key_rows": ";".join(sorted(set(selected["selected_key_policy_hash"].astype(str)))),
            "selected_key_manifest_policy": str(policy_fields.get("selected_key_policy_hash", "")),
            "outcome_rows": ";".join(sorted(set(outcomes["selected_key_policy_hash"].astype(str)))),
            "shard_manifest": str(result.get("selected_key_policy_hash", "")),
        }
        for source, value in sources.items():
            consistency_rows.append({"shard_id": shard["shard_id"], "source": source, "expected_canonical_hash": spec_hash, "observed_canonical_hash": value, "match": value == spec_hash, "status": "pass" if value == spec_hash else "fail"})
    status = pd.DataFrame(status_rows)
    consistency = pd.DataFrame(consistency_rows)
    ledgers = [pd.read_parquet(path) for path in sorted((smoke_root / "aggregate_shards").glob("*/outcome_events.parquet"))]
    events = funding_consumer.normalize_frozen_events(pd.concat(ledgers, ignore_index=True), "a1")
    boundaries = funding_consumer.build_event_boundary_rows(events)
    panel = funding_run.load_frozen_panel()
    panel, _ = funding_helpers.extend_frozen_panel(panel, boundaries)
    joined = funding_consumer.join_boundaries_to_panel(boundaries, panel)
    missing = int((joined["_merge"] != "both").sum())
    duplicate = int(joined.duplicated(["event_key", "boundary_ts"]).sum())
    write_csv(RUN_ROOT / "smoke/two_shard_hash_consistency_audit.csv", consistency)
    write_csv(RUN_ROOT / "smoke/two_shard_status_summary.csv", status)
    unique_hashes = int(first["selected_key_policy_hash"].nunique())
    idempotence_failures = int(audit["idempotence_failure"].sum())
    collisions = int(collision["collision"].sum())
    hash_mismatches = int((~consistency["match"]).sum())
    smoke_pass = bool(len(status) == 2 and status["status"].eq("complete").all() and hash_mismatches == 0 and missing == 0 and duplicate == 0 and status["protected_interval_violations"].sum() == 0 and status["decision_input_leak_violations"].sum() == 0)
    pass_gate = bool(len(first) == 1440 and unique_hashes == 180 and collision["definition_count"].eq(8).all() and collision["exit_policy_count"].eq(8).all() and idempotence_failures == 0 and collisions == 0 and all(row["status"] == "pass" for row in roundtrip_rows) and smoke_pass)
    summary = {
        "run_root": str(RUN_ROOT), "status": "complete" if pass_gate else "blocked", "code_modified": True,
        "unique_canonical_selected_key_hashes": unique_hashes, "definitions_audited": len(first),
        "idempotence_failures": idempotence_failures, "collision_count": collisions,
        "csv_roundtrip_failures": int(sum(not row["csv_roundtrip_match"] for row in roundtrip_rows)),
        "shuffled_order_failures": shuffled_failures, "prior_selected_events_changed": bool(impact["selected_events_changed"].fillna(False).any()),
        "prior_roots_requiring_rerun": impact.loc[impact["rerun_required"], "root"].tolist(),
        "two_shard_smoke_pass": smoke_pass, "two_shard_hash_mismatch_count": hash_mismatches,
        "missing_funding_joins": missing, "duplicate_funding_joins": duplicate,
        "balanced_50_fresh_restart_allowed": pass_gate,
        "runtime_seconds": time.monotonic() - started, "compact_bundle_path": str(RUN_ROOT / "compact_review_bundle"),
    }
    write_json(RUN_ROOT / "decision_summary.json", summary)
    required = [
        "diagnostic/hash_instability_root_cause.md", "contract/canonical_selected_key_policy_contract.md",
        "audit/full_manifest_hash_idempotence.csv", "audit/hash_collision_audit.csv", "audit/csv_roundtrip_hash_audit.csv",
        "audit/prior_root_impact_assessment.csv", "audit/completed_shard_selected_event_comparison.csv",
        "smoke/two_shard_hash_consistency_audit.csv", "smoke/two_shard_status_summary.csv", "decision_summary.json",
    ]
    bundle_rows = []
    for rel in required:
        source = RUN_ROOT / rel
        target = RUN_ROOT / "compact_review_bundle" / rel.replace("/", "__")
        shutil.copy2(source, target)
        bundle_rows.append({"source": rel, "bundle_path": str(target.relative_to(RUN_ROOT)), "sha256": sha256(target)})
    write_csv(RUN_ROOT / "compact_review_bundle/compact_bundle_manifest.csv", bundle_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if pass_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
