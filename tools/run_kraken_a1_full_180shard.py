#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from tools import run_kraken_family_engine_aggregate_first_sweep as runner


REFERENCE_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_funding_corrected_balanced_50shard_canonical_20260711_v1_20260711_140327")
FUNDING_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
EXPECTED_MODEL_HASH = "0054af0ee40740e39739bfade92f342867bb208a4fe7ed15b151a8a0a838d072"
CONTRACT_VERSION = "a1_selected_key_policy_v2_20260711"


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def verify_reference_funding() -> dict[str, Any]:
    audit = pd.read_csv(REFERENCE_ROOT / "funding/panel_extension_audit.csv").iloc[0].to_dict()
    if str(audit.get("status")) != "pass" or bool(audit.get("model_refit")):
        raise RuntimeError("reference funding model audit is not reusable")
    if str(audit.get("expected_model_hash")) != EXPECTED_MODEL_HASH or str(audit.get("actual_model_hash")) != EXPECTED_MODEL_HASH:
        raise RuntimeError("reference funding model hash mismatch")
    return audit


def verify_and_import_shard(root: Path, plan_row: pd.Series, funding_audit: dict[str, Any]) -> dict[str, Any]:
    shard_id = str(plan_row["shard_id"])
    source = REFERENCE_ROOT / "aggregate_shards" / shard_id
    destination = root / "aggregate_shards" / shard_id
    expected = str(plan_row["selected_key_policy_hash"])
    if not source.exists():
        raise RuntimeError(f"reference shard missing: {shard_id}")
    manifest = json.loads((source / "shard_manifest.json").read_text())
    selected_manifest = pd.read_csv(source / "selected_key_manifest.csv").iloc[0]
    outcome_manifest = pd.read_csv(source / "outcome_cache_manifest.csv").iloc[0]
    selected = pd.read_csv(source / "selected_keys.csv")
    outcomes = pd.read_parquet(source / "outcome_events.parquet")
    validation = runner.a1_validate_finalized_aggregate_shard(REFERENCE_ROOT, shard_id)
    checks = {
        "finalized_shard": validation.get("status") == "pass" and manifest.get("status") == "complete",
        "canonical_policy_hash": manifest.get("selected_key_policy_hash") == expected and set(selected["selected_key_policy_hash"].astype(str)) == {expected} and set(outcomes["selected_key_policy_hash"].astype(str)) == {expected},
        "contract_version": manifest.get("selected_key_policy_contract_version") == CONTRACT_VERSION and selected_manifest.get("selected_key_policy_contract_version") == CONTRACT_VERSION and outcome_manifest.get("selected_key_policy_contract_version") == CONTRACT_VERSION,
        "selected_key_content_hash": str(selected_manifest.get("content_hash")) == runner.canonical_frame_hash(selected, sort_keys=["candidate_identity_hash", "symbol_id", "decision_ts"]),
        "selected_key_manifest_hash": str(manifest.get("selected_key_manifest_hash")) == runner.sha256_file(source / "selected_key_manifest.csv"),
        "outcome_content_hash": str(outcome_manifest.get("content_hash")) == runner.canonical_frame_hash(outcomes, sort_keys=["candidate_definition_id", "symbol", "decision_ts", "event_id"]),
        "outcome_manifest_hash": str(manifest.get("outcome_cache_manifest_hash")) == runner.sha256_file(source / "outcome_cache_manifest.csv"),
        "aggregate_hash": validation.get("status") == "pass",
        "funding_model_hash": str(funding_audit.get("actual_model_hash")) == EXPECTED_MODEL_HASH,
        "protected_boundary": str(selected_manifest.get("protected_train_boundary")) == str(runner.PROTECTED_TS) and str(outcome_manifest.get("protected_train_boundary")) == str(runner.PROTECTED_TS) and int(manifest.get("protected_interval_violations", 1)) == 0,
        "definition_fanout": int(manifest.get("definition_count", 0)) == 8 and int(manifest.get("exit_policy_count", 0)) == 8,
        "cache_status": str(selected_manifest.get("status")) == "pass" and str(outcome_manifest.get("status")) == "pass",
    }
    if not all(checks.values()):
        raise RuntimeError(f"reference shard import audit failed: {shard_id}: {[key for key, value in checks.items() if not value]}")
    if destination.exists():
        current = runner.a1_validate_finalized_aggregate_shard(root, shard_id)
        if current.get("status") != "pass":
            raise RuntimeError(f"existing imported shard is invalid: {shard_id}")
    else:
        temporary = root / "aggregate_shards" / f".tmp_import_{shard_id}"
        if temporary.exists():
            raise RuntimeError(f"stale import directory: {temporary}")
        shutil.copytree(source, temporary)
        os.replace(temporary, destination)
    return {
        "shard_id": shard_id, "selected_key_policy_hash": expected, "definition_lane": plan_row["definition_lane"],
        "canonical_policy_hash_pass": checks["canonical_policy_hash"], "policy_contract_version_pass": checks["contract_version"],
        "selected_key_content_hash_pass": checks["selected_key_content_hash"], "selected_key_manifest_hash_pass": checks["selected_key_manifest_hash"],
        "outcome_content_hash_pass": checks["outcome_content_hash"], "outcome_manifest_hash_pass": checks["outcome_manifest_hash"],
        "aggregate_hash_pass": checks["aggregate_hash"], "funding_model_hash": EXPECTED_MODEL_HASH,
        "protected_boundary_pass": checks["protected_boundary"], "definition_count": 8, "exit_policy_count": 8,
        "source_root": str(REFERENCE_ROOT), "imported_immutable": True, "status": "pass",
    }


def notify(ctx: runner.Context, title: str, body: str, level: str = "info") -> None:
    ctx.notifier.send(title, body, level=level)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root)
    if root.exists() and any(root.iterdir()) and not args.resume:
        allowed = {"run.log"}
        if {path.name for path in root.iterdir()} - allowed:
            raise RuntimeError(f"fresh run root required: {root}")
    runner_args = runner.parse_args([
        "--phase-profile", runner.A1_COMPRESSION_PRODUCTION_SHARDED_AGGREGATE_PHASE_PROFILE,
        "--run-root", str(root), "--start", "2024-01-01", "--end", "2025-12-31",
        *( ["--resume"] if args.resume else [] ),
    ])
    ctx = runner.init_context(runner_args)
    started = time.monotonic()
    notify(ctx, "A1 full 180-shard train scan started", f"Run root: {root}\nPlan: 180 shards; import 50 verified; compute 130 new")
    manifest = runner.load_a1_compression_manifest()
    definitions = runner.a1_definitions_with_selected_key_hash(manifest, ctx)
    plan = runner.a1_build_full_shard_plan(ctx, manifest, None)
    if len(plan) != 180 or len(definitions) != 1440 or definitions["selected_key_policy_hash"].nunique() != 180 or not definitions.groupby("selected_key_policy_hash").size().eq(8).all():
        raise RuntimeError("full canonical manifest contract failed")
    reference_plan = pd.read_csv(REFERENCE_ROOT / "shards/selected_50_shard_plan.csv")
    imported_hashes = set(reference_plan["selected_key_policy_hash"].astype(str))
    if len(imported_hashes) != 50 or not imported_hashes.issubset(set(plan["selected_key_policy_hash"].astype(str))):
        raise RuntimeError("reference 50 hashes do not map exactly into full plan")
    plan["imported_reference_shard"] = plan["selected_key_policy_hash"].astype(str).isin(imported_hashes)
    plan["selected_key_policy_contract_version"] = CONTRACT_VERSION
    write_csv(root / "shards/full_manifest_shard_plan.csv", plan)
    recomputed = runner.a1_definitions_with_selected_key_hash(definitions, ctx)
    old_hashes = set(pd.read_csv("results/rebaseline/phase_kraken_a1_compression_funding_corrected_balanced_50shard_20260711_v1/shards/selected_50_shard_plan.csv")["selected_key_policy_hash"].astype(str))
    lineage = pd.DataFrame({
        "candidate_definition_id": definitions["candidate_definition_id"],
        "planned_canonical_hash": definitions["selected_key_policy_hash"],
        "recomputed_canonical_hash": recomputed["selected_key_policy_hash"],
    })
    lineage["contract_version"] = CONTRACT_VERSION
    lineage["old_v1_hash_reused"] = lineage["planned_canonical_hash"].astype(str).isin(old_hashes)
    lineage["status"] = ((lineage["planned_canonical_hash"] == lineage["recomputed_canonical_hash"]) & ~lineage["old_v1_hash_reused"]).map({True: "pass", False: "fail"})
    write_csv(root / "audit/canonical_hash_lineage_audit.csv", lineage)
    if not lineage["status"].eq("pass").all():
        raise RuntimeError("full manifest canonical lineage gate failed")
    funding_audit = verify_reference_funding()
    write_csv(root / "audit/funding_model_hash_audit.csv", [{"expected_model_hash": EXPECTED_MODEL_HASH, "reference_expected_hash": funding_audit.get("expected_model_hash"), "reference_actual_hash": funding_audit.get("actual_model_hash"), "model_refit": False, "status": "pass"}])

    import_rows = []
    for _, row in plan[plan["imported_reference_shard"]].iterrows():
        import_rows.append(verify_and_import_shard(root, row, funding_audit))
    write_csv(root / "shards/imported_50_shard_audit.csv", import_rows)
    if len(import_rows) != 50 or any(row["status"] != "pass" for row in import_rows):
        raise RuntimeError("50-shard import did not complete")
    runner.write_status(ctx, "running", "imported-50-verified")
    notify(ctx, "A1 full scan imported 50 verified shards", f"Run root: {root}\nImported: 50/50\nRemaining new shards: 130")
    windows = runner.a1_full_train_window_manifest(ctx, manifest)
    remaining = plan[~plan["imported_reference_shard"]].reset_index(drop=True)
    for index, (_, shard) in enumerate(remaining.iterrows(), start=1):
        spec_hash = str(shard["selected_key_policy_hash"])
        shard_defs = definitions[definitions["selected_key_policy_hash"].astype(str).eq(spec_hash)].copy()
        result = runner.a1_execute_economic_shard(ctx, shard_row=shard.to_dict(), definitions=shard_defs, feature_root=root, windows=windows)
        if result.get("status") not in {"complete", "pass"}:
            raise RuntimeError(f"new shard failed: {result}")
        total = 50 + index
        runner.a1_write_heartbeat(ctx, "a1-full-180-sharded-aggregate", shard_id=result.get("shard_id"), imported_shards=50, new_shards_completed=index, total_verified_shards=total, total_planned_shards=180)
        if total in {90, 135, 180}:
            notify(ctx, f"A1 full scan progress: {total}/180 shards", f"Run root: {root}\nImported: 50\nNew completed: {index}\nVerified total: {total}/180")
    validations = [runner.a1_validate_finalized_aggregate_shard(root, str(row.shard_id)) for row in plan.itertuples()]
    if len(validations) != 180 or any(row.get("status") != "pass" for row in validations):
        write_csv(root / "shards/shard_status_summary.csv", validations)
        raise RuntimeError("full reducer requires exactly 180/180 verified shards")
    manifests = [json.loads((root / "aggregate_shards" / str(row.shard_id) / "shard_manifest.json").read_text()) for row in plan.itertuples()]
    write_csv(root / "shards/shard_status_summary.csv", manifests)
    from tools.kraken_a1_full_streaming_reducer import finalize_full_scan
    summary = finalize_full_scan(root, plan, definitions, manifests, FUNDING_ROOT, EXPECTED_MODEL_HASH)
    summary["wall_runtime_seconds"] = time.monotonic() - started
    runner.write_json(root / "decision_summary.json", summary)
    runner.write_status(ctx, summary["status"], "full-180-complete")
    notify(ctx, "A1 full 180-shard train scan complete", f"Run root: {root}\nShards: 180/180\nDefinitions: {summary.get('definitions_scored')}\nEvents: {summary.get('events_scored')}\nStatus: {summary.get('status')}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        from tools.telegram_notify import TelegramNotifier, load_telegram_env_files
        load_telegram_env_files()
        class A:
            tg_bot_token = ""; tg_chat_id = ""; tg_auto_chat = False; disable_telegram = False; telegram_dry_run = False
        notifier = TelegramNotifier.from_args(A(), run_label="a1-full-180-shard")
        if notifier.enabled:
            notifier.send("A1 full 180-shard train scan failed", f"{type(exc).__name__}: {exc}")
        raise
