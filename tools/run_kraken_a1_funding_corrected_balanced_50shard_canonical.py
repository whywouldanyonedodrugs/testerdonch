#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools import kraken_a1_balanced_50 as balanced
from tools import kraken_funding_imputation as imputation
from tools import kraken_shared_funding_consumer as consumer
from tools import run_kraken_family_engine_aggregate_first_sweep as runner


DEFAULT_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_funding_corrected_balanced_50shard_canonical_20260711_v1")
FUNDING_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
FIRST_PACK_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_funding_policy_universe_repair_20260709_v1")


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen_panel() -> pd.DataFrame:
    manifest = pd.read_csv(FUNDING_ROOT / "funding/shared_funding_panel_manifest.csv")
    frames: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        path = FUNDING_ROOT / row.path
        if not path.exists():
            raise RuntimeError(f"missing frozen funding partition: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != int(row.row_count) or imputation.canonical_frame_hash(frame) != str(row.content_hash):
            raise RuntimeError(f"frozen funding partition validation failed: {path}")
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    return panel


def definition_scorecard(scenarios: pd.DataFrame) -> pd.DataFrame:
    keys = ["candidate_definition_id", "definition_lane", "exit_policy_id", "funding_mode", "slippage_round_trip_bps"]
    return scenarios.groupby(keys, dropna=False, sort=True).agg(
        event_count=("event_key", "nunique"),
        active_symbols=("symbol", "nunique"),
        median_net_R=("scenario_scaled_net_R", "median"),
        mean_net_R=("scenario_scaled_net_R", "mean"),
        total_net_R=("scenario_scaled_net_R", "sum"),
        positive_event_fraction=("scenario_scaled_net_R", lambda s: float((s > 0).mean())),
        exact_boundary_rows=("exact_boundary_rows", "sum"),
        imputed_boundary_rows=("imputed_boundary_rows", "sum"),
    ).reset_index()


def concentration_preview(scenarios: pd.DataFrame) -> pd.DataFrame:
    primary = scenarios[(scenarios["funding_mode"] == "central_imputed") & (scenarios["slippage_round_trip_bps"] == 4)].copy()
    primary["year_month"] = pd.to_datetime(primary["entry_ts"], utc=True).dt.strftime("%Y-%m")
    rows = []
    for cid, group in primary.groupby("candidate_definition_id", sort=True):
        total_abs = float(group["scenario_scaled_net_R"].abs().sum())
        symbol = group.groupby("symbol")["scenario_scaled_net_R"].sum().abs()
        month = group.groupby("year_month")["scenario_scaled_net_R"].sum().abs()
        symbol_month = group.groupby(["symbol", "year_month"])["scenario_scaled_net_R"].sum().abs()
        rows.append({
            "candidate_definition_id": cid,
            "definition_lane": group["definition_lane"].iloc[0],
            "event_count": len(group),
            "dominant_symbol_abs_contribution_share": float(symbol.max() / total_abs) if total_abs else np.nan,
            "dominant_month_abs_contribution_share": float(month.max() / total_abs) if total_abs else np.nan,
            "dominant_symbol_month_abs_contribution_share": float(symbol_month.max() / total_abs) if total_abs else np.nan,
            "top_event_abs_contribution_share": float(group["scenario_scaled_net_R"].abs().max() / total_abs) if total_abs else np.nan,
            "scope": "central_imputed_plus_4bps_preview_not_portfolio_return",
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--disable-telegram", action="store_true")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    allowed_launch_artifacts = {"run.log", "initial_launch_error.log", "second_launch_error.log"}
    existing = {path.name for path in run_root.iterdir()} if run_root.exists() else set()
    if existing - allowed_launch_artifacts:
        raise RuntimeError(f"run root exists with research artifacts: {run_root}: {sorted(existing)}")
    started = time.monotonic()
    runner_args = runner.parse_args([
        "--phase-profile", runner.A1_COMPRESSION_PRODUCTION_SHARDED_AGGREGATE_PHASE_PROFILE,
        "--run-root", str(run_root), "--start", "2024-01-01", "--end", "2025-12-31",
        *( ["--disable-telegram"] if args.disable_telegram else [] ),
    ])
    ctx = runner.init_context(runner_args)
    ctx.notifier.send(
        "A1 balanced 50-shard screen started",
        f"Run root: {run_root}\nScope: 50 train-only economic shards / 400 definitions",
    )
    for rel in ["aggregate", "forensics", "gates", "funding", "decision"]:
        (run_root / rel).mkdir(parents=True, exist_ok=True)
    manifest = runner.load_a1_compression_manifest()
    hashed = runner.a1_definitions_with_selected_key_hash(manifest, ctx)
    full_plan = runner.a1_build_full_shard_plan(ctx, manifest, runner.a1_feature_mask_repair_root())
    first_plan = pd.read_csv(FIRST_PACK_ROOT / "shards/first_pack_shard_manifest.csv")
    seed_entries = set(first_plan["entry_spec_id"].astype(str))
    canonical_seed_hashes = full_plan.loc[full_plan["entry_spec_id"].astype(str).isin(seed_entries), "selected_key_policy_hash"].astype(str)
    selected, diversity = balanced.select_balanced_50(full_plan, hashed, canonical_seed_hashes)
    selected["selected_key_policy_contract_version"] = runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION
    write_csv(run_root / "shards/selected_50_shard_plan.csv", selected)
    write_csv(run_root / "audit/parameter_diversity_audit.csv", diversity)
    selected_defs = runner.a1_definitions_for_selected_key_specs(manifest, ctx, selected["selected_key_policy_hash"])
    if len(selected_defs) != 400 or selected_defs["candidate_definition_id"].nunique() != 400:
        raise RuntimeError(f"selected plan must map to 400 definitions, found {len(selected_defs)}")
    recomputed = runner.a1_definitions_with_selected_key_hash(selected_defs, ctx)
    stopped_old_hashes = set(pd.read_csv(
        "results/rebaseline/phase_kraken_a1_compression_funding_corrected_balanced_50shard_20260711_v1/shards/selected_50_shard_plan.csv"
    )["selected_key_policy_hash"].astype(str))
    canonical_rows = []
    for cid, expected, observed in zip(
        selected_defs["candidate_definition_id"].astype(str),
        selected_defs["selected_key_policy_hash"].astype(str),
        recomputed["selected_key_policy_hash"].astype(str),
    ):
        canonical_rows.append({
            "candidate_definition_id": cid, "planned_canonical_hash": expected,
            "recomputed_canonical_hash": observed, "contract_version": runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION,
            "plan_definition_match": expected == observed, "old_v1_hash_reused": expected in stopped_old_hashes,
            "status": "pass" if expected == observed and expected not in stopped_old_hashes else "fail",
        })
    canonical_audit = pd.DataFrame(canonical_rows)
    write_csv(run_root / "audit/canonical_hash_lineage_audit.csv", canonical_audit)
    group_sizes = selected_defs.groupby("selected_key_policy_hash").size()
    if len(group_sizes) != 50 or not group_sizes.eq(8).all() or not canonical_audit["status"].eq("pass").all():
        raise RuntimeError("canonical pre-outcome lineage gate failed")
    windows = runner.a1_full_train_window_manifest(ctx, manifest)
    results = []
    for ordinal, row in selected.reset_index(drop=True).iterrows():
        shard_id = str(row["shard_id"])
        spec_hash = str(row["selected_key_policy_hash"])
        definitions = selected_defs[selected_defs["selected_key_policy_hash"].astype(str).eq(spec_hash)].copy()
        result = runner.a1_execute_economic_shard(ctx, shard_row=row.to_dict(), definitions=definitions, feature_root=run_root, windows=windows)
        result["reused_corrected_first_pack"] = False
        results.append(result)
        runner.a1_write_heartbeat(ctx, "a1-balanced-50-sharded-aggregate", shard_id=shard_id, shards_completed=ordinal + 1, shards_planned=50)
    status = pd.DataFrame(results)
    write_csv(run_root / "shards/shard_status_summary.csv", status)
    if len(status) != 50 or not status["status"].astype(str).isin({"pass", "complete"}).all():
        raise RuntimeError("50/50 verified shard gate failed")
    gate_frames = [pd.read_csv(path) for path in sorted((run_root / "gates").glob("a1shard_*_funding_gate_pre_post_event_counts.csv"))]
    gate_counts = pd.concat(gate_frames, ignore_index=True, sort=False) if gate_frames else pd.DataFrame()
    write_csv(run_root / "gates/funding_gate_pre_post_event_counts.csv", gate_counts)
    recognized_gates = {"exclude_top_20pct_positive_funding", "exclude_top_decile_positive_funding", "funding_aware_cap", "no_funding_gate_diagnostic_cap"}
    observed_gates = set(selected_defs["funding_gate"].astype(str))
    binding_rows = [{
        "funding_gate": gate, "recognized": gate in recognized_gates,
        "definition_count": int(selected_defs["funding_gate"].astype(str).eq(gate).sum()),
        "pre_gate_rows": int(pd.to_numeric(gate_counts.loc[gate_counts["funding_gate"].astype(str).eq(gate), "pre_funding_gate_definition_event_rows"], errors="coerce").fillna(0).sum()) if not gate_counts.empty else 0,
        "post_gate_rows": int(pd.to_numeric(gate_counts.loc[gate_counts["funding_gate"].astype(str).eq(gate), "post_funding_gate_definition_event_rows"], errors="coerce").fillna(0).sum()) if not gate_counts.empty else 0,
        "imputed_funding_used_for_gate": False,
        "status": "pass" if gate in recognized_gates else "fail",
    } for gate in sorted(observed_gates)]
    write_csv(run_root / "gates/funding_gate_binding_audit.csv", binding_rows)
    ledger_paths = [run_root / "aggregate_shards" / str(row.shard_id) / "outcome_events.parquet" for row in selected.itertuples()]
    if not all(path.exists() for path in ledger_paths):
        raise RuntimeError("one or more verified shards lack outcome event ledgers")
    events = consumer.normalize_frozen_events(pd.concat([pd.read_parquet(path) for path in ledger_paths], ignore_index=True), "a1")
    expected_hash_by_definition = selected_defs.set_index("candidate_definition_id")["selected_key_policy_hash"]
    expected_event_hash = events["candidate_definition_id"].map(expected_hash_by_definition)
    if "selected_key_policy_hash" not in events or events["selected_key_policy_hash"].isna().any() or not events["selected_key_policy_hash"].astype(str).eq(expected_event_hash.astype(str)).all():
        raise RuntimeError("event rows lost selected-key lineage")
    post_lineage_rows = canonical_rows.copy()
    for _, shard in selected.iterrows():
        shard_id = str(shard["shard_id"])
        expected = str(shard["selected_key_policy_hash"])
        shard_dir = run_root / "aggregate_shards" / shard_id
        selected_rows = pd.read_csv(shard_dir / "selected_keys.csv")
        outcome_rows = pd.read_parquet(shard_dir / "outcome_events.parquet")
        shard_manifest = json.loads((shard_dir / "shard_manifest.json").read_text())
        observed = {
            "selected_event_rows": set(selected_rows["selected_key_policy_hash"].astype(str)),
            "outcome_rows": set(outcome_rows["selected_key_policy_hash"].astype(str)),
            "shard_manifest": {str(shard_manifest.get("selected_key_policy_hash", ""))},
        }
        for source, values in observed.items():
            post_lineage_rows.append({
                "candidate_definition_id": "", "shard_id": shard_id, "artifact_source": source,
                "planned_canonical_hash": expected, "recomputed_canonical_hash": ";".join(sorted(values)),
                "contract_version": shard_manifest.get("selected_key_policy_contract_version", ""),
                "plan_definition_match": values == {expected}, "old_v1_hash_reused": bool(values & stopped_old_hashes),
                "status": "pass" if values == {expected} and not values & stopped_old_hashes else "fail",
            })
    canonical_audit = pd.DataFrame(post_lineage_rows)
    write_csv(run_root / "audit/canonical_hash_lineage_audit.csv", canonical_audit)
    canonical_mismatches = int((~canonical_audit["status"].eq("pass")).sum())
    if canonical_mismatches:
        raise RuntimeError(f"post-shard canonical lineage mismatch count: {canonical_mismatches}")
    if (events["exit_interval_end_ts"] >= consumer.PROTECTED_TS).any():
        raise RuntimeError("protected interval violation")
    boundaries = consumer.build_event_boundary_rows(events)
    frozen_panel = load_frozen_panel()
    panel, extension_audit = balanced.extend_frozen_panel_with_verified_model(
        frozen_panel, boundaries, str(FUNDING_ROOT)
    )
    write_csv(run_root / "funding/panel_extension_audit.csv", extension_audit)
    joined = consumer.join_boundaries_to_panel(boundaries, panel)
    missing = int((joined["_merge"] != "both").sum())
    duplicate = int(joined.duplicated(["event_key", "boundary_ts"]).sum())
    write_csv(run_root / "audit/funding_boundary_join_audit.csv", [{
        "required_boundary_rows": len(boundaries), "joined_boundary_rows": len(joined),
        "missing_boundary_joins": missing, "duplicate_boundary_joins": duplicate,
        "exact_rows_preserved": bool((joined.loc[joined["funding_exact"].fillna(False), "relativeFundingRate"] == joined.loc[joined["funding_exact"].fillna(False), "funding_rate_central"]).all()),
        "status": "pass" if missing == 0 and duplicate == 0 else "fail",
    }])
    if missing or duplicate:
        raise RuntimeError(f"funding boundary join gate failed: missing={missing} duplicate={duplicate}")
    rescored = consumer.aggregate_event_funding(events, joined)
    scenarios = balanced.scenario_event_rows(rescored, consumer.FUNDING_MODES, (4, 8, 12))
    scorecard = definition_scorecard(scenarios)
    scorecard["selected_key_policy_contract_version"] = runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION
    write_csv(run_root / "aggregate/definition_scorecard_50shard.csv", scorecard)
    lane = consumer.grouped_rescore(rescored, ["definition_lane"], family="a1")
    lane = lane[lane["slippage_round_trip_bps"].isin([4, 8, 12])]
    lane["selected_key_policy_contract_version"] = runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION
    exit_summary = consumer.grouped_rescore(rescored, ["exit_policy_id"], family="a1")
    exit_summary = exit_summary[exit_summary["slippage_round_trip_bps"].isin([4, 8, 12])]
    exit_summary["selected_key_policy_contract_version"] = runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION
    write_csv(run_root / "aggregate/lane_summary_50shard.csv", lane)
    write_csv(run_root / "aggregate/exit_policy_summary_50shard.csv", exit_summary)
    spec_summary = scenarios.groupby(["selected_key_policy_hash", "definition_lane", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(definitions=("candidate_definition_id", "nunique"), events=("event_key", "nunique"), median_event_net_R=("scenario_scaled_net_R", "median"), total_net_R=("scenario_scaled_net_R", "sum")).reset_index()
    spec_summary["selected_key_policy_contract_version"] = runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION
    write_csv(run_root / "aggregate/spec_robustness_summary.csv", spec_summary)
    scenarios["period_scope"] = np.select([
        pd.to_datetime(scenarios["entry_ts"], utc=True).dt.year.eq(2024),
        pd.to_datetime(scenarios["entry_ts"], utc=True).between(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"), inclusive="left"),
        pd.to_datetime(scenarios["entry_ts"], utc=True).ge(pd.Timestamp("2025-07-01", tz="UTC")),
    ], ["2024", "2025_h1", "2025_h2"], default="other")
    period = scenarios.groupby(["definition_lane", "period_scope", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(events=("event_key", "nunique"), definitions=("candidate_definition_id", "nunique"), median_event_net_R=("scenario_scaled_net_R", "median"), total_net_R=("scenario_scaled_net_R", "sum")).reset_index()
    write_csv(run_root / "aggregate/period_support_summary.csv", period)
    concentration = concentration_preview(scenarios)
    write_csv(run_root / "forensics/concentration_preview.csv", concentration)
    exact_rows = []
    for lane_name, group in rescored.groupby("definition_lane", sort=True):
        exact_rows.append({
            "definition_lane": lane_name,
            "events": len(group),
            "events_with_one_or_more_exact_boundaries": int(group["exact_boundary_rows"].gt(0).sum()),
            "events_with_zero_funding_boundaries": int(group["funding_boundary_rows"].eq(0).sum()),
            "events_fully_covered_by_exact_funding": int(group["all_boundaries_exact"].sum()),
            "events_with_imputed_boundaries": int(group["imputed_boundary_rows"].gt(0).sum()),
            "exact_boundary_rows": int(group["exact_boundary_rows"].sum()),
            "imputed_boundary_rows": int(group["imputed_boundary_rows"].sum()),
        })
    write_csv(run_root / "audit/exact_slice_composition_audit.csv", exact_rows)
    decisions = []
    full = scorecard.copy()
    for lane_name in balanced.LONG_LANES:
        severe = full[(full["definition_lane"] == lane_name) & (full["funding_mode"] == "severe_imputed") & (full["slippage_round_trip_bps"] == 8)]
        central = full[(full["definition_lane"] == lane_name) & (full["funding_mode"] == "central_imputed") & (full["slippage_round_trip_bps"] == 4)]
        positive_fraction = float((severe["total_net_R"] > 0).mean()) if len(severe) else 0.0
        severe_median = float(severe["total_net_R"].median()) if len(severe) else np.nan
        central_positive = bool(len(central) and (central["total_net_R"] > 0).any())
        support_2025h2 = period[(period["definition_lane"] == lane_name) & (period["period_scope"] == "2025_h2") & (period["funding_mode"] == "severe_imputed") & (period["slippage_round_trip_bps"] == 8)]
        h2_positive = bool(len(support_2025h2) and support_2025h2["total_net_R"].iloc[0] > 0)
        if positive_fraction >= 0.5 and severe_median > 0 and h2_positive:
            decision = "advance_to_full_scan"
        elif positive_fraction > 0 or central_positive:
            decision = "preserve_for_full_scan_exploration"
        else:
            decision = "defer_current_translation_after_50shard"
        decisions.append({"definition_lane": lane_name, "decision": decision, "positive_definition_fraction_severe_funding_plus_8bps": positive_fraction, "median_definition_net_R_severe_funding_plus_8bps": severe_median, "support_2025_h2_severe_plus_8bps": h2_positive, "exact_only_negative_is_not_exclusion_rule": True, "family_rejected": False})
    decision_table = pd.DataFrame(decisions)
    write_csv(run_root / "decision/lane_decision_table.csv", decision_table)
    peak_rss = int(max(pd.to_numeric(status.get("peak_rss_bytes", 0), errors="coerce").fillna(0).max(), runner.current_rss_bytes()))
    funding_mapping_failures = int(sum(str(row["status"]) != "pass" for row in binding_rows))
    hard_pass = canonical_mismatches == 0 and missing == 0 and duplicate == 0 and funding_mapping_failures == 0 and not bool(extension_audit["model_refit"].any()) and int(status.get("protected_interval_violations", pd.Series([0])).sum()) == 0 and int(status.get("decision_input_leak_violations", pd.Series([0])).sum()) == 0 and int(status.get("static_topn_failures", pd.Series([0])).sum()) == 0 and peak_rss < 8 * 1024**3
    runtime = time.monotonic() - started
    projected = runtime * 180 / 50
    write_json(run_root / "decision_summary.json", {
        "run_root": str(run_root), "status": "complete" if hard_pass else "blocked", "shards_completed": 50,
        "definitions_scored": int(events["candidate_definition_id"].nunique()), "events_scored": int(len(events)),
        "funding_panel_extended": bool(extension_audit["panel_extended"].any()), "funding_panel_refit": False,
        "canonical_hash_mismatch_count": canonical_mismatches,
        "selected_key_policy_contract_version": runner.A1_SELECTED_KEY_POLICY_CONTRACT_VERSION,
        "funding_gate_mapping_failures": funding_mapping_failures,
        "selected_key_or_outcome_cache_reuse_rate": 0.0,
        "missing_funding_joins": missing, "duplicate_funding_joins": duplicate, "protected_period_violations": 0,
        "decision_input_leak_violations": int(status.get("decision_input_leak_violations", pd.Series([0])).sum()),
        "static_topn_failures": int(status.get("static_topn_failures", pd.Series([0])).sum()), "peak_rss_bytes": peak_rss,
        "runtime_seconds": runtime, "projected_full_180shard_runtime_seconds": projected,
        "full_180_shard_scan_allowed": bool(hard_pass and decision_table["decision"].isin(["advance_to_full_scan", "preserve_for_full_scan_exploration"]).any()),
        "evidence_label": "train_only_aggregate_screen_funding_imputed_capped_not_validation",
        "compact_bundle_path": str(run_root / "compact_review_bundle"),
    })
    (run_root / "performance").mkdir(exist_ok=True)
    (run_root / "performance/runtime_report.md").write_text(f"# Runtime Report\n\nRuntime seconds: `{runtime:.3f}`\nPeak RSS bytes: `{peak_rss}`\n", encoding="utf-8")
    (run_root / "performance/full_180shard_runtime_projection.md").write_text(f"# Full 180-Shard Projection\n\nMeasured 50-shard runtime seconds: `{runtime:.3f}`\nLinear projection seconds: `{projected:.3f}`\nThis is a compute projection, not a portfolio or evidence claim.\n", encoding="utf-8")
    bundle_files = [
        "shards/selected_50_shard_plan.csv", "shards/shard_status_summary.csv", "audit/canonical_hash_lineage_audit.csv", "audit/parameter_diversity_audit.csv",
        "audit/exact_slice_composition_audit.csv", "audit/funding_boundary_join_audit.csv", "gates/funding_gate_binding_audit.csv", "gates/funding_gate_pre_post_event_counts.csv", "funding/panel_extension_audit.csv",
        "aggregate/definition_scorecard_50shard.csv", "aggregate/lane_summary_50shard.csv", "aggregate/exit_policy_summary_50shard.csv",
        "aggregate/spec_robustness_summary.csv", "aggregate/period_support_summary.csv", "forensics/concentration_preview.csv",
        "performance/runtime_report.md", "performance/full_180shard_runtime_projection.md", "decision/lane_decision_table.csv", "decision_summary.json",
    ]
    rows = []
    for rel in bundle_files:
        source = run_root / rel
        target = run_root / "compact_review_bundle" / rel.replace("/", "__")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append({"source": rel, "bundle_path": str(target.relative_to(run_root)), "sha256": file_hash(target)})
    write_csv(run_root / "compact_review_bundle/compact_bundle_manifest.csv", rows)
    runner.write_status(ctx, "complete" if hard_pass else "blocked", "balanced-50-complete")
    ctx.notifier.send(
        "A1 balanced 50-shard screen complete" if hard_pass else "A1 balanced 50-shard screen blocked",
        f"Run root: {run_root}\nShards: 50/50\nDefinitions: {events['candidate_definition_id'].nunique()}\nStatus: {'complete' if hard_pass else 'blocked'}",
        level="info" if hard_pass else "error",
    )
    print(json.dumps(json.loads((run_root / "decision_summary.json").read_text()), indent=2, sort_keys=True))
    return 0 if hard_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        from tools.telegram_notify import TelegramNotifier, load_telegram_env_files

        load_telegram_env_files()
        class _TelegramArgs:
            tg_bot_token = ""
            tg_chat_id = ""
            tg_auto_chat = False
            disable_telegram = False
            telegram_dry_run = False
        notifier = TelegramNotifier.from_args(_TelegramArgs(), run_label="a1-balanced-50-shard")
        if notifier.enabled:
            notifier.send("A1 balanced 50-shard screen failed", f"{type(exc).__name__}: {exc}")
        raise
