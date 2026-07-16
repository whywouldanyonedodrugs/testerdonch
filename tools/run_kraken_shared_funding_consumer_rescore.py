from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import tools.kraken_funding_imputation as imputation
import tools.kraken_shared_funding_consumer as consumer


DEFAULT_RUN_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_consumer_a1_tsmom_rescore_20260711_v1")
FUNDING_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
A1_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_funding_policy_universe_repair_20260709_v1")
TSMOM_FULL_ROOT = Path("results/rebaseline/phase_kraken_full_tsmom_v6_aggregate_20260707_v1")
TSMOM_OUTCOME_ROOT = Path("results/rebaseline/phase_kraken_tsmom_outcome_grouped_aggregate_20260707_v1")
TSMOM_FORENSIC_ROOT = Path("results/rebaseline/phase_kraken_tsmom_v6_survivor_forensic_decomposition_20260708_v1")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    value.to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_path = FUNDING_ROOT / "funding/shared_funding_panel_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    frames = []
    failures = []
    for row in manifest.itertuples(index=False):
        path = FUNDING_ROOT / row.path
        if not path.exists():
            failures.append({"path": row.path, "reason": "missing"})
            continue
        frame = pd.read_parquet(path)
        if len(frame) != int(row.row_count):
            failures.append({"path": row.path, "reason": "row_count_mismatch"})
        if imputation.canonical_frame_hash(frame) != str(row.content_hash):
            failures.append({"path": row.path, "reason": "content_hash_mismatch"})
        frames.append(frame)
    if failures:
        raise RuntimeError(f"funding panel manifest validation failed: {failures[:5]}")
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    return panel, manifest


def load_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    a1_files = sorted((A1_ROOT / "aggregate_shards").glob("*/outcome_events.parquet"))
    if not a1_files:
        raise RuntimeError("A1 frozen outcome ledgers missing")
    a1 = consumer.normalize_frozen_events(pd.concat([pd.read_parquet(path) for path in a1_files], ignore_index=True), "a1")
    tsmom_path = TSMOM_OUTCOME_ROOT / "cache/tsmom_interval_outcome.parquet"
    if not tsmom_path.exists():
        raise RuntimeError("TSMOM frozen interval outcome cache missing")
    tsmom = consumer.normalize_frozen_events(pd.read_parquet(tsmom_path), "tsmom")
    if tsmom["candidate_definition_id"].nunique() != 128:
        raise RuntimeError("TSMOM frozen outcomes do not cover exactly 128 v6 definitions")
    return a1, tsmom


def integration_audits(events: pd.DataFrame, boundaries: pd.DataFrame, joined: pd.DataFrame, rescored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = int((joined["_merge"] != "both").sum())
    duplicates = int(joined.duplicated(["event_key", "boundary_ts"]).sum())
    exact_mismatch = int((joined["funding_exact"].fillna(False) & ~np.isclose(joined["relativeFundingRate"], joined["funding_rate_central"], rtol=0.0, atol=0.0)).sum())
    legacy_exact = pd.to_numeric(rescored.get("funding_boundary_count_exact", 0), errors="coerce").fillna(0).astype(int)
    panel_exact = rescored["exact_boundary_rows"].astype(int)
    exact_reproduction = np.isclose(rescored["funding_exact_R_panel"], pd.to_numeric(rescored["raw_funding_R"], errors="coerce"), rtol=0.0, atol=1e-12)
    comparable = legacy_exact.gt(0) & legacy_exact.eq(panel_exact)
    exact_event_mismatch = int((comparable & ~exact_reproduction).sum())
    sample_keys = set()
    for _, family_events in events.groupby("source_family", sort=True):
        ranked = family_events[["event_key"]].drop_duplicates().copy()
        ranked["sample_hash"] = ranked["event_key"].map(lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest())
        sample_keys.update(ranked.sort_values("sample_hash", kind="mergesort").head(128)["event_key"])
    sample = joined[joined["event_key"].isin(sample_keys)]
    sample_exact_mismatch = int((sample["funding_exact"].fillna(False) & ~np.isclose(sample["relativeFundingRate"], sample["funding_rate_central"], rtol=0.0, atol=0.0)).sum())
    imputed_cap_mismatch = int((joined["funding_imputed"].fillna(False) & joined["label_cap_reason"].ne(imputation.IMPUTED_CAP)).sum())
    exactness = pd.DataFrame([
        {"audit": "exact_panel_rows_unchanged", "rows_checked": int(joined["funding_exact"].fillna(False).sum()), "mismatch_count": exact_mismatch, "pass": exact_mismatch == 0},
        {"audit": "deterministic_A1_TSMOM_sample_exact_rows", "rows_checked": int(len(sample)), "mismatch_count": sample_exact_mismatch, "pass": sample_exact_mismatch == 0},
        {"audit": "legacy_exact_event_funding_reproduction", "rows_checked": int(comparable.sum()), "mismatch_count": exact_event_mismatch, "pass": exact_event_mismatch == 0, "tolerance": 1e-12},
        {"audit": "imputed_rows_never_gate_eligible", "rows_checked": int(joined["funding_imputed"].fillna(False).sum()), "mismatch_count": int((joined["funding_imputed"].fillna(False) & joined["funding_gate_eligible"].fillna(False)).sum()), "pass": not bool((joined["funding_imputed"].fillna(False) & joined["funding_gate_eligible"].fillna(False)).any())},
        {"audit": "imputed_rows_carry_train_screen_cap", "rows_checked": int(joined["funding_imputed"].fillna(False).sum()), "mismatch_count": imputed_cap_mismatch, "pass": imputed_cap_mismatch == 0},
    ])
    boundary_audit = pd.DataFrame([
        {"audit": "required_boundary_join", "boundary_rows": len(boundaries), "missing_count": missing, "duplicate_count": duplicates, "pass": missing == 0 and duplicates == 0},
        {"audit": "boundary_strictly_after_entry", "boundary_rows": len(boundaries), "violation_count": int((boundaries["boundary_ts"] <= boundaries["entry_ts"]).sum()), "pass": bool((boundaries["boundary_ts"] > boundaries["entry_ts"]).all())},
        {"audit": "boundary_at_or_before_interval_end", "boundary_rows": len(boundaries), "violation_count": int((boundaries["boundary_ts"] > boundaries["exit_interval_end_ts"]).sum()), "pass": bool((boundaries["boundary_ts"] <= boundaries["exit_interval_end_ts"]).all())},
    ])
    monotonic = rescored["funding_central_R"].ge(rescored["funding_conservative_R"] - 1e-12) & rescored["funding_conservative_R"].ge(rescored["funding_severe_R"] - 1e-12)
    scenario = pd.DataFrame([{"audit": "adverse_scenario_monotonicity", "events_checked": len(rescored), "violation_count": int((~monotonic).sum()), "pass": bool(monotonic.all())}])
    sign_rows = []
    for side in ["long", "long_flat", "short", "short_diagnostic"]:
        sign = consumer.funding_side_sign(side)
        positive_rate_r = sign * 0.001 * 100.0
        expected_adverse = positive_rate_r < 0 if side in {"long", "long_flat"} else positive_rate_r > 0
        sign_rows.append({"side": side, "sign": sign, "positive_rate_funding_R_fixture": positive_rate_r, "expected_sign_pass": expected_adverse})
    sign_audit = pd.DataFrame(sign_rows)
    sign_audit["boundary_rule_pass"] = bool(boundary_audit["pass"].all())
    sign_audit["pass"] = sign_audit["expected_sign_pass"] & sign_audit["boundary_rule_pass"]
    return exactness, boundary_audit, scenario, sign_audit


def selected_rows(frame: pd.DataFrame, **filters: Any) -> pd.DataFrame:
    out = frame
    for key, value in filters.items():
        out = out[out[key] == value]
    return out


def build_decisions(a1_rescore: pd.DataFrame, tsmom_rescore: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    a1_full = a1_rescore[(a1_rescore["period_scope"] == "full_train")]
    lane_rows = []
    for lane in sorted(a1_full["definition_lane"].unique()):
        lane_frame = a1_full[a1_full["definition_lane"] == lane]
        def value(mode: str, bps: int = 0) -> float:
            row = lane_frame[(lane_frame["funding_mode"] == mode) & (lane_frame["slippage_round_trip_bps"] == bps)]
            return float(row["scaled_net_R"].iloc[0]) if len(row) else np.nan
        exact_net = value("exact_only_slice")
        central = value("central_imputed")
        conservative = value("conservative_imputed")
        severe = value("severe_imputed")
        severe_8 = value("severe_imputed", 8)
        allowed = all(np.isfinite([exact_net, central, conservative, severe, severe_8])) and min(exact_net, central, conservative, severe, severe_8) > 0
        lane_rows.append({"definition_lane": lane, "exact_only_scaled_net_R": exact_net, "central_scaled_net_R": central, "conservative_scaled_net_R": conservative, "severe_scaled_net_R": severe, "severe_plus8bps_scaled_net_R": severe_8, "allowed_into_50_shard_screen": allowed, "decision": "allow_train_only_50_shard_screen_capped" if allowed else "do_not_expand_from_funding_rescore"})
    lane_decisions = pd.DataFrame(lane_rows)

    previous = pd.read_csv(TSMOM_FULL_ROOT / "aggregate/tsmom_v6_definition_level_aggregate_summary.csv")[["candidate_definition_id", "aggregate_nonfutile"]]
    forensic = pd.read_csv(TSMOM_FORENSIC_ROOT / "decision/forensic_candidate_decision_table.csv")[["candidate_definition_id", "forensic_decision", "label_cap_reason"]]
    full = tsmom_rescore[tsmom_rescore["period_scope"] == "full_train"]
    pivot = full.pivot_table(index="candidate_definition_id", columns=["funding_mode", "slippage_round_trip_bps"], values="scaled_net_R", aggfunc="first")
    reopened_rows = []
    for cid in sorted(previous["candidate_definition_id"]):
        prev = bool(previous.loc[previous["candidate_definition_id"] == cid, "aggregate_nonfutile"].iloc[0])
        def score(mode: str, bps: int = 0) -> float:
            return float(pivot.loc[cid, (mode, bps)]) if (mode, bps) in pivot.columns else np.nan
        central = score("central_imputed")
        conservative = score("conservative_imputed")
        severe = score("severe_imputed")
        severe_8 = score("severe_imputed", 8)
        reopened_central = not prev and central > 0
        all_scenarios = not prev and min(central, conservative, severe) > 0
        stress_survival = all_scenarios and severe_8 > 0
        frow = forensic[forensic["candidate_definition_id"] == cid]
        forensic_decision = str(frow["forensic_decision"].iloc[0]) if len(frow) else "not_run_blocks_advancement"
        concentration_pass = forensic_decision == "advance_to_train_only_stability_review"
        reopened_rows.append({"candidate_definition_id": cid, "previous_aggregate_nonfutile": prev, "central_scaled_net_R": central, "conservative_scaled_net_R": conservative, "severe_scaled_net_R": severe, "severe_plus8bps_scaled_net_R": severe_8, "reopened_under_central_only": reopened_central, "newly_nonfutile_all_funding_and_slippage_gates": stress_survival, "forensic_concentration_status": forensic_decision, "survives_prior_concentration_gate": concentration_pass, "allowed_to_advance": stress_survival and concentration_pass, "decision_cap": "funding_imputed_train_screen_cap;concentration_review_required"})
    reopened = pd.DataFrame(reopened_rows)
    decisions = pd.DataFrame([
        {"family": "a1_compression", "decision": "allow_selected_lanes_into_50_shard_train_screen_capped" if lane_decisions["allowed_into_50_shard_screen"].any() else "do_not_expand_a1_from_current_rescore", "allowed_count": int(lane_decisions["allowed_into_50_shard_screen"].sum()), "evidence_label": "train_only_rescore_diagnostic_capped"},
        {"family": "tsmom_v6", "decision": "preserve_only_candidates_passing_funding_and_prior_concentration" if reopened["allowed_to_advance"].any() else "no_tsmom_candidate_advances_current_rescore", "allowed_count": int(reopened["allowed_to_advance"].sum()), "evidence_label": "train_only_rescore_diagnostic_capped"},
    ])
    counts = {"newly_nonfutile": int(reopened["newly_nonfutile_all_funding_and_slippage_gates"].sum()), "survive_concentration": int(reopened["survives_prior_concentration_gate"].sum()), "allowed": int(reopened["allowed_to_advance"].sum())}
    return lane_decisions, reopened, {"family_decisions": decisions, **counts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    args = parser.parse_args()
    run_root = Path(args.run_root)
    if run_root.exists() and any(run_root.iterdir()):
        raise RuntimeError(f"run root exists and is nonempty: {run_root}")
    started = time.monotonic()
    for rel in ["integration", "a1", "tsmom", "funding", "stress", "decision", "compact_review_bundle"]:
        (run_root / rel).mkdir(parents=True, exist_ok=True)

    panel, panel_manifest = load_panel()
    a1, tsmom = load_events()
    events = pd.concat([a1, tsmom], ignore_index=True, sort=False)
    if (events["exit_interval_end_ts"] >= consumer.PROTECTED_TS).any():
        raise RuntimeError("frozen event interval crosses protected period")
    boundaries = consumer.build_event_boundary_rows(events)
    joined = consumer.join_boundaries_to_panel(boundaries, panel)
    rescored_events = consumer.aggregate_event_funding(events, joined)
    exactness, boundary_audit, scenario_audit, sign_audit = integration_audits(events, boundaries, joined, rescored_events)
    integration_pass = bool(exactness["pass"].all() and boundary_audit["pass"].all() and scenario_audit["pass"].all() and sign_audit["pass"].all())
    write_csv(run_root / "integration/funding_consumer_exactness.csv", exactness)
    write_csv(run_root / "integration/event_boundary_join_audit.csv", boundary_audit)
    write_csv(run_root / "integration/scenario_monotonicity_audit.csv", scenario_audit)
    write_csv(run_root / "integration/funding_sign_and_boundary_audit.csv", sign_audit)
    if not integration_pass:
        summary = {"run_root": str(run_root), "status": "blocked_consumer_integration_mismatch", "signals_regenerated": False, "integration_pass": False, "rescore_run": False}
        write_json(run_root / "decision_summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 2

    a1_events = rescored_events[rescored_events["source_family"] == "a1"].copy()
    tsmom_events = rescored_events[rescored_events["source_family"] == "tsmom"].copy()
    a1_definition = consumer.grouped_rescore(a1_events, ["candidate_definition_id"], family="a1")
    a1_lane = consumer.grouped_rescore(a1_events, ["definition_lane"], family="a1")
    a1_exit = consumer.grouped_rescore(a1_events, ["exit_policy_id"], family="a1")
    tsmom_definition = consumer.grouped_rescore(tsmom_events, ["candidate_definition_id"], family="tsmom")
    if tsmom_definition["candidate_definition_id"].nunique() != 128:
        raise RuntimeError("TSMOM rescore did not retain all 128 definitions")
    write_csv(run_root / "a1/definition_rescore.csv", a1_definition)
    write_csv(run_root / "a1/lane_rescore.csv", a1_lane)
    write_csv(run_root / "a1/exit_rescore.csv", a1_exit)
    write_csv(run_root / "tsmom/definition_rescore.csv", tsmom_definition)

    lane_decisions, reopened, decision_data = build_decisions(a1_lane, tsmom_definition)
    write_csv(run_root / "a1/lane_50_shard_decision.csv", lane_decisions)
    write_csv(run_root / "tsmom/reopened_candidate_audit.csv", reopened)
    write_csv(run_root / "decision/family_decision_table.csv", decision_data["family_decisions"])

    combined_definition = pd.concat([a1_definition.assign(report_group="definition"), tsmom_definition.assign(report_group="definition")], ignore_index=True, sort=False)
    full_train_definition = combined_definition[combined_definition["period_scope"] == "full_train"]
    scenario_attr = full_train_definition.groupby(["source_family", "funding_mode", "slippage_round_trip_bps"], dropna=False, sort=True).agg(events=("events", "sum"), raw_funding_R=("raw_funding_R", "sum"), raw_net_R=("raw_net_R", "sum"), scaled_funding_R=("scaled_funding_R", "sum"), scaled_net_R=("scaled_net_R", "sum"), imputed_boundaries=("imputed_boundary_rows", "sum")).reset_index()
    year_attr = combined_definition[combined_definition["slippage_round_trip_bps"] == 0].groupby(["source_family", "period_scope", "funding_mode"], dropna=False, sort=True).agg(events=("events", "sum"), exact_boundaries=("exact_boundary_rows", "sum"), imputed_boundaries=("imputed_boundary_rows", "sum"), raw_funding_R=("raw_funding_R", "sum"), raw_net_R=("raw_net_R", "sum"), scaled_funding_R=("scaled_funding_R", "sum"), scaled_net_R=("scaled_net_R", "sum")).reset_index()
    slippage_attr = full_train_definition.groupby(["source_family", "funding_mode", "slippage_round_trip_bps"], dropna=False, sort=True).agg(events=("events", "sum"), raw_slippage_R=("raw_slippage_R", "sum"), raw_net_R=("raw_net_R", "sum"), scaled_slippage_R=("scaled_slippage_R", "sum"), scaled_net_R=("scaled_net_R", "sum")).reset_index()
    write_csv(run_root / "funding/scenario_attribution.csv", scenario_attr)
    write_csv(run_root / "funding/year_coverage_attribution.csv", year_attr)
    write_csv(run_root / "stress/slippage_scenario_attribution.csv", slippage_attr)

    exact_mismatch = int(exactness["mismatch_count"].sum())
    missing = int(boundary_audit.loc[boundary_audit["audit"] == "required_boundary_join", "missing_count"].iloc[0])
    duplicates = int(boundary_audit.loc[boundary_audit["audit"] == "required_boundary_join", "duplicate_count"].iloc[0])
    summary = {
        "run_root": str(run_root), "status": "complete", "code_modified": True,
        "signals_regenerated": False, "new_a1_shards_launched": False, "materialization_controls_validation_holdout_launched": False,
        "consumer_integration_pass": integration_pass, "exact_funding_reproduction_mismatch_count": exact_mismatch,
        "missing_boundary_joins": missing, "duplicate_boundary_joins": duplicates,
        "scenario_monotonicity_pass": bool(scenario_audit["pass"].all()), "protected_period_violations": 0,
        "a1_events_rescored": len(a1_events), "a1_definitions_rescored": a1_events["candidate_definition_id"].nunique(),
        "a1_lanes_allowed_into_50_shard_screen": lane_decisions.loc[lane_decisions["allowed_into_50_shard_screen"], "definition_lane"].tolist(),
        "tsmom_events_rescored": len(tsmom_events), "tsmom_definitions_rescored": tsmom_events["candidate_definition_id"].nunique(),
        "newly_nonfutile_tsmom_count": decision_data["newly_nonfutile"],
        "tsmom_candidates_surviving_prior_concentration_gates": decision_data["survive_concentration"],
        "tsmom_candidates_allowed_to_advance": decision_data["allowed"],
        "central_results_cap": "funding_imputed_train_screen_cap", "candidate_advances_solely_on_central_imputation": False,
        "runtime_seconds": time.monotonic() - started,
        "next_recommended_phase": "review_a1_50_shard_eligibility_and_tsmom_rescore_separately_next",
        "compact_bundle_path": str(run_root / "compact_review_bundle"),
    }
    write_json(run_root / "decision_summary.json", summary)
    bundle_rel = [
        "integration/funding_consumer_exactness.csv", "integration/event_boundary_join_audit.csv", "integration/scenario_monotonicity_audit.csv", "integration/funding_sign_and_boundary_audit.csv",
        "a1/definition_rescore.csv", "a1/lane_rescore.csv", "a1/exit_rescore.csv", "a1/lane_50_shard_decision.csv",
        "tsmom/definition_rescore.csv", "tsmom/reopened_candidate_audit.csv", "funding/scenario_attribution.csv", "funding/year_coverage_attribution.csv",
        "stress/slippage_scenario_attribution.csv", "decision/family_decision_table.csv", "decision_summary.json",
    ]
    bundle_rows = []
    for rel in bundle_rel:
        src = run_root / rel
        dst = run_root / "compact_review_bundle" / rel.replace("/", "__")
        shutil.copy2(src, dst)
        bundle_rows.append({"source": rel, "bundle_path": str(dst.relative_to(run_root)), "sha256": file_hash(dst), "size_bytes": dst.stat().st_size})
    write_csv(run_root / "compact_review_bundle/compact_bundle_manifest.csv", bundle_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
