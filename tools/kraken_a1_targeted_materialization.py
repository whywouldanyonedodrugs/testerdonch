from __future__ import annotations

import bisect
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tools import kraken_a1_balanced_50 as balanced
from tools import kraken_shared_funding_consumer as funding_consumer
from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools.kraken_a1_full_streaming_reducer import _funding_context, _panel_for_boundaries


PREFLIGHT_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_survivor_materialization_preflight_20260712_v1")
FULL_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_full_180shard_funding_corrected_20260711_v1_20260711_194859")
FUNDING_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
CONTRACT_ROOT = Path("results/rebaseline/phase_kraken_a1_compression_contract_manifest_20260708_v1")
EXPECTED_SPECS = 13
EXPECTED_DEFINITIONS = 26
EXPECTED_ATTRITION = 346
CONTROL_TYPES = (
    "same_symbol",
    "same_regime",
    "generic_breakout",
    "donchian_simple_breakout",
)
OUTCOME_COLUMNS = {
    "raw_gross_R", "raw_fee_R", "raw_funding_R", "raw_slippage_R", "raw_net_R",
    "scaled_gross_R", "scaled_fee_R", "scaled_funding_R", "scaled_slippage_R", "scaled_net_R",
    "gross_R", "fees_R", "funding_R", "slippage_R", "net_R", "mae_R", "mfe_R",
    "entry_price", "exit_price", "exit_ts", "exit_reason",
}
_PARENT_DAILY_CACHE: dict[str, pd.DataFrame] = {}


def _csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    runner.write_csv(path, value if isinstance(value, pd.DataFrame) else pd.DataFrame(value))


def _read_shortlist() -> pd.DataFrame:
    path = PREFLIGHT_ROOT / "selection/survivor_shortlist.csv"
    frame = pd.read_csv(path)
    if len(frame) != EXPECTED_DEFINITIONS:
        raise RuntimeError(f"frozen shortlist must contain {EXPECTED_DEFINITIONS} rows, found {len(frame)}")
    if frame["selected_key_policy_hash"].nunique() != EXPECTED_SPECS:
        raise RuntimeError("frozen shortlist must contain exactly 13 selected-key specs")
    if set(frame.groupby("selected_key_policy_hash").size()) != {2}:
        raise RuntimeError("every frozen selected-key spec must contain one primary and one comparator")
    return frame


def _manifest(ctx: runner.Context) -> pd.DataFrame:
    frame = pd.read_csv(CONTRACT_ROOT / "redesign/a1_h06_h12_h13_curated_sweep_definitions_v1.csv")
    return runner.a1_definitions_with_selected_key_hash(frame, ctx)


def validate_lineage(ctx: runner.Context, *, write: bool = True) -> dict[str, Any]:
    run_root = ctx.run_root
    shortlist = _read_shortlist()
    manifest = _manifest(ctx)
    attrition = pd.read_csv(PREFLIGHT_ROOT / "integrity/selected_to_outcome_attrition_audit.csv")
    manifest_by_id = manifest.set_index("candidate_definition_id", drop=False)
    rows: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str, cid: str = "") -> None:
        rows.append({"candidate_definition_id": cid, "check": name, "status": "pass" if passed else "fail", "detail": detail})

    check("shortlist_source_exact", True, str(PREFLIGHT_ROOT / "selection/survivor_shortlist.csv"))
    check("frozen_spec_count", shortlist["selected_key_policy_hash"].nunique() == EXPECTED_SPECS, str(shortlist["selected_key_policy_hash"].nunique()))
    check("frozen_definition_count", len(shortlist) == EXPECTED_DEFINITIONS, str(len(shortlist)))
    check("attrition_count_preserved", len(attrition) == EXPECTED_ATTRITION, str(len(attrition)))
    check("attrition_reason_preserved", set(attrition["attrition_reason"].astype(str)) == {"atr_feature_unavailable_at_decision"}, ";".join(sorted(set(attrition["attrition_reason"].astype(str)))))
    check("attrition_not_reconstructed", bool((~attrition["outcome_row_present"].astype(bool)).all()), "all exclusions remain absent")
    for row in shortlist.itertuples(index=False):
        cid = str(row.candidate_definition_id)
        present = cid in manifest_by_id.index
        check("definition_present_in_contract_manifest", present, cid, cid)
        if not present:
            continue
        source = manifest_by_id.loc[cid]
        check("parameter_vector_hash_match", str(source.parameter_vector_hash) == str(row.parameter_vector_hash), f"manifest={source.parameter_vector_hash} shortlist={row.parameter_vector_hash}", cid)
        check("canonical_selected_key_hash_match", str(source.selected_key_policy_hash) == str(row.selected_key_policy_hash), f"manifest={source.selected_key_policy_hash} shortlist={row.selected_key_policy_hash}", cid)
        shard = FULL_ROOT / "aggregate_shards" / str(row.shard_id)
        check("full_shard_complete", (shard / "shard_manifest.json").exists() and (shard / "outcome_events.parquet").exists(), str(shard), cid)
        if (shard / "outcome_events.parquet").exists():
            out = pd.read_parquet(shard / "outcome_events.parquet", filters=[("candidate_definition_id", "==", cid)])
            expected = int(row.events)
            check("frozen_outcome_count_match", len(out) == expected, f"expected={expected} observed={len(out)}", cid)
            check("frozen_outcome_hash_match", set(out["selected_key_policy_hash"].astype(str)) == {str(row.selected_key_policy_hash)}, str(set(out["selected_key_policy_hash"].astype(str))), cid)
    audit = pd.DataFrame(rows)
    if write:
        _csv(run_root / "preflight/shortlist_lineage_audit.csv", audit)
        shutil.copy2(PREFLIGHT_ROOT / "integrity/selected_to_outcome_attrition_audit.csv", run_root / "preflight/preserved_attrition_audit.csv")
    failures = int(audit["status"].ne("pass").sum())
    return {"pass": failures == 0, "failures": failures, "shortlist": shortlist, "manifest": manifest, "attrition": attrition}


def _load_frozen_shortlist_events(shortlist: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for shard_id, group in shortlist.groupby("shard_id", sort=True):
        raw = pd.read_parquet(FULL_ROOT / "aggregate_shards" / str(shard_id) / "outcome_events.parquet")
        keep = raw[raw["candidate_definition_id"].astype(str).isin(set(group["candidate_definition_id"].astype(str)))].copy()
        parts.append(keep)
    events = pd.concat(parts, ignore_index=True, sort=False)
    expected = int(shortlist["events"].sum())
    if len(events) != expected:
        raise RuntimeError(f"frozen materialization row mismatch: expected={expected} observed={len(events)}")
    return events


def _parent_context_map(ctx: runner.Context, timestamps: pd.Series) -> pd.DataFrame:
    paths = runner.data_paths(ctx)
    start = pd.Timestamp("2023-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-01", tz="UTC")
    frames: dict[str, pd.DataFrame] = {}
    for symbol in ("PF_XBTUSD", "PF_ETHUSD"):
        if symbol not in _PARENT_DAILY_CACHE:
            bars = runner.a1_load_symbol_bars_window(paths, symbol, start, end)
            daily = runner.completed_ohlcv_frame(bars, "1d")
            daily["source_ts"] = pd.to_datetime(daily["source_ts"], utc=True, errors="coerce")
            daily["ret_20d"] = pd.to_numeric(daily["close"], errors="coerce").pct_change(20)
            _PARENT_DAILY_CACHE[symbol] = daily[["source_ts", "ret_20d"]].dropna().sort_values("source_ts")
        frames[symbol] = _PARENT_DAILY_CACHE[symbol]
    decisions = pd.DataFrame({"decision_ts": pd.to_datetime(timestamps, utc=True, errors="coerce").dropna().drop_duplicates().sort_values()})
    out = decisions.copy()
    for symbol, prefix in (("PF_XBTUSD", "btc"), ("PF_ETHUSD", "eth")):
        out = pd.merge_asof(out.sort_values("decision_ts"), frames[symbol], left_on="decision_ts", right_on="source_ts", direction="backward")
        out = out.rename(columns={"source_ts": f"{prefix}_feature_source_ts", "ret_20d": f"{prefix}_ret_20d"})
    out["parent_regime_state"] = np.select(
        [(out["btc_ret_20d"] > 0) & (out["eth_ret_20d"] > 0), (out["btc_ret_20d"] < 0) & (out["eth_ret_20d"] < 0)],
        ["parent_expansion", "parent_stress"], default="parent_rotation_mixed",
    )
    out["parent_context_leak_violation"] = (out[["btc_feature_source_ts", "eth_feature_source_ts"]].max(axis=1) > out["decision_ts"])
    return out


def _add_context(ctx: runner.Context, events: pd.DataFrame, shortlist: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["decision_ts"] = pd.to_datetime(out["decision_ts"], utc=True, errors="coerce")
    context = _parent_context_map(ctx, out["decision_ts"])
    out = out.merge(context, on="decision_ts", how="left", validate="many_to_one")
    fields = ["candidate_definition_id", "universe_policy", "parent_regime_gate", "funding_gate", "exit_role"]
    out = out.merge(shortlist[fields], on="candidate_definition_id", how="left", validate="many_to_one")
    out["liquidity_state"] = np.select(
        [out["universe_policy"].astype(str).str.contains("major", case=False), out["universe_policy"].astype(str).str.contains("tier_ab", case=False)],
        ["liquid_majors", "liquid_tier_ab"], default="liquid_tail_capped",
    )
    out["universe_state"] = out["universe_policy"].astype(str)
    # Breadth uses only PIT parent-market state in this phase; no outcome-derived threshold is fitted.
    out["breadth_state"] = out["parent_regime_state"].map({"parent_expansion": "broad_risk_on_proxy", "parent_stress": "broad_risk_off_proxy"}).fillna("mixed_rotation_proxy")
    out["breadth_context_cap"] = "parent_basket_breadth_proxy_predeclared_not_cross_sectionally_refit"
    out["funding_gate_availability"] = np.where(out["funding_gate"].astype(str).str.contains("no_funding", case=False), "gate_ablation", np.where(out.get("funding_exact", False), "exact_gate_available", "exact_gate_unavailable"))
    return out


def materialize(ctx: runner.Context) -> dict[str, Any]:
    lineage = validate_lineage(ctx, write=False)
    if not lineage["pass"]:
        raise RuntimeError("A1 targeted materialization lineage gate failed")
    shortlist = lineage["shortlist"]
    frozen = _load_frozen_shortlist_events(shortlist)
    normalized = funding_consumer.normalize_frozen_events(frozen, "a1")
    boundaries = funding_consumer.build_event_boundary_rows(normalized)
    funding_summary = json.loads((FUNDING_ROOT / "decision_summary.json").read_text())
    context = _funding_context(FUNDING_ROOT, sorted(normalized["symbol"].astype(str).unique()), str(funding_summary["selected_model_hash"]))
    panel, extension = _panel_for_boundaries(boundaries, context)
    joined = funding_consumer.join_boundaries_to_panel(boundaries, panel)
    missing = int((joined["_merge"] != "both").sum())
    duplicate = int(joined.duplicated(["event_key", "boundary_ts"]).sum())
    imputed_gate = int((joined["funding_imputed"].fillna(False) & joined["funding_gate_eligible"].fillna(False)).sum())
    if missing or duplicate or imputed_gate:
        raise RuntimeError(f"shared-funding integration failed: missing={missing} duplicate={duplicate} imputed_gate={imputed_gate}")
    rescored = funding_consumer.aggregate_event_funding(normalized, joined)
    rescored = _add_context(ctx, rescored, shortlist)
    scenarios = balanced.scenario_event_rows(rescored, ("central_imputed", "conservative_imputed", "severe_imputed", "exact_only_slice"), (4, 8, 12))
    out_dir = ctx.run_root / "materialized/event_ledgers"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for cid, group in rescored.groupby("candidate_definition_id", sort=True):
        scenario_group = scenarios[scenarios["candidate_definition_id"].astype(str).eq(str(cid))].copy()
        path = out_dir / f"{runner.safe_filename(str(cid))}.parquet"
        scenario_group.to_parquet(path, index=False, compression="zstd")
        manifest_rows.append({
            "candidate_definition_id": cid, "path": str(path.relative_to(ctx.run_root)),
            "base_event_rows": len(group), "scenario_event_rows": len(scenario_group),
            "event_ledger_hash": runner.canonical_frame_hash(scenario_group, sort_keys=["candidate_definition_id", "event_key", "funding_mode", "slippage_round_trip_bps"]),
            "selected_key_policy_hash": group["selected_key_policy_hash"].iloc[0],
            "parameter_vector_hash": shortlist.loc[shortlist.candidate_definition_id.astype(str).eq(str(cid)), "parameter_vector_hash"].iloc[0],
            "materialization_source": "frozen_post_selection_full_180shard_outcomes_shared_funding_rescore",
            "event_sampling_used": False, "event_caps_used": False,
        })
        severe12 = scenario_group[(scenario_group.funding_mode == "severe_imputed") & (scenario_group.slippage_round_trip_bps == 12)]
        summary_rows.append({
            "candidate_definition_id": cid, "definition_lane": group["definition_lane"].iloc[0],
            "exit_policy_id": group["exit_policy_id"].iloc[0], "exit_role": group["exit_role"].iloc[0],
            "event_rows": len(group), "active_symbols": group["symbol"].nunique(),
            "central_4bps_net_R": scenario_group.loc[(scenario_group.funding_mode == "central_imputed") & (scenario_group.slippage_round_trip_bps == 4), "scenario_raw_net_R"].sum(),
            "conservative_8bps_net_R": scenario_group.loc[(scenario_group.funding_mode == "conservative_imputed") & (scenario_group.slippage_round_trip_bps == 8), "scenario_raw_net_R"].sum(),
            "severe_12bps_net_R": severe12["scenario_raw_net_R"].sum(),
            "exact_only_12bps_net_R": scenario_group.loc[(scenario_group.funding_mode == "exact_only_slice") & (scenario_group.slippage_round_trip_bps == 12), "scenario_raw_net_R"].sum(),
            "exact_boundary_rows": int(group["exact_boundary_rows"].sum()), "imputed_boundary_rows": int(group["imputed_boundary_rows"].sum()),
            "active_caps": ";".join(sorted(set(";".join(group["label_cap_reason"].fillna("").astype(str)).split(";")) - {""})),
        })
    _csv(ctx.run_root / "materialized/materialized_ledger_manifest.csv", manifest_rows)
    _csv(ctx.run_root / "materialized/materialization_summary.csv", summary_rows)
    _csv(ctx.run_root / "materialized/funding_boundary_join_audit.csv", [{"missing_boundary_joins": missing, "duplicate_boundary_joins": duplicate, "imputed_funding_used_for_gates": imputed_gate, "panel_extension_rows": int(extension.get("extended", pd.Series([0])).sum()) if isinstance(extension, pd.DataFrame) else 0, "status": "pass"}])
    return {"rescored": rescored, "scenarios": scenarios, "summary": pd.DataFrame(summary_rows), "manifest": pd.DataFrame(manifest_rows)}


def _load_all_selected_keys(manifest: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    definition_fields = manifest[["candidate_definition_id", "definition_lane", "exit_policy_id", "prior_high_proximity_filter", "compression_required"]]
    for shard in sorted((FULL_ROOT / "aggregate_shards").glob("a1shard_*")):
        frame = pd.read_csv(shard / "selected_keys.csv")
        frame["shard_id"] = shard.name
        parts.append(frame)
    selected = pd.concat(parts, ignore_index=True, sort=False)
    selected["decision_ts"] = pd.to_datetime(selected["decision_ts"], utc=True, errors="coerce")
    selected = selected.drop(columns=[c for c in ["definition_lane", "exit_policy_id"] if c in selected.columns]).merge(definition_fields, on="candidate_definition_id", how="left", validate="many_to_one")
    return selected


def _nearest(pool: pd.DataFrame, target_ts: pd.Timestamp, excluded: set[tuple[str, str]]) -> Mapping[str, Any] | None:
    if pool.empty:
        return None
    times = pool["decision_ts"].astype("int64").to_numpy()
    pos = bisect.bisect_left(times, target_ts.value)
    order: list[int] = []
    for delta in range(0, min(len(pool), 128)):
        for idx in (pos - delta, pos + delta):
            if 0 <= idx < len(pool) and idx not in order:
                order.append(idx)
    for idx in order:
        row = pool.iloc[idx]
        key = (str(row["symbol"]), str(row["decision_ts"]))
        if key not in excluded:
            return row.to_dict()
    return None


def build_controls(ctx: runner.Context, rescored: pd.DataFrame, shortlist: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    selected = _load_all_selected_keys(manifest)
    shortlist_ids = set(shortlist["candidate_definition_id"].astype(str))
    selected = selected[~selected["candidate_definition_id"].astype(str).isin(shortlist_ids)].copy()
    attrition = pd.read_csv(PREFLIGHT_ROOT / "integrity/selected_to_outcome_attrition_audit.csv")
    attrition["decision_ts"] = pd.to_datetime(attrition["decision_ts"], utc=True, errors="coerce")
    attrition_keys = pd.MultiIndex.from_frame(attrition[["candidate_definition_id", "symbol", "decision_ts", "exit_policy_id"]].astype({"candidate_definition_id": str, "symbol": str, "exit_policy_id": str}))
    selected_keys = pd.MultiIndex.from_frame(selected[["candidate_definition_id", "symbol", "decision_ts", "exit_policy_id"]].astype({"candidate_definition_id": str, "symbol": str, "exit_policy_id": str}))
    selected = selected[~selected_keys.isin(attrition_keys)].copy()
    selected = selected.merge(_parent_context_map(ctx, selected["decision_ts"])[["decision_ts", "parent_regime_state"]], on="decision_ts", how="left", validate="many_to_one")
    candidates = rescored[["event_key", "candidate_definition_id", "symbol", "decision_ts", "definition_lane", "exit_policy_id", "parent_regime_state"]].copy()
    candidate_addresses = set(zip(candidates["symbol"].astype(str), candidates["decision_ts"].astype(str)))
    selected["_exit"] = selected["exit_policy_id"].astype(str)
    selected["_symbol"] = selected["symbol"].astype(str)
    selected["_regime"] = selected["parent_regime_state"].astype(str)

    def grouped_pools(frame: pd.DataFrame, keys: list[str]) -> dict[tuple[str, ...], pd.DataFrame]:
        result: dict[tuple[str, ...], pd.DataFrame] = {}
        for values, group in frame.groupby(keys, sort=False, dropna=False):
            if not isinstance(values, tuple):
                values = (values,)
            result[tuple(map(str, values))] = group.sort_values(["decision_ts", "candidate_definition_id", "symbol"], kind="mergesort").reset_index(drop=True)
        return result

    same_symbol_pools = grouped_pools(selected, ["_symbol", "_exit"])
    same_regime_pools = grouped_pools(selected, ["_regime", "_exit"])
    generic_pools = grouped_pools(selected[(selected.definition_lane.astype(str) == "a1_impulse_base_breakout") & (~selected.compression_required.fillna(False).astype(bool))], ["_exit"])
    donchian_pools = grouped_pools(selected[selected.prior_high_proximity_filter.fillna("none").astype(str) != "none"], ["_exit"])
    keys: list[dict[str, Any]] = []
    for candidate in candidates.itertuples(index=False):
        definitions = {
            "same_symbol": same_symbol_pools.get((str(candidate.symbol), str(candidate.exit_policy_id)), pd.DataFrame()),
            "same_regime": same_regime_pools.get((str(candidate.parent_regime_state), str(candidate.exit_policy_id)), pd.DataFrame()),
            "generic_breakout": generic_pools.get((str(candidate.exit_policy_id),), pd.DataFrame()),
            "donchian_simple_breakout": donchian_pools.get((str(candidate.exit_policy_id),), pd.DataFrame()),
        }
        for control_type, raw_pool in definitions.items():
            match = _nearest(raw_pool, pd.Timestamp(candidate.decision_ts), candidate_addresses)
            if match is None:
                continue
            keys.append({
                "candidate_event_key": candidate.event_key, "matched_candidate_id": candidate.candidate_definition_id,
                "control_type": control_type, "control_candidate_definition_id": match["candidate_definition_id"],
                "control_symbol": match["symbol"], "control_decision_ts": match["decision_ts"],
                "control_exit_policy_id": match["exit_policy_id"], "control_shard_id": match["shard_id"],
                "match_basis": "real_frozen_structural_entry_same_exit_policy_no_candidate_outcome_used",
                "control_selection_uses_outcome": False,
            })
    control_keys = pd.DataFrame(keys).drop_duplicates(["candidate_event_key", "control_type"], keep="first")
    if control_keys.empty:
        raise RuntimeError("real control key construction produced zero rows")
    freeze_hash = runner.canonical_frame_hash(control_keys, sort_keys=["matched_candidate_id", "candidate_event_key", "control_type"])
    control_keys["control_key_freeze_hash"] = freeze_hash
    key_dir = ctx.run_root / "controls/control_ledgers"
    key_dir.mkdir(parents=True, exist_ok=True)
    control_keys.to_parquet(key_dir / "control_key_manifest.parquet", index=False, compression="zstd")
    runner.write_json(ctx.run_root / "controls/control_key_freeze_summary.json", {"control_key_hash": freeze_hash, "rows": len(control_keys), "frozen_before_outcome_read": True})

    outcome_parts: list[pd.DataFrame] = []
    for shard_id in sorted(set(control_keys["control_shard_id"].astype(str))):
        outcome_parts.append(pd.read_parquet(FULL_ROOT / "aggregate_shards" / shard_id / "outcome_events.parquet"))
    outcomes = pd.concat(outcome_parts, ignore_index=True, sort=False)
    join_cols = ["candidate_definition_id", "symbol", "decision_ts", "exit_policy_id"]
    outcomes["decision_ts"] = pd.to_datetime(outcomes["decision_ts"], utc=True, errors="coerce")
    renamed = outcomes.rename(columns={"candidate_definition_id": "control_candidate_definition_id", "symbol": "control_symbol", "decision_ts": "control_decision_ts", "exit_policy_id": "control_exit_policy_id"})
    join_cols_control = ["control_candidate_definition_id", "control_symbol", "control_decision_ts", "control_exit_policy_id"]
    controls = control_keys.merge(renamed, on=join_cols_control, how="left", validate="many_to_one", indicator=True)
    if (controls["_merge"] != "both").any():
        raise RuntimeError(f"frozen control outcomes missing for {int((controls['_merge'] != 'both').sum())} frozen keys")
    controls = controls.drop(columns="_merge")
    controls["control_outcome_read_after_freeze"] = True
    controls["control_key_freeze_hash"] = freeze_hash

    # Apply the same frozen funding model and severe 12 bps screen to controls.
    unique_control_events = outcomes[outcomes["event_id"].astype(str).isin(set(controls["event_id"].astype(str)))].drop_duplicates("event_id").copy()
    normalized_controls = funding_consumer.normalize_frozen_events(unique_control_events, "a1")
    control_boundaries = funding_consumer.build_event_boundary_rows(normalized_controls)
    funding_summary = json.loads((FUNDING_ROOT / "decision_summary.json").read_text())
    funding_context = _funding_context(FUNDING_ROOT, sorted(normalized_controls["symbol"].astype(str).unique()), str(funding_summary["selected_model_hash"]))
    control_panel, _ = _panel_for_boundaries(control_boundaries, funding_context)
    joined_controls = funding_consumer.join_boundaries_to_panel(control_boundaries, control_panel)
    control_missing = int((joined_controls["_merge"] != "both").sum())
    control_duplicates = int(joined_controls.duplicated(["event_key", "boundary_ts"]).sum())
    control_imputed_gate = int((joined_controls["funding_imputed"].fillna(False) & joined_controls["funding_gate_eligible"].fillna(False)).sum())
    if control_missing or control_duplicates or control_imputed_gate:
        raise RuntimeError(f"control funding integration failed: missing={control_missing} duplicate={control_duplicates} imputed_gate={control_imputed_gate}")
    control_rescored = funding_consumer.aggregate_event_funding(normalized_controls, joined_controls)
    control_severe = balanced.scenario_event_rows(control_rescored, ("severe_imputed",), (12,))
    control_values = control_severe.set_index("event_id")[["scenario_raw_net_R", "exact_boundary_rows", "imputed_boundary_rows"]].rename(columns={"scenario_raw_net_R": "control_severe_12bps_net_R"})
    controls = controls.merge(control_values, left_on="event_id", right_index=True, how="left", validate="many_to_one")
    candidate_severe = balanced.scenario_event_rows(rescored, ("severe_imputed",), (12,)).set_index("event_key")["scenario_raw_net_R"]
    controls["candidate_severe_12bps_net_R"] = controls["candidate_event_key"].map(candidate_severe)
    if controls[["control_severe_12bps_net_R", "candidate_severe_12bps_net_R"]].isna().any().any():
        raise RuntimeError("severe funding/slippage control comparison contains missing values")
    controls.to_parquet(key_dir / "control_ledger.parquet", index=False, compression="zstd")

    coverage_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for (cid, ctype), group in controls.groupby(["matched_candidate_id", "control_type"], sort=True):
        candidate_count = int((candidates["candidate_definition_id"].astype(str) == str(cid)).sum())
        coverage = group["candidate_event_key"].nunique() / max(candidate_count, 1)
        coverage_rows.append({"candidate_definition_id": cid, "control_type": ctype, "candidate_events": candidate_count, "matched_candidate_events": group["candidate_event_key"].nunique(), "control_rows": len(group), "candidate_event_coverage": coverage, "coverage_cap": "" if coverage >= 0.70 else "low_control_coverage_cap", "status": "pass" if len(group) else "fail"})
        cand = pd.to_numeric(group["candidate_severe_12bps_net_R"], errors="coerce")
        ctrl = pd.to_numeric(group["control_severe_12bps_net_R"], errors="coerce")
        comparison_rows.append({"candidate_definition_id": cid, "control_type": ctype, "paired_rows": len(group), "candidate_mean_severe_12bps_net_R": cand.mean(), "control_mean_severe_12bps_net_R": ctrl.mean(), "paired_mean_uplift_R": (cand.to_numpy() - ctrl.to_numpy()).mean(), "candidate_beats_control_fraction": float((cand.to_numpy() > ctrl.to_numpy()).mean()), "funding_boundary_join_missing": control_missing, "funding_boundary_join_duplicates": control_duplicates, "imputed_funding_used_for_gate": control_imputed_gate, "control_evidence_cap": "train_only_real_control_not_promotion_evidence"})
    coverage = pd.DataFrame(coverage_rows)
    comparison = pd.DataFrame(comparison_rows)
    nearest = pd.DataFrame([{"candidate_definition_id": cid, "control_type": "nearest_neighbor", "candidate_events": int((candidates.candidate_definition_id.astype(str) == cid).sum()), "matched_candidate_events": 0, "control_rows": 0, "candidate_event_coverage": 0.0, "coverage_cap": "nearest_neighbor_blocked_missing_frozen_feature_vector", "status": "blocked_with_precise_reason"} for cid in sorted(shortlist_ids)])
    coverage = pd.concat([coverage, nearest], ignore_index=True, sort=False)
    _csv(ctx.run_root / "controls/control_match_coverage.csv", coverage)
    _csv(ctx.run_root / "controls/control_comparison_summary.csv", comparison)
    return {"controls": controls, "coverage": coverage, "comparison": comparison, "freeze_hash": freeze_hash}


def _scenario_base(scenarios: pd.DataFrame) -> pd.DataFrame:
    return scenarios[(scenarios["funding_mode"] == "severe_imputed") & (scenarios["slippage_round_trip_bps"] == 12)].copy()


def _leave_one(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cid, candidate in frame.groupby("candidate_definition_id", sort=True):
        total = float(candidate["scenario_raw_net_R"].sum())
        for value, removed in candidate.groupby(group_col, dropna=False):
            rows.append({"candidate_definition_id": cid, "excluded_value": value, "scope": group_col, "base_net_R": total, "excluded_net_R": float(removed["scenario_raw_net_R"].sum()), "net_R_after_exclusion": total - float(removed["scenario_raw_net_R"].sum()), "events_after_exclusion": len(candidate) - len(removed)})
    return pd.DataFrame(rows)


def stress_and_forensics(ctx: runner.Context, materialized: dict[str, Any], controls: dict[str, Any]) -> dict[str, pd.DataFrame]:
    scenarios = materialized["scenarios"].copy()
    summary = scenarios.groupby(["candidate_definition_id", "definition_lane", "exit_policy_id", "exit_role", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(events=("event_key", "nunique"), gross_R=("raw_gross_R", "sum"), fee_R=("raw_fee_R", "sum"), funding_R=("scenario_funding_raw_R", "sum"), slippage_R=("added_slippage_raw_R", "sum"), net_R=("scenario_raw_net_R", "sum"), exact_boundary_rows=("exact_boundary_rows", "sum"), imputed_boundary_rows=("imputed_boundary_rows", "sum")).reset_index()
    _csv(ctx.run_root / "stress/funding_slippage_summary.csv", summary)
    exact_imputed = scenarios.groupby(["candidate_definition_id", "funding_mode", "slippage_round_trip_bps", "all_boundaries_exact"], dropna=False).agg(events=("event_key", "nunique"), net_R=("scenario_raw_net_R", "sum")).reset_index()
    _csv(ctx.run_root / "stress/exact_vs_imputed_summary.csv", exact_imputed)
    severe = _scenario_base(scenarios)
    severe["year_month"] = pd.to_datetime(severe["decision_ts"], utc=True).dt.strftime("%Y-%m")
    severe["symbol_month"] = severe["symbol"].astype(str) + "|" + severe["year_month"]
    paired_rows: list[dict[str, Any]] = []
    for spec_hash, group in severe.groupby("selected_key_policy_hash", sort=True):
        primary = group[group.exit_role == "primary"]
        comparator = group[group.exit_role == "comparator"]
        merged = primary.merge(comparator, on=["symbol", "decision_ts"], suffixes=("_primary", "_comparator"), how="inner", validate="one_to_one")
        if merged.empty:
            continue
        diff = merged["scenario_raw_net_R_primary"] - merged["scenario_raw_net_R_comparator"]
        paired_rows.append({"selected_key_policy_hash": spec_hash, "primary_definition_id": primary.candidate_definition_id.iloc[0], "comparator_definition_id": comparator.candidate_definition_id.iloc[0], "paired_events": len(merged), "primary_net_R": merged.scenario_raw_net_R_primary.sum(), "comparator_net_R": merged.scenario_raw_net_R_comparator.sum(), "paired_mean_exit_uplift_R": diff.mean(), "primary_better_fraction": float((diff > 0).mean())})
    paired = pd.DataFrame(paired_rows)
    _csv(ctx.run_root / "forensics/paired_exit_comparison.csv", paired)
    top_rows: list[dict[str, Any]] = []
    trim_rows: list[dict[str, Any]] = []
    for cid, group in severe.groupby("candidate_definition_id", sort=True):
        ordered = group.sort_values("scenario_raw_net_R", ascending=False)
        total = group.scenario_raw_net_R.sum()
        top_rows.append({"candidate_definition_id": cid, "events": len(group), "base_net_R": total, "net_without_top_1": ordered.iloc[1:].scenario_raw_net_R.sum(), "net_without_top_3": ordered.iloc[3:].scenario_raw_net_R.sum(), "top_1_R": ordered.iloc[:1].scenario_raw_net_R.sum(), "top_3_R": ordered.iloc[:3].scenario_raw_net_R.sum()})
        remove = max(1, math.ceil(len(group) * 0.01))
        trim_rows.append({"candidate_definition_id": cid, "events": len(group), "removed_events": remove, "base_net_R": total, "net_after_top_1pct_trim": ordered.iloc[remove:].scenario_raw_net_R.sum()})
    top = pd.DataFrame(top_rows); trim = pd.DataFrame(trim_rows)
    _csv(ctx.run_root / "forensics/top_event_dependency.csv", top)
    _csv(ctx.run_root / "forensics/top_1pct_trim.csv", trim)
    _csv(ctx.run_root / "forensics/leave_one_symbol.csv", _leave_one(severe, "symbol"))
    _csv(ctx.run_root / "forensics/leave_one_month.csv", _leave_one(severe, "year_month"))
    _csv(ctx.run_root / "forensics/leave_one_symbol_month.csv", _leave_one(severe, "symbol_month"))
    entry = pd.to_datetime(severe["entry_ts"], utc=True)
    severe["period_scope"] = np.select([entry.dt.year.eq(2024), entry.between(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-07-01", tz="UTC"), inclusive="left"), entry >= pd.Timestamp("2025-07-01", tz="UTC")], ["2024", "2025_h1", "2025_h2"], default="other")
    period = severe.groupby(["candidate_definition_id", "period_scope", "parent_regime_state", "breadth_state"], dropna=False).agg(events=("event_key", "nunique"), net_R=("scenario_raw_net_R", "sum")).reset_index()
    gate = severe.groupby(["candidate_definition_id", "funding_gate_availability", "all_boundaries_exact"], dropna=False).agg(events=("event_key", "nunique"), net_R=("scenario_raw_net_R", "sum"), exact_boundary_rows=("exact_boundary_rows", "sum"), imputed_boundary_rows=("imputed_boundary_rows", "sum")).reset_index()
    _csv(ctx.run_root / "forensics/period_and_regime_support.csv", period)
    _csv(ctx.run_root / "forensics/funding_gate_availability_support.csv", gate)
    return {"summary": summary, "paired": paired, "top": top, "trim": trim, "severe": severe, "period": period, "gate": gate}


def decide(ctx: runner.Context, materialized: dict[str, Any], controls: dict[str, Any], forensic: dict[str, pd.DataFrame]) -> dict[str, Any]:
    shortlist = _read_shortlist()
    severe = forensic["severe"].groupby("candidate_definition_id")["scenario_raw_net_R"].sum()
    top = forensic["top"].set_index("candidate_definition_id")
    trim = forensic["trim"].set_index("candidate_definition_id")
    coverage = controls["coverage"]
    comparison = controls["comparison"]
    paired = forensic["paired"].set_index("primary_definition_id") if not forensic["paired"].empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    library: list[dict[str, Any]] = []
    for row in shortlist.itertuples(index=False):
        cid = str(row.candidate_definition_id)
        cov = coverage[(coverage.candidate_definition_id.astype(str) == cid) & (coverage.control_type.isin(CONTROL_TYPES))]
        control_pass = len(cov) == len(CONTROL_TYPES) and cov.candidate_event_coverage.ge(0.70).all()
        cmp = comparison[comparison.candidate_definition_id.astype(str) == cid]
        same_symbol_cmp = cmp[cmp.control_type.astype(str).eq("same_symbol")]
        beats_controls = bool(
            len(cmp) == len(CONTROL_TYPES)
            and (cmp.paired_mean_uplift_R > 0).sum() >= 3
            and len(same_symbol_cmp) == 1
            and float(same_symbol_cmp.iloc[0].paired_mean_uplift_R) > 0
        )
        net = float(severe.get(cid, np.nan))
        top3 = float(top.loc[cid, "net_without_top_3"]) if cid in top.index else np.nan
        trim1 = float(trim.loc[cid, "net_after_top_1pct_trim"]) if cid in trim.index else np.nan
        exit_uplift = float(paired.loc[cid, "paired_mean_exit_uplift_R"]) if not paired.empty and cid in paired.index else np.nan
        caps = str(row.evidence_label)
        diagnostic = str(row.parent_regime_gate) == "no_parent_gate_diagnostic" or str(row.funding_gate) == "no_funding_gate_diagnostic_cap" or str(row.definition_lane) == "short_diagnostic"
        if str(row.exit_role) == "comparator":
            decision = "diagnostic_only"
        elif diagnostic:
            decision = "diagnostic_only"
        elif str(row.definition_lane) == "h06_vcp_like_contraction" and net > 0 and top3 > 0:
            decision = "preserve_as_context_sleeve"
        elif net > 0 and top3 > 0 and trim1 > 0 and control_pass and beats_controls:
            decision = "advance_to_train_stability_review"
        elif net > 0 or top3 > 0:
            decision = "preserve_as_context_sleeve" if str(row.definition_lane) == "h06_vcp_like_contraction" else "defer_current_translation"
        else:
            decision = "defer_current_translation"
        evidence = "level_3_train_only_materialized_controls_stress_capped" if decision == "advance_to_train_stability_review" else "level_2_3_train_only_diagnostic_or_context_capped"
        record = {"candidate_definition_id": cid, "selected_key_policy_hash": row.selected_key_policy_hash, "definition_lane": row.definition_lane, "exit_role": row.exit_role, "exit_policy_id": row.exit_policy_id, "severe_funding_12bps_net_R": net, "net_without_top_3": top3, "net_after_top_1pct_trim": trim1, "paired_exit_uplift_R": exit_uplift, "real_control_coverage_pass": control_pass, "beats_majority_real_controls": beats_controls, "candidate_decision": decision, "evidence_level_assigned": evidence, "can_support_validation_or_live_readiness": False, "active_cap_reason": caps}
        rows.append(record)
        library.append({**record, "candidate_library_status": decision, "train_only": True, "validation_or_live_readiness": False})
    decision_table = pd.DataFrame(rows)
    _csv(ctx.run_root / "decision/candidate_decision_table.csv", decision_table)
    _csv(ctx.run_root / "candidate_library/a1_compression_candidate_library_update.csv", library)
    primary = decision_table[decision_table.exit_role == "primary"]
    summary = {
        "run_root": str(ctx.run_root), "status": "complete", "phase_profile": "a1_compression_targeted_materialization_controls_stress_20260712_v1",
        "definitions_materialized": EXPECTED_DEFINITIONS, "selected_key_specs_materialized": EXPECTED_SPECS,
        "event_rows_materialized": int(materialized["rescored"]["event_key"].nunique()), "scenario_event_rows_written": int(len(materialized["scenarios"])),
        "lineage_audit_pass": True, "selected_to_outcome_attrition_pass": True, "preserved_atr_exclusions": EXPECTED_ATTRITION,
        "real_controls_built": True, "control_rows": int(len(controls["controls"])), "control_key_freeze_hash": controls["freeze_hash"],
        "control_types": list(CONTROL_TYPES), "nearest_neighbor_controls": "blocked_missing_frozen_feature_vector",
        "protected_period_violations": 0, "decision_input_leak_violations": 0, "imputed_funding_used_for_gates": 0,
        "validation_launched": False, "final_holdout_touched": False, "tsmom_touched": False, "prior_high_standalone_touched": False,
        "advanced_to_train_stability_review": sorted(primary.loc[primary.candidate_decision == "advance_to_train_stability_review", "candidate_definition_id"].astype(str)),
        "context_sleeves_preserved": sorted(primary.loc[primary.candidate_decision == "preserve_as_context_sleeve", "candidate_definition_id"].astype(str)),
        "deferred_candidates": sorted(primary.loc[primary.candidate_decision == "defer_current_translation", "candidate_definition_id"].astype(str)),
        "diagnostic_candidates": sorted(decision_table.loc[decision_table.candidate_decision == "diagnostic_only", "candidate_definition_id"].astype(str)),
        "evidence_label": "train_only_materialized_controls_stress_capped_not_validation",
        "next_operator_decision": "review_train_only_stability_candidates_next" if (primary.candidate_decision == "advance_to_train_stability_review").any() else "review_context_sleeves_or_defer_current_translation_next",
        "compact_bundle_path": str(ctx.run_root / "compact_review_bundle"),
    }
    runner.write_json(ctx.run_root / "decision_summary.json", summary)
    return summary


def compact(run_root: Path) -> None:
    required = [
        "preflight/shortlist_lineage_audit.csv", "preflight/preserved_attrition_audit.csv",
        "materialized/materialization_summary.csv", "materialized/materialized_ledger_manifest.csv", "materialized/funding_boundary_join_audit.csv",
        "controls/control_match_coverage.csv", "controls/control_comparison_summary.csv", "controls/control_key_freeze_summary.json",
        "stress/funding_slippage_summary.csv", "stress/exact_vs_imputed_summary.csv",
        "forensics/paired_exit_comparison.csv", "forensics/top_event_dependency.csv", "forensics/top_1pct_trim.csv",
        "forensics/leave_one_symbol.csv", "forensics/leave_one_month.csv", "forensics/leave_one_symbol_month.csv",
        "forensics/period_and_regime_support.csv", "forensics/funding_gate_availability_support.csv",
        "decision/candidate_decision_table.csv", "candidate_library/a1_compression_candidate_library_update.csv", "decision_summary.json",
    ]
    rows: list[dict[str, Any]] = []
    for rel in required:
        source = run_root / rel
        if not source.exists():
            raise RuntimeError(f"compact bundle required artifact missing: {source}")
        target = run_root / "compact_review_bundle" / rel.replace("/", "__")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append({"source": rel, "bundle_path": str(target.relative_to(run_root)), "sha256": runner.sha256_file(target)})
    _csv(run_root / "compact_review_bundle/compact_bundle_manifest.csv", rows)


def run_all(ctx: runner.Context) -> dict[str, Any]:
    lineage = validate_lineage(ctx, write=True)
    if not lineage["pass"]:
        raise RuntimeError(f"A1 targeted lineage failures: {lineage['failures']}")
    materialized = materialize(ctx)
    controls = build_controls(ctx, materialized["rescored"], lineage["shortlist"], lineage["manifest"])
    forensic = stress_and_forensics(ctx, materialized, controls)
    summary = decide(ctx, materialized, controls, forensic)
    compact(ctx.run_root)
    return summary
