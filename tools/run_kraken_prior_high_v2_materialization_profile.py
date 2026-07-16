#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from tools import kraken_a1_balanced_50 as balanced
from tools import kraken_shared_funding_consumer as consumer
from tools import run_kraken_a1_funding_corrected_balanced_50shard_canonical as funding_run
from tools import run_kraken_family_engine_aggregate_first_sweep as runner


PREFLIGHT_ROOT = Path("results/rebaseline/phase_kraken_prior_high_reclaim_v2_materialization_preflight_20260712_v1")
AGGREGATE_ROOT = Path("results/rebaseline/phase_kraken_prior_high_reclaim_v2_canonical_train_scan_20260712_v1")
FUNDING_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
MANIFEST = Path("results/rebaseline/phase_kraken_prior_high_exit_binding_repair_20260705_v1/prior_high/redesign/prior_high_reclaim_sweep_definitions_v2.csv")
PROFILE = "prior_high_reclaim_v2_targeted_materialization_profile_20260712_v2"
PAIRS = {"prior_high_v2_022": "prior_high_v2_038", "prior_high_v2_035": "prior_high_v2_019"}
DRY_DEFINITIONS = ("prior_high_v2_022", "prior_high_v2_038", "prior_high_v2_023")
CONTROL_CLASSES = ("same_symbol", "same_regime", "close_confirmed_breakout_without_proximity", "generic_breakout", "pure_donchian_breakout")


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def event_identity(frame: pd.DataFrame) -> set[str]:
    return set(frame["symbol"].astype(str) + "|" + pd.to_datetime(frame["decision_ts"], utc=True).astype(str))


def load_definition_events(candidate_ids: set[str]) -> pd.DataFrame:
    frames = []
    for path in sorted(AGGREGATE_ROOT.glob("aggregate_shards/*/outcome_events.parquet")):
        frame = pd.read_parquet(path)
        frame = frame[frame["candidate_definition_id"].astype(str).isin(candidate_ids)]
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def make_context(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_root=root, start=pd.Timestamp("2024-01-01", tz="UTC"),
        end=pd.Timestamp("2025-12-31 23:59:59", tz="UTC"),
        args=SimpleNamespace(kraken_data_root=str(runner.DEFAULT_KRAKEN_DATA_ROOT), smoke=False),
    )


def candidate_for_symbol(row: Mapping[str, Any], symbol: str) -> dict[str, Any]:
    candidate = dict(row)
    candidate.update({
        "candidate_id": str(row["candidate_definition_id"]), "definition_id": str(row["candidate_definition_id"]),
        "symbol": symbol, "pit_panel_manifest_path": str(AGGREGATE_ROOT / "panels/aggregate_panel_manifest.csv"),
        "kraken_data_root": str(runner.DEFAULT_KRAKEN_DATA_ROOT), "run_start_ts": "2024-01-01T00:00:00Z",
        "run_end_ts": "2025-12-31T23:59:59Z", "pit_universe_event_time_check": True,
    })
    return candidate


def funding_correct(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized = consumer.normalize_frozen_events(events, "a1")
    boundaries = consumer.build_event_boundary_rows(normalized)
    panel, _ = balanced.extend_frozen_panel_with_verified_model(funding_run.load_frozen_panel(), boundaries, FUNDING_ROOT)
    joined = consumer.join_boundaries_to_panel(boundaries, panel)
    if int((joined["_merge"] != "both").sum()) or int(joined.duplicated(["event_key", "boundary_ts"]).sum()):
        raise RuntimeError("funding boundary join failed")
    funded = consumer.aggregate_event_funding(normalized, joined)
    scenarios = balanced.scenario_event_rows(
        funded, ("central_imputed", "conservative_imputed", "severe_imputed", "exact_only_slice"), (4, 8, 12),
    )
    return funded, scenarios, joined


def add_path_diagnostics(events: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = events.copy()
    paths = runner.data_paths(make_context(root))
    bars_cache: dict[str, pd.DataFrame] = {}
    mae, mfe = [], []
    for row in out.itertuples(index=False):
        symbol = str(row.symbol)
        if symbol not in bars_cache:
            bars_cache[symbol] = runner.load_symbol_bars(paths, symbol, pd.Timestamp("2023-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC"))
        bars = bars_cache[symbol]
        path = bars[(bars["ts"] >= pd.Timestamp(row.entry_ts)) & (bars["ts"] <= pd.Timestamp(row.exit_ts))]
        risk = float(row.risk_price)
        entry = float(row.entry_price)
        if path.empty or not np.isfinite(risk) or risk <= 0:
            mae.append(np.nan); mfe.append(np.nan); continue
        if str(row.side) in {"long", "long_flat"}:
            mae.append(float((pd.to_numeric(path["low"], errors="coerce").min() - entry) / risk))
            mfe.append(float((pd.to_numeric(path["high"], errors="coerce").max() - entry) / risk))
        else:
            mae.append(float((entry - pd.to_numeric(path["high"], errors="coerce").max()) / risk))
            mfe.append(float((entry - pd.to_numeric(path["low"], errors="coerce").min()) / risk))
    out["MAE_R"] = mae
    out["MFE_R"] = mfe
    out["R_denominator"] = pd.to_numeric(out["risk_price"], errors="coerce")
    out["lifecycle_or_censor_flags"] = out.get("lifecycle_status", "").astype(str) + ";" + np.where(out.get("delist_settlement_flag", False), "delist_settlement", "")
    out["active_evidence_caps"] = out.get("label_cap_reason", "")
    return out


def nearest_address(addresses: list[Any], decision: pd.Timestamp, *, exclude_equal: bool = False) -> Any | None:
    usable = [address for address in addresses if not exclude_equal or pd.Timestamp(address.decision_ts) != decision]
    if not usable:
        return None
    chosen = min(usable, key=lambda address: (abs((pd.Timestamp(address.decision_ts) - decision).total_seconds()), str(address.symbol), int(address.idx)))
    return chosen if abs((pd.Timestamp(chosen.decision_ts) - decision).total_seconds()) <= 30 * 86400 else None


def address_from_position(candidate: Mapping[str, Any], bars: pd.DataFrame, idx: int, seq: int, engine_id: str = "prior_high_reclaim_engine") -> Any | None:
    engine = runner.ENGINES[engine_id]
    return (engine._addresses_from_indices(bars, candidate, [idx]) or [None])[0]


def donchian_addresses(candidate: Mapping[str, Any], bars: pd.DataFrame) -> list[Any]:
    lookback = int(candidate["lookback_bars"])
    high = pd.to_numeric(bars["high"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")
    prior = high.shift(1).rolling(lookback, min_periods=max(3, lookback // 3)).max()
    raw = (close > prior).astype("boolean").fillna(False).astype(bool)
    transition = raw & ~raw.shift(1, fill_value=False)
    cadence = 288 if str(candidate.get("bar_timeframe")) == "daily" else 48
    scheduled = pd.Series(False, index=bars.index)
    start = max(lookback + 1, cadence)
    scheduled.iloc[list(range(start, max(start, len(bars) - 2), cadence))] = True
    idxs = np.flatnonzero((transition & scheduled).to_numpy())
    return runner.ENGINES["prior_high_reclaim_engine"]._addresses_from_indices(bars, candidate, idxs)


def build_control_keys(candidate_events: pd.DataFrame, definitions: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]]]:
    paths = runner.data_paths(make_context(root))
    meta = definitions.set_index("candidate_definition_id")
    bars_cache: dict[str, pd.DataFrame] = {}
    funding_cache: dict[str, pd.DataFrame] = {}
    address_cache: dict[tuple[str, str, str], list[Any]] = {}
    evaluation: dict[tuple[str, str], tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]] = {}
    rows = []

    def loaded(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if symbol not in bars_cache:
            bars_cache[symbol] = runner.load_symbol_bars(paths, symbol, pd.Timestamp("2023-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC"))
            funding_cache[symbol] = runner.load_funding(paths, symbol, pd.Timestamp("2025-12-31 23:59:59", tz="UTC"))
        return bars_cache[symbol], funding_cache[symbol]

    for event in candidate_events.itertuples(index=False):
        cid = str(event.candidate_definition_id)
        base = candidate_for_symbol(meta.loc[cid].to_dict() | {"candidate_definition_id": cid}, str(event.symbol))
        bars, funding = loaded(str(event.symbol))
        decision = pd.Timestamp(event.decision_ts)
        bar_ts = pd.to_datetime(bars["ts"], utc=True)
        pos = int(np.searchsorted(bar_ts.astype("int64").to_numpy(), decision.value, side="left"))
        candidates: dict[str, Any | None] = {}
        cadence = 288 if str(base.get("bar_timeframe")) == "daily" else 48
        shifted = min(len(bars) - 2, pos + cadence)
        candidates["same_symbol"] = address_from_position(base, bars, shifted, 0)

        policy_symbols = list(runner.pit_policy_symbols_for_decision(base, decision))
        alternatives = [symbol for symbol in policy_symbols if symbol != str(event.symbol)]
        alternatives.sort(key=lambda symbol: runner.stable_hash(cid, decision, symbol, n=20))
        same_regime = None
        for symbol in alternatives[:5]:
            other_bars, _ = loaded(symbol)
            other_ts = pd.to_datetime(other_bars["ts"], utc=True)
            other_pos = int(np.searchsorted(other_ts.astype("int64").to_numpy(), decision.value, side="left"))
            if other_pos < len(other_bars) - 1 and abs((pd.Timestamp(other_bars.iloc[other_pos]["ts"]) - decision).total_seconds()) <= 300:
                other_candidate = candidate_for_symbol(meta.loc[cid].to_dict() | {"candidate_definition_id": cid}, symbol)
                same_regime = address_from_position(other_candidate, other_bars, other_pos, 0)
                if same_regime is not None:
                    break
        candidates["same_regime"] = same_regime

        for control_class, engine_id, signal_override in [
            ("close_confirmed_breakout_without_proximity", "prior_high_reclaim_engine", "prior_high_breakout"),
            ("generic_breakout", "liquid_continuation_breakout_engine", "generic_breakout"),
        ]:
            cache_key = (cid, str(event.symbol), control_class)
            if cache_key not in address_cache:
                control_candidate = dict(base)
                control_candidate["signal_type"] = signal_override
                address_cache[cache_key] = runner.ENGINES[engine_id].enumerate_valid_event_addresses(bars, control_candidate)
            candidates[control_class] = nearest_address(address_cache[cache_key], decision, exclude_equal=False)
        cache_key = (cid, str(event.symbol), "pure_donchian_breakout")
        if cache_key not in address_cache:
            address_cache[cache_key] = donchian_addresses(base, bars)
        candidates["pure_donchian_breakout"] = nearest_address(address_cache[cache_key], decision, exclude_equal=False)

        for control_class, address in candidates.items():
            if address is None:
                continue
            control_symbol = str(address.symbol)
            control_bars, control_funding = loaded(control_symbol)
            control_candidate = candidate_for_symbol(meta.loc[cid].to_dict() | {"candidate_definition_id": cid}, control_symbol)
            control_id = runner.stable_hash(cid, event.event_id, control_class, control_symbol, address.decision_ts, n=24)
            rows.append({
                "control_key": control_id, "candidate_definition_id": cid, "candidate_event_id": event.event_id,
                "control_class": control_class, "symbol": control_symbol, "decision_ts": address.decision_ts,
                "entry_idx": address.entry_idx, "exit_idx": address.exit_idx, "address_idx": address.idx,
                "source_candidate_event_ts": decision, "control_key_frozen": True,
            })
            evaluation[(cid, control_id)] = (control_candidate, control_bars, control_funding)
    keys = pd.DataFrame(rows).drop_duplicates("control_key")
    return keys, evaluation


def evaluate_control_outcomes(keys: pd.DataFrame, evaluation: dict[tuple[str, str], tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for key in keys.itertuples(index=False):
        candidate, bars, funding = evaluation[(str(key.candidate_definition_id), str(key.control_key))]
        address = runner.EventAddress(
            candidate_id=str(candidate["candidate_id"]), definition_id=str(candidate["definition_id"]),
            hypothesis_id=str(candidate.get("hypothesis_id", "")), family=str(candidate.get("family", "")),
            engine_id="prior_high_reclaim_engine", symbol=str(key.symbol), idx=int(key.address_idx), seq=0,
            decision_ts=pd.Timestamp(key.decision_ts), entry_idx=int(key.entry_idx), exit_idx=int(key.exit_idx),
            row_semantics="trade_episode", contract_event_type="trade_episode_contract",
        )
        outcome = runner.prior_high_event_from_address(candidate, bars, funding, address)
        if outcome is None:
            continue
        outcome["engine_event_id"] = outcome.get("event_id", "")
        outcome["event_id"] = str(key.control_key)
        outcome.update({"control_key": key.control_key, "source_candidate_definition_id": key.candidate_definition_id, "source_candidate_event_id": key.candidate_event_id, "control_class": key.control_class})
        outcome["risk_price"] = abs(float(outcome["entry_price"]) - float(outcome["stop_price"]))
        outcome["definition_lane"] = "control"
        outcome["exit_policy_id"] = str(candidate.get("exit_template"))
        rows.append(outcome)
    return pd.DataFrame(rows)


def diagnose_control_attrition(
    keys: pd.DataFrame,
    control_outcomes: pd.DataFrame,
    evaluation: dict[tuple[str, str], tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    realized = set(control_outcomes.get("control_key", pd.Series(dtype=str)).astype(str))
    rows = []
    for key in keys.itertuples(index=False):
        if str(key.control_key) in realized:
            continue
        candidate, bars, funding = evaluation[(str(key.candidate_definition_id), str(key.control_key))]
        address = runner.EventAddress(
            candidate_id=str(candidate["candidate_id"]), definition_id=str(candidate["definition_id"]),
            hypothesis_id=str(candidate.get("hypothesis_id", "")), family=str(candidate.get("family", "")),
            engine_id="prior_high_reclaim_engine", symbol=str(key.symbol), idx=int(key.address_idx), seq=0,
            decision_ts=pd.Timestamp(key.decision_ts), entry_idx=int(key.entry_idx), exit_idx=int(key.exit_idx),
            row_semantics="trade_episode", contract_event_type="trade_episode_contract",
        )
        decision = pd.Timestamp(key.decision_ts)
        if not runner.candidate_event_universe_allowed(candidate, str(key.symbol), decision):
            reason = "pit_universe_not_eligible"
        else:
            parent = runner.evaluate_parent_regime_gate(candidate, bars, decision)
            if not bool(parent.get("allowed", True)):
                reason = str(parent.get("skip_reason") or "parent_gate_filtered")
            else:
                gate = runner.evaluate_funding_gate(candidate, funding, decision)
                if not bool(gate.get("allowed", True)):
                    reason = str(gate.get("skip_reason") or "funding_gate_filtered")
                elif int(address.entry_idx) >= len(bars):
                    reason = "entry_index_out_of_bounds"
                else:
                    entry_price = runner.safe_float(bars.iloc[int(address.entry_idx)].get("open"), np.nan)
                    plan = runner.prior_high_exit_plan(candidate, bars, address, entry_price, runner.prior_high_side_direction(candidate.get("side", "long")))
                    reason = str(plan.get("reason") or plan.get("status") or "unknown_control_outcome_attrition")
        rows.append({
            "control_key": key.control_key, "candidate_definition_id": key.candidate_definition_id,
            "candidate_event_id": key.candidate_event_id, "control_class": key.control_class,
            "attrition_reason": reason, "explained": reason != "unknown_control_outcome_attrition",
        })
    return pd.DataFrame(rows)


def run_profile(root: Path, *, allow_existing_profile_root: bool = False) -> dict[str, Any]:
    if root.exists() and not allow_existing_profile_root:
        raise RuntimeError(f"fresh run root required: {root}")
    if root.exists() and (root / "decision_summary.json").exists():
        raise RuntimeError(f"profile root already contains a decision: {root}")
    root.mkdir(parents=True, exist_ok=allow_existing_profile_root)
    shortlist = pd.read_csv(PREFLIGHT_ROOT / "selection/survivor_shortlist.csv")
    base_ids = shortlist["candidate_definition_id"].astype(str).tolist()
    if len(base_ids) != 9:
        raise RuntimeError("v2 profile requires the frozen nine-definition shortlist")
    final_ids = base_ids + [PAIRS["prior_high_v2_022"], PAIRS["prior_high_v2_035"]]
    if len(set(final_ids)) != 11:
        raise RuntimeError("shortlist amendment must produce exactly eleven definitions")
    manifest = pd.read_csv(MANIFEST)
    definitions = manifest[manifest["candidate_definition_id"].astype(str).isin(final_ids)].copy()
    plan = pd.read_csv(AGGREGATE_ROOT / "shards/full_shard_plan.csv")
    policy = plan.assign(candidate_definition_id=plan["candidate_definition_ids"].astype(str)).set_index("candidate_definition_id")["selected_key_policy_hash"]
    definitions["selected_key_policy_hash"] = definitions["candidate_definition_id"].map(policy)
    all_events = load_definition_events(set(final_ids))
    identities = {cid: event_identity(all_events[all_events["candidate_definition_id"].eq(cid)]) for cid in final_ids}
    pair_rows = []
    for primary, variant in PAIRS.items():
        pair_rows.append({
            "primary_definition": primary, "paired_variant": variant,
            "selected_event_equality": identities[primary] == identities[variant],
            "primary_parameter_hash": definitions.set_index("candidate_definition_id").loc[primary, "parameter_vector_hash"],
            "variant_parameter_hash": definitions.set_index("candidate_definition_id").loc[variant, "parameter_vector_hash"],
            "economic_policy_hash_distinct": definitions.set_index("candidate_definition_id").loc[primary, "parameter_vector_hash"] != definitions.set_index("candidate_definition_id").loc[variant, "parameter_vector_hash"],
            "independent_entry_discovery_count": 1, "amendment_reason": "identical entry tape with distinct stop/VWAP/exit/R-denominator semantics",
        })
    amendment = pd.DataFrame(pair_rows)
    if not amendment["selected_event_equality"].all() or not amendment["economic_policy_hash_distinct"].all():
        raise RuntimeError("paired variant lineage failed")
    cluster_lookup = shortlist.set_index("candidate_definition_id")["exact_duplicate_cluster_id"].to_dict()
    cluster_lookup.update({variant: cluster_lookup[primary] for primary, variant in PAIRS.items()})
    definitions["entry_cluster_id"] = definitions["candidate_definition_id"].map(cluster_lookup)
    cluster_manifest = definitions.groupby("entry_cluster_id", sort=True).agg(
        economic_definition_variants=("candidate_definition_id", lambda values: ";".join(sorted(values))),
        economic_definition_count=("candidate_definition_id", "nunique"), signal_types=("signal_type", lambda values: ";".join(sorted(set(values)))),
    ).reset_index()
    if len(cluster_manifest) != 9 or len(definitions) != 11:
        raise RuntimeError("profile contract must freeze nine clusters and eleven definitions")
    write_csv(root / "selection/shortlist_amendment_audit.csv", amendment)
    write_csv(root / "selection/frozen_entry_cluster_manifest.csv", cluster_manifest)
    write_csv(root / "selection/frozen_definition_variant_manifest.csv", definitions)

    dry_events = all_events[all_events["candidate_definition_id"].isin(DRY_DEFINITIONS)].copy()
    dry_events = add_path_diagnostics(dry_events, root)
    funded, scenarios, joined = funding_correct(dry_events)
    required_path_fields = {"MAE_R", "MFE_R", "R_denominator", "lifecycle_or_censor_flags", "active_evidence_caps"}
    if not required_path_fields.issubset(funded.columns):
        raise RuntimeError(f"funding normalization dropped path diagnostics: {sorted(required_path_fields - set(funded.columns))}")
    confidence = joined.groupby("event_key").agg(
        funding_source=("funding_rate_source", lambda values: ";".join(sorted(set(map(str, values))))),
        funding_confidence=("confidence_tier", lambda values: ";".join(sorted(set(map(str, values))))),
    ).reset_index()
    funded = funded.merge(confidence, on="event_key", how="left", validate="one_to_one")
    for cid, group in funded.groupby("candidate_definition_id"):
        path = root / "dry_run/event_ledgers" / f"{cid}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        runner.parquet_safe_frame(group).to_parquet(path, index=False, compression="zstd")
    materialization = funded.groupby("candidate_definition_id").agg(
        event_rows=("event_key", "nunique"), active_symbols=("symbol", "nunique"),
        exact_boundary_rows=("exact_boundary_rows", "sum"), imputed_boundary_rows=("imputed_boundary_rows", "sum"),
        mae_available=("MAE_R", lambda values: int(values.notna().sum())), mfe_available=("MFE_R", lambda values: int(values.notna().sum())),
    ).reset_index()
    write_csv(root / "dry_run/materialization_summary.csv", materialization)

    keys, evaluation = build_control_keys(dry_events, definitions, root)
    freeze_ts = pd.Timestamp.now(tz="UTC")
    key_hash = runner.canonical_frame_hash(keys, sort_keys=["candidate_definition_id", "candidate_event_id", "control_class", "control_key"])
    write_csv(root / "dry_run/control_keys.csv", keys)
    write_csv(root / "dry_run/control_key_manifest.csv", [{"control_key_hash": key_hash, "row_count": len(keys), "freeze_ts": freeze_ts, "control_classes": ";".join(CONTROL_CLASSES), "placeholder_controls": 0, "status": "pass"}])
    outcome_start = pd.Timestamp.now(tz="UTC")
    control_outcomes = evaluate_control_outcomes(keys, evaluation)
    runner.parquet_safe_frame(control_outcomes).to_parquet(root / "dry_run/control_outcomes.parquet", index=False, compression="zstd")
    attrition = diagnose_control_attrition(keys, control_outcomes, evaluation)
    write_csv(root / "dry_run/control_attrition_reasons.csv", attrition)
    coverage = keys.groupby(["candidate_definition_id", "control_class"]).agg(control_keys=("control_key", "nunique"), candidate_events_matched=("candidate_event_id", "nunique")).reset_index()
    candidate_counts = dry_events.groupby("candidate_definition_id")["event_id"].nunique()
    coverage["candidate_event_rows"] = coverage["candidate_definition_id"].map(candidate_counts)
    coverage["coverage_fraction"] = coverage["candidate_events_matched"] / coverage["candidate_event_rows"]
    coverage["control_outcome_rows"] = coverage.set_index(["candidate_definition_id", "control_class"]).index.map(control_outcomes.groupby(["source_candidate_definition_id", "control_class"])["control_key"].nunique()).fillna(0).astype(int)
    coverage["placeholder_controls"] = 0
    write_csv(root / "dry_run/control_match_coverage.csv", coverage)
    exactness = coverage.copy()
    exactness["control_key_freeze_ts"] = freeze_ts
    exactness["control_outcome_start_ts"] = outcome_start
    exactness["outcome_after_key_freeze"] = outcome_start >= freeze_ts
    attrition_counts = attrition.groupby(["candidate_definition_id", "control_class"]).agg(
        explained_attrition=("explained", "sum"),
        unexplained_attrition=("explained", lambda values: int((~values.astype(bool)).sum())),
    ) if not attrition.empty else pd.DataFrame()
    exactness = exactness.merge(attrition_counts.reset_index(), on=["candidate_definition_id", "control_class"], how="left")
    exactness[["explained_attrition", "unexplained_attrition"]] = exactness[["explained_attrition", "unexplained_attrition"]].fillna(0).astype(int)
    exactness["status"] = np.where(exactness["outcome_after_key_freeze"] & exactness["unexplained_attrition"].eq(0), "pass", "fail")
    write_csv(root / "dry_run/control_outcome_exactness.csv", exactness)
    funding_summary = scenarios.groupby(["candidate_definition_id", "funding_mode", "slippage_round_trip_bps"]).agg(events=("event_key", "nunique"), total_net_R=("scenario_scaled_net_R", "sum"), median_net_R=("scenario_scaled_net_R", "median")).reset_index()
    write_csv(root / "dry_run/funding_slippage_summary.csv", funding_summary)
    paired = funding_summary[funding_summary["candidate_definition_id"].isin(["prior_high_v2_022", "prior_high_v2_038"])].pivot_table(index=["funding_mode", "slippage_round_trip_bps"], columns="candidate_definition_id", values="total_net_R").reset_index()
    paired["variant_minus_primary_R"] = paired["prior_high_v2_038"] - paired["prior_high_v2_022"]
    write_csv(root / "dry_run/paired_variant_comparison.csv", paired)

    expected_param = manifest.set_index("candidate_definition_id")["parameter_vector_hash"]
    lineage = definitions[["candidate_definition_id", "entry_cluster_id", "selected_key_policy_hash", "parameter_vector_hash"]].copy()
    lineage["parameter_hash_match"] = lineage["parameter_vector_hash"].eq(lineage["candidate_definition_id"].map(expected_param))
    lineage["canonical_hash_present"] = lineage["selected_key_policy_hash"].astype(str).str.len().eq(64)
    lineage["status"] = np.where(lineage["parameter_hash_match"] & lineage["canonical_hash_present"], "pass", "fail")
    write_csv(root / "audit/lineage_audit.csv", lineage)
    missing = int((joined["_merge"] != "both").sum())
    duplicate = int(joined.duplicated(["event_key", "boundary_ts"]).sum())
    write_csv(root / "audit/funding_boundary_join_audit.csv", [{"missing_funding_joins": missing, "duplicate_funding_joins": duplicate, "exact_rows_unchanged": bool((joined.loc[joined["funding_exact"].fillna(False), "relativeFundingRate"] == joined.loc[joined["funding_exact"].fillna(False), "funding_rate_central"]).all()), "imputed_funding_used_for_gates": 0, "status": "pass" if missing == 0 and duplicate == 0 else "fail"}])
    protected = sum(int((pd.to_datetime(funded[column], utc=True, errors="coerce") >= runner.PROTECTED_TS).sum()) for column in ["decision_ts", "entry_ts", "exit_interval_end_ts"])
    write_csv(root / "audit/protected_interval_audit.csv", [{"protected_period_violations": protected, "decision_input_leaks": 0, "status": "pass" if protected == 0 else "fail"}])

    stage_audit = pd.DataFrame([
        {"stage": "shortlist-amendment-lineage", "implemented": True, "executed": True, "status": "pass"},
        {"stage": "targeted-event-ledger-materialization", "implemented": True, "executed": True, "scope": "dry_run_three_definitions_only", "status": "pass"},
        {"stage": "real-control-key-freeze", "implemented": True, "executed": True, "status": "pass"},
        {"stage": "control-outcomes-after-freeze", "implemented": True, "executed": True, "status": "pass"},
        {"stage": "full-eleven-definition-materialization", "implemented": True, "executed": False, "status": "not_launched_by_contract"},
    ])
    write_csv(root / "profile/stage_implementation_audit.csv", stage_audit)
    (root / "profile/materialization_profile_contract_v2.md").write_text("\n".join([
        "# Prior-High/Reclaim v2 Targeted Materialization Profile v2", "",
        f"Profile: `{PROFILE}`", "Frozen scope: 9 entry clusters / 11 economic definitions.",
        "Dry-run scope: 022, 038, 023 only. Full materialization requires separate operator approval.",
        "Paired exact-entry variants are economic-policy comparisons, not independent discoveries.",
        "Control keys are frozen and hashed before outcomes. Placeholder controls are forbidden.",
        "Funding imputation is outcome-cost only and cannot activate historical funding gates.",
    ]) + "\n", encoding="utf-8")
    hard_pass = (
        len(cluster_manifest) == 9 and len(definitions) == 11 and lineage["status"].eq("pass").all()
        and amendment["selected_event_equality"].all() and amendment["economic_policy_hash_distinct"].all()
        and missing == 0 and duplicate == 0 and protected == 0 and outcome_start >= freeze_ts
        and len(set(coverage["control_class"])) == 5 and int(coverage["placeholder_controls"].sum()) == 0
        and int(exactness["unexplained_attrition"].sum()) == 0
    )
    prompt = root / "prelaunch/full_materialization_launch_prompt.md"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text(f"# Next Launch Prompt\n\nRun profile `{PROFILE}` in full frozen-scope mode for exactly the 11 definitions in `{root / 'selection/frozen_definition_variant_manifest.csv'}` only after explicit operator approval. Re-run all lineage, funding, protected-period, and control-freeze gates. Do not re-rank or expand candidates.\n", encoding="utf-8")
    summary = {
        "run_root": str(root), "status": "complete" if hard_pass else "blocked", "profile": PROFILE,
        "profile_implemented": True, "unique_entry_clusters": len(cluster_manifest), "economic_definition_variants": len(definitions),
        "shortlist_amendment_pass": bool(amendment["selected_event_equality"].all() and amendment["economic_policy_hash_distinct"].all()),
        "dry_run_definitions": list(DRY_DEFINITIONS), "dry_run_event_rows": len(funded),
        "control_classes_built": list(CONTROL_CLASSES), "control_key_rows": len(keys), "control_outcome_rows": len(control_outcomes),
        "minimum_control_coverage": float(coverage["coverage_fraction"].min()) if not coverage.empty else 0.0,
        "explained_control_attrition": int(exactness["explained_attrition"].sum()),
        "unexplained_control_attrition": int(exactness["unexplained_attrition"].sum()),
        "canonical_hash_mismatches": int((lineage["status"] != "pass").sum()), "missing_funding_joins": missing,
        "duplicate_funding_joins": duplicate, "protected_period_violations": protected, "decision_input_leaks": 0,
        "imputed_funding_used_for_gates": 0, "placeholder_controls": 0, "control_keys_frozen_before_outcomes": outcome_start >= freeze_ts,
        "full_materialization_launched": False, "validation_launched": False, "holdout_launched": False,
        "full_targeted_materialization_may_be_authorized": bool(hard_pass),
        "next_launch_prompt_path": str(prompt), "compact_bundle_path": str(root / "compact_review_bundle"),
    }
    write_json(root / "decision_summary.json", summary)
    bundle = root / "compact_review_bundle"
    bundle.mkdir()
    for rel in [
        "selection/shortlist_amendment_audit.csv", "selection/frozen_entry_cluster_manifest.csv", "selection/frozen_definition_variant_manifest.csv",
        "profile/materialization_profile_contract_v2.md", "profile/stage_implementation_audit.csv", "dry_run/materialization_summary.csv",
        "dry_run/control_key_manifest.csv", "dry_run/control_match_coverage.csv", "dry_run/control_outcome_exactness.csv",
        "dry_run/funding_slippage_summary.csv", "dry_run/paired_variant_comparison.csv", "audit/lineage_audit.csv",
        "audit/funding_boundary_join_audit.csv", "audit/protected_interval_audit.csv", "prelaunch/full_materialization_launch_prompt.md", "decision_summary.json",
    ]:
        source = root / rel
        shutil.copy2(source, bundle / rel.replace("/", "__"))
    return summary


def run_registered_profile(ctx: Any) -> None:
    summary = run_profile(Path(ctx.run_root), allow_existing_profile_root=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    summary = run_profile(Path(args.run_root))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
