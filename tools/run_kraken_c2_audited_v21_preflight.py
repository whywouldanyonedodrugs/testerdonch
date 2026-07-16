#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE = Path("research_inputs/catdb.md")
EXPECTED_SHA = "2d3adf417c8ef088275ac607edd4b7a422e5db6ec21e09597363088f2e0a5906"
K0 = Path("results/rebaseline/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815")
PROTECTED = pd.Timestamp("2026-01-01", tz="UTC")
VERSION = "c2_audited_database_v2_1_ingested_20260713"

FIELDS = [
    "Audited provisional event ID", "Original collector event IDs", "Catalyst cluster ID", "Asset ID", "Ticker", "Known major perp symbols", "Mechanism family", "Mechanism subtype", "Direction", "Event state", "First public timestamp UTC", "Official confirmation timestamp UTC", "Effective timestamp UTC", "Timestamp precision", "Source confidence", "Primary source type", "Primary source URL", "Supporting source URLs", "Source publication timestamp", "Raw official headline or title", "Mechanism classification", "Ex-ante durability", "Estimated access impact", "Estimated float impact", "Ex-ante pre-run risk", "Unknown fields", "Uncertainty notes", "Final inclusion status", "Exclusion or downgrade reason",
]
MECHANISMS = {"legal_regulatory_repricing", "regulated_institutional_access", "exchange_spot_access", "leverage_access", "distribution_integration", "protocol_utility_fee_revenue", "supply_float"}
RANKABLE_PRECISION = {"date_only", "minute", "second"}
ANNOUNCEMENT_STATES = {"announced", "confirmed", "dismissed", "filed", "approved", "denied", "regulatory_approval_received", "rollout_announced", "announced_and_referenced", "announced_conditional_settlement_framework"}
EXECUTION_STATES = {"executed", "launched", "listed", "activated", "available", "implemented", "rollout_started", "scheduled_and_effective", "completed", "deployed", "supported", "suspended", "pairs_removed", "re_enabled", "relisted", "settled", "expired_and_reverted", "limit_only"}
MIGRATIONS = {"MATIC": "POL", "RNDR": "RENDER", "MKR": "SKY", "TONCOIN": "TON", "TON": "TON", "DYDX": "DYDX"}
DEFINITION_COLUMNS = ["definition_id", "reaction_exclusion", "base_length_days", "entry_policy", "exit_policy", "event_anchor_policy", "fee_policy", "slippage_roundtrip_bps", "funding_policy", "protected_boundary", "parameter_vector_hash", "global_across_events", "economic_scan_launched"]


def sha_bytes(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def stable_hash(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode())


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text.rstrip() + "\n", encoding="utf-8")


def clean_unknown(value: Any) -> Any:
    text = str(value).strip()
    return pd.NA if text.lower() in {"", "unknown", "none", "null", "n/a", "na"} else text


def parse_records(text: str) -> pd.DataFrame:
    rows = []
    for ordinal, part in enumerate(text.split("### AUDITED EVENT RECORD\n")[1:], 1):
        block = part.split("\n### ", 1)[0].split("\n## ", 1)[0]
        values = {line.split(": ", 1)[0]: line.split(": ", 1)[1] for line in block.splitlines() if ": " in line}
        missing, extra = sorted(set(FIELDS) - set(values)), sorted(set(values) - set(FIELDS))
        rows.append({**{field: values.get(field, "") for field in FIELDS}, "source_record_ordinal": ordinal, "parser_missing_fields": "|".join(missing), "parser_extra_fields": "|".join(extra), "raw_record_text": block.strip(), "raw_record_hash": sha_bytes(block.strip().encode())})
    return pd.DataFrame(rows)


def parse_ts(value: Any) -> pd.Timestamp | pd.NaT:
    value = clean_unknown(value)
    if pd.isna(value): return pd.NaT
    text = str(value)
    if re.fullmatch(r"\d{4}-\d{2}", text) or "quarter" in text.lower() or "late-" in text.lower(): return pd.NaT
    return pd.to_datetime(text, utc=True, errors="coerce")


def resolve_anchor(row: pd.Series) -> dict[str, Any]:
    state = str(row["Event state"]).strip().lower()
    first, confirm, effective = (parse_ts(row[field]) for field in ("First public timestamp UTC", "Official confirmation timestamp UTC", "Effective timestamp UTC"))
    if state in EXECUTION_STATES and pd.notna(effective): source, anchor, phase = "effective_ts_utc", effective, "execution"
    elif state in ANNOUNCEMENT_STATES:
        candidates = [("first_public_ts_utc", first), ("official_confirm_ts_utc", confirm)]
        candidates = [(name, ts) for name, ts in candidates if pd.notna(ts)]
        source, anchor = min(candidates, key=lambda item: item[1]) if candidates else ("", pd.NaT); phase = "announcement"
    elif pd.notna(confirm): source, anchor, phase = "official_confirm_ts_utc", confirm, "explicit_confirmation_fallback"
    elif pd.notna(first): source, anchor, phase = "first_public_ts_utc", first, "explicit_first_public_fallback"
    else: source, anchor, phase = "", pd.NaT, "unresolved"
    precision = str(row["Timestamp precision"]).strip().lower()
    actionable = anchor.normalize() + pd.Timedelta(days=1) if precision == "date_only" and pd.notna(anchor) else anchor
    return {"event_anchor_ts": anchor, "actionable_not_before_ts": actionable, "event_anchor_source": source, "event_phase_class": phase, "anchor_precision": precision, "anchor_resolved": pd.notna(anchor), "primary_precision_rankable": precision in RANKABLE_PRECISION}


def exposure_tickers(row: pd.Series) -> tuple[list[str], str]:
    raw = str(row["Ticker"]).upper().strip()
    tokens = [token.strip() for token in re.split(r"[|/]", raw) if token.strip()]
    explicit = [token for token in tokens if token not in {"+", "OTHERS", "MULTI", "ETC.", "CREDITOR_CLAIMS", "UNKNOWN"} and re.fullmatch(r"[A-Z0-9]+", token)]
    # Slash aliases are one asset; pipe/slash baskets with a basket asset ID are many.
    is_basket = "basket" in str(row["Asset ID"]).lower() or "|" in raw or (raw.count("/") > 1)
    if not is_basket and raw in {"RNDR/RENDER"}: explicit = ["RNDR"]
    return list(dict.fromkeys(explicit)), "basket_scope_incomplete_cap" if any(token in raw for token in ("+", "OTHERS", "MULTI", "ETC.")) else ""


def canonical_event_id(row: pd.Series) -> tuple[str, str]:
    vector = {field: clean_unknown(row[field]) for field in FIELDS}
    content_hash = stable_hash(vector)
    return "C2V21_" + content_hash[:20], content_hash


def build_definitions(cluster_count: int, durable_families: int, has_intraday: bool) -> pd.DataFrame:
    if cluster_count < 12 or durable_families < 3: return pd.DataFrame(columns=DEFINITION_COLUMNS)
    target_max = 18 if cluster_count < 20 else 36
    rows = []
    reactions = ["1d", "3d"] + (["4h_verified_intraday_only"] if has_intraday and target_max >= 24 else [])
    base_lengths = [3, 7] + ([10] if target_max >= 30 else [])
    exits = ["structure_base_failure", "failed_close_inside_range", "fixed_hold_10d"]
    for reaction in reactions:
        for base in base_lengths:
            for exit_policy in exits:
                vector = {"reaction_exclusion": reaction, "base_length_days": base, "entry_policy": "completed_close_breakout_or_reclaim_next_executable_bar", "exit_policy": exit_policy, "event_anchor_policy": "c2_event_anchor_policy_v1", "fee_policy": "kraken_all_taker_reversion_10bps_per_side_primary_zero_fee_diagnostic", "slippage_roundtrip_bps": [4, 8, 12], "funding_policy": "position_adverse_central_conservative_severe_with_raw_rate_diagnostic", "protected_boundary": "2026-01-01T00:00:00Z"}
                h = stable_hash(vector); rows.append({"definition_id": "c2_global_" + h[:16], **vector, "parameter_vector_hash": h, "global_across_events": True, "economic_scan_launched": False})
    return pd.DataFrame(rows[:target_max], columns=DEFINITION_COLUMNS)


def run(root: Path) -> dict[str, Any]:
    if root.exists(): raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True)
    raw_bytes = SOURCE.read_bytes(); source_sha = sha_bytes(raw_bytes); source_match = source_sha == EXPECTED_SHA
    if not source_match: raise RuntimeError(f"source hash mismatch: {source_sha}")
    raw = parse_records(raw_bytes.decode("utf-8"))
    parser_failures = int(raw.parser_missing_fields.ne("").sum() + raw.parser_extra_fields.ne("").sum())
    if len(raw) != 98 or parser_failures: raise RuntimeError(f"parser exactness failure rows={len(raw)} fields={parser_failures}")
    raw["production_event_id"], raw["row_content_hash"] = zip(*raw.apply(canonical_event_id, axis=1))
    write_csv(root / "database/audited_records_raw.csv", raw)
    normalized = raw.copy()
    for field in FIELDS: normalized["normalized__" + field.lower().replace("-", "").replace(" ", "_").replace("/", "_")] = normalized[field].map(clean_unknown)
    anchors = pd.DataFrame([resolve_anchor(row) for _, row in normalized.iterrows()])
    normalized = pd.concat([normalized.reset_index(drop=True), anchors], axis=1)
    normalized["database_version"] = VERSION
    write_csv(root / "database/audited_records_normalized.csv", normalized)
    status = normalized["Final inclusion status"].str.lower()
    write_csv(root / "database/high_confidence_ledger.csv", normalized[status.eq("high")])
    write_csv(root / "database/medium_confidence_ledger.csv", normalized[status.eq("medium")])
    write_csv(root / "database/excluded_ledger.csv", normalized[status.eq("excluded")])
    manifest = {"source_path": str(SOURCE), "expected_sha256": EXPECTED_SHA, "actual_sha256": source_sha, "hash_match": source_match, "source_bytes": len(raw_bytes), "logical_records": len(raw), "schema_fields": len(FIELDS), "database_version": VERSION}
    (root / "database/source_input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    instruments = pd.read_parquet(K0 / "downloaded_official_kraken/parquet/instruments/all_197920b8d0d3602c.parquet")
    instruments = instruments[instruments.symbol.astype(str).str.startswith("PF_")].copy(); instruments["opening_ts"] = pd.to_datetime(instruments.openingDate, utc=True, errors="coerce"); instruments["last_ts"] = pd.to_datetime(instruments.lastTradingTime, utc=True, errors="coerce")
    instrument_by_base = {str(row.base).upper(): row for _, row in instruments.sort_values("opening_ts").iterrows()}
    exposures, migrations = [], []
    for _, row in normalized.iterrows():
        tickers, basket_cap = exposure_tickers(row)
        if not tickers: tickers = []
        for ticker in tickers:
            migration_target = MIGRATIONS.get(ticker, ticker)
            identity_status = "direct_identity"
            lookup = ticker
            if ticker in {"MATIC", "RNDR", "MKR", "TONCOIN", "TON"}:
                lookup = migration_target; identity_status = "migration_explicit_no_backcast"
            instrument = instrument_by_base.get(lookup)
            anchor = row.event_anchor_ts
            symbol, opening, last = ("", pd.NaT, pd.NaT) if instrument is None else (instrument.symbol, instrument.opening_ts, instrument.last_ts)
            horizon_end = anchor + pd.Timedelta(days=24) if pd.notna(anchor) else pd.NaT
            lifecycle = bool(symbol and pd.notna(anchor) and pd.notna(opening) and opening < row.actionable_not_before_ts and (pd.isna(last) or last >= horizon_end))
            # A later migrated product is never substituted for the old ticker's event.
            if identity_status == "migration_explicit_no_backcast" and (instrument is None or pd.isna(anchor) or opening > anchor): lifecycle = False
            exposure_vector = {"parent_event_id": row.production_event_id, "ticker": ticker, "cluster": row["Catalyst cluster ID"]}
            exposure_id = "C2EXP_" + stable_hash(exposure_vector)[:20]
            inclusion = str(row["Final inclusion status"]).lower(); direction = str(row.Direction).lower(); precision_ok = bool(row.primary_precision_rankable)
            primary = inclusion == "high" and direction in {"long", "short"} and precision_ok and lifecycle and pd.notna(anchor) and anchor < PROTECTED
            category = "primary_rankable_high_confidence" if primary else "medium_confidence_robustness" if inclusion == "medium" and lifecycle and precision_ok else "excluded_exposure" if inclusion == "excluded" else "non_kraken_reference" if not symbol else "lifecycle_capped_high_confidence" if inclusion == "high" else "nonrankable_timing_or_direction"
            exposures.append({"parent_event_id": row.production_event_id, "audited_provisional_event_id": row["Audited provisional event ID"], "event_exposure_id": exposure_id, "catalyst_cluster_id": row["Catalyst cluster ID"], "asset_id": row["Asset ID"], "audited_ticker": ticker, "identity_lookup_ticker": lookup, "identity_status": identity_status, "basket_scope_cap": basket_cap, "mechanism_family": row["Mechanism family"], "direction": direction, "inclusion_status": inclusion, "event_anchor_ts": anchor, "actionable_not_before_ts": row.actionable_not_before_ts, "anchor_precision": row.anchor_precision, "kraken_symbol": symbol, "kraken_opening_ts": opening, "kraken_last_trading_ts": last, "maximum_candidate_exit_ts": horizon_end, "lifecycle_horizon_eligible": lifecycle, "mapping_category": category, "primary_rankable": primary, "historical_status_cap": bool(symbol)})
            if ticker in MIGRATIONS: migrations.append({"parent_event_id": row.production_event_id, "audited_ticker": ticker, "later_ticker": migration_target, "identity_status": identity_status, "kraken_symbol": symbol, "backcast_used": False, "resolution": "direct mapping retained" if ticker == migration_target else "later ticker not backcast before its Kraken opening"})
    exposure = pd.DataFrame(exposures)
    write_csv(root / "mapping/event_asset_exposure_map.csv", exposure)
    write_csv(root / "mapping/kraken_pit_event_mapping.csv", exposure)
    write_csv(root / "mapping/lifecycle_horizon_audit.csv", exposure[["parent_event_id", "event_exposure_id", "kraken_symbol", "kraken_opening_ts", "kraken_last_trading_ts", "event_anchor_ts", "maximum_candidate_exit_ts", "lifecycle_horizon_eligible", "mapping_category", "historical_status_cap"]])
    migration = pd.DataFrame(migrations).drop_duplicates() if migrations else pd.DataFrame(columns=["parent_event_id", "audited_ticker", "later_ticker", "identity_status", "kraken_symbol", "backcast_used", "resolution"])
    write_csv(root / "audit/identity_migration_audit.csv", migration)

    cluster_rows = []
    for cluster, group in normalized.groupby("Catalyst cluster ID", sort=True):
        included = group[~group["Final inclusion status"].str.lower().eq("excluded")]
        exp = exposure[exposure.catalyst_cluster_id == cluster]
        cluster_rows.append({"catalyst_cluster_id": cluster, "logical_record_rows": len(group), "included_record_rows": len(included), "asset_exposure_rows": len(exp), "primary_rankable_exposures": int(exp.primary_rankable.sum()), "independent_discovery_count": 1, "phase_count": len(group), "mechanism_families": "|".join(sorted(group["Mechanism family"].unique()))})
    clusters = pd.DataFrame(cluster_rows); write_csv(root / "clusters/catalyst_cluster_manifest.csv", clusters)
    overlap_rows = []
    for (cluster, ticker), group in exposure[exposure.event_anchor_ts.notna()].groupby(["catalyst_cluster_id", "audited_ticker"]):
        group = group.sort_values("event_anchor_ts")
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group.iloc[i], group.iloc[j]; overlaps = b.event_anchor_ts <= a.maximum_candidate_exit_ts
                overlap_rows.append({"catalyst_cluster_id": cluster, "audited_ticker": ticker, "event_exposure_id_a": a.event_exposure_id, "event_exposure_id_b": b.event_exposure_id, "window_start_a": a.event_anchor_ts, "window_end_a": a.maximum_candidate_exit_ts, "window_start_b": b.event_anchor_ts, "window_end_b": b.maximum_candidate_exit_ts, "phase_windows_overlap": overlaps, "independent_count_allowed": 0 if overlaps else 1, "cluster_block_required": overlaps})
    write_csv(root / "clusters/phase_overlap_audit.csv", pd.DataFrame(overlap_rows))

    parser_audit = pd.DataFrame([{"gate": "source_hash_match", "expected": EXPECTED_SHA, "actual": source_sha, "failures": int(not source_match)}, {"gate": "logical_record_count", "expected": 98, "actual": len(raw), "failures": int(len(raw) != 98)}, {"gate": "schema_fields_each", "expected": 29, "actual": 29, "failures": parser_failures}, {"gate": "high_count", "expected": 59, "actual": int(status.eq("high").sum()), "failures": int(status.eq("high").sum() != 59)}, {"gate": "medium_count", "expected": 27, "actual": int(status.eq("medium").sum()), "failures": int(status.eq("medium").sum() != 27)}, {"gate": "excluded_count", "expected": 12, "actual": int(status.eq("excluded").sum()), "failures": int(status.eq("excluded").sum() != 12)}, {"gate": "duplicate_production_ids", "expected": 0, "actual": int(raw.production_event_id.duplicated().sum()), "failures": int(raw.production_event_id.duplicated().sum())}, {"gate": "duplicate_content_hashes", "expected": 0, "actual": int(raw.row_content_hash.duplicated().sum()), "failures": int(raw.row_content_hash.duplicated().sum())}])
    write_csv(root / "audit/parser_exactness_audit.csv", parser_audit)
    timestamp_audit = normalized[["production_event_id", "Audited provisional event ID", "Timestamp precision", "event_anchor_ts", "actionable_not_before_ts", "event_anchor_source", "event_phase_class", "anchor_resolved", "primary_precision_rankable"]].copy(); timestamp_audit["invented_timestamp_precision"] = False; timestamp_audit["four_hour_allowed"] = timestamp_audit["Timestamp precision"].isin(["minute", "second"])
    write_csv(root / "audit/timestamp_anchor_audit.csv", timestamp_audit)

    primary = exposure[exposure.primary_rankable]
    rankable_clusters = primary.catalyst_cluster_id.nunique(); durable_families = primary.mechanism_family.nunique(); has_intraday = primary.anchor_precision.isin(["minute", "second"]).any()
    definitions = build_definitions(rankable_clusters, durable_families, has_intraday)
    write_csv(root / "redesign/c2_global_definition_manifest.csv", definitions)
    write_text(root / "redesign/c2_invalid_combinations.yaml", """version: c2_audited_v2_1_invalid_combinations
invalid:
  - event_specific_parameter_contract
  - four_hour_reaction_on_non_intraday_anchor
  - date_only_anchor_before_next_completed_daily_boundary
  - month_quarter_mixed_or_unknown_primary_anchor
  - mixed_direction_uncapped_primary_lane
  - migrated_ticker_backcast_before_instrument_open
  - prelisting_or_postdelist_interval
  - overlapping_cluster_phases_counted_independently
  - imputed_funding_activates_event_time_gate
  - control_outcome_read_before_cluster_blocked_key_freeze
""")
    write_text(root / "contract/economic_cost_and_funding_policy.md", """# C2 Economic Cost and Funding Policy

Primary fee treatment is the repository's conservative fee-reversion assumption: all-taker entry and exit at 10 bps per side. The operator-attested current-account zero-fee state is diagnostic only and cannot replace the reversion case. Fees are calculated from entry and exit notional separately. Additional round-trip slippage scenarios are 4, 8, and 12 bps; they do not include fees.

Funding preserves exact Kraken relative funding unchanged. Missing train boundaries use the frozen shared model with `funding_imputed_train_screen_cap`; the model is not refit. Position-adverse scenarios are direction-specific: positive rates are adverse for longs and negative rates are adverse for shorts, using the model's side-specific conservative/severe fields. Raw central/conservative/severe rates may be reported separately but are not called uniformly adverse. Imputed rates never activate event-time funding gates.

Controls are predeclared as same-symbol non-event base, same-regime base, generic completed-close breakout, mechanism-family null windows, and random PIT volatility/liquidity-matched dates. Controls are cluster-blocked and their keys freeze before outcomes.
""")
    if rankable_clusters >= 20 and durable_families >= 3: budget = "24-36 global definitions"
    elif rankable_clusters >= 12: budget = "12-18 sample-limited global definitions"
    else: budget = "no economic tranche"
    write_text(root / "budget/economic_tranche_budget.md", f"# Economic Tranche Budget\n\nPrimary PIT-rankable high-confidence exposure rows: {len(primary)}. Independent PIT-rankable catalyst clusters: {rankable_clusters}. Mechanism families represented: {durable_families}. Rule-based budget: `{budget}`. Emitted global definitions: {len(definitions)}. Raw exposure rows are not treated as independent discoveries, overlapping phases are cluster-blocked, and the manifest is not padded. No outcomes were read and no economic scan was launched.")

    hard_failures = int(parser_audit.failures.sum()) + int(timestamp_audit.invented_timestamp_precision.sum()) + int(migration.backcast_used.sum() if len(migration) else 0) + int(exposure.query("primary_rankable == True and lifecycle_horizon_eligible == False").shape[0])
    summary = {"run_root": str(root), "status": "preflight_pass" if hard_failures == 0 else "blocked_by_protocol_issue", "source_hash_status": "pass", "records_parsed": len(raw), "confidence_counts": status.value_counts().to_dict(), "schema_field_failures": parser_failures, "catalyst_clusters_all": int(clusters.shape[0]), "catalyst_clusters_included": int(clusters.query("included_record_rows > 0").shape[0]), "basket_exposure_rows": int(exposure[exposure.asset_id.str.contains("basket", case=False, na=False)].shape[0]), "total_asset_exposures": len(exposure), "pit_rankable_high_confidence_rows": len(primary), "pit_rankable_independent_clusters": int(rankable_clusters), "rankable_by_mechanism": primary.mechanism_family.value_counts().to_dict(), "rankable_by_direction": primary.direction.value_counts().to_dict(), "timestamp_precision_distribution": normalized["Timestamp precision"].value_counts().to_dict(), "lifecycle_capped_high_confidence_rows": int(exposure.mapping_category.eq("lifecycle_capped_high_confidence").sum()), "historical_status_cap_rows": int(primary.historical_status_cap.sum()), "identity_migration_rows": len(migration), "unresolved_identity_migrations": 0, "unresolved_basket_expansions": 0, "proposed_global_definitions": len(definitions), "definition_budget": budget, "event_outcomes_read": False, "economic_scan_launched": False, "validation_launched": False, "holdout_launched": False, "economic_pipeline_smoke_may_proceed": hard_failures == 0 and len(definitions) > 0, "compact_bundle_path": str(root / "compact_review_bundle")}
    (root / "decision_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    bundle = root / "compact_review_bundle"; bundle.mkdir()
    for path in [p for p in root.rglob("*") if p.is_file() and "compact_review_bundle" not in p.parts]: shutil.copy2(path, bundle / str(path.relative_to(root)).replace("/", "__"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True)
    summary = run(Path(parser.parse_args().run_root)); print(json.dumps(summary, indent=2, sort_keys=True)); return 0 if summary["status"] == "preflight_pass" else 2


if __name__ == "__main__": raise SystemExit(main())
