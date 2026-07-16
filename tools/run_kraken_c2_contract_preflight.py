#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from tools import run_kraken_readiness_repair as readiness


RESEARCH = Path("research_inputs")
MAIN_NAME = "post_catalyst_c2_catalyst_db_2020_2025_main.csv"
EXCLUDED_NAME = "post_catalyst_c2_catalyst_db_2020_2025_excluded.csv"
EXTRACT_ROOT = Path("results/rebaseline/phase_qlmg_integrated_abcx_development_20260628_v2_20260628_134140")
PROTECTED = pd.Timestamp("2026-01-01", tz="UTC")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def precision(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "unknown", "nan", "none", "null"}: return "unknown"
    if "T" in text and (text.endswith("Z") or "+" in text): return "intraday_explicit_utc"
    if len(text) == 10 and text[4] == "-" and text[7] == "-": return "date_only"
    return "coarse_or_nonstandard"


def family(value: str) -> str:
    mapping = {
        "legal_regulatory_repricing": "legal_regulatory_repricing",
        "etf_institutional_access": "etf_institutional_access",
        "supply_shock": "supply_float_changes",
        "unlock_vesting_change": "supply_float_changes",
        "protocol_utility_fee_revenue_change": "protocol_utility_fee_revenue_changes",
        "exchange_access_expansion": "exchange_access_expansion",
        "leverage_access_expansion": "leverage_access_expansion",
        "major_integration_distribution_access": "protocol_utility_fee_revenue_changes",
    }
    return mapping.get(str(value), "attention_only_events")


def event_date(row: pd.Series) -> tuple[str, str, pd.Timestamp | pd.NaT]:
    for column in ("first_public_ts_utc", "official_confirm_ts_utc", "effective_ts_utc"):
        value = str(row.get(column, "")).strip()
        if value.lower() not in {"", "unknown", "nan", "none", "null"}:
            return column, value, pd.to_datetime(value, utc=True, errors="coerce")
    return "unknown", "unknown", pd.NaT


def run(root: Path) -> dict[str, Any]:
    if root.exists(): raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True)
    pdf_db = RESEARCH / "Post-Catalyst Continuation Base Catalyst Database-1.pdf"
    pdf_res = RESEARCH / "RES Research Base for Liquid-Sector Episodic Pivots and Post-Catalyst Continuation-1.pdf"
    main_csv, excluded_csv = RESEARCH / MAIN_NAME, RESEARCH / EXCLUDED_NAME
    raw_main = EXTRACT_ROOT / "md_extract/catalyst_main_raw.csv"
    raw_excluded = EXTRACT_ROOT / "md_extract/catalyst_excluded_raw.csv"
    audit = pd.DataFrame([
        {"source_role": "authoritative_main_companion", "path": str(main_csv), "exists": main_csv.exists(), "rows": len(pd.read_csv(main_csv)) if main_csv.exists() else 0, "sha256": sha(main_csv), "authority_status": "missing_required_companion" if not main_csv.exists() else "present"},
        {"source_role": "authoritative_excluded_companion", "path": str(excluded_csv), "exists": excluded_csv.exists(), "rows": len(pd.read_csv(excluded_csv)) if excluded_csv.exists() else 0, "sha256": sha(excluded_csv), "authority_status": "missing_required_companion" if not excluded_csv.exists() else "present"},
        {"source_role": "database_report", "path": str(pdf_db), "exists": pdf_db.exists(), "rows": 0, "sha256": sha(pdf_db), "authority_status": "report_claims_90_main_10_excluded_but_has_no_embedded_files"},
        {"source_role": "sector_catalyst_research", "path": str(pdf_res), "exists": pdf_res.exists(), "rows": 0, "sha256": sha(pdf_res), "authority_status": "design_reference_not_event_database"},
        {"source_role": "markdown_excerpt_extract", "path": str(raw_main), "exists": raw_main.exists(), "rows": len(pd.read_csv(raw_main)) if raw_main.exists() else 0, "sha256": sha(raw_main), "authority_status": "non_authoritative_36_row_excerpt"},
        {"source_role": "markdown_excluded_extract", "path": str(raw_excluded), "exists": raw_excluded.exists(), "rows": len(pd.read_csv(raw_excluded)) if raw_excluded.exists() else 0, "sha256": sha(raw_excluded), "authority_status": "non_authoritative_report_table_extract"},
    ])
    write_csv(root / "events/authoritative_database_audit.csv", audit)
    source_resolved = main_csv.exists() and excluded_csv.exists()
    main = pd.read_csv(main_csv) if source_resolved else pd.read_csv(raw_main)
    excluded = pd.read_csv(excluded_csv) if source_resolved else pd.read_csv(raw_excluded)
    if "official_confirm_ts_utc" not in main: main["official_confirm_ts_utc"] = "unknown"
    main["mechanism_family_c2"] = main["mechanism_family"].map(family)
    main["first_public_precision"] = main["first_public_ts_utc"].map(precision)
    main["official_confirm_precision"] = main["official_confirm_ts_utc"].map(precision)
    main["effective_precision"] = main["effective_ts_utc"].map(precision)
    main["invented_timestamp_precision"] = False
    main["source_quality"] = "unverifiable_report_citation_token_only" if not source_resolved else main.get("primary_source_type", "authoritative_companion")
    main["authoritative_source_set"] = source_resolved
    main["rankable_event"] = False if not source_resolved else True
    main["rankability_reason"] = "non_rankable_incomplete_source_set" if not source_resolved else "pending_kraken_lifecycle_mapping"
    main["current_only_or_future_aware_label"] = False
    main["event_anchor_policy"] = main.apply(lambda row: f"{event_date(row)[0]}:{precision(event_date(row)[1])};date_only_executes_no_earlier_than_next_completed_daily_boundary", axis=1)

    instrument_raw = readiness.load_instruments(readiness.DEFAULT_KRAKEN_DATA_ROOT, readiness.DEFAULT_K0_ROOT)
    if not instrument_raw.empty:
        instrument_raw["kraken_symbol"] = instrument_raw["venue_symbol"].where(
            instrument_raw["venue_symbol"].astype(str).str.len().gt(0), instrument_raw["symbol"]
        )
        instrument_raw["ticker"] = instrument_raw["base"].astype(str).str.upper()
        instrument_raw["kraken_first_tradable_ts"] = pd.to_datetime(instrument_raw["openingDate"], utc=True, errors="coerce")
        instruments = instrument_raw[instrument_raw["kraken_symbol"].astype(str).str.startswith("PF_")][["ticker", "kraken_symbol", "kraken_first_tradable_ts"]].copy()
    else:
        instruments = pd.DataFrame(columns=["ticker", "kraken_symbol", "kraken_first_tradable_ts"])
    mappings = []
    for _, row in main.iterrows():
        ticker = str(row.get("ticker", "")).upper(); lookup = "XBT" if ticker == "BTC" else ticker
        matches = instruments[instruments["ticker"].astype(str).str.upper().eq(lookup)]
        symbol = str(matches.iloc[0]["kraken_symbol"]) if len(matches) else ""
        first_live = matches.iloc[0]["kraken_first_tradable_ts"] if len(matches) else pd.NaT
        anchor_source, anchor_raw, anchor = event_date(row)
        precision_value = precision(anchor_raw)
        lifecycle = bool(symbol and pd.notna(first_live) and pd.notna(anchor) and pd.Timestamp(first_live) < anchor.normalize()) if precision_value == "date_only" else bool(symbol and pd.notna(first_live) and pd.notna(anchor) and pd.Timestamp(first_live) <= anchor)
        if not source_resolved: status, reason = "non_rankable_reference", "authoritative_companion_missing"
        elif pd.isna(anchor): status, reason = "non_rankable", "unknown_event_anchor"
        elif not symbol: status, reason = "non_rankable", "no_kraken_perpetual_mapping"
        elif not lifecycle: status, reason = "non_rankable", "pre_listing_or_same_day_lifecycle_ambiguous"
        else: status, reason = "rankable", "kraken_contract_live_before_observable_event_anchor"
        mappings.append({"event_id": row.event_id, "ticker": ticker, "kraken_symbol": symbol, "kraken_first_tradable_ts": first_live, "event_anchor_source": anchor_source, "event_anchor_raw": anchor_raw, "event_anchor_precision": precision_value, "kraken_lifecycle_eligible": lifecycle, "mapping_status": status, "mapping_reason": reason})
    mapping = pd.DataFrame(mappings)
    write_csv(root / "events/event_instrument_mapping.csv", mapping)
    ledger = main.merge(mapping, on=["event_id", "ticker"], how="left")
    ledger["rankable_event"] = ledger["mapping_status"].eq("rankable") & source_resolved
    ledger["rankability_reason"] = ledger["mapping_reason"]
    write_csv(root / "events/pit_event_ledger.csv", ledger)
    excluded["rankable_event"] = False; excluded["rankability_reason"] = "explicit_database_exclusion"
    write_csv(root / "events/excluded_event_ledger.csv", excluded)
    confidence = ledger[["event_id", "ticker", "first_public_ts_utc", "first_public_precision", "official_confirm_ts_utc", "official_confirm_precision", "effective_ts_utc", "effective_precision", "invented_timestamp_precision", "source_quality", "rankable_event"]]
    write_csv(root / "events/timestamp_confidence_audit.csv", confidence)

    write(root / "contract/c2_strategy_contract.md", """# C2 Post-Catalyst Continuation Base Contract

Hypothesis: after a publicly observable, ex-ante durable catalyst reprices access, legal status, float, or token utility, a close-confirmed breakout from a post-reaction base may continue. This phase creates no economic evidence.

Universe: Kraken linear perpetuals that were officially tradable before the observable event anchor and PIT-liquid at decision time. Pre-listing and same-day lifecycle-ambiguous mappings are non-rankable. Primary long lanes are legal/regulatory relief, ETF/institutional access, favorable supply/float changes, and direct protocol utility/fee/revenue changes. Exchange-access, leverage-access, and attention-only events are diagnostic/base-only.

Temporal contract: preserve first-public, official-confirmation, and effective timestamps separately. Unknown remains unknown. Date-only events are never assigned intraday times and cannot be acted upon before the next completed daily boundary. Decision inputs require source_close_ts <= decision_ts. Entry is the next executable bar after a completed-close signal. No event-day chase and no touch fills.

Costs: eventual outcomes must separately model fees, spread/slippage, and shared exact/imputed funding. Imputed funding cannot activate historical funding gates. Contract type, multiplier, margin/PnL currency, and Kraken symbol metadata must be frozen in the execution manifest.

Research protocol: train-only mechanical and aggregate stages precede any materialization. The protected cutoff is 2026-01-01. No outcome-derived event durability, pre-run-risk, context threshold, or catalyst classification is permitted.""")
    write(root / "contract/reaction_window_policy.md", "# Reaction Window Policy\n\nPredeclared alternatives are 4h, 1d, and 3d after the observable event anchor. Date-only anchors begin no earlier than the next completed daily boundary. Reaction windows are mutually exclusive definition parameters and are not selected using returns. No entry may occur inside the selected reaction exclusion.")
    write(root / "contract/base_formation_policy.md", "# Base Formation Policy\n\nBase lengths: 2, 3, 5, 7, 10, or 15 completed days. Compute event impulse from bars available after the anchor and reaction exclusion. Require a bounded retracement from impulse, declining range/ATR or explicit flat-range contraction, and a completed close above the frozen base high or reclaimed structure. The breakout decision bar is excluded from base-high construction. Entry is next executable bar open. No event-day chase, touch fill, passive-fill assumption, or current-bar leakage.")
    write(root / "contract/exit_stop_policy.md", "# Exit and Stop Policy\n\nExecutable modules: structure/base failure on completed close; failed close back inside range; signal-timeframe ATR initial/trailing stop with explicit timeframe and completed-source timestamp; EMA10/EMA20/SMA10/SMA20 close-confirmed trail; fixed-hold comparator. Earliest executable exit wins; same execution-bar ambiguity uses the most adverse valid fill. Five-minute OHLCV may detect stops but cannot define daily/4h ATR. No metadata-only exits or optimistic target-first fills.")
    write(root / "contract/control_policy.md", "# Control Policy\n\nFreeze control keys before outcomes. Required real classes: same-symbol non-event base; same-regime base breakout; generic close-confirmed breakout; mechanism-family matched null windows; random dates matched using PIT volatility and liquidity. Matching uses only decision-time features and excludes catalyst blackout windows. Candidate and control execution, exits, fees, slippage, and funding are identical. Zero unexplained controls fail closed; placeholders are forbidden.")

    parameter_yaml = """version: c2_parameter_space_v1
protected_cutoff: '2026-01-01T00:00:00Z'
reaction_exclusion: [4h, 1d, 3d]
base_length_days: [2, 3, 5, 7, 10, 15]
decision_timeframe: [4h, 1d]
entry: close_confirmed_next_executable_bar
retracement_max_fraction: [0.25, 0.40, 0.55]
compression: [range_atr_contraction, realized_vol_contraction, flat_range]
exit_policy: [structure_failure, failed_close_inside_range, atr_trail, ema10_trail, ema20_trail, sma10_trail, sma20_trail, fixed_hold]
parent_gate: [btc_eth_up, btc_eth_up_or_neutral]
funding_gate: [exact_only_when_available, report_only_when_unavailable]
primary_mechanisms: [legal_regulatory_repricing, etf_institutional_access, supply_float_changes, protocol_utility_fee_revenue_changes]
diagnostic_mechanisms: [exchange_access_expansion, leverage_access_expansion, attention_only_events]
"""
    write(root / "redesign/c2_parameter_space_v1.yaml", parameter_yaml)
    write(root / "redesign/c2_invalid_combinations_v1.yaml", """version: c2_invalid_combinations_v1
invalid:
  - unknown_event_anchor_for_rankable_definition
  - event_before_or_same_day_as_ambiguous_kraken_contract_open
  - date_only_anchor_with_invented_intraday_timestamp
  - event_day_entry
  - touch_or_same_bar_signal_fill
  - current_decision_bar_in_base_high
  - metadata_only_exit_or_stop
  - imputed_funding_activates_gate
  - attention_only_uncapped_primary_lane
  - exit_horizon_crosses_protected_cutoff_without_declared_censor
  - outcome_informed_control_matching
""")
    definition_columns = ["candidate_definition_id", "mechanism_lane", "reaction_exclusion", "base_length_days", "decision_timeframe", "retracement_max_fraction", "compression_policy", "entry_policy", "exit_policy", "universe_policy", "parent_gate", "funding_gate", "rankable", "parameter_vector_hash", "blocked_reason"]
    definitions = pd.DataFrame(columns=definition_columns)
    write_csv(root / "redesign/c2_curated_definitions_v1.csv", definitions)
    write(root / "budget/first_tranche_budget.md", "# First Tranche Budget\n\nTarget after authoritative-source repair: 60-120 valid definitions, allocated across durable primary mechanisms first and diagnostic listing/attention lanes second. Current emitted definitions: 0. The manifest is intentionally not padded because the authoritative 90-row main companion is missing. Runtime must be projected from PIT-valid event count and unique semantic cache specs before execution.")
    write(root / "preflight/mechanical_implementation_plan.md", "# Mechanical Implementation Plan\n\n1. Supply and hash the named 90-row main and 10-row excluded CSV companions.\n2. Validate schema, uniqueness, source URLs, ex-ante labels, and timestamp precision.\n3. Rebuild official Kraken instrument lifecycle mapping and PIT-liquid membership.\n4. Freeze event ledger and curated 60-120-definition manifest without outcome inspection.\n5. Implement event/reaction/base feature caches with source timestamps.\n6. Run deterministic fixtures for date-only handling, pre-listing exclusion, base-current-bar exclusion, next-bar entry, exits, and control-key freeze.\n7. Run a small real-data mechanical canary spanning exact and imputed funding periods.\n8. Only after zero leakage/lineage/execution mismatches, request a train-only aggregate screen.")
    family_counts = ledger.groupby("mechanism_family_c2").size().to_dict()
    precision_counts = ledger["first_public_precision"].value_counts(dropna=False).to_dict()
    summary = {"run_root": str(root), "status": "blocked_by_protocol_issue", "authoritative_source_set_resolved": source_resolved, "authoritative_events_found": 0 if not source_resolved else len(main), "reference_excerpt_events": len(main), "explicit_excluded_events": len(excluded), "pit_valid_kraken_events": int(ledger["rankable_event"].sum()), "rankable_events": int(ledger["rankable_event"].sum()), "diagnostic_reference_events": len(ledger), "events_by_mechanism_family": family_counts, "timestamp_confidence_distribution": precision_counts, "current_or_future_aware_event_labels": int(ledger["current_only_or_future_aware_label"].sum()), "invented_timestamp_precision": int(ledger["invented_timestamp_precision"].sum()), "prelisting_events_used_rankably": 0, "proposed_definition_target_after_repair": "60-120", "definitions_emitted": 0, "economic_scan_launched": False, "validation_launched": False, "holdout_launched": False, "mechanical_canary_may_proceed": False, "blockers": [f"missing {MAIN_NAME}", f"missing {EXCLUDED_NAME}", "36-row report excerpt cannot substitute for claimed 90-row main database"], "compact_bundle_path": str(root / "compact_review_bundle")}
    (root / "decision_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    bundle = root / "compact_review_bundle"; bundle.mkdir()
    for path in [p for p in root.rglob("*") if p.is_file() and "compact_review_bundle" not in p.parts]:
        shutil.copy2(path, bundle / str(path.relative_to(root)).replace("/", "__"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True); args = parser.parse_args()
    result = run(Path(args.run_root)); print(json.dumps(result, indent=2, sort_keys=True)); return 2 if result["status"].startswith("blocked") else 0


if __name__ == "__main__": raise SystemExit(main())
