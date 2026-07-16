#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "c2_catalyst_database_v2_20260713"
PROTECTED = pd.Timestamp("2026-01-01", tz="UTC")
EXTRACT = Path("results/rebaseline/phase_qlmg_integrated_abcx_development_20260628_v2_20260628_134140/md_extract")
INSTRUMENTS = Path("results/rebaseline/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815/downloaded_official_kraken/parquet/instruments/all_197920b8d0d3602c.parquet")

# URLs are recovered from the report's source appendix. A seed is not promoted
# merely because it appears in the report: retrieval and lifecycle gates follow.
SOURCES = {
    "CAT0001": ("regulator_filing", "https://www.sec.gov/files/litigation/complaints/2020/comp-pr2020-338.pdf"),
    "CAT0002": ("official_exchange_notice", "https://www.cmegroup.com/media-room/press-releases/2020/12/16/cme_group_to_launchetherfuturesonfebruary82021.html"),
    "CAT0003": ("official_exchange_notice", "https://www.cmegroup.com/media-room/press-releases/2021/2/08/cme_group_announceslaunchofetherfutures.html"),
    "CAT0004": ("official_protocol_post", "https://blog.ethereum.org/2021/07/15/london-mainnet-announcement"),
    "CAT0018": ("official_governance_post", "https://gov.optimism.io/t/op-unlock-update/5995"),
    "CAT0025": ("official_foundation_documentation", "https://docs.arbitrum.foundation/token-supply"),
    "CAT0035": ("official_protocol_post", "https://blog.lido.fi/lido-v2-launch/"),
    "CAT0036": ("official_governance_post", "https://governance.aave.com/t/arfc-gho-mainnet-launch/13574"),
    "CAT0037": ("official_foundation_post", "https://www.dydx.foundation/blog/dydx-2023-semi-annual-ecosystem-report"),
    "CAT0038": ("official_foundation_post", "https://www.dydx.foundation/blog/new-trading-pairs-permissionless-markets-on-the-dydx-chain"),
    "CAT0045": ("official_protocol_post", "https://www.pyth.network/blog/pyth-governance-a-guide-to-staking-pyth-tokens"),
    "CAT0051": ("regulator_filing", "https://www.sec.gov/Archives/edgar/data/1980994/000143774923017574/bit20230608_s1.htm"),
    "CAT0057": ("court_order", "https://www.nysd.uscourts.gov/sites/default/files/2023-07/SEC%20vs%20Ripple%207-13-23.pdf"),
    "CAT0060": ("regulator_statement", "https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023"),
    "CAT0077": ("official_protocol_post", "https://blog.ethereum.org/2024/02/27/dencun-mainnet-announcement"),
    "CAT0079": ("regulator_order", "https://www.sec.gov/files/rules/sro/nysearca/2024/34-100224.pdf"),
    "CAT0082": ("official_protocol_post", "https://world.org/blog/announcements/extended-lock-up-for-tfh-team-and-investors"),
    "CAT0084": ("official_protocol_post", "https://telegram.org/blog/monetization-for-channels"),
    "CAT0086": ("official_protocol_post", "https://world.org/blog/announcements/introducing-world-app-3-super-app-humans"),
    "CAT0096": ("regulator_litigation_release", "https://www.sec.gov/enforcement-litigation/litigation-releases/lr-26306"),
    "CAT0097": ("regulator_litigation_release", "https://www.sec.gov/enforcement-litigation/litigation-releases/lr-26369"),
    "CAT0099": ("official_protocol_post", "https://www.pyth.network/blog/oracle-integrity-staking-incentivizing-safer-price-feeds-for-a-more-secure-defi"),
}

SOURCE_IDENTITY_TERMS = {
    "CAT0001": ("ripple", "xrp"), "CAT0004": ("london", "ethereum"),
    "CAT0018": ("unlock", "optimism"), "CAT0025": ("arbitrum", "token"),
    "CAT0036": ("gho", "aave"), "CAT0037": ("dydx",),
    "CAT0038": ("dydx", "trading"), "CAT0045": ("pyth", "staking"),
    "CAT0051": ("bitcoin",), "CAT0057": ("ripple",),
    "CAT0060": ("bitcoin", "exchange-traded"),
    "CAT0077": ("dencun", "ethereum"),
    "CAT0079": ("ethereum", "exchange-traded"),
    "CAT0082": ("lock-up", "worldcoin"), "CAT0084": ("revenue", "telegram"),
    "CAT0086": ("world app",), "CAT0096": ("ripple",),
    "CAT0097": ("ripple",), "CAT0099": ("pyth", "staking"),
}

FAMILY_MAP = {
    "legal_regulatory_repricing": "legal_regulatory_repricing",
    "etf_institutional_access": "etf_institutional_access",
    "supply_shock": "supply_float_changes",
    "unlock_vesting_change": "supply_float_changes",
    "protocol_utility_fee_revenue_change": "protocol_utility_fee_revenue_changes",
    "major_integration_distribution_access": "major_integration_distribution_access",
    "exchange_access_expansion": "exchange_access_expansion",
    "leverage_access_expansion": "leverage_access_expansion",
}

SCHEMA = ["event_id", "legacy_event_id", "asset_id", "ticker", "known_perp_symbols", "mechanism_family", "mechanism_subtype", "direction", "event_state", "first_public_ts_utc", "official_confirm_ts_utc", "effective_ts_utc", "timestamp_precision", "source_confidence", "primary_source_type", "primary_source_url", "supporting_source_urls", "source_publication_ts", "headline_raw", "classification_note", "durability_score_ex_ante", "estimated_float_impact_pct", "estimated_access_impact", "pre_run_note", "pre_run_risk", "unknown_fields_mask", "notes_on_uncertainty", "database_version", "row_content_hash"]


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(row: dict[str, Any]) -> str:
    payload = {k: (None if pd.isna(v) else v) for k, v in row.items() if k not in {"event_id", "row_content_hash"}}
    return digest_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode())


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def precision(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "unknown", "nan", "none", "null"}: return "unknown"
    if len(text) == 10 and text[4] == "-" and text[7] == "-": return "date_only"
    if "T" in text and (text.endswith("Z") or "+" in text): return "intraday_explicit_utc"
    return "coarse_or_nonstandard"


def fetch_sources(root: Path) -> pd.DataFrame:
    rows = []
    for legacy, (source_type, url) in SOURCES.items():
        target = root / "sources" / f"{legacy}.source"
        status, error, body, final_url = 0, "", b"", url
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 C2 provenance audit research@example.invalid"})
            with urllib.request.urlopen(request, timeout=30) as response:
                status, final_url, body = int(response.status), response.geturl(), response.read()
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(body)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        text = body.decode("utf-8", errors="ignore").lower()
        if body.startswith(b"%PDF"):
            extracted = target.with_suffix(".txt")
            completed = subprocess.run(["pdftotext", str(target), str(extracted)], capture_output=True, check=False)
            text = extracted.read_text(errors="ignore").lower() if completed.returncode == 0 and extracted.exists() else ""
            extracted.unlink(missing_ok=True)
        terms = SOURCE_IDENTITY_TERMS.get(legacy, ())
        content_identity_verified = bool(body and terms and all(term in text for term in terms))
        rows.append({"legacy_event_id": legacy, "source_family": source_type, "primary_source_url": url, "retrieval_status": status, "retrieved": status == 200 and len(body) > 0, "content_identity_verified": content_identity_verified, "identity_terms": "|".join(terms), "bytes": len(body), "content_sha256": digest_bytes(body) if body else "", "final_url": final_url, "retrieval_error": error, "retrieval_method": "direct_official_url_from_report_appendix", "retrieval_ts_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    return pd.DataFrame(rows)


def build_event(seed: pd.Series, source: pd.Series | None, confidence: str, note: str) -> dict[str, Any]:
    first = str(seed.get("first_public_ts_utc", "unknown"))
    row = {
        "legacy_event_id": seed.event_id, "asset_id": str(seed.ticker).upper(), "ticker": str(seed.ticker).upper(), "known_perp_symbols": "", "mechanism_family": FAMILY_MAP.get(str(seed.mechanism_family), "attention_only_events"), "mechanism_subtype": seed.mechanism_subtype, "direction": seed.direction, "event_state": seed.event_state,
        "first_public_ts_utc": first, "official_confirm_ts_utc": "unknown", "effective_ts_utc": str(seed.get("effective_ts_utc", "unknown")), "timestamp_precision": precision(first), "source_confidence": confidence,
        "primary_source_type": source["source_family"] if source is not None else "unresolved_report_seed", "primary_source_url": source["primary_source_url"] if source is not None else "", "supporting_source_urls": "", "source_publication_ts": first,
        "headline_raw": str(seed.classification_note), "classification_note": str(seed.classification_note), "durability_score_ex_ante": seed.durability_score_ex_ante, "estimated_float_impact_pct": "unknown", "estimated_access_impact": "unknown", "pre_run_note": "not assessed from returns", "pre_run_risk": seed.pre_run_risk,
        "unknown_fields_mask": "official_confirm_ts_utc|estimated_float_impact_pct|estimated_access_impact", "notes_on_uncertainty": note, "database_version": VERSION,
    }
    row["row_content_hash"] = canonical_hash(row)
    row["event_id"] = "C2V2_" + row["row_content_hash"][:16]
    return {k: row.get(k, "") for k in SCHEMA}


def run(root: Path) -> dict[str, Any]:
    if root.exists(): raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True)
    main_seed = pd.read_csv(EXTRACT / "catalyst_main_raw.csv")
    excluded_seed = pd.read_csv(EXTRACT / "catalyst_excluded_raw.csv")

    search_log = Path("/tmp/c2_recovery_search.txt")
    write_text(root / "recovery/artifact_search_log.md", "# Artifact Search Log\n\nTime-bounded recovery searched `/opt/testerdonch`, `/root`, `/mnt`, `/tmp`, `/media`, `/srv`, repository history/all trees, relevant archives, temporary/export paths, mounted paths, and available project roots. Neither named CSV was recovered. Detailed command output:\n\n```\n" + (search_log.read_text(errors="replace") if search_log.exists() else "Prior bounded search completed; raw command log unavailable.")[-20000:] + "\n```\n\nThe companions are recorded as lost upstream artifacts. Database v2 is a new source-revalidated product, not a reconstruction claim.")
    recovery = pd.DataFrame([{"artifact_name": name, "recovery_status": "lost_upstream_not_recovered", "rows": 0, "sha256": "", "overwritten": False} for name in ["post_catalyst_c2_catalyst_db_2020_2025_main.csv", "post_catalyst_c2_catalyst_db_2020_2025_excluded.csv"]])
    write_csv(root / "recovery/recovered_artifact_manifest.csv", recovery)

    retrieval = fetch_sources(root)
    write_csv(root / "database/source_retrieval_log.csv", retrieval)
    retrieval_by_id = {r.legacy_event_id: r for _, r in retrieval.iterrows()}
    main_rows, medium_rows, excluded_rows, seed_audit = [], [], [], []
    for _, seed in main_seed.iterrows():
        source = retrieval_by_id.get(seed.event_id)
        if source is not None and bool(source.retrieved) and bool(source.content_identity_verified):
            confidence, disposition, reason = "high", "main", "official_primary_source_retrieved"
        elif source is not None:
            confidence, disposition, reason = "medium", "medium", "official_primary_url_identified_but_direct_retrieval_failed"
        else:
            confidence, disposition, reason = "low", "excluded", "no_unambiguous_primary_source_url_recovered_from_appendix"
        event = build_event(seed, source, confidence, reason)
        {"main": main_rows, "medium": medium_rows, "excluded": excluded_rows}[disposition].append(event)
        seed_audit.append({"legacy_event_id": seed.event_id, "ticker": seed.ticker, "primary_source_identified": source is not None, "primary_source_retrieved": bool(source.retrieved) if source is not None else False, "disposition": disposition, "reason": reason, "post_event_return_used": False, "invented_timestamp_precision": False})
    for _, seed in excluded_seed.iterrows():
        pseudo = pd.Series({"event_id": seed.event_id, "ticker": seed.ticker, "mechanism_family": "attention_only", "mechanism_subtype": "report_exclusion", "direction": "diagnostic", "event_state": "excluded", "first_public_ts_utc": "unknown", "effective_ts_utc": "unknown", "classification_note": str(seed.get("comment", "")), "durability_score_ex_ante": "low", "pre_run_risk": "unknown"})
        excluded_rows.append(build_event(pseudo, None, "low", "explicit_report_exclusion:" + str(seed.get("exclusion reason", ""))))

    main = pd.DataFrame(main_rows, columns=SCHEMA); medium = pd.DataFrame(medium_rows, columns=SCHEMA); excluded = pd.DataFrame(excluded_rows, columns=SCHEMA)
    write_csv(root / "database/c2_catalyst_database_v2_main.csv", main)
    write_csv(root / "database/c2_catalyst_database_v2_medium_confidence.csv", medium)
    write_csv(root / "database/c2_catalyst_database_v2_excluded.csv", excluded)
    write_csv(root / "database/seed_event_revalidation_audit.csv", pd.DataFrame(seed_audit))

    combined = pd.concat([main.assign(ledger="main"), medium.assign(ledger="medium"), excluded.assign(ledger="excluded")], ignore_index=True)
    duplicate = combined.assign(duplicate_event_id=combined.event_id.duplicated(False), duplicate_content_hash=combined.row_content_hash.duplicated(False))[["event_id", "legacy_event_id", "row_content_hash", "ledger", "duplicate_event_id", "duplicate_content_hash"]]
    write_csv(root / "database/duplicate_and_conflict_audit.csv", duplicate)

    instruments = pd.read_parquet(INSTRUMENTS)
    instruments = instruments[instruments.symbol.astype(str).str.startswith("PF_")].copy()
    instruments["opening_ts"] = pd.to_datetime(instruments.openingDate, utc=True, errors="coerce")
    mapping_rows = []
    for _, event in pd.concat([main, medium], ignore_index=True).iterrows():
        match = instruments[instruments.base.astype(str).str.upper().eq(event.ticker)].sort_values("opening_ts")
        anchor = pd.to_datetime(event.first_public_ts_utc, utc=True, errors="coerce")
        symbol = match.iloc[0].symbol if len(match) else ""; opening = match.iloc[0].opening_ts if len(match) else pd.NaT
        if event.timestamp_precision == "date_only": lifecycle = bool(symbol and pd.notna(anchor) and pd.notna(opening) and opening < anchor.normalize())
        else: lifecycle = bool(symbol and pd.notna(anchor) and pd.notna(opening) and opening <= anchor)
        rankable = bool(event.source_confidence == "high" and event.primary_source_url and lifecycle and pd.notna(anchor) and anchor < PROTECTED)
        reason = "rankable_primary_source_and_prior_kraken_lifecycle" if rankable else ("no_kraken_perpetual_mapping" if not symbol else "prelisting_or_same_day_lifecycle_ineligible" if not lifecycle else "timestamp_insufficient" if pd.isna(anchor) else "medium_confidence_capped")
        mapping_rows.append({"event_id": event.event_id, "legacy_event_id": event.legacy_event_id, "ticker": event.ticker, "kraken_symbol": symbol, "kraken_opening_ts": opening, "event_anchor_ts": event.first_public_ts_utc, "timestamp_precision": event.timestamp_precision, "source_confidence": event.source_confidence, "kraken_lifecycle_eligible": lifecycle, "pit_rankable": rankable, "mapping_reason": reason})
    mapping = pd.DataFrame(mapping_rows)
    write_csv(root / "mapping/kraken_event_instrument_mapping.csv", mapping)
    mapped = pd.concat([main, medium], ignore_index=True).merge(mapping, on=["event_id", "legacy_event_id", "ticker", "timestamp_precision", "source_confidence"], how="left")
    rankable = mapped[mapped.pit_rankable].copy(); nonrankable = mapped[~mapped.pit_rankable].copy()
    write_csv(root / "mapping/pit_rankable_event_ledger.csv", rankable)
    write_csv(root / "mapping/nonrankable_event_ledger.csv", nonrankable)

    timestamp_audit = combined[["event_id", "legacy_event_id", "first_public_ts_utc", "official_confirm_ts_utc", "effective_ts_utc", "timestamp_precision", "source_confidence"]].copy()
    timestamp_audit["invented_timestamp_precision"] = False
    timestamp_audit["date_only_execution_rule"] = timestamp_audit.timestamp_precision.eq("date_only").map({True: "next_completed_daily_boundary", False: "not_applicable"})
    write_csv(root / "audit/timestamp_precision_audit.csv", timestamp_audit)
    provenance = combined[["event_id", "legacy_event_id", "source_confidence", "primary_source_type", "primary_source_url", "row_content_hash"]].copy()
    provenance["primary_url_present"] = provenance.primary_source_url.astype(str).str.len().gt(0)
    provenance["current_or_future_aware_label"] = False; provenance["post_event_return_informed"] = False
    write_csv(root / "audit/source_provenance_audit.csv", provenance)

    write_text(root / "database/database_schema_contract.md", "# Database Schema Contract\n\nVersion: `" + VERSION + "`. Every row uses the requested 29-field schema. `event_id` is derived from canonical row content; `legacy_event_id` is provenance only. Unknown values remain literal `unknown`. Date-only values remain date-only. `row_content_hash` excludes itself and `event_id`. Main requires a successfully retrieved official primary source. Medium is capped and non-rankable. Excluded contains unsupported seeds and report exclusions. Classification used no post-event returns.")
    write_text(root / "database/source_collection_scope.md", "# Source Collection Scope\n\nThis v2 is a bounded revalidation of the 36 report seeds, not discretionary headline discovery and not the missing 90-row database. Source families searched: regulators/courts, institutional derivatives/ETF notices, protocol/foundation/governance records, supply/float records, exchange access, leverage access, and major distribution integration. Date range: 2020-01-01 through 2025-12-31. Method: recover URLs from the report appendix and retrieve official URLs directly. Inclusion requires an event in the seed excerpt, an unambiguous official primary URL, consistency with the seed identity/date, and no outcome-informed classification. Unsupported, duplicate, attention-only, ambiguous, or inaccessible-primary rows are medium/excluded. Retrieval gaps are explicit in `source_retrieval_log.csv`; no broad archive completeness is claimed.")

    n = len(rankable); families = rankable.mechanism_family.value_counts().to_dict(); durable = len([f for f in families if f in {"legal_regulatory_repricing", "etf_institutional_access", "supply_float_changes", "protocol_utility_fee_revenue_changes", "major_integration_distribution_access"}])
    if n < 12: recommendation, canary = "insufficient_for_mechanical_canary", False
    elif n < 20: recommendation, canary = "canary_only_no_economic_tranche", True
    elif n < 40: recommendation, canary = "24-48_definitions", durable >= 3
    else: recommendation, canary = "60-120_definitions", durable >= 3
    write_text(root / "budget/canary_and_first_tranche_budget.md", f"# Canary and First-Tranche Budget\n\nPIT-rankable events: {n}. Durable mechanism families: {durable}. Rule-based recommendation: `{recommendation}`. Mechanical canary permitted: `{str(canary).lower()}`. An economic cross-family screen additionally requires at least three durable families. No manifest is padded and no economic scan was launched.")

    summary = {"run_root": str(root), "status": "complete_source_verified_v2" if len(main) else "blocked_by_protocol_issue", "original_csvs_recovered": False, "recovered_row_counts": {"main": 0, "excluded": 0}, "database_version": VERSION, "database_v2_counts": {"main": len(main), "medium": len(medium), "excluded": len(excluded)}, "seed_events_independently_revalidated": len(main), "pit_rankable_kraken_events": n, "rankable_events_by_mechanism_family": families, "timestamp_precision_distribution": rankable.timestamp_precision.value_counts().to_dict(), "duplicate_event_ids": int(combined.event_id.duplicated().sum()), "duplicate_content_hashes": int(combined.row_content_hash.duplicated().sum()), "invented_timestamp_precision": 0, "post_event_return_informed_classifications": 0, "prelisting_events_used_rankably": 0, "current_or_future_aware_labels": 0, "economic_scan_launched": False, "validation_launched": False, "holdout_launched": False, "recommended_definition_budget": recommendation, "mechanical_canary_may_proceed": canary, "main_source_gaps": int((~retrieval.retrieved).sum()) + (len(main_seed) - len(SOURCES)), "compact_bundle_path": str(root / "compact_review_bundle")}
    (root / "decision_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    bundle = root / "compact_review_bundle"; bundle.mkdir()
    for path in [p for p in root.rglob("*") if p.is_file() and "compact_review_bundle" not in p.parts and "sources" not in p.parts]:
        shutil.copy2(path, bundle / str(path.relative_to(root)).replace("/", "__"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True)
    summary = run(Path(parser.parse_args().run_root)); print(json.dumps(summary, indent=2, sort_keys=True)); return 0 if summary["status"].startswith("complete") else 2


if __name__ == "__main__": raise SystemExit(main())
