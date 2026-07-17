#!/usr/bin/env python3
"""Build the outcome-free C16 official flow-data authority preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


OBSERVATION_START = pd.Timestamp("2024-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2026-01-01T00:00:00Z")
DECISION = "C16_flow_authority_unavailable"
TASK_ID = "donch_bt_stage_6a_c16_flow_authority_20260717_v1"
STARTING_COMMIT = "e905a25a82582b8b6e436329c73c3a2117e793a6"

# Full-day NYSE closures in the bounded observation interval. Early closes are
# still eligible trading dates for this daily authority matrix.
NYSE_CLOSED = {
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27",
    "2025-12-25",
}

PANEL_COLUMNS = [
    "product_id", "ticker", "underlying_asset", "effective_trading_date",
    "measure_type", "measure_value", "measure_unit", "derived_or_reported",
    "derivation_formula", "publication_ts_utc", "first_available_ts_utc",
    "timestamp_precision", "revision_id", "revision_published_ts_utc",
    "supersedes_revision_id", "source_id", "source_file_sha256",
]


@dataclass(frozen=True)
class DownloadContract:
    source_id: str
    historical_or_current: str
    guaranteed_start: pd.Timestamp | None
    guaranteed_end_exclusive: pd.Timestamp | None
    content_type: str
    dated_immutable_artifact: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_download_contract(contract: DownloadContract) -> None:
    if contract.historical_or_current not in {"dated_historical", "current_only", "mixed"}:
        raise ValueError("unknown source time scope")
    if contract.dated_immutable_artifact:
        if contract.guaranteed_end_exclusive is None:
            raise ValueError("immutable artifact missing guaranteed end")
        if contract.guaranteed_end_exclusive > PROTECTED_START:
            raise ValueError("dated artifact can expose protected observations")
        return
    if contract.historical_or_current != "dated_historical":
        raise ValueError("current-only or mixed observation payload rejected before open")
    if contract.guaranteed_end_exclusive is None or contract.guaranteed_end_exclusive > PROTECTED_START:
        raise ValueError("historical payload lacks a server-enforced protected bound")


def guarded_open(contract: DownloadContract, reader: Callable[[], object]) -> object:
    validate_download_contract(contract)
    return reader()


def validate_observations(frame: pd.DataFrame) -> None:
    missing = set(PANEL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"missing panel columns: {sorted(missing)}")
    if frame.empty:
        return
    dates = pd.to_datetime(frame["effective_trading_date"], utc=True, errors="raise")
    publications = pd.to_datetime(frame["first_available_ts_utc"], utc=True, errors="raise")
    if (dates < OBSERVATION_START).any() or (dates >= PROTECTED_START).any():
        raise ValueError("observation outside bounded interval")
    if (publications >= PROTECTED_START).any():
        raise ValueError("protected publication timestamp")
    if frame["source_file_sha256"].eq("").any():
        raise ValueError("observation missing source hash")


def first_published_panel(frame: pd.DataFrame) -> pd.DataFrame:
    validate_observations(frame)
    if frame.empty:
        return frame.copy()
    keys = ["product_id", "effective_trading_date", "measure_type"]
    ordered = frame.sort_values(keys + ["first_available_ts_utc", "revision_id"], kind="stable")
    return ordered.drop_duplicates(keys, keep="first").reset_index(drop=True)


def latest_revised_panel(frame: pd.DataFrame) -> pd.DataFrame:
    validate_observations(frame)
    if frame.empty:
        return frame.copy()
    keys = ["product_id", "effective_trading_date", "measure_type"]
    ordered = frame.sort_values(keys + ["revision_published_ts_utc", "revision_id"], kind="stable")
    return ordered.drop_duplicates(keys, keep="last").reset_index(drop=True)


def derive_share_change(current_shares: float, prior_shares: float, unit_size: float) -> float:
    if unit_size <= 0 or current_shares < 0 or prior_shares < 0:
        raise ValueError("invalid official share inputs")
    return (current_shares - prior_shares) / unit_size


def trading_days(start: str, end_inclusive: str = "2025-12-31") -> pd.DatetimeIndex:
    days = pd.date_range(start, end_inclusive, freq="B", tz="UTC")
    closed = pd.to_datetime(sorted(NYSE_CLOSED), utc=True)
    return days[~days.normalize().isin(closed)]


def products() -> list[dict[str, str]]:
    rows = [
        ("BTC_GBTC", "GBTC", "Grayscale", "BTC", "2024-01-11", "conversion", "NYSE Arca"),
        ("BTC_BITB", "BITB", "Bitwise", "BTC", "2024-01-11", "launch", "NYSE Arca"),
        ("BTC_DEFI", "DEFI", "Hashdex", "BTC", "2024-03-27", "conversion", "NYSE Arca"),
        ("BTC_IBIT", "IBIT", "BlackRock/iShares", "BTC", "2024-01-11", "launch", "Nasdaq"),
        ("BTC_BRRR", "BRRR", "Valkyrie/CoinShares", "BTC", "2024-01-11", "launch", "Nasdaq"),
        ("BTC_ARKB", "ARKB", "ARK/21Shares", "BTC", "2024-01-11", "launch", "Cboe BZX"),
        ("BTC_BTCO", "BTCO", "Invesco/Galaxy", "BTC", "2024-01-11", "launch", "Cboe BZX"),
        ("BTC_HODL", "HODL", "VanEck", "BTC", "2024-01-11", "launch", "Cboe BZX"),
        ("BTC_BTCW", "BTCW", "WisdomTree", "BTC", "2024-01-11", "launch", "Cboe BZX"),
        ("BTC_FBTC", "FBTC", "Fidelity", "BTC", "2024-01-11", "launch", "Cboe BZX"),
        ("BTC_EZBC", "EZBC", "Franklin Templeton", "BTC", "2024-01-11", "launch", "Cboe BZX"),
        ("ETH_ETHE", "ETHE", "Grayscale", "ETH", "2024-07-23", "conversion", "NYSE Arca"),
        ("ETH_ETHW", "ETHW", "Bitwise", "ETH", "2024-07-23", "launch", "NYSE Arca"),
        ("ETH_ETH", "ETH", "Grayscale", "ETH", "2024-07-23", "launch", "NYSE Arca"),
        ("ETH_ETHA", "ETHA", "BlackRock/iShares", "ETH", "2024-07-23", "launch", "Nasdaq"),
        ("ETH_ETHV", "ETHV", "VanEck", "ETH", "2024-07-23", "launch", "Cboe BZX"),
        ("ETH_CETH", "CETH", "21Shares", "ETH", "2024-07-23", "launch", "Cboe BZX"),
        ("ETH_QETH", "QETH", "Invesco/Galaxy", "ETH", "2024-07-23", "launch", "Cboe BZX"),
        ("ETH_FETH", "FETH", "Fidelity", "ETH", "2024-07-23", "launch", "Cboe BZX"),
        ("ETH_EZET", "EZET", "Franklin Templeton", "ETH", "2024-07-23", "launch", "Cboe BZX"),
    ]
    output = []
    for product_id, ticker, issuer, asset, start, transition, exchange in rows:
        approval = "sec_34_99306" if asset == "BTC" else "sec_34_100224"
        if product_id == "ETH_ETH":
            approval = "sec_34_100541"
        output.append({
            "product_id": product_id,
            "ticker": ticker,
            "issuer": issuer,
            "administrator_exchange": exchange,
            "underlying_asset": asset,
            "spot_backed_or_other": "spot_backed",
            "inception_or_trading_start": start,
            "termination_or_conversion": transition if transition == "conversion" else "none_through_2025_known_from_scope",
            "creation_redemption_method": "cash_at_initial_approval; later_product_state_not_individually_verified",
            "shares_outstanding_field_source": "official_current_surface_exists; versioned_2024_2025_history_not_located",
            "underlying_holdings_field_source": "official_current_surface_exists; versioned_2024_2025_history_not_located",
            "NAV_AUM_field_source": "official_current_or_periodic_disclosure; not_flow",
            "reported_flow_field_source": "none_located",
            "publication_convention": "daily_current_disclosure_claims_do_not_reconstruct_first_publication_timestamp",
            "revision_convention": "unversioned_or_unknown",
            "source_ids": approval,
            "source_hashes": "filled_from_source_ledger",
            "inclusion_status": "excluded_no_first_published_daily_authority",
            "inclusion_reason": "primary_spot_product_but_daily_flow_creation_history_and_revision_timing_unavailable",
        })
    return output


def source_rows(raw_root: Path, accessed_utc: str) -> list[dict[str, object]]:
    specs = [
        ("sec_34_99306", "SEC", "regulator_approval_order", "https://www.sec.gov/files/rules/sro/nysearca/2024/34-99306.pdf", "sec/sec_34_99306.pdf", "BTC products", "2024-01-10"),
        ("sec_34_100224", "SEC", "regulator_approval_order", "https://www.sec.gov/files/rules/sro/nysearca/2024/34-100224.pdf", "sec/sec_34_100224.pdf", "ETH products except ETH mini", "2024-05-23"),
        ("sec_34_100541", "SEC", "regulator_approval_order", "https://www.sec.gov/files/rules/sro/nysearca/2024/34-100541.pdf", "sec/sec_34_100541.pdf", "ETH mini and separate ProShares filing", "2024-07-17"),
    ]
    rows: list[dict[str, object]] = []
    for source_id, owner, source_type, url, relative, covered, covered_date in specs:
        path = raw_root / relative
        rows.append({
            "source_id": source_id,
            "source_owner": owner,
            "source_type": source_type,
            "URL": url,
            "terms_or_license_URL": "https://www.sec.gov/about/privacy-information",
            "accessed_utc": accessed_utc,
            "HTTP_file_metadata": "dated immutable SEC PDF; response headers archived",
            "local_path": str(path),
            "bytes": path.stat().st_size,
            "SHA256": sha256(path),
            "covered_products": covered,
            "covered_dates": covered_date,
            "historical_versus_current_only": "dated_historical",
            "revision_version_behavior": "immutable release-number artifact",
            "observation_value_status": "mechanics_and_product_scope_only; no daily flow values",
            "retention_status": "retained_official_government_record",
        })
    discovery = [
        ("issuer_blackrock", "BlackRock/iShares", "IBIT; ETHA", "https://www.ishares.com/us/products/333011/ishares-bitcoin-trust-etf"),
        ("issuer_fidelity", "Fidelity", "FBTC; FETH", "https://www.fidelity.com/etfs/fbtc"),
        ("issuer_grayscale", "Grayscale", "GBTC; ETHE; ETH", "https://etfs.grayscale.com/"),
        ("issuer_bitwise", "Bitwise", "BITB; ETHW", "https://bitbetf.com/"),
        ("issuer_21shares", "21Shares", "ARKB; CETH", "https://www.21shares.com/en-us"),
        ("issuer_invesco", "Invesco", "BTCO; QETH", "https://www.invesco.com/us/en/solutions/invesco-etfs.html"),
        ("issuer_vaneck", "VanEck", "HODL; ETHV", "https://www.vaneck.com/us/en/digital-assets/"),
        ("issuer_franklin", "Franklin Templeton", "EZBC; EZET", "https://www.franklintempleton.com/investments/options/exchange-traded-funds"),
        ("issuer_wisdomtree", "WisdomTree", "BTCW", "https://www.wisdomtree.com/investments/etfs/crypto/btcw"),
        ("issuer_coinshares", "CoinShares/Valkyrie", "BRRR", "https://coinshares.com/us/etf/brrr/"),
        ("issuer_hashdex", "Hashdex", "DEFI", "https://hashdex-etfs.com/defi"),
    ]
    for source_id, owner, covered, url in discovery:
        rows.append({
            "source_id": source_id,
            "source_owner": owner,
            "source_type": "official_issuer_current_surface_metadata_only",
            "URL": url,
            "terms_or_license_URL": url,
            "accessed_utc": accessed_utc,
            "HTTP_file_metadata": "body_not_opened; current or mixed observation surface",
            "local_path": "",
            "bytes": 0,
            "SHA256": "",
            "covered_products": covered,
            "covered_dates": "current_only_or_unproven_history",
            "historical_versus_current_only": "current_only_or_mixed",
            "revision_version_behavior": "no dated snapshot/version authority established",
            "observation_value_status": "rejected_before_observation_open",
            "retention_status": "not_archived; terms and protected-payload scope not needed after rejection",
        })
    quarantine = raw_root.parent / "quarantine" / "mixed_current_surface_not_parsed.html"
    rows.append({
        "source_id": "issuer_ibit_bounded_probe_rejected",
        "source_owner": "BlackRock/iShares",
        "source_type": "failed_date_bounded_endpoint_probe",
        "URL": "historical-date query attempted against official IBIT surface",
        "terms_or_license_URL": "https://www.ishares.com/us/about-us/legal-information",
        "accessed_utc": accessed_utc,
        "HTTP_file_metadata": "requested CSV but response body was HTML; body quarantined and not parsed",
        "local_path": str(quarantine),
        "bytes": 1263124,
        "SHA256": "b2d3f2f20a6a9e172a8a443c350d82fce4de2012a2bcf3c3467157b5ea25e373",
        "covered_products": "IBIT",
        "covered_dates": "not established",
        "historical_versus_current_only": "mixed_response_rejected",
        "revision_version_behavior": "unavailable",
        "observation_value_status": "zero_rows_parsed; zero_observation_values_opened",
        "retention_status": "local_quarantine_not_in_manifest_or_handoff",
    })
    return rows


def coverage_and_gaps(product_rows: list[dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = []
    gaps = []
    for row in product_rows:
        eligible = trading_days(row["inception_or_trading_start"])
        for day in eligible:
            coverage.append({
                "product_id": row["product_id"], "ticker": row["ticker"],
                "underlying_asset": row["underlying_asset"],
                "effective_trading_date": day.date().isoformat(),
                "product_active": True, "authoritative_value_available": False,
                "first_publication_reconstructable": False,
                "revision_versioned": False, "measure_type": "unavailable",
                "coverage_status": "missing_daily_authoritative_flow_creation_measure",
            })
        gaps.append({
            "product_id": row["product_id"], "underlying_asset": row["underlying_asset"],
            "gap_start": eligible.min().date().isoformat() if len(eligible) else "",
            "gap_end": eligible.max().date().isoformat() if len(eligible) else "",
            "missing_eligible_days": len(eligible), "consecutive_gap_days": len(eligible),
            "gap_reason": "no official versioned daily value with reconstructable first-publication timestamp",
            "revision_ambiguity": "unversioned_or_unknown",
        })
    return pd.DataFrame(coverage), pd.DataFrame(gaps)


def artifact_manifest(
    root: Path, local_panel_root: Path, status: str, generated_utc: str
) -> dict[str, object]:
    rows = []
    for path in sorted(root.iterdir()):
        if (
            path.is_file()
            and path.name not in {
                "ARTIFACT_MANIFEST.json", "BUILD_AS_OF_UTC.txt", "UPDATED_RUN_REGISTRY.csv"
            }
            and path.suffix != ".parquet"
        ):
            rows.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    for name in ["C16_FIRST_PUBLISHED_FLOW_PANEL.parquet", "C16_LATEST_REVISED_AUDIT_PANEL.parquet"]:
        path = local_panel_root / name
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256(path), "local_only": True})
    return {"task_id": TASK_ID, "status": status, "generated_utc": generated_utc, "artifacts": rows}


def build(raw_root: Path, run_root: Path, archive_root: Path, accessed_utc: str) -> None:
    expected_outputs = {
        "C16_SOURCE_LEDGER.csv", "C16_PRODUCT_REGISTRY.csv",
        "C16_FIRST_PUBLISHED_FLOW_PANEL.parquet", "ARTIFACT_MANIFEST.json",
    }
    if run_root.exists() and any((run_root / name).exists() for name in expected_outputs):
        raise FileExistsError(f"run outputs already exist: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    sources = pd.DataFrame(source_rows(raw_root, accessed_utc))
    product_rows = products()
    source_hash = dict(zip(sources["source_id"], sources["SHA256"]))
    for row in product_rows:
        row["source_hashes"] = source_hash.get(row["source_ids"], "")
    registry = pd.DataFrame(product_rows)
    coverage, gaps = coverage_and_gaps(product_rows)
    panel = pd.DataFrame(columns=PANEL_COLUMNS)
    validate_observations(panel)
    first = first_published_panel(panel)
    latest = latest_revised_panel(panel)

    sources.to_csv(run_root / "C16_SOURCE_LEDGER.csv", index=False)
    registry.to_csv(run_root / "C16_PRODUCT_REGISTRY.csv", index=False)
    coverage.to_csv(run_root / "C16_DAILY_COVERAGE_MATRIX.csv", index=False)
    gaps.to_csv(run_root / "C16_GAP_AND_REVISION_REGISTER.csv", index=False)
    publication = registry[["product_id", "ticker", "underlying_asset"]].copy()
    publication["eligible_days"] = publication.product_id.map(coverage.groupby("product_id").size())
    publication["authoritative_value_days"] = 0
    publication["first_publication_reconstructable_days"] = 0
    publication["first_publication_coverage_fraction"] = 0.0
    publication["revision_versioned_days"] = 0
    publication["audit_result"] = "fail_no_versioned_daily_authority"
    publication.to_csv(run_root / "C16_PUBLICATION_AND_REVISION_AUDIT.csv", index=False)
    atomic_parquet(first, run_root / "C16_FIRST_PUBLISHED_FLOW_PANEL.parquet")
    atomic_parquet(latest, run_root / "C16_LATEST_REVISED_AUDIT_PANEL.parquet")

    measures = """# C16 Measure Semantics\n\n+- `official_reported_net_flow`: an issuer/administrator value explicitly labeled net flow. None was located with versioned daily 2024-2025 history.\n+- `creation_redemption_units`: official creation units less redemption units for an effective date. Mechanics were confirmed, but no dated daily series was located.\n+- `shares_outstanding_change`: a derived share change only when consecutive official, dated, versioned shares and the creation-unit size are available. It is not used here.\n+- `underlying_holdings_change`: change in official underlying units. It mixes creations/redemptions with fees, in-kind mechanics, and other inventory changes unless the product contract resolves them; it is not relabeled as flow.\n+- `NAV_or_AUM_change`: never flow because it includes asset-price movement.\n+- `third_party_estimate`: discovery only and prohibited from authority.\n\nThe SEC orders confirm spot backing and initial cash creation/redemption mechanics. They do not supply daily creation values. Current issuer pages cannot reconstruct first publication or revisions, so both authority panels contain zero rows.\n"""
    (run_root / "C16_MEASURE_SEMANTICS.md").write_text(measures)
    terms = """# C16 Terms and License Review\n\n+Immutable SEC approval orders were retained as official U.S. government records. They establish product scope and mechanics, not daily values. Current issuer observation surfaces were not archived or parsed because they are current/mixed, lack proved pre-2026 server bounds, and do not provide dated revision authority. Their retention or derived-publication terms therefore do not need to be relied upon. No paid or third-party observation source was used.\n\n+One malformed historical-date probe returned HTML rather than the requested dated CSV. No observation row or value was parsed. The response is quarantined locally, excluded from panels, manifests, Git, and Drive. This probe proves that endpoint cannot be trusted as a bounded historical source.\n"""
    (run_root / "C16_TERMS_AND_LICENSE_REVIEW.md").write_text(terms)

    asset_summary = coverage.groupby("underlying_asset").agg(
        eligible_product_days=("product_id", "size"),
        active_product_count=("product_id", "nunique"),
        authoritative_value_days=("authoritative_value_available", "sum"),
        first_publication_days=("first_publication_reconstructable", "sum"),
    ).reset_index()
    asset_summary["usable_asset_days"] = 0
    asset_summary["first_publication_coverage_fraction"] = 0.0
    manifest = {
        "task_id": TASK_ID,
        "decision": DECISION,
        "starting_commit": STARTING_COMMIT,
        "observation_interval": "[2024-01-01,2026-01-01)",
        "protected_observations_opened": 0,
        "mixed_payload_observation_rows_parsed": 0,
        "Kraken_outcomes_opened": 0,
        "economic_outputs_computed": 0,
        "source_ledger_sha256": sha256(run_root / "C16_SOURCE_LEDGER.csv"),
        "product_registry_sha256": sha256(run_root / "C16_PRODUCT_REGISTRY.csv"),
        "first_published_panel_sha256": sha256(run_root / "C16_FIRST_PUBLISHED_FLOW_PANEL.parquet"),
        "latest_revised_panel_sha256": sha256(run_root / "C16_LATEST_REVISED_AUDIT_PANEL.parquet"),
        "source_count": len(sources),
        "product_count": len(registry),
        "panel_rows": 0,
        "coverage": asset_summary.to_dict("records"),
        "feasibility": {
            "economically_valid_flow_measure": False,
            "first_publication_coverage_at_least_80pct": False,
            "no_unversioned_ambiguity_on_usable_days": True,
            "BTC_usable_asset_days_at_least_150": False,
            "ETH_usable_asset_days_at_least_100": False,
            "identity_collisions": 0,
            "protected_observation_rows_opened": 0,
        },
    }
    write_json(run_root / "C16_DATA_MANIFEST.json", manifest)
    decision_text = f"""# C16 Authority Decision\n\n+Decision: `{DECISION}`\n\n+The 20-product primary cohort is identifiable from immutable SEC orders, and initial cash creation/redemption mechanics are authoritative. No free official source located in this bounded preflight supplied a versioned daily 2024-2025 flow, creation/redemption, shares, or holdings series with reconstructable first-publication timestamps. Official current surfaces overwrite or expose current state; SEC filings are dated but periodic and do not provide the required daily flow authority.\n\n+BTC usable asset-days: `0` (required 150). ETH usable asset-days: `0` (required 100). First-publication coverage: `0%` (required 80%). The failure is data authority, not economic evidence.\n"""
    (run_root / "C16_AUTHORITY_DECISION.md").write_text(decision_text)
    (run_root / "C16_NEXT_CONTRACT_RECOMMENDATION.md").write_text(
        "# C16 Next Contract Recommendation\n\nDo not design C16 signals or join Kraken outcomes. The exact non-economic remedy is a free official issuer/administrator archive that supplies dated daily creations/redemptions or shares/holdings, preserves first-publication timestamps and revisions, is server-bounded before 2026, and has clear retention/derived-publication rights. A manually preserved contemporaneous archive cannot be reconstructed retroactively from current pages.\n"
    )
    (run_root / "VALIDATION.md").write_text(
        "# Validation\n\nSynthetic tests cover publication clocks, revisions, current/mixed rejection before reader invocation, measure semantics, derived-share arithmetic, lifecycle coverage, protected boundaries, and deterministic hashes. Exact command and final counts are recorded in the task archive after execution.\n"
    )
    (run_root / "REVIEW.md").write_text(
        "# Independent Review\n\nPending final independent source, semantics, coverage, hash, and claims review. This placeholder cannot be treated as approval until the task archive records the review result.\n"
    )
    (run_root / "COMPLETION.md").write_text(
        f"# Completion\n\nStatus: `{DECISION}`. No Kraken outcome, economic output, capture, paid source, or protected observation value was opened.\n"
    )
    (run_root / "NEXT_ACTION.md").write_text(
        "# Next Action\n\nStop C16. Await an authoritative versioned daily official archive; otherwise choose a different pre-existing hypothesis only under separate human authorization.\n"
    )
    write_json(
        run_root / "ARTIFACT_MANIFEST.json",
        artifact_manifest(run_root, run_root, DECISION, accessed_utc),
    )

    for path in run_root.iterdir():
        if path.is_file() and path.suffix != ".parquet":
            shutil.copy2(path, archive_root / path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--as-of-utc", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.raw_root, args.run_root, args.archive_root, args.as_of_utc)
