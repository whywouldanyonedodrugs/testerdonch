#!/usr/bin/env python3
"""Build the outcome-free C03 PIT lifecycle authority and gated context inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from tools.build_kraken_u2_lifecycle_authority import (
        archived_observed_at, header_value, instrument_snapshot, normalize_http_date,
    )
    from tools.build_kraken_c01_reference_panel_authority import parse_terminal_lifecycle_html
except ModuleNotFoundError:
    from build_kraken_u2_lifecycle_authority import (
        archived_observed_at, header_value, instrument_snapshot, normalize_http_date,
    )
    from build_kraken_c01_reference_panel_authority import parse_terminal_lifecycle_html


TASK_ID = "donch_bt_c03_pit_cohort_breadth_20260717_v1"
TRAIN_START = pd.Timestamp("2023-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2026-01-01T00:00:00Z")
TRADE_DATASET = "historical_trade_candles_5m"
MARK_DATASET = "historical_mark_candles_5m"
SAFE_EVENT_FIELDS = {
    "event_id", "source_event_id", "candidate_id", "symbol", "PF_symbol",
    "decision_ts", "feature_available_ts", "canonical_episode_id",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1"}


def normalize_opening(values: Iterable[Any]) -> list[pd.Timestamp]:
    parsed = {pd.Timestamp(value).tz_convert("UTC") for value in values if value not in (None, "")}
    return sorted(parsed)


def has_unbounded_resumption(
    history: list[tuple[str, dict[str, Any], Path]], terminal_days: list[pd.Timestamp], explicit_resumption: bool,
) -> bool:
    if explicit_resumption:
        return True
    return any(
        row.get("tradeable") is True and pd.Timestamp(observed_at).tz_convert("UTC") > terminal
        for observed_at, row, _ in history for terminal in terminal_days
    )


def classify_day(
    day: pd.Timestamp,
    *,
    openings: list[pd.Timestamp],
    terminal_days: list[pd.Timestamp],
    resumed_without_boundary: bool,
    identity_collision: bool,
    verified_full_day_intervals: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = (),
    invalid_intervals: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = (),
) -> tuple[str, str, str]:
    """Classify a UTC day without carrying point observations across time."""
    start = pd.Timestamp(day).tz_convert("UTC").floor("D")
    end = start + pd.Timedelta(days=1)
    if identity_collision or len(openings) > 1:
        return "unknown", "identity_or_opening_collision", "fail_closed"
    opening = openings[0] if openings else None
    if opening is not None and end <= opening:
        return "verified_ineligible", "before_official_opening", "high"
    if any(start < invalid_end and end > invalid_start for invalid_start, invalid_end in invalid_intervals):
        return "verified_ineligible", "known_suspension_migration_or_settlement_interval", "high"
    settled = [terminal for terminal in terminal_days if start >= terminal.floor("D")]
    if settled:
        if resumed_without_boundary:
            return "unknown", "official_resumption_exists_but_boundary_is_unresolved", "fail_closed"
        return "verified_ineligible", "official_settlement_without_documented_resumption", "high"
    if any(start >= valid_start and end <= valid_end for valid_start, valid_end in verified_full_day_intervals):
        return "verified_eligible", "official_full_day_status_interval", "high"
    if opening is None:
        return "unknown", "official_opening_or_start_unavailable", "fail_closed"
    return "unknown", "continuous_status_between_point_snapshots_unproven", "fail_closed"


def feasibility(daily: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, Any]] = []
    for date, group in daily.groupby("date", sort=True):
        counts = group.status_class.value_counts()
        eligible = int(counts.get("verified_eligible", 0))
        ineligible = int(counts.get("verified_ineligible", 0))
        unknown = int(counts.get("unknown", 0))
        potential = eligible + unknown
        unknown_share = unknown / potential if potential else 1.0
        covered = int((group.status_class.eq("verified_eligible") & group.trade_bar_available & group.mark_bar_available).sum())
        collisions = int((group.status_class.eq("verified_eligible") & group.identity_collision).sum())
        usable = covered >= 30 and unknown_share <= 0.20 and collisions == 0
        rows.append({
            "level": "daily", "date_or_year": str(pd.Timestamp(date).date()),
            "potentially_eligible_identity_count": potential,
            "verified_eligible_count": eligible, "verified_ineligible_count": ineligible,
            "unknown_count": unknown, "unknown_share": unknown_share,
            "trade_mark_covered_eligible_count": covered,
            "included_identity_collision_count": collisions, "usable": usable,
        })
    matrix = pd.DataFrame(rows)
    yearly: list[dict[str, Any]] = []
    for year in (2023, 2024, 2025):
        subset = matrix[matrix.date_or_year.str.startswith(str(year))]
        fraction = float(subset.usable.mean()) if len(subset) else 0.0
        yearly.append({
            "level": "yearly", "date_or_year": str(year),
            "potentially_eligible_identity_count": int(subset.potentially_eligible_identity_count.max()) if len(subset) else 0,
            "verified_eligible_count": int(subset.verified_eligible_count.max()) if len(subset) else 0,
            "verified_ineligible_count": int(subset.verified_ineligible_count.max()) if len(subset) else 0,
            "unknown_count": int(subset.unknown_count.max()) if len(subset) else 0,
            "unknown_share": float(subset.unknown_share.max()) if len(subset) else 1.0,
            "trade_mark_covered_eligible_count": int(subset.trade_mark_covered_eligible_count.max()) if len(subset) else 0,
            "included_identity_collision_count": int(subset.included_identity_collision_count.max()) if len(subset) else 0,
            "usable": fraction >= 0.80, "usable_day_fraction": fraction,
        })
    result = pd.concat([matrix, pd.DataFrame(yearly)], ignore_index=True)
    return result, bool(all(row["usable"] for row in yearly))


def breadth_metrics(returns: pd.Series, expected_members: int) -> dict[str, Any]:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    coverage = len(values) / expected_members if expected_members else 0.0
    available = expected_members >= 20 and len(values) >= 20 and coverage >= 0.80
    if not available:
        return {"available": False, "valid_members": len(values), "expected_members": expected_members, "coverage": coverage}
    absolute = values.abs().sort_values(ascending=False)
    total_absolute = float(absolute.sum())
    positive = float(values.gt(0).mean()); negative = float(values.lt(0).mean())
    return {
        "available": True, "valid_members": len(values), "expected_members": expected_members,
        "coverage": coverage, "positive_return_fraction": positive,
        "negative_return_fraction": negative, "signed_breadth": positive - negative,
        "cross_sectional_return_median": float(values.median()),
        "cross_sectional_dispersion": float(values.std(ddof=0)),
        "top_1_absolute_move_share": float(absolute.iloc[:1].sum() / total_absolute) if total_absolute else 0.0,
        "top_5_absolute_move_share": float(absolute.iloc[:5].sum() / total_absolute) if total_absolute else 0.0,
        "participation_concentration": float((absolute / total_absolute).pow(2).sum()) if total_absolute else 0.0,
    }


def safe_event_identity(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    requested = set(columns)
    if not requested or not requested.issubset(SAFE_EVENT_FIELDS):
        raise ValueError("non-identity event field requested")
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns))
    if path.suffix == ".csv":
        return pd.read_csv(path, usecols=list(columns))
    raise ValueError("unsupported frozen-event identity format")


def attach_feature_availability(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    left = events.copy(); right = features.copy()
    left["decision_ts"] = pd.to_datetime(left["decision_ts"], utc=True, errors="raise")
    right["feature_available_ts"] = pd.to_datetime(right["feature_available_ts"], utc=True, errors="raise")
    if (left.decision_ts >= PROTECTED_START).any() or (right.feature_available_ts >= PROTECTED_START).any():
        raise ValueError("protected row in coverage join")
    joined = left.merge(right, left_on=["symbol", "decision_ts"], right_on=["symbol", "feature_available_ts"], how="left", validate="many_to_one")
    return joined


def merge_intervals(rows: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    intervals = sorted((row.chunk_start, row.chunk_end) for row in rows.itertuples(index=False) if row.chunk_start < row.chunk_end)
    merged: list[list[pd.Timestamp]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_days(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> set[pd.Timestamp]:
    days: set[pd.Timestamp] = set()
    for start, end in intervals:
        bounded_start = max(start, TRAIN_START); bounded_end = min(end, PROTECTED_START)
        if bounded_start >= bounded_end:
            continue
        days.update(pd.date_range(bounded_start.floor("D"), (bounded_end - pd.Timedelta(nanoseconds=1)).floor("D"), freq="D", tz="UTC"))
    return days


def load_manifest(path: Path) -> pd.DataFrame:
    columns = ["dataset", "symbol", "rows", "status", "chunk_start", "chunk_end", "rankable_pre_holdout", "contains_protected_period"]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame[
        frame.dataset.isin([TRADE_DATASET, MARK_DATASET])
        & frame.symbol.fillna("").str.startswith("PF_") & frame.status.eq("downloaded")
        & frame.rankable_pre_holdout.map(truthy) & ~frame.contains_protected_period.map(truthy)
        & pd.to_numeric(frame.rows, errors="coerce").fillna(0).gt(0)
    ].copy()
    frame["chunk_start"] = pd.to_datetime(frame.chunk_start, utc=True, errors="raise")
    frame["chunk_end"] = pd.to_datetime(frame.chunk_end, utc=True, errors="raise")
    if (frame.chunk_start >= PROTECTED_START).any() or (frame.chunk_end > PROTECTED_START).any():
        raise ValueError("protected manifest interval survived pre-open filter")
    return frame


def snapshots(source_dir: Path) -> list[tuple[str, dict[str, dict[str, Any]], Path]]:
    result = []
    for path in sorted(source_dir.glob("wayback_instruments_20*.body")):
        observed = archived_observed_at(path); result.append((observed, instrument_snapshot(path, observed), path))
    current = source_dir / "kraken_futures_instruments.body"
    observed = normalize_http_date(header_value(source_dir / "kraken_futures_instruments.headers.txt", "date"))
    result.append((observed, instrument_snapshot(current, observed), current))
    return result


def build(args: argparse.Namespace) -> str:
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output root already exists")
    output.mkdir(parents=True)
    snap = snapshots(args.instrument_sources)
    manifest = load_manifest(args.download_manifest)
    terminal_rows, resumed = parse_terminal_lifecycle_html(args.terminal_source.read_bytes())
    terminal = defaultdict(list)
    for row in terminal_rows:
        terminal[str(row["symbol"]).upper()].append(pd.to_datetime(row["settlement_date"], utc=True, dayfirst=True))

    observed: dict[str, list[tuple[str, dict[str, Any], Path]]] = defaultdict(list)
    for observed_at, table, path in snap:
        for symbol, row in table.items():
            if symbol.startswith("PF_") and row.get("type") == "flexible_futures":
                observed[symbol].append((observed_at, row, path))
    symbols = sorted(set(observed) | set(manifest.symbol.astype(str)) | set(terminal))
    coverage: dict[tuple[str, str], set[pd.Timestamp]] = {}
    for (symbol, dataset), group in manifest.groupby(["symbol", "dataset"], sort=True):
        coverage[(symbol, dataset)] = interval_days(merge_intervals(group))

    manifest_hash = sha256_file(args.download_manifest); terminal_hash = sha256_file(args.terminal_source)
    source_rows = []
    for observed_at, _, path in snap:
        source_rows.append({"source_id":path.stem,"source_kind":"official_current_instrument_snapshot" if path.name.startswith("kraken_futures") else "archived_official_instrument_snapshot","path":str(path),"observed_or_access_utc":observed_at,"sha256":sha256_file(path),"authority_use":"point_identity_and_status_only","continuous_status_claim":False})
    source_rows.extend([
        {"source_id":"official_terminal_event_ledger","source_kind":"official_cumulative_terminal_notice","path":str(args.terminal_source),"observed_or_access_utc":"2026-07-17","sha256":terminal_hash,"authority_use":"settlement_and_resumption_evidence","continuous_status_claim":False},
        {"source_id":"rankable_trade_mark_manifest","source_kind":"local_official_kraken_acquisition_manifest","path":str(args.download_manifest),"observed_or_access_utc":"2026-07-16","sha256":manifest_hash,"authority_use":"operational_bar_availability_only","continuous_status_claim":False},
        {"source_id":"stage2c1_volume_semantics","source_kind":"archived_official_specification_semantics","path":str(args.semantic_source),"observed_or_access_utc":"2026-07-17","sha256":sha256_file(args.semantic_source),"authority_use":"identity_and_base_volume_semantics_only","continuous_status_claim":False},
    ])
    pd.DataFrame(source_rows).to_csv(output / "C03_LIFECYCLE_SOURCE_LEDGER.csv", index=False)

    inventory_rows = []; daily_rows = []; days = pd.date_range(TRAIN_START, PROTECTED_START - pd.Timedelta(days=1), freq="D", tz="UTC")
    for symbol in symbols:
        history = observed.get(symbol, [])
        openings = normalize_opening([row.get("openingDate") for _, row, _ in history])
        bases = sorted({str(row.get("base")) for _, row, _ in history if row.get("base") not in (None, "")})
        quotes = sorted({str(row.get("quote")) for _, row, _ in history if row.get("quote") not in (None, "")})
        pairs = sorted({str(row.get("pair")) for _, row, _ in history if row.get("pair") not in (None, "")})
        collision = len(openings) > 1 or len(bases) > 1 or len(quotes) > 1 or len(pairs) > 1
        resumption_unbounded = has_unbounded_resumption(history, terminal.get(symbol,[]), symbol in resumed)
        point_status = [{"observed_at":at,"tradeable":row.get("tradeable"),"source_id":path.stem} for at,row,path in history]
        source_items = [f"{path.stem}:{sha256_file(path)}" for _,_,path in history]
        if symbol in terminal: source_items.append(f"official_terminal_event_ledger:{terminal_hash}")
        if symbol in set(manifest.symbol): source_items.append(f"rankable_trade_mark_manifest:{manifest_hash}")
        identity_version = "c03id_" + canonical_hash({"symbol":symbol,"openings":[str(x) for x in openings],"bases":bases,"quotes":quotes,"pairs":pairs})[:24]
        unknown = []
        if not history: unknown.append("official_instrument_identity_snapshot_missing")
        if not openings: unknown.append("official_opening_or_start_missing")
        if collision: unknown.append("identity_or_opening_collision")
        if resumption_unbounded: unknown.append("official_resumption_observed_but_exact_boundary_unresolved")
        unknown.append("continuous_status_between_sparse_point_snapshots_unproven")
        inventory_rows.append({
            "canonical_asset_id":bases[0] if len(bases)==1 else "", "PF_symbol":symbol,
            "identity_version":identity_version, "opening_or_start":";".join(x.isoformat() for x in openings),
            "terminal_or_end":";".join(x.isoformat() for x in sorted(terminal.get(symbol,[]))),
            "status_intervals":json.dumps(point_status,sort_keys=True,separators=(",",":")),
            "status_confidence":"point_observations_only_continuity_unknown",
            "source_ids_and_hashes":";".join(sorted(set(source_items))),
            "trade_coverage_day_count":len(coverage.get((symbol,TRADE_DATASET),set())),
            "mark_coverage_day_count":len(coverage.get((symbol,MARK_DATASET),set())),
            "identity_collision":collision, "unknown_reason":";".join(unknown),
        })
        for day in days:
            status, reason, confidence = classify_day(
                day, openings=openings, terminal_days=terminal.get(symbol,[]),
                resumed_without_boundary=resumption_unbounded, identity_collision=collision,
            )
            daily_rows.append({
                "date":day, "canonical_asset_id":bases[0] if len(bases)==1 else "", "PF_symbol":symbol,
                "identity_version":identity_version, "opening_or_start":";".join(x.isoformat() for x in openings),
                "terminal_or_end":";".join(x.isoformat() for x in sorted(terminal.get(symbol,[]))),
                "status_intervals":json.dumps(point_status,sort_keys=True,separators=(",",":")),
                "status_class":status, "status_confidence":confidence,
                "source_ids_and_hashes":";".join(sorted(set(source_items))),
                "trade_bar_available":day in coverage.get((symbol,TRADE_DATASET),set()),
                "mark_bar_available":day in coverage.get((symbol,MARK_DATASET),set()),
                "unknown_reason":reason if status=="unknown" else "", "identity_collision":collision,
            })
    inventory = pd.DataFrame(inventory_rows).sort_values(["PF_symbol","identity_version"],kind="mergesort")
    daily = pd.DataFrame(daily_rows).sort_values(["date","PF_symbol","identity_version"],kind="mergesort")
    if daily.duplicated(["date","PF_symbol","identity_version"]).any(): raise ValueError("duplicate daily identity")
    if set(daily.status_class) - {"verified_eligible","verified_ineligible","unknown"}: raise ValueError("invalid status class")
    inventory.to_csv(output / "C03_PIT_IDENTITY_INVENTORY.csv",index=False)
    daily.to_parquet(output / "C03_DAILY_MEMBERSHIP_AUTHORITY.parquet",index=False)
    matrix, gate = feasibility(daily); matrix.to_csv(output / "C03_PIT_FEASIBILITY_MATRIX.csv",index=False)
    yearly = matrix[matrix.level.eq("yearly")]
    (output / "C03_PIT_FEASIBILITY_REPORT.md").write_text(
        "# C03 PIT Feasibility Report\n\n"
        f"Status: `{'pass' if gate else 'C03_PIT_authority_unavailable'}`.\n\n"
        f"Observed PF identities: {len(inventory)}. Verified-eligible full UTC days: {int((daily.status_class=='verified_eligible').sum())}. "
        "Sparse official point snapshots were not carried across days. Current-roster membership and bar existence were not used as lifecycle proof.\n\n"
        + "\n".join(f"- {r.date_or_year}: usable-day fraction `{r.usable_day_fraction:.3%}`; maximum unknown share `{r.unknown_share:.3%}`; maximum covered eligible `{int(r.trade_mark_covered_eligible_count)}`." for r in yearly.itertuples())
        + "\n\nThe frozen gate requires at least 80% usable days in each year, at least 30 covered verified members, unknown share no more than 20%, and no included identity collision.\n"
    )
    status = "ready_for_C03_context_contract_review" if gate else "C03_PIT_authority_unavailable"
    (output / "C03_NEXT_CONTRACT_RECOMMENDATION.md").write_text(
        "# C03 Next Contract Recommendation\n\n"
        + ("The PIT gate passed; review the frozen context contract before any later use.\n" if gate else "Do not build the primary breadth panel. Acquire or recover official full-day/interval lifecycle-status authority sufficient to reduce unknown membership below the frozen threshold; do not substitute current-survivor or bar-existence breadth.\n")
    )
    (output / "VALIDATION.md").write_text(
        "# Validation\n\nOfficial point snapshots were kept as point evidence; unknown intervals were preserved. "
        "No market payload, C01/C02 economic ledger, protected outcome, or feature/outcome row was opened. "
        f"Identity collisions: {int(inventory.identity_collision.sum())}. Phase B executed: {'yes' if gate else 'no'}.\n"
    )
    (output / "REVIEW.md").write_text("# Review\n\nPending independent review.\n")
    (output / "COMPLETION.md").write_text(f"# Completion\n\nStatus: `{status}`. Economic outputs computed: zero. Protected outcomes opened: zero.\n")
    (output / "NEXT_ACTION.md").write_text((output / "C03_NEXT_CONTRACT_RECOMMENDATION.md").read_text())
    artifact_manifest(output)
    return status


def artifact_manifest(root: Path) -> None:
    rows=[]
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name!="ARTIFACT_MANIFEST.json"):
        rows.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    (root/"ARTIFACT_MANIFEST.json").write_text(json.dumps({"task_id":TASK_ID,"files":rows},indent=2,sort_keys=True)+"\n")


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser()
    parser.add_argument("--instrument-sources",type=Path,required=True)
    parser.add_argument("--terminal-source",type=Path,required=True)
    parser.add_argument("--download-manifest",type=Path,required=True)
    parser.add_argument("--semantic-source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    return parser.parse_args()


def main() -> int:
    args=parse_args(); status=build(args); print(status); return 0


if __name__=="__main__": raise SystemExit(main())
