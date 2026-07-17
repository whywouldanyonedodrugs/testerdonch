#!/usr/bin/env python3
"""Build official Kraken USD spot-reference authority without economic analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


TRAIN_START = pd.Timestamp("2023-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2026-01-01T00:00:00Z")
OFFICIAL_SUPPORT_URL = (
    "https://support.kraken.com/articles/"
    "360047543791-downloadable-historical-market-data-time-and-sales-"
)
OFFICIAL_ASSET_PAIRS_URL = "https://api.kraken.com/0/public/AssetPairs"
FULL_ARCHIVE_URL = (
    "https://drive.google.com/file/d/10zh3tDpqANYvVtYVgczwVz3UZFRUb1el/view"
)
QUARTERLY_FOLDER_URL = (
    "https://drive.google.com/drive/folders/188O9xQjZTythjyLNes_5zfMEFaMbTT22"
)
PILOT_WINDOWS = (
    (pd.Timestamp("2023-01-01T00:00:00Z"), pd.Timestamp("2023-01-08T00:00:00Z")),
    (pd.Timestamp("2024-06-01T00:00:00Z"), pd.Timestamp("2024-06-08T00:00:00Z")),
    (pd.Timestamp("2025-12-24T00:00:00Z"), PROTECTED_START),
)
PILOT_PF = ("PF_XBTUSD", "PF_ETHUSD", "PF_1INCHUSD", "PF_AAVEUSD")


@dataclass(frozen=True)
class ArchiveSpec:
    archive_id: str
    path: Path
    source_url: str
    eligible_start: pd.Timestamp
    eligible_end: pd.Timestamp


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(raw)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_archive_spec(spec: ArchiveSpec) -> None:
    if not spec.path.is_file():
        raise FileNotFoundError(spec.path)
    if spec.eligible_start >= spec.eligible_end:
        raise ValueError(f"invalid archive bounds: {spec.archive_id}")
    if spec.eligible_end > PROTECTED_START:
        raise ValueError(f"archive can expose protected rows: {spec.archive_id}")
    if "google.com" not in spec.source_url:
        raise ValueError(f"non-official archive source: {spec.source_url}")
    if not zipfile.is_zipfile(spec.path):
        raise ValueError(f"invalid ZIP archive: {spec.path}")


def archive_specs(args: argparse.Namespace) -> list[ArchiveSpec]:
    # The official complete archive supplies history through Q2 2025; the
    # separately published Q3 and Q4 increments provide non-overlapping tails.
    specs = [
        ArchiveSpec(
            "official_complete_through_2025q2",
            Path(args.full_archive),
            FULL_ARCHIVE_URL,
            TRAIN_START,
            pd.Timestamp("2025-07-01T00:00:00Z"),
        ),
        ArchiveSpec(
            "official_2025q3_increment",
            Path(args.q3_archive),
            QUARTERLY_FOLDER_URL,
            pd.Timestamp("2025-07-01T00:00:00Z"),
            pd.Timestamp("2025-10-01T00:00:00Z"),
        ),
        ArchiveSpec(
            "official_2025q4_increment",
            Path(args.q4_archive),
            QUARTERLY_FOLDER_URL,
            pd.Timestamp("2025-10-01T00:00:00Z"),
            PROTECTED_START,
        ),
    ]
    for spec in specs:
        validate_archive_spec(spec)
    return specs


def load_asset_pairs(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    if payload.get("error") != [] or not isinstance(payload.get("result"), dict):
        raise ValueError("invalid official AssetPairs response")
    return payload["result"]


def usd_pair_index(asset_pairs: dict[str, dict]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for api_key, row in asset_pairs.items():
        wsname = str(row.get("wsname", ""))
        if not wsname.endswith("/USD") or row.get("quote") != "ZUSD":
            continue
        canonical = wsname.removesuffix("/USD")
        item = {
            "api_key": api_key,
            "Kraken_spot_pair": row.get("altname"),
            "wsname": wsname,
            "spot_base": row.get("base"),
            "spot_quote": row.get("quote"),
            "current_status": row.get("status"),
        }
        index.setdefault(canonical, []).append(item)
    return index


def build_pair_authority(cohort: pd.DataFrame, asset_pairs: dict[str, dict]) -> pd.DataFrame:
    index = usd_pair_index(asset_pairs)
    cohort_rows = cohort[["symbol", "base"]].drop_duplicates().sort_values(["base", "symbol"])
    candidates = [
        {"symbol": "PF_XBTUSD", "base": "XBT", "pilot_anchor": True},
        {"symbol": "PF_ETHUSD", "base": "ETH", "pilot_anchor": True},
    ]
    candidates.extend(
        {"symbol": row.symbol, "base": row.base, "pilot_anchor": False}
        for row in cohort_rows.itertuples(index=False)
    )
    output = []
    for source in candidates:
        base = str(source["base"])
        lookup = "XBT" if base in {"BTC", "XBT"} else base
        matches = index.get(lookup, [])
        common = {
            "canonical_asset_id": lookup,
            "PF_symbol": source["symbol"],
            "mechanism_proof_cohort": not source["pilot_anchor"],
            "pilot_anchor": source["pilot_anchor"],
            "pair_identity_source": "official_Kraken_AssetPairs_snapshot",
            "observed_or_verified_start": "unknown_before_archive_read",
            "observed_or_verified_end": "unknown_before_archive_read",
            "symbol_migrations": "BTC_to_XBT" if base == "BTC" else "none_known",
            "known_status_intervals": "current_status_only_not_historical",
            "historical_authority_status": "pending_archive_observation",
        }
        if len(matches) == 1:
            row = matches[0]
            output.append(
                common
                | row
                | {
                    "identity_confidence": "high_current_identity_historical_interval_pending",
                    "inclusion_or_exclusion_reason": "official_USD_identity_candidate",
                }
            )
        else:
            output.append(
                common
                | {
                    "api_key": "",
                    "Kraken_spot_pair": "",
                    "wsname": "",
                    "spot_base": "",
                    "spot_quote": "",
                    "current_status": "",
                    "identity_confidence": "unresolved",
                    "inclusion_or_exclusion_reason": (
                        "no_unique_official_USD_pair" if not matches else "ambiguous_official_USD_pair"
                    ),
                }
            )
    frame = pd.DataFrame(output).drop_duplicates("PF_symbol", keep="first")
    return frame.sort_values(["canonical_asset_id", "PF_symbol"]).reset_index(drop=True)


def find_member(archive: zipfile.ZipFile, pair: str) -> str | None:
    expected = f"{pair}.csv"
    matches = [name for name in archive.namelist() if not name.endswith("/") and Path(name).name == expected]
    if len(matches) > 1:
        raise ValueError(f"ambiguous archive member for {pair}: {matches}")
    return matches[0] if matches else None


def aggregate_trade_chunks(chunks: Iterable[pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    pieces = []
    rows_seen = 0
    duplicate_tuple_rows = 0
    out_of_order_rows = 0
    previous_ts: float | None = None
    for sequence, chunk in enumerate(chunks):
        if list(chunk.columns) != ["timestamp", "price", "volume"]:
            raise ValueError("unexpected trade schema")
        for column in chunk.columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        if chunk.isna().any().any() or (chunk[["price", "volume"]] <= 0).any().any():
            raise ValueError("invalid trade row")
        rows_seen += len(chunk)
        duplicate_tuple_rows += int(chunk.duplicated(["timestamp", "price", "volume"]).sum())
        values = chunk["timestamp"].to_numpy()
        out_of_order_rows += int((values[1:] < values[:-1]).sum())
        if previous_ts is not None and len(values) and values[0] < previous_ts:
            out_of_order_rows += 1
        if len(values):
            previous_ts = float(values[-1])
        work = chunk.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], unit="s", utc=True)
        work["bar_ts"] = work["timestamp"].dt.floor("5min")
        work["_sequence"] = range(sequence * 10**9, sequence * 10**9 + len(work))
        grouped = work.groupby("bar_ts", sort=True, as_index=False).agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("volume", "sum"),
            trade_count=("price", "size"),
            first_sequence=("_sequence", "min"),
            last_sequence=("_sequence", "max"),
        )
        pieces.append(grouped)
    if not pieces:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "trade_count"]), {
            "trade_rows": 0,
            "duplicate_tuple_rows": 0,
            "out_of_order_rows": 0,
        }
    combined = pd.concat(pieces, ignore_index=True).sort_values(["bar_ts", "first_sequence"])
    final = combined.groupby("bar_ts", sort=True, as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trade_count=("trade_count", "sum"),
    ).rename(columns={"bar_ts": "timestamp"})
    return final, {
        "trade_rows": rows_seen,
        "duplicate_tuple_rows": duplicate_tuple_rows,
        "out_of_order_rows": out_of_order_rows,
    }


def validate_trade_order(stats: dict, archive_id: str, pair: str) -> None:
    if int(stats.get("out_of_order_rows", 0)):
        raise ValueError(f"out-of-order official trades: {archive_id}/{pair}")


def read_pair_archive(spec: ArchiveSpec, pair: str, windows: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] | None) -> tuple[pd.DataFrame, dict]:
    validate_archive_spec(spec)
    with zipfile.ZipFile(spec.path) as archive:
        member = find_member(archive, pair)
        if member is None:
            return pd.DataFrame(), {"member": "", "member_present": False, "trade_rows": 0}
        with archive.open(member) as source:
            reader = pd.read_csv(
                source,
                names=["timestamp", "price", "volume"],
                header=None,
                chunksize=500_000,
            )

            def selected_chunks() -> Iterable[pd.DataFrame]:
                for chunk in reader:
                    timestamps = pd.to_datetime(
                        pd.to_numeric(chunk["timestamp"], errors="coerce"), unit="s", utc=True
                    )
                    mask = timestamps.ge(spec.eligible_start) & timestamps.lt(spec.eligible_end)
                    if windows:
                        window_mask = pd.Series(False, index=chunk.index)
                        for start, end in windows:
                            window_mask |= timestamps.ge(start) & timestamps.lt(end)
                        mask &= window_mask
                    selected = chunk.loc[mask].copy()
                    if not selected.empty:
                        yield selected

            bars, stats = aggregate_trade_chunks(selected_chunks())
        return bars, stats | {"member": member, "member_present": True}


def finalize_bars(parts: list[pd.DataFrame], pair: str) -> pd.DataFrame:
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return pd.DataFrame()
    bars = pd.concat(nonempty, ignore_index=True).sort_values("timestamp")
    if bars["timestamp"].duplicated().any():
        raise ValueError(f"overlapping archive bars for {pair}")
    if (bars["timestamp"] < TRAIN_START).any() or (bars["timestamp"] >= PROTECTED_START).any():
        raise ValueError(f"rankable boundary violation for {pair}")
    bars["source_close_ts"] = bars["timestamp"] + pd.Timedelta(minutes=5)
    bars["feature_available_ts"] = bars["source_close_ts"]
    bars["Kraken_spot_pair"] = pair
    bars["venue"] = "kraken"
    bars["quote"] = "USD"
    return bars.reset_index(drop=True)


def internal_gap_intervals(bars: pd.DataFrame) -> pd.DataFrame:
    columns = ["gap_start_ts", "gap_end_exclusive_ts", "missing_5m_slots"]
    if bars.empty or len(bars) < 2:
        return pd.DataFrame(columns=columns)
    timestamps = bars["timestamp"].sort_values().drop_duplicates().reset_index(drop=True)
    differences = timestamps.diff()
    mask = differences > pd.Timedelta(minutes=5)
    return pd.DataFrame({
        "gap_start_ts": (timestamps.shift(1) + pd.Timedelta(minutes=5))[mask].reset_index(drop=True),
        "gap_end_exclusive_ts": timestamps[mask].reset_index(drop=True),
        "missing_5m_slots": (
            differences[mask].div(pd.Timedelta(minutes=5)).astype("int64") - 1
        ).reset_index(drop=True),
    }, columns=columns)


def coverage_row(
    pair_row: pd.Series,
    bars: pd.DataFrame,
    stats: list[dict],
    output_path: Path | None,
    gap_path: Path | None = None,
) -> dict:
    if bars.empty:
        return {
            "PF_symbol": pair_row.PF_symbol,
            "Kraken_spot_pair": pair_row.Kraken_spot_pair,
            "bar_count": 0,
            "first_usable_ts": "",
            "last_usable_ts": "",
            "expected_5m_slots_between_observed_bounds": 0,
            "missing_5m_slots_between_observed_bounds": 0,
            "trade_rows": sum(int(x.get("trade_rows", 0)) for x in stats),
            "duplicate_trade_tuples_reported_not_removed": sum(int(x.get("duplicate_tuple_rows", 0)) for x in stats),
            "out_of_order_rows": sum(int(x.get("out_of_order_rows", 0)) for x in stats),
            "normalized_path": "",
            "normalized_sha256": "",
            "internal_gap_interval_count": 0,
            "gap_mask_path": "",
            "gap_mask_sha256": "",
            "coverage_status": "no_official_archive_rows_observed",
        }
    first, last = bars.timestamp.min(), bars.timestamp.max()
    expected = int((last - first) / pd.Timedelta(minutes=5)) + 1
    return {
        "PF_symbol": pair_row.PF_symbol,
        "Kraken_spot_pair": pair_row.Kraken_spot_pair,
        "bar_count": len(bars),
        "first_usable_ts": first.isoformat(),
        "last_usable_ts": last.isoformat(),
        "expected_5m_slots_between_observed_bounds": expected,
        "missing_5m_slots_between_observed_bounds": expected - len(bars),
        "trade_rows": sum(int(x.get("trade_rows", 0)) for x in stats),
        "duplicate_trade_tuples_reported_not_removed": sum(int(x.get("duplicate_tuple_rows", 0)) for x in stats),
        "out_of_order_rows": sum(int(x.get("out_of_order_rows", 0)) for x in stats),
        "normalized_path": str(output_path) if output_path else "",
        "normalized_sha256": sha256(output_path) if output_path else "",
        "internal_gap_interval_count": len(internal_gap_intervals(bars)),
        "gap_mask_path": str(gap_path) if gap_path else "",
        "gap_mask_sha256": sha256(gap_path) if gap_path else "",
        "coverage_status": "official_archive_observed_bounded_sparse_5m",
    }


def source_ledger(specs: list[ArchiveSpec], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append({
            "source_id": spec.archive_id,
            "source_type": "official_Kraken_downloadable_time_and_sales",
            "url": spec.source_url,
            "accessed_utc": utc_now(),
            "local_path": str(spec.path),
            "bytes": spec.path.stat().st_size,
            "sha256": sha256(spec.path),
            "eligible_start": spec.eligible_start.isoformat(),
            "eligible_end_exclusive": spec.eligible_end.isoformat(),
            "protected_payload_opened": False,
        })
    for source_id, path, url in [
        ("official_support_article_snapshot", Path(args.support_snapshot), OFFICIAL_SUPPORT_URL),
        ("official_asset_pairs_snapshot", Path(args.assetpairs), OFFICIAL_ASSET_PAIRS_URL),
    ]:
        rows.append({
            "source_id": source_id,
            "source_type": "official_Kraken_metadata_or_documentation",
            "url": url,
            "accessed_utc": utc_now(),
            "local_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "eligible_start": "not_price_payload",
            "eligible_end_exclusive": "not_price_payload",
            "protected_payload_opened": False,
        })
    return pd.DataFrame(rows)


def validate_pilot_freeze(path: Path, authority: pd.DataFrame) -> None:
    freeze = json.loads(path.read_text())
    if not freeze.get("frozen_before_price_read"):
        raise ValueError("pilot freeze is not active")
    symbols = tuple(item["PF_symbol"] for item in freeze["pairs"])
    if symbols != PILOT_PF:
        raise ValueError(f"pilot identity mismatch: {symbols}")
    mapped = set(authority.loc[authority.Kraken_spot_pair.ne(""), "PF_symbol"])
    if not set(PILOT_PF).issubset(mapped):
        raise ValueError("frozen pilot lacks deterministic spot mapping")


def run(mode: str, args: argparse.Namespace) -> None:
    task_root = Path(args.task_root)
    task_root.mkdir(parents=True, exist_ok=True)
    specs = archive_specs(args)
    cohort = pd.read_csv(args.cohort)
    pairs = build_pair_authority(cohort, load_asset_pairs(Path(args.assetpairs)))
    validate_pilot_freeze(Path(args.pilot_freeze), pairs)
    source_ledger(specs, args).to_csv(task_root / "C02_SPOT_SOURCE_LEDGER.csv", index=False)
    pairs.to_csv(task_root / "C02_SPOT_PAIR_AUTHORITY.csv", index=False)

    if mode == "full":
        review = Path(args.pilot_review)
        if not review.is_file() or "Decision: `approve`" not in review.read_text():
            raise RuntimeError("full acquisition requires approved independent pilot review")

    selected = pairs[pairs.PF_symbol.isin(PILOT_PF)].copy() if mode == "pilot" else pairs[
        pairs.mechanism_proof_cohort & pairs.Kraken_spot_pair.ne("")
    ].copy()
    output_root = Path(args.data_root) / ("pilot_5m" if mode == "pilot" else "normalized_5m")
    coverage = []
    selected = selected.sort_values(["canonical_asset_id", "PF_symbol"]).reset_index(drop=True)
    for position, pair_row in enumerate(selected.itertuples(index=False), start=1):
        write_json(task_root / "PROGRESS.json", {
            "mode": mode,
            "status": "processing",
            "pairs_completed": position - 1,
            "pairs_total": len(selected),
            "current_PF_symbol": pair_row.PF_symbol,
            "current_Kraken_spot_pair": pair_row.Kraken_spot_pair,
            "updated_utc": utc_now(),
        })
        print(f"[{mode}] {position}/{len(selected)} {pair_row.PF_symbol} -> {pair_row.Kraken_spot_pair}", flush=True)
        parts, stats = [], []
        for spec in specs:
            bars, part_stats = read_pair_archive(
                spec,
                pair_row.Kraken_spot_pair,
                PILOT_WINDOWS if mode == "pilot" else None,
            )
            validate_trade_order(part_stats, spec.archive_id, pair_row.Kraken_spot_pair)
            parts.append(bars)
            stats.append({"archive_id": spec.archive_id} | part_stats)
        bars = finalize_bars(parts, pair_row.Kraken_spot_pair)
        output_path = None
        gap_path = None
        if not bars.empty:
            output_path = output_root / f"{pair_row.Kraken_spot_pair}.parquet"
            if output_path.exists():
                raise FileExistsError(f"refusing overwrite: {output_path}")
            atomic_parquet(bars, output_path)
            if mode == "full":
                gap_path = Path(args.data_root) / "gap_masks" / f"{pair_row.Kraken_spot_pair}.parquet"
                if gap_path.exists():
                    raise FileExistsError(f"refusing overwrite: {gap_path}")
                atomic_parquet(internal_gap_intervals(bars), gap_path)
        coverage.append(coverage_row(pd.Series(pair_row._asdict()), bars, stats, output_path, gap_path))

    coverage_frame = pd.DataFrame(coverage).sort_values(["PF_symbol"])
    coverage_frame.to_csv(task_root / "C02_SPOT_COVERAGE_MATRIX.csv", index=False)
    if mode == "pilot":
        expected_cells = len(PILOT_PF) * len(PILOT_WINDOWS)
        actual_cells = 0
        for row in coverage:
            path = Path(row["normalized_path"]) if row["normalized_path"] else None
            if path:
                bars = pd.read_parquet(path, columns=["timestamp"])
                for start, end in PILOT_WINDOWS:
                    actual_cells += int(bars.timestamp.between(start, end, inclusive="left").any())
        status = "pass" if actual_cells == expected_cells else "fail"
        (task_root / "C02_SPOT_PILOT_VALIDATION.md").write_text(
            "# C02 Spot Pilot Validation\n\n"
            f"Status: `{status}`.\n\n"
            f"Frozen pairs: `{list(PILOT_PF)}`.\n\n"
            f"Required non-empty pair/windows: `{expected_cells}`; observed: `{actual_cells}`.\n\n"
            "Source identity: official Kraken USD spot time-and-sales. Timestamps are UTC Unix seconds; "
            "price is executed spot price; volume is base-asset trade amount. Sparse 5-minute bars are "
            "constructed deterministically without gap filling. No row at or after 2026-01-01 is admitted.\n"
        )
        if status != "pass":
            raise RuntimeError("pilot coverage validation failed")
    else:
        observed = coverage_frame[coverage_frame.bar_count.gt(0)]
        pairs.loc[pairs.PF_symbol.isin(observed.PF_symbol), "historical_authority_status"] = "official_archive_observed_bounded"
        pairs.loc[pairs.PF_symbol.isin(observed.PF_symbol), "inclusion_or_exclusion_reason"] = "included_official_USD_archive_rows"
        attempted_without_rows = set(coverage_frame.loc[coverage_frame.bar_count.eq(0), "PF_symbol"])
        pairs.loc[pairs.PF_symbol.isin(attempted_without_rows), "historical_authority_status"] = "no_archive_rows_observed"
        pairs.loc[pairs.PF_symbol.isin(attempted_without_rows), "inclusion_or_exclusion_reason"] = "excluded_no_official_archive_rows_observed"
        firsts = observed.set_index("PF_symbol").first_usable_ts
        lasts = observed.set_index("PF_symbol").last_usable_ts
        pairs["observed_or_verified_start"] = pairs.PF_symbol.map(firsts).fillna(pairs.observed_or_verified_start)
        pairs["observed_or_verified_end"] = pairs.PF_symbol.map(lasts).fillna(pairs.observed_or_verified_end)
        pairs.to_csv(task_root / "C02_SPOT_PAIR_AUTHORITY.csv", index=False)

    manifest = {
        "task_id": "donch_bt_stage_3a_c02_spot_reference_20260717_v1",
        "mode": mode,
        "created_utc": utc_now(),
        "train_interval": [TRAIN_START.isoformat(), PROTECTED_START.isoformat()],
        "panel_name": "Kraken_USD_spot_bar_existence_panel",
        "source_ledger_sha256": sha256(task_root / "C02_SPOT_SOURCE_LEDGER.csv"),
        "pair_authority_sha256": sha256(task_root / "C02_SPOT_PAIR_AUTHORITY.csv"),
        "coverage_matrix_sha256": sha256(task_root / "C02_SPOT_COVERAGE_MATRIX.csv"),
        "normalized_files": [
            {"path": row["normalized_path"], "sha256": row["normalized_sha256"], "bars": int(row["bar_count"])}
            for row in coverage if row["normalized_path"]
        ],
        "gap_mask_files": [
            {"path": row["gap_mask_path"], "sha256": row["gap_mask_sha256"], "intervals": int(row["internal_gap_interval_count"])}
            for row in coverage if row["gap_mask_path"]
        ],
        "protected_rows_opened": 0,
        "economic_outputs_computed": 0,
    }
    manifest["manifest_content_hash"] = canonical_hash(manifest)
    write_json(task_root / "C02_SPOT_DATA_MANIFEST.json", manifest)
    write_json(task_root / "PROGRESS.json", {
        "mode": mode,
        "status": "complete",
        "pairs_completed": len(selected),
        "pairs_total": len(selected),
        "updated_utc": utc_now(),
    })


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["pilot", "full"], required=True)
    p.add_argument("--task-root", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--assetpairs", required=True)
    p.add_argument("--support-snapshot", required=True)
    p.add_argument("--pilot-freeze", required=True)
    p.add_argument("--pilot-review", required=True)
    p.add_argument("--full-archive", required=True)
    p.add_argument("--q3-archive", required=True)
    p.add_argument("--q4-archive", required=True)
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    run(arguments.mode, arguments)
