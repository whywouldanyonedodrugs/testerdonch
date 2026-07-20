#!/usr/bin/env python3
"""Build rankable-only funding calibration, unit audit, and frozen contracts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.qlmg_stage19_funding import equal_symbol_weighted_quantile, type7


UTC = timezone.utc
MINIMUM_ROWS = 720


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


def read_campaign(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["PF_symbol"] for row in csv.DictReader(handle) if row["final_campaign_eligible"] == "True"}


def read_instruments(path: Path) -> dict[str, dict[str, Any]]:
    columns = ["symbol", "type", "contractSize", "base", "quote", "openingDate", "source_url"]
    return {row["symbol"]: row for row in pq.read_table(path, columns=columns).to_pylist()}


def read_rankable_distributions(
    rankable_zip: Path, compatible_symbols: set[str], campaign_symbols: set[str]
) -> tuple[dict[str, list[Decimal]], dict[str, list[tuple[datetime, Decimal, Decimal]]]]:
    distributions: dict[str, list[Decimal]] = {}
    campaign_rows: dict[str, list[tuple[datetime, Decimal, Decimal]]] = {}
    with zipfile.ZipFile(rankable_zip) as archive:
        for name in sorted(archive.namelist()):
            symbol = Path(name).stem
            if symbol not in compatible_symbols:
                continue
            values: list[Decimal] = []
            detailed: list[tuple[datetime, Decimal, Decimal]] = []
            with archive.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="ascii", newline=""))
                for row in reader:
                    stamp = parse_ts(row["timestamp"])
                    if not datetime(2023, 1, 1, tzinfo=UTC) <= stamp < datetime(2026, 1, 1, tzinfo=UTC):
                        raise RuntimeError(f"rankable package boundary violation: {symbol}")
                    absolute = Decimal(row["absolute_rate"])
                    relative = Decimal(row["relative_rate"])
                    values.append(abs(relative) * Decimal(10000))
                    if symbol in campaign_symbols:
                        detailed.append((stamp, absolute, relative))
            if values:
                distributions[symbol] = sorted(values)
            if symbol in campaign_symbols:
                campaign_rows[symbol] = detailed
    return distributions, campaign_rows


def load_price_chunk_index(manifest: Path) -> dict[str, list[dict[str, str]]]:
    pairs: dict[tuple[str, str, str], dict[str, str]] = {}
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] not in {"historical_trade_candles_5m", "historical_mark_candles_5m"}:
                continue
            if row["status"] != "downloaded" or row["rankable_pre_holdout"] != "True" or row["contains_protected_period"] != "False":
                continue
            if int(row["rows"] or 0) <= 0:
                continue
            key = (row["symbol"], row["chunk_start"], row["chunk_end"])
            pairs.setdefault(key, {})[row["dataset"]] = row["parquet_path"]
    output: dict[str, list[dict[str, str]]] = {}
    for (symbol, start, end), paths in pairs.items():
        if len(paths) == 2:
            output.setdefault(symbol, []).append({"start": start, "end": end, **paths})
    for rows in output.values():
        rows.sort(key=lambda row: row["start"])
    return output


def price_open(path: Path, stamp: datetime) -> Decimal:
    table = pq.read_table(path, columns=["time", "open", "rankable_pre_holdout", "contains_protected_period"])
    target = int(stamp.timestamp() * 1000)
    matches = [row for row in table.to_pylist() if row["time"] == target]
    if len(matches) != 1 or not matches[0]["rankable_pre_holdout"] or matches[0]["contains_protected_period"]:
        raise RuntimeError(f"missing or unsafe exact contemporaneous price: {path}:{stamp.isoformat()}")
    return Decimal(matches[0]["open"])


def unit_audit(
    campaign: set[str], instruments: dict[str, dict[str, Any]],
    campaign_rows: dict[str, list[tuple[datetime, Decimal, Decimal]]],
    price_chunks: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    output = []
    for symbol in sorted(campaign):
        instrument = instruments.get(symbol)
        if not instrument or instrument["type"] != "flexible_futures" or instrument["contractSize"] != 1 or instrument["quote"] != "USD" or not instrument["base"]:
            raise RuntimeError(f"unresolved official PF contract/base unit: {symbol}")
        selected = None
        for chunk in price_chunks.get(symbol, []):
            start = datetime.fromisoformat(chunk["start"])
            end = datetime.fromisoformat(chunk["end"])
            for stamp, absolute, relative in campaign_rows.get(symbol, []):
                if start <= stamp < end and relative != 0:
                    try:
                        trade = price_open(Path(chunk["historical_trade_candles_5m"]), stamp)
                        mark = price_open(Path(chunk["historical_mark_candles_5m"]), stamp)
                    except Exception:
                        continue
                    selected = (chunk, stamp, absolute, relative, trade, mark)
                    break
            if selected:
                break
        if selected is None:
            raise RuntimeError(f"no contemporaneous rankable unit anchor: {symbol}")
        chunk, stamp, absolute, relative, trade, mark = selected
        implied = absolute / relative
        trade_error = abs(implied / trade - 1)
        mark_error = abs(implied / mark - 1)
        # Mark is the funding reference sanity anchor; trade is a deliberately
        # wider execution sanity bound for sparse/launch-period PF markets.
        if implied <= 0 or trade <= 0 or mark <= 0 or trade_error > Decimal("0.25") or mark_error > Decimal("0.10"):
            raise RuntimeError(f"absolute/relative unit sanity mismatch: {symbol}")
        output.append({
            "symbol": symbol, "instrument_type": instrument["type"], "contract_size": instrument["contractSize"],
            "base_unit": instrument["base"], "quote_unit": instrument["quote"],
            "anchor_timestamp_utc": stamp.isoformat().replace("+00:00", "Z"),
            "implied_price_absolute_over_relative": str(implied), "trade_open": str(trade), "mark_open": str(mark),
            "trade_relative_error": str(trade_error), "mark_relative_error": str(mark_error),
            "trade_error_limit": "0.25", "mark_error_limit": "0.10",
            "unit_status": "verified_one_contract_unit_equals_one_base_unit_no_hidden_multiplier",
        })
    return output


def build(
    rankable_zip: Path, output: Path, universe_csv: Path, instruments_path: Path,
    acquisition_manifest: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    campaign = read_campaign(universe_csv)
    instruments = read_instruments(instruments_path)
    compatible = {
        symbol for symbol, row in instruments.items()
        if symbol.startswith("PF_") and row["type"] == "flexible_futures" and row["contractSize"] == 1 and row["quote"] == "USD" and row["base"]
    }
    distributions, campaign_rows = read_rankable_distributions(rankable_zip, compatible, campaign)
    eligible = {symbol: values for symbol, values in distributions.items() if len(values) >= MINIMUM_ROWS}
    if not campaign <= distributions.keys() or not eligible:
        raise RuntimeError("incomplete campaign funding calibration coverage")
    pooled_q95 = equal_symbol_weighted_quantile(eligible, Decimal("0.95"))
    pooled_q99 = equal_symbol_weighted_quantile(eligible, Decimal("0.99"))
    allowances = []
    for symbol in sorted(campaign):
        values = distributions[symbol]
        if len(values) >= MINIMUM_ROWS:
            base, stress, source = type7(values, Decimal("0.95")), type7(values, Decimal("0.99")), "symbol_type7_rankable_only"
        else:
            base, stress, source = pooled_q95, pooled_q99, "equal_symbol_weighted_pooled_rankable_fallback"
        if base < 0 or stress < base:
            raise RuntimeError(f"invalid gap allowances: {symbol}")
        allowances.append({
            "symbol": symbol, "rankable_observations": len(values),
            "base_gap_allowance_bps_per_hour": str(base), "stress_gap_allowance_bps_per_hour": str(stress),
            "allowance_source": source,
        })
    audit = unit_audit(campaign, instruments, campaign_rows, load_price_chunk_index(acquisition_manifest))

    def write_csv(name: str, rows: list[dict[str, Any]]) -> str:
        path = output / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        return sha256_file(path)

    allowance_hash = write_csv("FUNDING_GAP_ALLOWANCE_TABLE.csv", allowances)
    unit_hash = write_csv("ABSOLUTE_RATE_UNIT_VERIFICATION.csv", audit)
    dual_contract = {
        "version": "stage19_v1", "row_timestamp_timezone": "UTC_hour_boundary",
        "unresolved_timestamp_semantics": True,
        "alignments": {"alignment_start": "row t applies to [t,t+1h)", "alignment_end": "row t applies to [t-1h,t)"},
        "cashflow_bps": "-position_sign * absolute_rate_usd_per_contract_unit_per_hour * overlap_seconds/3600 / entry_trade_open_usd_per_contract_unit * 10000",
        "selection_funding_bps": "min(0,signed_alignment_start_bps,signed_alignment_end_bps)",
        "favourable_credit_for_selection": False, "partial_hour_accrual": True,
        "gap_cost": "-allowance_bps_per_hour * missing_overlap_hours",
        "runtime_alignment_choice": False,
    }
    (output / "FUNDING_PERIOD_DUAL_ALIGNMENT_CONTRACT.json").write_bytes(canonical_bytes(dual_contract))
    gap_contract = {
        "version": "stage19_v1", "source_package_sha256": sha256_file(rankable_zip),
        "rankable_interval": "[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)",
        "measure": "abs(relative_rate)*10000 bps/hour", "quantile": "Hyndman-Fan type 7 Decimal linear interpolation",
        "minimum_symbol_observations": MINIMUM_ROWS,
        "fallback": "equal-symbol-weighted empirical mixture among official unit-compatible linear PF symbols; linear inverse-CDF interpolation",
        "eligible_pooled_symbols": len(eligible), "pooled_q95": str(pooled_q95), "pooled_q99": str(pooled_q99),
        "candidate_fold_context_or_return_conditioning": False, "protected_rows": 0,
        "table_sha256": allowance_hash,
    }
    (output / "FUNDING_GAP_ALLOWANCE_CONTRACT.json").write_bytes(canonical_bytes(gap_contract))
    result = {
        "campaign_symbols": len(campaign), "unit_verified": len(audit), "unit_audit_sha256": unit_hash,
        "unit_compatible_pool_symbols": len(eligible), "allowance_table_sha256": allowance_hash,
        "dual_alignment_contract_sha256": sha256_file(output / "FUNDING_PERIOD_DUAL_ALIGNMENT_CONTRACT.json"),
        "gap_contract_sha256": sha256_file(output / "FUNDING_GAP_ALLOWANCE_CONTRACT.json"),
        "rankable_package_sha256": sha256_file(rankable_zip), "protected_rows_used": 0,
    }
    (output / "FUNDING_AUTHORITY_BUILD_RESULT.json").write_bytes(canonical_bytes(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankable-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--universe-csv", type=Path, required=True)
    parser.add_argument("--instruments", type=Path, required=True)
    parser.add_argument("--acquisition-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.rankable_zip, args.output, args.universe_csv, args.instruments, args.acquisition_manifest), sort_keys=True))
