from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tools.kraken_protected_safe_funding import (
    MIXED,
    PROTECTED,
    SAFE,
    UNKNOWN,
    GuardedRowGroupReader,
    SourceAuthority,
    adverse_allowance_cost_bps,
    build_rankable_package,
    calibrate_allowances,
    exact_funding_cashflow_bps,
    inspect_source_file,
    overlap_fraction,
    selection_funding_metrics,
    sha256_file,
    validate_campaign_funding_source,
)


def write_funding(path: Path, timestamps: list[str], *, statistics: bool = True) -> None:
    table = pa.table({
        "timestamp": pa.array(pd.to_datetime(timestamps, utc=True)),
        "fundingRate": pa.array(np.arange(1, len(timestamps) + 1, dtype=float)),
        "relativeFundingRate": pa.array(np.zeros(len(timestamps))),
    })
    pq.write_table(table, path, write_statistics=statistics)


def authority(path: Path) -> SourceAuthority:
    return SourceAuthority(
        platform="kraken_derivatives",
        purpose="rankable_exact_funding",
        source_manifest_path="synthetic.csv",
        source_manifest_sha256="synthetic",
        file_path=str(path),
        file_sha256=sha256_file(path),
    )


class ProtectedSafeFundingTests(unittest.TestCase):
    def test_footer_classification_and_only_safe_payload_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = {
                SAFE: ["2025-12-31T22:00:00Z", "2025-12-31T23:00:00Z"],
                PROTECTED: ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
                MIXED: ["2025-12-31T23:00:00Z", "2026-01-01T00:00:00Z"],
            }
            rows = []
            paths = {}
            for label, stamps in cases.items():
                path = root / f"{label}.parquet"
                write_funding(path, stamps)
                _, current = inspect_source_file(authority(path))
                self.assertEqual(current[0]["classification"], label)
                rows.extend(current)
                paths[label] = path
            unknown_path = root / "unknown.parquet"
            write_funding(unknown_path, cases[SAFE], statistics=False)
            _, current = inspect_source_file(authority(unknown_path))
            self.assertEqual(current[0]["classification"], UNKNOWN)
            rows.extend(current)
            paths[UNKNOWN] = unknown_path

            reader = GuardedRowGroupReader(rows)
            frame = reader.read(paths[SAFE], 0)
            self.assertEqual(len(frame), 2)
            for label in (PROTECTED, MIXED, UNKNOWN):
                with self.assertRaisesRegex(RuntimeError, "before deserialization"):
                    reader.read(paths[label], 0)
            self.assertEqual(len(reader.requests), 1)

    def test_corrupt_footer_and_hash_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.parquet"
            path.write_bytes(b"not parquet")
            with self.assertRaisesRegex(RuntimeError, "invalid Parquet footer"):
                inspect_source_file(authority(path))
            good = Path(tmp) / "good.parquet"
            write_funding(good, ["2025-01-01T00:00:00Z"])
            bad_authority = SourceAuthority(**{**authority(good).__dict__, "file_sha256": "0" * 64})
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                inspect_source_file(bad_authority)

    def test_payload_contradiction_globally_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "safe.parquet"
            write_funding(path, ["2025-12-31T23:00:00Z"])
            _, rows = inspect_source_file(authority(path))

            def false_payload(index, columns):
                return pa.table({
                    "timestamp": pa.array(pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True)),
                    "fundingRate": pa.array([1.0]),
                }).select(columns)

            reader = GuardedRowGroupReader(rows, read_method=false_payload)
            with self.assertRaisesRegex(RuntimeError, "contradicts"):
                reader.read(path, 0)
            self.assertEqual(reader.requests[-1]["status"], "payload_footer_contradiction_global_stop")

    def test_absolute_funding_arithmetic_and_boundaries(self):
        self.assertEqual(overlap_fraction("2025-01-01T00:30:00Z", "2025-01-01T01:00:00Z", "2025-01-01T00:00:00Z"), 0.5)
        self.assertEqual(overlap_fraction("2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z", "2025-01-01T00:00:00Z"), 1.0)
        self.assertEqual(overlap_fraction("2025-01-01T01:00:00Z", "2025-01-01T02:00:00Z", "2025-01-01T00:00:00Z"), 0.0)
        self.assertAlmostEqual(exact_funding_cashflow_bps(1, 10.0, 0.5, 20000.0), -2.5)
        self.assertAlmostEqual(exact_funding_cashflow_bps(-1, 10.0, 0.5, 20000.0), 2.5)
        self.assertAlmostEqual(exact_funding_cashflow_bps(1, -10.0, 1.0, 20000.0), 5.0)
        self.assertEqual(adverse_allowance_cost_bps(3.0, 0.5), -1.5)

    def test_type7_allowance_fallback_and_no_favourable_credit(self):
        frame = pd.DataFrame({
            "symbol": ["A"] * 5 + ["B"] * 5 + ["C"],
            "absolute_hourly_funding_bps_on_mark_notional": [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 100],
        })
        table = calibrate_allowances(frame, minimum_rows=5).set_index("symbol")
        self.assertAlmostEqual(table.loc["A", "base_adverse_allowance_bps_per_hour"], np.quantile([0, 1, 2, 3, 4], 0.95, method="linear"))
        self.assertEqual(table.loc["C", "allowance_source"], "equal_symbol_weighted_eligible_symbol_quantile_fallback")
        self.assertTrue((table["stress_adverse_allowance_bps_per_hour"] >= table["base_adverse_allowance_bps_per_hour"]).all())
        with self.assertRaises(ValueError):
            adverse_allowance_cost_bps(-1.0, 1.0)

    def test_exact_diagnostic_cannot_change_selection_metric(self):
        first = selection_funding_metrics(40.0, [0.5, 1.0], 2.0, 4.0)
        second = selection_funding_metrics(40.0, [0.5, 1.0], 2.0, 4.0)
        self.assertEqual(first, second)
        self.assertNotIn("exact", " ".join(first))

    def test_builder_records_zero_payload_when_all_groups_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "mixed.parquet"
            write_funding(parquet, ["2025-12-31T23:00:00Z", "2026-01-01T00:00:00Z"])
            manifest = root / "manifest.csv"
            pd.DataFrame([{
                "dataset": "funding", "symbol": "PF_TESTUSD", "parquet_path": str(parquet),
                "parquet_sha256": sha256_file(parquet), "status": "downloaded",
            }]).to_csv(manifest, index=False)
            output = root / "output"
            result = build_rankable_package(manifest, output)
            self.assertEqual(result["status"], "blocked_no_safe_rankable_absolute_funding_row_groups")
            audit = json.loads((output / "PROTECTED_AUDIT.json").read_text())
            self.assertEqual(audit["payload_row_groups_read"], 0)
            self.assertFalse((output / "RANKABLE_EXACT_FUNDING.parquet").exists())
            with self.assertRaisesRegex(RuntimeError, "not a completed"):
                validate_campaign_funding_source(output)

    def test_source_contains_no_broad_parquet_payload_read(self):
        source = Path("tools/kraken_protected_safe_funding.py").read_text(encoding="utf-8")
        self.assertNotIn("read_parquet(", source)
        self.assertNotIn("read_table(", source)


if __name__ == "__main__":
    unittest.main()
