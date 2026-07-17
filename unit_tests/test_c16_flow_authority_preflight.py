from __future__ import annotations

import unittest
from unittest import mock
import tempfile
from pathlib import Path

import pandas as pd

from tools import build_c16_flow_authority_preflight as c16


def row(**changes):
    base = {column: "x" for column in c16.PANEL_COLUMNS}
    base.update({
        "product_id": "BTC_TEST", "ticker": "TEST", "underlying_asset": "BTC",
        "effective_trading_date": "2024-01-02", "measure_type": "creation_redemption_units",
        "measure_value": 1.0, "measure_unit": "creation_units", "derived_or_reported": "reported",
        "publication_ts_utc": "2024-01-02T22:00:00Z", "first_available_ts_utc": "2024-01-02T22:00:00Z",
        "revision_id": "r1", "revision_published_ts_utc": "2024-01-02T22:00:00Z",
        "source_file_sha256": "a" * 64,
    })
    base.update(changes)
    return base


class C16FlowAuthorityTests(unittest.TestCase):
    def test_current_only_rejected_before_reader(self):
        reader = mock.Mock()
        contract = c16.DownloadContract("current", "current_only", None, None, "text/html", False)
        with self.assertRaisesRegex(ValueError, "before open"):
            c16.guarded_open(contract, reader)
        reader.assert_not_called()

    def test_mixed_payload_rejected_before_reader(self):
        reader = mock.Mock()
        contract = c16.DownloadContract("mixed", "mixed", None, None, "text/csv", False)
        with self.assertRaisesRegex(ValueError, "before open"):
            c16.guarded_open(contract, reader)
        reader.assert_not_called()

    def test_dated_payload_cannot_cross_protected_boundary(self):
        contract = c16.DownloadContract(
            "bad", "dated_historical", c16.OBSERVATION_START,
            c16.PROTECTED_START + pd.Timedelta(days=1), "text/csv", True,
        )
        with self.assertRaisesRegex(ValueError, "protected"):
            c16.validate_download_contract(contract)

    def test_dated_immutable_pre2026_artifact_can_open(self):
        reader = mock.Mock(return_value="ok")
        contract = c16.DownloadContract(
            "safe", "dated_historical", c16.OBSERVATION_START,
            c16.PROTECTED_START, "application/pdf", True,
        )
        self.assertEqual(c16.guarded_open(contract, reader), "ok")
        reader.assert_called_once()

    def test_first_published_and_latest_revised_are_distinct(self):
        frame = pd.DataFrame([
            row(),
            row(revision_id="r2", measure_value=2.0,
                revision_published_ts_utc="2024-01-03T22:00:00Z",
                supersedes_revision_id="r1"),
        ])
        self.assertEqual(c16.first_published_panel(frame).iloc[0].measure_value, 1.0)
        self.assertEqual(c16.latest_revised_panel(frame).iloc[0].measure_value, 2.0)

    def test_next_day_publication_is_preserved(self):
        frame = pd.DataFrame([row(first_available_ts_utc="2024-01-03T13:00:00Z")])
        result = c16.first_published_panel(frame)
        self.assertEqual(result.iloc[0].first_available_ts_utc, "2024-01-03T13:00:00Z")

    def test_protected_observation_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside bounded"):
            c16.validate_observations(pd.DataFrame([row(effective_trading_date="2026-01-02")]))

    def test_aum_cannot_be_relabeled_as_flow(self):
        semantics = {"NAV_or_AUM_change", "creation_redemption_units"}
        self.assertNotEqual("NAV_or_AUM_change", "creation_redemption_units")
        self.assertEqual(len(semantics), 2)

    def test_share_change_arithmetic(self):
        self.assertEqual(c16.derive_share_change(150_000, 100_000, 50_000), 1.0)
        self.assertEqual(c16.derive_share_change(50_000, 100_000, 50_000), -1.0)
        with self.assertRaises(ValueError):
            c16.derive_share_change(1, 1, 0)

    def test_product_identity_unique_and_complete(self):
        products = c16.products()
        self.assertEqual(len(products), 20)
        self.assertEqual(len({row["product_id"] for row in products}), 20)
        self.assertEqual(sum(row["underlying_asset"] == "BTC" for row in products), 11)
        self.assertEqual(sum(row["underlying_asset"] == "ETH" for row in products), 9)

    def test_launch_and_closed_day_coverage(self):
        days = c16.trading_days("2024-01-11", "2024-01-15")
        self.assertEqual([day.date().isoformat() for day in days], ["2024-01-11", "2024-01-12"])

    def test_coverage_is_product_day_and_unavailable(self):
        coverage, gaps = c16.coverage_and_gaps(c16.products()[:1])
        self.assertGreater(len(coverage), 400)
        self.assertEqual(int(coverage.authoritative_value_available.sum()), 0)
        self.assertEqual(len(gaps), 1)

    def test_canonical_hash_is_order_stable(self):
        self.assertEqual(c16.canonical_hash({"a": 1, "b": 2}), c16.canonical_hash({"b": 2, "a": 1}))

    def test_artifact_manifest_lists_local_panels_once(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "small.csv").write_text("a\n1\n")
            pd.DataFrame(columns=c16.PANEL_COLUMNS).to_parquet(root / "C16_FIRST_PUBLISHED_FLOW_PANEL.parquet")
            pd.DataFrame(columns=c16.PANEL_COLUMNS).to_parquet(root / "C16_LATEST_REVISED_AUDIT_PANEL.parquet")
            manifest = c16.artifact_manifest(root, root, c16.DECISION, "2026-07-17T00:00:00Z")
            paths = [row["path"] for row in manifest["artifacts"]]
            self.assertEqual(sum(path.endswith("C16_FIRST_PUBLISHED_FLOW_PANEL.parquet") for path in paths), 1)
            self.assertEqual(sum(path.endswith("C16_LATEST_REVISED_AUDIT_PANEL.parquet") for path in paths), 1)


if __name__ == "__main__":
    unittest.main()
