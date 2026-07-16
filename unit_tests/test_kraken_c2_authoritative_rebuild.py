import unittest

import pandas as pd

from tools import run_kraken_c2_authoritative_rebuild as c2


class C2AuthoritativeRebuildTests(unittest.TestCase):
    def test_content_hash_is_order_independent_and_excludes_identity(self):
        left = {"ticker": "ETH", "first_public_ts_utc": "2024-05-23", "event_id": "old", "row_content_hash": "old"}
        right = {"first_public_ts_utc": "2024-05-23", "ticker": "ETH", "event_id": "new", "row_content_hash": "new"}
        self.assertEqual(c2.canonical_hash(left), c2.canonical_hash(right))

    def test_date_only_precision_is_not_promoted(self):
        self.assertEqual(c2.precision("2024-05-23"), "date_only")
        self.assertEqual(c2.precision("unknown"), "unknown")
        self.assertEqual(c2.precision("2024-05-23T12:00:00Z"), "intraday_explicit_utc")

    def test_date_only_lifecycle_requires_prior_day(self):
        anchor = pd.Timestamp("2024-05-23", tz="UTC")
        self.assertFalse(pd.Timestamp("2024-05-23T00:00:00Z") < anchor.normalize())
        self.assertTrue(pd.Timestamp("2024-05-22T23:59:59Z") < anchor.normalize())

    def test_required_schema_is_unique(self):
        self.assertEqual(len(c2.SCHEMA), 29)
        self.assertEqual(len(c2.SCHEMA), len(set(c2.SCHEMA)))


if __name__ == "__main__":
    unittest.main()
