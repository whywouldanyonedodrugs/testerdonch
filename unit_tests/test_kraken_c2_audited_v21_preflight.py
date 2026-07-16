import unittest
from pathlib import Path

import pandas as pd

from tools import run_kraken_c2_audited_v21_preflight as c2


class C2AuditedV21PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = c2.parse_records(Path("research_inputs/catdb.md").read_text())

    def test_parser_recovers_exact_record_schema(self):
        self.assertEqual(len(self.records), 98)
        self.assertTrue(self.records.parser_missing_fields.eq("").all())
        self.assertTrue(self.records.parser_extra_fields.eq("").all())

    def test_confidence_counts(self):
        self.assertEqual(self.records["Final inclusion status"].value_counts().to_dict(), {"high": 59, "medium": 27, "excluded": 12})

    def test_date_only_actionability_moves_to_next_daily_boundary(self):
        row = self.records[self.records["Timestamp precision"].eq("date_only")].iloc[0]
        anchor = c2.resolve_anchor(row)
        if pd.notna(anchor["event_anchor_ts"]):
            self.assertGreaterEqual(anchor["actionable_not_before_ts"], anchor["event_anchor_ts"].normalize() + pd.Timedelta(days=1))

    def test_basket_expands_only_explicit_assets(self):
        row = pd.Series({"Ticker": "BTC|ETH|SOL|USDC|+", "Asset ID": "btc_eth_sol_usdc_plus_basket"})
        tickers, cap = c2.exposure_tickers(row)
        self.assertEqual(tickers, ["BTC", "ETH", "SOL", "USDC"])
        self.assertEqual(cap, "basket_scope_incomplete_cap")

    def test_alias_does_not_become_two_exposures(self):
        row = pd.Series({"Ticker": "RNDR/RENDER", "Asset ID": "render"})
        self.assertEqual(c2.exposure_tickers(row)[0], ["RNDR"])


if __name__ == "__main__":
    unittest.main()
