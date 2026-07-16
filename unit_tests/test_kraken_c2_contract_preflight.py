from __future__ import annotations

import unittest

from tools import run_kraken_c2_contract_preflight as c2


class C2ContractPreflightTests(unittest.TestCase):
    def test_timestamp_precision_preserves_unknown_and_date_only(self):
        self.assertEqual(c2.precision("unknown"), "unknown")
        self.assertEqual(c2.precision("2024-01-10"), "date_only")
        self.assertEqual(c2.precision("2024-01-10T12:30:00Z"), "intraday_explicit_utc")

    def test_mechanism_normalization_is_predeclared(self):
        self.assertEqual(c2.family("unlock_vesting_change"), "supply_float_changes")
        self.assertEqual(c2.family("leverage_access_expansion"), "leverage_access_expansion")
        self.assertEqual(c2.family("unmapped_attention"), "attention_only_events")


if __name__ == "__main__":
    unittest.main()
