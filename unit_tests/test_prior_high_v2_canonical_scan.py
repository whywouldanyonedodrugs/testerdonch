from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from tools import run_kraken_family_engine_aggregate_first_sweep as runner
from tools.run_kraken_prior_high_v2_canonical_scan import (
    AUTHORITATIVE_MANIFEST,
    selected_key_policy_hash,
)


class PriorHighV2CanonicalScanTests(unittest.TestCase):
    def test_canonical_hash_is_order_independent_and_excludes_exit_cost_fields(self) -> None:
        row = pd.read_csv(AUTHORITATIVE_MANIFEST).iloc[0].to_dict()
        reordered = dict(reversed(list(row.items())))
        reordered.update({"exit_template": "different_exit", "fees_R": 999, "parameter_vector_hash": "old"})
        self.assertEqual(selected_key_policy_hash(row), selected_key_policy_hash(reordered))

    def test_parent_gate_uses_configured_full_window_for_cache_identity(self) -> None:
        candidate = {
            "parent_regime_gate": "btc_eth_trend_up",
            "run_start_ts": "2024-01-01T00:00:00Z",
            "run_end_ts": "2025-12-31T23:59:59Z",
        }
        bars = pd.DataFrame({"ts": pd.to_datetime(["2025-06-01", "2025-06-02"], utc=True), "close": [1, 2]})
        frame = pd.DataFrame({
            "source_ts": pd.to_datetime(["2025-05-31"], utc=True),
            "sma_40d": [1.0], "ret_20d": [0.1], "up": [True], "down": [False],
        })
        with patch.object(runner, "load_parent_gate_frame", return_value=frame) as mocked:
            result = runner.evaluate_parent_regime_gate(candidate, bars, pd.Timestamp("2025-06-02", tz="UTC"))
        self.assertTrue(result["allowed"])
        self.assertEqual(mocked.call_count, 2)
        for call in mocked.call_args_list:
            self.assertEqual(call.args[2], pd.Timestamp("2024-01-01", tz="UTC"))
            self.assertEqual(call.args[3], pd.Timestamp("2025-12-31 23:59:59", tz="UTC"))


if __name__ == "__main__":
    unittest.main()
