from __future__ import annotations

import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools.qlmg_kda02_level3 import (
    branch_side,
    cluster_bootstrap,
    equal_cluster_returns,
    gate_flags,
    positive_contribution_share,
    score_open_prices,
)
from tools.qlmg_kda01_timestamp_repair import locate_repaired_execution, repaired_execution_records
from tools.build_kda02_v2_prerun_freeze import add_market_clusters, provisional_definitions
from tools.run_kda02_level3 import add_estimand_weights, attach_funding_diagnostics, reconstruct_schedule


class KDA02Level3Tests(unittest.TestCase):
    def test_frozen_branch_sides(self):
        self.assertEqual(branch_side("primary_negative_active_purge_continuation"), -1)
        self.assertEqual(branch_side("primary_positive_active_purge_continuation"), 1)
        self.assertEqual(branch_side("robustness_negative_completed_purge_reversal"), 1)
        self.assertEqual(branch_side("robustness_positive_completed_purge_reversal"), -1)

    def test_long_short_and_cost_arithmetic(self):
        self.assertEqual(tuple(round(x, 10) for x in score_open_prices(100, 101, 1)), (100.0, 86.0, 68.0))
        self.assertAlmostEqual(score_open_prices(100, 99, -1)[0], 100.0)
        for invalid in (0, -1, math.nan):
            with self.assertRaises(ValueError):
                score_open_prices(invalid, 1, 1)

    def test_equal_market_day_is_not_trade_weighted(self):
        trades = pd.DataFrame({
            "definition_id": ["d"] * 3, "event_id": [1, 2, 3],
            "market_day_cluster_id": ["a", "a", "b"],
            "gross_bps": [0, 100, 0], "base_net_bps": [-14, 86, -14],
            "stress_net_bps": [-32, 68, -32],
        })
        result = equal_cluster_returns(trades, "market_day_cluster_id")
        self.assertEqual(result.trade_count.tolist(), [2, 1])
        self.assertEqual(result.base_net_bps.tolist(), [36, -14])

    def test_bootstrap_seed_and_concentration(self):
        first, low, high = cluster_bootstrap([1, 2, 3])
        second, low2, high2 = cluster_bootstrap([1, 2, 3])
        np.testing.assert_array_equal(first, second)
        self.assertEqual((low, high), (low2, high2))
        frame = pd.DataFrame({"symbol": ["a", "b", "c"], "base_net_bps": [4, 1, -9]})
        self.assertEqual(positive_contribution_share(frame, "symbol"), .8)

    def test_exact_gate_boundaries(self):
        row = {
            "accepted_count": 100, "trades_2023": 20, "trades_2024": 20, "trades_2025": 20,
            "equal_day_base_mean_bps": 1e-12, "equal_day_base_median_bps": 1e-12,
            "bootstrap_lower_bps": -5, "market_day_positive_share": .10,
            "symbol_positive_share": .25, "year_positive_share": .70,
            "equal_day_stress_mean_bps": -10,
        }
        self.assertTrue(gate_flags(row)["all_gates_pass"])
        row["equal_day_base_mean_bps"] = -1e-12
        self.assertFalse(gate_flags(row)["all_gates_pass"])

    def test_corrected_entry_is_at_decision_without_extra_bar(self):
        times = pd.date_range("2024-01-01T00:00:00Z", periods=100, freq="5min")
        result = locate_repaired_execution("2024-01-01T00:05:00Z", 1, times)
        self.assertEqual(result.entry_ts, pd.Timestamp("2024-01-01T00:05:00Z"))
        self.assertEqual(result.exit_ts, pd.Timestamp("2024-01-01T01:05:00Z"))

    def test_definition_local_actual_exit_nonoverlap(self):
        events = pd.DataFrame([
            {"event_id": "a", "economic_address": "a", "branch_id": "primary_negative_active_purge_continuation", "symbol": "PF_X", "decision_ts": "2024-01-01T00:05:00Z"},
            {"event_id": "b", "economic_address": "b", "branch_id": "primary_negative_active_purge_continuation", "symbol": "PF_X", "decision_ts": "2024-01-01T00:30:00Z"},
            {"event_id": "c", "economic_address": "c", "branch_id": "primary_negative_active_purge_continuation", "symbol": "PF_X", "decision_ts": "2024-01-01T01:05:00Z"},
        ])
        definitions = provisional_definitions().iloc[[0]].copy()
        definitions.loc[:, "branch_id"] = "primary_negative_active_purge_continuation"
        times = pd.date_range("2024-01-01T00:00:00Z", periods=100, freq="5min")
        result = repaired_execution_records(events, definitions, {"PF_X": times})
        self.assertEqual(result.accepted.tolist(), [True, False, True])
        self.assertEqual(result.status.tolist(), ["eligible", "actual_position_overlap", "eligible"])

    def test_market_cluster_identity_is_attempt_and_parent_onset_based(self):
        events = pd.DataFrame({
            "attempt": ["primary", "primary", "robustness"],
            "parent_onset_ts": pd.to_datetime(["2024-01-01T00:05Z", "2024-01-01T05:55Z", "2024-01-01T00:05Z"], utc=True),
        })
        result = add_market_clusters(events)
        self.assertEqual(result.market_day_cluster_id.iloc[0], result.market_day_cluster_id.iloc[1])
        self.assertNotEqual(result.market_day_cluster_id.iloc[0], result.market_day_cluster_id.iloc[2])
        self.assertEqual(result.market_6h_cluster_id.iloc[0], result.market_6h_cluster_id.iloc[1])

    def test_estimand_contribution_weights_equal_days(self):
        trades = pd.DataFrame({
            "definition_id": ["d", "d", "d"], "event_id": [1, 2, 3],
            "market_day_cluster_id": ["a", "a", "b"], "base_net_bps": [10, 20, 30],
        })
        result = add_estimand_weights(trades)
        self.assertEqual(result.equal_market_day_trade_weight.tolist(), [.25, .25, .5])
        self.assertAlmostEqual(result.estimand_base_contribution_bps.sum(), 22.5)

    def test_runner_requires_review_before_open_price_read(self):
        source = Path("tools/run_kda02_level3.py").read_text(encoding="utf-8")
        self.assertIn('if not review.get("approved")', source)
        main = source[source.index("def main()") :]
        self.assertLess(main.index("reconstruct_schedule"), main.index("price_and_score"))
        self.assertNotIn("mark_open", source)

    def test_schedule_reconciliation_ignores_omitted_infeasible_gate_rows(self):
        definitions = pd.DataFrame({
            "definition_id": ["primary_feasible", "robustness_feasible"],
            "attempt": ["primary", "robustness"],
            "branch_id": ["primary_branch", "robustness_branch"],
        })
        events = pd.DataFrame({"symbol": ["PF_X"], "branch_id": ["primary_branch"]})
        gates = pd.DataFrame({
            "definition_id": ["primary_feasible", "primary_omitted_infeasible"],
            "accepted_events": [1, 99],
        })
        records = pd.DataFrame({
            "definition_id": ["primary_feasible", "robustness_feasible"],
            "event_id": ["p", "r"],
            "accepted": [True, True],
        })
        with patch("tools.run_kda02_level3.load_timestamp_only_bars", return_value=(pd.DatetimeIndex([]), "ref")), patch(
            "tools.run_kda02_level3.repaired_execution_records", return_value=records
        ):
            actual, _ = reconstruct_schedule(definitions, events, gates, authority=[])
        self.assertEqual(actual.event_id.tolist(), ["p", "r"])

    def test_funding_is_diagnostic_not_in_gate_flags(self):
        source = Path("tools/qlmg_kda01_level3_economic.py").read_text(encoding="utf-8")
        gate_body = source[source.index("def gate_flags") :]
        self.assertNotIn("funding", gate_body)

    def test_funding_uses_unique_definition_event_address_across_horizons(self):
        trades = pd.DataFrame({
            "economic_address": ["candidate", "candidate"],
            "level3_economic_address": ["level3_1h", "level3_6h"],
            "symbol": ["PF_X", "PF_X"], "side": ["long", "long"],
            "entry_ts": pd.to_datetime(["2024-01-01T00:05Z"] * 2, utc=True),
            "exit_ts": pd.to_datetime(["2024-01-01T01:05Z", "2024-01-01T06:05Z"], utc=True),
        })
        fields = ["funding_rate_central", "funding_rate_conservative", "funding_rate_severe", "funding_rate_conservative_short", "funding_rate_severe_short"]
        panel = pd.DataFrame(columns=["symbol", "timestamp", "funding_exact", "funding_imputed", "funding_rate_source", *fields])
        location = {field: 0.0 for field in fields}
        funded, boundaries = attach_funding_diagnostics(trades, panel, location)
        self.assertEqual(funded.level3_economic_address.tolist(), ["level3_1h", "level3_6h"])
        self.assertEqual(funded.candidate_economic_address.tolist(), ["candidate", "candidate"])
        self.assertEqual(boundaries.economic_address.nunique(), 2)


if __name__ == "__main__":
    unittest.main()
