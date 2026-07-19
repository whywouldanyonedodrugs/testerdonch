from __future__ import annotations

import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools.build_kda03_v1_prerun_freeze import add_market_clusters, provisional_definitions, schedule_gates
from tools.qlmg_kda01_timestamp_repair import locate_repaired_execution, repaired_execution_records
from tools.qlmg_kda03_level3 import (
    assign_route,
    branch_side,
    cluster_bootstrap,
    equal_cluster_returns,
    route_flags,
    score_open_prices,
)
from tools import build_kraken_c01_foundation as foundation
from tools.qlmg_kraken_derivatives_state import sha256_file
from tools.run_kda03_level3 import (
    add_estimand_weights,
    attach_funding_diagnostics,
    reconstruct_schedule,
    verified_trade_authority_hash,
)


def route_row(**updates):
    row = {
        "equal_day_base_mean_bps": 1.0, "equal_day_base_median_bps": 1.0,
        "bootstrap_lower_bps": -5.0, "equal_day_stress_mean_bps": -10.0,
        "symbol_positive_share": .25, "year_positive_share": .70,
        "market_day_positive_share": .10, "market_day_clusters": 50,
        "material_estimand_or_context_dependence": False,
        "single_event_or_defect_explanation": False,
    }
    row.update(updates)
    return row


class KDA03Level3Tests(unittest.TestCase):
    def test_frozen_branch_sides(self):
        self.assertEqual(branch_side("primary_negative_reference_led_catchup"), 1)
        self.assertEqual(branch_side("primary_positive_reference_led_catchup"), -1)
        self.assertEqual(branch_side("robustness_negative_basis_impulse_continuation"), -1)
        self.assertEqual(branch_side("primary_positive_completed_basis_impulse_rejection"), -1)

    def test_long_short_and_cost_arithmetic(self):
        self.assertEqual(tuple(round(x, 10) for x in score_open_prices(100, 101, 1)), (100.0, 86.0, 68.0))
        self.assertAlmostEqual(score_open_prices(100, 99, -1)[0], 100.0)
        for invalid in (0, -1, math.nan):
            with self.assertRaises(ValueError):
                score_open_prices(invalid, 1, 1)

    def test_equal_market_day_is_not_trade_weighted_and_bootstrap_is_deterministic(self):
        trades = pd.DataFrame({
            "definition_id": ["d"] * 3, "event_id": [1, 2, 3],
            "market_day_cluster_id": ["a", "a", "b"],
            "gross_bps": [0, 100, 0], "base_net_bps": [-14, 86, -14],
            "stress_net_bps": [-32, 68, -32],
        })
        result = equal_cluster_returns(trades, "market_day_cluster_id")
        self.assertEqual(result.base_net_bps.tolist(), [36, -14])
        first = cluster_bootstrap([1, 2, 3])
        second = cluster_bootstrap([1, 2, 3])
        np.testing.assert_array_equal(first[0], second[0])

    def test_policy_route_priority_and_exact_boundaries(self):
        self.assertEqual(assign_route(route_row(equal_day_base_mean_bps=0)), "translation_rejected")
        self.assertEqual(assign_route(route_row(bootstrap_lower_bps=-5.0001)), "sample_limited_prospective_candidate")
        self.assertEqual(assign_route(route_row(equal_day_stress_mean_bps=-10.0001)), "execution_sensitive_candidate")
        self.assertEqual(assign_route(route_row(symbol_positive_share=.2501)), "narrow_sleeve_candidate")
        self.assertEqual(assign_route(route_row(year_positive_share=.7001)), "conditional_context_candidate_unvalidated")
        self.assertEqual(assign_route(route_row(material_estimand_or_context_dependence=True)), "conditional_context_candidate_unvalidated")
        self.assertEqual(assign_route(route_row()), "unconditional_control_candidate")
        self.assertTrue(route_flags(route_row())["control_eligible"])

    def test_mechanical_gate_is_exact_task_gate_not_concentration_or_year_gate(self):
        definitions = provisional_definitions()
        definition = definitions[definitions.definition_id.eq("kda03_v1_primary_negative_reference_led_catchup_timeout_1h")]
        records = pd.DataFrame({
            "definition_id": definition.definition_id.iloc[0],
            "branch_id": definition.branch_id.iloc[0],
            "year": [2025] * 100, "symbol": [f"PF_{i % 10}" for i in range(100)],
            "event_id": [f"e{i}" for i in range(100)], "economic_address": [f"a{i}" for i in range(100)],
            "exit_ts": pd.date_range("2025-01-01", periods=100, freq="D", tz="UTC"),
            "market_day_cluster_id": [f"d{i % 50}" for i in range(100)], "accepted": True,
        })
        _, gates = schedule_gates(records, definition)
        self.assertTrue(gates.definition_mechanically_feasible.iloc[0])

    def test_corrected_entry_and_definition_local_actual_exit_nonoverlap(self):
        times = pd.date_range("2024-01-01T00:00:00Z", periods=100, freq="5min")
        result = locate_repaired_execution("2024-01-01T00:05:00Z", 1, times)
        self.assertEqual(result.entry_ts, pd.Timestamp("2024-01-01T00:05:00Z"))
        events = pd.DataFrame([
            {"event_id": "a", "economic_address": "a", "branch_id": "b", "symbol": "PF_X", "decision_ts": "2024-01-01T00:05:00Z"},
            {"event_id": "b", "economic_address": "b", "branch_id": "b", "symbol": "PF_X", "decision_ts": "2024-01-01T00:30:00Z"},
            {"event_id": "c", "economic_address": "c", "branch_id": "b", "symbol": "PF_X", "decision_ts": "2024-01-01T01:05:00Z"},
        ])
        definitions = pd.DataFrame({"definition_id": ["d"], "definition_contract_hash": ["h"], "branch_id": ["b"], "timeout_hours": [1]})
        records = repaired_execution_records(events, definitions, {"PF_X": times})
        self.assertEqual(records.accepted.tolist(), [True, False, True])

    def test_cluster_identity_uses_attempt_and_onset(self):
        events = pd.DataFrame({
            "attempt": ["primary", "primary", "robustness"],
            "parent_onset_ts": pd.to_datetime(["2024-01-01T00:05Z", "2024-01-01T05:55Z", "2024-01-01T00:05Z"], utc=True),
        })
        result = add_market_clusters(events)
        self.assertEqual(result.market_day_cluster_id.iloc[0], result.market_day_cluster_id.iloc[1])
        self.assertNotEqual(result.market_day_cluster_id.iloc[0], result.market_day_cluster_id.iloc[2])

    def test_estimand_contribution_weights_equal_days(self):
        trades = pd.DataFrame({
            "definition_id": ["d", "d", "d"], "event_id": [1, 2, 3],
            "market_day_cluster_id": ["a", "a", "b"], "base_net_bps": [10, 20, 30],
        })
        result = add_estimand_weights(trades)
        self.assertEqual(result.equal_market_day_trade_weight.tolist(), [.25, .25, .5])
        self.assertAlmostEqual(result.estimand_base_contribution_bps.sum(), 22.5)

    def test_runner_requires_hash_bound_review_before_open_read(self):
        source = Path("tools/run_kda03_level3.py").read_text(encoding="utf-8")
        self.assertIn('if not review.get("approved")', source)
        main = source[source.index("def main()") :]
        self.assertLess(main.index("verified_trade_authority_hash"), main.index("reconstruct_schedule"))
        self.assertLess(main.index("reconstruct_schedule"), main.index("price_and_score"))
        self.assertNotIn("mark_open", source)

    def test_official_trade_authority_hash_fails_closed_before_payload_reader(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "bars.parquet"
            payload.write_bytes(b"frozen official bytes")
            row = foundation.AuthorityRow(
                dataset="historical_trade_candles_5m",
                symbol="PF_XUSD",
                chunk_start=pd.Timestamp("2023-01-01T00:00:00Z"),
                chunk_end=pd.Timestamp("2023-02-01T00:00:00Z"),
                parquet_path=payload,
                parquet_sha256=sha256_file(payload),
                rows=1,
            )
            first = verified_trade_authority_hash([row], ["PF_XUSD"])
            self.assertEqual(first, verified_trade_authority_hash([row], ["PF_XUSD"]))
            payload.write_bytes(b"drifted official bytes")
            with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                verified_trade_authority_hash([row], ["PF_XUSD"])

    def test_schedule_reconciliation_ignores_omitted_infeasible_rows(self):
        definitions = pd.DataFrame({"definition_id": ["primary_feasible", "robustness_feasible"], "attempt": ["primary", "robustness"], "branch_id": ["primary_branch", "robustness_branch"]})
        events = pd.DataFrame({"symbol": ["PF_X"], "branch_id": ["primary_branch"]})
        gates = pd.DataFrame({"definition_id": ["primary_feasible", "primary_omitted"], "accepted_events": [1, 99]})
        records = pd.DataFrame({"definition_id": ["primary_feasible", "robustness_feasible"], "event_id": ["p", "r"], "accepted": [True, True]})
        with patch("tools.run_kda03_level3.load_timestamp_only_bars", return_value=(pd.DatetimeIndex([]), "ref")), patch("tools.run_kda03_level3.repaired_execution_records", return_value=records):
            actual, _ = reconstruct_schedule(definitions, events, gates, authority=[])
        self.assertEqual(actual.event_id.tolist(), ["p", "r"])

    def test_funding_uses_unique_definition_event_address(self):
        trades = pd.DataFrame({
            "economic_address": ["candidate", "candidate"], "level3_economic_address": ["level3_1h", "level3_6h"],
            "symbol": ["PF_X", "PF_X"], "side": ["long", "long"],
            "entry_ts": pd.to_datetime(["2024-01-01T00:05Z"] * 2, utc=True),
            "exit_ts": pd.to_datetime(["2024-01-01T01:05Z", "2024-01-01T06:05Z"], utc=True),
        })
        fields = ["funding_rate_central", "funding_rate_conservative", "funding_rate_severe", "funding_rate_conservative_short", "funding_rate_severe_short"]
        panel = pd.DataFrame(columns=["symbol", "timestamp", "funding_exact", "funding_imputed", "funding_rate_source", *fields])
        funded, boundaries = attach_funding_diagnostics(trades, panel, {field: 0.0 for field in fields})
        self.assertEqual(funded.level3_economic_address.tolist(), ["level3_1h", "level3_6h"])
        self.assertEqual(boundaries.economic_address.nunique(), 2)


if __name__ == "__main__":
    unittest.main()
