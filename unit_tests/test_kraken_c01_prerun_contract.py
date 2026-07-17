from __future__ import annotations

from datetime import datetime, timezone
import unittest

from tools.kraken_c01_prerun_contract import (
    BOOTSTRAP_RESAMPLES,
    PRIMARY_MODEL,
    ROBUSTNESS_MODEL,
    canonical_episode_bootstrap_mean_ci,
    definition_local_non_overlap,
    definition_register,
    definitions_permitted_for_level4,
    fixed_notional_net_bps,
    funding_partition,
    interval_is_wholly_train_eligible,
    select_matched_non_event,
    validate_package_disposition,
)


def ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def passing_metrics(definition_id: str, model: str = PRIMARY_MODEL) -> dict:
    return {
        "definition_id": definition_id,
        "model": model,
        "executed_trades": 120,
        "trade_count_by_year": {2023: 40, 2024: 40, 2025: 40},
        "mean_net_bps": 2.0,
        "median_net_bps": 1.0,
        "bootstrap_ci_lower_bps": -4.0,
        "max_symbol_pnl_share": 0.20,
        "max_episode_pnl_share": 0.08,
        "max_year_positive_pnl_share": 0.60,
        "stress_mean_net_bps": -9.0,
    }


class C01PrerunContractTests(unittest.TestCase):
    def test_all_16_definitions_are_unique_and_retained(self) -> None:
        rows = definition_register()
        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(row["model_role"] == "primary" for row in rows), 8)
        self.assertEqual(sum(row["model_role"] == "robustness_only" for row in rows), 8)
        self.assertTrue(all(row["registered_even_if_zero_trades"] for row in rows))
        self.assertEqual(len({row["definition_policy_hash"] for row in rows}), 16)

    def test_actual_exit_non_overlap_is_definition_local(self) -> None:
        rows = [
            {"definition_id": "a", "symbol": "PF_X", "economic_address": "a1", "onset_ts": ts("2025-01-01T00:00:00Z"), "entry_ts": ts("2025-01-01T00:05:00Z"), "actual_exit_ts": ts("2025-01-01T01:00:00Z")},
            {"definition_id": "a", "symbol": "PF_X", "economic_address": "a2", "onset_ts": ts("2025-01-01T00:30:00Z"), "entry_ts": ts("2025-01-01T00:35:00Z"), "actual_exit_ts": ts("2025-01-01T02:00:00Z")},
            {"definition_id": "a", "symbol": "PF_X", "economic_address": "a3", "onset_ts": ts("2025-01-01T01:00:00Z"), "entry_ts": ts("2025-01-01T01:05:00Z"), "actual_exit_ts": ts("2025-01-01T02:00:00Z")},
            {"definition_id": "b", "symbol": "PF_X", "economic_address": "b1", "onset_ts": ts("2025-01-01T00:30:00Z"), "entry_ts": ts("2025-01-01T00:35:00Z"), "actual_exit_ts": ts("2025-01-01T00:45:00Z")},
        ]
        result = definition_local_non_overlap(rows)
        self.assertEqual({row["economic_address"] for row in result.accepted}, {"a1", "a3", "b1"})
        self.assertEqual(result.skipped[0]["prior_actual_exit_ts"], ts("2025-01-01T01:00:00Z"))
        self.assertEqual(len(result.accepted) + len(result.skipped), len(rows))

    def test_boundary_requires_every_timestamp_and_never_endpoint_close(self) -> None:
        fields = {
            "onset_ts": ts("2025-12-30T00:00:00Z"),
            "confirmation_ts": ts("2025-12-30T01:00:00Z"),
            "entry_ts": ts("2025-12-30T01:05:00Z"),
            "last_stop_monitor_ts": ts("2025-12-31T23:55:00Z"),
            "timeout_ts": ts("2025-12-31T23:55:00Z"),
            "funding_accounting_end_ts": ts("2025-12-31T16:00:00Z"),
            "exit_execution_ts": ts("2025-12-31T23:55:00Z"),
        }
        self.assertTrue(interval_is_wholly_train_eligible(fields))
        fields["exit_execution_ts"] = ts("2026-01-01T00:00:00Z")
        self.assertFalse(interval_is_wholly_train_eligible(fields))
        fields["exit_execution_ts"] = None
        self.assertFalse(interval_is_wholly_train_eligible(fields))

    def test_fixed_notional_arithmetic_is_not_stop_scaled(self) -> None:
        result = fixed_notional_net_bps(
            entry_price=100, exit_price=101, side="long",
            fee_bps=10, slippage_bps=4, funding_cashflow_bps=-1,
        )
        self.assertAlmostEqual(result["gross_return_bps"], 100)
        self.assertAlmostEqual(result["net_return_bps"], 85)
        self.assertNotIn("risk_denominator", result)

    def test_funding_partitions_cannot_be_pooled(self) -> None:
        self.assertEqual(funding_partition({"exact_boundary_count": 2, "imputed_boundary_count": 0}), "fully_exact")
        self.assertEqual(funding_partition({"exact_boundary_count": 1, "imputed_boundary_count": 1}), "mixed")
        self.assertEqual(funding_partition({"exact_boundary_count": 0, "imputed_boundary_count": 2}), "fully_imputed")
        self.assertEqual(funding_partition({"exact_boundary_count": 0, "imputed_boundary_count": 0}), "zero_boundary")

    def test_cluster_bootstrap_is_deterministic_and_frozen_to_10000(self) -> None:
        values = {"episode_a": [1.0, 2.0], "episode_b": [-1.0], "episode_c": [3.0]}
        self.assertEqual(canonical_episode_bootstrap_mean_ci(values), canonical_episode_bootstrap_mean_ci(values))
        with self.assertRaisesRegex(ValueError, "10,000"):
            canonical_episode_bootstrap_mean_ci(values, resamples=BOOTSTRAP_RESAMPLES - 1)

    def test_primary_gate_and_btc_only_cannot_rescue(self) -> None:
        primary = passing_metrics("primary")
        robustness = passing_metrics("robust", ROBUSTNESS_MODEL)
        self.assertEqual(definitions_permitted_for_level4([primary, robustness]), ["primary"])
        primary["median_net_bps"] = 0
        self.assertEqual(definitions_permitted_for_level4([primary, robustness]), [])

    def test_concentration_and_bootstrap_thresholds_are_all_hard_gates(self) -> None:
        fields = {
            "bootstrap_ci_lower_bps": -5.01,
            "max_symbol_pnl_share": 0.2501,
            "max_episode_pnl_share": 0.1001,
            "max_year_positive_pnl_share": 0.7001,
        }
        for field, failing_value in fields.items():
            with self.subTest(field=field):
                metrics = passing_metrics("primary")
                metrics[field] = failing_value
                self.assertEqual(definitions_permitted_for_level4([metrics]), [])

    def test_matched_control_calipers_never_widen(self) -> None:
        event = {
            "symbol": "PF_X", "calendar_year": 2024, "direction": "long",
            "onset_ts": ts("2024-06-01T00:00:00Z"), "lagged_volatility_24h": 0.10,
            "btc_return_6h_bps": 100, "eth_return_6h_bps": 80,
        }
        valid = {
            "control_address": "valid", "symbol": "PF_X", "calendar_year": 2024, "direction": "long",
            "timestamp": ts("2024-06-04T00:00:00Z"), "lagged_volatility_24h": 0.11,
            "btc_return_6h_bps": 110, "eth_return_6h_bps": 90,
            "inside_same_symbol_c01_episode": False,
        }
        outside = {**valid, "control_address": "outside", "lagged_volatility_24h": 0.121}
        self.assertEqual(select_matched_non_event(event, [outside, valid])["control_address"], "valid")
        self.assertIsNone(select_matched_non_event(event, [outside]))

    def test_claim_narrowing_keeps_every_gap_nonpassing(self) -> None:
        disposition = {
            "protocol_disposition": "closed_by_claim_narrowing",
            "package_role": "strategic_and_continuity_review_only",
            "package_release_ready_for_independent_reproduction": False,
            "missing_items": [
                {"item": "raw_extract", "status": "unavailable"},
                {"item": "hashes", "status": "deferred_with_exact_task", "exact_task": "task_v1"},
            ],
        }
        validate_package_disposition(disposition)
        disposition["missing_items"][0]["status"] = "pass"
        with self.assertRaisesRegex(ValueError, "cannot be converted"):
            validate_package_disposition(disposition)


if __name__ == "__main__":
    unittest.main()
