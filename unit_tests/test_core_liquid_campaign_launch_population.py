from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.core_liquid_campaign.launch_population_authority import (
    LaunchPopulationAuthorityError,
    canonical_hash,
    census_pit,
    validate_frozen_counts,
    validate_launch_population_authority,
)


class LaunchPopulationAuthorityTests(unittest.TestCase):
    def test_pit_census_counts_every_decision_and_exact_a4_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "pit.jsonl"
            rows = [
                {"symbol": "PF_A", "day_open_ms": 1_700_006_400_000, "decision_count_5m": 288, "average_liquidity_rank": 1.0, "eligible_population": 2, "top_10": True, "top_20": True, "top_40": True},
                {"symbol": "PF_B", "day_open_ms": 1_700_006_400_000, "decision_count_5m": 288, "average_liquidity_rank": 11.0, "eligible_population": 20, "top_10": False, "top_20": True, "top_40": True},
                {"symbol": "PF_A", "day_open_ms": 1_767_139_200_000, "decision_count_5m": 1, "average_liquidity_rank": 1.0, "eligible_population": 2, "top_10": True, "top_20": True, "top_40": True},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            partitions = [{"phase": "outer_evaluation", "outer_fold_id": "F", "inner_fold_id": None, "evaluation_start_ms": 1_700_006_400_000, "evaluation_end_exclusive_ms": 1_767_225_600_000}]
            report, membership = census_pit(path, partitions)
            self.assertEqual(3, len(membership))
            self.assertEqual({"symbol_days": 3, "decisions_5m": 577, "a4_8h": 7, "a4_1d": 3}, report["physical"]["all"])
            self.assertEqual({"symbol_days": 2, "decisions_5m": 289, "a4_8h": 4, "a4_1d": 2}, report["physical"]["10"])
            self.assertEqual(577, report["partitions"][0]["counts"]["all"]["decisions_5m"])

    def test_pit_census_fails_closed_on_unresolved_partial_day(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "pit.jsonl"
            path.write_text(json.dumps({"symbol": "PF_A", "day_open_ms": 1_700_006_400_000, "decision_count_5m": 287, "average_liquidity_rank": 1.0, "eligible_population": 1, "top_10": True, "top_20": True, "top_40": True}) + "\n", encoding="utf-8")
            with self.assertRaises(LaunchPopulationAuthorityError):
                census_pit(path, [])

    def test_pit_census_accepts_one_decision_only_at_frozen_terminal_day(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "pit.jsonl"
            path.write_text(json.dumps({"symbol": "PF_A", "day_open_ms": 1_700_006_400_000, "decision_count_5m": 1, "average_liquidity_rank": 1.0, "eligible_population": 1, "top_10": True, "top_20": True, "top_40": True}) + "\n", encoding="utf-8")
            with self.assertRaises(LaunchPopulationAuthorityError):
                census_pit(path, [])

    def test_frozen_count_validator_rejects_probe_substitution(self) -> None:
        payload = {
            "population_census": {
                "physical": {key: {**value, "a4_8h": 0, "a4_1d": value["symbol_days"]} for key, value in {
                    "all": {"symbol_days": 134_151, "decisions_5m": 38_582_680},
                    "10": {"symbol_days": 10_660, "decisions_5m": 3_067_210},
                    "20": {"symbol_days": 21_320, "decisions_5m": 6_134_420},
                    "40": {"symbol_days": 42_640, "decisions_5m": 12_268_840},
                }.items()},
                "fold_expanded": {
                    "all": {"symbol_days": 518_360, "decisions_5m": 149_234_872, "a4_8h": 1_554_712, "a4_1d": 518_360},
                    "10": {"symbol_days": 45_140, "decisions_5m": 12_997_450, "a4_8h": 135_400, "a4_1d": 45_140},
                    "20": {"symbol_days": 90_280, "decisions_5m": 25_994_900, "a4_8h": 270_800, "a4_1d": 90_280},
                    "40": {"symbol_days": 180_560, "decisions_5m": 51_989_800, "a4_8h": 541_600, "a4_1d": 180_560},
                },
            },
            "a3_sparse_census": {
                "raw_signature_rows": {"all": 179_689, "10": 13_759, "20": 27_309, "40": 54_732},
                "fold_expanded_signature_rows": {"all": 706_464, "10": 61_579, "20": 123_641, "40": 245_886},
                "unique_crossing_keys_without_atr": {"all": 45_266, "10": 3_456, "20": 6_860, "40": 13_780},
                "fold_expanded_unique_crossing_keys_without_atr": {"all": 177_763, "10": 15_401, "20": 30_944, "40": 61_654},
                "fold_expanded_by_phase": {},
            },
        }
        validate_frozen_counts(payload)
        payload["population_census"]["fold_expanded"]["40"]["decisions_5m"] = 396
        with self.assertRaises(LaunchPopulationAuthorityError):
            validate_frozen_counts(payload)

    def test_launch_validator_keeps_benchmark_probe_out_of_launch_authority(self) -> None:
        payload = {
            "schema": "stage24_launch_population_authority_v1",
            "status": "bound_outcome_free",
            "outcome_firewall": {"economic_outcomes_opened": False, "protected_rows_opened": 0, "capitalcom_payload_opened": False},
            "benchmark_semantic_frames": {"classification": "benchmark_probe_only", "launch_input_authority": False, "artifacts": 567},
            "source_authority": {"physical_parts": 81_906, "source_bytes": 3_528_014_133, "source_rows": 81_128_690, "raw_source_parts_physically_verified": True},
            "funding_authority": {"symbols": 187, "rows": 3_373_194},
            "population_census": {
                "physical": {
                    "all": {"symbol_days": 134_151, "decisions_5m": 38_582_680},
                    "10": {"symbol_days": 10_660, "decisions_5m": 3_067_210},
                    "20": {"symbol_days": 21_320, "decisions_5m": 6_134_420},
                    "40": {"symbol_days": 42_640, "decisions_5m": 12_268_840},
                },
                "fold_expanded": {
                    "all": {"symbol_days": 518_360, "decisions_5m": 149_234_872, "a4_8h": 1_554_712, "a4_1d": 518_360},
                    "10": {"symbol_days": 45_140, "decisions_5m": 12_997_450, "a4_8h": 135_400, "a4_1d": 45_140},
                    "20": {"symbol_days": 90_280, "decisions_5m": 25_994_900, "a4_8h": 270_800, "a4_1d": 90_280},
                    "40": {"symbol_days": 180_560, "decisions_5m": 51_989_800, "a4_8h": 541_600, "a4_1d": 180_560},
                },
            },
            "a3_sparse_census": {
                "raw_signature_rows": {"all": 179_689, "10": 13_759, "20": 27_309, "40": 54_732},
                "fold_expanded_signature_rows": {"all": 706_464, "10": 61_579, "20": 123_641, "40": 245_886},
                "unique_crossing_keys_without_atr": {"all": 45_266, "10": 3_456, "20": 6_860, "40": 13_780},
                "fold_expanded_unique_crossing_keys_without_atr": {"all": 177_763, "10": 15_401, "20": 30_944, "40": 61_654},
            },
        }
        payload["authority_inventory_sha256"] = canonical_hash(payload)
        validate_launch_population_authority(payload, verify_files=False)
        payload["benchmark_semantic_frames"]["launch_input_authority"] = True
        payload["authority_inventory_sha256"] = canonical_hash({key: value for key, value in payload.items() if key != "authority_inventory_sha256"})
        with self.assertRaises(LaunchPopulationAuthorityError):
            validate_launch_population_authority(payload, verify_files=False)


if __name__ == "__main__":
    unittest.main()
