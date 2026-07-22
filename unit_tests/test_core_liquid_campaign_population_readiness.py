from __future__ import annotations

import unittest

from tools.core_liquid_campaign.population_readiness import (
    PopulationReadinessError,
    reconcile_registered_population_routes,
)
from tools.core_liquid_campaign.schema import FAMILY_ORDER


class PopulationReadinessTests(unittest.TestCase):
    def _rows(self):
        rows = []
        counts = {
            "A4_TSMOM_V7": 2688,
            "A1_COMPRESSION_V2": 3280,
            "A2_PRIOR_HIGH_RS_CONTEXT_V1": 2654,
            "A3_STARTER_RETEST_V3": 3132,
            "KDA02B_SURVIVOR_ADJUDICATION_V1": 209,
        }
        parent_id = "A1_COMPRESSION_V2:0"
        for family in FAMILY_ORDER:
            for index in range(counts[family]):
                identity = f"{family}:{index}"
                config = {"PIT_liquidity_top_n": 10}
                row = {
                    "family_id": family,
                    "executable_attempt_id": identity,
                    "canonical_economic_address_sha256": f"address:{family}:{index}",
                    "config": config,
                }
                if family == "A4_TSMOM_V7":
                    config["rebalance"] = "8h" if index % 2 else "1d"
                elif family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
                    config.update({
                        "parent_binding_mode": "source_attempt",
                        "parent_family": "A1_COMPRESSION_V2",
                    })
                    row["resolved_parent_executable_attempt_id"] = parent_id
                elif family == "KDA02B_SURVIVOR_ADJUDICATION_V1":
                    config.pop("PIT_liquidity_top_n")
                rows.append(row)
        return rows

    @staticmethod
    def _launch():
        return {
            "authority_inventory_sha256": "launch-authority",
            "population_census": {"fold_expanded": {
                "10": {"decisions_5m": 100, "a4_8h": 3, "a4_1d": 1},
                "20": {"decisions_5m": 200, "a4_8h": 6, "a4_1d": 2},
                "40": {"decisions_5m": 400, "a4_8h": 12, "a4_1d": 4},
            }},
        }

    @staticmethod
    def _kda():
        return {"counts": {
            "configurations": 209,
            "eligible_event_rows": 466_348,
            "unavailable_event_rows": 16_571,
            "eligible_dispatch_units": 5_129_828,
            "unavailable_dispatch_units": 182_281,
        }}

    def test_reconciles_every_frozen_address_without_probe_substitution(self):
        report = reconcile_registered_population_routes(self._rows(), self._launch(), self._kda())
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["registered_execution_addresses"], 11_963)
        self.assertEqual(report["KDA02B"]["typed_unavailable_reason"], "stage14_kda02b_final_eligible_false")
        self.assertTrue(report["KDA02B"]["unavailable_is_not_economic_testing"])
        self.assertEqual(report["A2"]["source_attempt_addresses"], 2_654)

    def test_missing_exact_a2_parent_fails_closed(self):
        rows = self._rows()
        a2 = next(row for row in rows if row["family_id"] == "A2_PRIOR_HIGH_RS_CONTEXT_V1")
        a2["resolved_parent_executable_attempt_id"] = "absent"
        with self.assertRaisesRegex(PopulationReadinessError, "exact frozen parent"):
            reconcile_registered_population_routes(rows, self._launch(), self._kda())

    def test_kda_population_drift_fails_closed(self):
        kda = self._kda()
        kda["counts"]["unavailable_event_rows"] -= 1
        with self.assertRaisesRegex(PopulationReadinessError, "KDA02B"):
            reconcile_registered_population_routes(self._rows(), self._launch(), kda)


if __name__ == "__main__":
    unittest.main()
