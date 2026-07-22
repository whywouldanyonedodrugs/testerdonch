from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from tools.core_liquid_campaign.kda02b_denominator import (
    KDA02BDenominatorError,
    STAGE20_ELIGIBLE_SYMBOLS,
    STAGE20_KDA02B_OUTER_FOLDS,
    reconcile_stage20_kda02b_aggregate,
    stage20_kda02b_denominator_contract,
)
from tools.core_liquid_campaign.schema import baseline_config, normalize_config
from tools.core_liquid_campaign.selection import EventObservation, aggregate_streaming
from tools.core_liquid_campaign.synthetic import kda_frame
from tools.core_liquid_campaign.family_engines import kda02b_adjudication


UTC = timezone.utc
INTERVALS = (
    ("2023Q4", datetime(2023, 10, 1, tzinfo=UTC), datetime(2024, 1, 1, tzinfo=UTC)),
    ("2024Q1", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 4, 1, tzinfo=UTC)),
    ("2024Q2", datetime(2024, 4, 1, tzinfo=UTC), datetime(2024, 7, 1, tzinfo=UTC)),
    ("2024Q3", datetime(2024, 7, 1, tzinfo=UTC), datetime(2024, 10, 1, tzinfo=UTC)),
    ("2024Q4", datetime(2024, 10, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)),
    ("2025Q1", datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 4, 1, tzinfo=UTC)),
    ("2025Q2", datetime(2025, 4, 1, tzinfo=UTC), datetime(2025, 7, 1, tzinfo=UTC)),
    ("2025Q3", datetime(2025, 7, 1, tzinfo=UTC), datetime(2025, 10, 1, tzinfo=UTC)),
    ("2025Q4", datetime(2025, 10, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
)


class KDA02BDenominatorTests(unittest.TestCase):
    def _frames(self):
        config = normalize_config(
            "KDA02B_SURVIVOR_ADJUDICATION_V1",
            baseline_config("KDA02B_SURVIVOR_ADJUDICATION_V1"),
        )
        frames = []
        for fold, start, end in INTERVALS:
            anchor = start + timedelta(days=10)
            source = kda_frame(config, anchor=anchor)
            seconds = (end - start).total_seconds()
            frames.append(replace(
                source,
                fold_id=f"Q_{fold}",
                metadata={
                    **source.metadata,
                    "campaign_partition": {
                        "phase": "kda02b_adjudication",
                        "outer_fold_id": fold,
                        "inner_fold_id": None,
                        "evaluation_start": start,
                        "evaluation_end_exclusive": end,
                    },
                    "evaluation_start": start,
                    "evaluation_end_exclusive": end,
                    "eligible_days": int(seconds // 86400),
                    "eligible_symbol_seconds": seconds * STAGE20_ELIGIBLE_SYMBOLS,
                },
            ))
        return tuple(frames)

    @staticmethod
    def _observation(frame, index: int, *, event_id: str | None = None, exposure: float = 1.0):
        decision = frame.decision_ts
        start = frame.metadata["campaign_partition"]["evaluation_start"]
        end = frame.metadata["campaign_partition"]["evaluation_end_exclusive"]
        seconds = (end - start).total_seconds()
        entry = decision + timedelta(minutes=5)
        return EventObservation(
            event_id or f"event-{index}", frame.symbol, entry.date().isoformat(),
            entry.strftime("%Y-%m"), entry.year, float(index + 1), float(index),
            entry.date().isoformat(), decision, entry, entry + timedelta(hours=1),
            eligible_days=int(seconds // 86400),
            component_metrics=(("signed_component", -1.0 if index % 2 else 1.0),),
            holding_seconds_weighted=exposure * 3600.0,
            eligible_symbol_seconds=seconds * STAGE20_ELIGIBLE_SYMBOLS,
        )

    def test_reproduces_mixed_fold_denominator_and_reconciles_exact_stage20_union(self) -> None:
        frames = self._frames()
        raw = (self._observation(frames[0], 0), self._observation(frames[1], 1))
        with self.assertRaisesRegex(ValueError, "inconsistent aggregate denominator"):
            aggregate_streaming(raw)
        contract = stage20_kda02b_denominator_contract(frames)
        normalized, aggregate, trace = reconcile_stage20_kda02b_aggregate(raw, contract)
        self.assertEqual(823, contract["eligible_days"])
        self.assertEqual(823 * 86400 * STAGE20_ELIGIBLE_SYMBOLS, contract["eligible_symbol_seconds"])
        self.assertEqual({823}, {item.eligible_days for item in normalized})
        self.assertEqual(30 * len(raw) / 823, aggregate["opportunity_frequency_per_30d"])
        self.assertTrue(trace["aggregate_materialized_equal"])

    def test_component_sign_and_zero_exposure_preserve_the_denominator(self) -> None:
        frames = self._frames(); contract = stage20_kda02b_denominator_contract(frames)
        observations = (
            self._observation(frames[0], 0, exposure=0.0),
            self._observation(frames[-1], 1, exposure=1.0),
        )
        normalized, aggregate, trace = reconcile_stage20_kda02b_aggregate(observations, contract)
        self.assertEqual(1, trace["zero_exposure_observation_count"])
        self.assertEqual(3600.0 / contract["eligible_symbol_seconds"], aggregate["occupancy"])
        self.assertEqual(1.0, dict(normalized[0].component_metrics)["signed_component"])
        self.assertEqual(-1.0, dict(normalized[1].component_metrics)["signed_component"])

    def test_missing_duplicate_and_malformed_denominators_fail_closed(self) -> None:
        frames = self._frames()
        with self.assertRaisesRegex(KDA02BDenominatorError, "fold coverage differs"):
            stage20_kda02b_denominator_contract(frames[:-1])
        altered = replace(frames[0], metadata={**frames[0].metadata, "eligible_days": 91})
        with self.assertRaisesRegex(KDA02BDenominatorError, "eligible-day denominator differs"):
            stage20_kda02b_denominator_contract((altered, *frames[1:]))
        contract = stage20_kda02b_denominator_contract(frames)
        duplicate = self._observation(frames[0], 0, event_id="duplicate")
        with self.assertRaisesRegex(ValueError, "duplicate economic event ID"):
            reconcile_stage20_kda02b_aggregate((duplicate, duplicate), contract)

    def test_fold_order_is_the_exact_stage20_sequence(self) -> None:
        contract = stage20_kda02b_denominator_contract(tuple(reversed(self._frames())))
        self.assertEqual(
            STAGE20_KDA02B_OUTER_FOLDS,
            tuple(row["outer_fold_id"] for row in contract["partitions"]),
        )

    def test_component_filters_preserve_both_registered_side_cases(self) -> None:
        for cell, expected_side in (("KDA02B_009", 1), ("KDA02B_045", -1)):
            for variant in ("identity_replay", "price_only", "OI_removed", "liquidation_removed"):
                config = normalize_config("KDA02B_SURVIVOR_ADJUDICATION_V1", {
                    **baseline_config("KDA02B_SURVIVOR_ADJUDICATION_V1"),
                    "stage20_cell_id": cell,
                    "adjudication_variant": variant,
                })
                events = kda02b_adjudication.evaluate(kda_frame(config), config)
                self.assertEqual(1, len(events), (cell, variant))
                self.assertEqual(expected_side, events[0]["side"], (cell, variant))


if __name__ == "__main__":
    unittest.main()
