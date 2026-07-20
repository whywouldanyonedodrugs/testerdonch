import unittest
from datetime import datetime, timezone
from decimal import Decimal

from tools.qlmg_stage19_funding import (
    dual_alignment_cashflow_bps, equal_symbol_weighted_quantile,
    exact_cashflow_bps, period_for_row, type7,
)


UTC = timezone.utc


class Stage19FundingTests(unittest.TestCase):
    def test_type7_decimal_and_equal_symbol_weighting(self):
        values = [Decimal(i) for i in range(1, 11)]
        self.assertEqual(type7(values, Decimal("0.95")), Decimal("9.55"))
        mixture = {"A": [Decimal(0), Decimal(0)], "B": [Decimal(10)]}
        self.assertGreaterEqual(equal_symbol_weighted_quantile(mixture, Decimal("0.75")), Decimal(0))

    def test_period_alignments_are_both_explicit(self):
        t = datetime(2025, 1, 1, 2, tzinfo=UTC)
        self.assertEqual(period_for_row(t, "alignment_start")[0], t)
        self.assertEqual(period_for_row(t, "alignment_end")[1], t)

    def test_long_short_positive_negative_and_partial_hour(self):
        rate = Decimal("1")
        price = Decimal("100")
        self.assertEqual(exact_cashflow_bps(1, rate, Decimal("0.5"), price), Decimal("-50.000"))
        self.assertEqual(exact_cashflow_bps(-1, rate, Decimal("0.5"), price), Decimal("50.000"))
        self.assertEqual(exact_cashflow_bps(1, -rate, Decimal("0.5"), price), Decimal("50.000"))

    def test_favourable_funding_cannot_rescue_and_gap_is_nonpositive(self):
        entry = datetime(2025, 1, 1, 0, 30, tzinfo=UTC)
        exit_ = datetime(2025, 1, 1, 1, 30, tzinfo=UTC)
        rates = {datetime(2025, 1, 1, 0, tzinfo=UTC): Decimal("-1")}
        out = dual_alignment_cashflow_bps(
            entry=entry, exit_=exit_, position_sign=1, entry_trade_open=Decimal(100),
            absolute_rates=rates, base_gap_bps_per_hour=Decimal("2"),
            stress_gap_bps_per_hour=Decimal("4"),
        )
        self.assertLessEqual(out["adverse_exact_funding_bps"], 0)
        self.assertLessEqual(out["base_gap_cost_bps"], 0)
        self.assertLessEqual(out["stress_gap_cost_bps"], out["base_gap_cost_bps"])


if __name__ == "__main__":
    unittest.main()
