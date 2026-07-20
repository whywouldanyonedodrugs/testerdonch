import unittest
import hashlib
import tempfile
import zipfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from tools.qlmg_stage19_funding import (
    dual_alignment_cashflow_bps, equal_symbol_weighted_quantile,
    exact_cashflow_bps, period_for_row, Stage19FundingEngine, type7,
)


UTC = timezone.utc


class Stage19FundingTests(unittest.TestCase):
    @staticmethod
    def _hash(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
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
        with self.assertRaises(ValueError):
            dual_alignment_cashflow_bps(
                entry=entry, exit_=exit_, position_sign=1, entry_trade_open=Decimal(100),
                absolute_rates={}, base_gap_bps_per_hour=Decimal("-2"),
                stress_gap_bps_per_hour=Decimal("-4"),
            )

    def test_hash_bound_rankable_campaign_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "rankable.zip"
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr(
                    "rankable_2023_2025/PF_TESTUSD.csv",
                    "timestamp,tradeable,absolute_rate,relative_rate\n"
                    "2025-01-01 00:00:00,PF_TESTUSD,0.01,0.0001\n",
                )
            table = root / "allowances.csv"
            table.write_text(
                "symbol,rankable_observations,base_gap_allowance_bps_per_hour,stress_gap_allowance_bps_per_hour,allowance_source\n"
                "PF_TESTUSD,720,2,4,symbol_type7_rankable_only\n"
            )
            engine = Stage19FundingEngine(package, self._hash(package), table, self._hash(table))
            result = engine.evaluate_trade(
                symbol="PF_TESTUSD", entry=datetime(2025, 1, 1, 0, tzinfo=UTC),
                exit_=datetime(2025, 1, 1, 1, tzinfo=UTC), position_sign=1,
                entry_trade_open=Decimal(100),
            )
            self.assertLessEqual(result["adverse_exact_funding_bps"], 0)
            self.assertIs(engine.load_symbol("PF_TESTUSD"), engine.load_symbol("PF_TESTUSD"))
            with self.assertRaises(RuntimeError):
                Stage19FundingEngine(package, "0" * 64, table, self._hash(table))
            for bad_entry, bad_exit in [
                (datetime(2022, 12, 31, 23, tzinfo=UTC), datetime(2023, 1, 1, tzinfo=UTC)),
                (datetime(2025, 12, 31, 23, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC)),
                (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 1, tzinfo=UTC)),
                (datetime(2025, 1, 1), datetime(2025, 1, 1, 1)),
            ]:
                with self.assertRaises(RuntimeError):
                    engine.evaluate_trade(
                        symbol="PF_TESTUSD", entry=bad_entry, exit_=bad_exit,
                        position_sign=1, entry_trade_open=Decimal(100),
                    )


if __name__ == "__main__":
    unittest.main()
