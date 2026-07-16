from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.qlmg_screening_core import FundingEvent, ReplayConfig, replay_trade
from tools.qlmg_short_event_generators import (
    FINAL_HOLDOUT_START,
    f1_variants,
    g1_parent_variants,
    g1_variants,
    generate_a1_breakout_parents,
    generate_f1_events,
    generate_g1_events,
    revised_short_score,
    validate_no_protected,
)
from tools.run_qlmg_f1_g1_short_unblock import done_path, required_outputs_for_stage, stage_complete


def tiers() -> pd.DataFrame:
    return pd.DataFrame({"symbol": ["TESTUSDT"], "date": ["2025-01-01"], "liquidity_tier": ["C"]})


def base_df(n: int = 900) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    close = np.linspace(100, 120, n)
    return pd.DataFrame({
        "timestamp": idx,
        "open": close,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": np.full(n, 1000.0),
        "turnover": close * 1000,
        "open_interest": np.linspace(1000, 1200, n),
        "funding_rate": np.full(n, 0.0001),
    })


class F1G1ShortUnblockTests(unittest.TestCase):
    def test_protected_slice_rejected(self) -> None:
        df = pd.DataFrame({"decision_ts": [pd.Timestamp("2026-01-01", tz="UTC")]})
        with self.assertRaises(RuntimeError):
            validate_no_protected(df, ["decision_ts"])
        self.assertEqual(str(FINAL_HOLDOUT_START), "2026-01-01 00:00:00+00:00")

    def test_short_replay_mechanics_and_funding(self) -> None:
        idx = pd.date_range("2025-01-01", periods=6, freq="5min", tz="UTC")
        bars = pd.DataFrame({
            "open": [100] * 6,
            "high": [100, 101, 102, 103, 104, 105],
            "low": [100, 99, 97, 95, 94, 93],
            "close": [100, 99, 98, 96, 95, 94],
            "mark_high": [100, 101, 102, 103, 104, 105],
            "mark_low": [100, 99, 97, 95, 94, 93],
        }, index=idx)
        tp = replay_trade(bars, ReplayConfig("short", idx[0], idx[0], 100, 105, 96, 1), [])
        self.assertEqual(tp.exit_reason, "target")
        self.assertGreater(tp.net_R, 0)
        fund = replay_trade(bars, ReplayConfig("short", idx[0], idx[0], 100, 110, None, 1), [FundingEvent(idx[1], 0.01, 100)])
        self.assertGreater(fund.funding_pnl, 0)
        liq = replay_trade(bars.assign(mark_high=[100, 112, 112, 112, 112, 112]), ReplayConfig("short", idx[0], idx[0], 100, 130, 80, 1, leverage=10), [])
        self.assertEqual(liq.exit_reason, "liquidation")

    def test_f1_does_not_fire_before_backside_trigger(self) -> None:
        df = base_df(900)
        # Strong extension with no backside trigger: no events.
        df.loc[350:650, "close"] = np.linspace(130, 260, 301)
        df.loc[651:, "close"] = 260.0
        df["open"] = df["close"]
        df["high"] = df["close"] * 1.002
        df["low"] = df["close"] * 0.998
        events, _ = generate_f1_events("TESTUSDT", df, tiers(), f1_variants(1))
        self.assertTrue(events.empty)
        # Add a separate backside lower-low trigger after extension.
        df.loc[651, "close"] = df.loc[650, "low"] * 0.99
        df.loc[651, "open"] = df.loc[650, "close"]
        df.loc[651, "high"] = max(df.loc[651, "open"], df.loc[651, "close"]) * 1.001
        df.loc[651, "low"] = min(df.loc[651, "open"], df.loc[651, "close"]) * 0.999
        events, _ = generate_f1_events("TESTUSDT", df, tiers(), f1_variants(1))
        self.assertFalse(events.empty)
        self.assertTrue((pd.to_datetime(events["decision_ts"], utc=True) >= df.loc[651, "timestamp"]).all())
        self.assertTrue(events["backside_confirmation"].all())

    def test_g1_starts_from_parent_and_later_failure(self) -> None:
        df = base_df(1200)
        # Sideways base, then breakout, then later failure back inside.
        df.loc[:700, "close"] = 100.0
        df.loc[701:760, "close"] = np.linspace(100, 140, 60)
        df.loc[761:780, "close"] = 142.0
        df.loc[781:790, "close"] = 95.0
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1) * 1.003
        df["low"] = df[["open", "close"]].min(axis=1) * 0.997
        parents, _ = generate_a1_breakout_parents("TESTUSDT", df, tiers(), g1_parent_variants()[:1])
        self.assertFalse(parents.empty)
        events, _ = generate_g1_events("TESTUSDT", df, tiers(), parents, g1_variants(1))
        self.assertFalse(events.empty)
        self.assertTrue((pd.to_datetime(events["decision_ts"], utc=True) > pd.to_datetime(events["parent_decision_ts"], utc=True)).all())

    def test_revised_score_hard_penalizes_negative_pf_liquidation(self) -> None:
        good = revised_short_score({"net_R": 10, "PF": 1.5, "trades": 100, "liquidation_count": 0, "max_dd_R": -5})
        bad = revised_short_score({"net_R": -1, "PF": 3.0, "trades": 100, "liquidation_count": 0, "max_dd_R": -1})
        liq = revised_short_score({"net_R": 20, "PF": 2.0, "trades": 100, "liquidation_count": 1, "max_dd_R": -1})
        self.assertGreater(good, bad)
        self.assertLess(liq, 0)

    def test_stage_checkpoint_requires_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            done_path(root, "seal-guard").parent.mkdir(parents=True)
            done_path(root, "seal-guard").write_text("done")
            self.assertFalse(stage_complete(root, "seal-guard"))
            for p in required_outputs_for_stage(root, "seal-guard"):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x")
            self.assertTrue(stage_complete(root, "seal-guard"))

    def test_tmux_wrapper_exists(self) -> None:
        self.assertTrue(Path("tools/run_qlmg_f1_g1_short_unblock_tmux.sh").exists())


if __name__ == "__main__":
    unittest.main()
