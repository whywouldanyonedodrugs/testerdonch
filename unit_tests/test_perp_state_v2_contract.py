from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tools.perp_state_v2 import (
    ContextGroupRequirement,
    ContextJoinContract,
    build_symbol_context_capability_manifest,
    build_symbol_month_eligibility,
    join_context_asof,
    load_bybit_context,
    write_symbol_month_eligibility,
)
from tools.perp_state_v2_family_specs import FAMILY_MARK_INDEX_DISLOCATION, FAMILY_PREMIUM_COMPRESSION, FAMILY_SPECS


def _context_frame() -> pd.DataFrame:
    ts = pd.date_range("2025-01-01 00:00:00", periods=3, freq="5min", tz="UTC")
    close = ts + pd.Timedelta(minutes=5)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "mark_open": [100, 101, 102],
            "mark_high": [101, 102, 103],
            "mark_low": [99, 100, 101],
            "mark_close": [100.5, 101.5, 102.5],
            "mark_source_open_ts": ts,
            "mark_source_close_ts": close,
            "index_open": [100, 101, 102],
            "index_high": [101, 102, 103],
            "index_low": [99, 100, 101],
            "index_close": [100, 101, 102],
            "index_source_open_ts": ts,
            "index_source_close_ts": close,
            "premium_open": [0.0, -0.1, -0.2],
            "premium_high": [0.1, 0.0, -0.1],
            "premium_low": [-0.1, -0.2, -0.3],
            "premium_close": [0.0, -0.1, -0.2],
            "premium_source_open_ts": ts,
            "premium_source_close_ts": close,
            "long_account_ratio": [0.5, 0.4, 0.6],
            "short_account_ratio": [0.5, 0.6, 0.4],
            "long_short_account_ratio": [1.0, 2 / 3, 1.5],
            "lsr_source_ts": ts,
            "lsr_source_close_ts": close,
            "context_source_close_ts": close,
            "mark_index_spread_pct": [0.005, 0.00495, 0.0049],
            "premium_compression_1h": [None, None, None],
            "premium_compression_4h": [None, None, None],
            "lsr_delta_1h": [None, None, None],
            "lsr_delta_4h": [None, None, None],
        }
    )


def _long_context_frame(periods: int = 60) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01 00:00:00", periods=periods, freq="5min", tz="UTC")
    close = ts + pd.Timedelta(minutes=5)
    idx = pd.RangeIndex(periods)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "mark_open": 100 + idx,
            "mark_high": 101 + idx,
            "mark_low": 99 + idx,
            "mark_close": 100.5 + idx,
            "mark_source_open_ts": ts,
            "mark_source_close_ts": close,
            "index_open": 100 + idx,
            "index_high": 101 + idx,
            "index_low": 99 + idx,
            "index_close": 100 + idx,
            "index_source_open_ts": ts,
            "index_source_close_ts": close,
            "premium_open": idx / 10000.0,
            "premium_high": idx / 10000.0,
            "premium_low": idx / 10000.0,
            "premium_close": idx / 10000.0,
            "premium_source_open_ts": ts,
            "premium_source_close_ts": close,
            "long_account_ratio": 0.45 + (idx % 5) / 100.0,
            "short_account_ratio": 0.55 - (idx % 5) / 100.0,
            "long_short_account_ratio": (0.45 + (idx % 5) / 100.0) / (0.55 - (idx % 5) / 100.0),
            "lsr_source_ts": ts,
            "lsr_source_close_ts": close,
            "context_source_close_ts": close,
        }
    )


class PerpStateV2ContractTests(unittest.TestCase):
    def test_family_specific_groups(self) -> None:
        self.assertEqual(FAMILY_SPECS[FAMILY_MARK_INDEX_DISLOCATION].required_context_groups, ("mark", "index"))
        self.assertEqual(FAMILY_SPECS[FAMILY_PREMIUM_COMPRESSION].required_context_groups, ("premium",))

    def test_strict_join_requires_exact_source_close(self) -> None:
        ctx = _context_frame()
        decisions = pd.DataFrame({"decision_ts": [pd.Timestamp("2025-01-01 00:05:00Z"), pd.Timestamp("2025-01-01 00:06:00Z")]})
        contract = ContextJoinContract("unit", (ContextGroupRequirement("mark"), ContextGroupRequirement("index")))
        out = join_context_asof(decisions, ctx, contract, fail_closed=False)
        self.assertTrue(bool(out.loc[0, "bybit_context_available"]))
        self.assertFalse(bool(out.loc[1, "bybit_context_available"]))
        self.assertEqual(out.loc[1, "bybit_context_reject_reason"], "missing_required_context")

    def test_missing_required_group_rejects(self) -> None:
        ctx = _context_frame().drop(columns=["premium_close"])
        decisions = pd.DataFrame({"decision_ts": [pd.Timestamp("2025-01-01 00:05:00Z")]})
        contract = ContextJoinContract("unit", (ContextGroupRequirement("premium"),))
        with self.assertRaisesRegex(KeyError, "missing required"):
            join_context_asof(decisions, ctx, contract)

    def test_empty_required_group_rows_reject_without_type_error(self) -> None:
        ctx = _context_frame()
        for col in ["mark_open", "mark_high", "mark_low", "mark_close", "mark_source_open_ts", "mark_source_close_ts"]:
            ctx[col] = pd.NA
        decisions = pd.DataFrame({"decision_ts": [pd.Timestamp("2025-01-01 00:05:00Z")]})
        contract = ContextJoinContract("unit", (ContextGroupRequirement("mark"),))
        out = join_context_asof(decisions, ctx, contract, fail_closed=False)
        self.assertFalse(bool(out.loc[0, "bybit_context_available"]))
        self.assertEqual(out.loc[0, "bybit_context_reject_reason"], "missing_required_context")

    def test_stale_lsr_rejects_when_exact_required(self) -> None:
        ctx = _context_frame()
        ctx.loc[0, "lsr_source_close_ts"] = pd.Timestamp("2025-01-01 00:00:00Z")
        decisions = pd.DataFrame({"decision_ts": [pd.Timestamp("2025-01-01 00:05:00Z")]})
        contract = ContextJoinContract("unit", (ContextGroupRequirement("mark"), ContextGroupRequirement("lsr")))
        out = join_context_asof(decisions, ctx, contract, fail_closed=False)
        self.assertFalse(bool(out.loc[0, "bybit_context_available"]))
        self.assertEqual(out.loc[0, "bybit_context_reject_reason"], "stale_required_context")

    def test_symbol_month_eligibility_marks_partial_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _context_frame().to_parquet(root / "AAAUSDT.parquet", index=False)
            contract = ContextJoinContract("fam", (ContextGroupRequirement("mark"),), min_symbol_month_coverage=0.01)
            elig = build_symbol_month_eligibility(
                symbols=["AAAUSDT", "BBBUSDT"],
                phase_windows={"development": (pd.Timestamp("2025-01-01 00:00:00Z"), pd.Timestamp("2025-01-01 00:10:00Z"))},
                family_contracts={"fam": contract},
                context_root=root,
            )
            self.assertEqual(len(elig), 2)
            self.assertTrue(bool(elig.loc[elig["symbol"].eq("AAAUSDT"), "admissible"].iloc[0]))
            self.assertFalse(bool(elig.loc[elig["symbol"].eq("BBBUSDT"), "admissible"].iloc[0]))
            digest = write_symbol_month_eligibility(root / "elig.csv", elig)
            self.assertEqual(len(digest), 40)

    def test_capability_manifest_classifies_lsr_only_as_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lsr_only = _context_frame().copy()
            for col in [c for c in lsr_only.columns if c.startswith(("mark_", "index_", "premium_")) or c in {"mark_index_spread_pct", "premium_compression_1h", "premium_compression_4h"}]:
                lsr_only[col] = pd.NA
            lsr_only.to_parquet(root / "AAAUSDT.parquet", index=False)
            cap = build_symbol_context_capability_manifest(["AAAUSDT", "BBBUSDT"], context_root=root)
            row = cap[cap["symbol"].eq("AAAUSDT")].iloc[0]
            self.assertEqual(row["context_capability_status"], "partial_context_only")
            self.assertFalse(bool(row["has_mark_history"]))
            self.assertTrue(bool(row["has_lsr_history"]))
            missing = cap[cap["symbol"].eq("BBBUSDT")].iloc[0]
            self.assertEqual(missing["context_capability_status"], "no_context_file")

    def test_eligibility_uses_family_specific_required_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lsr_only = _context_frame().copy()
            for col in [c for c in lsr_only.columns if c.startswith(("mark_", "index_", "premium_")) or c in {"mark_index_spread_pct", "premium_compression_1h", "premium_compression_4h"}]:
                lsr_only[col] = pd.NA
            lsr_only.to_parquet(root / "AAAUSDT.parquet", index=False)
            contracts = {
                "mark_family": ContextJoinContract("mark_family", (ContextGroupRequirement("mark"),), min_symbol_month_coverage=0.01),
                "lsr_family": ContextJoinContract("lsr_family", (ContextGroupRequirement("lsr"),), min_symbol_month_coverage=0.01),
            }
            elig = build_symbol_month_eligibility(
                symbols=["AAAUSDT"],
                phase_windows={"development": (pd.Timestamp("2025-01-01 00:00:00Z"), pd.Timestamp("2025-01-01 00:10:00Z"))},
                family_contracts=contracts,
                context_root=root,
            )
            mark_row = elig[elig["family"].eq("mark_family")].iloc[0]
            lsr_row = elig[elig["family"].eq("lsr_family")].iloc[0]
            self.assertFalse(bool(mark_row["admissible"]))
            self.assertEqual(mark_row["ineligibility_reason"], "source_unavailable_mark")
            self.assertTrue(bool(lsr_row["admissible"]))

    def test_loader_keeps_source_column_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _context_frame().to_parquet(root / "AAAUSDT.parquet", index=False)
            out = load_bybit_context("AAAUSDT", root=root)
            self.assertIn("mark_source_close_ts", out.columns)
            self.assertIn("context_source_close_ts", out.columns)
            self.assertNotIn("ts_close", out.columns)

    def test_loader_recomputes_stale_persisted_derived_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stale = _long_context_frame()
            stale["mark_index_spread_pct"] = pd.NA
            stale["premium_compression_1h"] = pd.NA
            stale["premium_compression_4h"] = pd.NA
            stale["lsr_delta_1h"] = pd.NA
            stale["lsr_delta_4h"] = pd.NA
            stale.to_parquet(root / "AAAUSDT.parquet", index=False)
            out = load_bybit_context("AAAUSDT", root=root)
            self.assertGreater(int(out["mark_index_spread_pct"].notna().sum()), 0)
            self.assertGreater(int(out["premium_compression_1h"].notna().sum()), 0)
            self.assertGreater(int(out["premium_compression_4h"].notna().sum()), 0)
            self.assertGreater(int(out["lsr_delta_1h"].notna().sum()), 0)
            self.assertGreater(int(out["lsr_delta_4h"].notna().sum()), 0)
            expected = (pd.to_numeric(out["premium_close"], errors="coerce") - pd.to_numeric(out["premium_close"], errors="coerce").shift(48)).iloc[50]
            self.assertAlmostEqual(float(out.loc[50, "premium_compression_4h"]), float(expected))

    def test_raw_only_sidecar_schema_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_only = _long_context_frame()
            raw_only.to_parquet(root / "AAAUSDT.parquet", index=False)
            out = load_bybit_context("AAAUSDT", root=root)
            self.assertIn("premium_compression_4h", out.columns)
            cap = build_symbol_context_capability_manifest(["AAAUSDT"], context_root=root)
            self.assertEqual(cap.iloc[0]["context_capability_status"], "full_context_eligible")


if __name__ == "__main__":
    unittest.main()
