from __future__ import annotations

import inspect
import unittest
from unittest import mock

import pandas as pd

from tools import capitalcom_data_adapter as adapter
from tools.qlmg_rankable_source_contract import RankableSourceContractError


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


class CapitalcomDataAdapterTests(unittest.TestCase):
    @staticmethod
    def authority(**changes):
        authority = {
            "platform": "Capital.com",
            "source_dataset_id": "synthetic_capitalcom_fixture_v1",
            "purpose": "rankable_research",
            "minimum_event_time_utc": "2024-01-01T00:00:00Z",
            "maximum_event_time_utc": "2024-01-01T00:05:00Z",
            "schema_hash": HASH_A,
            "content_sha256": HASH_B,
            "volume_semantics_verified": False,
        }
        authority.update(changes)
        return authority

    @staticmethod
    def row(**changes):
        row = {
            "platform": "Capital.com",
            "platform_epic": "SYNTH.EPIC.A",
            "instrument_type": "CFD",
            "instrument_name": "Synthetic A",
            "currency": "USD",
            "contract_form": "undated_cash",
            "bid_open": 99.0,
            "bid_high": 101.0,
            "bid_low": 98.0,
            "bid_close": 100.0,
            "ask_open": 99.2,
            "ask_high": 101.2,
            "ask_low": 98.2,
            "ask_close": 100.2,
            "bar_start_utc": "2024-01-01T00:00:00Z",
            "bar_end_utc": "2024-01-01T00:05:00Z",
            "availability_utc": "2024-01-01T00:05:01Z",
            "calendar_id": "SYNTH_24X5",
            "metadata_snapshot_hash": HASH_C,
            "expiry_or_undated": "undated",
            "market_status": "open",
            "volume_semantic_status": "unverified",
            "financing_status": "unknown",
            "corporate_action_status": "unknown",
        }
        row.update(changes)
        return row

    def load(self, rows, authority=None):
        reader = mock.Mock(return_value=pd.DataFrame(rows))
        frame = adapter.load_capitalcom_bid_ask_bars(
            authority or self.authority(),
            payload_reader=reader,
        )
        reader.assert_called_once_with()
        return frame

    def test_bid_ask_are_preserved_and_not_labeled_exchange_trades(self):
        frame = self.load([self.row()])
        self.assertEqual(frame.loc[0, "bid_close"], 100.0)
        self.assertEqual(frame.loc[0, "ask_close"], 100.2)
        self.assertEqual(frame.loc[0, "price_semantics"], "otc_cfd_bid_ask")
        self.assertNotIn("trade_price", frame.columns)

    def test_buy_uses_ask_sell_uses_bid_and_no_midpoint_mode_exists(self):
        row = self.row()
        self.assertEqual(adapter.hypothetical_execution_price(row, side="buy"), 99.2)
        self.assertEqual(adapter.hypothetical_execution_price(row, side="sell"), 99.0)
        self.assertNotIn("midpoint", inspect.signature(adapter.hypothetical_execution_price).parameters)

    def test_required_identity_and_metadata_fields_fail_closed(self):
        for field in (
            "platform_epic",
            "instrument_type",
            "calendar_id",
            "metadata_snapshot_hash",
            "expiry_or_undated",
        ):
            with self.subTest(field=field):
                with self.assertRaises(RankableSourceContractError):
                    self.load([self.row(**{field: ""})])

    def test_invalid_bid_ask_ordering_fails_closed(self):
        with self.assertRaisesRegex(RankableSourceContractError, "bid/ask ordering"):
            self.load([self.row(ask_close=99.9)])

    def test_payload_must_reconcile_to_manifest_and_protected_boundary(self):
        with self.assertRaisesRegex(RankableSourceContractError, "exceeds manifest maximum"):
            self.load(
                [self.row()],
                authority=self.authority(maximum_event_time_utc="2024-01-01T00:04:59Z"),
            )
        with self.assertRaisesRegex(RankableSourceContractError, "outside the rankable boundary"):
            self.load(
                [self.row(availability_utc="2026-01-01T00:00:00Z")],
            )

    def test_volume_financing_and_corporate_action_semantics_are_explicit(self):
        with self.assertRaisesRegex(RankableSourceContractError, "volume semantics lack"):
            self.load([self.row(volume_semantic_status="verified")])
        verified = self.load(
            [self.row(volume_semantic_status="verified")],
            authority=self.authority(volume_semantics_verified=True),
        )
        self.assertEqual(verified.loc[0, "volume_semantic_status"], "verified")
        for field in ("financing_status", "corporate_action_status"):
            with self.subTest(field=field), self.assertRaises(RankableSourceContractError):
                self.load([self.row(**{field: 0})])

    def test_closed_target_maps_to_first_executable_quote_after_reopening(self):
        rows = [
            self.row(
                bar_start_utc="2024-01-01T00:00:00Z",
                bar_end_utc="2024-01-01T00:05:00Z",
                availability_utc="2024-01-01T00:05:01Z",
                market_status="closed",
            ),
            self.row(
                bar_start_utc="2024-01-01T00:05:00Z",
                bar_end_utc="2024-01-01T00:10:00Z",
                availability_utc="2024-01-01T00:10:01Z",
                market_status="open",
                ask_open=102.0,
                ask_high=103.0,
                ask_low=101.0,
                ask_close=102.5,
                bid_open=101.8,
                bid_high=102.8,
                bid_low=100.8,
                bid_close=102.3,
            ),
        ]
        authority = self.authority(maximum_event_time_utc="2024-01-01T00:10:00Z")
        frame = self.load(rows, authority=authority)
        target = adapter.first_executable_target_bar(
            frame,
            earliest_availability_utc="2024-01-01T00:05:01Z",
        )
        self.assertEqual(target["bar_end_utc"], pd.Timestamp("2024-01-01T00:10:00Z"))
        self.assertEqual(adapter.hypothetical_execution_price(target, side="buy"), 102.0)

    def test_adapter_has_no_api_account_order_or_credential_dependency(self):
        source = inspect.getsource(adapter).lower()
        for prohibited in ("requests", "httpx", "api_key", "password", "place_order", "account_id"):
            with self.subTest(prohibited=prohibited):
                self.assertNotIn(prohibited, source)


if __name__ == "__main__":
    unittest.main()
