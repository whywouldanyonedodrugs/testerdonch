from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from tools.qlmg_rankable_source_contract import (
    RankableSourceContractError,
    directed_cross_platform_contract_id,
    filter_rankable_source_rows,
    read_rankable_source_payload,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


class RankableSourceContractTests(unittest.TestCase):
    @staticmethod
    def authority(**changes):
        authority = {
            "platform": "Kraken",
            "source_dataset_id": "synthetic_kraken_fixture_v1",
            "purpose": "rankable_research",
            "minimum_event_time_utc": "2023-01-01T00:00:00Z",
            "interval_end_utc_exclusive": "2026-01-01T00:00:00Z",
            "schema_hash": HASH_A,
            "content_sha256": HASH_B,
            "funding_type": "exact",
        }
        authority.update(changes)
        return authority

    def assert_rejected_before_reader(self, authority, *, platform="kraken"):
        reader = mock.Mock(return_value="payload")
        with self.assertRaises(RankableSourceContractError):
            read_rankable_source_payload(
                authority,
                selected_platform=platform,
                payload_reader=reader,
            )
        reader.assert_not_called()

    def test_unrankable_purposes_and_unprovable_hashes_fail_before_reader(self):
        for purpose in (
            "mixed_rankable_protected",
            "unknown",
            "holdout",
            "execution_calibration_only",
            "data_engineering_only",
        ):
            with self.subTest(purpose=purpose):
                self.assert_rejected_before_reader(self.authority(purpose=purpose))
        for missing in ("schema_hash", "content_sha256", "source_dataset_id"):
            with self.subTest(missing=missing):
                authority = self.authority()
                authority.pop(missing)
                self.assert_rejected_before_reader(authority)

    def test_missing_unknown_and_wrong_platform_fail_before_reader(self):
        self.assert_rejected_before_reader(None)
        self.assert_rejected_before_reader(self.authority(platform="Bybit"))
        self.assert_rejected_before_reader(self.authority(), platform="capital.com")

    def test_pretrain_mixed_and_protected_boundaries_fail_before_reader(self):
        missing_cutoff = self.authority()
        missing_cutoff.pop("interval_end_utc_exclusive")
        self.assert_rejected_before_reader(missing_cutoff)
        self.assert_rejected_before_reader(
            self.authority(minimum_event_time_utc="2022-12-31T23:59:59Z")
        )
        self.assert_rejected_before_reader(
            self.authority(interval_end_utc_exclusive="2026-01-01T00:00:01Z")
        )
        inclusive = self.authority(maximum_event_time_utc="2026-01-01T00:00:00Z")
        inclusive.pop("interval_end_utc_exclusive")
        self.assert_rejected_before_reader(inclusive)

    def test_exclusive_cutoff_and_valid_kraken_fixture_reach_reader_unchanged(self):
        payload = {"identity": "PF_XBTUSD", "value": 3}
        reader = mock.Mock(return_value=payload)
        result = read_rankable_source_payload(
            self.authority(),
            selected_platform="kraken",
            payload_reader=reader,
        )
        reader.assert_called_once_with()
        self.assertIs(result, payload)

    def test_funding_uses_same_preopen_contract(self):
        reader = mock.Mock()
        with self.assertRaisesRegex(RankableSourceContractError, "funding type"):
            read_rankable_source_payload(
                self.authority(funding_type="imputed"),
                selected_platform="kraken",
                funding=True,
                payload_reader=reader,
            )
        reader.assert_not_called()

    def test_pretrain_and_wrong_platform_rows_do_not_reach_downstream(self):
        rows = pd.DataFrame(
            [
                {"platform": "Kraken", "event_ts": "2022-12-31T23:00:00Z", "value": 1},
                {"platform": "Capital.com", "event_ts": "2024-01-01T00:00:00Z", "value": 2},
                {"platform": "Kraken", "event_ts": "2024-01-01T01:00:00Z", "value": 3},
            ]
        )
        downstream = mock.Mock()
        downstream(
            filter_rankable_source_rows(
                rows,
                selected_platform="kraken",
                event_time_column="event_ts",
            )
        )
        consumed = downstream.call_args.args[0]
        self.assertEqual(consumed["value"].tolist(), [3])

    def test_protected_payload_row_fails_before_downstream(self):
        rows = pd.DataFrame(
            [{"platform": "Kraken", "event_ts": "2026-01-01T00:00:00Z"}]
        )
        downstream = mock.Mock()
        with self.assertRaisesRegex(RankableSourceContractError, "protected rows"):
            downstream(
                filter_rankable_source_rows(
                    rows,
                    selected_platform="kraken",
                    event_time_column="event_ts",
                )
            )
        downstream.assert_not_called()

    def test_directed_cross_platform_identity_is_not_symmetric(self):
        forward = directed_cross_platform_contract_id(
            source_platform="Capital.com",
            source_instrument_id="EPIC_A",
            target_platform="Kraken",
            target_instrument_id="PF_AUSD",
            contract_version="v1",
        )
        reverse = directed_cross_platform_contract_id(
            source_platform="Kraken",
            source_instrument_id="PF_AUSD",
            target_platform="Capital.com",
            target_instrument_id="EPIC_A",
            contract_version="v1",
        )
        self.assertNotEqual(forward, reverse)
        self.assertEqual(len(forward), 64)


if __name__ == "__main__":
    unittest.main()
