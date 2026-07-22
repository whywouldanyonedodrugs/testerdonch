from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.core_liquid_campaign.canonical import atomic_write_json, sha256_file
from tools.core_liquid_campaign.kda02b_lazy_family_input import (
    ECONOMIC_MODE,
    KDA02BLazyFamilyInputAdapter,
    KDA02BLazyFamilyInputError,
    SHADOW_MODE,
)
from tools.core_liquid_campaign.kda02b_population_index import build_kda02b_lazy_population_index
from tools.core_liquid_campaign.family_engines import kda02b_adjudication
from unit_tests.test_kda02b_population_index import KDA02BPopulationIndexTests


UTC = timezone.utc


class KDA02BLazyFamilyInputTests(unittest.TestCase):
    def _fixture(self, root: Path):
        sources, expected = KDA02BPopulationIndexTests()._fixture(root)
        event_decisions: dict[str, datetime] = {}
        event_manifest = json.loads(sources.event_manifest_path.read_text(encoding="utf-8"))
        for record in event_manifest["files"]:
            rows = pq.read_table(record["path"], columns=["symbol", "decision_ts"]).to_pylist()
            if rows:
                event_decisions[str(record["symbol"])] = rows[0]["decision_ts"].astimezone(UTC)

        feature_manifest = json.loads(sources.feature_manifest_path.read_text(encoding="utf-8"))
        for record in feature_manifest["partitions"]:
            symbol = str(record["symbol"]); decision = event_decisions[symbol]
            rows = []
            for offset in (15, 10, 5):
                rows.append({
                    "timestamp_utc": decision - timedelta(minutes=offset),
                    "trade_close": 100.0,
                    "trade_return_1h": -0.002,
                    "mark_return_1h": -0.002,
                    "oi_log_change_1h": -0.02,
                    "liquidation_base_units_1h": 5.0,
                    "liquidation_intensity_robust_z": 2.0,
                    "liquidation_normalization_valid": True,
                    "eligible": symbol != "PF_UUSD",
                    "known_lifecycle_mask": symbol != "PF_UUSD",
                    "trade_coverage": symbol != "PF_UUSD",
                    "mark_coverage": symbol != "PF_UUSD",
                    "analytics_coverage": symbol != "PF_UUSD",
                })
            path = Path(record["path"])
            pq.write_table(pa.Table.from_pylist(rows), path)
            record.update({"rows": len(rows), "sha256": sha256_file(path)})
        atomic_write_json(sources.feature_manifest_path, feature_manifest)

        threshold_document = json.loads(sources.fold_thresholds_path.read_text(encoding="utf-8"))
        for model_id, model in threshold_document["models"].items():
            evaluation_start = datetime.fromisoformat(model["evaluation_start"].replace("Z", "+00:00"))
            model.update({
                "training_start": "2023-01-01T00:00:00Z",
                "training_end": (evaluation_start - timedelta(hours=12)).isoformat().replace("+00:00", "Z"),
                "thresholds": {
                    "trade_abs_q80": 10.0, "trade_abs_q100": 500.0,
                    "mark_abs_q80": 10.0, "mark_abs_q100": 500.0,
                    "oi_q0": -0.12, "oi_q20": -0.01,
                    "liquidation_q80": 1.0, "liquidation_q100": 5.0,
                },
            })
        atomic_write_json(sources.fold_thresholds_path, threshold_document)

        acquisition = root / "acquisition.csv"
        acquisition_fields = [
            "dataset", "symbol", "status", "rankable_pre_holdout", "contains_protected_period",
            "parquet_path", "parquet_sha256", "rows", "chunk_start", "chunk_end",
        ]
        acquisition_rows = []
        for symbol, decision in sorted(event_decisions.items()):
            rows = []
            for index in range(90):
                open_ts = decision + timedelta(minutes=5 * index)
                price = 100.0 + index * 0.01
                rows.append({
                    "time": int(open_ts.timestamp() * 1000),
                    "open": price, "high": price + 0.02, "low": price - 0.02,
                    "close": price + 0.01, "volume": 1.0,
                    "source_url": "fixture", "venue_symbol": symbol,
                    "chunk_start_utc": decision.isoformat(),
                    "chunk_end_utc": (decision + timedelta(hours=8)).isoformat(),
                    "resolution": "5m", "historical_backfill": True,
                    "rankable_pre_holdout": True, "contains_protected_period": False,
                })
            path = root / f"candles-{symbol}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            for dataset in ("historical_trade_candles_5m", "historical_mark_candles_5m"):
                acquisition_rows.append({
                    "dataset": dataset, "symbol": symbol, "status": "downloaded",
                    "rankable_pre_holdout": "True", "contains_protected_period": "False",
                    "parquet_path": str(path), "parquet_sha256": sha256_file(path),
                    "rows": len(rows), "chunk_start": decision.isoformat(),
                    "chunk_end": (decision + timedelta(hours=8)).isoformat(),
                })
        with acquisition.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=acquisition_fields)
            writer.writeheader(); writer.writerows(acquisition_rows)

        funding = root / "funding.zip"
        with zipfile.ZipFile(funding, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for symbol, decision in sorted(event_decisions.items()):
                timestamp = (decision + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
                archive.writestr(
                    f"rankable_2023_2025/{symbol}.csv",
                    f"timestamp,tradeable,absolute_rate,relative_rate\n{timestamp},{symbol},0.001,0.00001\n",
                )
        feature_schema = root / "feature-schema.json"
        atomic_write_json(feature_schema, {"schema": "fixture_kda02b_feature_schema_v1"})

        role_paths = {
            "stage20_kda02b_event_tape_manifest": sources.event_manifest_path,
            "stage20_kda02b_fold_local_thresholds": sources.fold_thresholds_path,
            "stage8a_feature_cache_manifest": sources.feature_manifest_path,
            "stage8a_shared_feature_schema": feature_schema,
            "campaign_universe_reconciliation": sources.universe_path,
            "stage14_kda02b_retention_boundary": sources.retention_boundary_path,
            "kraken_acquisition_manifest": acquisition,
            "rankable_funding_package": funding,
        }
        records = [{
            "role": role, "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        } for role, path in sorted(role_paths.items())]
        authority = {
            "source_manifest_sha256": "a" * 64,
            "pit_universe_sha256": "b" * 64,
            "funding_manifest_sha256": "c" * 64,
            "cache_contract_sha256": "d" * 64,
            "fold_graph_sha256": "e" * 64,
            "rankable_funding_package_sha256": sha256_file(funding),
            "source_records": records,
            "kda02b_authority_inventory_sha256": "f" * 64,
        }
        atomic_write_json(sources.authority_path, authority)
        index_root = root / "index"
        build_kda02b_lazy_population_index(
            sources=sources, output_root=index_root, expectations=expected,
            selected_models=("Q_2024Q1", "Q_2024Q2"),
        )
        return sources, expected, index_root

    def test_shadow_stream_emits_exact_frames_and_typed_unavailable_without_outcome_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources, expected, index_root = self._fixture(root)
            adapter = KDA02BLazyFamilyInputAdapter(
                index_root=index_root, authority_path=sources.authority_path,
                repository_root=root, mode=SHADOW_MODE, expectations=expected,
            )
            records = list(adapter.stream())
            eligible = [record for record in records if record.status == "eligible"]
            unavailable = [record for record in records if record.status == "typed_unavailable"]
            self.assertEqual(3, len(eligible)); self.assertEqual(1, len(unavailable))
            self.assertIsNone(unavailable[0].frame)
            self.assertEqual("stage14_kda02b_final_eligible_false", unavailable[0].unavailable_reason)
            self.assertTrue(all(record.frame is not None for record in eligible))
            ordering = [(record.symbol, record.outer_fold_id, record.decision_ts, record.cell_id, record.event_id) for record in records]
            self.assertEqual(sorted(ordering), ordering)
            for record in eligible:
                frame = record.frame
                assert frame is not None
                frame.validate()
                self.assertEqual((), frame.funding)
                self.assertTrue(frame.metadata["shadow_no_outcome_execution_schedule_only"])
                self.assertEqual(0, frame.metadata["real_post_entry_price_rows_opened"])
                self.assertFalse(frame.metadata["economic_outcomes_opened"])
                self.assertEqual({bar.open for bar in frame.five_minute_bars}, {100.0})
                self.assertIsInstance(kda02b_adjudication.evaluate(frame, {
                    "stage20_cell_id": record.cell_id,
                    "adjudication_variant": "identity_replay",
                }), list)
            same_decision = [record.frame for record in eligible if record.symbol == "PF_AUSD"]
            self.assertIs(same_decision[0].metadata["kda02b_feature_history"], same_decision[1].metadata["kda02b_feature_history"])
            self.assertEqual({
                "eligible_frames": 3,
                "typed_unavailable_rows_without_frames": 1,
                "mode": SHADOW_MODE,
                "economic_outcomes_opened": False,
                "status": "pass",
            }, adapter.last_reconciliation)

    def test_economic_mode_requires_explicit_hash_and_then_uses_authorized_ohlc_and_funding(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources, expected, index_root = self._fixture(root)
            with self.assertRaisesRegex(KDA02BLazyFamilyInputError, "hash-bound authorization"):
                KDA02BLazyFamilyInputAdapter(
                    index_root=index_root, authority_path=sources.authority_path,
                    repository_root=root, mode=ECONOMIC_MODE, expectations=expected,
                )
            adapter = KDA02BLazyFamilyInputAdapter(
                index_root=index_root, authority_path=sources.authority_path,
                repository_root=root, mode=ECONOMIC_MODE,
                economic_authorization_sha256="1" * 64, expectations=expected,
            )
            eligible = [record for record in adapter.stream() if record.status == "eligible"]
            self.assertEqual(3, len(eligible))
            for record in eligible:
                frame = record.frame
                assert frame is not None
                horizon_hours = int(frame.metadata["stage20_tape_horizon"].removesuffix("h"))
                self.assertGreater(len({bar.open for bar in frame.five_minute_bars}), 1)
                self.assertGreaterEqual(frame.five_minute_bars[-1].open_ts, frame.decision_ts + timedelta(hours=horizon_hours + 1))
                self.assertTrue(frame.funding)
                self.assertTrue(frame.metadata["economic_outcomes_opened"])
                self.assertGreater(frame.metadata["real_post_entry_price_rows_opened"], 0)

    def test_unknown_mode_and_index_authority_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources, expected, index_root = self._fixture(root)
            with self.assertRaisesRegex(KDA02BLazyFamilyInputError, "unsupported"):
                KDA02BLazyFamilyInputAdapter(
                    index_root=index_root, authority_path=sources.authority_path,
                    repository_root=root, mode="implicit", expectations=expected,
                )
            sources.authority_path.write_bytes(sources.authority_path.read_bytes() + b" ")
            with self.assertRaisesRegex(KDA02BLazyFamilyInputError, "population authority failed"):
                KDA02BLazyFamilyInputAdapter(
                    index_root=index_root, authority_path=sources.authority_path,
                    repository_root=root, expectations=expected,
                )


if __name__ == "__main__":
    unittest.main()
