from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.core_liquid_campaign.canonical import atomic_write_json, canonical_hash, sha256_file
from tools.core_liquid_campaign.family_engines.kda02b_adjudication import cell_contract
from tools.core_liquid_campaign.kda02b_population_index import (
    KDA02BPopulationIndexError,
    LOCAL_UNAVAILABLE_REASON,
    PopulationExpectations,
    PopulationSources,
    build_kda02b_lazy_population_index,
    sources_from_authority,
    validate_kda02b_lazy_population_index,
)


UTC = timezone.utc


class KDA02BPopulationIndexTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[PopulationSources, PopulationExpectations]:
        models = {
            "Q_2024Q1": {
                "evaluation_start": "2024-01-01T00:00:00Z",
                "evaluation_end": "2024-04-01T00:00:00Z",
            },
            "Q_2024Q2": {
                "evaluation_start": "2024-04-01T00:00:00Z",
                "evaluation_end": "2024-07-01T00:00:00Z",
            },
        }
        thresholds = root / "thresholds.json"
        atomic_write_json(thresholds, {"models": models})
        symbols = {"PF_AUSD": True, "PF_BUSD": True, "PF_UUSD": False}
        universe = root / "universe.csv"
        with universe.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "PF_symbol", "final_campaign_eligible", "KDA02B_final_eligible",
                "causal_lookback_eligibility",
            ])
            writer.writeheader()
            for symbol, eligible in symbols.items():
                writer.writerow({
                    "PF_symbol": symbol, "final_campaign_eligible": "True",
                    "KDA02B_final_eligible": str(eligible),
                    "causal_lookback_eligibility": str(eligible),
                })
        retention = root / "retention.csv"
        with retention.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["symbol", "truncation_status"])
            writer.writeheader()
            for symbol in symbols:
                writer.writerow({"symbol": symbol, "truncation_status": "fixture"})

        feature_records = []
        decisions = {
            "PF_AUSD": datetime(2024, 2, 1, 0, 5, tzinfo=UTC),
            "PF_BUSD": datetime(2024, 5, 1, 0, 5, tzinfo=UTC),
            "PF_UUSD": datetime(2024, 2, 2, 0, 5, tzinfo=UTC),
        }
        for symbol, eligible in symbols.items():
            timestamp = decisions[symbol] - timedelta(minutes=5)
            path = root / f"features-{symbol}.parquet"
            pq.write_table(pa.Table.from_pylist([{
                "timestamp_utc": timestamp,
                "trade_return_1h": -0.01,
                "mark_return_1h": -0.01,
                "oi_log_change_1h": -0.02,
                "liquidation_base_units_1h": 5.0,
                "liquidation_intensity_robust_z": 2.0,
                "liquidation_normalization_valid": True,
                "eligible": eligible,
                "known_lifecycle_mask": eligible,
                "trade_coverage": eligible,
                "mark_coverage": eligible,
                "analytics_coverage": eligible,
            }]), path)
            feature_records.append({"symbol": symbol, "path": str(path), "rows": 1, "sha256": sha256_file(path)})
        feature_manifest = root / "features.json"
        atomic_write_json(feature_manifest, {"partitions": feature_records})

        cells = ("KDA02B_009", "KDA02B_011")
        events = {
            "PF_AUSD": [
                (cells[0], "Q_2024Q1", "a-1"),
                (cells[1], "Q_2024Q1", "a-2"),
            ],
            "PF_BUSD": [(cells[0], "Q_2024Q2", "b-1")],
            "PF_UUSD": [(cells[0], "Q_2024Q1", "u-1")],
        }
        tape_records = []
        for symbol in symbols:
            tape_rows = []
            for cell, model, event_id in events[symbol]:
                axes = cell_contract(cell)["axes"]
                price_side = "short" if axes["price_state"] == "negative" else "long"
                side = price_side if axes["branch"] == "continuation" else ("long" if price_side == "short" else "short")
                tape_rows.append({
                    "event_id": event_id, "translation_id": f"translation-{event_id}",
                    "cell_id": cell, "family": "KDA02B", "symbol": symbol,
                    "decision_ts": decisions[symbol], "side": side,
                    "horizon": axes["horizon"], "model_id": model,
                    "onset_ts": None if event_id == "a-1" else decisions[symbol] - timedelta(minutes=5),
                })
            path = root / f"events-{symbol}.parquet"
            pq.write_table(pa.Table.from_pylist(tape_rows), path)
            tape_records.append({
                "symbol": symbol, "path": str(path), "bytes": path.stat().st_size,
                "rows": len(tape_rows), "sha256": sha256_file(path),
            })
        event_manifest = root / "events.json"
        atomic_write_json(event_manifest, {
            "status": "pass", "economic_outcome_reader_opened": False,
            "protected_rows_opened": 0, "files": tape_records,
        })

        execution = root / "execution.jsonl"
        rows = []
        for cell in cells:
            for variant in ("identity_replay", "price_only"):
                attempt_id = f"{cell}-{variant}"
                rows.append({
                    "family_id": "KDA02B_SURVIVOR_ADJUDICATION_V1",
                    "executable_attempt_id": attempt_id,
                    "canonical_economic_address_sha256": canonical_hash(attempt_id),
                    "config": {"stage20_cell_id": cell, "adjudication_variant": variant},
                })
        execution.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

        authority = root / "authority.json"
        role_paths = {
            "stage20_kda02b_event_tape_manifest": event_manifest,
            "stage20_kda02b_fold_local_thresholds": thresholds,
            "stage8a_feature_cache_manifest": feature_manifest,
            "campaign_universe_reconciliation": universe,
            "stage14_kda02b_retention_boundary": retention,
        }
        source_records = [{
            "role": role, "path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        } for role, path in sorted(role_paths.items())]
        atomic_write_json(authority, {"source_records": source_records})
        sources = PopulationSources(
            authority, execution, event_manifest, feature_manifest, universe, retention, thresholds,
        )
        expected = PopulationExpectations(
            configurations=4, cells=2, variants_per_cell=2, folds=2, symbols=3,
            eligible_symbols=2, unavailable_symbols=1, event_rows=4,
            eligible_event_rows=3, unavailable_event_rows=1,
            eligible_unique_symbol_decisions=2, unavailable_unique_symbol_decisions=1,
            eligible_dispatch_units=6, unavailable_dispatch_units=2,
            eligible_coverage_positions=16, unavailable_coverage_positions=8,
            unavailable_symbols_with_events=1, unavailable_symbols_without_events=0,
        )
        return sources, expected

    def test_complete_lazy_index_preserves_rows_multiplicity_and_local_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources, expected = self._fixture(root)
            output = root / "index"
            manifest = build_kda02b_lazy_population_index(
                sources=sources, output_root=output, expectations=expected,
                selected_models=("Q_2024Q1", "Q_2024Q2"),
            )
            self.assertEqual(4, manifest["counts"]["event_rows"])
            self.assertEqual(6, manifest["counts"]["eligible_dispatch_units"])
            self.assertEqual(2, manifest["counts"]["unavailable_dispatch_units"])
            self.assertEqual(16, manifest["counts"]["eligible_coverage_positions"])
            self.assertEqual(8, manifest["counts"]["unavailable_coverage_positions"])
            event_rows = pq.read_table(output / "KDA02B_EVENT_INDEX.parquet").to_pylist()
            unavailable = [row for row in event_rows if row["status"] == "typed_unavailable"]
            self.assertEqual(1, len(unavailable))
            self.assertIsNone(next(row for row in event_rows if row["event_id"] == "a-1")["onset_ts"])
            self.assertEqual(LOCAL_UNAVAILABLE_REASON, unavailable[0]["unavailable_reason"])
            self.assertEqual(unavailable[0]["decision_ts"] - timedelta(minutes=5), unavailable[0]["feature_timestamp_utc"])
            self.assertFalse(manifest["economic_outcomes_opened"])
            validate_kda02b_lazy_population_index(output, expected)

    def test_feature_and_authority_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources, expected = self._fixture(root)
            resolved = sources_from_authority(sources.authority_path, sources.execution_registry_path, root)
            self.assertEqual(sources.feature_manifest_path, resolved.feature_manifest_path)
            feature_manifest = json.loads(sources.feature_manifest_path.read_text(encoding="utf-8"))
            feature_path = Path(feature_manifest["partitions"][0]["path"])
            feature_path.write_bytes(feature_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(KDA02BPopulationIndexError, "feature partition hash differs"):
                build_kda02b_lazy_population_index(
                    sources=sources, output_root=root / "index", expectations=expected,
                    selected_models=("Q_2024Q1", "Q_2024Q2"),
                )
            sources.fold_thresholds_path.write_bytes(sources.fold_thresholds_path.read_bytes() + b" ")
            with self.assertRaisesRegex(KDA02BPopulationIndexError, "authority drift"):
                sources_from_authority(sources.authority_path, sources.execution_registry_path, root)

    def test_index_validation_detects_physical_drift(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            sources, expected = self._fixture(root)
            output = root / "index"
            build_kda02b_lazy_population_index(
                sources=sources, output_root=output, expectations=expected,
                selected_models=("Q_2024Q1", "Q_2024Q2"),
            )
            event_path = output / "KDA02B_EVENT_INDEX.parquet"
            event_path.write_bytes(event_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(KDA02BPopulationIndexError, "file drift"):
                validate_kda02b_lazy_population_index(output, expected)


if __name__ == "__main__":
    unittest.main()
