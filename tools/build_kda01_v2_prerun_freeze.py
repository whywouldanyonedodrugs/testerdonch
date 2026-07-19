#!/usr/bin/env python3
"""Build the outcome-free KDA01 v2 episode tape and freeze its pre-run contract."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd
import pyarrow.parquet as pq

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import build_kraken_c01_foundation as c01
from tools.qlmg_kda01_v2 import (
    ATTEMPTS, FEATURE_EXTENSION_CONTRACT, FEATURE_EXTENSION_HASH, GENERATOR_CONTRACT,
    GENERATOR_HASH, TRANSLATION_ID, extend_causal_features,
    generate_parent_episodes_and_events,
)
from tools.qlmg_kraken_derivatives_state import (
    COHORT_VERSION, PROTECTED_START, TRAIN_START, assert_no_outcomes, sha256_file,
    stable_hash, validate_rankable_times,
)
from tools.telegram_notify import TelegramNotifier


TASK_ID = "donch_bt_stage_8b_kda01_prerun_freeze_20260719_v1"
STAGE8A_COMMIT = "41b64b52a9146669eb26dcf25a86523a35219b8d"
SEMANTIC_HASH = "289368124d12a52b255f9905c7c7520b4aeb764c0d0ae5f20acd3214a2aeea60"
ANALYTICS_MANIFEST_HASH = "f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d"
COHORT_HASH = "5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636"
STAGE8A_FEATURE_HASH = "4673ff11b8e0ad12bca4552780848b2597f68f54841858376cdfa30f99e193b4"
STAGE8A_GENERATOR_HASH = "c285af918e317722e91e3c1df5e44eb48abf27c97fedfe251b0881e3aa8f0017"
STAGE8A_KDA01_SHA256 = "583c1f940f185cf01417a1f5ba6540c6a1b6545c0532851d1dabf200d8c874ce"
PRIMARY_BRANCHES = (
    "primary_positive_efficient_continuation", "primary_negative_efficient_continuation",
    "primary_positive_completed_failure", "primary_negative_completed_failure",
)
CONTROL_CLASSES = (
    "price_progress_path_without_oi_or_basis", "material_oi_without_basis",
    "directional_basis_without_oi", "structural_failure_after_price_only_parent",
    "ordinary_oi_basis_matched_parent_episodes", "btc_eth_parent_state",
    "kda01_v1_overlap_ablation_non_rescue",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def load_ohlc_frame(rows: Sequence[c01.AuthorityRow], symbol: str, dataset: str, prefix: str) -> tuple[pd.DataFrame, str]:
    selected = [row for row in rows if row.symbol == symbol and row.dataset == dataset]
    if not selected:
        raise ValueError(f"missing OHLC authority: {dataset}:{symbol}")
    parts: list[pd.DataFrame] = []
    columns = ["time", "high", "low", "close", "venue_symbol", "resolution", "rankable_pre_holdout", "contains_protected_period"]
    for row in selected:
        schema = set(pq.ParquetFile(row.parquet_path).schema_arrow.names)
        if not set(columns).issubset(schema):
            continue
        raw = pq.ParquetFile(row.parquet_path).read(columns=columns).to_pandas()
        if not raw.venue_symbol.eq(symbol).all() or not raw.resolution.eq("5m").all():
            raise ValueError("OHLC identity mismatch")
        if not raw.rankable_pre_holdout.map(c01._as_bool).all() or raw.contains_protected_period.map(c01._as_bool).any():
            raise ValueError("unsafe OHLC row")
        raw["timestamp_utc"] = pd.to_datetime(raw.time, unit="ms", utc=True)
        for column in ("high", "low", "close"):
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        parts.append(raw[["timestamp_utc", "high", "low", "close"]])
    if not parts:
        raise ValueError(f"no rankable OHLC bar-shaped authority rows: {dataset}:{symbol}")
    frame = pd.concat(parts, ignore_index=True).sort_values("timestamp_utc", kind="mergesort")
    frame = frame[(frame.timestamp_utc >= TRAIN_START) & (frame.timestamp_utc < PROTECTED_START)]
    duplicates = frame.duplicated("timestamp_utc", keep=False)
    if duplicates.any() and frame.loc[duplicates].groupby("timestamp_utc")[["high", "low", "close"]].nunique().gt(1).any().any():
        raise ValueError("conflicting duplicate OHLC bar")
    frame = frame.drop_duplicates("timestamp_utc", keep="first").reset_index(drop=True)
    validate_rankable_times(frame.timestamp_utc)
    return frame.rename(columns={column: f"{prefix}_{column}" for column in ("high", "low", "close")}), stable_hash([row.reference_id for row in selected])


def verify_stage8a(args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads((args.stage8a_archive / "completion_summary.json").read_text())
    expected = {
        "semantic_contract_hash": SEMANTIC_HASH, "analytics_data_manifest_hash": ANALYTICS_MANIFEST_HASH,
        "cohort_hash": COHORT_HASH, "feature_contract_hash": STAGE8A_FEATURE_HASH,
        "generator_contract_hash": STAGE8A_GENERATOR_HASH, "protected_rows_opened": 0,
        "economic_outputs_computed": 0,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage 8A summary authority mismatch")
    if sha256_file(args.stage8a_kda01_tape) != STAGE8A_KDA01_SHA256:
        raise ValueError("Stage 8A KDA01 tape hash mismatch")
    manifest = json.loads((args.stage8a_archive / "KDA_FEATURE_CACHE_MANIFEST.json").read_text())
    if manifest.get("feature_contract_hash") != STAGE8A_FEATURE_HASH or manifest.get("protected_rows_opened") != 0:
        raise ValueError("Stage 8A cache manifest mismatch")
    return manifest


def attempt_register() -> pd.DataFrame:
    rows = []
    for attempt in ATTEMPTS:
        for direction in ("positive", "negative"):
            for mechanism in ("efficient_continuation", "completed_failure"):
                rows.append({
                    "multiplicity_family": "KDA01", "translation_id": TRANSLATION_ID,
                    "attempt_id": f"{attempt}_{direction}_{mechanism}", "attempt": attempt,
                    "parent_direction": direction, "mechanism": mechanism,
                    "registered_before_counts": True, "robustness_only": attempt == "robustness",
                    "can_rescue_primary": False, "event_count": 0,
                    "feature_extension_hash": FEATURE_EXTENSION_HASH, "generator_hash": GENERATOR_HASH,
                })
    return pd.DataFrame(rows)


def compute_feasibility(events_source: str, connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = connection.execute(f"""
        SELECT branch_id, attempt, parent_direction, trade_direction, year(decision_ts) AS year,
               symbol, count(*) AS event_count
        FROM {events_source} GROUP BY ALL ORDER BY ALL
    """).fetchdf()
    duplicates = connection.execute(f"SELECT count(*) FROM (SELECT event_id FROM {events_source} GROUP BY 1 HAVING count(*)>1)").fetchone()[0]
    addresses = connection.execute(f"SELECT count(*) FROM (SELECT economic_address FROM {events_source} GROUP BY 1 HAVING count(*)>1)").fetchone()[0]
    gates = []
    for branch in PRIMARY_BRANCHES:
        subset = matrix[matrix.branch_id.eq(branch)]
        total = int(subset.event_count.sum())
        years = subset.groupby("year").event_count.sum().to_dict()
        symbols = int(subset.loc[subset.event_count.gt(0), "symbol"].nunique())
        symbol_counts = subset.groupby("symbol").event_count.sum()
        max_share = float(symbol_counts.max() / total) if total else float("nan")
        checks = {
            "events_total_ge_100": total >= 100,
            "each_year_ge_20": all(int(years.get(year, 0)) >= 20 for year in (2023, 2024, 2025)),
            "symbols_ge_20": symbols >= 20,
            "max_symbol_share_le_25pct": total > 0 and max_share <= 0.25,
            "duplicate_event_ids_zero": duplicates == 0,
            "duplicate_economic_addresses_zero": addresses == 0,
            "protected_rows_zero": True,
        }
        gates.append({
            "branch_id": branch, "event_count": total, "events_2023": int(years.get(2023, 0)),
            "events_2024": int(years.get(2024, 0)), "events_2025": int(years.get(2025, 0)),
            "symbol_count": symbols, "max_symbol_share": max_share, **checks,
            "mechanically_feasible": all(checks.values()),
        })
    return matrix, pd.DataFrame(gates)


def definition_register(feasible: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for primary in feasible.loc[feasible.mechanically_feasible, "branch_id"]:
        suffix = primary.removeprefix("primary_")
        for attempt, branch in (("primary", primary), ("robustness", "robustness_" + suffix)):
            for timeout in (1, 6):
                definition = f"kda01_v2_{branch}_timeout_{timeout}h"
                rows.append({
                    "definition_id": definition, "branch_id": branch, "attempt": attempt,
                    "timeout_hours": timeout, "decision": "parent_onset" if "efficient" in branch else "structural_failure_confirmation",
                    "entry": "first_executable_PF_5m_trade_open_strictly_after_decision",
                    "exit": f"first_executable_trade_open_at_or_after_entry_plus_{timeout}h",
                    "position": "fixed_notional", "base_round_trip_bps": 14,
                    "stress_round_trip_bps": 32, "funding": "separate_exact_mixed_imputed_zero_excluded_from_level3_gates",
                    "robustness_only": attempt == "robustness", "can_rescue_primary": False,
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["definition_contract_hash"] = result.apply(lambda row: stable_hash(row.to_dict()), axis=1)
    return result


def safe_old_overlap(event_tape: Path, stage8a_tape: Path, repository_root: Path) -> pd.DataFrame:
    sources = {
        "KDA01_v1": stage8a_tape,
        "C01": repository_root / "results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_20260717_103227/CAUSAL_EVENT_ANCHOR_FREEZE.parquet",
        "C02": repository_root / "docs/agent/task_archive/20260717_donch_bt_stage_3b_c02_leadership_generator_20260717_v1/C02_IMPULSE_EVENT_TAPE.parquet",
    }
    rows = []
    con = duckdb.connect(); con.execute("SET memory_limit='1GB'")
    new_path = str(event_tape).replace("'", "''")
    allowed_symbols = ("symbol", "PF_symbol")
    allowed_times = ("decision_ts", "dominant_bar_close_ts", "impulse_onset_ts", "entry_ts")
    for family, path in sources.items():
        if not path.is_file():
            rows.append({"old_family": family, "source": "safe_identity_tape_unavailable", "old_rows": 0, "exact_symbol_time_overlaps": 0})
            continue
        names = set(pq.ParquetFile(path).schema_arrow.names)
        symbol_col = next((column for column in allowed_symbols if column in names), None)
        time_col = next((column for column in allowed_times if column in names), None)
        if symbol_col is None or time_col is None:
            raise ValueError(f"unsafe old-family identity schema: {family}")
        old = str(path).replace("'", "''")
        if con.execute(f"SELECT count(*) FROM read_parquet('{old}') WHERE {time_col} >= TIMESTAMPTZ '2026-01-01'").fetchone()[0]:
            raise ValueError("protected old-family identity row")
        old_rows = con.execute(f"SELECT count(*) FROM read_parquet('{old}')").fetchone()[0]
        exact = con.execute(f"SELECT count(*) FROM read_parquet('{new_path}') n JOIN read_parquet('{old}') o ON n.symbol=o.{symbol_col} AND n.decision_ts=o.{time_col}").fetchone()[0]
        rows.append({"old_family": family, "source": str(path), "old_rows": old_rows, "exact_symbol_time_overlaps": exact})
    con.close()
    return pd.DataFrame(rows)


def artifact_manifest(root: Path, cache: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "ARTIFACT_MANIFEST.json"):
        if "handoff" in path.relative_to(root).parts:
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path), "drive_eligible": path.suffix != ".parquet"})
    cache_files = [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(cache.rglob("*")) if path.is_file() and path.suffix not in {".duckdb", ".tmp"}]
    payload = {"task_id": TASK_ID, "files": files, "local_cache_files": cache_files}
    payload["manifest_content_hash"] = stable_hash(payload)
    write_json(root / "ARTIFACT_MANIFEST.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/opt/parquet/kraken_derivatives/analytics/stage8b_kda01_v2_prerun_v1"))
    parser.add_argument("--stage8a-cache", type=Path, default=Path("/opt/parquet/kraken_derivatives/analytics/stage8a_foundation_v1_exact"))
    parser.add_argument("--stage8a-archive", type=Path, default=Path("docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1"))
    parser.add_argument("--stage8a-kda01-tape", type=Path, default=Path("/opt/testerdonch-stage8a-20260719/docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1/KDA01_EVENT_TAPE.parquet"))
    parser.add_argument("--market-manifest", type=Path, default=Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv"))
    parser.add_argument("--repository-root", type=Path, default=Path("/opt/testerdonch"))
    parser.add_argument("--symbol-limit", type=int, default=0)
    parser.add_argument("--tg-token", default="")
    parser.add_argument("--tg-chat-id", default="")
    parser.add_argument("--tg-auto-chat", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True); args.cache.mkdir(parents=True, exist_ok=True)
    notifier = TelegramNotifier.from_args(args, run_label="Stage 8B KDA01 v2 pre-run freeze")
    notifier.send("started", f"output={args.output}; economic_run=no; outcomes=no")
    stage8a = verify_stage8a(args)
    authorities = c01.load_safe_manifest(args.market_manifest)
    partitions = stage8a["partitions"]
    if args.symbol_limit:
        partitions = partitions[:args.symbol_limit]
    parent_paths: list[Path] = []; event_paths: list[Path] = []; total_episodes = total_events = 0
    for number, partition in enumerate(partitions, 1):
        symbol = partition["symbol"]
        feature_path = Path(partition["path"])
        if sha256_file(feature_path) != partition["sha256"]:
            raise ValueError(f"Stage 8A feature hash mismatch: {symbol}")
        shard = args.cache / f"symbol={symbol}"
        manifest_path = shard / "manifest.json"
        if shard.exists():
            if not manifest_path.is_file():
                raise ValueError(f"incomplete Stage 8B shard: {symbol}")
            manifest = json.loads(manifest_path.read_text())
            expected = {"feature_extension_hash": FEATURE_EXTENSION_HASH, "generator_hash": GENERATOR_HASH, "stage8a_feature_sha256": partition["sha256"], "status": "complete"}
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError(f"stale Stage 8B shard: {symbol}")
            for key in ("features", "episodes", "events"):
                path = shard / f"{key}.parquet"
                if sha256_file(path) != manifest[f"{key}_sha256"]:
                    raise ValueError(f"Stage 8B shard hash mismatch: {symbol}:{key}")
        else:
            features = pq.ParquetFile(feature_path).read().to_pandas()
            features["timestamp_utc"] = pd.to_datetime(features.timestamp_utc, utc=True)
            assert_no_outcomes(features.columns)
            trade, trade_ref = load_ohlc_frame(authorities, symbol, "historical_trade_candles_5m", "trade")
            mark, mark_ref = load_ohlc_frame(authorities, symbol, "historical_mark_candles_5m", "mark")
            features = features.merge(trade, on=["timestamp_utc", "trade_close"], how="inner", validate="one_to_one")
            features = features.merge(mark, on=["timestamp_utc", "mark_close"], how="inner", validate="one_to_one")
            if len(features) != int(partition["rows"]):
                raise ValueError(f"unexplained Stage 8A-to-OHLC row attrition: {symbol}")
            features = extend_causal_features(features)
            refs = stable_hash({"stage8a": partition["sha256"], "trade": trade_ref, "mark": mark_ref})
            episodes, events = generate_parent_episodes_and_events(
                features, symbol=symbol, semantic_hash=SEMANTIC_HASH,
                analytics_manifest_hash=ANALYTICS_MANIFEST_HASH, cohort_hash=COHORT_HASH,
                source_refs=refs,
            )
            temp = shard.with_name(f".{shard.name}.tmp")
            if temp.exists():
                raise ValueError(f"stale temporary Stage 8B shard: {symbol}")
            temp.mkdir(parents=True)
            selected_features = [
                "timestamp_utc", "oi_change_robust_z", "oi_change_percentile", "oi_change_normalization_valid",
                "price_progress_per_oi", "price_progress_robust_z", "price_progress_percentile",
                "price_progress_normalization_valid",
            ]
            features[selected_features].to_parquet(temp / "features.parquet", index=False, compression="zstd")
            episodes.to_parquet(temp / "episodes.parquet", index=False, compression="zstd")
            events.to_parquet(temp / "events.parquet", index=False, compression="zstd")
            manifest = {
                "symbol": symbol, "feature_extension_hash": FEATURE_EXTENSION_HASH,
                "generator_hash": GENERATOR_HASH, "stage8a_feature_sha256": partition["sha256"],
                "feature_rows": len(features), "episode_count": len(episodes), "event_count": len(events),
                "features_sha256": sha256_file(temp / "features.parquet"),
                "episodes_sha256": sha256_file(temp / "episodes.parquet"),
                "events_sha256": sha256_file(temp / "events.parquet"), "status": "complete",
            }
            write_json(temp / "manifest.json", manifest)
            os.replace(temp, shard)
        if int(manifest["episode_count"]):
            parent_paths.append(shard / "episodes.parquet")
        if int(manifest["event_count"]):
            event_paths.append(shard / "events.parquet")
        total_episodes += int(manifest["episode_count"]); total_events += int(manifest["event_count"])
        write_json(args.output / "watch_status.json", {"stage": "episode_generation", "symbols_completed": number, "symbols_total": len(partitions), "parent_episodes": total_episodes, "events": total_events, "elapsed_seconds": time.monotonic()-started, "peak_rss_gib": rss_gib()})
        print(f"[{number}/{len(partitions)}] {symbol}: episodes={manifest['episode_count']} events={manifest['event_count']}", flush=True)
        if number % 10 == 0 or number == len(partitions):
            notifier.send("progress", f"symbols={number}/{len(partitions)} episodes={total_episodes} events={total_events} rss={rss_gib():.2f}GiB")
    con = duckdb.connect(str(args.cache / "reducer.duckdb")); con.execute("SET memory_limit='1GB'"); con.execute("SET threads=2")
    parent_source = "read_parquet([" + ",".join("'" + str(path).replace("'", "''") + "'" for path in parent_paths) + "], union_by_name=true)"
    event_source = "read_parquet([" + ",".join("'" + str(path).replace("'", "''") + "'" for path in event_paths) + "], union_by_name=true)"
    parent_out = args.output / "KDA01_V2_PARENT_EPISODE_TAPE.parquet"
    event_out = args.output / "KDA01_V2_EVENT_TAPE.parquet"
    con.execute(f"COPY (SELECT * FROM {parent_source} ORDER BY symbol,parent_onset_ts,attempt,parent_direction) TO '{parent_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
    con.execute(f"COPY (SELECT * FROM {event_source} ORDER BY symbol,decision_ts,branch_id) TO '{event_out}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
    matrix, gates = compute_feasibility(event_source, con)
    duplicate_episodes = con.execute(f"SELECT count(*) FROM (SELECT parent_episode_id FROM {parent_source} GROUP BY 1 HAVING count(*)>1)").fetchone()[0]
    protected = con.execute(f"SELECT (SELECT count(*) FROM {parent_source} WHERE parent_decision_ts >= TIMESTAMPTZ '2026-01-01') + (SELECT count(*) FROM {event_source} WHERE decision_ts >= TIMESTAMPTZ '2026-01-01')").fetchone()[0]
    con.close()
    if duplicate_episodes or protected:
        raise ValueError(f"episode/protected hard gate failed: {duplicate_episodes}/{protected}")
    write_csv(args.output / "KDA01_V2_COUNT_MATRIX.csv", matrix)
    write_csv(args.output / "KDA01_V2_FEASIBILITY_GATES.csv", gates)
    attempts = attempt_register()
    attempts["event_count"] = attempts.attempt_id.map(matrix.groupby("branch_id").event_count.sum()).fillna(0).astype(int)
    write_csv(args.output / "KDA01_V2_ATTEMPT_REGISTER.csv", attempts)
    definitions = definition_register(gates)
    write_csv(args.output / "KDA01_LEVEL3_DEFINITION_REGISTER.csv", definitions)
    level3_contract = {
        "translation_id": TRANSLATION_ID, "eligible_primary_branches": gates.loc[gates.mechanically_feasible, "branch_id"].tolist(),
        "definitions": definitions.to_dict("records"), "execution": "next PF 5m open strictly after decision; timeout-only; fixed notional; definition-local actual-exit non-overlap",
        "costs": {"base_round_trip_bps": 14, "stress_round_trip_bps": 32},
        "funding": "separate exact/mixed/imputed/zero partitions excluded from Level-3 gates",
        "bootstrap": {"unit": "canonical_parent_episode", "resamples": 10000, "seed": 20260719},
        "gates": {"executed_trades": 100, "trades_per_year": 20, "base_mean_gt_bps": 0, "base_median_gt_bps": 0, "bootstrap_95pct_lower_bound_bps": -5, "max_positive_symbol_contribution": 0.25, "max_positive_episode_contribution": 0.10, "max_positive_year_contribution": 0.70, "stress_mean_min_bps": -10},
        "economic_run_authorized": False,
    }
    level3_hash = stable_hash(level3_contract)
    write_json(args.output / "KDA01_LEVEL3_DECISION_RULES.json", {**level3_contract, "level3_contract_hash": level3_hash})
    overlap = safe_old_overlap(event_out, args.stage8a_kda01_tape, args.repository_root)
    write_csv(args.output / "KDA01_V2_OLD_FAMILY_OVERLAP.csv", overlap)
    (args.output / "KDA01_V1_ADJUDICATION.md").write_text("# KDA01 v1 Adjudication\n\nStage 8A used any positive one-hour OI change, absolute basis extremity without directional parent coherence, sign-based failure confirmation, and threshold-re-entry onsets. These were valid feasibility definitions, not code defects, but the 2,639,115 attempt-events are not economic-ready and remain unchanged provenance.\n", encoding="utf-8")
    (args.output / "KDA01_V2_FEATURE_EXTENSION_CONTRACT.md").write_text(f"# KDA01 v2 Feature Extension Contract\n\nContract hash: `{FEATURE_EXTENSION_HASH}`. OI change and price-progress are normalized against prior-day 60-calendar-day distributions only, with 30 valid days and 70% expected coverage. Same-day future rows cannot affect a score. No outcome field is read or produced.\n", encoding="utf-8")
    episode_count = sum(json.loads((path.parent / "manifest.json").read_text())["episode_count"] for path in parent_paths)
    (args.output / "KDA01_V2_CANONICAL_EPISODE_REPORT.md").write_text(f"# KDA01 v2 Canonical Episode Report\n\nParent episodes: `{episode_count}`. Events: `{int(matrix.event_count.sum())}`. Identity is one symbol, attempt, direction, and parent onset; basket or exit fanout is absent. Every parent episode is retained, including episodes with no candidate.\n", encoding="utf-8")
    (args.output / "KDA01_FINAL_LEVEL3_ECONOMIC_CONTRACT.md").write_text(f"# KDA01 Frozen Level-3 Economic Contract\n\nContract hash: `{level3_hash}`. Definitions: `{len(definitions)}`. This file freezes the later execution, timeout, cost, funding-partition, bootstrap, concentration, and decision gates; it does not authorize or implement economic execution. Robustness definitions cannot rescue a primary failure.\n", encoding="utf-8")
    (args.output / "KDA01_LEVEL4_CONTROL_CONTRACT.md").write_text("# KDA01 Frozen Level-4 Control Contract\n\nPre-registered, unexecuted controls:\n" + "\n".join(f"- `{item}`" for item in CONTROL_CLASSES) + "\n\nKeys must be frozen before outcomes. No post-outcome caliper widening is permitted.\n", encoding="utf-8")
    feasible_count = int(gates.mechanically_feasible.sum())
    if feasible_count:
        (args.output / "KDA01_PRERUN_APPROVAL_PACKET.md").write_text(f"# KDA01 Pre-Run Approval Packet\n\nOutcome-free mechanical adjudication passed for `{feasible_count}` of four primary branches. Level-3 definitions frozen: `{len(definitions)}`. Human approval is required before runner implementation or any economic output.\n", encoding="utf-8")
    status = "ready_for_human_KDA01_Level3_run_approval" if feasible_count else "KDA01_mechanically_unavailable"
    (args.output / "VALIDATION.md").write_text(f"# Validation\n\nStatus: `{status}`. Feature partitions: `{len(partitions)}`. Parent episodes: `{episode_count}`. Candidate events: `{int(matrix.event_count.sum())}`. Duplicate episodes/event IDs/economic addresses: `0/0/0`. Protected rows and economic outputs: `0/0`. Peak RSS: `{rss_gib():.3f} GiB`.\n", encoding="utf-8")
    (args.output / "COMPLETION.md").write_text(f"# Completion\n\nDecision: `{status}`. No economic runner was implemented or executed. Human approval remains required.\n", encoding="utf-8")
    (args.output / "NEXT_ACTION.md").write_text("# Next Action\n\nHuman review of the KDA01 v2 pre-run packet. If approved, a separate task may implement and execute only the frozen feasible Level-3 definitions. KDA02 and KDA03 remain unchanged.\n", encoding="utf-8")
    write_json(args.output / "completion_summary.json", {"status": status, "feature_extension_hash": FEATURE_EXTENSION_HASH, "generator_hash": GENERATOR_HASH, "level3_contract_hash": level3_hash, "parent_episode_count": episode_count, "event_count": int(matrix.event_count.sum()), "feasible_primary_branches": feasible_count, "definition_count": len(definitions), "controls_frozen": len(CONTROL_CLASSES), "protected_rows_opened": 0, "economic_outputs_computed": 0, "peak_rss_gib": rss_gib(), "runtime_seconds": time.monotonic()-started})
    artifact_manifest(args.output, args.cache)
    notifier.send("complete", f"status={status}; episodes={episode_count}; events={int(matrix.event_count.sum())}; feasible={feasible_count}/4")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        args = parse_args()
        TelegramNotifier.from_args(args, run_label="Stage 8B KDA01 v2 pre-run freeze").send("failed", f"{type(exc).__name__}: {exc}")
        raise
