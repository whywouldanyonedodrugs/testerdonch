#!/usr/bin/env python3
"""Persistent packet-faithful Stage 20 Phase 2-5 campaign supervisor."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_kraken_c01_foundation as market
from tools.build_kda01_contract_closure import load_timestamp_only_bars
from tools.qlmg_stage16_campaign import canonical_sha256
from tools.qlmg_stage19_funding import Stage19FundingEngine
from tools.qlmg_stage20_campaign import (
    CAMPAIGN_ID, FUNDING_PACKAGE, FUNDING_PACKAGE_SHA256, MARKET_MANIFEST,
    PROTECTED_START, STAGE19_REL, Stage20Error, atomic_json, attach_open_prices,
    choose_beam, file_sha256, metric_row, resolve_timestamp_schedule,
    score_executions, stable_hash, temporal_models, utc_now,
)
from tools.qlmg_stage20_launch_gates import (
    assert_gate_binds, final_launch_boundary_audit, validate_gate,
    validate_source_manifest,
)
from tools.run_kda03_level3 import load_open_map, verified_trade_authority_hash
from tools.telegram_notify import TelegramNotifier


MAX_WORKERS = 4
MAX_WALL_SECONDS = 14_400
MAX_RSS_BYTES = 5 * 1024**3
MAX_OUTPUT_BYTES = 5 * 1024**3
HEARTBEAT_SECONDS = 1_800
EXPECTED_CELLS = {"KDA02B": 96, "KDA02C": 48, "KDX01": 42}
DEFAULT_LIMITS = {
    "max_workers": MAX_WORKERS, "max_wall_seconds": MAX_WALL_SECONDS,
    "max_rss_bytes": MAX_RSS_BYTES, "max_output_bytes": MAX_OUTPUT_BYTES,
    "heartbeat_seconds": HEARTBEAT_SECONDS,
}

_WORKER: dict[str, Any] = {}


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def process_tree_rss(root_pid: int) -> int:
    table: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text().splitlines()
            parent = int(next(line.split()[1] for line in status if line.startswith("PPid:")))
            rss_kib = int(next(line.split()[1] for line in status if line.startswith("VmRSS:")))
            table[int(entry.name)] = (parent, rss_kib * 1024)
        except (FileNotFoundError, PermissionError, StopIteration, ValueError):
            continue
    selected, frontier = {root_pid}, {root_pid}
    while frontier:
        children = {pid for pid, (parent, _) in table.items() if parent in frontier and pid not in selected}
        selected.update(children)
        frontier = children
    return sum(table.get(pid, (0, 0))[1] for pid in selected)


def _worker_init(event_root: str, output_root: str) -> None:
    stage19 = ROOT / STAGE19_REL
    contract = json.loads((stage19 / "FUNDING_COST_AND_COVERAGE_CONTRACT.json").read_text())
    _WORKER.update({
        "event_root": Path(event_root), "output_root": Path(output_root),
        "authority": market.load_safe_manifest(MARKET_MANIFEST),
        "funding": Stage19FundingEngine(
            FUNDING_PACKAGE, FUNDING_PACKAGE_SHA256,
            stage19 / "FUNDING_GAP_ALLOWANCE_TABLE.csv",
            contract["gap_allowance_table_sha256"],
        ),
        "models": temporal_models(json.loads((stage19 / "INNER_FOLD_MAP.json").read_text())),
    })


def _atomic_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, target)


def _score_job(job: dict[str, Any]) -> dict[str, Any]:
    symbol = job["symbol"]
    model_id = job["model_id"]
    family = job["family"]
    if model_id.startswith("Q_"):
        verify_outer_freeze(job)
    event_path = _WORKER["event_root"] / "events" / f"{symbol}.parquet"
    job_id = stable_hash(job)
    marker = _WORKER["output_root"] / "state" / "jobs" / f"{job_id}.json"
    if _WORKER.get("cached_symbol") != symbol:
        previous_symbol = _WORKER.get("cached_symbol")
        if previous_symbol is not None:
            _WORKER["funding"]._rates_cache.pop(previous_symbol, None)
        _WORKER.update({"cached_symbol": symbol, "cached_event_sha": file_sha256(event_path),
                        "cached_events": pd.read_parquet(event_path), "cached_market": None})
    event_sha = _WORKER["cached_event_sha"]
    if marker.is_file():
        prior = json.loads(marker.read_text())
        if prior.get("event_sha256") == event_sha and all(
            Path(row["path"]).is_file() and file_sha256(Path(row["path"])) == row["sha256"]
            for row in prior.get("files", [])
        ):
            return {**prior, "resumed": True}

    events = _WORKER["cached_events"]
    events = events.loc[events.model_id.eq(model_id) & events.family.eq(family)].copy()
    selected = job.get("selected_cell_ids")
    if selected is not None:
        events = events.loc[events.cell_id.isin(selected)].copy()
    registered_cell_ids = sorted(events.cell_id.unique().tolist()) if not events.empty else []
    files: list[dict[str, Any]] = []
    rejected_total = scored_total = 0
    bar_hash = None
    if not events.empty:
        if _WORKER["cached_market"] is None:
            authority = _WORKER["authority"]
            verified_trade_authority_hash(authority, [symbol])
            bars, bar_hash = load_timestamp_only_bars(authority, symbol)
            opens = load_open_map(authority, symbol)
            _WORKER["cached_market"] = (bars, opens, bar_hash)
        else:
            bars, opens, bar_hash = _WORKER["cached_market"]
        model = _WORKER["models"][model_id]
        schedule, rejected = resolve_timestamp_schedule(
            events, {symbol: bars}, model["evaluation_start"], model["evaluation_end"], model_id,
        )
        rejected_total += len(rejected)
        if not schedule.empty:
            scored = score_executions(attach_open_prices(schedule, {symbol: opens}), _WORKER["funding"])
            target = Path(job["scored_root"]) / f"{symbol}.parquet"
            _atomic_parquet(scored, target)
            files.append({"path": str(target), "bytes": target.stat().st_size,
                          "sha256": file_sha256(target), "rows": len(scored)})
            scored_total += len(scored)
        if not rejected.empty:
            target = Path(job["rejection_root"]) / f"{symbol}.parquet"
            _atomic_parquet(rejected, target)
            files.append({"path": str(target), "bytes": target.stat().st_size,
                          "sha256": file_sha256(target), "rows": len(rejected)})
    result = {
        "status": "pass", "job_id": job_id, "symbol": symbol, "model_id": model_id,
        "family": family, "event_sha256": event_sha,
        "registered_cell_ids": registered_cell_ids,
        "bar_authority_hash": bar_hash, "scored_rows": scored_total,
        "rejected_rows": rejected_total, "files": files, "completed_at_utc": utc_now(),
        "protected_rows_opened": 0, "Capitalcom_payload_opened": False,
    }
    atomic_json(marker, result)
    return result


def verify_outer_freeze(job: dict[str, Any]) -> dict[str, Any]:
    freeze_path = Path(job.get("required_freeze_path") or "")
    if not freeze_path.is_file():
        raise Stage20Error("outer outcome access attempted before atomic freeze")
    freeze = json.loads(freeze_path.read_text())
    if (freeze.get("freeze_sha256") != job.get("required_freeze_sha256")
            or freeze.get("selected_cell_ids") != job.get("selected_cell_ids")):
        raise Stage20Error("outer outcome freeze identity drift")
    return freeze


def _score_job_bundle(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_score_job(job) for job in jobs]


def _load_scored_roots(roots: list[Path], model_ids: list[str]) -> pd.DataFrame:
    frames = []
    for root, model_id in zip(roots, model_ids):
        for path in sorted(root.glob("*.parquet")):
            frame = pd.read_parquet(path)
            if not frame.empty:
                frame["source_model_id"] = model_id
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cell_metrics(frame: pd.DataFrame, cells: list[dict[str, Any]], inner_ids: list[str],
                  eligible_days: int, interval_seconds: float) -> list[dict[str, Any]]:
    rows = []
    for cell in cells:
        subset = frame.loc[frame.cell_id.eq(cell["cell_id"])].copy() if not frame.empty else pd.DataFrame()
        inner_means = []
        inner_observations = []
        for model_id in inner_ids:
            part = (subset.loc[subset.source_model_id.eq(model_id)]
                    if not subset.empty else pd.DataFrame())
            available = not part.empty
            value = (float(part.assign(day=part.entry_ts.dt.strftime("%Y-%m-%d"))
                           .groupby("day").base_net_bps.mean().mean())
                     if available else float("nan"))
            inner_means.append(value)
            inner_observations.append({"inner_fold_id": model_id,
                                       "status": "available" if available else "empty_unavailable",
                                       "base_net_mean_bps": value})
        metrics = metric_row(subset, inner_means, int(cell["complexity"]), eligible_days, 187, interval_seconds)
        rows.append({**metrics, "cell_id": cell["cell_id"], "family": cell["family"],
                     "canonical_translation_id": cell["canonical_translation_id"],
                     "inner_fold_observations": inner_observations,
                     "unavailable_inner_fold_count": sum(
                         row["status"] == "empty_unavailable" for row in inner_observations
                     )})
    return rows


def write_attempt_and_multiplicity_registries(output_root: Path) -> dict[str, int]:
    stage19 = ROOT / STAGE19_REL
    translations = json.loads((stage19 / "ECONOMIC_TRANSLATION_REGISTRY.json").read_text())
    search = json.loads((stage19 / "SEARCH_SPACE_REGISTRY.json").read_text())
    executable = [
        {"attempt_id": row["cell_id"], "family": row["family"],
         "canonical_translation_id": row["canonical_translation_id"],
         "status": "registered_executable_attempt"}
        for row in translations["cells"]
    ]
    inherited = search["non_executable_inherited_attempts"]
    by_family = {family: sum(row["family"] == family for row in executable) for family in EXPECTED_CELLS}
    if by_family != EXPECTED_CELLS or len(executable) != 186 or len(inherited) != 42:
        raise Stage20Error("attempt registry or inherited multiplicity drift")
    attempts = {"executable_attempts": executable, "inherited_non_executable_attempts": inherited}
    multiplicity = {
        "status": "reconciled", "executable_attempts": 186,
        "inherited_non_executable_KDX_attempts": 42, "programme_attempts": 228,
        "cells_by_family": by_family, "programme_exposure_class": "program_exposed_historical",
    }
    atomic_json(output_root / "FULL_ATTEMPT_REGISTRY.json", attempts)
    atomic_json(output_root / "MULTIPLICITY_RECONCILIATION.json", multiplicity)
    return multiplicity


def notify_required(notifier: TelegramNotifier, title: str, body: str = "") -> None:
    if not notifier.send(title, body):
        raise Stage20Error(f"Telegram {title.lower()} delivery failed")


def operational_snapshot(args: argparse.Namespace, started: float,
                         limits: dict[str, int]) -> dict[str, int]:
    snapshot = {
        "elapsed_seconds": int(time.monotonic() - started),
        "aggregate_rss_bytes": process_tree_rss(os.getpid()),
        "output_bytes": directory_bytes(args.run_root),
        "free_disk_bytes": shutil.disk_usage(args.run_root).free,
    }
    if snapshot["elapsed_seconds"] >= limits["max_wall_seconds"]:
        raise Stage20Error("wall_time_limit")
    if snapshot["aggregate_rss_bytes"] >= limits["max_rss_bytes"]:
        raise Stage20Error("aggregate_rss_limit")
    if snapshot["output_bytes"] >= limits["max_output_bytes"]:
        raise Stage20Error("campaign_output_limit")
    if snapshot["free_disk_bytes"] < limits["max_output_bytes"]:
        raise Stage20Error("free_disk_limit")
    return snapshot


def verify_job_result(result: dict[str, Any]) -> None:
    if result.get("status") != "pass":
        raise Stage20Error("worker result did not pass")
    for row in result.get("files", []):
        path = Path(row["path"])
        if (not path.is_file() or path.stat().st_size != int(row["bytes"])
                or file_sha256(path) != row["sha256"]):
            raise Stage20Error(f"worker artifact reconciliation failed: {path}")


def maybe_release_health(args: argparse.Namespace, state: dict[str, Any]) -> None:
    if state.get("health_release_status") == "pass":
        return
    evidence = state.get("first_reconciled_real_cell")
    if not state.get("first_scheduled_heartbeat_delivered") or not evidence:
        return
    for row in evidence.get("files", []):
        path = Path(row["path"])
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise Stage20Error("health-release cell artifact drift")
    state["health_release_status"] = "pass"
    state["health_released_at_utc"] = utc_now()
    atomic_json(args.run_root / "CAMPAIGN_STATE.json", state)
    state_bytes = (args.run_root / "CAMPAIGN_STATE.json").read_bytes()
    persisted = json.loads(state_bytes)
    if persisted.get("generation") != state.get("generation") or persisted.get("health_release_status") != "pass":
        raise Stage20Error("campaign state failed health-release round trip")
    heartbeat = args.run_root / "HEARTBEAT.json"
    atomic_json(args.run_root / "HEALTH_RELEASE.json", {
        "status": "pass", "released_at_utc": state["health_released_at_utc"],
        "registered_cell_id": evidence["cell_id"], "job_id": evidence["job_id"],
        "campaign_state_sha256": file_sha256(args.run_root / "CAMPAIGN_STATE.json"),
        "heartbeat_sha256": file_sha256(heartbeat), "artifact_files": evidence["files"],
    })


def emit_heartbeat(args: argparse.Namespace, state: dict[str, Any], notifier: TelegramNotifier,
                   snapshot: dict[str, int]) -> None:
    heartbeat = {
        "status": "healthy", "elapsed_seconds": snapshot["elapsed_seconds"],
        "jobs_complete": state.get("jobs_complete", 0), "current_fold": state.get("current_fold"),
        "aggregate_rss_bytes": snapshot["aggregate_rss_bytes"],
        "output_bytes": snapshot["output_bytes"], "partial_rankings_included": False,
    }
    atomic_json(args.run_root / "HEARTBEAT.json", heartbeat)
    notify_required(notifier, "30-MINUTE HEARTBEAT", json.dumps(heartbeat, sort_keys=True))
    state["first_scheduled_heartbeat_delivered"] = True
    state["last_heartbeat_at_utc"] = utc_now()
    state["generation"] += 1
    atomic_json(args.run_root / "CAMPAIGN_STATE.json", state)
    maybe_release_health(args, state)


def execute_jobs(pool: ProcessPoolExecutor, jobs: list[list[dict[str, Any]]], args: argparse.Namespace,
                 state: dict[str, Any], notifier: TelegramNotifier, started: float,
                 next_heartbeat: float, results: list[dict[str, Any]],
                 *, worker: Any = _score_job_bundle,
                 limits: dict[str, int] | None = None) -> float:
    bound = dict(DEFAULT_LIMITS if limits is None else limits)
    iterator = iter(jobs)
    pending: dict[Any, list[dict[str, Any]]] = {}
    exhausted = False
    state.setdefault("scheduler_max_in_flight", 0)
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < bound["max_workers"]:
                snapshot = operational_snapshot(args, started, bound)
                if time.monotonic() >= next_heartbeat:
                    emit_heartbeat(args, state, notifier, snapshot)
                    next_heartbeat += bound["heartbeat_seconds"]
                try:
                    bundle = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = pool.submit(worker, bundle)
                pending[future] = bundle
                state["bundles_submitted"] = state.get("bundles_submitted", 0) + 1
                state["scheduler_max_in_flight"] = max(state["scheduler_max_in_flight"], len(pending))
                if len(pending) > bound["max_workers"]:
                    raise Stage20Error("lazy scheduler exceeded worker bound")
            if not pending:
                continue
            done, _ = wait(tuple(pending), timeout=1.0, return_when=FIRST_COMPLETED)
            if not done:
                snapshot = operational_snapshot(args, started, bound)
                if time.monotonic() >= next_heartbeat:
                    emit_heartbeat(args, state, notifier, snapshot)
                    next_heartbeat += bound["heartbeat_seconds"]
                continue
            for future in done:
                pending.pop(future)
                bundle_results = future.result()
                for result in bundle_results:
                    verify_job_result(result)
                    if (result.get("registered_cell_ids") and result.get("files")
                            and not result.get("synthetic")
                            and state.get("first_reconciled_real_cell") is None):
                        state["first_reconciled_real_cell"] = {
                            "cell_id": result["registered_cell_ids"][0], "job_id": result["job_id"],
                            "files": result.get("files", []), "reconciled_at_utc": utc_now(),
                        }
                results.extend(bundle_results)
                state["jobs_complete"] = len(results)
                state["registered_cells_with_completed_job"] = len({
                    cell_id for row in results for cell_id in row.get("registered_cell_ids", [])
                })
                state["generation"] += 1
                state["updated_at_utc"] = utc_now()
                atomic_json(args.run_root / "CAMPAIGN_STATE.json", state)
                maybe_release_health(args, state)
    except BaseException:
        state["scheduler_accepting_submissions"] = False
        state["scheduler_cancelled_queued_jobs"] = sum(future.cancel() for future in pending)
        state["scheduler_running_jobs_at_stop"] = sum(not future.cancelled() for future in pending)
        state["generation"] += 1
        atomic_json(args.run_root / "CAMPAIGN_STATE.json", state)
        raise
    return next_heartbeat


def score_roots(output_root: Path, symbols: list[str], model_ids: list[str], family: str,
                phase_key: str, selected_cell_ids: list[str] | None = None,
                freeze: dict[str, Any] | None = None, freeze_path: Path | None = None) -> tuple[list[list[dict[str, Any]]], list[Path]]:
    roots = [output_root / "staging" / phase_key / model_id / family for model_id in model_ids]
    bundles = []
    for symbol in symbols:
        bundle = []
        for model_id, scored_root in zip(model_ids, roots):
            rejection_root = output_root / "staging" / "rejections" / phase_key / model_id / family
            bundle.append({"symbol": symbol, "model_id": model_id, "family": family,
                           "selected_cell_ids": selected_cell_ids, "scored_root": str(scored_root),
                           "rejection_root": str(rejection_root),
                           "required_freeze_path": str(freeze_path) if freeze_path else None,
                           "required_freeze_sha256": freeze.get("freeze_sha256") if freeze else None})
        bundles.append(bundle)
    return bundles, roots


def phase2_to_5(output_root: Path, symbols: list[str], pool: ProcessPoolExecutor,
                args: argparse.Namespace, state: dict[str, Any], notifier: TelegramNotifier,
                started: float, next_heartbeat: float,
                job_results: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    stage19 = ROOT / STAGE19_REL
    registry = json.loads((stage19 / "ECONOMIC_TRANSLATION_REGISTRY.json").read_text())
    fold_map = json.loads((stage19 / "INNER_FOLD_MAP.json").read_text())
    cells_by_family = {family: [row for row in registry["cells"] if row["family"] == family]
                       for family in EXPECTED_CELLS}
    folds = sorted(fold_map["outer_folds"], key=lambda row: (row["outer_evaluation_start"], row["hypothesis_id"]))
    family_active = {family: True for family in EXPECTED_CELLS}
    surfaces, beams, outer_results, fold_decisions = [], [], [], []
    completed_development: set[tuple[str, str]] = set()
    for outer in folds:
        family = next(family for family in EXPECTED_CELLS if outer["hypothesis_id"].startswith(family))
        fold_id = outer["outer_fold_id"]
        if not family_active[family]:
            fold_decisions.append({"outer_fold_id": fold_id, "family": family, "status": "family_stopped_earlier"})
            continue
        inner_ids = [row["inner_fold_id"] for row in outer["inner_folds"]]
        new_ids = [model_id for model_id in inner_ids if (model_id, family) not in completed_development]
        if new_ids:
            jobs, _ = score_roots(output_root, symbols, new_ids, family, "development")
            state.update({"phase": 2, "current_fold": fold_id, "current_family": family})
            next_heartbeat = execute_jobs(pool, jobs, args, state, notifier, started,
                                          next_heartbeat, job_results)
            completed_development.update((model_id, family) for model_id in new_ids)
        development_roots = [output_root / "staging" / "development" / model_id / family
                             for model_id in inner_ids]
        development = _load_scored_roots(development_roots, inner_ids)
        start = min(pd.Timestamp(row["validation_start"]) for row in outer["inner_folds"])
        end = max(pd.Timestamp(row["validation_end_exclusive"]) for row in outer["inner_folds"])
        metrics = _cell_metrics(development, cells_by_family[family], inner_ids,
                                max(1, (end - start).days), float((end - start).total_seconds()))
        for row in metrics:
            row["outer_fold_id"] = fold_id
        surfaces.extend(metrics)
        notify_required(notifier, "PHASE", f"phase=3\nfold={fold_id}\nmanual_selection=no")
        selected = choose_beam(metrics)
        if not selected:
            family_active[family] = False
            decision = {"outer_fold_id": fold_id, "family": family,
                        "status": "family_stop_no_positive_development_candidate"}
            fold_decisions.append(decision)
            notify_required(notifier, "FAMILY STOP", json.dumps(decision, sort_keys=True))
            continue
        selected_ids = [row["cell_id"] for row in selected]
        freeze = {
            "outer_fold_id": fold_id, "family": family, "selected_cell_ids": selected_ids,
            "selected_translation_ids": [row["canonical_translation_id"] for row in selected],
            "freeze_sha256": canonical_sha256({"fold": fold_id, "cells": selected_ids}),
            "maximum_candidates": 5, "manual_selection": False,
        }
        beams.append(freeze)
        freeze_path = output_root / "freezes" / f"{stable_hash(fold_id)}.json"
        atomic_json(freeze_path, freeze)
        notify_required(notifier, "PHASE", f"phase=4-5\nfold={fold_id}\nfreeze_committed=yes")
        outer_model = "Q_" + fold_id.split(":", 1)[1]
        jobs, outer_roots = score_roots(output_root, symbols, [outer_model], family,
                                        f"outer/{stable_hash(fold_id)}", selected_ids, freeze, freeze_path)
        state.update({"phase": 4, "current_fold": fold_id, "current_family": family,
                      "current_freeze_sha256": freeze["freeze_sha256"]})
        next_heartbeat = execute_jobs(pool, jobs, args, state, notifier, started,
                                      next_heartbeat, job_results)
        outer_frame = _load_scored_roots(outer_roots, [outer_model])
        for cell in [row for row in cells_by_family[family] if row["cell_id"] in selected_ids]:
            subset = outer_frame.loc[outer_frame.cell_id.eq(cell["cell_id"])].copy() if not outer_frame.empty else pd.DataFrame()
            eval_start, eval_end = pd.Timestamp(outer["outer_evaluation_start"]), pd.Timestamp(outer["outer_evaluation_end_exclusive"])
            result = metric_row(subset, [], int(cell["complexity"]), max(1, (eval_end-eval_start).days), 187,
                                float((eval_end-eval_start).total_seconds()))
            result.update({"outer_fold_id": fold_id, "family": family, "cell_id": cell["cell_id"],
                           "canonical_translation_id": cell["canonical_translation_id"],
                           "freeze_sha256": freeze["freeze_sha256"]})
            outer_results.append(result)
        fold_decisions.append({"outer_fold_id": fold_id, "family": family, "status": "outer_evaluated",
                               "selected_candidates": len(selected_ids)})
        notify_required(notifier, "FOLD COMPLETE", f"fold={fold_id}\nselected={len(selected_ids)}")

    response = pd.DataFrame(surfaces)
    outer_frame = pd.DataFrame(outer_results)
    attempted_cells = sorted(response.cell_id.unique().tolist()) if not response.empty else []
    if len(attempted_cells) != 186:
        raise Stage20Error(f"executable cell attempt reconciliation failed: {len(attempted_cells)}")
    response.to_csv(output_root / "PHASE2_DEVELOPMENT_RESPONSE_SURFACE.csv", index=False)
    outer_frame.to_csv(output_root / "PHASE4_5_OUTER_ROLLING_RESULTS.csv", index=False)
    atomic_json(output_root / "PHASE3_FROZEN_BEAM_REGISTRY.json", {"beams": beams})
    atomic_json(output_root / "FOLD_AND_FAMILY_DECISIONS.json", {"decisions": fold_decisions})
    routes = []
    for family in EXPECTED_CELLS:
        family_surface = [row for row in surfaces if row["family"] == family]
        rows = outer_frame.loc[outer_frame.family.eq(family)] if not outer_frame.empty else pd.DataFrame()
        finite = rows.loc[np.isfinite(rows.aggregate_base_net_mean_bps)] if not rows.empty else pd.DataFrame()
        hard_positive = finite.loc[
            finite.aggregate_base_net_mean_bps.gt(0) & finite.base_net_median_bps.gt(0)
        ] if not finite.empty else pd.DataFrame()
        positive = len(hard_positive)
        if not rows.empty and positive:
            alignment_sensitive = bool(
                hard_positive.aggregate_base_net_alignment_start_mean_bps.le(0).any()
                or hard_positive.aggregate_base_net_alignment_end_mean_bps.le(0).any()
            )
            stress_sensitive = bool(hard_positive.aggregate_stress_net_mean_bps.le(0).any())
            if alignment_sensitive or stress_sensitive:
                route = "execution_sensitive_candidate"
            elif family == "KDA02C":
                route = "conditional_context_candidate_unvalidated"
            else:
                route = "sample_limited_prospective_candidate"
        elif rows.empty and family_surface and all(row["accepted_trade_count"] == 0 for row in family_surface):
            route = "mechanically_unavailable"
        elif rows.empty and any(
            np.isfinite(row["aggregate_base_net_mean_bps"])
            and row["aggregate_base_net_mean_bps"] > 0 for row in family_surface
        ):
            route = "sample_limited_prospective_candidate"
        else:
            route = "translation_rejected"
        routes.append({"family": family, "route": route, "positive_outer_candidates": positive,
                       "limitation_tags": ["program_exposed_historical", "no_controls",
                                           "no_independent_validation", "not_deployment_evidence"],
                       "controls_executed": False, "independent_validation": False,
                       "programme_exposure_class": "program_exposed_historical"})
    atomic_json(output_root / "FAMILY_ROUTES.json", {"routes": routes})
    atomic_json(output_root / "EXECUTION_RECONCILIATION.json", {
        "status": "reconciled", "planned_executable_cells": 186,
        "executed_or_mechanically_evaluated_cells": len(attempted_cells),
        "omitted_executable_cells": 0, "inherited_non_executable_KDX_attempts": 42,
        "programme_attempts": 228, "controls_executed": False,
    })
    return ({"response_surface_rows": len(response), "frozen_beams": len(beams),
             "outer_result_rows": len(outer_frame), "routes": routes}, next_heartbeat)


def validate_gates(args: argparse.Namespace) -> None:
    source = validate_source_manifest(args.source_manifest)
    gates = {
        "deterministic_event_replay": (args.deterministic_replay, validate_gate(
            args.deterministic_replay, "deterministic_event_replay")),
        "synthetic_supervisor_canary": (args.synthetic_canary, validate_gate(
            args.synthetic_canary, "synthetic_supervisor_canary")),
        "independent_preoutcome_review": (args.preoutcome_review, validate_gate(
            args.preoutcome_review, "independent_preoutcome_review")),
        "telegram_preflight": (args.telegram_validation, validate_gate(
            args.telegram_validation, "telegram_preflight")),
        "final_launch_authority": (args.launch_binding, validate_gate(
            args.launch_binding, "final_launch_authority")),
    }
    required = [args.approval, args.source_manifest,
                args.event_root / "PREOUTCOME_EVENT_TAPE_MANIFEST.json"]
    for gate_id, (_, gate) in gates.items():
        for path in required:
            assert_gate_binds(gate, path)
        if gate_id == "final_launch_authority":
            for path, _ in gates.values():
                if path != args.launch_binding:
                    assert_gate_binds(gate, path)
    final_launch_boundary_audit(
        approval=args.approval, source_manifest=source, event_root=args.event_root,
        launch_gate=gates["final_launch_authority"][1], run_root=args.run_root,
    )
    launch = json.loads((args.run_root / "CAMPAIGN_LAUNCH_MANIFEST.json").read_text())
    if launch.get("status") != "preoutcome_pending_review_and_telegram" or launch.get("registered_cells") != 186:
        raise Stage20Error("launch manifest state invalid")
    event_manifest = json.loads((args.event_root / "PREOUTCOME_EVENT_TAPE_MANIFEST.json").read_text())
    if event_manifest.get("status") != "pass" or event_manifest.get("registered_cells") != 186:
        raise Stage20Error("preoutcome event manifest invalid")
    if event_manifest.get("protected_rows_opened") != 0 or event_manifest.get("economic_outcome_reader_opened") is not False:
        raise Stage20Error("outcome firewall did not remain closed")
    if len(event_manifest.get("files", [])) != 187 or any(not row.get("symbol") for row in event_manifest["files"]):
        raise Stage20Error("native-PF-symbol event partition reconciliation failed")
    if shutil.disk_usage(args.run_root).free < MAX_OUTPUT_BYTES:
        raise Stage20Error("insufficient free disk for bound output limit")


def run(args: argparse.Namespace) -> int:
    validate_gates(args)
    notifier = TelegramNotifier.from_args(args, run_label="stage20-stage19-campaign")
    if not notifier.enabled:
        raise Stage20Error("blocked_telegram_notifier_unavailable")
    started = time.monotonic()
    notify_required(notifier, "CAMPAIGN START", "phases=2-5\nlanes=KDA02B,KDA02C,KDX01\ncontrols=no")
    event_manifest = json.loads((args.event_root / "PREOUTCOME_EVENT_TAPE_MANIFEST.json").read_text())
    symbols = sorted(Path(row["path"]).stem for row in event_manifest["files"] if row.get("symbol"))
    multiplicity = write_attempt_and_multiplicity_registries(args.run_root)
    state = {"status": "economic_running", "generation": 1, "started_at_utc": utc_now(),
             "symbols_total": len(symbols), "jobs_complete": 0, "workers": MAX_WORKERS,
             "protected_rows_opened": 0, "Capitalcom_payload_opened": False,
             "scheduler_accepting_submissions": True, "health_release_status": "pending",
             "first_scheduled_heartbeat_delivered": False,
             "first_reconciled_real_cell": None}
    atomic_json(args.run_root / "CAMPAIGN_STATE.json", state)
    notify_required(notifier, "PHASE", "phase=2\nregistered_development_cells=186")
    next_heartbeat = started + HEARTBEAT_SECONDS
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_worker_init,
                             initargs=(str(args.event_root), str(args.run_root))) as pool:
        terminal, next_heartbeat = phase2_to_5(
            args.run_root, symbols, pool, args, state, notifier, started, next_heartbeat, results
        )
    if not state.get("first_scheduled_heartbeat_delivered"):
        while time.monotonic() < next_heartbeat:
            operational_snapshot(args, started, DEFAULT_LIMITS)
            time.sleep(min(5.0, next_heartbeat - time.monotonic()))
        emit_heartbeat(args, state, notifier, operational_snapshot(args, started, DEFAULT_LIMITS))
    maybe_release_health(args, state)
    if (state.get("health_release_status") != "pass"
            or not (args.run_root / "HEALTH_RELEASE.json").is_file()):
        raise Stage20Error("campaign health release requirements not satisfied")
    atomic_json(args.run_root / "SYMBOL_SCORING_MANIFEST.json", {"jobs": results})
    state.update({"status": "terminal_complete", "generation": state["generation"] + 1,
                  "scheduler_accepting_submissions": False,
                  "completed_at_utc": utc_now(), "multiplicity": multiplicity, **terminal})
    atomic_json(args.run_root / "CAMPAIGN_STATE.json", state)
    notify_required(notifier, "CAMPAIGN COMPLETE", f"status=terminal_complete\nouter_rows={terminal['outer_result_rows']}")
    print(json.dumps(state, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run-root", type=Path, required=True)
    result.add_argument("--event-root", type=Path, required=True)
    result.add_argument("--preoutcome-review", type=Path, required=True)
    result.add_argument("--telegram-validation", type=Path, required=True)
    result.add_argument("--synthetic-canary", type=Path, required=True)
    result.add_argument("--deterministic-replay", type=Path, required=True)
    result.add_argument("--source-manifest", type=Path, required=True)
    result.add_argument("--approval", type=Path, required=True)
    result.add_argument("--launch-binding", type=Path, required=True)
    result.add_argument("--tg-bot-token", default="")
    result.add_argument("--tg-chat-id", default="")
    result.add_argument("--tg-auto-chat", action="store_true", default=True)
    return result


if __name__ == "__main__":
    ns = parser().parse_args()
    try:
        raise SystemExit(run(ns))
    except Exception as exc:
        state_path = ns.run_root / "CAMPAIGN_STATE.json"
        try:
            stopped = json.loads(state_path.read_text()) if state_path.is_file() else {}
        except Exception:
            stopped = {}
        stopped.update({"status": "global_stop", "reason": f"{type(exc).__name__}: {exc}",
                        "stopped_at_utc": utc_now(), "resumable_state_preserved": True,
                        "scheduler_accepting_submissions": False,
                        "generation": int(stopped.get("generation", 0)) + 1})
        try:
            atomic_json(state_path, stopped)
            notifier = TelegramNotifier.from_args(ns, run_label="stage20-stage19-campaign")
            stopped["Telegram_global_stop_delivered"] = notifier.send(
                "GLOBAL STOP", f"reason_type={type(exc).__name__}"
            )
            atomic_json(state_path, stopped)
        except Exception:
            stopped["Telegram_global_stop_delivered"] = False
        print(json.dumps(stopped, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
