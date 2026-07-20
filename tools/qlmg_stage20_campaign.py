"""Mechanical utilities for the approval-bound Stage 20 derivatives campaign.

This module does not choose economic semantics. It validates and executes only
the Stage 16 identities and Stage 19 funding contract bound by the approved
Stage 19 replacement packet.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tools.qlmg_stage16_campaign import collapse_edges, deterministic_beam, file_sha256, quantile_type7
from tools.qlmg_stage19_funding import Stage19FundingEngine


CAMPAIGN_ID = "kraken_derivatives_campaign_001_stage19_exact_funding"
START_COMMIT = "245b375b00167f1b4a81f6a4449e7de1d1db83a2"
APPROVAL_SHA256 = "57d521488e88373afd557eb457ffd119089aac0d863a0c319e73d70cf9f7690c"
MANIFEST_SHA256 = "e7d618a2a24c574c9ba83d323df605a6434c2e816fe24c465d183ba5d6256990"
PACKET_SHA256 = "3fde09f16efff2479ee847fe26c859a67ef1d516b0f0acbcbaec141154a87bd6"
TRAIN_START = pd.Timestamp("2023-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2026-01-01T00:00:00Z")
STAGE19_REL = Path("docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2")
STAGE14_REL = Path("docs/agent/task_archive/20260719_donch_bt_stage_14_close_derivatives_phase1_blockers_20260719_v1")
STAGE8A_REL = Path("docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1")
STAGE14_STATE_ROOT = Path("/opt/parquet/kraken_derivatives/analytics/stage14_phase1_v1")
FUNDING_PACKAGE = Path("/opt/testerdonch/results/rebaseline/phase_kraken_local_official_funding_export_20260720_v4/kraken_funding_rankable_2023_2025.zip")
FUNDING_PACKAGE_SHA256 = "6c0969727c882ca6439c57c3b7d03367e1b2d26ee46823e56b983756d026ef64"
MARKET_MANIFEST = Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv")
KDA02_EVENT_TAPE = Path("/opt/parquet/kraken_derivatives/analytics/stage9_kda02_v2_prerun_v4/KDA02_V2_EVENT_TAPE_RAW.parquet")
KDA02_EVENT_TAPE_SHA256 = "e3d59a21bcd19b93c69bce0f1e540506ec72e2f9f83d35265246ec2dc3c483e8"
class Stage20Error(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def guarded_parquet(path: Path, columns: Iterable[str], time_column: str) -> pd.DataFrame:
    """Reject protected row groups before payload read and enforce the rankable interval."""
    parquet = pq.ParquetFile(path)
    requested = list(columns)
    if not set(requested).issubset(parquet.schema_arrow.names):
        raise Stage20Error(f"required columns absent: {path}")
    index = parquet.schema_arrow.names.index(time_column)
    for group in range(parquet.metadata.num_row_groups):
        metadata = parquet.metadata.row_group(group)
        if metadata.num_rows == 0:
            continue
        stats = metadata.column(index).statistics
        if stats is None or not stats.has_min_max:
            raise Stage20Error(f"timestamp statistics unavailable before read: {path}")
        maximum = pd.Timestamp(stats.max)
        maximum = maximum.tz_localize("UTC") if maximum.tzinfo is None else maximum.tz_convert("UTC")
        if maximum >= PROTECTED_START:
            raise Stage20Error(f"protected row group rejected before read: {path}")
    frame = pd.read_parquet(path, columns=requested)
    times = pd.to_datetime(frame[time_column], utc=True, errors="raise")
    if ((times < TRAIN_START) | (times >= PROTECTED_START)).any():
        raise Stage20Error(f"non-rankable row reached reader: {path}")
    frame[time_column] = times
    return frame


def authority_audit(repo: Path, approval_path: Path) -> dict[str, Any]:
    stage19 = repo / STAGE19_REL
    manifest_path = stage19 / "CAMPAIGN_MANIFEST.json"
    packet_path = stage19 / "FUTURE_DERIVATIVES_CAMPAIGN_APPROVAL_PACKET.json"
    if file_sha256(approval_path) != APPROVAL_SHA256:
        raise Stage20Error("approval artifact mismatch")
    if file_sha256(manifest_path) != MANIFEST_SHA256 or file_sha256(packet_path) != PACKET_SHA256:
        raise Stage20Error("campaign manifest or packet mismatch")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    if approval.get("human_authorized") is not True or approval.get("status") != "approved":
        raise Stage20Error("external approval is not active")
    if approval.get("approved_repository_start") != START_COMMIT:
        raise Stage20Error("approval starting commit mismatch")
    if approval.get("campaign_manifest_file_sha256") != MANIFEST_SHA256 or approval.get("approval_packet_file_sha256") != PACKET_SHA256:
        raise Stage20Error("approval does not bind current packet")
    if approval.get("approved_phases") != [2, 3, 4, 5] or approval.get("approved_executable_cells", {}).get("total") != 186:
        raise Stage20Error("approval phase or cell scope mismatch")
    dependencies = approval.get("bound_dependency_file_sha256", {})
    if len(dependencies) != 18:
        raise Stage20Error("external approval dependency inventory mismatch")
    raw: dict[str, str] = {}
    for name, expected in dependencies.items():
        path = stage19 / name
        raw[name] = file_sha256(path)
        if raw[name] != expected:
            raise Stage20Error(f"dependency hash drift: {name}")
        if manifest["dependency_file_sha256"].get(name) != expected or packet["dependency_file_sha256"].get(name) != expected:
            raise Stage20Error(f"dependency binding drift: {name}")
    state_manifest_path = repo / STAGE14_REL / "LOCAL_STATE_TAPE_MANIFEST.json"
    state_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    if Path(state_manifest.get("root", "")) != STAGE14_STATE_ROOT:
        raise Stage20Error("Stage-14 state root mismatch")
    for row in state_manifest.get("files", []):
        target = STAGE14_STATE_ROOT / row["path"]
        if not target.is_file() or target.stat().st_size != int(row["bytes"]) or file_sha256(target) != row["sha256"]:
            raise Stage20Error(f"Stage-14 state tape drift: {row['path']}")
    if state_manifest.get("economic_outputs_computed") is not False or state_manifest.get("protected_rows_opened") != 0:
        raise Stage20Error("Stage-14 state authority is not outcome-free")
    feature_manifest = json.loads((repo / STAGE8A_REL / "KDA_FEATURE_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    if feature_manifest.get("analytics_manifest_hash") != "f1520fd3875578a9a2101b10d5e15b7b88c58a6ffcf6067dd91f582352d92a6d" or feature_manifest.get("cohort_hash") != "5df5b304d8e735d3579e220d25a000e81013435320c6e8fb2480c1ed565a3636":
        raise Stage20Error("feature authority mismatch")
    stage9_manifest = repo / "docs/agent/task_archive/20260719_donch_bt_stage_9_kda02_purge_adjudication_conditional_level3_20260719_v1/ARTIFACT_MANIFEST.json"
    stage9 = json.loads(stage9_manifest.read_text(encoding="utf-8"))
    bound = [row for row in stage9.get("files", []) + stage9.get("local_cache_files", []) if row.get("path") == str(KDA02_EVENT_TAPE)]
    if len(bound) != 1 or bound[0].get("sha256") != KDA02_EVENT_TAPE_SHA256 or file_sha256(KDA02_EVENT_TAPE) != KDA02_EVENT_TAPE_SHA256:
        raise Stage20Error("KDA02 completed-purge event tape authority mismatch")
    return {
        "status": "pass",
        "verified_at_utc": utc_now(),
        "approval_sha256": APPROVAL_SHA256,
        "campaign_manifest_sha256": MANIFEST_SHA256,
        "approval_packet_sha256": PACKET_SHA256,
        "dependency_file_sha256": raw,
        "dependency_count": len(raw),
        "stage14_state_manifest_sha256": file_sha256(state_manifest_path),
        "stage14_state_files_verified": len(state_manifest["files"]),
        "kda02_event_tape_sha256": KDA02_EVENT_TAPE_SHA256,
        "protected_rows_opened": 0,
        "Capitalcom_payload_opened": False,
        "narrative_estimator_hash_typo_resolution": "formal approval and machine dependency both bind b30b7c115d6d6e1ed765542c44d791c44117a0387f71ae4b33ec1219d4243a3b",
    }


def nonoverlap(frame: pd.DataFrame) -> pd.DataFrame:
    """Definition-local, symbol-local chronological non-overlap by actual exit."""
    if frame.empty:
        return frame.copy()
    work = frame.sort_values(["translation_id", "symbol", "decision_ts", "event_id"], kind="mergesort")
    keep: list[int] = []
    for _, group in work.groupby(["translation_id", "symbol"], sort=True):
        prior_exit: pd.Timestamp | None = None
        for index, row in group.iterrows():
            if prior_exit is None or row.entry_ts >= prior_exit:
                keep.append(index)
                prior_exit = row.exit_ts
    return work.loc[keep].reset_index(drop=True)


def metric_row(frame: pd.DataFrame, inner_means: list[float], complexity: int,
               eligible_days: int, eligible_symbols: int, interval_seconds: float) -> dict[str, Any]:
    if frame.empty:
        return {"accepted_trade_count": 0, "integrity_pass": True, "independent_market_day_clusters": 0,
                "independent_utc_hour_clusters": 0, "aggregate_base_net_mean_bps": float("nan"),
                "aggregate_stress_net_mean_bps": float("nan"), "base_net_median_bps": float("nan"),
                "aggregate_base_net_alignment_start_mean_bps": float("nan"),
                "aggregate_base_net_alignment_end_mean_bps": float("nan"),
                "median_inner_fold_base_net_mean_bps": float("nan"), "p20_inner_fold_base_net_mean_bps": float("nan"),
                "cluster_bootstrap_lower_bound_bps": float("nan"), "left_tail_utility_bps": float("nan"),
                "opportunity_frequency_per_30d": 0.0, "capital_occupancy": 0.0, "execution_margin_bps": float("nan"),
                "symbol_day_year_contribution": 1.0, "complexity": complexity, "market_day_returns": {}}
    work = frame.copy()
    work["day"] = work.entry_ts.dt.strftime("%Y-%m-%d")
    work["hour"] = work.entry_ts.dt.strftime("%Y-%m-%dT%H")
    day = work.groupby("day", sort=True).base_net_bps.mean()
    stress_day = work.groupby("day", sort=True).stress_net_bps.mean()
    alignment_start_day = work.groupby("day", sort=True).base_net_alignment_start_bps.mean()
    alignment_end_day = work.groupby("day", sort=True).base_net_alignment_end_bps.mean()
    rng = np.random.default_rng(20260720)
    values = day.to_numpy(float)
    bootstrap = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(2000)])
    cumulative = values.cumsum()
    drawdown = np.maximum.accumulate(np.r_[0.0, cumulative])[1:] - cumulative
    absolute = work.base_net_bps.abs().sum()
    shares: list[float] = []
    for column in ("symbol", "day"):
        grouped = work.groupby(column).base_net_bps.sum().abs()
        shares.append(float(grouped.max() / absolute) if absolute else 1.0)
    yearly = work.assign(year=work.entry_ts.dt.year).groupby("year").base_net_bps.sum().abs()
    shares.append(float(yearly.max() / absolute) if absolute else 1.0)
    return {
        "accepted_trade_count": int(len(work)), "integrity_pass": True,
        "independent_market_day_clusters": int(day.size), "independent_utc_hour_clusters": int(work.hour.nunique()),
        "aggregate_base_net_mean_bps": float(day.mean()), "aggregate_stress_net_mean_bps": float(stress_day.mean()),
        "aggregate_base_net_alignment_start_mean_bps": float(alignment_start_day.mean()),
        "aggregate_base_net_alignment_end_mean_bps": float(alignment_end_day.mean()),
        "base_net_median_bps": float(work.base_net_bps.median()),
        "median_inner_fold_base_net_mean_bps": float(np.quantile(inner_means, .5)) if inner_means else float("nan"),
        "p20_inner_fold_base_net_mean_bps": float(np.quantile(inner_means, .2)) if inner_means else float("nan"),
        "cluster_bootstrap_lower_bound_bps": float(np.quantile(bootstrap, .05)),
        "left_tail_utility_bps": -float(drawdown.max(initial=0)),
        "opportunity_frequency_per_30d": float(30 * len(work) / max(1, eligible_days)),
        "capital_occupancy": float((work.exit_ts - work.entry_ts).dt.total_seconds().sum() / max(1.0, eligible_symbols * interval_seconds)),
        "execution_margin_bps": float(stress_day.mean()), "symbol_day_year_contribution": max(shares),
        "complexity": int(complexity), "market_day_returns": {str(key): float(value) for key, value in day.items()},
    }


def eligible_for_beam(metrics: dict[str, Any]) -> bool:
    return (
        bool(metrics.get("integrity_pass"))
        and metrics.get("accepted_trade_count", 0) >= 30
        and metrics.get("independent_market_day_clusters", 0) >= 20
        and metrics.get("independent_utc_hour_clusters", 0) >= 20
        and math.isfinite(float(metrics.get("aggregate_base_net_mean_bps", float("nan"))))
        and float(metrics["aggregate_base_net_mean_bps"]) > 0
    )


def choose_beam(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [row for row in rows if eligible_for_beam(row)]
    return deterministic_beam(eligible, limit=5) if eligible else []


FEATURE_COLUMNS = [
    "timestamp_utc", "trade_close", "mark_close", "trade_return_1h", "mark_return_1h",
    "oi_log_change_1h", "liquidation_base_units_1h", "liquidation_intensity_robust_z",
    "liquidation_normalization_valid", "basis_bps", "basis_change_1h",
    "basis_level_normalization_valid", "basis_change_normalization_valid", "eligible",
    "known_lifecycle_mask", "trade_coverage", "mark_coverage", "analytics_coverage",
]


def valid_feature_rows(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.eligible.fillna(False).astype(bool)
        & frame.known_lifecycle_mask.fillna(False).astype(bool)
        & frame.trade_coverage.fillna(False).astype(bool)
        & frame.mark_coverage.fillna(False).astype(bool)
        & frame.analytics_coverage.fillna(False).astype(bool)
    )


def type7_threshold(values: pd.Series, probability: float) -> float:
    clean = pd.to_numeric(values, errors="coerce")
    clean = clean[np.isfinite(clean)]
    if clean.nunique() < 3:
        raise Stage20Error("fewer than three unique values for registered type-7 threshold")
    return float(quantile_type7(clean.tolist(), probability))


def registered_rank_edges(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce")
    clean = clean[np.isfinite(clean)]
    try:
        edges = [float(value) for value in np.quantile(clean.to_numpy(float), (0, .2, .4, .6, .8, 1), method="linear")]
        collapse_edges(edges)
    except Exception as exc:
        raise Stage20Error(f"invalid registered Type-7 grid: {exc}") from exc
    return {"q0": edges[0], "q20": edges[1], "q80": edges[4], "q100": edges[5]}


def fit_global_thresholds(repo: Path, windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> dict[str, dict[str, float]]:
    """Fit each registered Type-7 component threshold on the full PIT population."""
    primitive_paths = sorted((STAGE14_STATE_ROOT / "primitive_base").glob("*.parquet"))
    feature_manifest = json.loads((repo / STAGE8A_REL / "KDA_FEATURE_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    feature_paths = [Path(row["path"]) for row in feature_manifest["partitions"]]
    sources = {
        "oi": (primitive_paths, "oi_log_change_1h", 1.0, None),
        "trade_abs": (primitive_paths, "trade_abs_bps", 1.0, None),
        "mark_abs": (primitive_paths, "mark_abs_bps", 1.0, None),
        "liquidation": (feature_paths, "liquidation_intensity_robust_z", 1.0, "liquidation_normalization_valid"),
        "basis_level": (feature_paths, "basis_bps", 1.0, "basis_level_normalization_valid"),
        "basis_change": (feature_paths, "basis_change_1h", 10000.0, "basis_change_normalization_valid"),
    }
    result = {name: {} for name in windows}
    for threshold_name, (paths, field, scale, validity_field) in sources.items():
        timestamp_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        for path in paths:
            time_name = "timestamp_utc"
            columns = [time_name, field, "eligible"] + ([validity_field] if validity_field else [])
            part = pd.read_parquet(path, columns=columns)
            valid = part.eligible.fillna(False).astype(bool)
            if validity_field:
                valid &= part[validity_field].fillna(False).astype(bool)
            values = pd.to_numeric(part[field], errors="coerce") * scale
            valid &= np.isfinite(values)
            timestamp_parts.append(pd.to_datetime(part.loc[valid, time_name], utc=True).dt.as_unit("ns").astype("int64").to_numpy())
            value_parts.append(values.loc[valid].to_numpy(float))
        timestamps = np.concatenate(timestamp_parts)
        values = np.concatenate(value_parts)
        for name, (start, end) in windows.items():
            mask = (timestamps >= start.value) & (timestamps < end.value)
            try:
                for quantile, value in registered_rank_edges(pd.Series(values[mask], copy=False)).items():
                    result[name][f"{threshold_name}_{quantile}"] = value
            except Stage20Error as exc:
                result[name][f"{threshold_name}_error"] = str(exc)
        del timestamps, values, timestamp_parts, value_parts
    breadth = kdx_breadth_panel()
    breadth_values = breadth.breadth_share
    for name, (start, end) in windows.items():
        mask = breadth.timestamp_utc.ge(start) & breadth.timestamp_utc.lt(end)
        try:
            for quantile, value in registered_rank_edges(breadth_values.loc[mask]).items():
                result[name][f"breadth_{quantile}"] = value
        except Stage20Error as exc:
            result[name]["breadth_error"] = str(exc)
    return result


def kdx_breadth_panel() -> pd.DataFrame:
    panel = guarded_parquet(STAGE14_STATE_ROOT / "KDA02C_PIT_BREADTH_PANEL.parquet",
                            ["timestamp_utc", "eligible"], "timestamp_utc")
    events = guarded_parquet(KDA02_EVENT_TAPE, ["event_id", "attempt", "event_type", "parent_direction", "decision_ts"], "decision_ts")
    events = events.loc[
        events.attempt.eq("primary") & events.event_type.eq("completed_purge_reversal")
        & events.parent_direction.eq(-1)
    ].copy()
    events["timestamp_utc"] = events.decision_ts - pd.Timedelta(minutes=5)
    counts = events.groupby("timestamp_utc", sort=True).event_id.nunique().rename("directional_onset_count")
    result = panel.merge(counts, on="timestamp_utc", how="left", validate="one_to_one")
    result["directional_onset_count"] = result.directional_onset_count.fillna(0).astype(int)
    result["breadth_share"] = result.directional_onset_count / result.eligible.replace(0, np.nan)
    return result


def fit_component_thresholds(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    ts = pd.to_datetime(frame.timestamp_utc, utc=True, errors="raise")
    sample = frame.loc[valid_feature_rows(frame) & ts.ge(start) & ts.lt(end)]
    if sample.empty:
        raise Stage20Error("empty inner-training feature population")
    result: dict[str, float] = {}
    values = {
        "oi": sample.oi_log_change_1h,
        "trade_abs": sample.trade_return_1h.abs() * 10000.0,
        "mark_abs": sample.mark_return_1h.abs() * 10000.0,
        "liquidation": sample.loc[sample.liquidation_normalization_valid.fillna(False), "liquidation_intensity_robust_z"],
        "basis_level": sample.loc[sample.basis_level_normalization_valid.fillna(False), "basis_bps"],
        "basis_change": sample.loc[sample.basis_change_normalization_valid.fillna(False), "basis_change_1h"] * 10000.0,
    }
    for prefix, source in values.items():
        result.update({f"{prefix}_{key}": value for key, value in registered_rank_edges(source).items()})
    if "breadth_share" in sample:
        result.update({f"breadth_{key}": value for key, value in registered_rank_edges(sample.breadth_share).items()})
    return result


def _episode_onsets(mandatory: pd.DataFrame, timestamps: pd.Series, valid: pd.Series | None = None) -> list[int]:
    """First all-true bar; rearm only after every mandatory predicate is false."""
    if mandatory.empty:
        return []
    truth = mandatory.fillna(False).astype(bool).to_numpy()
    all_true = truth.all(axis=1)
    all_false = ~truth.any(axis=1)
    times = pd.to_datetime(timestamps, utc=True, errors="raise").reset_index(drop=True)
    valid_array = np.ones(len(times), dtype=bool) if valid is None else valid.fillna(False).astype(bool).to_numpy()
    nanos = times.dt.as_unit("ns").astype("int64").to_numpy()
    gap = np.r_[True, np.diff(nanos) != 300_000_000_000]
    barriers = gap | ~valid_array
    boundaries = barriers | all_false
    segments = np.cumsum(boundaries)
    authorized_segments = set(segments[np.flatnonzero(all_false & valid_array)])
    candidates = np.flatnonzero(all_true & valid_array)
    if not len(candidates):
        return []
    candidate_segments = segments[candidates]
    keep = np.r_[True, candidate_segments[1:] != candidate_segments[:-1]]
    keep &= np.asarray([segment in authorized_segments for segment in candidate_segments])
    return candidates[keep].tolist()


def kda02b_symbol_events(frame: pd.DataFrame, symbol: str, cell: dict[str, Any],
                         thresholds: dict[str, float], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    work = frame.copy()
    work["timestamp_utc"] = pd.to_datetime(work.timestamp_utc, utc=True, errors="raise")
    work = work.loc[work.timestamp_utc.ge(start - pd.Timedelta(minutes=5)) & work.timestamp_utc.lt(end)].reset_index(drop=True)
    axes = cell["search_axes"]
    sign = -1 if axes["price_state"] == "negative" else 1
    trade_bps = work.trade_return_1h * 10000.0
    mark_bps = work.mark_return_1h * 10000.0
    price = trade_bps.mul(sign).gt(0) & mark_bps.mul(sign).gt(0)
    if axes["price_axis"] == "raw_bps":
        price &= trade_bps.abs().between(14, 500) & mark_bps.abs().between(14, 500)
    else:
        price &= trade_bps.abs().between(thresholds["trade_abs_q80"], thresholds["trade_abs_q100"]) & mark_bps.abs().between(thresholds["mark_abs_q80"], thresholds["mark_abs_q100"])
    oi = work.oi_log_change_1h.between(-.12, -.01) if axes["oi_axis"] == "raw_oi_log_change" else work.oi_log_change_1h.between(thresholds["oi_q0"], thresholds["oi_q20"])
    if axes["liquidation_context"] == "present_absent":
        liquidation = work.liquidation_base_units_1h.gt(0)
    else:
        liquidation = work.liquidation_normalization_valid.fillna(False) & work.liquidation_intensity_robust_z.between(thresholds["liquidation_q80"], thresholds["liquidation_q100"])
    valid = valid_feature_rows(work)
    mandatory = pd.DataFrame({"price": price, "oi": oi, "liquidation": liquidation})
    rows = []
    for index in _episode_onsets(mandatory, work.timestamp_utc, valid):
        decision = work.timestamp_utc.iloc[index] + pd.Timedelta(minutes=5)
        if decision < start or decision >= end:
            continue
        side_sign = sign if axes["branch"] == "continuation" else -sign
        rows.append({"event_id": stable_hash([cell["canonical_translation_id"], symbol, decision.isoformat()]),
                     "translation_id": cell["canonical_translation_id"], "cell_id": cell["cell_id"],
                     "family": "KDA02B", "symbol": symbol, "decision_ts": decision,
                     "side": "long" if side_sign > 0 else "short", "horizon": axes["horizon"]})
    return pd.DataFrame(rows)


def kdx_symbol_events(frame: pd.DataFrame, symbol: str, cell: dict[str, Any],
                      thresholds: dict[str, float], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    work = frame.copy()
    work["timestamp_utc"] = pd.to_datetime(work.timestamp_utc, utc=True, errors="raise")
    work = work.loc[work.timestamp_utc.ge(start - pd.Timedelta(hours=7)) & work.timestamp_utc.lt(end)].reset_index(drop=True)
    axes = cell["search_axes"]
    rank = axes["component_scaling"] == "fold_local_rank"
    required = cell["feature_contract"]["required_components"]
    trade_limit = thresholds["trade_abs_q80"] if rank else 14.0
    mark_limit = thresholds["mark_abs_q80"] if rank else 14.0
    valid = valid_feature_rows(work).to_numpy()
    predicates: dict[str, pd.Series] = {
        "trade": work.trade_return_1h.mul(-10000).between(trade_limit, thresholds["trade_abs_q100"] if rank else 500.0),
        "mark": work.mark_return_1h.mul(-10000).between(mark_limit, thresholds["mark_abs_q100"] if rank else 500.0),
    }
    if "oi_contraction" in required:
        predicates["oi"] = work.oi_log_change_1h.between(thresholds["oi_q0"] if rank else -.12, thresholds["oi_q20"] if rank else -.01)
    if "liquidation_intensity" in required:
        predicates["liquidation"] = (work.liquidation_normalization_valid.fillna(False) & work.liquidation_intensity_robust_z.between(thresholds["liquidation_q80"], thresholds["liquidation_q100"])) if rank else work.liquidation_base_units_1h.gt(0)
    if "negative_basis_change" in required:
        predicates["basis_change"] = work.basis_change_normalization_valid.fillna(False) & work.basis_change_1h.mul(10000).between(thresholds["basis_change_q0"] if rank else -500, thresholds["basis_change_q20"] if rank else -14)
    if "negative_basis_level" in required:
        predicates["basis_level"] = work.basis_level_normalization_valid.fillna(False) & (work.basis_bps.between(thresholds["basis_level_q0"], thresholds["basis_level_q20"]) if rank else work.basis_bps.le(-14))
    if "directional_PIT_breadth" in required:
        if "breadth_share" not in work:
            raise Stage20Error("KDX01 breadth component missing from feature frame")
        predicates["breadth"] = work.breadth_share.between(thresholds["breadth_q80"], thresholds["breadth_q100"]) if rank else work.breadth_share.ge(.02)
    mandatory = pd.DataFrame(predicates)
    truth = mandatory.fillna(False).astype(bool).to_numpy()
    all_true = truth.all(axis=1)
    times = work.timestamp_utc
    nanos = times.dt.as_unit("ns").astype("int64").to_numpy()
    contiguous = np.r_[False, np.diff(nanos) == 300_000_000_000]
    true_indices = np.flatnonzero(all_true & valid)
    rearm_indices = np.flatnonzero((~all_true) & valid)
    invalid_indices = np.flatnonzero(~valid | ~contiguous)
    rows: list[dict[str, Any]] = []
    cursor = int(rearm_indices[0] + 1) if len(rearm_indices) else len(work)
    while cursor < len(work):
        position = int(np.searchsorted(true_indices, cursor, side="left"))
        if position >= len(true_indices):
            break
        open_index = int(true_indices[position])
        if not contiguous[open_index]:
            cursor = open_index + 1
            continue
        onset = times.iloc[open_index]
        last_allowed = int(np.searchsorted(nanos, (onset + pd.Timedelta(hours=6)).value, side="right"))
        invalid_position = int(np.searchsorted(invalid_indices, open_index + 1, side="left"))
        if invalid_position < len(invalid_indices):
            last_allowed = min(last_allowed, int(invalid_indices[invalid_position]))
        scan = np.arange(open_index + 1, min(last_allowed, len(work)))
        if len(scan):
            reclaim_mask = (valid[scan] & contiguous[scan]
                            & work.trade_close.iloc[scan].to_numpy().astype(float).__ge__(float(work.trade_close.iloc[open_index - 1]))
                            & work.mark_close.iloc[scan].to_numpy().astype(float).__ge__(float(work.mark_close.iloc[open_index - 1])))
            reclaim_indices = scan[reclaim_mask]
        else:
            reclaim_indices = np.array([], dtype=int)
        close_index = int(reclaim_indices[0]) if len(reclaim_indices) else min(last_allowed, len(work) - 1)
        if len(reclaim_indices):
            decision = times.iloc[close_index] + pd.Timedelta(minutes=5)
            if start <= decision < end:
                rows.append({"event_id": stable_hash([cell["canonical_translation_id"], symbol, onset.isoformat(), decision.isoformat()]),
                             "translation_id": cell["canonical_translation_id"], "cell_id": cell["cell_id"],
                             "family": "KDX01", "symbol": symbol, "decision_ts": decision,
                             "side": "long", "horizon": axes["horizon"], "onset_ts": onset})
        rearm_position = int(np.searchsorted(rearm_indices, close_index + 1, side="left"))
        cursor = int(rearm_indices[rearm_position] + 1) if rearm_position < len(rearm_indices) else len(work)
    return pd.DataFrame(rows)


def build_kda02c_feature_tape(events: pd.DataFrame, breadth: pd.DataFrame) -> pd.DataFrame:
    work = events.loc[events.event_type.eq("completed_purge_reversal")].copy().reset_index(drop=True)
    work["decision_ts"] = pd.to_datetime(work.decision_ts, utc=True, errors="raise")
    work["decision_source_ts"] = work.decision_ts - pd.Timedelta(minutes=5)
    panel = breadth[["timestamp_utc", "eligible"]].copy()
    panel["timestamp_utc"] = pd.to_datetime(panel.timestamp_utc, utc=True, errors="raise")
    work = work.merge(panel, left_on="decision_source_ts", right_on="timestamp_utc", how="left", validate="many_to_one")
    if work.eligible.isna().any() or work.eligible.le(0).any():
        raise Stage20Error("KDA02C PIT denominator missing or zero")
    identity = {"primary": "primary_z2", "robustness": "robust_pct95"}
    work["purge_identity"] = work.attempt.map(identity)
    if work.purge_identity.isna().any():
        raise Stage20Error("unknown KDA02C purge attempt")
    for window in (5, 15, 30, 60):
        counts = np.zeros(len(work), dtype=np.int64)
        for (attempt, direction), index in work.groupby(["attempt", "parent_direction"], sort=True).groups.items():
            loc = np.asarray(list(index), dtype=int)
            values = work.loc[loc, "decision_source_ts"].sort_values(kind="mergesort")
            nanos = values.astype("int64").to_numpy()
            left = np.searchsorted(nanos, nanos - window * 60 * 1_000_000_000, side="right")
            right = np.searchsorted(nanos, nanos, side="right")
            mapped = pd.Series(right - left, index=values.index)
            counts[mapped.index.to_numpy()] = mapped.to_numpy()
        work[f"count_{window}m"] = counts
        work[f"share_{window}m"] = counts / work.eligible.astype(float)
    if work.event_id.duplicated().any():
        raise Stage20Error("duplicate KDA02C base event identity")
    return work


def kda02c_events(feature_tape: pd.DataFrame, cell: dict[str, Any], training_start: pd.Timestamp,
                  training_end: pd.Timestamp, evaluation_start: pd.Timestamp,
                  evaluation_end: pd.Timestamp) -> pd.DataFrame:
    axes = cell["search_axes"]
    direction = -1 if axes["direction"] == "negative" else 1
    window = int(axes["diagnostic_window"].removesuffix("m"))
    population = feature_tape.loc[
        feature_tape.purge_identity.eq(axes["purge_identity"])
        & feature_tape.parent_direction.eq(direction)
    ]
    training = population.loc[population.decision_ts.ge(training_start) & population.decision_ts.lt(training_end)]
    evaluation = population.loc[population.decision_ts.ge(evaluation_start) & population.decision_ts.lt(evaluation_end)].copy()
    share = evaluation[f"share_{window}m"]
    form = axes["breadth_form"]
    if form == "raw_share":
        accepted = share.ge(.02)
    elif form == "isolated_vs_nonisolated":
        accepted = evaluation[f"count_{window}m"].ge(2)
    else:
        edges = registered_rank_edges(training[f"share_{window}m"])
        accepted = share.between(edges["q80"], edges["q100"])
    evaluation = evaluation.loc[accepted]
    if evaluation.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "event_id": evaluation.event_id,
        "translation_id": cell["canonical_translation_id"], "cell_id": cell["cell_id"],
        "family": "KDA02C", "symbol": evaluation.symbol, "decision_ts": evaluation.decision_ts,
        "side": "long" if direction < 0 else "short", "horizon": "1h",
    }).reset_index(drop=True)


def resolve_execution(events: pd.DataFrame, open_maps: dict[str, pd.Series], fold_start: pd.Timestamp,
                      fold_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    hours = {"1h": 1, "3h": 3, "6h": 6}
    for row in events.sort_values(["decision_ts", "translation_id", "symbol", "event_id"], kind="mergesort").itertuples(index=False):
        source = open_maps.get(row.symbol)
        reason = None
        entry_ts = exit_ts = None
        if source is None:
            reason = "authorized_open_map_unavailable"
        else:
            decision = pd.Timestamp(row.decision_ts)
            target = decision + pd.Timedelta(hours=hours[row.horizon])
            entry_pos = int(source.index.searchsorted(decision, side="left"))
            exit_pos = int(source.index.searchsorted(target, side="left"))
            if entry_pos >= len(source) or source.index[entry_pos] > decision + pd.Timedelta(minutes=10):
                reason = "missing_entry_within_10m"
            elif exit_pos >= len(source) or source.index[exit_pos] > target + pd.Timedelta(minutes=10):
                reason = "missing_exit_within_10m"
            else:
                entry_ts, exit_ts = source.index[entry_pos], source.index[exit_pos]
                if not (fold_start <= decision <= entry_ts <= exit_ts < fold_end and exit_ts < PROTECTED_START):
                    reason = "same_fold_or_protected_boundary_reject"
        if reason:
            rejected.append({"translation_id": row.translation_id, "event_id": row.event_id, "symbol": row.symbol, "reason": reason})
            continue
        record = row._asdict()
        record.update({"entry_ts": entry_ts, "exit_ts": exit_ts, "entry_open": float(source.loc[entry_ts]), "exit_open": float(source.loc[exit_ts])})
        accepted.append(record)
    frame = nonoverlap(pd.DataFrame(accepted)) if accepted else pd.DataFrame()
    return frame, pd.DataFrame(rejected)


def resolve_timestamp_schedule(events: pd.DataFrame, bar_maps: dict[str, pd.DatetimeIndex], fold_start: pd.Timestamp,
                               fold_end: pd.Timestamp, fold_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    hours = {"1h": 1, "3h": 3, "6h": 6}
    for row in events.sort_values(["decision_ts", "translation_id", "symbol", "event_id"], kind="mergesort").itertuples(index=False):
        source = bar_maps.get(row.symbol)
        reason = None
        entry_ts = exit_ts = None
        if source is None:
            reason = "authorized_timestamp_map_unavailable"
        else:
            decision = pd.Timestamp(row.decision_ts)
            target = decision + pd.Timedelta(hours=hours[row.horizon])
            entry_pos = int(source.searchsorted(decision, side="left"))
            exit_pos = int(source.searchsorted(target, side="left"))
            if entry_pos >= len(source) or source[entry_pos] > decision + pd.Timedelta(minutes=10):
                reason = "missing_entry_within_10m"
            elif exit_pos >= len(source) or source[exit_pos] > target + pd.Timedelta(minutes=10):
                reason = "missing_exit_within_10m"
            else:
                entry_ts, exit_ts = source[entry_pos], source[exit_pos]
                if not (fold_start <= decision <= entry_ts <= exit_ts < fold_end and exit_ts < PROTECTED_START):
                    reason = "same_fold_or_protected_boundary_reject"
        if reason:
            rejected.append({"translation_id": row.translation_id, "event_id": row.event_id, "symbol": row.symbol, "reason": reason})
            continue
        record = row._asdict()
        record.update({"entry_ts": entry_ts, "exit_ts": exit_ts,
                       "economic_address": stable_hash([row.translation_id, row.event_id, row.symbol, row.side,
                                                        entry_ts.isoformat(), fold_id])})
        accepted.append(record)
    frame = nonoverlap(pd.DataFrame(accepted)) if accepted else pd.DataFrame()
    if len(frame) and frame.economic_address.duplicated().any():
        raise Stage20Error("duplicate frozen economic address")
    return frame, pd.DataFrame(rejected)


def attach_open_prices(schedule: pd.DataFrame, open_maps: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for row in schedule.itertuples(index=False):
        source = open_maps.get(row.symbol)
        if source is None or row.entry_ts not in source.index or row.exit_ts not in source.index:
            raise Stage20Error(f"frozen timestamp missing from price reader: {row.symbol}:{row.event_id}")
        rows.append({**row._asdict(), "entry_open": float(source.loc[row.entry_ts]), "exit_open": float(source.loc[row.exit_ts])})
    return pd.DataFrame(rows)


def score_executions(executions: pd.DataFrame, funding_engine: Stage19FundingEngine) -> pd.DataFrame:
    if executions.empty:
        return executions.copy()
    rows = []
    for row in executions.itertuples(index=False):
        if row.side not in {"long", "short"}:
            raise Stage20Error(f"invalid frozen side: {row.side}")
        sign = 1 if row.side == "long" else -1
        gross = sign * (float(row.exit_open) / float(row.entry_open) - 1.0) * 10000.0
        funding = funding_engine.evaluate_trade(
            symbol=row.symbol,
            entry=pd.Timestamp(row.entry_ts).to_pydatetime(),
            exit_=pd.Timestamp(row.exit_ts).to_pydatetime(),
            position_sign=sign,
            entry_trade_open=Decimal(str(row.entry_open)),
        )
        adverse = float(funding["adverse_exact_funding_bps"])
        start_funding = min(0.0, float(funding["signed_alignment_start_bps"]))
        end_funding = min(0.0, float(funding["signed_alignment_end_bps"]))
        base_gap = float(funding["base_gap_cost_bps"])
        base = gross - 14 + adverse + base_gap
        stress = gross - 32 + adverse + float(funding["stress_gap_cost_bps"])
        rows.append({
            **row._asdict(), "gross_bps": gross, "base_net_bps": base,
            "stress_net_bps": stress, "funding_adverse_exact_bps": adverse,
            "base_net_alignment_start_bps": gross - 14 + start_funding + base_gap,
            "base_net_alignment_end_bps": gross - 14 + end_funding + base_gap,
            "funding_base_gap_cost_bps": base_gap,
            "funding_stress_gap_cost_bps": float(funding["stress_gap_cost_bps"]),
            "funding_start_alignment_bps": float(funding["signed_alignment_start_bps"]),
            "funding_end_alignment_bps": float(funding["signed_alignment_end_bps"]),
            "funding_missing_start_hours": float(funding["missing_alignment_start_hours"]),
            "funding_missing_end_hours": float(funding["missing_alignment_end_hours"]),
        })
    return pd.DataFrame(rows)


def temporal_models(fold_map: dict[str, Any]) -> dict[str, dict[str, pd.Timestamp]]:
    models: dict[str, dict[str, pd.Timestamp]] = {}
    for outer in fold_map["outer_folds"]:
        quarter = outer["outer_fold_id"].split(":", 1)[1]
        candidates = [(f"Q_{quarter}", outer["development_start"], outer["development_end_exclusive"],
                       outer["outer_evaluation_start"], outer["outer_evaluation_end_exclusive"])]
        candidates.extend((inner["inner_fold_id"], inner["training_start"], inner["training_latest_exit_exclusive"],
                           inner["validation_start"], inner["validation_end_exclusive"]) for inner in outer["inner_folds"])
        for name, train_start, train_end, eval_start, eval_end in candidates:
            row = {"training_start": pd.Timestamp(train_start), "training_end": pd.Timestamp(train_end),
                   "evaluation_start": pd.Timestamp(eval_start), "evaluation_end": pd.Timestamp(eval_end)}
            if name in models and models[name] != row:
                raise Stage20Error(f"temporal model identity drift: {name}")
            models[name] = row
    return dict(sorted(models.items()))


def load_symbol_features(partition: dict[str, Any], breadth: pd.DataFrame) -> pd.DataFrame:
    path = Path(partition["path"])
    if file_sha256(path) != partition["sha256"]:
        raise Stage20Error(f"feature partition drift: {partition['symbol']}")
    frame = guarded_parquet(path, FEATURE_COLUMNS, "timestamp_utc")
    context = breadth[["timestamp_utc", "eligible", "breadth_share"]].rename(columns={"eligible": "breadth_eligible"})
    frame = frame.merge(context, on="timestamp_utc", how="left", validate="one_to_one")
    return frame.sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)


def missing_cell_threshold(cell: dict[str, Any], thresholds: dict[str, Any]) -> str | None:
    axes = cell["search_axes"]
    prefixes: list[str] = []
    if cell["family"] == "KDA02B":
        if axes["oi_axis"] == "fold_local_percentile": prefixes.append("oi")
        if axes["price_axis"] == "fold_local_rank": prefixes.extend(["trade_abs", "mark_abs"])
        if axes["liquidation_context"] == "continuous_intensity": prefixes.append("liquidation")
    elif cell["family"] == "KDX01" and axes["component_scaling"] == "fold_local_rank":
        prefixes.extend(["trade_abs", "mark_abs"])
        required = cell["feature_contract"]["required_components"]
        if "oi_contraction" in required: prefixes.append("oi")
        if "liquidation_intensity" in required: prefixes.append("liquidation")
        if "negative_basis_change" in required: prefixes.append("basis_change")
        if "negative_basis_level" in required: prefixes.append("basis_level")
        if "directional_PIT_breadth" in required: prefixes.append("breadth")
    for prefix in prefixes:
        if f"{prefix}_error" in thresholds:
            return f"{prefix}: {thresholds[f'{prefix}_error']}"
    return None


def merge_kda02c_symbol_partitions(c_frame: pd.DataFrame, events_root: Path,
                                   files: list[dict[str, Any]]) -> None:
    """Merge KDA02C rows into the already-authorized native-symbol partitions."""
    if c_frame.empty:
        return
    records = {row["symbol"]: row for row in files}
    for symbol, additions in c_frame.groupby("symbol", sort=True):
        if symbol not in records:
            raise Stage20Error(f"KDA02C symbol outside the authorized PF cohort: {symbol}")
        target = events_root / f"{symbol}.parquet"
        existing = pd.read_parquet(target)
        combined = pd.concat([existing, additions], ignore_index=True)
        combined.to_parquet(target, index=False, compression="zstd")
        record = records[symbol]
        record.update({"rows": len(combined), "bytes": target.stat().st_size,
                       "sha256": file_sha256(target)})


def build_preoutcome_event_tapes(repo: Path, output: Path, thresholds_source: Path | None = None) -> dict[str, Any]:
    stage19 = repo / STAGE19_REL
    registry = json.loads((stage19 / "ECONOMIC_TRANSLATION_REGISTRY.json").read_text(encoding="utf-8"))
    folds = json.loads((stage19 / "INNER_FOLD_MAP.json").read_text(encoding="utf-8"))
    models = temporal_models(folds)
    if thresholds_source is None:
        thresholds = fit_global_thresholds(repo, {name: (row["training_start"], row["training_end"]) for name, row in models.items()})
    else:
        frozen = json.loads(thresholds_source.read_text(encoding="utf-8"))
        if set(frozen.get("models", {})) != set(models):
            raise Stage20Error("frozen threshold model identity mismatch")
        thresholds = {}
        for name, model in models.items():
            source = frozen["models"][name]
            for field in ("training_start", "training_end", "evaluation_start", "evaluation_end"):
                if pd.Timestamp(source[field]) != model[field]:
                    raise Stage20Error(f"frozen threshold temporal drift: {name}:{field}")
            thresholds[name] = {
                key: (str(value) if key.endswith("_error") else float(value))
                for key, value in source["thresholds"].items()
            }
    output.mkdir(parents=True, exist_ok=False)
    events_root = output / "events"
    events_root.mkdir()
    atomic_json(output / "FOLD_LOCAL_THRESHOLDS.json", {
        "method": "Hyndman-Fan type 7 on full eligible inner-training PIT observations",
        "models": {name: {**{key: value.isoformat() for key, value in models[name].items()}, "thresholds": thresholds[name]} for name in models},
    })
    breadth = guarded_parquet(STAGE14_STATE_ROOT / "KDA02C_PIT_BREADTH_PANEL.parquet",
                              pq.ParquetFile(STAGE14_STATE_ROOT / "KDA02C_PIT_BREADTH_PANEL.parquet").schema_arrow.names,
                              "timestamp_utc")
    kdx_breadth = kdx_breadth_panel()
    cells_b = [row for row in registry["cells"] if row["family"] == "KDA02B"]
    cells_x = [row for row in registry["cells"] if row["family"] == "KDX01"]
    groups_b: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    groups_x: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for cell in cells_b:
        axes = cell["search_axes"]
        key = tuple(axes[name] for name in ("price_state", "oi_axis", "price_axis", "liquidation_context"))
        groups_b.setdefault(key, []).append(cell)
    for cell in cells_x:
        axes = cell["search_axes"]
        key = (axes["component_ladder"], axes["component_scaling"])
        groups_x.setdefault(key, []).append(cell)
    feature_manifest = json.loads((repo / STAGE8A_REL / "KDA_FEATURE_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    files: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    total_rows = 0
    for partition in sorted(feature_manifest["partitions"], key=lambda row: row["symbol"]):
        frame = load_symbol_features(partition, kdx_breadth)
        symbol_rows: list[pd.DataFrame] = []
        for model_id, model in models.items():
            for group in groups_b.values():
                reason = missing_cell_threshold(group[0], thresholds[model_id])
                if reason:
                    skips.extend({"model_id": model_id, "cell_id": cell["cell_id"], "reason": reason} for cell in group)
                    continue
                base = kda02b_symbol_events(frame, partition["symbol"], group[0], thresholds[model_id],
                                            model["evaluation_start"], model["evaluation_end"])
                for cell in group:
                    if base.empty:
                        continue
                    expanded = base.copy()
                    axes = cell["search_axes"]
                    sign = -1 if axes["price_state"] == "negative" else 1
                    sign = sign if axes["branch"] == "continuation" else -sign
                    expanded["translation_id"] = cell["canonical_translation_id"]
                    expanded["cell_id"] = cell["cell_id"]
                    expanded["side"] = "long" if sign > 0 else "short"
                    expanded["horizon"] = axes["horizon"]
                    expanded["model_id"] = model_id
                    expanded["event_id"] = [stable_hash([cell["canonical_translation_id"], partition["symbol"], value.isoformat(), model_id]) for value in expanded.decision_ts]
                    symbol_rows.append(expanded)
            for group in groups_x.values():
                reason = missing_cell_threshold(group[0], thresholds[model_id])
                if reason:
                    skips.extend({"model_id": model_id, "cell_id": cell["cell_id"], "reason": reason} for cell in group)
                    continue
                base = kdx_symbol_events(frame, partition["symbol"], group[0], thresholds[model_id],
                                         model["evaluation_start"], model["evaluation_end"])
                for cell in group:
                    if base.empty:
                        continue
                    expanded = base.copy()
                    expanded["translation_id"] = cell["canonical_translation_id"]
                    expanded["cell_id"] = cell["cell_id"]
                    expanded["horizon"] = cell["search_axes"]["horizon"]
                    expanded["model_id"] = model_id
                    expanded["event_id"] = [stable_hash([cell["canonical_translation_id"], partition["symbol"], value.isoformat(), model_id]) for value in expanded.decision_ts]
                    symbol_rows.append(expanded)
        symbol_frame = pd.concat(symbol_rows, ignore_index=True) if symbol_rows else pd.DataFrame()
        target = events_root / f"{partition['symbol']}.parquet"
        symbol_frame.to_parquet(target, index=False, compression="zstd")
        files.append({"path": str(target), "symbol": partition["symbol"], "rows": len(symbol_frame),
                      "bytes": target.stat().st_size, "sha256": file_sha256(target)})
        total_rows += len(symbol_frame)
    base_events = guarded_parquet(KDA02_EVENT_TAPE,
                                  pq.ParquetFile(KDA02_EVENT_TAPE).schema_arrow.names,
                                  "decision_ts")
    tape_c = build_kda02c_feature_tape(base_events, breadth)
    c_rows: list[pd.DataFrame] = []
    for model_id, model in models.items():
        for cell in [row for row in registry["cells"] if row["family"] == "KDA02C"]:
            try:
                frame = kda02c_events(tape_c, cell, model["training_start"], model["training_end"],
                                      model["evaluation_start"], model["evaluation_end"])
            except Stage20Error as exc:
                skips.append({"model_id": model_id, "cell_id": cell["cell_id"], "reason": str(exc)})
                continue
            if not frame.empty:
                frame["model_id"] = model_id
                frame["event_id"] = [stable_hash([cell["canonical_translation_id"], value, model_id]) for value in frame.event_id]
                c_rows.append(frame)
    c_frame = pd.concat(c_rows, ignore_index=True) if c_rows else pd.DataFrame()
    # Keep every lane in the same native-PF-symbol partitions.  The economic
    # workers verify one official trade authority and one open map per symbol;
    # a synthetic KDA02C filename would bypass that ownership contract.
    merge_kda02c_symbol_partitions(c_frame, events_root, files)
    total_rows += len(c_frame)
    if total_rows and sum(row["rows"] for row in files) != total_rows:
        raise Stage20Error("preoutcome event-tape reconciliation failed")
    unique_skips = sorted({(row["model_id"], row["cell_id"], row["reason"]) for row in skips})
    atomic_json(output / "MECHANICAL_CELL_SKIPS.json", {"skips": [dict(model_id=a, cell_id=b, reason=c) for a, b, c in unique_skips]})
    result = {"status": "pass", "created_at_utc": utc_now(), "models": len(models), "registered_cells": 186,
              "event_rows": total_rows, "files": files, "protected_rows_opened": 0,
              "mechanical_cell_model_skips": len(unique_skips),
              "Capitalcom_payload_opened": False, "economic_outcome_reader_opened": False}
    atomic_json(output / "PREOUTCOME_EVENT_TAPE_MANIFEST.json", result)
    return result
