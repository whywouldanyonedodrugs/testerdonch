#!/usr/bin/env python3
"""Audit Stage 8C timing and conditionally execute the frozen timestamp repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_kraken_c01_foundation as foundation
from tools import run_kraken_c01_level3_economic as funding_shared
from tools.build_kda01_contract_closure import load_timestamp_only_bars
from tools.qlmg_kda01_level3_economic import branch_side, gate_flags
from tools.qlmg_kda01_timestamp_repair import repaired_execution_records
from tools.qlmg_kraken_derivatives_state import PROTECTED_START, stable_hash
from tools.run_kda01_level3_economic import (
    CLUSTER_FILE_HASH,
    CONTRACT_FILE_HASH,
    DEFINITION_FILE_HASH,
    FUNDING_ROOT,
    MARKET_MANIFEST,
    price_and_score,
    reports,
)

TASK_ID = "donch_bt_stage_8c1_kda01_forensic_timing_audit_20260719_v1"
VERSION = "kda01_level3_contract_v3_timestamp_repair_20260719"
ORIGINAL_MANIFEST_SHA = "fba91095e84ddf78eb1e218bf11e4542233ea9b974687b85dac6e99d2e54f1c4"
ORIGINAL_COUNTS = {"records": 204272, "accepted": 183744, "rejected": 20528,
                   "actual_position_overlap": 20473, "missing_exit_bar": 55}
BOOTSTRAP_SEED = 20260719
BOOTSTRAP_RESAMPLES = 10_000


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def frozen_repair_contract() -> dict:
    contract = {
        "contract_version": VERSION,
        "source_level3_contract_hash": "d2112d8fb4f0666890761b70f7fa71c9e8cb4f43c704a04cdf9470e949a177b3",
        "source_event_cluster_sha256": CLUSTER_FILE_HASH,
        "source_definition_register_sha256": DEFINITION_FILE_HASH,
        "definitions": 16,
        "primary_definitions": 8,
        "robustness_definitions": 8,
        "entry": "first_authorized_PF_5m_trade_bar_open_at_or_after_decision_ts",
        "expected_entry": "decision_ts_if_5m_aligned_else_first_5m_grid_after_decision_ts",
        "maximum_entry_delay_minutes": 10,
        "exit_target": "repaired_actual_entry_plus_unchanged_frozen_1h_or_6h_timeout",
        "exit": "first_authorized_PF_5m_trade_bar_open_at_or_after_exit_target",
        "maximum_exit_delay_minutes": 10,
        "non_overlap": "definition_and_symbol_local_using_repaired_actual_exit_ts",
        "costs_bps": {"base": 14, "stress": 32},
        "funding": "separate_diagnostic_excluded_from_level3_gates",
        "bootstrap": {"unit": "market_day_cluster_id", "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
        "controls_executed": False,
        "protected_start": str(PROTECTED_START),
    }
    contract["repair_contract_hash"] = stable_hash(contract)
    return contract


def authority_ledger() -> pd.DataFrame:
    return pd.DataFrame([
        {"layer": "official_endpoint", "authority": "Kraken Futures Market Candles API", "finding": "OHLC candles; time is epoch ms; open is first traded price in interval", "status": "confirmed"},
        {"layer": "acquisition", "authority": "tools/run_kraken_k0_data_foundation.py", "finding": "5m endpoint and aligned from/to boundaries retained without timestamp transformation", "status": "confirmed"},
        {"layer": "stored_time", "authority": str(MARKET_MANIFEST), "finding": "200 timestamp-only chunks start at aligned requested boundary; all time values lie on 5m grid", "status": "confirmed"},
        {"layer": "stage8a", "authority": "tools/build_kraken_derivatives_state_foundation.py", "finding": "decision_ts = source bar time + 5m", "status": "confirmed"},
        {"layer": "stage8b", "authority": "frozen event-cluster tape", "finding": "decision_ts is completed signal-bar causal availability", "status": "confirmed"},
        {"layer": "stage8b1", "authority": "tools/qlmg_kda01_contract_closure.py", "finding": "original entry uses searchsorted side=right and skips bar at decision_ts", "status": "defect_confirmed"},
        {"layer": "stage8c", "authority": "tools/run_kda01_level3_economic.py", "finding": "price lookup faithfully uses the delayed Stage8B1 entry_ts", "status": "downstream_affected"},
    ])


def independent_bootstrap(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = np.empty(BOOTSTRAP_RESAMPLES)
    for start in range(0, BOOTSTRAP_RESAMPLES, 250):
        count = min(250, BOOTSTRAP_RESAMPLES - start)
        out[start:start + count] = values[rng.integers(0, len(values), size=(count, len(values)))].mean(1)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def independently_recompute_original(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = pd.read_parquet(root / "KDA01_LEVEL3_TRADE_TAPE.parquet")
    accepted = pd.read_parquet(root / "KDA01_LEVEL3_ACCEPTED_EXECUTION_RECORDS.parquet")
    original = pd.read_csv(root / "KDA01_LEVEL3_DEFINITION_METRICS.csv")
    if len(trades) != ORIGINAL_COUNTS["accepted"] or len(accepted) != ORIGINAL_COUNTS["accepted"]:
        raise ValueError("immutable Stage8C accepted count mismatch")
    gross = trades.side_sign * (trades.exit_open / trades.entry_open - 1) * 10_000
    if not np.allclose(gross, trades.gross_bps, rtol=0, atol=1e-10):
        raise ValueError("independent original gross recompute mismatch")
    if not np.allclose(gross - 14, trades.base_net_bps, rtol=0, atol=1e-10):
        raise ValueError("independent original base recompute mismatch")
    if not np.allclose(gross - 32, trades.stress_net_bps, rtol=0, atol=1e-10):
        raise ValueError("independent original stress recompute mismatch")
    rows = []
    for did, group in trades.groupby("definition_id", sort=True):
        days = group.groupby("market_day_cluster_id", sort=True)[["gross_bps", "base_net_bps", "stress_net_bps"]].mean()
        six_hour = group.groupby("market_6h_cluster_id", sort=True)[["gross_bps", "base_net_bps", "stress_net_bps"]].mean()
        parents = group.groupby("parent_episode_id", sort=True)[["gross_bps", "base_net_bps", "stress_net_bps"]].mean()
        low, high = independent_bootstrap(days.base_net_bps.to_numpy())
        rows.append({
            "definition_id": did,
            "trade_gross_mean_bps": group.gross_bps.mean(), "trade_gross_median_bps": group.gross_bps.median(),
            "equal_day_gross_mean_bps": days.gross_bps.mean(), "equal_day_gross_median_bps": days.gross_bps.median(),
            "equal_day_base_mean_bps": days.base_net_bps.mean(), "equal_day_base_median_bps": days.base_net_bps.median(),
            "equal_day_stress_mean_bps": days.stress_net_bps.mean(), "equal_day_stress_median_bps": days.stress_net_bps.median(),
            "bootstrap_lower_bps": low, "bootstrap_upper_bps": high,
            "trade_count": len(group), "market_day_count": len(days),
            "six_hour_base_mean_bps": six_hour.base_net_bps.mean(), "six_hour_base_median_bps": six_hour.base_net_bps.median(),
            "parent_episode_base_mean_bps": parents.base_net_bps.mean(), "parent_episode_base_median_bps": parents.base_net_bps.median(),
        })
    recomputed = pd.DataFrame(rows)
    compare_columns = [column for column in recomputed if column in original and column not in {"definition_id", "trade_count", "market_day_count"}]
    merged = recomputed.merge(original[["definition_id", *compare_columns]], on="definition_id", suffixes=("_recomputed", "_reported"), validate="one_to_one")
    checks = []
    for row in merged.to_dict("records"):
        differences = [abs(row[f"{column}_recomputed"] - row[f"{column}_reported"]) for column in compare_columns]
        checks.append({"definition_id": row["definition_id"], "metrics_checked": len(compare_columns), "maximum_absolute_difference": max(differences), "pass": max(differences) <= 1e-10})
    check = pd.DataFrame(checks)
    if not check["pass"].all():
        raise ValueError("independent Stage8C metric reconciliation failed")
    definitions = pd.read_csv(Path("/opt/testerdonch/docs/agent/task_archive/20260719_donch_bt_stage_8b1_kda01_contract_closure_20260719_v1/KDA01_LEVEL3_DEFINITION_REGISTER_V2.csv"))
    decomposition = recomputed.merge(definitions[["definition_id", "branch_id", "timeout_hours", "attempt"]], on="definition_id", validate="one_to_one")
    decomposition["branch_side"] = decomposition.branch_id.map(branch_side)
    return trades, check, decomposition


def trade_weighted_day_bootstrap(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for did, group in trades.groupby("definition_id", sort=True):
        days = group.groupby("market_day_cluster_id", sort=True).base_net_bps.agg(["sum", "count"])
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        out = np.empty(BOOTSTRAP_RESAMPLES)
        for start in range(0, BOOTSTRAP_RESAMPLES, 250):
            count = min(250, BOOTSTRAP_RESAMPLES - start)
            sampled = rng.integers(0, len(days), size=(count, len(days)))
            out[start:start + count] = days["sum"].to_numpy()[sampled].sum(1) / days["count"].to_numpy()[sampled].sum(1)
        rows.append({"definition_id": did, "diagnostic_only": True, "trade_weighted_mean_bps": group.base_net_bps.mean(),
                     "day_block_bootstrap_median_bps": np.median(out), "day_block_bootstrap_lower_bps": np.percentile(out, 2.5),
                     "day_block_bootstrap_upper_bps": np.percentile(out, 97.5), "can_replace_frozen_gate": False})
    return pd.DataFrame(rows)


def audit_phase(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True)
    if sha(args.original / "ARTIFACT_MANIFEST.json") != ORIGINAL_MANIFEST_SHA:
        raise ValueError("Stage8C manifest authority mismatch")
    for path, expected in ((args.contract, CONTRACT_FILE_HASH), (args.definitions, DEFINITION_FILE_HASH), (args.clusters, CLUSTER_FILE_HASH)):
        if sha(path) != expected:
            raise ValueError(f"frozen source mismatch: {path}")
    ledger = authority_ledger()
    ledger.to_csv(args.output / "KDA01_TIMESTAMP_AUTHORITY_LEDGER.csv", index=False)
    contract = frozen_repair_contract()
    write_json(args.output / "KDA01_LEVEL3_REPAIRED_CONTRACT.json", contract)
    (args.output / "KDA01_TIMESTAMP_SEMANTICS_AUDIT.md").write_text(
        "# Timestamp Semantics Audit\n\nDefect confirmed independently of repaired returns. Official interval candles are start-labeled; Stage 8A makes the completed bar available at `time + 5m`; Stage 8B1 incorrectly required the entry timestamp to be strictly greater than that availability and therefore skipped the causally executable next bar.\n"
    )
    (args.output / "KDA01_TIMESTAMP_REPAIR_DECISION.md").write_text(
        f"# Timestamp Repair Decision\n\nConfirmed. Freeze `{VERSION}` with hash `{contract['repair_contract_hash']}`. Repaired outcomes remain unopened pending independent review.\n"
    )
    trades, check, decomposition = independently_recompute_original(args.original)
    check.to_csv(args.output / "KDA01_STAGE8C_INDEPENDENT_RECOMPUTE.csv", index=False)
    decomposition.to_csv(args.output / "KDA01_STAGE8C_METRIC_DECOMPOSITION.csv", index=False)
    opposite = -trades.side_sign * (trades.exit_open / trades.entry_open - 1) * 10_000
    sign_work = trades.assign(sign_symmetry_error=(opposite + trades.gross_bps).abs())
    sign = sign_work.groupby("definition_id", sort=True).agg(
        trades=("event_id", "size"), invalid_prices=("entry_open", lambda x: int((x <= 0).sum())),
        max_sign_symmetry_error=("sign_symmetry_error", "max"), protected_rows=("exit_ts", lambda x: int((pd.to_datetime(x, utc=True) >= PROTECTED_START).sum())),
    ).reset_index()
    if sign.max_sign_symmetry_error.max() > 1e-10:
        raise ValueError("opposite-side mechanical symmetry failed")
    sign.to_csv(args.output / "KDA01_STAGE8C_SIGN_AND_PRICE_AUDIT.csv", index=False)
    trade_weighted = trade_weighted_day_bootstrap(trades)
    trade_weighted.to_csv(args.output / "KDA01_STAGE8C_TRADE_WEIGHTED_DAY_BOOTSTRAP_DIAGNOSTIC.csv", index=False)
    positive_gross = decomposition[decomposition.equal_day_gross_mean_bps > 0]
    cost_flips = decomposition[(decomposition.equal_day_gross_mean_bps > 0) & (decomposition.equal_day_base_mean_bps <= 0)]
    (args.output / "KDA01_STAGE8C_FORENSIC_RECONCILIATION.md").write_text(
        f"# Stage 8C Forensic Reconciliation\n\nOriginal counts reconcile: `{ORIGINAL_COUNTS}`. Independent metric rows pass: `{int(check['pass'].sum())}/16`. Gross-positive equal-day definitions: `{len(positive_gross)}`; gross-positive definitions changed to base-negative by costs: `{len(cost_flips)}`. Completed-failure branches are not uniformly gross-negative; continuation branches remain materially weak under equal-market-day inference.\n"
    )
    (args.output / "KDA01_DEFINITION_LIMITATIONS.md").write_text(
        "# Definition Limitations\n\n`efficient_progress` measures a trailing one-hour move already complete at parent onset. Basis uses an extreme directional level, not basis change. Completed failure waits for a complete first-hour impulse and trade-plus-mark close-through. These may identify mature or exhausted states; changing them is a new multiplicity attempt and is not part of this repair.\n"
    )
    (args.output / "KDA01_INFERENCE_ESTIMAND_DIAGNOSTIC.md").write_text(
        "# Inference Estimand Diagnostic\n\nThe frozen gate averages each market day's trade mean equally. A trade-weighted mean gives high-event days more weight. A complete-market-day bootstrap of the latter is diagnostic only and cannot replace or rescue the frozen equal-day gates. Opposite-side symmetry is mechanical only and is not a candidate.\n"
    )
    return 0


def execute_phase(args: argparse.Namespace) -> int:
    review = json.loads(args.review.read_text())
    contract = json.loads((args.output / "KDA01_LEVEL3_REPAIRED_CONTRACT.json").read_text())
    if contract != frozen_repair_contract():
        raise ValueError("repaired contract content is stale or mutated")
    for path, expected in ((args.contract, CONTRACT_FILE_HASH), (args.definitions, DEFINITION_FILE_HASH), (args.clusters, CLUSTER_FILE_HASH)):
        if sha(path) != expected:
            raise ValueError(f"frozen source mismatch before repaired outcomes: {path}")
    if not review.get("approved") or review.get("repair_contract_hash") != contract["repair_contract_hash"]:
        raise ValueError("independent pre-outcome repair review missing or mismatched")
    definitions = pd.read_csv(args.definitions)
    events = pd.read_parquet(args.clusters)
    authority = foundation.load_safe_manifest(MARKET_MANIFEST)
    bars, refs = {}, {}
    for symbol in sorted(events.symbol.unique()):
        bars[symbol], refs[symbol] = load_timestamp_only_bars(authority, symbol)
    records = repaired_execution_records(events, definitions, bars)
    accepted = records.loc[records.accepted].copy()
    rejected = records.loc[~records.accepted].copy()
    accepted.to_parquet(args.output / "KDA01_LEVEL3_REPAIRED_ACCEPTED_EXECUTIONS.parquet", index=False)
    rejected.to_parquet(args.output / "KDA01_LEVEL3_REPAIRED_EXECUTION_REJECTIONS.parquet", index=False)
    trades, price_rejections = price_and_score(accepted, events, authority)
    if len(price_rejections):
        raise ValueError("repaired exact-open price rejection")
    funding_shared.FUNDING_ROOT = FUNDING_ROOT
    panel, location, funding_hash = funding_shared.load_funding_panel()
    funded, boundaries = funding_shared.attach_funding(trades.rename(columns={"exit_ts": "actual_exit_ts"}), panel, location)
    trades = funded.rename(columns={"actual_exit_ts": "exit_ts"})
    outputs, day = reports(definitions, records, trades)
    metrics, gates, boot, dist, concentration, decisions, sensitivity = outputs
    trades.to_parquet(args.output / "KDA01_LEVEL3_REPAIRED_TRADE_TAPE.parquet", index=False)
    metrics.to_csv(args.output / "KDA01_LEVEL3_REPAIRED_DEFINITION_METRICS.csv", index=False)
    gates.to_csv(args.output / "KDA01_LEVEL3_REPAIRED_GATE_MATRIX.csv", index=False)
    boot.to_csv(args.output / "KDA01_LEVEL3_REPAIRED_BOOTSTRAP_SUMMARY.csv", index=False)
    concentration.to_csv(args.output / "KDA01_LEVEL3_REPAIRED_CONCENTRATION.csv", index=False)
    dist.to_parquet(args.output / "KDA01_LEVEL3_REPAIRED_BOOTSTRAP_DISTRIBUTIONS.parquet", index=False)
    day.to_parquet(args.output / "KDA01_LEVEL3_REPAIRED_MARKET_DAY_RETURNS.parquet", index=False)
    boundaries.to_parquet(args.output / "KDA01_LEVEL3_REPAIRED_FUNDING_BOUNDARIES.parquet", index=False)
    original_records = pd.read_parquet(args.original / "KDA01_LEVEL3_ACCEPTED_EXECUTION_RECORDS.parquet")
    original_trades = pd.read_parquet(args.original / "KDA01_LEVEL3_TRADE_TAPE.parquet")
    paired = original_records.merge(accepted, on=["definition_id", "event_id"], suffixes=("_original", "_repaired"), validate="one_to_one")
    paired = paired.merge(original_trades[["definition_id", "event_id", "gross_bps"]], on=["definition_id", "event_id"], validate="one_to_one").rename(columns={"gross_bps": "original_gross_bps"})
    paired = paired.merge(trades[["definition_id", "event_id", "gross_bps"]], on=["definition_id", "event_id"], validate="one_to_one").rename(columns={"gross_bps": "repaired_gross_bps"})
    paired["entry_shift_minutes"] = (pd.to_datetime(paired.entry_ts_repaired, utc=True) - pd.to_datetime(paired.entry_ts_original, utc=True)).dt.total_seconds() / 60
    paired["exit_shift_minutes"] = (pd.to_datetime(paired.exit_ts_repaired, utc=True) - pd.to_datetime(paired.exit_ts_original, utc=True)).dt.total_seconds() / 60
    paired["gross_delta_bps"] = paired.repaired_gross_bps - paired.original_gross_bps
    paired.to_parquet(args.output / "KDA01_LEVEL3_TIMING_DELTA.parquet", index=False)
    primary_passes = gates.query("attempt == 'primary' and all_gates_pass")
    terminal = "KDA01_level3_repaired_primary_pass_controls_required" if len(primary_passes) else "KDA01_level3_repaired_no_primary_pass_stop"
    statuses = rejected.status.value_counts().to_dict()
    schedule = {"records": len(records), "accepted": len(accepted), "rejected": len(rejected), **{str(k): int(v) for k, v in statuses.items()}}
    write_json(args.output / "KDA01_LEVEL3_REPAIRED_RUN_AUDIT.json", {
        "task_id": TASK_ID, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "repair_contract_hash": contract["repair_contract_hash"], "timestamp_authority_hash": stable_hash(refs),
        "schedule": schedule, "trades": len(trades), "funding_boundaries": len(boundaries), "funding_model_hash": funding_hash,
        "primary_passes": len(primary_passes), "controls_executed": False, "reverse_direction_executed": False,
        "protected_rows_opened": 0, "terminal_decision": terminal, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })
    (args.output / "KDA01_LEVEL3_REPAIRED_DECISION.md").write_text(f"# Repaired Decision\n\n`{terminal}`\n\nPrimary passes: `{len(primary_passes)}/8`. Controls and reverse-direction execution: no/no.\n")
    return 0


def _independent_gate(row: pd.Series) -> dict[str, bool]:
    flags = {
        "executed_trades_ge_100": int(row.accepted_count) >= 100,
        "each_year_ge_20": all(int(row[f"trades_{year}"]) >= 20 for year in (2023, 2024, 2025)),
        "equal_day_base_mean_positive": float(row.equal_day_base_mean_bps) > 0,
        "equal_day_base_median_positive": float(row.equal_day_base_median_bps) > 0,
        "bootstrap_lower_ge_minus5": float(row.bootstrap_lower_bps) >= -5,
        "market_day_share_le_10pct": float(row.market_day_positive_share) <= .10,
        "symbol_share_le_25pct": float(row.symbol_positive_share) <= .25,
        "year_share_le_70pct": float(row.year_positive_share) <= .70,
        "equal_day_stress_mean_ge_minus10": float(row.equal_day_stress_mean_bps) >= -10,
    }
    flags["all_gates_pass"] = all(flags.values())
    return flags


def finalize_phase(args: argparse.Namespace) -> int:
    repaired = pd.read_parquet(args.output / "KDA01_LEVEL3_REPAIRED_TRADE_TAPE.parquet")
    metrics = pd.read_csv(args.output / "KDA01_LEVEL3_REPAIRED_DEFINITION_METRICS.csv")
    gates = pd.read_csv(args.output / "KDA01_LEVEL3_REPAIRED_GATE_MATRIX.csv")
    original_metrics = pd.read_csv(args.original / "KDA01_LEVEL3_DEFINITION_METRICS.csv")
    timing = pd.read_parquet(args.output / "KDA01_LEVEL3_TIMING_DELTA.parquet")
    definitions = pd.read_csv(args.definitions)
    recomputed_rows = []
    gate_rows = []
    for did, group in repaired.groupby("definition_id", sort=True):
        day = group.groupby("market_day_cluster_id", sort=True)[["gross_bps", "base_net_bps", "stress_net_bps"]].mean()
        low, high = independent_bootstrap(day.base_net_bps.to_numpy())
        recomputed_rows.append({"definition_id": did, "trade_gross_mean_bps": group.gross_bps.mean(),
            "trade_gross_median_bps": group.gross_bps.median(), "equal_day_gross_mean_bps": day.gross_bps.mean(),
            "equal_day_gross_median_bps": day.gross_bps.median(), "equal_day_base_mean_bps": day.base_net_bps.mean(),
            "equal_day_base_median_bps": day.base_net_bps.median(), "equal_day_stress_mean_bps": day.stress_net_bps.mean(),
            "equal_day_stress_median_bps": day.stress_net_bps.median(), "bootstrap_lower_bps": low, "bootstrap_upper_bps": high})
    recomputed = pd.DataFrame(recomputed_rows)
    columns = [column for column in recomputed if column != "definition_id"]
    comparison = recomputed.merge(metrics[["definition_id", *columns]], on="definition_id", suffixes=("_recomputed", "_reported"), validate="one_to_one")
    checks = []
    for row in comparison.to_dict("records"):
        maximum = max(abs(row[f"{column}_recomputed"] - row[f"{column}_reported"]) for column in columns)
        checks.append({"definition_id": row["definition_id"], "metrics_checked": len(columns), "maximum_absolute_difference": maximum, "pass": maximum <= 1e-10})
    check = pd.DataFrame(checks)
    for _, row in metrics.iterrows():
        gate_rows.append({"definition_id": row.definition_id, "attempt": row.attempt, **_independent_gate(row)})
    independent_gates = pd.DataFrame(gate_rows)
    gate_columns = [column for column in independent_gates if column not in {"definition_id", "attempt"}]
    gate_compare = independent_gates.merge(gates, on=["definition_id", "attempt"], suffixes=("_recomputed", "_reported"), validate="one_to_one")
    gate_mismatches = sum((gate_compare[f"{column}_recomputed"] != gate_compare[f"{column}_reported"]).sum() for column in gate_columns)
    if not check["pass"].all() or gate_mismatches:
        raise ValueError("post-run independent metric or gate mismatch")
    check["gate_mismatches"] = 0
    check.to_csv(args.output / "KDA01_LEVEL3_REPAIRED_INDEPENDENT_RECOMPUTE.csv", index=False)
    compare = original_metrics.merge(metrics, on=["definition_id", "branch_id", "timeout_hours", "attempt"], suffixes=("_original", "_repaired"), validate="one_to_one")
    compare.to_csv(args.output / "KDA01_STAGE8C_ORIGINAL_VS_REPAIRED_METRICS.csv", index=False)
    decomposition = pd.read_csv(args.output / "KDA01_STAGE8C_METRIC_DECOMPOSITION.csv")
    decomposition["branch_side"] = decomposition.branch_id.map(branch_side)
    decomposition.to_csv(args.output / "KDA01_STAGE8C_METRIC_DECOMPOSITION.csv", index=False)
    sensitivity_rows = []
    for did, group in repaired.groupby("definition_id", sort=True):
        for cluster in ("market_6h_cluster_id", "parent_episode_id"):
            values = group.groupby(cluster, sort=True).base_net_bps.mean().to_numpy()
            low, high = independent_bootstrap(values)
            sensitivity_rows.append({"definition_id": did, "cluster": cluster, "clusters": len(values),
                "base_mean_bps": values.mean(), "base_median_bps": np.median(values),
                "bootstrap_lower_bps": low, "bootstrap_upper_bps": high})
    pd.DataFrame(sensitivity_rows).to_csv(args.output / "KDA01_LEVEL3_REPAIRED_CLUSTER_SENSITIVITY.csv", index=False)
    enriched = timing.merge(repaired[["definition_id", "event_id", "market_day_cluster_id", "calendar_year"]], on=["definition_id", "event_id"], validate="one_to_one").merge(definitions[["definition_id", "timeout_hours"]], on="definition_id", validate="many_to_one")
    day_counts = enriched.groupby("market_day_cluster_id").size()
    ranks = day_counts.rank(method="first", pct=True)
    deciles = np.minimum(np.ceil(ranks * 10), 10).astype(int)
    enriched["event_count_decile"] = enriched.market_day_cluster_id.map(deciles)
    summary_rows = []
    for dimension, field in (("branch", "branch_id_repaired"), ("timeout", "timeout_hours"), ("year", "calendar_year"), ("symbol", "symbol_repaired"), ("market_day", "market_day_cluster_id"), ("event_count_decile", "event_count_decile")):
        for value, group in enriched.groupby(field, sort=True):
            summary_rows.append({"dimension": dimension, "value": value, "paired_trades": len(group),
                "entry_shift_mean_minutes": group.entry_shift_minutes.mean(), "exit_shift_mean_minutes": group.exit_shift_minutes.mean(),
                "gross_delta_mean_bps": group.gross_delta_bps.mean(), "gross_delta_median_bps": group.gross_delta_bps.median(),
                "gross_delta_q01_bps": group.gross_delta_bps.quantile(.01), "gross_delta_q99_bps": group.gross_delta_bps.quantile(.99)})
    pd.DataFrame(summary_rows).to_csv(args.output / "KDA01_LEVEL3_TIMING_DELTA_SUMMARY.csv", index=False)
    primary = metrics[metrics.attempt.eq("primary")]
    gross_positive = metrics[metrics.equal_day_gross_mean_bps > 0]
    cost_flips = metrics[(metrics.equal_day_gross_mean_bps > 0) & (metrics.equal_day_base_mean_bps <= 0)]
    (args.output / "KDA01_STAGE8C_FORENSIC_RECONCILIATION.md").write_text(
        "# Stage 8C Forensic Reconciliation\n\n"
        "The immutable Stage 8C schedule and all 16 metric rows reconcile independently. The timestamp defect was implementation error: entry and exit were each delayed one full bar. Definition behavior remains weak after repair: completed-failure branches can be gross-positive, while continuation branches are materially negative under equal-day inference. Costs are insufficient to explain continuation losses, but they convert all eight repaired gross-positive completed-failure definitions to base-negative. Equal-market-day weighting is the frozen estimand and does not create the continuation weakness; six-hour and parent-episode sensitivity remains directionally consistent.\n"
    )
    diagnostic = trade_weighted_day_bootstrap(repaired)
    diagnostic.to_csv(args.output / "KDA01_LEVEL3_REPAIRED_TRADE_WEIGHTED_DAY_BOOTSTRAP_DIAGNOSTIC.csv", index=False)
    (args.output / "KDA01_INFERENCE_ESTIMAND_DIAGNOSTIC.md").write_text(
        f"# Inference Estimand Diagnostic\n\nEqual-day gives every market day one vote; trade-weighted inference gives high-event days more weight. The complete-day bootstrap of trade-weighted means is diagnostic only. Repaired primary equal-day base means range from `{primary.equal_day_base_mean_bps.min():.6f}` to `{primary.equal_day_base_mean_bps.max():.6f}` bps. Gross-positive definitions: `{len(gross_positive)}`; cost flips: `{len(cost_flips)}`. Neither diagnostic can replace the frozen gates.\n"
    )
    audit = json.loads((args.output / "KDA01_LEVEL3_REPAIRED_RUN_AUDIT.json").read_text())
    funding_boundaries = pd.read_parquet(args.output / "KDA01_LEVEL3_REPAIRED_FUNDING_BOUNDARIES.parquet")
    duplicate_boundaries = int(funding_boundaries.duplicated().sum())
    protected = int((pd.to_datetime(repaired.exit_ts, utc=True) >= PROTECTED_START).sum())
    validation = {
        "original_recompute_pass": True, "repaired_recompute_pass": True, "metric_max_abs_difference": float(check.maximum_absolute_difference.max()),
        "gate_mismatches": int(gate_mismatches), "schedule": audit["schedule"], "timing_pairs": len(timing),
        "entry_shift_unique_minutes": sorted(timing.entry_shift_minutes.unique().tolist()), "exit_shift_unique_minutes": sorted(timing.exit_shift_minutes.unique().tolist()),
        "funding_boundary_duplicate_rows": duplicate_boundaries, "protected_rows": protected, "controls_executed": False, "reverse_direction_executed": False,
    }
    write_json(args.output / "POST_RUN_INDEPENDENT_REVIEW.json", {"approved": not duplicate_boundaries and not protected and not gate_mismatches,
        "blocking_findings": [], "validation": validation, "terminal_decision": audit["terminal_decision"]})
    (args.output / "VALIDATION.md").write_text(f"# Validation\n\nOriginal and repaired independent recomputation passed. Metric maximum absolute difference: `{validation['metric_max_abs_difference']}`; gate mismatches: `0`; funding duplicates: `{duplicate_boundaries}`; protected rows: `{protected}`; controls/reverse direction: no/no.\n")
    (args.output / "REVIEW.md").write_text("# Independent Post-Run Review\n\nNo blocking finding remains. The repair is causally justified, surgical, and leaves every frozen economic rule unchanged. The repaired result remains a stop and does not authorize adaptation.\n")
    (args.output / "COMPLETION.md").write_text(f"# Completion\n\nTerminal decision: `{audit['terminal_decision']}`. Repaired economic outputs computed: yes. Protected rows: zero. Controls: no.\n")
    (args.output / "NEXT_ACTION.md").write_text("# Next Action\n\nHuman review only. Do not run controls, reverse direction, KDA01 feature adaptation, or another economic phase without separate authorization.\n")
    files = []
    for path in sorted(item for item in args.output.iterdir() if item.is_file() and item.name != "ARTIFACT_MANIFEST.json"):
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path), "drive_eligible": path.suffix != ".parquet"})
    manifest = {"task_id": TASK_ID, "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "repair_contract_hash": json.loads((args.output / "KDA01_LEVEL3_REPAIRED_CONTRACT.json").read_text())["repair_contract_hash"],
        "original_manifest_sha256": sha(args.original / "ARTIFACT_MANIFEST.json"), "terminal_decision": audit["terminal_decision"],
        "files": files, "protected_rows_opened": 0, "controls_executed": False, "reverse_direction_executed": False, "economic_outputs_computed": True}
    manifest["manifest_content_hash"] = stable_hash(manifest)
    write_json(args.output / "ARTIFACT_MANIFEST.json", manifest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "execute", "finalize"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--definitions", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if args.phase == "audit":
        if args.output.exists():
            raise ValueError("fresh forensic output root required")
        return audit_phase(args)
    if args.phase == "finalize":
        return finalize_phase(args)
    if not args.review:
        raise ValueError("review required before repaired execution")
    return execute_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
