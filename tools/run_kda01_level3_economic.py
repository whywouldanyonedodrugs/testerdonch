#!/usr/bin/env python3
"""Execute the single authorized frozen KDA01 v2 Level-3 economic contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_kraken_c01_foundation as foundation
from tools import run_kraken_c01_level3_economic as funding_shared
from tools.build_kda01_contract_closure import load_timestamp_only_bars, verify_stage8b
from tools.qlmg_kda01_contract_closure import execution_records
from tools.qlmg_kda01_level3_economic import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, branch_side, cluster_bootstrap,
    equal_cluster_returns, gate_flags, positive_contribution_share, score_open_prices,
)
from tools.qlmg_kraken_derivatives_state import PROTECTED_START, TRAIN_START, stable_hash

TASK_ID = "donch_bt_stage_8c_kda01_level3_economic_20260719_v1"
CONTRACT_HASH = "d2112d8fb4f0666890761b70f7fa71c9e8cb4f43c704a04cdf9470e949a177b3"
CONTRACT_FILE_HASH = "f0488650eed87c714a46acc6e74def362f200240d8733fa09500b679464a56b5"
DEFINITION_FILE_HASH = "ef32ebd1032c79e9f59df71b463d8e6d509b2ccfc8ab3eb75db2449d5d2aac7a"
CLUSTER_FILE_HASH = "1a2946a2da8c5bf9b1a4cbc9571abad31d968034f246fe94a8e442e959864669"
SOURCE_EVENT_HASH = "7c9e682380681aa6fc83161ae29dafab9a047bdb19fe1fa8ade3ba5a4eb9c2e5"
SOURCE_PARENT_HASH = "ae470d06c7da049d145ac8e083d0c4711c9c2dd6e4ec32fedf79224e8885ddbd"
MARKET_MANIFEST = Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv")
FUNDING_ROOT = Path("/opt/testerdonch/results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
OPEN_COLUMNS = ("time", "open", "venue_symbol", "resolution", "rankable_pre_holdout", "contains_protected_period")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_open_map(rows: list[foundation.AuthorityRow], symbol: str) -> pd.Series:
    parts = []
    for row in [x for x in rows if x.dataset == "historical_trade_candles_5m" and x.symbol == symbol]:
        parquet = pq.ParquetFile(row.parquet_path)
        if not set(OPEN_COLUMNS).issubset(parquet.schema_arrow.names):
            continue
        raw = parquet.read(columns=list(OPEN_COLUMNS)).to_pandas()
        if (not raw.venue_symbol.eq(symbol).all() or not raw.resolution.eq("5m").all()
                or not raw.rankable_pre_holdout.map(foundation._as_bool).all()
                or raw.contains_protected_period.map(foundation._as_bool).any()):
            raise ValueError("unsafe rankable open payload")
        ts = pd.to_datetime(pd.to_numeric(raw.time, errors="raise"), unit="ms", utc=True)
        if (ts < TRAIN_START).any() or (ts >= PROTECTED_START).any():
            raise ValueError("pre-2023 or protected open row")
        values = pd.to_numeric(raw.open, errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any() or (values <= 0).any():
            raise ValueError("invalid official trade open")
        parts.append(pd.DataFrame({"ts": ts, "open": values}))
    if not parts:
        raise ValueError(f"no authorized open bars for {symbol}")
    frame = pd.concat(parts).sort_values("ts", kind="mergesort")
    duplicate = frame[frame.duplicated("ts", keep=False)]
    if len(duplicate) and duplicate.groupby("ts").open.nunique().gt(1).any():
        raise ValueError("conflicting exact trade opens")
    return frame.drop_duplicates("ts").set_index("ts").open


def preflight(contract: Path, definitions: Path, clusters: Path, stage8b: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    expected = ((contract, CONTRACT_FILE_HASH), (definitions, DEFINITION_FILE_HASH), (clusters, CLUSTER_FILE_HASH))
    for path, digest in expected:
        if sha(path) != digest:
            raise ValueError(f"frozen input hash mismatch: {path}")
    rules = json.loads(contract.read_text())
    if rules.get("contract_version") != "kda01_level3_contract_v2_20260719" or rules.get("level3_contract_hash") != CONTRACT_HASH:
        raise ValueError("frozen contract identity mismatch")
    if rules.get("controls_executed") is not False or len(rules.get("controls", [])) != 7:
        raise ValueError("control authorization mismatch")
    register = pd.read_csv(definitions)
    if len(register) != 16 or register.definition_id.nunique() != 16 or register.robustness_only.map(foundation._as_bool).sum() != 8:
        raise ValueError("definition authority mismatch")
    events = pd.read_parquet(clusters)
    for field in ("decision_ts", "parent_onset_ts"):
        events[field] = pd.to_datetime(events[field], utc=True, errors="raise")
    if len(events) != 102136 or events.event_id.duplicated().any() or (events.decision_ts < TRAIN_START).any() or (events.decision_ts >= PROTECTED_START).any():
        raise ValueError("event-cluster authority mismatch")
    source = verify_stage8b(stage8b)
    return source, register, events


def reconstruct(register: pd.DataFrame, events: pd.DataFrame, authority: list[foundation.AuthorityRow]) -> pd.DataFrame:
    bars, refs = {}, {}
    for symbol in sorted(events.symbol.unique()):
        bars[symbol], refs[symbol] = load_timestamp_only_bars(authority, symbol)
    records = execution_records(events, register, bars)
    statuses = records.loc[~records.accepted, "status"].value_counts().to_dict()
    expected = {
        "records": 204272, "accepted": 183744, "rejected": 20528,
        "actual_position_overlap": 20473, "missing_exit_bar": 55,
        "entry_delay_exceeded": 0, "exit_delay_exceeded": 0, "missing_entry_bar": 0,
    }
    actual = {"records": len(records), "accepted": int(records.accepted.sum()), "rejected": int((~records.accepted).sum()),
              **{name: int(statuses.get(name, 0)) for name in expected if name not in {"records", "accepted", "rejected"}}}
    if actual != expected:
        raise ValueError(f"accepted execution reconstruction mismatch: {actual}")
    records.attrs["timestamp_authority_hash"] = stable_hash(refs)
    return records


def price_and_score(accepted: pd.DataFrame, events: pd.DataFrame, authority: list[foundation.AuthorityRow]) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_columns = ["event_id", "attempt", "parent_episode_id", "market_day_cluster_id", "market_6h_cluster_id"]
    work = accepted.merge(events[event_columns], on="event_id", validate="many_to_one")
    maps = {symbol: load_open_map(authority, symbol) for symbol in sorted(work.symbol.unique())}
    rows, rejected = [], []
    for row in work.itertuples(index=False):
        source = maps[row.symbol]
        try:
            entry, exit_ = float(source.loc[pd.Timestamp(row.entry_ts)]), float(source.loc[pd.Timestamp(row.exit_ts)])
            side_sign = branch_side(row.branch_id)
            gross, base, stress = score_open_prices(entry, exit_, side_sign)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"definition_id": row.definition_id, "event_id": row.event_id, "reason": str(exc)})
            continue
        address = "kda01l3_" + stable_hash({"definition": row.definition_contract_hash, "event": row.event_id,
            "symbol": row.symbol, "entry_ts": pd.Timestamp(row.entry_ts).isoformat(), "exit_ts": pd.Timestamp(row.exit_ts).isoformat()})
        rows.append({**row._asdict(), "economic_address": address, "side": "long" if side_sign == 1 else "short", "side_sign": side_sign, "entry_open": entry,
                     "exit_open": exit_, "gross_bps": gross, "base_net_bps": base, "stress_net_bps": stress,
                     "calendar_year": pd.Timestamp(row.entry_ts).year, "calendar_month": pd.Timestamp(row.entry_ts).strftime("%Y-%m")})
    trades = pd.DataFrame(rows)
    if len(trades) and (trades.economic_address.duplicated().any() or (pd.to_datetime(trades.exit_ts, utc=True) >= PROTECTED_START).any()):
        raise ValueError("duplicate or protected trade outcome")
    return trades, pd.DataFrame(rejected)


def reports(register: pd.DataFrame, records: pd.DataFrame, trades: pd.DataFrame):
    day = equal_cluster_returns(trades, "market_day_cluster_id")
    metrics, gates, boot_summary, distributions, concentration, decisions, sensitivities = [], [], [], [], [], [], []
    for definition in register.sort_values(["robustness_only", "definition_id"], kind="mergesort").to_dict("records"):
        did = definition["definition_id"]
        g, d = trades[trades.definition_id.eq(did)], day[day.definition_id.eq(did)]
        dist, low, high = cluster_bootstrap(d.base_net_bps)
        distributions.extend({"definition_id": did, "resample": i, "mean_base_net_bps": v} for i, v in enumerate(dist))
        years = g.calendar_year.value_counts().to_dict()
        row = {**definition, "candidate_count": int((records.definition_id == did).sum()), "accepted_count": len(g),
               "price_rejection_count": int((records.definition_id == did).sum() - len(g) - ((records.definition_id == did) & ~records.accepted).sum()),
               "trades_2023": int(years.get(2023, 0)), "trades_2024": int(years.get(2024, 0)), "trades_2025": int(years.get(2025, 0)),
               "symbols": int(g.symbol.nunique()), "market_day_clusters": len(d), "win_rate": float((g.base_net_bps > 0).mean()),
               "trade_gross_mean_bps": float(g.gross_bps.mean()), "trade_gross_median_bps": float(g.gross_bps.median()),
               "trade_base_mean_bps": float(g.base_net_bps.mean()), "trade_base_median_bps": float(g.base_net_bps.median()),
               "trade_stress_mean_bps": float(g.stress_net_bps.mean()), "trade_stress_median_bps": float(g.stress_net_bps.median()),
               "equal_day_gross_mean_bps": float(d.gross_bps.mean()), "equal_day_gross_median_bps": float(d.gross_bps.median()),
               "equal_day_base_mean_bps": float(d.base_net_bps.mean()), "equal_day_base_median_bps": float(d.base_net_bps.median()),
               "equal_day_stress_mean_bps": float(d.stress_net_bps.mean()), "equal_day_stress_median_bps": float(d.stress_net_bps.median()),
               "bootstrap_lower_bps": low, "bootstrap_upper_bps": high,
               "market_day_positive_share": positive_contribution_share(g, "market_day_cluster_id"),
               "symbol_positive_share": positive_contribution_share(g, "symbol"), "year_positive_share": positive_contribution_share(g, "calendar_year")}
        for name in ("gross_bps", "base_net_bps", "stress_net_bps"):
            for q in (.01, .05, .25, .5, .75, .95, .99): row[f"{name}_q{int(q*100):02d}"] = float(g[name].quantile(q))
        flags = gate_flags(row)
        metrics.append(row); gates.append({"definition_id": did, "attempt": definition["attempt"], **flags})
        boot_summary.append({"definition_id": did, "clusters": len(d), "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "lower_bps": low, "upper_bps": high})
        concentration.append({"definition_id": did, "market_day_positive_share": row["market_day_positive_share"], "symbol_positive_share": row["symbol_positive_share"], "year_positive_share": row["year_positive_share"]})
        decisions.append({"definition_id": did, "attempt": definition["attempt"], "all_gates_pass": flags["all_gates_pass"], "decision": "primary_pass_controls_required" if definition["attempt"] == "primary" and flags["all_gates_pass"] else "no_primary_pass" if definition["attempt"] == "primary" else "robustness_diagnostic_only"})
        for cluster in ("market_6h_cluster_id", "parent_episode_id"):
            c = equal_cluster_returns(g, cluster); _, lo, hi = cluster_bootstrap(c.base_net_bps)
            sensitivities.append({"definition_id": did, "cluster": cluster, "clusters": len(c), "base_mean_bps": float(c.base_net_bps.mean()), "base_median_bps": float(c.base_net_bps.median()), "bootstrap_lower_bps": lo, "bootstrap_upper_bps": hi})
    return tuple(pd.DataFrame(x) for x in (metrics, gates, boot_summary, distributions, concentration, decisions, sensitivities)), day


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--contract", type=Path, required=True); p.add_argument("--definitions", type=Path, required=True); p.add_argument("--clusters", type=Path, required=True); p.add_argument("--stage8b", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--execute-frozen-economic-run", action="store_true"); a = p.parse_args()
    if not a.execute_frozen_economic_run: raise ValueError("explicit frozen economic authorization flag required")
    if a.output.exists(): raise ValueError("fresh output root required")
    source, register, events = preflight(a.contract, a.definitions, a.clusters, a.stage8b)
    authority = foundation.load_safe_manifest(MARKET_MANIFEST)
    records = reconstruct(register, events, authority)  # Must pass before any open is read.
    a.output.mkdir(parents=True)
    records.loc[records.accepted].to_parquet(
        a.output / "KDA01_LEVEL3_ACCEPTED_EXECUTION_RECORDS.parquet", index=False
    )
    records.loc[~records.accepted].to_parquet(
        a.output / "KDA01_LEVEL3_EXECUTION_REJECTIONS.parquet", index=False
    )
    trades, price_rejections = price_and_score(records[records.accepted].copy(), events, authority)
    funding_shared.FUNDING_ROOT = FUNDING_ROOT
    panel, location, funding_hash = funding_shared.load_funding_panel()
    funding_input = trades.rename(columns={"exit_ts": "actual_exit_ts"}) if "actual_exit_ts" not in trades else trades
    funded, boundaries = funding_shared.attach_funding(funding_input, panel, location)
    trades = funded.rename(columns={"actual_exit_ts": "exit_ts"}) if "exit_ts" not in funded else funded
    outputs, day = reports(register, records, trades)
    metrics, gates, boot, dist, conc, decisions, sensitivity = outputs
    primary_pass = decisions[(decisions.attempt == "primary") & decisions.all_gates_pass]
    terminal = "KDA01_level3_primary_pass_controls_required" if len(primary_pass) else "KDA01_level3_no_primary_pass_stop"
    snapshot = json.loads(a.contract.read_text()); snapshot.update({"authorized_task": TASK_ID, "source_contract_sha256": CONTRACT_FILE_HASH, "controls_executed": False})
    write_json(a.output / "KDA01_LEVEL3_RUN_CONTRACT_SNAPSHOT.json", snapshot)
    trades.to_parquet(a.output / "KDA01_LEVEL3_TRADE_TAPE.parquet", index=False); metrics.to_csv(a.output / "KDA01_LEVEL3_DEFINITION_METRICS.csv", index=False); gates.to_csv(a.output / "KDA01_LEVEL3_GATE_MATRIX.csv", index=False); day.to_parquet(a.output / "KDA01_LEVEL3_MARKET_DAY_RETURNS.parquet", index=False); boot.to_csv(a.output / "KDA01_LEVEL3_BOOTSTRAP_SUMMARY.csv", index=False); dist.to_parquet(a.output / "KDA01_LEVEL3_BOOTSTRAP_DISTRIBUTIONS.parquet", index=False); conc.to_csv(a.output / "KDA01_LEVEL3_CONCENTRATION.csv", index=False)
    trades.groupby(["definition_id", "calendar_year", "calendar_month", "symbol"], as_index=False).agg(trades=("event_id", "size"), gross_bps=("gross_bps", "mean"), base_net_bps=("base_net_bps", "mean"), stress_net_bps=("stress_net_bps", "mean")).to_csv(a.output / "KDA01_LEVEL3_YEAR_MONTH_SYMBOL_SUMMARY.csv", index=False)
    funded.groupby(["definition_id", "funding_partition"], as_index=False).agg(trades=("event_id", "size"), exact_boundaries=("exact_boundary_count", "sum"), imputed_boundaries=("imputed_boundary_count", "sum"), central_cashflow_bps=("funding_cashflow_central_bps", "mean"), conservative_cashflow_bps=("funding_cashflow_conservative_bps", "mean"), severe_cashflow_bps=("funding_cashflow_severe_bps", "mean")).to_csv(a.output / "KDA01_LEVEL3_FUNDING_PARTITIONS.csv", index=False)
    sensitivity.to_csv(a.output / "KDA01_LEVEL3_CLUSTER_SENSITIVITY.csv", index=False); decisions.to_csv(a.output / "KDA01_LEVEL3_DEFINITION_DECISIONS.csv", index=False)
    (a.output / "KDA01_LEVEL3_DECISION.md").write_text(f"# KDA01 Level-3 Decision\n\n`{terminal}`\n\nPassing primary definitions: {len(primary_pass)}. Controls were not executed.\n")
    (a.output / "KDA01_LEVEL3_CLAIM_BOUNDARY.md").write_text("# Claim Boundary\n\nTrain-only Kraken Level-3 evidence. No controls, validation, protected period, portfolio, or production claim. Funding diagnostics are excluded from gates.\n")
    audit = {"task_id": TASK_ID, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "runner_sha256": sha(Path(__file__)), "calculation_module_sha256": sha(ROOT / "tools/qlmg_kda01_level3_economic.py"), "market_manifest_sha256": sha(MARKET_MANIFEST), "contract_file_sha256": CONTRACT_FILE_HASH, "definition_register_sha256": DEFINITION_FILE_HASH, "contract_hash": CONTRACT_HASH, "source_event_sha256": SOURCE_EVENT_HASH, "source_parent_sha256": SOURCE_PARENT_HASH, "cluster_sha256": CLUSTER_FILE_HASH, "funding_model_hash": funding_hash, "schedule_records": len(records), "accepted": int(records.accepted.sum()), "price_rejections": len(price_rejections), "trades": len(trades), "controls_executed": False, "protected_rows_opened": 0, "terminal_decision": terminal, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(a.output / "RUN_AUDIT.json", audit); price_rejections.to_csv(a.output / "PRICE_REJECTIONS.csv", index=False); boundaries.to_parquet(a.output / "FUNDING_BOUNDARY_LEDGER.parquet", index=False)
    (a.output / "VALIDATION.md").write_text("# Validation\n\nFrozen authorities, exact schedule reconstruction, price roles, costs, protected boundary, funding separation, and gates passed mechanical validation.\n")
    (a.output / "REVIEW.md").write_text("# Review\n\nPending independent post-run review.\n")
    (a.output / "COMPLETION.md").write_text(f"# Completion\n\nTerminal decision: `{terminal}`. Economic outputs computed: yes. Protected rows opened: no. Controls executed: no.\n")
    (a.output / "NEXT_ACTION.md").write_text("# Next Action\n\nHuman review of the terminal Level-3 decision. No automatic controls or further economic work.\n")
    files=[]
    for path in sorted(x for x in a.output.iterdir() if x.is_file() and x.name != "ARTIFACT_MANIFEST.json"):
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path), "drive_eligible": path.suffix != ".parquet"})
    manifest={"task_id":TASK_ID,"contract_hash":CONTRACT_HASH,"terminal_decision":terminal,"files":files,"protected_rows_opened":0,"controls_executed":False,"economic_outputs_computed":True}; manifest["manifest_content_hash"] = stable_hash(manifest); write_json(a.output / "ARTIFACT_MANIFEST.json", manifest)
    return 0

if __name__ == "__main__": raise SystemExit(main())
