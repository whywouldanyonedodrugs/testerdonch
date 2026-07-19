#!/usr/bin/env python3
"""Execute the one conditionally authorized frozen KDA03 v1 Level-3 run."""

from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
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
from tools.build_kda01_contract_closure import load_timestamp_only_bars
from tools.qlmg_kda01_timestamp_repair import repaired_execution_records
from tools.qlmg_kda03_level3 import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    assign_route,
    branch_side,
    cluster_bootstrap,
    equal_cluster_returns,
    route_flags,
    score_open_prices,
)
from tools.qlmg_kraken_derivatives_state import PROTECTED_START, TRAIN_START, sha256_file, stable_hash


TASK_ID = "donch_bt_stage_11_kda03_basis_shock_20260719_v2"
MARKET_MANIFEST = Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv")
FUNDING_ROOT = Path("/opt/testerdonch/results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
OPEN_COLUMNS = ("time", "open", "venue_symbol", "resolution", "rankable_pre_holdout", "contains_protected_period")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def verified_trade_authority_hash(
    rows: list[foundation.AuthorityRow], symbols: list[str]
) -> str:
    """Verify and bind all official trade shards before timestamps or opens are read."""
    symbol_set = set(symbols)
    selected = sorted(
        (
            row for row in rows
            if row.dataset == "historical_trade_candles_5m" and row.symbol in symbol_set
        ),
        key=lambda row: (row.symbol, row.chunk_start, str(row.parquet_path)),
    )
    present = {row.symbol for row in selected}
    if present != symbol_set:
        raise ValueError(f"missing official trade-bar authority: {sorted(symbol_set - present)}")
    records = []
    for row in selected:
        actual = sha256_file(row.parquet_path)
        if actual != row.parquet_sha256:
            raise ValueError(f"official trade-bar payload hash mismatch: {row.parquet_path}")
        records.append({
            "dataset": row.dataset,
            "symbol": row.symbol,
            "chunk_start": row.chunk_start.isoformat(),
            "chunk_end": row.chunk_end.isoformat(),
            "parquet_path": str(row.parquet_path),
            "parquet_sha256": actual,
            "rows": row.rows,
        })
    return stable_hash(records)


def load_open_map(rows: list[foundation.AuthorityRow], symbol: str) -> pd.Series:
    parts = []
    for row in [item for item in rows if item.dataset == "historical_trade_candles_5m" and item.symbol == symbol]:
        parquet = pq.ParquetFile(row.parquet_path)
        if not set(OPEN_COLUMNS).issubset(parquet.schema_arrow.names):
            continue
        raw = parquet.read(columns=list(OPEN_COLUMNS)).to_pandas()
        if (
            not raw.venue_symbol.eq(symbol).all()
            or not raw.resolution.eq("5m").all()
            or not raw.rankable_pre_holdout.map(foundation._as_bool).all()
            or raw.contains_protected_period.map(foundation._as_bool).any()
        ):
            raise ValueError("unsafe rankable open payload")
        ts = pd.to_datetime(pd.to_numeric(raw.time, errors="raise"), unit="ms", utc=True)
        if (ts < TRAIN_START).any() or (ts >= PROTECTED_START).any():
            raise ValueError("pre-2023 or protected open row")
        values = pd.to_numeric(raw.open, errors="coerce")
        if values.isna().any() or (~np.isfinite(values)).any() or (values <= 0).any():
            raise ValueError("invalid official PF trade open")
        parts.append(pd.DataFrame({"ts": ts, "open": values}))
    if not parts:
        raise ValueError(f"no authorized PF trade opens: {symbol}")
    frame = pd.concat(parts).sort_values("ts", kind="mergesort")
    duplicate = frame[frame.duplicated("ts", keep=False)]
    if len(duplicate) and duplicate.groupby("ts").open.nunique().gt(1).any():
        raise ValueError("conflicting exact PF trade opens")
    return frame.drop_duplicates("ts").set_index("ts").open


def preflight(archive: Path, review_path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    review = json.loads(review_path.read_text())
    contract_path = archive / "KDA03_FINAL_LEVEL3_CONTRACT.json"
    definitions_path = archive / "KDA03_LEVEL3_DEFINITION_REGISTER.csv"
    event_path = archive / "KDA03_V1_EVENT_TAPE.parquet"
    parent_path = archive / "KDA03_V1_PARENT_EPISODE_TAPE.parquet"
    gate_path = archive / "KDA03_V1_FEASIBILITY_GATES.csv"
    contract = json.loads(contract_path.read_text())
    if not review.get("approved"):
        raise ValueError("independent KDA03 pre-outcome approval missing")
    if review.get("level3_contract_hash") != contract.get("level3_contract_hash"):
        raise ValueError("independent review contract hash mismatch")
    paths = {
        "contract": contract_path,
        "definitions": definitions_path,
        "events": event_path,
        "parents": parent_path,
        "feasibility_gates": gate_path,
        "feature_module": ROOT / "tools/qlmg_kda03_v1.py",
        "builder": ROOT / "tools/build_kda03_v1_prerun_freeze.py",
        "calculation_module": ROOT / "tools/qlmg_kda03_level3.py",
        "runner": Path(__file__),
    }
    expected = review.get("reviewed_sha256", {})
    if set(expected) != set(paths):
        raise ValueError("independent review file set mismatch")
    for key, path in paths.items():
        if sha256_file(path) != expected[key]:
            raise ValueError(f"frozen pre-outcome source mismatch: {key}")
    if contract.get("contract_version") != "kda03_level3_contract_v1_20260719":
        raise ValueError("unexpected KDA03 frozen contract version")
    if contract.get("controls_executed") is not False:
        raise ValueError("control authorization mismatch")
    definitions = pd.read_csv(definitions_path)
    events = pd.read_parquet(event_path)
    gates = pd.read_csv(gate_path)
    feasible = gates.groupby("branch_id").branch_mechanically_feasible.first()
    if int(feasible.sum()) < 1 or definitions.empty:
        raise ValueError("KDA03 conditional economic gate is not open")
    if events.event_id.duplicated().any() or events.economic_address.duplicated().any():
        raise ValueError("duplicate frozen KDA03 event identity")
    for column in ("decision_ts", "parent_onset_ts"):
        events[column] = pd.to_datetime(events[column], utc=True, errors="raise")
    if (events.decision_ts < TRAIN_START).any() or (events.decision_ts >= PROTECTED_START).any():
        raise ValueError("non-rankable frozen KDA03 event")
    return contract, definitions, events, gates


def reconstruct_schedule(
    definitions: pd.DataFrame,
    events: pd.DataFrame,
    gates: pd.DataFrame,
    authority: list[foundation.AuthorityRow],
) -> tuple[pd.DataFrame, str]:
    bars: dict[str, pd.DatetimeIndex] = {}
    refs: dict[str, str] = {}
    for symbol in sorted(events.loc[events.branch_id.isin(definitions.branch_id), "symbol"].unique()):
        bars[symbol], refs[symbol] = load_timestamp_only_bars(authority, symbol)
    records = repaired_execution_records(events, definitions, bars)
    primary_definition_ids = set(definitions.loc[definitions.attempt.eq("primary"), "definition_id"])
    frozen_gate_rows = gates.loc[gates.definition_id.isin(primary_definition_ids)]
    if frozen_gate_rows.definition_id.duplicated().any() or set(frozen_gate_rows.definition_id) != primary_definition_ids:
        raise ValueError("frozen primary definition/gate identity mismatch")
    primary_expected = frozen_gate_rows.set_index("definition_id").accepted_events.astype(int).to_dict()
    primary_actual = records.loc[records.accepted & records.definition_id.isin(primary_expected)].definition_id.value_counts().to_dict()
    if any(int(primary_actual.get(definition, 0)) != count for definition, count in primary_expected.items()):
        raise ValueError("timestamp-only schedule no longer matches mechanical freeze")
    if records.duplicated(["definition_id", "event_id"]).any():
        raise ValueError("duplicate definition-event schedule")
    return records, stable_hash(refs)


def price_and_score(
    accepted: pd.DataFrame,
    events: pd.DataFrame,
    authority: list[foundation.AuthorityRow],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_columns = [
        "event_id", "attempt", "parent_episode_id", "parent_direction", "trade_direction",
        "market_day_cluster_id", "market_6h_cluster_id", "parent_onset_ts",
        "eligible_denominator", "primary_signed_shock_breadth",
        "robustness_signed_shock_breadth", "btc_basis_decimal", "btc_basis_change_15m",
        "eth_basis_decimal", "eth_basis_change_15m",
    ]
    work = accepted.merge(events[event_columns], on="event_id", validate="many_to_one")
    maps = {symbol: load_open_map(authority, symbol) for symbol in sorted(work.symbol.unique())}
    rows, rejected = [], []
    for row in work.itertuples(index=False):
        source = maps[row.symbol]
        try:
            entry = float(source.loc[pd.Timestamp(row.entry_ts)])
            exit_ = float(source.loc[pd.Timestamp(row.exit_ts)])
            side = branch_side(row.branch_id)
            if side != int(row.trade_direction):
                raise ValueError("frozen branch side disagrees with event trade direction")
            gross, base, stress = score_open_prices(entry, exit_, side)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"definition_id": row.definition_id, "event_id": row.event_id, "reason": str(exc)})
            continue
        address = "kda03l3_" + stable_hash({
            "definition": row.definition_contract_hash,
            "event": row.event_id,
            "symbol": row.symbol,
            "entry_ts": pd.Timestamp(row.entry_ts).isoformat(),
            "exit_ts": pd.Timestamp(row.exit_ts).isoformat(),
        })[:32]
        rows.append({
            **row._asdict(),
            "level3_economic_address": address,
            "side": "long" if side == 1 else "short",
            "side_sign": side,
            "entry_open": entry,
            "exit_open": exit_,
            "gross_bps": gross,
            "base_net_bps": base,
            "stress_net_bps": stress,
            "calendar_year": pd.Timestamp(row.entry_ts).year,
            "calendar_month": pd.Timestamp(row.entry_ts).strftime("%Y-%m"),
            "basis_context": (
                "both_positive" if row.btc_basis_change_15m > 0 and row.eth_basis_change_15m > 0
                else "both_negative" if row.btc_basis_change_15m < 0 and row.eth_basis_change_15m < 0
                else "mixed_or_missing"
            ),
        })
    trades = pd.DataFrame(rows)
    if len(trades) and (
        trades.level3_economic_address.duplicated().any()
        or (pd.to_datetime(trades.exit_ts, utc=True) >= PROTECTED_START).any()
    ):
        raise ValueError("duplicate or protected KDA03 trade outcome")
    return trades, pd.DataFrame(rejected)


def add_estimand_weights(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    day_sizes = out.groupby(["definition_id", "market_day_cluster_id"]).event_id.transform("size")
    day_counts = out.groupby("definition_id").market_day_cluster_id.transform("nunique")
    out["equal_market_day_trade_weight"] = 1.0 / day_sizes / day_counts
    out["estimand_base_contribution_bps"] = out.base_net_bps * out.equal_market_day_trade_weight
    return out


def positive_share(frame: pd.DataFrame, group: str) -> float:
    grouped = frame.groupby(group, sort=True).estimand_base_contribution_bps.sum()
    positive = grouped[grouped > 0]
    denominator = float(positive.sum())
    return float(positive.max() / denominator) if math.isfinite(denominator) and denominator > 0 else math.nan


def attach_funding_diagnostics(
    trades: pd.DataFrame,
    panel: pd.DataFrame,
    location: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Route funding by unique definition-event address, preserving candidate identity."""
    if trades.level3_economic_address.duplicated().any():
        raise ValueError("duplicate KDA03 Level-3 address before funding")
    funding_input = trades.rename(columns={
        "economic_address": "candidate_economic_address",
        "level3_economic_address": "economic_address",
        "exit_ts": "actual_exit_ts",
    })
    funded, boundaries = funding_shared.attach_funding(funding_input, panel, location)
    result = funded.rename(columns={
        "economic_address": "level3_economic_address",
        "actual_exit_ts": "exit_ts",
    })
    if result.level3_economic_address.duplicated().any() or "candidate_economic_address" not in result:
        raise ValueError("KDA03 funding identity round-trip failed")
    return result, boundaries


def reports(definitions: pd.DataFrame, records: pd.DataFrame, trades: pd.DataFrame):
    weighted = add_estimand_weights(trades)
    day = equal_cluster_returns(weighted, "market_day_cluster_id")
    metrics, gates, bootstrap, distributions = [], [], [], []
    concentration, decisions, sensitivities, contexts = [], [], [], []
    for definition in definitions.sort_values(["robustness_only", "definition_id"], kind="mergesort").to_dict("records"):
        did = definition["definition_id"]
        trade = weighted.loc[weighted.definition_id.eq(did)]
        daily = day.loc[day.definition_id.eq(did)]
        dist, low, high = cluster_bootstrap(daily.base_net_bps)
        distributions.extend({"definition_id": did, "resample": index, "mean_base_net_bps": value} for index, value in enumerate(dist))
        sensitivity_rows = []
        for cluster in ("market_6h_cluster_id", "parent_episode_id"):
            sensitivity = equal_cluster_returns(trade, cluster)
            _, sensitivity_low, sensitivity_high = cluster_bootstrap(sensitivity.base_net_bps)
            entry = {
                "definition_id": did, "cluster": cluster, "clusters": len(sensitivity),
                "base_mean_bps": float(sensitivity.base_net_bps.mean()),
                "base_median_bps": float(sensitivity.base_net_bps.median()),
                "bootstrap_lower_bps": sensitivity_low, "bootstrap_upper_bps": sensitivity_high,
            }
            sensitivity_rows.append(entry); sensitivities.append(entry)
        context_rows = []
        for label, group in trade.groupby("basis_context", sort=True):
            entry = {
                "definition_id": did, "basis_context": label, "trades": len(group),
                "market_days": int(group.market_day_cluster_id.nunique()),
                "base_mean_bps": float(group.base_net_bps.mean()),
                "base_median_bps": float(group.base_net_bps.median()),
                "primary_signed_shock_breadth_mean": float(group.primary_signed_shock_breadth.mean()),
                "robustness_signed_shock_breadth_mean": float(group.robustness_signed_shock_breadth.mean()),
            }
            context_rows.append(entry); contexts.append(entry)
        primary_mean = float(daily.base_net_bps.mean())
        estimand_dependence = any((item["base_mean_bps"] > 0) != (primary_mean > 0) for item in sensitivity_rows)
        eligible_context_means = [item["base_mean_bps"] for item in context_rows if item["market_days"] >= 20]
        context_dependence = bool(eligible_context_means) and any(value > 0 for value in eligible_context_means) and any(value <= 0 for value in eligible_context_means)
        years = trade.calendar_year.value_counts().to_dict()
        row = {
            **definition, "candidate_count": int(records.definition_id.eq(did).sum()),
            "accepted_count": len(trade), "trades_2023": int(years.get(2023, 0)),
            "trades_2024": int(years.get(2024, 0)), "trades_2025": int(years.get(2025, 0)),
            "symbols": int(trade.symbol.nunique()), "market_day_clusters": len(daily),
            "trade_gross_mean_bps": float(trade.gross_bps.mean()),
            "trade_base_mean_bps": float(trade.base_net_bps.mean()),
            "trade_stress_mean_bps": float(trade.stress_net_bps.mean()),
            "equal_day_gross_mean_bps": float(daily.gross_bps.mean()),
            "equal_day_gross_median_bps": float(daily.gross_bps.median()),
            "equal_day_base_mean_bps": primary_mean,
            "equal_day_base_median_bps": float(daily.base_net_bps.median()),
            "equal_day_stress_mean_bps": float(daily.stress_net_bps.mean()),
            "equal_day_stress_median_bps": float(daily.stress_net_bps.median()),
            "bootstrap_lower_bps": low, "bootstrap_upper_bps": high,
            "market_day_positive_share": positive_share(trade, "market_day_cluster_id"),
            "symbol_positive_share": positive_share(trade, "symbol"),
            "year_positive_share": positive_share(trade, "calendar_year"),
            "estimand_sign_dependence": estimand_dependence,
            "context_sign_dependence": context_dependence,
            "material_estimand_or_context_dependence": estimand_dependence or context_dependence,
            "single_event_or_defect_explanation": False,
        }
        flags = route_flags(row)
        route = assign_route(row)
        metrics.append({**row, "policy_route": route if definition["attempt"] == "primary" else "robustness_diagnostic_only"})
        gates.append({"definition_id": did, "attempt": definition["attempt"], **flags})
        bootstrap.append({"definition_id": did, "market_day_clusters": len(daily), "resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "lower_bps": low, "upper_bps": high})
        concentration.append({"definition_id": did, "market_day_positive_share": row["market_day_positive_share"], "symbol_positive_share": row["symbol_positive_share"], "year_positive_share": row["year_positive_share"]})
        decisions.append({
            "definition_id": did, "attempt": definition["attempt"],
            "policy_route": route if definition["attempt"] == "primary" else "robustness_diagnostic_only",
            "control_eligible": bool(flags["control_eligible"]) if definition["attempt"] == "primary" else False,
        })
    frames = (metrics, gates, bootstrap, distributions, concentration, decisions, sensitivities, contexts)
    return tuple(pd.DataFrame(value) for value in frames), day, weighted


def contribution_ledger(weighted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dimension, column in (("market_day", "market_day_cluster_id"), ("symbol", "symbol"), ("year", "calendar_year")):
        grouped = weighted.groupby(["definition_id", column], sort=True).agg(
            trades=("event_id", "size"), base_contribution_bps=("estimand_base_contribution_bps", "sum")
        ).reset_index()
        for item in grouped.to_dict("records"):
            rows.append({
                "definition_id": item["definition_id"], "dimension": dimension,
                "contributor_id": str(item[column]), "trades": item["trades"],
                "base_contribution_bps": item["base_contribution_bps"],
            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-frozen-economic-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.execute_frozen_economic_run:
        raise ValueError("explicit frozen KDA03 economic authorization flag required")
    if args.output.exists():
        raise ValueError("fresh KDA03 economic output root required")
    contract, definitions, events, mechanical_gates = preflight(args.archive, args.review)
    if sha256_file(MARKET_MANIFEST) != contract.get("market_manifest_sha256"):
        raise ValueError("frozen official market-manifest hash mismatch")
    authority = foundation.load_safe_manifest(MARKET_MANIFEST)
    economic_symbols = sorted(events.loc[events.branch_id.isin(definitions.branch_id), "symbol"].unique())
    authority_hash = verified_trade_authority_hash(authority, economic_symbols)
    if authority_hash != contract.get("official_trade_bar_authority_hash"):
        raise ValueError("frozen official trade-bar authority hash mismatch")
    records, timestamp_hash = reconstruct_schedule(definitions, events, mechanical_gates, authority)
    if timestamp_hash != contract.get("timestamp_authority_hash"):
        raise ValueError("frozen timestamp authority hash mismatch")
    # No open-price column is read until the full timestamp-only schedule has reconciled above.
    args.output.mkdir(parents=True)
    accepted = records.loc[records.accepted].copy()
    accepted.to_parquet(args.output / "KDA03_LEVEL3_ACCEPTED_EXECUTIONS.parquet", index=False, compression="zstd")
    records.loc[~records.accepted].to_parquet(args.output / "KDA03_LEVEL3_EXECUTION_REJECTIONS.parquet", index=False, compression="zstd")
    trades, price_rejections = price_and_score(accepted, events, authority)
    if len(price_rejections):
        raise ValueError("KDA03 exact-open price rejection")
    funding_shared.FUNDING_ROOT = FUNDING_ROOT
    panel, location, funding_hash = funding_shared.load_funding_panel()
    trades, boundaries = attach_funding_diagnostics(trades, panel, location)
    funded = trades
    outputs, day, weighted = reports(definitions, records, trades)
    metrics, gates, bootstrap, distributions, concentration, decisions, sensitivity, context = outputs
    primary_routes = decisions.loc[decisions.attempt.eq("primary")]
    control_eligible = primary_routes.loc[primary_routes.control_eligible]
    terminal = "KDA03_level3_routes_assigned"
    weighted.to_parquet(args.output / "KDA03_LEVEL3_TRADE_TAPE.parquet", index=False, compression="zstd")
    metrics.to_csv(args.output / "KDA03_LEVEL3_DEFINITION_METRICS.csv", index=False)
    gates.to_csv(args.output / "KDA03_LEVEL3_GATE_MATRIX.csv", index=False)
    day.to_parquet(args.output / "KDA03_LEVEL3_MARKET_DAY_RETURNS.parquet", index=False, compression="zstd")
    bootstrap.to_csv(args.output / "KDA03_LEVEL3_BOOTSTRAP_SUMMARY.csv", index=False)
    distributions.to_parquet(args.output / "KDA03_LEVEL3_BOOTSTRAP_DISTRIBUTIONS.parquet", index=False, compression="zstd")
    concentration.to_csv(args.output / "KDA03_LEVEL3_CONCENTRATION.csv", index=False)
    sensitivity.to_csv(args.output / "KDA03_LEVEL3_CLUSTER_SENSITIVITY.csv", index=False)
    context.to_csv(args.output / "KDA03_LEVEL3_CONTEXT_DIAGNOSTICS.csv", index=False)
    contribution_ledger(weighted).to_csv(args.output / "KDA03_LEVEL3_CONTRIBUTION_LEDGER.csv", index=False)
    decisions.to_csv(args.output / "KDA03_LEVEL3_DEFINITION_DECISIONS.csv", index=False)
    funded.groupby(["definition_id", "funding_partition"], as_index=False).agg(
        trades=("event_id", "size"),
        exact_boundaries=("exact_boundary_count", "sum"),
        imputed_boundaries=("imputed_boundary_count", "sum"),
        central_cashflow_bps=("funding_cashflow_central_bps", "mean"),
        conservative_cashflow_bps=("funding_cashflow_conservative_bps", "mean"),
        severe_cashflow_bps=("funding_cashflow_severe_bps", "mean"),
    ).to_csv(args.output / "KDA03_LEVEL3_FUNDING_PARTITIONS.csv", index=False)
    boundaries.to_parquet(args.output / "KDA03_LEVEL3_FUNDING_BOUNDARY_LEDGER.parquet", index=False, compression="zstd")
    (args.output / "KDA03_LEVEL3_DECISION.md").write_text(
        f"# KDA03 Level-3 Decision\n\n`{terminal}`\n\nPrimary definitions routed: `{len(primary_routes)}`. "
        f"Control-eligible definitions: `{len(control_eligible)}`. Routes are assignments, not passes. "
        "All definitions remain frozen and controls were not executed.\n",
        encoding="utf-8",
    )
    (args.output / "KDA03_LEVEL3_CLAIM_BOUNDARY.md").write_text(
        "# KDA03 Level-3 Claim Boundary\n\nKraken train-only, current-roster/lifecycle-capped evidence under inferred "
        "analytics units. KDA03A is a reference-led directional proxy, not a spread or arbitrage test. "
        "Mark confirms state but is not a fill. "
        "Funding is diagnostic and excluded from gates. No control, validation, protected-period, portfolio, "
        "live, or production claim is authorized.\n",
        encoding="utf-8",
    )
    audit = {
        "task_id": TASK_ID,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "contract_hash": contract["level3_contract_hash"],
        "contract_file_sha256": sha256_file(args.archive / "KDA03_FINAL_LEVEL3_CONTRACT.json"),
        "definition_register_sha256": sha256_file(args.archive / "KDA03_LEVEL3_DEFINITION_REGISTER.csv"),
        "event_tape_sha256": sha256_file(args.archive / "KDA03_V1_EVENT_TAPE.parquet"),
        "parent_tape_sha256": sha256_file(args.archive / "KDA03_V1_PARENT_EPISODE_TAPE.parquet"),
        "runner_sha256": sha256_file(Path(__file__)),
        "calculation_module_sha256": sha256_file(ROOT / "tools/qlmg_kda03_level3.py"),
        "market_manifest_sha256": sha256_file(MARKET_MANIFEST),
        "official_trade_bar_authority_hash": authority_hash,
        "timestamp_authority_hash": timestamp_hash,
        "funding_model_hash": funding_hash,
        "schedule_records": len(records),
        "accepted": int(records.accepted.sum()),
        "price_rejections": len(price_rejections),
        "trades": len(trades),
        "primary_route_count": len(primary_routes),
        "primary_routes": primary_routes[["definition_id", "policy_route"]].to_dict("records"),
        "control_eligible_count": len(control_eligible),
        "control_eligible_definitions": control_eligible.definition_id.tolist(),
        "controls_executed": False,
        "protected_rows_opened": 0,
        "terminal_decision": terminal,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json(args.output / "RUN_AUDIT.json", audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
