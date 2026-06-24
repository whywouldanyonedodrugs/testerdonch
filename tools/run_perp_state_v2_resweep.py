#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from shared_utils import get_symbols_from_file  # noqa: E402
from tools.perp_state_v2 import (  # noqa: E402
    DEFAULT_CONTEXT_ROOT,
    PERP_STATE_CONTRACT_VERSION,
    audit_sidecar_readiness,
    build_symbol_month_eligibility,
    file_sha1,
    join_context_asof,
    load_bybit_context,
    schema_hash,
    write_symbol_month_eligibility,
)
from tools.perp_state_v2_diagnostics import (  # noqa: E402
    EventClusterSpec,
    assign_event_cluster_ids,
    concentration_summary,
    leave_one_group_out_summary,
)
from tools.perp_state_v2_family_specs import (  # noqa: E402
    FAMILY_BREADTH_FOLLOWER_V2,
    FAMILY_PANIC_RECLAIM_V2,
    FAMILY_SPECS,
    FAMILY_WASHED_REVERSAL_V2,
    context_contracts_for_specs,
    family_specs_hash,
    selected_family_specs,
)
from tools.run_entry_ledger_resweep import _load_parts  # noqa: E402


DEFAULT_RUN_PREFIX = "phase_perp_state_v2_resweep"
DEFAULT_BASELINE_RUN_ROOT = Path("results/rebaseline/phase_event_crowding_entry_sweep_event_crowding_20260416")
DEFAULT_GLOBAL_START = pd.Timestamp("2023-01-22 00:00:00+00:00")
DEFAULT_DEVELOPMENT_END = pd.Timestamp("2025-08-31 23:59:59+00:00")
DEFAULT_HOLDOUT_START = pd.Timestamp("2025-09-01 00:00:00+00:00")
DEFAULT_GLOBAL_END = pd.Timestamp("2026-03-05 23:59:59+00:00")

BASE_TO_V2_FAMILY = {
    "panic_deleveraging_reclaim": FAMILY_PANIC_RECLAIM_V2,
    "washed_out_crowding_reversal": FAMILY_WASHED_REVERSAL_V2,
    "breadth_follower_catchup": FAMILY_BREADTH_FOLLOWER_V2,
}

LANE_COSTS = {"H": 0.00310, "M": 0.00450, "L": 0.00750, "unknown": 0.00750}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Independent perp-state v2 readiness and controlled resweep runner.")
    p.add_argument("--stage", default="all", choices=["readiness", "eligibility", "phase1", "rank-train", "freeze-shortlist", "microstructure-smoke", "all"])
    p.add_argument("--run-id", default="")
    p.add_argument("--results-root", default="results/rebaseline")
    p.add_argument("--context-root", default=str(DEFAULT_CONTEXT_ROOT))
    p.add_argument("--baseline-run-root", default=str(DEFAULT_BASELINE_RUN_ROOT))
    p.add_argument("--global-start", default="2023-01-22")
    p.add_argument("--development-end", default="2025-08-31")
    p.add_argument("--holdout-start", default="2025-09-01")
    p.add_argument("--global-end", default="2026-03-05")
    p.add_argument("--family-filter", default="")
    p.add_argument("--phase-filter", default="")
    p.add_argument("--symbol-filter", default="")
    p.add_argument("--symbols-limit", type=int, default=0)
    p.add_argument("--min-sidecar-coverage", type=float, default=0.95)
    p.add_argument("--allow-partial-sidecar", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--shortlist-file", default="")
    return p.parse_args()


def _parse_day(s: str, *, end_of_day: bool = False) -> pd.Timestamp:
    ts = pd.Timestamp(str(s), tz="UTC")
    if end_of_day:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def _split_arg(s: str) -> list[str] | None:
    out = [x.strip() for x in str(s or "").split(",") if x.strip()]
    return out or None


def _run_root(args: argparse.Namespace) -> Path:
    run_id = args.run_id.strip() or pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    return (REPO / args.results_root / f"{DEFAULT_RUN_PREFIX}_{run_id}").resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _selected_symbols(args: argparse.Namespace) -> list[str]:
    requested = _split_arg(args.symbol_filter)
    if requested:
        symbols = [s.upper() for s in requested]
    else:
        symbols = get_symbols_from_file()
    if int(args.symbols_limit) > 0:
        symbols = symbols[: int(args.symbols_limit)]
    return symbols


def _family_specs(args: argparse.Namespace):
    specs = selected_family_specs(family_filter=_split_arg(args.family_filter), phase_filter=_split_arg(args.phase_filter))
    if args.min_sidecar_coverage != 0.95:
        # Preserve frozen family definitions but allow a runner-level readiness experiment.
        from dataclasses import replace

        specs = {k: replace(v, min_symbol_month_coverage=float(args.min_sidecar_coverage)) for k, v in specs.items()}
    return specs


def _phase_windows(args: argparse.Namespace) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        "development": (_parse_day(args.global_start), _parse_day(args.development_end, end_of_day=True)),
        "holdout_sealed": (_parse_day(args.holdout_start), _parse_day(args.global_end, end_of_day=True)),
    }


def _write_run_manifest(args: argparse.Namespace, root: Path, specs: Mapping[str, Any], eligibility_hash: str = "") -> None:
    manifest = {
        "contract_version": PERP_STATE_CONTRACT_VERSION,
        "run_root": str(root),
        "context_root": str(Path(args.context_root).resolve()),
        "baseline_run_root": str(Path(args.baseline_run_root).resolve()),
        "global_start": args.global_start,
        "development_end": args.development_end,
        "holdout_start": args.holdout_start,
        "global_end": args.global_end,
        "stage": args.stage,
        "family_specs_hash": family_specs_hash(specs),
        "sidecar_schema_hash": schema_hash(),
        "symbol_month_eligibility_hash": eligibility_hash,
        "holdout_policy": "sealed; no holdout reveal implemented in this runner",
        "canonical_sidecar_schema": "timestamp plus *_source_open_ts/*_source_close_ts, lsr_source_ts, lsr_source_close_ts, context_source_close_ts",
        "external_helper_dependency": "none",
        "cost_model_by_liquidity_lane": LANE_COSTS,
    }
    _write_json(root / "run_manifest.json", manifest)


def run_readiness(args: argparse.Namespace, root: Path, specs: Mapping[str, Any]) -> None:
    symbols = _selected_symbols(args)
    coverage, cadence = audit_sidecar_readiness(symbols=symbols, context_root=args.context_root)
    out = root / "readiness"
    out.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out / "sidecar_symbol_coverage.csv", index=False)
    cadence.to_csv(out / "sidecar_group_cadence.csv", index=False)
    summary = {
        "symbols_requested": int(len(symbols)),
        "context_files_present": int(coverage.get("context_file_exists", pd.Series(dtype=bool)).fillna(False).sum()) if not coverage.empty else 0,
        "schema_hash": schema_hash(),
        "coverage_csv_sha1": file_sha1(out / "sidecar_symbol_coverage.csv"),
        "cadence_csv_sha1": file_sha1(out / "sidecar_group_cadence.csv"),
    }
    _write_json(out / "sidecar_readiness_summary.json", summary)
    _write_run_manifest(args, root, specs)


def run_eligibility(args: argparse.Namespace, root: Path, specs: Mapping[str, Any]) -> str:
    symbols = _selected_symbols(args)
    contracts = context_contracts_for_specs(specs)
    eligibility = build_symbol_month_eligibility(
        symbols=symbols,
        phase_windows=_phase_windows(args),
        family_contracts=contracts,
        context_root=args.context_root,
    )
    out_path = root / "symbol_month_eligibility.csv"
    digest = write_symbol_month_eligibility(out_path, eligibility)
    summary = (
        eligibility.groupby(["phase", "family"], as_index=False)
        .agg(symbol_months=("symbol", "count"), admissible_symbol_months=("admissible", "sum"), median_coverage=("coverage_ratio", "median"))
        if not eligibility.empty
        else pd.DataFrame()
    )
    summary.to_csv(root / "symbol_month_eligibility_summary.csv", index=False)
    _write_run_manifest(args, root, specs, eligibility_hash=digest)
    if not bool(args.allow_partial_sidecar):
        dev = eligibility[eligibility["phase"].eq("development")]
        if dev.empty:
            raise RuntimeError("no development symbol-month eligibility rows were produced; refusing v2 run")
        bad = dev[~dev["admissible"].astype(bool)]
        if not bad.empty:
            examples = bad[["family", "symbol", "year_month", "reject_reason"]].head(5).to_dict("records")
            raise RuntimeError(
                "incomplete development sidecar eligibility under perp_state_v2; "
                f"bad_symbol_month_rows={len(bad)} examples={examples}. "
                "Use --allow-partial-sidecar only for smoke/planning runs."
            )
    return digest


def _read_baseline_shortlist(baseline_root: Path) -> pd.DataFrame:
    path = baseline_root / "shortlist_train_only.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing baseline shortlist: {path}")
    return pd.read_csv(path)


def _baseline_config_roots(baseline_root: Path, family: str, config_id: str) -> tuple[Path, Path]:
    ledger = baseline_root / "entry_ledger_parts" / f"family={family}" / f"config={config_id}"
    paths = baseline_root / "path_stats_parts" / f"family={family}" / f"config={config_id}"
    return ledger, paths


def run_phase1(args: argparse.Namespace, root: Path, specs: Mapping[str, Any]) -> None:
    baseline_root = Path(args.baseline_run_root)
    short = _read_baseline_shortlist(baseline_root)
    rows = []
    path_rows = []
    context_cache: dict[str, pd.DataFrame] = {}
    contracts = context_contracts_for_specs(specs)
    eligibility_path = root / "symbol_month_eligibility.csv"
    eligibility = pd.read_csv(eligibility_path) if eligibility_path.exists() else pd.DataFrame()
    for item in short.itertuples(index=False):
        base_family = str(item.family)
        v2_family = BASE_TO_V2_FAMILY.get(base_family)
        if v2_family not in specs:
            continue
        contract = contracts[v2_family]
        ledger_root, path_root = _baseline_config_roots(baseline_root, base_family, str(item.config_id))
        ledger = _load_parts(ledger_root)
        path_stats = _load_parts(path_root)
        if ledger.empty or path_stats.empty:
            continue
        ledger = ledger.copy()
        ledger["base_family"] = base_family
        ledger["base_config_id"] = str(item.config_id)
        ledger["family"] = v2_family
        ledger["config_id"] = str(item.config_id).replace(base_family, v2_family, 1)
        ledger["decision_ts"] = pd.to_datetime(ledger["decision_ts"], utc=True, errors="coerce")
        if not eligibility.empty:
            ledger["year_month"] = ledger["decision_ts"].dt.strftime("%Y-%m")
            elig = eligibility[(eligibility["phase"].eq("development")) & (eligibility["family"].eq(v2_family)) & (eligibility["admissible"].astype(bool))]
            keys = set(zip(elig["symbol"].astype(str), elig["year_month"].astype(str)))
            ledger = ledger[ledger.apply(lambda r: (str(r["symbol"]), str(r["year_month"])) in keys, axis=1)].copy()
        joined_parts = []
        for symbol, sym_df in ledger.groupby("symbol", sort=False):
            if symbol not in context_cache:
                context_cache[symbol] = load_bybit_context(symbol, root=args.context_root, require_schema=True)
            if context_cache[symbol].empty:
                continue
            joined = join_context_asof(sym_df, context_cache[symbol], contract, fail_closed=True)
            if not joined.empty:
                joined_parts.append(joined)
        if not joined_parts:
            continue
        joined_ledger = pd.concat(joined_parts, ignore_index=True)
        joined_ledger["state_consistency_ok"] = _state_consistency_ok(v2_family, joined_ledger)
        joined_ledger = assign_event_cluster_ids(joined_ledger, EventClusterSpec(trigger_hours_by_family={k: v.trigger_window_hours for k, v in specs.items()}))
        rows.append(joined_ledger)
        path_stats = path_stats[path_stats["entry_id"].isin(set(joined_ledger["entry_id"]))].copy()
        path_stats["family"] = v2_family
        path_stats["config_id"] = joined_ledger["config_id"].iloc[0]
        path_rows.append(path_stats)
    out = root / "phase1_augmented_v1"
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        ledger_out = pd.concat(rows, ignore_index=True)
        path_out = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
    else:
        ledger_out = pd.DataFrame()
        path_out = pd.DataFrame()
    ledger_out.to_parquet(out / "phase1_augmented_entries.parquet", index=False)
    path_out.to_parquet(out / "phase1_augmented_path_stats.parquet", index=False)
    _write_json(
        out / "phase1_summary.json",
        {
            "augmented_entry_rows": int(len(ledger_out)),
            "augmented_path_rows": int(len(path_out)),
            "families": sorted(ledger_out["family"].dropna().astype(str).unique().tolist()) if not ledger_out.empty else [],
            "holdout_revealed": False,
        },
    )


def _state_consistency_ok(family: str, df: pd.DataFrame) -> pd.Series:
    premium = pd.to_numeric(df.get("premium_close"), errors="coerce")
    premium_c1 = pd.to_numeric(df.get("premium_compression_1h"), errors="coerce")
    spread = pd.to_numeric(df.get("mark_index_spread_pct"), errors="coerce")
    lsr = pd.to_numeric(df.get("long_short_account_ratio"), errors="coerce")
    if family == FAMILY_PANIC_RECLAIM_V2:
        return (premium <= 0.0) | (premium_c1 <= 0.0) | (spread.abs() >= 0.001) | (lsr <= 1.0)
    if family == FAMILY_WASHED_REVERSAL_V2:
        return (premium <= 0.0) | (premium_c1 <= 0.0) | (lsr <= 1.0)
    if family == FAMILY_BREADTH_FOLLOWER_V2:
        return ~((premium > 0.0) & (lsr > 1.5))
    return pd.Series(True, index=df.index)


def run_rank_train(args: argparse.Namespace, root: Path) -> None:
    base = root / "phase1_augmented_v1"
    ledger_path = base / "phase1_augmented_entries.parquet"
    path_path = base / "phase1_augmented_path_stats.parquet"
    if not ledger_path.exists() or not path_path.exists():
        raise RuntimeError("phase1 must complete before rank-train")
    ledger = pd.read_parquet(ledger_path)
    path_stats = pd.read_parquet(path_path)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    if ledger.empty or path_stats.empty:
        pd.DataFrame().to_csv(out / "perp_state_v2_config_summary_train.csv", index=False)
        pd.DataFrame().to_csv(out / "perp_state_v2_concentration_summary.csv", index=False)
        return
    development_end = _parse_day(args.development_end, end_of_day=True)
    ledger["decision_ts"] = pd.to_datetime(ledger["decision_ts"], utc=True, errors="coerce")
    ledger = ledger[ledger["decision_ts"] <= development_end].copy()
    path_stats = path_stats[path_stats["entry_id"].isin(set(ledger["entry_id"]))].copy()
    summary_rows = []
    for (family, config_id), group in ledger.groupby(["family", "config_id"], dropna=False):
        ps = path_stats[path_stats["entry_id"].isin(set(group["entry_id"]))].copy()
        if ps.empty:
            continue
        cost = group[["entry_id", "liquidity_lane"]].drop_duplicates("entry_id").set_index("entry_id").reindex(ps["entry_id"])["liquidity_lane"].astype(str).map(LANE_COSTS).fillna(LANE_COSTS["unknown"]).to_numpy(dtype=float)
        r72 = pd.to_numeric(ps["fwd_ret_close_72h"], errors="coerce") - cost
        r24 = pd.to_numeric(ps["fwd_ret_close_24h"], errors="coerce") - cost
        mfe = pd.to_numeric(ps["mfe_72h"], errors="coerce") - cost
        mae = pd.to_numeric(ps["mae_72h"], errors="coerce") - cost
        ratio = mfe / mae.abs().clip(lower=1e-9)
        month_share = group["decision_ts"].dt.strftime("%Y-%m").value_counts(normalize=True)
        cluster_share = group["event_cluster_id"].value_counts(normalize=True)
        state_ok_share = float(group.get("state_consistency_ok", pd.Series(False, index=group.index)).fillna(False).mean())
        eligible = bool(
            len(group) >= 250
            and group["event_cluster_id"].nunique() >= 20
            and group["decision_ts"].dt.strftime("%Y-%m").nunique() >= 6
            and float(r72.median()) > 0.0
            and float(r72.mean()) > 0.0
            and float(ratio.median()) >= 1.10
            and state_ok_share >= 0.80
            and (float(month_share.max()) <= 0.40 if len(month_share) else False)
            and (float(cluster_share.max()) <= 0.25 if len(cluster_share) else False)
        )
        summary_rows.append(
            {
                "family": family,
                "config_id": config_id,
                "entries": int(len(group)),
                "distinct_event_clusters": int(group["event_cluster_id"].nunique()),
                "active_months": int(group["decision_ts"].dt.strftime("%Y-%m").nunique()),
                "median_ret24_cost": float(r24.median()),
                "median_ret72_cost": float(r72.median()),
                "mean_ret72_cost": float(r72.mean()),
                "median_mfe_mae_72h_cost": float(ratio.median()),
                "state_consistency_share": state_ok_share,
                "max_month_entry_share": float(month_share.max()) if len(month_share) else np.nan,
                "max_event_cluster_entry_share": float(cluster_share.max()) if len(cluster_share) else np.nan,
                "eligible_exploratory_v2": eligible,
                "ranking_score": float(r72.median() * 100.0 + ratio.median() + state_ok_share),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["eligible_exploratory_v2", "ranking_score", "entries"], ascending=[False, False, False]) if summary_rows else pd.DataFrame()
    summary.to_csv(out / "perp_state_v2_config_summary_train.csv", index=False)
    concentration_summary(ledger, path_stats).to_csv(out / "perp_state_v2_concentration_summary.csv", index=False)
    leave_one_group_out_summary(ledger.assign(year_month=ledger["decision_ts"].dt.strftime("%Y-%m")), path_stats, group_col="year_month").to_csv(out / "leave_one_month_out_summary.csv", index=False)
    leave_one_group_out_summary(ledger, path_stats, group_col="event_cluster_id").to_csv(out / "leave_one_event_cluster_out_summary.csv", index=False)


def run_freeze_shortlist(args: argparse.Namespace, root: Path) -> None:
    path = root / "analysis" / "perp_state_v2_config_summary_train.csv"
    if not path.exists():
        raise RuntimeError("rank-train must complete before freeze-shortlist")
    df = pd.read_csv(path) if path.stat().st_size else pd.DataFrame()
    if df.empty:
        out = df
    else:
        elig = df[df["eligible_exploratory_v2"].astype(bool)].copy()
        picks = []
        seen = set()
        for row in elig.sort_values(["ranking_score", "entries"], ascending=[False, False]).itertuples(index=False):
            fam = str(row.family)
            if fam in seen:
                continue
            picks.append(row._asdict())
            seen.add(fam)
            if len(picks) >= 3:
                break
        out = pd.DataFrame(picks)
    shortlist = Path(args.shortlist_file) if args.shortlist_file else root / "shortlist_train_only.csv"
    shortlist.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(shortlist, index=False)


def main() -> int:
    args = _parse_args()
    root = _run_root(args)
    root.mkdir(parents=True, exist_ok=True)
    specs = _family_specs(args)
    if args.stage in {"readiness", "all"}:
        run_readiness(args, root, specs)
    if args.stage in {"eligibility", "all"}:
        run_eligibility(args, root, specs)
    if args.stage in {"phase1", "all"}:
        run_phase1(args, root, specs)
    if args.stage in {"rank-train", "all"}:
        run_rank_train(args, root)
    if args.stage in {"freeze-shortlist", "all"}:
        run_freeze_shortlist(args, root)
    if args.stage == "microstructure-smoke":
        _write_json(root / "microstructure_smoke_note.json", {"status": "scaffold_only", "note": "Use tools.perp_microstructure_v2 capture helpers explicitly; historical microstructure sweeps require coverage first."})
    _write_json(root / "progress_summary.json", {"stage": args.stage, "run_root": str(root), "holdout_revealed": False})
    print(f"[perp-state-v2] completed stage={args.stage} run_root={root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
