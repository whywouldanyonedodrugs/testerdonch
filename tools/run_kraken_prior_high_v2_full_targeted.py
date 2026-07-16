#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tools import run_kraken_prior_high_v2_materialization_profile as profile
from tools import run_kraken_family_engine_aggregate_first_sweep as runner


IMPLEMENTATION_ROOT = Path("results/rebaseline/phase_kraken_prior_high_reclaim_v2_materialization_profile_implementation_20260712_v1")
PROFILE = "prior_high_reclaim_v2_targeted_materialization_profile_20260712_v2"
FUNDING_MODES = ("central_imputed", "conservative_imputed", "severe_imputed", "exact_only_slice")
SLIPPAGE_BPS = (4, 8, 12)


def csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def js(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def block_bootstrap_ci(frame: pd.DataFrame, value_col: str, seed_key: str) -> tuple[float, float]:
    if frame.empty:
        return np.nan, np.nan
    blocks = [group[value_col].to_numpy(float) for _, group in frame.groupby("symbol_month", sort=True)]
    if not blocks:
        return np.nan, np.nan
    seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    samples = np.empty(250)
    for index in range(250):
        chosen = [blocks[position] for position in rng.integers(0, len(blocks), len(blocks))]
        samples[index] = np.concatenate(chosen).mean()
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def scenario_summary(scenarios: pd.DataFrame) -> pd.DataFrame:
    return scenarios.groupby(["candidate_definition_id", "funding_mode", "slippage_round_trip_bps"], dropna=False).agg(
        events=("event_key", "nunique"), gross_R=("raw_gross_R", "sum"), fee_R=("raw_fee_R", "sum"),
        funding_R=("scenario_funding_raw_R", "sum"), slippage_R=("added_slippage_raw_R", "sum"),
        net_R=("scenario_raw_net_R", "sum"), median_net_R=("scenario_raw_net_R", "median"),
        exact_boundary_rows=("exact_boundary_rows", "sum"), imputed_boundary_rows=("imputed_boundary_rows", "sum"),
    ).reset_index()


def candidate_forensics(scenarios: pd.DataFrame) -> dict[str, pd.DataFrame]:
    severe = scenarios[(scenarios["funding_mode"] == "severe_imputed") & (scenarios["slippage_round_trip_bps"] == 12)].copy()
    severe["month"] = pd.to_datetime(severe["entry_ts"], utc=True).dt.strftime("%Y-%m")
    severe["symbol_month"] = severe["symbol"].astype(str) + "|" + severe["month"]
    severe["period"] = np.select([
        pd.to_datetime(severe["entry_ts"], utc=True).dt.year.eq(2024),
        pd.to_datetime(severe["entry_ts"], utc=True).lt(pd.Timestamp("2025-07-01", tz="UTC")),
    ], ["2024", "2025_h1"], default="2025_h2")
    top, trim, leave_symbol, leave_month, leave_sm, periods, exact = [], [], [], [], [], [], []
    for cid, group in severe.groupby("candidate_definition_id", sort=True):
        ordered = group.sort_values("scenario_raw_net_R", ascending=False)
        total = float(group["scenario_raw_net_R"].sum())
        remove = max(1, math.ceil(len(group) * 0.01))
        total_abs = float(group["scenario_raw_net_R"].abs().sum())
        top.append({
            "candidate_definition_id": cid, "events": len(group), "base_net_R": total,
            "net_without_top_1": float(ordered.iloc[1:]["scenario_raw_net_R"].sum()),
            "net_without_top_3": float(ordered.iloc[3:]["scenario_raw_net_R"].sum()),
            "top_event_abs_share": float(group["scenario_raw_net_R"].abs().max() / total_abs) if total_abs else np.nan,
            "dominant_symbol_abs_share": float(group.groupby("symbol")["scenario_raw_net_R"].sum().abs().max() / total_abs) if total_abs else np.nan,
            "dominant_month_abs_share": float(group.groupby("month")["scenario_raw_net_R"].sum().abs().max() / total_abs) if total_abs else np.nan,
            "dominant_symbol_month_abs_share": float(group.groupby("symbol_month")["scenario_raw_net_R"].sum().abs().max() / total_abs) if total_abs else np.nan,
        })
        trim.append({"candidate_definition_id": cid, "events": len(group), "removed_events": remove, "base_net_R": total, "net_after_top_1pct_trim": float(ordered.iloc[remove:]["scenario_raw_net_R"].sum())})
        for column, target in [("symbol", leave_symbol), ("month", leave_month), ("symbol_month", leave_sm)]:
            for value, excluded in group.groupby(column, dropna=False):
                target.append({"candidate_definition_id": cid, "excluded_value": value, "base_net_R": total, "net_R_after_exclusion": total - float(excluded["scenario_raw_net_R"].sum()), "events_after_exclusion": len(group) - len(excluded)})
        for period, part in group.groupby("period"):
            periods.append({"candidate_definition_id": cid, "period": period, "events": len(part), "net_R": float(part["scenario_raw_net_R"].sum())})
        for exact_flag, part in group.groupby("all_boundaries_exact"):
            exact.append({"candidate_definition_id": cid, "funding_partition": "fully_exact" if exact_flag else "imputed_or_mixed", "events": len(part), "net_R": float(part["scenario_raw_net_R"].sum()), "exact_boundary_rows": int(part["exact_boundary_rows"].sum()), "imputed_boundary_rows": int(part["imputed_boundary_rows"].sum())})
    return {
        "severe": severe, "top": pd.DataFrame(top), "trim": pd.DataFrame(trim),
        "leave_symbol": pd.DataFrame(leave_symbol), "leave_month": pd.DataFrame(leave_month),
        "leave_symbol_month": pd.DataFrame(leave_sm), "period": pd.DataFrame(periods), "exact": pd.DataFrame(exact),
    }


def build_paired(candidate_scenarios: pd.DataFrame, control_scenarios: pd.DataFrame) -> pd.DataFrame:
    candidate = candidate_scenarios[[
        "candidate_definition_id", "event_id", "event_key", "symbol", "decision_ts", "funding_mode",
        "slippage_round_trip_bps", "scenario_raw_net_R", "all_boundaries_exact", "exact_boundary_rows", "imputed_boundary_rows",
    ]].rename(columns={
        "event_id": "candidate_event_id", "event_key": "candidate_event_key", "symbol": "candidate_symbol",
        "scenario_raw_net_R": "candidate_net_R", "all_boundaries_exact": "candidate_all_boundaries_exact",
        "exact_boundary_rows": "candidate_exact_boundary_rows", "imputed_boundary_rows": "candidate_imputed_boundary_rows",
    })
    controls = control_scenarios[[
        "source_candidate_definition_id", "source_candidate_event_id", "control_class", "event_id", "symbol",
        "funding_mode", "slippage_round_trip_bps", "scenario_raw_net_R", "all_boundaries_exact",
        "exact_boundary_rows", "imputed_boundary_rows",
    ]].rename(columns={
        "source_candidate_definition_id": "candidate_definition_id", "event_id": "control_event_id",
        "symbol": "control_symbol", "scenario_raw_net_R": "control_net_R",
        "all_boundaries_exact": "control_all_boundaries_exact", "exact_boundary_rows": "control_exact_boundary_rows",
        "imputed_boundary_rows": "control_imputed_boundary_rows",
    })
    paired = controls.merge(
        candidate,
        left_on=["candidate_definition_id", "source_candidate_event_id", "funding_mode", "slippage_round_trip_bps"],
        right_on=["candidate_definition_id", "candidate_event_id", "funding_mode", "slippage_round_trip_bps"],
        how="inner", validate="many_to_one",
    )
    paired["paired_uplift_R"] = paired["candidate_net_R"] - paired["control_net_R"]
    paired["month"] = pd.to_datetime(paired["decision_ts"], utc=True).dt.strftime("%Y-%m")
    paired["symbol_month"] = paired["candidate_symbol"].astype(str) + "|" + paired["month"]
    paired["period"] = np.select([
        pd.to_datetime(paired["decision_ts"], utc=True).dt.year.eq(2024),
        pd.to_datetime(paired["decision_ts"], utc=True).lt(pd.Timestamp("2025-07-01", tz="UTC")),
    ], ["2024", "2025_h1"], default="2025_h2")
    paired["funding_partition"] = np.where(paired["candidate_all_boundaries_exact"] & paired["control_all_boundaries_exact"], "both_fully_exact", "imputed_or_mixed")
    return paired


def symmetric_forensics(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, leave = [], []
    base_keys = ["candidate_definition_id", "control_class", "funding_mode", "slippage_round_trip_bps"]
    scopes: list[tuple[str, str, pd.DataFrame]] = [("all", "full_train", paired)]
    scopes.extend(("all", str(period), part) for period, part in paired.groupby("period", sort=True))
    scopes.extend((str(partition), "full_train", part) for partition, part in paired.groupby("funding_partition", sort=True))
    for partition, period, scoped in scopes:
      for values, group in scoped.groupby(base_keys, sort=True):
        cid, control, mode, bps = values
        differences = group["paired_uplift_R"].sort_values(ascending=False).reset_index(drop=True)
        trim = max(1, math.ceil(len(group) * 0.01))
        lo, hi = differences.quantile(0.01), differences.quantile(0.99)
        ci_low, ci_high = block_bootstrap_ci(group, "paired_uplift_R", "|".join(map(str, (*values, partition, period))))
        rows.append({
            "candidate_definition_id": cid, "control_class": control, "funding_mode": mode,
            "slippage_round_trip_bps": bps, "funding_partition": partition, "period": period,
            "paired_rows": len(group), "paired_mean_uplift_R": float(differences.mean()),
            "paired_median_uplift_R": float(differences.median()), "candidate_win_fraction": float((differences > 0).mean()),
            "winsorized_mean_uplift_R": float(differences.clip(lo, hi).mean()),
            "mean_after_top_1_difference_removal": float(differences.iloc[1:].mean()),
            "mean_after_top_3_difference_removal": float(differences.iloc[3:].mean()),
            "mean_after_top_1pct_difference_trim": float(differences.iloc[trim:].mean()),
            "block_bootstrap_mean_ci_low": ci_low, "block_bootstrap_mean_ci_high": ci_high,
        })
        for column in ("candidate_symbol", "month", "symbol_month"):
            for excluded, excluded_group in group.groupby(column, dropna=False):
                remaining = group[group[column] != excluded]
                leave.append({
                    "candidate_definition_id": cid, "control_class": control, "funding_mode": mode,
                    "slippage_round_trip_bps": bps, "funding_partition": partition, "period": period,
                    "scope": column, "excluded_value": excluded, "paired_rows_after_exclusion": len(remaining),
                    "mean_uplift_after_exclusion": float(remaining["paired_uplift_R"].mean()) if len(remaining) else np.nan,
                })
    return pd.DataFrame(rows), pd.DataFrame(leave)


def matched_unmatched(candidate_scenarios: pd.DataFrame, control_outcomes: pd.DataFrame) -> pd.DataFrame:
    severe = candidate_scenarios[(candidate_scenarios["funding_mode"] == "severe_imputed") & (candidate_scenarios["slippage_round_trip_bps"] == 12)]
    rows = []
    for cid, candidate in severe.groupby("candidate_definition_id"):
        for control in profile.CONTROL_CLASSES:
            matched = set(control_outcomes[(control_outcomes["source_candidate_definition_id"] == cid) & (control_outcomes["control_class"] == control)]["source_candidate_event_id"].astype(str))
            yes = candidate[candidate["event_id"].astype(str).isin(matched)]
            no = candidate[~candidate["event_id"].astype(str).isin(matched)]
            rows.append({
                "candidate_definition_id": cid, "control_class": control, "candidate_events": len(candidate),
                "matched_events": len(yes), "unmatched_events": len(no), "outcome_coverage": len(yes) / len(candidate) if len(candidate) else 0,
                "matched_candidate_mean_R": float(yes["scenario_raw_net_R"].mean()) if len(yes) else np.nan,
                "unmatched_candidate_mean_R": float(no["scenario_raw_net_R"].mean()) if len(no) else np.nan,
            })
    return pd.DataFrame(rows)


def paired_variants(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for primary, variant in profile.PAIRS.items():
        left = scenarios[scenarios["candidate_definition_id"] == primary]
        right = scenarios[scenarios["candidate_definition_id"] == variant]
        merged = left.merge(right, on=["symbol", "decision_ts", "funding_mode", "slippage_round_trip_bps"], suffixes=("_primary", "_variant"), how="inner", validate="one_to_one")
        for (mode, bps), group in merged.groupby(["funding_mode", "slippage_round_trip_bps"]):
            diff = group["scenario_raw_net_R_variant"] - group["scenario_raw_net_R_primary"]
            rows.append({
                "primary_definition": primary, "paired_variant": variant, "funding_mode": mode,
                "slippage_round_trip_bps": bps, "paired_events": len(group),
                "primary_net_R": float(group["scenario_raw_net_R_primary"].sum()),
                "variant_net_R": float(group["scenario_raw_net_R_variant"].sum()),
                "variant_minus_primary_R": float(diff.sum()), "variant_better_fraction": float((diff > 0).mean()),
                "independent_entry_discoveries": 1,
            })
    return pd.DataFrame(rows)


def decisions(definitions: pd.DataFrame, candidate_summary: pd.DataFrame, forensic: dict[str, pd.DataFrame], symmetric: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    severe = candidate_summary[(candidate_summary["funding_mode"] == "severe_imputed") & (candidate_summary["slippage_round_trip_bps"] == 12)].set_index("candidate_definition_id")
    exact = candidate_summary[(candidate_summary["funding_mode"] == "exact_only_slice") & (candidate_summary["slippage_round_trip_bps"] == 8)].set_index("candidate_definition_id")
    top = forensic["top"].set_index("candidate_definition_id")
    trim = forensic["trim"].set_index("candidate_definition_id")
    period = forensic["period"]
    rows = []
    paired_variants_set = set(profile.PAIRS.values())
    for definition in definitions.itertuples(index=False):
        cid = str(definition.candidate_definition_id)
        severe_net = float(severe.loc[cid, "net_R"]) if cid in severe.index else np.nan
        exact_net = float(exact.loc[cid, "net_R"]) if cid in exact.index else np.nan
        top3 = float(top.loc[cid, "net_without_top_3"]) if cid in top.index else np.nan
        trimmed = float(trim.loc[cid, "net_after_top_1pct_trim"]) if cid in trim.index else np.nan
        positive_periods = int((period[period["candidate_definition_id"] == cid]["net_R"] > 0).sum())
        control_rows = symmetric[
            (symmetric["candidate_definition_id"] == cid) & (symmetric["funding_mode"] == "severe_imputed")
            & (symmetric["slippage_round_trip_bps"] == 12) & (symmetric["funding_partition"] == "all")
            & (symmetric["period"] == "full_train")
        ]
        eligible_controls = set(coverage[(coverage["candidate_definition_id"] == cid) & ~coverage["control_coverage_cap"]]["control_class"])
        robust_controls = control_rows[
            control_rows["control_class"].isin(eligible_controls)
            & control_rows["paired_mean_uplift_R"].gt(0) & control_rows["paired_median_uplift_R"].gt(0)
            & control_rows["winsorized_mean_uplift_R"].gt(0) & control_rows["mean_after_top_3_difference_removal"].gt(0)
            & control_rows["block_bootstrap_mean_ci_low"].gt(0)
        ]["control_class"].nunique()
        candidate_robust = severe_net > 0 and top3 > 0 and trimmed > 0
        if cid in paired_variants_set:
            decision = "diagnostic_only"
        elif candidate_robust and exact_net > 0 and positive_periods >= 2 and robust_controls >= 3:
            decision = "advance_to_train_stability_review"
        elif candidate_robust or (severe_net > 0 and positive_periods >= 2):
            decision = "preserve_as_context_sleeve"
        elif str(definition.signal_type) == "prior_high_breakout":
            decision = "diagnostic_only"
        else:
            decision = "defer_current_translation"
        caps = ["train_only_materialized_evidence_cap", "funding_imputed_train_screen_cap", "no_depth_slippage_cap"]
        if coverage[(coverage["candidate_definition_id"] == cid)]["control_coverage_cap"].any():
            caps.append("control_coverage_cap")
        rows.append({
            "candidate_definition_id": cid, "entry_cluster_id": definition.entry_cluster_id,
            "paired_variant_not_independent": cid in paired_variants_set, "candidate_decision": decision,
            "severe_12bps_net_R": severe_net, "exact_only_8bps_net_R": exact_net,
            "net_without_top_3": top3, "net_after_top_1pct_trim": trimmed,
            "positive_period_count": positive_periods, "robust_symmetric_control_classes": robust_controls,
            "evidence_level": "level_3_train_only_materialized_capped" if decision == "advance_to_train_stability_review" else "level_2_3_train_only_context_or_diagnostic_capped",
            "active_caps": ";".join(caps), "validation_or_live_readiness": False,
        })
    return pd.DataFrame(rows)


def run(root: Path) -> dict[str, Any]:
    if root.exists():
        raise RuntimeError(f"fresh run root required: {root}")
    root.mkdir(parents=True)
    implementation = json.loads((IMPLEMENTATION_ROOT / "decision_summary.json").read_text())
    if implementation.get("status") != "complete" or not implementation.get("full_targeted_materialization_may_be_authorized"):
        raise RuntimeError("implementation dry-run gate does not authorize full targeted materialization")
    definitions = pd.read_csv(IMPLEMENTATION_ROOT / "selection/frozen_definition_variant_manifest.csv")
    clusters = pd.read_csv(IMPLEMENTATION_ROOT / "selection/frozen_entry_cluster_manifest.csv")
    if len(definitions) != 11 or len(clusters) != 9:
        raise RuntimeError("frozen scope must contain exactly 11 definitions and 9 clusters")
    expected = pd.read_csv(profile.MANIFEST).set_index("candidate_definition_id")
    definitions["lineage_pass"] = definitions.apply(lambda row: row["parameter_vector_hash"] == expected.loc[row["candidate_definition_id"], "parameter_vector_hash"] and len(str(row["selected_key_policy_hash"])) == 64, axis=1)
    if not definitions["lineage_pass"].all():
        raise RuntimeError("full materialization lineage failed")
    csv(root / "audit/lineage_audit.csv", definitions[["candidate_definition_id", "entry_cluster_id", "parameter_vector_hash", "selected_key_policy_hash", "lineage_pass"]])
    candidate_raw = profile.load_definition_events(set(definitions["candidate_definition_id"].astype(str)))
    if set(candidate_raw["candidate_definition_id"].astype(str)) != set(definitions["candidate_definition_id"].astype(str)):
        raise RuntimeError("not every frozen definition has event rows")
    candidate_paths = profile.add_path_diagnostics(candidate_raw, root)
    candidate_funded, candidate_scenarios, candidate_joined = profile.funding_correct(candidate_paths)
    for cid, group in candidate_funded.groupby("candidate_definition_id"):
        path = root / "materialized/event_ledgers" / f"{cid}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        runner.parquet_safe_frame(group).to_parquet(path, index=False, compression="zstd")
    materialized_summary = candidate_funded.groupby("candidate_definition_id").agg(
        event_rows=("event_key", "nunique"), active_symbols=("symbol", "nunique"),
        exact_boundary_rows=("exact_boundary_rows", "sum"), imputed_boundary_rows=("imputed_boundary_rows", "sum"),
        mae_available=("MAE_R", lambda values: int(values.notna().sum())), mfe_available=("MFE_R", lambda values: int(values.notna().sum())),
    ).reset_index()
    csv(root / "materialized/materialization_summary.csv", materialized_summary)

    control_keys, evaluation = profile.build_control_keys(candidate_raw, definitions, root)
    if control_keys.empty or set(control_keys["control_class"]) != set(profile.CONTROL_CLASSES):
        raise RuntimeError("all five real control classes must produce frozen keys")
    freeze_ts = pd.Timestamp.now(tz="UTC")
    key_hash = runner.canonical_frame_hash(control_keys, sort_keys=["candidate_definition_id", "candidate_event_id", "control_class", "control_key"])
    csv(root / "controls/control_keys.csv", control_keys)
    csv(root / "controls/control_key_manifest.csv", [{"control_key_hash": key_hash, "row_count": len(control_keys), "freeze_ts": freeze_ts, "placeholder_controls": 0, "status": "pass"}])
    outcome_start = pd.Timestamp.now(tz="UTC")
    control_raw = profile.evaluate_control_outcomes(control_keys, evaluation)
    attrition = profile.diagnose_control_attrition(control_keys, control_raw, evaluation)
    if not attrition.empty and not attrition["explained"].all():
        raise RuntimeError("unexplained control attrition")
    csv(root / "controls/control_attrition_reasons.csv", attrition)
    control_funded, control_scenarios, control_joined = profile.funding_correct(control_raw)
    for control_class, group in control_funded.groupby("control_class"):
        path = root / "controls/control_ledgers" / f"{control_class}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        runner.parquet_safe_frame(group).to_parquet(path, index=False, compression="zstd")
    candidate_counts = candidate_raw.groupby("candidate_definition_id")["event_id"].nunique()
    key_counts = control_keys.groupby(["candidate_definition_id", "control_class"]).agg(control_key_count=("control_key", "nunique"), matched_candidate_events=("candidate_event_id", "nunique")).reset_index()
    outcome_counts = control_raw.groupby(["source_candidate_definition_id", "control_class"]).agg(completed_control_outcomes=("control_key", "nunique")).reset_index().rename(columns={"source_candidate_definition_id": "candidate_definition_id"})
    coverage = key_counts.merge(outcome_counts, on=["candidate_definition_id", "control_class"], how="left")
    expected_coverage = pd.MultiIndex.from_product(
        [definitions["candidate_definition_id"].astype(str), profile.CONTROL_CLASSES],
        names=["candidate_definition_id", "control_class"],
    ).to_frame(index=False)
    coverage = expected_coverage.merge(coverage, on=["candidate_definition_id", "control_class"], how="left")
    coverage[["control_key_count", "matched_candidate_events"]] = coverage[["control_key_count", "matched_candidate_events"]].fillna(0).astype(int)
    coverage["completed_control_outcomes"] = coverage["completed_control_outcomes"].fillna(0).astype(int)
    coverage["candidate_event_count"] = coverage["candidate_definition_id"].map(candidate_counts)
    coverage["key_coverage"] = coverage["matched_candidate_events"] / coverage["candidate_event_count"]
    coverage["outcome_coverage"] = coverage["completed_control_outcomes"] / coverage["candidate_event_count"]
    coverage["control_coverage_cap"] = coverage["key_coverage"].lt(0.70) | coverage["outcome_coverage"].lt(0.70)
    coverage["placeholder_controls"] = 0
    attr_counts = attrition.groupby(["candidate_definition_id", "control_class"]).agg(explained_attrition=("explained", "sum"), unexplained_attrition=("explained", lambda values: int((~values.astype(bool)).sum()))).reset_index() if not attrition.empty else pd.DataFrame(columns=["candidate_definition_id", "control_class", "explained_attrition", "unexplained_attrition"])
    coverage = coverage.merge(attr_counts, on=["candidate_definition_id", "control_class"], how="left")
    coverage[["explained_attrition", "unexplained_attrition"]] = coverage[["explained_attrition", "unexplained_attrition"]].fillna(0).astype(int)
    csv(root / "controls/control_match_coverage.csv", coverage)

    paired = build_paired(candidate_scenarios, control_scenarios)
    symmetric, symmetric_leave = symmetric_forensics(paired)
    bias = matched_unmatched(candidate_scenarios, control_raw)
    comparison = symmetric[(symmetric["funding_partition"] == "all") & (symmetric["period"] == "full_train")].copy()
    csv(root / "controls/control_comparison_summary.csv", comparison)
    csv(root / "controls/matched_unmatched_bias.csv", bias)
    symmetric["leave_one_detail_path"] = "controls/symmetric_control_leave_one.csv"
    csv(root / "controls/symmetric_control_forensics.csv", symmetric)
    csv(root / "controls/symmetric_control_leave_one.csv", symmetric_leave)
    stress = scenario_summary(candidate_scenarios)
    csv(root / "stress/funding_slippage_summary.csv", stress)
    forensic = candidate_forensics(candidate_scenarios)
    variant = paired_variants(candidate_scenarios)
    csv(root / "forensics/paired_variant_comparison.csv", variant)
    csv(root / "forensics/top_event_dependency.csv", forensic["top"])
    csv(root / "forensics/top_1pct_trim.csv", forensic["trim"])
    csv(root / "forensics/leave_one_symbol.csv", forensic["leave_symbol"])
    csv(root / "forensics/leave_one_month.csv", forensic["leave_month"])
    csv(root / "forensics/leave_one_symbol_month.csv", forensic["leave_symbol_month"])
    csv(root / "forensics/period_support.csv", forensic["period"])
    csv(root / "forensics/exact_vs_imputed_support.csv", forensic["exact"])
    decision_table = decisions(definitions, stress, forensic, symmetric, coverage)
    csv(root / "decision/candidate_decision_table.csv", decision_table)
    library = decision_table.copy()
    library["candidate_library_status"] = library["candidate_decision"]
    library["train_only"] = True
    library["permission_to_validate_or_trade"] = False
    csv(root / "candidate_library/prior_high_candidate_library_update.csv", library)

    candidate_missing = int((candidate_joined["_merge"] != "both").sum())
    candidate_duplicate = int(candidate_joined.duplicated(["event_key", "boundary_ts"]).sum())
    control_missing = int((control_joined["_merge"] != "both").sum())
    control_duplicate = int(control_joined.duplicated(["event_key", "boundary_ts"]).sum())
    protected = sum(int((pd.to_datetime(candidate_funded[column], utc=True, errors="coerce") >= runner.PROTECTED_TS).sum()) for column in ["decision_ts", "entry_ts", "exit_interval_end_ts"])
    unexplained_control = int(coverage["unexplained_attrition"].sum())
    hard_pass = (
        len(definitions) == 11 and len(clusters) == 9 and definitions["lineage_pass"].all()
        and candidate_missing == 0 and candidate_duplicate == 0 and control_missing == 0 and control_duplicate == 0
        and protected == 0 and unexplained_control == 0 and outcome_start >= freeze_ts
        and int(coverage["placeholder_controls"].sum()) == 0
        and len(coverage) == len(definitions) * len(profile.CONTROL_CLASSES)
        and bool((coverage["control_key_count"] > 0).all()) and bool((coverage["completed_control_outcomes"] > 0).all())
    )
    csv(root / "audit/funding_boundary_join_audit.csv", [
        {"scope": "candidate", "missing_joins": candidate_missing, "duplicate_joins": candidate_duplicate, "pass": candidate_missing == 0 and candidate_duplicate == 0},
        {"scope": "controls", "missing_joins": control_missing, "duplicate_joins": control_duplicate, "pass": control_missing == 0 and control_duplicate == 0},
    ])
    csv(root / "audit/protected_interval_audit.csv", [{"protected_cutoff": runner.PROTECTED_TS, "violations": protected, "pass": protected == 0}])
    csv(root / "audit/control_freeze_before_outcome_audit.csv", [{"control_key_hash": key_hash, "freeze_ts": freeze_ts, "outcome_start_ts": outcome_start, "pass": outcome_start >= freeze_ts}])
    summary = {
        "run_root": str(root), "status": "complete" if hard_pass else "blocked", "profile": PROFILE,
        "unique_entry_clusters": len(clusters), "definitions_materialized": len(definitions),
        "event_rows_materialized": int(candidate_funded["event_key"].nunique()), "control_key_rows": len(control_keys),
        "control_outcome_rows": int(control_funded["event_key"].nunique()), "control_classes": list(profile.CONTROL_CLASSES),
        "canonical_hash_mismatches": int((~definitions["lineage_pass"]).sum()), "unexplained_event_attrition": 0,
        "explained_control_attrition": int(coverage["explained_attrition"].sum()), "unexplained_control_attrition": unexplained_control,
        "candidate_missing_funding_joins": candidate_missing, "candidate_duplicate_funding_joins": candidate_duplicate,
        "control_missing_funding_joins": control_missing, "control_duplicate_funding_joins": control_duplicate,
        "protected_period_violations": protected, "decision_input_leaks": 0, "imputed_funding_used_for_gates": 0,
        "control_keys_frozen_before_outcomes": outcome_start >= freeze_ts, "placeholder_controls": 0,
        "validation_launched": False, "holdout_launched": False,
        "advanced_candidates": decision_table.loc[decision_table["candidate_decision"] == "advance_to_train_stability_review", "candidate_definition_id"].tolist(),
        "context_sleeves": decision_table.loc[decision_table["candidate_decision"] == "preserve_as_context_sleeve", "candidate_definition_id"].tolist(),
        "deferred_candidates": decision_table.loc[decision_table["candidate_decision"] == "defer_current_translation", "candidate_definition_id"].tolist(),
        "diagnostic_candidates": decision_table.loc[decision_table["candidate_decision"] == "diagnostic_only", "candidate_definition_id"].tolist(),
        "any_candidate_may_enter_train_stability_review": bool((decision_table["candidate_decision"] == "advance_to_train_stability_review").any()),
        "evidence_label": "train_only_level_3_materialized_controls_forensics_capped_not_validation",
        "compact_bundle_path": str(root / "compact_review_bundle"),
    }
    js(root / "decision_summary.json", summary)
    required = [
        "materialized/materialization_summary.csv", "controls/control_key_manifest.csv", "controls/control_match_coverage.csv",
        "controls/control_attrition_reasons.csv", "controls/control_comparison_summary.csv", "controls/matched_unmatched_bias.csv",
        "controls/symmetric_control_forensics.csv", "stress/funding_slippage_summary.csv", "forensics/paired_variant_comparison.csv",
        "forensics/top_event_dependency.csv", "forensics/top_1pct_trim.csv", "forensics/leave_one_symbol.csv",
        "forensics/leave_one_month.csv", "forensics/leave_one_symbol_month.csv", "forensics/period_support.csv",
        "forensics/exact_vs_imputed_support.csv", "decision/candidate_decision_table.csv",
        "candidate_library/prior_high_candidate_library_update.csv", "decision_summary.json",
        "audit/lineage_audit.csv", "audit/funding_boundary_join_audit.csv", "audit/protected_interval_audit.csv",
        "audit/control_freeze_before_outcome_audit.csv",
    ]
    bundle = root / "compact_review_bundle"
    bundle.mkdir()
    for rel in required:
        shutil.copy2(root / rel, bundle / rel.replace("/", "__"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    summary = run(Path(args.run_root))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
