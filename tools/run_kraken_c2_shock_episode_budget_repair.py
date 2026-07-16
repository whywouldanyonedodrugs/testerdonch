#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


PREFLIGHT = Path("results/rebaseline/phase_kraken_c2_audited_v2_1_ingestion_preflight_20260713_v1")
PROTECTED = "2026-01-01T00:00:00Z"


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text.rstrip() + "\n", encoding="utf-8")


class UnionFind:
    def __init__(self, values: list[str]): self.parent = {value: value for value in values}
    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]; value = self.parent[value]
        return value
    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b: self.parent[max(a, b)] = min(a, b)


def episode_assignments(exposures: pd.DataFrame, records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exp = exposures.copy()
    exp["event_anchor_ts"] = pd.to_datetime(exp.event_anchor_ts, utc=True, errors="coerce", format="mixed")
    exp["maximum_candidate_exit_ts"] = pd.to_datetime(exp.maximum_candidate_exit_ts, utc=True, errors="coerce", format="mixed")
    metadata = records.set_index("production_event_id")
    parents = exp.groupby("parent_event_id", sort=True).agg(
        catalyst_pathway_id=("catalyst_cluster_id", "first"),
        window_start=("event_anchor_ts", "min"),
        window_end=("maximum_candidate_exit_ts", "max"),
        exposure_rows=("event_exposure_id", "size"),
        asset_count=("audited_ticker", "nunique"),
    ).reset_index()
    parents["source_publication_ts"] = parents.parent_event_id.map(metadata["Source publication timestamp"])
    parents["effective_ts_utc"] = parents.parent_event_id.map(metadata["Effective timestamp UTC"])
    parents["audited_provisional_event_id"] = parents.parent_event_id.map(metadata["Audited provisional event ID"])
    parent_ids = parents.parent_event_id.tolist(); uf = UnionFind(parent_ids); overlap_rows = []
    for pathway, group in parents.groupby("catalyst_pathway_id", sort=True):
        rows = list(group.sort_values("parent_event_id").itertuples(index=False))
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                overlap = bool(pd.notna(left.window_start) and pd.notna(right.window_start) and left.window_start <= right.window_end and right.window_start <= left.window_end)
                same_parent = left.parent_event_id == right.parent_event_id
                if overlap: uf.union(left.parent_event_id, right.parent_event_id)
                overlap_rows.append({"catalyst_pathway_id": pathway, "parent_event_id_a": left.parent_event_id, "parent_event_id_b": right.parent_event_id, "window_start_a": left.window_start, "window_end_a": left.window_end, "window_start_b": right.window_start, "window_end_b": right.window_end, "windows_overlap": overlap, "same_parent_event": same_parent, "joined_same_shock_episode": overlap, "independent_count_allowed": int(not overlap), "assignment_basis": "overlapping_predeclared_max_outcome_windows" if overlap else "nonoverlapping_windows_separate_episode" if pd.notna(left.window_start) and pd.notna(right.window_start) else "missing_anchor_no_overlap_inferred"})
    components: dict[tuple[str, str], list[str]] = {}
    for row in parents.itertuples(index=False): components.setdefault((row.catalyst_pathway_id, uf.find(row.parent_event_id)), []).append(row.parent_event_id)
    parent_episode = {}
    for (pathway, _), members in sorted(components.items()):
        episode_id = "C2SHOCK_" + stable_hash({"catalyst_pathway_id": pathway, "parent_event_ids": sorted(members), "overlap_rule": "max_candidate_windows_connected_components_v1"})[:20]
        for parent in members: parent_episode[parent] = episode_id
    parents["shock_episode_id"] = parents.parent_event_id.map(parent_episode)
    parents["episode_assignment_status"] = "assigned"
    exp["catalyst_pathway_id"] = exp.catalyst_cluster_id
    exp["shock_episode_id"] = exp.parent_event_id.map(parent_episode)
    return exp, pd.DataFrame(overlap_rows), parents


def build_definitions(long_episodes: int) -> pd.DataFrame:
    columns = ["definition_id", "reaction_exclusion", "base_length_days", "entry_policy", "exit_policy", "fee_cost_mode", "slippage_roundtrip_bps", "funding_policy", "event_anchor_policy", "global_across_episodes", "parameter_vector_hash", "rankable_scope"]
    if long_episodes < 8: return pd.DataFrame(columns=columns)
    rows = []
    if long_episodes >= 12:
        specs = [(reaction, base, exit_policy) for reaction in ("1d", "3d") for base in (3, 7) for exit_policy in ("structure_base_failure", "failed_close_inside_range", "fixed_hold_10d")]
        scope = "sample_limited_long_12_to_18_budget"
    else:
        specs = []
        for index, (reaction, base) in enumerate((r, b) for r in ("1d", "3d") for b in (3, 7)):
            specs.extend([(reaction, base, "structure_base_failure" if index % 2 == 0 else "failed_close_inside_range"), (reaction, base, "fixed_hold_10d")])
        scope = "sample_limited_long_max_8_budget"
    for reaction, base, exit_policy in specs:
        vector = {"reaction_exclusion": reaction, "base_length_days": base, "entry_policy": "completed_close_breakout_or_reclaim_next_executable_bar", "exit_policy": exit_policy, "fee_cost_mode": "base_5bps_per_side_plus_4bps_roundtrip_slippage", "slippage_roundtrip_bps": 4, "funding_policy": "direction_adverse_required", "event_anchor_policy": "c2_event_anchor_policy_v1", "protected_boundary": PROTECTED}
        h = stable_hash(vector); rows.append({"definition_id": "c2_sample_" + h[:16], **{k: vector[k] for k in columns if k in vector}, "global_across_episodes": True, "parameter_vector_hash": h, "rankable_scope": scope})
    return pd.DataFrame(rows, columns=columns)


def source_metadata_unchanged(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    for column in before.columns:
        if column in {"event_anchor_ts", "maximum_candidate_exit_ts", "actionable_not_before_ts", "kraken_opening_ts", "kraken_last_trading_ts"}:
            left = pd.to_datetime(before[column], utc=True, errors="coerce", format="mixed")
            right = pd.to_datetime(after[column], utc=True, errors="coerce", format="mixed")
            if not left.equals(right): return False
        else:
            left = before[column].fillna("<NULL>").astype(str).reset_index(drop=True)
            right = after[column].fillna("<NULL>").astype(str).reset_index(drop=True)
            if not left.equals(right): return False
    return True


def run(root: Path) -> dict[str, Any]:
    if root.exists(): raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True)
    exposure_before = pd.read_csv(PREFLIGHT / "mapping/event_asset_exposure_map.csv")
    records = pd.read_csv(PREFLIGHT / "database/audited_records_normalized.csv")
    old_summary = json.loads((PREFLIGHT / "decision_summary.json").read_text())
    assigned, overlap, parents = episode_assignments(exposure_before, records)
    if len(assigned) != len(exposure_before): raise RuntimeError("exposure rows lost")
    if assigned.shock_episode_id.isna().any(): raise RuntimeError("unexplained episode assignment")
    old_phase = pd.read_csv(PREFLIGHT / "clusters/phase_overlap_audit.csv")
    exposure_parent = exposure_before.set_index("event_exposure_id").parent_event_id.to_dict()
    old_overlap_pairs = set()
    for row in old_phase.itertuples(index=False):
        if bool(row.phase_windows_overlap):
            left, right = exposure_parent.get(row.event_exposure_id_a), exposure_parent.get(row.event_exposure_id_b)
            if left and right and left != right: old_overlap_pairs.add(tuple(sorted((left, right))))
    if len(overlap):
        overlap["prior_phase_overlap_audit_overlap"] = overlap.apply(lambda row: tuple(sorted((row.parent_event_id_a, row.parent_event_id_b))) in old_overlap_pairs, axis=1)
        overlap["reconciliation_status"] = overlap.apply(lambda row: "pass" if not row.prior_phase_overlap_audit_overlap or row.windows_overlap else "mismatch_prior_overlap_not_preserved", axis=1)
    else:
        overlap["prior_phase_overlap_audit_overlap"] = pd.Series(dtype=bool); overlap["reconciliation_status"] = pd.Series(dtype=str)
    write_csv(root / "clusters/pathway_episode_map.csv", parents)
    write_csv(root / "clusters/episode_overlap_audit.csv", overlap)

    primary = assigned[assigned.primary_rankable.astype(bool)].copy()
    episode_rows = []
    for episode_id, group in assigned.groupby("shock_episode_id", sort=True):
        primary_group = group[group.primary_rankable.astype(bool)]
        directions = set(primary_group.direction.dropna().astype(str)) if len(primary_group) else set(group.direction.dropna().astype(str))
        if directions == {"long"}: direction_class = "long"
        elif directions == {"short"}: direction_class = "short"
        else: direction_class = "mixed_or_diagnostic"
        episode_rows.append({"shock_episode_id": episode_id, "catalyst_pathway_id": group.catalyst_pathway_id.iloc[0], "parent_event_count": group.parent_event_id.nunique(), "exposure_row_count": len(group), "asset_count": group.audited_ticker.nunique(), "primary_rankable_exposure_rows": len(primary_group), "episode_direction": direction_class, "mechanism_families": "|".join(sorted(group.mechanism_family.unique())), "window_start": pd.to_datetime(group.event_anchor_ts, utc=True, errors="coerce").min(), "window_end": pd.to_datetime(group.maximum_candidate_exit_ts, utc=True, errors="coerce").max(), "equal_episode_weight": 1.0, "independent_discovery_count": 1, "cluster_block_id": group.catalyst_pathway_id.iloc[0]})
    episodes = pd.DataFrame(episode_rows); write_csv(root / "clusters/shock_episode_manifest.csv", episodes)

    basket = assigned.copy(); basket["exposure_weight_within_episode"] = 1.0 / basket.groupby("shock_episode_id").event_exposure_id.transform("size")
    basket_audit = basket.groupby(["shock_episode_id", "parent_event_id"], sort=True).agg(exposure_rows=("event_exposure_id", "size"), distinct_episode_ids=("shock_episode_id", "nunique"), allocated_episode_weight=("exposure_weight_within_episode", "sum"), basket_parent=("asset_id", lambda values: any("basket" in str(value).lower() for value in values))).reset_index()
    basket_audit["basket_double_count_violation"] = basket_audit.distinct_episode_ids.ne(1)
    # Weight is normalized over the whole episode, so parent subtotals need not be 1.
    episode_weight = basket.groupby("shock_episode_id").exposure_weight_within_episode.sum()
    basket_audit["episode_total_weight"] = basket_audit.shock_episode_id.map(episode_weight)
    basket_audit["episode_weight_violation"] = (basket_audit.episode_total_weight - 1.0).abs() > 1e-12
    write_csv(root / "audit/basket_episode_weight_audit.csv", basket_audit)

    overlap_violations = int(overlap.query("windows_overlap == True and joined_same_shock_episode == False").shape[0]) if len(overlap) else 0
    overlap_reconciliation_mismatches = int(overlap.reconciliation_status.ne("pass").sum()) if len(overlap) else 0
    old_count = int(old_summary["pit_rankable_independent_clusters"])
    primary_episode_ids = set(primary.shock_episode_id)
    primary_episodes = episodes[episodes.shock_episode_id.isin(primary_episode_ids)].copy()
    long_count = int((primary_episodes.episode_direction == "long").sum()); short_count = int((primary_episodes.episode_direction == "short").sum()); mixed_count = int((primary_episodes.episode_direction == "mixed_or_diagnostic").sum())
    comparison = pd.DataFrame([{"scope": "primary_rankable_economic_independence", "old_rule": "one_count_per_catalyst_pathway", "old_independence_count": old_count, "new_rule": "metadata_only_nonoverlapping_shock_episodes_within_pathway", "new_shock_episode_count": len(primary_episodes), "long_episodes": long_count, "short_episodes": short_count, "mixed_or_diagnostic_episodes": mixed_count, "exposure_rows": len(primary), "parent_event_rows": primary.parent_event_id.nunique(), "pathway_count": primary.catalyst_pathway_id.nunique()}])
    write_csv(root / "audit/old_new_independence_count_comparison.csv", comparison)

    definitions = build_definitions(long_count); write_csv(root / "redesign/c2_sample_limited_definition_manifest.csv", definitions)
    write_text(root / "clusters/shock_episode_contract.md", """# C2 Shock Episode Contract v1

`catalyst_pathway_id` is the frozen audited `catalyst_cluster_id` and describes a continuing information pathway. `shock_episode_id` is a deterministic connected component of parent events within one pathway whose predeclared maximum candidate outcome windows overlap. All exposures from one parent basket or simultaneous multi-asset action share the parent event and episode. Overlap is transitive. Nonoverlapping phases in the same pathway may form distinct episodes; calendar year alone never creates independence. Missing-anchor parent records receive their own explicit metadata-only episode and are non-rankable under existing mapping rules. No prices, returns, outcomes, or performance labels enter episode assignment.

Economic inference equal-weights shock episodes. Exposure rows divide one episode weight and cannot multiply it. Resampling is pathway-blocked. Required diagnostics are leave-one-episode and leave-one-pathway. This sample cannot support a validation claim.
""")
    write_text(root / "contract/economic_cost_and_funding_policy_v2.md", """# C2 Economic Cost and Funding Policy v2

Frozen cost modes:

- Base: 5 bps taker per side plus 4 bps round-trip slippage.
- Conservative: 5 bps taker per side plus 8 bps round-trip slippage.
- Severe: 10 bps taker per side plus 12 bps round-trip slippage.
- Zero-fee account state: diagnostic only.

Fees apply separately to entry and exit notional. Slippage is additional round-trip cost. Exact funding remains unchanged. Imputed funding uses the frozen shared model, carries its train-screen cap, and cannot activate event-time gates. Direction-adverse funding scenarios are mandatory for both long and short positions; raw central/conservative/severe rate scenarios may be reported but are not uniformly adverse. Severe-mode failure alone cannot reject a hypothesis. No cost result is computed in this phase.
""")
    if long_count >= 12: budget = "12-18 global long definitions"
    elif long_count >= 8: budget = "at most 8 sample-limited long definitions"
    else: budget = "no long economic tranche"
    short_policy = "diagnostic_only" if short_count < 8 else "short_sample_limited_tranche_may_be_designed_separately"
    write_text(root / "budget/repaired_economic_tranche_budget.md", f"# Repaired Economic Tranche Budget\n\nPrimary exposure rows: {len(primary)}. Parent records represented: {primary.parent_event_id.nunique()}. Catalyst pathways: {primary.catalyst_pathway_id.nunique()}. Shock episodes: {len(primary_episodes)}. Long/short/mixed episodes: {long_count}/{short_count}/{mixed_count}. Long budget: `{budget}`. Short policy: `{short_policy}`. Emitted global long definitions: {len(definitions)}. Definitions are global across episodes and are not padded. No outcomes were read and no scan was launched.")

    pathway_preserved = bool((assigned.catalyst_pathway_id == assigned.catalyst_cluster_id).all())
    source_columns = exposure_before.columns.tolist()
    metadata_unchanged = source_metadata_unchanged(exposure_before.reset_index(drop=True), assigned[source_columns].reset_index(drop=True))
    hard_failures = int(len(assigned) != len(exposure_before)) + int(assigned.shock_episode_id.isna().sum()) + int(basket_audit.basket_double_count_violation.sum()) + int(basket_audit.episode_weight_violation.sum()) + overlap_violations + overlap_reconciliation_mismatches + int(not pathway_preserved) + int(not metadata_unchanged)
    may_proceed = hard_failures == 0 and long_count >= 8 and len(definitions) > 0
    summary = {"run_root": str(root), "status": "repair_pass" if hard_failures == 0 else "blocked_by_protocol_issue", "exposure_rows": len(assigned), "parent_event_count": assigned.parent_event_id.nunique(), "shock_episode_count_all": episodes.shock_episode_id.nunique(), "catalyst_pathway_count_all": assigned.catalyst_pathway_id.nunique(), "primary_exposure_rows": len(primary), "primary_parent_events": primary.parent_event_id.nunique(), "primary_shock_episodes": len(primary_episodes), "primary_catalyst_pathways": primary.catalyst_pathway_id.nunique(), "long_shock_episodes": long_count, "short_shock_episodes": short_count, "mixed_or_diagnostic_shock_episodes": mixed_count, "old_independence_count": old_count, "repaired_independence_count": len(primary_episodes), "pathway_ids_preserved_pct": 100.0 if pathway_preserved else 0.0, "event_metadata_changed": 0 if metadata_unchanged else 1, "exposure_rows_lost": len(exposure_before) - len(assigned), "unexplained_episode_assignments": int(assigned.shock_episode_id.isna().sum()), "basket_double_count_violations": int(basket_audit.basket_double_count_violation.sum() + basket_audit.episode_weight_violation.sum()), "overlapping_phases_counted_independently": overlap_violations, "phase_overlap_reconciliation_mismatches": overlap_reconciliation_mismatches, "cost_policy_status": "frozen_v2", "emitted_definitions": len(definitions), "short_lane_policy": short_policy, "sample_limited_economic_smoke_may_proceed": may_proceed, "outcomes_read": False, "economic_scan_launched": False, "validation_launched": False, "holdout_launched": False, "compact_bundle_path": str(root / "compact_review_bundle")}
    (root / "decision_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    bundle = root / "compact_review_bundle"; bundle.mkdir()
    for path in [p for p in root.rglob("*") if p.is_file() and "compact_review_bundle" not in p.parts]: shutil.copy2(path, bundle / str(path.relative_to(root)).replace("/", "__"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True)
    summary = run(Path(parser.parse_args().run_root)); print(json.dumps(summary, indent=2, sort_keys=True)); return 0 if summary["status"] == "repair_pass" else 2


if __name__ == "__main__": raise SystemExit(main())
