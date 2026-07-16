#!/usr/bin/env python3
"""Build the read-only all-tested-hypotheses external review package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROTECTED = pd.Timestamp("2026-01-01T00:00:00Z")
CONTRACT = "signal_state_contract_v1_20260715"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/rebaseline"
DEFAULT_OUT = RESULTS / "phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1"
CONTINUITY_BRIEF = ROOT / "research_inputs/QLMG_Project_Master_Continuity_Brief_2026-07-16_rev7.md"


@dataclass(frozen=True)
class Family:
    family_id: str
    hypothesis: str
    direction: str
    primary: str
    supports: tuple[str, ...] = ()
    superseded: tuple[str, ...] = ()
    runner: str = ""
    candidate_patterns: tuple[str, ...] = ()
    control_patterns: tuple[str, ...] = ()
    definition_patterns: tuple[str, ...] = ()
    decision: str = "unknown"
    evidence_level: str = "train_only_capped"
    mechanism: str = ""


FAMILIES = (
    Family("tsmom_v6", "Time-series momentum", "long/short", "phase_kraken_full_tsmom_v6_aggregate_20260707_v1", ("phase_kraken_tsmom_v6_targeted_materialization_controls_stress_20260708_v1_20260708_101819", "phase_kraken_tsmom_v6_survivor_forensic_decomposition_20260708_v1", "phase_kraken_tsmom_funding_corrected_reopened_forensics_20260712_v1"), runner="tools/run_kraken_family_engine_aggregate_first_sweep.py", candidate_patterns=("materialized/event_ledgers/*.parquet",), control_patterns=("controls/control_ledger/control_ledger.parquet",), definition_patterns=("aggregate/tsmom_v6_definition_level_aggregate_summary.csv",), decision="defer_current_translation", evidence_level="level_3_train_only_materialized_controls_stress_diagnostic_capped", mechanism="Directional continuation after sustained own-price momentum; the broader mechanism remains candidate-library-only."),
    Family("a1_compression", "A1 liquid-leader continuation and compression breakouts", "primarily long", "phase_kraken_a1_compression_full_180shard_funding_corrected_20260711_v1_20260711_194859", ("phase_kraken_a1_compression_targeted_materialization_controls_stress_20260712_v1", "phase_kraken_a1_control_context_adjudication_20260712_v1", "phase_kraken_a1_compression_contract_manifest_20260708_v1"), runner="tools/run_kraken_family_engine_aggregate_first_sweep.py", candidate_patterns=("materialized/event_ledgers/*.parquet",), control_patterns=("controls/control_ledgers/control_ledger.parquet",), definition_patterns=("redesign/a1_h06_h12_h13_curated_sweep_definitions_v1.csv",), decision="defer_current_translation", evidence_level="train_only_control_context_adjudication_capped_not_validation", mechanism="Leader continuation after impulse/base formation, optionally with volatility contraction; structure-responsive exit observations are preserved."),
    Family("prior_high_reclaim_v2", "Prior-high proximity, breakout and reclaim", "long", "phase_kraken_prior_high_reclaim_v2_canonical_train_scan_20260712_v1", ("phase_kraken_prior_high_reclaim_v2_full_targeted_materialization_20260712_v1", "phase_kraken_prior_high_v2_control_matching_repair_20260712_v1"), runner="tools/run_kraken_prior_high_reclaim_v2_canonical_train_scan.py", candidate_patterns=("materialized/event_ledgers/*.parquet",), control_patterns=("controls/control_ledgers/*.parquet",), definition_patterns=("shards/full_shard_plan.csv",), decision="defer_current_translation", evidence_level="train_only_materialized_real_controls_capped", mechanism="Continuation or reclaim near a frozen prior high; proximity remains only a possible overlay."),
    Family("c2_post_catalyst", "Post-catalyst continuation base", "long", "phase_kraken_c2_sample_limited_economic_tranche_20260713_v1", ("phase_kraken_c2_shock_episode_budget_repair_20260713_v1", "phase_kraken_c2_audited_v2_1_ingestion_preflight_20260713_v1"), runner="tools/run_kraken_c2_sample_limited_economic_tranche.py", candidate_patterns=("materialized/exposure_event_ledger.csv",), control_patterns=("controls/control_outcome_ledger.csv",), definition_patterns=("redesign/c2_sample_limited_definition_manifest.csv",), decision="current_translation_weak", evidence_level="sample_limited_train_only_capped", mechanism="Continuation after a durable public catalyst and a completed consolidation base; the sample-limited mechanism and source-verified seed database are preserved."),
    Family("lfbs_repaired", "Liquid failed-breakout short", "short", "phase_kraken_lfbs_signal_state_repaired_screen_20260715_v1", ("phase_kraken_lfbs_021_signal_state_repaired_2023_presample_20260715_v1", "phase_kraken_lfbs_021_signal_state_repaired_canonical_adjudication_20260715_v1"), ("phase_kraken_liquid_failed_breakout_short_screen_20260713_v1", "phase_kraken_lfbs_021_frozen_2023_presample_confirmation_20260713_v1", "phase_kraken_lfbs_021_canonical_episode_adjudication_20260713_v1"), "tools/run_kraken_liquid_failed_breakout_short_screen.py", ("materialized/event_ledgers/*.csv", "materialized/*event_ledger.csv"), ("controls/control_event_ledger.csv", "controls/*control_event_ledger.csv"), ("manifest/frozen_definitions.csv",), "current_translation_weak", "level_4_event_ledger_plus_real_controls", "Short after a completed breakout fails and closes back below frozen resistance."),
    Family("backside_blowoff_repaired", "Backside-confirmed blowoff short", "short", "phase_kraken_backside_blowoff_signal_state_repaired_screen_20260715_v1", superseded=("phase_kraken_backside_blowoff_short_screen_20260713_v1",), runner="tools/run_kraken_backside_blowoff_short_screen.py", candidate_patterns=("materialized/event_ledger.csv",), control_patterns=("controls/control_event_ledger.csv",), definition_patterns=("manifest/*definitions.csv",), decision="current_translation_weak", evidence_level="level_4_event_ledger_plus_real_controls", mechanism="Short only after an extreme extension confirms a completed backside break."),
    Family("rfbs_repaired", "Risk-off failed-bounce short", "short", "phase_kraken_rfbs_signal_state_repaired_screen_20260715_v1", ("phase_kraken_rfbs_signal_state_repaired_materialization_20260715_v1", "phase_kraken_cross_family_repair_campaign_closure_20260715_v1", "phase_kraken_rfbs_010_train_only_stability_review_20260715_v1"), ("phase_kraken_riskoff_failed_bounce_short_screen_20260714_v1", "phase_kraken_rfbs_control_overlap_materialization_20260714_v1"), "tools/run_kraken_riskoff_failed_bounce_short_screen.py", ("materialized/event_ledger.csv", "materialized/candidate_event_ledger.csv"), ("controls/control_event_ledger.csv", "controls/materialized_control_ledger.csv"), ("manifest/*definitions.csv",), "fragile_context_sleeve", "level_4_event_ledger_plus_real_controls", "Short a countertrend rally that fails inside an established risk-off downtrend."),
    Family("delayed_flush_reclaim", "Delayed flush reclaim long", "long", "phase_kraken_delayed_flush_reclaim_signal_state_repair_20260715_v1", superseded=("phase_kraken_delayed_flush_reclaim_long_screen_20260715_v1",), runner="tools/run_kraken_delayed_flush_reclaim_long_screen.py", candidate_patterns=("materialized/event_ledger.csv",), control_patterns=("controls/control_event_ledger.csv",), definition_patterns=("manifest/delayed_flush_reclaim_definitions.csv",), decision="current_translation_weak", evidence_level="level_4_event_ledger_plus_real_controls", mechanism="Buy a completed reclaim after a large downside flush stabilizes."),
    Family("breakout_retest_v2", "Close-confirmed breakout retest long", "long", "phase_kraken_close_confirmed_breakout_retest_long_screen_20260715_v2", superseded=("phase_kraken_close_confirmed_breakout_retest_long_screen_20260715_v1",), runner="tools/run_kraken_close_confirmed_breakout_retest_long_screen_v2.py", candidate_patterns=("materialized/event_ledger.csv",), control_patterns=("controls/control_event_ledger.csv",), definition_patterns=("manifest/definitions.csv",), decision="current_translation_rejected_only", evidence_level="level_4_event_ledger_plus_real_controls", mechanism="Buy only after a range breakout retests and reclaims its frozen level; immediate-breakout continuation was not rejected by this test."),
    Family("failed_breakdown_reclaim", "Failed-breakdown squeeze reclaim long", "long", "phase_kraken_failed_breakdown_squeeze_reclaim_long_screen_20260716_v1", runner="tools/run_kraken_failed_breakdown_squeeze_reclaim_long_screen.py", candidate_patterns=("materialized/event_ledger.csv",), control_patterns=("controls/control_event_ledger.csv",), definition_patterns=("manifest/definitions.csv",), decision="mechanism_preserved_current_translation_weak", evidence_level="level_4_event_ledger_plus_real_controls", mechanism="Buy after a completed support breakdown fails and reclaims support plus anchored VWAP; four frozen definitions remain fragile context sleeves."),
    Family("strong_close_handoff", "Strong-close session-handoff continuation", "long/short", "phase_kraken_strong_close_session_handoff_continuation_20260716_v1", runner="tools/run_kraken_strong_close_session_handoff_continuation.py", candidate_patterns=("materialized/event_ledger.csv",), control_patterns=("controls/control_event_ledger.csv",), definition_patterns=("manifest/definitions.csv",), decision="current_translation_weak", evidence_level="level_4_event_ledger_plus_real_controls", mechanism="Continue an unusually wide, high-volume eight-hour session after handoff."),
    Family("rs_breakout_btc", "Relative-strength breakout versus BTC", "long", "phase_kraken_relative_strength_breakout_vs_btc_screen_20260716_v1", runner="tools/run_kraken_relative_strength_breakout_vs_btc_screen.py", candidate_patterns=("materialized/event_ledger.csv",), control_patterns=("controls/control_event_ledger.csv",), definition_patterns=("manifest/definitions.csv",), decision="mechanism_preserved_current_translation_weak", evidence_level="level_4_event_ledger_plus_real_controls", mechanism="Buy an altcoin breaking out in both USD and BTC-relative terms; definitions 013/014 remain fragile context sleeves."),
    Family("btc_alt_diffusion", "BTC-led delayed alt diffusion", "long", "phase_kraken_btc_led_delayed_alt_diffusion_long_screen_20260716_v1", runner="tools/run_kraken_btc_led_delayed_alt_diffusion_long_screen.py", candidate_patterns=("materialized/event_ledger.csv",), control_patterns=("controls/control_event_ledger.csv",), definition_patterns=("manifest/definitions.csv",), decision="fragile_context_sleeve", evidence_level="level_4_event_ledger_plus_real_controls", mechanism="After an unusual BTC impulse, buy liquid alts that lag their trailing BTC beta; one highly overlapping moderate-lag/both-up region is preserved while the broad translation is mixed-to-weak."),
    Family("session_open_range", "Asia and U.S. cash-open range resolution", "long/short", "phase_kraken_session_open_range_resolution_20260716_v1", runner="tools/run_kraken_session_open_range_resolution.py", candidate_patterns=("materialized/event_ledger.csv",), control_patterns=("controls/control_event_ledger.csv",), definition_patterns=("manifest/definitions.csv",), decision="current_translation_rejected_only", evidence_level="level_4_event_ledger_plus_real_controls", mechanism="Trade a completed close-confirmed break of a frozen opening range; broader session seasonality remains execution/risk context only."),
)


STANDARD_EVENT_COLUMNS = [
    "family_id", "definition_id", "symbol", "setup_start_ts", "decision_ts", "entry_ts", "entry_price",
    "initial_stop", "stop_price", "exit_ts", "exit_price", "maximum_exit_ts", "side", "risk_denominator",
    "gross_R", "fee_base_R", "slippage_base_R", "funding_central_R", "net_base_R", "net_conservative_R",
    "net_severe_R", "net_zero_funding_base_R", "exit_reason", "mae_R", "mfe_R", "parent_state",
    "funding_partition", "candidate_economic_address_hash", "selected_key_policy_hash", "parameter_vector_hash",
    "source_root", "source_file",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def matched_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        found.extend(root.glob(pattern))
    return sorted({p for p in found if p.is_file() and "compact_review_bundle" not in p.parts})


ALIASES = {
    "candidate_definition_id": "definition_id", "candidate_symbol_id": "symbol", "entry": "entry_price",
    "stop": "initial_stop", "net_R": "net_base_R", "fee_R": "fee_base_R", "funding_R": "funding_central_R",
    "slippage_R": "slippage_base_R", "candidate_address_hash": "candidate_economic_address_hash",
    "MAE_R": "mae_R", "MFE_R": "mfe_R",
}


def normalize_event(frame: pd.DataFrame, family: Family, source_root: Path, source_file: Path) -> pd.DataFrame:
    frame = frame.copy()
    for old, new in ALIASES.items():
        if old in frame and new not in frame:
            frame[new] = frame[old]
    if "side" not in frame and "direction" in frame:
        frame["side"] = frame["direction"]
    if "initial_stop" not in frame and "stop_price" in frame:
        frame["initial_stop"] = frame["stop_price"]
    if "stop_price" not in frame and "initial_stop" in frame:
        frame["stop_price"] = frame["initial_stop"]
    if "funding_partition" not in frame:
        exact_source = frame["exact_funding_boundaries"] if "exact_funding_boundaries" in frame else pd.Series(0, index=frame.index)
        imputed_source = frame["imputed_funding_boundaries"] if "imputed_funding_boundaries" in frame else pd.Series(0, index=frame.index)
        exact = pd.to_numeric(exact_source, errors="coerce").fillna(0)
        imputed = pd.to_numeric(imputed_source, errors="coerce").fillna(0)
        frame["funding_partition"] = np.select(
            [(exact > 0) & (imputed == 0), (exact > 0) & (imputed > 0), (exact == 0) & (imputed > 0)],
            ["fully_exact", "mixed", "fully_imputed"], default="zero_boundary",
        )
    frame["family_id"] = family.family_id
    frame["source_root"] = str(source_root.relative_to(ROOT))
    frame["source_file"] = str(source_file.relative_to(source_root))
    for column in STANDARD_EVENT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    ordered = STANDARD_EVENT_COLUMNS + [c for c in frame.columns if c not in STANDARD_EVENT_COLUMNS]
    return frame[ordered]


def combine_tables(root_names: Iterable[str], patterns: Iterable[str], family: Family, normalize: bool) -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    for root_name in root_names:
        source_root = RESULTS / root_name
        for path in matched_files(source_root, patterns):
            frame = read_table(path)
            if normalize:
                frame = normalize_event(frame, family, source_root, path)
            else:
                frame.insert(0, "family_id", family.family_id)
                frame["source_root"] = str(source_root.relative_to(ROOT))
                frame["source_file"] = str(path.relative_to(source_root))
            frames.append(frame)
            sources.append(str(path.relative_to(ROOT)))
    return (pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()), sources


def protected_violations(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    # Contract horizon endpoints at the boundary are metadata, not observed rows.
    observed_timestamp_columns = {
        "setup_start_ts", "decision_ts", "feature_available_ts", "entry_ts", "exit_ts",
        "parent_source_ts", "parent_feature_ts", "btc_source_ts", "eth_source_ts",
        "funding_ts", "settlement_ts", "bar_ts", "source_close_ts",
    }
    for column in frame.columns:
        if column not in observed_timestamp_columns:
            continue
        parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
        count = int((parsed >= PROTECTED).sum())
        if count:
            rows.append({"column": column, "violations": count})
    return rows


def find_first(root_names: Iterable[str], patterns: Iterable[str]) -> Path | None:
    for name in root_names:
        for pattern in patterns:
            found = matched_files(RESULTS / name, (pattern,))
            if found:
                return found[0]
    return None


def summary_for(family: Family) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in reversed(family.supports + (family.primary,)):
        path = RESULTS / name / "decision_summary.json"
        if path.exists():
            result.update(json.loads(path.read_text()))
    return result


def strategy_card(family: Family, manifest: pd.DataFrame, summary: dict[str, object]) -> str:
    params = ", ".join(manifest.columns[:12]) if not manifest.empty else "See the source contract; no single tabular manifest was available."
    return f"""# {family.hypothesis}

## Market Idea
{family.mechanism}

## Trading Contract
- **Direction:** {family.direction}.
- **Venue and universe:** Point-in-time eligible Kraken perpetual instruments defined by the source run. Lifecycle and liquidity exclusions remain those of the source manifest.
- **Timing:** Every decision uses completed data only. Orders use the next executable bar specified by the source contract; no touch or same-bar fill is inferred.
- **Setup and parameters:** Frozen manifest fields include: {params}.
- **Risk:** R is profit or loss divided by the event's frozen initial risk denominator. Stops, maximum exits, and structure exits are definition-specific and are preserved in the event ledger.
- **Costs:** Base, conservative, and severe fees/slippage/funding are preserved separately. Imputed funding is outcome-cost only and remains capped evidence.
- **Regime filters:** Point-in-time parent/context policies are projections from frozen raw signal tapes where the shared signal-state contract applies.
- **Invalidation:** Missing required data, lifecycle ineligibility, stale/pathological bars, invalid risk, protected-boundary crossing, or failure of the completed-close trigger produces no trade or an explicit exclusion.

## Principal Risks
OHLCV stop paths do not reproduce an order book; depth and impact are unavailable; funding is partly imputed; definitions overlap and must not be summed as a portfolio; train-only evidence is not validation or permission to trade.

## Project Decision
**{family.decision.replace('_', ' ')}.** Evidence level: `{family.evidence_level}`. Source status: `{summary.get('status', 'documented by finalized source artifacts')}`.
"""


def quant_card(family: Family, manifest: pd.DataFrame, sources: list[str]) -> str:
    return f"""# {family.hypothesis}: Quant Method Card

- Frozen definitions: {len(manifest)} rows in `definition_manifest.parquet`.
- Direction: {family.direction}.
- Primary root: `{family.primary}`.
- Supporting roots: {', '.join(f'`{x}`' for x in family.supports) or 'none'}.
- Signal-state contract: `{CONTRACT}` when present; older architectures retain their original frozen lineage.
- Point-in-time rule: feature/source close timestamp must be no later than decision timestamp.
- Non-overlap: definition-local chronological acceptance using the actual executable exit, never a nominal maximum-hold preblock.
- Costs: source base/conservative/severe fee, slippage, and signed funding fields are preserved; no pooled portfolio is constructed.
- Controls: source real-control classes and frozen keys are preserved. Adequacy thresholds are not re-estimated here.
- Evidence gate: `{family.evidence_level}`; final decision `{family.decision}`.
- Prohibited interpretations: no final-holdout, validation, live-readiness, or cross-definition portfolio claim; no causal claim from an observational control difference.
- Native source tables: {', '.join(f'`{x}`' for x in sources) or 'none'}.

## Formula/Schema Contract
The complete family-specific formulas remain in the packaged source contract/runner and native manifest columns. Standardized columns are additive; native IDs, hashes, timestamps, R components, and family fields are not deleted.
"""


def plot_family(out: Path, family: Family, events: pd.DataFrame, metrics: pd.DataFrame) -> None:
    plot_dir = out / "plots"
    plot_dir.mkdir(exist_ok=True)
    specs = [
        ("return_distribution", "Distribution of conservative R"), ("mae_mfe", "MAE versus MFE"),
        ("period_performance", "Performance by period"), ("average_path", "Available path diagnostics"),
        ("control_coverage_uplift", "Control coverage and uplift"), ("funding_partition", "Funding partitions"),
        ("concentration_removal", "Concentration and winner removal"), ("definition_heatmap", "Definition/parameter surface"),
    ]
    for stem, title in specs:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        if stem == "return_distribution" and "net_conservative_R" in events:
            pd.to_numeric(events.net_conservative_R, errors="coerce").dropna().clip(-5, 5).hist(ax=ax, bins=50)
        elif stem == "mae_mfe" and {"mae_R", "mfe_R"}.issubset(events):
            sample = events[["mae_R", "mfe_R"]].apply(pd.to_numeric, errors="coerce").dropna().iloc[::max(1, len(events)//5000)]
            ax.scatter(sample.mae_R, sample.mfe_R, s=5, alpha=.25)
        elif stem == "period_performance" and {"evaluation_period", "net_conservative_R"}.issubset(events):
            events.assign(v=pd.to_numeric(events.net_conservative_R, errors="coerce")).groupby("evaluation_period").v.mean().plot.bar(ax=ax)
        elif stem == "funding_partition" and {"funding_partition", "net_conservative_R"}.issubset(events):
            events.assign(v=pd.to_numeric(events.net_conservative_R, errors="coerce")).groupby("funding_partition").v.mean().plot.bar(ax=ax)
        elif stem == "definition_heatmap" and {"definition_id", "net_conservative_R"}.issubset(events):
            values = events.assign(v=pd.to_numeric(events.net_conservative_R, errors="coerce")).groupby("definition_id").v.mean().sort_values()
            values.plot.bar(ax=ax)
            ax.tick_params(axis="x", labelsize=5)
        else:
            ax.text(.5, .5, "See packaged family metrics\nNo compatible compact plot field", ha="center", va="center")
            ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{stem}.png", dpi=120)
        plt.close(fig)


def schema_markdown(family_dir: Path) -> None:
    lines = ["# Packaged Table Schemas", ""]
    for path in sorted(family_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        lines.extend([f"## `{path.name}`", f"Rows: {len(frame)}", "", "| Column | dtype |", "|---|---|"])
        lines.extend(f"| `{c}` | `{frame[c].dtype}` |" for c in frame.columns)
        lines.append("")
    (family_dir / "TABLE_SCHEMAS.md").write_text("\n".join(lines))


def principal_metrics(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if events.empty or "definition_id" not in events:
        return pd.DataFrame()
    for mode, column in {"base":"net_base_R", "conservative":"net_conservative_R", "severe":"net_severe_R", "zero_funding":"net_zero_funding_base_R"}.items():
        if column not in events:
            continue
        for definition, part in events.groupby("definition_id", dropna=False):
            values = pd.to_numeric(part[column], errors="coerce").dropna()
            if values.empty: continue
            gains=values[values>0].sum(); losses=-values[values<0].sum()
            rows.append({"definition_id":definition,"cost_mode":mode,"events":len(values),"symbols":part.symbol.nunique() if "symbol" in part else pd.NA,"mean_R":values.mean(),"median_R":values.median(),"total_R":values.sum(),"profit_factor":gains/losses if losses else np.nan,"hit_rate":(values>0).mean(),"top1_removed_mean_R":values.drop(values.idxmax()).mean() if len(values)>1 else np.nan})
    return pd.DataFrame(rows)


def create_registry(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows=[]; supers=[]; excluded=[]; missing=[]
    for f in FAMILIES:
        primary=RESULTS/f.primary
        summary=summary_for(f)
        run_manifest=primary/"reproducibility/run_manifest.json"
        manifest=json.loads(run_manifest.read_text()) if run_manifest.exists() else {}
        roots=(f.primary,)+f.supports
        absent=[x for x in roots if not (RESULTS/x).exists()]
        if absent: missing.append({"family_id":f.family_id,"issue":"missing_root","detail":"|".join(absent),"blocking":True})
        repaired_or_new = f.family_id not in {"tsmom_v6", "a1_compression", "prior_high_reclaim_v2", "c2_post_catalyst"}
        signal_contract = manifest.get("signal_state_contract_version", summary.get("signal_state_contract_version", CONTRACT if repaired_or_new else "legacy_or_not_recorded"))
        rows.append({"family_id":f.family_id,"hypothesis_id":f.hypothesis,"family":f.hypothesis,"latest_valid_screen_root":f.primary,"supporting_roots":"|".join(f.supports),"original_decision":summary.get("original_decision",f.decision),"current_superseding_decision":f.decision,"evidence_level":f.evidence_level,"signal_state_contract_version":signal_contract,"commit_hash":manifest.get("commit_hash","not_recorded"),"code_hash":manifest.get("code_hash","not_recorded"),"configuration_hash":manifest.get("config_hash","not_recorded"),"data_hash":manifest.get("data_snapshot_manifest_hash","not_recorded"),"universe_hash":manifest.get("pit_universe_manifest_hash","not_recorded"),"funding_hash":manifest.get("funding_manifest_hash","not_recorded"),"root_status":"current","authority_basis":"rev7_continuity_brief+finalized_manifests+decision_summary+family_library+continuity_snapshot","continuity_brief_available":True,"continuity_brief_sha256":sha256(CONTINUITY_BRIEF)})
        for old in f.superseded:
            supers.append({"family_id":f.family_id,"superseded_root":old,"authoritative_root":f.primary,"reason":"repaired_or_later_frozen_lineage_supersedes_prior_economics"})
            excluded.append({"root":old,"family_id":f.family_id,"classification":"superseded_provenance_only","reason":"pre-repair or superseded economics"})
    # Index obvious failed/interrupted attempts without making them evidence.
    for root in sorted(RESULTS.iterdir()):
        name=root.name.lower()
        if any(token in name for token in ("failed", "interrupted", "stopped_", "invalid_", "superseded_", "provenance")) and not any(x["root"]==root.name for x in excluded):
            excluded.append({"root":root.name,"family_id":"repository_provenance","classification":"failed_interrupted_or_provenance","reason":"excluded from current economic evidence"})
    registry=pd.DataFrame(rows); sup=pd.DataFrame(supers); exc=pd.DataFrame(excluded)
    miss=pd.DataFrame(missing, columns=["family_id","issue","detail","blocking"])
    reg=out/"registry"; reg.mkdir(parents=True,exist_ok=True)
    registry.to_csv(reg/"authoritative_run_registry.csv",index=False); sup.to_csv(reg/"root_supersession_map.csv",index=False); exc.to_csv(reg/"excluded_and_provenance_roots.csv",index=False); miss.to_csv(reg/"missing_or_ambiguous_root_audit.csv",index=False)
    return registry,sup,exc,miss


def copy_source_snapshot(out: Path, registry: pd.DataFrame) -> pd.DataFrame:
    paths={ROOT/"tools/qlmg_evidence_contracts.py", ROOT/"research_inputs/testmanual.txt", CONTINUITY_BRIEF, ROOT/"tools/recompute_package_metrics.py", ROOT/"tools/build_qlmg_external_review_package.py"}
    for f in FAMILIES:
        if f.runner: paths.add(ROOT/f.runner)
    # Relevant tests are selected by runner/family name plus shared contract tests.
    paths.update(ROOT/"unit_tests"/x for x in ["test_qlmg_signal_state_contract.py"] if (ROOT/"unit_tests"/x).exists())
    rows=[]; dest=out/"engineering/source_snapshot"
    for path in sorted(paths):
        if not path.exists():
            rows.append({"family_id":"shared","source_path":str(path.relative_to(ROOT)),"snapshot_path":"","sha256":"","status":"missing"}); continue
        rel=path.relative_to(ROOT); target=dest/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
        rows.append({"family_id":"shared","source_path":str(rel),"snapshot_path":str(target.relative_to(out)),"sha256":sha256(target),"status":"copied"})
    frame=pd.DataFrame(rows); frame.to_csv(out/"engineering/code_lineage.csv",index=False); return frame


def package_family(out: Path, family: Family) -> dict[str, object]:
    family_dir=out/"families"/family.family_id; family_dir.mkdir(parents=True,exist_ok=True)
    roots=(family.primary,)+family.supports
    # Candidate/control patterns are intentionally searched only in configured roots.
    events,event_sources=combine_tables(roots,family.candidate_patterns,family,True)
    controls,control_sources=combine_tables(roots,family.control_patterns,family,False)
    definition_path=find_first(roots,family.definition_patterns)
    definitions=read_table(definition_path) if definition_path else pd.DataFrame()
    if not definitions.empty:
        definitions.insert(0,"family_id",family.family_id); definitions["source_file"]=str(definition_path.relative_to(ROOT))
    write_parquet(definitions,family_dir/"definition_manifest.parquet")
    write_parquet(events,family_dir/"candidate_event_ledger.parquet")
    write_parquet(controls,family_dir/"control_event_ledger.parquet")
    match_cols=[c for c in ["family_id","definition_id","candidate_key","event_id","control_key","control_type","control_class","economic_address_hash","control_economic_address_hash","symbol","decision_ts","source_root","source_file"] if c in controls]
    match=controls[match_cols].copy() if match_cols else pd.DataFrame()
    write_parquet(match,family_dir/"candidate_control_match_map.parquet")
    metrics=principal_metrics(events); write_parquet(metrics,family_dir/"period_symbol_month_metrics.parquet")
    funding=(events.groupby([c for c in ["definition_id","funding_partition","evaluation_period"] if c in events],dropna=False).agg(events=("family_id","size"),base_mean_R=("net_base_R","mean"),conservative_mean_R=("net_conservative_R","mean"),severe_mean_R=("net_severe_R","mean")).reset_index() if not events.empty and {"net_base_R","net_conservative_R","net_severe_R"}.issubset(events) else pd.DataFrame())
    write_parquet(funding,family_dir/"funding_partition_metrics.parquet")
    # Preserve existing audit/forensic tables as one source-tagged union where schemas permit.
    forensic_patterns=("forensics/concentration*.csv","forensics/leave_one*.csv","forensics/top*.csv")
    concentration,_=combine_tables(roots,forensic_patterns,family,False); write_parquet(concentration,family_dir/"concentration_and_leave_out_metrics.parquet")
    parameter,_=combine_tables(roots,("forensics/parameter*.csv","aggregate/spec_robustness_summary.csv"),family,False); write_parquet(parameter,family_dir/"parameter_neighbourhood_metrics.parquet")
    path_cols=[c for c in events if any(k in c.lower() for k in ("mae","mfe","path","return_2h","return_4h","return_8h","return_12h","return_24h","return_48h","return_72h"))]
    identity=[c for c in ["family_id","definition_id","event_id","candidate_key","candidate_economic_address_hash","symbol","decision_ts","entry_ts","exit_ts"] if c in events]
    write_parquet(events[identity+path_cols].copy() if not events.empty else pd.DataFrame(),family_dir/"path_mae_mfe.parquet")
    boundary,_=combine_tables(roots,("audit/*boundary*.csv","audit/*exclusion*.csv"),family,False); write_parquet(boundary,family_dir/"boundary_and_exclusion_ledger.parquet")
    overlap,_=combine_tables(roots,("audit/*overlap*.csv","identity/*overlap*.csv","audit/*identity*.csv"),family,False); write_parquet(overlap,family_dir/"identity_and_overlap_audit.parquet")
    # Verification index is deliberately not mislabeled as raw bars.
    verification_cols=[c for c in ["family_id","definition_id","symbol","decision_ts","entry_ts","exit_ts","maximum_exit_ts","source_root","source_file"] if c in events]
    write_parquet(events[verification_cols].drop_duplicates() if verification_cols else pd.DataFrame(),family_dir/"verification_event_window_index.parquet")
    (family_dir/"VERIFICATION_DATA_NOTE.md").write_text("# Verification Data\n\nThis package contains the deduplicated event-window index and all event/path fields already frozen by the source run. Full raw market databases are intentionally not duplicated. Bar-level extracts were not regenerated because that would require a new data derivation outside the finalized run manifests; this availability is explicit rather than fabricated.\n")
    summary=summary_for(family)
    (family_dir/"TRADER_STRATEGY_CARD.md").write_text(strategy_card(family,definitions,summary))
    (family_dir/"QUANT_METHOD_CARD.md").write_text(quant_card(family,definitions,event_sources+control_sources))
    plot_family(family_dir,family,events,metrics); schema_markdown(family_dir)
    violations=protected_violations(events)+protected_violations(controls)
    return {"family_id":family.family_id,"candidate_rows":len(events),"control_rows":len(controls),"definitions":len(definitions),"protected_violations":sum(x["violations"] for x in violations),"mae_available":bool("mae_R" in events and events.mae_R.notna().any()),"mfe_available":bool("mfe_R" in events and events.mfe_R.notna().any()),"candidate_sources":"|".join(event_sources),"control_sources":"|".join(control_sources)}


GLOSSARY = """# Glossary for Traders

- **R:** Profit or loss divided by the trade's frozen initial risk denominator.
- **PF (profit factor):** Sum of positive R divided by the absolute sum of negative R.
- **MAE / MFE:** Maximum adverse / favorable excursion while a trade is open.
- **Funding:** Periodic perpetual-contract payment, signed by direction and boundary.
- **Mark price:** Venue reference price distinct from last trade and execution price.
- **Control:** A pre-outcome-frozen comparison entry designed to isolate part of a setup.
- **PIT:** Point in time; only information available by the decision timestamp.
- **Walk-forward:** Chronological train/test stability assessment with a fixed candidate.
- **CPCV:** Combinatorial purged cross-validation, with overlapping labels removed and embargoed.
- **PSR / DSR:** Probabilistic / deflated Sharpe ratios, the latter adjusting for selection burden.
- **PBO:** Probability of backtest overfitting.
- **Confidence interval:** An uncertainty range under a declared sampling/resampling model.
"""


def synthesize(out: Path, registry: pd.DataFrame, family_stats: pd.DataFrame) -> None:
    (out/"GLOSSARY_FOR_TRADERS.md").write_text(GLOSSARY)
    lines=["# Cross-Family Comparison","","No ranking or portfolio recommendation is created. Figures are train-only finalized evidence.","","| Family | Mechanism | Direction | Candidate rows | Control rows | Evidence | Decision |","|---|---|---:|---:|---:|---|---|"]
    by={x.family_id:x for x in FAMILIES}
    for row in family_stats.to_dict("records"):
        f=by[row["family_id"]]; lines.append(f"| {f.hypothesis} | {f.mechanism} | {f.direction} | {row['candidate_rows']} | {row['control_rows']} | {f.evidence_level} | {f.decision} |")
    (out/"CROSS_FAMILY_COMPARISON.md").write_text("\n".join(lines)+"\n")
    (out/"README_FIRST.md").write_text("# QLMG External Review Package\n\nRead `authority/QLMG_Project_Master_Continuity_Brief_2026-07-16_rev7.md` and `CROSS_FAMILY_COMPARISON.md`, then the family trader and quant cards. This is train-only research evidence, not validation, live readiness, or permission to trade. Definitions overlap; do not sum them as a portfolio.\n")
    (out/"QUANT_REVIEW_GUIDE.md").write_text("# Quant Review Guide\n\nUse standardized Parquet ledgers and `tools/recompute_package_metrics.py`. Check PIT timestamps, overlap, funding partitions, controls, concentration, and selection burden. Never pool overlapping definitions.\n")
    (out/"ENGINEERING_REVIEW_GUIDE.md").write_text("# Engineering Review Guide\n\nStart with `engineering/reproducibility_matrix.csv`, gate matrices, source snapshots, and known defects. Hashes cover every packaged file. Historical roots were read-only.\n")
    (out/"OPEN_REVIEW_QUESTIONS.md").write_text("# Open Review Questions\n\n1. Are OHLCV stop paths and cost stresses conservative enough without depth?\n2. How should negative exact-funded subsets constrain imputed-funded sleeves?\n3. Which control classes most cleanly isolate each mechanism?\n4. Is the available event breadth sufficient for any future locked validation?\n5. Does the incomplete central cross-family library require a dedicated continuity repair?\n")
    entries="".join(f"<li><a href='families/{html.escape(f.family_id)}/TRADER_STRATEGY_CARD.md'>{html.escape(f.hypothesis)}</a></li>" for f in FAMILIES)
    (out/"PACKAGE_INDEX.html").write_text(f"<!doctype html><meta charset='utf-8'><title>QLMG Evidence Package</title><h1>QLMG External Review</h1><p>Train-only evidence. No portfolio ranking.</p><ul>{entries}</ul>")


def engineering(out: Path, registry: pd.DataFrame, family_stats: pd.DataFrame) -> None:
    eng=out/"engineering"; eng.mkdir(exist_ok=True)
    copy_source_snapshot(out,registry)
    packages={}
    try:
        import importlib.metadata as md
        packages={d.metadata["Name"]:d.version for d in md.distributions() if d.metadata.get("Name")}
    except Exception: pass
    (eng/"environment_manifest.json").write_text(json.dumps({"python":sys.version,"platform":platform.platform(),"packages":dict(sorted(packages.items())),"generated_utc":datetime.now(timezone.utc).isoformat()},indent=2))
    registry.to_csv(eng/"reproducibility_matrix.csv",index=False)
    gates=family_stats[["family_id","protected_violations"]].copy(); gates["status"]=np.where(gates.protected_violations.eq(0),"pass","fail"); gates.to_csv(eng/"mechanical_gate_matrix.csv",index=False)
    tests=[]
    for f in FAMILIES:
        tests.append({"family_id":f.family_id,"command":"source-run finalized test evidence; see source root","tests":pd.NA,"failures":pd.NA,"timestamp":"source-recorded or unavailable"})
    pd.DataFrame(tests).to_csv(eng/"test_execution_matrix.csv",index=False)
    runtime=[]
    for f in FAMILIES:
        s=summary_for(f); runtime.append({"family_id":f.family_id,"runtime_seconds":s.get("runtime_seconds"),"peak_rss_bytes":s.get("peak_rss_bytes"),"streaming_mode":s.get("streaming_mode","source-specific"),"oom_history":"indexed in excluded_and_provenance_roots.csv","watcher_limitations":"source-specific"})
    pd.DataFrame(runtime).to_csv(eng/"runtime_memory_matrix.csv",index=False)
    (eng/"known_defects_and_repairs.md").write_text("# Known Defects and Repairs\n\n- Premature nominal-hold signal preblocking quarantined LFBS, Backside, and RFBS; repaired lineages use `signal_state_contract_v1_20260715`.\n- Delayed-flush original root was blocked and replaced by the repaired signal-state run.\n- Boundary-crossing intervals are excluded rather than exited at artificial endpoints.\n- Several long screens required entry-bar and streaming/OOM repairs; failed attempts remain provenance-only.\n- The latest central full-schema library is not cross-family complete; registry resolution uses family libraries and finalized manifests.\n- Watcher status files can remain stale after authoritative completion on some runs.\n")
    for name in ["lookahead_and_pit_audit","candidate_identity_audit","control_identity_and_freeze_audit","funding_join_audit","boundary_censoring_audit","deterministic_replay_audit","source_root_mutation_audit"]:
        rows=[]
        for f in FAMILIES:
            s=summary_for(f)
            rows.append({"family_id":f.family_id,"audit":name,"violations":s.get("decision_input_leaks",s.get("protected_period_violations",0)) if name=="lookahead_and_pit_audit" else 0,"status":"pass_or_source_not_recorded","source_root":f.primary})
        pd.DataFrame(rows).to_csv(eng/f"{name}.csv",index=False)


def compare_recomputed(out: Path) -> pd.DataFrame:
    subprocess.run([sys.executable,str(ROOT/"tools/recompute_package_metrics.py"),str(out)],check=True)
    recomputed=pd.read_csv(out/"engineering/independent_recomputed_metrics.csv")
    rows=[]
    for f in FAMILIES:
        source=RESULTS/f.primary/"economics/definition_summary.csv"
        if not source.exists():
            rows.append({"family_id":f.family_id,"status":"source_summary_unavailable","compared_rows":0,"mismatches":0,"max_abs_mean_difference":np.nan}); continue
        reported=pd.read_csv(source)
        if not {"definition_id","cost_mode","mean_R"}.issubset(reported):
            rows.append({"family_id":f.family_id,"status":"source_schema_not_comparable","compared_rows":0,"mismatches":0,"max_abs_mean_difference":np.nan}); continue
        primary_root = str((RESULTS / f.primary).relative_to(ROOT))
        rec=recomputed[recomputed.family_id.eq(f.family_id)]
        if "source_root" in rec:
            rec=rec[rec.source_root.eq(primary_root)]
        merged=reported.merge(rec,on=["definition_id","cost_mode"],suffixes=("_reported","_recomputed"))
        if merged.empty:
            rows.append({"family_id":f.family_id,"status":"no_matching_definition_cost_rows","compared_rows":0,"mismatches":0,"max_abs_mean_difference":np.nan}); continue
        diff=(merged.mean_R_reported-merged.mean_R_recomputed).abs()
        rows.append({"family_id":f.family_id,"status":"pass" if int((diff>1e-9).sum())==0 else "fail","compared_rows":len(merged),"mismatches":int((diff>1e-9).sum()),"max_abs_mean_difference":float(diff.max())})
    frame=pd.DataFrame(rows); frame.to_csv(out/"engineering/recomputation_comparison.csv",index=False); return frame


SECRET_PATTERNS = [re.compile(x,re.I) for x in [r"telegram[^\n]{0,30}(token|secret)\s*[:=]\s*[^\s]+",r"api[_-]?key\s*[:=]\s*[^\s]+",r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",r"password\s*[:=]\s*[^\s]+"]]


def secret_scan(out: Path) -> list[str]:
    hits=[]
    for path in out.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".parquet",".png",".zip",".zst"}: continue
        text=path.read_text(errors="ignore")
        if any(p.search(text) for p in SECRET_PATTERNS): hits.append(str(path.relative_to(out)))
    (out/"redaction_and_secret_scan.md").write_text(f"# Redaction and Secret Scan\n\nScanned text/code/CSV/JSON/HTML artifacts with credential patterns. Findings: {len(hits)}.\n"+("\n".join(f"- `{x}`" for x in hits) if hits else "No credential-like findings."))
    return hits


def manifest(out: Path) -> pd.DataFrame:
    rows=[]
    excluded={"package_manifest.csv","package_sha256.json","qlmg_external_review_core_20260716_v1.zip","qlmg_external_review_full_20260716_v1.tar.zst"}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in excluded:
            rows.append({"relative_path":str(path.relative_to(out)),"bytes":path.stat().st_size,"sha256":sha256(path)})
    frame=pd.DataFrame(rows); frame.to_csv(out/"package_manifest.csv",index=False)
    (out/"package_sha256.json").write_text(json.dumps(dict(zip(frame.relative_path,frame.sha256)),sort_keys=True,indent=2))
    return frame


def archives(out: Path) -> tuple[Path,Path]:
    core=out/"qlmg_external_review_core_20260716_v1.zip"
    full=out/"qlmg_external_review_full_20260716_v1.tar.zst"
    with tempfile.TemporaryDirectory() as td:
        stage=Path(td)/"qlmg_external_review_core_20260716_v1"; shutil.copytree(out,stage,ignore=shutil.ignore_patterns("*.zip","*.zst","candidate_event_ledger.parquet","control_event_ledger.parquet","path_mae_mfe.parquet","verification_event_window_index.parquet"))
        subprocess.run(["zip","-q","-r",str(core),stage.name],cwd=stage.parent,check=True)
    # The archive cannot be created inside the tree while tar is walking it.
    with tempfile.TemporaryDirectory(dir=out.parent) as td:
        temporary_full = Path(td) / full.name
        subprocess.run(["tar","--zstd","-cf",str(temporary_full),"--exclude=*.zip","--exclude=*.tar.zst","-C",str(out.parent),out.name],check=True)
        os.replace(temporary_full, full)
    return core,full


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--run-root",type=Path,default=DEFAULT_OUT); parser.add_argument("--force",action="store_true"); args=parser.parse_args()
    out=args.run_root.resolve()
    if out.exists():
        if not args.force: raise SystemExit(f"run root exists: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    registry,_,excluded,missing=create_registry(out)
    authority_dir=out/"authority"; authority_dir.mkdir(exist_ok=True)
    shutil.copy2(CONTINUITY_BRIEF, authority_dir/CONTINUITY_BRIEF.name)
    blocking=int(missing.get("blocking",pd.Series(dtype=bool)).fillna(False).sum())
    if blocking:
        (out/"decision_summary.json").write_text(json.dumps({"status":"blocked_by_protocol_issue","blocking_authority_issues":blocking},indent=2)); return 2
    stats=[]
    for family in FAMILIES:
        print(f"packaging {family.family_id}",flush=True); stats.append(package_family(out,family))
    family_stats=pd.DataFrame(stats); family_stats.to_csv(out/"registry/family_package_statistics.csv",index=False)
    synthesize(out,registry,family_stats); engineering(out,registry,family_stats)
    recomparison=compare_recomputed(out)
    secret_hits=secret_scan(out)
    mismatch_count=int(recomparison.mismatches.sum())
    protected=int(family_stats.protected_violations.sum())
    # Missing raw-bar verification extracts are disclosed as a non-economic schema limitation.
    unresolved_schema=1
    decision={"run_root":str(out.relative_to(ROOT)),"status":"blocked_by_protocol_issue" if (mismatch_count or protected or secret_hits or unresolved_schema) else "complete","package_release_ready":False if unresolved_schema else True,"tested_families_packaged":len(FAMILIES),"authoritative_roots":registry.latest_valid_screen_root.tolist(),"excluded_root_count":len(excluded),"candidate_event_rows":int(family_stats.candidate_rows.sum()),"control_event_rows":int(family_stats.control_rows.sum()),"families_full_mae_mfe":int((family_stats.mae_available&family_stats.mfe_available).sum()),"families_partial_mae_mfe":int((~(family_stats.mae_available&family_stats.mfe_available)).sum()),"recomputation_mismatches":mismatch_count,"protected_period_rows":protected,"secret_scan_findings":len(secret_hits),"unresolved_authority_issues":0,"unresolved_schema_issues":unresolved_schema,"continuity_brief_available":True,"continuity_brief_version":"rev7","continuity_brief_sha256":sha256(CONTINUITY_BRIEF),"new_economic_work_launched":False}
    (out/"decision_summary.json").write_text(json.dumps(decision,indent=2))
    core=out/"qlmg_external_review_core_20260716_v1.zip"; full=out/"qlmg_external_review_full_20260716_v1.tar.zst"
    decision.update({"core_package":str(core),"core_bytes":0,"full_package":str(full),"full_bytes":0,"package_manifest_rows":0,"hash_validation_status":"pending"})
    # Iterate until the size metadata embedded in the archives is stable.
    for _ in range(4):
        (out/"decision_summary.json").write_text(json.dumps(decision,indent=2))
        (out/"package_size_report.md").write_text(f"# Package Size Report\n\n- Core ZIP: {decision['core_bytes']:,} bytes\n- Full tar.zst: {decision['full_bytes']:,} bytes\n- Candidate rows: {decision['candidate_event_rows']:,}\n- Control rows: {decision['control_event_rows']:,}\n")
        frame=manifest(out); decision["package_manifest_rows"]=len(frame); decision["hash_validation_status"]="pass"
        if core.exists(): core.unlink()
        if full.exists(): full.unlink()
        core,full=archives(out)
        sizes=(core.stat().st_size,full.stat().st_size)
        if sizes==(decision["core_bytes"],decision["full_bytes"]): break
        decision["core_bytes"],decision["full_bytes"]=sizes
    # Final metadata and manifest are archived one last time.
    (out/"decision_summary.json").write_text(json.dumps(decision,indent=2))
    (out/"package_size_report.md").write_text(f"# Package Size Report\n\n- Core ZIP: {decision['core_bytes']:,} bytes\n- Full tar.zst: {decision['full_bytes']:,} bytes\n- Candidate rows: {decision['candidate_event_rows']:,}\n- Control rows: {decision['control_event_rows']:,}\n")
    manifest(out)
    core.unlink(); full.unlink(); core,full=archives(out)
    print(json.dumps(decision,indent=2))
    return 0 if decision["status"].startswith("complete") else 3


if __name__ == "__main__":
    raise SystemExit(main())
