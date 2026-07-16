#!/usr/bin/env python3
"""Rebuild real event-ledger controls for A2/A3/B1/C2 after placeholder invalidation."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from qlmg_match_feature_builder import enrich_event_pool_with_match_features
from qlmg_real_controls import (
    CONTROL_TYPES,
    apply_real_control_labels,
    build_real_controls,
    standardize_event_ledger,
    validate_no_protected_rows,
)
from qlmg_evidence_contracts import assert_pass, validate_control_rows, validate_pit_feature_timestamps

ROOT = Path(__file__).resolve().parents[1]
BASE_RUN_ROOT = ROOT / "results/rebaseline/phase_qlmg_real_control_rebuild_20260629_v1"
CORRECTED_ROOT = ROOT / "results/rebaseline/phase_qlmg_corrected_event_level_development_sweep_20260629_v1_20260629_052114"
REPAIR_ROOT = ROOT / "results/rebaseline/phase_qlmg_b1_c2_ledger_quality_a3_failure_audit_20260629_v1_20260629_062257"
INVALIDATION_ROOT = ROOT / "results/rebaseline/phase_qlmg_global_result_invalidation_audit_20260629_v1_20260629_161953"

INPUT_LEDGERS = [
    ("A3", CORRECTED_ROOT / "a3_sweep/a3_event_level_replay.parquet"),
    ("A2_redesign_only", CORRECTED_ROOT / "a2_sweep/a2_event_level_replay.parquet"),
    ("B1", REPAIR_ROOT / "b1_repair/b1_event_level_replay.parquet"),
    ("C2", REPAIR_ROOT / "c2_repair/c2_event_level_replay.parquet"),
]


def set_input_roots(corrected_root: Path | None = None, repair_root: Path | None = None) -> None:
    """Override source roots for chained reruns."""
    global CORRECTED_ROOT, REPAIR_ROOT, INPUT_LEDGERS
    if corrected_root is not None:
        CORRECTED_ROOT = corrected_root
    if repair_root is not None:
        REPAIR_ROOT = repair_root
    INPUT_LEDGERS = [
        ("A3", CORRECTED_ROOT / "a3_sweep/a3_event_level_replay.parquet"),
        ("A2_redesign_only", CORRECTED_ROOT / "a2_sweep/a2_event_level_replay.parquet"),
        ("B1", REPAIR_ROOT / "b1_repair/b1_event_level_replay.parquet"),
        ("C2", REPAIR_ROOT / "c2_repair/c2_event_level_replay.parquet"),
    ]


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def make_run_root(base: Path) -> Path:
    if not base.exists():
        return base
    return base.with_name(base.name + "_" + utc())


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_csv(path: Path, df: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df, list):
        df = pd.DataFrame(df)
    df.to_csv(path, index=False)


def load_pool(max_symbols: int | None = None, smoke: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    chunks = []
    manifest = []
    for fam, path in INPUT_LEDGERS:
        row = {"family_hint": fam, "path": str(path), "exists": path.exists(), "rows_raw": 0, "rows_standardized": 0}
        if not path.exists():
            manifest.append(row)
            continue
        raw = pd.read_parquet(path)
        row["rows_raw"] = len(raw)
        if max_symbols and "symbol" in raw.columns:
            syms = sorted(raw["symbol"].dropna().astype(str).unique())[:max_symbols]
            raw = raw[raw["symbol"].astype(str).isin(syms)].copy()
        if smoke:
            # Keep at least enough rows to exercise all control paths without a long run.
            raw = raw.groupby(raw.get("candidate_id", pd.Series("candidate", index=raw.index)), group_keys=False).head(80)
        std = standardize_event_ledger(raw, str(path), family_hint=fam)
        row["rows_standardized"] = len(std)
        chunks.append(std)
        manifest.append(row)
    pool = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not pool.empty:
        validate_no_protected_rows(pool, ["decision_ts", "entry_ts", "exit_ts"])
    return pool, manifest


def choose_candidate_keys(pool: pd.DataFrame, top_per_family: int) -> list[str]:
    if pool.empty:
        return []
    rows = []
    for key, g in pool.groupby("candidate_key"):
        fam = str(g["family"].iloc[0])
        vals = pd.to_numeric(g["source_net_R"], errors="coerce")
        rows.append({"candidate_key": key, "family": fam, "events": vals.notna().sum(), "net_R": vals.sum()})
    df = pd.DataFrame(rows)
    selected = []
    for fam, g in df.sort_values(["net_R", "events"], ascending=[False, False]).groupby("family", sort=False):
        selected.extend(g.head(top_per_family)["candidate_key"].astype(str).tolist())
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default=str(BASE_RUN_ROOT))
    ap.add_argument("--corrected-root", default=str(CORRECTED_ROOT))
    ap.add_argument("--repair-root", default=str(REPAIR_ROOT))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--nulls-per-event", type=int, default=3)
    ap.add_argument("--top-per-family", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--max-control-rows-per-candidate", type=int, default=20000)
    args = ap.parse_args()
    set_input_roots(Path(args.corrected_root).resolve(), Path(args.repair_root).resolve())

    run_root = make_run_root(Path(args.run_root))
    for d in ["preflight", "controls", "recomputed", "triage", "quarantine", "compact_review_bundle"]:
        (run_root / d).mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(ROOT).free / 1024**3
    if free_gb < 5:
        raise SystemExit(f"free disk below 5GB hard stop: {free_gb:.2f}GB")

    pool, manifest = load_pool(max_symbols=args.max_symbols, smoke=args.smoke)
    write_csv(run_root / "preflight/input_event_ledger_manifest.csv", manifest)
    if pool.empty:
        raise SystemExit("no event-level ledgers available")
    pool, feature_coverage = enrich_event_pool_with_match_features(pool)
    (run_root / "features").mkdir(parents=True, exist_ok=True)
    pool.to_parquet(run_root / "features/event_match_feature_panel.parquet", index=False)
    write_csv(run_root / "features/match_feature_coverage_summary.csv", feature_coverage)
    write_text(run_root / "features/match_feature_report.md", "# Match Feature Build Report\n\nFeatures are joined as-of from `/opt/parquet/5m` with `feature_source_ts <= decision_ts`. Buckets use fixed predeclared thresholds; missing vol/liquidity/funding/OI remains explicit and can block nearest-neighbor controls.\n")
    pool.to_parquet(run_root / "recomputed/combined_source_event_pool.parquet", index=False)

    candidate_keys = choose_candidate_keys(pool, top_per_family=args.top_per_family)
    cand, control_ledger, control_summary = build_real_controls(
        pool,
        candidate_keys=candidate_keys,
        control_types=CONTROL_TYPES,
        nulls_per_event=args.nulls_per_event,
        seed=args.seed,
        max_control_rows_per_candidate=args.max_control_rows_per_candidate,
    )
    assert_pass(validate_pit_feature_timestamps(pool, feature_ts_cols=("feature_source_ts",)))
    assert_pass(validate_control_rows(control_ledger, allow_empty=False))
    labelled = apply_real_control_labels(cand, control_summary)
    write_csv(run_root / "recomputed/candidate_event_level_summary.csv", cand)
    control_ledger.to_parquet(run_root / "controls/real_control_event_ledger.parquet", index=False)
    write_csv(run_root / "controls/real_control_summary.csv", control_summary)
    write_csv(run_root / "triage/recomputed_candidate_labels.csv", labelled)

    # Family-level interpretation keeps B1/C2 capped unless broad true ledgers and controls exist.
    family_rows = []
    for fam, g in labelled.groupby("family"):
        labels = sorted(g["real_control_label"].dropna().astype(str).unique())
        family_rows.append({
            "family": fam,
            "candidate_count": len(g),
            "events_total": int(g["events"].sum()),
            "labels": ";".join(labels),
            "control_types_built_min": int(g["real_control_type_count"].min()) if "real_control_type_count" in g else 0,
            "min_control_coverage_ratio": float(g["min_real_control_coverage_ratio"].min()) if "min_real_control_coverage_ratio" in g and g["min_real_control_coverage_ratio"].notna().any() else float("nan"),
            "promotion_cap": "seed_or_proxy_limited" if fam in {"B1", "C2"} else "mark_funding_proxy_cap_if_present",
        })
    family_df = pd.DataFrame(family_rows)
    write_csv(run_root / "triage/family_recomputed_control_status.csv", family_df)

    deprecated = pd.DataFrame([
        {"source": str(INVALIDATION_ROOT / "quarantine/quarantined_artifacts_do_not_use_for_ranking.csv"), "applied": (INVALIDATION_ROOT / "quarantine/quarantined_artifacts_do_not_use_for_ranking.csv").exists()},
        {"source": str(INVALIDATION_ROOT / "quarantine/deprecated_promotion_labels.csv"), "applied": (INVALIDATION_ROOT / "quarantine/deprecated_promotion_labels.csv").exists()},
    ])
    write_csv(run_root / "quarantine/inherited_quarantine_inputs.csv", deprecated)

    report = f"""# QLMG Real Control Rebuild Report

Run root: `{run_root}`

## Scope

This phase rebuilds controls from event-level rows only. It does not use placeholder formulas, summary projections, or copied distributions.

## Inputs

- Corrected A2/A3 event ledgers: `{CORRECTED_ROOT}`
- B1/C2 repair ledgers: `{REPAIR_ROOT}`
- Invalidation quarantine source: `{INVALIDATION_ROOT}`

## Results

- Source event rows loaded: `{len(pool)}`
- Candidate definitions recomputed: `{len(cand)}`
- Real control rows selected: `{len(control_ledger)}`
- Control summary rows: `{len(control_summary)}`

## Control Contract

Each selected control row includes `control_event_id`, `control_symbol`, `control_decision_ts`, `control_source_row_id`, `control_window_id`, `match_basis`, and `source_path`.
Controls are normalized to candidate event count in `controls/real_control_summary.csv`.

## B1/C2 Cap

B1 and C2 remain seed/proxy-limited sidecars. Even where real event-ledger controls are built, labels are capped because the underlying ledgers are Markdown/seed-limited and use proxy mark/funding context.

## Outputs

- `features/event_match_feature_panel.parquet`
- `features/match_feature_coverage_summary.csv`
- `controls/real_control_event_ledger.parquet`
- `controls/real_control_summary.csv`
- `recomputed/candidate_event_level_summary.csv`
- `triage/recomputed_candidate_labels.csv`
- `triage/family_recomputed_control_status.csv`

## Operator Decision

`repair_controls_completed_recompute_next_review`

No output from this phase is live-ready, sealed-ready, validated, production-ready, or a trading recommendation.
"""
    write_text(run_root / "QLMG_REAL_CONTROL_REBUILD_REPORT.md", report)
    decision = {
        "run_root": str(run_root),
        "event_rows_loaded": int(len(pool)),
        "match_feature_coverage_rows": int(len(feature_coverage)),
        "match_feature_pit_ok_share": float(pool["match_feature_pit_ok"].mean()) if "match_feature_pit_ok" in pool.columns and len(pool) else 0.0,
        "candidate_definitions_recomputed": int(len(cand)),
        "real_control_rows": int(len(control_ledger)),
        "placeholder_controls_used": False,
        "controls_have_source_ids": bool(len(control_ledger) and control_ledger["control_source_row_id"].notna().all() and control_ledger["control_window_id"].notna().all()),
        "b1_c2_cap": "seed_limited_proxy_context",
        "operator_decision": "repair_controls_completed_recompute_next_review",
        "final_holdout_untouched": True,
    }
    write_text(run_root / "decision_summary.json", json.dumps(decision, indent=2, sort_keys=True))
    for rel in ["QLMG_REAL_CONTROL_REBUILD_REPORT.md", "decision_summary.json", "features/match_feature_coverage_summary.csv", "controls/real_control_summary.csv", "triage/recomputed_candidate_labels.csv", "triage/family_recomputed_control_status.csv"]:
        src = run_root / rel
        if src.exists():
            dst = run_root / "compact_review_bundle" / rel.replace("/", "__")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
