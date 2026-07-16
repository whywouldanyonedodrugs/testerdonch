#!/usr/bin/env python3
"""Create compact text/graphics-only variants of the QLMG review package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_20260716_v1"
DEFAULT_OUTPUT = ROOT / "results/rebaseline/phase_kraken_all_tested_hypotheses_external_review_package_reduced_20260716_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return float(gains / losses) if losses > 0 else np.nan


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def definition_overview(family_id: str, frame: pd.DataFrame) -> dict[str, object]:
    excluded = {"family_id", "source_file", "parameter_vector_json"}
    fields: list[str] = []
    for column in frame.columns:
        if column in excluded:
            continue
        values = frame[column].dropna().astype(str).drop_duplicates()
        if len(values) <= 12:
            rendered = " | ".join(values.tolist())
        else:
            rendered = f"{len(values)} unique values"
        fields.append(f"{column}={rendered}")
    return {"family_id": family_id, "definition_rows": len(frame), "parameter_and_policy_overview": "; ".join(fields)}


def build(source: Path, reduced: Path, include_plots: bool) -> dict[str, int]:
    if reduced.exists():
        shutil.rmtree(reduced)
    reduced.mkdir(parents=True)

    top_files = [
        "README_FIRST.md", "PACKAGE_INDEX.html", "GLOSSARY_FOR_TRADERS.md",
        "QUANT_REVIEW_GUIDE.md", "ENGINEERING_REVIEW_GUIDE.md",
        "OPEN_REVIEW_QUESTIONS.md", "CROSS_FAMILY_COMPARISON.md",
        "redaction_and_secret_scan.md", "decision_summary.json",
    ]
    for relative in top_files:
        path = source / relative
        if path.exists():
            copy_file(path, reduced / relative)

    for path in (source / "authority").glob("*.md"):
        copy_file(path, reduced / "authority" / path.name)
    for path in (source / "registry").glob("*.csv"):
        copy_file(path, reduced / "registry" / path.name)

    engineering_names = [
        "code_lineage.csv", "reproducibility_matrix.csv", "mechanical_gate_matrix.csv",
        "test_execution_matrix.csv", "runtime_memory_matrix.csv", "known_defects_and_repairs.md",
        "lookahead_and_pit_audit.csv", "candidate_identity_audit.csv",
        "control_identity_and_freeze_audit.csv", "funding_join_audit.csv",
        "boundary_censoring_audit.csv", "deterministic_replay_audit.csv",
        "source_root_mutation_audit.csv", "recomputation_comparison.csv",
        "independent_recomputed_metrics.csv",
    ]
    for name in engineering_names:
        path = source / "engineering" / name
        if path.exists():
            copy_file(path, reduced / "engineering" / name)

    registry = pd.read_csv(source / "registry/authoritative_run_registry.csv")
    family_stats = pd.read_csv(source / "registry/family_package_statistics.csv")
    decisions = registry[["family_id", "hypothesis_id", "current_superseding_decision", "evidence_level"]].merge(
        family_stats[["family_id", "definitions", "candidate_rows", "control_rows", "mae_available", "mfe_available"]],
        on="family_id", how="left",
    )

    definition_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    funding_rows: list[pd.DataFrame] = []
    control_rows: list[dict[str, object]] = []
    economics_rows: list[dict[str, object]] = []

    for family_dir in sorted((source / "families").iterdir()):
        if not family_dir.is_dir():
            continue
        family_id = family_dir.name
        destination = reduced / "families" / family_id
        for name in ["TRADER_STRATEGY_CARD.md", "QUANT_METHOD_CARD.md", "VERIFICATION_DATA_NOTE.md"]:
            path = family_dir / name
            if path.exists():
                copy_file(path, destination / name)
        if include_plots:
            for plot in (family_dir / "plots").glob("*.png"):
                copy_file(plot, destination / "plots" / plot.name)

        definitions = pd.read_parquet(family_dir / "definition_manifest.parquet")
        definition_rows.append(definition_overview(family_id, definitions))

        events = pd.read_parquet(family_dir / "candidate_event_ledger.parquet")
        for mode, column in {
            "base": "net_base_R", "conservative": "net_conservative_R",
            "severe": "net_severe_R", "zero_funding": "net_zero_funding_base_R",
        }.items():
            if column not in events:
                continue
            values = pd.to_numeric(events[column], errors="coerce").dropna()
            if values.empty:
                continue
            economics_rows.append({
                "family_id": family_id, "cost_mode": mode, "event_rows": len(values),
                "mean_R": values.mean(), "median_R": values.median(), "total_R": values.sum(),
                "profit_factor": profit_factor(values), "hit_rate": (values > 0).mean(),
            })
            if "evaluation_period" in events:
                temporary = events.assign(_value=pd.to_numeric(events[column], errors="coerce"))
                for period, part in temporary.groupby("evaluation_period", dropna=False):
                    period_rows.append({
                        "family_id": family_id, "period": period, "cost_mode": mode,
                        "event_rows": int(part._value.notna().sum()), "mean_R": part._value.mean(),
                        "median_R": part._value.median(), "profit_factor": profit_factor(part._value),
                    })

        funding = pd.read_parquet(family_dir / "funding_partition_metrics.parquet")
        if not funding.empty:
            funding_rows.append(funding.assign(family_id=family_id))

        controls = pd.read_parquet(family_dir / "control_event_ledger.parquet")
        control_class = next((x for x in ["control_class", "control_type", "control_pool_class"] if x in controls), None)
        control_value = next((x for x in ["net_conservative_R", "control_net_conservative_R", "net_R"] if x in controls), None)
        if control_class:
            for name, part in controls.groupby(control_class, dropna=False):
                values = pd.to_numeric(part[control_value], errors="coerce").dropna() if control_value else pd.Series(dtype=float)
                control_rows.append({
                    "family_id": family_id, "control_class": name, "rows": len(part),
                    "unique_control_addresses": part["control_economic_address_hash"].nunique() if "control_economic_address_hash" in part else (part["control_key"].nunique() if "control_key" in part else pd.NA),
                    "conservative_mean_R": values.mean() if not values.empty else pd.NA,
                    "conservative_profit_factor": profit_factor(values) if not values.empty else pd.NA,
                })

    tables = reduced / "review_tables"
    tables.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(tables / "family_decision_and_evidence_summary.csv", index=False)
    pd.DataFrame(economics_rows).to_csv(tables / "cross_family_economic_summary.csv", index=False)
    pd.DataFrame(period_rows).to_csv(tables / "family_period_summary.csv", index=False)
    if funding_rows:
        funding_columns = sorted(set().union(*(frame.columns for frame in funding_rows)))
        funding_records = [
            record
            for frame in funding_rows
            for record in frame.reindex(columns=funding_columns).to_dict(orient="records")
        ]
        pd.DataFrame.from_records(funding_records, columns=funding_columns).to_csv(
            tables / "family_funding_partition_summary.csv", index=False
        )
    pd.DataFrame(control_rows).to_csv(tables / "family_control_summary.csv", index=False)
    pd.DataFrame(definition_rows).to_csv(tables / "definition_parameter_overview.csv", index=False)

    note = """# Reduced Package Scope

This reduced package contains textual reports, compact numerical summaries, registries, audits, and optional plots only.

It intentionally excludes:
- Parquet files;
- candidate and control event ledgers;
- raw market, mark, index, funding, lifecycle, and verification-window data;
- source-code snapshots;
- package build environments;
- full definition-level row expansions.

Use the full evidence archive when row-level verification is required. Do not infer portfolio returns by adding overlapping definition results.
"""
    (reduced / "REDUCED_PACKAGE_SCOPE.md").write_text(note)

    rows = []
    for path in sorted(reduced.rglob("*")):
        if path.is_file() and path.name not in {"reduced_package_manifest.csv", "reduced_package_sha256.json"}:
            rows.append({"relative_path": str(path.relative_to(reduced)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(reduced / "reduced_package_manifest.csv", index=False)
    (reduced / "reduced_package_sha256.json").write_text(json.dumps(dict(zip(manifest.relative_path, manifest.sha256)), indent=2, sort_keys=True))
    forbidden = [p for p in reduced.rglob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".feather", ".arrow"}]
    if forbidden:
        raise RuntimeError(f"columnar files entered reduced package: {forbidden[:3]}")
    return {"files": len(manifest), "bytes": sum(int(x) for x in manifest.bytes), "families": len(decisions)}


def archive(directory: Path, output: Path) -> None:
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                handle.write(path, Path(directory.name) / path.relative_to(directory))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.package_root.resolve()
    output_root = args.output_root.resolve()
    if output_root == source or source in output_root.parents:
        raise ValueError("reduced output root must be separate from the audited source package")
    output_root.mkdir(parents=True, exist_ok=True)
    graphical = output_root / "reduced_review_package"
    text_only = output_root / "text_only_review_package"
    graphical_stats = build(source, graphical, include_plots=True)
    text_stats = build(source, text_only, include_plots=False)
    graphical_zip = output_root / "qlmg_external_review_reduced_20260716_v1.zip"
    text_zip = output_root / "qlmg_external_review_text_only_20260716_v1.zip"
    archive(graphical, graphical_zip)
    archive(text_only, text_zip)
    result = {
        "status": "complete",
        "source_package_root": str(source),
        "source_package_modified": False,
        "contains_parquet_or_raw_ledgers": False,
        "graphical_directory": str(graphical), "graphical_zip": str(graphical_zip),
        "graphical_zip_bytes": graphical_zip.stat().st_size, **{f"graphical_{k}": v for k, v in graphical_stats.items()},
        "text_directory": str(text_only), "text_zip": str(text_zip),
        "text_zip_bytes": text_zip.stat().st_size, **{f"text_{k}": v for k, v in text_stats.items()},
    }
    (output_root / "decision_summary.json").write_text(json.dumps(result, indent=2))
    archive_rows = [
        {"archive": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in (graphical_zip, text_zip)
    ]
    pd.DataFrame(archive_rows).to_csv(output_root / "reduced_archives_manifest.csv", index=False)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
