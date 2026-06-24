#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "phase_project_deep_cleanup_20260624_v1"
ARCHIVE_NAME = "legacy_donch_research_20260624"
SMALL_ARCHIVE_MAX_BYTES = 2 * 1024 * 1024
MAX_ARCHIVED_FILES_PER_ROOT = 250

LEGACY_REBASELINE_PATTERNS = (
    "phase_state_transition", "phase_rebound", "phase_event_crowding", "phase_v3", "phase_s1",
    "phase_multivariant", "phase_reset", "phase_oldcfg", "phase_liquid_sentiment",
    "phase_high_activity", "phase_two_sleeve", "phase_three_sleeve",
    "phase2_", "phase3_", "phase4_", "phase5_", "phase6_", "phase7_", "phase8_",
    "phase9_", "phase10_", "phase11_", "phase12_", "phase13_", "phase14", "phase15",
    "phase17", "phase18", "phase20", "feature_repair",
)
TOP_LEVEL_RESULTS_DELETE_NAMES = (
    "policy_sweeps", "jt016_entry_sweeps", "jt029_fast_burst_sweeps", "jt014_regime_exits",
    "jt026_sentiment_riskon", "jt013_risk_off_sweeps", "jt_riskon_rule_fast", "walkforward_oos",
    "simulations", "offline_releases", "investigation", "_deprecated_runs", "donch_autopar",
    "leakage_audit", "signals_fixed_dir", "sweeps", "meta", "meta_labeling", "meta_export",
    "meta_export old", "Full trades 1221", "bt_signals_p060", "bt_signals_p065", "bt_signals_p070",
    "bt_signals_p075", "bt_signals_p080", "bt_signals_p085", "bt_signals_p090", "policy_sweeps_fixed_only",
    "_probe_patch_check", "_speed_sanity",
)
ARTIFACT_DELETE_ROOTS = ("artifacts/rebaseline", "artifacts/meta")
KEEP_NAME_PATTERNS = ("phase_qlmg_perp_project_reset", "phase_project_deep_cleanup")
ARCHIVE_FILE_KEYWORDS = (
    "report", "manifest", "contract", "summary", "readme", "decision", "policy", "index", "audit",
    "schema", "status", "verdict", "plan", "memo", "root", "state", "settings", "config",
)
ARCHIVE_SUFFIXES = {".md", ".json", ".yaml", ".yml", ".txt", ".csv"}
EXCLUDED_ARCHIVE_KEYWORDS = ("ledger", "signal_decisions", "trades", "path_stats", "entry_ledger", "part-", "extract")
NEVER_DELETE_PARTS = {
    ".git", ".venv", "venv", "tools", "unit_tests", "tests", "live", "deploy", "docs", "parquet",
}
NEVER_DELETE_PREFIXES = (
    "/opt/parquet",
    str(REPO / "docs"),
    str(REPO / "tools"),
    str(REPO / "unit_tests"),
    str(REPO / "live"),
    str(REPO / "deploy"),
    str(REPO / ".git"),
    str(REPO / ".venv"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep cleanup and GitHub-ready tidy for legacy Donch research artifacts.")
    p.add_argument("--execute", action="store_true", help="Actually delete planned legacy artifacts. Default is dry-run.")
    p.add_argument("--run-root", default="")
    p.add_argument("--results-root", default="results/rebaseline")
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--archive-root", default=str(REPO / "archive" / ARCHIVE_NAME))
    return p.parse_args()


def run_root_from_args(args: argparse.Namespace) -> tuple[Path, str]:
    if args.run_root:
        root = Path(args.run_root)
        return (root if root.is_absolute() else (REPO / root).resolve()), "explicit_run_root"
    base = (REPO / args.results_root / args.run_id).resolve()
    if not base.exists():
        return base, "default_root_available"
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return base.with_name(f"{base.name}_{suffix}"), f"default_root_existed_created_suffix_{suffix}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    if fieldnames is None:
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def shell(args: Sequence[str], timeout: float = 60.0) -> str:
    try:
        p = subprocess.run(args, cwd=str(REPO), text=True, capture_output=True, timeout=timeout, check=False)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def file_count(path: Path) -> int:
    if path.is_file():
        return 1
    return sum(1 for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def human(n: int | float) -> str:
    x = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if x < 1024 or unit == "TB":
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}TB"


def sha256_file(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = limit_bytes
        while True:
            read_size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if read_size <= 0:
                break
            chunk = f.read(read_size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def is_active_keep(path: Path, run_root: Path, archive_root: Path) -> bool:
    rp = str(path.resolve())
    if rp == str(run_root.resolve()) or rp.startswith(str(run_root.resolve()) + os.sep):
        return True
    if rp == str(archive_root.resolve()) or rp.startswith(str(archive_root.resolve()) + os.sep):
        return True
    return any(pattern in path.name for pattern in KEEP_NAME_PATTERNS)


def classify_candidate(path: Path, run_root: Path, archive_root: Path) -> tuple[bool, str]:
    if not path.exists() or is_active_keep(path, run_root, archive_root):
        return False, "active_or_missing"
    try:
        resolved = str(path.resolve())
    except Exception:
        resolved = str(path)
    if any(resolved == p or resolved.startswith(p + os.sep) for p in NEVER_DELETE_PREFIXES):
        return False, "protected_prefix"
    if (REPO / "results/rebaseline") in path.parents and any(x in path.name for x in LEGACY_REBASELINE_PATTERNS):
        return True, "legacy_rebaseline_pattern"
    if path.parent == REPO / "results" and path.name in TOP_LEVEL_RESULTS_DELETE_NAMES:
        return True, "legacy_top_level_results"
    if rel(path) in ARTIFACT_DELETE_ROOTS:
        return True, "legacy_artifact_root"
    if path.is_file() and path.name == "trade_log_and_ledger_file_index.txt" and path.with_suffix(path.suffix + ".gz").exists():
        return True, "redundant_uncompressed_index_with_gzip"
    if path.is_file() and path.suffix == ".zip" and path.with_suffix("").exists():
        return True, "redundant_zip_with_unpacked_folder"
    return False, "not_legacy_candidate"


def guard_deletion(path: Path, run_root: Path, archive_root: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "path_missing"
    ok, reason = classify_candidate(path, run_root, archive_root)
    if not ok:
        return False, reason
    resolved = str(path.resolve())
    if any(resolved == p or resolved.startswith(p + os.sep) for p in NEVER_DELETE_PREFIXES):
        return False, "protected_prefix_guard"
    parts = set(path.resolve().parts)
    if parts & NEVER_DELETE_PARTS:
        return False, "protected_part_guard"
    return True, reason


def candidate_roots(run_root: Path, archive_root: Path) -> list[Path]:
    candidates: list[Path] = []
    rebase = REPO / "results/rebaseline"
    if rebase.exists():
        for p in sorted(rebase.iterdir()):
            ok, _ = classify_candidate(p, run_root, archive_root)
            if ok:
                candidates.append(p)
    top = REPO / "results"
    if top.exists():
        for p in sorted(top.iterdir()):
            ok, _ = classify_candidate(p, run_root, archive_root)
            if ok:
                candidates.append(p)
    for rel_path in ARTIFACT_DELETE_ROOTS:
        p = REPO / rel_path
        ok, _ = classify_candidate(p, run_root, archive_root)
        if ok:
            candidates.append(p)
    for p in [REPO / "reports/project_handover_due_diligence_20260621/trade_log_and_ledger_file_index.txt", REPO / "reports/project_handover_due_diligence_20260621.zip", REPO / "reports/project_handover_qa_20260618.zip"]:
        ok, _ = classify_candidate(p, run_root, archive_root)
        if ok:
            candidates.append(p)
    # Deduplicate while preserving order.
    out: list[Path] = []
    seen = set()
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def archive_should_copy(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > SMALL_ARCHIVE_MAX_BYTES:
        return False
    low = path.name.lower()
    if path.suffix.lower() not in ARCHIVE_SUFFIXES:
        return False
    if any(x in low for x in EXCLUDED_ARCHIVE_KEYWORDS):
        return False
    return any(x in low for x in ARCHIVE_FILE_KEYWORDS) or path.suffix.lower() in {".md", ".json", ".yaml", ".yml"}


def archive_compact_files(root: Path, archive_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if root.is_file():
        files = [root] if archive_should_copy(root) else []
    else:
        files = [p for p in sorted(root.rglob("*")) if archive_should_copy(p)]
    for p in files[:MAX_ARCHIVED_FILES_PER_ROOT]:
        try:
            rel_under_repo = rel(p)
            dest = archive_root / "preserved_files" / rel_under_repo
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            rows.append({"source_path": rel_under_repo, "archive_path": rel(dest), "size_bytes": p.stat().st_size, "sha256": sha256_file(p), "status": "copied"})
        except Exception as exc:
            rows.append({"source_path": rel(p), "archive_path": "", "size_bytes": p.stat().st_size if p.exists() else 0, "sha256": "", "status": f"copy_failed:{type(exc).__name__}:{exc}"})
    return rows


def build_manifests(run_root: Path, archive_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = candidate_roots(run_root, archive_root)
    cleanup_rows = []
    planned_rows = []
    archive_rows = []
    for p in candidates:
        ok, reason = guard_deletion(p, run_root, archive_root)
        s = size_bytes(p)
        cleanup_rows.append({"path": rel(p), "absolute_path": str(p.resolve()), "is_planned_delete": ok, "classification": reason, "size_bytes": s, "size_human": human(s), "file_count": file_count(p)})
        if ok:
            archive_rows.extend(archive_compact_files(p, archive_root))
            planned_rows.append({"path": rel(p), "absolute_path": str(p.resolve()), "classification": reason, "size_bytes": s, "size_human": human(s), "file_count": file_count(p), "archive_indexed": True})
    return cleanup_rows, archive_rows, planned_rows


def execute_deletions(planned_rows: Sequence[Mapping[str, Any]], execute: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in planned_rows:
        p = Path(str(row["absolute_path"]))
        status = "dry_run_not_deleted"
        err = ""
        if execute:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
                status = "deleted"
            except Exception as exc:
                status = "delete_failed"
                err = f"{type(exc).__name__}: {exc}"
        rows.append({**dict(row), "status": status, "error": err, "deleted_at_utc": utc_now() if status == "deleted" else ""})
    return rows


def docs_index(archive_root: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted((REPO / "docs").rglob("*.md")) if (REPO / "docs").exists() else []:
        rows.append({"path": rel(p), "active_doc": p.name.startswith("QLMG_") or p.name == "README.md", "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    tracked_deleted = subprocess.run(["git", "status", "--short", "docs"], cwd=str(REPO), text=True, capture_output=True, check=False).stdout.splitlines()
    for line in tracked_deleted:
        if line.startswith(" D "):
            rows.append({"path": line[3:], "active_doc": False, "sha256": "deleted_from_active_docs", "size_bytes": "", "note": "legacy Donch doc deleted from active docs; retained in git history and archive index"})
    write_csv(archive_root / "legacy_docs_index.csv", rows)
    return rows


def main() -> None:
    args = parse_args()
    run_root, root_reason = run_root_from_args(args)
    archive_root = Path(args.archive_root)
    if not archive_root.is_absolute():
        archive_root = (REPO / archive_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    before_df = subprocess.run(["df", "-h", str(REPO)], text=True, capture_output=True, check=False).stdout.strip()
    cleanup_rows, archive_rows, planned_rows = build_manifests(run_root, archive_root)
    deleted_rows = execute_deletions(planned_rows, execute=bool(args.execute))
    doc_rows = docs_index(archive_root)
    write_csv(run_root / "cleanup_candidates.csv", cleanup_rows)
    write_csv(run_root / "archive_preservation_manifest.csv", archive_rows)
    write_csv(run_root / "planned_deletions.csv", planned_rows)
    write_csv(run_root / "deleted_manifest.csv", deleted_rows)
    write_csv(run_root / "legacy_docs_index.csv", doc_rows)
    after_df = subprocess.run(["df", "-h", str(REPO)], text=True, capture_output=True, check=False).stdout.strip()
    projected = sum(int(r["size_bytes"]) for r in planned_rows)
    actual = sum(int(r["size_bytes"]) for r in deleted_rows if r.get("status") == "deleted")
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "archive_root": str(archive_root), "root_reason": root_reason, "execute": bool(args.execute), "created_at_utc": utc_now(), "projected_recovery_bytes": projected, "actual_deleted_bytes": actual})
    write_text(run_root / "do_not_delete_guard_report.md", f"""# Do Not Delete Guard Report

- execute: `{bool(args.execute)}`
- planned deletion rows: `{len(planned_rows)}`
- projected recovery: `{human(projected)}`
- actual deleted: `{human(actual)}`

Protected prefixes include `/opt/parquet`, `docs`, `tools`, `unit_tests`, `live`, `deploy`, `.git`, and `.venv`. Active QLMG reset/cleanup roots are also protected.
""")
    write_text(run_root / "cleanup_report.md", f"""# Deep Cleanup Report

- mode: `{'execute' if args.execute else 'dry-run'}`
- run_root: `{run_root}`
- archive_root: `{archive_root}`
- projected recovery: `{human(projected)}`
- actual deleted: `{human(actual)}`

## df before

```text
{before_df}
```

## df after

```text
{after_df}
```

Manifests: `cleanup_candidates.csv`, `archive_preservation_manifest.csv`, `planned_deletions.csv`, `deleted_manifest.csv`, `legacy_docs_index.csv`.
""")
    print(run_root)


if __name__ == "__main__":
    main()
