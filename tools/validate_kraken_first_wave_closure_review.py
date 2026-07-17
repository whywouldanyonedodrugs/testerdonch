#!/usr/bin/env python3
"""Independently validate the closed Stage 5A review package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "results/rebaseline/phase_kraken_first_wave_closure_review_20260717_v1"
C01_METRICS = ROOT / "results/rebaseline/phase_kraken_c01_level3_economic_20260717_v1_20260717_103227/DEFINITION_METRICS.csv"
C02_METRICS = ROOT / "results/rebaseline/phase_kraken_c02_positive_spot_led_level3_20260717_v1_20260717_161958/DEFINITION_METRICS.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(package: Path) -> dict:
    manifest = rows(package / "PACKAGE_MANIFEST.csv")
    for row in manifest:
        path = package / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise AssertionError(f"manifest mismatch: {row['path']}")

    summary = json.loads((package / "PACKAGE_SHA256.json").read_text())
    archive = package / "qlmg_first_wave_closure_review_20260717_v01.zip"
    if archive.stat().st_size != summary["archive_bytes"] or sha256(archive) != summary["archive_sha256"]:
        raise AssertionError("archive sidecar mismatch")
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad:
            raise AssertionError(f"bad ZIP member: {bad}")
        if any(name.lower().endswith((".parquet", ".zst", ".tar")) for name in zf.namelist()):
            raise AssertionError("raw/binary tape included")

    decisions = {row["family"]: row["decision"] for row in rows(package / "FIRST_WAVE_DECISION_MATRIX.csv")}
    expected = {"C01": "level3_no_primary_pass_stop", "C02": "level3_no_primary_pass_stop", "C03": "C03_PIT_authority_unavailable"}
    if decisions != expected:
        raise AssertionError(f"decision drift: {decisions}")

    metrics = rows(package / "DEFINITION_METRICS_COMPARISON.csv")
    source_ids = {row["definition_id"] for row in rows(C01_METRICS)} | {row["definition_id"] for row in rows(C02_METRICS)}
    if len(metrics) != 20 or {row["definition_id"] for row in metrics} != source_ids:
        raise AssertionError("definition identity mismatch")
    if any(row["all_level3_gates_pass"] == "True" for row in metrics):
        raise AssertionError("terminal gate drift")

    program = rows(package / "PROGRAM_DECISION_REGISTER.csv")
    if len(program) != 17 or sum(row["status"] == "prior_lineage_strategic_continuity_only" for row in program) != 14:
        raise AssertionError("lineage integration mismatch")
    next_rows = rows(package / "NEXT_WAVE_READINESS_MATRIX.csv")
    selected = [row for row in next_rows if row["recommended_priority"] == "1"]
    if len(selected) != 1 or not selected[0]["candidate"].startswith("C16 "):
        raise AssertionError("next-preflight selection mismatch")

    corpus = "\n".join(path.read_text(errors="ignore") for path in package.rglob("*") if path.is_file() and path.suffix in {".md", ".csv", ".json"})
    secret_patterns = [r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", r"(?i)telegram[_ -]?bot[_ -]?token\s*[:=]\s*[^,\s]+", r"(?i)client_secret\s*[:=]\s*[^,\s]+"]
    findings = sum(bool(re.search(pattern, corpus)) for pattern in secret_patterns)
    if findings:
        raise AssertionError(f"secret findings: {findings}")
    if summary["protected_outcomes_opened"] or summary["economic_outputs_computed"]:
        raise AssertionError("prohibited work status")

    return {
        "status": "approve",
        "manifest_rows_verified": len(manifest),
        "definition_rows_verified": len(metrics),
        "prior_lineages_verified": 14,
        "protected_outcomes_opened": 0,
        "economic_outputs_computed": 0,
        "secret_findings": 0,
        "archive_sha256": summary["archive_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    print(json.dumps(validate(args.package_root.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
