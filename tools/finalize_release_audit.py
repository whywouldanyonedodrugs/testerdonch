#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_map_hash(obj: Dict[str, str]) -> str:
    """
    Deterministic hash for the checksum map itself.
    This avoids recursive self-reference of hashing a file that contains its own hash.
    """
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Finalize release governance artifacts: checksums + release_manifest."
    )
    p.add_argument("--release-id", required=True)
    p.add_argument("--release-dir", required=True, help="Path to release root directory.")
    p.add_argument("--bundle-dir", required=True, help="Path to bundle directory (meta_export...).")
    p.add_argument("--target-threshold", type=float, default=None)
    p.add_argument("--target-scope", default="")
    p.add_argument("--notes", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    release_dir = Path(args.release_dir).resolve()
    bundle_dir = Path(args.bundle_dir).resolve()

    if not release_dir.exists():
        raise SystemExit(f"release-dir not found: {release_dir}")
    if not bundle_dir.exists():
        raise SystemExit(f"bundle-dir not found: {bundle_dir}")

    required = [
        "model.joblib",
        "feature_manifest.json",
        "calibration.json",
        "thresholds.json",
        "sizing_curve.csv",
        "deployment_config.json",
        "golden_features.parquet",
        "regimes_report.json",
        "bundle_smoke.json",
    ]
    missing = [fn for fn in required if not (bundle_dir / fn).exists()]
    if missing:
        raise SystemExit(f"bundle missing required files: {missing}")

    checksums_path = bundle_dir / "checksums_sha256.json"
    dep_cfg = _read_json(bundle_dir / "deployment_config.json")
    feat_manifest = _read_json(bundle_dir / "feature_manifest.json")
    smoke = _read_json(bundle_dir / "bundle_smoke.json")

    features = feat_manifest.get("features", {})
    n_num = len(features.get("numeric_cols", []) or [])
    n_cat = len(features.get("cat_cols", []) or [])
    threshold = dep_cfg.get("decision", {}).get("threshold")
    scope = dep_cfg.get("decision", {}).get("scope")
    if args.target_threshold is not None:
        threshold = float(args.target_threshold)
    if args.target_scope.strip():
        scope = args.target_scope.strip()

    manifest = {
        "release_id": args.release_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_dir": str(release_dir),
        "bundle_dir": str(bundle_dir),
        "bundle_files_required": required,
        "checksums_path": str(checksums_path),
        "checksums_self_hash_rule": (
            "checksums_sha256.json entry is sha256(canonical JSON of checksums map without pretty-print variance)."
        ),
        "schema": {
            "numeric_cols": int(n_num),
            "cat_cols": int(n_cat),
            "raw_cols": int(n_num + n_cat),
        },
        "decision": {
            "threshold": threshold,
            "scope": scope,
            "criterion": dep_cfg.get("decision", {}).get("criterion"),
        },
        "smoke": smoke,
        "notes": args.notes,
    }

    release_manifest_path = release_dir / "release_manifest.json"
    _write_json(release_manifest_path, manifest)
    # Convenience copy next to bundle for handoffs that move only bundle dir.
    _write_json(bundle_dir / "release_manifest.json", manifest)

    # IMPORTANT: compute checksums AFTER writing release manifests to avoid stale hash entries.
    files: List[Path] = sorted([p for p in bundle_dir.iterdir() if p.is_file() and p.name != checksums_path.name])
    checksum_map: Dict[str, str] = {p.name: _sha256_file(p) for p in files}
    # Deterministic self-hash entry rule documented in release_manifest.
    checksum_map[checksums_path.name] = _canonical_map_hash(checksum_map)
    _write_json(checksums_path, checksum_map)

    print(f"[finalize_release_audit] updated checksums: {checksums_path}")
    print(f"[finalize_release_audit] wrote release manifest: {release_manifest_path}")
    print(f"[finalize_release_audit] raw_cols={n_num+n_cat} scope={scope} threshold={threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
