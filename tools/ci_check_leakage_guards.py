#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from feature_registry import registry_feature_names, registry_rows_by_test_type


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    file: str
    line: int = 0


def _rel_or_abs(path: Path) -> str:
    p = path.resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except Exception:
        return str(p)


def _parse_source(path: Path) -> ast.AST:
    src = path.read_text(encoding="utf-8")
    return ast.parse(src, filename=str(path))


def _literal_str(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_bool(node: Optional[ast.AST]) -> Optional[bool]:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return bool(node.value)
    return None


def _iter_merge_asof_calls(tree: ast.AST) -> List[Tuple[int, Dict[str, ast.AST]]]:
    out: List[Tuple[int, Dict[str, ast.AST]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_merge_asof = False
        if isinstance(fn, ast.Attribute) and fn.attr == "merge_asof":
            is_merge_asof = True
        elif isinstance(fn, ast.Name) and fn.id == "merge_asof":
            is_merge_asof = True
        if not is_merge_asof:
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        out.append((getattr(node, "lineno", 0), kw))
    return out


def _function_segment(src: str, tree: ast.AST, fn_name: str) -> Optional[str]:
    for node in tree.body:  # type: ignore[attr-defined]
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return ast.get_source_segment(src, node)
    return None


def check_regime_filtered_only(regime_path: Path) -> List[Violation]:
    violations: List[Violation] = []
    src = regime_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(regime_path))

    required_fns = ["compute_markov_regime_4h", "compute_daily_combined_regime"]
    for fn_name in required_fns:
        seg = _function_segment(src, tree, fn_name)
        if not seg:
            violations.append(
                Violation(
                    code="REGIME_FN_MISSING",
                    message=f"required function not found: {fn_name}",
                    file=_rel_or_abs(regime_path),
                    line=0,
                )
            )
            continue
        if "filtered_marginal_probabilities" not in seg:
            violations.append(
                Violation(
                    code="REGIME_FILTERED_REQUIRED",
                    message=f"{fn_name} must use filtered_marginal_probabilities",
                    file=_rel_or_abs(regime_path),
                    line=0,
                )
            )
        if "smoothed_marginal_probabilities" in seg:
            violations.append(
                Violation(
                    code="REGIME_SMOOTHED_FORBIDDEN",
                    message=f"{fn_name} uses smoothed_marginal_probabilities (look-ahead leak)",
                    file=_rel_or_abs(regime_path),
                    line=0,
                )
            )
    return violations


def check_merge_asof_contract(path: Path, require_tolerance: bool) -> List[Violation]:
    violations: List[Violation] = []
    tree = _parse_source(path)
    rel = _rel_or_abs(path)
    calls = _iter_merge_asof_calls(tree)
    if not calls:
        violations.append(
            Violation(
                code="MERGE_ASOF_MISSING",
                message="expected at least one merge_asof call",
                file=rel,
                line=0,
            )
        )
        return violations

    for line, kw in calls:
        direction = _literal_str(kw.get("direction"))
        exact = _literal_bool(kw.get("allow_exact_matches"))
        if direction != "backward":
            violations.append(
                Violation(
                    code="MERGE_ASOF_DIRECTION",
                    message=f"merge_asof must use direction='backward' (found: {direction})",
                    file=rel,
                    line=line,
                )
            )
        if exact is not True:
            violations.append(
                Violation(
                    code="MERGE_ASOF_EXACT",
                    message=f"merge_asof must set allow_exact_matches=True (found: {exact})",
                    file=rel,
                    line=line,
                )
            )
        if require_tolerance and "tolerance" not in kw:
            violations.append(
                Violation(
                    code="MERGE_ASOF_TOL",
                    message="merge_asof must pass explicit tolerance keyword in this file",
                    file=rel,
                    line=line,
                )
            )
    return violations


def check_htf_resample_contract(indicators_path: Path) -> List[Violation]:
    """
    Static guard: resample_ohlcv and MACD-TF resampling must use close-labeled bins.
    """
    violations: List[Violation] = []
    rel = _rel_or_abs(indicators_path)
    src = indicators_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(indicators_path))

    for fn_name in ("resample_ohlcv", "macd_histogram_tf"):
        seg = _function_segment(src, tree, fn_name)
        if not seg:
            violations.append(
                Violation(
                    code="HTF_FN_MISSING",
                    message=f"required function not found: {fn_name}",
                    file=rel,
                    line=0,
                )
            )
            continue
        if 'label="right"' not in seg or 'closed="right"' not in seg:
            violations.append(
                Violation(
                    code="HTF_RESAMPLE_LABEL",
                    message=f"{fn_name} must use resample(..., label='right', closed='right')",
                    file=rel,
                    line=0,
                )
            )
    return violations


def check_no_left_labeled_resample(path: Path) -> List[Violation]:
    violations: List[Violation] = []
    src = path.read_text(encoding="utf-8")
    rel = _rel_or_abs(path)
    if 'label="left"' in src or "label='left'" in src:
        violations.append(
            Violation(
                code="LEFT_LABELLED_RESAMPLE",
                message="active decision-time code must not use label='left' resampling",
                file=rel,
                line=0,
            )
        )
    if 'closed="left"' in src or "closed='left'" in src:
        violations.append(
            Violation(
                code="LEFT_CLOSED_RESAMPLE",
                message="active decision-time code must not use closed='left' resampling",
                file=rel,
                line=0,
            )
        )
    return violations


def check_no_inline_ffill_alignment(path: Path) -> List[Violation]:
    violations: List[Violation] = []
    src = path.read_text(encoding="utf-8")
    rel = _rel_or_abs(path)
    if 'method="ffill"' in src or "method='ffill'" in src:
        violations.append(
            Violation(
                code="INLINE_FFILL_ALIGNMENT",
                message="active decision-time code must use canonical point-in-time helpers instead of inline reindex(..., method='ffill')",
                file=rel,
                line=0,
            )
        )
    if ".ffill().bfill()" in src:
        violations.append(
            Violation(
                code="FFILL_BFILL_LEAK",
                message="active decision-time code must not backfill from future values",
                file=rel,
                line=0,
            )
        )
    return violations


def _load_manifest_features(path: Path) -> Tuple[List[str], List[str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    feats = obj.get("features", obj)
    if not isinstance(feats, dict):
        return [], []
    num = list(feats.get("numeric_cols") or feats.get("num_cols") or [])
    cat = list(feats.get("cat_cols") or [])
    return num, cat


def check_feature_registry_contract(repo_root: Path) -> List[Violation]:
    violations: List[Violation] = []
    reg_names = set(registry_feature_names())
    rows_by_test = registry_rows_by_test_type()
    rel = "feature_registry.py"

    for required in (
        "decision_bar_close",
        "htf_last_closed",
        "daily_snapshot",
        "truncation_equivalence",
        "exogenous_publish",
        "derived_from_causal",
        "trade_history",
        "same_timestamp_cross_sectional",
    ):
        if not rows_by_test.get(required):
            violations.append(
                Violation(
                    code="FEATURE_REGISTRY_TESTTYPE",
                    message=f"feature registry missing required test_type bucket: {required}",
                    file=rel,
                    line=0,
                )
            )

    manifest_paths = [
        repo_root / "research_outputs" / "07_deployment_artifacts" / "feature_manifest.json",
        repo_root / "research_outputs_rs30" / "07_deployment_artifacts" / "feature_manifest.json",
    ]
    for mp in manifest_paths:
        if not mp.exists():
            continue
        num, cat = _load_manifest_features(mp)
        missing = sorted(set(num + cat) - reg_names)
        if missing:
            violations.append(
                Violation(
                    code="FEATURE_REGISTRY_MISSING",
                    message=f"registry missing manifest features from {mp.name}: {missing[:10]}",
                    file=rel,
                    line=0,
                )
            )

    extra_required = {
        "recent_winrate_20",
        "recent_winrate_50",
        "recent_winrate_ewm_20",
        "asset4h_share_at_timestamp",
        "liq30_eqw_ret_72h",
        "liq30_pos_share_24h",
        "liq30_new_20d_high_share",
    }
    missing_extra = sorted(extra_required - reg_names)
    if missing_extra:
        violations.append(
            Violation(
                code="FEATURE_REGISTRY_ACTIVE_GAP",
                message=f"registry missing active non-manifest features: {missing_extra}",
                file=rel,
                line=0,
            )
        )
    return violations


def run_checks(repo_root: Path) -> Dict[str, object]:
    violations: List[Violation] = []

    regime_path = (repo_root / "regime_detector.py").resolve()
    violations.extend(check_regime_filtered_only(regime_path))

    merge_specs = [
        ("backtester.py", True),
        ("fill_entry_quality_features.py", True),
        ("backfill_trade_features.py", True),
        ("pull.py", False),
        ("tools/run_v3_frozen_oos.py", True),
    ]
    for rel, req_tol in merge_specs:
        p = (repo_root / rel).resolve()
        if not p.exists():
            violations.append(
                Violation(
                    code="FILE_MISSING",
                    message=f"required file missing: {rel}",
                    file=rel,
                    line=0,
                )
            )
            continue
        violations.extend(check_merge_asof_contract(p, require_tolerance=req_tol))

    indicators_path = (repo_root / "indicators.py").resolve()
    if indicators_path.exists():
        violations.extend(check_htf_resample_contract(indicators_path))
    else:
        violations.append(
            Violation(
                code="FILE_MISSING",
                message="required file missing: indicators.py",
                file="indicators.py",
                line=0,
            )
        )

    violations.extend(check_feature_registry_contract(repo_root))

    left_label_specs = [
        "backtester.py",
        "backtester_b_gemin.py",
        "live/live_trader.py",
        "live/regime_features.py",
        "live/feature_builder.py",
        "fill_entry_quality_features.py",
        "scout.py",
        "backfill_trade_features.py",
        "regime_detector.py",
        "tools/run_v3_frozen_oos.py",
    ]
    for rel in left_label_specs:
        p = (repo_root / rel).resolve()
        if not p.exists():
            continue
        violations.extend(check_no_left_labeled_resample(p))
        violations.extend(check_no_inline_ffill_alignment(p))

    report = {
        "status": "ok" if not violations else "fail",
        "checked_files": ["regime_detector.py", "indicators.py", "feature_registry.py"] + [x[0] for x in merge_specs] + left_label_specs,
        "violation_count": len(violations),
        "violations": [
            {"code": v.code, "message": v.message, "file": v.file, "line": int(v.line)}
            for v in violations
        ],
    }
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CI leakage guards for regime probabilities and asof merge alignment.")
    p.add_argument("--repo-root", default=str(REPO_ROOT))
    p.add_argument("--out", default="results/ci_leakage_guards/report.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.repo_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    report = run_checks(root)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = report["status"]
    print(f"[jt008] status={status} violations={report['violation_count']} out={out}", flush=True)
    if status != "ok":
        for row in report["violations"]:
            print(
                f"[jt008] {row['file']}:{row['line']} {row['code']} {row['message']}",
                flush=True,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
