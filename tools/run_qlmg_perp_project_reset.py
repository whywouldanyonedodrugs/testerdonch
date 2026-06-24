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
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_RUN_ID = "phase_qlmg_perp_project_reset_20260624_v1"
STAGES = (
    "repo-and-artifact-inventory",
    "legacy-archive-plan",
    "documentation-reset",
    "infrastructure-reuse-audit",
    "long-short-engine-audit",
    "data-inventory",
    "data-acquisition-feasibility",
    "new-sealed-policy",
    "strategy-contract-scaffold",
    "test-plan",
    "final-report",
    "all",
)
DATA_ROOTS = {
    "bybit_5m_ohlcv": Path("/opt/parquet/5m"),
    "bybit_1m_hot": Path("/opt/parquet/1m_hot"),
    "bybit_1m": Path("/opt/parquet/1m"),
    "bybit_context_5m": Path("/opt/parquet/bybit_context_5m"),
}
DOC_PATHS = [
    REPO / "docs/QLMG_PERP_BACKTESTING_MANUAL.md",
    REPO / "docs/QLMG_PERP_PROJECT_STATE.md",
    REPO / "docs/QLMG_PERP_DATA_CONTRACT.md",
    REPO / "docs/QLMG_PERP_STRATEGY_CATALOG.md",
    REPO / "docs/QLMG_PERP_VALIDATION_PROTOCOL.md",
    REPO / "docs/QLMG_PERP_MIGRATION_FROM_DONCH.md",
]
DONCH_LEGACY_PATTERNS = ("phase_v3", "phase_s1", "phase_state_transition", "phase_rebound", "phase_event_crowding", "phase_reset", "phase_oldcfg", "phase2_", "phase3_", "phase4_", "phase5_", "phase10_", "phase11_", "phase13_", "phase15_", "phase17", "phase18", "phase20", "phase_multivariant")
SAFE_CACHE_NAMES = {"shared_cache", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
DURABLE_NAME_FRAGMENTS = ("manifest", "contract", "report", "ledger", "parquet", "csv", "json", "md", "shortlist", "summary", "trades", "path_stats", "entry_ledger")
STRATEGY_FAMILIES = {
    "D1_low_volume_short_horizon_reversal": {"direction": "long", "tier": "Tier C small tradable", "data": "5m OHLCV, turnover, lifecycle, costs; 1m preferred for final execution", "mechanism": "thin-name mean reversion after local liquidity vacuum without leverage/liquidation denial", "blocker": "cannot promote without honest spread/slippage and delist handling"},
    "D3_small_cap_liquidity_shock_bounce": {"direction": "long", "tier": "Tier B/C", "data": "OHLCV, turnover, mark/index/funding/OI, liquidation proxy if available", "mechanism": "forced selling/liquidity shock followed by stabilization", "blocker": "blocked if leverage clearing cannot be observed or costs dominate"},
    "A1_large_liquid_smooth_path_continuation_breakout": {"direction": "long", "tier": "Tier A liquid", "data": "OHLCV, turnover, ATR/ADR, leader ranks, fees/funding", "mechanism": "Qullamaggie-style leader breakout after prior thrust and tight base", "blocker": "blocked if breakout is raw extension without consolidation or execution realism"},
    "A2_prior_high_proximity_momentum": {"direction": "long", "tier": "Tier A/B", "data": "OHLCV, relative strength, prior highs, turnover", "mechanism": "controlled pullback/reclaim near prior high in supportive regime", "blocker": "blocked if only broad Donchian continuation is recreated"},
    "B1_sector_episodic_pivot": {"direction": "long", "tier": "theme basket", "data": "sector map, catalyst timestamps, OHLCV, turnover", "mechanism": "theme ignition after catalyst and early volume confirmation", "blocker": "blocked without point-in-time sector/catalyst data"},
    "C2_post_catalyst_continuation_base": {"direction": "long", "tier": "event names", "data": "catalyst DB, OHLCV, turnover, sidecar optional", "mechanism": "post-event base instead of event-day chase", "blocker": "blocked if catalyst timestamp is not point-in-time"},
    "E1_liquidation_flush_long": {"direction": "long", "tier": "perp names", "data": "mark/index/funding/OI, liquidation data preferred, OHLCV", "mechanism": "leverage cleared, price stabilizes, no anticipatory catch", "blocker": "blocked without liquidation/clearing evidence or mark-price safety"},
    "F1_parabolic_blowoff_short": {"direction": "short", "tier": "overextended names", "data": "OHLCV, VWAP, borrow/perp funding, mark liquidation model", "mechanism": "backside confirmation after parabolic extension", "blocker": "blocked if anticipatory shorting or liquidation risk is unmodeled"},
    "G1_failed_continuation_breakout_short": {"direction": "short", "tier": "failed leaders", "data": "OHLCV, breakout levels, volume/turnover, funding/OI optional", "mechanism": "failed breakout traps late longs and confirms downside", "blocker": "blocked without short stop/TP and gap risk handling"},
    "E3_price_oi_matrix_overlay": {"direction": "both/overlay", "tier": "all eligible", "data": "price, OI, funding, turnover", "mechanism": "state conditioner for parent entries, not standalone alpha", "blocker": "blocked if used as standalone direction model"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0 reset for QLMG-inspired crypto perp research.")
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--run-root", default="")
    p.add_argument("--results-root", default="results/rebaseline")
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--dry-run-deletion", action="store_true")
    p.add_argument("--skip-metadata-probes", action="store_true")
    p.add_argument("--metadata-timeout", type=float, default=12.0)
    return p.parse_args()


def stage_list(stage: str) -> list[str]:
    return [s for s in STAGES if s != "all"] if stage == "all" else [stage]


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
    if fieldnames is None:
        keys: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict(r))


def append_command(run_root: Path, stage: str, argv: Sequence[str]) -> None:
    p = run_root / "command_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts_utc": utc_now(), "stage": stage, "argv": list(argv), "cwd": str(REPO)}, sort_keys=True) + "\n")


def mark_done(run_root: Path, stage: str) -> None:
    write_text(run_root / "stage_status" / f"{stage}.done", utc_now())


def shell(args: Sequence[str], *, cwd: Path = REPO, timeout: float = 60.0) -> str:
    try:
        p = subprocess.run(args, cwd=str(cwd), text=True, capture_output=True, timeout=timeout, check=False)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def sha256_file(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = limit_bytes
        while True:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if size <= 0:
                break
            b = f.read(size)
            if not b:
                break
            h.update(b)
            if remaining is not None:
                remaining -= len(b)
    return h.hexdigest()


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def file_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*") if _.is_file())


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def human_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def parquet_metadata(path: Path) -> dict[str, Any]:
    out = {"row_count": None, "timestamp_min": None, "timestamp_max": None, "schema_fields": "", "schema_fingerprint": "", "error": ""}
    if pq is None:
        out["error"] = "pyarrow_unavailable"
        return out
    try:
        pf = pq.ParquetFile(path)
        schema_names = list(pf.schema_arrow.names)
        out["row_count"] = int(pf.metadata.num_rows)
        out["schema_fields"] = "|".join(schema_names)
        out["schema_fingerprint"] = hashlib.sha256(out["schema_fields"].encode()).hexdigest()[:16]
        ts_col = "timestamp" if "timestamp" in schema_names else ("decision_ts" if "decision_ts" in schema_names else None)
        mins: list[pd.Timestamp] = []
        maxs: list[pd.Timestamp] = []
        if ts_col:
            idx = schema_names.index(ts_col)
            for rg_i in range(pf.metadata.num_row_groups):
                col = pf.metadata.row_group(rg_i).column(idx)
                stats = col.statistics
                if stats is not None and stats.min is not None and stats.max is not None:
                    mins.append(pd.to_datetime(stats.min, utc=True, errors="coerce"))
                    maxs.append(pd.to_datetime(stats.max, utc=True, errors="coerce"))
            if not mins or not maxs:
                s = pd.read_parquet(path, columns=[ts_col])[ts_col]
                mins.append(pd.to_datetime(s.min(), utc=True, errors="coerce"))
                maxs.append(pd.to_datetime(s.max(), utc=True, errors="coerce"))
        clean_min = [x for x in mins if pd.notna(x)]
        clean_max = [x for x in maxs if pd.notna(x)]
        if clean_min:
            out["timestamp_min"] = str(min(clean_min))
        if clean_max:
            out["timestamp_max"] = str(max(clean_max))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def scan_parquet_root(root: Path, dataset: str, max_files: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not root.exists():
        return [], {"dataset": dataset, "path": str(root), "exists": False, "file_count": 0, "symbol_count": 0, "total_rows": 0, "earliest_timestamp": "", "latest_timestamp": "", "schema_fields": "", "size_bytes": 0}
    files = sorted(root.glob("*.parquet"))
    if max_files is not None:
        files = files[:max_files]
    rows = []
    total_rows = 0
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    schemas: dict[str, int] = {}
    for f in files:
        meta = parquet_metadata(f)
        rows.append({"dataset": dataset, "symbol": f.stem, "path": str(f), "size_bytes": f.stat().st_size, **meta})
        if meta.get("row_count") is not None:
            total_rows += int(meta["row_count"])
        if meta.get("timestamp_min"):
            mins.append(pd.to_datetime(meta["timestamp_min"], utc=True, errors="coerce"))
        if meta.get("timestamp_max"):
            maxs.append(pd.to_datetime(meta["timestamp_max"], utc=True, errors="coerce"))
        schema = str(meta.get("schema_fields") or "")
        schemas[schema] = schemas.get(schema, 0) + 1
    clean_min = [x for x in mins if pd.notna(x)]
    clean_max = [x for x in maxs if pd.notna(x)]
    summary = {
        "dataset": dataset,
        "path": str(root),
        "exists": True,
        "file_count": len(files),
        "symbol_count": len({r["symbol"] for r in rows}),
        "total_rows": total_rows,
        "earliest_timestamp": str(min(clean_min)) if clean_min else "",
        "latest_timestamp": str(max(clean_max)) if clean_max else "",
        "schema_fields": max(schemas.items(), key=lambda kv: kv[1])[0] if schemas else "",
        "schema_variant_count": len(schemas),
        "size_bytes": dir_size(root),
        "suitable_for_screening": bool(files),
        "suitable_for_final_selection": "requires_contract_specific_quality_audit" if files else "no",
        "immutable_or_mutable": "mutable_local_store_unless_manifest_locked",
    }
    return rows, summary


def stage_repo_inventory(run_root: Path, args: argparse.Namespace) -> None:
    out = run_root / "repo"
    out.mkdir(parents=True, exist_ok=True)
    write_text(out / "git_status_short.txt", shell(["git", "status", "--short"], timeout=20))
    write_text(out / "git_diff_stat.txt", shell(["git", "diff", "--stat"], timeout=20))
    write_text(out / "current_branch_and_head.txt", f"branch: {shell(['git','rev-parse','--abbrev-ref','HEAD'])}\nhead: {shell(['git','rev-parse','HEAD'])}")
    write_text(out / "top_level_tree.txt", shell(["bash", "-lc", "find . -maxdepth 1 -mindepth 1 -printf '%f\\n' | sort"], timeout=20))
    code_patterns = [
        ("backtest_runner", "tools/run_*.py"), ("core_backtester", "backtester*.py"), ("feature", "*feature*.py"),
        ("live", "live/**/*.py"), ("tests", "unit_tests/test_*.py"), ("docs", "docs/*.md"),
    ]
    rows = []
    for category, pattern in code_patterns:
        for p in sorted(REPO.glob(pattern)):
            if p.is_file() and ".venv" not in p.parts:
                rows.append({"category": category, "path": safe_rel(p), "size_bytes": p.stat().st_size, "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()})
    write_csv(out / "key_code_inventory.csv", rows)
    art_rows = []
    for base in [REPO / "results/rebaseline", REPO / "reports/rebaseline", REPO / "reports", REPO / "artifacts", REPO / "archive"]:
        if not base.exists():
            continue
        for p in sorted([x for x in base.iterdir() if x.is_dir()]):
            art_rows.append({"path": safe_rel(p), "category": "legacy_result_or_report", "size_bytes": dir_size(p), "size_human": human_bytes(dir_size(p)), "file_count": file_count(p), "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()})
    write_csv(out / "key_artifact_inventory.csv", art_rows)
    write_text(out / "repo_state.md", f"""# Repo State

- generated_at_utc: `{utc_now()}`
- branch/head: see `current_branch_and_head.txt`
- worktree: dirty; this is recorded, not cleaned.
- purpose: Phase 0 QLMG project reset inventory only.

## Notes
Old Donch/V3/S1/state-transition files are treated as legacy evidence and reusable infrastructure, not active alpha candidates.
""")


def is_legacy_artifact(path: Path) -> bool:
    name = path.name.lower()
    return any(p.lower() in name for p in DONCH_LEGACY_PATTERNS)


def is_safe_cache_candidate(path: Path) -> tuple[bool, str]:
    parts = set(path.parts)
    name = path.name
    if name in SAFE_CACHE_NAMES:
        if any(fragment in str(path).lower() for fragment in DURABLE_NAME_FRAGMENTS):
            return False, "name_contains_durable_fragment"
        return True, f"allowlisted_cache_name_{name}"
    return False, "not_allowlisted_cache"


def stage_legacy_archive(run_root: Path, args: argparse.Namespace) -> None:
    out = run_root / "archive"
    out.mkdir(parents=True, exist_ok=True)
    artifact_index = []
    do_not_delete = []
    archive_rows = []
    for root in [REPO / "results/rebaseline", REPO / "reports/rebaseline", REPO / "reports", REPO / "artifacts", REPO / "archive"]:
        if not root.exists():
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            legacy = is_legacy_artifact(p)
            row = {"path": safe_rel(p), "is_legacy_donch_related": legacy, "size_bytes": dir_size(p), "size_human": human_bytes(dir_size(p)), "file_count": file_count(p), "action": "index_only_preserve" if legacy else "preserve"}
            artifact_index.append(row)
            if legacy:
                archive_rows.append({**row, "reason": "legacy strategy result; preserve reproducibility; not active"})
                do_not_delete.append(safe_rel(p))
    write_csv(out / "legacy_artifact_index.csv", artifact_index)
    write_csv(out / "safe_to_archive_or_index_only.csv", archive_rows)
    write_text(out / "do_not_delete_list.txt", "\n".join(sorted(set(do_not_delete))))
    candidates = []
    for root in [REPO / "results/rebaseline", REPO / "reports", REPO / "artifacts", REPO / ".pytest_cache"]:
        if not root.exists():
            continue
        paths = [root] if root.name in SAFE_CACHE_NAMES else list(root.rglob("*"))
        for p in paths:
            if not p.is_dir():
                continue
            ok, reason = is_safe_cache_candidate(p)
            if ok:
                candidates.append({"path": str(p), "relative_path": safe_rel(p), "size_bytes": dir_size(p), "size_human": human_bytes(dir_size(p)), "file_count": file_count(p), "safe_delete_reason": reason})
    write_csv(out / "safe_cache_deletion_candidates.csv", candidates)
    deleted = []
    for row in candidates:
        p = Path(row["path"])
        status = "dry_run_not_deleted" if args.dry_run_deletion else "deleted"
        err = ""
        if not args.dry_run_deletion:
            try:
                shutil.rmtree(p)
            except Exception as exc:
                status = "delete_failed"
                err = f"{type(exc).__name__}: {exc}"
        deleted.append({**row, "status": status, "error": err, "deleted_at_utc": utc_now() if status == "deleted" else ""})
    write_csv(out / "deleted_safe_cache_manifest.csv", deleted)
    write_text(out / "legacy_archive_plan.md", f"""# Legacy Archive Plan

Old Donch/V3/S1/state-transition artifacts are legacy reference only. Phase 0 uses an index-only archive by default because moving huge run roots risks breaking artifact references.

- physical movement of run roots: `false`
- raw market data moved: `false`
- safe-cache deletion dry run: `{args.dry_run_deletion}`
- safe-cache candidates: `{len(candidates)}`
- deleted rows: `{sum(1 for r in deleted if r['status'] == 'deleted')}`

Durable manifests, contracts, reports, ledgers, path stats, parquet results, summaries, and full run roots are on the do-not-delete list.
""")
    write_text(out / "deletion_guard_report.md", f"""# Deletion Guard Report

Deletion was restricted to allowlisted rebuildable cache directory names: `{', '.join(sorted(SAFE_CACHE_NAMES))}`.

The guard refuses paths whose names include durable evidence fragments such as `{', '.join(DURABLE_NAME_FRAGMENTS)}`. The manifest `deleted_safe_cache_manifest.csv` is the audit trail for any actual deletions.
""")


def qlmg_docs() -> dict[Path, str]:
    return {
        DOC_PATHS[0]: """# QLMG Perp Backtesting Manual

Status: active program documentation as of 2026-06-24 UTC.

The active research program is Qullamaggie-inspired crypto perpetual research across long and short Bybit USDT linear perpetuals. Donchian/V3/S1 work is legacy reference only.

Core rules:
- long and short backtests are required;
- mark, index, last, fill, and liquidation prices are different objects;
- OHLCV-only results are screening only;
- final promotion requires funding cashflow, exchange fees, spread/slippage, symbol lifecycle, delisting, and mark-price liquidation modeling;
- small-cap and lower-liquidity symbols may be strategically valuable for small accounts, but only with conservative execution modeling;
- no live trading is authorized by this document.
""",
        DOC_PATHS[1]: """# QLMG Perp Project State

Old active direction: Donchian/V3/S1 long-only breakout/pullback continuation.

New active direction: QLMG-inspired long/short crypto perpetual strategy research for small-account aggressive growth.

Current stage: Phase 0 reset, infrastructure audit, data inventory, documentation reset, short-support readiness, and sealed-policy drafting. No alpha sweeps or live deployment are authorized.
""",
        DOC_PATHS[2]: """# QLMG Perp Data Contract

Primary venue: Bybit linear USDT perpetuals. Binance USDⓈ-M data may be secondary research support.

Required semantics:
- all features must be known at decision_ts;
- higher-timeframe bars use last-closed bars only;
- sidecar mark/index/premium/LSR source_close_ts must be <= decision_ts and fresh;
- no last-price substitution for mark/index;
- no default premium=0 or LSR=1;
- listing/delisting/status must be handled point-in-time when available, otherwise marked proxy/unknown;
- OHLCV-only is screening, not final selection.
""",
        DOC_PATHS[3]: """# QLMG Perp Strategy Catalog

Priority families for future Phase 1, not executed in Phase 0:
1. D1 low-volume short-horizon reversal.
2. D3 small-cap liquidity-shock bounce.
3. A1 large/liquid smooth-path continuation breakout.
4. A2 prior-high proximity momentum.
5. B1 sector episodic pivot/theme ignition.
6. C2 post-catalyst continuation base.
7. E1 liquidation flush long.
8. F1 parabolic blowoff short.
9. G1 failed continuation breakout short.
10. E3 price + OI matrix overlay.

Funding, OI, premium, and mark/index are state variables, conditioners, vetoes, or risk scalers unless a separate contract proves otherwise.
""",
        DOC_PATHS[4]: """# QLMG Perp Validation Protocol

The sealed-data policy is reset for this new program. Old Donch contamination does not automatically invalidate all old dates, but overlapping momentum/perp-state concepts mean prior windows are not perfectly clean.

Phase 0 does not run final validation. First-pass screening must use older data with walk-forward/purged validation. Any final candidate must be frozen before final holdout access.
""",
        DOC_PATHS[5]: """# Migration From Donch To QLMG Perp Research

Deprecated as active alpha direction:
- Donchian/V3/S1 long-only continuation as the main development path.

Preserved:
- data loaders, sidecar contracts, funding/cost accounting lessons, CPCV/walk-forward tooling, audit/report discipline, and selected legacy artifacts as reference evidence.

No longer valid assumptions:
- long-only engine suffices;
- OHLCV-only promotion is acceptable;
- liquidation can be treated as a stop;
- current active universe is a safe historical universe.

Legacy artifacts are indexed under the Phase 0 archive outputs.
""",
    }


def stage_documentation_reset(run_root: Path, args: argparse.Namespace) -> None:
    docs_out = run_root / "docs"
    rows = []
    for path, text in qlmg_docs().items():
        write_text(path, text)
        rows.append({"path": safe_rel(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_csv(docs_out / "created_docs_manifest.csv", rows)
    write_text(docs_out / "documentation_reset_report.md", "# Documentation Reset\n\nCreated QLMG-specific docs and migration note. Old Donch/V3/S1 docs remain legacy references where present, not active strategy contracts.")


def component_rows() -> list[dict[str, str]]:
    return [
        {"component": "OHLCV parquet loaders", "status": "reusable_with_minor_modification", "reason": "local 5m/1m_hot stores exist; need lifecycle and long/short contract wrappers"},
        {"component": "Bybit sidecar context loader", "status": "reusable_with_minor_modification", "reason": "raw mark/index/premium/LSR contract exists; must extend to QLMG family contracts"},
        {"component": "funding cashflow model", "status": "reusable_with_minor_modification", "reason": "parent-first repair implemented long funding; must add side-aware long/short tests"},
        {"component": "cost model", "status": "reusable_with_minor_modification", "reason": "lane priors/decomposed costs exist; need QLMG small-cap/slippage calibration"},
        {"component": "portfolio ledger", "status": "reusable_with_major_modification", "reason": "legacy engine is long-biased; needs long/short exposure/margin semantics"},
        {"component": "liquidation modeling", "status": "missing", "reason": "mark-price liquidation and margin utilization not complete"},
        {"component": "symbol lifecycle handling", "status": "reusable_with_major_modification", "reason": "proxy first/last timestamps exist; official historical status needed"},
        {"component": "sector/catalyst support", "status": "missing", "reason": "no point-in-time sector/catalyst database established"},
        {"component": "orderbook/spread/depth support", "status": "partial", "reason": "some intrabar fixtures exist; historical depth coverage not proven"},
        {"component": "validation/CPCV tooling", "status": "reusable_with_minor_modification", "reason": "CPCV/purged patterns exist in prior runners"},
        {"component": "reporting artifacts", "status": "reusable_without_change", "reason": "manifest/report pattern is established"},
    ]


def stage_infrastructure_audit(run_root: Path, args: argparse.Namespace) -> None:
    out = run_root / "infrastructure"
    rows = component_rows()
    write_csv(out / "reusable_components.csv", [r for r in rows if r["status"] in {"reusable_without_change", "reusable_with_minor_modification"}])
    write_csv(out / "components_to_modify.csv", [r for r in rows if "modification" in r["status"] or r["status"] == "partial"])
    write_csv(out / "components_to_deprecate.csv", [{"component": "V3/S1 alpha-specific assumptions", "status": "deprecated", "reason": "legacy strategy direction no longer active"}])
    write_csv(out / "missing_components.csv", [r for r in rows if r["status"] == "missing"])
    write_text(out / "infrastructure_reuse_audit.md", "# Infrastructure Reuse Audit\n\n" + pd.DataFrame(rows).to_markdown(index=False))


def stage_short_audit(run_root: Path, args: argparse.Namespace) -> None:
    out = run_root / "shorts"
    checks = [
        ("long_entries", "present", "legacy engine supports long entries"),
        ("short_entries", "partial", "Trade dataclass has side field and fixtures mention shorts, but main path logic is long-biased"),
        ("short_stop_loss", "partial", "intrabar fixtures exist; main replay math must be audited side-aware"),
        ("short_take_profit", "partial", "needs explicit side-inverted trigger tests"),
        ("short_trailing_stop", "missing", "no complete side-aware trailing-stop contract found"),
        ("vwap_fail_reclaim_short", "missing", "strategy-specific trigger not implemented as reusable QLMG module"),
        ("funding_sign_by_side", "missing", "long funding exists; short receive/pay convention needs implementation/tests"),
        ("maker_taker_by_side", "partial", "fees exist but side/order-type fidelity incomplete"),
        ("mark_price_liquidation_long_short", "missing", "liquidation is not a stop; mark-price margin model missing"),
        ("isolated_margin_baseline", "missing", "needs explicit position margin model"),
        ("forced_liquidation", "missing", "no complete mark-price forced liquidation path"),
        ("delist_while_short", "missing", "requires lifecycle data and settlement semantics"),
        ("portfolio_long_short_exposure", "missing", "needs concurrent long/short exposure and margin constraints"),
    ]
    rows = [{"capability": a, "status": b, "evidence_or_gap": c} for a, b, c in checks]
    write_csv(out / "short_support_matrix.csv", rows)
    write_text(out / "long_short_engine_audit.md", "# Long/Short Engine Audit\n\nVerdict: short support is incomplete. Existing components can seed implementation, but QLMG short families must not run promotion-grade tests until side-aware execution, funding, and liquidation tests pass.\n\n" + pd.DataFrame(rows).to_markdown(index=False))
    write_text(out / "required_short_engine_changes.md", "# Required Short Engine Changes\n\n- Implement side-aware PnL, stop/TP/trailing logic.\n- Implement funding sign by side: longs pay positive funding, shorts receive positive funding.\n- Implement mark-price liquidation for long and short isolated-margin diagnostics.\n- Add delist/settlement handling while short.\n- Add portfolio long/short exposure and margin accounting.")
    write_text(out / "short_engine_test_plan.md", "# Short Engine Test Plan\n\nRequired tests: short stop, short TP, short same-bar ambiguity, short trailing stop, funding sign by side, mark-price liquidation long/short, forced liquidation, delist while short, order side inversion, concurrent long/short exposure.")


def data_inventory_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_symbol: list[dict[str, Any]] = []
    stores: list[dict[str, Any]] = []
    for name, root in DATA_ROOTS.items():
        rows, summary = scan_parquet_root(root, name)
        per_symbol.extend(rows)
        stores.append(summary)
    return per_symbol, stores


def stage_data_inventory(run_root: Path, args: argparse.Namespace) -> None:
    out = run_root / "data"
    per_symbol, stores = data_inventory_rows()
    write_csv(out / "data_store_inventory.csv", stores)
    write_csv(out / "symbol_coverage_by_dataset.csv", per_symbol)
    tf_rows = [{"dataset": s["dataset"], "timeframe": "5m" if "5m" in s["dataset"] else ("1m" if "1m" in s["dataset"] else "context_5m"), "earliest_timestamp": s.get("earliest_timestamp", ""), "latest_timestamp": s.get("latest_timestamp", ""), "symbol_count": s.get("symbol_count", 0), "total_rows": s.get("total_rows", 0)} for s in stores]
    write_csv(out / "timeframe_coverage.csv", tf_rows)
    context = [r for r in per_symbol if r["dataset"] == "bybit_context_5m"]
    write_csv(out / "mark_index_funding_oi_coverage.csv", context)
    write_csv(out / "instrument_metadata_inventory.csv", [{"source": "local_due_diligence_or_probe", "status": "partial_or_pending", "note": "official historical metadata not proven local; current metadata probe may exist under data_acquisition"}])
    write_csv(out / "orderbook_trade_data_inventory.csv", [{"dataset": "orderbook_public_trade_history", "status": "not_found_as_complete_local_history", "promotion_use": "blocked_until_coverage_proven"}])
    write_csv(out / "liquidation_data_inventory.csv", [{"dataset": "liquidation_history", "status": "not_found_as_complete_local_history", "promotion_use": "required_or_strongly_preferred_for_E1"}])
    write_csv(out / "event_catalyst_data_inventory.csv", [{"dataset": "catalyst_database", "status": "not_found_as_point_in_time_db", "promotion_use": "required_for_B1_C2"}])
    write_csv(out / "sector_map_inventory.csv", [{"dataset": "sector_map", "status": "not_found_as_point_in_time_db", "promotion_use": "required_for_B1"}])
    usage = shell(["bash", "-lc", "df -h /opt/testerdonch /opt/parquet 2>/dev/null; du -sh /opt/testerdonch/results /opt/testerdonch/reports /opt/parquet/* 2>/dev/null | sort -h"], timeout=120)
    write_text(out / "storage_usage_report.md", "# Storage Usage\n\n```\n" + usage + "\n```")
    write_text(out / "data_inventory_report.md", "# Data Inventory Report\n\n" + pd.DataFrame(stores).to_markdown(index=False) + "\n\nFinal-selection suitability remains contract-specific; OHLCV-only is screening only.")


def fetch_json(url: str, timeout: float) -> tuple[dict[str, Any] | None, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "donch-phase0-metadata-probe"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), "ok"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def stage_data_acquisition(run_root: Path, args: argparse.Namespace) -> None:
    out = run_root / "data_acquisition"
    cache = out / "metadata_probe_cache"
    cache.mkdir(parents=True, exist_ok=True)
    probes = []
    if not args.skip_metadata_probes:
        targets = {
            "bybit_linear_instruments_info_current": "https://api.bybit.com/v5/market/instruments-info?category=linear&limit=1000",
            "binance_usdm_exchange_info_current": "https://fapi.binance.com/fapi/v1/exchangeInfo",
        }
        for name, url in targets.items():
            obj, status = fetch_json(url, args.metadata_timeout)
            probes.append({"name": name, "url": url, "status": status, "cache_path": str(cache / f"{name}.json") if obj is not None else ""})
            if obj is not None:
                write_json(cache / f"{name}.json", obj)
    gaps = [
        {"data_type": "Bybit historical instrument metadata", "status": "missing_or_partial", "required_for": "lifecycle/survivorship/delist handling", "recommended_action": "source official snapshots or begin daily capture"},
        {"data_type": "Bybit 1m OHLCV backfill", "status": "local /opt/parquet/1m empty; 1m_hot partial", "required_for": "execution/timing and short triggers", "recommended_action": "targeted backfill for candidate windows after contracts"},
        {"data_type": "top-of-book/depth", "status": "not complete locally", "required_for": "final execution realism", "recommended_action": "target candidate symbols/windows; start live capture"},
        {"data_type": "liquidation data", "status": "not complete locally", "required_for": "E1 liquidation flush", "recommended_action": "evaluate paid/free sources; capture future"},
        {"data_type": "sector map", "status": "missing point-in-time", "required_for": "B1", "recommended_action": "create versioned sector taxonomy"},
        {"data_type": "catalyst DB", "status": "missing point-in-time", "required_for": "B1/C2", "recommended_action": "define source and timestamp policy before tests"},
    ]
    write_csv(out / "data_gap_matrix.csv", gaps)
    write_csv(out / "vendor_or_api_options.md.csv", probes)
    write_text(out / "vendor_or_api_options.md", "# Vendor Or API Options\n\nMetadata probes allowed in Phase 0 only. Historical downloads were not performed.\n\n" + pd.DataFrame(probes).to_markdown(index=False))
    write_text(out / "storage_budget_scenarios.md", "# Storage Budget Scenarios\n\n- Full deep-book/tick history for all symbols is likely infeasible.\n- Full OHLCV plus mark/index/funding/OI is high priority.\n- 1m and depth should be targeted to candidate symbols/windows.\n- Live capture should start once engineering policy is approved.")
    write_text(out / "recommended_minimum_data_plan.md", "# Recommended Minimum Data Plan\n\n1. Lock 5m OHLCV and context manifests.\n2. Obtain/record historical symbol lifecycle metadata.\n3. Backfill 1m OHLCV selectively for candidate windows.\n4. Start live top-of-book/depth/public-trade capture.\n5. Build point-in-time sector and catalyst stores before B1/C2 tests.")
    write_text(out / "acquisition_feasibility_report.md", "# Acquisition Feasibility\n\n" + pd.DataFrame(gaps).to_markdown(index=False))


def latest_from_store_inventory(run_root: Path) -> str:
    p = run_root / "data/data_store_inventory.csv"
    if not p.exists():
        return "unknown"
    df = pd.read_csv(p)
    ts = pd.to_datetime(df.get("latest_timestamp"), utc=True, errors="coerce")
    return str(ts.max()) if ts.notna().any() else "unknown"


def stage_sealed_policy(run_root: Path, args: argparse.Namespace) -> None:
    out = run_root / "seal"
    latest = latest_from_store_inventory(run_root)
    splits = {
        "policy_status": "draft_only_phase0_no_final_holdout_access",
        "latest_local_timestamp_observed": latest,
        "options": [
            {"name": "conservative_gold_holdout", "description": "latest 6 months after data inventory if not previously used for QLMG selection", "use": "final frozen validation only"},
            {"name": "screening_walkforward", "description": "older data with purged walk-forward/CPCV", "use": "Phase 1 screening"},
            {"name": "forward_collection", "description": "if final holdout is too short, collect new shadow data", "use": "future validation"},
        ],
    }
    write_json(out / "proposed_data_splits.json", splits)
    registry = {"registry_version": 1, "status": "draft", "created_at_utc": utc_now(), "note": "QLMG final holdout is not activated in Phase 0.", "latest_local_timestamp_observed": latest, "slices": []}
    write_json(out / "sealed_registry_draft.json", registry)
    write_text(out / "new_sealed_policy.md", "# New Sealed Policy\n\nOld Donch contamination does not automatically invalidate every date for QLMG strategies, but overlapping momentum and perp-state concepts mean prior windows are not perfectly clean. Final holdout must be chosen only after data inventory and must not be touched during Phase 0 or screening. Prefer at least the latest six months as final holdout if data depth supports it, with a separate gold holdout if possible.")


def yaml_quote(v: str) -> str:
    return json.dumps(v)


def contract_text(family: str, spec: Mapping[str, str]) -> str:
    return f"""family_id: {family}
status: draft_phase0_not_executable
direction: {yaml_quote(spec['direction'])}
universe_tiers: {yaml_quote(spec['tier'])}
hypothesis: {yaml_quote(spec['mechanism'])}
economic_mechanism: {yaml_quote(spec['mechanism'])}
required_data: {yaml_quote(spec['data'])}
optional_data: "sidecar perp-state variables as conditioners where contract declares them"
forbidden_shortcuts: "no future data, no current-universe survivorship, no OHLCV-only promotion, no liquidation-as-stop"
entry_logic_sketch: "to be frozen in Phase 1 before any alpha run"
exit_logic_sketch: "single predeclared exit family per test; no broad exit rescue"
stop_logic_sketch: "must be side-aware and mark/liquidation-aware for perps"
execution_model_tier_required: "screening can use 5m conservative costs; promotion requires funding, mark, spread/slippage, lifecycle, and side-aware fills"
search_parameters: "small bounded grid only; exact budget to be frozen before run"
validation_plan: "older-data screening, purged walk-forward/CPCV, then frozen final holdout only after contract approval"
promotion_gates: "positive robust net expectancy after costs/funding, concentration limits, lifecycle-safe, no sealed-policy breach"
rejection_conditions: {yaml_quote(spec['blocker'])}
data_blockers: {yaml_quote(spec['blocker'])}
"""


def stage_strategy_contracts(run_root: Path, args: argparse.Namespace) -> None:
    out = run_root / "contracts"
    tmpl = """candidate_id: TBD
family_id: TBD
status: draft
hypothesis: TBD
direction: long|short|both
venue: Bybit linear USDT perpetuals
required_data: []
forbidden_data_or_shortcuts: []
decision_ts_contract: last fully closed bar unless explicitly changed
execution_contract: TBD
cost_funding_liquidation_contract: required for promotion
sealed_policy: no final holdout access until candidate contract is frozen
"""
    write_text(out / "strategy_contract_template.yaml", tmpl)
    fam_dir = out / "family_contracts_draft"
    for family, spec in STRATEGY_FAMILIES.items():
        write_text(fam_dir / f"{family}.yaml", contract_text(family, spec))


def stage_test_plan(run_root: Path, args: argparse.Namespace) -> None:
    out = run_root / "tests"
    tests = [
        "listing timestamp gate", "delist mid-trade", "missing funding interval", "mark-price liquidation not last-price liquidation", "long funding sign", "short funding sign", "short stop and take-profit", "short trailing stop", "order side inversion", "survivorship-free universe", "point-in-time rolling feature", "sector map point-in-time membership", "catalyst timestamp not visible before event", "no prelisting contract features", "stale sidecar fail-closed", "cost attribution reconciliation", "deterministic replay fixed seed",
    ]
    unit_rows = [{"test_name": t, "priority": "required", "status": "planned"} for t in tests]
    integ_rows = [{"scenario": "end_to_end_long_short_replay_fixture", "status": "planned"}, {"scenario": "qlmg_family_contract_smoke_no_holdout", "status": "planned"}, {"scenario": "sealed_guard_blocks_final_holdout", "status": "planned"}]
    write_csv(out / "required_unit_tests.csv", unit_rows)
    write_csv(out / "required_integration_tests.csv", integ_rows)
    write_text(out / "qlmg_test_plan.md", "# QLMG Test Plan\n\n" + pd.DataFrame(unit_rows).to_markdown(index=False) + "\n\n## Integration\n\n" + pd.DataFrame(integ_rows).to_markdown(index=False))


def stage_final_report(run_root: Path, args: argparse.Namespace, root_reason: str) -> None:
    docs = [safe_rel(p) for p in DOC_PATHS if p.exists()]
    data_inv = run_root / "data/data_store_inventory.csv"
    data_table = pd.read_csv(data_inv).to_markdown(index=False) if data_inv.exists() else "Data inventory missing."
    short_matrix = run_root / "shorts/short_support_matrix.csv"
    shorts = pd.read_csv(short_matrix)
    short_ready = "not_ready" if (shorts["status"].isin(["missing", "partial"]).any()) else "ready"
    report = f"""# QLMG Phase 0 Reset Report

Generated: `{utc_now()}`

## Executive Verdict

- Active strategy direction changed to QLMG-inspired long/short Bybit USDT perp research.
- Old Donch/V3/S1/state-transition artifacts are legacy reference only.
- Old artifacts were preserved/indexed. Only strict rebuildable cache paths were eligible for deletion.
- Short-engine readiness: `{short_ready}`.
- Data readiness: `screening_partial; final_selection_blocked_until lifecycle, 1m/depth/liquidation/catalyst requirements are resolved per family`.
- Alpha tests should not start until Phase 1 contracts acknowledge these blockers.

## Root

- run_root: `{run_root}`
- root_reason: `{root_reason}`

## Documentation Created

{chr(10).join('- `' + d + '`' for d in docs)}

## Data Inventory Summary

{data_table}

## Highest Priority Missing Data

1. Historical instrument lifecycle/status metadata.
2. Point-in-time sector/catalyst stores for B1/C2.
3. Targeted 1m OHLCV and top-of-book/depth for execution validation.
4. Liquidation data or a defensible liquidation-clearing proxy for E1.
5. Short-side funding/liquidation/margin engine implementation.

## Recommended Next Coding Tasks

1. Implement side-aware execution and funding tests before any short alpha run.
2. Build lifecycle/tradability manifest with official/proxy separation.
3. Add QLMG family runner skeleton that consumes draft contracts but blocks final holdout.
4. Start with D1/D3 screening only if lifecycle/cost controls are available; otherwise start with engine/data fixes.

## Recommended First Alpha-Test Prompt

Freeze a D1/D3 screening contract using only pre-final-holdout data, with no final validation, no broad exit grid, explicit lifecycle/tradability filters, and conservative costs. Produce activation, matched-null, path-shape, concentration, and fail-closed diagnostics before any shortlist.

## Blockers

- Short support incomplete.
- Mark-price liquidation model missing.
- Current `/opt/parquet/1m` is empty; `/opt/parquet/1m_hot` is partial.
- Historical orderbook/trade/liquidation/catalyst/sector data are not proven complete.
- Final QLMG sealed holdout remains draft-only and must not be touched.
"""
    write_text(run_root / "QLMG_PHASE0_RESET_REPORT.md", report)


def run_stage(stage: str, run_root: Path, args: argparse.Namespace, root_reason: str) -> None:
    append_command(run_root, stage, sys.argv)
    if stage == "repo-and-artifact-inventory":
        stage_repo_inventory(run_root, args)
    elif stage == "legacy-archive-plan":
        stage_legacy_archive(run_root, args)
    elif stage == "documentation-reset":
        stage_documentation_reset(run_root, args)
    elif stage == "infrastructure-reuse-audit":
        stage_infrastructure_audit(run_root, args)
    elif stage == "long-short-engine-audit":
        stage_short_audit(run_root, args)
    elif stage == "data-inventory":
        stage_data_inventory(run_root, args)
    elif stage == "data-acquisition-feasibility":
        stage_data_acquisition(run_root, args)
    elif stage == "new-sealed-policy":
        stage_sealed_policy(run_root, args)
    elif stage == "strategy-contract-scaffold":
        stage_strategy_contracts(run_root, args)
    elif stage == "test-plan":
        stage_test_plan(run_root, args)
    elif stage == "final-report":
        stage_final_report(run_root, args, root_reason)
    else:
        raise ValueError(stage)
    mark_done(run_root, stage)


def main() -> None:
    args = parse_args()
    run_root, root_reason = run_root_from_args(args)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_manifest.json", {"run_root": str(run_root), "root_reason": root_reason, "created_at_utc": utc_now(), "dry_run_deletion": bool(args.dry_run_deletion), "metadata_probes_enabled": not bool(args.skip_metadata_probes), "stage_requested": args.stage})
    for stage in stage_list(args.stage):
        print(f"[qlmg-phase0] {stage}", flush=True)
        run_stage(stage, run_root, args, root_reason)
    print(run_root)


if __name__ == "__main__":
    main()
