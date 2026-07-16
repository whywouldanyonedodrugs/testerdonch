from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


DECISION_INPUT_FIELDS = {
    "decision_ts",
    "feature_available_ts",
    "feature_source_end_ts",
    "liquidity_source_end_ts",
    "max_decision_input_available_ts",
}

OUTCOME_FIELDS = {
    "entry_ts",
    "entry_price",
    "exit_ts",
    "exit_price",
    "raw_gross_R",
    "raw_fee_R",
    "raw_funding_R",
    "raw_slippage_R",
    "raw_net_R",
    "scaled_gross_R",
    "scaled_fee_R",
    "scaled_funding_R",
    "scaled_slippage_R",
    "scaled_net_R",
}

SEMANTIC_CACHE_CONTRACT_VERSION = "semantic_cache_contract_v1_20260707"

DECISION_INPUT_CACHE_CLASSES = {
    "decision_calendar",
    "pit_universe_membership",
    "universe_membership",
    "tsmom_rank_feature",
    "tsmom_topn",
    "tsmom_signal",
    "parent_gate",
    "funding_gate",
    "vol_scale",
    "a1_leader_rank",
    "a1_topn",
    "a1_impulse_base_feature",
    "a1_compression_feature",
    "a1_breakout_signal",
    "selected_event_key",
}

OUTCOME_CACHE_CLASSES = {
    "interval_outcome",
    "tsmom_interval_outcome",
}


def canonical_json_hash(payload: Mapping[str, Any], *, n: int = 64) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:n]


def cache_manifest_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Kraken Semantic Cache Manifest",
        "type": "object",
        "required": [
            "cache_class",
            "semantic_cache_hash",
            "schema_hash",
            "cache_contract_version",
            "input_manifest_hash",
            "protected_train_boundary",
            "policy_fields",
            "content_hash",
            "row_count",
            "hash_basis",
        ],
        "properties": {
            "cache_class": {"type": "string"},
            "semantic_cache_hash": {"type": "string"},
            "schema_hash": {"type": "string"},
            "cache_contract_version": {"type": "string"},
            "input_manifest_hash": {"type": "string"},
            "data_hash": {"type": "string"},
            "protected_train_boundary": {"type": "string"},
            "policy_fields": {"type": "object"},
            "selected_event_key_hash": {"type": "string"},
            "content_hash": {"type": "string"},
            "row_count": {"type": "integer"},
            "min_decision_ts": {"type": "string"},
            "max_decision_ts": {"type": "string"},
            "hash_basis": {"type": "string"},
            "canonical_sort_keys": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    }


def schema_hash_for_columns(columns: Sequence[str], *, cache_class: str = "") -> str:
    payload = {
        "cache_class": str(cache_class),
        "columns": sorted(str(c) for c in columns),
        "schema_contract": SEMANTIC_CACHE_CONTRACT_VERSION,
    }
    return canonical_json_hash(payload, n=32)


def semantic_cache_hash(
    *,
    cache_class: str,
    schema_hash: str,
    input_manifest_hash: str,
    protected_train_boundary: str,
    policy_fields: Mapping[str, Any] | None = None,
    data_hash: str = "",
    cache_contract_version: str = SEMANTIC_CACHE_CONTRACT_VERSION,
    selected_event_key_hash: str = "",
) -> str:
    """Hash only semantic cache reuse inputs, not wrapper/reporting code."""
    payload = {
        "cache_class": str(cache_class),
        "schema_hash": str(schema_hash),
        "input_manifest_hash": str(input_manifest_hash),
        "data_hash": str(data_hash),
        "protected_train_boundary": str(protected_train_boundary),
        "policy_fields": dict(policy_fields or {}),
        "cache_contract_version": str(cache_contract_version),
    }
    if str(cache_class) in OUTCOME_CACHE_CLASSES:
        payload["selected_event_key_hash"] = str(selected_event_key_hash)
    return canonical_json_hash(payload, n=32)


def _decision_bounds(df: pd.DataFrame) -> tuple[str, str]:
    if df is None or df.empty or "decision_ts" not in df.columns:
        return "", ""
    ts = pd.to_datetime(df["decision_ts"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return "", ""
    return str(ts.min()), str(ts.max())


def semantic_cache_manifest(
    *,
    cache_class: str,
    df: pd.DataFrame,
    sort_keys: Sequence[str],
    policy_fields: Mapping[str, Any] | None,
    input_manifest_hash: str,
    protected_train_boundary: str,
    data_hash: str = "",
    selected_event_key_hash: str = "",
    path: str = "",
    schema_hash: str | None = None,
    content_hash: str | None = None,
    cache_contract_version: str = SEMANTIC_CACHE_CONTRACT_VERSION,
) -> dict[str, Any]:
    cache_class_s = str(cache_class)
    schema = schema_hash or schema_hash_for_columns(list(df.columns), cache_class=cache_class_s)
    if cache_class_s in DECISION_INPUT_CACHE_CLASSES:
        outcome_cols = sorted(set(df.columns) & OUTCOME_FIELDS)
        if outcome_cols:
            raise ValueError(f"decision-input cache {cache_class_s} contains outcome fields: {outcome_cols}")
    if cache_class_s in OUTCOME_CACHE_CLASSES and not str(selected_event_key_hash):
        raise ValueError(f"outcome cache {cache_class_s} requires selected_event_key_hash")
    use_sort_keys = [str(c) for c in sort_keys if c in df.columns]
    content = content_hash or (canonical_frame_hash(df, sort_keys=use_sort_keys) if use_sort_keys else canonical_frame_hash(df))
    min_decision, max_decision = _decision_bounds(df)
    semantic_hash = semantic_cache_hash(
        cache_class=cache_class_s,
        schema_hash=schema,
        input_manifest_hash=input_manifest_hash,
        data_hash=data_hash,
        protected_train_boundary=protected_train_boundary,
        policy_fields=policy_fields or {},
        cache_contract_version=cache_contract_version,
        selected_event_key_hash=selected_event_key_hash,
    )
    return {
        "cache_class": cache_class_s,
        "semantic_cache_hash": semantic_hash,
        "schema_hash": schema,
        "cache_contract_version": cache_contract_version,
        "input_manifest_hash": str(input_manifest_hash),
        "data_hash": str(data_hash),
        "protected_train_boundary": str(protected_train_boundary),
        "policy_fields": dict(policy_fields or {}),
        "selected_event_key_hash": str(selected_event_key_hash),
        "path": str(path),
        "content_hash": content,
        "row_count": int(len(df)),
        "min_decision_ts": min_decision,
        "max_decision_ts": max_decision,
        "hash_basis": "canonical_sorted_csv_rows",
        "canonical_sort_keys": use_sort_keys,
        "status": "pass",
    }


def validate_semantic_cache_manifest(manifest: Mapping[str, Any], *, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    required = cache_manifest_schema()["required"]
    missing = [k for k in required if k not in manifest or str(manifest.get(k, "")) == ""]
    if missing:
        return {**dict(manifest), "status": "fail", "reason": f"missing_required_fields:{';'.join(missing)}"}
    cache_class = str(manifest.get("cache_class", ""))
    if cache_class in OUTCOME_CACHE_CLASSES and not str(manifest.get("selected_event_key_hash", "")):
        return {**dict(manifest), "status": "fail", "reason": "outcome_cache_missing_selected_event_key_hash"}
    policy_fields = manifest.get("policy_fields", {})
    if isinstance(policy_fields, str):
        try:
            policy_fields = json.loads(policy_fields)
        except Exception:
            return {**dict(manifest), "status": "fail", "reason": "invalid_policy_fields_json"}
    expected_hash = semantic_cache_hash(
        cache_class=cache_class,
        schema_hash=str(manifest.get("schema_hash", "")),
        input_manifest_hash=str(manifest.get("input_manifest_hash", "")),
        data_hash=str(manifest.get("data_hash", "")),
        protected_train_boundary=str(manifest.get("protected_train_boundary", "")),
        policy_fields=policy_fields,
        cache_contract_version=str(manifest.get("cache_contract_version", SEMANTIC_CACHE_CONTRACT_VERSION)),
        selected_event_key_hash=str(manifest.get("selected_event_key_hash", "")),
    )
    if expected_hash != str(manifest.get("semantic_cache_hash", "")):
        return {**dict(manifest), "status": "fail", "reason": "semantic_cache_hash_mismatch", "expected_semantic_cache_hash": expected_hash}
    if expected:
        for key, value in expected.items():
            if str(manifest.get(key, "")) != str(value):
                return {**dict(manifest), "status": "fail", "reason": f"expected_{key}_mismatch", "expected": value}
    return {**dict(manifest), "status": "pass"}


def write_semantic_cache_shard(
    temp_path: Path,
    final_path: Path,
    df: pd.DataFrame,
    *,
    cache_class: str,
    policy_fields: Mapping[str, Any] | None,
    input_manifest_hash: str,
    protected_train_boundary: str,
    sort_keys: Sequence[str],
    data_hash: str = "",
    selected_event_key_hash: str = "",
) -> dict[str, Any]:
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    safe_sort_keys = [str(c) for c in sort_keys if c in df.columns]
    manifest = semantic_cache_manifest(
        cache_class=cache_class,
        df=df,
        sort_keys=safe_sort_keys,
        policy_fields=policy_fields,
        input_manifest_hash=input_manifest_hash,
        protected_train_boundary=protected_train_boundary,
        data_hash=data_hash,
        selected_event_key_hash=selected_event_key_hash,
        path=str(final_path),
    )
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, final_path)
    manifest_path = final_path.with_suffix(final_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def canonical_frame_hash(df: pd.DataFrame, *, sort_keys: Sequence[str] | None = None, columns: Sequence[str] | None = None) -> str:
    """Hash row content deterministically without depending on parquet metadata."""
    if df is None or df.empty:
        payload = "empty\n"
    else:
        work = df.copy()
        if columns:
            missing = [c for c in columns if c not in work.columns]
            if missing:
                raise ValueError(f"missing columns for canonical hash: {missing}")
            work = work[list(columns)]
        else:
            work = work[sorted(work.columns)]
        if sort_keys:
            missing = [c for c in sort_keys if c not in work.columns]
            if missing:
                raise ValueError(f"missing sort keys for canonical hash: {missing}")
            work = work.sort_values(list(sort_keys), kind="mergesort")
        payload = work.fillna("").astype(str).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def stable_hash(*parts: object, n: int = 24) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:n]


def build_candidate_identity_hash(row: Mapping[str, Any]) -> str:
    fields = [
        "run_id",
        "candidate_definition_id",
        "candidate_symbol_id",
        "symbol_id",
        "symbol",
        "parameter_vector_hash",
        "definition_hash",
        "universe_policy_hash",
        "rank_policy_hash",
        "signal_policy_hash",
        "parent_gate_policy_hash",
        "funding_gate_policy_hash",
        "vol_scale_policy_hash",
        "interval_policy_hash",
        "feature_policy_hash",
        "event_semantics_version",
        "code_hash",
        "config_hash",
        "input_manifest_hash",
        "train_window_hash",
    ]
    return stable_hash(*(row.get(f, "") for f in fields), n=32)


def freeze_selected_event_keys(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    required = ["candidate_identity_hash", "symbol_id", "decision_ts"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"selected event keys missing required columns: {missing}")
    keys = df[required].copy()
    keys["decision_ts"] = pd.to_datetime(keys["decision_ts"], utc=True, errors="coerce")
    if keys["decision_ts"].isna().any():
        raise ValueError("selected event keys contain invalid decision_ts")
    frozen_hash = canonical_frame_hash(keys, sort_keys=required)
    keys["selected_event_keys_hash"] = frozen_hash
    keys["selected_event_keys_frozen"] = True
    return keys, frozen_hash


def assert_decision_input_no_leak(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{"check": "decision_input_no_leak", "status": "pass", "rows": 0, "violations": 0}])
    if "decision_ts" not in df.columns:
        return pd.DataFrame([{"check": "decision_input_no_leak", "status": "fail", "rows": len(df), "violations": len(df), "reason": "missing_decision_ts"}])
    decision = pd.to_datetime(df["decision_ts"], utc=True, errors="coerce")
    rows = []
    total_violations = 0
    for col in sorted(DECISION_INPUT_FIELDS & set(df.columns) - {"decision_ts"}):
        ts = pd.to_datetime(df[col], utc=True, errors="coerce")
        violations = int(((ts.notna()) & (decision.notna()) & (ts > decision)).sum())
        total_violations += violations
        rows.append({"check": f"{col}_lte_decision_ts", "status": "fail" if violations else "pass", "rows": len(df), "violations": violations})
    if not rows:
        rows.append({"check": "decision_input_no_leak", "status": "fail", "rows": len(df), "violations": len(df), "reason": "no_feature_timestamp_columns"})
    rows.append({"check": "decision_input_no_leak_overall", "status": "fail" if total_violations else "pass", "rows": len(df), "violations": total_violations})
    return pd.DataFrame(rows)


def assert_outcome_cache_not_used_pre_freeze(access_log: pd.DataFrame) -> pd.DataFrame:
    if access_log.empty:
        return pd.DataFrame([{"check": "outcome_cache_pre_freeze_access", "status": "pass", "rows": 0, "violations": 0}])
    required = {"cache_class", "access_phase", "selected_event_keys_frozen"}
    missing = required - set(access_log.columns)
    if missing:
        return pd.DataFrame([{"check": "outcome_cache_pre_freeze_access", "status": "fail", "rows": len(access_log), "violations": len(access_log), "reason": f"missing:{';'.join(sorted(missing))}"}])
    outcome = access_log["cache_class"].astype(str).eq("outcome")
    pre_freeze = ~access_log["selected_event_keys_frozen"].astype(str).str.lower().isin({"true", "1", "yes"})
    selection_phase = access_log["access_phase"].astype(str).isin({"universe", "ranking", "signal", "gate", "selection", "tuning", "control_pre_freeze"})
    violations = int((outcome & pre_freeze & selection_phase).sum())
    return pd.DataFrame([{"check": "outcome_cache_pre_freeze_access", "status": "fail" if violations else "pass", "rows": len(access_log), "violations": violations}])


def write_atomic_shard(temp_path: Path, final_path: Path, df: pd.DataFrame, manifest_payload: Mapping[str, Any], *, sort_keys: Sequence[str]) -> dict[str, Any]:
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_sort_keys = list(sort_keys)
    content_hash = canonical_frame_hash(df, sort_keys=canonical_sort_keys)
    df.to_parquet(temp_path, index=False)
    os.replace(temp_path, final_path)
    manifest = {
        **dict(manifest_payload),
        "path": str(final_path),
        "row_count": int(len(df)),
        "content_hash": content_hash,
        "hash_basis": "canonical_sorted_csv_rows",
        "canonical_sort_keys": canonical_sort_keys,
    }
    manifest_path = final_path.with_suffix(final_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def validate_shard_manifest(manifest_path: Path, *, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not manifest_path.exists():
        return {"status": "fail", "reason": "missing_manifest", "manifest_path": str(manifest_path)}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "fail", "reason": f"invalid_manifest:{type(exc).__name__}", "manifest_path": str(manifest_path)}
    shard_path = Path(str(manifest.get("path", "")))
    if not shard_path.exists():
        return {"status": "fail", "reason": "missing_shard_path", **manifest}
    try:
        df = pd.read_parquet(shard_path)
        manifest_sort_keys = manifest.get("canonical_sort_keys")
        if isinstance(manifest_sort_keys, str):
            sort_keys = [c for c in manifest_sort_keys.split(";") if c]
        elif isinstance(manifest_sort_keys, list):
            sort_keys = [str(c) for c in manifest_sort_keys]
        else:
            sort_keys = [c for c in ["candidate_identity_hash", "symbol_id", "decision_ts"] if c in df.columns]
        actual = canonical_frame_hash(df, sort_keys=sort_keys)
    except Exception as exc:
        return {"status": "fail", "reason": f"unreadable_shard:{type(exc).__name__}", **manifest}
    if actual != manifest.get("content_hash"):
        return {"status": "fail", "reason": "content_hash_mismatch", "actual_content_hash": actual, **manifest}
    if expected:
        for key, value in expected.items():
            if str(manifest.get(key, "")) != str(value):
                return {"status": "fail", "reason": f"expected_{key}_mismatch", "expected": value, **manifest}
    return {"status": "pass", "actual_content_hash": actual, **manifest}
