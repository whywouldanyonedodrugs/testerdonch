#!/usr/bin/env python3
"""Build the outcome-free C01 onset tape and frozen later economic contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from tools.build_kraken_c01_foundation import (
        FAMILY_ID, PRIMARY_MODEL, PROTECTED_START, REFERENCE_PANEL_HASH, ROBUSTNESS_MODEL,
        TRAIN_START, assert_no_outcome_columns, canonical_json,
        deterministic_hash, iso_utc, load_safe_manifest, sha256_file,
    )
    from tools.kraken_candle_volume_authority import (
        PROXY_FIELD, daily_close_based_proxy, lagged_top_n_membership,
    )
except ModuleNotFoundError:
    from build_kraken_c01_foundation import (
        FAMILY_ID, PRIMARY_MODEL, PROTECTED_START, REFERENCE_PANEL_HASH, ROBUSTNESS_MODEL,
        TRAIN_START, assert_no_outcome_columns, canonical_json,
        deterministic_hash, iso_utc, load_safe_manifest, sha256_file,
    )
    from kraken_candle_volume_authority import (
        PROXY_FIELD, daily_close_based_proxy, lagged_top_n_membership,
    )


TASK_ID = "donch_bt_stage_2c1_volume_authority_resume_20260717_v1"
FEATURE_HASH = "c0d2955d6447f360beff528fc0985d328c2853cf9a45bf03958d2a61216470bb"
ONSET_VERSION = "c01_causal_onset_v1_20260717"
COHORT_VERSION = "c01_mechanism_proof_top100_close_based_proxy_v1_20260717"
RESET = pd.Timedelta(hours=6)
BAR = pd.Timedelta(minutes=5)
EXCLUDED_CATEGORIES = {"xstocks", "stablecoin", "forex", "pre-ipo", "dtf", "commodities"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1"}


def extract_onsets(tape: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_id", "symbol", "decision_ts", "residual_model_version", "sign",
        "path_state", "feature_version", "reference_panel_hash", "canonical_episode_id",
    }
    if not required.issubset(tape.columns):
        raise ValueError(f"Stage 2B tape missing onset fields: {sorted(required - set(tape.columns))}")
    work = tape.copy()
    work["decision_ts"] = pd.to_datetime(work["decision_ts"], utc=True, errors="raise")
    work = work.sort_values(
        ["symbol", "residual_model_version", "sign", "decision_ts", "candidate_id"], kind="mergesort",
    )
    groups = ["symbol", "residual_model_version", "sign"]
    previous = work.groupby(groups, sort=False)["decision_ts"].shift()
    work["prior_same_sign_active_ts"] = previous
    inactive = ((work["decision_ts"] - previous) / BAR - 1).clip(lower=0).fillna(72).astype(int)
    work["reset_inactive_bar_count"] = inactive
    onsets = work[previous.isna() | ((work["decision_ts"] - previous) > RESET)].copy()
    onsets["onset_version"] = ONSET_VERSION
    return onsets.reset_index(drop=True)


def _instrument_identity(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("instruments")
    if not isinstance(rows, list):
        raise ValueError("official instrument source has no instruments list")
    output = []
    for row in rows:
        symbol = str(row.get("symbol", ""))
        if not symbol.startswith("PF_"):
            continue
        category = str(row.get("category", "")).strip()
        reason = ""
        if symbol in {"PF_XBTUSD", "PF_ETHUSD"}:
            reason = "reference_factor_not_candidate"
        elif _bool(row.get("tradfi")) or category.lower() in EXCLUDED_CATEGORIES:
            reason = f"excluded_noncrypto_or_stable_category:{category}"
        elif row.get("type") != "flexible_futures":
            reason = f"excluded_nonperpetual_type:{row.get('type')}"
        output.append({
            "symbol": symbol, "base": str(row.get("base", "")), "category": category,
            "tradfi": _bool(row.get("tradfi")), "asset_identity_eligible": not reason,
            "asset_identity_reason": reason or "eligible_crypto_perpetual",
        })
    return pd.DataFrame(output).drop_duplicates("symbol")


def _semantic_summary(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows["snapshot_ts"] = pd.to_datetime(rows["snapshot_ts"], utc=True, errors="raise")
    grouped = rows.groupby("symbol", sort=True).agg(
        semantic_first_authority_ts=("snapshot_ts", "min"),
        semantic_last_authority_ts=("snapshot_ts", "max"),
        semantic_consistent=("semantic_consistent", "all"),
        observed_semantic_versions=("semantic_version", "nunique"),
        observed_official_snapshots=("snapshot_ts", "count"),
    ).reset_index()
    return grouped


def read_symbol_daily_proxy(authority_rows: Sequence[Any], symbol: str) -> pd.DataFrame:
    selected = [row for row in authority_rows if row.symbol == symbol and row.dataset == "historical_trade_candles_5m"]
    if not selected:
        return pd.DataFrame(columns=["symbol", "utc_day", PROXY_FIELD])
    parts: list[pd.DataFrame] = []
    required = {"time", "close", "volume", "venue_symbol", "resolution", "rankable_pre_holdout", "contains_protected_period"}
    for row in selected:
        schema = set(pq.ParquetFile(row.parquet_path).schema_arrow.names)
        if not required.issubset(schema):
            if row.rows <= 1:
                continue
            raise ValueError(f"volume shard schema mismatch: {row.parquet_path}")
        frame = pd.read_parquet(row.parquet_path, columns=sorted(required))
        if frame.empty:
            continue
        if not frame["venue_symbol"].astype(str).eq(symbol).all() or not frame["resolution"].astype(str).eq("5m").all():
            raise ValueError("volume shard symbol/resolution mismatch")
        if not frame["rankable_pre_holdout"].map(_bool).all() or frame["contains_protected_period"].map(_bool).any():
            raise ValueError("unrankable/protected volume row reached reader")
        frame["source_open_ts"] = pd.to_datetime(pd.to_numeric(frame["time"], errors="raise"), unit="ms", utc=True)
        frame = frame[(frame["source_open_ts"] >= TRAIN_START) & (frame["source_open_ts"] < PROTECTED_START)]
        parts.append(frame[["source_open_ts", "close", "volume"]].assign(symbol=symbol))
    if not parts:
        return pd.DataFrame(columns=["symbol", "utc_day", PROXY_FIELD])
    bars = pd.concat(parts, ignore_index=True).sort_values("source_open_ts", kind="mergesort")
    duplicates = bars.duplicated("source_open_ts", keep=False)
    if duplicates.any():
        conflicts = bars.loc[duplicates].groupby("source_open_ts")[["close", "volume"]].nunique().gt(1).any(axis=1)
        if conflicts.any():
            raise ValueError(f"conflicting duplicate volume bars for {symbol}")
        bars = bars.drop_duplicates("source_open_ts", keep="first")
    return daily_close_based_proxy(bars)


def build_cohort(
    authority_rows: Sequence[Any], symbols: Sequence[str], instrument_path: Path, semantic_path: Path,
) -> tuple[pd.DataFrame, str]:
    identity = _instrument_identity(instrument_path)
    semantic = _semantic_summary(semantic_path)
    metadata = identity.merge(semantic, on="symbol", how="left", validate="one_to_one")
    metadata["semantic_consistent"] = metadata["semantic_consistent"].fillna(False)
    metadata["semantic_authority_available"] = metadata["semantic_first_authority_ts"].notna()
    eligible_symbols = metadata.loc[
        metadata["symbol"].isin(symbols) & metadata["asset_identity_eligible"]
        & metadata["semantic_consistent"] & metadata["semantic_authority_available"], "symbol",
    ].sort_values().tolist()
    daily_parts = [read_symbol_daily_proxy(authority_rows, symbol) for symbol in eligible_symbols]
    daily = pd.concat([part for part in daily_parts if not part.empty], ignore_index=True)
    daily = daily.merge(
        metadata[["symbol", "semantic_first_authority_ts"]], on="symbol", how="left", validate="many_to_one",
    )
    daily = daily[daily["utc_day"] >= daily["semantic_first_authority_ts"].dt.floor("D")].copy()
    ranks = lagged_top_n_membership(daily, top_n=100, lookback_days=30, minimum_valid_days=20)
    ranks = ranks.merge(metadata, on="symbol", how="left", validate="many_to_one")
    ranks["cohort_version"] = COHORT_VERSION
    ranks["liquidity_field"] = PROXY_FIELD
    ranks["rank_uses_current_day"] = False
    ranks["exact_quote_volume_claim"] = False
    cohort_hash = deterministic_hash({
        "version": COHORT_VERSION,
        "volume_authority_sha256": sha256_file(semantic_path),
        "instrument_source_sha256": sha256_file(instrument_path),
        "excluded_categories": sorted(EXCLUDED_CATEGORIES),
        "membership": ranks.loc[ranks["top_100_eligible"], ["utc_day", "symbol", "rank"]].astype(str).to_dict("records"),
    })
    ranks["cohort_hash"] = cohort_hash
    return ranks, cohort_hash


def apply_cohort(onsets: pd.DataFrame, cohort: pd.DataFrame, cohort_hash: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = onsets.copy()
    work["utc_day"] = work["decision_ts"].dt.floor("D")
    membership = cohort[["utc_day", "symbol", "top_100_eligible", "rank", "valid_prior_days"]]
    work = work.merge(membership, on=["utc_day", "symbol"], how="left", validate="many_to_one")
    work["top_100_eligible"] = work["top_100_eligible"].map(
        lambda value: bool(value) if pd.notna(value) else False,
    )
    work["cohort_exclusion_reason"] = np.where(
        work["top_100_eligible"], "", "not_top100_or_insufficient_20day_proxy_history",
    )
    excluded = work[~work["top_100_eligible"]].copy()
    accepted = work[work["top_100_eligible"]].copy()
    accepted["candidate_cohort_version"] = COHORT_VERSION
    accepted["candidate_cohort_hash"] = cohort_hash
    accepted["stage2b_candidate_id"] = accepted["candidate_id"]
    accepted["event_id"] = accepted.apply(
        lambda row: "c01event_" + deterministic_hash({
            "stage2b_candidate_id": row["candidate_id"], "onset_version": ONSET_VERSION,
            "cohort_hash": cohort_hash,
        })[:24], axis=1,
    )
    accepted["dominant_residual_bar_sign_alignment"] = np.where(
        accepted["path_state"].eq("jump_dominated"),
        "aligned_implied_by_largest_absolute_share_gte_0.5_and_nonzero_shock",
        "not_applicable",
    )
    accepted["post_onset_rows_read"] = 0
    accepted["protected_rows_read"] = 0
    accepted["economic_outputs_computed"] = False
    if accepted["event_id"].duplicated().any():
        raise ValueError("duplicate C01 onset event identity")
    assert_no_outcome_columns(accepted.columns)
    return accepted, excluded


def model_agreement(events: pd.DataFrame) -> pd.DataFrame:
    primary = events[events["residual_model_version"] == PRIMARY_MODEL].copy()
    robust = events[events["residual_model_version"] == ROBUSTNESS_MODEL].copy()
    exact_keys = ["symbol", "decision_ts", "sign"]
    exact = primary.merge(
        robust[exact_keys + ["path_state", "event_id", "canonical_episode_id"]],
        on=exact_keys, how="inner", suffixes=("_primary", "_robust"), validate="many_to_many",
    )
    exact_ids = set(exact["event_id_primary"])
    robust_exact_ids = set(exact["event_id_robust"])
    within = 0
    same_episode = 0
    for keys, group in primary.groupby(["symbol", "sign"], sort=True):
        pool = robust[(robust["symbol"] == keys[0]) & (robust["sign"] == keys[1])]
        values = pool["decision_ts"].sort_values().to_numpy(dtype="datetime64[ns]")
        episode_ids = set(pool["canonical_episode_id"])
        for row in group.itertuples(index=False):
            ts = np.datetime64(row.decision_ts.to_datetime64())
            if len(values) and np.min(np.abs(values - ts)) <= np.timedelta64(30, "m"):
                within += 1
            if row.canonical_episode_id in episode_ids:
                same_episode += 1
    return pd.DataFrame([
        {"metric": "primary_events", "count": len(primary)},
        {"metric": "robustness_events", "count": len(robust)},
        {"metric": "exact_timestamp_and_sign", "count": len(exact_ids)},
        {"metric": "exact_timestamp_sign_and_path", "count": int((exact["path_state_primary"] == exact["path_state_robust"]).sum())},
        {"metric": "same_sign_onset_within_30m", "count": within},
        {"metric": "same_canonical_episode", "count": same_episode},
        {"metric": "primary_only_no_exact_match", "count": len(primary) - len(exact_ids)},
        {"metric": "robustness_only_no_exact_match", "count": len(robust) - len(robust_exact_ids)},
    ])


def branch_register(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in (PRIMARY_MODEL, ROBUSTNESS_MODEL):
        for sign in ("positive", "negative"):
            for path in ("smooth", "jump_dominated", "intermediate"):
                if path == "intermediate":
                    role, direction = "diagnostic_control_only", "none"
                elif path == "smooth" and sign == "positive":
                    role, direction = ("primary_economic_prior" if model == PRIMARY_MODEL else "robustness_only"), "long_continuation"
                elif path == "smooth":
                    role, direction = "secondary_symmetric_diagnostic", "short_continuation"
                elif sign == "positive":
                    role, direction = "failure_branch", "short_after_completed_failure"
                else:
                    role, direction = "failure_branch", "long_after_completed_failure"
                mask = (
                    events["residual_model_version"].eq(model) & events["sign"].eq(sign)
                    & events["path_state"].eq(path)
                )
                rows.append({
                    "attempt_id": f"c01_{'primary' if model == PRIMARY_MODEL else 'robust'}_{sign}_{path}",
                    "residual_model": model, "sign": sign, "path_state": path,
                    "frozen_role": role, "frozen_direction": direction,
                    "onset_event_count": int(mask.sum()), "retained": True,
                })
    return pd.DataFrame(rows)


def bounded_cross_family_identity_report(repository_root: Path, event_count: int) -> pd.DataFrame:
    """Inspect only headers/metadata for the three explicitly authorized families."""
    specs = [
        {
            "prior_family": "A1_compression_continuation",
            "source_path": "results/rebaseline/phase_kraken_a1_compression_targeted_materialization_controls_stress_20260712_v1/materialized/event_ledgers",
            "blocker": "decision identity exists but no causal episode_input_start is retained in the event-ledger schema",
        },
        {
            "prior_family": "relative_strength_breakout_vs_BTC",
            "source_path": "results/rebaseline/phase_kraken_relative_strength_breakout_vs_btc_screen_20260716_v1/signals/raw_signal_manifest.csv",
            "blocker": "decision identity exists but the causal breakout episode start is not retained in the raw-signal schema",
        },
        {
            "prior_family": "repaired_Backside",
            "source_path": "results/rebaseline/phase_kraken_backside_blowoff_signal_state_repaired_screen_20260715_v1/signals/raw_signal_manifest.csv",
            "blocker": "decision identity exists but the causal blowoff/extension episode start is not retained in the raw-signal schema",
        },
    ]
    rows = []
    for spec in specs:
        path = repository_root / spec["source_path"]
        columns: list[str] = []
        if path.is_file() and path.suffix == ".csv":
            columns = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
        elif path.is_dir():
            files = sorted(path.glob("*.parquet"))
            if files:
                columns = pq.ParquetFile(files[0]).schema_arrow.names
        rows.append({
            **spec, "source_exists": path.exists(), "schema_columns_inspected": ";".join(columns),
            "mapping_status": "blocked", "c01_candidate_rows": event_count,
            "economic_rows_read": 0, "protected_rows_read": 0,
            "multiplicity_treatment": "same_broad_umbrella_pending_causal_start_export",
        })
    return pd.DataFrame(rows)


def write_contracts(
    output: Path, cohort_hash: str, events: pd.DataFrame, cost_policy_source: Path,
) -> tuple[str, str]:
    generator = f"""# C01 Frozen Generator Contract

Family: `{FAMILY_ID}`. Stage 2B feature hash: `{FEATURE_HASH}`. Reference panel hash: `{REFERENCE_PANEL_HASH}`.

An event is the first active row for a symbol/residual-model/sign after the prior 72 completed five-minute bars contain no same-sign activation. No later peak or cleaner row may replace it. The mechanism-proof cohort uses `{PROXY_FIELD}` and cohort hash `{cohort_hash}`. Membership is top 100 by prior-30-calendar-day median with at least 20 valid days, ranked once per UTC day using only prior UTC days. Current 2026 calibration is unit evidence only.

The proxy is not exact quote volume, traded USD notional, capacity, spread, depth, or slippage evidence. Current-roster, survivorship-free, and continuous-tradeability caps remain. Historical semantic ambiguity fails closed. All 12 Stage 2B attempts remain registered. Primary model is `{PRIMARY_MODEL}`; `{ROBUSTNESS_MODEL}` is robustness only. Event rows: {len(events):,}. No post-onset row or outcome is read.
"""
    cost_hash = sha256_file(cost_policy_source)
    economic = f"""# C01 Economic Contract Draft

Status: frozen draft only; no economic run is authorized.

## Entries

- Positive smooth: long at the next executable five-minute trade-bar open after onset.
- Negative smooth: symmetric short diagnostic at the next executable open.
- Positive jump-dominated: within 24h, first completed trade bar closing below the dominant residual bar low; short next executable open.
- Negative jump-dominated: within 24h, first completed trade bar closing above the dominant residual bar high; long next executable open.
- No confirmation means no jump-failure trade. Intermediate events are diagnostic/control only.

The dominant bar must be recomputed from the accepted causal residual component tape and frozen before outcome reads. For jump-dominated onsets its sign alignment is mathematically implied by largest absolute residual share >= 0.5 and nonzero cumulative shock; timestamp and OHLC extreme still require deterministic pre-outcome extraction.

## Exits

- Primary timeout: 6h after entry; robustness timeout: 24h.
- Smooth stop: completed mark close through the opposite six-hour shock-window extreme, executing next trade-bar open.
- Jump-failure stop: completed mark close beyond the dominant jump-bar extreme in the original shock direction, executing next trade-bar open.
- No partial exits, adds, passive/touch fills, leverage optimization, maximum-hold preblocking, or artificial boundary close.

## Costs and funding

- Authority: `{cost_policy_source}` with SHA-256 `{cost_hash}`.
- Base: 5 bps taker per side plus 4 bps round-trip slippage.
- Frozen stress: 10 bps taker per side plus 12 bps round-trip slippage.
- Exact funding where available. Imputed funding is a separately capped cost scenario, never signal or promotion evidence.
- Missing execution/depth evidence remains a claim cap.

## Controls and ablations

1. Raw 6h USD-return shock without residualization.
2. Residual shock without path separation.
3. Raw-return path classification at the same causal timestamps.
4. Matched symbol/year/lagged-volatility/parent-return non-events.
5. BTC-only residual robustness.

Continuation and failure branches remain separate. Run Level 3 kill screen first; only a survivor may receive Level 4 controls. Stop if residualization/path adds no increment, one year/symbol/episode dominates, costs remove the result, or threshold changes are required.
"""
    (output / "C01_FROZEN_GENERATOR_CONTRACT.md").write_text(generator, encoding="utf-8")
    (output / "C01_ECONOMIC_CONTRACT_DRAFT.md").write_text(economic, encoding="utf-8")
    return sha256_bytes(generator.encode()), sha256_bytes(economic.encode())


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def build(args: argparse.Namespace) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage2b = Path(args.stage2b_root).resolve()
    manifest = json.loads((stage2b / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("feature_contract_hash") != FEATURE_HASH or manifest.get("reference_panel_hash") != REFERENCE_PANEL_HASH:
        raise ValueError("Stage 2B immutable feature/reference hash mismatch")
    tape_path = stage2b / "C01_GENERATOR_DIAGNOSTIC_TAPE.parquet"
    expected = next(item["sha256"] for item in manifest["files"] if item["path"] == tape_path.name)
    if sha256_file(tape_path) != expected:
        raise ValueError("Stage 2B tape hash mismatch")
    tape = pd.read_parquet(tape_path)
    if not tape["feature_version"].eq("c01_residual_path_features_v1_20260717").all():
        raise ValueError("Stage 2B feature version mismatch")
    onsets = extract_onsets(tape)
    safe_rows = load_safe_manifest(Path(args.market_manifest))
    cohort, cohort_hash = build_cohort(
        safe_rows, sorted(onsets["symbol"].unique()), Path(args.instrument_source), Path(args.volume_authority),
    )
    events, excluded = apply_cohort(onsets, cohort, cohort_hash)
    preferred = [
        "event_id", "stage2b_candidate_id", "candidate_id", "canonical_episode_id",
        "canonical_episode_input_start", "canonical_episode_input_end", "symbol", "venue",
        "decision_ts", "shock_window_start", "shock_window_end", "residual_model_version", "sign", "path_state",
        "residual_shock_6h", "residual_scale_6h", "residual_shock_z_6h", "largest_bar_share", "path_efficiency",
        "dominant_residual_bar_sign_alignment", "feature_version", "reference_panel_hash", "onset_version",
        "candidate_cohort_version", "candidate_cohort_hash", "rank", "valid_prior_days",
        "prior_same_sign_active_ts", "reset_inactive_bar_count", "post_onset_rows_read", "protected_rows_read",
        "economic_outputs_computed",
    ]
    events = events[preferred].sort_values(["symbol", "decision_ts", "residual_model_version", "sign"], kind="mergesort")
    assert_no_outcome_columns(events.columns)
    events.to_parquet(output / "C01_ONSET_EVENT_TAPE.parquet", index=False)
    cohort.to_parquet(output / "C01_DAILY_LIQUIDITY_MEMBERSHIP.parquet", index=False)
    cohort_summary = cohort.groupby("symbol", sort=True).agg(
        base=("base", "first"), category=("category", "first"),
        semantic_first_authority_ts=("semantic_first_authority_ts", "first"),
        semantic_last_authority_ts=("semantic_last_authority_ts", "first"),
        observed_official_snapshots=("observed_official_snapshots", "first"),
        first_rank_day=("utc_day", "min"), last_rank_day=("utc_day", "max"),
        ranked_days=("utc_day", "nunique"), top_100_eligible_days=("top_100_eligible", "sum"),
        minimum_rank=("rank", "min"), maximum_rank=("rank", "max"),
        maximum_valid_prior_days=("valid_prior_days", "max"),
        cohort_version=("cohort_version", "first"), cohort_hash=("cohort_hash", "first"),
    ).reset_index()
    cohort_summary["survivorship_free_claim"] = False
    cohort_summary["continuous_tradeability_claim"] = False
    cohort_summary["exact_quote_volume_claim"] = False
    write_csv(output / "C01_MECHANISM_PROOF_COHORT.csv", cohort_summary)
    counts = events.groupby([events["decision_ts"].dt.year.rename("year"), "residual_model_version", "sign", "path_state"]).size().rename("onset_event_count").reset_index()
    write_csv(output / "C01_ONSET_EVENT_COUNTS.csv", counts)
    write_csv(output / "C01_MODEL_AGREEMENT_REPORT.csv", model_agreement(events))
    write_csv(output / "C01_BRANCH_ROLE_REGISTER.csv", branch_register(events))
    cross = bounded_cross_family_identity_report(Path(args.repository_root).resolve(), len(events))
    write_csv(output / "C01_CROSS_FAMILY_EPISODE_UPDATE.csv", cross)
    generator_hash, economic_hash = write_contracts(
        output, cohort_hash, events, Path(args.cost_policy_source).resolve(),
    )
    (output / "C01_COHORT_HYGIENE_REPORT.md").write_text(
        "# C01 Cohort Hygiene Report\n\n"
        f"Eligible onset events: {len(events):,}; excluded onsets: {len(excluded):,}. "
        f"Cohort hash: `{cohort_hash}`. Asset exclusions are `{sorted(EXCLUDED_CATEGORIES)}` plus BTC/ETH reference factors. "
        "Historical semantic inconsistencies and pre-authority intervals fail closed. The cohort remains current-roster capped, "
        "not survivorship-free, and does not claim continuous tradeability. No economic outcome was read.\n",
        encoding="utf-8",
    )
    (output / "C01_MULTIPLICITY_BUDGET.md").write_text(
        "# C01 Multiplicity Budget\n\nAll 12 Stage 2B attempts remain registered: 2 residual models x 2 signs x 3 path states. "
        "The BTC-only model is robustness, not a second portfolio or winner-selection source. C01 remains in the broad shared "
        "multiplicity umbrella where prior-family causal starts are unavailable.\n",
        encoding="utf-8",
    )
    write_csv(output / "KRAKEN_DATA_CAPABILITY_AMENDMENT.csv", pd.DataFrame([
        {
            "field": "base_volume", "status": "verified_for_listed_PF_symbol_semantic_intervals",
            "authority_path": str(Path(args.volume_authority)), "rankable_use": "causal_cohort_hygiene_only",
            "claim_cap": "historical_semantic_version_and_current_roster_caps",
        },
        {
            "field": "exact_quote_volume", "status": "unavailable", "authority_path": str(Path(args.volume_authority)),
            "rankable_use": "none", "claim_cap": "do_not_infer_from_candle_volume",
        },
        {
            "field": PROXY_FIELD, "status": "permitted", "authority_path": str(Path(args.volume_authority)),
            "rankable_use": "prior_day_top100_cohort_hygiene_only",
            "claim_cap": "not_notional_capacity_spread_depth_or_slippage_evidence",
        },
    ]))
    summary = {
        "task_id": TASK_ID, "status": "ready_for_explicit_C01_economic_run_approval",
        "stage2b_tape_sha256": expected, "feature_contract_hash": FEATURE_HASH,
        "reference_panel_hash": REFERENCE_PANEL_HASH, "cohort_hash": cohort_hash,
        "onset_event_count": len(events), "excluded_onset_count": len(excluded),
        "canonical_episode_count": int(events["canonical_episode_id"].nunique()),
        "generator_contract_hash": generator_hash, "economic_contract_draft_hash": economic_hash,
        "protected_outcomes_opened": False, "economic_outputs_computed": False,
    }
    (output / "C01_STAGE2C_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2b-root", required=True)
    parser.add_argument("--market-manifest", required=True)
    parser.add_argument("--instrument-source", required=True)
    parser.add_argument("--volume-authority", required=True)
    parser.add_argument("--cost-policy-source", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
