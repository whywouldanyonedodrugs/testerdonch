from __future__ import annotations

from collections.abc import Iterable
import json
from typing import Any

import numpy as np
import pandas as pd


LONG_LANES = (
    "a1_impulse_base_breakout",
    "a1_plus_compression",
    "h12_rv_compression_breakout",
    "h13_flat_range_escape",
    "h06_vcp_like_contraction",
)

DIVERSITY_FIELDS = (
    "decision_timeframe",
    "universe_policy",
    "leader_rank_metric",
    "leader_top_n",
    "parent_regime_gate",
    "funding_gate",
    "prior_high_proximity_filter",
    "impulse_lookback_days",
    "base_window_days",
    "pullback_max_pct",
    "path_smoothness_metric",
    "rv_reference_days",
    "rv_percentile_threshold",
    "box_width_atr_max",
    "contraction_legs_required",
)


def select_balanced_50(
    plan: pd.DataFrame,
    definitions: pd.DataFrame,
    reused_hashes: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select ten specs per lane using coverage, never prior economic results."""
    if "selected_key_policy_hash" not in definitions:
        raise ValueError("definitions require selected_key_policy_hash")
    representative = definitions.drop_duplicates("selected_key_policy_hash").set_index("selected_key_policy_hash")
    reused = set(map(str, reused_hashes))
    selected_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for lane in LONG_LANES:
        lane_plan = plan[plan["definition_lane"].astype(str).eq(lane)].copy()
        if len(lane_plan) < 10:
            raise ValueError(f"lane {lane} has fewer than ten selected-key specs")
        lane_reused = lane_plan[lane_plan["selected_key_policy_hash"].astype(str).isin(reused)].copy()
        if len(lane_reused) != 2:
            raise ValueError(f"lane {lane} must have exactly two reusable first-pack specs, found {len(lane_reused)}")
        chosen = [row.to_dict() for _, row in lane_reused.sort_values("selected_key_policy_hash", kind="mergesort").iterrows()]
        for row in chosen:
            row["selection_role"] = "reused_corrected_first_pack"
            row["diversity_score"] = np.nan
        remaining = lane_plan[~lane_plan["selected_key_policy_hash"].astype(str).isin(reused)].copy()
        while len(chosen) < 10:
            chosen_hashes = [str(row["selected_key_policy_hash"]) for row in chosen]
            chosen_specs = representative.loc[chosen_hashes]
            scored: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
            for _, candidate in remaining.iterrows():
                candidate_hash = str(candidate["selected_key_policy_hash"])
                if candidate_hash in chosen_hashes:
                    continue
                spec = representative.loc[candidate_hash]
                uncovered = 0
                min_distance = len(DIVERSITY_FIELDS)
                for field in DIVERSITY_FIELDS:
                    value = str(spec.get(field, ""))
                    covered = set(chosen_specs.get(field, pd.Series(dtype=str)).fillna("").astype(str))
                    uncovered += int(value not in covered)
                for _, existing in chosen_specs.iterrows():
                    distance = sum(str(spec.get(field, "")) != str(existing.get(field, "")) for field in DIVERSITY_FIELDS)
                    min_distance = min(min_distance, distance)
                score = float(uncovered * 100 + min_distance)
                scored.append((score, candidate_hash, candidate.to_dict(), spec.to_dict()))
            if not scored:
                raise ValueError(f"unable to complete deterministic diversity selection for {lane}")
            score, _, picked, _ = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
            picked["selection_role"] = "new_parameter_space_diversity"
            picked["diversity_score"] = score
            chosen.append(picked)
        for ordinal, row in enumerate(chosen, start=1):
            spec = representative.loc[str(row["selected_key_policy_hash"])]
            output = dict(row)
            output["lane_selection_ordinal"] = ordinal
            output["selection_uses_prior_pnl"] = False
            selected_rows.append(output)
            audit = {
                "definition_lane": lane,
                "selected_key_policy_hash": row["selected_key_policy_hash"],
                "selection_role": row["selection_role"],
                "diversity_score": row["diversity_score"],
                "selection_uses_prior_pnl": False,
            }
            audit.update({field: spec.get(field, "") for field in DIVERSITY_FIELDS})
            audit_rows.append(audit)
    selected = pd.DataFrame(selected_rows)
    if len(selected) != 50 or selected["selected_key_policy_hash"].nunique() != 50:
        raise ValueError("balanced selection must contain 50 unique selected-key specs")
    if not selected.groupby("definition_lane").size().eq(10).all():
        raise ValueError("balanced selection must contain ten specs per long lane")
    if selected["definition_lane"].astype(str).eq("short_diagnostic").any():
        raise ValueError("short diagnostic specs are forbidden")
    return selected.reset_index(drop=True), pd.DataFrame(audit_rows)


def extend_frozen_panel(
    panel: pd.DataFrame,
    required_boundaries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extend only by a frozen symbol/day estimate; never fit new parameters."""
    work = panel.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    required = required_boundaries[["symbol", "boundary_ts"]].drop_duplicates().copy()
    required["boundary_ts"] = pd.to_datetime(required["boundary_ts"], utc=True, errors="coerce")
    existing = work[["symbol", "timestamp"]].rename(columns={"timestamp": "boundary_ts"})
    missing = required.merge(existing, on=["symbol", "boundary_ts"], how="left", indicator=True)
    missing = missing[missing["_merge"].eq("left_only")].drop(columns="_merge")
    if missing.empty:
        return work, pd.DataFrame([{"panel_extended": False, "requested_missing_boundaries": 0, "extended_rows": 0, "unresolved_rows": 0, "model_refit": False, "status": "pass"}])
    source = work.copy()
    source["utc_date"] = source["timestamp"].dt.floor("D")
    scenario_columns = [
        "relativeFundingRate", "funding_rate_central", "funding_rate_conservative",
        "funding_rate_severe", "funding_rate_conservative_short", "funding_rate_severe_short",
        "funding_exact", "funding_imputed", "funding_source", "confidence_tier",
        "model_version", "label_cap_reason", "funding_gate_eligible", "liquidity_tier",
    ]
    daily = source.sort_values(["symbol", "utc_date", "timestamp"], kind="mergesort").drop_duplicates(["symbol", "utc_date"])
    missing["utc_date"] = missing["boundary_ts"].dt.floor("D")
    extensions = missing.merge(daily[["symbol", "utc_date", *[c for c in scenario_columns if c in daily]]], on=["symbol", "utc_date"], how="left", validate="many_to_one")
    unresolved = extensions["funding_rate_central"].isna()
    if unresolved.any():
        # The selected v1 model is a frozen symbol estimate shrunk to liquidity tier;
        # its imputed scenario values are constant by symbol and can be applied to a
        # missing date without fitting or observing strategy outcomes.
        symbol_model = source[source["funding_imputed"].fillna(False)].sort_values(["symbol", "timestamp"], kind="mergesort").drop_duplicates("symbol")
        fallback = extensions.loc[unresolved, ["symbol"]].merge(
            symbol_model[["symbol", *[c for c in scenario_columns if c in symbol_model]]],
            on="symbol", how="left", validate="many_to_one", suffixes=("", "_symbol_model"),
        )
        for column in scenario_columns:
            fallback_column = f"{column}_symbol_model"
            if fallback_column not in fallback and column in fallback:
                fallback_column = column
            if fallback_column in fallback:
                extensions.loc[unresolved, column] = fallback[fallback_column].to_numpy()
        unresolved = extensions["funding_rate_central"].isna()
    if unresolved.any():
        examples = extensions.loc[unresolved, ["symbol", "boundary_ts"]].head(10).to_dict("records")
        raise ValueError(f"frozen funding model cannot extend {int(unresolved.sum())} boundaries without refit: {examples}")
    extensions["timestamp"] = extensions.pop("boundary_ts")
    extensions["relativeFundingRate"] = np.nan
    extensions["funding_exact"] = False
    extensions["funding_imputed"] = True
    extensions["funding_source"] = "frozen_model_symbol_day_extension"
    extensions["label_cap_reason"] = "funding_imputed_train_screen_cap"
    extensions["funding_gate_eligible"] = False
    extensions = extensions.drop(columns=["utc_date"], errors="ignore")
    extended = pd.concat([work, extensions.reindex(columns=work.columns)], ignore_index=True, sort=False)
    extended = extended.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    if extended.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("panel extension created duplicate boundaries")
    audit = pd.DataFrame([{
        "panel_extended": True,
        "requested_missing_boundaries": int(len(missing)),
        "extended_rows": int(len(extensions)),
        "unresolved_rows": 0,
        "model_refit": False,
        "model_version": ";".join(sorted(set(extensions["model_version"].astype(str)))) if "model_version" in extensions else "",
        "status": "pass",
    }])
    return extended, audit


def extend_frozen_panel_with_verified_model(
    panel: pd.DataFrame,
    required_boundaries: pd.DataFrame,
    funding_root: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extend unseen symbols by rehydrating, never refitting, the frozen model contract."""
    try:
        return extend_frozen_panel(panel, required_boundaries)
    except ValueError as original_error:
        from pathlib import Path
        from tools import kraken_funding_imputation as model_lib
        from tools import run_kraken_shared_funding_imputation_model as builder

        root = Path(funding_root)
        summary = json.loads((root / "decision_summary.json").read_text(encoding="utf-8"))
        expected_model_hash = str(summary["selected_model_hash"])
        selected_model_name = str(summary["selected_central_model"])
        work = panel.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
        original_symbols = sorted(work["symbol"].dropna().astype(str).unique())
        exact_training, _ = builder.load_exact_rates(original_symbols)
        training_manifest = pd.read_csv(root / "funding/model_training_dataset_manifest.csv")
        boundary_contract = training_manifest[
            training_manifest["artifact"].astype(str).eq("required_unique_hourly_boundaries")
        ]
        if len(boundary_contract) != 1:
            raise ValueError("frozen funding training manifest lacks one required-boundary contract row")
        daily_training = builder.load_daily_market_features(
            original_symbols,
            pd.to_datetime(boundary_contract.iloc[0]["min_entry_ts"], utc=True),
            pd.to_datetime(boundary_contract.iloc[0]["max_interval_end_ts"], utc=True),
        )
        exact_features = builder.attach_daily_features(exact_training, daily_training)
        frozen_model = model_lib.fit_funding_model(exact_features, selected_model_name)
        if frozen_model.model_hash != expected_model_hash:
            raise ValueError(
                f"frozen model rehydration hash mismatch: expected={expected_model_hash} "
                f"actual={frozen_model.model_hash}; original extension error={original_error}"
            )
        required = required_boundaries[["symbol", "boundary_ts"]].drop_duplicates().copy()
        required["boundary_ts"] = pd.to_datetime(required["boundary_ts"], utc=True, errors="coerce")
        existing = work[["symbol", "timestamp"]].rename(columns={"timestamp": "boundary_ts"})
        missing = required.merge(existing, on=["symbol", "boundary_ts"], how="left", indicator=True)
        missing = missing[missing["_merge"].eq("left_only")].drop(columns="_merge")
        current_symbols = sorted(set(original_symbols) | set(required["symbol"].astype(str)))
        daily_current = builder.load_daily_market_features(
            current_symbols, missing["boundary_ts"].min(), missing["boundary_ts"].max()
        )
        missing_for_model = missing.rename(columns={"boundary_ts": "timestamp"})
        missing_features = builder.attach_daily_features(missing_for_model, daily_current)
        exact_current, _ = builder.load_exact_rates(current_symbols)
        imputed_panel = work[work["funding_imputed"].fillna(False)]
        q75 = float((imputed_panel["funding_rate_conservative"] - imputed_panel["funding_rate_central"]).abs().median())
        q95 = float((imputed_panel["funding_rate_severe"] - imputed_panel["funding_rate_central"]).abs().median())
        extension = model_lib.build_funding_scenarios(
            missing_for_model, exact_current, missing_features, frozen_model, (q75, q95)
        )
        extended = pd.concat([work, extension.reindex(columns=work.columns)], ignore_index=True, sort=False)
        extended = extended.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
        if extended.duplicated(["symbol", "timestamp"]).any():
            raise ValueError("verified frozen-model extension created duplicate boundaries")
        audit = pd.DataFrame([{
            "panel_extended": True,
            "requested_missing_boundaries": int(len(missing)),
            "extended_rows": int(len(extension)),
            "unresolved_rows": int(extension["funding_rate_central"].isna().sum()),
            "model_refit": False,
            "model_rehydrated_from_original_exact_manifest": True,
            "expected_model_hash": expected_model_hash,
            "actual_model_hash": frozen_model.model_hash,
            "exact_extension_rows": int(extension["funding_exact"].sum()),
            "imputed_extension_rows": int(extension["funding_imputed"].sum()),
            "imputed_gate_eligible_rows": int((extension["funding_imputed"] & extension["funding_gate_eligible"]).sum()),
            "status": "pass" if not extension["funding_rate_central"].isna().any() and not (extension["funding_imputed"] & extension["funding_gate_eligible"]).any() else "fail",
        }])
        return extended, audit


def scenario_event_rows(events: pd.DataFrame, funding_modes: Iterable[str], slippage_bps_values: Iterable[int]) -> pd.DataFrame:
    from tools import kraken_shared_funding_consumer as consumer

    rows: list[pd.DataFrame] = []
    for mode in funding_modes:
        funding, include = consumer.funding_mode_values(events, mode)
        for bps in slippage_bps_values:
            frame = events.loc[include].copy()
            frame["funding_mode"] = mode
            frame["slippage_round_trip_bps"] = int(bps)
            frame["scenario_funding_raw_R"] = funding.loc[include]
            frame["scenario_funding_scaled_R"] = frame["scenario_funding_raw_R"] * frame["vol_scale"]
            frame["added_slippage_raw_R"] = -float(bps) / 10_000.0 * frame["risk_ratio"]
            frame["scenario_raw_net_R"] = frame["raw_gross_R"] + frame["raw_fee_R"] + frame["raw_slippage_R"] + frame["added_slippage_raw_R"] + frame["scenario_funding_raw_R"]
            frame["scenario_scaled_net_R"] = frame["scaled_gross_R"] + frame["scaled_fee_R"] + frame["scaled_slippage_R"] + frame["added_slippage_raw_R"] * frame["vol_scale"] + frame["scenario_funding_scaled_R"]
            rows.append(frame)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
