from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


PROTECTED_TS = pd.Timestamp("2026-01-01", tz="UTC")
FUNDING_MODES = (
    "exact_only_slice",
    "central_imputed",
    "conservative_imputed",
    "severe_imputed",
    "zero_funding_diagnostic",
    "legacy_adverse_proxy_stress",
)
SLIPPAGE_STRESS_BPS = (0, 4, 8, 12)


def funding_side_sign(side: object) -> float:
    value = str(side or "").strip().lower()
    if value in {"long", "long_flat"}:
        return -1.0
    if value in {"short", "short_diagnostic"}:
        return 1.0
    raise ValueError(f"unsupported funding side semantics: {side!r}")


def normalize_frozen_events(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    required = {
        "event_id", "candidate_definition_id", "symbol", "entry_ts", "exit_interval_end_ts",
        "side", "raw_gross_R", "raw_fee_R", "raw_slippage_R", "raw_funding_R",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{family} frozen events missing fields: {sorted(missing)}")
    out = frame.copy()
    out["source_family"] = family
    out["event_key"] = family + "::" + out["event_id"].astype(str)
    if out["event_key"].duplicated().any():
        raise ValueError(f"{family} event identities are not unique")
    for col in ["decision_ts", "entry_ts", "exit_interval_end_ts"]:
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    if out[["entry_ts", "exit_interval_end_ts"]].isna().any().any():
        raise ValueError(f"{family} event timestamps contain null values")
    if (out["exit_interval_end_ts"] < out["entry_ts"]).any():
        raise ValueError(f"{family} has interval end before entry")
    if family == "a1":
        out["risk_ratio"] = pd.to_numeric(out["entry_price"], errors="coerce") / pd.to_numeric(out["risk_price"], errors="coerce")
        out["vol_scale"] = pd.to_numeric(out.get("vol_scale", 1.0), errors="coerce").fillna(1.0) if "vol_scale" in out else 1.0
        out["definition_lane"] = out.get("definition_lane", "unknown")
        out["exit_policy_id"] = out.get("exit_policy_id", "unknown")
    elif family == "tsmom":
        out["risk_ratio"] = pd.to_numeric(out["entry_notional_usd"], errors="coerce") / pd.to_numeric(out["risk_usd"], errors="coerce")
        out["vol_scale"] = pd.to_numeric(out["vol_scale"], errors="coerce")
        out["definition_lane"] = "tsmom"
        out["exit_policy_id"] = out.get("exit_reason", "scheduled_interval")
    else:
        raise ValueError(f"unsupported family: {family}")
    if (~np.isfinite(out["risk_ratio"]) | (out["risk_ratio"] <= 0)).any():
        raise ValueError(f"{family} has invalid funding risk ratios")
    if out["vol_scale"].isna().any():
        raise ValueError(f"{family} has missing vol scale")
    out["funding_side_sign"] = out["side"].map(funding_side_sign)
    for raw, scaled in [
        ("raw_gross_R", "scaled_gross_R"),
        ("raw_fee_R", "scaled_fee_R"),
        ("raw_slippage_R", "scaled_slippage_R"),
    ]:
        out[raw] = pd.to_numeric(out[raw], errors="coerce")
        if scaled not in out:
            out[scaled] = out[raw] * out["vol_scale"]
        else:
            out[scaled] = pd.to_numeric(out[scaled], errors="coerce")
    return out


def build_event_boundary_rows(events: pd.DataFrame) -> pd.DataFrame:
    first = events["entry_ts"].dt.floor("h") + pd.Timedelta(hours=1)
    last = events["exit_interval_end_ts"].dt.floor("h")
    counts = (((last - first).dt.total_seconds() // 3600) + 1).clip(lower=0).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return pd.DataFrame(columns=["event_key", "source_family", "symbol", "boundary_ts", "entry_ts", "exit_interval_end_ts"])
    repeated_positions = np.repeat(np.arange(len(events), dtype=np.int64), counts.to_numpy())
    starts = np.repeat((np.cumsum(counts.to_numpy()) - counts.to_numpy()), counts.to_numpy())
    offsets = np.arange(total, dtype=np.int64) - starts
    first_ns = first.astype("int64").to_numpy()
    boundary_ns = first_ns[repeated_positions] + offsets * 3_600_000_000_000
    selected = events.iloc[repeated_positions]
    return pd.DataFrame({
        "event_key": selected["event_key"].to_numpy(),
        "source_family": selected["source_family"].to_numpy(),
        "symbol": selected["symbol"].astype(str).to_numpy(),
        "boundary_ts": pd.to_datetime(boundary_ns, utc=True),
        "entry_ts": selected["entry_ts"].to_numpy(),
        "exit_interval_end_ts": selected["exit_interval_end_ts"].to_numpy(),
    })


def join_boundaries_to_panel(boundaries: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    required = {
        "symbol", "timestamp", "relativeFundingRate", "funding_rate_central",
        "funding_rate_conservative", "funding_rate_severe",
        "funding_rate_conservative_short", "funding_rate_severe_short",
        "funding_exact", "funding_imputed", "label_cap_reason", "funding_gate_eligible",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"shared funding panel missing fields: {sorted(missing)}")
    work = panel.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    if work.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("shared funding panel has duplicate symbol/timestamp rows")
    joined = boundaries.merge(
        work,
        left_on=["symbol", "boundary_ts"],
        right_on=["symbol", "timestamp"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    return joined


def aggregate_event_funding(events: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    work = joined.copy()
    event_map = events.set_index("event_key")
    work["side_sign"] = work["event_key"].map(event_map["funding_side_sign"])
    work["risk_ratio"] = work["event_key"].map(event_map["risk_ratio"])
    work["rate_conservative_adverse"] = np.where(
        work["side_sign"] < 0,
        work["funding_rate_conservative"],
        work["funding_rate_conservative_short"],
    )
    work["rate_severe_adverse"] = np.where(
        work["side_sign"] < 0,
        work["funding_rate_severe"],
        work["funding_rate_severe_short"],
    )
    for mode, rate_col in [
        ("central", "funding_rate_central"),
        ("conservative", "rate_conservative_adverse"),
        ("severe", "rate_severe_adverse"),
    ]:
        work[f"funding_{mode}_R_component"] = work["side_sign"] * pd.to_numeric(work[rate_col], errors="coerce") * work["risk_ratio"]
    work["funding_exact_R_component"] = np.where(
        work["funding_exact"].fillna(False),
        work["side_sign"] * pd.to_numeric(work["relativeFundingRate"], errors="coerce") * work["risk_ratio"],
        0.0,
    )
    work["legacy_proxy_R_component"] = np.where(work["funding_imputed"].fillna(False), -0.05, work["funding_exact_R_component"])
    grouped = work.groupby("event_key", sort=False).agg(
        funding_boundary_rows=("boundary_ts", "size"),
        exact_boundary_rows=("funding_exact", "sum"),
        imputed_boundary_rows=("funding_imputed", "sum"),
        missing_boundary_rows=("_merge", lambda s: int((s != "both").sum())),
        funding_exact_R_panel=("funding_exact_R_component", "sum"),
        funding_central_R=("funding_central_R_component", "sum"),
        funding_conservative_R=("funding_conservative_R_component", "sum"),
        funding_severe_R=("funding_severe_R_component", "sum"),
        funding_legacy_proxy_R=("legacy_proxy_R_component", "sum"),
    ).reset_index()
    out = events.merge(grouped, on="event_key", how="left", validate="one_to_one")
    zero_cols = [
        "funding_boundary_rows", "exact_boundary_rows", "imputed_boundary_rows", "missing_boundary_rows",
        "funding_exact_R_panel", "funding_central_R", "funding_conservative_R",
        "funding_severe_R", "funding_legacy_proxy_R",
    ]
    out[zero_cols] = out[zero_cols].fillna(0.0)
    out["all_boundaries_exact"] = out["imputed_boundary_rows"].eq(0)
    out["central_imputation_cap"] = out["imputed_boundary_rows"].gt(0)
    return out


def period_masks(events: pd.DataFrame) -> dict[str, pd.Series]:
    entry = pd.to_datetime(events["entry_ts"], utc=True)
    return {
        "full_train": pd.Series(True, index=events.index),
        "2024": entry.dt.year.eq(2024),
        "2025_h1": entry.ge(pd.Timestamp("2025-01-01", tz="UTC")) & entry.lt(pd.Timestamp("2025-07-01", tz="UTC")),
        "2025_h2": entry.ge(pd.Timestamp("2025-07-01", tz="UTC")) & entry.lt(PROTECTED_TS),
        "exact_funded_events_only": events["all_boundaries_exact"].astype(bool),
    }


def funding_mode_values(events: pd.DataFrame, mode: str) -> tuple[pd.Series, pd.Series]:
    include = pd.Series(True, index=events.index)
    if mode == "exact_only_slice":
        include = events["all_boundaries_exact"].astype(bool)
        values = events["funding_exact_R_panel"]
    elif mode == "central_imputed":
        values = events["funding_central_R"]
    elif mode == "conservative_imputed":
        values = events["funding_conservative_R"]
    elif mode == "severe_imputed":
        values = events["funding_severe_R"]
    elif mode == "zero_funding_diagnostic":
        values = pd.Series(0.0, index=events.index)
    elif mode == "legacy_adverse_proxy_stress":
        values = events["funding_legacy_proxy_R"]
    else:
        raise ValueError(f"unknown funding mode: {mode}")
    return pd.to_numeric(values, errors="coerce").fillna(0.0), include


def grouped_rescore(
    events: pd.DataFrame,
    group_columns: Iterable[str],
    *,
    family: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = list(group_columns)
    masks = period_masks(events)
    for period_scope, period_mask in masks.items():
        for funding_mode in FUNDING_MODES:
            funding_raw, mode_include = funding_mode_values(events, funding_mode)
            base_include = period_mask & mode_include
            for slippage_bps in SLIPPAGE_STRESS_BPS:
                subset = events.loc[base_include].copy()
                if subset.empty:
                    continue
                subset["scenario_funding_raw_R"] = funding_raw.loc[base_include]
                subset["scenario_funding_scaled_R"] = subset["scenario_funding_raw_R"] * subset["vol_scale"]
                subset["added_slippage_raw_R"] = -float(slippage_bps) / 10_000.0 * subset["risk_ratio"]
                subset["added_slippage_scaled_R"] = subset["added_slippage_raw_R"] * subset["vol_scale"]
                subset["scenario_raw_slippage_R"] = subset["raw_slippage_R"] + subset["added_slippage_raw_R"]
                subset["scenario_scaled_slippage_R"] = subset["scaled_slippage_R"] + subset["added_slippage_scaled_R"]
                subset["scenario_raw_net_R"] = subset["raw_gross_R"] + subset["raw_fee_R"] + subset["scenario_funding_raw_R"] + subset["scenario_raw_slippage_R"]
                subset["scenario_scaled_net_R"] = subset["scaled_gross_R"] + subset["scaled_fee_R"] + subset["scenario_funding_scaled_R"] + subset["scenario_scaled_slippage_R"]
                grouped = subset.groupby(group_columns, dropna=False, sort=True)
                for keys, group in grouped:
                    if not isinstance(keys, tuple):
                        keys = (keys,)
                    row = dict(zip(group_columns, keys))
                    row.update({
                        "source_family": family,
                        "period_scope": period_scope,
                        "funding_mode": funding_mode,
                        "slippage_round_trip_bps": slippage_bps,
                        "events": len(group),
                        "active_symbols": group["symbol"].nunique(),
                        "exact_boundary_rows": int(group["exact_boundary_rows"].sum()),
                        "imputed_boundary_rows": int(group["imputed_boundary_rows"].sum()),
                        "raw_gross_R": group["raw_gross_R"].sum(),
                        "raw_fee_R": group["raw_fee_R"].sum(),
                        "raw_funding_R": group["scenario_funding_raw_R"].sum(),
                        "raw_slippage_R": group["scenario_raw_slippage_R"].sum(),
                        "raw_net_R": group["scenario_raw_net_R"].sum(),
                        "scaled_gross_R": group["scaled_gross_R"].sum(),
                        "scaled_fee_R": group["scaled_fee_R"].sum(),
                        "scaled_funding_R": group["scenario_funding_scaled_R"].sum(),
                        "scaled_slippage_R": group["scenario_scaled_slippage_R"].sum(),
                        "scaled_net_R": group["scenario_scaled_net_R"].sum(),
                        "funding_imputed_train_screen_cap": bool(group["central_imputation_cap"].any()),
                        "evidence_label": "train_only_rescore_diagnostic_capped",
                    })
                    rows.append(row)
    return pd.DataFrame(rows)
