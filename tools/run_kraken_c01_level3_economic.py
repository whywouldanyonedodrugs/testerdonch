#!/usr/bin/env python3
"""Execute the single authorized frozen C01 Level-3 economic screen.

The CLI deliberately exposes only the four frozen task inputs. Market, lifecycle,
and funding authorities are repository-owned constants and are hash-audited in the
run root. No control construction is implemented in this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from tools import build_kraken_c01_foundation as foundation
    from tools import kraken_c01_prerun_contract as contract
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    import build_kraken_c01_foundation as foundation
    import kraken_c01_prerun_contract as contract


TASK_ID = "donch_bt_stage_2e_c01_level3_economic_20260717_v1"
FAMILY_ID = "C01_debetaed_residual_shock_path_bifurcation"
CONTRACT_SHA256 = "c655e94c35412354356bb7f89c07ca17b71c2ae6537a2a1c42aa3dce928ba77d"
EVENT_TAPE_SHA256 = "e4587653aec82fb66ab6775284501ca768b6689a6b14cdb17a90799f32cea6b7"
REGISTER_SHA256 = "b13175a3728c5940a430d425df47c304a16674fc2343af6a63d226e22fbb37c4"
DECISION_RULES_SHA256 = "2f32eb0c67c23c743c85b679a66b8d05d2492871a8128654b5ea3ee998754415"
CONTROL_CONTRACT_SHA256 = "2449fc625fd68b61fc8010bae580e41188c5fb2d41d14a2bc61de91e4e7d3206"
GENERATOR_SHA256 = "3464e79a79956c881c7418840068a61e3f3a47776a5a4d3a669e98df124fd970"
FEATURE_SHA256 = "c0d2955d6447f360beff528fc0985d328c2853cf9a45bf03958d2a61216470bb"
COHORT_SHA256 = "768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15"
REFERENCE_PANEL_SHA256 = "2c0cae827c6f83361ea347796d0323b20d83c3acd222c506aac40c0e97b73763"
ECONOMIC_DRAFT_SHA256 = "f1c8c612ea9f7ffcc2abad3f2efde36b5dfb68fde20d2769fdc5ce40ab306c13"
DATA_AUTHORITY_SHA256 = "b51ca71c3a9cd6425ade8255e443cc76bfc5b5882ee80c3eccfc8de11bad1a1a"

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE2C_ROOT = REPO_ROOT / "docs/agent/task_archive/20260717_donch_bt_stage_2c1_volume_authority_resume_20260717_v1"
STAGE2D_ROOT = REPO_ROOT / "docs/agent/task_archive/20260717_donch_bt_stage_2d_c01_prerun_closure_20260717_v1"
MARKET_MANIFEST = Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv")
LIFECYCLE_SOURCE = REPO_ROOT / "docs/agent/task_archive/20260717_donch_bt_stage_2a1_c01_reference_panel_20260717_v1/sources/terminal_lifecycle/kraken_derivatives_delistings.body"
REFERENCE_MANIFEST = REPO_ROOT / "docs/agent/task_archive/20260717_donch_bt_stage_2a1_c01_reference_panel_20260717_v1/ARTIFACT_MANIFEST.json"
INSTRUMENT_SOURCE = REPO_ROOT / "docs/agent/task_archive/20260716_donch_bt_stage_2a_u2_lifecycle_20260716_v1/sources/kraken_futures_instruments.body"
FUNDING_ROOT = REPO_ROOT / "results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1"
TRAIN_START = pd.Timestamp("2023-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2026-01-01T00:00:00Z")
BAR_DELTA = pd.Timedelta(minutes=5)
OUTCOME_COLUMNS = {"open", "high", "low", "close"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def utc(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return result.tz_convert("UTC")


def iso(value: Any) -> str:
    return utc(value).isoformat().replace("+00:00", "Z")


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1"}


def validate_event_tape(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_id", "candidate_id", "canonical_episode_id", "symbol", "venue", "decision_ts",
        "shock_window_start", "shock_window_end", "residual_model_version", "sign", "path_state",
        "feature_version", "reference_panel_hash", "candidate_cohort_hash", "protected_rows_read",
        "economic_outputs_computed",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"event tape missing columns: {sorted(required - set(frame.columns))}")
    work = frame.copy()
    for field in ("decision_ts", "shock_window_start", "shock_window_end"):
        work[field] = pd.to_datetime(work[field], utc=True, errors="raise")
    if not work["venue"].eq("Kraken").all() or not work["symbol"].str.startswith("PF_").all():
        raise ValueError("non-Kraken event row")
    if (work["decision_ts"] < TRAIN_START).any() or (work["decision_ts"] >= TRAIN_END).any():
        raise ValueError("pre-2023 or protected event row")
    if int(pd.to_numeric(work["protected_rows_read"], errors="raise").sum()) != 0:
        raise ValueError("event tape reports protected rows read")
    if work["economic_outputs_computed"].map(_bool).any():
        raise ValueError("event tape already contains economic output state")
    if work["event_id"].duplicated().any():
        raise ValueError("duplicate onset identity")
    if not work["reference_panel_hash"].eq(REFERENCE_PANEL_SHA256).all():
        raise ValueError("reference-panel hash mismatch")
    if not work["candidate_cohort_hash"].eq(COHORT_SHA256).all():
        raise ValueError("candidate-cohort hash mismatch")
    if not work["feature_version"].eq("c01_residual_path_features_v1_20260717").all():
        raise ValueError("feature-version mismatch")
    return work.sort_values(["symbol", "decision_ts", "residual_model_version"], kind="mergesort")


def validate_definition_register(frame: pd.DataFrame) -> pd.DataFrame:
    expected = pd.DataFrame(contract.definition_register()).astype({"registered_even_if_zero_trades": str})
    work = frame.copy().astype({"registered_even_if_zero_trades": str})
    columns = list(expected.columns)
    if list(work.columns) != columns or not work[columns].reset_index(drop=True).equals(expected[columns]):
        raise ValueError("definition register differs from frozen machine contract")
    return frame.copy()


def load_authorized_ohlc(rows: Sequence[foundation.AuthorityRow], symbol: str, dataset: str) -> pd.DataFrame:
    selected = [row for row in rows if row.symbol == symbol and row.dataset == dataset]
    if not selected:
        raise ValueError(f"no safe {dataset} rows for {symbol}")
    parts: list[pd.DataFrame] = []
    columns = ["time", "open", "high", "low", "close", "venue_symbol", "resolution", "rankable_pre_holdout", "contains_protected_period"]
    for row in selected:
        schema = set(pq.ParquetFile(row.parquet_path).schema_arrow.names)
        if not set(columns).issubset(schema):
            if "candles" in schema and row.rows <= 1:
                continue
            raise ValueError(f"unsafe candle schema: {row.parquet_path}")
        raw = pd.read_parquet(row.parquet_path, columns=columns)
        if raw.empty:
            continue
        if not raw["venue_symbol"].eq(symbol).all() or not raw["resolution"].eq("5m").all():
            raise ValueError("candle symbol or resolution mismatch")
        if not raw["rankable_pre_holdout"].map(_bool).all() or raw["contains_protected_period"].map(_bool).any():
            raise ValueError("mixed or unrankable candle payload")
        ts = pd.to_datetime(pd.to_numeric(raw["time"], errors="raise"), unit="ms", utc=True)
        if (ts < TRAIN_START).any() or (ts >= TRAIN_END).any():
            raise ValueError("out-of-range candle row")
        part = pd.DataFrame({"source_open_ts": ts})
        for field in OUTCOME_COLUMNS:
            part[field] = pd.to_numeric(raw[field], errors="coerce")
        if part[list(OUTCOME_COLUMNS)].isna().any().any() or (part[list(OUTCOME_COLUMNS)] <= 0).any().any():
            raise ValueError("missing or non-positive OHLC")
        parts.append(part)
    if not parts:
        raise ValueError(f"no non-empty {dataset} rows for {symbol}")
    frame = pd.concat(parts, ignore_index=True).sort_values("source_open_ts", kind="mergesort")
    dup = frame.duplicated("source_open_ts", keep=False)
    if dup.any() and frame.loc[dup].groupby("source_open_ts")[list(OUTCOME_COLUMNS)].nunique().gt(1).any().any():
        raise ValueError("conflicting duplicate candle")
    return frame.drop_duplicates("source_open_ts").reset_index(drop=True)


def residual_components(
    candidate_trade: pd.DataFrame, candidate_mark: pd.DataFrame,
    btc_trade: pd.DataFrame, btc_mark: pd.DataFrame,
    eth_trade: pd.DataFrame, eth_mark: pd.DataFrame,
    model: str, invalidations: Sequence[tuple[pd.Timestamp, pd.Timestamp]], opening_date: pd.Timestamp,
) -> pd.DataFrame:
    """Reconstruct the frozen causal per-bar residual component tape."""
    def returns(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        out = frame[["source_open_ts", "close"]].copy()
        out[name] = np.log(out["close"]).diff()
        out.loc[out["source_open_ts"].diff().ne(BAR_DELTA), name] = np.nan
        return out[["source_open_ts", name]]

    frame = returns(candidate_trade, "candidate_ret").merge(
        returns(btc_trade, "btc_ret"), on="source_open_ts", how="outer", validate="one_to_one",
    ).merge(returns(eth_trade, "eth_ret"), on="source_open_ts", how="outer", validate="one_to_one")
    for name, source in (("candidate_mark", candidate_mark), ("btc_mark", btc_mark), ("eth_mark", eth_mark)):
        frame = frame.merge(source[["source_open_ts", "close"]].rename(columns={"close": name}), on="source_open_ts", how="outer", validate="one_to_one")
    frame = frame.sort_values("source_open_ts", kind="mergesort")
    frame = frame[(frame.source_open_ts >= TRAIN_START) & (frame.source_open_ts < TRAIN_END)].copy()
    frame["decision_ts"] = frame.source_open_ts + BAR_DELTA
    frame.loc[frame.source_open_ts < opening_date.ceil("5min"), ["candidate_ret", "candidate_mark"]] = np.nan
    invalid = pd.Series(False, index=frame.index)
    for start, end in invalidations:
        invalid |= frame.decision_ts.ge(start) & frame.decision_ts.lt(end)
    frame.loc[invalid, ["candidate_ret", "candidate_mark"]] = np.nan
    frame["decision_day"] = frame.decision_ts.dt.floor("D")
    coefficients = foundation._daily_coefficients(frame, model)  # frozen Stage 2B implementation
    frame = frame.merge(coefficients, on="decision_day", how="left", validate="many_to_one")
    if model == contract.PRIMARY_MODEL:
        frame["residual"] = frame.candidate_ret - frame.alpha - frame.beta_btc * frame.btc_ret - frame.beta_eth * frame.eth_ret
        required = ["candidate_ret", "btc_ret", "eth_ret", "candidate_mark", "btc_mark", "eth_mark", "residual"]
    else:
        frame["residual"] = frame.candidate_ret - frame.alpha - frame.beta_btc * frame.btc_ret
        required = ["candidate_ret", "btc_ret", "candidate_mark", "btc_mark", "residual"]
    frame.loc[frame[required].isna().any(axis=1), "residual"] = np.nan
    return frame[["source_open_ts", "decision_ts", "residual"]].reset_index(drop=True)


def dominant_bar(event: Mapping[str, Any], residuals: pd.DataFrame, trade: pd.DataFrame) -> dict[str, Any]:
    window = residuals[(residuals.source_open_ts >= utc(event["shock_window_start"])) & (residuals.decision_ts <= utc(event["decision_ts"]))]
    if len(window) != 72 or window.residual.isna().any():
        raise ValueError("dominant residual window unavailable")
    residual_sum = float(window.residual.sum())
    absolute_sum = float(window.residual.abs().sum())
    largest_share = float(window.residual.abs().max() / absolute_sum)
    path_efficiency = float(abs(residual_sum) / absolute_sum)
    if not np.isclose(residual_sum, float(event["residual_shock_6h"]), rtol=1e-10, atol=1e-12):
        raise ValueError("recomputed residual shock mismatch")
    if not np.isclose(largest_share, float(event["largest_bar_share"]), rtol=1e-10, atol=1e-12):
        raise ValueError("recomputed largest-bar share mismatch")
    if not np.isclose(path_efficiency, float(event["path_efficiency"]), rtol=1e-10, atol=1e-12):
        raise ValueError("recomputed path-efficiency mismatch")
    maximum = window.residual.abs().max()
    row = window[window.residual.abs().eq(maximum)].sort_values("source_open_ts", kind="mergesort").iloc[0]
    bar = trade[trade.source_open_ts.eq(row.source_open_ts)]
    if len(bar) != 1:
        raise ValueError("dominant trade bar unavailable")
    price = bar.iloc[0]
    return {
        "dominant_bar_source_open_ts": row.source_open_ts,
        "dominant_bar_close_ts": row.decision_ts,
        "dominant_residual": float(row.residual),
        "dominant_bar_high": float(price.high),
        "dominant_bar_low": float(price.low),
    }


def freeze_event_anchor(event: Mapping[str, Any], residuals: pd.DataFrame, trade: pd.DataFrame) -> dict[str, Any]:
    dom = dominant_bar(event, residuals, trade)
    shock = trade[(trade.source_open_ts >= utc(event["shock_window_start"])) & ((trade.source_open_ts + BAR_DELTA) <= utc(event["decision_ts"]))]
    if len(shock) != 72:
        raise ValueError("shock-window anchor bars unavailable")
    return {
        "event_id": event["event_id"], "symbol": event["symbol"],
        "residual_model_version": event["residual_model_version"],
        "shock_window_low": float(shock.low.min()), "shock_window_high": float(shock.high.max()), **dom,
    }


def first_trade_open(trade: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    rows = trade[trade.source_open_ts >= timestamp]
    return None if rows.empty else rows.iloc[0]


def prepare_candidate(
    event: Mapping[str, Any], definition: Mapping[str, Any], trade: pd.DataFrame,
    mark: pd.DataFrame, residuals: pd.DataFrame,
    lifecycle_invalidations: Sequence[tuple[pd.Timestamp, pd.Timestamp]] = (),
    frozen_anchor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    onset = utc(event["decision_ts"])
    anchor = dict(frozen_anchor) if frozen_anchor is not None else freeze_event_anchor(event, residuals, trade)
    dom = {field: anchor[field] for field in ("dominant_bar_source_open_ts", "dominant_bar_close_ts", "dominant_residual", "dominant_bar_high", "dominant_bar_low")}
    side = str(definition["side"])
    if str(definition["path_state"]) == "smooth":
        confirmation = onset
    else:
        horizon = onset + pd.Timedelta(hours=24)
        post = trade[(trade.source_open_ts >= onset) & ((trade.source_open_ts + BAR_DELTA) <= horizon)].copy()
        if str(definition["shock_sign"]) == "positive":
            matched = post[post.close < dom["dominant_bar_low"]]
        else:
            matched = post[post.close > dom["dominant_bar_high"]]
        if matched.empty:
            raise CandidateInvalid("jump_confirmation_unavailable")
        confirmation = utc(matched.iloc[0].source_open_ts) + BAR_DELTA
    entry = first_trade_open(trade, confirmation)
    if entry is None or utc(entry.source_open_ts) >= TRAIN_END:
        raise CandidateInvalid("next_trade_open_unavailable")
    entry_ts, entry_price = utc(entry.source_open_ts), float(entry.open)
    if str(definition["path_state"]) == "smooth":
        stop = float(anchor["shock_window_low"]) if side == "long" else float(anchor["shock_window_high"])
    else:
        stop = dom["dominant_bar_low"] if side == "long" else dom["dominant_bar_high"]
    risk = entry_price - stop if side == "long" else stop - entry_price
    if not math.isfinite(risk) or risk <= 0:
        raise CandidateInvalid("non_positive_structural_stop_distance")
    prior_mark = mark[(mark.source_open_ts + BAR_DELTA) <= entry_ts]
    if prior_mark.empty:
        raise CandidateInvalid("pre_entry_mark_unavailable")
    prior_close = float(prior_mark.iloc[-1].close)
    if (side == "long" and prior_close <= stop) or (side == "short" and prior_close >= stop):
        raise CandidateInvalid("stop_breached_before_entry")
    timeout_target = entry_ts + pd.Timedelta(hours=int(definition["timeout_hours"]))
    timeout_bar = first_trade_open(trade, timeout_target)
    if timeout_bar is None or utc(timeout_bar.source_open_ts) >= TRAIN_END:
        raise CandidateInvalid("timeout_next_trade_open_unavailable")
    timeout_execution = utc(timeout_bar.source_open_ts)
    stop_trigger: pd.Timestamp | None = None
    expected = pd.date_range(entry_ts, timeout_execution - BAR_DELTA, freq=BAR_DELTA)
    mark_index = mark.set_index("source_open_ts", drop=False)
    for source_ts in expected:
        if source_ts not in mark_index.index:
            raise CandidateInvalid("missing_mark_bar_during_monitoring")
        row = mark_index.loc[source_ts]
        if isinstance(row, pd.DataFrame):
            raise CandidateInvalid("duplicate_mark_bar_during_monitoring")
        close_ts = source_ts + BAR_DELTA
        breached = float(row.close) <= stop if side == "long" else float(row.close) >= stop
        if breached:
            stop_trigger = close_ts
            break
    if stop_trigger is None:
        exit_ts, exit_price, exit_reason = timeout_execution, float(timeout_bar.open), "fixed_timeout"
        last_monitor = timeout_execution
    else:
        stop_bar = first_trade_open(trade, stop_trigger)
        if stop_bar is None or utc(stop_bar.source_open_ts) >= TRAIN_END:
            raise CandidateInvalid("stop_next_trade_open_unavailable")
        stop_execution = utc(stop_bar.source_open_ts)
        if stop_execution == timeout_execution:
            raise CandidateInvalid("same_bar_stop_timeout_ambiguity")
        exit_ts, exit_price, exit_reason = stop_execution, float(stop_bar.open), "mark_close_stop_next_trade_open"
        last_monitor = stop_trigger
    required_ts = {
        "onset_ts": onset, "confirmation_ts": confirmation, "entry_ts": entry_ts,
        "last_stop_monitor_ts": last_monitor, "timeout_ts": timeout_target,
        "funding_accounting_end_ts": exit_ts, "exit_execution_ts": exit_ts,
    }
    if not contract.interval_is_wholly_train_eligible(required_ts):
        raise CandidateInvalid("interval_not_wholly_train_eligible")
    if any(entry_ts < end and exit_ts >= start for start, end in lifecycle_invalidations):
        raise CandidateInvalid("known_lifecycle_invalid_interval")
    address = "c01trade_" + stable_hash({
        "definition_policy_hash": definition["definition_policy_hash"], "event_id": event["event_id"],
        "symbol": event["symbol"], "entry_ts": iso(entry_ts), "entry_price": entry_price,
        "stop_price": stop, "risk_denominator": risk, "timeout_target": iso(timeout_target),
    })
    return {
        **dict(event), **dict(definition), **dom,
        "onset_ts": onset, "confirmation_ts": confirmation, "entry_ts": entry_ts,
        "entry_price": entry_price, "stop_price": stop, "risk_denominator": risk,
        "timeout_target_ts": timeout_target, "actual_exit_ts": exit_ts, "exit_price": exit_price,
        "exit_reason": exit_reason, "last_stop_monitor_ts": last_monitor,
        "economic_address": address, "calendar_year": entry_ts.year,
    }


class CandidateInvalid(ValueError):
    pass


def concentration_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"total_net_bps": 0.0, "max_symbol_pnl_share": math.nan, "max_episode_pnl_share": math.nan, "max_year_positive_pnl_share": math.nan}
    total = float(frame.base_fee_slippage_net_bps.sum())
    symbols = frame.groupby("symbol").base_fee_slippage_net_bps.sum()
    episodes = frame.groupby("canonical_episode_id").base_fee_slippage_net_bps.sum()
    years = frame.groupby("calendar_year").base_fee_slippage_net_bps.sum()
    positive_year_total = float(years.clip(lower=0).sum())
    return {
        "total_net_bps": total,
        "max_symbol_pnl_share": float(symbols.clip(lower=0).max() / total) if total > 0 else math.nan,
        "max_episode_pnl_share": float(episodes.clip(lower=0).max() / total) if total > 0 else math.nan,
        "max_year_positive_pnl_share": float(years.clip(lower=0).max() / positive_year_total) if positive_year_total > 0 else math.nan,
    }


def load_funding_panel() -> tuple[pd.DataFrame, dict[str, float], str]:
    manifest_path = FUNDING_ROOT / "funding/shared_funding_panel_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    parts: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        if str(row.status) != "pass" or utc(row.max_timestamp) >= TRAIN_END:
            raise ValueError("unsafe funding partition metadata")
        path = FUNDING_ROOT / str(row.path)
        if sha256_file(path) != str(row.file_sha256):
            raise ValueError("funding partition hash mismatch")
        parts.append(pd.read_parquet(path))
    panel = pd.concat(parts, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel.timestamp, utc=True, errors="raise")
    if (panel.timestamp < TRAIN_START).any() or (panel.timestamp >= TRAIN_END).any():
        raise ValueError("unsafe funding row")
    if panel.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("duplicate funding boundary")
    rate_fields = ["funding_rate_central", "funding_rate_conservative", "funding_rate_severe", "funding_rate_conservative_short", "funding_rate_severe_short"]
    global_location = {field: float(pd.to_numeric(panel[field], errors="raise").median()) for field in rate_fields}
    model_hashes = panel.model_hash.dropna().astype(str).unique()
    if len(model_hashes) != 1:
        raise ValueError("ambiguous frozen funding model")
    return panel.sort_values(["symbol", "timestamp"]), global_location, str(model_hashes[0])


def attach_funding(trades: pd.DataFrame, panel: pd.DataFrame, global_location: Mapping[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        empty = trades.copy()
        for field in ("exact_boundary_count", "imputed_boundary_count", "funding_cashflow_central_bps", "funding_cashflow_conservative_bps", "funding_cashflow_severe_bps"):
            empty[field] = pd.Series(dtype="float64")
        empty["funding_partition"] = pd.Series(dtype="object")
        return empty, pd.DataFrame()
    index = panel.set_index(["symbol", "timestamp"])
    fields = ["funding_rate_central", "funding_rate_conservative", "funding_rate_severe", "funding_rate_conservative_short", "funding_rate_severe_short"]
    by_symbol = panel.groupby("symbol")[fields].median()
    rows: list[dict[str, Any]] = []
    for trade in trades.itertuples(index=False):
        first_boundary = utc(trade.entry_ts).ceil("h")
        if first_boundary <= utc(trade.entry_ts):
            first_boundary += pd.Timedelta(hours=1)
        boundaries = pd.date_range(first_boundary, utc(trade.actual_exit_ts).floor("h"), freq="h")
        for boundary in boundaries:
            try:
                source = index.loc[(trade.symbol, boundary)]
                if isinstance(source, pd.DataFrame):
                    raise ValueError("duplicate funding boundary")
                exact, imputed, extension = bool(source.funding_exact), bool(source.funding_imputed), False
                rates = {field: float(source[field]) for field in fields}
                source_name = str(source.funding_rate_source)
            except KeyError:
                location = by_symbol.loc[trade.symbol] if trade.symbol in by_symbol.index else pd.Series(global_location)
                rates = {field: float(location[field]) for field in fields}
                exact, imputed, extension = False, True, True
                source_name = "frozen_model_location_extension"
            if not all(math.isfinite(value) for value in rates.values()):
                raise ValueError("missing non-finite funding estimate")
            direction = -1.0 if trade.side == "long" else 1.0
            conservative_field = "funding_rate_conservative" if trade.side == "long" else "funding_rate_conservative_short"
            severe_field = "funding_rate_severe" if trade.side == "long" else "funding_rate_severe_short"
            rows.append({
                "economic_address": trade.economic_address, "symbol": trade.symbol, "boundary_ts": boundary,
                "funding_exact": exact, "funding_imputed": imputed, "panel_extension": extension,
                "funding_source": source_name, "central_cashflow_bps": direction * rates["funding_rate_central"] * 10_000,
                "conservative_cashflow_bps": direction * rates[conservative_field] * 10_000,
                "severe_cashflow_bps": direction * rates[severe_field] * 10_000,
            })
    boundary = pd.DataFrame(rows)
    if boundary.empty:
        sums = pd.DataFrame({"economic_address": trades.economic_address})
        for field in ("exact_boundary_count", "imputed_boundary_count", "funding_cashflow_central_bps", "funding_cashflow_conservative_bps", "funding_cashflow_severe_bps"):
            sums[field] = 0
    else:
        sums = boundary.groupby("economic_address", sort=True).agg(
            exact_boundary_count=("funding_exact", "sum"), imputed_boundary_count=("funding_imputed", "sum"),
            funding_cashflow_central_bps=("central_cashflow_bps", "sum"),
            funding_cashflow_conservative_bps=("conservative_cashflow_bps", "sum"),
            funding_cashflow_severe_bps=("severe_cashflow_bps", "sum"),
        ).reset_index()
    out = trades.merge(sums, on="economic_address", how="left", validate="one_to_one")
    out[["exact_boundary_count", "imputed_boundary_count"]] = out[["exact_boundary_count", "imputed_boundary_count"]].fillna(0).astype(int)
    for field in ("funding_cashflow_central_bps", "funding_cashflow_conservative_bps", "funding_cashflow_severe_bps"):
        out[field] = out[field].fillna(0.0)
    out["funding_partition"] = [contract.funding_partition(row) for row in out.to_dict("records")]
    return out, boundary


def score_trades(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    if out.empty:
        return out
    gross, base, stress = [], [], []
    for row in out.itertuples(index=False):
        gross_row = contract.fixed_notional_net_bps(entry_price=row.entry_price, exit_price=row.exit_price, side=row.side, fee_bps=0, slippage_bps=0, funding_cashflow_bps=0)
        base_row = contract.fixed_notional_net_bps(entry_price=row.entry_price, exit_price=row.exit_price, side=row.side, fee_bps=10, slippage_bps=4, funding_cashflow_bps=0)
        stress_row = contract.fixed_notional_net_bps(entry_price=row.entry_price, exit_price=row.exit_price, side=row.side, fee_bps=20, slippage_bps=12, funding_cashflow_bps=0)
        gross.append(gross_row["gross_return_bps"]); base.append(base_row["net_return_bps"]); stress.append(stress_row["net_return_bps"])
    out["gross_return_bps"] = gross
    out["base_fee_slippage_net_bps"] = base
    out["stress_fee_slippage_net_bps"] = stress
    out["base_funding_adjusted_net_bps"] = out.base_fee_slippage_net_bps + out.funding_cashflow_central_bps
    out["conservative_funding_adjusted_net_bps"] = out.base_fee_slippage_net_bps + out.funding_cashflow_conservative_bps
    out["severe_funding_adjusted_net_bps"] = out.stress_fee_slippage_net_bps + out.funding_cashflow_severe_bps
    out["structural_R"] = np.where(out.side.eq("long"), (out.exit_price-out.entry_price)/out.risk_denominator, (out.entry_price-out.exit_price)/out.risk_denominator)
    return out


def compute_reports(register: pd.DataFrame, trades: pd.DataFrame, eligibility: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, concentration_rows, bootstrap_rows, gate_rows, funding_rows = [], [], [], [], []
    for definition in register.to_dict("records"):
        did = definition["definition_id"]
        group = trades[trades.definition_id.eq(did)] if not trades.empty else trades
        eligible_group = eligibility[eligibility.definition_id.eq(did)]
        counts = eligible_group.status.value_counts().to_dict()
        years = group.calendar_year.value_counts().to_dict() if not group.empty else {}
        concentration = concentration_metrics(group)
        if group.empty:
            low = high = math.nan
        else:
            episode_values = group.groupby("canonical_episode_id").base_fee_slippage_net_bps.apply(list).to_dict()
            low, high = contract.canonical_episode_bootstrap_mean_ci(episode_values)
        metrics = {
            **definition, "onset_count": len(eligible_group), "confirmed_count": int((eligible_group.confirmed == True).sum()),
            "invalid_count": int(counts.get("invalid", 0)), "skipped_count": int(counts.get("skipped_actual_overlap", 0)),
            "executed_trades": len(group), "trade_count_2023": int(years.get(2023, 0)),
            "trade_count_2024": int(years.get(2024, 0)), "trade_count_2025": int(years.get(2025, 0)),
            "trade_count_by_symbol_json": json.dumps(group.symbol.value_counts().sort_index().to_dict() if len(group) else {}, sort_keys=True),
            "gross_mean_bps": float(group.gross_return_bps.mean()) if len(group) else math.nan,
            "gross_median_bps": float(group.gross_return_bps.median()) if len(group) else math.nan,
            "mean_net_bps": float(group.base_fee_slippage_net_bps.mean()) if len(group) else math.nan,
            "median_net_bps": float(group.base_fee_slippage_net_bps.median()) if len(group) else math.nan,
            "stress_mean_net_bps": float(group.stress_fee_slippage_net_bps.mean()) if len(group) else math.nan,
            "stress_median_net_bps": float(group.stress_fee_slippage_net_bps.median()) if len(group) else math.nan,
            "bootstrap_ci_lower_bps": low, "bootstrap_ci_upper_bps": high, **concentration,
        }
        gate_input = {**metrics, "trade_count_by_year": years}
        flags = contract.level3_gate_flags(gate_input)
        metric_rows.append(metrics)
        concentration_rows.append({"definition_id": did, **concentration})
        bootstrap_rows.append({"definition_id": did, "canonical_episode_count": int(group.canonical_episode_id.nunique()) if len(group) else 0, "resamples": contract.BOOTSTRAP_RESAMPLES, "seed": contract.BOOTSTRAP_SEED, "ci_lower_bps": low, "ci_upper_bps": high})
        gate_rows.append({"definition_id": did, "model": definition["model"], **flags, "all_gates_pass": all(flags.values())})
        for partition in ("fully_exact", "mixed", "fully_imputed", "zero_boundary"):
            part = group[group.funding_partition.eq(partition)] if len(group) else group
            for period, period_group in [("full", part), ("2023", part[part.calendar_year.eq(2023)]), ("2024", part[part.calendar_year.eq(2024)]), ("2025", part[part.calendar_year.eq(2025)])]:
                funding_rows.append({
                    "definition_id": did, "funding_partition": partition, "calendar_period": period,
                    "trade_count": len(period_group),
                    "central_mean_net_bps": float(period_group.base_funding_adjusted_net_bps.mean()) if len(period_group) else math.nan,
                    "conservative_mean_net_bps": float(period_group.conservative_funding_adjusted_net_bps.mean()) if len(period_group) else math.nan,
                    "severe_mean_net_bps": float(period_group.severe_funding_adjusted_net_bps.mean()) if len(period_group) else math.nan,
                })
    return tuple(pd.DataFrame(rows) for rows in (metric_rows, gate_rows, funding_rows, concentration_rows, bootstrap_rows))  # type: ignore[return-value]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def artifact_manifest(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "ARTIFACT_MANIFEST.json"):
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {"task_id": TASK_ID, "file_count": len(rows), "files": rows, "self_hash_excluded": True}
    write_json(root / "ARTIFACT_MANIFEST.json", manifest)
    return manifest


def verify_authorities(contract_path: Path, register_path: Path, event_path: Path) -> dict[str, Any]:
    checks = {
        str(contract_path): (sha256_file(contract_path), CONTRACT_SHA256),
        str(register_path): (sha256_file(register_path), REGISTER_SHA256),
        str(event_path): (sha256_file(event_path), EVENT_TAPE_SHA256),
        str(STAGE2D_ROOT / "C01_LEVEL3_DECISION_RULES.json"): (sha256_file(STAGE2D_ROOT / "C01_LEVEL3_DECISION_RULES.json"), DECISION_RULES_SHA256),
        str(STAGE2D_ROOT / "C01_LEVEL4_CONTROL_CONTRACT.md"): (sha256_file(STAGE2D_ROOT / "C01_LEVEL4_CONTROL_CONTRACT.md"), CONTROL_CONTRACT_SHA256),
        str(STAGE2C_ROOT / "C01_FROZEN_GENERATOR_CONTRACT.md"): (sha256_file(STAGE2C_ROOT / "C01_FROZEN_GENERATOR_CONTRACT.md"), GENERATOR_SHA256),
        str(STAGE2C_ROOT / "C01_ECONOMIC_CONTRACT_DRAFT.md"): (sha256_file(STAGE2C_ROOT / "C01_ECONOMIC_CONTRACT_DRAFT.md"), ECONOMIC_DRAFT_SHA256),
    }
    failures = [path for path, (actual, expected) in checks.items() if actual != expected]
    if failures:
        raise ValueError(f"frozen authority hash mismatch: {failures}")
    result = {path: {"actual": actual, "expected": expected, "pass": actual == expected} for path, (actual, expected) in checks.items()}
    stage2b_manifest = json.loads((REPO_ROOT / "docs/agent/task_archive/20260717_donch_bt_stage_2b_c01_foundation_20260717_v1/ARTIFACT_MANIFEST.json").read_text())
    stage2c_summary = json.loads((STAGE2C_ROOT / "C01_STAGE2C_SUMMARY.json").read_text())
    semantic = {
        "feature_contract_hash": (str(stage2b_manifest.get("feature_contract_hash")), FEATURE_SHA256),
        "cohort_hash": (str(stage2c_summary.get("cohort_hash")), COHORT_SHA256),
        "reference_panel_hash": (str(stage2c_summary.get("reference_panel_hash")), REFERENCE_PANEL_SHA256),
    }
    semantic_failures = [name for name, (actual, expected) in semantic.items() if actual != expected]
    if semantic_failures:
        raise ValueError(f"frozen semantic authority mismatch: {semantic_failures}")
    result.update({name: {"actual": actual, "expected": expected, "pass": True, "hash_scope": "semantic"} for name, (actual, expected) in semantic.items()})
    return result


def execute(args: argparse.Namespace) -> None:
    if not args.execute_economic_run:
        raise ValueError("economic execution requires --execute-economic-run")
    root = Path(args.run_root)
    if root.exists():
        raise FileExistsError(f"fresh run root required: {root}")
    authorities = verify_authorities(Path(args.contract), Path(args.definition_register), Path(args.event_tape))
    schema = set(pq.ParquetFile(args.event_tape).schema_arrow.names)
    if OUTCOME_COLUMNS & schema:
        raise ValueError("mixed event/outcome tape rejected before reader")
    root.mkdir(parents=True)
    started = time.monotonic()
    runner_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    register = validate_definition_register(pd.read_csv(args.definition_register))
    events = validate_event_tape(pd.read_parquet(args.event_tape))
    economic_events = events[events.path_state.isin(["smooth", "jump_dominated"])].copy()
    authority_rows = foundation.load_safe_manifest(MARKET_MANIFEST)
    data_authority_hash = foundation.authority_hash(authority_rows, sha256_file(REFERENCE_MANIFEST), sha256_file(LIFECYCLE_SOURCE))
    if data_authority_hash != DATA_AUTHORITY_SHA256:
        raise ValueError("market/lifecycle data authority hash mismatch")
    invalidations = foundation.load_known_lifecycle_invalidations(LIFECYCLE_SOURCE)
    instrument_payload = json.loads(INSTRUMENT_SOURCE.read_text(encoding="utf-8"))
    opening_dates = {
        str(row["symbol"]): utc(row["openingDate"])
        for row in instrument_payload.get("instruments", [])
        if isinstance(row, Mapping) and row.get("symbol") and row.get("openingDate")
    }
    if set(economic_events.symbol) - set(opening_dates):
        raise ValueError("event symbol missing official opening date")
    btc_trade = load_authorized_ohlc(authority_rows, "PF_XBTUSD", "historical_trade_candles_5m")
    btc_mark = load_authorized_ohlc(authority_rows, "PF_XBTUSD", "historical_mark_candles_5m")
    eth_trade = load_authorized_ohlc(authority_rows, "PF_ETHUSD", "historical_trade_candles_5m")
    eth_mark = load_authorized_ohlc(authority_rows, "PF_ETHUSD", "historical_mark_candles_5m")
    anchors: list[dict[str, Any]] = []
    for number, symbol in enumerate(sorted(economic_events.symbol.unique()), start=1):
        trade = load_authorized_ohlc(authority_rows, symbol, "historical_trade_candles_5m")
        mark = load_authorized_ohlc(authority_rows, symbol, "historical_mark_candles_5m")
        for model in (contract.PRIMARY_MODEL, contract.ROBUSTNESS_MODEL):
            residuals = residual_components(trade, mark, btc_trade, btc_mark, eth_trade, eth_mark, model, invalidations.get(symbol, ()), opening_dates[symbol])
            model_events = economic_events[(economic_events.symbol == symbol) & (economic_events.residual_model_version == model)]
            for event in model_events.to_dict("records"):
                anchors.append(freeze_event_anchor(event, residuals, trade))
        write_json(root / "PROGRESS.json", {"stage": "causal_anchor_freeze", "symbols_completed": number, "symbols_total": economic_events.symbol.nunique(), "anchors": len(anchors), "elapsed_seconds": time.monotonic()-started})
    anchor_frame = pd.DataFrame(anchors).sort_values(["symbol", "event_id"], kind="mergesort")
    if len(anchor_frame) != len(economic_events) or anchor_frame.event_id.duplicated().any():
        raise ValueError("causal event-anchor freeze does not reconcile")
    anchor_frame.to_parquet(root / "CAUSAL_EVENT_ANCHOR_FREEZE.parquet", index=False)
    anchor_hash = sha256_file(root / "CAUSAL_EVENT_ANCHOR_FREEZE.parquet")
    write_json(root / "CAUSAL_EVENT_ANCHOR_FREEZE.json", {"rows": len(anchor_frame), "sha256": anchor_hash, "frozen_before_post_onset_outcome_simulation": True})
    anchor_index = anchor_frame.set_index("event_id")
    prepared, ledger = [], []
    for number, symbol in enumerate(sorted(economic_events.symbol.unique()), start=1):
        trade = load_authorized_ohlc(authority_rows, symbol, "historical_trade_candles_5m")
        mark = load_authorized_ohlc(authority_rows, symbol, "historical_mark_candles_5m")
        for model in (contract.PRIMARY_MODEL, contract.ROBUSTNESS_MODEL):
            model_events = economic_events[(economic_events.symbol == symbol) & (economic_events.residual_model_version == model)]
            definitions = register[register.model.eq(model)]
            for event in model_events.to_dict("records"):
                matching = definitions[(definitions.path_state == event["path_state"]) & (definitions.shock_sign == event["sign"])]
                for definition in matching.to_dict("records"):
                    base = {"definition_id": definition["definition_id"], "event_id": event["event_id"], "symbol": symbol, "onset_ts": event["decision_ts"], "confirmed": False}
                    try:
                        candidate = prepare_candidate(event, definition, trade, mark, pd.DataFrame(), invalidations.get(symbol, ()), anchor_index.loc[event["event_id"]].to_dict())
                        prepared.append(candidate)
                        ledger.append({**base, "economic_address": candidate["economic_address"], "confirmed": True, "status": "prepared_for_actual_exit_non_overlap", "reason": ""})
                    except CandidateInvalid as exc:
                        ledger.append({**base, "status": "invalid", "reason": str(exc)})
        write_json(root / "PROGRESS.json", {"stage": "candidate_path_simulation", "symbols_completed": number, "symbols_total": economic_events.symbol.nunique(), "prepared_rows": len(prepared), "elapsed_seconds": time.monotonic()-started})
    overlap = contract.definition_local_non_overlap(prepared)
    accepted = pd.DataFrame(overlap.accepted)
    skipped = pd.DataFrame(overlap.skipped)
    accepted_addresses = set(accepted.economic_address) if len(accepted) else set()
    skipped_addresses = set(skipped.economic_address) if len(skipped) else set()
    for row in ledger:
        address = row.get("economic_address")
        if not address:
            continue
        if address in accepted_addresses:
            candidate = accepted[accepted.economic_address.eq(address)].iloc[0]
            row.update({"status": "executed", "actual_exit_ts": candidate.actual_exit_ts, "reason": ""})
        elif address in skipped_addresses:
            prior = skipped[skipped.economic_address.eq(address)].iloc[0]
            row.update({"economic_address": address, "status": "skipped_actual_overlap", "actual_exit_ts": candidate["actual_exit_ts"], "reason": prior.skip_reason, "prior_economic_address": prior.prior_economic_address})
    eligibility = pd.DataFrame(ledger)
    if len(accepted) and accepted.economic_address.duplicated().any():
        raise ValueError("duplicate accepted economic address")
    accepted_hash = stable_hash(sorted(accepted.economic_address.tolist())) if len(accepted) else stable_hash([])
    panel, global_location, funding_model_hash = load_funding_panel()
    funded, funding_boundaries = attach_funding(accepted, panel, global_location)
    trades = score_trades(funded)
    if len(trades):
        trades["runner_commit"] = runner_commit
        trades["contract_sha256"] = CONTRACT_SHA256
        trades["event_tape_sha256"] = EVENT_TAPE_SHA256
        trades["market_manifest_sha256"] = sha256_file(MARKET_MANIFEST)
        trades["funding_model_hash"] = funding_model_hash
        trades["trade_path_reference"] = "safe_market_manifest:historical_trade_candles_5m:" + trades.symbol
        trades["mark_path_reference"] = "safe_market_manifest:historical_mark_candles_5m:" + trades.symbol
        trades["eligibility_status"] = "executed"
        trades["invalid_or_skip_reason"] = ""
        trades["protected_row_count"] = 0
    metrics, gates, funding_report, concentrations, bootstraps = compute_reports(register, trades, eligibility)
    permitted = contract.definitions_permitted_for_level4(metrics.to_dict("records"))
    decision = "level3_primary_pass_controls_pending_separate_approval" if permitted else "level3_no_primary_pass_stop"
    register.to_csv(root / "DEFINITION_REGISTER.csv", index=False)
    eligibility.to_parquet(root / "EVENT_ELIGIBILITY_AND_SKIP_LEDGER.parquet", index=False)
    trades.to_parquet(root / "TRADE_LEDGER.parquet", index=False)
    metrics.to_csv(root / "DEFINITION_METRICS.csv", index=False)
    gates.to_csv(root / "LEVEL3_GATE_MATRIX.csv", index=False)
    funding_report.to_csv(root / "FUNDING_PARTITION_REPORT.csv", index=False)
    concentrations.to_csv(root / "CONCENTRATION_REPORT.csv", index=False)
    bootstraps.to_csv(root / "BOOTSTRAP_REPORT.csv", index=False)
    funding_boundaries.to_parquet(root / "FUNDING_BOUNDARY_LEDGER.parquet", index=False)
    input_audit = {
        "frozen_hashes": authorities, "market_manifest": str(MARKET_MANIFEST), "market_manifest_sha256": sha256_file(MARKET_MANIFEST),
        "safe_market_shards": len(authority_rows), "funding_model_hash": funding_model_hash,
        "funding_manifest_sha256": sha256_file(FUNDING_ROOT / "funding/shared_funding_panel_manifest.csv"),
        "generator_hash": GENERATOR_SHA256, "feature_hash": FEATURE_SHA256, "cohort_hash": COHORT_SHA256,
        "reference_panel_hash": REFERENCE_PANEL_SHA256,
        "data_authority_hash": data_authority_hash, "causal_event_anchor_freeze_hash": anchor_hash,
    }
    write_json(root / "INPUT_AND_HASH_AUDIT.json", input_audit)
    period_audit = {
        "train_start": iso(TRAIN_START), "train_end_exclusive": iso(TRAIN_END),
        "event_tape_protected_rows_opened": 0, "market_protected_rows_opened": 0,
        "funding_protected_rows_opened": 0, "trade_ledger_protected_rows": int((trades.actual_exit_ts >= TRAIN_END).sum()) if len(trades) else 0,
        "artificial_endpoint_exits": int(trades.exit_reason.astype(str).str.contains("artificial", case=False).sum()) if len(trades) else 0,
    }
    write_json(root / "PERIOD_AND_PROTECTED_AUDIT.json", period_audit)
    manifest = {
        "task_id": TASK_ID, "family_id": FAMILY_ID, "status": "complete", "runner_commit": runner_commit,
        "contract_sha256": CONTRACT_SHA256, "event_tape_sha256": EVENT_TAPE_SHA256,
        "definition_count": len(register), "economic_onset_count": len(economic_events),
        "executed_trade_rows": len(trades), "accepted_trade_hash": accepted_hash,
        "family_decision": decision, "level4_permitted_definitions": permitted,
        "level4_controls_run": False, "protected_rows_opened": 0,
        "runtime_seconds": time.monotonic()-started,
    }
    write_json(root / "RUN_MANIFEST.json", manifest)
    (root / "DECISION.md").write_text(
        f"# C01 Level-3 Decision\n\nDecision: `{decision}`.\n\nPrimary definitions passing every frozen gate: `{permitted}`. "
        "This is train-period Level-3 kill-screen evidence only. Level-4 controls were not run and require separate human approval.\n",
        encoding="utf-8",
    )
    validation_pass = bool(
        len(register) == 16 and len(metrics) == 16 and len(gates) == 16
        and period_audit["trade_ledger_protected_rows"] == 0 and period_audit["artificial_endpoint_exits"] == 0
        and (trades.empty or not trades.economic_address.duplicated().any())
    )
    (root / "VALIDATION.md").write_text(
        f"# Validation\n\nStatus: `{'pass' if validation_pass else 'fail'}`. Definitions: `{len(register)}`; trade rows: `{len(trades)}`; "
        f"protected rows opened: `0`; artificial endpoint exits: `{period_audit['artificial_endpoint_exits']}`; Level-4 controls: `not run`.\n",
        encoding="utf-8",
    )
    (root / "REVIEW.md").write_text(
        "# Post-Run Review\n\nPending independent artifact recomputation. The generated result is not publishable until this file is replaced by the post-run review.\n",
        encoding="utf-8",
    )
    artifact_manifest(root)
    if not validation_pass:
        raise RuntimeError("post-run validation failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--definition-register", required=True)
    parser.add_argument("--event-tape", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--execute-economic-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    execute(parse_args())
