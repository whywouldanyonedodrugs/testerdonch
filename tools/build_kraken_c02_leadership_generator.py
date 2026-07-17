#!/usr/bin/env python3
"""Build the outcome-free C02 Kraken spot/perpetual leadership tape."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_kraken_c01_foundation as c01


TASK_ID = "donch_bt_stage_3b_c02_leadership_generator_20260717_v1"
FAMILY_ID = "C02_spot_led_vs_perp_led_impulse"
FEATURE_VERSION = "c02_causal_features_v1_20260717"
IDENTITY_VERSION = "c02_identity_v1_20260717"
TRAIN_START = pd.Timestamp("2023-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2026-01-01T00:00:00Z")
SPOT_MANIFEST_HASH = "3de3b533a390f04590ae458ac661d4fc10d299df1a1833734911fa609b0a7046"
COHORT_HASH = "768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15"
BAR = pd.Timedelta(minutes=5)
PROHIBITED_OUTPUT_TOKENS = (
    "forward_return", "post_decision", "pnl", "net_r", "gross_r", "mae", "mfe",
    "expectancy", "profit", "promotion", "exit_price",
)

CONTRACT_TEXT = """# C02 Generator Contract

Status: frozen before generation. Family: `C02_spot_led_vs_perp_led_impulse`.

## Authority and boundaries

Only official sparse Kraken USD spot five-minute bars, manifest-authorized Kraken PF trade/mark five-minute bars, the Stage 2C prior-day Top-100 panel, and known terminal lifecycle invalidations are inputs. The interval is `[2023-01-01, 2026-01-01)`. No sparse interval is filled. The cohort remains current-roster capped and not survivorship-free.

## Causal rules

Exact aligned completed bars use `feature_available_ts = interval_open + 5m`. Fifteen-minute returns require four consecutive observed five-minute boundaries. Daily sample scales use the preceding 30 UTC days only and require 2,000 observations. Activation requires same sign, maximum directional z at least 3.0, minimum at least 1.5, and a 60-minute same-direction reset. Primary/robustness leadership lookbacks are 15/30 minutes and use the first follower-threshold crossing. Perp-led failure requires the first PF trade and mark close beyond the three-bar impulse-window extreme within six hours.

## Eligibility and identity

A symbol-day requires Stage 2C Top-100 membership, exact spot identity and USD unit identity, at least 20 of the prior 30 days at 70% exact intersection, at least 70% complete-window intersection, valid scales, and no known lifecycle-invalid overlap. Event, economic-address, and overlapping same-symbol episode identities are deterministic and outcome-free.

## Prohibitions

No post-decision return, exit, PnL, MAE/MFE, control result, promotion metric, funding, OI, index, breadth, session, catalyst, prior-high, C01 residual, interpolation, event selection by magnitude, or economic ranking is permitted.
"""


def write_prefrozen_contracts(output: Path) -> None:
    (output / "C02_GENERATOR_CONTRACT.md").write_text(CONTRACT_TEXT, encoding="utf-8")
    schema = {
        "version": FEATURE_VERSION,
        "train_interval": [TRAIN_START.isoformat(), PROTECTED_START.isoformat()],
        "decision_semantics": "completed_exact_5m_bar; feature_available_ts<=decision_ts",
        "features": {
            "spot_r_15m": "log close ratio across four consecutive observed spot intervals",
            "perp_r_15m": "log close ratio across four consecutive observed PF trade intervals",
            "spot_z_15m": "return divided by prior-30-UTC-day sample std; min 2000",
            "perp_z_15m": "return divided by prior-30-UTC-day sample std; min 2000",
            "leadership_state": "first directional z>=1.5 crossing in frozen lookback",
            "volume_surprise": "15m close-based USD-volume proxy / prior-30-day median; diagnostic",
            "price_gaps": "same-decision spot/PF trade/mark close diagnostics",
        },
        "prohibited_output_tokens": list(PROHIBITED_OUTPUT_TOKENS),
        "outcome_fields": [],
    }
    (output / "C02_FEATURE_SCHEMA.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_outcome_fields(columns: Iterable[str]) -> None:
    bad = sorted(c for c in columns if any(token in c.lower() for token in PROHIBITED_OUTPUT_TOKENS))
    if bad:
        raise ValueError(f"outcome-derived fields prohibited: {bad}")


def make_attempt_register() -> pd.DataFrame:
    rows = []
    states = ["spot_led_continuation", "simultaneous_impulse", "perp_led_continuation",
              "perp_led_completed_failure", "ambiguous_diagnostic"]
    for shift in (-5, 0, 5):
        for lookback in (15, 30):
            for direction in ("positive", "negative"):
                for state in states:
                    policy = {"alignment_shift_minutes": shift, "leadership_lookback_minutes": lookback,
                              "direction": direction, "state": state}
                    rows.append({"family_id": FAMILY_ID, "attempt_id": "c02attempt_" + stable_hash(policy)[:20],
                                 **policy, "attempt_role": "primary" if shift == 0 and lookback == 15 else "robustness_or_sensitivity",
                                 "registered_before_generation": True, "event_count": 0, "retention_status": "retained"})
    return pd.DataFrame(rows)


def validate_spot_frame(frame: pd.DataFrame, pair: str) -> pd.DataFrame:
    required = {"timestamp", "close", "volume", "source_close_ts", "feature_available_ts", "Kraken_spot_pair", "venue"}
    if not required.issubset(frame.columns):
        raise ValueError(f"spot schema missing: {sorted(required - set(frame.columns))}")
    out = frame.copy()
    for col in ("timestamp", "source_close_ts", "feature_available_ts"):
        out[col] = pd.to_datetime(out[col], utc=True, errors="raise")
    if out.empty or not out["Kraken_spot_pair"].astype(str).eq(pair).all() or not out["venue"].astype(str).str.lower().eq("kraken").all():
        raise ValueError(f"spot identity mismatch: {pair}")
    if (out["timestamp"] < TRAIN_START).any() or (out["timestamp"] >= PROTECTED_START).any():
        raise ValueError("protected/pretrain spot row reached normalization")
    if not out["source_close_ts"].eq(out["timestamp"] + BAR).all() or not out["feature_available_ts"].eq(out["source_close_ts"]).all():
        raise ValueError("spot availability semantics mismatch")
    out["close"] = pd.to_numeric(out["close"], errors="raise")
    out["volume"] = pd.to_numeric(out["volume"], errors="raise")
    if (out["close"] <= 0).any() or (out["volume"] < 0).any():
        raise ValueError("invalid spot close/volume")
    out = out.sort_values("timestamp", kind="mergesort")
    if out["timestamp"].duplicated().any():
        raise ValueError("duplicate normalized spot interval")
    return out[["timestamp", "close", "volume", "source_close_ts", "feature_available_ts"]].reset_index(drop=True)


def read_spot(path: Path, pair: str, expected_hash: str) -> pd.DataFrame:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"spot content hash mismatch: {pair}")
    return validate_spot_frame(pd.read_parquet(path), pair)


def read_pf_bars(rows: Sequence[c01.AuthorityRow], symbol: str, dataset: str) -> tuple[pd.DataFrame, str]:
    selected = [row for row in rows if row.symbol == symbol and row.dataset == dataset]
    if not selected:
        raise ValueError(f"no authorized {dataset} rows for {symbol}")
    parts = []
    base_required = {"time", "close", "venue_symbol", "resolution", "rankable_pre_holdout", "contains_protected_period"}
    requested = set(base_required)
    if dataset == "historical_trade_candles_5m":
        requested |= {"high", "low", "volume"}
    for authority in selected:
        schema = set(pq.ParquetFile(authority.parquet_path).schema_arrow.names)
        if "time" not in schema and authority.rows <= 1:
            continue
        if not requested.issubset(schema):
            raise ValueError(f"PF schema mismatch before payload read: {authority.parquet_path}")
        raw = pd.read_parquet(authority.parquet_path, columns=sorted(requested))
        if raw.empty:
            continue
        if not raw["venue_symbol"].astype(str).eq(symbol).all() or not raw["resolution"].astype(str).eq("5m").all():
            raise ValueError("PF identity/resolution mismatch")
        if not raw["rankable_pre_holdout"].map(c01._as_bool).all() or raw["contains_protected_period"].map(c01._as_bool).any():
            raise ValueError("unrankable PF row reached normalization")
        raw["timestamp"] = pd.to_datetime(pd.to_numeric(raw["time"], errors="raise"), unit="ms", utc=True)
        if (raw["timestamp"] < TRAIN_START).any() or (raw["timestamp"] >= PROTECTED_START).any():
            raise ValueError("protected/pretrain PF row reached normalization")
        keep = ["timestamp", "close"] + (["high", "low", "volume"] if dataset.endswith("trade_candles_5m") else [])
        parts.append(raw[keep])
    if not parts:
        raise ValueError(f"no nonempty authorized {dataset} payload for {symbol}")
    out = pd.concat(parts, ignore_index=True).sort_values("timestamp", kind="mergesort")
    duplicated = out.duplicated("timestamp", keep=False)
    if duplicated.any():
        value_cols = [c for c in out.columns if c != "timestamp"]
        if out.loc[duplicated].groupby("timestamp")[value_cols].nunique().gt(1).any(axis=None):
            raise ValueError(f"conflicting duplicate PF interval: {dataset}:{symbol}")
        out = out.drop_duplicates("timestamp", keep="first")
    for col in out.columns.drop("timestamp"):
        out[col] = pd.to_numeric(out[col], errors="raise")
    if (out["close"] <= 0).any():
        raise ValueError("non-positive PF close")
    ref = stable_hash([row.reference_id for row in selected])
    return out.reset_index(drop=True), f"c02:{dataset}:{symbol}:{ref}"


def align_exact(spot: pd.DataFrame, trade: pd.DataFrame, mark: pd.DataFrame, *, spot_shift_minutes: int = 0) -> pd.DataFrame:
    s = spot.rename(columns={"close": "spot_close", "volume": "spot_volume"}).copy()
    s["timestamp"] = s["timestamp"] + pd.Timedelta(minutes=spot_shift_minutes)
    t = trade.rename(columns={"close": "perp_close", "high": "perp_high", "low": "perp_low", "volume": "perp_volume"})
    m = mark.rename(columns={"close": "mark_close"})
    out = s.merge(t, on="timestamp", how="inner", validate="one_to_one").merge(m, on="timestamp", how="inner", validate="one_to_one")
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if out["timestamp"].duplicated().any():
        raise ValueError("duplicate exact intersection")
    out["feature_available_ts"] = out["timestamp"] + BAR
    return out


def complete_return(frame: pd.DataFrame, close_col: str) -> pd.Series:
    result = np.log(frame[close_col] / frame[close_col].shift(3))
    complete = frame["timestamp"].sub(frame["timestamp"].shift(3)).eq(pd.Timedelta(minutes=15))
    for lag in (1, 2, 3):
        complete &= frame["timestamp"].diff().eq(BAR).shift(lag - 1, fill_value=False)
    return result.where(complete)


def prior_daily_scale(returns: pd.Series, timestamps: pd.Series) -> pd.DataFrame:
    work = pd.DataFrame({"utc_day": timestamps.dt.floor("D"), "r": returns}).dropna()
    work["r2"] = work["r"] ** 2
    daily = work.groupby("utc_day", sort=True).agg(n=("r", "size"), total=("r", "sum"), total2=("r2", "sum"))
    calendar = pd.date_range(TRAIN_START.floor("D"), PROTECTED_START.floor("D"), freq="D", inclusive="left")
    daily = daily.reindex(calendar, fill_value=0)
    prior = daily.shift(1).rolling(30, min_periods=1).sum()
    variance = (prior.total2 - prior.total.pow(2) / prior.n.where(prior.n > 0)) / (prior.n - 1).where(prior.n > 1)
    scale = np.sqrt(variance.clip(lower=0)).where(prior.n >= 2000)
    return pd.DataFrame({"utc_day": calendar, "valid_observations": prior.n.fillna(0).astype(int).values, "scale": scale.values})


def prior_daily_median(values: pd.Series, timestamps: pd.Series, days: Sequence[pd.Timestamp] | None = None) -> pd.DataFrame:
    valid = values.notna()
    value_array = values.loc[valid].to_numpy(dtype=float)
    time_array = timestamps.loc[valid].astype("int64").to_numpy()
    calendar = pd.DatetimeIndex(days) if days is not None else pd.date_range(TRAIN_START.floor("D"), PROTECTED_START.floor("D"), freq="D", inclusive="left")
    medians = []
    for day in calendar:
        left = np.searchsorted(time_array, (day - pd.Timedelta(days=30)).value, side="left")
        right = np.searchsorted(time_array, day.value, side="left")
        medians.append(float(np.median(value_array[left:right])) if right > left else np.nan)
    return pd.DataFrame({"utc_day": calendar, "median": medians})


def daily_intersection_audit(frame: pd.DataFrame, symbol: str, pair: str) -> pd.DataFrame:
    observed = frame.assign(utc_day=frame.timestamp.dt.floor("D")).groupby("utc_day").size()
    calendar = pd.date_range(TRAIN_START.floor("D"), PROTECTED_START.floor("D"), freq="D", inclusive="left")
    count = observed.reindex(calendar, fill_value=0).astype(int)
    ratio = count / 288.0
    prior_good_days = ratio.ge(0.70).astype(int).shift(1).rolling(30, min_periods=30).sum()
    prior_rows = count.shift(1).rolling(30, min_periods=30).sum()
    prior_ratio = prior_rows / (30 * 288)
    out = pd.DataFrame({"utc_day": calendar, "PF_symbol": symbol, "Kraken_spot_pair": pair,
                        "exact_intersection_rows": count.values, "daily_intersection_ratio": ratio.values,
                        "prior_30d_days_ge_70pct": prior_good_days.values,
                        "prior_30d_complete_intersection_ratio": prior_ratio.values})
    out["coverage_eligible"] = out.prior_30d_days_ge_70pct.ge(20) & out.prior_30d_complete_intersection_ratio.ge(0.70)
    out["price_unit_equivalence_verified"] = True
    out["price_unit_authority"] = "exact_same_canonical_base;official_USD_spot;Kraken_PF_USD_linear_perpetual"
    out["survivorship_free_claim"] = False
    return out


def complete_daily_eligibility(audit: pd.DataFrame, cohort: pd.DataFrame,
                               invalidations: Sequence[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.DataFrame:
    out = audit.merge(cohort[["utc_day", "top_100_eligible", "rank"]], on="utc_day", how="left", validate="one_to_one")
    out["top_100_eligible"] = out.top_100_eligible.eq(True)
    out["lifecycle_window_valid"] = [
        not any(day - pd.Timedelta(days=30) < end and day + pd.Timedelta(days=1) > start
                for start, end in invalidations)
        for day in out.utc_day
    ]
    out["eligible_before_scale"] = out.coverage_eligible & out.top_100_eligible & out.lifecycle_window_valid & out.price_unit_equivalence_verified
    out["eligibility_reason"] = np.select(
        [~out.coverage_eligible, ~out.top_100_eligible, ~out.lifecycle_window_valid, ~out.price_unit_equivalence_verified],
        ["insufficient_exact_intersection_history", "not_stage2c_daily_top100", "known_lifecycle_invalid_interval", "price_unit_equivalence_unverified"],
        default="eligible_before_prior_scale_check",
    )
    return out


def classify_leadership(frame: pd.DataFrame, index: int, direction: int, lookback_minutes: int) -> str:
    start_ts = frame.at[index, "timestamp"] - pd.Timedelta(minutes=lookback_minutes)
    start_position = int(frame["timestamp"].searchsorted(start_ts, side="left"))
    def first_crossing(column: str) -> pd.Timestamp | None:
        qualified = direction * frame[column] >= 1.5
        candidates = []
        for position in range(max(1, start_position), index + 1):
            timestamp = frame.at[position, "timestamp"]
            if (qualified.iloc[position] and not qualified.iloc[position - 1]
                    and timestamp - frame.at[position - 1, "timestamp"] == BAR):
                candidates.append(timestamp)
        return candidates[0] if candidates else None
    s, p = first_crossing("spot_z_15m"), first_crossing("perp_z_15m")
    if s is None or p is None:
        return "ambiguous"
    if s == p:
        return "simultaneous"
    return "spot_led" if s <= p - BAR else "perp_led"


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["spot_r_15m"] = complete_return(out, "spot_close")
    out["perp_r_15m"] = complete_return(out, "perp_close")
    out["mark_r_15m"] = complete_return(out, "mark_close")
    sscale = prior_daily_scale(out.spot_r_15m, out.timestamp).rename(columns={"valid_observations": "spot_scale_observations", "scale": "spot_prior_scale"})
    pscale = prior_daily_scale(out.perp_r_15m, out.timestamp).rename(columns={"valid_observations": "perp_scale_observations", "scale": "perp_prior_scale"})
    out["utc_day"] = out.timestamp.dt.floor("D")
    out = out.merge(sscale, on="utc_day", how="left", validate="many_to_one").merge(pscale, on="utc_day", how="left", validate="many_to_one")
    out["spot_z_15m"] = out.spot_r_15m / out.spot_prior_scale
    out["perp_z_15m"] = out.perp_r_15m / out.perp_prior_scale
    out["spot_usd_volume_15m"] = (out.spot_close * out.spot_volume).rolling(3, min_periods=3).sum()
    out["perp_usd_volume_15m"] = (out.perp_close * out.perp_volume).rolling(3, min_periods=3).sum()
    out["perp_spot_gap_bps"] = 10000 * np.log(out.perp_close / out.spot_close)
    out["perp_spot_gap_change_15m_bps"] = out.perp_spot_gap_bps - out.perp_spot_gap_bps.shift(3)
    out["mark_trade_gap_bps"] = 10000 * np.log(out.mark_close / out.perp_close)
    return out


def onset_indices(frame: pd.DataFrame) -> list[tuple[int, int]]:
    same_sign = np.sign(frame.spot_r_15m) == np.sign(frame.perp_r_15m)
    valid_sign = same_sign & frame.spot_r_15m.ne(0) & frame.perp_r_15m.ne(0)
    directions = np.sign(frame.spot_r_15m).where(valid_sign, 0).fillna(0).astype(int)
    active = pd.Series(False, index=frame.index)
    for direction in (-1, 1):
        sz = direction * frame.spot_z_15m
        pz = direction * frame.perp_z_15m
        active |= directions.eq(direction) & np.maximum(sz, pz).ge(3.0) & np.minimum(sz, pz).ge(1.5)
    rows: list[tuple[int, int]] = []
    for i in np.flatnonzero(active.to_numpy()):
        if i < 12:
            continue
        prior = frame.iloc[i - 12:i]
        direction = int(directions.iloc[i])
        if prior.timestamp.iloc[-1] - prior.timestamp.iloc[0] != pd.Timedelta(minutes=55):
            continue
        prior_same = active.iloc[i - 12:i] & directions.iloc[i - 12:i].eq(direction)
        if not prior_same.any():
            rows.append((int(i), direction))
    return rows


def generate_events(frame: pd.DataFrame, *, symbol: str, pair: str, canonical_asset: str,
                    eligibility: pd.DataFrame, cohort: pd.DataFrame, refs: dict[str, str],
                    lifecycle_invalid: Sequence[tuple[pd.Timestamp, pd.Timestamp]] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    featured = add_features(frame)
    eligible = eligibility.set_index("utc_day")
    liquid = cohort.set_index("utc_day")
    onset_rows = onset_indices(featured)
    onset_days = sorted({featured.at[index, "utc_day"] for index, _ in onset_rows})
    spot_medians = prior_daily_median(featured.spot_usd_volume_15m, featured.timestamp, onset_days).set_index("utc_day")["median"].to_dict()
    perp_medians = prior_daily_median(featured.perp_usd_volume_15m, featured.timestamp, onset_days).set_index("utc_day")["median"].to_dict()
    rows = []
    excluded = []
    for i, direction in onset_rows:
        row = featured.iloc[i]
        day = row.utc_day
        reasons = []
        if day not in eligible.index or not bool(eligible.at[day, "coverage_eligible"]): reasons.append("insufficient_exact_intersection_history")
        if day not in liquid.index or not bool(liquid.at[day, "top_100_eligible"]): reasons.append("not_stage2c_daily_top100")
        required_start = row.timestamp - pd.Timedelta(days=30)
        if any(required_start < end and row.timestamp + BAR > start for start, end in lifecycle_invalid): reasons.append("known_lifecycle_invalid_interval")
        if not np.isfinite(row.spot_prior_scale) or not np.isfinite(row.perp_prior_scale): reasons.append("scale_unavailable_or_under_2000")
        if reasons:
            excluded.append({"PF_symbol": symbol, "decision_ts": row.feature_available_ts, "reason": ";".join(reasons)})
            continue
        primary = classify_leadership(featured, i, direction, 15)
        robust = classify_leadership(featured, i, direction, 30)
        impulse_start = row.timestamp - pd.Timedelta(minutes=15)
        identity = {"family_id": FAMILY_ID, "canonical_asset_id": canonical_asset, "PF_symbol": symbol,
                    "Kraken_spot_pair": pair, "direction": direction, "impulse_start": impulse_start.isoformat(),
                    "impulse_onset_ts": row.feature_available_ts.isoformat(), "leadership_state": primary,
                    "feature_version": FEATURE_VERSION, "spot_manifest_hash": SPOT_MANIFEST_HASH,
                    "cohort_hash": COHORT_HASH}
        event_id = "c02event_" + stable_hash(identity)[:24]
        branch = {"spot_led": "spot_led_continuation", "simultaneous": "simultaneous_impulse",
                  "perp_led": "perp_led_continuation", "ambiguous": "ambiguous_diagnostic"}[primary]
        s_med = spot_medians.get(day, np.nan)
        p_med = perp_medians.get(day, np.nan)
        rows.append({**identity, "event_id": event_id, "economic_address": "c02addr_" + stable_hash({**identity, "branch": branch})[:24],
                     "attempt_id": "c02attempt_" + stable_hash({"alignment_shift_minutes": 0, "leadership_lookback_minutes": 15,
                         "direction": "positive" if direction > 0 else "negative", "state": branch})[:20],
                     "direction_label": "positive" if direction > 0 else "negative", "impulse_start": impulse_start,
                     "impulse_onset_ts": row.feature_available_ts, "decision_ts": row.feature_available_ts,
                     "leadership_state": primary, "leadership_30m": robust, "leadership_lookback": "15m_primary",
                     "branch": branch, "failure_state": "not_applicable" if primary != "perp_led" else "unconfirmed",
                     "spot_r_15m": row.spot_r_15m, "perp_r_15m": row.perp_r_15m, "spot_z_15m": row.spot_z_15m,
                     "perp_z_15m": row.perp_z_15m, "spot_usd_volume_15m": row.spot_usd_volume_15m,
                     "perp_usd_volume_15m": row.perp_usd_volume_15m,
                     "spot_volume_surprise": row.spot_usd_volume_15m / s_med if s_med > 0 else np.nan,
                     "perp_volume_surprise": row.perp_usd_volume_15m / p_med if p_med > 0 else np.nan,
                     "perp_spot_gap_bps": row.perp_spot_gap_bps, "perp_spot_gap_change_15m_bps": row.perp_spot_gap_change_15m_bps,
                     "mark_trade_gap_bps": row.mark_trade_gap_bps, "prior_day_pf_liquidity_rank": int(liquid.at[day, "rank"]),
                     "exact_intersection_coverage": float(eligible.at[day, "prior_30d_complete_intersection_ratio"]),
                     "feature_available_ts": row.feature_available_ts, "reference_hash": stable_hash(refs),
                     "data_hash": stable_hash(refs), "cohort_hash": COHORT_HASH, "feature_hash": stable_hash({"version": FEATURE_VERSION}),
                     "spot_path_reference": refs["spot"], "PF_trade_path_reference": refs["trade"],
                     "PF_mark_path_reference": refs["mark"], "protected_row_count": 0})
    return pd.DataFrame(rows), pd.DataFrame(excluded)


def generate_failures(events: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    indexed = frame.set_index("timestamp")
    rows = []
    for event in events.loc[events.leadership_state.eq("perp_led")].itertuples(index=False):
        onset_open = pd.Timestamp(event.impulse_onset_ts) - BAR
        impulse = indexed.loc[onset_open - pd.Timedelta(minutes=10):onset_open]
        if len(impulse) != 3:
            continue
        low, high = float(impulse.perp_low.min()), float(impulse.perp_high.max())
        future = indexed.loc[onset_open + BAR:onset_open + pd.Timedelta(hours=6)]
        if event.direction > 0:
            confirmed = future[(future.perp_close < low) & (future.mark_close < low)]
        else:
            confirmed = future[(future.perp_close > high) & (future.mark_close > high)]
        if confirmed.empty:
            continue
        ts = confirmed.index[0] + BAR
        identity = {"source_event_id": event.event_id, "decision_ts": ts.isoformat(), "failure_state": "completed_trade_and_mark"}
        rows.append({**event._asdict(), "source_event_id": event.event_id,
                     "event_id": "c02failure_" + stable_hash(identity)[:24],
                     "economic_address": "c02addr_" + stable_hash(identity)[:24], "decision_ts": ts,
                     "attempt_id": "c02attempt_" + stable_hash({"alignment_shift_minutes": 0,
                         "leadership_lookback_minutes": 15, "direction": getattr(event, "direction_label", "positive" if event.direction > 0 else "negative"),
                         "state": "perp_led_completed_failure"})[:20],
                     "branch": "perp_led_completed_failure", "failure_state": "completed_trade_and_mark",
                     "impulse_window_low": low, "impulse_window_high": high, "feature_available_ts": ts})
    return pd.DataFrame(rows)


def cluster_episodes(events: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy().sort_values(["PF_symbol", "impulse_start", "event_id"], kind="mergesort")
    failure_end = failures.groupby("source_event_id").decision_ts.max().to_dict() if not failures.empty else {}
    out["episode_start"] = pd.to_datetime(out.impulse_start, utc=True)
    out["episode_end"] = [max(pd.Timestamp(row.impulse_onset_ts) + pd.Timedelta(hours=6), failure_end.get(row.event_id, TRAIN_START)) for row in out.itertuples()]
    ids = {}
    for symbol, group in out.groupby("PF_symbol", sort=True):
        current_start = current_end = None
        members: list[str] = []
        def freeze() -> None:
            if members:
                eid = "c02episode_" + stable_hash({"symbol": symbol, "start": current_start.isoformat(), "end": current_end.isoformat()})[:24]
                ids.update({member: eid for member in members})
        for row in group.itertuples():
            if current_end is None or row.episode_start > current_end:
                freeze(); current_start, current_end, members = row.episode_start, row.episode_end, [row.event_id]
            else:
                current_end = max(current_end, row.episode_end); members.append(row.event_id)
        freeze()
    out["canonical_episode_id"] = out.event_id.map(ids)
    out["canonical_episode_size"] = out.groupby("canonical_episode_id").event_id.transform("size")
    return out


def alignment_comparison(exact: pd.DataFrame, shifted: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    exact_core = exact[["event_id", "PF_symbol", "direction", "impulse_onset_ts", "leadership_state"]]
    for shift, frame in shifted.items():
        lookup: dict[tuple[str, int, pd.Timestamp], list[tuple[str, str]]] = {}
        if not frame.empty:
            for candidate in frame[["event_id", "PF_symbol", "direction", "impulse_onset_ts", "leadership_state"]].itertuples(index=False):
                key = (candidate.PF_symbol, int(candidate.direction), pd.Timestamp(candidate.impulse_onset_ts))
                lookup.setdefault(key, []).append((candidate.event_id, candidate.leadership_state))
        for event in exact_core.itertuples(index=False):
            onset = pd.Timestamp(event.impulse_onset_ts)
            candidates = []
            for delta in (pd.Timedelta(0), -BAR, BAR):
                for candidate_id, state in lookup.get((event.PF_symbol, int(event.direction), onset + delta), []):
                    candidates.append((abs(delta), candidate_id, state))
            matched = min(candidates) if candidates else None
            rows.append({"exact_event_id": event.event_id, "spot_shift_minutes": shift,
                         "same_episode_and_direction": matched is not None,
                         "same_leadership_state": matched is not None and matched[2] == event.leadership_state,
                         "exact_leadership_state": event.leadership_state,
                         "shifted_leadership_state": matched[2] if matched is not None else "missing"})
    return pd.DataFrame(rows)


def add_anchor_returns(events: pd.DataFrame, btc_trade: pd.DataFrame, eth_trade: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    def anchor(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        work = frame[["timestamp", "close"]].copy()
        work[name] = complete_return(work, "close")
        work["decision_ts"] = work.timestamp + BAR
        return work[["decision_ts", name]]
    out = events.merge(anchor(btc_trade, "BTC_r_15m"), on="decision_ts", how="left", validate="many_to_one")
    return out.merge(anchor(eth_trade, "ETH_r_15m"), on="decision_ts", how="left", validate="many_to_one")


def update_attempt_counts(register: pd.DataFrame, exact: pd.DataFrame, shifted: dict[int, pd.DataFrame], failures: pd.DataFrame) -> pd.DataFrame:
    out = register.copy()
    sources = {0: exact, **shifted}
    for idx, row in out.iterrows():
        source = sources[int(row.alignment_shift_minutes)]
        if source.empty:
            continue
        direction = 1 if row.direction == "positive" else -1
        state_col = "leadership_state" if int(row.leadership_lookback_minutes) == 15 else "leadership_30m"
        desired = {"spot_led_continuation": "spot_led", "simultaneous_impulse": "simultaneous",
                   "perp_led_continuation": "perp_led", "ambiguous_diagnostic": "ambiguous"}.get(row.state)
        if row.state == "perp_led_completed_failure":
            count = len(failures[failures.direction.eq(direction)]) if int(row.alignment_shift_minutes) == 0 else 0
        else:
            count = int((source.direction.eq(direction) & source[state_col].eq(desired)).sum())
        out.at[idx, "event_count"] = count
    return out


def write_nearest_family_overlap(events: pd.DataFrame, output: Path) -> None:
    sources = [
        ("H43_BTC_led_alt_diffusion", Path("results/rebaseline/phase_kraken_btc_led_delayed_alt_diffusion_long_screen_20260716_v1/materialized/event_ledger.csv"), "symbol", "decision_ts"),
        ("relative_strength_breakout", Path("results/rebaseline/phase_kraken_relative_strength_breakout_vs_btc_screen_20260716_v1/materialized/event_ledger.csv"), "symbol", "decision_ts"),
        ("RFBS_repaired", Path("results/rebaseline/phase_kraken_rfbs_signal_state_repaired_screen_20260715_v1/materialized/event_ledger.csv"), "symbol", "decision_ts"),
        ("Backside_repaired", Path("results/rebaseline/phase_kraken_backside_blowoff_signal_state_repaired_screen_20260715_v1/materialized/event_ledger.csv"), "symbol", "decision_ts"),
        ("C01_onset", Path("docs/agent/task_archive/20260717_donch_bt_stage_2c1_volume_authority_resume_20260717_v1/C01_ONSET_EVENT_TAPE.parquet"), "symbol", "decision_ts"),
    ]
    rows = []
    c02_intervals = events[["event_id", "canonical_episode_id", "PF_symbol", "episode_start", "episode_end"]].drop_duplicates("event_id") if not events.empty else pd.DataFrame()
    for family, path, symbol_col, time_col in sources:
        if not path.exists():
            rows.append({"nearest_family": family, "source_path": str(path), "status": "blocked_source_missing",
                         "safe_source_rows": 0, "unique_causal_addresses": 0, "overlapping_C02_events": 0, "overlapping_C02_episodes": 0})
            continue
        if path.suffix == ".parquet":
            other = pd.read_parquet(path, columns=[symbol_col, time_col])
        else:
            other = pd.read_csv(path, usecols=[symbol_col, time_col])
        other = other.rename(columns={symbol_col: "PF_symbol", time_col: "other_decision_ts"})
        other["other_decision_ts"] = pd.to_datetime(other.other_decision_ts, utc=True, errors="raise")
        if (other.other_decision_ts >= PROTECTED_START).any():
            raise ValueError(f"protected causal identity in nearest-family source: {path}")
        other = other.drop_duplicates(["PF_symbol", "other_decision_ts"])
        overlap_events: set[str] = set(); overlap_episodes: set[str] = set()
        for symbol, group in other.groupby("PF_symbol", sort=False):
            intervals = c02_intervals[c02_intervals.PF_symbol.eq(symbol)] if not c02_intervals.empty else c02_intervals
            times = group.other_decision_ts.astype("int64").to_numpy()
            for event in intervals.itertuples(index=False):
                if ((times >= pd.Timestamp(event.episode_start).value) & (times <= pd.Timestamp(event.episode_end).value)).any():
                    overlap_events.add(event.event_id); overlap_episodes.add(event.canonical_episode_id)
        rows.append({"nearest_family": family, "source_path": str(path), "status": "safe_causal_fields_only",
                     "safe_source_rows": len(other), "unique_causal_addresses": len(other),
                     "overlapping_C02_events": len(overlap_events), "overlapping_C02_episodes": len(overlap_episodes)})
    pd.DataFrame(rows).to_csv(output / "C02_NEAREST_FAMILY_OVERLAP_PREFLIGHT.csv", index=False)


def parse_args() -> argparse.Namespace:
    root = Path("docs/agent/task_archive/20260717_donch_bt_stage_3b_c02_leadership_generator_20260717_v1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root)
    parser.add_argument("--pf-manifest", type=Path, default=Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv"))
    parser.add_argument("--spot-manifest", type=Path, default=Path("docs/agent/task_archive/20260717_donch_bt_stage_3a_c02_spot_reference_20260717_v1/C02_SPOT_DATA_MANIFEST.json"))
    parser.add_argument("--pair-authority", type=Path, default=Path("docs/agent/task_archive/20260717_donch_bt_stage_3a_c02_spot_reference_20260717_v1/C02_SPOT_PAIR_AUTHORITY.csv"))
    parser.add_argument("--cohort", type=Path, default=Path("docs/agent/task_archive/20260717_donch_bt_stage_2c1_volume_authority_resume_20260717_v1/C01_DAILY_LIQUIDITY_MEMBERSHIP.parquet"))
    parser.add_argument("--lifecycle-source", type=Path, default=Path("docs/agent/task_archive/20260717_donch_bt_stage_2a1_c01_reference_panel_20260717_v1/sources/terminal_lifecycle/kraken_derivatives_delistings.body"))
    return parser.parse_args()


def main() -> int:
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    write_prefrozen_contracts(args.output)
    started = time.monotonic()
    spot_manifest = json.loads(args.spot_manifest.read_text())
    if spot_manifest.get("manifest_content_hash") != SPOT_MANIFEST_HASH or spot_manifest.get("protected_rows_opened") != 0:
        raise ValueError("Stage 3A spot authority mismatch")
    authority_rows = c01.load_safe_manifest(args.pf_manifest)
    pair = pd.read_csv(args.pair_authority)
    pair = pair[pair.inclusion_or_exclusion_reason.eq("included_official_USD_archive_rows") & pair.mechanism_proof_cohort.map(c01._as_bool)].copy()
    cohort = pd.read_parquet(args.cohort)
    if cohort.cohort_hash.nunique() != 1 or cohort.cohort_hash.iloc[0] != COHORT_HASH or cohort.rank_uses_current_day.any():
        raise ValueError("Stage 2C cohort authority mismatch")
    lifecycle = c01.load_known_lifecycle_invalidations(args.lifecycle_source)
    normalized = {Path(row["path"]).stem: row for row in spot_manifest["normalized_files"]}
    pair = pair[pair.Kraken_spot_pair.isin(normalized)].sort_values("PF_symbol")
    register = make_attempt_register(); register.to_csv(args.output / "C02_FAMILY_AND_ATTEMPT_REGISTER.csv", index=False)
    all_events=[]; all_failures=[]; all_eligibility=[]; all_exclusions=[]; shifted_events={-5:[],5:[]}
    for number, mapping in enumerate(pair.itertuples(index=False), 1):
        spot_row = normalized[mapping.Kraken_spot_pair]
        spot = read_spot(Path(spot_row["path"]), mapping.Kraken_spot_pair, spot_row["sha256"])
        trade, trade_ref = read_pf_bars(authority_rows, mapping.PF_symbol, "historical_trade_candles_5m")
        mark, mark_ref = read_pf_bars(authority_rows, mapping.PF_symbol, "historical_mark_candles_5m")
        exact = align_exact(spot, trade, mark)
        symbol_cohort = cohort[cohort.symbol.eq(mapping.PF_symbol)][["utc_day", "top_100_eligible", "rank"]]
        audit = complete_daily_eligibility(
            daily_intersection_audit(exact, mapping.PF_symbol, mapping.Kraken_spot_pair),
            symbol_cohort, lifecycle.get(mapping.PF_symbol, []),
        )
        refs={"spot": f"stage3a:{mapping.Kraken_spot_pair}:{spot_row['sha256']}", "trade": trade_ref, "mark": mark_ref}
        events, excluded = generate_events(exact, symbol=mapping.PF_symbol, pair=mapping.Kraken_spot_pair,
            canonical_asset=mapping.canonical_asset_id, eligibility=audit, cohort=symbol_cohort, refs=refs,
            lifecycle_invalid=lifecycle.get(mapping.PF_symbol, []))
        failures = generate_failures(events, exact)
        all_events.append(events); all_failures.append(failures); all_eligibility.append(audit); all_exclusions.append(excluded)
        for shift in (-5, 5):
            shifted = align_exact(spot, trade, mark, spot_shift_minutes=shift)
            shift_audit = complete_daily_eligibility(
                daily_intersection_audit(shifted, mapping.PF_symbol, mapping.Kraken_spot_pair),
                symbol_cohort, lifecycle.get(mapping.PF_symbol, []),
            )
            se, _ = generate_events(shifted, symbol=mapping.PF_symbol, pair=mapping.Kraken_spot_pair,
                canonical_asset=mapping.canonical_asset_id, eligibility=shift_audit, cohort=symbol_cohort, refs=refs,
                lifecycle_invalid=lifecycle.get(mapping.PF_symbol, []))
            shifted_events[shift].append(se)
        (args.output / "watch_status.json").write_text(json.dumps({
            "stage": "outcome_free_generation", "symbols_completed": number, "symbols_total": len(pair),
            "impulse_events": sum(len(part) for part in all_events), "failure_events": sum(len(part) for part in all_failures),
            "elapsed_seconds": time.monotonic() - started,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[{number}/{len(pair)}] {mapping.PF_symbol}: exact_events={len(events)}", flush=True)
    concat=lambda frames: pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()
    events=concat(all_events); failures=concat(all_failures); eligibility=concat(all_eligibility); exclusions=concat(all_exclusions)
    if not events.empty:
        confirmed_sources = set(failures.source_event_id) if not failures.empty else set()
        events["failure_state"] = np.where(
            events.leadership_state.ne("perp_led"), "not_applicable",
            np.where(events.event_id.isin(confirmed_sources), "confirmed", "unconfirmed"),
        )
        events=cluster_episodes(events, failures)
        if not failures.empty:
            episode_fields = events[["event_id", "canonical_episode_id", "episode_start", "episode_end"]].rename(columns={"event_id": "source_event_id"})
            failures = failures.merge(episode_fields, on="source_event_id", how="left", validate="many_to_one")
            if failures.canonical_episode_id.isna().any():
                raise ValueError("failure-to-episode reconciliation failed")
        btc_trade, btc_ref = read_pf_bars(authority_rows, "PF_XBTUSD", "historical_trade_candles_5m")
        eth_trade, eth_ref = read_pf_bars(authority_rows, "PF_ETHUSD", "historical_trade_candles_5m")
        events = add_anchor_returns(events, btc_trade, eth_trade)
        events["asset_group"] = np.where(events.canonical_asset_id.isin(["BTC", "XBT", "ETH"]), "BTC_ETH", "alt")
    shifts={k:concat(v) for k,v in shifted_events.items()}
    alignment=alignment_comparison(events, shifts) if not events.empty else pd.DataFrame()
    register = update_attempt_counts(register, events, shifts, failures)
    register.to_csv(args.output / "C02_FAMILY_AND_ATTEMPT_REGISTER.csv", index=False)
    for output in (events, failures): assert_no_outcome_fields(output.columns)
    eligibility.to_parquet(args.output / "C02_DAILY_ELIGIBILITY_AUDIT.parquet", index=False)
    events.to_parquet(args.output / "C02_IMPULSE_EVENT_TAPE.parquet", index=False)
    failures.to_parquet(args.output / "C02_FAILURE_EVENT_TAPE.parquet", index=False)
    alignment.to_csv(args.output / "C02_ALIGNMENT_SENSITIVITY.csv", index=False)
    if not events.empty:
        counts=events.groupby([events.impulse_onset_ts.dt.year.rename("year"), "asset_group", "PF_symbol", "direction_label", "leadership_state", "leadership_30m", "branch", "failure_state"], dropna=False).size().rename("event_count").reset_index()
    else: counts=pd.DataFrame(columns=["year","asset_group","PF_symbol","direction_label","leadership_state","leadership_30m","branch","failure_state","event_count"])
    counts.to_csv(args.output / "C02_EVENT_COUNT_MATRIX.csv", index=False)
    write_nearest_family_overlap(events, args.output)
    if not exclusions.empty: exclusions.to_csv(args.output / "C02_ELIGIBILITY_EXCLUSIONS.csv", index=False)
    if not alignment.empty:
        by_event = alignment.groupby("exact_event_id", sort=False).agg(
            present_under_both=("same_episode_and_direction", "all"),
            same_state_under_both=("same_leadership_state", "all"),
        )
        exact_present = float(by_event.present_under_both.mean())
        same_state = float(by_event.loc[by_event.present_under_both, "same_state_under_both"].mean()) if by_event.present_under_both.any() else 0.0
    else:
        exact_present = same_state = 0.0
    status = "alignment_stable_for_contract_review" if exact_present >= .8 and same_state >= .7 else "alignment_fragile_requires_review"
    (args.output / "C02_LEADERSHIP_AGREEMENT_REPORT.md").write_text(f"# C02 Leadership Agreement\n\n- alignment_status: `{status}`\n- same episode and direction: `{exact_present:.6f}`\n- same leadership state conditional on presence: `{same_state:.6f}`\n", encoding="utf-8")
    episode_count=int(events.canonical_episode_id.nunique()) if not events.empty else 0
    (args.output / "C02_EPISODE_IDENTITY_REPORT.md").write_text(f"# C02 Episode Identity\n\nOverlapping same-symbol causal intervals only. Events: `{len(events)}`. Episodes: `{episode_count}`. Outcomes and exits are not inputs.\n", encoding="utf-8")
    summary={"family_id":FAMILY_ID,"symbols_processed":len(pair),"events":len(events),"failures":len(failures),"episodes":episode_count,
             "alignment_status":status,"same_episode_direction_rate":exact_present,"same_leadership_rate":same_state,
             "economic_outputs_computed":0,"protected_outcomes_opened":0,"protected_rows_opened":0,"spot_manifest_hash":SPOT_MANIFEST_HASH,
             "cohort_hash":COHORT_HASH,"outcome_fields":[]}
    (args.output / "C02_GENERATION_SUMMARY.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (args.output / "watch_status.json").write_text(json.dumps({
        "stage": "complete", "symbols_completed": len(pair), "symbols_total": len(pair),
        "impulse_events": len(events), "failure_events": len(failures), "elapsed_seconds": time.monotonic() - started,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
