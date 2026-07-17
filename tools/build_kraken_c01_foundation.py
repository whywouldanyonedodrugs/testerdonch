#!/usr/bin/env python3
"""Build the outcome-free C01 feature, identity, and episode foundation.

This module is intentionally incapable of calculating forward returns or trade
outcomes.  It opens only manifest-authorized Kraken 5m trade/mark shards wholly
before the protected boundary and emits causal diagnostic candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from tools.build_kraken_c01_reference_panel_authority import _TableParser, parse_terminal_lifecycle_html
except ModuleNotFoundError:  # Direct `python tools/...py` execution.
    from build_kraken_c01_reference_panel_authority import _TableParser, parse_terminal_lifecycle_html


TASK_ID = "donch_bt_stage_2b_c01_foundation_20260717_v1"
FAMILY_ID = "C01_debetaed_residual_shock_path_bifurcation"
REFERENCE_PANEL_ID = "kraken_c01_reference_panel_v1"
REFERENCE_PANEL_HASH = "2c0cae827c6f83361ea347796d0323b20d83c3acd222c506aac40c0e97b73763"
CANDIDATE_COHORT_VERSION = "current_roster_bar_existence_cohort"
FEATURE_VERSION = "c01_residual_path_features_v1_20260717"
PRIMARY_MODEL = "btc_eth_ols_daily_v1"
ROBUSTNESS_MODEL = "btc_only_ols_daily_v1"
RESIDUAL_MODELS = (PRIMARY_MODEL, ROBUSTNESS_MODEL)
SIGNS = ("positive", "negative")
PATH_STATES = ("smooth", "jump_dominated", "intermediate")
TRAIN_START = pd.Timestamp("2023-01-01T00:00:00Z")
PROTECTED_START = pd.Timestamp("2026-01-01T00:00:00Z")
BAR_DELTA = pd.Timedelta(minutes=5)
SHOCK_BARS = 72
DAY_BARS = 288
ESTIMATION_DAYS = 30
EXPECTED_ESTIMATION_ROWS = ESTIMATION_DAYS * DAY_BARS
MIN_ESTIMATION_ROWS = math.ceil(0.70 * EXPECTED_ESTIMATION_ROWS)
MIN_SCALE_BLOCKS = 80
REFERENCE_SYMBOLS = {"PF_XBTUSD", "PF_ETHUSD"}
FORBIDDEN_OUTPUT_TOKENS = (
    "forward_return", "future_return", "exit", "pnl", "profit", "mae", "mfe",
    "expectancy", "sharpe", "promotion", "target", "label_return",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def deterministic_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def utc_timestamp(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    return result.tz_convert("UTC")


def iso_utc(value: Any) -> str:
    return utc_timestamp(value).isoformat().replace("+00:00", "Z")


def classify_path_state(largest_bar_share: float, path_efficiency: float) -> str:
    if not np.isfinite(largest_bar_share) or not np.isfinite(path_efficiency):
        raise ValueError("path state requires finite causal measures")
    if largest_bar_share <= 0.25 and path_efficiency >= 0.50:
        return "smooth"
    if largest_bar_share >= 0.50:
        return "jump_dominated"
    return "intermediate"


def classify_shock(z_value: float) -> str | None:
    if not np.isfinite(z_value):
        return None
    if z_value >= 3.0:
        return "positive"
    if z_value <= -3.0:
        return "negative"
    return None


def candidate_identity_payload(row: Mapping[str, Any]) -> dict[str, str]:
    """Return only causal identity fields; reporting/runtime fields are excluded."""
    fields = (
        "family_id", "definition_id", "attempt_id", "symbol", "venue", "decision_ts",
        "shock_window_start", "shock_window_end", "residual_model_version", "feature_version",
        "reference_panel_id", "reference_panel_hash", "candidate_cohort_version",
        "data_authority_hash",
    )
    return {field: str(row[field]) for field in fields}


def assign_candidate_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    digest = deterministic_hash(candidate_identity_payload(row))
    return f"c01cand_{digest[:24]}", f"c01econ_{digest}"


def cluster_intervals(rows: pd.DataFrame) -> pd.DataFrame:
    """Cluster overlapping same-symbol intervals without an added gap."""
    required = {"candidate_id", "symbol", "canonical_episode_input_start", "canonical_episode_input_end"}
    if not required.issubset(rows.columns):
        raise ValueError(f"episode input missing columns: {sorted(required - set(rows.columns))}")
    if rows.empty:
        return rows.assign(
            canonical_episode_id=pd.Series(dtype="object"),
            episode_cluster_start=pd.Series(dtype="datetime64[ns, UTC]"),
            episode_cluster_end=pd.Series(dtype="datetime64[ns, UTC]"),
            episode_member_count=pd.Series(dtype="int64"),
        )
    work = rows.copy()
    for field in ("canonical_episode_input_start", "canonical_episode_input_end"):
        work[field] = pd.to_datetime(work[field], utc=True, errors="raise")
    work = work.sort_values(
        ["symbol", "canonical_episode_input_start", "canonical_episode_input_end", "candidate_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    assignments: list[tuple[str, pd.Timestamp, pd.Timestamp, int]] = [None] * len(work)  # type: ignore[list-item]
    for symbol, indices in work.groupby("symbol", sort=True).groups.items():
        clusters: list[list[int]] = []
        current: list[int] = []
        current_end: pd.Timestamp | None = None
        for idx in list(indices):
            start = work.at[idx, "canonical_episode_input_start"]
            end = work.at[idx, "canonical_episode_input_end"]
            if end < start:
                raise ValueError("episode interval ends before it starts")
            if current and start > current_end:
                clusters.append(current)
                current = []
                current_end = None
            current.append(idx)
            current_end = end if current_end is None else max(current_end, end)
        if current:
            clusters.append(current)
        for members in clusters:
            cluster_start = min(work.at[idx, "canonical_episode_input_start"] for idx in members)
            cluster_end = max(work.at[idx, "canonical_episode_input_end"] for idx in members)
            episode_hash = deterministic_hash({
                "venue": "Kraken", "symbol": symbol,
                "cluster_start": iso_utc(cluster_start), "cluster_end": iso_utc(cluster_end),
                "member_candidate_ids": sorted(work.at[idx, "candidate_id"] for idx in members),
            })
            assignment = (f"episode_{episode_hash[:24]}", cluster_start, cluster_end, len(members))
            for idx in members:
                assignments[idx] = assignment
    work[["canonical_episode_id", "episode_cluster_start", "episode_cluster_end", "episode_member_count"]] = pd.DataFrame(
        assignments, index=work.index,
    )
    return work


@dataclass(frozen=True)
class AuthorityRow:
    dataset: str
    symbol: str
    chunk_start: pd.Timestamp
    chunk_end: pd.Timestamp
    parquet_path: Path
    parquet_sha256: str
    rows: int

    @property
    def reference_id(self) -> str:
        return f"{self.dataset}:{self.symbol}:{self.chunk_start.isoformat()}:{self.parquet_sha256[:16]}"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1"}


def load_safe_manifest(path: Path) -> list[AuthorityRow]:
    required = {
        "dataset", "symbol", "chunk_start", "chunk_end", "resolution", "rankable_pre_holdout",
        "contains_protected_period", "parquet_path", "parquet_sha256", "rows", "status",
    }
    frame = pd.read_csv(path, usecols=sorted(required), low_memory=False)
    if not required.issubset(frame.columns):
        raise ValueError(f"manifest missing authority columns: {sorted(required - set(frame.columns))}")
    rows: list[AuthorityRow] = []
    allowed_datasets = {"historical_trade_candles_5m", "historical_mark_candles_5m"}
    selected_scope = frame["dataset"].isin(allowed_datasets) & frame["symbol"].astype(str).str.startswith("PF_")
    for raw in frame.loc[selected_scope].to_dict("records"):
        if int(raw["rows"]) == 0:
            # Empty acquisition envelopes contain no candidate observations and
            # are excluded from the payload reader using manifest metadata.
            continue
        if not _as_bool(raw["rankable_pre_holdout"]) or _as_bool(raw["contains_protected_period"]):
            continue
        start, end = utc_timestamp(raw["chunk_start"]), utc_timestamp(raw["chunk_end"])
        symbol = str(raw["symbol"])
        parquet_path = Path(str(raw["parquet_path"]))
        if (
            str(raw["resolution"]) != "5m" or str(raw["status"]) != "downloaded"
            or not symbol.startswith("PF_") or start < TRAIN_START or end > PROTECTED_START
            or not parquet_path.is_absolute() or not parquet_path.exists()
            or "/downloaded_official_kraken/parquet/" not in str(parquet_path)
        ):
            raise ValueError(f"unsafe or ambiguous manifest row rejected before reader: {raw}")
        rows.append(AuthorityRow(
            dataset=str(raw["dataset"]), symbol=symbol, chunk_start=start, chunk_end=end,
            parquet_path=parquet_path, parquet_sha256=str(raw["parquet_sha256"]), rows=int(raw["rows"]),
        ))
    if not rows:
        raise ValueError("no safe Kraken 5m authority rows")
    return sorted(rows, key=lambda row: (row.dataset, row.symbol, row.chunk_start, str(row.parquet_path)))


def authority_hash(
    rows: Sequence[AuthorityRow], reference_manifest_hash: str, lifecycle_source_hash: str,
) -> str:
    return deterministic_hash({
        "safe_manifest_rows": [
            {
                "dataset": row.dataset, "symbol": row.symbol,
                "chunk_start": iso_utc(row.chunk_start), "chunk_end": iso_utc(row.chunk_end),
                "parquet_path": str(row.parquet_path), "parquet_sha256": row.parquet_sha256,
                "rows": row.rows,
            }
            for row in rows
        ],
        "reference_panel_hash": REFERENCE_PANEL_HASH,
        "reference_final_day_manifest_hash": reference_manifest_hash,
        "known_lifecycle_source_hash": lifecycle_source_hash,
        "train_bounds": [iso_utc(TRAIN_START), iso_utc(PROTECTED_START)],
    })


def current_candidate_cohort(
    instrument_source: Path, authority_rows: Sequence[AuthorityRow],
) -> tuple[list[str], dict[str, pd.Timestamp], pd.DataFrame]:
    payload = json.loads(instrument_source.read_text(encoding="utf-8"))
    instruments = payload.get("instruments") if isinstance(payload, Mapping) else None
    if not isinstance(instruments, list):
        raise ValueError("official current instrument source is ambiguous")
    cohort: list[str] = []
    opening_dates: dict[str, pd.Timestamp] = {}
    audit_rows: list[dict[str, Any]] = []
    for raw in instruments:
        if not isinstance(raw, Mapping):
            continue
        symbol = str(raw.get("symbol", ""))
        if (
            raw.get("type") != "flexible_futures" or raw.get("tradeable") is not True
            or raw.get("tradfi") is True or not symbol.startswith("PF_") or symbol in REFERENCE_SYMBOLS
        ):
            continue
        if not raw.get("openingDate"):
            raise ValueError(f"candidate has no official opening date: {symbol}")
        if symbol in opening_dates:
            raise ValueError(f"duplicate current instrument: {symbol}")
        opening_date = utc_timestamp(raw["openingDate"])
        has_bars: dict[str, bool] = {}
        for dataset in ("historical_trade_candles_5m", "historical_mark_candles_5m"):
            eligible_shards = [
                row for row in authority_rows
                if row.symbol == symbol and row.dataset == dataset and row.chunk_end > opening_date
            ]
            has_bars[dataset] = any(
                "time" in pq.ParquetFile(row.parquet_path).schema_arrow.names for row in reversed(eligible_shards)
            )
        included = all(has_bars.values()) and opening_date < PROTECTED_START
        if included:
            reason = "included_current_roster_with_preprotected_trade_and_mark_bars"
        elif opening_date >= PROTECTED_START:
            reason = "excluded_opening_on_or_after_protected_cutoff"
        else:
            reason = "excluded_no_preprotected_trade_and_mark_bar_existence"
        audit_rows.append({
            "symbol": symbol, "official_opening_date": iso_utc(opening_date),
            "current_roster_member": True, "trade_bar_exists_before_protected_cutoff": has_bars["historical_trade_candles_5m"],
            "mark_bar_exists_before_protected_cutoff": has_bars["historical_mark_candles_5m"],
            "included": included, "candidate_cohort_version": CANDIDATE_COHORT_VERSION,
            "reason": reason,
            "survivorship_free_claim": False, "continuous_tradeability_claim": False,
        })
        if not included:
            continue
        opening_dates[symbol] = opening_date
        cohort.append(symbol)
    if not cohort:
        raise ValueError("current roster/bar-existence cohort is empty")
    return sorted(cohort), opening_dates, pd.DataFrame(audit_rows).sort_values("symbol").reset_index(drop=True)


def load_known_lifecycle_invalidations(path: Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Parse conservative date-precision terminal-to-resumption intervals."""
    payload = path.read_bytes()
    terminal_rows, _ = parse_terminal_lifecycle_html(payload)
    parser = _TableParser()
    parser.feed(payload.decode("utf-8", errors="strict"))
    visible = " ".join(parser.visible)
    resumed: dict[str, list[pd.Timestamp]] = {}
    for symbol, month_day, year in re.findall(
        r"\b(PF_[A-Z0-9]+USD)\s+resumed trading on\s+([A-Za-z]+\s+\d{1,2}),\s+(\d{4})",
        visible,
        flags=re.I,
    ):
        resumed.setdefault(symbol.upper(), []).append(
            utc_timestamp(pd.to_datetime(f"{month_day} {year}", format="mixed", utc=True).floor("D"))
        )
    result: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for row in terminal_rows:
        symbol = row["symbol"]
        start = utc_timestamp(pd.to_datetime(row["settlement_date"], format="%d-%b-%Y", utc=True).floor("D"))
        later_resumptions = sorted(value for value in resumed.get(symbol, []) if value > start)
        # Date-only resumption authority becomes usable after that UTC date ends.
        end = later_resumptions[0] + pd.Timedelta(days=1) if later_resumptions else PROTECTED_START
        if start < PROTECTED_START:
            result.setdefault(symbol, []).append((max(start, TRAIN_START), min(end, PROTECTED_START)))
    return result


def _validate_bar_frame(frame: pd.DataFrame, row: AuthorityRow) -> pd.DataFrame:
    required = {
        "time", "close", "venue_symbol", "resolution", "rankable_pre_holdout",
        "contains_protected_period",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"bar shard schema missing: {sorted(required - set(frame.columns))}")
    if frame.empty:
        return pd.DataFrame(columns=["source_open_ts", "close"])
    if not frame["venue_symbol"].astype(str).eq(row.symbol).all() or not frame["resolution"].astype(str).eq("5m").all():
        raise ValueError("bar shard venue/symbol/resolution mismatch")
    if not frame["rankable_pre_holdout"].map(_as_bool).all() or frame["contains_protected_period"].map(_as_bool).any():
        raise ValueError("unrankable or protected bar row reached normalization")
    times = pd.to_datetime(pd.to_numeric(frame["time"], errors="raise"), unit="ms", utc=True)
    if (times < TRAIN_START).any() or (times >= PROTECTED_START).any():
        raise ValueError("out-of-bound bar row reached normalization")
    close = pd.to_numeric(frame["close"], errors="coerce")
    if close.isna().any() or (close <= 0).any():
        raise ValueError("non-positive or missing close")
    return pd.DataFrame({"source_open_ts": times, "close": close.astype(float)})


def read_authorized_bars(
    rows: Sequence[AuthorityRow], symbol: str, dataset: str, *, not_before: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, str]:
    selected = [
        row for row in rows
        if row.symbol == symbol and row.dataset == dataset
        and (not_before is None or row.chunk_end > not_before)
    ]
    if not selected:
        raise ValueError(f"no safe authority rows for {dataset}:{symbol}")
    parts: list[pd.DataFrame] = []
    for row in selected:
        schema_names = set(pq.ParquetFile(row.parquet_path).schema_arrow.names)
        if "time" not in schema_names:
            if "candles" in schema_names and row.rows <= 1:
                # Official empty response envelope, identified without reading rows.
                continue
            raise ValueError(f"ambiguous non-candle shard schema: {row.parquet_path}")
        # The manifest boundary is checked before this payload reader is called.
        columns = ["time", "close", "venue_symbol", "resolution", "rankable_pre_holdout", "contains_protected_period"]
        parts.append(_validate_bar_frame(pd.read_parquet(row.parquet_path, columns=columns), row))
    if not parts:
        raise ValueError(f"no non-empty safe candle shards for {dataset}:{symbol}")
    frame = pd.concat(parts, ignore_index=True).sort_values("source_open_ts", kind="mergesort")
    duplicated = frame.duplicated("source_open_ts", keep=False)
    if duplicated.any():
        conflicts = frame.loc[duplicated].groupby("source_open_ts")["close"].nunique().gt(1)
        if conflicts.any():
            raise ValueError(f"conflicting duplicate bar boundary for {dataset}:{symbol}")
        frame = frame.drop_duplicates("source_open_ts", keep="first")
    refs = deterministic_hash([row.reference_id for row in selected])
    return frame.reset_index(drop=True), f"c01_authority:{dataset}:{symbol}:{refs}"


def append_reference_final_day(frame: pd.DataFrame, path: Path, symbol: str) -> pd.DataFrame:
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    required = {"time", "close", "venue_symbol", "rankable_pre_holdout", "contains_protected_period"}
    if not required.issubset(schema):
        raise ValueError("reference final-day shard schema mismatch")
    raw = pd.read_parquet(path, columns=sorted(required | {"resolution"}))
    authority = AuthorityRow(
        dataset="reference_final_day", symbol=symbol, chunk_start=pd.Timestamp("2025-12-31T00:00:00Z"),
        chunk_end=PROTECTED_START, parquet_path=path, parquet_sha256=sha256_file(path), rows=len(raw),
    )
    extra = _validate_bar_frame(raw, authority)
    combined = pd.concat([frame, extra], ignore_index=True).sort_values("source_open_ts", kind="mergesort")
    duplicated = combined.duplicated("source_open_ts", keep=False)
    if duplicated.any() and combined.loc[duplicated].groupby("source_open_ts")["close"].nunique().gt(1).any():
        raise ValueError("reference old/new boundary conflict")
    return combined.drop_duplicates("source_open_ts", keep="first").reset_index(drop=True)


def _returns(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy()
    out[name] = np.log(out["close"]).diff()
    # A return spanning a missing five-minute bar is not an aligned observation.
    out.loc[out["source_open_ts"].diff().ne(BAR_DELTA), name] = np.nan
    out = out.rename(columns={"close": f"{name}_close"})
    return out


def _consecutive_window(series: pd.Series, size: int) -> pd.Series:
    index = pd.DatetimeIndex(series.index)
    values = series.notna().astype(np.int16)
    count = values.rolling(size, min_periods=size).sum().eq(size)
    span = pd.Series(index, index=index).diff(size - 1).eq(BAR_DELTA * (size - 1))
    return count & span


def _daily_coefficients(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    factors = ["btc_ret", "eth_ret"] if model == PRIMARY_MODEL else ["btc_ret"]
    validity = frame[["candidate_ret", *factors, "candidate_mark", "btc_mark"]].notna().all(axis=1)
    if model == PRIMARY_MODEL:
        validity &= frame["eth_mark"].notna()
    valid = frame.loc[validity, ["source_open_ts", "candidate_ret", *factors]].copy()
    valid["fit_day"] = valid["source_open_ts"].dt.floor("D")
    dimension = 1 + len(factors)
    daily_rows: list[dict[str, Any]] = []
    for day, group in valid.groupby("fit_day", sort=True):
        x = np.column_stack([np.ones(len(group)), *[group[name].to_numpy(float) for name in factors]])
        y = group["candidate_ret"].to_numpy(float)
        daily_rows.append({"fit_day": day, "count": len(group), "xtx": x.T @ x, "xty": x.T @ y})
    days = pd.date_range(TRAIN_START.floor("D"), PROTECTED_START.floor("D"), freq="D", inclusive="left")
    by_day = {row["fit_day"]: row for row in daily_rows}
    coefficients: list[dict[str, Any]] = []
    for day in days:
        members = [by_day[d] for d in pd.date_range(day - pd.Timedelta(days=ESTIMATION_DAYS), day - pd.Timedelta(days=1), freq="D") if d in by_day]
        count = sum(int(item["count"]) for item in members)
        beta = np.full(dimension, np.nan)
        if count >= MIN_ESTIMATION_ROWS:
            xtx = sum((item["xtx"] for item in members), start=np.zeros((dimension, dimension)))
            xty = sum((item["xty"] for item in members), start=np.zeros(dimension))
            try:
                beta = np.linalg.solve(xtx, xty)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
        row: dict[str, Any] = {"decision_day": day, "fit_observations": count, "alpha": beta[0], "beta_btc": beta[1]}
        row["beta_eth"] = beta[2] if model == PRIMARY_MODEL else np.nan
        coefficients.append(row)
    return pd.DataFrame(coefficients)


def _causal_scale(residual: pd.Series, shock_start: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    residual = residual.copy()
    block = residual.index.floor("6h")
    block_frame = pd.DataFrame({"residual": residual.to_numpy(), "block_start": block}, index=residual.index)
    summaries: list[tuple[pd.Timestamp, float]] = []
    for start, group in block_frame.groupby("block_start", sort=True):
        if len(group) != SHOCK_BARS or group["residual"].isna().any():
            continue
        if group.index[-1] - group.index[0] != BAR_DELTA * (SHOCK_BARS - 1):
            continue
        summaries.append((start + pd.Timedelta(hours=6), float(group["residual"].sum())))
    if not summaries:
        return np.full(len(residual), np.nan), np.zeros(len(residual), dtype=int)
    ends = np.array([item[0].value for item in summaries], dtype=np.int64)
    values = np.array([item[1] for item in summaries], dtype=float)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    cumulative_sq = np.concatenate([[0.0], np.cumsum(values * values)])
    starts_ns = pd.to_datetime(shock_start, utc=True).astype("int64").to_numpy()
    lo = np.searchsorted(ends, starts_ns - pd.Timedelta(days=30).value, side="left")
    hi = np.searchsorted(ends, starts_ns, side="right")
    counts = hi - lo
    sums = cumulative[hi] - cumulative[lo]
    sums_sq = cumulative_sq[hi] - cumulative_sq[lo]
    variance = np.full(len(residual), np.nan)
    eligible = counts >= MIN_SCALE_BLOCKS
    variance[eligible] = (sums_sq[eligible] - sums[eligible] * sums[eligible] / counts[eligible]) / (counts[eligible] - 1)
    variance[variance < 0] = 0
    return np.sqrt(variance), counts


def compute_symbol_features(
    symbol: str,
    candidate_trade: pd.DataFrame,
    candidate_mark: pd.DataFrame,
    btc_trade: pd.DataFrame,
    btc_mark: pd.DataFrame,
    eth_trade: pd.DataFrame,
    eth_mark: pd.DataFrame,
    model: str,
    opening_date: pd.Timestamp,
    lifecycle_invalidations: Sequence[tuple[pd.Timestamp, pd.Timestamp]] = (),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Compute causal observable states for one symbol/model; no forward data."""
    candidate = _returns(candidate_trade, "candidate_ret")
    btc = _returns(btc_trade, "btc_ret")
    eth = _returns(eth_trade, "eth_ret")
    frame = candidate[["source_open_ts", "candidate_ret"]].merge(
        btc[["source_open_ts", "btc_ret"]], on="source_open_ts", how="outer", validate="one_to_one",
    ).merge(eth[["source_open_ts", "eth_ret"]], on="source_open_ts", how="outer", validate="one_to_one")
    for name, mark in (("candidate_mark", candidate_mark), ("btc_mark", btc_mark), ("eth_mark", eth_mark)):
        values = mark[["source_open_ts", "close"]].rename(columns={"close": name})
        frame = frame.merge(values, on="source_open_ts", how="outer", validate="one_to_one")
    frame = frame.sort_values("source_open_ts", kind="mergesort")
    frame = frame[(frame["source_open_ts"] >= TRAIN_START) & (frame["source_open_ts"] < PROTECTED_START)].copy()
    frame["decision_ts"] = frame["source_open_ts"] + BAR_DELTA
    first_complete_candidate_open = opening_date.ceil("5min")
    before_listing = frame["source_open_ts"] < first_complete_candidate_open
    frame.loc[before_listing, ["candidate_ret", "candidate_mark"]] = np.nan
    lifecycle_invalid = pd.Series(False, index=frame.index)
    for invalid_start, invalid_end in lifecycle_invalidations:
        lifecycle_invalid |= frame["decision_ts"].ge(invalid_start) & frame["decision_ts"].lt(invalid_end)
    frame.loc[lifecycle_invalid, ["candidate_ret", "candidate_mark"]] = np.nan
    frame["decision_day"] = frame["decision_ts"].dt.floor("D")
    coefficients = _daily_coefficients(frame, model)
    frame = frame.merge(coefficients, on="decision_day", how="left", validate="many_to_one")
    if model == PRIMARY_MODEL:
        frame["residual"] = frame["candidate_ret"] - frame["alpha"] - frame["beta_btc"] * frame["btc_ret"] - frame["beta_eth"] * frame["eth_ret"]
        required_current = ["candidate_ret", "btc_ret", "eth_ret", "candidate_mark", "btc_mark", "eth_mark", "residual"]
    else:
        frame["residual"] = frame["candidate_ret"] - frame["alpha"] - frame["beta_btc"] * frame["btc_ret"]
        required_current = ["candidate_ret", "btc_ret", "candidate_mark", "btc_mark", "residual"]
    frame = frame.set_index("source_open_ts", drop=False)
    complete_current = frame[required_current].notna().all(axis=1)
    contiguous = _consecutive_window(frame["residual"].where(complete_current), SHOCK_BARS)
    abs_residual = frame["residual"].abs()
    frame["residual_shock_6h"] = frame["residual"].where(complete_current).rolling(SHOCK_BARS, min_periods=SHOCK_BARS).sum().where(contiguous)
    denominator = abs_residual.where(complete_current).rolling(SHOCK_BARS, min_periods=SHOCK_BARS).sum().where(contiguous)
    frame["largest_bar_share"] = abs_residual.where(complete_current).rolling(SHOCK_BARS, min_periods=SHOCK_BARS).max().where(contiguous) / denominator
    frame["path_efficiency"] = frame["residual_shock_6h"].abs() / denominator
    frame["shock_window_start"] = frame["source_open_ts"] - BAR_DELTA * (SHOCK_BARS - 1)
    scale, scale_counts = _causal_scale(frame["residual"].where(complete_current), frame["shock_window_start"])
    frame["residual_scale_6h"] = scale
    frame["scale_block_count"] = scale_counts
    frame["residual_shock_z_6h"] = frame["residual_shock_6h"] / frame["residual_scale_6h"].replace(0, np.nan)
    for name in ("btc_ret", "eth_ret"):
        complete = _consecutive_window(frame[name], SHOCK_BARS)
        frame[name.replace("_ret", "_return_6h")] = frame[name].rolling(SHOCK_BARS, min_periods=SHOCK_BARS).sum().where(complete)
    for name in ("btc_ret", "candidate_ret"):
        complete = _consecutive_window(frame[name], DAY_BARS)
        frame[name.replace("_ret", "_rv_24h")] = np.sqrt(frame[name].pow(2).rolling(DAY_BARS, min_periods=DAY_BARS).sum()).where(complete)
    frame["candidate_lagged_trade_bar_availability"] = frame["candidate_ret"].notna().rolling(EXPECTED_ESTIMATION_ROWS, min_periods=1).mean()
    frame["candidate_lagged_mark_bar_availability"] = frame["candidate_mark"].notna().rolling(EXPECTED_ESTIMATION_ROWS, min_periods=1).mean()
    shock_sign = frame["residual_shock_z_6h"].map(classify_shock)
    eligible = (
        shock_sign.notna() & frame["largest_bar_share"].notna() & frame["path_efficiency"].notna()
        & frame["decision_ts"].lt(PROTECTED_START) & frame["source_open_ts"].ge(max(TRAIN_START, first_complete_candidate_open))
        & frame[["btc_return_6h", "eth_return_6h", "btc_rv_24h", "candidate_rv_24h"]].notna().all(axis=1)
    )
    out = frame.loc[eligible, [
        "decision_ts", "shock_window_start", "residual_shock_6h", "residual_scale_6h",
        "residual_shock_z_6h", "largest_bar_share", "path_efficiency", "scale_block_count",
        "fit_observations", "alpha", "beta_btc", "beta_eth", "btc_return_6h", "eth_return_6h",
        "btc_rv_24h", "candidate_rv_24h", "candidate_lagged_trade_bar_availability",
        "candidate_lagged_mark_bar_availability",
    ]].copy()
    out["symbol"] = symbol
    out["residual_model"] = model
    out["sign"] = shock_sign.loc[eligible].astype(str)
    out["path_state"] = [classify_path_state(a, b) for a, b in zip(out["largest_bar_share"], out["path_efficiency"])]
    unavailable = {
        "source_rows": int(len(frame)),
        "pre_open_or_outside_train_rows": int((frame["source_open_ts"] < max(TRAIN_START, first_complete_candidate_open)).sum()),
        "known_lifecycle_invalidated_rows": int(lifecycle_invalid.sum()),
        "insufficient_estimation_rows": int(frame["alpha"].isna().sum()),
        "missing_trade_or_mark_shock_window": int((~contiguous).sum()),
        "insufficient_scale_blocks": int((frame["scale_block_count"] < MIN_SCALE_BLOCKS).sum()),
        "missing_parent_diagnostic_window": int(frame[["btc_return_6h", "eth_return_6h", "btc_rv_24h", "candidate_rv_24h"]].isna().any(axis=1).sum()),
        "eligible_shock_candidates": int(len(out)),
    }
    return out.reset_index(drop=True), unavailable


def make_attempt_registry(feature_hash: str, data_hash: str, cohort_hash: str, counts: Mapping[str, int] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counts = counts or {}
    for model in RESIDUAL_MODELS:
        for sign in SIGNS:
            for path_state in PATH_STATES:
                attempt_id = f"c01_{'primary' if model == PRIMARY_MODEL else 'robust'}_{sign}_{path_state}"
                rows.append({
                    "attempt_id": attempt_id, "definition_id": attempt_id, "family_id": FAMILY_ID,
                    "component_dimensions": json.dumps({"residual_model": model, "sign": sign, "path_state": path_state}, sort_keys=True),
                    "fixed_values": json.dumps({"shock_horizon": "6h", "activation": "abs_z_gte_3", "estimation": "30_calendar_days_daily_refit"}, sort_keys=True),
                    "status": "retained_for_generator_contract_review", "candidate_count": int(counts.get(attempt_id, 0)),
                    "killed_or_retained_for_later_review": "retained", "reason": "predeclared_attempt_retained_regardless_of_count",
                    "same_sample_prohibitions": "A1_or_compression_relabel;H43_BTC_impulse_laggard_relabel;relative_strength_hard_gate;generic_large_candle_fade;RFBS_Backside_LFBS_or_failed_breakdown_recombination;prior_high_retest_session_or_failure_threshold_reuse",
                    "source_commit": "9949b29ead0e6d6e17543ddd955bff0234805006", "feature_hash": feature_hash,
                    "data_authority_hash": data_hash, "candidate_cohort_hash": cohort_hash,
                    "candidate_cohort_version": CANDIDATE_COHORT_VERSION, "reference_panel_hash": REFERENCE_PANEL_HASH,
                })
    return pd.DataFrame(rows)


def assert_no_outcome_columns(columns: Iterable[str]) -> None:
    lowered = [str(column).lower() for column in columns]
    found = sorted({column for column in lowered if any(token in column for token in FORBIDDEN_OUTPUT_TOKENS)})
    if found:
        raise ValueError(f"outcome-derived columns forbidden in C01 diagnostic output: {found}")


def count_interval_overlaps(left: pd.DataFrame, right: pd.DataFrame) -> tuple[int, int]:
    """Return unique left rows and total pairs overlapping on symbol."""
    overlapping_left: set[str] = set()
    pairs = 0
    for symbol, lgroup in left.groupby("symbol", sort=True):
        rgroup = right[right["symbol"].eq(symbol)]
        if rgroup.empty:
            continue
        rstarts = pd.to_datetime(rgroup["episode_input_start"], utc=True).astype("int64").to_numpy()
        rends = pd.to_datetime(rgroup["episode_input_end"], utc=True).astype("int64").to_numpy()
        order = np.argsort(rstarts, kind="mergesort")
        rstarts, rends = rstarts[order], rends[order]
        for row in lgroup.itertuples(index=False):
            start = utc_timestamp(row.canonical_episode_input_start).value
            end = utc_timestamp(row.canonical_episode_input_end).value
            possible = np.searchsorted(rstarts, end, side="right")
            matches = int(np.count_nonzero(rends[:possible] >= start))
            if matches:
                overlapping_left.add(str(row.candidate_id))
                pairs += matches
    return len(overlapping_left), pairs


def cross_family_overlap_preflight(c01_tape: pd.DataFrame, repository_root: Path) -> pd.DataFrame:
    """Read only predeclared safe identity/timestamp columns from nearest families."""
    specs = [
        {
            "family": "A1_compression_continuation",
            "path": repository_root / "results/rebaseline/phase_kraken_a1_compression_targeted_materialization_controls_stress_20260712_v1/materialized/event_ledgers",
            "kind": "parquet_directory", "start": None, "decision": "decision_ts", "symbol": "symbol", "id": "event_id",
            "blocker": "decision identity exists but no causal episode_input_start is retained in the event-ledger schema",
        },
        {
            "family": "BTC_led_alt_diffusion_H43",
            "path": repository_root / "results/rebaseline/phase_kraken_btc_led_delayed_alt_diffusion_long_screen_20260716_v1/signals/raw_signal_manifest.csv",
            "kind": "csv", "start": "btc_source_ts", "decision": "decision_ts", "symbol": "symbol", "id": "raw_signal_id", "blocker": "",
        },
        {
            "family": "relative_strength_breakout_vs_BTC",
            "path": repository_root / "results/rebaseline/phase_kraken_relative_strength_breakout_vs_btc_screen_20260716_v1/signals/raw_signal_manifest.csv",
            "kind": "csv", "start": None, "decision": "decision_ts", "symbol": "symbol", "id": "raw_signal_id",
            "blocker": "decision identity exists but the causal breakout episode start is not retained in the raw-signal schema",
        },
        {
            "family": "RFBS_completed_failure",
            "path": repository_root / "results/rebaseline/phase_kraken_rfbs_signal_state_repaired_screen_20260715_v1/signals/raw_signal_manifest.csv",
            "kind": "csv", "start": "rally_anchor_ts", "decision": "decision_ts", "symbol": "symbol", "id": "raw_signal_address_hash", "blocker": "",
        },
        {
            "family": "repaired_Backside",
            "path": repository_root / "results/rebaseline/phase_kraken_backside_blowoff_signal_state_repaired_screen_20260715_v1/signals/raw_signal_manifest.csv",
            "kind": "csv", "start": None, "decision": "decision_ts", "symbol": "symbol", "id": "raw_signal_address_hash",
            "blocker": "decision identity exists but the causal blowoff/extension episode start is not retained in the raw-signal schema",
        },
    ]
    report: list[dict[str, Any]] = []
    for spec in specs:
        path = Path(spec["path"])
        base = {
            "prior_family": spec["family"], "source_path": str(path.relative_to(repository_root)) if path.exists() else str(path),
            "safe_columns_read": "", "mapping_status": "blocked", "mapping_blocker": "", "prior_identity_rows": 0,
            "c01_candidate_rows": len(c01_tape), "c01_candidates_with_overlap": 0, "overlap_pair_count": 0,
            "economic_columns_read": 0, "protected_rows_read": 0,
        }
        if not path.exists():
            base["mapping_blocker"] = "declared nearest-family identity source does not exist"
            report.append(base)
            continue
        if spec["start"] is None:
            base["mapping_blocker"] = spec["blocker"]
            report.append(base)
            continue
        columns = [spec["id"], spec["symbol"], spec["start"], spec["decision"]]
        if spec["kind"] == "csv":
            prior = pd.read_csv(path, usecols=columns)
        else:
            prior = pd.read_parquet(path, columns=columns)
        prior = prior.rename(columns={spec["id"]: "prior_id", spec["symbol"]: "symbol", spec["start"]: "episode_input_start", spec["decision"]: "decision_ts"})
        prior["decision_ts"] = pd.to_datetime(prior["decision_ts"], utc=True, errors="raise")
        prior["episode_input_start"] = pd.to_datetime(prior["episode_input_start"], utc=True, errors="raise")
        if (prior["decision_ts"] >= PROTECTED_START).any() or (prior["decision_ts"] < TRAIN_START).any():
            raise ValueError(f"out-of-bound prior identity row in {spec['family']}")
        prior = prior.drop_duplicates("prior_id").copy()
        prior["episode_input_end"] = prior["decision_ts"] + pd.Timedelta(hours=24)
        if (prior["episode_input_start"] > prior["decision_ts"]).any():
            raise ValueError(f"non-causal prior interval in {spec['family']}")
        overlapping, pairs = count_interval_overlaps(c01_tape, prior)
        base.update({
            "safe_columns_read": ";".join(columns), "mapping_status": "mapped_safe_identity_interval",
            "mapping_blocker": "", "prior_identity_rows": len(prior),
            "c01_candidates_with_overlap": overlapping, "overlap_pair_count": pairs,
        })
        report.append(base)
    return pd.DataFrame(report)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _definition_id(model: str, sign: str, path_state: str) -> str:
    return f"c01_{'primary' if model == PRIMARY_MODEL else 'robust'}_{sign}_{path_state}"


def build(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe_rows = load_safe_manifest(Path(args.market_manifest))
    reference_manifest = json.loads(Path(args.reference_manifest).read_text(encoding="utf-8"))
    if reference_manifest.get("reference_panel_hash") != REFERENCE_PANEL_HASH:
        raise ValueError("Stage 2A1 reference-panel hash mismatch")
    reference_manifest_hash = sha256_file(Path(args.reference_manifest))
    lifecycle_source_hash = sha256_file(Path(args.lifecycle_source))
    data_hash = authority_hash(safe_rows, reference_manifest_hash, lifecycle_source_hash)
    cohort, opening_dates, cohort_audit = current_candidate_cohort(Path(args.instrument_source), safe_rows)
    lifecycle_invalidations = load_known_lifecycle_invalidations(Path(args.lifecycle_source))
    cohort_audit["known_lifecycle_invalidation_intervals"] = cohort_audit["symbol"].map(
        lambda symbol: json.dumps([[iso_utc(start), iso_utc(end)] for start, end in lifecycle_invalidations.get(symbol, [])])
    )
    cohort_hash = deterministic_hash({"version": CANDIDATE_COHORT_VERSION, "symbols": cohort, "opening_dates": {s: iso_utc(opening_dates[s]) for s in cohort}})
    feature_contract = {
        "family_id": FAMILY_ID, "feature_version": FEATURE_VERSION, "reference_panel_id": REFERENCE_PANEL_ID,
        "reference_panel_hash": REFERENCE_PANEL_HASH, "candidate_cohort_version": CANDIDATE_COHORT_VERSION,
        "residual_models": list(RESIDUAL_MODELS), "estimation_days": ESTIMATION_DAYS,
        "minimum_estimation_observations": MIN_ESTIMATION_ROWS, "daily_refit": "prior_UTC_days_only",
        "shock_bars": SHOCK_BARS, "scale_blocks_minimum": MIN_SCALE_BLOCKS,
        "scale_blocks": "UTC_anchored_nonoverlapping_6h_blocks_ending_on_or_before_shock_window_start",
        "activation": {"positive": ">=3.0", "negative": "<=-3.0"},
        "path_states": {"smooth": "largest_bar_share<=0.25 and path_efficiency>=0.50", "jump_dominated": "largest_bar_share>=0.50", "intermediate": "all_other"},
        "known_lifecycle_policy": "official_terminal_date_through_end_of_documented_resumption_date; open_ended_to_protected_cutoff",
        "lifecycle_source_sha256": lifecycle_source_hash,
        "train_bounds": [iso_utc(TRAIN_START), iso_utc(PROTECTED_START)], "outcomes_authorized": False,
    }
    feature_hash = deterministic_hash(feature_contract)
    attempts = make_attempt_registry(feature_hash, data_hash, cohort_hash)
    _write_csv(output / "C01_FAMILY_AND_ATTEMPT_REGISTER.csv", attempts)
    _write_csv(output / "C01_CANDIDATE_COHORT_AUDIT.csv", cohort_audit)

    authority_by_key = {(row.symbol, row.dataset): row for row in safe_rows}
    del authority_by_key  # Duplicate key existence is intentionally not used; rows are sharded.
    ref_dir = Path(args.reference_final_day_dir)
    reference: dict[str, tuple[pd.DataFrame, pd.DataFrame, str, str]] = {}
    for symbol in sorted(REFERENCE_SYMBOLS):
        trade, trade_ref = read_authorized_bars(safe_rows, symbol, "historical_trade_candles_5m")
        mark, mark_ref = read_authorized_bars(safe_rows, symbol, "historical_mark_candles_5m")
        trade_path = ref_dir / f"{symbol}_trade.parquet"
        mark_path = ref_dir / f"{symbol}_mark.parquet"
        trade = append_reference_final_day(trade, trade_path, symbol)
        mark = append_reference_final_day(mark, mark_path, symbol)
        reference[symbol] = (trade, mark, trade_ref, mark_ref)

    candidates: list[pd.DataFrame] = []
    unavailable_rows: list[dict[str, Any]] = []
    authority_ledger: list[dict[str, Any]] = []
    for symbol in cohort:
        trade, trade_ref = read_authorized_bars(
            safe_rows, symbol, "historical_trade_candles_5m", not_before=opening_dates[symbol],
        )
        mark, mark_ref = read_authorized_bars(
            safe_rows, symbol, "historical_mark_candles_5m", not_before=opening_dates[symbol],
        )
        authority_ledger.extend([
            {"ref_id": trade_ref, "symbol": symbol, "dataset": "trade", "manifest_safe_shards": sum(row.symbol == symbol and row.dataset == "historical_trade_candles_5m" for row in safe_rows), "protected_shards_opened": 0},
            {"ref_id": mark_ref, "symbol": symbol, "dataset": "mark", "manifest_safe_shards": sum(row.symbol == symbol and row.dataset == "historical_mark_candles_5m" for row in safe_rows), "protected_shards_opened": 0},
        ])
        for model in RESIDUAL_MODELS:
            features, unavailable = compute_symbol_features(
                symbol, trade, mark, reference["PF_XBTUSD"][0], reference["PF_XBTUSD"][1],
                reference["PF_ETHUSD"][0], reference["PF_ETHUSD"][1], model, opening_dates[symbol],
                lifecycle_invalidations.get(symbol, ()),
            )
            unavailable_rows.extend({"symbol": symbol, "residual_model": model, "reason": reason, "row_count": count} for reason, count in unavailable.items())
            if features.empty:
                continue
            features["family_id"] = FAMILY_ID
            features["definition_id"] = [_definition_id(model, sign, path) for sign, path in zip(features["sign"], features["path_state"])]
            features["attempt_id"] = features["definition_id"]
            features["venue"] = "Kraken"
            features["shock_window_end"] = features["decision_ts"]
            features["canonical_episode_input_start"] = features["shock_window_start"]
            features["canonical_episode_input_end"] = features["decision_ts"] + pd.Timedelta(hours=24)
            features["residual_model_version"] = model
            features["feature_version"] = FEATURE_VERSION
            features["reference_panel_id"] = REFERENCE_PANEL_ID
            features["reference_panel_hash"] = REFERENCE_PANEL_HASH
            features["candidate_cohort_version"] = CANDIDATE_COHORT_VERSION
            features["candidate_cohort_hash"] = cohort_hash
            features["data_authority_hash"] = data_hash
            features["trade_path_refs"] = trade_ref
            features["mark_path_refs"] = mark_ref
            features["protected_row_count"] = 0
            identities = [assign_candidate_identity(row) for row in features.to_dict("records")]
            features["candidate_id"] = [item[0] for item in identities]
            features["economic_address"] = [item[1] for item in identities]
            candidates.append(features)
    tape = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    if tape.empty:
        raise ValueError("complete cohort produced no diagnostic shocks; inspect unavailable counts without changing thresholds")
    tape = cluster_intervals(tape)
    preferred = [
        "family_id", "definition_id", "attempt_id", "candidate_id", "economic_address",
        "canonical_episode_id", "canonical_episode_input_start", "canonical_episode_input_end",
        "episode_cluster_start", "episode_cluster_end", "episode_member_count", "symbol", "venue",
        "decision_ts", "shock_window_start", "shock_window_end", "residual_model_version", "feature_version",
        "reference_panel_id", "reference_panel_hash", "candidate_cohort_version", "candidate_cohort_hash",
        "data_authority_hash", "trade_path_refs", "mark_path_refs", "protected_row_count", "sign", "path_state",
        "residual_shock_6h", "residual_scale_6h", "residual_shock_z_6h", "largest_bar_share",
        "path_efficiency", "scale_block_count", "fit_observations", "alpha", "beta_btc", "beta_eth",
        "btc_return_6h", "eth_return_6h", "btc_rv_24h", "candidate_rv_24h",
        "candidate_lagged_trade_bar_availability", "candidate_lagged_mark_bar_availability",
    ]
    tape = tape[preferred].sort_values(["symbol", "decision_ts", "residual_model_version", "definition_id"], kind="mergesort").reset_index(drop=True)
    assert_no_outcome_columns(tape.columns)
    if tape["candidate_id"].duplicated().any() or tape["economic_address"].duplicated().any():
        raise ValueError("duplicate C01 candidate identity")
    if not tape["decision_ts"].lt(PROTECTED_START).all() or not tape["decision_ts"].ge(TRAIN_START).all():
        raise ValueError("out-of-bound C01 candidate")
    tape.to_parquet(output / "C01_GENERATOR_DIAGNOSTIC_TAPE.parquet", index=False)
    _write_csv(output / "C01_DATA_AUTHORITY.csv", pd.DataFrame(authority_ledger))
    unavailable_frame = pd.DataFrame(unavailable_rows)
    _write_csv(output / "C01_UNAVAILABLE_FEATURE_COUNTS.csv", unavailable_frame)

    observed = tape.groupby([tape["decision_ts"].dt.year.rename("year"), "symbol", "residual_model_version", "sign", "path_state"]).size().rename("candidate_count").reset_index()
    matrix = pd.MultiIndex.from_product(
        [range(2023, 2026), cohort, RESIDUAL_MODELS, SIGNS, PATH_STATES],
        names=["year", "symbol", "residual_model_version", "sign", "path_state"],
    ).to_frame(index=False).merge(observed, how="left", on=["year", "symbol", "residual_model_version", "sign", "path_state"])
    matrix["candidate_count"] = matrix["candidate_count"].fillna(0).astype(int)
    _write_csv(output / "C01_CANDIDATE_COUNT_MATRIX.csv", matrix)
    counts = tape["attempt_id"].value_counts().to_dict()
    attempts = make_attempt_registry(feature_hash, data_hash, cohort_hash, counts)
    _write_csv(output / "C01_FAMILY_AND_ATTEMPT_REGISTER.csv", attempts)
    overlap = cross_family_overlap_preflight(tape, Path(args.repository_root).resolve())
    _write_csv(output / "C01_CROSS_FAMILY_OVERLAP_PREFLIGHT.csv", overlap)

    episode_sizes = tape[["canonical_episode_id", "episode_member_count"]].drop_duplicates()["episode_member_count"]
    (output / "C01_EPISODE_IDENTITY_REPORT.md").write_text(
        "# C01 Episode Identity Report\n\n"
        f"- Candidate rows: {len(tape):,}.\n- Canonical same-symbol overlap episodes: {tape['canonical_episode_id'].nunique():,}.\n"
        f"- Duplicate candidate IDs: {int(tape['candidate_id'].duplicated().sum())}.\n"
        f"- Duplicate economic addresses: {int(tape['economic_address'].duplicated().sum())}.\n"
        f"- Episode member count min/median/max: {int(episode_sizes.min())}/{float(episode_sizes.median()):.1f}/{int(episode_sizes.max())}.\n"
        "- Interval: shock-window start through decision plus 24 hours; overlap uses no added gap.\n"
        "- Identity uses no outcome or future-return field.\n",
        encoding="utf-8",
    )
    _write_feature_docs(output, feature_contract, feature_hash, data_hash, cohort_hash, cohort, tape, unavailable_frame)


def _write_feature_docs(
    output: Path, contract: Mapping[str, Any], feature_hash: str, data_hash: str, cohort_hash: str,
    cohort: Sequence[str], tape: pd.DataFrame, unavailable: pd.DataFrame,
) -> None:
    (output / "C01_FEATURE_CONTRACT.md").write_text(f"""# C01 Feature Contract

Family: `{FAMILY_ID}`

Feature version: `{FEATURE_VERSION}`
Feature-contract hash: `{feature_hash}`

Five-minute source timestamps are candle opens. A source row becomes available at `source_open_ts + 5m`; that completed close is `decision_ts`. The daily OLS used on UTC day D is fitted from aligned valid observations in `[D-30d,D)`. Valid observations require candidate and required-factor trade returns plus distinct mark rows. At least `{MIN_ESTIMATION_ROWS}` of `{EXPECTED_ESTIMATION_ROWS}` expected observations are required.

The 6h shock uses 72 consecutive completed residuals. Scale is the sample standard deviation of valid UTC-anchored, non-overlapping 6h residual sums whose block ends are on or before the current shock-window start and within its prior 30 calendar days. At least 80 blocks are required. This removes overlap between the current shock and scale inputs.

Mark closes are eligibility/quality inputs only and are never substituted for trade returns. Parent returns/RV are diagnostics, not gates. No funding, OI, basis, spread, session, catalyst, prior-high, relative-strength, forward return, exit, or PnL field is computed.

Candidate cohort is exactly `{CANDIDATE_COHORT_VERSION}`. It is current-roster capped and is not survivorship-free. Official opening dates and complete event-time bars fail closed; continuous tradeability is not claimed. Known official terminal intervals are masked from the settlement date through the end of a later documented resumption date, or through the protected cutoff when no later resumption exists. Date-only lifecycle authority is conservatively interpreted as whole UTC dates.
""", encoding="utf-8")
    schema = {
        "schema_version": "c01_generator_diagnostic_tape_v1",
        "feature_contract_hash": feature_hash,
        "columns": [{"name": name, "dtype": str(dtype), "causal": True, "outcome": False} for name, dtype in tape.dtypes.items()],
        "forbidden_output_tokens": list(FORBIDDEN_OUTPUT_TOKENS),
    }
    (output / "C01_FEATURE_SCHEMA.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = tape.groupby(["residual_model_version", "sign", "path_state"]).size().rename("count").reset_index()
    (output / "C01_GENERATOR_REVIEW.md").write_text(
        "# C01 Generator Review\n\nStatus: generator-only, non-economic evidence.\n\n"
        f"- Cohort symbols: {len(cohort)}.\n- Candidate rows: {len(tape):,}.\n- Attempts retained: 12/12.\n"
        f"- Candidate cohort hash: `{cohort_hash}`.\n- Data authority hash: `{data_hash}`.\n"
        f"- Protected rows opened/emitted: 0/0.\n- Economic outputs: 0.\n\n"
        "## Counts\n\n" + summary.to_markdown(index=False) + "\n\n"
        "Counts are feasibility diagnostics and did not alter thresholds or attempts. Unavailability is recorded in `C01_UNAVAILABLE_FEATURE_COUNTS.csv`.\n",
        encoding="utf-8",
    )
    (output / "C01_NEXT_CONTRACT_RECOMMENDATION.md").write_text("""# Next C01 Contract Recommendation

Status: `ready_for_C01_generator_contract_review` only if validation and independent review remain approved.

The later contract should freeze this exact 12-attempt generator, cohort cap, causal feature hash, candidate identity, and episode interface before any outcome access. It must separately predeclare continuation/failure entries, actual-exit non-overlap, controls, costs, funding, boundary handling, multiplicity correction, and decision vocabulary. Candidate counts in this task are not authority to prune branches, alter z/path thresholds, or run economics.
""", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-manifest", required=True)
    parser.add_argument("--instrument-source", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--reference-final-day-dir", required=True)
    parser.add_argument("--lifecycle-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
