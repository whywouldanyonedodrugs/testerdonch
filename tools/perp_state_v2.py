from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PERP_STATE_CONTRACT_VERSION = "perp_state_v2"
DEFAULT_CONTEXT_ROOT = Path("/opt/parquet/bybit_context_5m")
BAR_STEP = pd.Timedelta(minutes=5)
DEFAULT_SYMBOL_MONTH_MIN_COVERAGE = 0.95
CAPABILITY_FULL = "full_context_eligible"
CAPABILITY_PARTIAL = "partial_context_only"
CAPABILITY_NONE = "no_context_file"

GROUP_SOURCE_CLOSE_COL = {
    "mark": "mark_source_close_ts",
    "index": "index_source_close_ts",
    "premium": "premium_source_close_ts",
    "lsr": "lsr_source_close_ts",
}

GROUP_REQUIRED_COLUMNS = {
    "mark": ["mark_open", "mark_high", "mark_low", "mark_close", "mark_source_open_ts", "mark_source_close_ts"],
    "index": ["index_open", "index_high", "index_low", "index_close", "index_source_open_ts", "index_source_close_ts"],
    "premium": ["premium_open", "premium_high", "premium_low", "premium_close", "premium_source_open_ts", "premium_source_close_ts"],
    "lsr": ["long_account_ratio", "short_account_ratio", "long_short_account_ratio", "lsr_source_ts", "lsr_source_close_ts"],
}

TIMESTAMP_COLUMNS = (
    "timestamp",
    "mark_source_open_ts",
    "mark_source_close_ts",
    "index_source_open_ts",
    "index_source_close_ts",
    "premium_source_open_ts",
    "premium_source_close_ts",
    "lsr_source_ts",
    "lsr_source_close_ts",
    "context_source_close_ts",
)

RAW_CONTEXT_SCHEMA_COLUMNS = (
    "timestamp",
    "mark_open",
    "mark_high",
    "mark_low",
    "mark_close",
    "mark_source_open_ts",
    "mark_source_close_ts",
    "index_open",
    "index_high",
    "index_low",
    "index_close",
    "index_source_open_ts",
    "index_source_close_ts",
    "premium_open",
    "premium_high",
    "premium_low",
    "premium_close",
    "premium_source_open_ts",
    "premium_source_close_ts",
    "long_account_ratio",
    "short_account_ratio",
    "long_short_account_ratio",
    "lsr_source_ts",
    "lsr_source_close_ts",
    "context_source_close_ts",
)

DERIVED_CONTEXT_COLUMNS = (
    "mark_index_spread_pct",
    "premium_compression_1h",
    "premium_compression_4h",
    "lsr_delta_1h",
    "lsr_delta_4h",
)

# The durable sidecar contract is raw data + provenance only.  Derived columns
# may exist in older parquet files, but they are not schema truth and are
# recomputed on every load.
CANONICAL_SCHEMA_COLUMNS = RAW_CONTEXT_SCHEMA_COLUMNS


@dataclass(frozen=True)
class ContextGroupRequirement:
    group: str
    max_staleness: pd.Timedelta = pd.Timedelta(0)

    def __post_init__(self) -> None:
        g = str(self.group).strip().lower()
        if g not in GROUP_REQUIRED_COLUMNS:
            raise ValueError(f"unknown Bybit context group: {self.group!r}")
        object.__setattr__(self, "group", g)
        object.__setattr__(self, "max_staleness", pd.to_timedelta(self.max_staleness))


@dataclass(frozen=True)
class ContextJoinContract:
    family: str
    requirements: tuple[ContextGroupRequirement, ...]
    min_symbol_month_coverage: float = DEFAULT_SYMBOL_MONTH_MIN_COVERAGE

    @property
    def required_groups(self) -> tuple[str, ...]:
        return tuple(r.group for r in self.requirements)

    @property
    def max_join_lag(self) -> pd.Timedelta:
        if not self.requirements:
            return pd.Timedelta(0)
        return max((r.max_staleness for r in self.requirements), default=pd.Timedelta(0))

    def freshness_by_group(self) -> dict[str, pd.Timedelta]:
        return {r.group: pd.to_timedelta(r.max_staleness) for r in self.requirements}


def context_path(symbol: str, root: Path | str = DEFAULT_CONTEXT_ROOT) -> Path:
    return Path(root) / f"{str(symbol).upper()}.parquet"


def schema_hash(columns: Sequence[str] = CANONICAL_SCHEMA_COLUMNS) -> str:
    payload = json.dumps(list(columns), separators=(",", ":"), sort_keys=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def dataframe_signature(df: pd.DataFrame) -> str:
    payload = {
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "row_count": int(len(df)),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def context_required_columns(groups: Iterable[str]) -> list[str]:
    cols: list[str] = []
    for raw in groups:
        group = str(raw).strip().lower()
        if group not in GROUP_REQUIRED_COLUMNS:
            raise ValueError(f"unknown Bybit context group: {raw!r}")
        for col in GROUP_REQUIRED_COLUMNS[group]:
            if col not in cols:
                cols.append(col)
    return cols


def group_history_available(df: pd.DataFrame, group: str) -> bool:
    """Return True only when the sidecar has non-null history for a context group."""
    group = str(group).strip().lower()
    cols = GROUP_REQUIRED_COLUMNS[group]
    if any(c not in df.columns for c in cols):
        return False
    source_col = GROUP_SOURCE_CLOSE_COL[group]
    if source_col not in df.columns:
        return False
    source_close = pd.to_datetime(df[source_col], utc=True, errors="coerce")
    values_present = df[cols].notna().all(axis=1)
    return bool((source_close.notna() & values_present).any())


def group_history_available_in_window(df: pd.DataFrame, group: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    group = str(group).strip().lower()
    if df.empty or any(c not in df.columns for c in GROUP_REQUIRED_COLUMNS[group]):
        return False
    source_col = GROUP_SOURCE_CLOSE_COL[group]
    if source_col not in df.columns:
        return False
    source_close = pd.to_datetime(df[source_col], utc=True, errors="coerce")
    values_present = df[GROUP_REQUIRED_COLUMNS[group]].notna().all(axis=1)
    start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
    return bool((source_close.notna() & values_present & (source_close >= start) & (source_close <= end)).any())


def source_unavailable_reason(groups: Sequence[str], instrument_status: str = "") -> str:
    groups = tuple(str(g).strip().lower() for g in groups)
    if not groups:
        return ""
    status = str(instrument_status or "").strip().lower()
    if status == "closed" and any(g in {"mark", "index", "premium"} for g in groups):
        return "context_ineligible_source_unavailable_closed_symbol"
    mip = [g for g in ("mark", "index", "premium") if g in groups]
    if len(mip) == 3:
        return "source_unavailable_mark_index_premium"
    if len(groups) == 1:
        return f"source_unavailable_{groups[0]}"
    return "source_unavailable_" + "_".join(groups)


def build_symbol_context_capability_manifest(
    symbols: Sequence[str],
    *,
    context_root: Path | str = DEFAULT_CONTEXT_ROOT,
    instrument_status_by_symbol: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """
    Classify sidecar capability for each symbol without requiring a perfect schema.

    This deliberately separates ETL completion/schema shape from research
    eligibility. A sidecar file may exist and be schema-valid while still being
    research-ineligible for families that need unavailable source groups.
    """
    root = Path(context_root)
    status_map = {str(k).upper(): str(v) for k, v in dict(instrument_status_by_symbol or {}).items()}
    rows: list[dict[str, Any]] = []
    for symbol in sorted({str(s).upper() for s in symbols}):
        path = context_path(symbol, root)
        row: dict[str, Any] = {
            "symbol": symbol,
            "active_5m_universe_flag": True,
            "sidecar_file_exists": path.exists(),
            "sidecar_schema_valid": False,
            "has_mark_history": False,
            "has_index_history": False,
            "has_premium_history": False,
            "has_lsr_history": False,
            "instrument_status": status_map.get(symbol, "unknown"),
            "context_capability_status": CAPABILITY_NONE,
            "schema_missing": ",".join(CANONICAL_SCHEMA_COLUMNS),
            "row_count": 0,
            "etl_completion_status": "file_missing",
            "research_eligibility_status": "no_context_file",
        }
        if not path.exists():
            rows.append(row)
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            row.update(
                {
                    "sidecar_file_exists": True,
                    "etl_completion_status": f"read_error:{type(exc).__name__}",
                    "research_eligibility_status": "schema_invalid",
                }
            )
            rows.append(row)
            continue
        missing = [c for c in CANONICAL_SCHEMA_COLUMNS if c not in df.columns]
        row["row_count"] = int(len(df))
        row["schema_missing"] = ",".join(missing)
        row["sidecar_schema_valid"] = not missing
        for group in GROUP_REQUIRED_COLUMNS:
            row[f"has_{group}_history"] = group_history_available(df, group)
        has_all = all(bool(row[f"has_{g}_history"]) for g in GROUP_REQUIRED_COLUMNS)
        has_any = any(bool(row[f"has_{g}_history"]) for g in GROUP_REQUIRED_COLUMNS)
        row["etl_completion_status"] = "file_written"
        if has_all and not missing:
            row["context_capability_status"] = CAPABILITY_FULL
            row["research_eligibility_status"] = "full_context_eligible"
        elif has_any:
            row["context_capability_status"] = CAPABILITY_PARTIAL
            row["research_eligibility_status"] = "partial_context_only"
        else:
            row["context_capability_status"] = CAPABILITY_PARTIAL if not missing else CAPABILITY_NONE
            row["research_eligibility_status"] = "schema_invalid" if missing else "no_required_group_history"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("symbol", kind="mergesort").reset_index(drop=True)


def load_bybit_context(
    symbol: str,
    *,
    root: Path | str = DEFAULT_CONTEXT_ROOT,
    start_date: object | None = None,
    end_date: object | None = None,
    require_schema: bool = True,
) -> pd.DataFrame:
    path = context_path(symbol, root)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if require_schema:
        missing = [c for c in CANONICAL_SCHEMA_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"Bybit context sidecar schema mismatch for {symbol}: missing {missing}")
    for col in TIMESTAMP_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    if "timestamp" not in df.columns:
        raise KeyError(f"Bybit context sidecar missing timestamp for {symbol}")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    df = df.drop_duplicates("timestamp", keep="last")
    if start_date is not None:
        df = df[df["timestamp"] >= pd.to_datetime(start_date, utc=True, errors="coerce")]
    if end_date is not None:
        df = df[df["timestamp"] <= pd.to_datetime(end_date, utc=True, errors="coerce")]
    return add_context_derived_columns(df.reset_index(drop=True))


def add_context_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def num_col(name: str) -> pd.Series:
        if name not in out.columns:
            return pd.Series(np.nan, index=out.index)
        return pd.to_numeric(out[name], errors="coerce")

    mark = num_col("mark_close")
    index = num_col("index_close")
    premium = num_col("premium_close")
    lsr = num_col("long_short_account_ratio")
    out["mark_index_spread_pct"] = (mark - index) / index.replace(0.0, np.nan)
    out["premium_compression_1h"] = premium - premium.shift(12)
    out["premium_compression_4h"] = premium - premium.shift(48)
    out["lsr_delta_1h"] = lsr - lsr.shift(12)
    out["lsr_delta_4h"] = lsr - lsr.shift(48)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def validate_context_schema(df: pd.DataFrame, *, required_groups: Iterable[str] | None = None) -> list[str]:
    groups = tuple(required_groups or ())
    required = list(CANONICAL_SCHEMA_COLUMNS if not groups else ["timestamp", *context_required_columns(groups), "context_source_close_ts"])
    missing = [c for c in required if c not in df.columns]
    return missing


def prepare_context_for_join(context: pd.DataFrame, contract: ContextJoinContract) -> pd.DataFrame:
    groups = contract.required_groups
    if not groups:
        return pd.DataFrame()
    missing = validate_context_schema(context, required_groups=groups)
    if missing:
        raise KeyError(f"Bybit context missing required columns for family={contract.family}: {missing}")
    out = add_context_derived_columns(context)
    close_cols = [GROUP_SOURCE_CLOSE_COL[g] for g in groups]
    for col in close_cols + ["context_source_close_ts"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out["context_asof_ts"] = out[close_cols].max(axis=1)
    if "context_source_close_ts" in out.columns:
        out["context_source_close_ts"] = pd.to_datetime(out["context_source_close_ts"], utc=True, errors="coerce")
        # Preserve source column but make the as-of key the max of required groups.
        out["context_required_source_close_ts"] = out["context_asof_ts"]
    required_value_cols = context_required_columns(groups)
    out["bybit_context_missing_required"] = out[required_value_cols].isna().any(axis=1)
    out = out[~out["bybit_context_missing_required"]].dropna(subset=["context_asof_ts"]).copy()
    out = out.sort_values("context_asof_ts", kind="mergesort").drop_duplicates("context_asof_ts", keep="last")
    return out.reset_index(drop=True)


def join_context_asof(
    decisions: pd.DataFrame,
    context: pd.DataFrame,
    contract: ContextJoinContract,
    *,
    decision_ts_col: str = "decision_ts",
    fail_closed: bool = True,
) -> pd.DataFrame:
    if decision_ts_col not in decisions.columns:
        raise KeyError(f"decision dataframe missing {decision_ts_col!r}")
    left = decisions.copy()
    left[decision_ts_col] = pd.to_datetime(left[decision_ts_col], utc=True, errors="coerce")
    left["_decision_original_order"] = np.arange(len(left), dtype=np.int64)
    left = left.sort_values(decision_ts_col, kind="mergesort")
    if left.empty:
        left["bybit_context_available"] = False
        left["bybit_context_reject_reason"] = ""
        return left.drop(columns=["_decision_original_order"]).reset_index(drop=True)

    prepared = prepare_context_for_join(context, contract)
    if prepared.empty:
        out = left.copy()
        out["bybit_context_available"] = False
        out["bybit_context_reject_reason"] = "missing_required_context"
    else:
        tolerance = contract.max_join_lag
        out = pd.merge_asof(
            left,
            prepared,
            left_on=decision_ts_col,
            right_on="context_asof_ts",
            direction="backward",
            tolerance=tolerance,
            suffixes=("", "_ctx"),
        )
        out["bybit_context_available"] = out["context_asof_ts"].notna()
        out["bybit_context_reject_reason"] = np.where(out["bybit_context_available"], "", "missing_required_context")
        after = out["context_asof_ts"].notna() & (pd.to_datetime(out["context_asof_ts"], utc=True, errors="coerce") > out[decision_ts_col])
        if bool(after.any()):
            raise RuntimeError("Bybit context causality violation: context_asof_ts > decision_ts")

    freshness = contract.freshness_by_group()
    stale_any = pd.Series(False, index=out.index)
    source_after_any = pd.Series(False, index=out.index)
    for group in contract.required_groups:
        close_col = GROUP_SOURCE_CLOSE_COL[group]
        staleness_col = f"{group}_context_staleness_seconds"
        if close_col in out.columns:
            close_ts = pd.to_datetime(out[close_col], utc=True, errors="coerce")
        else:
            close_ts = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        out[staleness_col] = (out[decision_ts_col] - close_ts).dt.total_seconds()
        group_after = close_ts.notna() & (close_ts > out[decision_ts_col])
        source_after_any |= group_after.fillna(False)
        max_stale_sec = freshness[group].total_seconds()
        group_stale = out["bybit_context_available"].fillna(False) & (
            close_ts.isna() | (out[staleness_col] < 0) | (out[staleness_col] > max_stale_sec)
        )
        stale_any |= group_stale.fillna(False)
    if bool(source_after_any.any()):
        raise RuntimeError("Bybit context causality violation: required group source_close_ts > decision_ts")
    out.loc[stale_any, "bybit_context_available"] = False
    out.loc[stale_any, "bybit_context_reject_reason"] = "stale_required_context"
    out = out.sort_values("_decision_original_order", kind="mergesort").drop(columns=["_decision_original_order"])
    if fail_closed:
        out = out[out["bybit_context_available"].fillna(False)].copy()
    return out.reset_index(drop=True)


def context_availability_for_decisions(
    decisions: pd.Series | pd.DatetimeIndex,
    context: pd.DataFrame,
    contract: ContextJoinContract,
) -> pd.DataFrame:
    """
    Lightweight availability scanner for eligibility preflight.

    This follows the same source-close and max-staleness contract as
    `join_context_asof` but returns only availability/reason/staleness columns,
    avoiding wide dataframe joins during full-universe audits.
    """
    decision_ts = pd.to_datetime(pd.Series(decisions), utc=True, errors="coerce")
    out = pd.DataFrame({"decision_ts": decision_ts})
    out["bybit_context_available"] = False
    out["bybit_context_reject_reason"] = "missing_required_context"
    for group in contract.required_groups:
        out[f"{group}_context_staleness_seconds"] = np.nan
    if out.empty:
        return out
    try:
        prepared = prepare_context_for_join(context, contract)
    except Exception:
        return out
    if prepared.empty:
        return out
    prepared = prepared.sort_values("context_asof_ts", kind="mergesort").reset_index(drop=True)
    asof = pd.to_datetime(prepared["context_asof_ts"], utc=True, errors="coerce")
    valid_asof = asof.notna()
    prepared = prepared.loc[valid_asof].reset_index(drop=True)
    asof = asof.loc[valid_asof].reset_index(drop=True)
    if prepared.empty:
        return out
    asof_ns = asof.astype("int64").to_numpy()
    dec_ns = decision_ts.astype("int64").to_numpy()
    idx = np.searchsorted(asof_ns, dec_ns, side="right") - 1
    has_match = idx >= 0
    safe_idx = np.clip(idx, 0, max(0, len(prepared) - 1))
    tolerance_ns = int(contract.max_join_lag.value)
    asof_age = dec_ns - asof_ns[safe_idx]
    available = has_match & (asof_age >= 0) & (asof_age <= tolerance_ns)
    stale_any = np.zeros(len(out), dtype=bool)
    source_after_any = np.zeros(len(out), dtype=bool)
    freshness = contract.freshness_by_group()
    for group in contract.required_groups:
        close_col = GROUP_SOURCE_CLOSE_COL[group]
        close = pd.to_datetime(prepared[close_col], utc=True, errors="coerce")
        close_ns = close.astype("int64").to_numpy()
        group_age = dec_ns - close_ns[safe_idx]
        group_age_sec = group_age / 1_000_000_000.0
        group_age_sec[~has_match] = np.nan
        out[f"{group}_context_staleness_seconds"] = group_age_sec
        max_ns = int(freshness[group].value)
        group_after = has_match & (group_age < 0)
        group_stale = available & ((group_age < 0) | (group_age > max_ns))
        source_after_any |= group_after
        stale_any |= group_stale
    if bool(source_after_any.any()):
        raise RuntimeError("Bybit context causality violation: required group source_close_ts > decision_ts")
    available = available & (~stale_any)
    out["bybit_context_available"] = available
    out.loc[available, "bybit_context_reject_reason"] = ""
    out.loc[stale_any, "bybit_context_reject_reason"] = "stale_required_context"
    return out


def exact_required_source_close_set(context: pd.DataFrame, groups: Sequence[str]) -> set[int]:
    """Return source-close nanoseconds where every required group is complete.

    Eligibility uses this stricter closed-bar check by default. A decision is
    covered only when the required group source close timestamps exactly equal
    `decision_ts`; no stale carry-forward is counted as symbol-month coverage.
    """
    if context.empty or not groups:
        return set()
    group_sets: list[set[int]] = []
    for group in groups:
        group = str(group).strip().lower()
        cols = GROUP_REQUIRED_COLUMNS[group]
        if any(c not in context.columns for c in cols):
            return set()
        close_col = GROUP_SOURCE_CLOSE_COL[group]
        present = context[cols].notna().all(axis=1)
        close_ts = pd.to_datetime(context[close_col], utc=True, errors="coerce")
        vals = close_ts[present & close_ts.notna()].astype("int64").to_numpy().tolist()
        group_sets.append(set(vals))
    if not group_sets:
        return set()
    valid = group_sets[0]
    for sub in group_sets[1:]:
        valid = valid.intersection(sub)
        if not valid:
            break
    return valid


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
    months = pd.period_range(start=start.tz_localize(None).to_period("M"), end=end.tz_localize(None).to_period("M"), freq="M")
    return [str(p) for p in months]


def expected_5m_closes_for_month(year_month: str, phase_start: pd.Timestamp, phase_end: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(f"{year_month}-01", tz="UTC")
    end = start + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=55)
    start = max(start, pd.Timestamp(phase_start))
    end = min(end, pd.Timestamp(phase_end))
    if end < start:
        return pd.DatetimeIndex([], tz="UTC")
    # decision_ts is a closed 5m bar timestamp. Keep the first timestamp on the 5m grid at/after start.
    start = start.ceil("5min")
    end = end.floor("5min")
    if end < start:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(start, end, freq="5min", tz="UTC")


def build_symbol_month_eligibility(
    *,
    symbols: Sequence[str],
    phase_windows: Mapping[str, tuple[pd.Timestamp, pd.Timestamp]],
    family_contracts: Mapping[str, ContextJoinContract],
    context_root: Path | str = DEFAULT_CONTEXT_ROOT,
    capability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = Path(context_root)
    capability = capability.copy() if capability is not None else build_symbol_context_capability_manifest(symbols, context_root=root)
    capability_by_symbol = {str(r.symbol).upper(): r._asdict() for r in capability.itertuples(index=False)}
    selected_symbols = sorted({str(s).upper() for s in symbols})

    for symbol in selected_symbols:
        path = context_path(symbol, root)
        ctx = load_bybit_context(symbol, root=root, require_schema=False) if path.exists() else pd.DataFrame()
        cap = capability_by_symbol.get(symbol, {})
        schema_valid = bool(cap.get("sidecar_schema_valid", False))
        file_exists = bool(cap.get("sidecar_file_exists", False))
        instrument_status = str(cap.get("instrument_status", "unknown"))

        for phase, (phase_start_raw, phase_end_raw) in phase_windows.items():
            phase_start = pd.Timestamp(phase_start_raw).tz_convert("UTC") if pd.Timestamp(phase_start_raw).tzinfo else pd.Timestamp(phase_start_raw, tz="UTC")
            phase_end = pd.Timestamp(phase_end_raw).tz_convert("UTC") if pd.Timestamp(phase_end_raw).tzinfo else pd.Timestamp(phase_end_raw, tz="UTC")
            month_expected = {ym: expected_5m_closes_for_month(ym, phase_start, phase_end) for ym in month_range(phase_start, phase_end)}
            expected_frames = []
            for ym, idx in month_expected.items():
                if len(idx):
                    expected_frames.append(pd.DataFrame({"decision_ts": idx, "year_month": ym}))
            expected_all = pd.concat(expected_frames, ignore_index=True) if expected_frames else pd.DataFrame(columns=["decision_ts", "year_month"])

            for family, contract in family_contracts.items():
                req_groups = list(contract.required_groups)
                join_error_reason = ""
                exact_close_set: set[int] = set()
                if file_exists and not ctx.empty and schema_valid and not expected_all.empty:
                    exact_close_set = exact_required_source_close_set(ctx, req_groups)
                    if not exact_close_set:
                        join_error_reason = "missing_required_group"

                for ym, expected in month_expected.items():
                    expected_count = int(len(expected))
                    month_start = expected.min() if expected_count else pd.Timestamp(f"{ym}-01", tz="UTC")
                    month_end = expected.max() if expected_count else month_start
                    has_by_group = {
                        g: group_history_available_in_window(ctx, g, month_start, month_end) if file_exists and not ctx.empty else False
                        for g in GROUP_REQUIRED_COLUMNS
                    }
                    missing_history_groups = [g for g in req_groups if not bool(has_by_group.get(g, False))]
                    row: dict[str, Any] = {
                        "phase": phase,
                        "family": family,
                        "symbol": symbol,
                        "year_month": ym,
                        "month": ym,
                        "active_5m_universe_flag": True,
                        "required_groups": ",".join(req_groups),
                        "expected_5m_decisions": expected_count,
                        "sidecar_file_exists": file_exists,
                        "context_file_exists": file_exists,
                        "sidecar_schema_valid": schema_valid,
                        "has_mark_history": bool(has_by_group["mark"]),
                        "has_index_history": bool(has_by_group["index"]),
                        "has_premium_history": bool(has_by_group["premium"]),
                        "has_lsr_history": bool(has_by_group["lsr"]),
                        "instrument_status": instrument_status,
                        "eligibility_status": "ineligible",
                        "admissible": False,
                        "coverage_ratio": 0.0,
                        "valid_context_rows": 0,
                        "missing_required_rows": expected_count,
                        "freshness_failure_rows": 0,
                        "ineligibility_reason": "",
                        "reject_reason": "",
                    }
                    if expected_count == 0:
                        row.update({"coverage_ratio": np.nan, "missing_required_rows": 0, "reject_reason": "empty_phase_month", "ineligibility_reason": "empty_phase_month"})
                        rows.append(row)
                        continue
                    if not file_exists or ctx.empty:
                        row.update({"reject_reason": "missing_sidecar_file", "ineligibility_reason": "missing_sidecar_file"})
                        rows.append(row)
                        continue
                    if not schema_valid:
                        row.update({"reject_reason": "schema_invalid", "ineligibility_reason": "schema_invalid"})
                        rows.append(row)
                        continue
                    if missing_history_groups:
                        reason = source_unavailable_reason(missing_history_groups, instrument_status)
                        row.update({"reject_reason": reason, "ineligibility_reason": reason})
                        rows.append(row)
                        continue
                    if join_error_reason:
                        row.update({"reject_reason": join_error_reason, "ineligibility_reason": join_error_reason})
                        rows.append(row)
                        continue
                    expected_ns = pd.Series(expected).astype("int64").to_numpy()
                    valid = int(sum(int(x) in exact_close_set for x in expected_ns))
                    coverage = float(valid / expected_count) if expected_count else np.nan
                    missing = int(expected_count - valid)
                    stale = 0
                    admissible = bool(coverage >= float(contract.min_symbol_month_coverage))
                    if admissible:
                        reject = ""
                    elif stale:
                        reject = "freshness_failure"
                    elif missing:
                        reject = "missing_required_group"
                    else:
                        reject = "insufficient_coverage" if valid else (reasons[~available].mode().iloc[0] if (~available).any() else "insufficient_coverage")
                    row.update(
                        {
                            "coverage_ratio": coverage,
                            "valid_context_rows": valid,
                            "missing_required_rows": missing,
                            "freshness_failure_rows": stale,
                            "admissible": admissible,
                            "eligibility_status": "eligible" if admissible else "ineligible",
                            "ineligibility_reason": reject,
                            "reject_reason": reject,
                        }
                    )
                    for group in req_groups:
                        row[f"{group}_max_staleness_seconds"] = 0.0 if valid else np.nan
                        row[f"{group}_source_close_col"] = GROUP_SOURCE_CLOSE_COL[group]
                    rows.append(row)
    return pd.DataFrame(rows)

def write_symbol_month_eligibility(path: Path, eligibility: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    eligibility = eligibility.sort_values(["phase", "family", "symbol", "year_month"], kind="mergesort").reset_index(drop=True)
    eligibility.to_csv(path, index=False)
    return file_sha1(path)


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_sidecar_readiness(
    *,
    symbols: Sequence[str],
    context_root: Path | str = DEFAULT_CONTEXT_ROOT,
    max_symbols: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(context_root)
    selected = sorted({str(s).upper() for s in symbols})
    if max_symbols is not None and int(max_symbols) > 0:
        selected = selected[: int(max_symbols)]
    coverage_rows: list[dict[str, Any]] = []
    cadence_rows: list[dict[str, Any]] = []
    for symbol in selected:
        path = context_path(symbol, root)
        if not path.exists():
            coverage_rows.append({"symbol": symbol, "context_file_exists": False, "row_count": 0, "schema_missing": ",".join(CANONICAL_SCHEMA_COLUMNS)})
            continue
        df = pd.read_parquet(path)
        missing = [c for c in CANONICAL_SCHEMA_COLUMNS if c not in df.columns]
        for col in TIMESTAMP_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        ts = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce") if "timestamp" in df.columns else pd.Series(pd.NaT, index=df.index)
        coverage_rows.append(
            {
                "symbol": symbol,
                "context_file_exists": True,
                "row_count": int(len(df)),
                "schema_missing": ",".join(missing),
                "first_timestamp": pd.Timestamp(ts.min()).isoformat() if ts.notna().any() else "",
                "last_timestamp": pd.Timestamp(ts.max()).isoformat() if ts.notna().any() else "",
                "duplicate_timestamp_count": int(ts.duplicated().sum()) if ts.notna().any() else 0,
                "schema_hash": schema_hash(),
            }
        )
        if missing:
            continue
        for group, close_col in GROUP_SOURCE_CLOSE_COL.items():
            cols = GROUP_REQUIRED_COLUMNS[group]
            source_close = pd.to_datetime(df[close_col], utc=True, errors="coerce")
            present = df[cols].notna().all(axis=1)
            changed_ref = pd.to_numeric(df[cols[0]], errors="coerce") if cols[0] in df.columns else pd.Series(np.nan, index=df.index)
            nn = changed_ref.dropna()
            changed = nn.ne(nn.shift()).fillna(True) if len(nn) else pd.Series(dtype=bool)
            change_times = pd.Series(nn.index[changed.to_numpy()], dtype="int64") if len(nn) else pd.Series(dtype="int64")
            if len(change_times) and "timestamp" in df.columns:
                cts = pd.to_datetime(df.loc[change_times.to_numpy(), "timestamp"], utc=True, errors="coerce")
                intervals = cts.diff().dt.total_seconds().dropna() / 60.0
            else:
                intervals = pd.Series(dtype=float)
            dup_conflicts = 0
            if source_close.notna().any() and bool(source_close.duplicated().any()):
                tmp = pd.DataFrame({"source_close": source_close, "value": changed_ref})
                dup_conflicts = int(tmp.groupby("source_close")["value"].nunique(dropna=True).gt(1).sum())
            cadence_rows.append(
                {
                    "symbol": symbol,
                    "group": group,
                    "required_row_count": int(present.sum()),
                    "missing_required_share": float(1.0 - present.mean()) if len(present) else np.nan,
                    "first_source_close_ts": pd.Timestamp(source_close.min()).isoformat() if source_close.notna().any() else "",
                    "last_source_close_ts": pd.Timestamp(source_close.max()).isoformat() if source_close.notna().any() else "",
                    "median_update_interval_minutes": float(intervals.median()) if len(intervals) else np.nan,
                    "duplicate_source_close_count": int(source_close.duplicated().sum()) if source_close.notna().any() else 0,
                    "duplicate_conflicting_value_count": dup_conflicts,
                    "revision_proof_status": "not_provable_from_single_parquet_snapshot",
                }
            )
    return pd.DataFrame(coverage_rows), pd.DataFrame(cadence_rows)
