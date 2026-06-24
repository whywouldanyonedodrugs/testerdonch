from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventClusterSpec:
    default_trigger_hours: int = 24
    trigger_hours_by_family: Mapping[str, int] | None = None

    def hours_for_family(self, family: str) -> int:
        return int((self.trigger_hours_by_family or {}).get(str(family), self.default_trigger_hours))


def assign_event_cluster_ids(entries: pd.DataFrame, spec: EventClusterSpec | None = None) -> pd.DataFrame:
    if entries.empty:
        out = entries.copy()
        out["event_cluster_id"] = pd.Series(dtype=str)
        out["event_cluster_start_ts"] = pd.Series(dtype="datetime64[ns, UTC]")
        out["event_cluster_end_ts"] = pd.Series(dtype="datetime64[ns, UTC]")
        out["event_trigger_window_hours"] = pd.Series(dtype=float)
        return out
    spec = spec or EventClusterSpec()
    required = {"symbol", "family", "decision_ts"}
    missing = required.difference(entries.columns)
    if missing:
        raise KeyError(f"entries missing required columns for event cluster IDs: {sorted(missing)}")
    out = entries.copy()
    out["decision_ts"] = pd.to_datetime(out["decision_ts"], utc=True, errors="coerce")
    out = out.dropna(subset=["decision_ts", "symbol", "family"]).sort_values(["symbol", "family", "decision_ts", "entry_id" if "entry_id" in out.columns else "decision_ts"], kind="mergesort")
    ids: list[str] = []
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    hours_out: list[int] = []
    current_key: tuple[str, str] | None = None
    cluster_start: pd.Timestamp | None = None
    cluster_end: pd.Timestamp | None = None
    cluster_seq = 0
    current_cluster_id = ""
    for row in out.itertuples(index=False):
        symbol = str(getattr(row, "symbol"))
        family = str(getattr(row, "family"))
        decision = pd.Timestamp(getattr(row, "decision_ts"))
        hours = spec.hours_for_family(family)
        trigger_start = decision - pd.Timedelta(hours=hours)
        trigger_end = decision
        key = (symbol, family)
        if current_key != key or cluster_end is None or trigger_start > cluster_end:
            current_key = key
            cluster_start = trigger_start
            cluster_end = trigger_end
            cluster_seq += 1
            current_cluster_id = _cluster_hash(symbol, family, cluster_start, cluster_seq)
        else:
            cluster_end = max(cluster_end, trigger_end)
            cluster_start = min(cluster_start, trigger_start) if cluster_start is not None else trigger_start
        ids.append(current_cluster_id)
        starts.append(cluster_start)
        ends.append(cluster_end)
        hours_out.append(hours)
    out["event_cluster_id"] = ids
    out["event_cluster_start_ts"] = starts
    out["event_cluster_end_ts"] = ends
    out["event_trigger_window_hours"] = hours_out
    return out.sort_index(kind="mergesort")


def _cluster_hash(symbol: str, family: str, cluster_start: pd.Timestamp, seq: int) -> str:
    raw = f"{symbol}|{family}|{pd.Timestamp(cluster_start).isoformat()}|{int(seq)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def concentration_summary(
    ledger: pd.DataFrame,
    path_stats: pd.DataFrame,
    *,
    return_col: str = "fwd_ret_close_72h",
    cost_col: str | None = None,
) -> pd.DataFrame:
    if ledger.empty or path_stats.empty:
        return pd.DataFrame()
    needed = ["entry_id", "symbol", "decision_ts", "family", "event_cluster_id"]
    missing = [c for c in needed if c not in ledger.columns]
    if missing:
        raise KeyError(f"ledger missing concentration columns: {missing}")
    if return_col not in path_stats.columns:
        raise KeyError(f"path stats missing {return_col!r}")
    left = ledger[needed].drop_duplicates("entry_id").copy()
    right = path_stats[["entry_id", return_col]].copy()
    df = left.merge(right, on="entry_id", how="inner")
    df["decision_ts"] = pd.to_datetime(df["decision_ts"], utc=True, errors="coerce")
    df["year_month"] = df["decision_ts"].dt.strftime("%Y-%m")
    ret = pd.to_numeric(df[return_col], errors="coerce")
    if cost_col and cost_col in path_stats.columns:
        ret = ret - pd.to_numeric(path_stats.set_index("entry_id").reindex(df["entry_id"])[cost_col].to_numpy(), errors="coerce")
    df["gross_positive_return"] = ret.clip(lower=0.0).fillna(0.0)
    rows: list[dict[str, object]] = []
    for family, fam_df in df.groupby("family", dropna=False):
        for dim, col in [("symbol", "symbol"), ("month", "year_month"), ("event_cluster", "event_cluster_id")]:
            entries = fam_df[col].value_counts(dropna=True)
            gpnl = fam_df.groupby(col, dropna=True)["gross_positive_return"].sum().sort_values(ascending=False)
            gpnl_sum = float(gpnl.sum())
            rows.append(
                {
                    "family": family,
                    "dimension": dim,
                    "bucket_count": int(entries.size),
                    "max_entry_share": float(entries.iloc[0] / entries.sum()) if int(entries.sum()) else np.nan,
                    "effective_entry_count": _effective_count(entries.astype(float) / float(entries.sum())) if int(entries.sum()) else np.nan,
                    "max_gross_positive_return_share": float(gpnl.iloc[0] / gpnl_sum) if gpnl_sum > 0 and len(gpnl) else np.nan,
                    "effective_gross_positive_return_count": _effective_count(gpnl / gpnl_sum) if gpnl_sum > 0 and len(gpnl) else np.nan,
                    "top_bucket": str(entries.index[0]) if len(entries) else "",
                }
            )
    return pd.DataFrame(rows)


def leave_one_group_out_summary(
    ledger: pd.DataFrame,
    path_stats: pd.DataFrame,
    *,
    group_col: str,
    return_col: str = "fwd_ret_close_72h",
) -> pd.DataFrame:
    if ledger.empty or path_stats.empty:
        return pd.DataFrame()
    if group_col not in ledger.columns:
        raise KeyError(f"ledger missing {group_col!r}")
    df = ledger[["entry_id", "family", group_col]].drop_duplicates("entry_id").merge(path_stats[["entry_id", return_col]], on="entry_id", how="inner")
    rows: list[dict[str, object]] = []
    for (family, held_out), _ in df.groupby(["family", group_col], dropna=True):
        keep = df[(df["family"] == family) & (df[group_col] != held_out)]
        vals = pd.to_numeric(keep[return_col], errors="coerce")
        rows.append(
            {
                "family": family,
                "held_out_group": held_out,
                "group_col": group_col,
                "remaining_entries": int(vals.notna().sum()),
                "remaining_mean_return": float(vals.mean()) if vals.notna().any() else np.nan,
                "remaining_median_return": float(vals.median()) if vals.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _effective_count(shares: pd.Series) -> float:
    x = pd.to_numeric(shares, errors="coerce").dropna()
    denom = float((x * x).sum())
    return float(1.0 / denom) if denom > 0 else np.nan
