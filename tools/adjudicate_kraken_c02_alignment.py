#!/usr/bin/env python3
"""Adjudicate C02 leadership at the observed five-minute resolution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_kraken_c02_leadership_generator as c02
import build_kraken_c01_event_contract as c01


TASK_ID = "donch_bt_stage_3c_c02_alignment_adjudication_20260717_v1"
ROOT = Path("docs/agent/task_archive/20260717_donch_bt_stage_3c_c02_alignment_adjudication_20260717_v1")
STAGE3B = Path("docs/agent/task_archive/20260717_donch_bt_stage_3b_c02_leadership_generator_20260717_v1")
SOURCE_CONTRACT_HASH = "25ecea746dae447a6db3967c3183afb920aa144f8c2324c5224bc5d929a5befb"
SOURCE_EVENT_HASH = "50a6c486012c2bf4b97bcc29af69ee10b1b52f1588f7fcc129f476a628b20c3e"
SOURCE_FAILURE_HASH = "a754ef591a06534a1231a42ade895a7934bc8322172c1e1ee0796fb6ff2e32f3"
CONTRACT_VERSION = "c02_resolution_aware_v1_20260717"
BAR = pd.Timedelta(minutes=5)
RESOLUTION = pd.Timedelta(minutes=10)
PROHIBITED = ("pnl", "mae", "mfe", "expectancy", "profit", "return_after", "exit_", "control_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_crossings(
    frame: pd.DataFrame, index: int, direction: int, lookback_minutes: int
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return first observed threshold-crossing bar opens on the exact grid."""
    start = frame.at[index, "timestamp"] - pd.Timedelta(minutes=lookback_minutes)
    start_position = int(frame["timestamp"].searchsorted(start, side="left"))

    def crossing(column: str) -> pd.Timestamp | None:
        qualified = direction * frame[column] >= 1.5
        for position in range(max(1, start_position), index + 1):
            if frame.at[position, "timestamp"] - frame.at[position - 1, "timestamp"] != BAR:
                continue
            if bool(qualified.iloc[position]) and not bool(qualified.iloc[position - 1]):
                return pd.Timestamp(frame.at[position, "timestamp"])
        return None

    return crossing("spot_z_15m"), crossing("perp_z_15m")


def resolution_aware_state(
    frame: pd.DataFrame, index: int, direction: int, lookback_minutes: int
) -> tuple[str, pd.Timestamp | None, pd.Timestamp | None]:
    spot, perp = first_crossings(frame, index, direction, lookback_minutes)
    if spot is None or perp is None:
        return "coincident_or_unresolved", spot, perp
    if spot <= perp - RESOLUTION:
        return "resolved_spot_led", spot, perp
    if perp <= spot - RESOLUTION:
        return "resolved_perp_led", spot, perp
    return "coincident_or_unresolved", spot, perp


def retained_failure_state(leadership_state: str, has_completed_failure: bool) -> str:
    if leadership_state != "resolved_perp_led":
        return "not_applicable"
    return "completed_trade_and_mark" if has_completed_failure else "unconfirmed"


def transition_category(row: pd.Series) -> str:
    if not bool(row["same_episode_and_direction"]):
        return "event_disappearance_or_episode_direction_change"
    exact, shifted, shift = row["exact_leadership_state"], row["shifted_leadership_state"], int(row["spot_shift_minutes"])
    if exact == "simultaneous" and shifted == "spot_led" and shift == -5:
        return "simultaneous_to_spot_led_under_minus5m"
    if exact == "simultaneous" and shifted == "perp_led" and shift == 5:
        return "simultaneous_to_perp_led_under_plus5m"
    if {exact, shifted} == {"spot_led", "perp_led"}:
        return "genuine_leader_reversal"
    if exact == "ambiguous" or shifted in {"ambiguous", "missing"}:
        return "ambiguous_case"
    if exact == shifted:
        return "unchanged"
    return "other_state_transition"


def build_transition_matrix(alignment: pd.DataFrame) -> pd.DataFrame:
    work = alignment.copy()
    work["transition_category"] = work.apply(transition_category, axis=1)
    return (
        work.groupby(
            ["spot_shift_minutes", "exact_leadership_state", "shifted_leadership_state", "same_episode_and_direction", "transition_category"],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("event_count")
        .reset_index()
    )


def assert_safe_schema(columns: Iterable[str]) -> None:
    findings = [name for name in columns if any(token in name.lower() for token in PROHIBITED)]
    if findings:
        raise ValueError(f"prohibited outcome fields: {findings}")


def classify_events(
    frozen_events: pd.DataFrame,
    frozen_failures: pd.DataFrame,
    authority_rows: list[c01.AuthorityRow],
    normalized_spot: dict[str, dict],
) -> pd.DataFrame:
    failure_map = frozen_failures.set_index("source_event_id") if not frozen_failures.empty else pd.DataFrame()
    output: list[dict] = []
    for symbol, group in frozen_events.groupby("PF_symbol", sort=True):
        pair = str(group["Kraken_spot_pair"].iloc[0])
        spot_record = normalized_spot[pair]
        spot = c02.read_spot(Path(spot_record["path"]), pair, spot_record["sha256"])
        trade, _ = c02.read_pf_bars(authority_rows, symbol, "historical_trade_candles_5m")
        mark, _ = c02.read_pf_bars(authority_rows, symbol, "historical_mark_candles_5m")
        featured = c02.add_features(c02.align_exact(spot, trade, mark))
        position = pd.Series(featured.index.to_numpy(), index=featured["timestamp"]).to_dict()
        onset_set = set(c02.onset_indices(featured))
        for event in group.sort_values("impulse_onset_ts", kind="mergesort").itertuples(index=False):
            onset_open = pd.Timestamp(event.impulse_onset_ts) - BAR
            if onset_open not in position:
                raise ValueError(f"frozen onset absent from exact grid: {event.event_id}")
            idx = int(position[onset_open])
            if (idx, int(event.direction)) not in onset_set:
                raise ValueError(f"frozen onset no longer satisfies mechanical activation: {event.event_id}")
            old15 = c02.classify_leadership(featured, idx, int(event.direction), 15)
            old30 = c02.classify_leadership(featured, idx, int(event.direction), 30)
            if old15 != event.leadership_state or old30 != event.leadership_30m:
                raise ValueError(f"Stage 3B label reconstruction mismatch: {event.event_id}")
            state15, spot15, perp15 = resolution_aware_state(featured, idx, int(event.direction), 15)
            state30, spot30, perp30 = resolution_aware_state(featured, idx, int(event.direction), 30)
            source = event._asdict()
            source_event_id = source.pop("event_id")
            source_address = source.pop("economic_address")
            source["source_branch"] = source.pop("branch")
            source["source_failure_state"] = source.pop("failure_state")
            source["source_leadership_state"] = source.pop("leadership_state")
            source["source_leadership_30m"] = source.pop("leadership_30m")
            identity = {
                "source_event_id": source_event_id,
                "contract_version": CONTRACT_VERSION,
                "leadership_state": state15,
                "leadership_30m": state30,
            }
            event_id = "c02raevent_" + c02.stable_hash(identity)[:24]
            branch = {
                "resolved_spot_led": "resolved_spot_led_continuation",
                "resolved_perp_led": "resolved_perp_led_continuation",
                "coincident_or_unresolved": "coincident_or_unresolved_diagnostic",
            }[state15]
            confirmed = state15 == "resolved_perp_led" and source_event_id in failure_map.index
            failure = failure_map.loc[source_event_id] if confirmed else None
            output.append(
                {
                    **source,
                    "source_event_id": source_event_id,
                    "source_economic_address": source_address,
                    "event_id": event_id,
                    "economic_address": "c02raaddr_" + c02.stable_hash({**identity, "branch": branch})[:24],
                    "contract_version": CONTRACT_VERSION,
                    "leadership_state": state15,
                    "leadership_30m": state30,
                    "branch": branch,
                    "spot_crossing_bar_open_15m": spot15,
                    "perp_crossing_bar_open_15m": perp15,
                    "spot_crossing_bar_open_30m": spot30,
                    "perp_crossing_bar_open_30m": perp30,
                    "crossing_interval_minutes": 5,
                    "minimum_resolved_bar_open_separation_minutes": 10,
                    "failure_state": retained_failure_state(state15, confirmed),
                    "failure_event_id": str(failure["event_id"]) if confirmed else None,
                    "failure_decision_ts": pd.Timestamp(failure["decision_ts"]) if confirmed else pd.NaT,
                    "protected_row_count": 0,
                }
            )
    result = pd.DataFrame(output).sort_values(["impulse_onset_ts", "PF_symbol", "event_id"], kind="mergesort").reset_index(drop=True)
    assert_safe_schema(result.columns)
    if result["event_id"].duplicated().any() or result["economic_address"].duplicated().any():
        raise ValueError("resolution-aware identity collision")
    if len(result) != len(frozen_events) or set(result.source_event_id) != set(frozen_events.event_id):
        raise ValueError("frozen event population reconciliation failed")
    return result


def count_matrix(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(
            [events.impulse_onset_ts.dt.year.rename("year"), "asset_group", "PF_symbol", "direction_label", "leadership_state", "leadership_30m", "failure_state"],
            dropna=False,
            sort=True,
        )
        .size()
        .rename("event_count")
        .reset_index()
    )


def agreement_report(events: pd.DataFrame) -> pd.DataFrame:
    resolved = events[events.leadership_state.isin(["resolved_spot_led", "resolved_perp_led"])].copy()
    rows: list[dict] = []
    for state in ("resolved_spot_led", "resolved_perp_led"):
        for direction in ("all", "positive", "negative"):
            part = resolved[resolved.leadership_state.eq(state)]
            if direction != "all":
                part = part[part.direction_label.eq(direction)]
            years = part.impulse_onset_ts.dt.year.value_counts().to_dict()
            same = part.leadership_30m.eq(state)
            total = len(part)
            rate = float(same.mean()) if total else 0.0
            rows.append(
                {
                    "primary_state": state,
                    "direction": direction,
                    "primary_resolved_events": total,
                    "events_2023": int(years.get(2023, 0)),
                    "events_2024": int(years.get(2024, 0)),
                    "events_2025": int(years.get(2025, 0)),
                    "same_leader_30m_events": int(same.sum()),
                    "same_leader_30m_rate": rate,
                    "passes_total_100": total >= 100,
                    "passes_each_year_20": all(int(years.get(year, 0)) >= 20 for year in (2023, 2024, 2025)),
                    "passes_30m_agreement_80pct": rate >= 0.80,
                    "mechanically_sufficient": total >= 100 and all(int(years.get(year, 0)) >= 20 for year in (2023, 2024, 2025)) and rate >= 0.80,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--pf-manifest", type=Path, default=Path("/opt/parquet/kraken_derivatives/manifests/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815_download_manifest.csv"))
    parser.add_argument("--spot-manifest", type=Path, default=Path("docs/agent/task_archive/20260717_donch_bt_stage_3a_c02_spot_reference_20260717_v1/C02_SPOT_DATA_MANIFEST.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    contract = STAGE3B / "C02_GENERATOR_CONTRACT.md"
    event_path = STAGE3B / "C02_IMPULSE_EVENT_TAPE.parquet"
    failure_path = STAGE3B / "C02_FAILURE_EVENT_TAPE.parquet"
    if sha256(contract) != SOURCE_CONTRACT_HASH or sha256(event_path) != SOURCE_EVENT_HASH or sha256(failure_path) != SOURCE_FAILURE_HASH:
        raise ValueError("Stage 3B source hash mismatch")
    spot_manifest = json.loads(args.spot_manifest.read_text(encoding="utf-8"))
    if spot_manifest.get("manifest_content_hash") != c02.SPOT_MANIFEST_HASH or spot_manifest.get("protected_rows_opened") != 0:
        raise ValueError("Stage 3A spot authority mismatch")
    frozen_events = pd.read_parquet(event_path)
    frozen_failures = pd.read_parquet(failure_path)
    assert_safe_schema(frozen_events.columns)
    assert_safe_schema(frozen_failures.columns)
    if int(frozen_events.protected_row_count.sum()) != 0 or (pd.to_datetime(frozen_events.decision_ts, utc=True) >= c02.PROTECTED_START).any():
        raise ValueError("protected Stage 3B event")
    frozen_events["impulse_onset_ts"] = pd.to_datetime(frozen_events.impulse_onset_ts, utc=True)
    alignment = pd.read_csv(STAGE3B / "C02_ALIGNMENT_SENSITIVITY.csv")
    build_transition_matrix(alignment).to_csv(args.output / "C02_ORIGINAL_ALIGNMENT_TRANSITION_MATRIX.csv", index=False)
    authority_rows = c01.load_safe_manifest(args.pf_manifest)
    normalized = {Path(row["path"]).stem: row for row in spot_manifest["normalized_files"]}
    resolved = classify_events(frozen_events, frozen_failures, authority_rows, normalized)
    resolved.to_parquet(args.output / "C02_RESOLUTION_AWARE_EVENT_TAPE.parquet", index=False)
    count_matrix(resolved).to_csv(args.output / "C02_RESOLUTION_AWARE_COUNT_MATRIX.csv", index=False)
    agreement_report(resolved).to_csv(args.output / "C02_15M_30M_AGREEMENT.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
