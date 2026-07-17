#!/usr/bin/env python3
"""Freeze the outcome-free C02 positive spot-led Level-3 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK_ID = "donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1"
LINEAGE_ID = "C02_positive_resolution_aware_spot_led_continuation_v1"
ROOT = Path("docs/agent/task_archive/20260717_donch_bt_stage_3d_c02_positive_spot_led_prerun_20260717_v1")
STAGE3C = Path("docs/agent/task_archive/20260717_donch_bt_stage_3c_c02_alignment_adjudication_20260717_v1")
EVENT_TAPE = STAGE3C / "C02_RESOLUTION_AWARE_EVENT_TAPE.parquet"
EVENT_TAPE_HASH = "c73344b1bd104c0816d731a1002f729b49100385a34b9b56ec4b2be66dad71ad"
RESOLUTION_CONTRACT_HASH = "ce65c62edfb80f5fb83e9b8b6bae1d3eb9c981f8e9a1bcad3b285fdce46cca51"
PROTECTED_START = pd.Timestamp("2026-01-01T00:00:00Z")
BOOTSTRAP_SEED = 20260717
SAFE_EVENT_COLUMNS = [
    "event_id", "economic_address", "source_event_id", "PF_symbol", "canonical_asset_id",
    "Kraken_spot_pair", "direction_label", "leadership_state", "leadership_30m",
    "leadership_lookback", "decision_ts", "impulse_onset_ts", "canonical_episode_id",
    "spot_z_15m", "perp_z_15m", "prior_day_pf_liquidity_rank", "feature_available_ts",
    "contract_version", "protected_row_count",
]
FORBIDDEN_TOKENS = ("pnl", "mae", "mfe", "profit", "exit_price", "entry_price", "net_bps", "gross_bps", "control_outcome")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def assert_no_outcome_columns(columns: Iterable[str]) -> None:
    bad = [column for column in columns if any(token in column.lower() for token in FORBIDDEN_TOKENS)]
    if bad:
        raise ValueError(f"outcome columns prohibited in pre-run freeze: {bad}")


def load_safe_stage3c_events(path: Path = EVENT_TAPE) -> pd.DataFrame:
    if sha256(path) != EVENT_TAPE_HASH:
        raise ValueError("Stage 3C event tape hash mismatch")
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    if not set(SAFE_EVENT_COLUMNS).issubset(schema):
        raise ValueError("Stage 3C safe event schema incomplete")
    assert_no_outcome_columns(SAFE_EVENT_COLUMNS)
    events = pd.read_parquet(path, columns=SAFE_EVENT_COLUMNS)
    for column in ("decision_ts", "impulse_onset_ts", "feature_available_ts"):
        events[column] = pd.to_datetime(events[column], utc=True, errors="raise")
    if events.protected_row_count.fillna(0).astype(int).sum() or (events.decision_ts >= PROTECTED_START).any():
        raise ValueError("protected event reached pre-run freeze")
    if not events.Kraken_spot_pair.str.endswith("USD").all() or not events.PF_symbol.str.startswith("PF_").all():
        raise ValueError("non-Kraken identity reached pre-run freeze")
    if (events.feature_available_ts > events.decision_ts).any():
        raise ValueError("feature availability after decision")
    return events


def select_event_sets(events: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    primary = events[
        events.direction_label.eq("positive")
        & events.leadership_state.eq("resolved_spot_led")
        & events.leadership_lookback.eq("15m_primary")
    ].copy()
    primary = primary.sort_values(["decision_ts", "PF_symbol", "event_id"], kind="mergesort").reset_index(drop=True)
    primary["in_30m_agreement_subset"] = primary.leadership_30m.eq("resolved_spot_led")
    if len(primary) != 489 or int(primary.in_30m_agreement_subset.sum()) != 425:
        raise ValueError("frozen C02 event-set count mismatch")
    if primary.event_id.duplicated().any() or primary.economic_address.duplicated().any():
        raise ValueError("duplicate C02 event identity")
    primary_hash = canonical_hash(primary.event_id.tolist())
    robustness_hash = canonical_hash(primary.loc[primary.in_30m_agreement_subset, "event_id"].tolist())
    primary["lineage_id"] = LINEAGE_ID
    primary["primary_event_set_hash"] = primary_hash
    primary["robustness_event_set_hash"] = robustness_hash
    return primary, primary_hash, robustness_hash


def definition_register(primary_hash: str, robustness_hash: str) -> pd.DataFrame:
    specifications = [
        ("c02_l3_primary_all_1h", "primary_all", 1, "primary"),
        ("c02_l3_primary_all_6h", "primary_all", 6, "primary"),
        ("c02_l3_30m_agreement_1h", "30m_agreement", 1, "robustness_only"),
        ("c02_l3_30m_agreement_6h", "30m_agreement", 6, "robustness_only"),
    ]
    rows = []
    for definition_id, event_set, horizon, role in specifications:
        policy = {
            "definition_id": definition_id,
            "lineage_id": LINEAGE_ID,
            "event_set": event_set,
            "event_set_hash": primary_hash if event_set == "primary_all" else robustness_hash,
            "timeout_hours": horizon,
            "entry": "first_executable_PF_5m_open_strictly_after_decision",
            "exit": "first_executable_PF_5m_open_at_or_after_entry_plus_timeout",
            "role": role,
        }
        rows.append({**policy, "definition_hash": canonical_hash(policy), "can_earn_level3_permission": role == "primary"})
    result = pd.DataFrame(rows)
    if len(result) != 4 or result.definition_id.duplicated().any():
        raise ValueError("definition freeze failed")
    return result


def executable_interval(bar_opens: pd.DatetimeIndex, decision_ts: pd.Timestamp, timeout_hours: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    bars = pd.DatetimeIndex(pd.to_datetime(bar_opens, utc=True)).sort_values().unique()
    decision = pd.Timestamp(decision_ts)
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    else:
        decision = decision.tz_convert("UTC")
    entries = bars[bars > decision]
    if not len(entries):
        raise ValueError("missing next-open entry")
    entry = pd.Timestamp(entries[0])
    exits = bars[bars >= entry + pd.Timedelta(hours=timeout_hours)]
    if not len(exits):
        raise ValueError("missing timeout exit")
    exit_ts = pd.Timestamp(exits[0])
    if entry >= PROTECTED_START or exit_ts >= PROTECTED_START:
        raise ValueError("protected/sample-boundary crossing")
    return entry, exit_ts


def definition_local_nonoverlap(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted, skipped = [], []
    for symbol, group in trades.sort_values(["PF_symbol", "entry_ts", "event_id"], kind="mergesort").groupby("PF_symbol", sort=True):
        open_until = None
        prior_event = None
        for row in group.itertuples(index=False):
            if open_until is not None and pd.Timestamp(row.entry_ts) < open_until:
                skipped.append({**row._asdict(), "skip_reason": "actual_timeout_position_open", "blocking_event_id": prior_event})
                continue
            accepted.append(row._asdict())
            open_until = pd.Timestamp(row.exit_ts)
            prior_event = row.event_id
    return pd.DataFrame(accepted), pd.DataFrame(skipped)


def fixed_notional_bps(entry_price: float, exit_price: float) -> dict[str, float]:
    if not np.isfinite(entry_price) or not np.isfinite(exit_price) or entry_price <= 0 or exit_price <= 0:
        raise ValueError("invalid execution price")
    gross = 10_000.0 * (exit_price / entry_price - 1.0)
    return {"gross_bps": gross, "base_net_bps_ex_funding": gross - 14.0, "stress_net_bps_ex_funding": gross - 32.0}


def funding_partition(exact_boundaries: int, imputed_boundaries: int) -> str:
    if exact_boundaries < 0 or imputed_boundaries < 0:
        raise ValueError("negative funding boundary count")
    total = exact_boundaries + imputed_boundaries
    if total == 0:
        return "zero_boundary"
    if exact_boundaries == total:
        return "fully_exact_funded"
    if imputed_boundaries == total:
        return "fully_imputed"
    return "mixed"


def episode_bootstrap_ci(values: np.ndarray, episodes: np.ndarray, *, resamples: int = 10_000, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    episodes = np.asarray(episodes)
    unique = np.unique(episodes)
    if not len(values) or len(values) != len(episodes) or not len(unique):
        raise ValueError("invalid bootstrap inputs")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    by_episode = {episode: values[episodes == episode] for episode in unique}
    for index in range(resamples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        means[index] = np.concatenate([by_episode[episode] for episode in sampled]).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def concentration_metrics(trades: pd.DataFrame) -> dict[str, float]:
    total = float(trades.base_net_bps_ex_funding.sum())
    if total <= 0:
        raise ValueError("non-positive concentration denominator")
    symbol = trades.groupby("PF_symbol").base_net_bps_ex_funding.sum()
    episode = trades.groupby("canonical_episode_id").base_net_bps_ex_funding.sum()
    year = trades.groupby("year").base_net_bps_ex_funding.sum()
    positive_year = year[year > 0]
    if positive_year.sum() <= 0:
        raise ValueError("non-positive positive-year denominator")
    return {
        "max_positive_symbol_share": float(symbol.clip(lower=0).max() / total),
        "max_positive_episode_share": float(episode.clip(lower=0).max() / total),
        "max_positive_year_share": float(positive_year.max() / positive_year.sum()),
    }


def level3_gate_result(trades: pd.DataFrame, bootstrap_lower_bps: float) -> dict[str, bool]:
    years = trades.year.value_counts().to_dict()
    concentration = concentration_metrics(trades)
    gates = {
        "executed_trades_ge_100": len(trades) >= 100,
        "each_year_ge_20": all(int(years.get(year, 0)) >= 20 for year in (2023, 2024, 2025)),
        "mean_base_net_positive": float(trades.base_net_bps_ex_funding.mean()) > 0,
        "median_base_net_positive": float(trades.base_net_bps_ex_funding.median()) > 0,
        "bootstrap_lower_ge_minus5": bootstrap_lower_bps >= -5,
        "symbol_share_le_25pct": concentration["max_positive_symbol_share"] <= 0.25,
        "episode_share_le_10pct": concentration["max_positive_episode_share"] <= 0.10,
        "year_share_le_70pct": concentration["max_positive_year_share"] <= 0.70,
        "stress_mean_ge_minus10": float(trades.stress_net_bps_ex_funding.mean()) >= -10,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def primary_permission(primary_results: dict[str, dict[str, bool]], robustness_results: dict[str, dict[str, bool]]) -> bool:
    del robustness_results
    return any(result.get("all_pass", False) for result in primary_results.values())


def match_leadership_control(treated: pd.Series, pool: pd.DataFrame) -> pd.Series | None:
    candidates = pool[
        pool.PF_symbol.eq(treated.PF_symbol)
        & pool.year.eq(treated.year)
        & pool.spot_z_15m.sub(treated.spot_z_15m).abs().le(0.5)
        & pool.perp_z_15m.sub(treated.perp_z_15m).abs().le(0.5)
        & pool.prior_day_pf_liquidity_rank.sub(treated.prior_day_pf_liquidity_rank).abs().le(10)
        & pool.lagged_pf_vol_24h.div(treated.lagged_pf_vol_24h).sub(1).abs().le(0.20)
        & pool.canonical_episode_id.ne(treated.canonical_episode_id)
        & pool.decision_ts.sub(treated.decision_ts).abs().ge(pd.Timedelta(hours=24))
    ].copy()
    if candidates.empty:
        return None
    candidates["distance"] = candidates.decision_ts.sub(treated.decision_ts).abs()
    return candidates.sort_values(["distance", "decision_ts"], kind="mergesort").iloc[0]


def write_contracts(output: Path, primary_hash: str, robustness_hash: str) -> str:
    contract = f"""# C02 Final Level-3 Economic Contract

Lineage: `{LINEAGE_ID}`.

This document freezes a later, separately authorized test of positive resolved spot-led continuation only. Stage 3B one-bar alignment remains failed. Negative spot-led, perp-led, completed failure, shifted clocks, alternate thresholds, and alternate horizons are excluded.

Primary identities: `{primary_hash}` (489 source events). Robustness identities: `{robustness_hash}` (425 source events).

Definitions are exactly `c02_l3_primary_all_1h`, `c02_l3_primary_all_6h`, `c02_l3_30m_agreement_1h`, and `c02_l3_30m_agreement_6h`. Agreement definitions are robustness-only and cannot rescue primary failure.

Decision is the Stage 3C onset-bar availability time. Entry is the first executable Kraken PF five-minute trade-bar open strictly after decision. Exit is the first executable PF five-minute trade-bar open at or after entry plus one or six hours. Exposure is fixed notional; exits are timeout-only. Each definition applies symbol-local non-overlap using its actual timeout exit.

Base cost is 5 bps taker per side plus 4 bps round-trip slippage (14 bps total). Stress cost is 10 bps per side plus 12 bps round-trip slippage (32 bps total). Primary gates use base net bps excluding funding. Funding is partitioned separately as fully exact, mixed, fully imputed, or zero-boundary and cannot rescue a definition.

The frozen Level-3 gates and seed are machine-readable in `C02_LEVEL3_DECISION_RULES.json`. Passing permits later controls only; it is not validation or promotion. No economic run is authorized by this contract freeze.
"""
    path = output / "C02_FINAL_LEVEL3_ECONOMIC_CONTRACT.md"
    path.write_text(contract, encoding="utf-8")
    return sha256(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if sha256(STAGE3C / "C02_RESOLUTION_AWARE_GENERATOR_CONTRACT.md") != RESOLUTION_CONTRACT_HASH:
        raise ValueError("Stage 3C contract hash mismatch")
    events, primary_hash, robustness_hash = select_event_sets(load_safe_stage3c_events())
    events.to_csv(args.output / "C02_POSITIVE_SPOT_LED_EVENT_SET.csv", index=False)
    definitions = definition_register(primary_hash, robustness_hash)
    definitions.to_csv(args.output / "C02_LEVEL3_DEFINITION_REGISTER.csv", index=False)
    rules = {
        "lineage_id": LINEAGE_ID,
        "primary_event_set_hash": primary_hash,
        "robustness_event_set_hash": robustness_hash,
        "bootstrap": {"unit": "canonical_episode", "resamples": 10000, "seed": BOOTSTRAP_SEED, "ci": 0.95},
        "costs_bps": {"base": {"taker_per_side": 5, "round_trip_slippage": 4}, "stress": {"taker_per_side": 10, "round_trip_slippage": 12}},
        "funding_partitions": ["fully_exact_funded", "mixed", "fully_imputed", "zero_boundary"],
        "gates": {"executed_trades_min": 100, "each_year_min": 20, "mean_base_net_bps_gt": 0, "median_base_net_bps_gt": 0,
                  "bootstrap_ci_lower_bps_gte": -5, "max_positive_symbol_share_lte": .25, "max_positive_episode_share_lte": .10,
                  "max_positive_year_share_lte": .70, "stress_mean_bps_gte": -10},
        "primary_only_permission": True,
        "robustness_cannot_rescue": True,
    }
    (args.output / "C02_LEVEL3_DECISION_RULES.json").write_text(json.dumps(rules, indent=2, sort_keys=True) + "\n")
    contract_hash = write_contracts(args.output, primary_hash, robustness_hash)
    control = """# C02 Frozen Level-4 Control Contract

Pre-registered only; execution requires separate approval and a primary Level-3 all-pass result.

Leadership control selects at most one positive coincident/unresolved event with the same PF symbol and calendar year, spot/perp z within 0.5, prior-day rank within 10, causal lagged PF 24h volatility within 20%, a different canonical episode, and at least 24h onset separation. Choose nearest timestamp and break ties by timestamp. Calipers never widen.

Leadership ablation uses the same positive confirmed-impulse generator and execution rules without resolved leadership. Measurement robustness is the frozen 30m-agreement subset and cannot substitute for the primary set. Control identities must freeze before outcomes.
"""
    (args.output / "C02_LEVEL4_CONTROL_CONTRACT.md").write_text(control, encoding="utf-8")
    multiplicity = f"""# C02 Multiplicity and Lineage Record

- Distinct lineage: `{LINEAGE_ID}`.
- Frozen source population: 489 primary events; 425 in the robustness subset.
- Definitions: four total; two primary horizons and two robustness-only expressions.
- Search dimensions added here: none beyond the predeclared 1h/6h horizons and frozen 30m subset.
- Excluded lineages: negative spot-led, perp-led, failure, shifted-clock, alternate-threshold, and alternate-horizon definitions.
- Stage 3B alignment remains failed; Stage 3C resolution-aware spot-led feasibility remains mechanical only.
"""
    (args.output / "C02_MULTIPLICITY_AND_LINEAGE_RECORD.md").write_text(multiplicity, encoding="utf-8")
    packet = f"""# C02 Pre-Run Approval Packet

Status: `ready_for_human_C02_Level3_run_approval`.

- Final contract SHA-256: `{contract_hash}`.
- Stage 3C resolution contract: `{RESOLUTION_CONTRACT_HASH}`.
- Stage 3C event tape: `{EVENT_TAPE_HASH}`.
- Stage 3B source contract: `25ecea746dae447a6db3967c3183afb920aa144f8c2324c5224bc5d929a5befb`.
- Spot manifest: `3de3b533a390f04590ae458ac661d4fc10d299df1a1833734911fa609b0a7046`.
- Stage 2C cohort: `768b09c731a728e31ce1d882862878c698cbf19e6883b1d0fe02505edb619f15`.

Proposed later command interface (runner intentionally not implemented):

`./.venv/bin/python tools/run_kraken_c02_level3_economic.py --contract {ROOT}/C02_FINAL_LEVEL3_ECONOMIC_CONTRACT.md --definitions {ROOT}/C02_LEVEL3_DEFINITION_REGISTER.csv --event-set {ROOT}/C02_POSITIVE_SPOT_LED_EVENT_SET.csv --output-root results/rebaseline/phase_kraken_c02_positive_spot_led_level3_<UTC_SUFFIX>`

Rollback: revert later task commits normally and preserve every result root. Forbidden: protected data, branch expansion, alternate thresholds/horizons, negative/perp/failure branches, controls without a primary all-pass result and human approval, validation, portfolio, or live work.
"""
    (args.output / "C02_PRERUN_APPROVAL_PACKET.md").write_text(packet, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
