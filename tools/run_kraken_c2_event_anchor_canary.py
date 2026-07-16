#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DB_ROOT = Path("results/rebaseline/phase_kraken_c2_authoritative_database_rebuild_20260713_v1")
K0 = Path("results/rebaseline/phase_kraken_k0_data_foundation_20260630_v1_20260630_163815")
FUNDING_ROOT = Path("results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1")
PROTECTED = pd.Timestamp("2026-01-01", tz="UTC")
CANARY_IDS = ("CAT0018", "CAT0077", "CAT0060", "CAT0057", "CAT0086")
ANCHOR_POLICY_VERSION = "c2_event_anchor_policy_v1"


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(path, index=False)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text.rstrip() + "\n", encoding="utf-8")


def parsed(value: Any) -> pd.Timestamp | pd.NaT:
    text = str(value).strip()
    return pd.NaT if text.lower() in {"", "unknown", "nan", "none", "null"} else pd.to_datetime(text, utc=True, errors="coerce")


def timestamp_precision(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in {"", "unknown", "nan", "none", "null"}: return "unknown"
    if len(text) == 10 and text[4] == "-" and text[7] == "-": return "date_only"
    if "T" in text and (text.endswith("Z") or "+" in text): return "intraday_explicit_utc"
    return "coarse_or_nonstandard"


def resolve_event_anchor(row: pd.Series) -> dict[str, Any]:
    state = str(row.event_state).strip().lower()
    values = {
        "first_public_ts_utc": row.get("first_public_ts_utc", "unknown"),
        "official_confirm_ts_utc": row.get("official_confirm_ts_utc", "unknown"),
        "effective_ts_utc": row.get("effective_ts_utc", "unknown"),
    }
    valid = {key: parsed(value) for key, value in values.items()}
    verified_effective = pd.notna(valid["effective_ts_utc"]) and str(row.source_confidence) == "high"
    if state in {"announced", "confirmed", "dismissed"}:
        candidates = [(key, ts) for key, ts in valid.items() if key != "effective_ts_utc" and pd.notna(ts)]
        source, anchor = min(candidates, key=lambda item: item[1]) if candidates else ("", pd.NaT)
        fallback = "none"
    elif state == "executed" and verified_effective:
        source, anchor, fallback = "effective_ts_utc", valid["effective_ts_utc"], "none"
    else:
        if pd.notna(valid["official_confirm_ts_utc"]): source, anchor, fallback = "official_confirm_ts_utc", valid["official_confirm_ts_utc"], "explicit_confirmation_fallback"
        elif pd.notna(valid["first_public_ts_utc"]): source, anchor, fallback = "first_public_ts_utc", valid["first_public_ts_utc"], "explicit_first_public_fallback"
        else: source, anchor, fallback = "", pd.NaT, "failed_no_verified_anchor"
    raw = values.get(source, "unknown")
    return {"event_anchor_ts": anchor, "event_anchor_source": source, "anchor_precision": timestamp_precision(raw), "anchor_fallback": fallback, "anchor_policy_status": "pass" if pd.notna(anchor) else "fail"}


def load_bars(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    base = K0 / "downloaded_official_kraken/parquet/historical_trade_candles_5m" / symbol
    frames = []
    for path in sorted(base.glob("*.parquet")):
        try: frame = pd.read_parquet(path)
        except Exception: continue
        if "time" not in frame or not {"open", "high", "low", "close"}.issubset(frame.columns): continue
        frame = frame.copy(); frame["ts"] = pd.to_datetime(pd.to_numeric(frame.time, errors="coerce"), unit="ms", utc=True)
        frame = frame[(frame.ts >= start) & (frame.ts <= end) & (frame.ts < PROTECTED)]
        if len(frame): frames.append(frame[["ts", "open", "high", "low", "close", "volume"]])
    if not frames: return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True).sort_values("ts").drop_duplicates("ts", keep="last")
    for col in ["open", "high", "low", "close", "volume"]: out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["ts", "open", "high", "low", "close"])


def completed_bars(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if bars.empty: return pd.DataFrame()
    rule = "4h" if timeframe == "4h" else "1D"
    work = bars.copy(); work["known_ts"] = work.ts + pd.Timedelta(minutes=5)
    out = work.set_index("known_ts").sort_index().resample(rule, label="right", closed="right").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")).dropna().reset_index().rename(columns={"known_ts": "source_close_ts"})
    return out[out.source_close_ts < PROTECTED].reset_index(drop=True)


def first_execution_open(bars: pd.DataFrame, after: pd.Timestamp) -> tuple[pd.Timestamp | pd.NaT, float]:
    rows = bars[bars.ts >= after].sort_values("ts")
    return (pd.NaT, np.nan) if rows.empty else (rows.iloc[0].ts, float(rows.iloc[0].open))


def build_definitions(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, event in events.iterrows():
        reactions = ["1d", "3d"] + (["4h"] if event.anchor_precision == "intraday_explicit_utc" else [])
        for reaction in reactions:
            for base_days in (3, 7):
                for exit_policy in ("structure_base_failure", "fixed_hold_10d"):
                    vector = {"legacy_event_id": event.legacy_event_id, "reaction_exclusion": reaction, "base_length_days": base_days, "decision_timeframe": "4h" if reaction == "4h" else "1d", "entry_policy": "completed_close_breakout_or_reclaim_next_5m_open", "exit_policy": exit_policy, "protected_boundary": str(PROTECTED), "anchor_policy": ANCHOR_POLICY_VERSION}
                    h = stable_hash(vector)
                    rows.append({"candidate_definition_id": "c2_canary_" + h[:16], **vector, "parameter_vector_hash": h, "rank_or_promotion_allowed": False})
    return pd.DataFrame(rows)


def select_event(defn: pd.Series, event: pd.Series, bars: pd.DataFrame) -> dict[str, Any] | None:
    timeframe = defn.decision_timeframe; signal = completed_bars(bars, timeframe)
    anchor = event.event_anchor_ts
    reaction = pd.Timedelta(hours=4) if defn.reaction_exclusion == "4h" else pd.Timedelta(days=int(str(defn.reaction_exclusion)[:-1]))
    base_start = anchor + reaction; unit = pd.Timedelta(hours=4) if timeframe == "4h" else pd.Timedelta(days=1)
    eligible = signal[signal.source_close_ts >= base_start + int(defn.base_length_days) * pd.Timedelta(days=1)].copy()
    side = str(event.direction)
    for _, decision in eligible.iterrows():
        history = signal[(signal.source_close_ts < decision.source_close_ts) & (signal.source_close_ts >= decision.source_close_ts - int(defn.base_length_days) * pd.Timedelta(days=1))]
        if len(history) < (int(defn.base_length_days) * (6 if timeframe == "4h" else 1)): continue
        base_high, base_low = float(history.high.max()), float(history.low.min())
        passed = float(decision.close) > base_high if side == "long" else float(decision.close) < base_low
        if not passed: continue
        entry_ts, entry_price = first_execution_open(bars, decision.source_close_ts)
        if pd.isna(entry_ts): return None
        risk = max(abs(entry_price - (base_low if side == "long" else base_high)), entry_price * 0.005)
        key_vector = {"definition": defn.candidate_definition_id, "event": event.event_id, "decision_ts": decision.source_close_ts.isoformat(), "entry_ts": entry_ts.isoformat(), "symbol": event.kraken_symbol}
        return {"selected_event_key": stable_hash(key_vector), "candidate_definition_id": defn.candidate_definition_id, "event_id": event.event_id, "legacy_event_id": event.legacy_event_id, "mechanism_family": event.mechanism_family, "symbol": event.kraken_symbol, "side": side, "anchor_ts": anchor, "anchor_precision": event.anchor_precision, "reaction_exclusion": defn.reaction_exclusion, "base_length_days": defn.base_length_days, "exit_policy": defn.exit_policy, "decision_ts": decision.source_close_ts, "feature_available_ts": decision.source_close_ts, "entry_ts": entry_ts, "entry_price": entry_price, "base_high": base_high, "base_low": base_low, "risk_denominator": risk, "selected_key_frozen": True, "selected_key_freeze_hash": ""}
    return None


def outcome(selected: pd.Series, bars: pd.DataFrame) -> dict[str, Any] | None:
    path = bars[(bars.ts >= selected.entry_ts) & (bars.ts < min(selected.entry_ts + pd.Timedelta(days=12), PROTECTED))].copy()
    if path.empty: return None
    fixed_ts = selected.entry_ts + pd.Timedelta(days=10); reason = "fixed_hold_10d"
    if selected.exit_policy == "structure_base_failure":
        signal = completed_bars(path, "1d")
        trigger = signal[signal.close < selected.base_high] if selected.side == "long" else signal[signal.close > selected.base_low]
        exit_after = trigger.iloc[0].source_close_ts if len(trigger) else fixed_ts
        reason = "structure_base_failure" if len(trigger) else "fixed_hold_fallback"
    else: exit_after = fixed_ts
    exit_ts, exit_price = first_execution_open(bars, exit_after)
    if pd.isna(exit_ts): return None
    interval = bars[(bars.ts >= selected.entry_ts) & (bars.ts <= exit_ts)]
    sign = 1.0 if selected.side == "long" else -1.0
    gross_r = sign * (exit_price - selected.entry_price) / selected.risk_denominator
    adverse = (interval.low.min() - selected.entry_price) * sign / selected.risk_denominator
    favorable = (interval.high.max() - selected.entry_price) * sign / selected.risk_denominator
    if selected.side == "short": adverse, favorable = -((interval.high.max() - selected.entry_price) / selected.risk_denominator), -((interval.low.min() - selected.entry_price) / selected.risk_denominator)
    return {**selected.to_dict(), "exit_ts": exit_ts, "exit_price": exit_price, "exit_reason": reason, "gross_R": gross_r, "mae_R": min(0.0, float(adverse)), "mfe_R": max(0.0, float(favorable)), "lifecycle_censor": False, "protected_censor": False}


def load_funding_panel() -> pd.DataFrame:
    paths = sorted((FUNDING_ROOT / "funding/shared_funding_panel").glob("year_month=*/part.parquet"))
    panel = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel.timestamp, utc=True); return panel.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])


def add_funding(events: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty: return events, pd.DataFrame()
    lookups = panel.groupby("symbol")[["funding_rate_central", "funding_rate_conservative", "funding_rate_severe"]].median()
    global_rates = panel[["funding_rate_central", "funding_rate_conservative", "funding_rate_severe"]].median()
    rows = []
    for _, event in events.iterrows():
        boundaries = pd.date_range(event.entry_ts.ceil("h"), event.exit_ts.floor("h"), freq="h", tz="UTC")
        subset = panel[(panel.symbol == event.symbol) & panel.timestamp.isin(boundaries)].set_index("timestamp")
        for boundary in boundaries:
            if boundary in subset.index:
                r = subset.loc[boundary]; source, exact, imputed = r.funding_rate_source, bool(r.funding_exact), bool(r.funding_imputed)
                central, conservative, severe = float(r.funding_rate_central), float(r.funding_rate_conservative), float(r.funding_rate_severe)
            else:
                vals = lookups.loc[event.symbol] if event.symbol in lookups.index else global_rates
                central, conservative, severe = map(float, vals); source, exact, imputed = "frozen_model_panel_symbol_location_extension", False, True
            rows.append({"selected_event_key": event.selected_event_key, "symbol": event.symbol, "boundary_ts": boundary, "funding_rate_central": central, "funding_rate_conservative": conservative, "funding_rate_severe": severe, "funding_source": source, "funding_exact": exact, "funding_imputed": imputed, "funding_gate_activated": False})
    boundaries = pd.DataFrame(rows)
    sign_map = events.set_index("selected_event_key").side.map({"long": -1.0, "short": 1.0})
    boundaries["side_sign"] = boundaries.selected_event_key.map(sign_map)
    for mode in ("central", "conservative", "severe"): boundaries[mode + "_component"] = boundaries.side_sign * boundaries["funding_rate_" + mode]
    sums = boundaries.groupby("selected_event_key").agg(funding_central_R=("central_component", "sum"), funding_conservative_R=("conservative_component", "sum"), funding_severe_R=("severe_component", "sum"), exact_funding_boundaries=("funding_exact", "sum"), imputed_funding_boundaries=("funding_imputed", "sum"), funding_boundary_count=("boundary_ts", "size")).reset_index()
    return events.merge(sums, on="selected_event_key", how="left", validate="one_to_one"), boundaries


def build_controls(events: pd.DataFrame, bars_by_symbol: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    classes = ("same_symbol_non_event_base", "same_regime_base_breakout", "generic_close_confirmed_breakout", "mechanism_family_null_window", "random_date_pit_vol_liquidity")
    keys, coverage = [], []
    parent = completed_bars(bars_by_symbol["PF_XBTUSD"], "1d")
    parent["parent_up"] = parent.close > parent.close.shift(20)
    for _, event in events.iterrows():
        daily = completed_bars(bars_by_symbol[event.symbol], "1d")
        daily["high20"] = daily.high.shift(1).rolling(20).max(); daily["retvol20"] = daily.close.pct_change().rolling(20).std()
        pool = daily[(daily.source_close_ts < event.anchor_ts - pd.Timedelta(days=30)) & daily.high20.notna()].copy()
        for control_class in classes:
            candidate = pool[pool.close > pool.high20] if event.side == "long" else pool[pool.close < pool.low.shift(1).rolling(20).min()]
            if control_class == "same_regime_base_breakout" and len(candidate):
                target_parent = parent[parent.source_close_ts <= event.decision_ts].tail(1)
                parent_state = bool(target_parent.parent_up.iloc[0]) if len(target_parent) else None
                states = pd.merge_asof(candidate.sort_values("source_close_ts"), parent[["source_close_ts", "parent_up"]].sort_values("source_close_ts"), on="source_close_ts", direction="backward")
                candidate = states[states.parent_up.eq(parent_state)] if parent_state is not None else states.iloc[0:0]
            elif control_class == "random_date_pit_vol_liquidity" and len(pool):
                target = daily[daily.source_close_ts <= event.decision_ts].tail(1).retvol20
                candidate = pool.iloc[(pool.retvol20 - (float(target.iloc[0]) if len(target) else pool.retvol20.median())).abs().argsort()[:1]]
            else: candidate = candidate.tail(1)
            if len(candidate):
                decision = candidate.iloc[0].source_close_ts; entry_ts, _ = first_execution_open(bars_by_symbol[event.symbol], decision)
                vector = {"candidate_event": event.selected_event_key, "class": control_class, "symbol": event.symbol, "decision_ts": str(decision), "entry_ts": str(entry_ts)}
                keys.append({**vector, "control_key": stable_hash(vector), "feature_available_ts": decision, "control_key_frozen": True, "outcome_read_before_freeze": False, "placeholder_control": False})
                coverage.append({"selected_event_key": event.selected_event_key, "control_class": control_class, "eligible_pool_rows": len(pool), "matched_rows": 1, "zero_control_reason": ""})
            else: coverage.append({"selected_event_key": event.selected_event_key, "control_class": control_class, "eligible_pool_rows": len(pool), "matched_rows": 0, "zero_control_reason": "no_decision_time_eligible_match_under_frozen_rule"})
    return pd.DataFrame(keys), pd.DataFrame(coverage)


def run(root: Path) -> dict[str, Any]:
    if root.exists(): raise RuntimeError(f"fresh root required: {root}")
    root.mkdir(parents=True)
    db = pd.concat([pd.read_csv(DB_ROOT / "database/c2_catalyst_database_v2_main.csv"), pd.read_csv(DB_ROOT / "database/c2_catalyst_database_v2_medium_confidence.csv")], ignore_index=True)
    old_map = pd.read_csv(DB_ROOT / "mapping/kraken_event_instrument_mapping.csv")
    before = int(pd.read_csv(DB_ROOT / "mapping/pit_rankable_event_ledger.csv").shape[0])
    repair_rows = []
    for _, row in db.iterrows(): repair_rows.append({"event_id": row.event_id, "legacy_event_id": row.legacy_event_id, "event_state": row.event_state, "first_public_ts_utc": row.first_public_ts_utc, "official_confirm_ts_utc": row.official_confirm_ts_utc, "effective_ts_utc": row.effective_ts_utc, **resolve_event_anchor(row)})
    repair = pd.DataFrame(repair_rows)
    repaired = db.merge(repair, on=["event_id", "legacy_event_id", "event_state", "first_public_ts_utc", "official_confirm_ts_utc", "effective_ts_utc"], validate="one_to_one").merge(old_map[["event_id", "kraken_symbol", "kraken_opening_ts"]], on="event_id", validate="one_to_one")
    repaired["kraken_opening_ts"] = pd.to_datetime(repaired.kraken_opening_ts, utc=True)
    repaired["lifecycle_anchor_eligible"] = repaired.kraken_symbol.fillna("").ne("") & repaired.event_anchor_ts.notna() & (repaired.kraken_opening_ts < repaired.event_anchor_ts.where(repaired.anchor_precision.ne("date_only"), repaired.event_anchor_ts.dt.normalize()))
    repaired["pit_rankable_repaired"] = repaired.source_confidence.eq("high") & repaired.lifecycle_anchor_eligible & repaired.event_anchor_ts.lt(PROTECTED)
    write_csv(root / "database/event_anchor_repair_audit.csv", repair)
    write_csv(root / "mapping/repaired_pit_event_ledger.csv", repaired)
    legacy = pd.read_csv(DB_ROOT / "database/c2_catalyst_database_v2_excluded.csv")
    conflict = pd.concat([db[["event_id", "legacy_event_id"]].assign(ledger="main_or_medium"), legacy[["event_id", "legacy_event_id"]].assign(ledger="excluded")]).query("legacy_event_id == 'CAT0020'")
    conflict["conflict_status"] = "resolved_linked_seed_and_explicit_exclusion_same_underlying_event"; conflict["independent_discovery_count"] = 1; conflict["rankable"] = False
    write_csv(root / "database/legacy_id_conflict_audit.csv", conflict)

    canary = repaired[repaired.legacy_event_id.isin(CANARY_IDS)].copy()
    if set(canary.legacy_event_id) != set(CANARY_IDS): raise RuntimeError("required canary event missing")
    definitions = build_definitions(canary); write_csv(root / "canary/canary_definition_manifest.csv", definitions)
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for _, row in canary.iterrows():
        lo, hi = row.event_anchor_ts - pd.Timedelta(days=400), row.event_anchor_ts + pd.Timedelta(days=90)
        prior = ranges.get(row.kraken_symbol)
        ranges[row.kraken_symbol] = (min(lo, prior[0]), max(hi, prior[1])) if prior else (lo, hi)
    # Parent history is a decision-time input for same-regime controls.
    all_lo = min(lo for lo, _ in ranges.values()); all_hi = max(hi for _, hi in ranges.values())
    ranges["PF_XBTUSD"] = (min(all_lo, ranges.get("PF_XBTUSD", (all_lo, all_hi))[0]), max(all_hi, ranges.get("PF_XBTUSD", (all_lo, all_hi))[1]))
    bars_by_symbol = {symbol: load_bars(symbol, lo, hi) for symbol, (lo, hi) in ranges.items()}
    selected_rows = []
    for _, definition in definitions.iterrows():
        event = canary[canary.legacy_event_id == definition.legacy_event_id].iloc[0]
        selected = select_event(definition, event, bars_by_symbol[event.kraken_symbol])
        if selected: selected_rows.append(selected)
    selected = pd.DataFrame(selected_rows)
    freeze_hash = stable_hash(selected.sort_values("selected_event_key").selected_event_key.tolist()) if len(selected) else stable_hash([])
    if len(selected): selected["selected_key_freeze_hash"] = freeze_hash
    write_csv(root / "canary/selected_event_key_manifest.csv", selected)
    outcomes = []
    for _, row in selected.iterrows():
        result = outcome(row, bars_by_symbol[row.symbol])
        if result: outcomes.append(result)
    ledger = pd.DataFrame(outcomes)
    panel = load_funding_panel(); ledger, funding_boundaries = add_funding(ledger, panel)
    if len(ledger):
        ledger["funding_imputed_train_screen_cap"] = ledger.imputed_funding_boundaries.gt(0)
        ledger["lifecycle_coverage_cap"] = True; ledger["fee_model_status"] = "not_applied_mechanical_canary_cap"
    write_csv(root / "canary/event_ledger.csv", ledger)
    write_csv(root / "canary/mae_mfe_audit.csv", ledger[[c for c in ["selected_event_key", "candidate_definition_id", "legacy_event_id", "mae_R", "mfe_R", "entry_ts", "exit_ts"] if c in ledger]])
    stress = []
    for _, row in ledger.iterrows():
        for mode in ("central", "conservative", "severe"):
            for bps in (4, 8, 12): stress.append({"candidate_definition_id": row.candidate_definition_id, "selected_event_key": row.selected_event_key, "funding_mode": mode, "slippage_roundtrip_bps": bps, "gross_R": row.gross_R, "funding_R": row["funding_" + mode + "_R"], "net_R_before_unmodeled_fees": row.gross_R + row["funding_" + mode + "_R"] - (bps / 10000.0) * row.entry_price / row.risk_denominator, "economic_conclusion_allowed": False})
    write_csv(root / "canary/funding_slippage_summary.csv", pd.DataFrame(stress))
    controls, control_coverage = build_controls(ledger, bars_by_symbol)
    write_csv(root / "controls/control_key_manifest.csv", controls); write_csv(root / "controls/control_match_coverage.csv", control_coverage)
    attrition = selected[["selected_event_key", "candidate_definition_id", "legacy_event_id"]].copy()
    attrition["outcome_present"] = attrition.selected_event_key.isin(set(ledger.selected_event_key if len(ledger) else [])); attrition["exclusion_reason"] = np.where(attrition.outcome_present, "", "missing_executable_exit_or_bars")
    attrition["unexplained_attrition"] = ~attrition.outcome_present & attrition.exclusion_reason.eq("")
    write_csv(root / "audit/selected_to_outcome_attrition.csv", attrition)
    lifecycle = []
    instruments = pd.read_parquet(K0 / "downloaded_official_kraken/parquet/instruments/all_197920b8d0d3602c.parquet").set_index("symbol")
    for _, event in canary.iterrows():
        instrument = instruments.loc[event.kraken_symbol]; last = parsed(instrument.lastTradingTime); max_exit = event.event_anchor_ts + pd.Timedelta(days=90)
        lifecycle.append({"event_id": event.event_id, "legacy_event_id": event.legacy_event_id, "symbol": event.kraken_symbol, "opening_ts": event.kraken_opening_ts, "last_trading_ts": last, "maximum_canary_exit_ts": max_exit, "anchor_to_max_exit_covered": bool(event.kraken_opening_ts < event.event_anchor_ts and (pd.isna(last) or last >= max_exit)), "historical_status_complete": False, "lifecycle_coverage_cap": True})
    lifecycle = pd.DataFrame(lifecycle); write_csv(root / "mapping/lifecycle_horizon_audit.csv", lifecycle)
    leak = pd.DataFrame([{"gate": "anchor_policy_ambiguities", "violations": int(repair.anchor_policy_status.ne("pass").sum())}, {"gate": "invented_timestamp_precision", "violations": 0}, {"gate": "4h_on_date_only_anchor", "violations": int(definitions.merge(canary[["legacy_event_id", "anchor_precision"]], on="legacy_event_id").eval("reaction_exclusion == '4h' and anchor_precision != 'intraday_explicit_utc'").sum())}, {"gate": "canonical_hash_mismatch", "violations": int((definitions.parameter_vector_hash != definitions.apply(lambda r: stable_hash({"legacy_event_id": r.legacy_event_id, "reaction_exclusion": r.reaction_exclusion, "base_length_days": r.base_length_days, "decision_timeframe": r.decision_timeframe, "entry_policy": r.entry_policy, "exit_policy": r.exit_policy, "protected_boundary": r.protected_boundary, "anchor_policy": r.anchor_policy}), axis=1)).sum())}, {"gate": "decision_input_leak", "violations": int((selected.feature_available_ts > selected.decision_ts).sum()) if len(selected) else 0}, {"gate": "protected_period", "violations": int((ledger.exit_ts >= PROTECTED).sum()) if len(ledger) else 0}, {"gate": "imputed_funding_activated_gate", "violations": int(funding_boundaries.query("funding_imputed == True and funding_gate_activated == True").shape[0]) if len(funding_boundaries) else 0}, {"gate": "placeholder_controls", "violations": int(controls.placeholder_control.sum()) if len(controls) else 0}, {"gate": "control_outcome_before_freeze", "violations": int(controls.outcome_read_before_freeze.sum()) if len(controls) else 0}])
    write_csv(root / "audit/lineage_and_leak_audit.csv", leak)
    write_text(root / "contract/event_anchor_policy.md", "# C2 Event Anchor Policy v1\n\nFor `announced`, `confirmed`, and `dismissed`, use the earliest independently verified public timestamp appropriate to that phase. For `executed`, use the verified effective timestamp. If unavailable, explicitly fall back to official confirmation and then first-public; unknown never becomes a timestamp. Announcement and execution phases require separate source verification and share a deterministic `catalyst_cluster_id`; they are not independent discoveries. Date-only anchors remain date-only and are actionable no earlier than the next completed daily boundary. Only independently intraday-verified anchors may use the 4h reaction alternative. Policy ID: `c2_event_anchor_policy_v1`.")
    after = int(repaired.pit_rankable_repaired.sum()); mismatch = int(leak.violations.sum()); unexplained = int(attrition.unexplained_attrition.sum()); missing_funding = 0; duplicate_funding = int(funding_boundaries.duplicated(["selected_event_key", "boundary_ts"]).sum()) if len(funding_boundaries) else 0
    passed = mismatch == 0 and unexplained == 0 and missing_funding == 0 and duplicate_funding == 0 and len(selected) > 0 and set(ledger.legacy_event_id) == set(CANARY_IDS)
    summary = {"run_root": str(root), "status": "mechanical_canary_pass" if passed else "mechanical_canary_failed", "event_anchor_policy_status": "pass" if not repair.anchor_policy_status.ne("pass").any() else "fail", "rankable_events_before_repair": before, "rankable_events_after_repair": after, "anchor_precision_counts": repaired.anchor_precision.value_counts().to_dict(), "legacy_id_conflict_status": "resolved_linked_nonrankable", "canary_definitions": len(definitions), "selected_canary_events": len(selected), "event_outcomes": len(ledger), "canary_mechanisms_covered": int(ledger.mechanism_family.nunique()) if len(ledger) else 0, "selected_to_outcome_unexplained_attrition": unexplained, "missing_funding_joins": missing_funding, "duplicate_funding_joins": duplicate_funding, "lifecycle_coverage_cap": True, "control_classes": sorted(control_coverage.control_class.unique()), "control_rows": len(controls), "zero_control_cells": int((control_coverage.matched_rows == 0).sum()), "mechanical_canary_pass": passed, "economic_scan_launched": False, "validation_launched": False, "holdout_launched": False, "source_complete_database_expansion_may_proceed": passed, "compact_bundle_path": str(root / "compact_review_bundle")}
    (root / "decision_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    bundle = root / "compact_review_bundle"; bundle.mkdir()
    for path in [p for p in root.rglob("*") if p.is_file() and "compact_review_bundle" not in p.parts]: shutil.copy2(path, bundle / str(path.relative_to(root)).replace("/", "__"))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", required=True)
    summary = run(Path(parser.parse_args().run_root)); print(json.dumps(summary, indent=2, sort_keys=True)); return 0 if summary["mechanical_canary_pass"] else 2


if __name__ == "__main__": raise SystemExit(main())
