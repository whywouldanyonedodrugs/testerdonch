#!/usr/bin/env python3
"""Build Stage 14 outcome-free measurements and local state tapes."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from qlmg_derivatives_phase1 import (
    FEATURE_COLUMNS, PARENT_EVENT_COLUMNS, GRAMMAR_LADDERS, OutcomeReadSpy,
    causal_oi_normalization, completed_purge_states, episode_table,
    oi_retention_gap_counts, onset_mask, reconcile_universe, stable_hash, strict_base_valid,
    validate_grammar,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/agent/task_archive/20260719_donch_bt_stage_14_close_derivatives_phase1_blockers_20260719_v1"
FEATURE_MANIFEST = ROOT / "docs/agent/task_archive/20260719_donch_bt_stage_8a_kraken_derivatives_state_foundation_20260719_v1/KDA_FEATURE_CACHE_MANIFEST.json"
INVENTORY = ROOT / "docs/agent/task_archive/20260717_donch_bt_stage_7b_resumable_analytics_acquisition_20260717_v1/KRAKEN_ANALYTICS_FROZEN_SYMBOL_INVENTORY.csv"
FUNDING_MANIFEST = Path("/opt/testerdonch/results/rebaseline/phase_kraken_shared_funding_imputation_model_20260711_v1/funding/shared_funding_panel_manifest.csv")
PARENT_ROOTS = {
    "KDA01": Path("/opt/parquet/kraken_derivatives/analytics/stage8b_kda01_v2_prerun_v1_final"),
    "KDA02": Path("/opt/parquet/kraken_derivatives/analytics/stage9_kda02_v2_prerun_v4"),
    "KDA03": Path("/opt/parquet/kraken_derivatives/analytics/stage11_kda03_v1"),
}
FIXED_AT = "2026-07-19T00:00:00Z"


def q(series: pd.Series, value: float) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return None if numeric.empty else float(numeric.quantile(value))


def hash_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parents(spy: OutcomeReadSpy) -> dict[str, set[tuple[str, int]]]:
    result: dict[str, set[tuple[str, int]]] = {}
    for family, root in PARENT_ROOTS.items():
        keys: set[tuple[str, int]] = set()
        for path in sorted(root.glob("symbol=*/events.parquet")):
            schema = set(pq.ParquetFile(path).schema_arrow.names)
            if not set(PARENT_EVENT_COLUMNS).issubset(schema):
                if pq.ParquetFile(path).metadata.num_rows:
                    raise ValueError(f"nonempty {family} parent tape lacks safe identity columns: {path}")
                continue
            frame = spy.read(path, PARENT_EVENT_COLUMNS, kind="parent")
            if len(frame):
                ts = pd.to_datetime(frame.decision_ts, utc=True).dt.as_unit("ns").astype("int64")
                keys.update(zip(frame.symbol.astype(str), ts.astype(int)))
        result[family] = keys
    return result


def write_measurement_docs(out: Path, measurement: dict, funding: pd.DataFrame) -> None:
    k = measurement["KDA02B"]
    (out / "KDA02B_PHASE1_MEASUREMENT.md").write_text(f"""# KDA02B Phase-1 Measurement

Status: `mechanism_development_ready`; Phase-1 admission: `{measurement['admissions']['KDA02B_v2_oi_vacuum_redevelopment']}`.

The unsigned Kraken OI quantity identifies contraction but cannot identify whether new longs, new shorts, or liquidation caused it. The registered continuation and reversal branches therefore remain symmetric development branches. No branch was chosen from historical outcomes.

The tape contains `{k['episodes']:,}` causally admissible false-to-true episodes across `{k['symbols']}` symbols. Retention starts were suppressed unless the immediately prior row was valid, contiguous, and false. Median duration is `{k['median_duration_minutes']}` minutes. Contemporaneous absolute trade displacement is `{k['trade_abs_bps_median']:.3f}` bps median; `{k['trade_abs_ge_14_share']:.4f}` and `{k['trade_abs_ge_32_share']:.4f}` of onsets meet or exceed 14/32 bps. These are state magnitudes, not expected profits or remaining movement estimates.

Authority is the frozen Stage-8 feature cache. OI is base-unit, unsigned, and `inferred_authoritative_v1`; the universe is current-roster/lifecycle capped. Large onset and episode tapes remain local under the hash-manifested run root.
""", encoding="utf-8")
    b = measurement["KDA02C"]
    (out / "KDA02C_PIT_BREADTH_CONTRACT.md").write_text(f"""# KDA02C PIT Breadth Contract

Status: `{measurement['admissions']['KDA02C_v1_purge_breadth_context']}`. Labels remain `post_hoc_context_hypothesis` and `program_exposed_historical`.

At each completed five-minute source timestamp, the denominator is only the authorized Stage-8 cohort members whose lifecycle, trade, mark, analytics, and eligibility masks are true at that timestamp. The stored decision time is source timestamp plus five minutes. Primary and robustness completed-purge identities exactly reproduce the Stage-8 causal definitions. Counts are split by contemporaneous price sign; BTC (`PF_XBTUSD`) and ETH (`PF_ETHUSD`) identifiers are retained. Diagnostic 5/15/30/60-minute windows are fixed before outcomes; no breadth cutoff is selected here.

The panel has `{b['timestamps']:,}` timestamps, median eligible denominator `{b['median_denominator']}`, and `{b['primary_onsets']:,}` / `{b['robust_onsets']:,}` primary/robust onsets across `{b['conservative_UTC_hour_clusters']:,}` conservative UTC-hour clusters. Primary-onset contemporaneous absolute trade displacement is `{b['primary_onset_trade_abs_bps_median']:.3f}` bps median; `{b['primary_onset_trade_abs_ge_14_share']:.4f}` and `{b['primary_onset_trade_abs_ge_32_share']:.4f}` meet or exceed 14/32 bps. The cohort remains current-roster/lifecycle capped and is not survivorship-free. Plausible remaining movement is intentionally unknown until authorized Phase 2; no profit is inferred from raw state magnitude.
""", encoding="utf-8")
    x = measurement["KDX01"]
    (out / "KDX01_PRIMITIVE_STATE_CONTRACT.md").write_text(f"""# KDX01 Primitive State Contract

Status: `{measurement['admissions']['KDX01_v1_downside_completed_derivatives_state_rejection']}`. Contamination label: `cross_family_program_exposed_redevelopment`.

The price anchor separately registers contemporaneous downside trade displacement, downside mark displacement, and structural rejection completion. Incremental primitives are OI contraction, elevated liquidation, extreme basis level, negative basis change, and primary purge breadth. The fixed grammar is `{json.dumps([list(x) for x in GRAMMAR_LADDERS])}` with maximum interaction depth six. It is not a union of prior favourable branches.

The price-only anchor produced `{x['price_onsets']:,}` onsets and the price-plus-OI ladder `{x['price_oi_onsets']:,}`; therefore OI is mechanically incremental without reading outcomes. Even the sparsest grammar ladder retains `{x['minimum_grammar_UTC_hour_clusters']:,}` conservative UTC-hour clusters. Price-plus-OI contemporaneous absolute trade displacement is `{x['price_oi_trade_abs_bps_median']:.3f}` bps median; `{x['price_oi_trade_abs_ge_14_share']:.4f}` and `{x['price_oi_trade_abs_ge_32_share']:.4f}` meet or exceed 14/32 bps. Actor direction remains a proxy: OI is unsigned and liquidation side is price-inferred. The family is distinguishable from price-only state mechanically, but plausible remaining movement and any economic increment remain untested until authorized Phase 2.
""", encoding="utf-8")
    exact = int(funding.exact_rows.sum()); imputed = int(funding.imputed_rows.sum())
    (out / "CAMPAIGN_FUNDING_AND_COST_CONTRACT.md").write_text(f"""# Campaign Funding and Cost Contract

This proposal is non-authorizing. Pre-funding all-in round-trip costs remain 14 bps base and 32 bps stress.

The authoritative shared funding panel (`0054af0ee40740e39739bfade92f342867bb208a4fe7ed15b151a8a0a838d072`) contains `{exact:,}` exact and `{imputed:,}` imputed boundary rows in its published monthly manifest. Exact coverage is concentrated from June 2025 onward, so an exact-only primary contract would silently discard most earlier folds.

The published required-boundary panel begins `2024-01-01`; it does not cover the registered 2023 development/evaluation boundaries. Before any Phase-2 outcome reader opens, the later authorized campaign must extend the boundary panel to every registered 2023-2025 event using the already frozen selected model, unchanged parameters, PIT features, and the same exact/mixed/imputed labels. This is outcome-free model application, not retraining. A 2023 boundary lacking the frozen model's required PIT inputs fails that event closed and is reported by fold; the campaign stops the affected fold if the loss breaches its predeclared coverage floor. No missing boundary is relabelled `zero_boundary`.

Future Phase 2-5 primary development/evaluation metric: base net expectancy after 14 bps plus signed exact funding where available and the authoritative conservative adverse imputation elsewhere. Each event must be labelled `zero_boundary`, `exact`, `mixed`, or `imputed`; no imputed value may activate a signal, pass a funding gate, select a direction, or improve a candidate relative to its zero-funding result. Required sensitivities are 32 bps stress plus conservative funding, exact-boundary-only slice, zero-funding diagnostic, and severe adverse imputation. Missing panel joins fail closed after the required extension; missing exact observations do not fail a fold when the approved partitioned contract permits conservative imputation. Exact funding remains mandatory only for an explicitly approved exact-only claim.
""", encoding="utf-8")


def build(output: Path) -> None:
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    (output / "kda02b_onsets").mkdir(); (output / "kda02b_episodes").mkdir(); (output / "primitive_base").mkdir()
    feature_manifest = json.loads(FEATURE_MANIFEST.read_text())
    parts = sorted(feature_manifest["partitions"], key=lambda x: x["symbol"])
    symbols = {x["symbol"] for x in parts}
    if feature_manifest["protected_rows_opened"] != 0 or len(parts) != 187:
        raise ValueError("feature authority mismatch")
    spy = OutcomeReadSpy()
    grid = pd.date_range("2023-01-01", "2026-01-01", freq="5min", inclusive="left", tz="UTC")
    n = len(grid); origin = grid[0].value; step = 300_000_000_000
    fields = {name: np.zeros(n, dtype=np.int32) for name in (
        "eligible", "known", "excluded", "membership_change", "primary_state", "robust_state",
        "primary_onset", "robust_onset", "primary_negative", "primary_positive",
        "robust_negative", "robust_positive")}
    xbt_primary = np.zeros(n, np.int8); eth_primary = np.zeros(n, np.int8)
    retention: list[dict] = []; episode_summaries: list[dict] = []
    all_onset_samples: list[pd.DataFrame] = []
    rows_read = 0; bytes_read = 0
    for number, part in enumerate(parts, 1):
        path = Path(part["path"]); bytes_read += path.stat().st_size
        if hash_file(path) != part["sha256"]:
            raise ValueError(f"feature partition hash mismatch: {path}")
        frame = spy.read(path, FEATURE_COLUMNS, kind="feature").sort_values("timestamp_utc", kind="mergesort").reset_index(drop=True)
        if len(frame) != part["rows"] or frame.timestamp_utc.duplicated().any():
            raise ValueError(f"partition identity mismatch: {part['symbol']}")
        rows_read += len(frame); ts = pd.to_datetime(frame.timestamp_utc, utc=True).dt.as_unit("ns")
        ts_ns = ts.dt.as_unit("ns")
        pos = ((ts_ns.astype("int64").to_numpy() - origin) // step).astype(int)
        if (pos < 0).any() or (pos >= n).any() or not np.array_equal(grid.asi8[pos], ts_ns.astype("int64").to_numpy()):
            raise ValueError(f"timestamp grid mismatch: {part['symbol']}")
        base = strict_base_valid(frame)
        oi_norm = causal_oi_normalization(ts, frame.oi_log_change_1h)
        valid = base & oi_norm.oi_normalization_valid & frame.oi_log_change_1h.notna() & frame.trade_return_1h.notna() & frame.mark_return_1h.notna() & frame.realized_vol_24h.notna()
        state = valid & (frame.oi_log_change_1h < 0) & (frame.trade_return_1h.abs() <= 2 * frame.realized_vol_24h)
        onsets = onset_mask(state, valid, ts)
        extra = pd.DataFrame({
            "directional_price_state": np.where(frame.trade_return_1h < 0, "negative", np.where(frame.trade_return_1h > 0, "positive", "flat")),
            "oi_log_change_1h": frame.oi_log_change_1h,
            "oi_change_percentile": oi_norm.oi_change_percentile,
            "oi_change_robust_z": oi_norm.oi_change_robust_z,
            "trade_displacement_bps": frame.trade_return_1h * 10000,
            "mark_displacement_bps": frame.mark_return_1h * 10000,
            "realized_vol_ratio": frame.trade_return_1h.abs() / frame.realized_vol_24h,
            "liquidation_present": frame.liquidation_base_units_1h.fillna(0).gt(0),
            "year": ts.dt.year, "liquidity_cohort": frame.major_vs_alt.astype(str),
        })
        onset_frame = pd.concat([pd.DataFrame({"symbol": part["symbol"], "state_ts": ts[onsets], "decision_ts": ts[onsets] + pd.Timedelta(minutes=5)}).reset_index(drop=True), extra.loc[onsets].reset_index(drop=True)], axis=1)
        episodes = episode_table(part["symbol"], ts, state, valid, extra)
        onset_frame.to_parquet(output / "kda02b_onsets" / f"{part['symbol']}.parquet", index=False, compression="zstd")
        episodes.to_parquet(output / "kda02b_episodes" / f"{part['symbol']}.parquet", index=False, compression="zstd")
        all_onset_samples.append(onset_frame)
        gaps,oi_missing_bars=oi_retention_gap_counts(ts,frame.oi_close_base_units)
        first_valid = ts[valid].min() if valid.any() else pd.NaT
        first_onset = ts[onsets].min() if onsets.any() else pd.NaT
        first_oi = ts[frame.oi_close_base_units.notna()].min() if frame.oi_close_base_units.notna().any() else pd.NaT
        retention.append({"symbol": part["symbol"], "first_raw_OI_timestamp": first_oi, "first_complete_causal_lookback": first_valid,
                          "first_admissible_onset_timestamp": first_onset, "invalid_boundary_start": ts.min(),
                          "invalid_boundary_end": first_valid, "gap_count": gaps,"OI_missing_bars_inside_retention":oi_missing_bars,
                          "truncation_status": "left_truncated_OI" if pd.notna(first_oi) and first_oi > pd.Timestamp("2023-01-01", tz="UTC") else "rankable_start_present"})
        if len(episodes):
            for (year, cohort), group in episodes.groupby([episodes.onset_ts.dt.year, "liquidity_cohort"], dropna=False):
                episode_summaries.append({"group_type":"symbol_year_liquidity", "group_value":f"{part['symbol']}|{year}|{cohort}", "episodes":len(group),
                                          "duration_minutes_median":q(group.duration_minutes,.5), "duration_minutes_p95":q(group.duration_minutes,.95),
                                          "gap_minutes_median":q(group.minutes_since_prior_episode,.5), "oi_log_change_median":q(group.oi_log_change_1h,.5),
                                          "trade_abs_bps_median":q(group.trade_displacement_bps.abs(),.5), "mark_abs_bps_median":q(group.mark_displacement_bps.abs(),.5),
                                          "realized_vol_ratio_median":q(group.realized_vol_ratio,.5), "liquidation_present_share":float(group.liquidation_present.mean())})
        primary, robust, direction = completed_purge_states(frame)
        p_on = onset_mask(primary, base, ts); r_on = onset_mask(robust, base, ts)
        known = frame.known_lifecycle_mask.fillna(False).astype(bool)
        fields["eligible"][pos] += base.to_numpy(np.int32); fields["known"][pos] += known.to_numpy(np.int32)
        fields["excluded"][pos] += (known & ~base).to_numpy(np.int32)
        membership = base.astype(np.int8).diff().fillna(0).ne(0)
        fields["membership_change"][pos] += membership.to_numpy(np.int32)
        for key, values in (("primary_state",primary),("robust_state",robust),("primary_onset",p_on),("robust_onset",r_on)):
            fields[key][pos] += values.to_numpy(np.int32)
        for prefix, values in (("primary", primary), ("robust", robust)):
            fields[f"{prefix}_negative"][pos] += (values & direction.lt(0)).to_numpy(np.int32)
            fields[f"{prefix}_positive"][pos] += (values & direction.gt(0)).to_numpy(np.int32)
        if part["symbol"] == "PF_XBTUSD": xbt_primary[pos] = primary.to_numpy(np.int8)
        if part["symbol"] == "PF_ETHUSD": eth_primary[pos] = primary.to_numpy(np.int8)
        primitive = pd.DataFrame({"timestamp_utc":ts, "eligible":base, "trade_downside":frame.trade_return_1h.lt(0),"mark_downside":frame.mark_return_1h.lt(0),"structural_rejection":frame.trade_log_return_5m.lt(0),
                                  "oi":frame.oi_log_change_1h.lt(0), "liquidation":frame.liquidation_normalization_valid.fillna(False) & frame.liquidation_intensity_robust_z.ge(2),
                                  "basis_level":frame.basis_level_normalization_valid.fillna(False) & frame.basis_level_robust_z.abs().ge(2),"basis_change":frame.basis_change_normalization_valid.fillna(False) & frame.basis_change_robust_z.le(-2),
                                  "trade_abs_bps":frame.trade_return_1h.abs()*10000, "mark_abs_bps":frame.mark_return_1h.abs()*10000,
                                  "oi_log_change_1h":frame.oi_log_change_1h, "liquidation_intensity_robust_z":frame.liquidation_intensity_robust_z,
                                  "basis_bps":frame.basis_bps, "basis_change_robust_z":frame.basis_change_robust_z})
        primitive.to_parquet(output / "primitive_base" / f"{part['symbol']}.parquet", index=False, compression="zstd")
        if number % 20 == 0: print(f"stage14 pass1 {number}/{len(parts)}", flush=True)
    breadth = pd.DataFrame({"timestamp_utc":grid, **fields, "BTC_primary_state":xbt_primary, "ETH_primary_state":eth_primary})
    denom = breadth.eligible.replace(0, np.nan)
    breadth["primary_cohort_share"] = breadth.primary_state / denom
    breadth["robust_cohort_share"] = breadth.robust_state / denom
    for minutes in (5,15,30,60):
        bars = minutes // 5
        breadth[f"primary_onsets_{minutes}m"] = breadth.primary_onset.rolling(bars, min_periods=1).sum()
        breadth[f"primary_share_max_{minutes}m"] = breadth.primary_cohort_share.rolling(bars, min_periods=1).max()
        breadth[f"robust_onsets_{minutes}m"] = breadth.robust_onset.rolling(bars, min_periods=1).sum()
    breadth.to_parquet(output / "KDA02C_PIT_BREADTH_PANEL.parquet", index=False, compression="zstd")
    breadth_rows=[]
    for year, group in breadth.groupby(breadth.timestamp_utc.dt.year):
        for identity in ("primary","robust"):
            on=group[f"{identity}_onset"]; share=group[f"{identity}_cohort_share"]
            breadth_rows.append({"year":year,"identity":identity,"timestamps":len(group),"eligible_denominator_min":int(group.eligible.min()),"eligible_denominator_median":q(group.eligible,.5),
                                 "eligible_denominator_max":int(group.eligible.max()),"onsets":int(on.sum()),"isolated_timestamp_frequency":float((on==1).mean()),
                                 "broad_timestamp_frequency_ge_2":float((on>=2).mean()),"cohort_share_p95":q(share,.95),"cohort_share_p99":q(share,.99),
                                 "active_episode_count_p95":q(group[f"{identity}_state"],.95),"membership_changes":int(group.membership_change.sum())})
    pd.DataFrame(breadth_rows).to_csv(ARCHIVE/"KDA02C_BREADTH_SUMMARY.csv",index=False)
    all_onsets = pd.concat(all_onset_samples, ignore_index=True)
    pd.DataFrame(episode_summaries).to_csv(ARCHIVE/"KDA02B_ONSET_AND_EPISODE_SUMMARY.csv",index=False)
    pd.DataFrame(retention).to_csv(ARCHIVE/"KDA02B_RETENTION_BOUNDARY.csv",index=False)
    # KDX01 fixed grammar and overlap measurement.
    components=("trade_downside","mark_downside","structural_rejection","oi","liquidation","basis_level","basis_change","breadth")
    component_counts={(a,b):0 for a in components for b in components}
    incidence=[]; parent_keys=load_parents(spy); parent_overlap={x:0 for x in parent_keys}; address_counts={}
    total_ladder={"+".join(x):0 for x in GRAMMAR_LADDERS}; total_onsets={"+".join(x):0 for x in GRAMMAR_LADDERS}
    breadth_flag = breadth.primary_cohort_share.fillna(0).gt(0).to_numpy()
    for number, part in enumerate(parts,1):
        p=pd.read_parquet(output/"primitive_base"/f"{part['symbol']}.parquet")
        ts=pd.to_datetime(p.timestamp_utc,utc=True); pos=((ts.dt.as_unit("ns").astype("int64").to_numpy()-origin)//step).astype(int)
        p["breadth"]=breadth_flag[pos]; valid=p.eligible.astype(bool)
        for a in components:
            for b in components:
                component_counts[(a,b)] += int((valid & p[a].astype(bool) & p[b].astype(bool)).sum())
        for cell in validate_grammar():
            key="+".join(cell); state=valid.copy()
            for comp in cell: state &= p[comp].astype(bool)
            onset=onset_mask(state,valid,ts); total_ladder[key]+=int(state.sum()); total_onsets[key]+=int(onset.sum())
            for year in (2023,2024,2025):
                y=ts.dt.year.eq(year); incidence.append({"symbol":part["symbol"],"year":year,"grammar_cell":key,"state_rows":int((state&y).sum()),"onsets":int((onset&y).sum())})
            if cell == GRAMMAR_LADDERS[1]:
                for raw_ns in ts[onset].dt.as_unit("ns").astype("int64") + 300_000_000_000:
                    value=pd.Timestamp(int(raw_ns),tz="UTC")
                    addr=stable_hash({"symbol":part["symbol"],"decision_ts":value.isoformat(),"grammar":key}); address_counts[addr]=address_counts.get(addr,0)+1
                    pair=(part["symbol"],int(value.value))
                    for family, keys in parent_keys.items(): parent_overlap[family]+=int(pair in keys)
        if number % 40 == 0: print(f"stage14 pass2 {number}/{len(parts)}",flush=True)
    pd.DataFrame([{"component_a":a,"component_b":b,"eligible_overlap_rows":value} for (a,b),value in component_counts.items()]).to_csv(ARCHIVE/"KDX01_COMPONENT_OVERLAP_MATRIX.csv",index=False)
    pd.DataFrame(incidence).to_csv(ARCHIVE/"KDX01_INCIDENCE_SUMMARY.csv",index=False)
    universe=reconcile_universe(pd.read_csv(INVENTORY),symbols); universe.to_csv(ARCHIVE/"KRAKEN_CAMPAIGN_UNIVERSE_RECONCILIATION.csv",index=False)
    funding=pd.read_csv(FUNDING_MANIFEST)
    elapsed=time.monotonic()-started; kdx_price=total_onsets["+".join(GRAMMAR_LADDERS[0])]; kdx_oi=total_onsets["+".join(GRAMMAR_LADDERS[1])]
    admissions={"KDA02B_v2_oi_vacuum_redevelopment":"phase_2_ready" if len(all_onsets)>=1000 else "mechanism_underidentified",
                "KDA02C_v1_purge_breadth_context":"phase_2_ready" if int(breadth.primary_onset.sum())>=1000 else "mechanically_unavailable",
                "KDX01_v1_downside_completed_derivatives_state_rejection":"phase_2_ready" if kdx_oi>=1000 and kdx_oi<kdx_price else "mechanism_underidentified"}
    measurement={"generated_at":FIXED_AT,"economic_outputs_computed":False,"protected_rows_opened":0,"Capitalcom_payload_opened":0,"feature_rows_read":rows_read,
                 "feature_bytes_read":bytes_read,"outcome_read_spy":{"request_count":len(spy.requests),"requests":spy.requests,"allowed_feature_columns":list(FEATURE_COLUMNS),"allowed_parent_columns":list(PARENT_EVENT_COLUMNS)},
                 "KDA02B":{"episodes":len(all_onsets),"symbols":int(all_onsets.symbol.nunique()),"median_duration_minutes":q(pd.concat([pd.read_parquet(x,columns=['duration_minutes']) for x in sorted((output/'kda02b_episodes').glob('*.parquet'))]),.5),
                            "trade_abs_bps_median":q(all_onsets.trade_displacement_bps.abs(),.5),"trade_abs_ge_14_share":float(all_onsets.trade_displacement_bps.abs().ge(14).mean()),"trade_abs_ge_32_share":float(all_onsets.trade_displacement_bps.abs().ge(32).mean())},
                 "KDA02C":{"timestamps":len(breadth),"median_denominator":q(breadth.eligible,.5),"primary_onsets":int(breadth.primary_onset.sum()),"robust_onsets":int(breadth.robust_onset.sum())},
                 "KDX01":{"price_onsets":kdx_price,"price_oi_onsets":kdx_oi,"grammar_state_rows":total_ladder,"grammar_onsets":total_onsets,"parent_event_exact_timestamp_overlap":parent_overlap,
                          "duplicate_economic_addresses":int(sum(v>1 for v in address_counts.values()))},"admissions":admissions,
                 "benchmark":{"wall_seconds":elapsed,"peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"feature_rows_per_second":rows_read/elapsed,"workers":1,"deterministic_cell_projection_method":"measured full outcome-free primitive scans"}}
    (output/"PHASE1_MEASUREMENT.json").write_text(json.dumps(measurement,indent=2,sort_keys=True,default=str)+"\n")
    (ARCHIVE/"PHASE1_ADMISSION_DECISIONS.json").write_text(json.dumps({"generated_at":FIXED_AT,"decisions":admissions,"criteria":"outcome-free incidence, causal completeness, raw scale, exact bounded grammar, funding and benchmark contracts"},indent=2,sort_keys=True)+"\n")
    write_measurement_docs(ARCHIVE,measurement,funding)
    print(json.dumps({"status":"complete","admissions":admissions,"wall_seconds":elapsed,"rows":rows_read},sort_keys=True))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); build(args.output)
