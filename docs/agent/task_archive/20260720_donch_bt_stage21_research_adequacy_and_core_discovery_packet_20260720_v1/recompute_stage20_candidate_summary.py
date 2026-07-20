#!/usr/bin/env python3
"""Recompute the 45-row Stage 20 selected-candidate audit from terminal shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path("/opt/testerdonch/results/rebaseline/phase_kraken_derivatives_campaign_stage20_20260720_v01")
STAGE19 = Path("/opt/testerdonch/docs/agent/task_archive/20260720_donch_bt_stage_19_local_official_funding_export_packet_20260720_v2")
COLS = ["cell_id", "symbol", "entry_ts", "gross_bps", "base_net_bps", "stress_net_bps",
        "funding_adverse_exact_bps", "funding_base_gap_cost_bps", "funding_stress_gap_cost_bps",
        "base_net_alignment_start_bps", "base_net_alignment_end_bps"]


def load_root(root: Path, selected: list[str]) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=COLS)
        frame = frame.loc[frame.cell_id.isin(selected)]
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)


def extras(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    keys = ["symbol_contribution", "day_contribution", "year_contribution", "gross_mean_bps",
            "fixed_cost_bps", "adverse_exact_funding_mean_bps", "base_gap_allowance_mean_bps",
            "stress_gap_allowance_mean_bps", "base_net_recomputed_mean_bps",
            "stress_net_recomputed_mean_bps", "alignment_start_mean_bps", "alignment_end_mean_bps"]
    if frame.empty:
        return {prefix + key: np.nan for key in keys}
    work = frame.copy(); work["entry_ts"] = pd.to_datetime(work.entry_ts, utc=True)
    work["day"] = work.entry_ts.dt.strftime("%Y-%m-%d"); work["year"] = work.entry_ts.dt.year
    absolute = work.base_net_bps.abs().sum()

    def share(column: str) -> float:
        grouped = work.groupby(column, sort=True).base_net_bps.sum().abs()
        return float(grouped.max() / absolute) if absolute else 1.0

    def day_mean(column: str) -> float:
        return float(work.groupby("day", sort=True)[column].mean().mean())

    return {prefix + "symbol_contribution": share("symbol"), prefix + "day_contribution": share("day"),
            prefix + "year_contribution": share("year"), prefix + "gross_mean_bps": day_mean("gross_bps"),
            prefix + "fixed_cost_bps": -14.0,
            prefix + "adverse_exact_funding_mean_bps": day_mean("funding_adverse_exact_bps"),
            prefix + "base_gap_allowance_mean_bps": day_mean("funding_base_gap_cost_bps"),
            prefix + "stress_gap_allowance_mean_bps": day_mean("funding_stress_gap_cost_bps"),
            prefix + "base_net_recomputed_mean_bps": day_mean("base_net_bps"),
            prefix + "stress_net_recomputed_mean_bps": day_mean("stress_net_bps"),
            prefix + "alignment_start_mean_bps": day_mean("base_net_alignment_start_bps"),
            prefix + "alignment_end_mean_bps": day_mean("base_net_alignment_end_bps")}


def build(output: Path) -> None:
    folds = {row["outer_fold_id"]: row for row in json.loads((STAGE19 / "INNER_FOLD_MAP.json").read_text())["outer_folds"]}
    beams = json.loads((RUN / "PHASE3_FROZEN_BEAM_REGISTRY.json").read_text())["beams"]
    development = pd.read_csv(RUN / "PHASE2_DEVELOPMENT_RESPONSE_SURFACE.csv")
    outer = pd.read_csv(RUN / "PHASE4_5_OUTER_ROLLING_RESULTS.csv")
    selected = sorted({cell for beam in beams for cell in beam["selected_cell_ids"]})
    model_ids = sorted({inner["inner_fold_id"] for fold in folds.values() if fold["hypothesis_id"].startswith("KDA02B") for inner in fold["inner_folds"]})
    models = {model_id: load_root(RUN / "staging/development" / model_id / "KDA02B", selected) for model_id in model_ids}
    rows = []
    for beam in beams:
        fold_id = beam["outer_fold_id"]; quarter = fold_id.split(":", 1)[1]
        inner_ids = [row["inner_fold_id"] for row in folds[fold_id]["inner_folds"]]
        dev_frame = pd.concat([models[model_id] for model_id in inner_ids], ignore_index=True)
        outer_root = next((RUN / "staging/outer").glob(f"*/Q_{quarter}/KDA02B"))
        outer_frame = load_root(outer_root, beam["selected_cell_ids"])
        for rank, cell_id in enumerate(beam["selected_cell_ids"], 1):
            dev_row = development.loc[(development.outer_fold_id == fold_id) & (development.cell_id == cell_id)].iloc[0]
            outer_row = outer.loc[(outer.outer_fold_id == fold_id) & (outer.cell_id == cell_id)].iloc[0]
            result = {"family": "KDA02B", "outer_quarter": quarter, "outer_fold_id": fold_id,
                      "beam_rank": rank, "cell_id": cell_id,
                      "canonical_translation_id": dev_row.canonical_translation_id,
                      "freeze_sha256": beam["freeze_sha256"]}
            for column in ["accepted_trade_count", "aggregate_base_net_mean_bps", "aggregate_stress_net_mean_bps",
                           "base_net_median_bps", "median_inner_fold_base_net_mean_bps",
                           "p20_inner_fold_base_net_mean_bps", "cluster_bootstrap_lower_bound_bps",
                           "left_tail_utility_bps", "opportunity_frequency_per_30d", "capital_occupancy",
                           "execution_margin_bps", "symbol_day_year_contribution", "unavailable_inner_fold_count"]:
                result["development_" + column] = dev_row[column]
            for column in ["accepted_trade_count", "aggregate_base_net_mean_bps", "aggregate_stress_net_mean_bps",
                           "base_net_median_bps", "cluster_bootstrap_lower_bound_bps", "left_tail_utility_bps",
                           "opportunity_frequency_per_30d", "capital_occupancy", "execution_margin_bps",
                           "symbol_day_year_contribution"]:
                result["outer_" + column] = outer_row[column]
            result.update(extras(dev_frame.loc[dev_frame.cell_id == cell_id], "development_"))
            result.update(extras(outer_frame.loc[outer_frame.cell_id == cell_id], "outer_"))
            rows.append(result)
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.12g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
