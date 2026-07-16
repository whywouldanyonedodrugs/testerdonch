from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


MODEL_VERSION = "kraken_shared_funding_imputation_v1_20260711"
PROTECTED_TRAIN_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
IMPUTED_CAP = "funding_imputed_train_screen_cap"
MODEL_NAMES = (
    "global_robust_median",
    "liquidity_tier_robust_median",
    "symbol_shrunk_to_liquidity_tier",
    "symbol_robust_mean_shrunk_to_liquidity_tier",
    "symbol_robust_location_shrunk_to_liquidity_tier",
    "symbol_shrunk_to_liquidity_tier_bias_calibrated",
    "liquidity_tier_parent_regime_vol_return_state",
)
FORBIDDEN_OUTCOME_FEATURES = {
    "gross_R",
    "net_R",
    "raw_gross_R",
    "raw_net_R",
    "scaled_gross_R",
    "scaled_net_R",
    "mae_R",
    "mfe_R",
    "candidate_performance",
    "trade_pnl",
    "strategy_return",
}
FEATURE_COLUMNS = (
    "symbol",
    "timestamp",
    "liquidity_tier",
    "parent_regime",
    "realized_vol_state",
    "return_state",
)


def canonical_frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"empty").hexdigest()
    work = frame.copy()
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]):
            work[col] = pd.to_datetime(work[col], utc=True, errors="coerce").astype(str)
    columns = sorted(work.columns)
    work = work.reindex(columns=columns).sort_values(columns, kind="mergesort").reset_index(drop=True)
    return hashlib.sha256(work.to_csv(index=False).encode("utf-8")).hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_no_strategy_outcome_features(columns: Iterable[str]) -> None:
    found = sorted(set(columns) & FORBIDDEN_OUTCOME_FEATURES)
    if found:
        raise ValueError(f"strategy outcome leakage fields are forbidden: {found}")


def robust_median(values: pd.Series, default: float = 0.0) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.median()) if len(clean) else float(default)


def robust_winsorized_mean(values: pd.Series, default: float = 0.0) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if not len(clean):
        return float(default)
    lower, upper = clean.quantile([0.01, 0.99])
    return float(clean.clip(lower=lower, upper=upper).mean())


@dataclass(frozen=True)
class FundingModel:
    name: str
    global_median: float
    tier_medians: Mapping[str, float]
    symbol_medians: Mapping[str, float]
    symbol_counts: Mapping[str, int]
    state_medians: Mapping[str, float]
    state_counts: Mapping[str, int]
    aggregate_bias_offset: float = 0.0
    shrinkage_observations: int = 168
    minimum_state_observations: int = 96

    @property
    def model_hash(self) -> str:
        return stable_json_hash(
            {
                "name": self.name,
                "global_median": self.global_median,
                "tier_medians": dict(self.tier_medians),
                "symbol_medians": dict(self.symbol_medians),
                "symbol_counts": dict(self.symbol_counts),
                "state_medians": dict(self.state_medians),
                "state_counts": dict(self.state_counts),
                "aggregate_bias_offset": self.aggregate_bias_offset,
                "shrinkage_observations": self.shrinkage_observations,
                "minimum_state_observations": self.minimum_state_observations,
                "version": MODEL_VERSION,
            }
        )


def _state_key(frame: pd.DataFrame) -> pd.Series:
    fields = ["liquidity_tier", "parent_regime", "realized_vol_state", "return_state"]
    values = frame.reindex(columns=fields, fill_value="unknown").fillna("unknown").astype(str)
    return values[fields[0]].str.cat([values[field] for field in fields[1:]], sep="|")


def fit_funding_model(train: pd.DataFrame, name: str) -> FundingModel:
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown funding model: {name}")
    assert_no_strategy_outcome_features(train.columns)
    required = {"symbol", "relativeFundingRate"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"missing model fields: {sorted(missing)}")
    work = train.copy()
    work["relativeFundingRate"] = pd.to_numeric(work["relativeFundingRate"], errors="coerce")
    work = work.dropna(subset=["relativeFundingRate"])
    if work.empty:
        raise ValueError("cannot fit funding model without exact rates")
    work["liquidity_tier"] = work.get("liquidity_tier", "unknown").fillna("unknown").astype(str)
    global_median = robust_median(work["relativeFundingRate"])
    if name == "symbol_robust_mean_shrunk_to_liquidity_tier":
        tier = work.groupby("liquidity_tier", sort=True)["relativeFundingRate"].agg(robust_winsorized_mean).to_dict()
    elif name == "symbol_robust_location_shrunk_to_liquidity_tier":
        tier_groups = work.groupby("liquidity_tier", sort=True)["relativeFundingRate"]
        tier_median = tier_groups.median()
        tier_mean = tier_groups.agg(robust_winsorized_mean)
        tier = (0.5 * tier_median + 0.5 * tier_mean).to_dict()
    else:
        tier = work.groupby("liquidity_tier", sort=True)["relativeFundingRate"].median().to_dict() if name != "global_robust_median" else {}
    symbol_medians: dict[str, float] = {}
    symbol_counts: dict[str, int] = {}
    if name in {"symbol_shrunk_to_liquidity_tier", "symbol_robust_mean_shrunk_to_liquidity_tier", "symbol_robust_location_shrunk_to_liquidity_tier", "symbol_shrunk_to_liquidity_tier_bias_calibrated"}:
        symbol_groups = work.groupby("symbol", sort=True)["relativeFundingRate"]
        if name == "symbol_robust_mean_shrunk_to_liquidity_tier":
            symbol_medians = symbol_groups.agg(robust_winsorized_mean).to_dict()
        elif name == "symbol_robust_location_shrunk_to_liquidity_tier":
            symbol_medians = (0.5 * symbol_groups.median() + 0.5 * symbol_groups.agg(robust_winsorized_mean)).to_dict()
        else:
            symbol_medians = symbol_groups.median().to_dict()
        symbol_counts = symbol_groups.size().astype(int).to_dict()
    state_medians: dict[str, float] = {}
    state_counts: dict[str, int] = {}
    if name == "liquidity_tier_parent_regime_vol_return_state":
        work["state_key"] = _state_key(work)
        state_groups = work.groupby("state_key", sort=True)["relativeFundingRate"]
        state_medians = state_groups.median().to_dict()
        state_counts = state_groups.size().astype(int).to_dict()
    fitted = FundingModel(
        name=name,
        global_median=global_median,
        tier_medians={str(k): float(v) for k, v in tier.items()},
        symbol_medians={str(k): float(v) for k, v in symbol_medians.items()},
        symbol_counts={str(k): int(v) for k, v in symbol_counts.items()},
        state_medians={str(k): float(v) for k, v in state_medians.items()},
        state_counts={str(k): int(v) for k, v in state_counts.items()},
    )
    if name == "symbol_shrunk_to_liquidity_tier_bias_calibrated":
        raw = predict_funding_model(fitted, work)
        lower, upper = work["relativeFundingRate"].quantile([0.01, 0.99])
        robust_target_mean = float(work["relativeFundingRate"].clip(lower=lower, upper=upper).mean())
        # A fixed half-correction keeps the robust location estimator dominant while
        # addressing persistent signed aggregate-cost bias without fold tuning.
        offset = 0.5 * (robust_target_mean - float(raw.mean()))
        fitted = FundingModel(**{**fitted.__dict__, "aggregate_bias_offset": offset})
    return fitted


def predict_funding_model(model: FundingModel, frame: pd.DataFrame) -> pd.Series:
    assert_no_strategy_outcome_features(frame.columns)
    index = frame.index
    tier = frame.get("liquidity_tier", pd.Series("unknown", index=index)).fillna("unknown").astype(str)
    tier_prediction = tier.map(model.tier_medians).fillna(model.global_median).astype(float)
    if model.name == "global_robust_median":
        return pd.Series(model.global_median, index=index, dtype=float)
    if model.name == "liquidity_tier_robust_median":
        return tier_prediction
    if model.name in {"symbol_shrunk_to_liquidity_tier", "symbol_robust_mean_shrunk_to_liquidity_tier", "symbol_robust_location_shrunk_to_liquidity_tier", "symbol_shrunk_to_liquidity_tier_bias_calibrated"}:
        symbol = frame["symbol"].astype(str)
        med = symbol.map(model.symbol_medians)
        count = symbol.map(model.symbol_counts).fillna(0.0).astype(float)
        weight = count / (count + float(model.shrinkage_observations))
        prediction = med.fillna(tier_prediction) * weight + tier_prediction * (1.0 - weight)
        return prediction + float(model.aggregate_bias_offset)
    state = _state_key(frame)
    state_prediction = state.map(model.state_medians)
    state_count = state.map(model.state_counts).fillna(0).astype(int)
    return state_prediction.where(state_count >= model.minimum_state_observations, tier_prediction).fillna(tier_prediction).astype(float)


def prediction_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    y = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    valid = y.notna() & p.notna()
    y = y[valid].astype(float)
    p = p[valid].astype(float)
    if not len(y):
        return {"rows": 0, "mae": np.nan, "rmse": np.nan, "median_absolute_error": np.nan, "aggregate_bias": np.nan, "aggregate_bias_abs_rate_share": np.nan, "positive_cost_bias": np.nan}
    error = p - y
    abs_rate_sum = float(y.abs().sum())
    positive_actual = float(y.clip(lower=0).sum())
    return {
        "rows": int(len(y)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "median_absolute_error": float(error.abs().median()),
        "aggregate_bias": float(error.sum()),
        "aggregate_bias_abs_rate_share": float(error.sum() / abs_rate_sum) if abs_rate_sum else 0.0,
        "positive_cost_bias": float((p.clip(lower=0).sum() - positive_actual) / positive_actual) if positive_actual else 0.0,
    }


def expanding_blocked_time_validation(frame: pd.DataFrame, minimum_train_months: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["year_month"] = work["timestamp"].dt.strftime("%Y-%m")
    months = sorted(work["year_month"].dropna().unique())
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for idx in range(minimum_train_months, len(months)):
        test_month = months[idx]
        train_months = months[:idx]
        train = work[work["year_month"].isin(train_months)]
        test = work[work["year_month"] == test_month]
        if train.empty or test.empty:
            continue
        for name in MODEL_NAMES:
            model = fit_funding_model(train, name)
            pred = predict_funding_model(model, test)
            metrics = prediction_metrics(test["relativeFundingRate"], pred)
            rows.append({"validation": "blocked_expanding_time", "fold": test_month, "train_start": train["timestamp"].min(), "train_end": train["timestamp"].max(), "test_start": test["timestamp"].min(), "test_end": test["timestamp"].max(), "model": name, **metrics})
            predictions.append(pd.DataFrame({"row_id": test.index, "model": name, "actual": test["relativeFundingRate"].values, "predicted": pred.values, "fold": test_month, "validation": "blocked_expanding_time"}))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def leave_group_out_validation(frame: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for group in sorted(frame[group_col].dropna().astype(str).unique()):
        test_mask = frame[group_col].astype(str) == group
        train = frame[~test_mask]
        test = frame[test_mask]
        if train.empty or test.empty:
            continue
        for name in MODEL_NAMES:
            model = fit_funding_model(train, name)
            pred = predict_funding_model(model, test)
            metrics = prediction_metrics(test["relativeFundingRate"], pred)
            rows.append({"validation": f"leave_{group_col}_out", "fold": group, "model": name, **metrics})
            predictions.append(pd.DataFrame({"row_id": test.index, "model": name, "actual": test["relativeFundingRate"].values, "predicted": pred.values, "fold": group, "validation": f"leave_{group_col}_out"}))
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()


def select_central_model(blocked_predictions: pd.DataFrame, leave_symbol_predictions: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        blocked = blocked_predictions[blocked_predictions["model"] == name]
        leave_symbol = leave_symbol_predictions[leave_symbol_predictions["model"] == name]
        bm = prediction_metrics(blocked["actual"], blocked["predicted"])
        sm = prediction_metrics(leave_symbol["actual"], leave_symbol["predicted"])
        comparison_rows.append({"model": name, "complexity_rank": MODEL_NAMES.index(name), **{f"blocked_{k}": v for k, v in bm.items()}, **{f"leave_symbol_{k}": v for k, v in sm.items()}})
    comparison = pd.DataFrame(comparison_rows)
    baseline = comparison[comparison["model"] == "global_robust_median"].iloc[0]
    chosen = "global_robust_median"
    for name in MODEL_NAMES[1:]:
        row = comparison[comparison["model"] == name].iloc[0]
        improves_error = float(row["blocked_mae"]) <= float(baseline["blocked_mae"]) * 0.9975
        symbol_safe = float(row["leave_symbol_mae"]) <= float(baseline["leave_symbol_mae"]) * 1.02
        bias_safe = abs(float(row["blocked_aggregate_bias_abs_rate_share"])) <= max(abs(float(baseline["blocked_aggregate_bias_abs_rate_share"])) * 0.95, 0.01)
        if improves_error and symbol_safe and bias_safe:
            chosen = name
            break
    comparison["selected_central_model"] = comparison["model"] == chosen
    comparison["selection_rule"] = "simplest_model_with_0.25pct_blocked_mae_improvement_lso_mae_within_2pct_and_aggregate_bias_improved_else_global_median"
    return chosen, comparison


def build_funding_scenarios(
    required_boundaries: pd.DataFrame,
    exact_rates: pd.DataFrame,
    features: pd.DataFrame,
    model: FundingModel,
    residual_absolute_quantiles: tuple[float, float],
) -> pd.DataFrame:
    assert_no_strategy_outcome_features(features.columns)
    keys = ["symbol", "timestamp"]
    panel = required_boundaries[keys].drop_duplicates().copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    feature_cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    panel = panel.merge(features[feature_cols].drop_duplicates(keys), on=keys, how="left", validate="one_to_one")
    exact = exact_rates[keys + ["relativeFundingRate"]].drop_duplicates(keys)
    panel = panel.merge(exact, on=keys, how="left", validate="one_to_one")
    predicted = predict_funding_model(model, panel)
    is_exact = panel["relativeFundingRate"].notna()
    q75, q95 = map(float, residual_absolute_quantiles)
    panel["funding_rate_central"] = panel["relativeFundingRate"].where(is_exact, predicted)
    panel["funding_rate_conservative"] = panel["relativeFundingRate"].where(is_exact, predicted + q75)
    panel["funding_rate_severe"] = panel["relativeFundingRate"].where(is_exact, predicted + q95)
    panel["funding_rate_conservative_short"] = panel["relativeFundingRate"].where(is_exact, predicted - q75)
    panel["funding_rate_severe_short"] = panel["relativeFundingRate"].where(is_exact, predicted - q95)
    panel["funding_rate_source"] = np.where(is_exact, "exact_relativeFundingRate", f"imputed_{model.name}")
    panel["confidence_tier"] = np.where(is_exact, "exact", "low_train_screen_imputation")
    panel["funding_exact"] = is_exact
    panel["funding_imputed"] = ~is_exact
    panel["model_version"] = MODEL_VERSION
    panel["model_hash"] = model.model_hash
    panel["label_cap_reason"] = np.where(is_exact, "", IMPUTED_CAP)
    panel["funding_gate_eligible"] = is_exact
    panel["funding_gate_use_policy"] = "exact_rows_only_imputation_forbidden_for_gate_activation"
    return panel.sort_values(keys, kind="mergesort").reset_index(drop=True)


def exact_rows_unchanged_audit(panel: pd.DataFrame) -> pd.DataFrame:
    exact = panel[panel["funding_exact"]].copy()
    mismatch = ~np.isclose(
        pd.to_numeric(exact["relativeFundingRate"], errors="coerce"),
        pd.to_numeric(exact["funding_rate_central"], errors="coerce"),
        rtol=0.0,
        atol=0.0,
        equal_nan=False,
    )
    return pd.DataFrame(
        [{
            "audit": "exact_relativeFundingRate_unchanged",
            "exact_rows": int(len(exact)),
            "mismatch_count": int(mismatch.sum()),
            "pass": bool(not mismatch.any()),
        }]
    )
