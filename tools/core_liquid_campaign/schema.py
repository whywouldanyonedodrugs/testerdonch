from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_hash


CAMPAIGN_ID = "kraken_core_liquid_discovery_campaign_003_code_first_stage22"
SCHEMA_VERSION = "core_liquid_family_axis_schema_v1"
TYPED_NULL = None
FAMILY_ORDER = (
    "A4_TSMOM_V7",
    "A1_COMPRESSION_V2",
    "A2_PRIOR_HIGH_RS_CONTEXT_V1",
    "A3_STARTER_RETEST_V3",
    "KDA02B_SURVIVOR_ADJUDICATION_V1",
)
OUTER_FOLDS = ("2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4")


class SchemaError(ValueError):
    pass


ALWAYS: dict[str, Any] = {"op": "always"}


def eq(field: str, value: Any) -> dict[str, Any]:
    return {"op": "eq", "field": field, "value": value}


def ne(field: str, value: Any) -> dict[str, Any]:
    return {"op": "ne", "field": field, "value": value}


def isin(field: str, values: Sequence[Any]) -> dict[str, Any]:
    return {"op": "in", "field": field, "values": list(values)}


def all_of(*args: Mapping[str, Any]) -> dict[str, Any]:
    return {"op": "and", "args": [dict(arg) for arg in args]}


def any_of(*args: Mapping[str, Any]) -> dict[str, Any]:
    return {"op": "or", "args": [dict(arg) for arg in args]}


def evaluate_expression(expression: Mapping[str, Any], values: Mapping[str, Any]) -> bool:
    op = expression["op"]
    if op == "always":
        return True
    if op == "eq":
        return values.get(expression["field"]) == expression["value"]
    if op == "ne":
        return values.get(expression["field"]) != expression["value"]
    if op == "in":
        return values.get(expression["field"]) in expression["values"]
    if op == "and":
        return all(evaluate_expression(item, values) for item in expression["args"])
    if op == "or":
        return any(evaluate_expression(item, values) for item in expression["args"])
    raise SchemaError(f"unknown expression operator: {op!r}")


@dataclass(frozen=True)
class AxisSpec:
    name: str
    value_type: str
    ordered: bool
    allowed_values: tuple[Any, ...] | None
    numeric_bounds: tuple[float, float] | None
    default_policy: str
    active_if: Mapping[str, Any]
    valid_if: Mapping[str, Any]
    formula_id: str
    feature_availability: str
    threshold_population_scope: str
    missingness: str
    economic_identity: bool
    distance_inclusion: bool
    distance_scale: str
    complexity_contribution: float
    control_compatibility: tuple[str, ...]
    serialization: str
    classification: str
    search_new_broad: bool = True

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_values"] = list(self.allowed_values) if self.allowed_values is not None else None
        value["numeric_bounds"] = list(self.numeric_bounds) if self.numeric_bounds is not None else None
        value["control_compatibility"] = list(self.control_compatibility)
        return value


@dataclass(frozen=True)
class FamilySchema:
    family_id: str
    engine_id: str
    event_type: str
    axes: tuple[AxisSpec, ...]
    priority_pairs: tuple[tuple[str, str], ...]
    valid_rules: tuple[Mapping[str, Any], ...]

    @property
    def axis_map(self) -> dict[str, AxisSpec]:
        return {axis.name: axis for axis in self.axes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "engine_id": self.engine_id,
            "event_type": self.event_type,
            "axes": [axis.to_dict() for axis in self.axes],
            "priority_pairs": [list(pair) for pair in self.priority_pairs],
            "valid_rules": [dict(rule) for rule in self.valid_rules],
        }


def axis(
    name: str,
    values: Sequence[Any] | None,
    *,
    value_type: str | None = None,
    ordered: bool = True,
    active_if: Mapping[str, Any] = ALWAYS,
    valid_if: Mapping[str, Any] = ALWAYS,
    formula_id: str,
    availability: str = "completed point-in-time input with feature_available_ts<=decision_ts",
    population: str = "not_applicable",
    missingness: str = "unavailable; no fallback or imputation",
    identity: bool = True,
    distance: bool = True,
    scale: str = "listed_level_index_over_max_index",
    complexity: float = 1.0,
    controls: Sequence[str] = ("main_null", "component_ablation"),
    classification: str = "economic_search_axis",
    search_new_broad: bool = True,
) -> AxisSpec:
    if value_type is None:
        sample = next((item for item in values or () if item is not None), "")
        value_type = "boolean" if isinstance(sample, bool) else "integer" if isinstance(sample, int) else "number" if isinstance(sample, float) else "string"
    bounds = None
    if values and value_type in {"integer", "number"}:
        numeric = [float(item) for item in values]
        bounds = (min(numeric), max(numeric))
    return AxisSpec(
        name=name,
        value_type=value_type,
        ordered=ordered,
        allowed_values=tuple(values) if values is not None else None,
        numeric_bounds=bounds,
        default_policy="prohibited; legacy projection must be explicit",
        active_if=dict(active_if),
        valid_if=dict(valid_if),
        formula_id=formula_id,
        feature_availability=availability,
        threshold_population_scope=population,
        missingness=missingness,
        economic_identity=identity,
        distance_inclusion=distance,
        distance_scale=scale,
        complexity_contribution=complexity,
        control_compatibility=tuple(controls),
        serialization="canonical JSON scalar; inactive is JSON null",
        classification=classification,
        search_new_broad=search_new_broad,
    )


ATR_EXITS = ("ATR_stop_1.5", "ATR_stop_2.5", "ATR_trail_2", "ATR_trail_3", "ATR_trail_4")
TARGET_EXITS = ("ATR_stop_1.5", "ATR_stop_2.5")
CONTEXTS = ("none", "prior_high_RS", "BTC_ETH", "breadth_dispersion", "funding_veto")


def _schemas() -> dict[str, FamilySchema]:
    a4 = FamilySchema(
        "A4_TSMOM_V7", "a4_tsmom_engine_v1", "definition_symbol_episode",
        (
            axis("PIT_liquidity_top_n", (10, 20, 40), formula_id="pit_lagged_liquidity_top_n_v1"),
            axis("annualized_vol_target", ("none", 0.1, 0.2, 0.3), value_type="string_or_number", formula_id="frozen_episode_vol_target_v1"),
            axis("context_overlay", ("none", "BTC_ETH", "breadth_dispersion", "funding_veto"), ordered=False, formula_id="named_context_multiplier_v3"),
            axis("direction", ("long_short", "long_flat", "short_flat"), ordered=False, formula_id="strict_scalar_sign_side_v1"),
            axis("exit", ("time_1d", "time_3d", "time_5d", "time_10d", "signal_reversal", "ATR_trail_2", "ATR_trail_3", "ATR_trail_4"), ordered=False, formula_id="completed_close_next_open_exit_v3"),
            axis("lookback_days", (5, 10, 20, 40, 60, 90, 120, 180), formula_id="tsmom_lookback_v1"),
            axis("path_smoothness_quantile_min", ("none", "q20", "q40", "q60"), formula_id="path_smoothness_gate_v1", population="symbol_side_fold_training"),
            axis("rebalance", ("8h", "1d"), ordered=False, formula_id="utc_rebalance_v1"),
            axis("signal_estimator", ("signed_return", "ema_slope", "breakout_distance_rank", "equal_weight_ensemble"), ordered=False, formula_id="a4_scalar_estimator_v1"),
            axis("vol_window_days", (10, 20, 40, 60), formula_id="completed_daily_volatility_window_v1"),
            axis("ATR_window_days_for_ATR_exits", (10, 20, 40, 60), active_if=isin("exit", ("ATR_trail_2", "ATR_trail_3", "ATR_trail_4")), formula_id="wilder_daily_atr_v1"),
            axis("volatility_estimator", ("close_to_close", "parkinson"), ordered=False, formula_id="a4_volatility_estimator_v1"),
        ),
        (("lookback_days", "signal_estimator"), ("signal_estimator", "volatility_estimator"), ("exit", "ATR_window_days_for_ATR_exits"), ("direction", "context_overlay")),
        (),
    )
    a1 = FamilySchema(
        "A1_COMPRESSION_V2", "a1_compression_engine_v1", "impulse_base_confirmation_episode",
        (
            axis("PIT_liquidity_top_n", (10, 20, 40), formula_id="pit_lagged_liquidity_top_n_v1"),
            axis("base_duration", ("2h", "6h", "12h", "1d", "3d"), formula_id="contiguous_base_clock_v1"),
            axis("confirmation", ("one_close", "two_closes", "close_plus_bounded_15m_delay"), ordered=False, formula_id="a1_confirmation_v1"),
            axis("context_overlay", CONTEXTS, ordered=False, formula_id="named_context_multiplier_v3"),
            axis("contraction_rank_max", ("none", "q20", "q30", "q40", "q50"), formula_id="base_contraction_rank_gate_v1", population="registered contraction_baseline and shape_rank_scope"),
            axis("direction", ("long", "short", "symmetric"), ordered=False, formula_id="a1_side_grammar_v1"),
            axis("exit", ("time_1d", "time_3d", "time_5d", "time_10d", "base_failure", "ATR_stop_1.5", "ATR_stop_2.5", "ATR_trail_2", "ATR_trail_3"), ordered=False, formula_id="completed_close_next_open_exit_v3"),
            axis("fixed_target_R", ("none", 1.5, 3.0), value_type="string_or_number", active_if=isin("exit", TARGET_EXITS), formula_id="fixed_target_from_initial_risk_v1"),
            axis("impulse_rank_min", ("q60", "q70", "q80", "q90"), formula_id="side_signed_impulse_rank_gate_v1", population="registered impulse_rank_scope"),
            axis("impulse_window", ("6h", "12h", "1d", "3d", "7d"), formula_id="a1_impulse_window_v1"),
            axis("smoothness_rank_min", ("none", "q40", "q60", "q80"), formula_id="base_path_smoothness_gate_v1", population="registered shape_rank_scope"),
            axis("ATR_window_days", (10, 20, 40, 60), active_if=isin("exit", ATR_EXITS), formula_id="wilder_daily_atr_v1"),
            axis("contraction_baseline", ("adjacent_equal_duration", "trailing_5x_base_duration"), ordered=False, formula_id="a1_contraction_baseline_v1"),
            axis("impulse_rank_scope", ("symbol_side", "liquidity_decile_side", "global_side"), ordered=False, formula_id="fold_local_threshold_population_v1"),
            axis("shape_rank_scope", ("symbol", "liquidity_decile", "global_PIT"), ordered=False, formula_id="fold_local_threshold_population_v1"),
        ),
        (("impulse_window", "base_duration"), ("contraction_rank_max", "contraction_baseline"), ("impulse_rank_min", "impulse_rank_scope"), ("exit", "fixed_target_R")),
        ({"rule": "fixed_target_requires_atr_stop"},),
    )
    a2 = FamilySchema(
        "A2_PRIOR_HIGH_RS_CONTEXT_V1", "a2_context_engine_v1", "parent_bound_context_overlay",
        (
            axis("parent_binding_mode", ("source_attempt", "beam_slot"), ordered=False, formula_id="a2_parent_binding_mode_v1", classification="fixed_mechanism_semantic", search_new_broad=False),
            axis("parent_family", ("A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"), ordered=False, formula_id="a2_parent_family_v1"),
            axis("parent_source_attempt_id", None, value_type="string", ordered=False, active_if=eq("parent_binding_mode", "source_attempt"), formula_id="exact_source_parent_id_v1", distance=False, scale="excluded_identifier", search_new_broad=False),
            axis("parent_fold_id", OUTER_FOLDS, ordered=True, active_if=eq("parent_binding_mode", "beam_slot"), formula_id="exact_parent_outer_fold_slot_v1"),
            axis("parent_beam_rank", tuple(range(1, 9)), active_if=eq("parent_binding_mode", "beam_slot"), formula_id="exact_parent_beam_rank_v1"),
            axis("BTC_ETH_context", ("none", "trend", "volatility", "drawdown"), ordered=False, formula_id="btc_eth_context_component_v1"),
            axis("RS_lookback_days", (5, 10, 20, 60), active_if=ne("RS_rank", "none"), formula_id="btc_relative_strength_v1"),
            axis("RS_rank", ("none", "continuous", "q40", "q60", "q80"), formula_id="component_threshold_v1", population="registered RS_population_scope"),
            axis("breadth_dispersion", ("none", "breadth", "dispersion", "both"), ordered=False, formula_id="breadth_dispersion_component_v1"),
            axis("overlay_action", ("permission", "linear_size_0_to_1", "tercile_size", "parent_only"), ordered=False, formula_id="a2_overlay_action_v1"),
            axis("prior_high_lookback_days", (20, 60, 120, 250), active_if=any_of(ne("proximity_rank", "none"), ne("reclaim_state", "none")), formula_id="completed_daily_prior_level_v1"),
            axis("proximity_rank", ("none", "continuous", "q40", "q60", "q80"), formula_id="component_threshold_v1", population="parent symbol fold training"),
            axis("reclaim_state", ("none", "continuous_distance", "first_close_above"), ordered=False, formula_id="prior_level_reclaim_v1"),
            axis("ATR_window_days_for_proximity", (10, 20, 40, 60), active_if=any_of(ne("proximity_rank", "none"), ne("reclaim_state", "none")), formula_id="wilder_daily_atr_v1"),
            axis("BTC_ETH_drawdown_lookback_days", (20, 60, 120), active_if=eq("BTC_ETH_context", "drawdown"), formula_id="btc_eth_drawdown_v1"),
            axis("BTC_ETH_trend_pair_days", ("5_20", "20_60", "60_120"), active_if=eq("BTC_ETH_context", "trend"), formula_id="btc_eth_trend_pair_v1"),
            axis("BTC_ETH_volatility_lookback_days", (10, 20, 40, 60), active_if=eq("BTC_ETH_context", "volatility"), formula_id="btc_eth_realized_volatility_v1"),
            axis("RS_population_scope", ("global_PIT", "liquidity_decile", "parent_liquidity_universe"), ordered=False, active_if=ne("RS_rank", "none"), formula_id="fold_local_threshold_population_v1"),
            axis("breadth_return_lookback_days", (5, 20, 60), active_if=isin("breadth_dispersion", ("breadth", "both")), formula_id="pit_breadth_v1"),
            axis("dispersion_return_lookback_days", (5, 20, 60), active_if=isin("breadth_dispersion", ("dispersion", "both")), formula_id="pit_dispersion_v1"),
        ),
        (("parent_family", "overlay_action"), ("RS_rank", "RS_population_scope"), ("proximity_rank", "prior_high_lookback_days"), ("BTC_ETH_context", "overlay_action")),
        ({"rule": "at_least_one_context_component"}, {"rule": "parent_binding_complete"}),
    )
    a3 = FamilySchema(
        "A3_STARTER_RETEST_V3", "a3_starter_retest_engine_v1", "directional_starter_retest_parent",
        (
            axis("PIT_liquidity_top_n", (10, 20, 40), formula_id="pit_lagged_liquidity_top_n_v1"),
            axis("direction", ("long", "short"), ordered=False, formula_id="a3_directional_identity_v1"),
            axis("add_fraction", (0.0, 0.25, 0.5, 0.75), formula_id="a3_add_fraction_v1"),
            axis("add_requires_reclaim", (True,), active_if=ne("add_fraction", 0.0), formula_id="a3_reclaim_required_v1", classification="fixed_mechanism_semantic"),
            axis("breakout_lookback_days", (5, 10, 20, 60), formula_id="completed_daily_prior_level_v1"),
            axis("breakout_rank_min", ("q60", "q70", "q80", "q90"), formula_id="a3_breakout_rank_gate_v1", population="registered breakout_rank_scope"),
            axis("confirmation", ("one_close", "two_closes", "close_plus_15m_delay"), ordered=False, formula_id="a3_confirmation_v1"),
            axis("context_overlay", CONTEXTS, ordered=False, formula_id="named_context_multiplier_v3"),
            axis("exit", ("time_1d", "time_3d", "time_5d", "time_10d", "breakout_failure", "ATR_stop_1.5", "ATR_stop_2.5", "ATR_trail_2", "ATR_trail_3"), ordered=False, formula_id="completed_close_next_open_exit_v3"),
            axis("fixed_target_R", ("none", 1.5, 3.0), value_type="string_or_number", active_if=isin("exit", TARGET_EXITS), formula_id="fixed_target_from_initial_risk_v1"),
            axis("retest_depth_ATR", (0.25, 0.5, 1.0, 1.5), active_if=ne("add_fraction", 0.0), formula_id="a3_retest_band_v1"),
            axis("retest_window", ("6h", "1d", "3d"), active_if=ne("add_fraction", 0.0), formula_id="a3_retest_window_v1"),
            axis("starter_fraction", (0.25, 0.5, 0.75, 1.0), formula_id="a3_starter_fraction_v1"),
            axis("ATR_window_days", (10, 20, 40, 60), active_if=isin("exit", ATR_EXITS), formula_id="wilder_daily_atr_v1"),
            axis("breakout_rank_scope", ("symbol_side", "liquidity_decile_side", "global_side"), ordered=False, formula_id="fold_local_threshold_population_v1"),
        ),
        (("starter_fraction", "add_fraction"), ("retest_depth_ATR", "retest_window"), ("breakout_rank_min", "breakout_rank_scope"), ("direction", "context_overlay")),
        ({"rule": "starter_plus_add_le_one"}, {"rule": "fixed_target_requires_atr_stop"}),
    )
    kda = FamilySchema(
        "KDA02B_SURVIVOR_ADJUDICATION_V1", "kda02b_adjudication_engine_v1", "exact_stage20_cell_adjudication",
        (
            axis("stage20_cell_id", ("KDA02B_009", "KDA02B_011", "KDA02B_031", "KDA02B_032", "KDA02B_033", "KDA02B_034", "KDA02B_035", "KDA02B_036", "KDA02B_045", "KDA02B_047", "KDA02B_048", "KDA02B_058", "KDA02B_060", "KDA02B_079", "KDA02B_081", "KDA02B_082", "KDA02B_083", "KDA02B_084", "KDA02B_093"), ordered=False, formula_id="stage20_exact_cell_identity_v1", classification="fixed_mechanism_semantic", search_new_broad=False),
            axis("adjudication_variant", ("identity_replay", "price_only", "OI_removed", "liquidation_removed", "generic_structure_control", "stress_cost_32bps", "funding_start_alignment", "funding_end_alignment", "funding_zero", "entry_delay_15m", "entry_delay_60m"), ordered=False, formula_id="kda02b_exact_adjudication_variant_v1", search_new_broad=False),
        ),
        (("stage20_cell_id", "adjudication_variant"),),
        (),
    )
    return {item.family_id: item for item in (a4, a1, a2, a3, kda)}


family_schemas = _schemas()


def schema_document() -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "typed_null": None,
        "rank_normalization": {
            "time_series_empirical": "weak_percentile=(strictly_below+equal)/N; Type-7 quantile thresholds; equality passes",
            "cross_sectional": "average_tie_rank normalized as (rank-1)/(N-1); N=1 is unavailable",
            "liquidity_decile": "1+min(9,floor(10*normalized_cross_sectional_rank)); exact 1.0 maps to decile 10; tied ranks share a decile",
        },
        "inactive_axis_contract": "inactive fields serialize as JSON null and are omitted from engine kwargs; provenance is a separate object",
        "removed_axes": {
            "A4_TSMOM_V7.signal_rank_scope": {
                "classification": "removed_as_unsupported",
                "reason": "V4/V5 supplied populations but no scalar transform or economic consumer; Stage 22 forbids inventing one",
            }
        },
        "families": [family_schemas[family].to_dict() for family in FAMILY_ORDER],
    }


def schema_hash() -> str:
    return canonical_hash(schema_document())


def _valid_type(value: Any, declared: str) -> bool:
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared == "string":
        return isinstance(value, str)
    if declared == "string_or_number":
        return (isinstance(value, str) or isinstance(value, (int, float))) and not isinstance(value, bool)
    return False


def normalize_config(family_id: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    if family_id not in family_schemas:
        raise SchemaError(f"unknown family: {family_id}")
    schema = family_schemas[family_id]
    known = set(schema.axis_map)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise SchemaError(f"unknown fields for {family_id}: {unknown}")
    output: dict[str, Any] = {}
    context = dict(raw)
    for spec in schema.axes:
        active = evaluate_expression(spec.active_if, context)
        if not active:
            output[spec.name] = TYPED_NULL
            context[spec.name] = TYPED_NULL
            continue
        if spec.name not in raw:
            raise SchemaError(f"missing active field {family_id}.{spec.name}")
        value = raw[spec.name]
        if value is None:
            raise SchemaError(f"active field cannot be null: {family_id}.{spec.name}")
        if not _valid_type(value, spec.value_type):
            raise SchemaError(f"wrong type for {family_id}.{spec.name}: {value!r}")
        if spec.allowed_values is not None and value not in spec.allowed_values:
            raise SchemaError(f"unsupported value for {family_id}.{spec.name}: {value!r}")
        if not evaluate_expression(spec.valid_if, {**context, spec.name: value}):
            raise SchemaError(f"valid_if failed for {family_id}.{spec.name}: {value!r}")
        output[spec.name] = value
        context[spec.name] = value
    _validate_family_rules(family_id, output)
    return output


def _validate_family_rules(family_id: str, config: Mapping[str, Any]) -> None:
    if family_id in {"A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"}:
        if config.get("fixed_target_R") not in (None, "none") and config.get("exit") not in TARGET_EXITS:
            raise SchemaError("fixed target requires an ATR stop exit")
    if family_id == "A3_STARTER_RETEST_V3":
        if float(config["starter_fraction"]) + float(config["add_fraction"]) > 1.0 + 1e-12:
            raise SchemaError("starter_fraction + add_fraction exceeds 1")
    if family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
        components = (
            config.get("BTC_ETH_context") != "none",
            config.get("RS_rank") != "none",
            config.get("breadth_dispersion") != "none",
            config.get("proximity_rank") != "none",
            config.get("reclaim_state") != "none",
        )
        if not any(components) and config.get("overlay_action") != "parent_only":
            raise SchemaError("A2 requires at least one enabled context component")
        if config["parent_binding_mode"] == "source_attempt":
            parent = str(config["parent_source_attempt_id"])
            if not parent.startswith(str(config["parent_family"]) + ":"):
                raise SchemaError("A2 source parent family and exact parent ID disagree")
        elif config["parent_fold_id"] not in OUTER_FOLDS or not 1 <= int(config["parent_beam_rank"]) <= 8:
            raise SchemaError("A2 beam-slot parent binding is incomplete")


def economic_address(family_id: str, config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    normalized = normalize_config(family_id, config)
    schema = family_schemas[family_id]
    economic = {axis.name: normalized[axis.name] for axis in schema.axes if axis.economic_identity}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "family_id": family_id,
        "economic_config": economic,
    }
    return payload, canonical_hash(payload)


def engine_kwargs(family_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = normalize_config(family_id, config)
    return {key: value for key, value in normalized.items() if value is not None}


def gower_distance(family_id: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = normalize_config(family_id, left)
    b = normalize_config(family_id, right)
    contributions: list[float] = []
    for spec in family_schemas[family_id].axes:
        if not spec.distance_inclusion:
            continue
        av, bv = a[spec.name], b[spec.name]
        if av is None and bv is None:
            continue
        if (av is None) != (bv is None):
            contributions.append(1.0)
            continue
        levels = spec.allowed_values
        if levels is not None and spec.ordered and len(levels) > 1:
            contributions.append(abs(levels.index(av) - levels.index(bv)) / (len(levels) - 1))
        else:
            contributions.append(0.0 if av == bv else 1.0)
    return sum(contributions) / len(contributions) if contributions else 0.0


def complexity(family_id: str, config: Mapping[str, Any]) -> float:
    normalized = normalize_config(family_id, config)
    total = 0.0
    for spec in family_schemas[family_id].axes:
        value = normalized[spec.name]
        if value is None or value in ("none", False, 0, 0.0):
            continue
        total += spec.complexity_contribution
    return total


def search_axes(family_id: str) -> tuple[AxisSpec, ...]:
    return tuple(spec for spec in family_schemas[family_id].axes if spec.search_new_broad and spec.allowed_values is not None)


def generation_levels(family_id: str, spec: AxisSpec) -> tuple[Any, ...]:
    if family_id in {"A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"} and spec.name == "context_overlay":
        return ("none", "funding_veto")
    if family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1" and spec.name == "overlay_action":
        return ("permission", "linear_size_0_to_1", "tercile_size")
    assert spec.allowed_values is not None
    return spec.allowed_values


def mapped_config(family_id: str, coordinates: Sequence[float]) -> dict[str, Any]:
    specs = search_axes(family_id)
    if len(coordinates) != len(specs):
        raise SchemaError(f"coordinate dimension mismatch for {family_id}")
    raw: dict[str, Any] = {}
    for spec, coordinate in zip(specs, coordinates):
        levels = generation_levels(family_id, spec)
        index = min(len(levels) - 1, int(coordinate * len(levels)))
        raw[spec.name] = levels[index]
    if family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
        raw["parent_binding_mode"] = "beam_slot"
    return raw


def marginal_levels(family_id: str) -> Iterable[tuple[str, Any]]:
    for spec in family_schemas[family_id].axes:
        if spec.search_new_broad and spec.allowed_values is not None:
            for value in generation_levels(family_id, spec):
                yield spec.name, value


def baseline_config(family_id: str) -> dict[str, Any]:
    baselines: dict[str, dict[str, Any]] = {
        "A4_TSMOM_V7": {
            "PIT_liquidity_top_n": 20,
            "annualized_vol_target": "none",
            "context_overlay": "none",
            "direction": "long_short",
            "exit": "ATR_trail_2",
            "lookback_days": 20,
            "path_smoothness_quantile_min": "none",
            "rebalance": "1d",
            "signal_estimator": "signed_return",
            "vol_window_days": 20,
            "ATR_window_days_for_ATR_exits": 20,
            "volatility_estimator": "close_to_close",
        },
        "A1_COMPRESSION_V2": {
            "PIT_liquidity_top_n": 20,
            "base_duration": "12h",
            "confirmation": "one_close",
            "context_overlay": "none",
            "contraction_rank_max": "q40",
            "direction": "long",
            "exit": "ATR_stop_1.5",
            "fixed_target_R": "none",
            "impulse_rank_min": "q70",
            "impulse_window": "1d",
            "smoothness_rank_min": "none",
            "ATR_window_days": 20,
            "contraction_baseline": "adjacent_equal_duration",
            "impulse_rank_scope": "symbol_side",
            "shape_rank_scope": "symbol",
        },
        "A2_PRIOR_HIGH_RS_CONTEXT_V1": {
            "parent_binding_mode": "beam_slot",
            "parent_family": "A1_COMPRESSION_V2",
            "parent_fold_id": "2024Q1",
            "parent_beam_rank": 1,
            "BTC_ETH_context": "none",
            "RS_lookback_days": 20,
            "RS_rank": "q40",
            "breadth_dispersion": "none",
            "overlay_action": "permission",
            "prior_high_lookback_days": 60,
            "proximity_rank": "none",
            "reclaim_state": "none",
            "ATR_window_days_for_proximity": 20,
            "BTC_ETH_drawdown_lookback_days": 60,
            "BTC_ETH_trend_pair_days": "20_60",
            "BTC_ETH_volatility_lookback_days": 20,
            "RS_population_scope": "global_PIT",
            "breadth_return_lookback_days": 20,
            "dispersion_return_lookback_days": 20,
        },
        "A3_STARTER_RETEST_V3": {
            "PIT_liquidity_top_n": 20,
            "direction": "long",
            "add_fraction": 0.25,
            "add_requires_reclaim": True,
            "breakout_lookback_days": 20,
            "breakout_rank_min": "q70",
            "confirmation": "one_close",
            "context_overlay": "none",
            "exit": "ATR_stop_1.5",
            "fixed_target_R": "none",
            "retest_depth_ATR": 0.5,
            "retest_window": "1d",
            "starter_fraction": 0.5,
            "ATR_window_days": 20,
            "breakout_rank_scope": "symbol_side",
        },
        "KDA02B_SURVIVOR_ADJUDICATION_V1": {
            "stage20_cell_id": "KDA02B_009",
            "adjudication_variant": "identity_replay",
        },
    }
    return dict(baselines[family_id])


def axis_fixture_config(family_id: str, field: str, value: Any) -> dict[str, Any]:
    config = baseline_config(family_id)
    if family_id == "A4_TSMOM_V7" and field == "ATR_window_days_for_ATR_exits":
        config["exit"] = "ATR_trail_2"
    if family_id == "A1_COMPRESSION_V2" and field in {"ATR_window_days", "fixed_target_R"}:
        config["exit"] = "ATR_stop_1.5"
    if family_id == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
        if field == "parent_source_attempt_id" or (field == "parent_binding_mode" and value == "source_attempt"):
            config["parent_binding_mode"] = "source_attempt"
            config["parent_source_attempt_id"] = "A1_COMPRESSION_V2:anchor_ablation:0001"
        if field in {"parent_fold_id", "parent_beam_rank"}:
            config["parent_binding_mode"] = "beam_slot"
        if field in {"ATR_window_days_for_proximity", "prior_high_lookback_days"}:
            config["proximity_rank"] = "q40"
        if field == "BTC_ETH_drawdown_lookback_days":
            config["BTC_ETH_context"] = "drawdown"
        if field == "BTC_ETH_trend_pair_days":
            config["BTC_ETH_context"] = "trend"
        if field == "BTC_ETH_volatility_lookback_days":
            config["BTC_ETH_context"] = "volatility"
        if field in {"RS_lookback_days", "RS_population_scope"}:
            config["RS_rank"] = "q40"
        if field == "breadth_return_lookback_days":
            config["breadth_dispersion"] = "breadth"
        if field == "dispersion_return_lookback_days":
            config["breadth_dispersion"] = "dispersion"
        if field == "RS_rank" and value == "none":
            config["proximity_rank"] = "q40"
        if field in {"BTC_ETH_context", "breadth_dispersion", "proximity_rank", "reclaim_state"} and value == "none":
            config["RS_rank"] = "q40"
    if family_id == "A3_STARTER_RETEST_V3":
        if field == "add_fraction" and float(value) > 0.5:
            config["starter_fraction"] = 0.25
        if field in {"retest_depth_ATR", "retest_window", "add_requires_reclaim"}:
            config["add_fraction"] = 0.25
        if field in {"ATR_window_days", "fixed_target_R"}:
            config["exit"] = "ATR_stop_1.5"
        if field == "starter_fraction" and float(value) > 0.25:
            config["add_fraction"] = 0.0
    config[field] = value
    return config
