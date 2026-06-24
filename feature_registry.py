from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    builder: str
    source: str
    availability: str
    test_type: str


def _mk_many(
    names: List[str],
    *,
    family: str,
    builder: str,
    source: str,
    availability: str,
    test_type: str,
) -> List[FeatureSpec]:
    return [
        FeatureSpec(
            name=n,
            family=family,
            builder=builder,
            source=source,
            availability=availability,
            test_type=test_type,
        )
        for n in names
    ]


_REGISTRY: List[FeatureSpec] = []

_REGISTRY += _mk_many(
    ["atr_at_entry"],
    family="decision_bar",
    builder="live.live_trader/_atr_pre_at_ts",
    source="5m/ATR timeframe bars",
    availability="decision bar close only",
    test_type="decision_bar_close",
)

_REGISTRY += _mk_many(
    [
        "atr_1h",
        "rsi_1h",
        "adx_1h",
        "eth_macd_both_pos_4h",
        "asset_rsi_15m",
        "asset_rsi_4h",
        "asset_macd_line_1h",
        "asset_macd_signal_1h",
        "asset_macd_hist_1h",
        "asset_macd_slope_1h",
        "asset_macd_hist_slope_1h",
        "asset_macd_line_4h",
        "asset_macd_signal_4h",
        "asset_macd_hist_4h",
        "asset_macd_slope_4h",
        "asset_macd_hist_slope_4h",
        "asset_vol_1h",
        "asset_vol_4h",
        "eth_macd_line_4h",
        "eth_macd_signal_4h",
        "eth_macd_hist_4h",
        "eth_macd_hist_slope_4h",
        "eth_macd_hist_slope_1h",
        "markov_prob_up_4h",
        "markov_state_4h",
    ],
    family="htf_indicator",
    builder="indicators.py / scout.py / live.feature_builder / regime_detector.py",
    source="15m/1h/4h close-labeled bars",
    availability="latest fully closed source bar only",
    test_type="htf_last_closed",
)

_REGISTRY += _mk_many(
    [
        "vol_prob_low_1d",
        "regime_code_1d",
        "regime_up",
        "trend_regime_1d",
        "vol_regime_1d",
        "regime_1d",
        "daily_regime_str_1d",
        "trend_regime_code_1d",
        "vol_regime_code_1d",
        "btc_vol_regime_level",
        "btc_trend_slope",
        "eth_vol_regime_level",
        "eth_trend_slope",
        "btcusdt_vol_regime_level",
        "btcusdt_trend_slope",
        "ethusdt_vol_regime_level",
        "ethusdt_trend_slope",
        "btc_ret_24h",
        "btc_ret_72h",
        "liq30_eqw_ret_24h",
        "liq30_eqw_ret_72h",
        "liq30_pos_share_24h",
        "liq30_gt2pct_share_24h",
        "liq30_above_1dma_share",
        "liq30_new_20d_high_share",
        "liq30_member_count",
        "liq30_breadth_accel_24h",
        "liq30_minus_btc_24h",
        "liq30_minus_btc_72h",
        "liq30_macd4h_pos_share",
        "rs_pct",
        "usd_vol_med_24h",
    ],
    family="daily_snapshot",
    builder="scout.py / regime_detector.py / tools.run_v3_frozen_oos",
    source="completed daily snapshots and daily-ranked cross-section",
    availability="latest fully closed daily snapshot only",
    test_type="daily_snapshot",
)

_REGISTRY += _mk_many(
    [
        "vol_mult",
        "atr_pct",
        "days_since_prev_break",
        "consolidation_range_atr",
        "prior_1d_ret",
        "rv_3d",
        "don_break_level",
        "donch_break_level",
        "don_break_len",
        "donch_break_len",
        "don_dist_atr",
        "gap_from_1d_ma",
        "prebreak_congestion",
    ],
    family="entry_quality",
    builder="scout.py / live.feature_builder / fill_entry_quality_features.py",
    source="5m + higher timeframe completed bars",
    availability="decision bar close only; higher timeframe inputs last-closed only",
    test_type="truncation_equivalence",
)

_REGISTRY += _mk_many(
    [
        "oi_level",
        "oi_notional_est",
        "oi_pct_1h",
        "oi_pct_4h",
        "oi_pct_1d",
        "oi_z_7d",
        "oi_chg_norm_vol_1h",
        "oi_price_div_1h",
        "funding_rate",
        "funding_abs",
        "funding_z_7d",
        "funding_rollsum_3d",
        "funding_oi_div",
        "est_leverage",
        "btc_funding_rate",
        "btc_oi_z_7d",
        "eth_funding_rate",
        "eth_oi_z_7d",
        "btcusdt_funding_rate",
        "btcusdt_oi_z_7d",
        "ethusdt_funding_rate",
        "ethusdt_oi_z_7d",
        "crowded_long",
        "crowded_short",
        "crowd_side",
    ],
    family="exogenous_oi_funding",
    builder="live.oi_funding / scout.py / live.live_trader",
    source="open interest and funding event series",
    availability="available on or after publish timestamp; funding delayed by one 5m bar by default",
    test_type="exogenous_publish",
)

_REGISTRY += _mk_many(
    [
        "funding_regime_code",
        "oi_regime_code",
        "btc_risk_regime_code",
        "risk_on",
        "risk_on_1",
        "S1_regime_code_1d",
        "S2_markov_x_vol1d",
        "S3_funding_x_oi",
        "S4_crowd_x_trend1d",
        "S5_btcRisk_x_regimeUp",
        "S6_fresh_x_compress",
    ],
    family="derived_regime_set",
    builder="live.live_trader._augment_meta_with_regime_sets / research.02_make_regimes",
    source="already-causal regime, OI/funding, and freshness inputs",
    availability="same decision timestamp as upstream causal inputs",
    test_type="derived_from_causal",
)

_REGISTRY += _mk_many(
    [
        "recent_winrate_20",
        "recent_winrate_50",
        "recent_winrate_ewm_20",
    ],
    family="trade_history",
    builder="backtester.py / backtester_b_gemin.py",
    source="strictly earlier closed trades",
    availability="only trades closed before decision timestamp",
    test_type="trade_history",
)

_REGISTRY += _mk_many(
    [
        "asset4h_share_at_timestamp",
    ],
    family="cross_sectional_same_ts",
    builder="tools.run_v3_frozen_oos",
    source="same-timestamp aggregation of already-causal per-symbol 4h state",
    availability="same decision timestamp as upstream causal inputs",
    test_type="same_timestamp_cross_sectional",
)


ACTIVE_FEATURE_REGISTRY: List[Dict[str, str]] = [asdict(x) for x in _REGISTRY]


def registry_feature_names() -> List[str]:
    return [row["name"] for row in ACTIVE_FEATURE_REGISTRY]


def registry_rows_by_test_type() -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {}
    for row in ACTIVE_FEATURE_REGISTRY:
        out.setdefault(row["test_type"], []).append(row)
    return out
