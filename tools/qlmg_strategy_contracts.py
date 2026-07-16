#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FAMILY_BUDGET_GROUPS = {
    "D3_D4_E1": ["D3", "D4", "E1"],
    "A1_A3": ["A1_A3"],
    "A2": ["A2"],
    "F1_G1": ["F1", "G1"],
}

DEFAULT_ALLOCATIONS = {
    "D3_D4_E1": 160,
    "A1_A3": 90,
    "A2": 50,
    "F1_G1": 60,
}


def allocate_sweep_budget(total: int, include_shorts: bool = True) -> dict[str, int]:
    total = int(total)
    groups = dict(DEFAULT_ALLOCATIONS)
    if not include_shorts:
        groups["F1_G1"] = 0
    base_total = sum(groups.values()) or 1
    alloc = {k: int(total * v / base_total) for k, v in groups.items()}
    remainder = total - sum(alloc.values())
    for key in ["D3_D4_E1", "A1_A3", "A2", "F1_G1"]:
        if remainder <= 0:
            break
        if groups.get(key, 0) > 0:
            alloc[key] += 1
            remainder -= 1
    return alloc


def family_contract(family: str) -> dict[str, Any]:
    common = {
        "protected_holdout_start": "2026-01-01T00:00:00Z",
        "allowed_data_end": "2025-12-31T23:59:59Z",
        "no_live_trading": True,
        "no_sealed_validation": True,
        "matched_null_design": "same/neighboring month, same symbol where possible, liquidity tier, volatility, turnover, parent regime, session, listing age",
        "validation_plan": "pre-holdout matched nulls, local neighborhood, purged walk-forward/CPCV, cost/funding/liquidation stress",
        "forbidden_shortcuts": ["future sector membership", "current-active universe leakage", "post-result threshold edits", "final holdout access"],
    }
    specs: dict[str, dict[str, Any]] = {
        "D3": {
            "hypothesis": "Tier C liquidity-shock selloffs rebound after stabilization and micro-reclaim when deleveraging conditions are supportive.",
            "economic_mechanism": "thin perp liquidity overshoots during forced selling; after leverage clears, small-account capacity can exploit short rebound paths.",
            "active_regimes": ["Tier C", "sharp selloff", "price down + OI down or funding reset", "post-shock stabilization", "spread/range proxy normalizes"],
            "reduce_size_regimes": ["parent neutral_down", "funding window", "high bad-wick proxy"],
            "disable_regimes": ["price down + OI up without reclaim", "severe data integrity flags", "protected holdout"],
            "side": "long",
        },
        "D4": {
            "hypothesis": "Price+OI+funding deleveraging reclaim has stronger rebound paths than price-only shocks.",
            "economic_mechanism": "OI collapse and funding reset indicate leverage cleared before reclaim.",
            "active_regimes": ["Tier B/C", "price down hard", "OI down hard", "funding normalized or negative", "reclaim after stabilization"],
            "reduce_size_regimes": ["parent down", "wide spread proxy"],
            "disable_regimes": ["price down + OI up unless separate short-squeeze contract", "missing OI/funding"],
            "side": "long",
        },
        "E1": {
            "hypothesis": "Large drops with OI collapse and funding reset produce tradable post-deleveraging reclaim paths.",
            "economic_mechanism": "liquidation/deleveraging flush removes crowded positioning before mean reversion.",
            "active_regimes": ["Tier A/B primary", "large local-high drawdown", "OI collapse", "funding reset/negative", "reclaim after stabilization"],
            "reduce_size_regimes": ["Tier C", "missing mark path"],
            "disable_regimes": ["first impulse chase", "proxy liquidation only with adverse 1m overlay"],
            "side": "long",
        },
        "A1_A3": {
            "hypothesis": "Liquid continuation breakouts/retests work mainly in supportive parent trend and broad participation regimes.",
            "economic_mechanism": "leaders continue when market participation broadens and leverage is not euphoric.",
            "active_regimes": ["Tier A/B", "BTC/ETH positive or neutral-positive", "breadth expanding", "funding not euphoric", "moderate OI rise"],
            "reduce_size_regimes": ["neutral parent", "high funding percentile"],
            "disable_regimes": ["parent downtrend", "breadth collapse", "crowded/euphoric leverage"],
            "side": "long",
            "event_source_status": "proxy_from_A2_until_true_A1_A3_event_ledger_exists",
        },
        "A2": {
            "hypothesis": "Prior-high proximity momentum may be viable only in positive parent trend and non-crisis volatility regimes.",
            "economic_mechanism": "near-high leaders attract continuation flows when broad market structure supports risk-on behavior.",
            "active_regimes": ["Tier A/B", "parent trend positive", "breadth supportive", "smooth path", "funding not extreme"],
            "reduce_size_regimes": ["volatility high", "funding high"],
            "disable_regimes": ["parent down", "crisis volatility"],
            "side": "long",
        },
        "F1": {
            "hypothesis": "Parabolic blowoff shorts require crowded derivatives and backside confirmation.",
            "economic_mechanism": "late leverage chase unwinds after failed extension.",
            "active_regimes": ["large extension", "funding top decile", "strong OI build", "breadth narrowing", "backside trigger"],
            "reduce_size_regimes": [],
            "disable_regimes": ["anticipatory short without backside confirmation", "short engine or borrow semantics unavailable"],
            "side": "short",
            "status": "contract_only_unless_event_generation_safe",
        },
        "G1": {
            "hypothesis": "Failed continuation breakouts short better in weak parent/narrow breadth/crowded breakout regimes.",
            "economic_mechanism": "crowded breakout attempts fail when market participation does not confirm.",
            "active_regimes": ["failed breakout", "weak or neutral parent", "narrow breadth", "crowded OI/funding into breakout", "retest failure"],
            "reduce_size_regimes": [],
            "disable_regimes": ["strong breadth confirmation", "no retest failure"],
            "side": "short",
            "status": "contract_only_unless_event_generation_safe",
        },
    }
    out = {"family": family, **common, **specs[family]}
    out["sweep_parameter_ranges"] = "bounded deterministic samples only; no full Cartesian grid"
    out["maximum_candidate_budget"] = "allocated by run-level budget contract"
    out["rejection_gates"] = ["fails regime-matched null", "fails cost/funding stress", "liquidation/data quality unacceptable", "validation paths <55% positive", "concentration breach"]
    return out


def write_contracts(root: Path, families: list[str]) -> list[dict[str, Any]]:
    out_dir = root / "contracts/strategy_contracts"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for fam in families:
        contract = family_contract(fam)
        path = out_dir / f"{fam}.json"
        path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append({"family": fam, "path": str(path), "side": contract.get("side"), "status": contract.get("status", contract.get("event_source_status", "implemented_or_proxy")), "active_regimes": ";".join(contract.get("active_regimes", []))})
    return rows
