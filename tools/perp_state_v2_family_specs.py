from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from tools.perp_state_v2 import ContextGroupRequirement, ContextJoinContract


PHASE_AUGMENT_V1 = "phase1_augment_v1_survivors"
PHASE_RETEST = "phase2_sidecar_retest"
PHASE_SIDECAR_NATIVE = "phase3_sidecar_native"
PHASE_MICROSTRUCTURE = "phase4_microstructure_timing"

FAMILY_PANIC_RECLAIM_V2 = "panic_deleveraging_reclaim_v2"
FAMILY_WASHED_REVERSAL_V2 = "washed_out_crowding_reversal_v2"
FAMILY_BREADTH_FOLLOWER_V2 = "breadth_follower_catchup_v2"
FAMILY_CHEAP_STRENGTH_V2 = "cheap_strength_funding_discount_v2"
FAMILY_UNDEROWNED_TREND_V2 = "underowned_trend_continuation_v2"
FAMILY_SECOND_LEG_V2 = "second_leg_rerisking_v2"
FAMILY_MARK_INDEX_DISLOCATION = "mark_index_dislocation_resolution_v1"
FAMILY_PREMIUM_COMPRESSION = "premium_compression_continuation_v1"
FAMILY_LSR_WASHOUT = "lsr_washout_reversal_v1"
FAMILY_ABSORPTION_PANIC = "absorption_confirmed_panic_reclaim_v1"
FAMILY_TRADE_EXHAUSTION = "trade_imbalance_exhaustion_v1"


@dataclass(frozen=True)
class PerpStateV2FamilySpec:
    family: str
    phase: str
    required_context_groups: tuple[str, ...]
    freshness: Mapping[str, pd.Timedelta]
    trigger_window_hours: int
    state_consistency: str
    requires_orderbook: bool = False
    requires_public_trades: bool = False
    min_symbol_month_coverage: float = 0.95

    def context_contract(self) -> ContextJoinContract:
        requirements = tuple(
            ContextGroupRequirement(group=g, max_staleness=self.freshness.get(g, pd.Timedelta(0)))
            for g in self.required_context_groups
        )
        return ContextJoinContract(
            family=self.family,
            requirements=requirements,
            min_symbol_month_coverage=float(self.min_symbol_month_coverage),
        )


_DEFAULT_EXACT = pd.Timedelta(0)
_LSR_STEPWISE_5M = pd.Timedelta(minutes=5)

FAMILY_SPECS: dict[str, PerpStateV2FamilySpec] = {
    FAMILY_PANIC_RECLAIM_V2: PerpStateV2FamilySpec(
        family=FAMILY_PANIC_RECLAIM_V2,
        phase=PHASE_AUGMENT_V1,
        required_context_groups=("mark", "index", "premium", "lsr"),
        freshness={"mark": _DEFAULT_EXACT, "index": _DEFAULT_EXACT, "premium": _DEFAULT_EXACT, "lsr": _DEFAULT_EXACT},
        trigger_window_hours=12,
        state_consistency="panic reclaim must show prior dislocation/premium damage or non-crowded LSR state",
    ),
    FAMILY_WASHED_REVERSAL_V2: PerpStateV2FamilySpec(
        family=FAMILY_WASHED_REVERSAL_V2,
        phase=PHASE_AUGMENT_V1,
        required_context_groups=("premium", "lsr"),
        freshness={"premium": _DEFAULT_EXACT, "lsr": _DEFAULT_EXACT},
        trigger_window_hours=24,
        state_consistency="washed reversal must show damaged premium or washed/non-crowded LSR state",
    ),
    FAMILY_BREADTH_FOLLOWER_V2: PerpStateV2FamilySpec(
        family=FAMILY_BREADTH_FOLLOWER_V2,
        phase=PHASE_AUGMENT_V1,
        required_context_groups=("premium", "lsr"),
        freshness={"premium": _DEFAULT_EXACT, "lsr": _DEFAULT_EXACT},
        trigger_window_hours=72,
        state_consistency="breadth follower must not be rich-premium and crowded-long at entry",
    ),
    FAMILY_CHEAP_STRENGTH_V2: PerpStateV2FamilySpec(
        family=FAMILY_CHEAP_STRENGTH_V2,
        phase=PHASE_RETEST,
        required_context_groups=("mark", "index", "premium"),
        freshness={"mark": _DEFAULT_EXACT, "index": _DEFAULT_EXACT, "premium": _DEFAULT_EXACT},
        trigger_window_hours=72,
        state_consistency="cheap strength must occur with non-rich premium/mark-index state",
    ),
    FAMILY_UNDEROWNED_TREND_V2: PerpStateV2FamilySpec(
        family=FAMILY_UNDEROWNED_TREND_V2,
        phase=PHASE_RETEST,
        required_context_groups=("premium", "lsr"),
        freshness={"premium": _DEFAULT_EXACT, "lsr": _DEFAULT_EXACT},
        trigger_window_hours=72,
        state_consistency="underowned trend must occur with cooled premium or non-crowded LSR",
    ),
    FAMILY_SECOND_LEG_V2: PerpStateV2FamilySpec(
        family=FAMILY_SECOND_LEG_V2,
        phase=PHASE_RETEST,
        required_context_groups=("premium", "lsr"),
        freshness={"premium": _DEFAULT_EXACT, "lsr": _DEFAULT_EXACT},
        trigger_window_hours=24,
        state_consistency="second leg must show premium/LSR normalization after stress",
    ),
    FAMILY_MARK_INDEX_DISLOCATION: PerpStateV2FamilySpec(
        family=FAMILY_MARK_INDEX_DISLOCATION,
        phase=PHASE_SIDECAR_NATIVE,
        required_context_groups=("mark", "index"),
        freshness={"mark": _DEFAULT_EXACT, "index": _DEFAULT_EXACT},
        trigger_window_hours=12,
        state_consistency="mark-index dislocation family must show prior mark/index spread dislocation and normalization",
    ),
    FAMILY_PREMIUM_COMPRESSION: PerpStateV2FamilySpec(
        family=FAMILY_PREMIUM_COMPRESSION,
        phase=PHASE_SIDECAR_NATIVE,
        required_context_groups=("premium",),
        freshness={"premium": _DEFAULT_EXACT},
        trigger_window_hours=72,
        state_consistency="premium compression continuation must occur after premium compression/normalization",
    ),
    FAMILY_LSR_WASHOUT: PerpStateV2FamilySpec(
        family=FAMILY_LSR_WASHOUT,
        phase=PHASE_SIDECAR_NATIVE,
        required_context_groups=("lsr",),
        freshness={"lsr": _DEFAULT_EXACT},
        trigger_window_hours=24,
        state_consistency="LSR washout reversal must occur in washed-out or non-crowded LSR state",
    ),
    FAMILY_ABSORPTION_PANIC: PerpStateV2FamilySpec(
        family=FAMILY_ABSORPTION_PANIC,
        phase=PHASE_MICROSTRUCTURE,
        required_context_groups=("mark", "index", "premium", "lsr"),
        freshness={"mark": _DEFAULT_EXACT, "index": _DEFAULT_EXACT, "premium": _DEFAULT_EXACT, "lsr": _DEFAULT_EXACT},
        trigger_window_hours=12,
        state_consistency="absorption panic must combine panic state with valid orderbook/trade absorption evidence",
        requires_orderbook=True,
        requires_public_trades=True,
    ),
    FAMILY_TRADE_EXHAUSTION: PerpStateV2FamilySpec(
        family=FAMILY_TRADE_EXHAUSTION,
        phase=PHASE_MICROSTRUCTURE,
        required_context_groups=("premium",),
        freshness={"premium": _DEFAULT_EXACT},
        trigger_window_hours=4,
        state_consistency="trade exhaustion must show sell imbalance exhaustion and premium damage/normalization",
        requires_public_trades=True,
    ),
}

PHASE_ORDER = (PHASE_AUGMENT_V1, PHASE_RETEST, PHASE_SIDECAR_NATIVE, PHASE_MICROSTRUCTURE)
FAMILY_ORDER = tuple(FAMILY_SPECS.keys())


def selected_family_specs(family_filter: Sequence[str] | None = None, phase_filter: Sequence[str] | None = None) -> dict[str, PerpStateV2FamilySpec]:
    allow_family = set(family_filter or FAMILY_ORDER)
    allow_phase = set(phase_filter or PHASE_ORDER)
    unknown = sorted(allow_family.difference(FAMILY_SPECS))
    if unknown:
        raise ValueError(f"unknown perp-state v2 families requested: {unknown}")
    return {name: spec for name, spec in FAMILY_SPECS.items() if name in allow_family and spec.phase in allow_phase}


def context_contracts_for_specs(specs: Mapping[str, PerpStateV2FamilySpec]) -> dict[str, ContextJoinContract]:
    return {name: spec.context_contract() for name, spec in specs.items()}


def family_specs_hash(specs: Mapping[str, PerpStateV2FamilySpec]) -> str:
    payload: dict[str, Any] = {}
    for name, spec in sorted(specs.items()):
        payload[name] = {
            "phase": spec.phase,
            "required_context_groups": list(spec.required_context_groups),
            "freshness_seconds": {g: int(pd.to_timedelta(v).total_seconds()) for g, v in spec.freshness.items()},
            "trigger_window_hours": int(spec.trigger_window_hours),
            "state_consistency": spec.state_consistency,
            "requires_orderbook": bool(spec.requires_orderbook),
            "requires_public_trades": bool(spec.requires_public_trades),
            "min_symbol_month_coverage": float(spec.min_symbol_month_coverage),
        }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
