"""Pure frozen KDA02 v2 Level-3 arithmetic and inference gates."""

from __future__ import annotations

from tools.qlmg_kda01_level3_economic import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    cluster_bootstrap,
    equal_cluster_returns,
    gate_flags,
    positive_contribution_share,
    score_open_prices,
)


def branch_side(branch_id: str) -> int:
    mapping = {
        "negative_active_purge_continuation": -1,
        "positive_active_purge_continuation": 1,
        "negative_completed_purge_reversal": 1,
        "positive_completed_purge_reversal": -1,
    }
    suffix = branch_id.removeprefix("primary_").removeprefix("robustness_")
    if suffix not in mapping:
        raise ValueError(f"unknown frozen KDA02 branch side: {branch_id}")
    return mapping[suffix]


__all__ = [
    "BOOTSTRAP_RESAMPLES", "BOOTSTRAP_SEED", "branch_side", "cluster_bootstrap",
    "equal_cluster_returns", "gate_flags", "positive_contribution_share",
    "score_open_prices",
]
