from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .canonical import canonical_hash
from .schema import FAMILY_ORDER


class PopulationReadinessError(RuntimeError):
    pass


def reconcile_registered_population_routes(
    execution_rows: Sequence[Mapping[str, Any]],
    launch_authority: Mapping[str, Any],
    kda02b_population_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile every frozen execution address to its exact lazy population route.

    This is a configuration-free census.  It never reads payoffs and never
    substitutes the 567 benchmark frames for the launch population.
    """
    if len(execution_rows) != 11_963:
        raise PopulationReadinessError("execution registry is not the frozen 11,963-address inventory")
    by_id = {str(row["executable_attempt_id"]): row for row in execution_rows}
    if len(by_id) != len(execution_rows):
        raise PopulationReadinessError("execution registry contains duplicate attempt identities")
    expanded = launch_authority.get("population_census", {}).get("fold_expanded", {})
    for top in (10, 20, 40):
        if str(top) not in expanded:
            raise PopulationReadinessError("launch population omits a frozen PIT top-N census")

    def route_count(row: Mapping[str, Any]) -> int:
        family = str(row["family_id"]); config = row["config"]
        try:
            census = expanded[str(int(config["PIT_liquidity_top_n"]))]
        except (KeyError, TypeError, ValueError) as exc:
            raise PopulationReadinessError(f"attempt lacks a frozen PIT route: {row.get('executable_attempt_id')}") from exc
        if family in {"A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"}:
            return int(census["decisions_5m"])
        if family == "A4_TSMOM_V7":
            rebalance = str(config.get("rebalance"))
            if rebalance not in {"8h", "1d"}:
                raise PopulationReadinessError("A4 attempt has an invalid rebalance route")
            return int(census[f"a4_{rebalance}"])
        raise PopulationReadinessError("direct population route requested for a parent-bound family")

    direct = Counter(); direct_addresses = Counter(); route_groups = Counter()
    a2_source_attempts = 0; a2_beam_templates = 0; a2_parent_scan_upper_bound = 0
    kda_addresses = 0
    address_records = []
    for row in execution_rows:
        family = str(row["family_id"])
        if family in {"A4_TSMOM_V7", "A1_COMPRESSION_V2", "A3_STARTER_RETEST_V3"}:
            count = route_count(row)
            top = int(row["config"]["PIT_liquidity_top_n"])
            clock = str(row["config"].get("rebalance", "5m"))
            direct[family] += count; direct_addresses[family] += 1
            route_groups[(family, top, clock)] += 1
            route = "complete_PIT_lazy_schedule"
            population_units = count
        elif family == "A2_PRIOR_HIGH_RS_CONTEXT_V1":
            binding_mode = str(row["config"].get("parent_binding_mode"))
            if binding_mode == "source_attempt":
                parent_id = str(row.get("resolved_parent_executable_attempt_id"))
                parent = by_id.get(parent_id)
                if parent is None or parent.get("family_id") != row["config"].get("parent_family"):
                    raise PopulationReadinessError("A2 source-attempt route lacks its exact frozen parent")
                population_units = route_count(parent)
                a2_source_attempts += 1; a2_parent_scan_upper_bound += population_units
                route = "exact_source_parent_event_route"
            elif binding_mode == "beam_slot":
                population_units = None
                a2_beam_templates += 1
                route = "conditional_exact_beam_parent_event_route"
            else:
                raise PopulationReadinessError("A2 attempt has an unknown parent-binding mode")
        elif family == "KDA02B_SURVIVOR_ADJUDICATION_V1":
            kda_addresses += 1
            population_units = None
            route = "complete_stage20_cell_event_index"
        else:
            raise PopulationReadinessError(f"unknown registered family: {family}")
        address_records.append({
            "executable_attempt_id": str(row["executable_attempt_id"]),
            "canonical_economic_address_sha256": str(row["canonical_economic_address_sha256"]),
            "family_id": family,
            "route": route,
            "population_units_or_parent_scan_upper_bound": population_units,
        })

    kda_counts = kda02b_population_manifest.get("counts", {})
    expected_kda = {
        "configurations": 209,
        "eligible_event_rows": 466_348,
        "unavailable_event_rows": 16_571,
        "eligible_dispatch_units": 5_129_828,
        "unavailable_dispatch_units": 182_281,
    }
    if any(int(kda_counts.get(key, -1)) != value for key, value in expected_kda.items()) or kda_addresses != 209:
        raise PopulationReadinessError("KDA02B registered population census differs")
    families = Counter(str(row["family_id"]) for row in execution_rows)
    if set(families) != set(FAMILY_ORDER):
        raise PopulationReadinessError("registered population routes omit a frozen family")
    records_identity = canonical_hash(address_records)
    return {
        "schema": "stage24_registered_population_route_reconciliation_v1",
        "status": "pass",
        "registered_execution_addresses": len(execution_rows),
        "families": dict(sorted(families.items())),
        "direct_PIT_addresses": dict(sorted(direct_addresses.items())),
        "direct_PIT_attempt_fold_decision_units": dict(sorted(direct.items())),
        "direct_route_groups": [
            {"family_id": family, "PIT_liquidity_top_n": top, "clock": clock, "addresses": count}
            for (family, top, clock), count in sorted(route_groups.items())
        ],
        "A2": {
            "source_attempt_addresses": a2_source_attempts,
            "beam_slot_templates": a2_beam_templates,
            "source_parent_decision_scan_upper_bound": a2_parent_scan_upper_bound,
            "opportunity_identity": "exact parent engine event; never generic PIT substitution",
        },
        "KDA02B": {
            **expected_kda,
            "typed_unavailable_reason": "stage14_kda02b_final_eligible_false",
            "unavailable_is_not_economic_testing": True,
        },
        "population_authority_sha256": str(launch_authority.get("authority_inventory_sha256")),
        "address_route_inventory_sha256": records_identity,
        "economic_outcomes_opened": False,
        "protected_rows_opened": 0,
        "capitalcom_payload_opened": False,
    }


__all__ = ["PopulationReadinessError", "reconcile_registered_population_routes"]
