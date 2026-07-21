from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from .canonical import atomic_write_json, canonical_hash, sha256_file
from .family_engines.common import type7_quantile_with_negative_infinity
from .selection import adjudicate_route, family_outer_vector, paired_control_pass, registered_bootstrap_seed


TERMINAL_STATUSES = frozenset({
    "completed", "unavailable_data", "unavailable_no_parent", "unavailable_duplicate_address",
    "invalid_combination", "family_stopped", "global_bound_stop_incomplete", "mechanical_failure",
})


class TerminalContractError(RuntimeError):
    pass


def reconcile_identities(expected_ids: Sequence[str], rows: Sequence[Mapping[str, Any]], id_field: str) -> dict[str, Any]:
    expected = list(expected_ids)
    if len(expected) != len(set(expected)):
        raise TerminalContractError("terminal expected identity inventory contains duplicates")
    seen: dict[str, str] = {}
    for row in rows:
        identity = str(row[id_field]); status = str(row["terminal_status"])
        if identity in seen:
            raise TerminalContractError("terminal identity appears more than once")
        if status not in TERMINAL_STATUSES:
            raise TerminalContractError(f"unknown terminal status: {status}")
        seen[identity] = status
    missing = sorted(set(expected) - set(seen)); extra = sorted(set(seen) - set(expected))
    counts = Counter(seen.values())
    return {
        "expected": len(expected), "observed": len(seen), "missing": missing, "extra": extra,
        "status_counts": dict(sorted(counts.items())), "pass": not missing and not extra and len(seen) == len(expected),
    }


def forensic_summary(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [row for row in observations if row.get("status", "complete") == "complete"]
    by_symbol: dict[str, float] = defaultdict(float); by_month: dict[str, float] = defaultdict(float)
    by_symbol_month: dict[str, float] = defaultdict(float); by_day: dict[str, float] = defaultdict(float); by_year: dict[str, float] = defaultdict(float)
    total = 0.0
    for row in complete:
        value = float(row["base_net_bps"]); symbol = str(row["symbol"]); month = str(row["month"])
        day = str(row.get("market_day", row.get("entry_day", month)))
        year = day[:4]
        by_symbol[symbol] += value; by_month[month] += value; by_symbol_month[f"{symbol}|{month}"] += value; by_day[day] += value; by_year[year] += value; total += value
    leave_symbol = {key: total - value for key, value in sorted(by_symbol.items())}
    leave_month = {key: total - value for key, value in sorted(by_month.items())}
    leave_symbol_month = {key: total - value for key, value in sorted(by_symbol_month.items())}
    absolute = sum(abs(float(row["base_net_bps"])) for row in complete)
    concentrations = {
        "symbol": max((abs(value) / absolute for value in by_symbol.values()), default=0.0) if absolute else 0.0,
        "day": max((abs(value) / absolute for value in by_day.values()), default=0.0) if absolute else 0.0,
        "year": max((abs(value) / absolute for value in by_year.values()), default=0.0) if absolute else 0.0,
    }
    concentration = max(concentrations.values())
    return {
        "event_count": len(complete), "symbol_contributions": dict(sorted(by_symbol.items())),
        "month_contributions": dict(sorted(by_month.items())), "symbol_month_contributions": dict(sorted(by_symbol_month.items())),
        "day_contributions": dict(sorted(by_day.items())), "year_contributions": dict(sorted(by_year.items())),
        "leave_one_symbol": leave_symbol, "leave_one_month": leave_month, "leave_one_symbol_month": leave_symbol_month,
        "concentrations": concentrations, "maximum_concentration": concentration,
        "concentrated": concentration > 0.50 or min((*leave_symbol.values(), *leave_month.values()), default=0.0) <= 0,
        "severely_concentrated": concentration > 0.80,
    }


def route_record(
    family: str,
    fold_values: Mapping[str, Sequence[float]],
    fold_order: Sequence[str],
    controls: Mapping[str, Mapping[str, Any]],
    *,
    stress_positive: bool,
    delay_positive: bool,
    sample_sufficient: bool,
    add_fraction: float | None = None,
    common_stress_ok: bool = True,
) -> dict[str, Any]:
    vector = family_outer_vector(fold_values, fold_order)
    finite = [value for value in vector if value != float("-inf")]
    common = (
        len(finite) >= 5 and sum(value > 0 for value in finite) >= 5 and common_stress_ok
        and median(finite) > 0 and type7_quantile_with_negative_infinity(vector, 0.20) > 0
    )
    main = controls.get("main_null", {"pass": False})
    components = {key: bool(value.get("pass")) for key, value in controls.items() if key != "main_null"}
    base_positive = bool(finite and median(finite) > 0)
    route = adjudicate_route(
        family, common, bool(main.get("pass")), components,
        base_positive=base_positive, stress_positive=stress_positive,
        delay_positive=delay_positive, sample_sufficient=sample_sufficient,
        add_fraction=add_fraction,
        sample_limited_eligible=base_positive and common_stress_ok and (len(finite) < 5 or not sample_sufficient),
    )
    return {"family": family, "outer_vector": vector, "common_gate": common, "common_stress_ok": common_stress_ok, "main_null": main, "component_controls": components, "route": route}


def terminal_package(
    output_root: Path,
    *,
    attempt_ids: Sequence[str],
    control_ids: Sequence[str],
    attempt_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    forensics: Sequence[Mapping[str, Any]],
    all_workers_stopped: bool,
    bound_stop: bool = False,
    job_reconciliation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    attempt_reconciliation = reconcile_identities(attempt_ids, attempt_rows, "attempt_id")
    control_reconciliation = reconcile_identities(control_ids, control_rows, "control_attempt_id")
    if not all_workers_stopped:
        raise TerminalContractError("terminal package cannot commit while a worker remains live")
    status = "global_bound_stop_incomplete" if bound_stop else "completed"
    if not bound_stop and (not attempt_reconciliation["pass"] or not control_reconciliation["pass"]):
        raise TerminalContractError("completion reconciliation is incomplete")
    if not bound_stop and (not isinstance(job_reconciliation, Mapping) or job_reconciliation.get("pass") is not True):
        raise TerminalContractError("completion stage-job reconciliation is incomplete")
    if not bound_stop:
        blocking = {"global_bound_stop_incomplete", "mechanical_failure"}
        if any(row["terminal_status"] in blocking for row in (*attempt_rows, *control_rows)):
            raise TerminalContractError("completion contains an incomplete or mechanically failed terminal identity")
        if not routes or any(not row.get("route") for row in routes):
            raise TerminalContractError("completion lacks a frozen terminal route")
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "ATTEMPT_TERMINAL_ROWS.json", {"rows": list(attempt_rows)})
    atomic_write_json(output_root / "CONTROL_TERMINAL_ROWS.json", {"rows": list(control_rows)})
    atomic_write_json(output_root / "ROUTE_RECORDS.json", {"rows": list(routes)})
    atomic_write_json(output_root / "FORENSIC_RECORDS.json", {"rows": list(forensics)})
    supporting = []
    for name in ("ATTEMPT_TERMINAL_ROWS.json", "CONTROL_TERMINAL_ROWS.json", "ROUTE_RECORDS.json", "FORENSIC_RECORDS.json"):
        path = output_root / name
        supporting.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    supporting_inventory = {"schema": "stage23_terminal_supporting_artifact_inventory_v1", "files": supporting, "inventory_sha256": canonical_hash(supporting)}
    atomic_write_json(output_root / "TERMINAL_SUPPORTING_ARTIFACT_INVENTORY.json", supporting_inventory)
    payload = {
        "schema": "stage23_terminal_package_v1", "status": status,
        "attempt_reconciliation": attempt_reconciliation, "control_reconciliation": control_reconciliation,
        "routes": list(routes), "forensics": list(forensics), "all_workers_stopped": all_workers_stopped,
        "resumable": bool(bound_stop), "job_reconciliation": dict(job_reconciliation or {}), "artifact_inventory_sha256": sha256_file(output_root / "TERMINAL_SUPPORTING_ARTIFACT_INVENTORY.json"),
    }
    atomic_write_json(output_root / "TERMINAL_PACKAGE.json", payload)
    files = []
    for path in sorted(output_root.glob("*.json")):
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    inventory = {"schema": "stage23_terminal_artifact_inventory_v1", "files": files, "inventory_sha256": canonical_hash(files)}
    atomic_write_json(output_root / "TERMINAL_ARTIFACT_INVENTORY.json", inventory)
    return payload


def _terminal_status(value: object) -> str:
    status = str(value)
    return {
        "complete": "completed", "completed": "completed",
        "unavailable_data": "unavailable_data", "unavailable_no_parent": "unavailable_no_parent",
        "unavailable_duplicate_address": "unavailable_duplicate_address",
        "invalid_combination": "invalid_combination", "family_stopped": "family_stopped",
        "global_bound_stop_incomplete": "global_bound_stop_incomplete",
    }.get(status, "mechanical_failure")


def campaign_terminal_records(
    *,
    strategy_rows: Sequence[Mapping[str, Any]],
    execution_rows: Sequence[Mapping[str, Any]],
    control_registry: Sequence[Mapping[str, Any]],
    stage_payloads: Sequence[Mapping[str, Any]],
    outer_payloads: Sequence[Mapping[str, Any]],
    control_payloads: Sequence[Mapping[str, Any]],
    fold_order: Sequence[str],
) -> dict[str, Any]:
    """Independently derive all terminal identities, routes and forensics."""
    execution_status: dict[str, str] = {}
    for payload in stage_payloads:
        identity = payload.get("registered_attempt_id")
        if identity is None:
            continue
        candidate = _terminal_status(payload.get("status"))
        prior = execution_status.get(str(identity))
        if prior is None or (prior != "completed" and candidate == "completed"):
            execution_status[str(identity)] = candidate
    attempt_rows = []
    for index, row in enumerate(strategy_rows):
        registered_id = canonical_hash({"registered_registry_index": index, "registered_row": row})
        if row.get("execution_disposition") == "multiplicity_only_duplicate":
            status = "unavailable_duplicate_address"
        else:
            status = execution_status.get(str(row["executable_attempt_id"]), "family_stopped")
        attempt_rows.append({
            "attempt_id": registered_id, "terminal_status": status,
            "executable_attempt_id": row["executable_attempt_id"], "family_id": row["family_id"],
            "canonical_economic_address_sha256": row["canonical_economic_address_sha256"],
        })
    control_by_id = {str(row["control_attempt_id"]): row for row in control_registry}
    payload_by_control = {str(row["control_attempt_id"]): row for row in control_payloads if row.get("control_attempt_id") is not None}
    control_rows = []
    control_gates: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for control_id, registered in sorted(control_by_id.items()):
        payload = payload_by_control.get(control_id)
        status = "global_bound_stop_incomplete" if payload is None else _terminal_status(payload.get("status"))
        control_rows.append({"control_attempt_id": control_id, "terminal_status": status, "family_id": registered["family"], "fold": registered["fold"], "control_id": registered["control_id"]})
        if payload is None or status != "completed" or not isinstance(payload.get("paired_control"), Mapping):
            continue
        paired = payload["paired_control"]
        if str(registered["control_id"]) == {
            "A4_TSMOM_V7": "A4_SIGN_PERMUTED_MAIN_NULL", "A1_COMPRESSION_V2": "A1_MATCHED_PSEUDO_EVENT_MAIN_NULL",
            "A2_PRIOR_HIGH_RS_CONTEXT_V1": "A2_CONTEXT_PERMUTED_MAIN_NULL", "A3_STARTER_RETEST_V3": "A3_RETEST_TIME_PERMUTED_MAIN_NULL",
        }.get(str(registered["family"])):
            day_values = list(paired.get("parent_minus_control_by_utc_day", {}).values())
            gate = paired_control_pass(day_values, float(paired.get("coverage", 0.0)), int(registered["effective_seed"]))
            gate["gate"] = "registered_main_null"
        else:
            parent = payload.get("parent_aggregate", {}); control = payload.get("aggregate", {})
            parent_primary = parent.get("base_net_bps"); control_primary = control.get("base_net_bps")
            parent_stress = parent.get("stress_net_bps"); control_stress = control.get("stress_net_bps")
            passed = None not in {parent_primary, control_primary, parent_stress, control_stress} and float(parent_primary) - float(control_primary) > 0 and float(parent_stress) >= float(control_stress) - 5.0
            gate = {"pass": passed, "parent_primary_minus_control": None if parent_primary is None or control_primary is None else float(parent_primary) - float(control_primary), "parent_stress": parent_stress, "control_stress": control_stress, "coverage": float(paired.get("coverage", 0.0)), "gate": "registered_component_ablation"}
        gate.update({"control_attempt_id": control_id, "fold": registered["fold"], "parent_slot": registered["parent_slot"]})
        control_gates[str(registered["family"])][str(registered["control_id"])].append(gate)
    fold_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for payload in outer_payloads:
        aggregate = payload.get("aggregate")
        if payload.get("status") == "complete" and isinstance(aggregate, Mapping) and aggregate.get("base_net_bps") is not None:
            fold_values[str(payload["family_id"])][str(payload["outer_fold_id"])].append(float(aggregate["base_net_bps"]))
    execution_by_id = {str(row["executable_attempt_id"]): row for row in execution_rows}
    main_ids = {
        "A4_TSMOM_V7": "A4_SIGN_PERMUTED_MAIN_NULL", "A1_COMPRESSION_V2": "A1_MATCHED_PSEUDO_EVENT_MAIN_NULL",
        "A2_PRIOR_HIGH_RS_CONTEXT_V1": "A2_CONTEXT_PERMUTED_MAIN_NULL", "A3_STARTER_RETEST_V3": "A3_RETEST_TIME_PERMUTED_MAIN_NULL",
    }
    route_component_ids = {
        "A4_TSMOM_V7": {"A4_GENERIC_SIGNED_RETURN"}, "A1_COMPRESSION_V2": {"A1_PRICE_ONLY_IMPULSE"},
        "A2_PRIOR_HIGH_RS_CONTEXT_V1": {"A2_PARENT_ONLY"}, "A3_STARTER_RETEST_V3": {"A3_STARTER_ONLY"},
    }
    routes = []
    for family in sorted(set(fold_values) | set(control_gates)):
        controls: dict[str, Mapping[str, Any]] = {}
        for control_id, gates in control_gates[family].items():
            passing_folds = len({str(row["fold"]) for row in gates if row["pass"]})
            controls["main_null" if control_id == main_ids.get(family) else control_id] = {
                "pass": passing_folds >= 5, "passing_folds": passing_folds, "registered_results": gates,
            }
        aggregates = [payload["aggregate"] for payload in outer_payloads if payload.get("family_id") == family and payload.get("status") == "complete" and isinstance(payload.get("aggregate"), Mapping)]
        stress_positive = bool(aggregates) and median(float(row.get("stress_net_bps") or float("-inf")) for row in aggregates) > 0
        common_stress_ok = bool(aggregates) and median(float(row.get("stress_net_bps") or float("-inf")) for row in aggregates) > -18
        delay_values = [float(row.get("component_metrics", {}).get("entry_delay_15m_net_bps")) for row in aggregates if row.get("component_metrics", {}).get("entry_delay_15m_net_bps") is not None]
        delay_positive = bool(delay_values) and median(delay_values) > 0
        sample_sufficient = len(fold_values[family]) >= 5 and all(int(row.get("event_count", 0)) >= 30 for row in aggregates)
        route_controls = {key: value for key, value in controls.items() if key == "main_null" or key in route_component_ids.get(family, set())}
        selected_add_fractions = [float(execution_by_id[str(payload.get("registered_attempt_id"))]["config"].get("add_fraction", 0.0)) for payload in outer_payloads if payload.get("family_id") == family and str(payload.get("registered_attempt_id")) in execution_by_id]
        add_fraction = median(selected_add_fractions) if family == "A3_STARTER_RETEST_V3" and selected_add_fractions else None
        route = route_record(family, fold_values[family], fold_order, route_controls, stress_positive=stress_positive, delay_positive=delay_positive, sample_sufficient=sample_sufficient, common_stress_ok=common_stress_ok, add_fraction=add_fraction)
        route["address_fold_decisions"] = [{
            "executable_attempt_id": payload.get("registered_attempt_id"),
            "canonical_economic_address_sha256": payload.get("canonical_economic_address_sha256"),
            "outer_fold_id": payload.get("outer_fold_id"), "terminal_status": _terminal_status(payload.get("status")),
        } for payload in outer_payloads if payload.get("family_id") == family]
        route["control_gate_inventory_sha256"] = canonical_hash(control_gates[family]); routes.append(route)
    kda_by_cell: dict[str, dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(dict)
    for payload in stage_payloads:
        registered = execution_by_id.get(str(payload.get("registered_attempt_id")))
        if registered is None or registered.get("family_id") != "KDA02B_SURVIVOR_ADJUDICATION_V1":
            continue
        variant = str(registered["config"]["adjudication_variant"])
        kda_by_cell[str(registered["config"]["stage20_cell_id"])][variant] = (registered, payload)
    for cell, variants in sorted(kda_by_cell.items()):
        identity_pair = variants.get("identity_replay")
        if identity_pair is None:
            routes.append({"family": "KDA02B_SURVIVOR_ADJUDICATION_V1", "stage20_cell_id": cell, "route": "survivor_rejected", "reason": "identity_unavailable"})
            continue
        identity_row, identity = identity_pair
        identity_days = {str(key): float(value) for key, value in identity.get("day_base_net_bps", {}).items()}
        if not identity_days:
            for observation in identity.get("observations", ()):
                day = str(observation.get("market_day", observation.get("entry_day")))
                identity_days[day] = identity_days.get(day, 0.0) + float(observation["base_net_bps"])
        identity_base = identity.get("aggregate", {}).get("base_net_bps")
        control_results: dict[str, Mapping[str, Any]] = {}
        for variant in ("price_only", "OI_removed", "generic_structure_control"):
            pair = variants.get(variant)
            other_days = {} if pair is None else {str(key): float(value) for key, value in pair[1].get("day_base_net_bps", {}).items()}
            if pair is not None and not other_days:
                for observation in pair[1].get("observations", ()):
                    day = str(observation.get("market_day", observation.get("entry_day")))
                    other_days[day] = other_days.get(day, 0.0) + float(observation["base_net_bps"])
            common_days = sorted(set(identity_days) & set(other_days))
            uplifts = [identity_days[day] - other_days[day] for day in common_days]
            control_results[variant] = paired_control_pass(
                uplifts, len(common_days) / len(identity_days) if identity_days else 0.0,
                registered_bootstrap_seed(str(identity_row["canonical_economic_address_sha256"]), f"{cell}:{variant}"),
            )
        stress = variants.get("stress_cost_32bps", ({}, {"aggregate": {}}))[1].get("aggregate", {}).get("base_net_bps")
        delay = variants.get("entry_delay_15m", ({}, {"aggregate": {}}))[1].get("aggregate", {}).get("base_net_bps")
        supported = identity_base is not None and float(identity_base) > 0 and all(result["pass"] for result in control_results.values()) and stress is not None and float(stress) > 0 and delay is not None and float(delay) > 0
        execution_sensitive = identity_base is not None and float(identity_base) > 0 and ((stress is not None and float(stress) <= 0) or (delay is not None and float(delay) <= 0))
        route_name = "component_supported_survivor" if supported else "execution_sensitive_candidate" if execution_sensitive else "sample_limited_candidate" if identity_base is not None and float(identity_base) > 0 else "survivor_rejected"
        routes.append({
            "family": "KDA02B_SURVIVOR_ADJUDICATION_V1", "stage20_cell_id": cell,
            "canonical_economic_address_sha256": identity_row["canonical_economic_address_sha256"],
            "identity_base_net_bps": identity_base, "component_controls": control_results,
            "stress_32bps": stress, "entry_delay_15m": delay,
            "registered_forensics": {
                name: variants.get(name, ({}, {"aggregate": {}}))[1].get("aggregate", {}).get("base_net_bps")
                for name in ("liquidation_removed", "funding_zero", "funding_start_alignment", "funding_end_alignment", "entry_delay_60m")
            },
            "route": route_name,
        })
    observations = [{**row, "family_id": payload.get("family_id")} for payload in outer_payloads for row in payload.get("observations", ()) if isinstance(row, Mapping)]
    complete_surfaces: dict[tuple[str, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for payload in stage_payloads:
        aggregate = payload.get("aggregate")
        registered = execution_by_id.get(str(payload.get("registered_attempt_id")))
        if payload.get("status") != "complete" or not isinstance(aggregate, Mapping) or registered is None:
            continue
        fold = str(payload.get("outer_fold_id", payload.get("fold_id", "development")))
        complete_surfaces[(str(registered["family_id"]), fold)].append((registered, aggregate))
    neighborhood_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in outer_payloads:
        registered = execution_by_id.get(str(payload.get("registered_attempt_id")))
        aggregate = payload.get("aggregate")
        if registered is None or payload.get("status") != "complete" or not isinstance(aggregate, Mapping):
            continue
        family = str(registered["family_id"]); fold = str(payload.get("outer_fold_id", payload.get("fold_id", "development")))
        config = registered["config"]
        neighbors = []
        for other, other_aggregate in complete_surfaces[(family, fold)]:
            differing = [key for key in sorted(set(config) | set(other["config"])) if config.get(key) != other["config"].get(key)]
            if len(differing) == 1 and other_aggregate.get("base_net_bps") is not None:
                neighbors.append({"axis": differing[0], "base_net_bps": float(other_aggregate["base_net_bps"]), "canonical_economic_address_sha256": other["canonical_economic_address_sha256"]})
        neighborhood_by_family[family].append({
            "canonical_economic_address_sha256": registered["canonical_economic_address_sha256"], "outer_fold_id": fold,
            "one_axis_neighbor_count": len(neighbors), "positive_neighbor_fraction": (sum(row["base_net_bps"] > 0 for row in neighbors) / len(neighbors) if neighbors else None),
            "neighbors": neighbors,
        })
    forensics = []
    for payload in outer_payloads:
        if payload.get("status") != "complete" or not isinstance(payload.get("aggregate"), Mapping):
            continue
        family = str(payload.get("family_id")); address = str(payload.get("canonical_economic_address_sha256")); aggregate = payload["aggregate"]
        selected_observations = [row for row in payload.get("observations", ()) if isinstance(row, Mapping)]
        aggregates = [aggregate]
        funding_metrics: dict[str, list[float]] = defaultdict(list)
        for aggregate in aggregates:
            for name in ("funding_zero_net_bps", "funding_start_alignment_net_bps", "funding_end_alignment_net_bps"):
                value = aggregate.get("component_metrics", {}).get(name)
                if value is not None:
                    funding_metrics[name].append(float(value))
        funding_summary = {name: {"observations": len(values), "median_net_bps": median(values)} for name, values in sorted(funding_metrics.items())}
        base_value = aggregate.get("base_net_bps")
        funding_sign_changed = bool(base_value is not None and any(item["median_net_bps"] * float(base_value) <= 0 for item in funding_summary.values()))
        neighborhood_rows = [row for row in neighborhood_by_family.get(family, []) if row["canonical_economic_address_sha256"] == address and row["outer_fold_id"] == str(payload.get("outer_fold_id"))]
        positive_neighbor_fractions = [float(row["positive_neighbor_fraction"]) for row in neighborhood_rows if row["positive_neighbor_fraction"] is not None]
        summary = forensic_summary(selected_observations)
        forensics.append({
            "family": family, "canonical_economic_address_sha256": address, "outer_fold_id": payload.get("outer_fold_id"), **summary,
            "funding_zero_and_alignment": funding_summary,
            "parameter_neighborhood": neighborhood_rows,
            "tags": {
                "concentrated": summary["concentrated"], "severely_concentrated": summary["severely_concentrated"],
                "execution_sensitive": aggregate.get("stress_net_bps") is None or float(aggregate.get("stress_net_bps")) <= 0 or aggregate.get("component_metrics", {}).get("entry_delay_15m_net_bps") is None or float(aggregate["component_metrics"]["entry_delay_15m_net_bps"]) <= 0,
                "funding_sensitive": funding_sign_changed,
                "parameter_sensitive": bool(positive_neighbor_fractions) and median(positive_neighbor_fractions) < 0.60,
                "sample_limited": int(aggregate.get("event_count", 0)) < 30,
            },
        })
    return {
        "attempt_rows": attempt_rows, "control_rows": control_rows, "routes": routes,
        "forensics": forensics, "control_gate_inventory": {family: dict(values) for family, values in control_gates.items()},
    }


__all__ = ["TERMINAL_STATUSES", "TerminalContractError", "campaign_terminal_records", "forensic_summary", "reconcile_identities", "route_record", "terminal_package"]
