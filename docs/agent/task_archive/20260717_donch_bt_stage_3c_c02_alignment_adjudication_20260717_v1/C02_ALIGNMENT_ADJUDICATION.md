# C02 Alignment Adjudication

Status: `ready_for_C02_resolution_aware_contract_review`.

## Original frozen failure

The Stage 3B failure remains authoritative. Under the `-5m` perturbation, 19,754 exact simultaneous events become spot-led and 6,374 events disappear or change episode/direction. Under `+5m`, 19,969 simultaneous events become perp-led and 5,592 disappear or change episode/direction. Genuine spot/perp leader reversals number 116 and 141 respectively. The dominant instability is bar-resolution ordering and event-grid sensitivity, not a defensible alternative clock.

## Resolution-aware population

The exact frozen population remains 32,686 events across 87 symbols. The interval-censored primary states are:

- `resolved_spot_led`: 1,017 events; 201/332/484 in 2023/2024/2025.
- `resolved_perp_led`: 609 events; 98/217/294 in 2023/2024/2025.
- `coincident_or_unresolved`: 31,060 events.

Resolved spot-led events retain the same leader under the 30-minute lookback in 841 of 1,017 cases (`82.6942%`) and pass all frozen aggregate feasibility gates. Positive spot-led events also pass at `86.9121%`; negative spot-led events are below the robustness gate at `78.7879%`.

Resolved perp-led events retain the same leader in 412 of 609 cases (`67.6519%`) and fail the frozen robustness gate in aggregate and by direction. Their 264 completed trade-and-mark failures remain diagnostic only.

## Decision

Five-minute data can represent a bounded resolution-aware spot-led branch for contract review. It cannot support the frozen resolved-perp-led branch under the predeclared robustness gate. This is not economic evidence and does not authorize a C02 screen.
