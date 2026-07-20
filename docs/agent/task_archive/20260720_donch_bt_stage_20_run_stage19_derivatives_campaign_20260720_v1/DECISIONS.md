# Stage 20 decisions

## Scope

- Execute only KDA02B, KDA02C, and KDX01 through Phases 2-5.
- Retain 186 executable cells, 42 inherited non-executable KDX attempts, and 228 programme attempts.
- Do not execute Phase 6, controls, independent validation, Capital.com access, protected-period access, or live/private actions.

## Pre-outcome implementation decisions

1. Recovered the deleted Stage 17 mechanical campaign source from the local Codex audit log, verified its recorded byte size and SHA-256, and adapted only the authority, funding, and Stage 20 execution surfaces required by the approved packet.
2. Replaced the superseded Stage 17 funding extension with the hash-bound Stage 19 runtime. The runtime always uses the adverse minimum of zero and both exact hourly alignments plus nonpositive q95/q99 missing-hour charges; it exposes no alignment choice.
3. Partitioned KDA02C event identities under their native PF symbols. A synthetic `KDA02C` partition would not map to an official trade authority and would silently exclude that approved lane.
4. Ordered economic access as development scoring, atomic beam freeze, then selected outer scoring for each fold. Materializing outer returns before the freeze, even without reading them during selection, was rejected as inconsistent with the packet's freeze-before-outer contract.
5. Kept large event and scored tapes local and hash-manifested. Only compact evidence is eligible for the approved Drive handoff.

## Outcome firewall

All implementation, event construction, threshold fitting, deterministic replay, funding validation, and the Phase 2-5 canary are pre-outcome operations. No real forward return, PnL, protected strategy outcome, or Capital.com payload may be opened until all pre-outcome gates and the independent review pass.
# Decisions

- Reused both existing 187-symbol event-tape trees after their complete
  row/SHA deterministic replay passed; no large rebuild was performed.
- Implemented only the seven authorized launch-safety remediations and did not
  alter frozen economic or selection semantics.
- Launched only after repeated independent review, secure Telegram preflight,
  and the final atomic authority audit passed.
- Treated the transient `ProcessLookupError` as a genuine global bound stop.
  Did not patch or relaunch because that runtime robustness defect was outside
  the seven fixed criteria for this bounded task.
- Limited closure to metadata-only reconciliation. No partial response surface,
  candidate ranking, return, route, or protected payload was inspected.
- Preserve the run root for an explicitly authorized idempotent resume; do not
  claim 186-cell completion from the 126 represented cell identities.
