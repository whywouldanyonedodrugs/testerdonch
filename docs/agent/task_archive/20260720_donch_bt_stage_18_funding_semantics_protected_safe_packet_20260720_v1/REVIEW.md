# Stage 18 Adversarial Review

Status: `rejected_for_packet_publication`; code repair accepted within bounded scope.

## Determinations

- Protected-safe loader: accepted. Hash, platform, purpose, schema, footer statistics, row-group index, and post-read timestamp bounds fail closed. Tests cover safe, protected, mixed, missing-statistics, corrupt, hash-drifted, and footer/payload contradiction cases.
- Absolute funding cashflow arithmetic: accepted synthetically for sign, long/short, positive/negative rates, partial hour, exact boundary, and no-overlap behavior.
- Historical funding-period timestamp semantics: rejected as unresolved. Official documentation proves continuous next-period funding mechanics but does not bind the historical endpoint row timestamp to the applicable period start or end.
- Adverse allowance: rejected for real calibration. No absolute-funding row group can be safely opened, leaving zero authorized observations.
- Calendar selection privilege: the preferred uniform-allowance contract removes the privilege by construction, but no numerical allowance is frozen.
- Packet completeness: rejected. Economic addresses, manifest, approval packet, and resource projection were not regenerated because doing so would embed unresolved semantics.
- Index/spot acquisition: not required by the intended absolute-rate formula, but this does not cure the physical-source and timestamp blockers.

## Blocking findings

1. `blocking` — all 305 authoritative funding row groups are mixed or protected. Smallest repair: produce a separately authorized physical pre-2026 source upstream, without deserializing the current mixed payload under this task.
2. `blocking` — API row timestamp meaning is not authoritative. Smallest repair: obtain an official endpoint schema statement or a separately authorized exact source whose period-start field is explicit.
3. `blocking` — q95/q99 allowance has no admissible calibration sample. It must not be replaced with zero, relative funding, or a hand-selected constant.

No independent reviewer was available under the active no-delegation constraint. This self-adversarial review therefore cannot satisfy the task's independent-acceptance gate even if the data blockers are later repaired.
