# Stage 15 validation

Status: `blocked_preoutcome_packet_semantics`.

- PASS: exact starting authority and Stage-14 raw hashes matched.
- PASS: approval attachment SHA-256 matched `c526bd3e1d47ddcec5c17494dcbb3230a20d7b3e081df0fa6a2f0f984fd2ac6b`.
- PASS: packet/manifest canonical hashes matched the supplied approval.
- PASS: Stage-14 closure validator reported 13 checks passing and 228 exact cells.
- PASS: compatibility/adversarial unit suite reported 14 tests passing.
- PASS: combined campaign and derivatives suite reported 23 tests passing; changed modules compiled.
- PASS: legacy override, approval substitution, coordinated Phase-6 widening, alias substitution, missing state/Telegram, inadequate coverage, `NaN`, positive/negative infinity, negative coverage, and coverage above one fail closed.
- PASS: independent re-review accepted the validator repair with no residual code finding.
- BLOCK: approved packet lacks deterministic search/translation and complete payoff/execution semantics; independent review rejected outcome access.
- PASS: `git diff --check`.
- PASS: protected rows opened `0`; Capital.com payloads opened `0`; economic outputs computed `0`.

Funding extension, Telegram delivery tests, synthetic campaign canary, and campaign launch were not attempted because their order follows a passing packet-level pre-outcome review.
