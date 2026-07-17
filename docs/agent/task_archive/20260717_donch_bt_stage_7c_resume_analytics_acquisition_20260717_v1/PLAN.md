# Stage 7C Resume Analytics Acquisition Plan

Status: in_progress
Owner: backtesting agent
Created UTC: 2026-07-17
Repository: `/opt/testerdonch` at `35882880317f691219041d22daaf28c9e582a2bb`

## Objective

Implement reviewed shard-month publication and launch the exact approved 2023-2025 public Kraken analytics acquisition as a persistent resume-safe worker.

## Non-goals

No signals, returns, rankings, prices, funding acquisition, protected rows, private endpoints, or economic interpretation.

## Authority and gates

- Stage 7B inventory/data hashes verified.
- Current free bytes: 56,093,011,968; total: 160,970,244,096.
- Revised 25% threshold: 40,242,561,024; projected free after contingency: 41,754,721,971; prestart gate passes.
- Runtime hard stop: 32,194,048,819 bytes.
- Exact scope: three metrics, 460 frozen identities at 300 seconds, BTC/ETH at 60 seconds, `[2023-01-01,2026-01-01)`.

## Milestones

1. Add raw-field semantics, deterministic shards, unit publication, storage/heartbeat guards, and synthetic tests. Fail closed on any mismatch.
2. Run compile and applicable non-economic tests; independently inspect diff and projection.
3. Commit implementation, launch persistent worker, verify PID/session/log/heartbeat/Telegram and first progress.
4. At completion or resume-safe running checkpoint, archive compact evidence, push non-force, and hand off to approved Drive root.

## Rollback

Stop the worker with SIGINT at a unit boundary; retain ledger, staged files, published units, and old Stage 7B evidence. Revert task commits without deleting acquired evidence.
