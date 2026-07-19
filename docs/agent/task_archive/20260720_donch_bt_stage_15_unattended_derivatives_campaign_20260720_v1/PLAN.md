# Stage 15 — Approved unattended derivatives campaign

Status: blocked
Owner: Codex backtesting agent
Created UTC: 2026-07-20
Repository root and commit: `/opt/testerdonch-stage15-20260720` at `50dffb791c146b359cb210532e5f7291774e26f0`

## Received task and archive context

- Task specification: `TASK_SPEC.md`, SHA-256 `2fbf5088807c0bd20deda7c63d13517be921936e2f8f23f13e446fe3375442f6`.
- Human approval: `HUMAN_APPROVAL.json`, SHA-256 `c526bd3e1d47ddcec5c17494dcbb3230a20d7b3e081df0fa6a2f0f984fd2ac6b`.
- Approval is limited to the exact Stage-14 packet and manifest, three named hypotheses, and Phases 2–5.
- Durable task archive: this directory. Large result ledgers will use a new local versioned run root and remain out of Git.
- Approved Drive target: `qlmg_sweep_drive:` under folder ID `1vMuf1EJbojoeIjJjz8Xj1fhwLizNgLfl`, unique task folder, no overwrite.

## Objective

Repair only the launch-binding defects without changing economic semantics, independently verify the repair and funding extension before outcomes, then execute and close the exact authorized 228-cell, 27-fold Kraken derivatives campaign with deterministic, auditable Phases 2–5 evidence.

## Non-goals

- No Phase 6 controls or Phase 7 validation/deployment.
- No C17, protected-period, Capital.com, acquisition, capture, account, order, or live-trading action.
- No search-space, fold, cost, execution, hypothesis, beam, objective, tie-break, or multiplicity change.
- No outcome-conditioned intervention or partial-winner inspection.

## Verified authority and assumptions

| Item | Status | Evidence | Failure response |
|---|---|---|---|
| Starting authority | verified | exact worktree base `50dffb791c146b359cb210532e5f7291774e26f0`; `origin/main` matched at preflight | global stop before outcomes on mismatch |
| Human approval bytes | verified | SHA-256 `c526bd3e...fd2ac6b` | stop before outcomes |
| Stage-14 packet raw/canonical | verified | `c0298305...1c313` / `57079141...4bfd` | stop before outcomes |
| Stage-14 manifest raw/canonical | verified | `61119ce3...5f632` / `018a274e...d941` | stop before outcomes |
| Search/resource/state-tape hashes | verified | Stage-14 task archive | stop before outcomes |
| Protected cutoff | verified | `2026-01-01T00:00:00Z`, exclusive upper train bound | global stop |
| Telegram secure configuration | verified presence only | existing ignored `.telegram.env` exposes no values | require successful dry-run before outcomes |
| Economic claims | verified bounded | programme-exposed historical; not independent validation; not live-ready | preserve labels in every summary |

## Repository state preservation

- Original checkout: `/opt/testerdonch`, branch `main`, local commit `baaa10c224807e1dc7e32bfee7227711cb0c1279`; tracked dirty paths and three untracked paths preserved unchanged.
- Recovery bundle: `/opt/testerdonch-stage15-dirty-recovery-20260720`; hash ledger SHA-256 `769ba512c5f517f943e1589bf532e83a1a7e9e266f68f5c5baf110aebde5055f` and all listed hashes verified.
- Sensitive ignored Telegram configuration was excluded from content capture.
- Isolated worktree: `/opt/testerdonch-stage15-20260720`; branch `agent/stage15-unattended-derivatives-campaign-20260720`; exact base `50dffb791c146b359cb210532e5f7291774e26f0`.
- Rationale: the user-authorized starting commit is the verified remote Stage-14 authority; unrelated local-main and dirty-checkout work must not be overlaid.

## Scope and boundaries

- Venue/instrument: Kraken linear PF derivatives in the frozen 187-symbol authorized cohort.
- Rankable interval: `[2023-01-01T00:00:00Z, 2026-01-01T00:00:00Z)`; registered development begins 2023-04-01.
- Economic run: authorized only by the exact supplied human approval.
- Remote writes: non-force Git push and approved-default Drive handoff are authorized.
- Resources: at most four workers, 14,400 seconds, and 5 GiB campaign output.

## Frozen hypothesis-development object

- Lanes: KDA02B 96 cells, KDA02C 48 cells, KDX01 84 cells.
- Mechanisms, axes, 27 quarterly forward records, 14/32-bps pre-funding costs, first-authorized-PF-five-minute-open execution, actual-exit non-overlap, funding partitions, Pareto objectives, deterministic tie-break, and maximum-five family/fold beam are those in the byte-bound Stage-14 packet.
- Candidate economic addresses are complete registered cell IDs plus fold-local fitted state; every freeze precedes its evaluation block.
- Funding extension applies the frozen model unchanged using PIT inputs. Coverage/missingness is a fail-closed eligibility rule and cannot rank candidates.
- Evaluation information cannot flow backward. Failed folds remain explicit and are not pooled away.

## Files expected to change

- `tools/qlmg_research_campaign.py`: dual raw/canonical approval validation and external-approval override after exact validation.
- `tools/qlmg_derivatives_campaign.py`: frozen mechanics, funding-coverage, execution, replay, and evidence helpers.
- `tools/run_derivatives_campaign.py`: resumable supervised campaign runner and notifications.
- `tools/validate_stage15_campaign.py`: pre/post-run contract and artifact checks.
- `unit_tests/test_qlmg_research_campaign.py` and `unit_tests/test_derivatives_campaign.py`: adversarial and deterministic fixtures.
- This task archive and compact handoff records.

## Milestones

### M1 — Pre-outcome launch repair

- Implement exact Stage-14 aliases, raw plus canonical hash validation, validated external-approval override, numeric funding constraints, and mandatory notifier state.
- Acceptance: adversarial tests reject byte/canonical alteration, aliases, widening, missing state/notifier/coverage, and direct bypass.
- Failure: stop without opening outcome data.

### M2 — Funding extension and canary

- Extend the frozen funding model to registered pre-2024 boundaries without refit; verify PIT timestamps, partitions, coverage floors, and outcome independence.
- Acceptance: independent review accepts repair and extension; synthetic canary and resume replay pass.
- Failure: stop before outcomes; regenerate packet if economic semantics would need to change.

### M3 — Persistent campaign

- Validate Telegram delivery, launch under `systemd --user` when usable or `tmux` fallback, persist state/log/PID/session, and execute all registered work or exact family stops.
- Acceptance: one real cell, scheduled heartbeat, state/artifact reconciliation, then terminal 228-cell/fold reconciliation within caps.
- Failure: automatic family isolation or global stop according to the frozen contract.

### M4 — Review, publication, and handoff

- Independently recompute frozen selections/results/routes, validate hashes and secrets, create reviewed commits, non-force push, build a closed compact ZIP, and round-trip verify Drive.
- Acceptance: no blocking review findings; package manifest passes; Drive bytes and hashes match.

## Validation commands

Repository-supported unit-test pattern is `python3 -m unittest`. Exact focused and broader command results will be recorded in `COMMANDS_AND_RESULTS.md`. Packet/funding/campaign validation uses the task-specific validators added here and deterministic fixture/replay modes only before outcome authorization is opened.

## Risk and rollback

- Source rollback is branch-local commit reversion; no finalized source or run root will be deleted or overwritten.
- Campaign output uses a new versioned local root and atomic state files.
- A compatibility or funding defect before the first outcome read terminates the launch. A post-launch shared authority, timestamp, protected-exposure, storage, or replay defect produces a global stop without tuning.
- Drive collision increments `vNN`; remote overwrite and deletion are prohibited.

## Decision and progress log

| UTC | Decision/result | Consequence |
|---|---|---|
| 2026-07-20 | Exact approval attachment SHA-256 matched the task | approval may enter compatibility validation |
| 2026-07-20 | Original checkout was dirty and local `main` was ahead of the user-authorized remote base | isolated worktree created from exact `50dffb7`; original preserved |
| 2026-07-20 | Packet and manifest raw and canonical hashes matched the approval | bounded repair may proceed; no outcome reader opened |
| 2026-07-20 | Host has 4 CPUs, about 11 GiB available RAM, 38 GiB disk free, and `tmux`; user systemd is degraded | plan for `tmux` persistent fallback after all pre-outcome gates |
| 2026-07-20 | First independent review found three validator bypasses | repaired exact approval anchoring, legacy override, and numeric coverage validation |
| 2026-07-20 | Independent re-review accepted the validator repair; 14 focused tests passed | compatibility milestone closed |
| 2026-07-20 | Independent review found the approved packet lacks deterministic search/translation and payoff/execution identities | global prelaunch stop; funding, Telegram, outcomes, and launch not attempted; regenerated packet and new approval required |

## Completion record

- Acceptance criteria met: compatibility repair only; campaign execution blocked before outcomes.
- Economic runs launched: no.
- Protected outcomes inspected: no.
- Remote writes: pending compact blocker handoff only; no economic artifacts exist.
- Unresolved blocker: the approved packet must change, which invalidates the supplied exact approval.
