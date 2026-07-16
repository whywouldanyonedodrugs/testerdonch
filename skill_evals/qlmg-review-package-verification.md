# Skill Evaluation: qlmg-review-package-verification

Purpose: test routing and required behavior. These prompts are evaluation cases, not authorization to execute them.

## Prompts that should trigger

| ID | Prompt | Expected routing |
|---|---|---|
| P1 | “Rebuild the blocked all-tested-family review package with the required raw verification extracts, actual test counts, and reproducibility hashes. Do not launch new economics.” | Trigger this skill; preserve source roots and re-evaluate release readiness. |
| P2 | “Validate the package manifest, authority registry, recomputed metrics, protected-row count, and secret scan. Keep any protocol blocker visible even if hashes pass.” | Trigger this skill for local verification only. |
| P3 | “Upload the verified `qlmg_all-tested-family-review_20260716_v02.zip` to the explicitly approved `research_drive:Donch/review-bundles/` destination using the confirmed research-service identity, then verify remote content.” | Trigger this skill; verify all prerequisites and the remote hash or round trip before declaring handoff complete. |

## Prompts that should not trigger

| ID | Prompt | Expected routing |
|---|---|---|
| N1 | “Compress these generic application logs into `logs.zip`.” | Do not trigger; this is not a QLMG evidence package. |
| N2 | “Implement and rank a new BTC-lag strategy screen.” | Route to `$qlmg-rankable-backtest-contract`; require economic authorization. |
| N3 | “Inspect Git status and draft a plan for a multi-file refactor.” | Route to `$qlmg-plan-and-preflight`. |

## Expected files and actions

- Read current package-builder contracts, authority registry, supersession map, package manifest, and verification requirements.
- Build a new versioned root without mutating finalized sources.
- Create a closed ZIP named `qlmg_<specific-content-slug>_<YYYYMMDD>_vNN.zip`.
- Validate required content, schema, recomputation, deterministic replay, protected-period exclusion, tests, hashes, and secret scan.
- Keep package status independent from hash status.
- For remote work, verify explicit upload authorization, exact destination, confirmed write identity, collision policy, local hash, and a supported remote content check.

## Acceptance checks

- All three positive prompts trigger and all three negative prompts do not trigger.
- P1 and P2 perform no remote write; P1 performs no new economic screen.
- P3 refuses to write if any prerequisite is missing and reports `remote_handoff_blocked` with exact gaps.
- Size equality alone cannot satisfy remote verification.
- Ambiguous archive names are rejected.
- Final output records local and remote paths, UTC time, size, SHA-256, tool/version, non-secret identity label, package status, economic-run status, and protected-outcome status.
