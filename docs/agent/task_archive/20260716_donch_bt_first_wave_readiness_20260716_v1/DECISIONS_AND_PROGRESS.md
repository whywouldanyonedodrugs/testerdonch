# Decisions And Progress

## Prerequisite

- Verified the former `main` and `origin/main` at `f81bb0719bdfe2e9b6e22b627f7e43ad8cb47587`.
- Applied only governance commit `858e88d8bae4d5e92e025d59983eb0f7ee2a15d0` on an isolated integration branch. Original and applied patch IDs were identical.
- Non-economic governance validation passed. Local `main` was fast-forwarded and pushed to `8cf3e227105fd7626445d27c8caf4c28bccc2ecb` under the operator's explicit authorization.

## Authority Intake

- Transfer ZIP SHA-256 matched `06afecffde74d4fecb32b6a5859ea13d0d8700e398fdc106d4e8b1bc5b2dc5be`.
- ZIP integrity passed.
- `TRANSFER_MANIFEST.json` declared exactly eight authority files. Every byte count and SHA-256 matched.
- Exact source copies are under `received/authority/`. The ZIP itself remains under `research_inputs/` and was not duplicated into Git.

## Readiness Verification

- Repository/archive preflight: pass after the authorized governance prerequisite.
- External-review package protocol: blocked; release readiness remains false.
- Protected-boundary milestone: fail.
- Failure is pre-read partitioning, not merely an output-filter failure. Synthetic reader spies demonstrated that pre-2023 files and mixed pre-/post-cutoff files enter `pd.read_parquet` before row filtering. Funding behaves the same way. A non-Kraken venue tag can pass through a Kraken-named symbol path.
- In accordance with the received task, the audit stopped at Milestone 3. U2/C01/C02/C03 were not promoted based on proxies or prior narrative.

Decision: `blocked_by_protocol_issue`. New economic work remains unauthorized.
