# Commands and results

## Preflight and preservation

- Git root, HEAD, branch, upstream, worktrees, remotes, submodules, sparse-checkout, instruction chain, and dirty state: verified.
- Expected starting commit: matched `baaa10c224807e1dc7e32bfee7227711cb0c1279`.
- Stage 8C1 repaired run manifest SHA-256: matched `2e8d8cce4ed5507bbac11b86d98158e7bcfc3ce60905c82c1cc22224636eff05`.
- External dirty-state recovery bundle: Git bundle verified, safe-copy TAR listed successfully, manifest JSON validated.
- Stage 8A commit, semantic/feature/generator/data/cohort identities, event-tape SHA-256, and six KDA02 attempt counts: matched.

## Outcome-free implementation and generation

- Focused compile and KDA02 v2 unit suites: passed.
- Two-symbol outcome-free smoke: passed; counts reproduced by the full run.
- The first complete freeze and its blocked independent review are preserved under `attempts/pre_review_blocked_v1`; no outcome was opened. The review identified strict-window, retained-material-OI, and cross-horizon funding-address defects.
- The three defects were repaired with exact irregular/duplicate/timestamp-resolution boundaries, a frozen `current_oi <= onset_oi_close` retained-reset rule, and unique definition-event funding addresses. Focused repaired suites: 27 tests passed.
- Superseding full 187-symbol generator command: `/usr/bin/time -v /opt/testerdonch/.venv/bin/python tools/build_kda02_v2_prerun_freeze.py --output docs/agent/task_archive/20260719_donch_bt_stage_9_kda02_purge_adjudication_conditional_level3_20260719_v1 --cache /opt/parquet/kraken_derivatives/analytics/stage9_kda02_v2_prerun_v4`.
- Superseding generator result: exit 0; 25m26.50s wall; 2,062,844 KiB maximum RSS; 8,281 parent episodes; 5,128 events; 2 feasible primary branches; 8 frozen definitions; 0 protected rows; 0 economic outputs.
- Cache-only deterministic replay through `/tmp/kda02-v4-replay.jhp1FN`: exit 0 in 48.12s. Parent tape `f5bc2dfb4b315d005e4444656cd69ef17a48751c4711ffe8409cdaef45280206`, event tape `ed777c58a3fac84007d49be8b9a7f7d04b7ab50911b12363fc1a5a575005a378`, contract file `1d595603a46359dd61552280f7ce169d2455f19f7515dad69bab9f6da672b255`, definition register `6f0165c58009b554d0877db58282cad204d837444c16e39fb17cc264a55846fb`, and feasibility gates `866fd4f740c1ac22accba1a1bfd964c659948946d0a26fa5548152fbc9133f33` reproduced byte-for-byte.
- Relevant regression command covering Stage 8A, KDA01 timestamp/non-overlap/economic primitives, and KDA02: 102 tests passed.
- Fresh independent pre-outcome review: approved with no blockers; all nine runner-preflight hashes matched; 27 focused reviewer tests passed; no outcome was opened.
- `git diff --check`: passed before and after independent review.

## Authorization state

- PF trade-open outcomes read before independent approval: no.
- Funding outcomes read: no.
- Controls/KDA02B/KDA01/KDA03/Capital.com outcomes: no.
- Protected rows opened: no.
