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

## Pre-price runner repair

- Reviewed freeze commit: `f7a8a63ce41a718a51d3b4168e4e4b61e6603d46`.
- First runner invocation stopped in timestamp-only schedule reconciliation before output-root creation. PF trade opens, returns, funding outcomes, protected rows, controls, KDA01, KDA02B, and Capital.com outcomes remained unopened.
- Root cause: the runner compared frozen economic definitions against all mechanical gate rows, including four deliberately omitted definitions belonging to infeasible continuation branches.
- The zero-outcome failure is preserved under `attempts/preprice_schedule_preflight_failure_v1`.
- Smallest repair: intersect the reconciliation gate rows with the complete frozen primary-definition identities and fail closed if that identity set differs. A regression covers an omitted infeasible gate row.
- Superseding v5 cache-only freeze: exit 0 in 46.60s; parent/event identities remained byte-identical; runner SHA-256 is `aa46407b43e1feb85fe79c9dc05b3b6c340192fbd70e01425a5d160649f579cf`; contract file SHA-256 is `2833fbab498ebdf3bf3d86801e442779ef1f3396fd5f32d5c8f3658402eb671d`; Level-3 contract hash is `5ca2ba8b762c4aa06b3c880a68112826764979d4e3f2f555316cece4248d280c`.
- Expanded relevant regression suite after the repair: 103 tests passed. Economic output root remained absent.
- Owner-side timestamp-only dry reconstruction after repair: 9,274 schedule rows, 8,999 accepted, all 8 definitions, and timestamp-authority hash `5fb1e9bbe2497f1d914f8f5d70f148039fa5879eb23b176b92756b057bbab50b`. Frozen primary accepted counts reproduced as `2812/2643/1017/988`; robustness counts were `617/604/159/159`. No outcome column or output root was opened.

## Frozen Level-3 run and post-run validation

- Reviewed repair commit: `971f694a4ab2cfcfb91e19c5d161d751c91d8e1a`.
- Single economic runner: exit 0 in 66.12s; maximum RSS 1,119,184 KiB; 9,274 schedules; 8,999 accepted trades; 275 overlap exclusions; 0 price rejections; 0 protected rows.
- Terminal decision: `KDA02_level3_no_primary_pass_stop`; primary passes: 0/4. Robustness remains diagnostic-only.
- Full official-open comparison: all 17,998 entry/exit uses across 54 symbols matched; owner check had zero mismatches, while the independent review identified only 12 repeated-use binary display differences below `4.1e-16` relative error and no substantive mismatch.
- Owner recomputation: exact long/short return and cost arithmetic, 3,087 market-day rows, all 80,000 bootstrap draws, concentration, and every gate flag matched.
- Relevant repository suite: 103 tests passed. Independent post-run focused suite: 28 tests passed.
- Independent post-run review: approved with no blockers; pricing, arithmetic, clustering, bootstrap, concentration, funding diagnostics, gates, claims, provenance, and artifacts matched.
- Controls, KDA02B outcomes, KDA01 outcomes, Capital.com payloads, and protected rows were not executed or opened.
