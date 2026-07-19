# Commands and Results

All commands ran in the isolated worktree unless stated otherwise. No market, outcome, protected, or Capital.com payload reader was invoked.

| Command or check | Result |
|---|---|
| `git fetch --prune origin`; commit/worktree/status inspection | `origin/main` verified at supplied Stage 9 commit `3ea0d320...`; original checkout dirty and three commits behind before fetch. |
| Dirty recovery patch, untracked inventory, Git bundle, and manifest checks | Recovery created outside the dirty checkout; reverse patch and Git bundle checks passed. An initial forward patch check against the already-modified source checkout correctly failed and was replaced by the applicable reverse check. |
| `git worktree add -b agent/stage10-documentation-gate-policy-20260719 ... 3ea0d320...` | Clean isolated task worktree created. |
| Received package byte/SHA-256 validation | 13 package-declared files matched; all 14 received files including `PACKAGE_MANIFEST.json` archived. |
| Stage 7C/8C1/9 authority and hash inspection | Analytics content hash, exact terminal tokens, Stage 9 manifest hash, and Stage 9 ZIP hash matched. |
| JSON/CSV parser and uniform-width checks | Active policy JSON parsed; 7 active CSVs parsed with uniform schemas. |
| Changed-document link checker and source-map resolver | Passed; historical broken links outside changed/active Stage 10 files were excluded and remain provenance. |
| Policy invariant assertions and route-enum checks | Passed after independent-review repairs. |
| `python3 -m pytest ...` | Not runnable: repository environment lacks `pytest`. |
| `python3 -m unittest unit_tests.test_project_deep_cleanup_20260624 unit_tests.test_sealed_slice_guard` | 5 tests passed; sealed-slice module import blocked because the environment lacks `pandas`. |
| `python3 -m unittest unit_tests.test_project_deep_cleanup_20260624` | 5/5 passed. |
| Secret scan of changed files | Passed with zero findings. |
| `git -c core.whitespace=cr-at-eol diff --cached --check` | Passed while retaining the exact received CRLF bytes as fully diffable evidence. |
| Scope check | Documentation/registry/archive only; zero code, data, test, result-root, or finalized-root changes. |

No economic run, return computation, controls, data acquisition, protected access, live action, order action, or Capital.com payload access occurred.
