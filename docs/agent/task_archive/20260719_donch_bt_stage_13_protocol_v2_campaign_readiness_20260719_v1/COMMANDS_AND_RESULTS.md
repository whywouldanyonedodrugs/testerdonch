# Commands and results

- `git fetch origin --prune`: exact expected start found at `origin/main`.
- dirty recovery creation and SHA-256 verification: pass; 41 untracked paths recorded, ignored source ZIP recorded, tracked-ref bundle complete.
- safe ZIP path and `unzip -t`: pass; 21 entries extracted; 20 payload records verified against package manifest.
- `python3 tools/build_research_campaign_readiness.py --output ...`: pass and deterministic on replay.
- `python3 -m unittest unit_tests.test_qlmg_research_campaign -v`: 10/10 pass after adversarial-review repairs.
- `python3 -m unittest unit_tests.test_project_deep_cleanup_20260624 -v`: 5/5 pass.
- `unit_tests.test_sealed_slice_guard`: not runnable in this environment because the pre-existing test imports unavailable `pandas`; the new standard-library boundary tests passed.
- JSON parse, active-policy hash binding, historical CSV field comparison, protected-boundary assertions, package verification, and changed-Markdown link check: pass.
- `git diff --check`: pass.
