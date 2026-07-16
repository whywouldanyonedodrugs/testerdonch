# Commands And Results

All commands ran from `/opt/testerdonch` on 2026-07-16 UTC unless noted.

| Command or inspection | Purpose | Exit | Result |
|---|---|---:|---|
| `git status`, `git rev-parse`, `git fetch origin main` | Verify clean main, local/remote identity, governance object | 0 | Former local and remote main both `f81bb071...`; governance commit exists |
| `git cherry-pick 858e88d...` on temporary branch | Apply only governance commit patch | 0 | New commit `8cf3e227...`; no conflicts |
| Stable `git patch-id` comparison | Prove applied delta equals governance commit delta | 0 | Both `ea1b08144f887796c4427f76a77fd1a913acfdbe` |
| Governance scope/JSON/link/frontmatter/secret checks | Non-economic structural validation | 0 | 73 paths permitted; 7 JSON files parsed; 36 markdown files checked; 4 skills checked; 0 secret hits |
| Scoped `git diff --check` | Exclude documented exact-provenance whitespace | 0 | Pass |
| `./.venv/bin/python -m unittest unit_tests.test_project_deep_cleanup_20260624 unit_tests.test_sealed_slice_guard` | Governance/protection smoke | 0 | 9 run, 9 pass, 0 fail |
| `git merge --ff-only ...`; `git push origin main` | Authorized governance activation | 0 | Local and origin main `8cf3e227...`; no force push |
| `sha256sum` and `unzip -t` on authority ZIP | Verify transfer container | 0 | Expected SHA-256; all 10 entries structurally valid |
| Manifest verification script | Verify eight authority files | 0 | 8/8 byte counts and hashes pass; no undeclared entries |
| Metadata-only external-package inspection | Trace release status and gaps | 0 | `blocked_by_protocol_issue`, `release_ready=false`, full archive present |
| Synthetic `load_symbol_bars` reader-spy fixture | Verify pre-read time/venue exclusion | 0 | Protected-start chunk skipped, but pre-2023 and mixed 2025/2026 chunks were read; non-Kraken venue tag returned |
| Synthetic `load_funding` reader-spy fixture | Verify funding partition timing | 0 | Mixed 2025/2026 file was read before row filtering |
| `./.venv/bin/python -m unittest unit_tests.test_sealed_slice_guard unit_tests.test_qlmg_mechanical_qa_evidence_contract unit_tests.test_qlmg_d4_liquidation_execution_audit unit_tests.test_qlmg_d4_survivability_redesign` | Existing non-economic guard suite | 0 | 37 run, 37 pass, 0 fail |

No command read a market parquet payload, a candidate-return table, a 2026 outcome payload, or a capture root. Existing external-package candidate/event parquet files were listed by filename only and not opened.
