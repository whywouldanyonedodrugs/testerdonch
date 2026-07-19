# Commands and Results

All commands ran from the isolated Stage 12 worktree unless noted.

| Command or check | Result |
|---|---|
| `git fetch origin main --prune`; commit/remotes/worktree/status inspection | `origin/main` and requested base matched `0fb0880…`; primary checkout dirty and isolated. |
| SHA-256 checks for policy and Stage 11 manifest; manifest JSON/decision CSV inspection | Supplied policy, contract, manifest file/content hashes and terminal status matched. |
| `python3 -m unittest unit_tests.test_hypothesis_development_protocol -v` | 6 tests passed. |
| `python3 -m unittest unit_tests.test_project_deep_cleanup_20260624 -v` | 5 tests passed. |
| `python3 -m pytest …` | Not started: optional `pytest` module absent. No installation attempted. |
| `python3 -m unittest …test_sealed_slice_guard…` | Module could not import because optional `pandas` is absent; the five cleanup tests in the same invocation passed. No installation attempted. |
| Dependency-free JSON/CSV/link validator | 2 changed JSON files, 5 changed CSV files, and local links in 12 changed Markdown files passed at the recorded checkpoint. |
| Focused secret-pattern scan | 0 findings at the recorded checkpoint. |
| `git diff --check` | Passed at the recorded checkpoint. |
| Policy SHA-256 | `c54d4a3445da249c6dbd34613b770ad9624f861e0d7c345b3cba944aa6cdf1aa` (unchanged). |
| Protocol SHA-256 | `1d405f6cc94f6c5d9409751bafc5d911e26b508f320506343f66ab1de52957a9` at the recorded checkpoint. |
| Limitation-tag registry SHA-256 | `06d6705cf6c81f60a82fc3f7936974bd187025d96a28ca931efe24281573ef97` at the recorded checkpoint. |
| Recovery validation | 5/5 bundle-file hashes, tracked-refs bundle, reverse patch check, and 41/41 untracked metadata hashes passed. |
| Local package | ZIP integrity and 32/32 embedded artifact-manifest entries passed; direct secret findings `0`. |
| Drive collision/read/upload/round trip | Exact `v01` folder absent before write; 5 files uploaded without overwrite; 5/5 downloaded sizes and SHA-256 values matched. |

No command imported a market reader, strategy runner, return calculator, bootstrap module, or protected/Capital.com payload.
