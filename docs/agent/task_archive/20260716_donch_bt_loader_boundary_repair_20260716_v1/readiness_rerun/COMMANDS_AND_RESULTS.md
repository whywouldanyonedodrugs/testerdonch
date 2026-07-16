# Commands and Results

All inspections were run from `/opt/testerdonch`; no network, capture, or real market/protected payload was accessed.

| Command/inspection | Result |
|---|---|
| Git root/branch/HEAD/main/origin-main/status | root verified; task branch; authorized `992e7928...`; clean before task changes |
| Original readiness `ARTIFACT_MANIFEST.json` verification | 22/22 entries present and SHA-256 matched |
| Eight transferred authority source SHA-256 checks | 8/8 matched archived manifest |
| External-package `decision_summary.json`, stat, and schema matrices | metadata only; package remains blocked |
| Repository AST scan for raw parquet readers taking `paths` | five functions found; two repaired, three active siblings remain unguarded |
| Focused synthetic boundary tests | 8/8 passed after repair |
| Owning aggregate module | 286/286 passed |
| Repository guard modules | 9/9 passed |
| Compile | passed |

The repository-map `pytest` command was unavailable because the venv has no pytest. No dependency was installed; unittest-compatible repository modules were used.
