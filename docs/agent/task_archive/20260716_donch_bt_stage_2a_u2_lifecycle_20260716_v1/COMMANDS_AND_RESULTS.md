# Commands and Results

| UTC | Command/inspection | Exit | Result |
|---|---|---:|---|
| 2026-07-16 | Git root/status/main/authority preflight | 0 | clean aligned main at c198cf0 |
| 2026-07-16 | Inspected local instrument metadata and the official Kraken download-manifest metadata for PF_XBTUSD/PF_ETHUSD trade and mark coverage | 0 | Stable current identities; 219 rankable chunks per symbol/dataset; metadata coverage ends exclusively at 2025-12-31T00:00:00Z; no market payload opened |
| 2026-07-16 | Retrieved current official Kraken instruments, status metadata, official support search responses, and the Wayback CDX index | 0 | Cached URL, headers, body, access time, response status, and SHA-256 under `sources/` |
| 2026-07-16 | Retrieved all ten archived official instrument API snapshots listed by CDX for 2023-2025 | 0 | PF_XBTUSD and PF_ETHUSD present and tradeable in every checkpoint; archive cadence still has unresolved gaps |
| 2026-07-16 | `.venv/bin/python -m py_compile tools/build_kraken_u2_lifecycle_authority.py unit_tests/test_kraken_u2_lifecycle_authority.py` | 0 | PASS |
| 2026-07-16 | `.venv/bin/python -m unittest -v unit_tests.test_kraken_u2_lifecycle_authority` | 0 | PASS, 8 tests |
| 2026-07-16 | `.venv/bin/python -m unittest -v unit_tests.test_rankable_loader_boundary unit_tests.test_sealed_slice_guard` | 0 | PASS, 15 tests |
| 2026-07-16 | `.venv/bin/python -m pytest ...` and `python3 -m pytest ...` | 1 | Not run: pytest is not installed in either interpreter; no dependency was installed |
| 2026-07-16 | Generated all seven required U2 outputs from metadata-only inputs | 0 | 2 considered, 0 included; protected payloads opened 0; economic outputs 0 |
| 2026-07-16 | Regenerated all seven outputs into an isolated temporary directory and compared bytes | 0 | PASS, 7/7 deterministic comparisons |
| 2026-07-16 | Independent schema/source/hash audit | 0 | PASS: 20 source entries, all source hashes/URLs/access times valid, required columns present, all artifact hashes valid |
