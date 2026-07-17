# Decisions and Progress

- Verified clean synchronized `main` at `7cd2444c92c5aecedeb1a897c919c25cbfd8749d` and created an isolated task branch.
- Verified all three accepted Stage 7A hashes exactly.
- Resolved the authoritative C03 identity inventory to finalized run `_173151` with 479 PF-symbol rows.
- Filesystem `/dev/sda1` is ext4 with approximately 38 GiB free and 8.5 million free inodes. The mandatory post-completion reserve is 50 GiB, so full acquisition is storage-blocked before projection.
- Phase A bounded audit and downloader preservation remain in scope; Phases B/C will not run.
- Implemented the resumable downloader and passed 58/58 synthetic/applicable checks after fixing single-file Parquet validation and SQLite test metadata extraction.
- Independent pre-acquisition review approved the bounded Phase A matrix and exact replay only; full acquisition remains prohibited.
- The first Phase A attempt was quarantined after 292 complete jobs when the final-window query used the end-exclusive boundary as inclusive `to`, allowing a protected-boundary timestamp into timestamp-only inspection. No protected data value was traversed, normalized, or written.
- Corrected URL construction to use `to=end_exclusive-interval`, require every query `to < 2026-01-01`, and added regression tests. All 60 tests pass.
- Preserved the partial run/data roots unchanged and created fresh attempt-2 roots for the corrected audit.
- Corrected Phase A completed 1,080/1,080 jobs with zero retries, 1,659,672 page rows, 178,593,792-byte peak RSS, and exact 144/144 logical replay.
- Verified 2,000-row response cap, inclusive lower/upper bounds, 792 continuation pages and 792 identical cross-page boundary duplicates.
- OI was empty for all six symbols in the first January 2023 window; liquidation and basis populated all audit windows.
- No metric passed the official unit/sign gate. Storage projection also fails the 50 GiB post-completion reserve. Final decision frozen as `blocked_by_units_pagination_or_storage`; Phases B/C did not launch.
- Added fail-closed Phase B/C entry points that require an independently reviewed plan with approved metrics and a passing storage gate. Monthly production requests are strictly pre-protected and were tested without launching acquisition.
- Final focused and applicable repository suite passed 64/64 tests.
