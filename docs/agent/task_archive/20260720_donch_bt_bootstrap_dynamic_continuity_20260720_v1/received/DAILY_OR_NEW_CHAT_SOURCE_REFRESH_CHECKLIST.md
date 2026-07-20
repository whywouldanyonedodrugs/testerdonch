# Daily or New-Chat Project Source Refresh Checklist

Run this only once per day when material events exist, immediately after a material authority/protected-policy change, or before a new project chat.

1. Read and verify the dynamic continuity pointer and latest snapshot.
2. Enumerate immutable events after the prior `project_source_refresh_watermark`.
3. Compare every event with its immutable task handoff and repository evidence.
4. Classify each change:
   - dynamic-only;
   - current-state/decision update;
   - data-capability update;
   - method/policy update;
   - protected incident update;
   - campaign-packet lineage update.
5. Edit existing merged canonical sources. Do not add one file per stage.
6. Update machine-readable registries only where facts changed.
7. Keep project source count at 17–20 and never above 30.
8. Build a replacement bundle and manifest.
9. Upload replacements before deleting predecessors.
10. Verify project readback.
11. Publish a `source_refresh` event and advance the watermark.
12. Start the new chat with the supplied compact start prompt.

Routine negative tasks and handoff URLs remain in the dynamic ledger unless they change a terminal decision, reusable learning, authority, incident, or next programme route.
