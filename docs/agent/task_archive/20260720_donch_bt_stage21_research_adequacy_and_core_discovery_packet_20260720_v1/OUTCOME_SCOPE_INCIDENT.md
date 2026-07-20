# Outcome-scope incident

Status: `review_isolation_accepted; final_packet_review_pending`

At 2026-07-20T17:45:36Z, a read-only command intended to locate and inspect
prior definition manifests also printed the first two rows of three prior
2023–2025 economic summary CSVs. This was outside the task's Stage-20-terminal-
only outcome-review scope.

No protected-period or Capital.com payload was opened, no new economic run was
started, and no printed prior value is used in this packet. The complete
3,521-address economic registry had already been frozen and byte-replay
validated at SHA-256
`55c5af1750fe93c694eb99c152cff7f4a70e778f476197fe717fa2c57c7457b1`.
The pre-incident campaign manifest and approval-request hashes remain
`00dec3bf855ce9e87e0f448880687ce68cf83ce61af28665761287376694e9b6` and
`16fffa70b4231cbcb82ef1efc6f0eb3fec1ff37a6ae4e51dca59befb6bb9728a`.

The independent reviewer initially accepted isolation for the unchanged freeze,
then correctly blocked the first post-incident correction because its new exact
anchors were not independently chosen. That version was not published. A
separate outcome-isolated constructor then specified the complete anchor/null
table, source IDs, family-specific controls, distinct KDA variants, and A4/A3
aggregation rules using source manifests and mechanism rules only. The parent
applied that specification verbatim. No displayed economic value was used to
choose a field, level, anchor, address, or rule. The final registry is
`d582d8e2e5390f7f833849494cd7b4751eff667db99221ce7a8a8c27bf42f8b3`;
the final manifest and approval-request hashes are
`edeb26248831791b90c7aede801afd5a258ecf4f61710eb3146852b94b4f34c7` and
`a21f01f18525134db4c2c2b017e6a155ebe99d76ab886cb031fda797f011b977`.
Publication remains conditional on a fresh independent review of this exact
post-correction packet and the incident history.
