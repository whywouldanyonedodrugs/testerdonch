# Drive Identifier Retention

The task specification explicitly provided exact Google Drive folder and file IDs and required preserving the exact received task, transfer manifest, and handoff evidence in the repository task archive.

Security classification:

- These IDs are stable external resource identifiers, not credentials.
- No OAuth tokens, refresh tokens, passwords, private keys, or rclone configuration contents were copied.
- The branch was not pushed or published.

Decision:

- Preserve exact IDs in `TASK_SPEC.md`, `received/TRANSFER_MANIFEST.json`, and handoff records because they are required provenance and verification evidence.
- Avoid relying on the IDs as an authorization substitute; every remote write still requires the exact task authorization, collision check, local ZIP hash, and round-trip verification.
- Treat the IDs as private project metadata if this branch or archive is later shared beyond the approved destination.

Residual risk: if the branch is later pushed to a broader audience, the external resource identifiers become visible to that audience. This task did not authorize a push.
