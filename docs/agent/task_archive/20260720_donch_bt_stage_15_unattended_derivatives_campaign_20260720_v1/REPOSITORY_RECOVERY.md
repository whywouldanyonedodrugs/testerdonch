# Repository recovery and isolation

- Recovery root: `/opt/testerdonch-stage15-dirty-recovery-20260720`
- Recovery hash ledger: `RECOVERY_HASHES.sha256`, SHA-256 `769ba512c5f517f943e1589bf532e83a1a7e9e266f68f5c5baf110aebde5055f`
- Validation: every file named by the ledger passed `sha256sum -c`.
- Preserved forms: porcelain-v2 state, staged and unstaged binary-capable patches, untracked size/hash inventory, worktree inventory, and a Git bundle of original HEAD.
- Secret exclusion: ignored `.telegram.env` content was not copied or archived.
- Original checkout modified by Stage 15: no.
- Isolated worktree: `/opt/testerdonch-stage15-20260720`
- Branch/base: `agent/stage15-unattended-derivatives-campaign-20260720` from `50dffb791c146b359cb210532e5f7291774e26f0`.
