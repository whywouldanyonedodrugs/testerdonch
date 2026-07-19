# Decisions and Progress

- Exact Stage 8C economic authorization confirmed; controls remain prohibited.
- Stage 8B1 contract, register, and cluster-tape hashes verified before price access.
- Analytics manifest, semantic contract, cohort, source event/parent, and Stage 8B manifest identities located and reconciled.
- Implementation will reuse Stage 8B1 scheduling and C01 funding authority; no alternative logic is permitted.
- The first economic invocation failed after open-price scoring but before funding/gates because the C01 funding root was worktree-relative. The partial root was preserved; a focused absolute-authority-path repair was applied without changing economics.
- One duplicate process was started accidentally while the original invocation was still active. It was stopped during pre-price schedule reconstruction, before creating a run root or opening price payloads; it was not a second economic execution.
- Repaired authoritative run completed in 4m24.69s with 2,463,012 KiB peak RSS and no swap.
- All 16 definitions executed; zero of eight primary definitions passed. Terminal decision: `KDA01_level3_no_primary_pass_stop`.
- Independent recomputation matched metrics, gates, cluster returns, and 160,000 fixed-seed bootstrap draws; 100 exact-open source checks passed.
