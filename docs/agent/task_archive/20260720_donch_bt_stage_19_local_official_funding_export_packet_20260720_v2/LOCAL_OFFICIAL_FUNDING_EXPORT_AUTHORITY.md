# Local Official Kraken Funding Export Authority

The source is the human-transferred file `/opt/testerdonch/research_inputs/exports.zip` (109,980,750 bytes; SHA-256 `65ba6712a6ab657389d2795d3ed77bedb4270841dfe711147ae9df16e366edab`). Filesystem mtime and ctime are both `2026-07-20 09:15:35.633840485 +0000`.

Kraken's official support page `https://support.kraken.com/articles/export-historical-funding-rates` was retrieved at `2026-07-20T09:40:56Z`; the retrieved HTML SHA-256 was `0d849b0b783af9e9a222292d58763f3275d0cb41f6e83bdfb098e893ca07a5bf`. Its current export link is hosted by `assets-cms.kraken.com`.

Authority classification: `human_transferred_official_export + official_support_page_provenance + local_content_hash`. No published upstream content hash or exact remote-byte comparison was available, so independent remote-byte identity is not claimed.

The 964 members comprise 480 funding CSVs, 482 AppleDouble metadata files, one `.DS_Store`, and one directory entry. AppleDouble and `.DS_Store` are explicitly excluded Finder metadata; every other non-directory member is required to be an exact `exports/*.csv` funding payload.

The final immutable streaming packages are under `/opt/testerdonch/results/rebaseline/phase_kraken_local_official_funding_export_20260720_v4`. Campaign code may access only `kraken_funding_rankable_2023_2025.zip`.
