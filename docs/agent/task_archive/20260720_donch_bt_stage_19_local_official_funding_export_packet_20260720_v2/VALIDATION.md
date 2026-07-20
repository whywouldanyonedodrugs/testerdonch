# Stage 19 Validation

Status: `pass`; packet remains non-authorizing.

## Authority and partition

- Source ZIP: `/opt/testerdonch/research_inputs/exports.zip`, 109,980,750 bytes, SHA-256 `65ba6712a6ab657389d2795d3ed77bedb4270841dfe711147ae9df16e366edab`.
- ZIP inventory: 964 members, including 480 funding CSVs. All three supplied sample hashes matched.
- Rankable package: 5,658,890 rows, 476 symbols, interval `[2023-01-01T00:00:00Z,2026-01-01T00:00:00Z)`, SHA-256 `6c0969727c882ca6439c57c3b7d03367e1b2d26ee46823e56b983756d026ef64`.
- Protected quarantine: 236,786 rows, 320 symbols, SHA-256 `0957b7253e76840f6062f23290ed8b53aea4714f5c3cbaa314e7fba0975b37d0`.
- Pre-rankable exclusion: 277,640 rows, SHA-256 `5e3ca567998e0e35f9f4e6db027533a14bc8f778738fcc1eff744ec6bc7e73e3`.
- Unknown or invalid rows: zero.
- Protected funding values used for statistics: zero.
- Protected strategy price or return rows opened: zero.

## Funding and campaign authority

- Campaign symbol mapping, unit verification, and allowance coverage each contain the same 187 symbols.
- Unit audit SHA-256: `9729b8a87947be3303a3e00febef47413b4c66c1d4395b1c1ac82b0b076bdc87`.
- Unit-source manifest SHA-256: `9074a6bbefe9ab7fd3c579c6b90260f1b244c8e4f52f79e60f180a170e4110f0`.
- Dual-alignment contract SHA-256: `840729c812eed312759a63c9e85a68c2b91075309d45b9cca4849c92d6fccc50`.
- Gap allowance table SHA-256: `4ff2c6bcbe125e6f966348a029cd2cc63f55deb752b89c454a28642c41f119b5`.
- Funding cost contract SHA-256: `c0d5bf091080a681055a792a8a7cf4a1084dcd60deb17c60381339df7297c3e7`.
- Economic translation registry SHA-256: `e48fce5186238064496c71457b72ee3beeb8f420f009b8bb66508c773d813774`.
- Campaign manifest SHA-256: `e7d618a2a24c574c9ba83d323df605a6434c2e816fe24c465d183ba5d6256990`.
- Replacement approval packet SHA-256: `3fde09f16efff2479ee847fe26c859a67ef1d516b0f0acbcbaec141154a87bd6`.
- Synthetic canary: pass; both alignments, partial hours, both position signs, both rate signs, nonpositive gap charges, ignored favourable credit, unique economic addresses, and 186-cell determinism were exercised. Economic outputs computed: zero.

## Executed checks

The independent validator returned `status=pass`, 186 registered cells, 187 campaign symbols, and passes for dual alignment, gap allowance, unit verification, runtime adapter, and synthetic canary. It also confirmed zero protected strategy price/return rows and that a new external human approval is required.

Focused unit coverage includes ZIP security and streaming, Decimal funding arithmetic, hash-bound runtime behavior, temporal boundaries, nonnegative allowance validation, packet regeneration, semantic rebound mutation rejection, protected-safe funding regression, and the inherited Stage 16 semantic suite.

No campaign economics, Capital.com payload, Telegram message, order, or live-trading action occurred.
