# Decisions And Progress

- Task scope frozen to four official candle slices and bounded terminal-lifecycle authority.
- No mixed-chunk fallback, economic analysis, protected payload access, or capture access authorized.
- Four exact official candle requests returned HTTP 200 with 288 rows each, `more_candles=false`, and no timestamp at or after the protected boundary.
- The official cumulative Kraken derivatives delistings page was cached and parsed. XBT and ETH are absent from its terminal-event table; that absence is explicitly not a no-outage claim.
- The first real parser pass failed closed on an official trailing footnote marker (`PF_RIVERUSD*`). The parser now preserves the source symbol while removing only trailing asterisks for canonical lookup; a synthetic regression covers that form.
- Independent review found and repaired an initially implicit prior-coverage assumption. The builder now requires the accepted Stage 2A authority rows to end exactly at `2025-12-31T00:00:00Z` before joining the new bounded day.
- The reference panel is factor/reference-only. Continuous-tradeability and survivorship-free claims remain `no`.
- Transient public CDN `Set-Cookie` values were redacted from retained response-header files; raw response bodies remain byte-exact.
