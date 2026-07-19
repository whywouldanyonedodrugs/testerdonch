# Timestamp Semantics Audit

Defect confirmed independently of repaired returns. Official interval candles are start-labeled; Stage 8A makes the completed bar available at `time + 5m`; Stage 8B1 incorrectly required the entry timestamp to be strictly greater than that availability and therefore skipped the causally executable next bar.
