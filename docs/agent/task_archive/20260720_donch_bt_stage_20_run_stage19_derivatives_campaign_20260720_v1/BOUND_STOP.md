# Bound-stop record

Status: `reconciled_bound_stop_incomplete_campaign`

At `2026-07-20T15:10:26Z`, the campaign supervisor stopped after a transient
`ProcessLookupError: [Errno 3] No such process` during process-tree resource
sampling. The stop was fail-closed: new submissions were disabled, no queued
future remained, the at-most-three already-running futures were allowed to
exit, atomic state was preserved, and the Telegram global-stop notification
was delivered.

The attempt is not terminal-complete and no promotion, route, or economic
claim is made. Health release remained pending because the first scheduled
30-minute heartbeat had not occurred. Metadata-only reconciliation found
6,459 completed job markers, 3,193 artifact claims, 126 registered cell
identities represented, zero missing or hash-mismatched artifacts, zero
protected rows, and no Capital.com payload access. No partial economic values
were opened for reconciliation.

The resumable root remains
`/opt/testerdonch/results/rebaseline/phase_kraken_derivatives_campaign_stage20_20260720_v01`.
