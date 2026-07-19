# KDA03 Stage 11 Closure and Lessons

Status: closed documentation of the published Stage 11 authority; no new economics.

## Authority and validity

- Published main: `0fb08802a1eaa44d379618d10882198a3c9d0e9a`.
- Stage 11 terminal status: `KDA03_level3_routes_assigned`.
- Contract hash: `5f1abdc7e21ab9e3a6851b21930a7af745cbb865be334f8be45c9a561d2411e7`.
- Artifact-manifest SHA-256: `07a9a18f75320d44703b42b3ed0d0a03fe143bba89c24a875ec5e4ac6a9b2856`; manifest content hash: `0bc85e5056db8ddb38e7977761e2fe657647cf2cb632e0447153bf64e1cd3af7`.
- The independent post-run review found no sign, timestamp, PF-open, cost, bootstrap, protected-data, or policy-routing defect. The first pre-outcome freeze was correctly blocked and repaired before outcomes.

These facts establish technical validity and reproducibility, not a strategy pass.

## All primary definitions and routes

| Direction | Mechanism | Horizon | Policy-v1.0 route | Control eligible |
|---|---|---:|---|---|
| Negative | basis impulse continuation | 1h | `translation_rejected` | no |
| Negative | basis impulse continuation | 6h | `translation_rejected` | no |
| Negative | completed-basis impulse rejection | 1h | `translation_rejected` | no |
| Negative | completed-basis impulse rejection | 6h | `sample_limited_prospective_candidate` | no |
| Negative | reference-led catch-up | 1h | `translation_rejected` | no |
| Negative | reference-led catch-up | 6h | `translation_rejected` | no |
| Positive | basis impulse continuation | 1h | `translation_rejected` | no |
| Positive | basis impulse continuation | 6h | `translation_rejected` | no |
| Positive | completed-basis impulse rejection | 1h | `translation_rejected` | no |
| Positive | completed-basis impulse rejection | 6h | `translation_rejected` | no |
| Positive | reference-led catch-up | 1h | `translation_rejected` | no |
| Positive | reference-led catch-up | 6h | `translation_rejected` | no |

KDA03A reference-led basis catch-up, KDA03B immediate leverage-backed continuation, and positive KDA03C completed rejection were rejected as exact translations. Eleven of twelve primary definitions were rejected; none may be reinterpreted as a pass.

## Sole preserved prospective object

The negative completed-basis rejection six-hour definition recorded:

```text
equal-market-day base mean:   +9.1570 bps
equal-market-day base median: +2.9323 bps
bootstrap lower:              -8.2953 bps
stress mean:                  -8.8430 bps
route:                        sample_limited_prospective_candidate
control eligibility:          no
```

The limitation is not literally a tiny raw trade count: the object contained 1,839 accepted trades, 691 market-day clusters, 1,839 parent episodes, and 49 symbols. Precision and threshold robustness were weak. Its separate evidence-limitation tags are `high_variance`, `wide_cluster_uncertainty`, `threshold_sensitive`, and `not_control_eligible`. All four are descriptive; `not_control_eligible` restates the binding policy gate and grants no route or evidence promotion.

The object is unvalidated, not live-ready, and cannot receive same-sample threshold, direction, symbol, year, cost, horizon, context, control, or robustness rescue.

## Hypothesis-design lessons

1. Most KDA03 definitions were near-flat gross rather than strongly wrong after removing the frozen 14-bps cost.
2. Standardized extremes were not tied to minimum raw economic magnitudes.
3. Normalization against prior daily summaries may make an “extreme” intraday state broader than its label implies.
4. A mechanism described as an extreme shock generated very frequent parent episodes.
5. KDA03A was underidentified because a complete executable reference-leg panel was not directly observed.
6. OI expansion cannot identify new-long, new-short, hedge, or arbitrage actor direction.
7. Positive and negative derivatives shocks are not structurally symmetric.
8. Slower-horizon negative completed derivatives-state rejection appeared in both KDA02 and KDA03, but neither object is validated.
9. A first plausible hand-written translation is not automatically the definitive confirmatory translation.
10. These lessons authorize no same-sample threshold rescue.

The prospective response is the staged [`Hypothesis Development Protocol`](docs/agent/HYPOTHESIS_DEVELOPMENT_PROTOCOL.md), not a reopening of KDA03.
