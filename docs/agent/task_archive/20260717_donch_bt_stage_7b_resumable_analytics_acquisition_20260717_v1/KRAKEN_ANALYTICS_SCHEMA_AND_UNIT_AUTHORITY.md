# Kraken Analytics Schema and Unit Authority

| Metric | Observed exact value shape | Field authority | Unit/sign authority | Acquisition gate |
|---|---|---|---|---|
| `open-interest` | array of four strings per timestamp | endpoint/schema observed | meanings, contract interpretation, and units unavailable | blocked |
| `liquidation-volume` | one string per timestamp | endpoint/schema observed | volume unit and directional structure unavailable | blocked |
| `future-basis` | `data.basis`, one string per timestamp | field name verified by official schema | basis unit and sign convention unavailable | blocked |

Request timestamp units and interval alignment are verified. Value units were not inferred from prices, returns, or profitability. The official Stage 7A schema snapshot names fields but does not supply the minimum economic unit/sign semantics required by Stage 7B. Therefore no metric is approved for Phase B or C acquisition.
