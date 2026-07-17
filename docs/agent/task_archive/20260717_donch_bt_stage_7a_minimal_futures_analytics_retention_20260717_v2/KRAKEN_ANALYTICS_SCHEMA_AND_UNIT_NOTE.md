# Kraken Analytics Schema and Unit Note

- Request `since` and `to`: **verified** as epoch seconds from official documentation.
- Request `interval=3600`: **verified** as a documented allowed interval.
- Response timestamp unit: recorded per cell as documented seconds, inferred milliseconds, ambiguous, or unavailable.
- Open-interest value unit: **unavailable**.
- Funding value unit and sign: **unavailable**.
- Liquidation-volume value unit: **unavailable**.
- Future-basis value unit and sign: **unavailable**.

No value semantics were inferred from prices, returns, or outcomes.
