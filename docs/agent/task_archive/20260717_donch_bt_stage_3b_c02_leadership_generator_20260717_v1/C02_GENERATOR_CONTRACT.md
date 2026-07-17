# C02 Generator Contract

Status: frozen before generation. Family: `C02_spot_led_vs_perp_led_impulse`.

## Authority and boundaries

Only official sparse Kraken USD spot five-minute bars, manifest-authorized Kraken PF trade/mark five-minute bars, the Stage 2C prior-day Top-100 panel, and known terminal lifecycle invalidations are inputs. The interval is `[2023-01-01, 2026-01-01)`. No sparse interval is filled. The cohort remains current-roster capped and not survivorship-free.

## Causal rules

Exact aligned completed bars use `feature_available_ts = interval_open + 5m`. Fifteen-minute returns require four consecutive observed five-minute boundaries. Daily sample scales use the preceding 30 UTC days only and require 2,000 observations. Activation requires same sign, maximum directional z at least 3.0, minimum at least 1.5, and a 60-minute same-direction reset. Primary/robustness leadership lookbacks are 15/30 minutes and use the first follower-threshold crossing. Perp-led failure requires the first PF trade and mark close beyond the three-bar impulse-window extreme within six hours.

## Eligibility and identity

A symbol-day requires Stage 2C Top-100 membership, exact spot identity and USD unit identity, at least 20 of the prior 30 days at 70% exact intersection, at least 70% complete-window intersection, valid scales, and no known lifecycle-invalid overlap. Event, economic-address, and overlapping same-symbol episode identities are deterministic and outcome-free.

## Prohibitions

No post-decision return, exit, PnL, MAE/MFE, control result, promotion metric, funding, OI, index, breadth, session, catalyst, prior-high, C01 residual, interpolation, event selection by magnitude, or economic ranking is permitted.
