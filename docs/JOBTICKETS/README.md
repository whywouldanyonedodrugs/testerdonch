# Jobticket System

This folder tracks implementation work as jobtickets.

## Ticket Id Format

- `JT-XXX`, zero-padded integer.

## Status Values

- `todo`
- `in_progress`
- `blocked`
- `done`

## Priority Values

- `P0` critical parity/safety risk
- `P1` high impact, near-term
- `P2` medium impact
- `P3` nice-to-have

## Estimation

- `S` <= 0.5 day
- `M` 1-2 days
- `L` 3-5 days
- `XL` > 5 days

## Evaluation Impact

Evaluation impact is estimated as expected increase in confidence that:
- live and backtest use equivalent decision mechanics
- research outputs are realistic and not leak-contaminated

Use:
- `confidence_delta_points` on a 0-100 confidence scale
- optional notes on risk reduction and false-alert reduction

## Files

- `BACKLOG_*.md`: active prioritized backlog
- `TEMPLATE.md`: ticket template for new items

