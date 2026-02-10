# Corrected Mini-Sweep Spec

Date: 2026-02-09  
Owner: Offline team

## Objective

Find a deployable policy after leakage/sizing-contamination fixes, with minimal re-optimization bias.

## Why mini-sweep is the correct quant approach now

1. The prior full-grid ranking was contaminated by replay-feature leakage and sizing artifacts.
2. Re-running the full grid immediately is expensive and increases multiple-testing risk.
3. A pre-registered, smaller hypothesis set with explicit controls is better science for urgent decisions.

## Data windows

1. Corrected IS window: `2023-01-01` to `2025-11-15` (same historical base for comparability).
2. OOS acceptance window: `2025-11-14` to `2026-02-08` using newer parquets (for forward check).

## Pre-registered variant set (13 runs)

Run these exactly; do not add variants mid-run.

| id | purpose | META_PROB_THRESHOLD | META_GATE_SCOPE | META_GATE_FAIL_CLOSED | META_SIZING_ENABLED | REGIME_BLOCK_WHEN_DOWN | RISK_OFF_PROBE_MULT | REGIME_SLOPE_FILTER_ENABLED |
|---|---|---:|---|---:|---:|---:|---:|---:|
| C0 | pure baseline, full-size | `None` | `all` | `False` | `False` | `False` | `1.00` | `False` |
| C1 | pure baseline, probe risk | `None` | `all` | `False` | `False` | `False` | `0.05` | `False` |
| S0 | sizing-only effect | `None` | `all` | `False` | `True` | `False` | `0.05` | `False` |
| G0 | gate+size all | `pstar` | `all` | `False` | `True` | `False` | `0.05` | `False` |
| G1 | gate+size scoped | `pstar` | `RISK_ON_1` | `True` | `True` | `False` | `0.05` | `False` |
| R0 | blockdown sensitivity | `None` | `all` | `False` | `True` | `True` | `0.05` | `False` |
| R1 | slope sensitivity | `None` | `all` | `False` | `True` | `False` | `0.05` | `True` |
| R2 | probe sensitivity low | `None` | `all` | `False` | `True` | `False` | `0.01` | `False` |
| R3 | probe sensitivity zero | `None` | `all` | `False` | `True` | `False` | `0.00` | `False` |
| RG0 | gated + blockdown | `pstar` | `all` | `False` | `True` | `True` | `0.05` | `False` |
| RG1 | gated + slope | `pstar` | `all` | `False` | `True` | `False` | `0.05` | `True` |
| RS0 | scoped + blockdown | `pstar` | `RISK_ON_1` | `True` | `True` | `True` | `0.05` | `False` |
| RS1 | scoped + slope | `pstar` | `RISK_ON_1` | `True` | `True` | `False` | `0.05` | `True` |

Notes:

1. `pstar` means the threshold from the newly exported bundle, not the old one.
2. Keep `BT_META_REPLAY_ENABLED=False` for every run.

## Ranking and acceptance rules

Primary ranking metric:

1. OOS risk-adjusted utility using corrected mechanics.

Hard filters:

1. OOS trade count must be sufficient for signal confidence (set minimum before run, for example >= 100).
2. OOS max drawdown must be within pre-declared limit.
3. Flat-size control (`C0`) must show non-negative edge over OOS (or strategy is not production-ready).

Tie-breakers:

1. Monthly consistency (avoid one-month-only profits).
2. Lower dependence on extreme size concentration.

## Required diagnostics per run

1. Standard metrics (PnL, DD, Sharpe, PF, win rate).
2. Size diagnostics (`size_mult` distribution).
3. Unit-risk diagnostics (`pnl / size_mult`) to separate alpha from sizing.
4. Loss-size profile to detect asymmetric tiny-loss artifacts.

## Decision policy

1. If no variant passes hard filters, do not promote full-size live.
2. If one variant passes, deploy as canary first.
3. Full-size only after canary parity and stability checks pass.

