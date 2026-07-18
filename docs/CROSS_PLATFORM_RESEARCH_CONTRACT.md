# Directed Cross-Platform Research Contract

This document defines identity and authority boundaries only. It does not authorize an economic run.

Cross-platform hypotheses are directed. `Capital.com -> Kraken` and `Kraken -> Capital.com` are different contracts and receive different canonical IDs. Every directed contract must freeze:

- source and target platform;
- source and target instrument identity and mapping authority;
- source availability time and target decision/execution time;
- platform-specific calendar, lifecycle, price, cost, financing/funding, and fill semantics;
- point-in-time universe and protected-period rules;
- candidate and control identities before outcomes.

Platform-specific mechanics must not be collapsed into one generic execution model. Capital.com bid/ask CFD quotes cannot be relabeled as Kraken trades; Kraken funding cannot be relabeled as Capital.com financing. Current metadata cannot establish historical lifecycle state.

An approved acquisition inventory is not an approved research universe. Each economic run requires its own frozen hypothesis, universe, directed mapping, multiplicity record, cost model, and explicit authorization. Existing Kraken IDs, hashes, decisions, and artifacts remain unchanged.
