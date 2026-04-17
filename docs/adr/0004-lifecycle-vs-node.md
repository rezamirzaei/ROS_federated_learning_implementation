# ADR 0004: Lifecycle node for aggregation, plain nodes elsewhere

- Status: Accepted
- Date: 2026-04-17

## Context

Not every component in the system benefits equally from ROS lifecycle
management. Aggregation owns a critical readiness boundary, while several other
nodes are closer to stateless adapters or visualizers.

## Decision

The aggregator remains a ROS `LifecycleNode`, while coordinator, monitor,
digital twin, and robot agents stay as plain nodes. This gives the global-model
publisher explicit activation semantics without forcing every component through
the additional lifecycle complexity.

## Consequences

- Aggregation readiness is explicit and testable.
- Simpler nodes keep a lower operational burden.
- If another node needs lifecycle semantics later, the default is to justify it
  with a new ADR instead of adopting it implicitly.
