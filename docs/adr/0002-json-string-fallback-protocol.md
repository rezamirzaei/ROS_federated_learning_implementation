# ADR 0002: JSON-over-String fallback protocol

- Status: Accepted
- Date: 2026-04-17

## Context

The repository is migrating toward custom ROS interfaces, but dashboards,
standalone mode, and local tooling still rely on a transport that works without
the generated message/service packages.

## Decision

The system keeps a JSON-serialized fallback protocol over `std_msgs/String`
topics for cross-mode compatibility. Typed custom interfaces are used where
available, but JSON payloads remain the compatibility floor until every
consumer has migrated.

## Consequences

- Standalone and ROS dashboards can share payload schemas.
- Contract drift must be treated as an API change because JSON topics are
  consumed by tests, tooling, and browser code.
- New fields should be added compatibly and documented in architecture/API docs.
