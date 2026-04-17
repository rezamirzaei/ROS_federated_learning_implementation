# ADR 0001: Dual-mode ROS shim

- Status: Accepted
- Date: 2026-04-17

## Context

The project must run both as a real ROS 2 deployment and as a pure-Python
standalone simulation for local development, CI, and lightweight demos. Direct
imports of `rclpy` in first-party modules would make the standalone path and
most tests fail on machines without ROS 2 installed.

## Decision

All first-party ROS-facing modules import ROS primitives only through
`fl_robots.ros_compat`. The shim re-exports real ROS types when `rclpy` is
available and otherwise provides local stubs that preserve importability and
basic type shapes.

## Consequences

- Standalone tests and tooling can import the package without ROS 2.
- The seam between ROS mode and stub mode stays centralized.
- Any future ROS dependency should be added to `ros_compat`, not imported
  directly from node modules.
