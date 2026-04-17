# ADR-0005: Standalone CSRF Strategy

**Status:** Accepted  
**Date:** 2026-04-17

## Context

The standalone dashboard exposes mutating endpoints (`/api/command`) over plain
HTTP. When bearer-token auth (`FL_ROBOTS_API_TOKEN`) is not configured, the
endpoints are open to any local user. A CSRF attack from a malicious page could
issue commands (reset, inject disturbance) on behalf of a logged-in operator.

## Decision

We implement a **double-submit cookie** CSRF pattern:

1. On first response, set a `SameSite=Strict` cookie with a random token.
2. Mutating POST endpoints require an `X-CSRF-Token` header matching the cookie.
3. When bearer auth is configured, CSRF is not required (the `Authorization`
   header is sufficient and cannot be set cross-origin by a browser).

## Consequences

- Zero-config local use is protected against CSRF without requiring a login flow.
- The standalone JS bundle reads the cookie and attaches the header automatically.
- Bearer-auth deployments are unaffected.

