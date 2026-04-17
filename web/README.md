# Frontend smoke tests

Playwright-driven checks for the standalone dashboard. Validates the shipped
static bundle wires up correctly, the API returns the expected shapes, and
security headers are present on every response.

## Install

```bash
cd web
npm install
npm run install-browsers
```

## Run

```bash
# Starts the Python server automatically (see playwright.config.ts).
npm test
```

Override the base URL (e.g. when pointing at a remote deployment):

```bash
FL_ROBOTS_BASE_URL=https://dash.example.com npm test
```

## CI

Not wired into the default CI workflow by default (keeps PRs fast). Enable
by adding a `frontend-smoke` job to `.github/workflows/ci.yml` that runs
`npm ci && npm run install-browsers && npm test` inside `web/`.
