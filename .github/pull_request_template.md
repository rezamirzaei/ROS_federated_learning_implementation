## Summary

<!-- What does this PR change and why? Link related issues. -->

Closes #

## Changes

- [ ] Code change (behaviour-impacting)
- [ ] Refactor (no behaviour change)
- [ ] Documentation
- [ ] Tests / CI
- [ ] Dependency / tooling bump

## Checklist

- [ ] `uv run python -m pytest` passes locally
- [ ] `uv run ruff check src tests` is clean
- [ ] `uv run mypy src/fl_robots` is clean (or the new scope is opted in)
- [ ] Coverage did not drop below the 60 % gate
- [ ] New public APIs are documented (docstring + README/ARCHITECTURE)
- [ ] Benchmarks re-run if `scripts/benchmark.py` or aggregation logic changed
- [ ] Security-sensitive change? Updated `docs/SECURITY.md` threat model.

## Reproduction / manual test

<!--
Commands you ran and what you observed. Paste relevant snippets of
`pytest` / `benchmark.py` output — especially for non-IID experiments.
-->

```bash
uv run python -m pytest -q
```

## Screenshots / traces

<!-- Dashboard screenshots, Grafana panels, or perf traces if relevant. -->
