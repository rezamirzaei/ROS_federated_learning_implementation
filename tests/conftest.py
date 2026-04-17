"""Pytest configuration for repository tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fl_robots.testing.fake_ros import FakeROSEnvironment


@pytest.fixture
def fake_ros() -> Iterator[FakeROSEnvironment]:
    """Install functional ROS fakes for the duration of a test.

    Any module that imports from :mod:`fl_robots.ros_compat` will see
    functional (non-raising) stand-ins for ``Node``, ``rclpy``, executors,
    and action servers. The fixture yields the environment so tests can
    drive subscribers, inspect publications, and fire timers directly.
    """
    with FakeROSEnvironment() as env:
        yield env


@pytest.fixture
def csrf_headers():
    """Return a helper that bootstraps the standalone app's CSRF cookie."""

    def _factory(
        client,
        *,
        cookie_name: str = "fl_robots_csrf_token",
        header_name: str = "X-CSRF-Token",
    ) -> dict[str, str]:
        client.get("/")
        cookie = client.get_cookie(cookie_name)
        assert cookie is not None, f"expected CSRF cookie {cookie_name!r} to be issued on GET /"
        return {header_name: cookie.value}

    return _factory

