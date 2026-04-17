"""Security regressions for the ROS dashboard Flask app."""

from __future__ import annotations

import threading
from pathlib import Path

from fl_robots.web_dashboard import build_dashboard_app


class _FakeDashboardNode:
    def __init__(self, output_dir: Path) -> None:
        self.state_lock = threading.Lock()
        self.robots: dict[str, dict] = {}
        self.output_dir = str(output_dir)
        self.socketio = None
        self.commands: list[str] = []
        self.hp_updates: list[tuple[float, int, int]] = []

    def _get_status(self) -> dict:
        return {
            "coordinator_state": "IDLE",
            "current_round": 0,
            "total_aggregations": 0,
            "active_robots": 0,
            "mean_divergence": 0.0,
            "avg_loss": None,
            "avg_accuracy": None,
            "best_accuracy": None,
            "training_time": 0.0,
            "robots": {},
            "events": [],
            "loss_history": [],
            "acc_history": [],
        }

    def _send_command(self, command: str) -> None:
        self.commands.append(command)

    def _call_trigger_aggregation(self) -> dict:
        return {"success": True, "message": "Aggregation triggered (test)"}

    def _call_update_hyperparameters(self, lr: float, bs: int, ep: int) -> dict:
        self.hp_updates.append((lr, bs, ep))
        return {"success": True, "message": "updated"}


def _client(tmp_path: Path):
    node = _FakeDashboardNode(tmp_path)
    app, _socketio = build_dashboard_app(node)
    return node, app.test_client()


def test_dashboard_security_headers_and_csp(tmp_path: Path):
    _node, client = _client(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
    csp = resp.headers["Content-Security-Policy"]
    assert "'unsafe-inline'" not in csp
    assert "object-src 'none'" in csp
    assert "https://cdn.jsdelivr.net" not in csp
    assert "https://cdn.socket.io" not in csp
    assert "script-src 'self'" in csp
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert client.get_cookie("fl_robots_dashboard_csrf_token") is not None


def test_dashboard_command_requires_csrf(tmp_path: Path):
    node, client = _client(tmp_path)
    resp = client.post("/api/command", json={"command": "start_training"})
    assert resp.status_code == 403
    assert node.commands == []


def test_dashboard_command_accepts_valid_csrf(tmp_path: Path, csrf_headers):
    node, client = _client(tmp_path)
    headers = csrf_headers(client, cookie_name="fl_robots_dashboard_csrf_token")
    resp = client.post("/api/command", json={"command": "start_training"}, headers=headers)
    assert resp.status_code == 200
    assert node.commands == ["start_training"]


def test_dashboard_command_requires_bearer_when_token_set(tmp_path: Path, monkeypatch, csrf_headers):
    monkeypatch.setenv("FL_ROBOTS_API_TOKEN", "dash-secret")
    node, client = _client(tmp_path)
    headers = csrf_headers(client, cookie_name="fl_robots_dashboard_csrf_token")

    unauthorized = client.post("/api/command", json={"command": "start_training"}, headers=headers)
    assert unauthorized.status_code == 401
    assert node.commands == []

    authorized = client.post(
        "/api/command",
        json={"command": "start_training"},
        headers={**headers, "Authorization": "Bearer dash-secret"},
    )
    assert authorized.status_code == 200
    assert node.commands == ["start_training"]


def test_dashboard_template_has_no_inline_handlers(tmp_path: Path):
    _node, client = _client(tmp_path)
    html = client.get("/").data.decode("utf-8")
    assert "onclick=" not in html
    assert "onerror=" not in html
    assert "unsafe-inline" not in client.get("/").headers["Content-Security-Policy"]

