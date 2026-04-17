"""WSGI entrypoint for production servers such as gunicorn."""

from __future__ import annotations

import atexit

from .simulation import SimulationEngine
from .standalone_web import create_app

_simulation = SimulationEngine()
app = create_app(_simulation)
atexit.register(_simulation.shutdown)

