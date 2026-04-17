"""WSGI entrypoint for production servers such as gunicorn.

Configuration is read from environment variables:

- ``FL_ROBOTS_NUM_ROBOTS`` — number of simulated robots (default 4)
- ``FL_ROBOTS_SEED`` — deterministic seed (default 42)

Example::

    gunicorn --bind 0.0.0.0:5000 --workers 2 fl_robots.wsgi:app
"""

from __future__ import annotations

import atexit
import os

from .simulation import SimulationEngine
from .standalone_web import create_app

_num_robots = int(os.environ.get("FL_ROBOTS_NUM_ROBOTS", "4"))
_seed = int(os.environ.get("FL_ROBOTS_SEED", "42"))

_simulation = SimulationEngine(num_robots=_num_robots, seed=_seed)
app = create_app(_simulation)
atexit.register(_simulation.shutdown)
