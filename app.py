"""Vercel WSGI entrypoint.

Vercel serverless does not support WebSockets/Socket.IO, so we expose
only the standard Flask HTTP routes here (pages, sessions API, leaderboard).
"""
import sys
import os
from pathlib import Path

# Make sure repo root is importable (needed for `from src.*` and `from webapp.*`)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Patch out eventlet/flask_socketio so the import doesn't crash in a
# serverless environment that has no async loop.
import unittest.mock as _mock
sys.modules.setdefault('eventlet', _mock.MagicMock())
sys.modules.setdefault('eventlet.green', _mock.MagicMock())
sys.modules.setdefault('flask_socketio', _mock.MagicMock())

from webapp.app import app  # noqa: E402
