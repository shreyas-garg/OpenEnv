"""
Root-level server/app.py — re-exports the FastAPI app from email_env.
This file exists to satisfy the OpenEnv validator convention.
"""

from email_env.server.app import app  # noqa: F401
