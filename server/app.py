"""
Root-level server/app.py — re-exports the FastAPI app from drift_env
and provides a main() entry point for OpenEnv validator / uvicorn.
"""

import uvicorn
from drift_env.server.app import app  # noqa: F401


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
