"""
Backend entrypoint.

Run:
  python backend/app.py

This reuses the Flask API defined in `backend/flask_server.py`.
"""

from __future__ import annotations

from flask_server import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)