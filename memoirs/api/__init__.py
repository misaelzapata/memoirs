"""REST API for memoirs (FastAPI). Optional `[api]` extra."""
from .server import app, run

__all__ = ["app", "run"]
