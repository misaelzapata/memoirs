"""Local memory ingestion engine."""

__all__ = ["__version__"]

__version__ = "0.1.0"

# Bootstrap structured logging on import (idempotent; safe for re-imports).
from .observability import setup_logging as _setup_logging  # noqa: E402
_setup_logging()
