"""Adapters for the head-to-head other engine bench.

Each adapter wraps one other engine memory engine in the
``EngineAdapter`` contract from :mod:`scripts.adapters.base`. New
adapters land here as a single file plus a re-export below.
"""
from __future__ import annotations

from scripts.adapters.base import AdapterStatus, EngineAdapter

# Re-exports are lazy: pulling in every adapter at import time would
# force every other engine library (mem0, cognee, langmem, llama-index, …) to
# load even when the bench only runs one engine. Tests and the CLI
# import the concrete classes from their own module, so this file
# stays light.

__all__ = [
    "AdapterStatus",
    "EngineAdapter",
]
