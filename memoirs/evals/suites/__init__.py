"""Bundled eval suites.

Each suite exposes a `build(db) -> EvalSuite` factory that seeds the given
DB with the memories the suite expects and returns the matching cases.
"""
from __future__ import annotations

# Re-exported for convenience: ``from memoirs.evals.suites import build_synthetic_basic``.
from .synthetic_basic import build as build_synthetic_basic  # noqa: F401


__all__ = ["build_synthetic_basic"]
