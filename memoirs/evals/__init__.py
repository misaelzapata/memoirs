"""Memoirs eval harness (P0-3).

Lightweight, dependency-free benchmarking utilities so memoirs can publish
metrics comparable with Mem0 / Zep / Letta on standard datasets
(LongMemEval, LoCoMo) and on custom synthetic suites.

Public surface:

    EvalCase, EvalSuite, EvalResults, run_eval

The suites are plain dataclasses so they round-trip through JSON for
regression tracking. See ``memoirs/evals/suites/synthetic_basic.py`` for a
worked example and ``memoirs/evals/longmemeval_adapter.py`` for the public
LongMemEval adapter.
"""
from __future__ import annotations

from .harness import (
    EvalCase,
    EvalResults,
    EvalSuite,
    ModeResults,
    QueryResult,
    compute_metrics,
    run_eval,
)


__all__ = [
    "EvalCase",
    "EvalSuite",
    "EvalResults",
    "ModeResults",
    "QueryResult",
    "compute_metrics",
    "run_eval",
]
