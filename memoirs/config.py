"""Central configuration constants.

Anything that was a magic number or hardcoded threshold lives here so
behavior changes go through a single edit and tests can patch them.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = Path(os.environ.get("MEMOIRS_DB", ".memoirs/memoirs.sqlite"))

_MODELS_DIR = Path.home() / ".local/share/memoirs/models"

_DEFAULT_QWEN3_PATH = _MODELS_DIR / "qwen3-4b-instruct-q4_k_m.gguf"
# Legacy alias kept for callers that still import GEMMA_MODEL_PATH (e.g.
# setup_helpers, cli download command). Resolves to whichever model the
# auto-detect picks below — Qwen3-4B by default — so the "gemma" name no
# longer implies the actual model in use.
GEMMA_MODEL_PATH = Path(
    os.environ.get(
        "MEMOIRS_GEMMA_MODEL",
        _DEFAULT_QWEN3_PATH if _DEFAULT_QWEN3_PATH.exists()
        else _MODELS_DIR / "gemma-2-2b-it-Q4_K_M.gguf",
    )
)

# ---------------------------------------------------------------------------
# Curator backend selection
# ---------------------------------------------------------------------------
# Validated by scripts/bench_models_known_cases.py over 15 hand-graded cases:
#   - qwen3-4b-instruct-2507: 7/10 contradiction + 4/5 consolidation (winner)
#   - qwen3.5-4b:             7/10 + 2/5  (newer but not better, 60% slower)
#   - qwen2.5-3b:             6/10 + 2/5  (previous default)
#   - gemma-2-2b / phi-3.5:   6/10 + 2/5  (legacy fallbacks)
# Auto-detect picks the first GGUF found in this priority order. Override with
# MEMOIRS_CURATOR_BACKEND ∈ {qwen3, qwen3.5, qwen, phi, gemma}.

_CURATOR_CANDIDATES = {
    "qwen3":   _MODELS_DIR / "qwen3-4b-instruct-q4_k_m.gguf",
    "qwen3.5": _MODELS_DIR / "qwen3.5-4b-q4_k_m.gguf",
    "qwen":    _MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf",
    "phi":     _MODELS_DIR / "Phi-3.5-mini-instruct-Q4_K_M.gguf",
    "gemma":   GEMMA_MODEL_PATH,
}


def _resolve_curator_backend() -> str:
    explicit = os.environ.get("MEMOIRS_CURATOR_BACKEND", "").strip().lower()
    if explicit in _CURATOR_CANDIDATES:
        return explicit
    # auto: bench-validated priority qwen3 > qwen3.5 > qwen2.5 > phi > gemma
    for name in ("qwen3", "qwen3.5", "qwen", "phi", "gemma"):
        if _CURATOR_CANDIDATES[name].exists():
            return name
    return "gemma"  # last resort, even if file missing — error surfaces at load


CURATOR_BACKEND = _resolve_curator_backend()
CURATOR_MODEL_PATH = Path(
    os.environ.get("MEMOIRS_CURATOR_MODEL", str(_CURATOR_CANDIDATES[CURATOR_BACKEND]))
)

# ---------------------------------------------------------------------------
# Engine thresholds
# ---------------------------------------------------------------------------
SEMANTIC_DUPLICATE_THRESHOLD = 0.92    # cosine similarity for "this is the same memory"
LOW_VALUE_SCORE_THRESHOLD = 0.15        # below this, archive after MIN_AGE_DAYS
LOW_VALUE_MIN_AGE_DAYS = 30             # only archive low-value if older than this

# Scoring weights — must sum to 1.0
SCORE_WEIGHTS = {
    "importance": 0.35,
    "confidence": 0.20,
    "recency":    0.15,
    "usage":      0.15,
    "user_signal":0.15,
}
RECENCY_HALF_LIFE_DAYS = 30

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
WATCH_POLL_INTERVAL = 2.0
WATCH_DEBOUNCE_SECONDS = 1.5
TOOL_USE_PREVIEW_LEN = 500              # truncation for [tool_use] blocks in flatten

# ---------------------------------------------------------------------------
# Gemma
# ---------------------------------------------------------------------------
GEMMA_MAX_CONTEXT_CHARS = 6000          # ~1500 tokens of conversation, leaves headroom for system prompt
GEMMA_DEFAULT_CTX_TOKENS = 4096         # enough for system + truncated convo + 512 output
GEMMA_DEFAULT_BATCH = 512               # logical batch size for prompt prefill
GEMMA_MAX_OUTPUT_TOKENS = 512           # JSON array is rarely >300 tokens; 512 is plenty
# Use ~80% of available cores by default (leaves room for OS + watcher).
# Override with MEMOIRS_GEMMA_THREADS=N.
GEMMA_DEFAULT_THREADS = max(4, (os.cpu_count() or 4) - 4)

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------
MCP_PROTOCOL_VERSION = "2025-06-18"
MCP_LOG_ARG_PREVIEW_LEN = 200
