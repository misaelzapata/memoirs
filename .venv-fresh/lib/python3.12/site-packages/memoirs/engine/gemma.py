"""Backward-compat shim — re-exports the curator module.

The curator LLM abstraction lived here under the name ``gemma`` while the
default model was Gemma 2 2B. Auto-detect now picks Qwen3-4B-Instruct-2507
(or Phi-3.5, or Gemma) based on which GGUFs are installed, so the module
moved to ``memoirs.engine.curator`` and the public functions were renamed
``gemma_*`` -> ``curator_*``.

Existing imports of ``from memoirs.engine.gemma import gemma_extract`` (or
``_have_gemma``, ``gemma_consolidate``, …) keep working because we re-bind
the new names to their legacy aliases here. New code should import from
``memoirs.engine.curator`` directly.
"""
from __future__ import annotations

# Re-export everything (constants, dataclasses, helpers, validators, …) so
# ``from memoirs.engine.gemma import Candidate`` still resolves.
from .curator import *  # noqa: F401,F403

# Map the renamed identifiers back to their legacy gemma_* names. We import
# explicitly so the shim still exposes the legacy names even when the new
# module's ``__all__`` doesn't list them.
from .curator import (  # noqa: F401
    _have_curator as _have_gemma,
    curator_consolidate as gemma_consolidate,
    curator_detect_contradiction as gemma_detect_contradiction,
    curator_extract as gemma_extract,
    curator_extract_entities as gemma_extract_entities,
    curator_extract_relationships as gemma_extract_relationships,
    curator_resolve_conflict as gemma_resolve_conflict,
    curator_summarize as gemma_summarize,
    curator_summarize_project as gemma_summarize_project,
)

# Re-export private helpers that callers (and bench scripts) historically
# imported by name. ``from .curator import *`` skips underscore-prefixed
# names, so we list them here.
from .curator import (  # noqa: F401
    _build_curator_chat_prompt,
    _candidate_dedup_key,
    _candidates_from_text,
    _chat_stops,
    _chat_user_turn,
    _chunk_user_turns,
    _content_token_budget,
    _count_tokens,
    _get_llm,
    _is_user_meaningful,
    _model_ctx,
    _strip_fences,
    _validate_summary,
    _wrap_prompt,
    _wrapper_overhead_tokens,
)

# Older code referenced ``_build_gemma_chat_prompt``; keep the legacy name.
_build_gemma_chat_prompt = _build_curator_chat_prompt
