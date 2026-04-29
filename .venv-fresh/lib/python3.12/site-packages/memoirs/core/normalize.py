"""Single source of truth for converting heterogeneous content payloads to text.

Replaces three duplicated implementations that lived in claude_code.py, cursor.py
and importers.py. Anthropic-style content arrays (text / thinking / tool_use /
tool_result) and free-form dicts/lists all collapse to readable plain text here.

Also hosts `should_skip_extraction` (Fix #1 of GAP) — the pre-extract noise
filter that drops code snippets, file paths, tool output, stack traces, etc.
before Gemma ever sees them, and `canonicalize_for_dedup` used by
`memory_engine.detect_exact_duplicate` to compare across whitespace/case/URL
variations.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..config import TOOL_USE_PREVIEW_LEN
from .redact import apply_redaction_if_enabled


def flatten_content(content: Any) -> str:
    """Collapse any content payload (str | list of blocks | dict | other) to text.

    Recognised block types from Anthropic SDK:
      - text          → text
      - thinking      → "[thinking]\\n..."
      - tool_use      → "[tool_use:NAME] {input json, truncated}"
      - tool_result   → "[tool_result]\\n<recursively flattened>"
    Anything else is rendered as "[type] {json preview}" so nothing is silently lost.
    """
    return apply_redaction_if_enabled(_flatten_raw(content))


def _flatten_raw(content: Any) -> str:
    """Internal flatten without redaction (kept simple so the redact hook only runs once)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            t = block.get("type")
            if t == "text":
                parts.append(block.get("text", ""))
            elif t == "thinking":
                parts.append(f"[thinking]\n{block.get('thinking', '')}")
            elif t == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {})
                preview = json.dumps(inp, ensure_ascii=False)[:TOOL_USE_PREVIEW_LEN]
                parts.append(f"[tool_use:{name}] {preview}")
            elif t == "tool_result":
                parts.append(f"[tool_result]\n{_flatten_raw(block.get('content'))}")
            else:
                preview = json.dumps(block, ensure_ascii=False)[:300]
                parts.append(f"[{t}] {preview}")
        return "\n\n".join(p for p in parts if p)
    if isinstance(content, dict):
        # Common cases first
        if "text" in content and isinstance(content["text"], str):
            return content["text"]
        if "content" in content:
            return _flatten_raw(content["content"])
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def normalize_role(value: Any) -> str:
    """Map heterogeneous role labels to {user, assistant, system, tool, document, event}."""
    role = str(value or "unknown").strip().lower()
    return {"human": "user", "ai": "assistant", "bot": "assistant"}.get(role, role)


def normalize_timestamp(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        from datetime import datetime, timezone

        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")
    return str(value)


# ----------------------------------------------------------------------
# Pre-extract noise filter (Fix #1 of GAP audit Fase 5C)
# ----------------------------------------------------------------------
#
# The audit found that 4058 active memorias contained 221 high-similarity
# pairs that slipped past the heuristic curator: code snippets, file paths,
# tool output, stack traces. These are not "memories" — they are transient
# environment state. We want them rejected BEFORE Gemma is invoked so we
# don't burn tokens generating low-value candidates.
#
# `should_skip_extraction(content)` is the canonical filter. It returns
# `(True, reason)` for any content that should not be extracted, and
# `(False, "")` for anything plausibly memorable.

# 1) Code snippet detector — multi-line content with at least one canonical
#    keyword indicates a code block, not a sentence about the user.
_CODE_KEYWORD_RE = re.compile(
    r"\b(def|func|class|import|from|const|var|let|fn|pub|return|if|else|elif|for|while|switch|case)\b"
    r"|=>|::"
)
_CODE_DECL_RE = re.compile(
    r"\b(def|func|class|import |from |const |var |let |fn |pub )\b"
)

# 2) File-path / URL prefix detector. A content that *starts* with one of
#    these prefixes is almost always a path or URL, not a durable fact.
_PATH_PREFIX_RE = re.compile(r"^(/tmp/|/home/|/var/|/etc/|/usr/|/opt/|/root/|http[s]?://)")
_PATH_TOKEN_RE = re.compile(r"(?:^|\s)(/[\w./\-_~]+|[a-zA-Z]:[\\/][\w./\-_~\\]+|http[s]?://\S+)")

# 3) Tool output / agent transcript markers.
_TOOL_OUTPUT_PREFIXES = (
    "[tool_use:",
    "[tool_result",
    "[thinking]",
    "<bash-",
    "<local-command-",
    "<command-",
    "<ide_",
    "<system-reminder>",
)
_TRACEBACK_PREFIXES = (
    "Traceback (most recent",
    "Error:",
    "ERROR:",
    "Exception:",
    "panic:",
)

# 4) Stack-trace anywhere in the body (multi-line) — Python "File ..., line N"
#    or generic "at file:line:col" patterns from JS/Go.
_STACK_TRACE_RE = re.compile(
    r'File "[^"]+", line \d+'
    r"|line \d+, in "
    r"|at \S+:\d+:\d+"
)

# 5) Hex-dump / random byte sequences.
_HEXDUMP_RE = re.compile(r"\b[0-9a-fA-F]{40,}\b")

# Hard length bounds. Min was 20 but rejected legit short prefs like
# "user prefers Python" (19) or "I am Misael" (11). Dropped to 8 — still
# filters single-token noise ("hi", "yes", "thanks") while keeping any
# 2+ word durable signal.
_MIN_CONTENT_LEN = 8
_MAX_CONTENT_LEN = 2000


def should_skip_extraction(content: str) -> tuple[bool, str]:
    """Return ``(True, reason)`` when ``content`` should not be turned into a memory.

    Reasons (stable strings — used in log lines and tests):
      - ``"too short"``               : len < 8
      - ``"too long; needs summarization first"`` : len > 2000
      - ``"code snippet"``            : multi-line block with code keywords
      - ``"path"``                    : content is dominated by file paths or URLs
      - ``"tool output"``             : starts with [tool_use / [tool_result / Traceback / Error
      - ``"stack trace"``             : Python/Go/JS stack trace markers anywhere
      - ``"hex dump"``                : 40+ hex chars w/o whitespace

    Always returns ``(False, "")`` for plain durable text such as
    ``"user prefers dark mode in IDE"``.
    """
    if content is None:
        return True, "empty"
    s = content if isinstance(content, str) else str(content)

    # 6) Length bounds first — cheapest checks.
    stripped = s.strip()
    if len(stripped) < _MIN_CONTENT_LEN:
        return True, "too short"
    if len(s) > _MAX_CONTENT_LEN:
        return True, "too long; needs summarization first"

    # 3) Tool output / transcript prefixes.
    head = stripped.lstrip()
    for pref in _TOOL_OUTPUT_PREFIXES:
        if head.startswith(pref):
            return True, "tool output"
    for pref in _TRACEBACK_PREFIXES:
        if head.startswith(pref):
            return True, "tool output"

    # 4) Stack trace markers anywhere in the body.
    if _STACK_TRACE_RE.search(s):
        return True, "stack trace"

    # 2) File path / URL as the dominant content.
    if _PATH_PREFIX_RE.match(stripped):
        return True, "path"
    # If more than half the characters live inside path/URL tokens, treat as path.
    path_chars = sum(len(m.group(1)) for m in _PATH_TOKEN_RE.finditer(s))
    if path_chars > 0 and path_chars > len(s) * 0.5:
        return True, "path"

    # 5) Hex dump / random bytes.
    if _HEXDUMP_RE.search(s):
        return True, "hex dump"

    # 1) Code snippet — multi-line block AND a code declaration keyword.
    line_count = s.count("\n") + 1
    if line_count >= 2 and _CODE_DECL_RE.search(s):
        return True, "code snippet"
    # Single-line but obviously code:
    #   - either a Python/JS/Rust declaration combined with `=` / `:=`
    #     (`def …`, `const x = …`, `let foo: Bar = baz` …)
    #   - or any line containing Go's `:=` short-declaration AND a string
    #     literal / function call (catches `serverURL := fs.String(...)`).
    if line_count == 1:
        head_norm = stripped
        if _CODE_DECL_RE.search(s) and (":=" in s or " = " in s) \
                and _CODE_KEYWORD_RE.search(s):
            return True, "code snippet"
        # Go-style short var decl: `name := expr(...)` or `name := "..."`.
        if ":=" in head_norm and (
            "(" in head_norm or '"' in head_norm or "'" in head_norm
        ) and not head_norm.lower().startswith(
            ("user ", "the ", "we ", "i ", "they ", "memoirs ", "decision",
             "preference", "goal ", "task ")
        ):
            return True, "code snippet"

    return False, ""


# ----------------------------------------------------------------------
# Cross-type / whitespace-insensitive dedup canonicalization (Fix #1.C)
# ----------------------------------------------------------------------
#
# `detect_exact_duplicate` originally hashed the raw content. Two candidates
# that differ only in whitespace, case, trailing punctuation, or an
# embedded URL would slip past the dupe check and create near-clones. The
# canonical form below is what the dedup layer compares against.

_URL_SCAN_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"\s+")
_PUNCT_STRIP = " \t\r\n.,;:!?\"'`()[]{}<>"


def canonicalize_for_dedup(content: str) -> str:
    """Lowercase, collapse whitespace, strip leading/trailing punctuation, drop URLs.

    Two strings whose canonical forms compare equal are considered identical
    by :func:`memory_engine.detect_exact_duplicate`.
    """
    if not content:
        return ""
    s = content if isinstance(content, str) else str(content)
    # Drop URLs entirely — they're rarely the load-bearing identity of a memory.
    s = _URL_SCAN_RE.sub("", s)
    # Lowercase + collapse whitespace.
    s = _WS_RE.sub(" ", s).lower().strip()
    # Strip leading/trailing punctuation noise.
    s = s.strip(_PUNCT_STRIP)
    return s
