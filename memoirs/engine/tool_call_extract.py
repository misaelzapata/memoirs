"""Automatic ``tool_call`` memory extraction from conversation transcripts.

P1-8 wired the storage layer (``record_tool_call``, the ``tool_call`` memory
type and its dedicated columns) but nothing converts Anthropic-style
``tool_use`` / ``tool_result`` content blocks into those memorias. This
module closes that gap so the ``extract`` flow (Layer 2) also yields a
durable record of every tool invocation it sees in the raw corpus.

Public API
----------
* :func:`extract_tool_calls_from_message` — pure parser. Takes one stored
  message dict (``raw_json`` round-tripped) and returns zero or more
  :class:`ToolCallEvent` records describing each tool invocation it
  contains. Tool results that arrive in *follow-up* messages are matched
  in :func:`record_tool_calls_for_conversation`, not here, because a
  single message rarely carries both halves.
* :func:`record_tool_calls_for_conversation` — orchestrator. Walks every
  active message of one conversation, pairs ``tool_use`` blocks with
  their matching ``tool_result`` blocks via ``tool_use_id``, and writes
  one ``type='tool_call'`` memory per pair through
  :func:`memoirs.engine.memory_engine.record_tool_call`. Idempotent: the
  underlying ``content_hash`` collision plus a per-conversation
  ``tool_result_hash`` lookup mean a re-run does not duplicate rows.

Design notes
------------
* We deliberately read ``messages.raw_json`` rather than re-flatten
  ``messages.content`` — the flattened text already lossily renders
  ``[tool_use:NAME] {...}`` and would force us to re-parse it. The
  raw blob is the source of truth for tool blocks.
* Result summaries cap at 200 chars; bodies longer than 500 chars get a
  ``... [truncated, sha256=<16hex>]`` suffix so the original payload
  stays addressable from the memory text without bloating the row.
* When a tool_use has no matching tool_result in the conversation
  (cancelled, in-flight, truncated transcript) we still emit an event
  with ``status='cancelled'`` and an empty result so the corpus reflects
  the attempt.

This module never imports ``gemma`` or ``extract_spacy`` so it stays
cheap (no model load) and safe to call from sync paths.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from ..db import MemoirsDB
from . import memory_engine as _me

log = logging.getLogger(__name__)


# Status values mirror ``memory_engine._TOOL_CALL_STATUSES`` — keeping a
# local copy avoids importing a private name and lets the extractor stay
# decoupled from upstream churn.
_STATUSES = {"success", "error", "cancelled"}

# Truncation thresholds for the human-readable ``result_summary`` we
# carry in the memory's content text.
SUMMARY_MAX = 200
TRUNCATE_THRESHOLD = 500


@dataclass(frozen=True)
class ToolCallEvent:
    """One parsed tool invocation extracted from a transcript."""

    tool_name: str
    args: dict
    result_summary: str
    status: str
    tool_use_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "args": self.args,
            "result_summary": self.result_summary,
            "status": self.status,
            "tool_use_id": self.tool_use_id,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_result_content(content: Any) -> str:
    """Render a ``tool_result.content`` payload as plain text.

    Anthropic permits the content to be a bare string, a list of blocks
    (each with ``type='text'`` / ``type='image'`` / etc), or a free-form
    dict. We collapse to text so the summary heuristic always gets a
    string; non-text blocks are rendered as JSON previews so nothing
    silently disappears.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("type")
                if t == "text" and isinstance(block.get("text"), str):
                    out.append(block["text"])
                elif "content" in block and isinstance(block["content"], str):
                    out.append(block["content"])
                else:
                    try:
                        out.append(json.dumps(block, ensure_ascii=False))
                    except (TypeError, ValueError):
                        out.append(str(block))
            else:
                out.append(str(block))
        return "\n".join(s for s in out if s)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        try:
            return json.dumps(content, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(content)
    return str(content)


def _summarize_result(text: str) -> str:
    """Cap a result at ``SUMMARY_MAX`` chars; long ones get a sha256 suffix.

    Matches the heuristic from the GAP brief:
      * ``len(text) <= SUMMARY_MAX``                   → text as-is
      * ``SUMMARY_MAX < len(text) <= TRUNCATE_THRESHOLD`` → first SUMMARY_MAX chars
      * ``len(text) > TRUNCATE_THRESHOLD``             → first SUMMARY_MAX chars +
        ``... [truncated, sha256=<16hex>]``
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= SUMMARY_MAX:
        return text
    head = text[:SUMMARY_MAX]
    if len(text) <= TRUNCATE_THRESHOLD:
        return head
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{head}... [truncated, sha256={digest}]"


def _resolve_message_obj(message: dict) -> dict:
    """Return the inner Anthropic ``message`` dict regardless of wrapping.

    Stored messages can come from two shapes:
      1. The full Claude Code JSONL line — ``{type, message: {role, content},
         ...}``. This is what ``messages.raw_json`` carries.
      2. A pre-unwrapped Anthropic message — ``{role, content: [...]}``.
         Mostly useful for tests and ad-hoc callers.
    """
    if not isinstance(message, dict):
        return {}
    inner = message.get("message")
    if isinstance(inner, dict):
        return inner
    return message


# ---------------------------------------------------------------------------
# Parser — single message
# ---------------------------------------------------------------------------


def extract_tool_calls_from_message(message: dict) -> list[ToolCallEvent]:
    """Parse one stored message into zero or more :class:`ToolCallEvent` records.

    Handles three cases that may appear in a single message:

    * ``tool_use`` block with no co-located result (assistant message that
      *issued* the call). Emitted with ``status='cancelled'`` and an empty
      summary; the orchestrator may upgrade it later when the matching
      ``tool_result`` is found.
    * ``tool_result`` block alone (user/tool message that *returned* the
      call). Skipped here — emitted by the orchestrator paired with the
      original ``tool_use``.
    * Both blocks in the same message (rare, but legal). Paired via
      ``tool_use_id`` and emitted with the resolved status.
    """
    msg = _resolve_message_obj(message)
    content = msg.get("content")
    if not isinstance(content, list):
        return []

    uses: list[dict] = []
    results_by_id: dict[str, dict] = {}
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "tool_use":
            uses.append(block)
        elif t == "tool_result":
            tid = block.get("tool_use_id")
            if isinstance(tid, str) and tid:
                results_by_id[tid] = block

    events: list[ToolCallEvent] = []
    for use in uses:
        name = use.get("name") or "?"
        args = use.get("input") if isinstance(use.get("input"), dict) else {}
        tid = use.get("id") if isinstance(use.get("id"), str) else None

        result = results_by_id.get(tid) if tid else None
        if result is not None:
            summary = _summarize_result(_flatten_result_content(result.get("content")))
            status = "error" if bool(result.get("is_error")) else "success"
        else:
            summary = ""
            status = "cancelled"

        events.append(
            ToolCallEvent(
                tool_name=str(name),
                args=dict(args),
                result_summary=summary,
                status=status,
                tool_use_id=tid,
            )
        )

    return events


# ---------------------------------------------------------------------------
# Orchestrator — full conversation
# ---------------------------------------------------------------------------


def _iter_conversation_messages(db: MemoirsDB, conversation_id: str) -> list[dict]:
    """Return every active message of the conversation, ordered, with raw + meta.

    Each entry carries the parsed ``raw_json`` plus the row-level fields the
    extractor needs to enrich tool_call memorias with context (``ordinal``,
    ``ts``). The raw payload is unwrapped to a plain dict; missing fields
    default to ``None`` rather than raising.
    """
    rows = db.conn.execute(
        """
        SELECT id, ordinal, role, raw_json,
               COALESCE(created_at, first_seen_at, updated_at) AS ts
        FROM messages
        WHERE conversation_id = ? AND is_active = 1
        ORDER BY ordinal ASC
        """,
        (conversation_id,),
    ).fetchall()
    out: list[dict] = []
    for r in rows:
        try:
            raw = json.loads(r["raw_json"]) if r["raw_json"] else {}
        except (TypeError, ValueError):
            raw = {}
        if isinstance(raw, dict):
            raw = dict(raw)
            raw.setdefault("_ordinal", r["ordinal"])
            raw.setdefault("_ts", r["ts"])
        out.append(raw)
    return out


def _derive_project_name(cwd: str | None) -> str | None:
    """Pick the most useful path component to label a project."""
    if not isinstance(cwd, str) or not cwd:
        return None
    parts = [p for p in cwd.split("/") if p]
    if not parts:
        return None
    for marker in ("projects", "Desktop", "src", "code", "workspace"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return parts[-1]


def _conversation_context(db: MemoirsDB, conversation_id: str) -> dict:
    """Read ``cwd`` / project label from ``conversations.metadata_json``."""
    row = db.conn.execute(
        "SELECT metadata_json FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    md: dict = {}
    if row and row["metadata_json"]:
        try:
            md = json.loads(row["metadata_json"])
        except (TypeError, ValueError):
            md = {}
    return md if isinstance(md, dict) else {}


def _existing_result_hashes(db: MemoirsDB, conversation_id: str) -> set[str]:
    """Tool-result hashes already recorded for ``conversation_id``.

    Used to short-circuit re-emit on idempotent re-runs without depending
    on the global ``content_hash`` (different ``record_tool_call`` calls
    can collapse two distinct invocations whose serialized form is equal,
    but per-conversation ``tool_result_hash`` is the strongest dedup
    signal we have for the extractor).
    """
    rows = db.conn.execute(
        """
        SELECT tool_result_hash
        FROM memories
        WHERE type = 'tool_call'
          AND archived_at IS NULL
          AND tool_result_hash IS NOT NULL
          AND json_extract(metadata_json, '$.conversation_id') = ?
        """,
        (conversation_id,),
    ).fetchall()
    return {r["tool_result_hash"] for r in rows if r["tool_result_hash"]}


def _augment_tool_call_metadata(
    db: MemoirsDB,
    memory_id: str,
    *,
    cwd: str | None,
    project_name: str | None,
    conversation_id: str | None,
    message_ordinal: int | None,
    timestamp: str | None,
    tool_use_id: str | None,
) -> None:
    """Patch ``cwd``/``project_name``/``message_ordinal``/``timestamp`` into
    a tool_call's ``metadata_json`` after ``record_tool_call`` has run.

    We extend the existing JSON object instead of overwriting it so the
    ``conversation_id`` already written by ``record_tool_call`` survives.
    Existing keys are preserved (idempotent re-runs don't clobber a
    backfilled value).
    """
    if not memory_id:
        return
    extras = {
        "cwd": cwd,
        "project_name": project_name,
        "conversation_id": conversation_id,
        "message_ordinal": message_ordinal,
        "timestamp": timestamp,
        "tool_use_id": tool_use_id,
    }
    extras = {k: v for k, v in extras.items() if v is not None and v != ""}
    if not extras:
        return
    row = db.conn.execute(
        "SELECT metadata_json FROM memories WHERE id = ?",
        (memory_id,),
    ).fetchone()
    if row is None:
        return
    try:
        md = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    except (TypeError, ValueError):
        md = {}
    if not isinstance(md, dict):
        md = {}
    changed = False
    for k, v in extras.items():
        if md.get(k) in (None, ""):
            md[k] = v
            changed = True
    if not changed:
        return
    try:
        new_json = json.dumps(md, ensure_ascii=False)
    except (TypeError, ValueError):
        return
    with db.conn:
        db.conn.execute(
            "UPDATE memories SET metadata_json = ? WHERE id = ?",
            (new_json, memory_id),
        )


def _is_extraction_enabled() -> bool:
    """Honor the ``MEMOIRS_EXTRACT_TOOL_CALLS`` env switch (default ``on``)."""
    raw = (os.environ.get("MEMOIRS_EXTRACT_TOOL_CALLS") or "on").strip().lower()
    return raw not in {"off", "false", "0", "no", "disable", "disabled"}


def record_tool_calls_for_conversation(
    db: MemoirsDB,
    conversation_id: str,
) -> int:
    """Walk one conversation and persist a ``tool_call`` memory per invocation.

    Returns the number of memorias **inserted by this call** (re-runs
    typically return 0). Honors ``MEMOIRS_EXTRACT_TOOL_CALLS=off`` by
    returning ``0`` without touching the DB.

    Implementation:

    1. Gathers every active message ordered by ``ordinal`` and unwraps
       the Anthropic ``message`` dict from each row's ``raw_json``.
    2. First pass collects ``tool_result`` blocks indexed by
       ``tool_use_id`` so the matcher works across messages — Claude
       Code consistently splits the call (assistant message) and the
       result (user message) onto separate JSONL lines.
    3. Second pass walks ``tool_use`` blocks in order and resolves each
       to (status, result_summary) via the index from step 2.
    4. Skips invocations whose ``tool_result_hash`` already exists for
       this ``conversation_id`` so the function is safe to re-run.
    """
    if not conversation_id:
        return 0
    if not _is_extraction_enabled():
        log.debug(
            "tool_call extraction disabled via MEMOIRS_EXTRACT_TOOL_CALLS for conv=%s",
            conversation_id,
        )
        return 0

    messages = _iter_conversation_messages(db, conversation_id)
    if not messages:
        return 0

    conv_md = _conversation_context(db, conversation_id)
    conv_cwd = conv_md.get("cwd") if isinstance(conv_md.get("cwd"), str) else None
    conv_project = conv_md.get("project_name") or conv_md.get("project_dir")
    if not isinstance(conv_project, str):
        conv_project = None

    # Pass 1: build a tool_use_id → result-block index across the whole
    # conversation. Last-writer-wins on duplicate ids (legal per the
    # protocol; should never happen in practice).
    results_by_id: dict[str, dict] = {}
    for raw in messages:
        msg = _resolve_message_obj(raw)
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
            tid = block.get("tool_use_id")
            if isinstance(tid, str) and tid:
                results_by_id[tid] = block

    seen_hashes = _existing_result_hashes(db, conversation_id)

    def _row_count() -> int:
        return db.conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE type='tool_call' AND archived_at IS NULL "
            "AND json_extract(metadata_json, '$.conversation_id') = ?",
            (conversation_id,),
        ).fetchone()["c"]

    before_total = _row_count()

    # Pass 2: walk tool_use blocks in transcript order and persist.
    for raw in messages:
        msg = _resolve_message_obj(raw)
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        msg_ordinal = raw.get("_ordinal") if isinstance(raw, dict) else None
        msg_ts = raw.get("_ts") if isinstance(raw, dict) else None
        msg_cwd = raw.get("cwd") if isinstance(raw, dict) and isinstance(raw.get("cwd"), str) else None
        if msg_ts is None and isinstance(raw, dict):
            msg_ts = raw.get("timestamp")
        msg_cwd = msg_cwd or conv_cwd
        project_name = conv_project or _derive_project_name(msg_cwd)

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue

            name = block.get("name") or "?"
            args = block.get("input") if isinstance(block.get("input"), dict) else {}
            tid = block.get("id") if isinstance(block.get("id"), str) else None

            result_block = results_by_id.get(tid) if tid else None
            if result_block is not None:
                rendered = _flatten_result_content(result_block.get("content"))
                summary = _summarize_result(rendered)
                status = "error" if bool(result_block.get("is_error")) else "success"
                # ``record_tool_call`` hashes whatever we hand it; pass
                # the *full* rendered text so the hash dedups against
                # an identical re-extraction even if SUMMARY_MAX changes.
                result_payload: Any = rendered
            else:
                summary = ""
                status = "cancelled"
                result_payload = None

            rh = _me._hash_tool_result(result_payload)
            if rh in seen_hashes:
                continue

            try:
                mid = _me.record_tool_call(
                    db,
                    tool_name=str(name),
                    args=dict(args),
                    result=result_payload if result_payload is not None else summary,
                    status=status,
                    conversation_id=conversation_id,
                )
            except ValueError:
                log.debug(
                    "skipped malformed tool_use in conv=%s name=%r",
                    conversation_id, name,
                )
                continue

            _augment_tool_call_metadata(
                db, mid,
                cwd=msg_cwd,
                project_name=project_name,
                conversation_id=conversation_id,
                message_ordinal=msg_ordinal,
                timestamp=msg_ts,
                tool_use_id=tid,
            )
            seen_hashes.add(rh)

    inserted = max(0, _row_count() - before_total)
    if inserted:
        log.info(
            "tool_call extract: conv=%s inserted=%d (env=%s)",
            conversation_id, inserted,
            os.environ.get("MEMOIRS_EXTRACT_TOOL_CALLS", "on"),
        )
    return inserted


__all__ = [
    "ToolCallEvent",
    "extract_tool_calls_from_message",
    "record_tool_calls_for_conversation",
    "SUMMARY_MAX",
    "TRUNCATE_THRESHOLD",
]
