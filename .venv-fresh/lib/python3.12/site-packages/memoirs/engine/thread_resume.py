"""Auto-resume thread (P-resume).

When a conversation pauses for an extended period (lunch, meeting, end-of-day)
and the agent later returns, it shouldn't have to re-read every message to get
oriented. This module persists a durable, per-conversation summary the next
agent turn can pull via :func:`resume_thread` (or the MCP wrapper).

Pipeline
--------
1. ``detect_idle_conversations`` walks the messages table and yields the
   conversations whose last activity is older than ``idle_minutes`` AND
   whose latest ``thread_summaries`` row is missing or stale.
2. ``generate_thread_summary`` runs Qwen (via :func:`curator_summarize_project`)
   over a compact view of the conversation — user turns + last decisions /
   pending actions — validates the output via :func:`_validate_summary`,
   and persists one ``thread_summaries`` row.
3. ``resume_thread`` returns ``{ summary, salient_memories, last_decisions,
   pending_actions }`` for the agent's system prompt.

Heuristic fallback
------------------
If the curator is unavailable or its retry path also fails validation we
fall back to a simple heuristic summary built from the first/last user
turns and the count of "decision" / "fix" mentions. Better something than
nothing — the agent will still know the rough shape of the conversation.

This module imports lazily from ``.curator`` so a minimal install (no Qwen
weights) can still load + run the heuristic path.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from ..db import MemoirsDB

log = logging.getLogger("memoirs.thread_resume")


# ---------------------------------------------------------------------------
# Public constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_IDLE_MINUTES = 30
DEFAULT_MAX_CONVS_PER_TICK = 10
SUMMARY_MAX_CHARS = 480
PENDING_ACTION_MAX = 8
SALIENT_ENTITY_MAX = 12


# ---------------------------------------------------------------------------
# Idle detection
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _ensure_thread_summaries_table(conn: sqlite3.Connection) -> None:
    """Belt-and-suspenders fallback when migration 011 hasn't run yet
    (e.g. a test opens a raw sqlite3 connection bypassing ``MemoirsDB.init``).
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS thread_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            generated_at TEXT NOT NULL DEFAULT (datetime('now')),
            message_count_at_summary INTEGER NOT NULL,
            last_message_ts TEXT,
            pending_actions_json TEXT,
            salient_entity_ids_json TEXT,
            user_id TEXT NOT NULL DEFAULT 'local',
            UNIQUE(conversation_id, generated_at)
        );
        CREATE INDEX IF NOT EXISTS idx_thread_summaries_conv
            ON thread_summaries(conversation_id);
        CREATE INDEX IF NOT EXISTS idx_thread_summaries_recent
            ON thread_summaries(last_message_ts DESC);
        """
    )


def detect_idle_conversations(
    db: MemoirsDB,
    *,
    idle_minutes: int = DEFAULT_IDLE_MINUTES,
    limit: int = 100,
    now: Optional[datetime] = None,
) -> list[dict]:
    """Conversations whose last message is older than ``idle_minutes`` and
    that don't yet have a fresh ``thread_summaries`` row.

    A summary is considered fresh if its ``last_message_ts`` is at least as
    new as the conversation's latest message timestamp — i.e. nothing has
    happened since we last summarized.

    Returns a list of dicts shaped::

        {"conversation_id": str, "last_message_ts": str,
         "message_count": int, "age_minutes": float}
    """
    _ensure_thread_summaries_table(db.conn)

    cutoff_dt = (now or _utc_now()) - timedelta(minutes=int(idle_minutes))
    cutoff_iso = cutoff_dt.isoformat()

    # We compute the "last message time" per conversation from the messages
    # table directly so the result is robust to backdated ``conversations.
    # updated_at`` rows. ``COALESCE(created_at, first_seen_at, updated_at)``
    # covers all the fields the various ingesters populate.
    rows = db.conn.execute(
        """
        SELECT m.conversation_id AS conversation_id,
               MAX(COALESCE(m.created_at, m.first_seen_at, m.updated_at))
                   AS last_message_ts,
               COUNT(*) AS message_count
        FROM messages m
        WHERE m.is_active = 1
        GROUP BY m.conversation_id
        HAVING last_message_ts IS NOT NULL
           AND last_message_ts <= ?
        ORDER BY last_message_ts DESC
        LIMIT ?
        """,
        (cutoff_iso, int(limit) * 4),
    ).fetchall()

    out: list[dict] = []
    now_dt = now or _utc_now()
    for r in rows:
        cid = r["conversation_id"]
        last_ts = r["last_message_ts"]
        # Fast check: do we already have a summary newer than the latest
        # message? If so, skip.
        latest = db.conn.execute(
            "SELECT last_message_ts FROM thread_summaries "
            "WHERE conversation_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (cid,),
        ).fetchone()
        if latest is not None and latest["last_message_ts"]:
            existing = _parse_iso(latest["last_message_ts"])
            current = _parse_iso(last_ts)
            if existing is not None and current is not None and existing >= current:
                continue
        last_dt = _parse_iso(last_ts)
        age_minutes = (
            (now_dt - last_dt).total_seconds() / 60.0
            if last_dt is not None else 0.0
        )
        out.append({
            "conversation_id": cid,
            "last_message_ts": last_ts,
            "message_count": int(r["message_count"]),
            "age_minutes": round(age_minutes, 2),
        })
        if len(out) >= int(limit):
            break
    return out


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

_DECISION_RE = re.compile(
    r"\b(decided|decision|chose|will use|going to|fix(?:ed)?|bug|implement(?:ed)?|"
    r"merged|approved|rejected|switched|landed|shipped|deployed|adopted)\b",
    re.IGNORECASE,
)
_PENDING_RE = re.compile(
    r"\b(todo|TODO|next:|next step|follow[- ]?up|pending|"
    r"awaiting|waiting for|need to|must|will|should)\b",
    re.IGNORECASE,
)

# Lines that are flattened tool blocks ("[tool_use:Read] {..}", "[tool_result:...]"
# and the JSON / table / markdown noise around them) carry no prose value for
# decisions / pending-action mining. We strip them so the heuristic only sees
# real human language.
_TOOL_NOISE_RE = re.compile(
    r"^\s*(?:\[tool_(?:use|result)\b|\[ ?[xX ]?\]|\{|\}|\||"
    r"```|---+|===+|<\?xml|</?[a-z][^>]*>)",
)


def _is_tool_noise(line: str) -> bool:
    """True if the line is a flattened tool-use block or rendered JSON/markup."""
    if not line:
        return True
    stripped = line.strip()
    if not stripped:
        return True
    if "[tool_use" in stripped or "[tool_result" in stripped:
        return True
    return bool(_TOOL_NOISE_RE.match(stripped))


def _heuristic_summary(messages: list[dict]) -> str:
    """Best-effort summary when the LLM is unavailable.

    Takes the first ~200 chars of the first user turn, the last ~200 chars
    of the most recent meaningful turn, and a coarse mention count.
    """
    if not messages:
        return ""
    user_turns = [m for m in messages if m.get("role") == "user" and m.get("content")]
    if not user_turns:
        user_turns = [m for m in messages if m.get("content")]
    if not user_turns:
        return ""
    first = (user_turns[0].get("content") or "").strip()
    last = (user_turns[-1].get("content") or "").strip()
    head = first[:200].rstrip()
    tail = last[:200].rstrip() if last and last is not first else ""
    decision_count = sum(1 for m in messages if _DECISION_RE.search(m.get("content") or ""))
    pending_count = sum(1 for m in messages if _PENDING_RE.search(m.get("content") or ""))
    pieces = [
        f"Conversation opened with: {head}".rstrip(".") + ".",
    ]
    if tail and tail != head:
        pieces.append(f"Latest user input: {tail}".rstrip(".") + ".")
    pieces.append(
        f"{len(messages)} messages, {decision_count} decision-like and "
        f"{pending_count} pending mentions."
    )
    out = " ".join(pieces)
    if len(out) > SUMMARY_MAX_CHARS:
        out = out[: SUMMARY_MAX_CHARS - 1].rstrip() + "…"
    return out


def _extract_pending_actions(messages: list[dict]) -> list[str]:
    """Pull short human-readable pending-action strings from message text.

    Skips lines that are flattened tool_use / tool_result blocks or rendered
    JSON / table noise — those are not pending actions, even if they happen
    to contain a "will" or "should" inside their payload.
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in messages:
        content = m.get("content") or ""
        if not content:
            continue
        for line in content.splitlines():
            line = line.strip(" -*•\t")
            if not line or _is_tool_noise(line):
                continue
            if not _PENDING_RE.search(line):
                continue
            short = line[:120]
            key = short.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(short)
            if len(out) >= PENDING_ACTION_MAX:
                return out
    return out


def _extract_last_decisions(messages: list[dict], *, max_n: int = 5) -> list[str]:
    """Latest few decision-like statements, in chronological order.

    Filters out flattened tool block lines so a serialized ``tool_use`` /
    ``tool_result`` payload that happens to contain a decision-like verb
    cannot pollute the result.
    """
    hits: list[str] = []
    for m in messages:
        content = m.get("content") or ""
        if not content:
            continue
        for line in content.splitlines():
            line = line.strip(" -*•\t")
            if not line or _is_tool_noise(line):
                continue
            if _DECISION_RE.search(line):
                hits.append(line[:200])
    return hits[-max_n:]


def _pending_tasks_from_corpus(db: MemoirsDB, conversation_id: str) -> list[str]:
    """Return ``type='task'`` memorias scoped to this conversation."""
    try:
        rows = db.conn.execute(
            """
            SELECT content
              FROM memories
             WHERE archived_at IS NULL
               AND type = 'task'
               AND json_extract(metadata_json, '$.conversation_id') = ?
             ORDER BY created_at DESC
             LIMIT ?
            """,
            (conversation_id, PENDING_ACTION_MAX),
        ).fetchall()
    except sqlite3.Error:
        return []
    out: list[str] = []
    for r in rows:
        c = (r["content"] or "").strip().splitlines()[0:1]
        if c:
            short = c[0][:160]
            if short and short not in out:
                out.append(short)
    return out


def _recent_tool_calls_from_corpus(
    db: MemoirsDB, conversation_id: str, *, limit: int = 8,
) -> list[dict]:
    """Recent ``type='tool_call'`` memorias scoped to this conversation.

    Newest-first; each row dict carries enough fields for the CLI renderer
    to format a clean table without re-parsing the flattened content.
    """
    try:
        rows = db.conn.execute(
            """
            SELECT id, tool_name, tool_status, tool_args_json, content,
                   created_at, metadata_json
              FROM memories
             WHERE archived_at IS NULL
               AND type = 'tool_call'
               AND json_extract(metadata_json, '$.conversation_id') = ?
             ORDER BY COALESCE(json_extract(metadata_json, '$.timestamp'),
                               created_at) DESC
             LIMIT ?
            """,
            (conversation_id, int(limit)),
        ).fetchall()
    except sqlite3.Error:
        return []
    return [dict(r) for r in rows]


def _collect_messages(db: MemoirsDB, conversation_id: str) -> list[dict]:
    rows = db.conn.execute(
        "SELECT id, role, content, ordinal, "
        "       COALESCE(created_at, first_seen_at, updated_at) AS ts "
        "  FROM messages "
        " WHERE conversation_id = ? AND is_active = 1 "
        " ORDER BY ordinal",
        (conversation_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _conversation_user_id(db: MemoirsDB, conversation_id: str) -> str:
    row = db.conn.execute(
        "SELECT user_id FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    if row is None:
        return "local"
    try:
        return str(row["user_id"]) or "local"
    except (KeyError, IndexError):
        return "local"


def _scoped_memories(db: MemoirsDB, conversation_id: str, *, limit: int = 10) -> list[dict]:
    """Top-N memorias scoped to this conversation via metadata.conversation_id.

    Falls back to scoring-only ordering when no metadata-scoped memorias
    exist (early in a conversation's life). Read-only — does not bump usage.
    """
    rows = db.conn.execute(
        """
        SELECT id, type, content, score, importance, confidence, created_at
          FROM memories
         WHERE archived_at IS NULL
           AND json_extract(metadata_json, '$.conversation_id') = ?
         ORDER BY score DESC
         LIMIT ?
        """,
        (conversation_id, int(limit)),
    ).fetchall()
    return [dict(r) for r in rows]


def _build_summary_input(messages: list[dict]) -> list[dict]:
    """Compress the conversation into a memory-shaped list for the LLM.

    The :func:`curator_summarize_project` prompt expects a list of memorias
    each with ``content`` + (optional) ``entities``. We synthesize that
    list from the conversation's user/assistant turns, prioritising user
    turns and de-duplicating on a coarse hash so a fanout assistant reply
    doesn't dominate the budget.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # Collapse runs of whitespace + truncate per-line for the prompt.
        snippet = " ".join(content.split())[:400]
        key = snippet[:80].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"content": snippet, "type": m.get("role") or "msg"})
        if len(out) >= 30:
            break
    return out


def generate_thread_summary(
    db: MemoirsDB,
    conversation_id: str,
    *,
    llm: Any | None = None,
    use_llm: bool = True,
) -> Optional[dict]:
    """Run the curator (or fall back to a heuristic), persist, return the row.

    Returns ``None`` if the conversation has no messages at all.
    """
    _ensure_thread_summaries_table(db.conn)

    messages = _collect_messages(db, conversation_id)
    if not messages:
        return None

    summary_text: Optional[str] = None
    if use_llm:
        try:
            from .curator import (
                _have_curator,
                _validate_summary,
                curator_summarize_project,
            )

            llm_obj = llm
            if llm_obj is None and not _have_curator():
                llm_obj = None
            if llm_obj is not None or _have_curator():
                synthetic_memories = _build_summary_input(messages)
                # Reuse the validated chat-template + retry plumbing —
                # ``curator_summarize_project`` already handles validate+retry+fallback.
                project_label = (
                    f"thread:{conversation_id[:24]}"
                )
                produced = curator_summarize_project(
                    project_label,
                    synthetic_memories,
                    llm=llm_obj,
                    max_chars=SUMMARY_MAX_CHARS,
                )
                if isinstance(produced, str) and produced.strip():
                    # _validate_summary already ran, but enforce length one
                    # more time defensively (the helper trims with an ellipsis).
                    summary_text = produced.strip()[:SUMMARY_MAX_CHARS]
        except Exception:  # noqa: BLE001
            log.exception(
                "generate_thread_summary: curator path failed for %s",
                conversation_id[:24],
            )

    if not summary_text:
        summary_text = _heuristic_summary(messages)

    if not summary_text:
        return None

    pending_actions = _extract_pending_actions(messages)

    # Salient entities — derived from memorias linked to the conversation.
    salient_entity_ids: list[str] = []
    try:
        rows = db.conn.execute(
            """
            SELECT DISTINCT e.id
              FROM entities e
              JOIN memory_entities me ON me.entity_id = e.id
              JOIN memories m ON m.id = me.memory_id
             WHERE m.archived_at IS NULL
               AND json_extract(m.metadata_json, '$.conversation_id') = ?
             LIMIT ?
            """,
            (conversation_id, SALIENT_ENTITY_MAX),
        ).fetchall()
        salient_entity_ids = [r["id"] for r in rows]
    except sqlite3.Error:
        # Schema in flight (memory_entities or entities may not exist on
        # raw sqlite3 connections that bypass MemoirsDB.init); soldier on.
        salient_entity_ids = []

    last_message_ts = messages[-1].get("ts") if messages else None
    user_id = _conversation_user_id(db, conversation_id)
    now_iso = _utc_now().isoformat()

    db.conn.execute(
        """
        INSERT INTO thread_summaries (
            conversation_id, summary, generated_at,
            message_count_at_summary, last_message_ts,
            pending_actions_json, salient_entity_ids_json, user_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            conversation_id,
            summary_text,
            now_iso,
            len(messages),
            last_message_ts,
            json.dumps(pending_actions, ensure_ascii=False),
            json.dumps(salient_entity_ids, ensure_ascii=False),
            user_id,
        ),
    )
    db.conn.commit()

    row = db.conn.execute(
        """
        SELECT id, conversation_id, summary, generated_at,
               message_count_at_summary, last_message_ts,
               pending_actions_json, salient_entity_ids_json, user_id
          FROM thread_summaries
         WHERE conversation_id = ?
         ORDER BY id DESC LIMIT 1
        """,
        (conversation_id,),
    ).fetchone()
    log.info(
        "thread_resume: summary persisted conv=%s msgs=%d len=%d",
        conversation_id[:24], len(messages), len(summary_text),
    )
    return _row_to_dict(row)


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict]:
    if row is None:
        return None
    out = dict(row)
    for jkey in ("pending_actions_json", "salient_entity_ids_json"):
        raw = out.get(jkey)
        if isinstance(raw, str) and raw:
            try:
                out[jkey.replace("_json", "")] = json.loads(raw)
            except json.JSONDecodeError:
                out[jkey.replace("_json", "")] = []
        else:
            out[jkey.replace("_json", "")] = []
    return out


def latest_thread_summary(db: MemoirsDB, conversation_id: str) -> Optional[dict]:
    """Most recent ``thread_summaries`` row for ``conversation_id``, or None."""
    _ensure_thread_summaries_table(db.conn)
    row = db.conn.execute(
        """
        SELECT id, conversation_id, summary, generated_at,
               message_count_at_summary, last_message_ts,
               pending_actions_json, salient_entity_ids_json, user_id
          FROM thread_summaries
         WHERE conversation_id = ?
         ORDER BY id DESC LIMIT 1
        """,
        (conversation_id,),
    ).fetchone()
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Resume entry point
# ---------------------------------------------------------------------------

def resume_thread(
    db: MemoirsDB,
    conversation_id: str,
    *,
    salient_limit: int = 8,
    generate_if_missing: bool = True,
    llm: Any | None = None,
) -> dict:
    """Hand back what an agent needs to re-orient on a paused thread.

    Shape::

        {
          "conversation_id": str,
          "summary": str,
          "generated_at": str | None,
          "salient_memories": [ { id, type, content, score } ],
          "last_decisions": [ str ],
          "pending_actions": [ str ],
          "project_context": { ... } | None,
        }

    If no thread summary exists yet and ``generate_if_missing`` is True we
    generate one on the fly (heuristic fallback if the curator is offline).
    """
    if not conversation_id:
        raise ValueError("conversation_id is required")

    summary_row = latest_thread_summary(db, conversation_id)
    if summary_row is None and generate_if_missing:
        summary_row = generate_thread_summary(db, conversation_id, llm=llm)

    messages = _collect_messages(db, conversation_id)

    last_decisions = _extract_last_decisions(messages)

    pending_tasks = _pending_tasks_from_corpus(db, conversation_id)
    if pending_tasks:
        pending_actions = pending_tasks
    elif summary_row is not None and summary_row.get("pending_actions"):
        pending_actions = [
            p for p in (summary_row.get("pending_actions") or [])
            if not _is_tool_noise(p)
        ]
    else:
        pending_actions = _extract_pending_actions(messages)

    recent_tool_calls = _recent_tool_calls_from_corpus(
        db, conversation_id, limit=PENDING_ACTION_MAX,
    )

    salient_memories = _scoped_memories(db, conversation_id, limit=salient_limit)

    # Optional project context — only attach when the conversation maps to a
    # known project (Claude Code transcripts usually do via metadata.cwd).
    project_context: Optional[dict] = None
    try:
        conv_md_row = db.conn.execute(
            "SELECT metadata_json FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conv_md_row and conv_md_row["metadata_json"]:
            try:
                conv_md = json.loads(conv_md_row["metadata_json"])
            except json.JSONDecodeError:
                conv_md = {}
            cwd = conv_md.get("cwd")
            project_name = None
            if isinstance(cwd, str):
                parts = [p for p in cwd.split("/") if p]
                for marker in ("projects", "Desktop", "src", "code"):
                    if marker in parts:
                        idx = parts.index(marker)
                        if idx + 1 < len(parts):
                            project_name = parts[idx + 1]
                            break
                if not project_name and parts:
                    project_name = parts[-1]
            if project_name:
                try:
                    from .graph import get_project_context
                    project_context = get_project_context(db, project_name, limit=10)
                except Exception:  # noqa: BLE001
                    project_context = None
    except sqlite3.Error:
        project_context = None

    return {
        "conversation_id": conversation_id,
        "summary": (summary_row or {}).get("summary"),
        "generated_at": (summary_row or {}).get("generated_at"),
        "message_count_at_summary": (summary_row or {}).get("message_count_at_summary"),
        "salient_memories": salient_memories,
        "last_decisions": last_decisions,
        "pending_actions": pending_actions,
        "recent_tool_calls": recent_tool_calls,
        "project_context": project_context,
    }


# ---------------------------------------------------------------------------
# Sleep cron job
# ---------------------------------------------------------------------------

def sleep_thread_summaries_job(
    db: MemoirsDB,
    *,
    idle_minutes: int = DEFAULT_IDLE_MINUTES,
    max_convs: int = DEFAULT_MAX_CONVS_PER_TICK,
    llm: Any | None = None,
) -> dict[str, Any]:
    """Cron-style job: detect idle conversations, summarize up to ``max_convs``.

    Returns ``{"detected": int, "summarized": int, "errors": int,
    "skipped": int, "convs": [...]}`` for the sleep_runs ledger.
    """
    detected = detect_idle_conversations(
        db, idle_minutes=idle_minutes, limit=max_convs * 2,
    )
    summarized = 0
    errors = 0
    skipped = 0
    convs: list[dict] = []
    for entry in detected[:max_convs]:
        cid = entry["conversation_id"]
        try:
            row = generate_thread_summary(db, cid, llm=llm)
        except Exception as e:  # noqa: BLE001
            errors += 1
            log.exception("sleep_thread_summaries: failed for %s", cid[:24])
            convs.append({"conversation_id": cid, "status": "error", "error": str(e)})
            continue
        if row is None:
            skipped += 1
            convs.append({"conversation_id": cid, "status": "skipped"})
        else:
            summarized += 1
            convs.append({
                "conversation_id": cid,
                "status": "summarized",
                "summary_id": row.get("id"),
                "len": len(row.get("summary") or ""),
            })
    return {
        "detected": len(detected),
        "summarized": summarized,
        "errors": errors,
        "skipped": skipped,
        "convs": convs,
        "max_convs_per_tick": int(max_convs),
        "idle_minutes": int(idle_minutes),
    }


# ---------------------------------------------------------------------------
# JSONL auto-detection (Claude Code projects)
# ---------------------------------------------------------------------------

def encode_cwd_for_claude(cwd: str | os.PathLike) -> str:
    """Return the encoded-cwd directory name Claude Code uses.

    The convention is ``-<absolute-path-with-slashes-replaced-by-dashes>``.
    Example: ``/home/me/code`` → ``-home-me-code``.
    """
    p = str(Path(cwd).resolve())
    if not p.startswith("/"):
        return "-" + p.replace("/", "-")
    return "-" + p[1:].replace("/", "-")


def find_latest_jsonl_for_cwd(
    cwd: str | os.PathLike | None = None,
    *,
    claude_root: Path | None = None,
) -> Optional[Path]:
    """Locate the most recently modified JSONL transcript for ``cwd``.

    Returns ``None`` if no transcript directory exists for the cwd.
    """
    cwd = cwd or os.getcwd()
    root = claude_root or (Path.home() / ".claude" / "projects")
    encoded = encode_cwd_for_claude(cwd)
    project_dir = Path(root) / encoded
    if not project_dir.exists() or not project_dir.is_dir():
        return None
    jsonls = sorted(
        project_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not jsonls:
        return None
    return jsonls[0]


def find_conversation_id_for_cwd(
    db: MemoirsDB,
    cwd: str | os.PathLike | None = None,
) -> Optional[str]:
    """Return the conversation_id matching the latest JSONL for ``cwd``.

    Resolution order:
      1. Latest JSONL filename → session id → look up via source_uri /
         conversations.external_id.
      2. Fallback: most recently updated conversation that points to a
         source under the cwd's encoded project dir.
    """
    jsonl = find_latest_jsonl_for_cwd(cwd)
    if jsonl is None:
        return None
    session_id = jsonl.stem  # claude_code uses the session uuid as the filename

    row = db.conn.execute(
        "SELECT c.id FROM conversations c "
        "JOIN sources s ON s.id = c.source_id "
        "WHERE c.external_id = ? "
        "ORDER BY c.updated_at DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if row:
        return row["id"]

    # Fallback: any conv whose source uri starts with the encoded project dir.
    encoded = jsonl.parent.name
    row = db.conn.execute(
        "SELECT c.id FROM conversations c "
        "JOIN sources s ON s.id = c.source_id "
        "WHERE s.uri LIKE ? "
        "ORDER BY c.updated_at DESC LIMIT 1",
        (f"%{encoded}%",),
    ).fetchone()
    return row["id"] if row else None
