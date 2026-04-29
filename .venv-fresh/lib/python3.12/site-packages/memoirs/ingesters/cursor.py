"""Cursor chat history parser.

Cursor stores chat data inside per-workspace SQLite databases at:
    ~/.config/Cursor/User/workspaceStorage/<hash>/state.vscdb

Chat payloads are JSON blobs in `ItemTable` under keys like
`workbench.panel.aichat.view.aichat.chatdata`. Schema drifts between Cursor
versions, so this parser is defensive: it pulls candidate keys, walks the JSON
tree, and extracts anything that looks like a chat message. Raw JSON is kept in
`raw` so future versions can reparse without re-reading the .vscdb.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ..core.ids import stable_id
from ..models import RawConversation, RawMessage

if TYPE_CHECKING:  # pragma: no cover
    from ..db import MemoirsDB

log = logging.getLogger("memoirs.ingesters.cursor")


CURSOR_WORKSPACE_STORAGE = Path.home() / ".config" / "Cursor" / "User" / "workspaceStorage"
SOURCE_KIND = "cursor"

_CANDIDATE_KEYS = (
    "workbench.panel.aichat.view.aichat.chatdata",
    "aiService.prompts",
    "composer.composerData",
)


def is_cursor_path(path: Path) -> bool:
    """True if the path is a Cursor state.vscdb (by name or by location)."""
    try:
        path = Path(path).resolve()
    except OSError:
        return False
    if path.name == "state.vscdb":
        return True
    try:
        path.relative_to(CURSOR_WORKSPACE_STORAGE.resolve())
    except (ValueError, OSError):
        return False
    return path.suffix.lower() == ".vscdb"


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


from ..core.normalize import flatten_content as _flatten  # alias keeps existing call sites


def _extract_messages_from_blob(blob: Any) -> list[dict]:
    """Best-effort extraction of message dicts from a Cursor blob."""
    out: list[dict] = []

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 20:
            return
        if isinstance(node, dict):
            role = node.get("role") or node.get("type") or node.get("speaker")
            content_field = node.get("text") or node.get("content") or node.get("message")
            if (
                isinstance(role, str)
                and role.lower() in {"user", "assistant", "system", "human", "ai", "bot"}
                and content_field is not None
            ):
                role_norm = {"human": "user", "ai": "assistant", "bot": "assistant"}.get(
                    role.lower(), role.lower()
                )
                flat = _flatten(content_field)
                if flat.strip():
                    out.append(
                        {
                            "role": role_norm,
                            "content": flat,
                            "external_id": str(node.get("id") or node.get("messageId") or "") or None,
                            "created_at": node.get("timestamp") or node.get("createdAt"),
                            "raw": node,
                        }
                    )
            for v in node.values():
                walk(v, depth + 1)
        elif isinstance(node, list):
            for v in node:
                walk(v, depth + 1)

    walk(blob)
    return out


def load_cursor_state(path: Path) -> list[RawConversation]:
    """Read a Cursor state.vscdb file and return one RawConversation per chat blob."""
    path = Path(path).resolve()
    if not path.exists():
        return []

    workspace_hash = path.parent.name
    source_uri = str(path)

    try:
        ro = _open_readonly(path)
    except sqlite3.Error:
        return []

    try:
        rows = ro.execute(
            "SELECT key, value FROM ItemTable WHERE key IN ({})".format(
                ",".join("?" * len(_CANDIDATE_KEYS))
            ),
            _CANDIDATE_KEYS,
        ).fetchall()
    except sqlite3.Error:
        ro.close()
        return []
    ro.close()

    conversations: list[RawConversation] = []
    for key, raw in rows:
        try:
            blob = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        message_dicts = _extract_messages_from_blob(blob)
        if not message_dicts:
            continue
        messages = [
            RawMessage(
                role=m["role"],
                content=m["content"],
                ordinal=i,
                created_at=str(m["created_at"]) if m.get("created_at") else None,
                external_id=m.get("external_id"),
                metadata={"format": "cursor_state", "key": key},
                raw=m["raw"] if isinstance(m["raw"], dict) else {"value": str(m["raw"])[:500]},
            )
            for i, m in enumerate(message_dicts)
        ]
        conversations.append(
            RawConversation(
                external_id=stable_id("cursor", workspace_hash, key),
                title=f"cursor: {workspace_hash}",
                source_kind=SOURCE_KIND,
                source_uri=source_uri,
                messages=messages,
                metadata={
                    "format": "cursor_state",
                    "workspace_hash": workspace_hash,
                    "key": key,
                },
            )
        )
    return conversations


# ---------------------------------------------------------------------------
# Save + event_queue hook (P0-4 GAP)
# ---------------------------------------------------------------------------


def ingest_cursor_state(path: Path, db: "MemoirsDB") -> dict[str, int]:
    """Parse a Cursor ``state.vscdb``, persist its conversations, and emit
    one ``messages_ingested`` event per conversation that gained new rows.

    Returns ``{conversations, messages, events_enqueued, new_messages}``.
    Idempotent — re-running on an unchanged file produces zero events.
    """
    path = Path(path).resolve()
    conversations = load_cursor_state(path)
    if not conversations:
        return {
            "conversations": 0,
            "messages": 0,
            "events_enqueued": 0,
            "new_messages": 0,
        }
    try:
        from .importers import _save_and_emit_events, file_fingerprint
    except ImportError:  # pragma: no cover
        log.exception("ingesters.importers helper unavailable")
        return {
            "conversations": 0,
            "messages": 0,
            "events_enqueued": 0,
            "new_messages": 0,
        }
    try:
        mtime_ns, size_bytes, hash_value = file_fingerprint(path)
    except OSError:
        mtime_ns, size_bytes, hash_value = None, None, None
    return _save_and_emit_events(
        db,
        conversations,
        source_name=path.name,
        source_kind=SOURCE_KIND,
        source_uri=str(path),
        importer="cursor",
        hash_value=hash_value,
        mtime_ns=mtime_ns,
        size_bytes=size_bytes,
    )
