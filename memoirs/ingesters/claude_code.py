"""Claude Code transcript parser.

Claude Code stores per-session JSONL transcripts at:
    ~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl

Each line is a JSON object. We keep `user` / `assistant` / `system` lines and skip
runtime artifacts (`queue-operation`, `progress`, `file-history-snapshot`,
`ai-title`, `last-prompt`). Anthropic-style content arrays (text / thinking /
tool_use / tool_result) are flattened into plain text so the raw layer keeps a
readable record without losing tool traces.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.normalize import flatten_content
from ..models import RawConversation, RawMessage

if TYPE_CHECKING:  # pragma: no cover
    from ..db import MemoirsDB


CLAUDE_CODE_PROJECTS = Path.home() / ".claude" / "projects"
SOURCE_KIND = "claude_code"
_RELEVANT_TYPES = {"user", "assistant", "system"}

log = logging.getLogger("memoirs.ingesters.claude_code")


def is_claude_code_path(path: Path) -> bool:
    """True if the path lives under ~/.claude/projects/ and is a .jsonl file."""
    try:
        path = Path(path).resolve()
    except OSError:
        return False
    if path.suffix.lower() != ".jsonl":
        return False
    try:
        path.relative_to(CLAUDE_CODE_PROJECTS.resolve())
    except ValueError:
        return False
    return True


def load_claude_code_jsonl(path: Path) -> list[RawConversation]:
    """Parse a single Claude Code session JSONL file."""
    path = Path(path).resolve()
    if not path.exists():
        return []

    session_id = path.stem  # the session-uuid filename
    project_dir = path.parent.name  # encoded cwd, e.g. -home-misael-Desktop-projects-foo
    source_uri = str(path)

    messages: list[RawMessage] = []
    cwd: str | None = None
    git_branch: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            t = obj.get("type")
            if t not in _RELEVANT_TYPES:
                continue

            msg = obj.get("message") or {}
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or t
            content = flatten_content(msg.get("content"))
            if not content.strip():
                continue

            # Capture session-wide metadata from first messages that carry it
            cwd = cwd or obj.get("cwd")
            git_branch = git_branch or obj.get("gitBranch")

            metadata = {
                "parentUuid": obj.get("parentUuid"),
                "cwd": obj.get("cwd"),
                "gitBranch": obj.get("gitBranch"),
                "model": msg.get("model"),
                "version": obj.get("version"),
                "userType": obj.get("userType"),
                "permissionMode": obj.get("permissionMode"),
                "line": line_num,
                "format": "claude_code_jsonl",
            }
            metadata = {k: v for k, v in metadata.items() if v is not None}

            messages.append(
                RawMessage(
                    role=role,
                    content=content,
                    ordinal=len(messages),
                    created_at=obj.get("timestamp"),
                    external_id=obj.get("uuid"),
                    metadata=metadata,
                    raw=obj,
                )
            )

    if not messages:
        return []

    title = f"claude_code: {project_dir}"
    return [
        RawConversation(
            external_id=session_id,
            title=title,
            source_kind=SOURCE_KIND,
            source_uri=source_uri,
            messages=messages,
            created_at=messages[0].created_at,
            metadata={
                "format": "claude_code_jsonl",
                "project_dir": project_dir,
                "session_id": session_id,
                "cwd": cwd,
                "gitBranch": git_branch,
            },
        )
    ]


# ---------------------------------------------------------------------------
# Save + event_queue hook (P0-4 GAP)
# ---------------------------------------------------------------------------


def ingest_claude_code_jsonl(path: Path, db: "MemoirsDB") -> dict[str, int]:
    """Parse a Claude Code JSONL session, persist it, and emit one
    ``messages_ingested`` event per conversation that gained new rows.

    Returns ``{conversations, messages, events_enqueued, new_messages}``.
    Idempotent: re-running on an unchanged file yields zero new events
    because :meth:`MemoirsDB._upsert_message` is keyed on
    ``(conversation_id, ordinal)``.

    The ``event_queue`` import is lazy and best-effort — a missing module or
    a queue failure must NOT break ingest, so we swallow those errors at the
    module boundary.
    """
    path = Path(path).resolve()
    conversations = load_claude_code_jsonl(path)
    if not conversations:
        return {
            "conversations": 0,
            "messages": 0,
            "events_enqueued": 0,
            "new_messages": 0,
        }
    try:
        from .importers import _save_and_emit_events, file_fingerprint
    except ImportError:  # pragma: no cover — same package, should always import
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
        importer="claude_code",
        hash_value=hash_value,
        mtime_ns=mtime_ns,
        size_bytes=size_bytes,
    )
