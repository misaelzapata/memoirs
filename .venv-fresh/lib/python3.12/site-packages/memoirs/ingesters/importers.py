from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from .claude_code import is_claude_code_path, load_claude_code_jsonl
from .cursor import is_cursor_path, load_cursor_state
from ..db import MemoirsDB, content_hash, stable_id
from ..models import RawConversation, RawMessage


log = logging.getLogger("memoirs.ingesters.importers")


SUPPORTED_SUFFIXES = {".md", ".jsonl", ".json", ".zip", ".vscdb"}


class ImportErrorWithPath(RuntimeError):
    pass


def file_fingerprint(path: Path) -> tuple[int, int, str]:
    data = path.read_bytes()
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size, content_hash(data)


def load_conversations(path: Path) -> list[RawConversation]:
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix == ".jsonl" and is_claude_code_path(path):
        return load_claude_code_jsonl(path)
    if suffix == ".vscdb" or is_cursor_path(path):
        return load_cursor_state(path)
    if suffix == ".md":
        return load_markdown(path)
    if suffix == ".jsonl":
        return load_jsonl_file(path)
    if suffix == ".json":
        return load_json_file(path)
    if suffix == ".zip":
        return load_zip(path)
    raise ImportErrorWithPath(f"unsupported file type: {path}")


def load_markdown(path: Path) -> list[RawConversation]:
    text = path.read_text(encoding="utf-8")
    messages, prose = parse_jsonl_lines(text.splitlines(), source="markdown")
    source_uri = str(path.resolve())
    conversation_id = stable_id("markdown", source_uri)
    all_messages: list[RawMessage] = []
    ordinal = 0
    if prose.strip():
        all_messages.append(
            RawMessage(
                role="document",
                content=prose.strip(),
                ordinal=ordinal,
                metadata={"parser": "markdown_prose"},
                raw={"kind": "markdown_prose"},
            )
        )
        ordinal += 1
    for message in messages:
        all_messages.append(
            RawMessage(
                role=message.role,
                content=message.content,
                ordinal=ordinal,
                created_at=message.created_at,
                external_id=message.external_id,
                metadata={**message.metadata, "parser": "embedded_jsonl"},
                raw=message.raw,
            )
        )
        ordinal += 1
    return [
        RawConversation(
            external_id=conversation_id,
            title=path.stem,
            source_kind="markdown",
            source_uri=source_uri,
            messages=all_messages,
            metadata={"format": "markdown", "embedded_jsonl_messages": len(messages)},
        )
    ]


def load_jsonl_file(path: Path) -> list[RawConversation]:
    text = path.read_text(encoding="utf-8")
    messages, prose = parse_jsonl_lines(text.splitlines(), source="jsonl")
    if prose.strip():
        raise ImportErrorWithPath(f"non-jsonl content found in {path}")
    return [
        RawConversation(
            external_id=stable_id("jsonl", str(path.resolve())),
            title=path.stem,
            source_kind="jsonl",
            source_uri=str(path.resolve()),
            messages=messages,
            metadata={"format": "jsonl"},
        )
    ]


def parse_jsonl_lines(lines: list[str], *, source: str) -> tuple[list[RawMessage], str]:
    messages: list[RawMessage] = []
    prose_lines: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            prose_lines.append(line)
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            prose_lines.append(line)
            continue
        if not isinstance(payload, dict) or "role" not in payload or "content" not in payload:
            prose_lines.append(line)
            continue
        content = normalize_content(payload.get("content"))
        if not content.strip():
            continue
        messages.append(
            RawMessage(
                role=normalize_role(payload.get("role")),
                content=content,
                ordinal=len(messages),
                created_at=normalize_timestamp(payload.get("created_at") or payload.get("timestamp")),
                external_id=str(payload.get("id")) if payload.get("id") else None,
                metadata={"line": line_number, "source": source},
                raw=payload,
            )
        )
    return messages, "\n".join(prose_lines)


def load_json_file(path: Path) -> list[RawConversation]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_uri = str(path.resolve())
    if isinstance(payload, dict) and "messages" in payload:
        return [normalized_conversation(payload, source_uri, fallback_title=path.stem)]
    if isinstance(payload, list) and all(isinstance(item, dict) and "role" in item for item in payload):
        messages = messages_from_list(payload)
        return [
            RawConversation(
                external_id=stable_id("json", source_uri),
                title=path.stem,
                source_kind="json",
                source_uri=source_uri,
                messages=messages,
                metadata={"format": "message_list"},
            )
        ]
    if isinstance(payload, list) and payload and all(is_chatgpt_conversation(item) for item in payload):
        return [chatgpt_conversation(item, source_uri) for item in payload]
    if isinstance(payload, dict) and "conversations" in payload and isinstance(payload["conversations"], list):
        return [normalized_conversation(item, source_uri, fallback_title=path.stem) for item in payload["conversations"]]
    # Unknown JSON format (e.g. Claude Code subagent meta.json files): skip silently
    # rather than raising — this lets the watcher keep running without noise.
    return []


def load_zip(path: Path) -> list[RawConversation]:
    source_uri = str(path.resolve())
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        conversations_name = next((name for name in names if name.endswith("conversations.json")), None)
        if conversations_name is None:
            raise ImportErrorWithPath(f"zip does not contain conversations.json: {path}")
        payload = json.loads(archive.read(conversations_name).decode("utf-8"))
    if not isinstance(payload, list):
        raise ImportErrorWithPath(f"conversations.json is not a list: {path}")
    return [chatgpt_conversation(item, source_uri) for item in payload if is_chatgpt_conversation(item)]


def normalized_conversation(payload: dict[str, Any], source_uri: str, *, fallback_title: str) -> RawConversation:
    raw_messages = payload.get("messages", [])
    if not isinstance(raw_messages, list):
        raise ImportErrorWithPath("normalized conversation has non-list messages")
    return RawConversation(
        external_id=str(payload.get("conversation_id") or payload.get("id") or stable_id("json", source_uri, fallback_title)),
        title=str(payload.get("title") or fallback_title),
        source_kind="json",
        source_uri=source_uri,
        messages=messages_from_list(raw_messages),
        created_at=normalize_timestamp(payload.get("created_at")),
        metadata={"format": "normalized_json"},
    )


def messages_from_list(items: list[Any]) -> list[RawMessage]:
    messages: list[RawMessage] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = normalize_content(item.get("content"))
        if not content.strip():
            continue
        messages.append(
            RawMessage(
                role=normalize_role(item.get("role")),
                content=content,
                ordinal=len(messages),
                created_at=normalize_timestamp(item.get("created_at") or item.get("timestamp")),
                external_id=str(item.get("id")) if item.get("id") else None,
                metadata={"format": "message_list"},
                raw=item,
            )
        )
    return messages


def is_chatgpt_conversation(item: Any) -> bool:
    return isinstance(item, dict) and isinstance(item.get("mapping"), dict)


def chatgpt_conversation(payload: dict[str, Any], source_uri: str) -> RawConversation:
    mapping = payload.get("mapping", {})
    nodes = []
    if isinstance(mapping, dict):
        for node_id, node in mapping.items():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if not isinstance(message, dict):
                continue
            content = extract_chatgpt_content(message.get("content"))
            if not content.strip():
                continue
            author = message.get("author") or {}
            nodes.append(
                {
                    "node_id": node_id,
                    "message_id": message.get("id") or node_id,
                    "role": normalize_role(author.get("role") if isinstance(author, dict) else None),
                    "content": content,
                    "created_at": message.get("create_time"),
                    "raw": message,
                }
            )
    nodes.sort(key=lambda item: (item["created_at"] is None, item["created_at"] or 0, item["node_id"]))
    messages = [
        RawMessage(
            role=node["role"],
            content=node["content"],
            ordinal=index,
            created_at=normalize_timestamp(node["created_at"]),
            external_id=str(node["message_id"]),
            metadata={"format": "chatgpt_export", "node_id": node["node_id"]},
            raw=node["raw"],
        )
        for index, node in enumerate(nodes)
    ]
    external_id = str(payload.get("id") or payload.get("conversation_id") or stable_id("chatgpt", payload.get("title"), payload.get("create_time")))
    return RawConversation(
        external_id=external_id,
        title=str(payload.get("title") or "Untitled ChatGPT conversation"),
        source_kind="chatgpt",
        source_uri=source_uri,
        messages=messages,
        created_at=normalize_timestamp(payload.get("create_time")),
        metadata={"format": "chatgpt_export"},
    )


def extract_chatgpt_content(content_payload: Any) -> str:
    if not isinstance(content_payload, dict):
        return normalize_content(content_payload)
    parts = content_payload.get("parts")
    if isinstance(parts, list):
        return "\n".join(normalize_content(part) for part in parts if normalize_content(part).strip())
    return normalize_content(content_payload.get("text") or content_payload)


def normalize_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(normalize_content(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def normalize_role(value: Any) -> str:
    role = str(value or "unknown").strip().lower()
    aliases = {
        "human": "user",
        "ai": "assistant",
        "bot": "assistant",
    }
    return aliases.get(role, role)


def normalize_timestamp(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        from datetime import datetime, timezone

        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")
    return str(value)


# ---------------------------------------------------------------------------
# Save + per-conversation event_queue hook (P0-4 GAP)
# ---------------------------------------------------------------------------
#
# The four ingesters in this package (claude_code, cursor, this dispatcher's
# markdown / jsonl / json / chatgpt-zip paths) share the same need: parse a
# file, persist via :meth:`MemoirsDB.save_conversations`, and drop one
# ``messages_ingested`` event per conversation that actually grew. The hook
# is centralized here so the per-ingester wrappers stay tiny.
#
# Idempotency story:
#   * we count rows in ``messages`` per conversation BEFORE and AFTER the
#     save. ``_upsert_message`` uses ``ON CONFLICT(conversation_id, ordinal)``
#     so a re-run of the same file produces UPDATEs and no row count delta.
#   * we only enqueue when ``after - before > 0``, so re-running an ingester
#     on an unchanged source is a queue no-op.


def _conversation_message_count(db: "MemoirsDB", conversation_id: str) -> int:
    row = db.conn.execute(
        "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    return int(row["n"]) if row else 0


def _save_and_emit_events(
    db: "MemoirsDB",
    conversations: list[RawConversation],
    *,
    source_name: str,
    source_kind: str,
    source_uri: str,
    importer: str,
    hash_value: str | None = None,
    mtime_ns: int | None = None,
    size_bytes: int | None = None,
) -> dict[str, int]:
    """Persist ``conversations`` and emit per-conversation ``messages_ingested``.

    Returns a dict ``{conversations, messages, events_enqueued, new_messages}``.

    The function is *idempotent* with respect to the queue: a second call with
    no new content emits zero events because per-conversation message counts
    don't change.

    ``event_queue`` is imported lazily and any failure is swallowed so the
    ingest path is never blocked on telemetry.
    """
    if not conversations:
        return {
            "conversations": 0,
            "messages": 0,
            "events_enqueued": 0,
            "new_messages": 0,
        }

    # Snapshot per-conversation row counts BEFORE save so we can compute the
    # per-conversation delta. We use the stable_id formula that
    # ``save_conversations`` will apply. Legacy rows under a different id
    # would mean the BEFORE count is 0 here; that's fine — those rows are
    # rewritten as UPDATEs and the AFTER count picks them up too, so the
    # delta still reflects only NEW inserts.
    before_counts: dict[str, int] = {}
    for conv in conversations:
        cid = stable_id("conv", source_uri, conv.external_id)
        before_counts[cid] = _conversation_message_count(db, cid)

    conversation_count, message_count = db.save_conversations(
        conversations,
        source_name=source_name,
        source_kind=source_kind,
        source_uri=source_uri,
        hash_value=hash_value,
        mtime_ns=mtime_ns,
        size_bytes=size_bytes,
    )

    # Per-conversation enqueue. Best-effort: a queue failure must not break
    # ingest, so the enqueue module is imported lazily and any error is
    # logged-but-ignored.
    events_enqueued = 0
    new_messages_total = 0
    try:
        from ..engine.event_queue import enqueue_messages_ingested
    except ImportError:
        log.debug("event_queue module unavailable; skipping enqueue")
        return {
            "conversations": conversation_count,
            "messages": message_count,
            "events_enqueued": 0,
            "new_messages": 0,
        }

    source_id_row = db.conn.execute(
        "SELECT id FROM sources WHERE uri = ?", (source_uri,)
    ).fetchone()
    source_id = int(source_id_row["id"]) if source_id_row else None

    for conv in conversations:
        cid = stable_id("conv", source_uri, conv.external_id)
        after = _conversation_message_count(db, cid)
        delta = after - before_counts.get(cid, 0)
        if delta <= 0:
            continue
        try:
            enqueue_messages_ingested(
                db,
                conversation_id=cid,
                source_id=source_id,
                message_count=delta,
                extra={
                    "importer": importer,
                    "source_kind": source_kind,
                    "external_id": conv.external_id,
                },
            )
            events_enqueued += 1
            new_messages_total += delta
        except Exception:  # noqa: BLE001 — telemetry must never block ingest
            log.exception("event_queue enqueue failed for conv %s", cid)

    return {
        "conversations": conversation_count,
        "messages": message_count,
        "events_enqueued": events_enqueued,
        "new_messages": new_messages_total,
    }


def ingest_file_with_events(
    path: Path,
    db: "MemoirsDB",
    *,
    importer: str = "importers",
) -> dict[str, int]:
    """Dispatch ``load_conversations`` on ``path``, save, and emit events.

    Thin wrapper around :func:`load_conversations` + :func:`_save_and_emit_events`.
    Used by tests and by callers that want the full parse-save-enqueue flow
    without going through the watcher's polling logic.
    """
    path = path.resolve()
    conversations = load_conversations(path)
    if not conversations:
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
        source_kind=conversations[0].source_kind,
        source_uri=str(path),
        importer=importer,
        hash_value=hash_value,
        mtime_ns=mtime_ns,
        size_bytes=size_bytes,
    )
