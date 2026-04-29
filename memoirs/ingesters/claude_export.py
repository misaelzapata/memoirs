"""Claude.ai official export importer.

The user can request an export of their Claude.ai data from
Settings → Privacy → Export data. Anthropic emails them a `.zip` containing
(at minimum) `conversations.json` plus `users.json`, `projects.json` and
attachment files. Only `conversations.json` is required to reconstruct the
chat history; everything else is optional and ignored defensively.

Shape (verified against the public schema as of 2026-04):

    [
      {
        "uuid": "...",
        "name": "Conversation title",
        "created_at": "ISO",
        "updated_at": "ISO",
        "account": {"uuid": "..."},
        "chat_messages": [
          {
            "uuid": "...",
            "text": "...",                 # legacy / fallback
            "content": [                    # newer exports use a content array
              {"type": "text", "text": "..."}
            ],
            "sender": "human" | "assistant",
            "created_at": "ISO",
            "updated_at": "ISO",
            "attachments": [...],
            "files": [...]
          }
        ]
      }
    ]

Public docs are sparse, so the parser is permissive: unknown keys are dropped,
missing optional keys default to safe values, and malformed entries are
skipped instead of raising.

This is **distinct** from `claude_code.py`, which parses the per-session JSONL
transcripts that Claude Code (the CLI) writes locally.
"""
from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..db import MemoirsDB, content_hash
from ..models import RawConversation, RawMessage


SOURCE_KIND = "claude_export"
CONVERSATIONS_FILENAME = "conversations.json"

log = logging.getLogger("memoirs.ingesters.claude_export")


@dataclass
class ImportStats:
    """Result of an `import_claude_export` call."""

    conversations: int = 0
    messages: int = 0
    skipped_conversations: int = 0
    skipped_messages: int = 0
    source_uri: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversations": self.conversations,
            "messages": self.messages,
            "skipped_conversations": self.skipped_conversations,
            "skipped_messages": self.skipped_messages,
            "source_uri": self.source_uri,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_claude_export(zip_or_dir_path: Path, conn: MemoirsDB) -> ImportStats:
    """Import a Claude.ai official data export.

    Args:
        zip_or_dir_path: Path to either the `.zip` Anthropic emails the user, or
            an already-extracted directory containing `conversations.json`.
        conn: Initialised `MemoirsDB` (caller is responsible for `init()` and
            `close()`).

    Returns:
        ImportStats with counts. Idempotent: calling twice with the same input
        yields the same row counts thanks to the `(source_id, external_id)` and
        `(conversation_id, ordinal)` UNIQUE indexes.
    """
    path = Path(zip_or_dir_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"claude export path does not exist: {path}")

    payload, source_uri = _load_conversations_payload(path)
    if not isinstance(payload, list):
        raise ValueError(
            f"claude export {CONVERSATIONS_FILENAME} is not a JSON array: {path}"
        )

    conversations: list[RawConversation] = []
    skipped_conversations = 0
    skipped_messages = 0

    for raw in payload:
        if not isinstance(raw, dict):
            skipped_conversations += 1
            continue
        try:
            conv, n_skipped_msgs = _build_conversation(raw, source_uri)
        except _SkipConversation as exc:
            log.debug("skipping conversation: %s", exc)
            skipped_conversations += 1
            continue
        skipped_messages += n_skipped_msgs
        if not conv.messages:
            # No usable messages — don't pollute the DB with empty conversations.
            skipped_conversations += 1
            continue
        conversations.append(conv)

    if not conversations:
        return ImportStats(
            conversations=0,
            messages=0,
            skipped_conversations=skipped_conversations,
            skipped_messages=skipped_messages,
            source_uri=source_uri,
        )

    # Source fingerprint: when the input is a zip, hash its bytes; otherwise
    # leave hash_value=None — the directory may not have a stable identity.
    hash_value: str | None = None
    mtime_ns: int | None = None
    size_bytes: int | None = None
    if path.is_file():
        try:
            stat = path.stat()
            mtime_ns = stat.st_mtime_ns
            size_bytes = stat.st_size
            hash_value = content_hash(path.read_bytes())
        except OSError:
            pass

    run_id = conn.begin_import_run(
        source_uri,
        importer=SOURCE_KIND,
        file_mtime_ns=mtime_ns,
        file_size=size_bytes,
        hash_value=hash_value,
    )
    try:
        conv_count, msg_count = conn.save_conversations(
            conversations,
            source_name=path.name,
            source_kind=SOURCE_KIND,
            source_uri=source_uri,
            hash_value=hash_value,
            mtime_ns=mtime_ns,
            size_bytes=size_bytes,
        )
    except Exception as exc:
        conn.finish_import_run(run_id, status="failed", error=str(exc))
        raise

    conn.finish_import_run(
        run_id,
        status="completed",
        conversation_count=conv_count,
        message_count=msg_count,
    )

    # P0-4: drop a `messages_ingested` event so the sleep scheduler / extract
    # daemon notice the new corpus without polling. Best-effort — a failure to
    # enqueue must not break the import flow.
    if msg_count > 0:
        try:
            from ..engine.event_queue import enqueue_messages_ingested

            source_id_row = conn.conn.execute(
                "SELECT id FROM sources WHERE uri = ?", (source_uri,)
            ).fetchone()
            source_id = int(source_id_row["id"]) if source_id_row else None
            enqueue_messages_ingested(
                conn,
                conversation_id=None,  # batch import covers many conversations
                source_id=source_id,
                message_count=msg_count,
                extra={"importer": SOURCE_KIND, "conversations": conv_count},
            )
        except Exception:  # noqa: BLE001 — never block ingest on telemetry
            log.exception("event_queue enqueue failed for %s", source_uri)

    return ImportStats(
        conversations=conv_count,
        messages=msg_count,
        skipped_conversations=skipped_conversations,
        skipped_messages=skipped_messages,
        source_uri=source_uri,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


class _SkipConversation(Exception):
    """Raised internally to abort a single conversation without crashing the run."""


def _load_conversations_payload(path: Path) -> tuple[Any, str]:
    """Locate and load `conversations.json` from a zip or directory.

    Returns the parsed payload and the canonical `source_uri` (absolute path of
    the zip or the directory).
    """
    source_uri = str(path)
    if path.is_file():
        if path.suffix.lower() != ".zip":
            raise ValueError(
                f"claude export file must be a .zip (got {path.suffix}): {path}"
            )
        try:
            with zipfile.ZipFile(path) as archive:
                name = _find_conversations_member(archive.namelist())
                if name is None:
                    raise ValueError(
                        f"zip does not contain {CONVERSATIONS_FILENAME}: {path}"
                    )
                data = archive.read(name)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"not a valid zip archive: {path}") from exc
        try:
            return json.loads(data.decode("utf-8")), source_uri
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"malformed {CONVERSATIONS_FILENAME} in {path}: {exc}") from exc

    if path.is_dir():
        candidate = _find_conversations_in_dir(path)
        if candidate is None:
            raise FileNotFoundError(
                f"directory does not contain {CONVERSATIONS_FILENAME}: {path}"
            )
        try:
            return json.loads(candidate.read_text(encoding="utf-8")), source_uri
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed {candidate}: {exc}") from exc

    raise ValueError(f"claude export path is neither file nor directory: {path}")


def _find_conversations_member(names: Iterable[str]) -> str | None:
    """Return the zip member ending in `conversations.json` (root or subdir)."""
    # Prefer a top-level match; fall back to nested.
    candidates = [n for n in names if n.endswith(CONVERSATIONS_FILENAME)]
    if not candidates:
        return None
    candidates.sort(key=lambda n: (n.count("/"), len(n)))
    return candidates[0]


def _find_conversations_in_dir(directory: Path) -> Path | None:
    direct = directory / CONVERSATIONS_FILENAME
    if direct.is_file():
        return direct
    # Look one level deep (some users unzip into a wrapping folder).
    for child in directory.iterdir():
        if child.is_dir():
            nested = child / CONVERSATIONS_FILENAME
            if nested.is_file():
                return nested
    return None


def _build_conversation(
    raw: dict[str, Any], source_uri: str
) -> tuple[RawConversation, int]:
    """Convert a single Claude.ai conversation dict to RawConversation.

    Returns the conversation plus the count of messages that were skipped
    (empty content, malformed entries, etc.).
    """
    external_id = raw.get("uuid") or raw.get("id")
    if not external_id:
        raise _SkipConversation("conversation missing uuid")
    external_id = str(external_id)

    title = str(raw.get("name") or raw.get("title") or "Untitled Claude conversation")
    created_at = _normalize_ts(raw.get("created_at"))
    updated_at = _normalize_ts(raw.get("updated_at"))

    chat_messages = raw.get("chat_messages")
    if not isinstance(chat_messages, list):
        chat_messages = []

    messages: list[RawMessage] = []
    skipped = 0
    for item in chat_messages:
        if not isinstance(item, dict):
            skipped += 1
            continue
        msg = _build_message(item, ordinal=len(messages))
        if msg is None:
            skipped += 1
            continue
        messages.append(msg)

    account = raw.get("account") if isinstance(raw.get("account"), dict) else {}
    metadata = {
        "format": "claude_export",
        "claude_uuid": external_id,
        "account_uuid": account.get("uuid") if isinstance(account, dict) else None,
        "updated_at": updated_at,
    }
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return (
        RawConversation(
            external_id=external_id,
            title=title,
            source_kind=SOURCE_KIND,
            source_uri=source_uri,
            messages=messages,
            created_at=created_at,
            metadata=metadata,
        ),
        skipped,
    )


def _build_message(item: dict[str, Any], *, ordinal: int) -> RawMessage | None:
    sender = item.get("sender") or item.get("role")
    role = _normalize_role(sender)
    content = _extract_message_text(item)
    if not content.strip():
        return None

    external_id = item.get("uuid") or item.get("id")
    metadata = {
        "format": "claude_export",
        "claude_uuid": str(external_id) if external_id else None,
        "sender": str(sender) if sender is not None else None,
        "updated_at": _normalize_ts(item.get("updated_at")),
        "n_attachments": _safe_len(item.get("attachments")),
        "n_files": _safe_len(item.get("files")),
    }
    metadata = {k: v for k, v in metadata.items() if v not in (None, 0)}

    return RawMessage(
        role=role,
        content=content,
        ordinal=ordinal,
        created_at=_normalize_ts(item.get("created_at")),
        external_id=str(external_id) if external_id else None,
        metadata=metadata,
        raw=item,
    )


def _extract_message_text(item: dict[str, Any]) -> str:
    """Pull text content from either the legacy `text` field or the newer
    `content: [{type, text}]` array. Defensive against unknown block types."""
    parts: list[str] = []
    content_blocks = item.get("content")
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            elif btype == "tool_use":
                # Capture tool invocations as a readable line so the raw layer
                # keeps a trace without losing information.
                name = block.get("name") or "tool"
                parts.append(f"[tool_use: {name}]")
            elif btype == "tool_result":
                tr_content = block.get("content")
                if isinstance(tr_content, str) and tr_content.strip():
                    parts.append(f"[tool_result] {tr_content}")
            # Unknown block types are ignored on purpose.

    if parts:
        return "\n".join(parts).strip()

    # Fallback to the flat `text` field used by older exports.
    text = item.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


def _normalize_role(value: Any) -> str:
    role = str(value or "unknown").strip().lower()
    aliases = {
        "human": "user",
        "ai": "assistant",
        "bot": "assistant",
    }
    return aliases.get(role, role)


def _normalize_ts(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _safe_len(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    return 0
