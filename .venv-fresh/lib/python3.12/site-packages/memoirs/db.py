from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import importlib

from .core.ids import content_hash, stable_id, utc_now
from .migrations import (
    current_version as _migrations_current_version,
    run_pending_migrations as _run_pending_migrations,
    target_version as _migrations_target_version,
)
from .models import RawConversation, RawMessage


# ---------------------------------------------------------------------------
# Optional encryption-at-rest via SQLCipher (P3-2)
# ---------------------------------------------------------------------------
# sqlcipher3-binary ships a precompiled drop-in replacement for the stdlib
# sqlite3 module that links against SQLCipher 4.x. We import it lazily and
# only use it when MEMOIRS_ENCRYPT_KEY is set; without the key we behave
# exactly like before (plain sqlite3, no overhead).
try:  # pragma: no cover - import guard, exercised via tests
    import sqlcipher3.dbapi2 as _sqlcipher  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _sqlcipher = None  # type: ignore[assignment]


_ENCRYPT_ENV = "MEMOIRS_ENCRYPT_KEY"
# SQLite header magic — encrypted DBs intentionally don't expose this.
_SQLITE_MAGIC = b"SQLite format 3\x00"


def _is_hex_key(key: str) -> bool:
    """Return True if ``key`` is exactly 64 hex chars (raw 256-bit key)."""
    if len(key) != 64:
        return False
    try:
        int(key, 16)
        return True
    except ValueError:
        return False


def _format_key_pragma(key: str) -> str:
    """Render a ``PRAGMA key`` value safely.

    SQLCipher's ``PRAGMA key`` does NOT support parameterized binding, so we
    must inline the value. We accept two formats:

    * 64 hex chars  → emitted as ``x'<hex>'`` (raw key, no KDF).
    * any other str → emitted as ``'<passphrase>'`` with single-quote doubling
      (the standard SQL string-literal escape).
    """
    if _is_hex_key(key):
        return f"\"x'{key}'\""
    escaped = key.replace("'", "''")
    return f"'{escaped}'"


def is_encrypted(path: Path | str) -> bool:
    """Return True if ``path`` looks like a SQLCipher-encrypted DB.

    A plaintext SQLite DB starts with ``SQLite format 3\\x00``; SQLCipher
    encrypts the entire file (header included) so a healthy encrypted DB
    will *not* match that magic. Empty / missing files return False.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size < 16:
        return False
    with p.open("rb") as fh:
        head = fh.read(16)
    return head != _SQLITE_MAGIC


def _open_encrypted(
    path: Path,
    key: str,
    *,
    timeout: float = 30.0,
    check_same_thread: bool = False,
):
    """Open an encrypted connection via sqlcipher3 and apply cipher PRAGMAs.

    Raises :class:`ImportError` with a descriptive message if sqlcipher3 is
    unavailable. The caller (``MemoirsDB.__init__``) catches this and
    propagates it — we deliberately do NOT fall back to plain sqlite3 here:
    silently dropping encryption when the user asked for it would be a
    serious security footgun.
    """
    if _sqlcipher is None:
        raise ImportError(
            f"{_ENCRYPT_ENV} is set but sqlcipher3 is not installed. "
            "Install with: pip install 'memoirs[encryption]' "
            "(or pip install sqlcipher3-binary). "
            "Refusing to open the DB without encryption."
        )
    conn = _sqlcipher.connect(
        str(path), timeout=timeout, check_same_thread=check_same_thread
    )
    # Set the key BEFORE any other access — SQLCipher requires this.
    conn.execute(f"PRAGMA key = {_format_key_pragma(key)}")
    # Modern SQLCipher 4 defaults: 4 KiB pages and 256k KDF iterations.
    # These match SQLCipher 4's secure defaults (2024-2026 baseline).
    conn.execute("PRAGMA cipher_page_size = 4096")
    conn.execute("PRAGMA kdf_iter = 256000")
    # Validate the key by forcing a real read; a wrong key raises
    # DatabaseError("file is not a database") on the first query.
    conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
    return conn


# ---------------------------------------------------------------------------
# SQLite PRAGMA tuning
# ---------------------------------------------------------------------------
# These defaults trade a small amount of durability (synchronous=NORMAL with
# WAL is fsync-on-checkpoint instead of every commit — losing the very last
# transaction on power loss is possible but the DB stays consistent) for a
# substantial improvement in retrieval latency. Tunable via env vars:
#   MEMOIRS_SQLITE_MMAP_MB   (default 256)  — mmap_size in MiB
#   MEMOIRS_SQLITE_CACHE_MB  (default 64)   — cache_size in MiB (negative form)
_DEFAULT_MMAP_MB = 256
_DEFAULT_CACHE_MB = 64


def _env_int(name: str, default: int) -> int:
    """Read a non-negative int env var, falling back to ``default`` on bad values."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply memoirs' standard PRAGMAs to ``conn``.

    Idempotent: safe to call multiple times on the same connection. Exposed as
    a module-level helper so tests and tooling that open raw ``sqlite3``
    connections (bypassing :class:`MemoirsDB`) can still benefit from the
    same tuning.

    Order matters: ``journal_mode = WAL`` should be set before
    ``synchronous = NORMAL`` (NORMAL is only safe under WAL).
    """
    mmap_mb = _env_int("MEMOIRS_SQLITE_MMAP_MB", _DEFAULT_MMAP_MB)
    cache_mb = _env_int("MEMOIRS_SQLITE_CACHE_MB", _DEFAULT_CACHE_MB)

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    # Memory-map up to N MiB of the DB file: read-heavy paths (vec0 ANN scans,
    # FTS5 lookups) avoid the read() syscall.
    conn.execute(f"PRAGMA mmap_size = {mmap_mb * 1024 * 1024}")
    # Keep transient B-tree / sort scratch in RAM rather than spilling to a
    # temp file. sqlite-vec's auxiliary tables are regular tables, not temp
    # tables, so this does not affect ANN correctness.
    conn.execute("PRAGMA temp_store = MEMORY")
    # Negative cache_size means "KiB", so -65536 ≈ 64 MiB of page cache.
    conn.execute(f"PRAGMA cache_size = -{cache_mb * 1024}")


# Back-compat: external code (and older tests) imported the SQL text via
# ``from memoirs.db import SCHEMA``. The single source of truth now lives in
# the baseline migration; we re-export it so legacy imports keep working.
# Module names starting with a digit can't be imported with regular ``from``
# syntax, so we go through importlib.
_initial = importlib.import_module("memoirs.migrations.001_initial")
SCHEMA = _initial.SCHEMA


class MemoirsDB:
    def __init__(
        self,
        path: Path | str,
        *,
        auto_migrate: bool = True,
        encryption_key: Optional[str] = None,
    ) -> None:
        """Open the SQLite database at ``path``.

        When ``auto_migrate`` is True (the default), :meth:`init` runs every
        pending migration. Set it to False to open the DB without touching
        the schema (useful in tests, when inspecting a stale DB, or when the
        caller wants to drive migrations explicitly via the ``memoirs db``
        CLI).

        Encryption-at-rest (P3-2) is enabled when either ``encryption_key`` is
        passed or the ``MEMOIRS_ENCRYPT_KEY`` env var is set. The DB is then
        opened via sqlcipher3 (drop-in replacement for sqlite3) with
        SQLCipher 4 defaults (4 KiB page size, 256k KDF iterations). Without
        a key, behavior is identical to a stdlib sqlite3 connection.

        If a key is requested but ``sqlcipher3`` isn't installed, opening
        raises :class:`ImportError` — we do NOT silently downgrade to
        plaintext, since the caller explicitly asked for encryption.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = encryption_key
        if key is None:
            env_key = os.environ.get(_ENCRYPT_ENV)
            if env_key:
                key = env_key
        self._encrypted = key is not None
        if self._encrypted:
            # Encrypted path: route through sqlcipher3. Failure to import is
            # surfaced as ImportError to the caller (no silent fallback).
            try:
                self.conn = _open_encrypted(
                    self.path,
                    key,  # type: ignore[arg-type]
                    timeout=30.0,
                    check_same_thread=False,
                )
            except ImportError:
                logging.getLogger("memoirs.db").error(
                    "encryption requested but sqlcipher3 unavailable; refusing to open"
                )
                raise
            # sqlcipher3 builds against its own bundled sqlite, so its Row
            # class lives on sqlcipher3.dbapi2 — fall back to sqlite3.Row if
            # missing (older bindings).
            row_factory = getattr(_sqlcipher, "Row", sqlite3.Row)  # type: ignore[union-attr]
            self.conn.row_factory = row_factory
        else:
            # 30s timeout: avoids "database is locked" when watcher + MCP server +
            # CLI run concurrently. WAL mode lets readers proceed during writes.
            self.conn = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
        _apply_pragmas(self.conn)
        self._auto_migrate = auto_migrate

    def close(self) -> None:
        self.conn.close()

    def init(self) -> None:
        """Apply pending schema migrations.

        Delegates to :mod:`memoirs.migrations`. The migration runner uses
        ``PRAGMA user_version`` to track which migrations have been applied
        and is fully idempotent on a current DB.

        If ``auto_migrate=False`` was passed to the constructor, this method
        is a no-op (the caller is responsible for invoking the migration
        CLI / runner themselves).
        """
        if not self._auto_migrate:
            return
        current = _migrations_current_version(self.conn)
        target = _migrations_target_version()
        if current > target:
            # The DB was created by a newer memoirs version — don't try to
            # downgrade silently. Warn and leave the schema alone.
            import logging
            logging.getLogger("memoirs.db").warning(
                "DB at %s has user_version=%d but code expects %d — please upgrade memoirs",
                self.path, current, target,
            )
            return
        _run_pending_migrations(self.conn)

    def begin_import_run(
        self,
        source_uri: str,
        importer: str,
        *,
        file_mtime_ns: int | None = None,
        file_size: int | None = None,
        hash_value: str | None = None,
    ) -> int:
        now = utc_now()
        cursor = self.conn.execute(
            """
            INSERT INTO import_runs (
                source_uri, importer, status, started_at, file_mtime_ns,
                file_size, content_hash
            )
            VALUES (?, ?, 'running', ?, ?, ?, ?)
            """,
            (source_uri, importer, now, file_mtime_ns, file_size, hash_value),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def finish_import_run(
        self,
        run_id: int,
        *,
        status: str,
        conversation_count: int = 0,
        message_count: int = 0,
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE import_runs
            SET status = ?, finished_at = ?, conversation_count = ?,
                message_count = ?, error = ?
            WHERE id = ?
            """,
            (status, utc_now(), conversation_count, message_count, error, run_id),
        )
        self.conn.commit()

    def upsert_source(
        self,
        *,
        uri: str,
        kind: str,
        name: str,
        hash_value: str | None,
        mtime_ns: int | None,
        size_bytes: int | None,
    ) -> int:
        now = utc_now()
        self.conn.execute(
            """
            INSERT INTO sources (
                uri, kind, name, content_hash, mtime_ns, size_bytes,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uri) DO UPDATE SET
                kind = excluded.kind,
                name = excluded.name,
                content_hash = excluded.content_hash,
                mtime_ns = excluded.mtime_ns,
                size_bytes = excluded.size_bytes,
                updated_at = excluded.updated_at
            """,
            (uri, kind, name, hash_value, mtime_ns, size_bytes, now, now),
        )
        row = self.conn.execute("SELECT id FROM sources WHERE uri = ?", (uri,)).fetchone()
        if row is None:
            raise RuntimeError(f"source was not saved: {uri}")
        return int(row["id"])

    def save_conversations(
        self,
        conversations: Iterable[RawConversation],
        *,
        source_name: str,
        source_kind: str,
        source_uri: str,
        hash_value: str | None,
        mtime_ns: int | None,
        size_bytes: int | None,
    ) -> tuple[int, int]:
        conversation_list = list(conversations)
        source_id = self.upsert_source(
            uri=source_uri,
            kind=source_kind,
            name=source_name,
            hash_value=hash_value,
            mtime_ns=mtime_ns,
            size_bytes=size_bytes,
        )

        conversation_count = 0
        message_count = 0
        now = utc_now()
        with self.conn:
            for conversation in conversation_list:
                # Calculate the would-be ID for new rows.
                computed_id = stable_id("conv", source_uri, conversation.external_id)
                metadata_json = json.dumps(conversation.metadata, ensure_ascii=False, sort_keys=True)
                self.conn.execute(
                    """
                    INSERT INTO conversations (
                        id, source_id, external_id, title, created_at,
                        updated_at, message_count, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, external_id) DO UPDATE SET
                        title = excluded.title,
                        created_at = COALESCE(conversations.created_at, excluded.created_at),
                        updated_at = excluded.updated_at,
                        message_count = excluded.message_count,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        computed_id,
                        source_id,
                        conversation.external_id,
                        conversation.title,
                        conversation.created_at,
                        now,
                        len(conversation.messages),
                        metadata_json,
                    ),
                )
                # CRITICAL: ON CONFLICT may have UPDATEd an existing row that has a
                # DIFFERENT id (legacy rows from before stable_id changed format).
                # We must fetch the actual id used by the conversations row to
                # avoid FOREIGN KEY failures when inserting messages.
                actual_row = self.conn.execute(
                    "SELECT id FROM conversations WHERE source_id = ? AND external_id = ?",
                    (source_id, conversation.external_id),
                ).fetchone()
                conversation_id = actual_row["id"] if actual_row else computed_id
                self.conn.execute(
                    "UPDATE messages SET is_active = 0, updated_at = ? WHERE conversation_id = ?",
                    (now, conversation_id),
                )
                for message in conversation.messages:
                    self._upsert_message(conversation_id, message, now)
                conversation_count += 1
                message_count += len(conversation.messages)
        return conversation_count, message_count

    def ingest_event(self, event: dict[str, object]) -> dict[str, object]:
        event_type = str(event.get("type") or "chat_message")
        source_name = str(event.get("source") or event.get("client") or "mcp")
        source_uri = str(event.get("source_uri") or f"mcp://{source_name}")
        source_kind = str(event.get("source_kind") or "mcp")
        conversation_external_id = str(
            event.get("conversation_id")
            or event.get("thread_id")
            or event.get("session_id")
            or f"events:{source_name}"
        )
        title = str(event.get("title") or event.get("project") or conversation_external_id)
        content = normalize_event_content(event.get("content") or event.get("text") or event)
        if not content.strip():
            raise ValueError("event content is required")
        role = normalize_event_role(event.get("role"), event_type)
        created_at = normalize_event_timestamp(event.get("created_at") or event.get("timestamp"))
        external_id = event.get("message_id") or event.get("event_id") or event.get("id")
        if external_id is None:
            external_id = stable_id("event", source_uri, conversation_external_id, event_type, role, content, created_at)
        external_id = str(external_id)

        metadata = {
            "event_type": event_type,
            "project": event.get("project"),
            "client": event.get("client"),
            "metadata": event.get("metadata"),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        now = utc_now()
        event_hash = content_hash(f"{role}\n{content}")
        with self.conn:
            source_id = self.upsert_source(
                uri=source_uri,
                kind=source_kind,
                name=source_name,
                hash_value=None,
                mtime_ns=None,
                size_bytes=None,
            )
            computed_id = stable_id("conv", source_uri, conversation_external_id)
            self.conn.execute(
                """
                INSERT INTO conversations (
                    id, source_id, external_id, title, created_at,
                    updated_at, message_count, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(source_id, external_id) DO UPDATE SET
                    title = COALESCE(excluded.title, conversations.title),
                    updated_at = excluded.updated_at
                """,
                (
                    computed_id,
                    source_id,
                    conversation_external_id,
                    title,
                    created_at,
                    now,
                    json.dumps({"source": "mcp_event"}, ensure_ascii=False, sort_keys=True),
                ),
            )
            # Fetch actual id (handles legacy rows with different stable_id format)
            actual = self.conn.execute(
                "SELECT id FROM conversations WHERE source_id = ? AND external_id = ?",
                (source_id, conversation_external_id),
            ).fetchone()
            conversation_id = actual["id"] if actual else computed_id
            existing = self.conn.execute(
                """
                SELECT id, ordinal
                FROM messages
                WHERE conversation_id = ? AND external_id = ?
                """,
                (conversation_id, external_id),
            ).fetchone()
            raw_json = json.dumps(event, ensure_ascii=False, sort_keys=True)
            metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            if existing:
                message_id = str(existing["id"])
                ordinal = int(existing["ordinal"])
                self.conn.execute(
                    """
                    UPDATE messages
                    SET role = ?, content = ?, created_at = ?, content_hash = ?,
                        raw_json = ?, metadata_json = ?, is_active = 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (role, content, created_at, event_hash, raw_json, metadata_json, now, message_id),
                )
                action = "updated"
            else:
                row = self.conn.execute(
                    "SELECT COALESCE(MAX(ordinal), -1) + 1 AS next_ordinal FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
                ordinal = int(row["next_ordinal"])
                message_id = stable_id("msg", conversation_id, external_id)
                self.conn.execute(
                    """
                    INSERT INTO messages (
                        id, conversation_id, external_id, role, content, ordinal,
                        created_at, content_hash, raw_json, metadata_json, is_active,
                        first_seen_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        message_id,
                        conversation_id,
                        external_id,
                        role,
                        content,
                        ordinal,
                        created_at,
                        event_hash,
                        raw_json,
                        metadata_json,
                        now,
                        now,
                    ),
                )
                action = "inserted"
            self.conn.execute(
                """
                UPDATE conversations
                SET message_count = (
                    SELECT COUNT(*) FROM messages
                    WHERE conversation_id = conversations.id AND is_active = 1
                ),
                updated_at = ?
                WHERE id = ?
                """,
                (now, conversation_id),
            )
        return {
            "action": action,
            "source_uri": source_uri,
            "conversation_id": conversation_id,
            "conversation_external_id": conversation_external_id,
            "message_id": message_id,
            "message_external_id": external_id,
            "ordinal": ordinal,
        }

    def ingest_conversation_event(self, payload: dict[str, object]) -> dict[str, object]:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        results = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            event = {
                **payload,
                **message,
                "type": message.get("type") or "chat_message",
                "conversation_id": payload.get("conversation_id") or payload.get("thread_id"),
                "title": payload.get("title"),
                "source": payload.get("source") or message.get("source"),
                "source_uri": payload.get("source_uri") or message.get("source_uri"),
                "metadata": {
                    "batch_index": index,
                    "conversation_metadata": payload.get("metadata"),
                    "message_metadata": message.get("metadata"),
                },
            }
            event.pop("messages", None)
            results.append(self.ingest_event(event))
        return {
            "conversation_id": results[0]["conversation_id"] if results else None,
            "message_count": len(results),
            "results": results,
        }

    def _upsert_message(self, conversation_id: str, message: RawMessage, now: str) -> None:
        message_hash = content_hash(f"{message.role}\n{message.content}")
        message_id = stable_id("msg", conversation_id, message.ordinal, message.external_id or message_hash)
        raw_json = json.dumps(message.raw, ensure_ascii=False, sort_keys=True)
        metadata_json = json.dumps(message.metadata, ensure_ascii=False, sort_keys=True)
        self.conn.execute(
            """
            INSERT INTO messages (
                id, conversation_id, external_id, role, content, ordinal,
                created_at, content_hash, raw_json, metadata_json, is_active,
                first_seen_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(conversation_id, ordinal) DO UPDATE SET
                external_id = excluded.external_id,
                role = excluded.role,
                content = excluded.content,
                created_at = excluded.created_at,
                content_hash = excluded.content_hash,
                raw_json = excluded.raw_json,
                metadata_json = excluded.metadata_json,
                is_active = 1,
                updated_at = excluded.updated_at
            """,
            (
                message_id,
                conversation_id,
                message.external_id,
                message.role,
                message.content,
                message.ordinal,
                message.created_at,
                message_hash,
                raw_json,
                metadata_json,
                now,
                now,
            ),
        )

    def status(self) -> dict[str, object]:
        tables = {
            "sources": "SELECT COUNT(*) AS count FROM sources",
            "conversations": "SELECT COUNT(*) AS count FROM conversations",
            "active_messages": "SELECT COUNT(*) AS count FROM messages WHERE is_active = 1",
            "messages_total": "SELECT COUNT(*) AS count FROM messages",
        }
        result: dict[str, object] = {}
        for key, query in tables.items():
            result[key] = int(self.conn.execute(query).fetchone()["count"])
        result["recent_runs"] = [
            dict(row)
            for row in self.conn.execute(
                """
                SELECT id, source_uri, importer, status, started_at, finished_at,
                       conversation_count, message_count, error
                FROM import_runs
                ORDER BY id DESC
                LIMIT 5
                """
            ).fetchall()
        ]
        return result

    def list_conversations(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT c.id, c.title, c.message_count, c.updated_at, s.name AS source_name,
                   s.kind AS source_kind, s.uri AS source_uri
            FROM conversations c
            JOIN sources s ON s.id = c.source_id
            ORDER BY c.updated_at DESC
            """
        ).fetchall()

    def list_messages(self, conversation_id: str | None = None, limit: int = 20) -> list[sqlite3.Row]:
        params: list[object] = []
        where = "WHERE is_active = 1"
        if conversation_id:
            where += " AND conversation_id = ?"
            params.append(conversation_id)
        params.append(limit)
        return self.conn.execute(
            f"""
            SELECT conversation_id, ordinal, role, content, created_at
            FROM messages
            {where}
            ORDER BY conversation_id, ordinal
            LIMIT ?
            """,
            params,
        ).fetchall()


def normalize_event_content(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_event_role(value: object, event_type: str) -> str:
    if value is None:
        return "event" if event_type != "chat_message" else "unknown"
    role = str(value).strip().lower()
    aliases = {"human": "user", "ai": "assistant", "bot": "assistant"}
    return aliases.get(role, role)


def normalize_event_timestamp(value: object) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")
    return str(value)
