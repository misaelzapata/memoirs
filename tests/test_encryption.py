"""Tests for P3-2 encryption-at-rest via SQLCipher.

Most tests are gated by ``pytest.importorskip("sqlcipher3")`` so they're a
no-op on systems without the optional ``encryption`` extra installed. The
final guardrail test (``test_missing_sqlcipher_raises_importerror``) is
intentionally NOT skipped — it exercises the "key set but binding absent"
guardrail by monkeypatching the binding to ``None``, which is the same
condition users hit when the dep isn't installed.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys
import importlib.util
from pathlib import Path

import pytest

from memoirs.db import MemoirsDB, is_encrypted, _SQLITE_MAGIC  # noqa: E402

# Skip the SQLCipher-dependent tests when the binding isn't installed.
_HAS_SQLCIPHER = importlib.util.find_spec("sqlcipher3") is not None
requires_sqlcipher = pytest.mark.skipif(
    not _HAS_SQLCIPHER,
    reason="sqlcipher3 not installed (extras 'encryption'); skipping live SQLCipher tests",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _seed_basic(db: MemoirsDB) -> None:
    """Force schema migrations to run + insert one trivial row so we can
    later assert the round-trip preserved data."""
    db.init()
    db.conn.execute(
        "INSERT INTO sources (uri, kind, name, created_at, updated_at) "
        "VALUES ('test://x', 'test', 'x', '2026-01-01', '2026-01-01')"
    )
    db.conn.commit()


def _row_count(conn) -> int:
    return int(conn.execute("SELECT count(*) FROM sources").fetchone()[0])


# ---------------------------------------------------------------------------
# core encryption behavior
# ---------------------------------------------------------------------------


@requires_sqlcipher
def test_encrypted_db_does_not_have_sqlite_magic(tmp_path: Path, monkeypatch):
    """A DB created with MEMOIRS_ENCRYPT_KEY set must not begin with the
    plaintext SQLite header — that's the whole point."""
    monkeypatch.setenv("MEMOIRS_ENCRYPT_KEY", "test-passphrase-abc123")
    db_path = tmp_path / "secret.sqlite"
    db = MemoirsDB(db_path)
    try:
        _seed_basic(db)
    finally:
        db.close()

    assert db_path.exists()
    head = db_path.read_bytes()[:16]
    assert head != _SQLITE_MAGIC, "encrypted DB should not start with SQLite magic header"


@requires_sqlcipher
def test_open_encrypted_db_without_key_fails(tmp_path: Path, monkeypatch):
    """Once a DB is encrypted, opening it without the env key must NOT yield
    a usable connection — sqlite3 will see garbage and raise on first read."""
    monkeypatch.setenv("MEMOIRS_ENCRYPT_KEY", "right-key")
    db_path = tmp_path / "secret.sqlite"
    db = MemoirsDB(db_path)
    try:
        _seed_basic(db)
    finally:
        db.close()

    # Drop the env var → MemoirsDB will use plain sqlite3, which can't read
    # an encrypted file. Either the open or the first query must raise.
    monkeypatch.delenv("MEMOIRS_ENCRYPT_KEY")
    with pytest.raises((sqlite3.DatabaseError, sqlite3.OperationalError)):
        db2 = MemoirsDB(db_path)
        try:
            db2.conn.execute("SELECT count(*) FROM sources").fetchone()
        finally:
            db2.close()


@requires_sqlcipher
def test_open_encrypted_db_with_correct_key_works(tmp_path: Path, monkeypatch):
    """Closing and reopening with the same env key must restore full access
    (all migration tables present, row count preserved)."""
    monkeypatch.setenv("MEMOIRS_ENCRYPT_KEY", "secret123")
    db_path = tmp_path / "secret.sqlite"
    db = MemoirsDB(db_path)
    try:
        _seed_basic(db)
        before = _row_count(db.conn)
    finally:
        db.close()

    db2 = MemoirsDB(db_path)
    try:
        # All baseline tables (from migration 001) must be present.
        rows = db2.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r[0] for r in rows}
        assert "sources" in names
        assert "conversations" in names
        assert "messages" in names
        assert _row_count(db2.conn) == before
    finally:
        db2.close()


@requires_sqlcipher
def test_is_encrypted_detects_plain_vs_encrypted(tmp_path: Path, monkeypatch):
    """The is_encrypted() helper must reliably distinguish the two formats."""
    plain_path = tmp_path / "plain.sqlite"
    enc_path = tmp_path / "enc.sqlite"

    # Plain DB
    db = MemoirsDB(plain_path)
    try:
        _seed_basic(db)
    finally:
        db.close()
    assert is_encrypted(plain_path) is False

    # Encrypted DB
    monkeypatch.setenv("MEMOIRS_ENCRYPT_KEY", "k")
    db = MemoirsDB(enc_path)
    try:
        _seed_basic(db)
    finally:
        db.close()
    assert is_encrypted(enc_path) is True

    # Missing file
    assert is_encrypted(tmp_path / "nope.sqlite") is False


# ---------------------------------------------------------------------------
# CLI round-trip: encrypt → decrypt
# ---------------------------------------------------------------------------


def _run_cli(*argv: str, env_overrides: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``memoirs <argv>`` as a subprocess so the env (in particular
    MEMOIRS_ENCRYPT_KEY) is fully isolated from the parent test process."""
    import os
    env = dict(os.environ)
    # Strip the encryption env so the CLI uses plain sqlite3 unless explicitly
    # asked otherwise; tests that need it set re-add it via env_overrides.
    env.pop("MEMOIRS_ENCRYPT_KEY", None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "memoirs", *argv],
        capture_output=True, text=True, env=env, check=False,
    )


@requires_sqlcipher
def test_cli_encrypt_decrypt_roundtrip(tmp_path: Path):
    """memoirs db encrypt → memoirs db decrypt must preserve every row."""
    plain_path = tmp_path / "plain.sqlite"
    enc_path = tmp_path / "enc.sqlite"
    decrypted_path = tmp_path / "back.sqlite"

    # Create + populate a plain DB through MemoirsDB (no env var).
    db = MemoirsDB(plain_path)
    try:
        _seed_basic(db)
        # Add a few extra rows to make the round-trip nontrivial.
        for i in range(5):
            db.conn.execute(
                "INSERT INTO sources (uri, kind, name, created_at, updated_at) "
                "VALUES (?, 'test', ?, '2026-01-01', '2026-01-01')",
                (f"test://r{i}", f"row-{i}"),
            )
        db.conn.commit()
        original = _row_count(db.conn)
    finally:
        db.close()

    # Encrypt via CLI
    res = _run_cli(
        "--db", str(plain_path),
        "db", "encrypt",
        "--key", "pass-1",
        "--out", str(enc_path),
    )
    assert res.returncode == 0, f"encrypt failed: {res.stderr}"
    assert enc_path.exists()
    assert is_encrypted(enc_path)

    # Decrypt via CLI
    res = _run_cli(
        "--db", str(enc_path),
        "db", "decrypt",
        "--key", "pass-1",
        "--out", str(decrypted_path),
    )
    assert res.returncode == 0, f"decrypt failed: {res.stderr}"
    assert decrypted_path.exists()
    assert not is_encrypted(decrypted_path)

    # Verify row count is preserved end-to-end.
    db2 = MemoirsDB(decrypted_path, auto_migrate=False)
    try:
        assert _row_count(db2.conn) == original
    finally:
        db2.close()


# ---------------------------------------------------------------------------
# CLI rekey
# ---------------------------------------------------------------------------


@requires_sqlcipher
def test_cli_rekey_changes_passphrase(tmp_path: Path):
    """After rekey, the old key must fail and the new key must succeed."""
    plain_path = tmp_path / "plain.sqlite"
    enc_path = tmp_path / "enc.sqlite"

    db = MemoirsDB(plain_path)
    try:
        _seed_basic(db)
    finally:
        db.close()

    # Encrypt with old-key
    res = _run_cli("--db", str(plain_path), "db", "encrypt",
                   "--key", "old-key", "--out", str(enc_path))
    assert res.returncode == 0, res.stderr

    # Rekey old → new
    res = _run_cli("--db", str(enc_path), "db", "rekey",
                   "--old", "old-key", "--new", "new-key")
    assert res.returncode == 0, f"rekey failed: {res.stderr}"

    # Old key must now fail to read
    import sqlcipher3.dbapi2 as _sql
    conn = _sql.connect(str(enc_path))
    try:
        conn.execute("PRAGMA key = 'old-key'")
        with pytest.raises(Exception):
            conn.execute("SELECT count(*) FROM sources").fetchone()
    finally:
        conn.close()

    # New key must succeed
    conn = _sql.connect(str(enc_path))
    try:
        conn.execute("PRAGMA key = 'new-key'")
        # Should not raise
        rows = conn.execute("SELECT count(*) FROM sources").fetchone()
        assert rows[0] >= 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Guardrail: env var set, binding absent → ImportError
# ---------------------------------------------------------------------------


def test_missing_sqlcipher_raises_importerror(tmp_path: Path, monkeypatch):
    """When MEMOIRS_ENCRYPT_KEY is set but sqlcipher3 is unavailable, opening
    the DB must NOT silently fall back to plaintext. We simulate this by
    monkeypatching memoirs.db._sqlcipher to None."""
    import memoirs.db as _dbmod

    monkeypatch.setattr(_dbmod, "_sqlcipher", None)
    monkeypatch.setenv("MEMOIRS_ENCRYPT_KEY", "any-key")

    with pytest.raises(ImportError, match="sqlcipher3"):
        MemoirsDB(tmp_path / "should-fail.sqlite")
