"""Tests for the SQLite PRAGMA tuning applied at connection time.

These verify that the latency-oriented PRAGMAs (mmap_size, cache_size,
temp_store) are actually set on the live connection, that the env-var
overrides are honored, and that the helper is idempotent.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memoirs.db import MemoirsDB, _apply_pragmas


def _pragma_int(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute(f"PRAGMA {name}").fetchone()
    # Row factory may or may not be set; access by index.
    return int(row[0])


def _pragma_str(conn: sqlite3.Connection, name: str) -> str:
    row = conn.execute(f"PRAGMA {name}").fetchone()
    return str(row[0])


def test_default_mmap_size_applied(tmp_path: Path):
    db = MemoirsDB(tmp_path / "m.sqlite")
    try:
        # Default 256 MiB == 268_435_456 bytes. Some platforms cap mmap_size
        # at a smaller hard limit; accept anything > 0 (i.e. not the default
        # 0) and equal to what we asked for if the platform allowed it.
        mmap = _pragma_int(db.conn, "mmap_size")
        assert mmap > 0, "mmap_size should be enabled (non-zero) by default"
        assert mmap == 256 * 1024 * 1024
    finally:
        db.close()


def test_default_cache_size_negative(tmp_path: Path):
    db = MemoirsDB(tmp_path / "m.sqlite")
    try:
        cache = _pragma_int(db.conn, "cache_size")
        # Negative form == KiB. Default 64 MiB → -65536.
        assert cache == -65536
    finally:
        db.close()


def test_default_temp_store_memory(tmp_path: Path):
    db = MemoirsDB(tmp_path / "m.sqlite")
    try:
        # 0 = DEFAULT (file), 1 = FILE, 2 = MEMORY
        ts = _pragma_int(db.conn, "temp_store")
        assert ts == 2
    finally:
        db.close()


def test_synchronous_normal_under_wal(tmp_path: Path):
    db = MemoirsDB(tmp_path / "m.sqlite")
    try:
        sync = _pragma_int(db.conn, "synchronous")
        # 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
        assert sync == 1
        mode = _pragma_str(db.conn, "journal_mode")
        assert mode.lower() == "wal"
    finally:
        db.close()


def test_mmap_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMOIRS_SQLITE_MMAP_MB", "128")
    db = MemoirsDB(tmp_path / "m.sqlite")
    try:
        mmap = _pragma_int(db.conn, "mmap_size")
        assert mmap == 128 * 1024 * 1024
    finally:
        db.close()


def test_cache_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMOIRS_SQLITE_CACHE_MB", "16")
    db = MemoirsDB(tmp_path / "m.sqlite")
    try:
        cache = _pragma_int(db.conn, "cache_size")
        assert cache == -(16 * 1024)  # -16384 KiB
    finally:
        db.close()


def test_invalid_env_falls_back_to_default(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMOIRS_SQLITE_MMAP_MB", "not-a-number")
    db = MemoirsDB(tmp_path / "m.sqlite")
    try:
        mmap = _pragma_int(db.conn, "mmap_size")
        assert mmap == 256 * 1024 * 1024  # default
    finally:
        db.close()


def test_apply_pragmas_idempotent(tmp_path: Path):
    """Calling _apply_pragmas multiple times should not change settings or error."""
    raw = sqlite3.connect(tmp_path / "raw.sqlite")
    try:
        _apply_pragmas(raw)
        first_mmap = _pragma_int(raw, "mmap_size")
        first_cache = _pragma_int(raw, "cache_size")
        first_ts = _pragma_int(raw, "temp_store")
        first_sync = _pragma_int(raw, "synchronous")

        # Re-apply: should be a no-op for state, and must not raise.
        _apply_pragmas(raw)
        _apply_pragmas(raw)

        assert _pragma_int(raw, "mmap_size") == first_mmap
        assert _pragma_int(raw, "cache_size") == first_cache
        assert _pragma_int(raw, "temp_store") == first_ts
        assert _pragma_int(raw, "synchronous") == first_sync
    finally:
        raw.close()


def test_apply_pragmas_on_raw_connection(tmp_path: Path):
    """Helper works on a vanilla sqlite3.connect() connection."""
    raw = sqlite3.connect(tmp_path / "raw.sqlite")
    try:
        # Pre-condition: defaults — mmap_size=0, temp_store=0
        assert _pragma_int(raw, "mmap_size") == 0
        _apply_pragmas(raw)
        assert _pragma_int(raw, "mmap_size") == 256 * 1024 * 1024
        assert _pragma_int(raw, "temp_store") == 2
        assert _pragma_int(raw, "cache_size") == -65536
    finally:
        raw.close()
