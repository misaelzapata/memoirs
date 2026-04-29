"""Versioned schema migrations for memoirs.

Storage model
-------------
The schema version lives in ``PRAGMA user_version`` (a per-DB integer that
SQLite persists for free — no extra metadata table required).

Each migration is a numbered Python module in this package whose name starts
with a zero-padded integer, e.g. ``001_initial.py``, ``002_add_tags.py``. The
module MUST expose two callables:

    def up(conn: sqlite3.Connection) -> None: ...
    def down(conn: sqlite3.Connection) -> None: ...  # rollback

``up`` should be idempotent whenever feasible (use ``CREATE TABLE IF NOT
EXISTS`` / ``CREATE INDEX IF NOT EXISTS`` / ``ALTER TABLE`` guarded by
``PRAGMA table_info``). ``down`` may raise ``NotImplementedError`` for
migrations that intentionally cannot be reversed (e.g. data destruction).

Public API
----------
- :func:`discover_migrations` — returns the ordered list of migrations.
- :func:`current_version`     — reads ``PRAGMA user_version``.
- :func:`target_version`      — highest available migration version.
- :func:`run_pending_migrations` — applies every migration > current.
- :func:`migrate_to`          — apply or rollback to an exact target.
- :func:`rollback`            — undo the most recently applied migration.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, Iterable

log = logging.getLogger("memoirs.migrations")

# Filenames look like ``001_initial.py`` — leading digits are the version.
_FILENAME_RE = re.compile(r"^(\d+)_([a-zA-Z0-9_]+)$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    module_name: str
    up: Callable[[sqlite3.Connection], None]
    down: Callable[[sqlite3.Connection], None]

    @property
    def label(self) -> str:
        return f"{self.version:03d}_{self.name}"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_migrations() -> list[Migration]:
    """Scan this package for ``NNN_name.py`` modules and return them sorted by
    version. Duplicate version numbers raise ``RuntimeError``.
    """
    package = __name__
    found: dict[int, Migration] = {}
    for info in pkgutil.iter_modules(__path__):
        if info.ispkg:
            continue
        match = _FILENAME_RE.match(info.name)
        if not match:
            continue
        version = int(match.group(1))
        name = match.group(2)
        module = importlib.import_module(f"{package}.{info.name}")
        up = getattr(module, "up", None)
        down = getattr(module, "down", None)
        if not callable(up) or not callable(down):
            raise RuntimeError(
                f"migration {info.name} must define up(conn) and down(conn)"
            )
        if version in found:
            raise RuntimeError(
                f"duplicate migration version {version}: "
                f"{found[version].module_name} and {info.name}"
            )
        found[version] = Migration(
            version=version,
            name=name,
            module_name=info.name,
            up=up,
            down=down,
        )
    return [found[v] for v in sorted(found)]


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

def current_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _set_version(conn: sqlite3.Connection, version: int) -> None:
    # PRAGMA user_version does not accept parameters; the value is bounded by
    # ``int`` so direct interpolation is safe.
    conn.execute(f"PRAGMA user_version = {int(version)}")


def target_version(migrations: Iterable[Migration] | None = None) -> int:
    migs = list(migrations) if migrations is not None else discover_migrations()
    if not migs:
        return 0
    return migs[-1].version


# ---------------------------------------------------------------------------
# Apply / rollback
# ---------------------------------------------------------------------------

def run_pending_migrations(conn: sqlite3.Connection) -> list[int]:
    """Apply every migration whose version > ``PRAGMA user_version``.

    Returns the ordered list of versions applied (empty if already current).
    """
    migrations = discover_migrations()
    return _apply_forward(conn, migrations, target=target_version(migrations))


def migrate_to(conn: sqlite3.Connection, target: int) -> list[int]:
    """Apply or rollback migrations until ``user_version == target``.

    Returns the list of versions touched (positive = applied, negative =
    rolled back).
    """
    migrations = discover_migrations()
    available = {m.version for m in migrations}
    if target != 0 and target not in available:
        raise ValueError(
            f"unknown migration version {target} (available: "
            f"{sorted(available) or 'none'})"
        )
    current = current_version(conn)
    if target > current:
        return _apply_forward(conn, migrations, target=target)
    if target < current:
        return _apply_backward(conn, migrations, target=target)
    return []


def rollback(conn: sqlite3.Connection, steps: int = 1) -> list[int]:
    """Roll back the most recently applied migration(s).

    ``steps`` defaults to 1. Returns the list of versions that were rolled
    back (most recent first), as negative integers for symmetry with
    :func:`migrate_to`.
    """
    if steps <= 0:
        return []
    migrations = discover_migrations()
    current = current_version(conn)
    applied_descending = [m for m in reversed(migrations) if m.version <= current]
    if not applied_descending:
        return []
    target_index = min(steps, len(applied_descending))
    target_version_value = (
        applied_descending[target_index].version
        if target_index < len(applied_descending)
        else 0
    )
    return _apply_backward(conn, migrations, target=target_version_value)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _apply_forward(
    conn: sqlite3.Connection,
    migrations: list[Migration],
    *,
    target: int,
) -> list[int]:
    current = current_version(conn)
    applied: list[int] = []
    for mig in migrations:
        if mig.version <= current:
            continue
        if mig.version > target:
            break
        log.info("migrate up: %s", mig.label)
        try:
            mig.up(conn)
            _set_version(conn, mig.version)
            conn.commit()
        except Exception:
            conn.rollback()
            log.exception("migration %s failed during up()", mig.label)
            raise
        applied.append(mig.version)
    return applied


def _apply_backward(
    conn: sqlite3.Connection,
    migrations: list[Migration],
    *,
    target: int,
) -> list[int]:
    current = current_version(conn)
    rolled: list[int] = []
    # Walk migrations in descending order.
    for mig in reversed(migrations):
        if mig.version > current:
            continue
        if mig.version <= target:
            break
        log.info("migrate down: %s", mig.label)
        try:
            mig.down(conn)
            # After rolling back version N, user_version becomes N-1 (the
            # previous migration's version, or 0 if N is the baseline).
            previous = _previous_version(migrations, mig.version)
            _set_version(conn, previous)
            conn.commit()
        except Exception:
            conn.rollback()
            log.exception("migration %s failed during down()", mig.label)
            raise
        rolled.append(-mig.version)
    return rolled


def _previous_version(migrations: list[Migration], version: int) -> int:
    prev = 0
    for mig in migrations:
        if mig.version >= version:
            break
        prev = mig.version
    return prev
