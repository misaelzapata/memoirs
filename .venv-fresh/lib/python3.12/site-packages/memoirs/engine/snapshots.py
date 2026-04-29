"""Point-in-Time snapshots — create, list, diff, restore.

A snapshot is a full SQLite copy of the live DB at a moment in time. We
use ``VACUUM INTO`` so the operation is atomic, defragments along the
way, and works on an open DB without requiring a write lock on the live
file.

Why not lean only on bi-temporal queries?
-----------------------------------------
Memoirs already supports time-travel via ``valid_from``/``valid_to`` and
``as_of=t`` retrieval. That covers "what did the engine know at time t"
for **memorias**. Snapshots cover three things bi-temporal can't:

- **DELETE / VACUUM** — when we hard-delete or compact, bi-temporal
  history is gone. A snapshot keeps the file as it was.
- **Schema drift** — a snapshot is the entire DB including future
  migrations, conflicts, candidates, conversation events.
- **Restore** — bi-temporal can show old state but cannot rewind the
  live DB. Snapshots can.

Configuration
-------------
- ``MEMOIRS_SNAPSHOT_DIR`` — directory for snapshot files. Default:
  ``<db-parent>/snapshots``.
- ``MEMOIRS_AUTO_SNAPSHOT`` — when set to ``daily``, ``hourly``, or a
  number-of-seconds string, the maintenance job creates a snapshot if
  the most recent one is older than that interval. Default: off.
- ``MEMOIRS_SNAPSHOT_KEEP`` — keep N most recent snapshots and prune
  older ones. Default: 10.

API
---
- :func:`create` — atomic ``VACUUM INTO`` to a new snapshot file.
- :func:`list_snapshots` — directory listing with mtime + memory count.
- :func:`diff` — counts of added / archived / changed memorias between
  two snapshots (or snapshot + live DB).
- :func:`restore` — copy a snapshot back over the live DB. Always
  creates a safety snapshot of the current live state first.
- :func:`maybe_auto_snapshot` — call from maintenance cycle; respects
  ``MEMOIRS_AUTO_SNAPSHOT`` cadence.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

log = logging.getLogger("memoirs.snapshots")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _snapshot_dir(db_path: Path) -> Path:
    raw = os.environ.get("MEMOIRS_SNAPSHOT_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path(db_path).parent / "snapshots"


def _keep_count() -> int:
    try:
        return max(1, int(os.environ.get("MEMOIRS_SNAPSHOT_KEEP", "10")))
    except ValueError:
        return 10


_AUTO_INTERVALS = {
    "off": None,
    "hourly": 3600,
    "daily": 86400,
    "weekly": 7 * 86400,
}


def _auto_interval_seconds() -> int | None:
    raw = (os.environ.get("MEMOIRS_AUTO_SNAPSHOT") or "off").strip().lower()
    if raw in _AUTO_INTERVALS:
        return _AUTO_INTERVALS[raw]
    # Numeric — seconds.
    try:
        n = int(raw)
        return max(60, n)  # floor at 1 minute
    except ValueError:
        log.warning("MEMOIRS_AUTO_SNAPSHOT=%r unrecognized; treating as off", raw)
        return None


# ---------------------------------------------------------------------------
# Snapshot file naming
# ---------------------------------------------------------------------------

_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-]+")


def _safe_name(name: str | None) -> str:
    if not name:
        return "snap"
    return _NAME_RE.sub("_", name)[:48] or "snap"


def _new_filename(name: str | None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}__{_safe_name(name)}.sqlite"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class SnapshotInfo:
    path: Path
    name: str
    created_at: str
    size_bytes: int
    memory_count: int


def create(db_path: Path | str, *, name: str | None = None) -> SnapshotInfo:
    """Atomic ``VACUUM INTO`` snapshot. Returns the new snapshot file's metadata."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"db not found: {db_path}")
    out_dir = _snapshot_dir(db_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _new_filename(name)
    # Use a fresh connection so we don't hold any txn on the caller's conn.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(f"VACUUM INTO ?", (str(out_path),))
    finally:
        conn.close()
    info = _info_from_path(out_path)
    log.info(
        "snapshots.create: name=%s path=%s size=%d memories=%d",
        info.name, info.path, info.size_bytes, info.memory_count,
    )
    _prune(out_dir, keep=_keep_count())
    return info


def list_snapshots(db_path: Path | str) -> list[SnapshotInfo]:
    out_dir = _snapshot_dir(Path(db_path))
    if not out_dir.exists():
        return []
    out: list[SnapshotInfo] = []
    for p in sorted(out_dir.glob("*.sqlite")):
        try:
            out.append(_info_from_path(p))
        except Exception as e:
            log.warning("snapshots.list_snapshots: skipping %s (%s)", p, e)
    out.sort(key=lambda s: s.created_at, reverse=True)
    return out


def diff(a_path: Path | str, b_path: Path | str) -> dict:
    """Compare two snapshot DBs (or a snapshot and the live DB).

    Returns counts of memorias *added*, *removed*, *changed* (same id,
    different content_hash). Uses two read-only connections; no writes.
    """
    a = sqlite3.connect(f"file:{a_path}?mode=ro", uri=True)
    b = sqlite3.connect(f"file:{b_path}?mode=ro", uri=True)
    try:
        a.row_factory = sqlite3.Row
        b.row_factory = sqlite3.Row
        a_rows = {r["id"]: r["content_hash"] for r in a.execute(
            "SELECT id, content_hash FROM memories WHERE archived_at IS NULL")}
        b_rows = {r["id"]: r["content_hash"] for r in b.execute(
            "SELECT id, content_hash FROM memories WHERE archived_at IS NULL")}
    finally:
        a.close(); b.close()
    added = [k for k in b_rows if k not in a_rows]
    removed = [k for k in a_rows if k not in b_rows]
    changed = [k for k in a_rows if k in b_rows and a_rows[k] != b_rows[k]]
    return {
        "a_count": len(a_rows),
        "b_count": len(b_rows),
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def restore(snapshot_path: Path | str, db_path: Path | str) -> SnapshotInfo:
    """Copy a snapshot over the live DB. Always takes a safety snapshot
    of the current live state first so a wrong restore is recoverable.
    """
    snapshot_path = Path(snapshot_path)
    db_path = Path(db_path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"snapshot not found: {snapshot_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"live db not found: {db_path}")
    safety = create(db_path, name=f"pre-restore-{snapshot_path.stem}")
    log.info("snapshots.restore: safety snapshot at %s", safety.path)
    # Atomic rename via tmp file in same directory.
    tmp = db_path.with_suffix(db_path.suffix + ".restore.tmp")
    shutil.copy2(snapshot_path, tmp)
    os.replace(tmp, db_path)
    log.info("snapshots.restore: %s -> %s", snapshot_path, db_path)
    return _info_from_path(snapshot_path)


def maybe_auto_snapshot(db_path: Path | str) -> SnapshotInfo | None:
    """Create a snapshot if the last one is older than ``MEMOIRS_AUTO_SNAPSHOT``.

    Returns the new snapshot info, or ``None`` if not yet due / disabled.
    """
    interval = _auto_interval_seconds()
    if interval is None:
        return None
    snaps = list_snapshots(db_path)
    if snaps:
        latest_mtime = snaps[0].path.stat().st_mtime
        if (time.time() - latest_mtime) < interval:
            return None
    return create(db_path, name="auto")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _info_from_path(p: Path) -> SnapshotInfo:
    st = p.stat()
    created = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    # Try to count active memorias; fall back to -1 on any error.
    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE archived_at IS NULL"
            ).fetchone()[0]
        finally:
            conn.close()
    except Exception:
        n = -1
    name_part = p.stem.split("__", 1)
    name = name_part[1] if len(name_part) == 2 else p.stem
    return SnapshotInfo(
        path=p, name=name, created_at=created, size_bytes=st.st_size, memory_count=int(n),
    )


def _prune(out_dir: Path, *, keep: int) -> None:
    files = sorted(out_dir.glob("*.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in files[keep:]:
        try:
            stale.unlink()
            log.info("snapshots.prune: removed %s", stale)
        except Exception as e:
            log.warning("snapshots.prune: could not remove %s (%s)", stale, e)


__all__ = [
    "SnapshotInfo",
    "create",
    "list_snapshots",
    "diff",
    "restore",
    "maybe_auto_snapshot",
]
