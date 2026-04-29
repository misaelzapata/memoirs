from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from .db import MemoirsDB
from .ingesters.importers import SUPPORTED_SUFFIXES, file_fingerprint, load_conversations


Reporter = Callable[[str], None]

log = logging.getLogger("memoirs.watch")


def _is_watchdog_available() -> bool:
    try:
        import watchdog  # noqa: F401
        import watchdog.events  # noqa: F401
        import watchdog.observers  # noqa: F401
    except ImportError:
        return False
    return True


def iter_targets(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    if not path.exists():
        return []
    targets: list[Path] = []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        if ".memoirs" in item.parts:
            continue
        if item.suffix.lower() in SUPPORTED_SUFFIXES:
            targets.append(item)
    return sorted(targets)


def ingest_path(db: MemoirsDB, path: Path, *, reporter: Reporter = print) -> tuple[int, int]:
    path = path.resolve()
    mtime_ns, size_bytes, hash_value = file_fingerprint(path)
    run_id = db.begin_import_run(
        str(path),
        importer=path.suffix.lower().lstrip(".") or "file",
        file_mtime_ns=mtime_ns,
        file_size=size_bytes,
        hash_value=hash_value,
    )
    # Snapshot active message count BEFORE saving so we can report the delta
    # (the watcher re-ingests the whole file every time it changes; only the
    # newly-appended messages actually become INSERTs thanks to UNIQUE indexes).
    before = db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    try:
        conversations = load_conversations(path)
        if not conversations:
            db.finish_import_run(run_id, status="skipped", conversation_count=0, message_count=0)
            return 0, 0
        conversation_count, message_count = db.save_conversations(
            conversations,
            source_name=path.name,
            source_kind=conversations[0].source_kind,
            source_uri=str(path),
            hash_value=hash_value,
            mtime_ns=mtime_ns,
            size_bytes=size_bytes,
        )
    except Exception as exc:
        db.finish_import_run(run_id, status="failed", error=str(exc))
        raise
    after = db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    delta = after - before
    db.finish_import_run(
        run_id,
        status="completed",
        conversation_count=conversation_count,
        message_count=message_count,
    )
    # P0-4: drop a `messages_ingested` event so downstream consumers (sleep
    # scheduler, extract daemon) notice the new corpus. We use ``delta`` (the
    # NEW message count, not the file total) so re-ingesting an unchanged
    # file is a no-op for the queue. Best-effort — a failure here must not
    # break ingestion.
    if delta > 0:
        try:
            from .engine.event_queue import enqueue_messages_ingested

            source_id_row = db.conn.execute(
                "SELECT id FROM sources WHERE uri = ?", (str(path),)
            ).fetchone()
            source_id = int(source_id_row["id"]) if source_id_row else None
            conv_id = conversations[0].external_id if conversations else None
            enqueue_messages_ingested(
                db,
                conversation_id=conv_id,
                source_id=source_id,
                message_count=int(delta),
                extra={
                    "importer": conversations[0].source_kind if conversations else None,
                    "path": str(path),
                },
            )
        except Exception:  # noqa: BLE001 — telemetry must never block ingest
            log.exception("event_queue enqueue failed for %s", path)
    if delta > 0:
        reporter(f"ingested {path.name}: +{delta} new messages (file total: {message_count})")
    else:
        # File touched but no new messages — typical when watcher polls during a
        # write that hasn't appended yet, or when nothing changed semantically.
        log.debug("touched %s: no new messages (total: %d)", path.name, message_count)
    return conversation_count, message_count


def scan_once(db: MemoirsDB, path: Path, *, reporter: Reporter = print) -> tuple[int, int]:
    total_conversations = 0
    total_messages = 0
    failed = 0
    targets = iter_targets(path)
    if not targets:
        reporter(f"no supported files found at {path}")
        return 0, 0
    for target in targets:
        try:
            conversation_count, message_count = ingest_path(db, target, reporter=reporter)
        except Exception as exc:
            failed += 1
            reporter(f"skip {target.name}: {exc}")
            continue
        total_conversations += conversation_count
        total_messages += message_count
    if failed:
        reporter(f"scan complete: {total_conversations} conversations, {total_messages} messages, {failed} files skipped")
    return total_conversations, total_messages


def watch_path(
    db: MemoirsDB,
    path: Path,
    *,
    interval: float = 2.0,
    once: bool = False,
    realtime: bool = False,
    reporter: Reporter = print,
) -> None:
    """Watch a file or folder and ingest on change.

    Default mode: polling every `interval` seconds (zero deps).
    `realtime=True` uses inotify via `watchdog` if installed; falls back to
    polling otherwise.
    """
    if realtime and not once:
        if _is_watchdog_available():
            _watch_with_watchdog(db, path, reporter=reporter)
            return
        reporter("watchdog not installed; falling back to polling. Install with: pip install -e '.[realtime]'")

    seen: dict[Path, tuple[int, int]] = {}
    reporter(f"watching {path.resolve()} every {interval:g}s (polling)")
    while True:
        targets = iter_targets(path)
        for target in targets:
            try:
                stat = target.stat()
            except FileNotFoundError:
                continue
            fingerprint = (stat.st_mtime_ns, stat.st_size)
            if seen.get(target) == fingerprint:
                continue
            seen[target] = fingerprint
            try:
                ingest_path(db, target, reporter=reporter)
            except Exception as exc:
                reporter(f"ingest failed for {target}: {exc}")
        if once:
            return
        time.sleep(interval)


def _watch_with_watchdog(db: MemoirsDB, path: Path, *, reporter: Reporter) -> None:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    target_root = path.resolve()
    reporter(f"watching {target_root} (watchdog/inotify, real-time)")

    class _Debouncer:
        def __init__(self, delay: float = 1.5) -> None:
            self.delay = delay
            self._timers: dict[str, threading.Timer] = {}
            self._lock = threading.Lock()

        def schedule(self, key: str, fn: Callable[[], None]) -> None:
            with self._lock:
                old = self._timers.pop(key, None)
                if old:
                    old.cancel()
                t = threading.Timer(self.delay, fn)
                t.daemon = True
                self._timers[key] = t
                t.start()

    debouncer = _Debouncer(delay=1.5)

    def _do_ingest(p: Path) -> None:
        try:
            ingest_path(db, p, reporter=reporter)
        except Exception as exc:
            reporter(f"ingest failed for {p}: {exc}")

    class Handler(FileSystemEventHandler):
        def _consider(self, event: FileSystemEvent) -> None:
            if event.is_directory:
                return
            p = Path(event.src_path)
            if p.suffix.lower() not in SUPPORTED_SUFFIXES and p.name != "state.vscdb":
                return
            if ".memoirs" in p.parts:
                return
            debouncer.schedule(str(p), lambda: _do_ingest(p))

        def on_modified(self, event: FileSystemEvent) -> None:
            self._consider(event)

        def on_created(self, event: FileSystemEvent) -> None:
            self._consider(event)

    # Initial sweep so anything written while we were down is captured.
    scan_once(db, path, reporter=reporter)

    observer = Observer()
    if target_root.is_file():
        observer.schedule(Handler(), str(target_root.parent), recursive=False)
    else:
        observer.schedule(Handler(), str(target_root), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        reporter("stopping watcher")
    finally:
        observer.stop()
        observer.join()
