"""Source-specific ingesters + the dispatcher.

Each module knows how to turn one kind of file into RawConversation rows:
  - claude_code.py    — ~/.claude/projects/**/*.jsonl
  - claude_export.py  — official Claude.ai data export (zip or extracted dir)
  - cursor.py         — ~/.config/Cursor/**/state.vscdb
  - importers.py      — generic .md / .json / .jsonl / .zip dispatcher
"""
from .claude_code import ingest_claude_code_jsonl
from .claude_export import ImportStats, import_claude_export
from .cursor import ingest_cursor_state
from .importers import (
    SUPPORTED_SUFFIXES,
    ImportErrorWithPath,
    file_fingerprint,
    ingest_file_with_events,
    load_conversations,
)

__all__ = [
    "SUPPORTED_SUFFIXES",
    "ImportErrorWithPath",
    "ImportStats",
    "file_fingerprint",
    "ingest_claude_code_jsonl",
    "ingest_cursor_state",
    "ingest_file_with_events",
    "import_claude_export",
    "load_conversations",
]
