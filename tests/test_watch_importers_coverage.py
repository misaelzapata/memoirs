"""Targeted coverage tests for memoirs/watch.py + memoirs/ingesters/importers.py.

Both modules are the high-traffic ingestion glue but most of their lines
(pollers, file-system observers) are exercised by integration tests at most.
We can still cover the pure helpers and the happy paths cheaply.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memoirs import watch
from memoirs.ingesters import importers


# ---------------------------------------------------------------------------
# importers — pure helpers
# ---------------------------------------------------------------------------


def test_file_fingerprint_returns_tuple(tmp_path: Path):
    p = tmp_path / "x.md"
    p.write_text("hello")
    mtime_ns, size, h = importers.file_fingerprint(p)
    assert mtime_ns > 0
    assert size == 5
    assert isinstance(h, str) and len(h) > 16


def test_load_conversations_unsupported_suffix(tmp_path: Path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,c")
    with pytest.raises(importers.ImportErrorWithPath):
        importers.load_conversations(p)


def test_load_markdown_with_prose(tmp_path: Path):
    p = tmp_path / "note.md"
    p.write_text("# Heading\n\nFree-form prose without JSONL.")
    convs = importers.load_markdown(p)
    assert len(convs) == 1
    assert convs[0].source_kind == "markdown"
    assert any(m.role == "document" for m in convs[0].messages)


def test_load_markdown_with_embedded_jsonl(tmp_path: Path):
    p = tmp_path / "log.md"
    p.write_text(
        '\n'.join([
            "Some prose",
            json.dumps({"role": "user", "content": "hi"}),
            json.dumps({"role": "assistant", "content": "hello"}),
        ])
    )
    convs = importers.load_markdown(p)
    assert len(convs) == 1
    msgs = convs[0].messages
    # 1 prose document + 2 jsonl messages
    assert any(m.role == "document" for m in msgs)
    assert any(m.role == "user" for m in msgs)
    assert any(m.role == "assistant" for m in msgs)


def test_load_jsonl_file_strict(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"role": "user", "content": "hi"}))
    convs = importers.load_jsonl_file(p)
    assert convs[0].source_kind == "jsonl"
    assert convs[0].messages[0].role == "user"


def test_load_jsonl_file_rejects_prose(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    p.write_text("not actually jsonl")
    with pytest.raises(importers.ImportErrorWithPath):
        importers.load_jsonl_file(p)


def test_parse_jsonl_lines_drops_invalid_payloads():
    lines = [
        json.dumps({"role": "user", "content": "ok"}),
        "{ not json",
        json.dumps({"no_role": "skip me"}),
    ]
    msgs, prose = importers.parse_jsonl_lines(lines, source="jsonl")
    assert len(msgs) == 1
    assert "{ not json" in prose
    assert "no_role" in prose


# ---------------------------------------------------------------------------
# watch.py — iter_targets
# ---------------------------------------------------------------------------


def test_iter_targets_file_supported(tmp_path: Path):
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"role": "user", "content": "hi"}))
    out = watch.iter_targets(p)
    assert out == [p.resolve()]


def test_iter_targets_file_unsupported(tmp_path: Path):
    p = tmp_path / "log.csv"
    p.write_text("a,b")
    assert watch.iter_targets(p) == []


def test_iter_targets_missing_path(tmp_path: Path):
    assert watch.iter_targets(tmp_path / "nonexistent") == []


def test_iter_targets_directory_filters_unsupported(tmp_path: Path):
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "b.csv").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.json").write_text("[]")
    out = watch.iter_targets(tmp_path)
    names = {p.name for p in out}
    assert "a.md" in names
    assert "c.json" in names
    assert "b.csv" not in names


def test_iter_targets_excludes_memoirs_internal(tmp_path: Path):
    """Files under .memoirs/ should never be picked up."""
    internal = tmp_path / ".memoirs"
    internal.mkdir()
    (internal / "x.md").write_text("nope")
    (tmp_path / "ok.md").write_text("yes")
    out = watch.iter_targets(tmp_path)
    names = {p.name for p in out}
    assert "ok.md" in names
    assert "x.md" not in names


# ---------------------------------------------------------------------------
# watch.py — ingest_path + scan_once on real DB
# ---------------------------------------------------------------------------


def test_ingest_path_jsonl_round_trip(tmp_db, tmp_path: Path):
    p = tmp_path / "log.jsonl"
    p.write_text(json.dumps({"role": "user", "content": "hi there"}))
    convs, msgs = watch.ingest_path(tmp_db, p, reporter=lambda _m: None)
    assert convs == 1
    assert msgs == 1
    # Re-ingest is idempotent (no extra rows).
    convs2, msgs2 = watch.ingest_path(tmp_db, p, reporter=lambda _m: None)
    assert convs2 == 1


def test_scan_once_empty_directory(tmp_db, tmp_path: Path, capsys):
    captured = []

    def reporter(msg: str) -> None:
        captured.append(msg)

    convs, msgs = watch.scan_once(tmp_db, tmp_path, reporter=reporter)
    assert convs == 0
    assert msgs == 0
    assert captured and "no supported files" in captured[0]


# ---------------------------------------------------------------------------
# importers — JSON dispatch (load_json_file branches)
# ---------------------------------------------------------------------------


def test_load_json_file_message_list(tmp_path: Path):
    p = tmp_path / "msgs.json"
    p.write_text(json.dumps([
        {"role": "user", "content": "alpha"},
        {"role": "assistant", "content": "beta"},
    ]))
    convs = importers.load_json_file(p)
    assert len(convs) == 1
    assert convs[0].source_kind == "json"
    assert len(convs[0].messages) == 2


def test_load_json_file_normalized_format(tmp_path: Path):
    p = tmp_path / "norm.json"
    p.write_text(json.dumps({
        "id": "conv-42",
        "title": "Hello",
        "messages": [{"role": "user", "content": "ok"}],
    }))
    convs = importers.load_json_file(p)
    assert len(convs) == 1
    assert convs[0].title == "Hello"


def test_load_json_file_unknown_returns_empty(tmp_path: Path):
    p = tmp_path / "weird.json"
    p.write_text(json.dumps({"some_other_shape": True}))
    assert importers.load_json_file(p) == []


def test_load_json_file_conversations_wrapper(tmp_path: Path):
    p = tmp_path / "wrapped.json"
    p.write_text(json.dumps({
        "conversations": [
            {"id": "c1", "messages": [{"role": "user", "content": "hi"}]},
            {"id": "c2", "messages": [{"role": "user", "content": "hi2"}]},
        ]
    }))
    convs = importers.load_json_file(p)
    assert len(convs) == 2


# ---------------------------------------------------------------------------
# importers — ChatGPT export
# ---------------------------------------------------------------------------


def test_is_chatgpt_conversation_detection():
    assert importers.is_chatgpt_conversation({"mapping": {"x": {}}}) is True
    assert importers.is_chatgpt_conversation({"mapping": "not-a-dict"}) is False
    assert importers.is_chatgpt_conversation([]) is False


def test_chatgpt_conversation_builds_messages():
    payload = {
        "id": "conv1",
        "title": "Test",
        "create_time": 1714233600,
        "mapping": {
            "n1": {"message": {
                "id": "m1", "author": {"role": "user"}, "create_time": 1,
                "content": {"parts": ["hello"]},
            }},
            "n2": {"message": {
                "id": "m2", "author": {"role": "assistant"}, "create_time": 2,
                "content": {"parts": ["hi back"]},
            }},
        },
    }
    out = importers.chatgpt_conversation(payload, "src://x")
    assert out.title == "Test"
    assert len(out.messages) == 2


def test_extract_chatgpt_content_parts():
    out = importers.extract_chatgpt_content({"parts": ["a", "b", ""]})
    assert "a" in out
    assert "b" in out


def test_extract_chatgpt_content_dict_text():
    out = importers.extract_chatgpt_content({"text": "hello"})
    assert "hello" in out


def test_normalize_content_handles_str_and_none():
    assert importers.normalize_content("hi") == "hi"
    assert importers.normalize_content(None) == ""


def test_messages_from_list_skips_empty_content():
    out = importers.messages_from_list([
        {"role": "user", "content": ""},
        {"role": "user", "content": "real content"},
        "not a dict",
    ])
    assert len(out) == 1
    assert out[0].content == "real content"
