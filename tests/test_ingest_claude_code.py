"""Tests for the Claude Code transcript parser."""
import json
from pathlib import Path

import pytest

from memoirs.ingesters.claude_code import (
    is_claude_code_path,
    load_claude_code_jsonl,
)
from memoirs.core.normalize import flatten_content


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with path.open("w") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


def test_is_claude_code_path_recognizes_jsonl_under_projects(tmp_path, monkeypatch):
    # Patch the global so tests don't depend on the real ~/.claude/projects
    fake_root = tmp_path / "claude_projects"
    fake_root.mkdir(parents=True)
    monkeypatch.setattr("memoirs.ingesters.claude_code.CLAUDE_CODE_PROJECTS", fake_root)
    real_jsonl = fake_root / "encoded-cwd" / "session.jsonl"
    real_jsonl.parent.mkdir(parents=True)
    real_jsonl.write_text("")
    assert is_claude_code_path(real_jsonl)


def test_is_claude_code_path_rejects_other_paths(tmp_path):
    other = tmp_path / "other.jsonl"
    other.write_text("")
    assert not is_claude_code_path(other)


def test_is_claude_code_path_rejects_non_jsonl(tmp_path, monkeypatch):
    fake_root = tmp_path / "claude_projects"
    fake_root.mkdir(parents=True)
    monkeypatch.setattr("memoirs.ingesters.claude_code.CLAUDE_CODE_PROJECTS", fake_root)
    f = fake_root / "x" / "session.json"
    f.parent.mkdir(parents=True)
    f.write_text("")
    assert not is_claude_code_path(f)


def test_load_returns_only_relevant_types(tmp_path):
    jsonl = tmp_path / "abc.jsonl"
    _write_jsonl(jsonl, [
        {"type": "queue-operation", "operation": "enqueue"},
        {"type": "progress", "step": 1},
        {"type": "user", "message": {"role": "user", "content": "hello"}, "uuid": "u1"},
        {"type": "assistant", "message": {"role": "assistant",
            "content": [{"type": "text", "text": "world"}]}, "uuid": "u2"},
        {"type": "file-history-snapshot", "x": 1},
    ])
    convs = load_claude_code_jsonl(jsonl)
    assert len(convs) == 1
    msgs = convs[0].messages
    # Only user + assistant with content survive
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[0].content == "hello"
    assert msgs[1].role == "assistant"
    assert msgs[1].content == "world"


def test_load_empty_returns_empty(tmp_path):
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("")
    assert load_claude_code_jsonl(jsonl) == []


def test_load_skips_invalid_lines(tmp_path):
    jsonl = tmp_path / "abc.jsonl"
    jsonl.write_text(
        '{"type":"user","message":{"role":"user","content":"ok"},"uuid":"a"}\n'
        'BROKEN_JSON_LINE\n'
        '{"type":"user","message":{"role":"user","content":"ok2"},"uuid":"b"}\n'
    )
    convs = load_claude_code_jsonl(jsonl)
    assert len(convs[0].messages) == 2


def test_load_captures_metadata(tmp_path):
    jsonl = tmp_path / "abc.jsonl"
    _write_jsonl(jsonl, [
        {"type": "user", "message": {"role": "user", "content": "x"},
         "uuid": "u1", "cwd": "/home/x", "gitBranch": "main", "version": "2.1"},
    ])
    convs = load_claude_code_jsonl(jsonl)
    md = convs[0].messages[0].metadata
    assert md["cwd"] == "/home/x"
    assert md["gitBranch"] == "main"
    assert md["version"] == "2.1"
