"""Targeted coverage tests for memoirs/core/normalize.py.

Most of the file is single-call helpers used by ingesters. We exercise the
content-flatten branches that are otherwise only hit by full integration runs.
"""
from __future__ import annotations

import json

from memoirs.core.normalize import (
    flatten_content,
    normalize_role,
    normalize_timestamp,
)


# ---------------------------------------------------------------------------
# flatten_content — all the Anthropic block types
# ---------------------------------------------------------------------------


def test_flatten_content_str_passes_through():
    assert flatten_content("hello") == "hello"


def test_flatten_content_none_returns_empty():
    assert flatten_content(None) == ""


def test_flatten_content_text_block():
    out = flatten_content([{"type": "text", "text": "hello world"}])
    assert "hello world" in out


def test_flatten_content_thinking_block():
    out = flatten_content([{"type": "thinking", "thinking": "internal thoughts"}])
    assert "[thinking]" in out
    assert "internal thoughts" in out


def test_flatten_content_tool_use_block():
    out = flatten_content([
        {"type": "tool_use", "name": "Read", "input": {"path": "/tmp/x"}},
    ])
    assert "[tool_use:Read]" in out
    assert "/tmp/x" in out


def test_flatten_content_tool_result_recurses():
    out = flatten_content([{
        "type": "tool_result",
        "content": [{"type": "text", "text": "stdout: hi"}],
    }])
    assert "[tool_result]" in out
    assert "stdout: hi" in out


def test_flatten_content_unknown_block_preserves_type_label():
    out = flatten_content([{"type": "weird", "foo": 1}])
    assert "[weird]" in out


def test_flatten_content_non_dict_block_str_fallback():
    out = flatten_content(["raw string in list"])
    assert "raw string in list" in out


def test_flatten_content_dict_with_text_field():
    assert flatten_content({"text": "direct"}) == "direct"


def test_flatten_content_dict_with_nested_content():
    assert "nested" in flatten_content({"content": "nested"})


def test_flatten_content_unknown_dict_serialized():
    out = flatten_content({"foo": 1, "bar": [2, 3]})
    parsed = json.loads(out)
    assert parsed["foo"] == 1


def test_flatten_content_other_type_str_fallback():
    assert flatten_content(42) == "42"


# ---------------------------------------------------------------------------
# normalize_role
# ---------------------------------------------------------------------------


def test_normalize_role_aliases():
    assert normalize_role("human") == "user"
    assert normalize_role("AI") == "assistant"
    assert normalize_role("bot") == "assistant"


def test_normalize_role_passthrough():
    assert normalize_role("system") == "system"
    assert normalize_role(None) == "unknown"


# ---------------------------------------------------------------------------
# normalize_timestamp
# ---------------------------------------------------------------------------


def test_normalize_timestamp_none_and_empty():
    assert normalize_timestamp(None) is None
    assert normalize_timestamp("") is None


def test_normalize_timestamp_int_to_iso():
    out = normalize_timestamp(1714233600)
    # Just verify it's an ISO-formatted UTC string with timezone.
    assert "T" in out
    assert "+00:00" in out or "Z" in out


def test_normalize_timestamp_str_passthrough():
    out = normalize_timestamp("2026-04-27T00:00:00Z")
    assert out == "2026-04-27T00:00:00Z"
