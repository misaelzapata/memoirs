"""Tests for core/normalize.flatten_content."""
from memoirs.core.normalize import flatten_content, normalize_role


def test_flatten_string_passthrough():
    assert flatten_content("hello") == "hello"


def test_flatten_text_block():
    blocks = [{"type": "text", "text": "hi"}]
    assert flatten_content(blocks) == "hi"


def test_flatten_thinking_block():
    blocks = [{"type": "thinking", "thinking": "deep"}]
    assert flatten_content(blocks) == "[thinking]\ndeep"


def test_flatten_tool_use_truncates():
    big_input = {"x": "a" * 5000}
    blocks = [{"type": "tool_use", "name": "Bash", "input": big_input}]
    out = flatten_content(blocks)
    assert "[tool_use:Bash]" in out
    assert len(out) < 1500  # truncated to TOOL_USE_PREVIEW_LEN


def test_flatten_tool_result_recursive():
    blocks = [{"type": "tool_result", "content": [{"type": "text", "text": "out"}]}]
    assert "[tool_result]" in flatten_content(blocks)
    assert "out" in flatten_content(blocks)


def test_flatten_mixed_blocks_concat():
    blocks = [
        {"type": "text", "text": "first"},
        {"type": "thinking", "thinking": "thoughts"},
    ]
    out = flatten_content(blocks)
    assert "first" in out and "[thinking]" in out


def test_flatten_unknown_block_type_kept():
    blocks = [{"type": "weird", "foo": "bar"}]
    out = flatten_content(blocks)
    assert "[weird]" in out


def test_flatten_empty_returns_empty():
    assert flatten_content(None) == ""
    assert flatten_content([]) == ""


def test_normalize_role_aliases():
    assert normalize_role("human") == "user"
    assert normalize_role("AI") == "assistant"
    assert normalize_role("BOT") == "assistant"
    assert normalize_role("user") == "user"
    assert normalize_role(None) == "unknown"
