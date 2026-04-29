"""Targeted coverage tests for memoirs/mcp/server.py.

The server is hand-rolled JSON-RPC 2.0. We exercise the dispatch / error /
notification branches without spinning up real stdin/stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memoirs.mcp.server import McpServer, MethodNotFound, _tool_result


@pytest.fixture
def server(tmp_path: Path):
    s = McpServer(tmp_path / "memoirs.sqlite")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# initialize / ping / tools/list
# ---------------------------------------------------------------------------


def test_dispatch_initialize_returns_capabilities(server):
    out = server._dispatch("initialize", {
        "protocolVersion": "2025-06-18",
        "clientInfo": {"name": "test-client", "version": "0.0.1"},
    })
    assert out["protocolVersion"] == "2025-06-18"
    assert "capabilities" in out
    assert out["serverInfo"]["name"] == "memoirs"


def test_dispatch_ping(server):
    assert server._dispatch("ping", {}) == {}


def test_dispatch_tools_list(server):
    out = server._dispatch("tools/list", {})
    assert "tools" in out
    names = {t["name"] for t in out["tools"]}
    assert "mcp_status" in names


def test_dispatch_unknown_method_raises(server):
    with pytest.raises(MethodNotFound):
        server._dispatch("not_a_method", {})


def test_call_tool_via_dispatch(server):
    out = server._dispatch("tools/call", {"name": "mcp_status", "arguments": {}})
    # MCP tool result envelope.
    assert "content" in out
    assert "structuredContent" in out
    assert "sources" in out["structuredContent"]


def test_call_tool_invalid_arguments_raises(server):
    with pytest.raises(ValueError):
        server._dispatch("tools/call", {"name": "mcp_status", "arguments": ["not", "a", "dict"]})


def test_dispatch_tools_call_requires_object_params(server):
    with pytest.raises(ValueError):
        server._dispatch("tools/call", "not-a-dict")


# ---------------------------------------------------------------------------
# _handle_single — JSON-RPC envelope
# ---------------------------------------------------------------------------


def test_handle_single_returns_result(server):
    msg = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}}
    out = server._handle_single(msg)
    assert out["id"] == 1
    assert out["result"] == {}


def test_handle_single_notification_returns_none(server):
    """Messages without `id` are notifications — server returns None."""
    msg = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
    assert server._handle_single(msg) is None


def test_handle_single_invalid_request_returns_error(server):
    out = server._handle_single("not a dict")
    assert out["error"]["code"] == -32600


def test_handle_single_missing_method_errors(server):
    out = server._handle_single({"jsonrpc": "2.0", "id": 99, "params": {}})
    assert out["error"]["code"] == -32600


def test_handle_single_unknown_method_returns_method_not_found(server):
    out = server._handle_single({"jsonrpc": "2.0", "id": 7, "method": "no_such"})
    assert out["error"]["code"] == -32601


def test_handle_single_value_error_returns_invalid_params(server):
    """tools/call with a missing `name` triggers ValueError → -32602."""
    out = server._handle_single({
        "jsonrpc": "2.0", "id": 8,
        "method": "tools/call",
        "params": {"name": "mcp_get_context", "arguments": {"query": ""}},
    })
    assert out["error"]["code"] == -32602


def test_handle_message_batch_filters_notifications(server):
    """A list mixing requests + notifications keeps only the request results."""
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    out = server._handle_message(batch)
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["id"] == 1


# ---------------------------------------------------------------------------
# _tool_result helper
# ---------------------------------------------------------------------------


def test_tool_result_envelope():
    out = _tool_result({"a": 1})
    assert out["structuredContent"] == {"a": 1}
    assert out["content"][0]["type"] == "text"
    assert json.loads(out["content"][0]["text"]) == {"a": 1}
