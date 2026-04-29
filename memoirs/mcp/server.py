"""JSON-RPC 2.0 server over stdio.

This file is *only* the JSON-RPC plumbing. Tool definitions and dispatch live in
`mcp.tools`. The server is intentionally hand-rolled (zero deps) — if the user
later installs `[mcp]` we can swap to the official SDK without touching tools.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from .. import __version__
from ..config import MCP_LOG_ARG_PREVIEW_LEN, MCP_PROTOCOL_VERSION
from ..db import MemoirsDB
from ..observability import with_trace_context
from . import tools as tool_module


log = logging.getLogger("memoirs.mcp")


class MethodNotFound(Exception):
    pass


class McpServer:
    def __init__(self, db_path: Path | str) -> None:
        self.db = MemoirsDB(db_path)
        self.db.init()
        self.protocol_version = MCP_PROTOCOL_VERSION

    def close(self) -> None:
        self.db.close()

    def run(self) -> None:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                self._send_error(None, -32700, "Parse error", str(exc))
                continue
            responses = self._handle_message(message)
            if responses is None:
                continue
            if isinstance(responses, list):
                if responses:
                    self._send(responses)
            else:
                self._send(responses)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _handle_message(self, message: Any) -> dict | list[dict] | None:
        if isinstance(message, list):
            return [r for r in (self._handle_single(item) for item in message) if r is not None]
        return self._handle_single(message)

    def _handle_single(self, message: Any) -> dict | None:
        if not isinstance(message, dict):
            return self._error(None, -32600, "Invalid Request", "message must be an object")
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        if not isinstance(method, str):
            return self._error(request_id, -32600, "Invalid Request", "method is required")
        if request_id is None:
            self._handle_notification(method, params)
            return None
        try:
            result = self._dispatch(method, params)
        except MethodNotFound as exc:
            return self._error(request_id, -32601, "Method not found", str(exc))
        except ValueError as exc:
            return self._error(request_id, -32602, "Invalid params", str(exc))
        except Exception as exc:
            log.exception("MCP internal error in method=%s", method)
            return self._error(request_id, -32603, "Internal error", str(exc))
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def _handle_notification(self, method: str, params: Any) -> None:
        if method in {"notifications/initialized", "notifications/cancelled"}:
            return
        log.debug("ignored notification: %s", method)

    def _dispatch(self, method: str, params: Any) -> dict:
        if method == "initialize":
            if isinstance(params, dict) and params.get("protocolVersion"):
                self.protocol_version = str(params["protocolVersion"])
            client_info = params.get("clientInfo", {}) if isinstance(params, dict) else {}
            log.info(
                "MCP initialize: client=%s/%s protocol=%s",
                client_info.get("name", "?"),
                client_info.get("version", "?"),
                self.protocol_version,
            )
            return {
                "protocolVersion": self.protocol_version,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "memoirs", "version": __version__},
            }
        if method == "ping":
            return {}
        if method == "tools/list":
            log.debug("MCP tools/list")
            return {"tools": tool_module.TOOL_SCHEMAS}
        if method == "tools/call":
            if not isinstance(params, dict):
                raise ValueError("tools/call params must be an object")
            # Wrap each tool call in its own trace context so structured
            # logs emitted by tools are auto-tagged with trace_id/span_id.
            # An inbound `_meta.trace_id` (some clients propagate one) is
            # honored; otherwise a fresh id is generated.
            inbound_tid = None
            meta = params.get("_meta") if isinstance(params, dict) else None
            if isinstance(meta, dict):
                inbound_tid = meta.get("trace_id") or meta.get("traceId")
            with with_trace_context(trace_id=inbound_tid):
                return self._call_tool(params)
        raise MethodNotFound(method)

    def _call_tool(self, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        try:
            arg_preview = json.dumps(arguments, ensure_ascii=False).replace("\n", "\\n")[:MCP_LOG_ARG_PREVIEW_LEN]
        except (TypeError, ValueError):
            arg_preview = str(arguments)[:MCP_LOG_ARG_PREVIEW_LEN]
        log.info("MCP tool=%s args=%s", name, arg_preview)
        payload = tool_module.call_tool(self.db, name, arguments)
        return _tool_result(payload)

    # ------------------------------------------------------------------
    # Wire format helpers
    # ------------------------------------------------------------------

    def _send(self, message: dict | list[dict]) -> None:
        sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    def _send_error(self, request_id: Any, code: int, message: str, data: Any = None) -> None:
        self._send(self._error(request_id, code, message, data))

    def _error(self, request_id: Any, code: int, message: str, data: Any = None) -> dict:
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _tool_result(payload: dict) -> dict:
    return {
        "content": [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False, sort_keys=True)}
        ],
        "structuredContent": payload,
    }


def main(db_path: Path | str) -> int:
    server = McpServer(db_path)
    try:
        server.run()
    finally:
        server.close()
    return 0
