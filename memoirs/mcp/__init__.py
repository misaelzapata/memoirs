"""MCP server (JSON-RPC 2.0 over stdio).

Entry point: `from memoirs.mcp import main; main(db_path)`.
"""
from .server import McpServer, main

__all__ = ["McpServer", "main"]
