"""Helpers for `memoirs setup` — install paths, snippet writing, MCP config wiring.

Idempotent: every operation can be re-run safely. Snippets are wrapped between
`<!-- memoirs:start -->` and `<!-- memoirs:end -->` markers so re-running setup
replaces only the memoirs block, leaving user content intact.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .config import GEMMA_MODEL_PATH


# ---------------------------------------------------------------------------
# Persistent instruction snippet — same text everywhere, format adapts
# ---------------------------------------------------------------------------

INSTRUCTION_BODY = """## Memoirs (memory engine)

This workspace has the **Memoirs** MCP server available. Use it proactively:

1. **At task start:** call `mcp_get_context` with a query summarizing the goal.
   The result reduces the long history to relevant past decisions, preferences,
   and project facts in ~600-1500 tokens — feed it into your reasoning.

2. **When detecting a project / repo:** call `mcp_list_projects` and
   `mcp_get_project_context` for that project to load its scoped memory.

3. **At task end (or whenever the user states a durable preference, decision, or
   constraint):** call `mcp_add_memory` with `type` ∈
   {preference, fact, project, task, decision, style, credential_pointer},
   `importance: 1..5, confidence: 0..1`. Aim for type=preference / decision /
   project — avoid type=fact for generic statements.

4. **When you correct yourself or invalidate an earlier statement:** call
   `mcp_score_feedback` with `useful=false` on the stale memory and add the
   corrected version with `mcp_add_memory`.

5. **For long conversations (50+ messages):** call `mcp_summarize_thread` to
   compress them into a single durable memory.

Memory types: `preference, fact, project, task, decision, style, credential_pointer`.
"""

START = "<!-- memoirs:start -->"
END = "<!-- memoirs:end -->"


def _wrapped_snippet() -> str:
    return f"{START}\n{INSTRUCTION_BODY}\n{END}\n"


def write_or_replace_snippet(path: Path) -> str:
    """Write the snippet block to `path`, replacing any existing memoirs block.

    Returns one of: "created", "updated", "unchanged".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_block = _wrapped_snippet()
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        # Replace existing block if present
        if START in existing and END in existing:
            updated = re.sub(
                rf"{re.escape(START)}.*?{re.escape(END)}\n?",
                new_block,
                existing,
                flags=re.DOTALL,
                count=1,
            )
            if updated == existing:
                return "unchanged"
            path.write_text(updated, encoding="utf-8")
            return "updated"
        # Append to end with a separator
        path.write_text(existing.rstrip() + "\n\n" + new_block, encoding="utf-8")
        return "updated"
    path.write_text(new_block, encoding="utf-8")
    return "created"


# Cursor uses `.mdc` files with frontmatter — same body, different wrapper
def write_cursor_rule(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_block = (
        "---\n"
        "description: Use Memoirs MCP for persistent memory across tasks\n"
        "alwaysApply: true\n"
        "---\n\n"
        + _wrapped_snippet()
    )
    if path.exists() and START in path.read_text():
        existing = path.read_text(encoding="utf-8")
        updated = re.sub(
            rf"{re.escape(START)}.*?{re.escape(END)}\n?",
            _wrapped_snippet(),
            existing,
            flags=re.DOTALL,
            count=1,
        )
        if updated == existing:
            return "unchanged"
        path.write_text(updated, encoding="utf-8")
        return "updated"
    path.write_text(new_block, encoding="utf-8")
    return "created"


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def detect_clients(cwd: Path | None = None, home: Path | None = None) -> dict[str, dict]:
    """Detect which MCP clients exist on this system + their config paths."""
    cwd = cwd or Path.cwd()
    home = home or Path.home()
    result: dict[str, dict] = {}

    # Claude Code (looks for the binary OR the per-project file OR the global)
    claude_bin = shutil.which("claude")
    result["claude_code"] = {
        "installed": bool(claude_bin),
        "binary": claude_bin,
        "mcp_config": cwd / ".mcp.json",
        "instructions_workspace": cwd / "CLAUDE.md",
        "instructions_global": home / ".claude" / "CLAUDE.md",
    }

    # Codex CLI
    codex_bin = shutil.which("codex")
    codex_cfg = home / ".codex" / "config.toml"
    result["codex"] = {
        "installed": bool(codex_bin) or codex_cfg.exists(),
        "binary": codex_bin,
        "mcp_config": codex_cfg,
        "instructions": home / ".codex" / "AGENTS.md",
    }

    # VS Code (Copilot Chat or Continue)
    code_bin = shutil.which("code")
    has_copilot = False
    if code_bin:
        try:
            ext = subprocess.check_output([code_bin, "--list-extensions"], text=True, timeout=5)
            has_copilot = "github.copilot-chat" in ext or "anthropic.claude-code" in ext
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            pass
    result["vscode"] = {
        "installed": bool(code_bin),
        "binary": code_bin,
        "has_copilot": has_copilot,
        "mcp_config": cwd / ".vscode" / "mcp.json",
        "instructions": cwd / ".github" / "copilot-instructions.md",
    }

    # Cursor
    cursor_bin = shutil.which("cursor")
    cursor_storage = home / ".config" / "Cursor"
    result["cursor"] = {
        "installed": bool(cursor_bin) or cursor_storage.exists(),
        "binary": cursor_bin,
        "rule_file": cwd / ".cursor" / "rules" / "memoirs.mdc",
    }

    return result


# ---------------------------------------------------------------------------
# MCP config writers
# ---------------------------------------------------------------------------


def write_mcp_config_claude_code(path: Path, memoirs_bin: Path, db_path: Path) -> str:
    cfg = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
        except json.JSONDecodeError:
            cfg = {}
    servers = cfg.setdefault("mcpServers", {})
    new = {
        "command": str(memoirs_bin),
        "args": ["--db", str(db_path), "mcp"],
    }
    if servers.get("memoirs") == new:
        return "unchanged"
    servers["memoirs"] = new
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return "updated" if "memoirs" in servers else "created"


def write_mcp_config_vscode(path: Path, memoirs_bin: Path, db_path: Path) -> str:
    cfg = {}
    if path.exists():
        try:
            cfg = json.loads(path.read_text())
        except json.JSONDecodeError:
            cfg = {}
    servers = cfg.setdefault("servers", {})
    new = {
        "type": "stdio",
        "command": str(memoirs_bin),
        "args": ["--db", str(db_path), "mcp"],
    }
    if servers.get("memoirs") == new:
        return "unchanged"
    servers["memoirs"] = new
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    return "updated"


def write_mcp_config_codex(path: Path, memoirs_bin: Path, db_path: Path) -> str:
    """Append [mcp_servers.memoirs] to ~/.codex/config.toml if missing."""
    block = (
        f'\n[mcp_servers.memoirs]\n'
        f'command = "{memoirs_bin}"\n'
        f'args = ["--db", "{db_path}", "mcp"]\n'
    )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(block.lstrip())
        return "created"
    text = path.read_text()
    if "[mcp_servers.memoirs]" in text:
        # Idempotent: skip if already there
        return "unchanged"
    path.write_text(text.rstrip() + "\n" + block)
    return "updated"


# ---------------------------------------------------------------------------
# Misc setup tasks
# ---------------------------------------------------------------------------


def python_dep_missing(name: str) -> bool:
    try:
        __import__(name)
        return False
    except ImportError:
        return True


def gemma_model_present() -> bool:
    return GEMMA_MODEL_PATH.exists() and GEMMA_MODEL_PATH.stat().st_size > 1_000_000_000


def sqlcipher_available() -> bool:
    """Return True if sqlcipher3 (encryption-at-rest, P3-2) is importable."""
    try:
        import sqlcipher3.dbapi2  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


def encryption_setup_warning() -> str | None:
    """Surface a setup-time warning when MEMOIRS_ENCRYPT_KEY is requested but
    sqlcipher3 is not installed.

    Returns the warning string (suitable for printing in `memoirs setup` /
    `memoirs doctor`) or ``None`` when the configuration is consistent
    (either the env var is unset, or the dep is present).
    """
    import os as _os
    if not _os.environ.get("MEMOIRS_ENCRYPT_KEY"):
        return None
    if sqlcipher_available():
        return None
    return (
        "MEMOIRS_ENCRYPT_KEY is set but sqlcipher3 is not installed. "
        "Install with: pip install 'memoirs[encryption]' — otherwise "
        "memoirs will refuse to open the DB."
    )


def gpu_offload_available() -> bool:
    try:
        import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload())
    except ImportError:
        return False


def vulkan_buildable() -> bool:
    return bool(shutil.which("glslc"))
