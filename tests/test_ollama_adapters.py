"""Tests for the Ollama integration shared by the external adapters.

What these tests prove:

  1. ``ollama_is_up()`` returns False when nothing answers on
     ``localhost:11434`` (mocked via ``urllib.request.urlopen`` so the
     test is deterministic offline).
  2. ``ollama_env_for_openai_clients()`` produces the canonical env-var
     dict — the contract the external adapters rely on.
  3. The ``CogneeAdapter`` mutates ``os.environ`` with that shim when
     ``MEMOIRS_USE_OLLAMA=on`` and Ollama is reachable.
  4. The ``Mem0Adapter`` (Docker path) and ``LettaAdapter`` follow the
     same contract — the env shim runs BEFORE any provider library is
     imported.
  5. ``bench_vs_others --ollama`` with Ollama down prints the install
     hint and lets other engines SKIP cleanly instead of crashing.

The tests are intentionally hermetic: no real HTTP, no Docker, no other engine
package imports. They patch ``urllib.request.urlopen`` and stub out the
adapter helpers when needed so the suite passes on a sandbox.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest


# Resolve the project root so `scripts.*` works under pytest.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# 1. ollama_is_up() returns False when nothing is listening
# ---------------------------------------------------------------------------


def test_ollama_is_up_false_when_localhost_is_unreachable():
    """Patch ``urlopen`` to raise — the helper must swallow and return False."""
    from scripts.adapters import _ollama

    with mock.patch("urllib.request.urlopen", side_effect=OSError("refused")):
        assert _ollama.ollama_is_up(timeout=0.1) is False


def test_ollama_is_up_true_on_200():
    """A 200 response on /api/tags counts as up."""
    from scripts.adapters import _ollama

    fake = mock.MagicMock()
    fake.__enter__.return_value.status = 200
    fake.__exit__.return_value = False
    with mock.patch("urllib.request.urlopen", return_value=fake):
        assert _ollama.ollama_is_up(timeout=0.1) is True


# ---------------------------------------------------------------------------
# 2. ollama_env_for_openai_clients() shape
# ---------------------------------------------------------------------------


def test_ollama_env_has_all_documented_keys():
    """Every adapter relies on this exact set; test the contract directly."""
    from scripts.adapters._ollama import (
        OLLAMA_OPENAI_BASE_URL,
        ollama_env_for_openai_clients,
    )

    env = ollama_env_for_openai_clients(model="qwen2.5:3b")
    # OpenAI SDK shape (Mem0, Letta, Zep)
    assert env["OPENAI_API_KEY"] == "ollama"
    assert env["OPENAI_BASE_URL"] == OLLAMA_OPENAI_BASE_URL
    assert env["OPENAI_API_BASE"] == OLLAMA_OPENAI_BASE_URL
    assert env["OPENAI_MODEL"] == "qwen2.5:3b"
    # Cognee / LiteLLM shape
    assert env["LLM_PROVIDER"] == "openai"
    assert env["LLM_API_KEY"] == "ollama"
    assert env["LLM_ENDPOINT"] == OLLAMA_OPENAI_BASE_URL
    assert env["LLM_MODEL"] == "qwen2.5:3b"
    # Embedding side
    assert env["EMBEDDING_API_KEY"] == "ollama"
    assert env["EMBEDDING_ENDPOINT"] == OLLAMA_OPENAI_BASE_URL
    assert env["EMBEDDING_MODEL"] == "nomic-embed-text"


def test_use_ollama_requested_recognises_both_flag_names():
    """Adapters tolerate ``MEMOIRS_OLLAMA`` and ``MEMOIRS_USE_OLLAMA``."""
    from scripts.adapters._ollama import use_ollama_requested

    assert use_ollama_requested({"MEMOIRS_USE_OLLAMA": "on"}) is True
    assert use_ollama_requested({"MEMOIRS_OLLAMA": "1"}) is True
    assert use_ollama_requested({"MEMOIRS_USE_OLLAMA": "false"}) is False
    assert use_ollama_requested({}) is False


def test_ollama_list_models_parses_payload():
    """``ollama_list_models`` returns the names from /api/tags JSON."""
    from scripts.adapters import _ollama

    payload = json.dumps({
        "models": [{"name": "qwen2.5:3b"}, {"name": "nomic-embed-text"}],
    }).encode("utf-8")
    fake = mock.MagicMock()
    fake.__enter__.return_value.read.return_value = payload
    fake.__exit__.return_value = False
    with mock.patch("urllib.request.urlopen", return_value=fake):
        models = _ollama.ollama_list_models(timeout=0.1)
    assert "qwen2.5:3b" in models
    assert "nomic-embed-text" in models


# ---------------------------------------------------------------------------
# 3. Cognee adapter applies env when MEMOIRS_USE_OLLAMA is on
# ---------------------------------------------------------------------------


def test_cognee_adapter_applies_ollama_env_when_flag_on(monkeypatch):
    """With the flag on and Ollama up, env vars must be set in process."""
    monkeypatch.setenv("MEMOIRS_USE_OLLAMA", "on")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_ENDPOINT", raising=False)

    # Pretend Ollama is up so the env shim runs.
    with mock.patch("scripts.adapters.cognee_adapter.ollama_is_up",
                    return_value=True):
        from scripts.adapters.cognee_adapter import CogneeAdapter
        # Build adapter — even if cognee isn't installed, the env shim
        # has already run by the time we hit the import block.
        CogneeAdapter()

    assert os.environ["OPENAI_BASE_URL"] == "http://localhost:11434/v1"
    assert os.environ["OPENAI_API_KEY"] == "ollama"
    assert os.environ["LLM_ENDPOINT"] == "http://localhost:11434/v1"


def test_cognee_adapter_skips_with_install_hint_when_ollama_down(monkeypatch):
    """``--ollama`` + Ollama unreachable → status.ok=False with install hint."""
    monkeypatch.setenv("MEMOIRS_USE_OLLAMA", "on")
    with mock.patch("scripts.adapters.cognee_adapter.ollama_is_up",
                    return_value=False):
        from scripts.adapters.cognee_adapter import CogneeAdapter
        adapter = CogneeAdapter()
    assert adapter.status.ok is False
    assert "ollama" in adapter.status.reason.lower()
    assert "install" in adapter.status.reason.lower()


# ---------------------------------------------------------------------------
# 4. Mem0 + Letta apply the same contract
# ---------------------------------------------------------------------------


def test_mem0_adapter_applies_ollama_env_before_docker(monkeypatch):
    """Mem0 adapter must mutate env even though it then skips on missing
    Docker — order of operations matters because the OpenAI SDK reads
    env at import time."""
    monkeypatch.setenv("MEMOIRS_USE_OLLAMA", "on")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with mock.patch("scripts.adapters.mem0_adapter.ollama_is_up",
                    return_value=True):
        from scripts.adapters.mem0_adapter import Mem0Adapter
        adapter = Mem0Adapter()

    # Docker path will skip (no docker / no key), but env was applied first.
    assert os.environ.get("OPENAI_BASE_URL") == "http://localhost:11434/v1"
    assert os.environ.get("OPENAI_API_KEY") == "ollama"
    # Adapter itself ends up SKIP because OPENAI_API_KEY=ollama is the
    # dummy and Docker may not be present — either way it doesn't crash.
    assert isinstance(adapter.status.ok, bool)


def test_letta_adapter_applies_ollama_env_when_flag_on(monkeypatch):
    """Same shim wiring for the Letta adapter."""
    monkeypatch.setenv("MEMOIRS_USE_OLLAMA", "on")
    monkeypatch.delenv("LETTA_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with mock.patch("scripts.adapters.letta_adapter.ollama_is_up",
                    return_value=True):
        from scripts.adapters.letta_adapter import LettaAdapter
        LettaAdapter()  # will skip on missing LETTA_BASE_URL

    assert os.environ["OPENAI_BASE_URL"] == "http://localhost:11434/v1"
    assert os.environ["LLM_ENDPOINT"] == "http://localhost:11434/v1"


def test_zep_adapter_applies_ollama_env_when_flag_on(monkeypatch):
    """Zep follows the same contract for symmetry — handy for future
    Graphiti-on-Ollama runs."""
    monkeypatch.setenv("MEMOIRS_USE_OLLAMA", "on")
    monkeypatch.delenv("ZEP_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with mock.patch("scripts.adapters.zep_adapter.ollama_is_up",
                    return_value=True):
        from scripts.adapters.zep_adapter import ZepAdapter
        ZepAdapter()

    assert os.environ["OPENAI_BASE_URL"] == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# 5. bench_vs_others --ollama with Ollama down prints install hint
# ---------------------------------------------------------------------------


def test_bench_with_ollama_down_prints_hint_and_skips_others(
    tmp_path, capsys, monkeypatch,
):
    """With ``--ollama`` and Ollama unreachable, the bench must:
      * print the install hint to stdout
      * keep running (memoirs still benches)
      * record SKIP rows for any other engine that needs the LLM.
    """
    # Make sure no real key sneaks in and triggers an unintended path.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from scripts.bench_vs_others import main

    # Force every Ollama probe to report "down".
    with mock.patch("scripts.bench_vs_others.ollama_is_up", return_value=False), \
         mock.patch("scripts.adapters.cognee_adapter.ollama_is_up",
                    return_value=False), \
         mock.patch("scripts.adapters.mem0_adapter.ollama_is_up",
                    return_value=False):
        out_json = tmp_path / "report.json"
        rc = main([
            "--engines", "memoirs,cognee",
            "--top-k", "5",
            "--out", str(out_json),
            "--ollama",
            "--quiet",
        ])
    assert rc == 0
    captured = capsys.readouterr().out
    # The install hint surfaces verbatim so operators know what to run.
    assert "Ollama not up" in captured
    assert "ollama.com/install.sh" in captured
    # Cognee skip row references the install hint phrase.
    data = json.loads(out_json.read_text())
    statuses = {e["engine"]: e["status"] for e in data["engines"]}
    assert statuses["memoirs"] == "OK"
    assert statuses["cognee"].startswith("SKIP")
    assert "ollama" in statuses["cognee"].lower()
