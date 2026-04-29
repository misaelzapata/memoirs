"""Tests for the LangMem / LlamaIndex / Memori bench adapters.

What this suite proves:

  1. Each adapter skips cleanly when its backing pip package isn't
     importable — no crash, ``status.ok=False`` with a non-empty
     reason that names the missing package.
  2. With the package mocked, an ingest + query round-trip recovers
     the bench IDs we wrote (i.e. the bench_id ↔ engine_id mapping
     works in both directions).
  3. ``shutdown()`` releases the backing handle so a second adapter
     instance in the same process is unaffected.
  4. The adapters honour ``MEMOIRS_USE_OLLAMA=on`` via the shared
     env-shim helper from ``scripts.adapters._ollama``.

Tests run hermetically — no real network, no real OpenAI / Ollama,
no other engine package imports at the module level. The cloud-mode skip
tests use ``monkeypatch.delenv('OPENAI_API_KEY', ...)``; the
mock-API tests stub the other engine libraries via ``sys.modules`` patches.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers — fake bench data
# ---------------------------------------------------------------------------


def _fake_memories():
    """Three BenchMemory rows with deterministic IDs / contents."""
    from scripts.bench_vs_others_dataset import BenchMemory

    return [
        BenchMemory(id="mem_test_a", type="event",
                    content="The capital of France is Paris."),
        BenchMemory(id="mem_test_b", type="event",
                    content="Rex is a golden retriever."),
        BenchMemory(id="mem_test_c", type="event",
                    content="My favourite tea is Earl Grey."),
    ]


def _fake_query(text: str = "What is the capital of France?"):
    from scripts.bench_vs_others_dataset import BenchQuery

    return BenchQuery(query=text, gold_memory_ids=["mem_test_a"],
                       category="single-hop")


def _reload(mod_name: str):
    """Drop the adapter from ``sys.modules`` so a re-import re-runs init."""
    sys.modules.pop(mod_name, None)


# ===========================================================================
# LangMem
# ===========================================================================


def test_langmem_adapter_skips_when_package_missing(monkeypatch):
    """``import langmem`` failing must surface as a clean status string."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")  # bypass the cloud-key skip
    monkeypatch.delenv("MEMOIRS_USE_OLLAMA", raising=False)

    # Stash any real langmem module and force the import to fail.
    saved_lm = sys.modules.pop("langmem", None)
    sys.modules["langmem"] = None  # ImportError on `import langmem`
    try:
        _reload("scripts.adapters.langmem_adapter")
        from scripts.adapters.langmem_adapter import LangMemAdapter

        adapter = LangMemAdapter()
        try:
            assert adapter.status.ok is False
            assert "langmem" in adapter.status.reason.lower()
        finally:
            adapter.shutdown()
    finally:
        sys.modules.pop("langmem", None)
        if saved_lm is not None:
            sys.modules["langmem"] = saved_lm
        _reload("scripts.adapters.langmem_adapter")


def test_langmem_adapter_skips_without_openai_key(monkeypatch):
    """Cloud mode (no Ollama) requires OPENAI_API_KEY — skip cleanly."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MEMOIRS_USE_OLLAMA", raising=False)

    _reload("scripts.adapters.langmem_adapter")
    from scripts.adapters.langmem_adapter import LangMemAdapter

    adapter = LangMemAdapter()
    try:
        assert adapter.status.ok is False
        assert "openai" in adapter.status.reason.lower()
    finally:
        adapter.shutdown()


def test_langmem_adapter_add_and_query_roundtrip_with_mock(monkeypatch):
    """Stub langmem + langgraph stores so we can verify add/query maps IDs."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")
    monkeypatch.delenv("MEMOIRS_USE_OLLAMA", raising=False)

    # Build a fake langgraph.store.memory.InMemoryStore that ranks by
    # exact-substring match against the query — deterministic and
    # avoids any real embedding call.
    class _FakeItem:
        def __init__(self, key, value):
            self.key = key
            self.value = value

    class _FakeStore:
        def __init__(self, **_kwargs):
            self._rows: list[tuple[tuple, str, dict]] = []

        def put(self, namespace, key, value, **_kw):
            self._rows.append((namespace, key, value))

        def search(self, namespace_prefix, *, query=None, limit=10, **_kw):
            q = (query or "").lower()
            scored = []
            for ns, key, value in self._rows:
                if ns[: len(namespace_prefix)] != namespace_prefix:
                    continue
                content = str(value.get("content", "")).lower()
                # naive ranking: count token overlaps
                score = sum(1 for tok in q.split() if tok in content)
                scored.append((score, key, value))
            scored.sort(key=lambda r: -r[0])
            return [_FakeItem(k, v) for s, k, v in scored[:limit] if s > 0]

    fake_lg_store_memory = types.ModuleType("langgraph.store.memory")
    fake_lg_store_memory.InMemoryStore = _FakeStore
    fake_lg_store_base = types.ModuleType("langgraph.store.base")
    fake_lg_store = types.ModuleType("langgraph.store")
    fake_lg = types.ModuleType("langgraph")
    fake_langmem = types.ModuleType("langmem")
    fake_lc_openai = types.ModuleType("langchain_openai")

    class _FakeEmbeddings:
        def __init__(self, **kw):
            self.kw = kw

    fake_lc_openai.OpenAIEmbeddings = _FakeEmbeddings

    saved = {}
    for name, mod in [
        ("langgraph", fake_lg),
        ("langgraph.store", fake_lg_store),
        ("langgraph.store.memory", fake_lg_store_memory),
        ("langgraph.store.base", fake_lg_store_base),
        ("langmem", fake_langmem),
        ("langchain_openai", fake_lc_openai),
    ]:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    try:
        _reload("scripts.adapters.langmem_adapter")
        from scripts.adapters.langmem_adapter import LangMemAdapter

        adapter = LangMemAdapter()
        assert adapter.status.ok, adapter.status.reason

        adapter.add_memories(_fake_memories())
        ranked = adapter.query(_fake_query("capital of France Paris"), top_k=10)
        assert "mem_test_a" in ranked

        adapter.shutdown()
        assert adapter._store is None
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        _reload("scripts.adapters.langmem_adapter")


def test_langmem_adapter_applies_ollama_env_when_flag_on(monkeypatch):
    """With MEMOIRS_USE_OLLAMA=on the adapter mutates env BEFORE the import."""
    monkeypatch.setenv("MEMOIRS_USE_OLLAMA", "on")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    _reload("scripts.adapters.langmem_adapter")
    with mock.patch("scripts.adapters.langmem_adapter.ollama_is_up",
                    return_value=True):
        from scripts.adapters.langmem_adapter import LangMemAdapter
        # Adapter may skip on missing langmem in CI, but the env shim
        # MUST have run before the import attempt.
        LangMemAdapter()

    assert os.environ["OPENAI_BASE_URL"] == "http://localhost:11434/v1"
    assert os.environ["OPENAI_API_KEY"] == "ollama"


# ===========================================================================
# LlamaIndex
# ===========================================================================


def test_llamaindex_adapter_skips_when_package_missing(monkeypatch):
    """No llama_index in sys.modules → adapter skips cleanly."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")
    monkeypatch.delenv("MEMOIRS_USE_OLLAMA", raising=False)

    saved_li = {}
    for name in list(sys.modules):
        if name == "llama_index" or name.startswith("llama_index."):
            saved_li[name] = sys.modules.pop(name)
    sys.modules["llama_index"] = None
    sys.modules["llama_index.core"] = None
    try:
        _reload("scripts.adapters.llamaindex_adapter")
        from scripts.adapters.llamaindex_adapter import LlamaIndexAdapter

        adapter = LlamaIndexAdapter()
        try:
            assert adapter.status.ok is False
            assert "llama" in adapter.status.reason.lower()
        finally:
            adapter.shutdown()
    finally:
        sys.modules.pop("llama_index", None)
        sys.modules.pop("llama_index.core", None)
        for name, mod in saved_li.items():
            sys.modules[name] = mod
        _reload("scripts.adapters.llamaindex_adapter")


def test_llamaindex_adapter_skips_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MEMOIRS_USE_OLLAMA", raising=False)

    _reload("scripts.adapters.llamaindex_adapter")
    from scripts.adapters.llamaindex_adapter import LlamaIndexAdapter

    adapter = LlamaIndexAdapter()
    try:
        assert adapter.status.ok is False
        assert "openai" in adapter.status.reason.lower()
    finally:
        adapter.shutdown()


def test_llamaindex_adapter_add_and_query_roundtrip_with_mock(monkeypatch):
    """Stub llama_index.core so we can verify the add/query roundtrip."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stub")
    monkeypatch.delenv("MEMOIRS_USE_OLLAMA", raising=False)

    class _FakeDocument:
        def __init__(self, *, text, metadata=None, **_kw):
            self.text = text
            self.metadata = dict(metadata or {})

    class _FakeNode:
        def __init__(self, doc):
            self._doc = doc
            self.metadata = doc.metadata
            self.text = doc.text

    class _FakeNodeWithScore:
        def __init__(self, node, score):
            self.node = node
            self.score = score

    class _FakeRetriever:
        def __init__(self, docs, similarity_top_k=10):
            self._docs = docs
            self.similarity_top_k = similarity_top_k

        def retrieve(self, query):
            q = query.lower()
            scored = []
            for d in self._docs:
                score = sum(1 for tok in q.split() if tok in d.text.lower())
                scored.append((score, d))
            scored.sort(key=lambda r: -r[0])
            return [
                _FakeNodeWithScore(_FakeNode(d), s)
                for s, d in scored[: self.similarity_top_k] if s > 0
            ]

    class _FakeIndex:
        def __init__(self, docs):
            self._docs = docs

        @classmethod
        def from_documents(cls, docs, **_kw):
            return cls(list(docs))

        def as_retriever(self, similarity_top_k=10, **_kw):
            return _FakeRetriever(self._docs, similarity_top_k=similarity_top_k)

    class _FakeSettings:
        llm = None
        embed_model = None

    fake_li_core = types.ModuleType("llama_index.core")
    fake_li_core.VectorStoreIndex = _FakeIndex
    fake_li_core.Document = _FakeDocument
    fake_li_core.Settings = _FakeSettings
    fake_li = types.ModuleType("llama_index")
    fake_li.core = fake_li_core
    fake_li_llms = types.ModuleType("llama_index.llms")
    fake_li_llms_openai = types.ModuleType("llama_index.llms.openai")
    fake_li_llms_openai.OpenAI = lambda **kw: object()
    fake_li_emb = types.ModuleType("llama_index.embeddings")
    fake_li_emb_openai = types.ModuleType("llama_index.embeddings.openai")
    fake_li_emb_openai.OpenAIEmbedding = lambda **kw: object()

    saved = {}
    for name, mod in [
        ("llama_index", fake_li),
        ("llama_index.core", fake_li_core),
        ("llama_index.llms", fake_li_llms),
        ("llama_index.llms.openai", fake_li_llms_openai),
        ("llama_index.embeddings", fake_li_emb),
        ("llama_index.embeddings.openai", fake_li_emb_openai),
    ]:
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    try:
        _reload("scripts.adapters.llamaindex_adapter")
        from scripts.adapters.llamaindex_adapter import LlamaIndexAdapter

        adapter = LlamaIndexAdapter()
        assert adapter.status.ok, adapter.status.reason

        adapter.add_memories(_fake_memories())
        ranked = adapter.query(_fake_query("capital France Paris"), top_k=10)
        assert "mem_test_a" in ranked

        adapter.shutdown()
        assert adapter._index is None
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
        _reload("scripts.adapters.llamaindex_adapter")


def test_llamaindex_adapter_applies_ollama_env_when_flag_on(monkeypatch):
    monkeypatch.setenv("MEMOIRS_USE_OLLAMA", "on")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    _reload("scripts.adapters.llamaindex_adapter")
    with mock.patch("scripts.adapters.llamaindex_adapter.ollama_is_up",
                    return_value=True):
        from scripts.adapters.llamaindex_adapter import LlamaIndexAdapter
        LlamaIndexAdapter()

    assert os.environ["OPENAI_BASE_URL"] == "http://localhost:11434/v1"
    assert os.environ["OPENAI_API_KEY"] == "ollama"


# ===========================================================================
# Memori
# ===========================================================================


def test_memori_adapter_skips_when_package_missing(monkeypatch):
    """No `memori` package → status.ok=False with the package name in reason."""
    monkeypatch.delenv("MEMOIRS_USE_OLLAMA", raising=False)

    saved = sys.modules.pop("memori", None)
    sys.modules["memori"] = None
    try:
        _reload("scripts.adapters.memori_adapter")
        from scripts.adapters.memori_adapter import MemoriAdapter

        adapter = MemoriAdapter()
        try:
            assert adapter.status.ok is False
            assert "memori" in adapter.status.reason.lower()
        finally:
            adapter.shutdown()
    finally:
        sys.modules.pop("memori", None)
        if saved is not None:
            sys.modules["memori"] = saved
        _reload("scripts.adapters.memori_adapter")


def test_memori_adapter_add_and_query_roundtrip_with_mock(monkeypatch, tmp_path):
    """Stub a tiny in-memory `memori` module and verify the round-trip."""
    monkeypatch.delenv("MEMOIRS_USE_OLLAMA", raising=False)

    # Tiny in-process driver that mimics SQLite enough for the adapter.
    class _FactSearchResult:
        def __init__(self, fid, content, similarity):
            self.id = fid
            self.content = content
            self.similarity = similarity

    class _Conn:
        def __init__(self, store):
            self._store = store

        def execute(self, sql, params=()):
            entity_id = params[0]
            rows = [(fid, c) for (eid, fid, c) in self._store
                    if eid == entity_id]
            class _Cur:
                def __init__(self, r):
                    self._r = r
                def fetchall(self):
                    return list(self._r)
            return _Cur(rows)

    class _EntityDriver:
        def __init__(self):
            self._next = 1
        def create(self, external_id):
            pk = self._next
            self._next += 1
            return pk

    class _EntityFactDriver:
        def __init__(self, store):
            self._store = store
            self._next = 1
            self.conn = _Conn(store)

        def create(self, *, entity_id, facts, fact_embeddings=None,
                   conversation_id=None):
            for f in facts:
                self._store.append((entity_id, self._next, f))
                self._next += 1

    class _FakeDriver:
        def __init__(self):
            self._store: list[tuple[int, int, str]] = []
            self.entity = _EntityDriver()
            self.entity_fact = _EntityFactDriver(self._store)

    class _FakeStorage:
        def __init__(self, driver):
            self.driver = driver
        def build(self):
            return self

    class _FakeEmbeddings:
        model = "fake-embed-model"

    class _FakeConfig:
        def __init__(self, driver):
            self.storage = _FakeStorage(driver)
            self.embeddings = _FakeEmbeddings()

    class _FakeMemori:
        last_instance = None

        def __init__(self, conn=None, debug_truncate=True):
            self._driver = _FakeDriver()
            self.config = _FakeConfig(self._driver)
            self._entity_id = None
            type(self).last_instance = self

        def attribution(self, entity_id, process_id=None):
            self._entity_id = entity_id
            return self

        def new_session(self):
            return self

        def recall(self, query, limit=None):
            q = (query or "").lower()
            results = []
            for eid, fid, content in self._driver._store:
                score = sum(1 for tok in q.split() if tok in content.lower())
                if score > 0:
                    results.append(_FactSearchResult(fid, content, score / 10.0))
            results.sort(key=lambda r: -r.similarity)
            return results[:(limit or 10)]

        def close(self):
            pass

    def _fake_embed_texts(texts, *, model, **_kw):
        return [[0.0] * 4 for _ in (texts if isinstance(texts, list) else [texts])]

    fake_memori = types.ModuleType("memori")
    fake_memori.Memori = _FakeMemori
    fake_memori.embed_texts = _fake_embed_texts

    saved_memori = sys.modules.get("memori")
    sys.modules["memori"] = fake_memori
    try:
        _reload("scripts.adapters.memori_adapter")
        from scripts.adapters.memori_adapter import MemoriAdapter

        adapter = MemoriAdapter(db_path=tmp_path / "x" / "memori.db")
        assert adapter.status.ok, adapter.status.reason

        adapter.add_memories(_fake_memories())
        ranked = adapter.query(_fake_query("capital France Paris"), top_k=10)
        assert "mem_test_a" in ranked

        adapter.shutdown()
        assert adapter._memori is None
    finally:
        if saved_memori is not None:
            sys.modules["memori"] = saved_memori
        else:
            sys.modules.pop("memori", None)
        _reload("scripts.adapters.memori_adapter")


def test_memori_adapter_applies_ollama_env_when_flag_on(monkeypatch):
    monkeypatch.setenv("MEMOIRS_USE_OLLAMA", "on")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    _reload("scripts.adapters.memori_adapter")
    with mock.patch("scripts.adapters.memori_adapter.ollama_is_up",
                    return_value=True):
        from scripts.adapters.memori_adapter import MemoriAdapter
        MemoriAdapter()

    assert os.environ["OPENAI_BASE_URL"] == "http://localhost:11434/v1"
    assert os.environ["OPENAI_API_KEY"] == "ollama"


def test_memori_adapter_skips_with_install_hint_when_ollama_down(monkeypatch):
    """``MEMOIRS_USE_OLLAMA=on`` + Ollama down → skip with install hint."""
    monkeypatch.setenv("MEMOIRS_USE_OLLAMA", "on")

    _reload("scripts.adapters.memori_adapter")
    with mock.patch("scripts.adapters.memori_adapter.ollama_is_up",
                    return_value=False):
        from scripts.adapters.memori_adapter import MemoriAdapter
        adapter = MemoriAdapter()
    try:
        assert adapter.status.ok is False
        assert "ollama" in adapter.status.reason.lower()
    finally:
        adapter.shutdown()


# ===========================================================================
# Registry
# ===========================================================================


def test_registry_resolves_three_new_engine_names():
    """``build_adapter('langmem')`` etc must instantiate the right class."""
    from scripts.bench_vs_others import build_adapter, DEFAULT_ENGINES

    for name in ("langmem", "llamaindex", "memori"):
        # Build returns the adapter class even when the engine will SKIP
        # at runtime — the mapping itself is what we're asserting here.
        adapter = build_adapter(name)
        try:
            assert adapter.name == name
        finally:
            adapter.shutdown()
        assert name in DEFAULT_ENGINES
