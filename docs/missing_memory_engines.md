# Missing Memory Engines — Adapter Candidates

Researched 2026-04-28. Already covered (skipped): memoirs, mem0, cognee, memori,
langmem, llamaindex, letta, zep / graphiti.

## Summary table

Ranked by **signal-to-effort** for a memoirs-style adapter (`add_memories`,
`query`, optional `ingest_conversation`). All entries are Python-importable
libraries with locally runnable engines, last commit in 2025-2026, and either
≥100 stars or a recent peer-reviewed paper.

| # | Engine | Stars | Last push | Pkg / import | API fit | Effort | Notes |
|---|--------|-------|-----------|--------------|---------|--------|-------|
| 1 | **A-MEM** (agiresearch/A-mem) | 989 | 2025-12-12 | `pip install .` → `from agentic_memory.memory_system import AgenticMemorySystem` | clean `add_note` / `search_agentic` | Low | NeurIPS '25, Zettelkasten + ChromaDB. Cited in our spec. |
| 2 | **MemoryOS** (BAI-LAB/MemoryOS) | 1,347 | 2026-04-28 | `pip install MemoryOS` → `from memoryos import Memoryos` | `add_memory(user_input, agent_response)` + retrieval | Low-Med | EMNLP '25 oral, hierarchical heat-driven. QA-pair shape needs message synthesis. |
| 3 | **Memoripy** (caspianmoon/memoripy) | 690 | 2026-03-18 | `pip install memoripy` → `from memoripy import MemoryManager` | `add_interaction` + `retrieve_relevant_interactions` | Low | Spreading activation + concept graph. Exactly the engine our spec calls out. |
| 4 | **HippoRAG** (OSU-NLP-Group/HippoRAG) | 3,460 | 2025-09-04 | `pip install hipporag` → `from hipporag import HippoRAG` | `index(docs)` + `retrieve(queries, num_to_retrieve)` | Low | NeurIPS '24 + ICML '25 (HippoRAG 2). KG + Personalized PageRank, no chat-pair concept (treat each memory as a doc). |
| 5 | **MIRIX** (Mirix-AI/MIRIX) | 3,527 | 2026-04-28 | `pip install mirix-client` (server backend Docker) | `client.add(messages)` + `client.retrieve_with_conversation` | Medium | SOTA 85.4% on LoCoMo, 6 specialized memory components. Needs Docker backend → similar to letta/zep adapters. |
| 6 | **General Agentic Memory / GAM** (VectorSpaceLab) | 846 | 2026-03-14 | `pip install -e ".[all]"` → `from gam import Workflow` | `wf.add(input_file)` + `wf.request(question)` | Medium | arXiv 2511.18423, dual-agent Memorizer+Researcher, claims SOTA on LoCoMo/HotpotQA/RULER/NarrativeQA. Doc-oriented — text snippets fit if synthesized as input. |
| 7 | **LightRAG** (HKUDS/LightRAG) | 34,499 | 2026-04-28 | `pip install lightrag-hku` → `from lightrag import LightRAG` | `lightrag.insert(text)` + `lightrag.query(q, mode="hybrid")` | Low-Med | EMNLP '25, KG + dual-level retrieval. Massive adoption — strong baseline even though framed as "RAG". |
| 8 | **txtai** (neuml/txtai) | 12,432 | 2026-04-21 | `pip install txtai` → `from txtai import Embeddings` (+ agent memory) | `embeddings.index([(id, text, meta)])` + `embeddings.search(q, k)` | Low | All-in-one embeddings DB with new agent/memory module. Cleanest "ChromaDB-equivalent" baseline. |
| 9 | **agno** (formerly phidata, agno-agi/agno) | 39,756 | 2026-04-28 | `pip install agno` → `from agno.memory.manager import MemoryManager` | LLM-driven memory capture + `MemorySearchResponse` ID-based search | Medium | Largest framework; memory is an opinionated submodule not a standalone engine, so adapter has to drive an `Agent` to record memories. |
| 10 | **CrewAI memory** (crewAIInc/crewAI) | 50,204 | 2026-04-28 | `pip install crewai` → `from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory` | per-class `save()` / `search()` | Medium | Three-tier memory (short/long/entity) backed by ChromaDB. Adapter has to instantiate each tier and merge results. |

## Engines investigated and rejected

- **Memary** (kingjulio8238) — last commit **2024-10-22**, fails the 2025-2026 maintenance bar.
- **Supermemory** (supermemoryai) — primarily TypeScript / SaaS; the PyPI `supermemory` package is a thin REST client to their hosted API (not local-first).
- **CaviraOSS/OpenMemory** — TypeScript, MCP-only; the `OpenMemoryMCP` flavors from mem0ai are the same engine we already cover via mem0.
- **rohitg00/agentmemory** — TypeScript; PyPI `agentmemory` is a different (older, unrelated) package, last meaningful release 2023.
- **WujiangXu/A-mem** — paper repro fork, the maintained library lives at `agiresearch/A-mem` (covered above).
- **microsoft/graphrag** — RAG over docs, no chat-pair / per-user memory concept; would need substantial shimming and LightRAG already covers the graph-RAG axis.
- **microsoft/autogen** — `autogen_core.memory.ListMemory` is a stub list, not a real engine; the production wrapper is `autogen_ext.memory.mem0` which we already cover.
- **deepset-ai/haystack** — `InMemoryChatMessageStore` is a chat history buffer, not a retrieval-ranked memory engine. Would only proxy other backends.
- **infiniflow/ragflow** — service/UI app (Docker compose, no Python lib API), large but not Python-importable.
- **stanford-oval/storm** — knowledge curation / report writer, not an agent memory engine.
- **mcp-memory-service** (doobidoo) — MCP server, no in-process Python API.
- **a-mem-mcp-server** (tobs-code) — 7 stars, fails star floor.
- **Helicone / Langfuse / PromptLayer** — observability, not memory engines.
- **Neo4j-LLM-Graph-Builder** — KG builder, not a memory layer.

---

## Per-engine notes

### 1. A-MEM (agiresearch/A-mem)

- URL: <https://github.com/agiresearch/A-mem>
- Stars: **989** · Last push: **2025-12-12** · License: MIT · Lang: Python
- Paper: NeurIPS 2025 *"A-Mem: Agentic Memory for LLM Agents"* (arXiv 2502.12110).
- Description: Zettelkasten-style memory with dynamic linking. ChromaDB for
  vectors, LLM auto-generates structured notes (tags, keywords, context) and
  evolves the link graph as new memories arrive.
- Python API
  ```python
  from agentic_memory.memory_system import AgenticMemorySystem
  m = AgenticMemorySystem(model_name="all-MiniLM-L6-v2",
                          llm_backend="openai", llm_model="gpt-4o-mini")
  mid = m.add_note("Deep learning neural networks", tags=["ml"])
  hits = m.search_agentic("neural networks", k=5)
  ```
- Fit: very clean. `add_note` returns the engine ID, `search_agentic` returns
  `[{"id", "content", "tags", ...}]` — direct map to `BenchMemory.id`.
- Adapter effort: **Low**. ~120 lines, mirrors our `cognee_adapter.py`.

### 2. MemoryOS (BAI-LAB/MemoryOS)

- URL: <https://github.com/BAI-LAB/MemoryOS>
- Stars: **1,347** · Last push: **2026-04-28** · License: Apache-2.0
- Paper: EMNLP 2025 Oral *"Memory OS of AI Agent"* (arXiv 2506.06326).
- Description: Hierarchical store/update/retrieve/generate stack with
  short→mid→long-term promotion driven by a heat score. Reports +49.11% F1 /
  +46.18% BLEU-1 on LoCoMo.
- Python API
  ```python
  from memoryos import Memoryos
  os_ = Memoryos(user_id="u", assistant_id="a", openai_api_key=…, data_path="/tmp/m")
  os_.add_memory(user_input="…", agent_response="…")
  reply = os_.get_response(query="…")
  ```
- Fit: requires QA-pair shape. We already synthesize pairs for letta — reuse
  that pattern. Retrieval likely surfaced via internal context dict; will
  need a probe like `scripts/probe_cognee.py` to enumerate IDs.
- Adapter effort: **Low-Medium**. ~180 lines, modeled after
  `letta_adapter.py`.

### 3. Memoripy (caspianmoon/memoripy)

- URL: <https://github.com/caspianmoon/memoripy>
- Stars: **690** · Last push: **2026-03-18** · License: Apache-2.0
- Description: Concept graph with spreading activation + decay/reinforcement
  + hierarchical clustering. Exactly the spreading-activation comparator the
  GAP doc references.
- Python API
  ```python
  from memoripy import MemoryManager, JSONStorage
  from memoripy.implemented_models import OpenAIChatModel, OllamaEmbeddingModel
  mm = MemoryManager(OpenAIChatModel(key, "gpt-4o-mini"),
                     OllamaEmbeddingModel("mxbai-embed-large"),
                     storage=JSONStorage("hist.json"))
  mm.add_interaction(prompt, response, embedding, concepts)
  hits = mm.retrieve_relevant_interactions(query, exclude_last_n=0)
  ```
- Fit: pair-shaped (prompt+response). For dataset memories we synthesize
  prompt="" / response=content. Retrieval returns interaction objects; we
  map each interaction back to its source `BenchMemory.id` via metadata.
- Adapter effort: **Low**. ~150 lines.

### 4. HippoRAG 2 (OSU-NLP-Group/HippoRAG)

- URL: <https://github.com/OSU-NLP-Group/HippoRAG>
- Stars: **3,460** · Last push: **2025-09-04** · License: MIT
- Papers: NeurIPS '24 (HippoRAG 1) + ICML '25 (HippoRAG 2). PyPI: `hipporag`.
- Description: KG + Personalized PageRank inspired by hippocampal indexing.
  Continual learning benchmark (NarrativeQA, MuSiQue, 2Wiki, HotpotQA).
- Python API
  ```python
  from hipporag import HippoRAG
  hr = HippoRAG(save_dir="./hr", llm_model_name="gpt-4o-mini",
                embedding_model_name="nvidia/NV-Embed-v2")
  hr.index(docs=[...])
  retrieval = hr.retrieve(queries=[...], num_to_retrieve=5)
  ```
- Fit: doc-shape; we already do that for cognee. `retrieve` returns ranked
  doc IDs we map 1:1 to `BenchMemory.id`.
- Adapter effort: **Low**. ~140 lines. Heavy first-time install (needs
  vllm/transformers); guard the import like other heavy adapters.

### 5. MIRIX (Mirix-AI/MIRIX)

- URL: <https://github.com/Mirix-AI/MIRIX>
- Stars: **3,527** · Last push: **2026-04-28** · License: Apache-2.0
- Paper: arXiv 2507.07957. Reports **85.4%** on LoCoMo.
- Description: Six specialized memory components (Core, Episodic, Semantic,
  Procedural, Resource, Knowledge Vault) with a meta-agent. PostgreSQL +
  BM25 + pgvector backend.
- Python API (server-mediated)
  ```python
  from mirix import MirixClient
  c = MirixClient(api_key=…, base_url="http://localhost:8531")
  c.initialize_meta_agent(config=…)
  c.add(user_id="u", messages=[…])
  hits = c.retrieve_with_conversation(user_id="u", messages=[…], limit=5)
  ```
- Fit: needs `docker compose up` (postgres + API). Retrieval returns memory
  objects keyed by Mirix IDs — we maintain `_conv_index` like letta.
- Adapter effort: **Medium**. ~220 lines + a docker bootstrap helper.

### 6. General Agentic Memory / GAM (VectorSpaceLab)

- URL: <https://github.com/VectorSpaceLab/general-agentic-memory>
- Stars: **846** · Last push: **2026-03-14** · License: MIT
- Paper: arXiv 2511.18423.
- Description: Just-in-Time memory framework with dual agents (Memorizer +
  Researcher). Operates on a "GAM directory" — text + video.
- Python API
  ```python
  from gam import Workflow
  wf = Workflow("text", gam_dir="./my_gam", model="gpt-4o-mini", api_key=…)
  wf.add(input_file="paper.pdf")
  res = wf.request("…")
  ```
- Fit: file-oriented. We'd write a temp file per `BenchMemory` (or stream a
  jsonl) and parse `res.answer` / `res.contexts` for memory IDs. The LoCoMo
  benchmark mode in `research/` is closer to what we want and ships
  `MemoryAgent` / `ResearchAgent` directly.
- Adapter effort: **Medium**. ~200 lines using the `research/gam_research`
  path rather than the file-IO Workflow.

### 7. LightRAG (HKUDS/LightRAG)

- URL: <https://github.com/HKUDS/LightRAG>
- Stars: **34,499** · Last push: **2026-04-28** · License: MIT
- Paper: EMNLP 2025.
- Description: Graph-augmented RAG with dual-level (entity/relation) retrieval.
  Pluggable storage backends (postgres, OpenSearch, JSON).
- Python API
  ```python
  from lightrag import LightRAG, QueryParam
  rag = LightRAG(working_dir="./lrag", llm_model_func=…, embedding_func=…)
  await rag.ainsert(text)
  ans = await rag.aquery("…", param=QueryParam(mode="hybrid"))
  ```
- Fit: doc-shape, async-only. Retrieval returns synthesized answer + citation
  context; we'd use the new `return_contexts=True` flag (Nov 2025) to recover
  source IDs.
- Adapter effort: **Low-Medium**. ~180 lines plus an asyncio bridge.

### 8. txtai (neuml/txtai)

- URL: <https://github.com/neuml/txtai>
- Stars: **12,432** · Last push: **2026-04-21** · License: Apache-2.0
- Description: All-in-one embeddings DB. Recently grew an `agent/` module
  with skill/memory hooks. Cleanest baseline for "just plain semantic search
  with no extraction pipeline".
- Python API
  ```python
  from txtai import Embeddings
  emb = Embeddings(path="sentence-transformers/all-MiniLM-L6-v2", content=True)
  emb.index([(m.id, m.content, None) for m in memories])
  hits = emb.search("…", limit=10)   # → [{"id", "text", "score"}]
  ```
- Fit: trivial — we already keep IDs as strings, embeddings index returns
  them verbatim.
- Adapter effort: **Low**. ~90 lines, the simplest of the bunch. Useful as a
  *retrieval floor* baseline (no extraction, no consolidation).

### 9. agno (agno-agi/agno, formerly phidata)

- URL: <https://github.com/agno-agi/agno>
- Stars: **39,756** · Last push: **2026-04-28** · License: MPL-2.0
- Description: Agent framework with a first-class `agno.memory.manager`
  module + `MemoryOptimizationStrategy` types. Memory is captured by an
  LLM-driven manager.
- Python API
  ```python
  from agno.memory.manager import MemoryManager
  from agno.db.sqlite import SqliteDb
  mm = MemoryManager(model=…, db=SqliteDb(db_file="agno.db"))
  mm.create_user_memories(user_id="u", messages=[…])
  hits = mm.search_user_memories(query="…", user_id="u")
  ```
- Fit: requires LLM at ingest time (which is fine — we have the same cost
  for cognee/memori). IDs are agno-internal UUIDs → we keep an
  `external_id → bench_id` map.
- Adapter effort: **Medium**. ~220 lines. Worth it because the framework
  has 39k stars and is a popular default.

### 10. CrewAI memory (crewAIInc/crewAI)

- URL: <https://github.com/crewAIInc/crewAI>
- Stars: **50,204** · Last push: **2026-04-28** · License: MIT
- Description: Three-tier memory (Short / Long / Entity) backed by ChromaDB
  + RAG. Each tier has its own `save()` / `search()`.
- Python API
  ```python
  from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
  from crewai.memory.storage.rag_storage import RAGStorage
  ltm = LongTermMemory(storage=RAGStorage(type="long_term", embedder_config=…))
  ltm.save(value=…, metadata={"id": bench_id})
  results = ltm.search(query, limit=10)
  ```
- Fit: need to fan out across all three stores (short/long/entity) and merge
  by score. Each store accepts arbitrary metadata so we can stash
  `BenchMemory.id` and recover it on retrieve.
- Adapter effort: **Medium**. ~250 lines because of the tier fan-out.

---

## Top 3 recommended next adapters

1. **txtai** — fastest win, clearest "retrieval floor" baseline.
2. **A-MEM** — academic SOTA reference, near-trivial Python API, MIT.
3. **Memoripy** — exactly the spreading-activation engine our docs cite as a
   comparator and currently missing.

(MemoryOS is a strong honorable mention — slightly more wiring because of
the QA-pair shape but high benchmark numbers.)
