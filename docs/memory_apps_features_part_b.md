# Memory Apps Feature Catalog — Part B (Long-tail / Niche / Novel)

> Catalog of declared features for 13 less-mainstream but relevant memory apps for LLMs / agents.
> Sources: GitHub repository metadata + READMEs (fetched 2026-04-28 via GitHub API), official docs, vendor blog posts.
> Where a feature is not mentioned in public sources we mark it as `?` (not declared) instead of guessing.

## Apps covered

| # | App | Repo / source | Stars | License | Last push |
|---|---|---|---|---|---|
| 1 | **Supermemory** | [supermemoryai/supermemory](https://github.com/supermemoryai/supermemory) | 22,279 | MIT | 2026-04-28 |
| 2 | **MemoryOS** | [BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS) | 1,348 | Apache-2.0 | 2026-04-28 |
| 3 | **A-MEM (Agentic Memory)** | [agiresearch/A-mem](https://github.com/agiresearch/A-mem) (+ [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem)) | 988 / 866 | MIT | 2025-12-12 / 2026-03-05 |
| 4 | **HippoRAG / HippoRAG 2** | [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) | 3,459 | MIT | 2025-09-04 |
| 5 | **OpenMemory (CaviraOSS)** | [CaviraOSS/OpenMemory](https://github.com/CaviraOSS/OpenMemory) | 4,033 | Apache-2.0 | 2026-04-25 |
| 6 | **Graphlit** | [graphlit/graphlit-client-python](https://github.com/graphlit/graphlit-client-python) (closed-source platform) | 20 (client) | MIT (client) | 2026-04-28 |
| 7 | **mcp-memory-service** | [doobidoo/mcp-memory-service](https://github.com/doobidoo/mcp-memory-service) | 1,744 | Apache-2.0 | 2026-04-28 |
| 8 | **Redis Agent Memory Server** | [redis/agent-memory-server](https://github.com/redis/agent-memory-server) | 238 | (see repo, NOASSERTION/Apache-style) | 2026-04-26 |
| 9 | **Pieces for Developers (LTM-2 / 2.5 / 2.7)** | closed source — [pieces.app](https://pieces.app/features/long-term-memory) / [docs.pieces.app](https://docs.pieces.app/products/core-dependencies/pieces-os/long-term-memory) | n/a | Proprietary | n/a |
| 10 | **ReMe (AgentScope)** | [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) | 2,846 | Apache-2.0 | 2026-04-28 |
| 11 | **Memori (MemoriLabs)** | [MemoriLabs/Memori](https://github.com/MemoriLabs/Memori) | 13,957 | NOASSERTION (Apache-style per badge) | 2026-04-28 |
| 12 | **agentmemory** | [rohitg00/agentmemory](https://github.com/rohitg00/agentmemory) | 2,070 | Apache-2.0 | 2026-04-28 |
| 13 | **Engram** | [Gentleman-Programming/engram](https://github.com/Gentleman-Programming/engram) | 2,944 | MIT | 2026-04-28 |
| 14 | **claude-mem** (bonus, viral 2026) | [thedotmack/claude-mem](https://github.com/thedotmack/claude-mem) | 69,130 | NOASSERTION | 2026-04-28 |

---

## 1. Comparative feature matrix

Legend: ✅ declared / ❌ explicitly absent or contradicted / ⚠️ partial / `?` not declared in public sources.

| Feature | Supermemory | MemoryOS | A-MEM | HippoRAG 2 | OpenMemory | Graphlit | mcp-memory-service | Redis AMS | Pieces LTM-2 | ReMe | Memori | agentmemory | Engram | claude-mem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Memory tiers** (short / mid / long) | facts + dynamic profile | short / mid / long-term persona | flat agentic notes | long-term graph memory | episodic / semantic / procedural / emotional / reflective | semantic memory + KG | working memory + long-term + KG | working (session) + long-term | OS-wide capture + timeline | file-based + vector | structured persistent state | hierarchical (event/observation/decision) | Save/Update/Search w/ topic keys + sessions | per-session compressed archive |
| **Episodic memory** | ⚠️ implicit | ✅ short-term QA pairs | ⚠️ as notes | ⚠️ via passages | ✅ explicit sector | ⚠️ feeds | `?` | ✅ as session messages | ✅ via timeline | ✅ via dialog/ JSONL | ✅ session-scoped | ✅ "events" | ✅ session + observation | ✅ session archives |
| **Semantic memory** | ✅ facts | ✅ long-term persona knowledge | ✅ notes | ✅ entities/passages | ✅ explicit sector | ✅ explicit | ✅ semantic/keyword/hybrid | ✅ semantic memory | ⚠️ derived | ✅ via memory_search | ✅ structured persistent | ✅ via vector + graph | ✅ FTS5 + topic keys | ✅ via Chroma |
| **Procedural memory** | `?` | `?` | `?` | `?` | ✅ explicit sector | `?` | `?` | `?` | `?` | `?` | ⚠️ via process_id | `?` | `?` | `?` |
| **Emotional / reflective memory** | `?` | `?` | `?` | `?` | ✅ explicit sectors | `?` | `?` | `?` | `?` | `?` | `?` | `?` | `?` | `?` |
| **Storage backends** | Postgres + Cloudflare KV/Workers + Drizzle | local files; Chromadb integration | ChromaDB | local files + custom embedding store | SQLite (default) / Postgres | hybrid: vector DB (Azure AI Search / Pinecone / Qdrant) + cloud object storage + graph DB | SQLite-vec / Cloudflare / Hybrid / Milvus (Lite/self-host/Zilliz) | Redis (RedisVL) — pluggable vector DB factory | local SQLite-style on-device store | file system + vector DB | BYO DB (Memori-Cloud or BYODB) | SQLite (no external DB) | SQLite + FTS5 (single binary) | SQLite + ChromaDB (`~/.claude-mem/`) |
| **LLM providers** | provider-agnostic (called via Vercel/OpenAI/Mastra/etc.) | OpenAI, Anthropic, Deepseek-R1, Qwen/Qwen3, vLLM, Llama-Factory | OpenAI / Ollama | OpenAI + any vLLM-served model + OpenAI-compatible endpoints | OpenAI, Gemini, Ollama, AWS, synthetic fallback | not enumerated in client README | works via REST so any | OpenAI, Anthropic, AWS Bedrock, Ollama, Azure, Gemini (via LiteLLM, 100+) | local + cloud LLMs (RAG mode) | Qwen / OpenAI-compatible (LLM_BASE_URL) | Anthropic, Bedrock, DeepSeek, Gemini, Grok, OpenAI | provider-agnostic (LiteLLM-style); skills ride on host agent | provider-agnostic | rides on Claude Code's agent-sdk (Anthropic) |
| **Embedders** | managed | BAAI/bge-m3, Qwen3-Embedding-0.6B, all-MiniLM-L6-v2 | all-MiniLM-L6-v2 (default) | NV-Embed-v2, GritLM, Contriever | OpenAI / Gemini / Ollama / AWS / synthetic | configurable | ONNX local (default) | LiteLLM-supported (text-embedding-3-small, Bedrock Titan, Ollama nomic, etc.) | on-device nano-models (LTM-2.5 distilled) | OpenAI-compatible (EMBEDDING_BASE_URL) | (Memori manages) | all-MiniLM-L6-v2 (local, free) | (configurable) | Chroma defaults |
| **Retrieval modes** | hybrid (memory + RAG), memories-only, document search | retriever spans short/mid/long | search_agentic semantic | dense + KG (PageRank) + filtering rerank | composite: salience + recency + coactivation; explainable trace | semantic search w/ entity context, GraphRAG | semantic + keyword + hybrid (BM25 + vector) | semantic, keyword, hybrid w/ metadata filters | conversational search across timeline | vector + BM25 hybrid | structured recall + Intelligent Recall + Advanced Augmentation | BM25 + Vector + Graph w/ RRF fusion | FTS5, semantic context, timeline | semantic search + per-session retrieval |
| **Decay / forgetting** | ✅ "automatic forgetting" of expired info | ⚠️ heat-based promotion (no explicit decay) | ⚠️ via evolution | ❌ continual addition (no decay) | ✅ adaptive decay per sector + reinforcement | `?` | ✅ "auto consolidation: decay + compression" | `?` | ⚠️ 9-month rolling window | `?` (compaction-driven) | `?` | ⚠️ via lifecycle/audit | ⚠️ session-end summary | ⚠️ compression at session end |
| **Consolidation / compaction** | ✅ extraction + profile | ✅ short→mid→long promotion via heat | ✅ "memory evolution" / linking | ✅ OpenIE → KG | ✅ via decay + reinforcement | ✅ enrichment pipeline | ✅ "autonomous consolidation compresses old memories" | ✅ summary memory strategy | ✅ timeline aggregation | ✅ Compactor (ReActAgent), tool-result compactor | ✅ background extraction | ✅ session-end consolidation | ✅ session summary | ✅ compression w/ Claude agent-sdk |
| **Deduplication** | ✅ implied via fact tracking | `?` | `?` | `?` | `?` | `?` | ✅ (mentioned, with `conversation_id` to bypass dedup) | ✅ "deduplication" listed | `?` | `?` | `?` | `?` | `?` | `?` |
| **Versioning** | ✅ contradictions / updates handled | `?` | ⚠️ update API | `?` | ✅ temporal evolution closes prior facts | ✅ via KG temporal edges (claimed) | `?` | `?` | `?` | `?` | `?` | `?` | ⚠️ via mem_update | `?` |
| **Expire / TTL** | ✅ "expires after the date passes" | `?` | `?` | `?` | ❌ "no dumb TTLs" — uses decay instead | `?` | ⚠️ via consolidation | `?` | ✅ 9-month retention | ✅ tool-result TTL (N-day) | `?` | `?` | `?` | `?` |
| **Archive** | `?` | `?` | `?` | `?` | `?` | `?` | `?` | `?` | ✅ timeline view | ✅ dialog/ folder JSONL | `?` | `?` | ✅ `engram export`/JSON | ✅ session archives |
| **Knowledge graph** | ⚠️ "single memory structure and ontology" | ⚠️ persona profile graph | ⚠️ Zettelkasten links between notes | ✅ explicit KG + Personalized PageRank | ✅ "waypoint graph" (associative, traversable) | ✅ first-class KG with typed entity-to-entity edges | ✅ "Yes (typed edges: causes, fixes, contradicts)" | ⚠️ entity recognition (no explicit KG) | ❌ not declared | ❌ not declared | `?` | ✅ BM25 + Vector + **Graph** (RRF fusion) | ❌ not declared | ❌ not declared |
| **Temporal queries / time-travel** | ⚠️ "handles temporal changes" | `?` | `?` | `?` | ✅ `valid_from`/`valid_to`, point-in-time queries, timelines, change detection | ✅ temporal KG edges (per platform glossary) | `?` | `?` | ✅ timeline view, time-range summaries (LTM-2.5 roadmap) | `?` | `?` | `?` | ✅ `mem_timeline` chronological context | ⚠️ session-ordered |
| **Conflict resolution** | ✅ "resolves contradictions automatically" | `?` | `?` | `?` | ✅ auto-evolution closes prior facts | `?` | ⚠️ "contradicts" edge type | `?` | `?` | `?` | `?` | ⚠️ audit policy on deletes | ✅ `mem_judge` conflict surfacing tool | `?` |
| **Multi-tenant / scoping / ACL** | ✅ `containerTag`, projects | ✅ user_id + assistant_id | `?` | `?` | ✅ `user_id` | ✅ org / environment IDs + JWT | ✅ `X-Agent-ID` header, tag-based scoping, OAuth 2.0 + DCR, multi-user | ✅ `user_id` + auth (OAuth2/JWT) | ✅ per-source access control | ✅ working_dir per agent | ✅ `entity_id` + `process_id` attribution + sessions | ✅ projects + leases + signals | ✅ project-scoped, project consolidation | ✅ per-project SQLite |
| **Privacy / encryption** | `?` | `?` | `?` | `?` | ✅ self-hosted, local-first | ⚠️ JWT + cloud platform | ✅ ONNX local embeddings, OAuth 2.0 + DCR, on-prem | ✅ OAuth2/JWT | ✅ "captures, indexes, and encrypts data **locally**"; nothing leaves device unless explicit | ✅ local file system | ✅ BYODB option | ✅ local-only by default | ✅ single binary, local SQLite | ✅ local SQLite |
| **Transports** | npm + PyPI SDK + REST API + MCP (`mcp.supermemory.ai/mcp`) + browser ext + plugins | PyPI (`memoryos-pro`) + MCP server (stdio) + Docker + Chromadb variant + Playground UI | PyPI (`pip install .`) | PyPI (`pip install hipporag`) | PyPI (`openmemory-py`) + npm (`openmemory-js`) + REST + MCP + Docker + VS Code ext | Python client + GraphQL API + MCP server (open source) | PyPI (`mcp-memory-service`) + REST (15 endpoints) + MCP stdio + Streamable HTTP / SSE + Cloudflare Tunnel + Web Dashboard | PyPI (`agent-memory-client`) + REST + MCP (stdio + SSE) + Docker / Docker Hub | PiecesOS desktop runtime + MCP + IDE plugins | PyPI (`reme-ai`) | PyPI (`memori`) + npm (`@memorilabs/memori`) + REST + MCP (`api.memorilabs.ai/mcp/`) + OpenClaw plugin + Dashboard | npm (`@agentmemory/agentmemory`) + 104-endpoint REST + MCP server + 12 hooks + AgentSDKProvider | Single Go binary; CLI + REST (port 7437) + MCP stdio + TUI + git sync + opt-in Cloud | npm/Claude Code plugin (`/plugin install claude-mem`) + Worker Service (Express :37777) + MCP |
| **IDE / agent integrations** | Claude Desktop, Cursor, Windsurf, VS Code, Claude Code, OpenCode, OpenClaw, Hermes; Vercel AI SDK, LangChain, LangGraph, OpenAI Agents, Mastra, Agno, Claude Memory Tool, n8n | Claude Desktop, Cline, Cursor | (library) | (library) | LangChain, CrewAI, AutoGen, Streamlit, MCP, VS Code, Cursor, Windsurf | Claude Desktop, Goose, Cline, Cursor, Windsurf | LangGraph, CrewAI, AutoGen, Claude Desktop, Claude Code, OpenCode, Cursor, Windsurf, Gemini CLI/Code Assist, Codex CLI, Goose, Aider, Copilot CLI, Amp, Continue, Zed, Cody, JetBrains, Replit, Sourcegraph, Qodo, Raycast, Kilo Code, ChatGPT (Dev Mode), claude.ai (Remote MCP) | Claude Desktop (MCP), LangChain (auto tool conversion), LangGraph | Claude Desktop, Cursor, VS Code, JetBrains, etc. via Pieces OS | QwenPaw / CoPaw, generic agent loops | Claude Code, Cursor, Codex, Warp, Antigravity, Agno; OpenClaw plugin | Claude Code, OpenClaw, Hermes, Cursor, Gemini CLI, OpenCode, Codex, Cline, Goose, Kilo Code, Aider, Claude Desktop, Windsurf, Roo Code, Claude SDK | Claude Code (plugin marketplace), OpenCode, Gemini CLI, Codex, VS Code (Copilot), Antigravity, Cursor, Windsurf, "any MCP client" | Claude Code only (plugin) |
| **Audit / explainability / provenance** | ⚠️ profile.static / profile.dynamic separation | `?` | ⚠️ generated tags + context | ⚠️ retrieval results show supporting passages | ✅ "explainable traces — see *why* something was recalled" (waypoint trace) | ⚠️ KG provides relationship context | ⚠️ tags + agent ID | `?` | ⚠️ timeline aggregation | `?` | ⚠️ Dashboard analytics | ✅ "audit policy codified across every delete path" + iii Console (trace-level engine inspection) | ✅ `mem_judge`, `mem_get_observation`, timeline | ⚠️ stored archives are inspectable |
| **Streaming retrieval / events** | `?` | `?` | `?` | `?` | `?` | `?` | ✅ "SSE events — real-time notifications when any agent stores or deletes a memory" | ⚠️ Docket task backend (background) | `?` | ⚠️ async summary tasks | `?` | ⚠️ real-time viewer | `?` | `?` |
| **Auto thread / project summarization** | ✅ profile dynamic part | ✅ via Updater promotion | ⚠️ via note generation | ❌ not the focus | ⚠️ via reflective sector | ✅ enrichment | ✅ autonomous consolidation | ✅ "automatic conversation summarization" + topic extraction | ✅ "intuitive dynamic summary generation" (LTM-2.5 roadmap) | ✅ Compactor (ReActAgent), `summary_memory`, `pre_reasoning_hook` | ✅ session grouping | ⚠️ via session lifecycle | ✅ `mem_session_end`, `mem_session_summary` | ✅ session summarization w/ Claude agent-sdk |
| **Tool-call memory** | ⚠️ tool integrations capture | `?` | `?` | `?` | ✅ "store dialog + tool calls as episodic memory" (AutoGen pattern) | ⚠️ via content ingestion | `?` | ⚠️ via working memory messages | ✅ workflow capture across apps | ✅ ToolResultCompactor — long tool outputs cached in `tool_result/<uuid>.txt` | ✅ "memory from what agents do, not just what they say" | ✅ via PostToolUse hook (Claude Code) | ⚠️ via observations | ✅ PostToolUse hook |
| **Multi-modal** | ✅ PDFs, images (OCR), videos (transcription), code (AST-aware chunking) | ❌ text only | ❌ text only | ❌ text only | ⚠️ ingest connectors handle docs/sheets/slides | ✅ "documents, audio transcripts, video, web pages, API data" | ⚠️ document ingestion (web dashboard) | ⚠️ depends on input | ✅ "code you copy, screens you view, audio you hear" | ❌ text/files | `?` | ❌ text-focused | ❌ text-focused | ❌ text-focused |
| **Connectors / ingestion** | Google Drive, Gmail, Notion, OneDrive, GitHub, Web Crawler (real-time webhooks) | ❌ | ❌ | ❌ | GitHub, Notion, Google Drive, Google Sheets, Google Slides, OneDrive, Web Crawler | broad: documents, audio, video, web pages, API data, Slack, etc. | document ingestion via Web Dashboard | n/a | OS-wide passive capture (every app) | ❌ | `?` | filesystem connector (`@agentmemory/fs-watcher`) | ❌ | ❌ |
| **Eval harness included** | ✅ MemoryBench (`npx skills add supermemoryai/memorybench`) | ✅ LoCoMo reproduction scripts | ⚠️ separate AgenticMemory repo | ✅ `tests_openai.py` / `tests_local.py` + reproduce/dataset | `?` | `?` | ✅ benchmark/QUALITY/SCALE/COMPARISON via LongMemEval-S | `?` | `?` | ⚠️ tests/light/test_reme_light.py | `?` | ✅ `benchmark/LONGMEMEVAL.md`, QUALITY, SCALE, COMPARISON | `?` | `?` |
| **Published benchmarks** | LongMemEval **#1 / 81.6%**, LoCoMo #1, ConvoMem #1 | LoCoMo: F1 +49.11%, BLEU-1 +46.18% (vs baseline) | "superior performance vs SOTA across 6 foundation models" | NaturalQuestions, PopQA, NarrativeQA, MuSiQue, 2Wiki, HotpotQA, LV-Eval (associativity + sense-making + factual memory) | ❌ none | ❌ none | LongMemEval R@5 86.0% (session) / 80.4% (turn); compares vs MemPalace, Mem0, Zep | `?` | `?` | LoCoMo & HaluMem SOTA claims | LoCoMo 81.95% accuracy / 1,294 tokens per query / 4.97% of full-context | LongMemEval-S R@5 95.2%, R@10 98.6%, MRR 88.2%; vs mem0 / Letta | `?` | `?` |
| **Migration / interop** | n/a | n/a | n/a | n/a | ✅ migration tool: imports from Mem0, Zep, Supermemory | n/a | ✅ SHODH Unified Memory API spec compatible | ⚠️ via REST | n/a | n/a | n/a | n/a | git sync between machines | n/a |
| **Web dashboard / UI** | App + Dashboard (`app.supermemory.ai`, `console.supermemory.ai`) | Playground (`baijia.online/memoryos`) | ❌ | ❌ | Optional UI profile (Docker) | ✅ Graphlit Portal | ✅ Web Dashboard, Tag Browser, Analytics, API docs, Quality scoring | ❌ | ✅ Pieces Desktop + Timeline UI | ❌ | ✅ `app.memorilabs.ai` Dashboard, Analytics, Playground | ✅ real-time viewer + iii Console | ✅ TUI (Catppuccin) + opt-in cloud dashboard | ❌ |
| **Auto-capture (no manual `add()`)** | ⚠️ via plugins | ❌ explicit add | ❌ explicit add | ❌ explicit index | ❌ explicit add | ⚠️ via ingestion pipeline | ⚠️ agent-driven, X-Agent-ID auto-tag | ⚠️ working memory messages → background extraction | ✅ OS-wide passive capture | ❌ explicit | ⚠️ background via `register(client)` | ✅ "12 hooks (zero manual effort)" | ⚠️ via Memory Protocol prompts | ✅ 5 lifecycle hooks (SessionStart→End) |
| **Cross-session / cross-agent sharing** | ✅ via `containerTag` | ✅ via persistent IDs | ⚠️ same DB | ⚠️ same `save_dir` | ✅ self-hosted central server | ✅ multi-tenant cloud | ✅ "shared across all agents and runs" + tag-as-bus pattern | ✅ Redis-backed shared store | ✅ across all your apps | ⚠️ files in shared dir | ✅ team / shared context | ✅ "MCP + REST + leases + signals" | ✅ git sync + opt-in cloud | ❌ Claude Code only |
| **Open-source license** | MIT | Apache-2.0 | MIT | MIT | Apache-2.0 | MIT (client only; platform proprietary) | Apache-2.0 | repo NOASSERTION (Apache-style per badge) | ❌ proprietary | Apache-2.0 | NOASSERTION (Apache-style) | Apache-2.0 | MIT | NOASSERTION |

---

## 2. Per-app detail

### 2.1 Supermemory — supermemoryai/supermemory
- **Source:** [README](https://github.com/supermemoryai/supermemory), [docs](https://supermemory.ai/docs)
- **Pitch:** "State-of-the-art memory and context engine for AI." Memory + RAG + connectors + file processing in one API.
- **Memory model:** auto-extraction of facts, dynamic+static user profiles, contradictions and "automatic forgetting" of expired info.
- **Storage / infra:** Postgres + Cloudflare KV / Workers / Pages, Drizzle ORM, Remix, Tailwind, Vite, TypeScript.
- **Search modes:** `hybrid` (RAG + Memory), `memories`, `documents`.
- **Profiles API:** `client.profile()` returns `{static, dynamic, searchResults}` in ~50ms (claim).
- **Connectors:** Google Drive, Gmail, Notion, OneDrive, GitHub, Web Crawler — real-time webhooks.
- **Multi-modal:** PDFs, images (OCR), videos (transcription), code (AST-aware chunking).
- **Frameworks / SDKs:** Vercel AI SDK, Mastra, LangChain, LangGraph, OpenAI Agents SDK, Agno, Claude Memory Tool, n8n. NPM `supermemory`, PyPI `supermemory`.
- **MCP:** `https://mcp.supermemory.ai/mcp`, OAuth or Bearer; clients = Claude Desktop, Cursor, Windsurf, VS Code, Claude Code, OpenCode, OpenClaw, Hermes.
- **Plugins (open-source):** [openclaw-supermemory](https://github.com/supermemoryai/openclaw-supermemory), [claude-supermemory](https://github.com/supermemoryai/claude-supermemory), [opencode-supermemory](https://github.com/supermemoryai/opencode-supermemory).
- **Benchmarks:** LongMemEval **81.6% — #1**, LoCoMo #1, ConvoMem #1 (per [README](https://github.com/supermemoryai/supermemory)).
- **Eval harness:** ships [MemoryBench](https://supermemory.ai/docs/memorybench/overview) — open-source reproducible benchmark framework comparing Supermemory / Mem0 / Zep / etc.
- **Differentiators:**
  1. SOTA across all three major memory benchmarks simultaneously (claim).
  2. Hybrid search (RAG + Memory) in a single query — not two separate stacks.
  3. Multi-modal extractors built in (OCR / video / AST chunking) without separate pipeline config.

### 2.2 MemoryOS — BAI-LAB/MemoryOS
- **Source:** [README](https://github.com/BAI-LAB/MemoryOS), [paper arXiv 2506.06326](https://arxiv.org/abs/2506.06326), EMNLP 2025 Oral.
- **Pitch:** "Memory operating system for personalized AI agents" — OS-inspired hierarchical storage.
- **Memory tiers:** **short-term**, **mid-term**, **long-term persona** memory; modules = Storage / Updating / Retrieval / Generation.
- **Lifecycle:** "heat" score on mid-term segments; when heat > threshold, content is promoted (user profile insights → long-term user profile, facts → user knowledge, assistant-relevant info → assistant KB).
- **LLM providers:** OpenAI, Anthropic (Claude), Deepseek-R1, Qwen/Qwen3, vLLM, Llama-Factory; all via OpenAI-compatible API + base URL.
- **Embedders:** BAAI/bge-m3, Qwen3-Embedding-0.6B, all-MiniLM-L6-v2 (config via `embedding_model_name`).
- **Storage backends:** in-repo files; ChromaDB integration (`memoryos-chromadb`); Docker.
- **Transports:** PyPI `memoryos-pro`; MCP server (`memoryos-mcp`) with tools `add_memory`, `retrieve_memory`, `get_user_profile`; Docker; Playground (`baijia.online/memoryos`).
- **IDE clients:** Claude Desktop, Cline, Cursor.
- **Benchmarks:** **+49.11% F1**, **+46.18% BLEU-1** on LoCoMo vs baseline. Reproduction scripts published.
- **Roadmap / family:** [Survey on AI Memory](http://github.com/BAI-LAB/Survey-on-AI-Memory), LightSearcher (experiential memory).
- **Differentiators:**
  1. OS-style hierarchical promotion (short → mid → long-term) driven by "heat" scoring.
  2. Separate user-side and assistant-side long-term knowledge bases.
  3. 5x latency reduction via parallelization (per 2025-07-07 release note).

### 2.3 A-MEM (Agentic Memory) — agiresearch/A-mem (+ WujiangXu/A-mem)
- **Source:** [agiresearch README](https://github.com/agiresearch/A-mem), [paper arXiv 2502.12110](https://arxiv.org/abs/2502.12110), [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/119020).
- **Pitch:** "Dynamic memory operations and flexible agent-memory interactions" — Zettelkasten-style notes with agent-driven evolution.
- **Memory model:** notes with content, tags, context, keywords, timestamp; semantic linking via ChromaDB; "memory evolution" updates tags/context/links automatically.
- **Storage:** ChromaDB.
- **LLM backends:** OpenAI (gpt-4o-mini etc.), Ollama.
- **Embedder:** all-MiniLM-L6-v2 default.
- **API:** `add_note`, `read`, `search_agentic`, `update`, `delete`.
- **Eval:** "superior performance vs existing SOTA baselines on six foundation models" (paper); reproduction = separate [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory) repo.
- **Differentiators:**
  1. Zettelkasten principle applied to LLM agent memory: every note has structured attributes + dynamic links.
  2. "Memory evolution" — adding a note can rewrite tags/context of related historical memories.
  3. Pure Python research code, MIT-licensed, very small surface area (good for embedding into custom agents).

### 2.4 HippoRAG / HippoRAG 2 — OSU-NLP-Group/HippoRAG
- **Source:** [README](https://github.com/OSU-NLP-Group/HippoRAG), [HippoRAG 1 arXiv 2405.14831](https://arxiv.org/abs/2405.14831) NeurIPS '24, [HippoRAG 2 arXiv 2502.14802](https://arxiv.org/abs/2502.14802) ICML '25.
- **Pitch:** Neurobiologically-inspired long-term memory framework for LLMs (mirrors hippocampal indexing).
- **Architecture:** OpenIE → knowledge graph → Personalized PageRank for retrieval; embedding store for passages, entities, facts.
- **Embedders:** NV-Embed-v2, GritLM, Contriever (and OpenAI-compatible).
- **LLM backends:** OpenAI, vLLM (Llama-3.1/3.3-Instruct), OpenAI-compatible endpoints; vLLM offline batch mode for >3x faster indexing.
- **API:** `index`, `retrieve`, `rag_qa`; supports incremental updates and document deletion.
- **Datasets / benchmarks:** NaturalQuestions, PopQA, NarrativeQA, MuSiQue, 2Wiki, HotpotQA, LV-Eval — covers factual memory, sense-making, associativity (multi-hop). Beats GraphRAG, RAPTOR, LightRAG in cost/latency.
- **Eval scripts:** `tests_openai.py`, `tests_local.py`, full `reproduce/` suite, [HuggingFace dataset](https://huggingface.co/datasets/osunlp/HippoRAG_2).
- **Differentiators:**
  1. Personalized PageRank over a KG for retrieval (not just dense retrieval).
  2. Online cost / latency comparable to vanilla RAG; offline indexing way cheaper than GraphRAG / RAPTOR.
  3. Three-axis evaluation: factual memory + sense-making + associativity — a novel framing for "true long-term memory".

### 2.5 OpenMemory (CaviraOSS) — CaviraOSS/OpenMemory
- **Source:** [README](https://github.com/CaviraOSS/OpenMemory), [docs](https://openmemory.cavira.app/docs).
- **Pitch:** "Real long-term memory for AI agents. Not RAG. Not a vector DB." — cognitive memory engine, self-hosted, local-first.
- **Memory sectors:** **episodic, semantic, procedural, emotional, reflective**.
- **Lifecycle:** adaptive **decay & reinforcement** instead of TTLs; composite scoring (salience + recency + coactivation).
- **Temporal KG:** `valid_from` / `valid_to`; auto-evolution closes prior facts; confidence decay; **point-in-time queries**; timelines; change detection.
- **Waypoint graph:** associative, traversable links; **explainable recall traces** show which nodes were used.
- **Storage:** SQLite default; Postgres; external DBs via config.
- **Embeddings:** OpenAI, Gemini, Ollama, AWS, synthetic fallback.
- **Transports:** PyPI `openmemory-py`, npm `openmemory-js`, REST (`/api/memory/*`, `/api/temporal/*`), MCP (`/mcp`), Docker, dashboard UI (optional `ui` profile), `opm` CLI, VS Code extension.
- **Connectors:** GitHub, Notion, Google Drive/Sheets/Slides, OneDrive, Web Crawler.
- **Migration tool:** imports from Mem0, Zep, Supermemory, etc.
- **MCP tools:** `openmemory_query`, `openmemory_store`, `openmemory_list`, `openmemory_get`, `openmemory_reinforce`.
- **Note:** Repo header says "currently being fully rewritten — expect breaking changes" (as of fetch date).
- **Differentiators:**
  1. Multi-sector memory taxonomy (episodic / semantic / procedural / emotional / reflective) is unusual outside research papers.
  2. First-class temporal KG with `valid_from`/`valid_to` and point-in-time queries.
  3. Built-in migration from competitor memory systems.

### 2.6 Graphlit — graphlit/graphlit-client-python (closed-source platform)
- **Source:** [client README](https://github.com/graphlit/graphlit-client-python), [graphlit.com](https://www.graphlit.com), [docs.graphlit.dev](https://docs.graphlit.dev), [glossary/knowledge-graph](https://www.graphlit.com/glossary/knowledge-graph), [blog/graphlit-mcp-server](https://www.graphlit.com/blog/graphlit-mcp-server).
- **Pitch:** Cloud-native semantic memory platform — one API for content ingestion, extraction, enrichment, storage, retrieval. "Knowledge fabric."
- **Storage:** hybrid system — vector DB (Azure AI Search, Pinecone, Qdrant w/ HNSW) + cloud object storage + graph database.
- **Knowledge graph:** entity-to-content + entity-to-entity; multi-hop reasoning; queryable connected knowledge.
- **Ingestion:** PDFs, emails, meeting transcripts, Slack, audio transcripts, video, web pages, API data.
- **Auth / scoping:** `GRAPHLIT_ENVIRONMENT_ID`, `GRAPHLIT_ORGANIZATION_ID`, `GRAPHLIT_JWT_SECRET`.
- **Transports:** Python client (open source MIT), JS/TS clients, GraphQL API, **open-source MCP server** for Claude Desktop, Goose, Cline, Cursor, Windsurf.
- **Differentiators:**
  1. Knowledge fabric framing — KG is the storage primitive, not an add-on.
  2. Tightly integrated multi-modal ingestion pipeline (audio/video transcription baked into the platform).
  3. GraphQL API as the primary surface (most peers use REST/MCP only).

### 2.7 mcp-memory-service — doobidoo/mcp-memory-service
- **Source:** [README](https://github.com/doobidoo/mcp-memory-service).
- **Pitch:** Open-source persistent shared memory for multi-agent systems — REST API + KG + autonomous consolidation; works with LangGraph, CrewAI, AutoGen, Claude Desktop, OpenCode.
- **Knowledge graph:** typed edges (causes, fixes, contradicts).
- **Auto-consolidation:** decay + compression of old memories.
- **Search:** hybrid (BM25 + vector); semantic + keyword + hybrid w/ metadata filters.
- **Embeddings:** ONNX, **local — never leaves infra**.
- **Storage backends:** SQLite (default), Cloudflare (cloud sync), Hybrid (5 ms local + background cloud sync), Milvus (Lite / self-hosted / Zilliz Cloud).
- **Transports:** REST (15 endpoints), MCP stdio, Streamable HTTP (SSE deprecated), OAuth 2.0 + DCR, Cloudflare Tunnel for browser claude.ai.
- **Multi-agent features:** `X-Agent-ID` header auto-tags by agent identity; `conversation_id` to bypass dedup; SSE events on store/delete; tags-as-inter-agent-bus pattern.
- **Web dashboard:** semantic search, tag browser, document ingestion, analytics, quality scoring.
- **IDE/agent compat:** LangGraph, CrewAI, AutoGen, Claude Desktop, Claude Code, OpenCode, Cursor, Windsurf, Gemini CLI, Codex CLI, Goose, Aider, Copilot CLI, Amp, Continue, Zed, Cody, JetBrains, Raycast, Replit, Sourcegraph, Qodo, ChatGPT Developer Mode, claude.ai (Remote MCP).
- **Spec compat:** [SHODH Unified Memory API v1.0.0](https://github.com/varun29ankuS/shodh-memory).
- **Benchmarks:** LongMemEval R@5 86.0% (session-level) / 80.4% (turn-level); critical comparison vs MemPalace.
- **Differentiators:**
  1. Native browser Remote MCP for claude.ai (not just desktop) with OAuth 2.0 + DCR.
  2. Typed-edge knowledge graph (causes / fixes / contradicts) — rare.
  3. SSE event stream for real-time multi-agent coordination ("tags as inter-agent signals" pattern).

### 2.8 Redis Agent Memory Server — redis/agent-memory-server
- **Source:** [README](https://github.com/redis/agent-memory-server), [docs site](https://redis.github.io/agent-memory-server/).
- **Pitch:** Memory layer for AI agents on Redis — REST + MCP, working memory + long-term memory.
- **Two-tier model:** working memory (session-scoped) + long-term memory (persistent).
- **Memory strategies:** discrete, summary, preferences, custom (configurable extraction).
- **Search:** semantic (vector), keyword (full-text), hybrid; metadata filtering.
- **Backends:** Redis (RedisVL); pluggable vector DB factory.
- **LLM providers:** OpenAI, Anthropic, AWS Bedrock, Ollama, Azure, Gemini — via [LiteLLM](https://docs.litellm.ai/) (100+ providers).
- **AI features:** automatic topic extraction, entity recognition, conversation summarization, deduplication.
- **Transports:** PyPI `agent-memory-client`, REST API, MCP (stdio + SSE), Docker (`redislabs/agent-memory-server`), GHCR. Auth: OAuth2/JWT (production).
- **Task backends:** asyncio (dev) and Docket (prod, separate worker container); Redis Cluster supported.
- **LangChain integration:** `get_memory_tools` auto-converts memory client tools to LangChain tools.
- **Differentiators:**
  1. First-class working-memory vs long-term-memory split as the architectural primitive.
  2. Configurable memory **strategies** (discrete / summary / preferences / custom) — declarative extraction policy.
  3. Production-grade Redis + Docket task worker model + 100+ LLM providers via LiteLLM.

### 2.9 Pieces for Developers — LTM-2 / 2.5 / 2.7 (closed source)
- **Source:** [pieces.app/features/long-term-memory](https://pieces.app/features/long-term-memory), [docs.pieces.app — LTM engine](https://docs.pieces.app/products/core-dependencies/pieces-os/long-term-memory), [blog: LTM-2 announce](https://pieces.app/blog/what-is-new-ltm-2), [blog: nano-models LTM-2.5](https://pieces.app/blog/nano-models), [MCP](https://pieces.app/features/mcp).
- **Pitch:** "Infinite Artificial Memory for your Digital Workers and Agents" — OS-wide passive workflow memory.
- **Capture:** **every application you use** — code copied, screens viewed, audio heard.
- **Retention:** 9 months of detailed workflow history (LTM-2).
- **Privacy:** captures, indexes, and **encrypts data locally**; nothing uploaded unless you explicitly send a prompt to a cloud LLM.
- **Architecture:** RAG-based, runs entirely on PiecesOS (on-device); LTM-2.5 introduces ~11 on-device **nano-models** (knowledge-distilled, quantized, pruned) — including a temporal-understanding nano-model.
- **Access control:** opt in/out per directory or project source.
- **Surfaces:** Timeline view, Conversational Search, MCP integration with IDEs, ability to extract URLs/snippets and bootstrap new chats from stored summaries.
- **Roadmap (LTM-2.5 / 2.7):** dynamic summary generation tailored to topics / tags / time ranges (not fixed intervals).
- **Differentiators:**
  1. OS-wide passive capture (apps, screens, audio) — no other tool in this catalog does this.
  2. On-device **nano-models** for memory tasks (temporal etc.) — not just embeddings, actual specialized small models.
  3. Closed-source **but** local-first encryption with explicit upload gating.

### 2.10 ReMe (AgentScope) — agentscope-ai/ReMe
- **Source:** [README](https://github.com/agentscope-ai/ReMe).
- **Pitch:** Memory management toolkit for AI agents — "Remember Me, Refine Me." File-based + vector-based modes.
- **ReMeLight (file-based):** memory as files (Markdown). Layout = `MEMORY.md` (long-term), `memory/YYYY-MM-DD.md` (daily), `dialog/YYYY-MM-DD.jsonl` (raw), `tool_result/<uuid>.txt` (long tool outputs w/ N-day TTL).
- **Components:** `ContextChecker`, `Compactor` (ReActAgent w/ structured summaries), `ToolResultCompactor`, `Summarizer` (ReActAgent with file `read`/`write`/`edit` tools), `MemorySearch` (vector + BM25 hybrid).
- **API:** `check_context`, `compact_memory`, `compact_tool_result`, `pre_reasoning_hook`, `summary_memory`, `memory_search`, `get_in_memory_memory`, `await_summary_tasks`.
- **LLMs / embeddings:** Qwen + OpenAI-compatible via `LLM_BASE_URL` / `EMBEDDING_BASE_URL` (e.g. DashScope).
- **Compression demo:** 223,838 tokens → 1,105 tokens (99.5%) on test trace.
- **Vector mode:** also exists ("vector-based memory system") — separate path.
- **Benchmarks:** "SOTA on LoCoMo and HaluMem" (claim); details in the repo's experimental section.
- **Differentiators:**
  1. Memory-as-files (Markdown + JSONL) — readable, editable, copyable; trivial migration.
  2. ReActAgent-driven compaction & summarization — the memory engine itself uses an LLM agent loop.
  3. Tool-result TTL bucket isolating noisy long tool outputs from main history.

### 2.11 Memori (MemoriLabs) — MemoriLabs/Memori
- **Source:** [README](https://github.com/MemoriLabs/Memori), [memorilabs.ai/docs](https://memorilabs.ai/docs/memori-cloud/), [paper arXiv 2603.19935](https://arxiv.org/abs/2603.19935).
- **Pitch:** "Memory from what agents do, not just what they say." Agent-native memory infrastructure — LLM, datastore, framework agnostic.
- **Modes:** Memori Cloud (managed) + BYODB (bring your own DB) — Python and TypeScript SDKs.
- **Attribution model:** every interaction tagged with `entity_id` (person/place/thing) and `process_id` (your agent/program); without attribution, no memories are made.
- **Sessions:** automatic; manual `new_session()` / `set_session(id)` available.
- **LLM support:** Anthropic, Bedrock, DeepSeek, Gemini, Grok (xAI), OpenAI Chat Completions + Responses API. Streaming + async + sync.
- **Frameworks:** Agno (and more — list truncated in fetched README).
- **Transports:** PyPI `memori`, npm `@memorilabs/memori`, REST, MCP (`api.memorilabs.ai/mcp/`), OpenClaw plugin (`@memorilabs/openclaw-memori`).
- **MCP IDE integrations:** Claude Code, Cursor, Codex, Warp, Antigravity.
- **Benchmark:** **LoCoMo 81.95%** accuracy with avg 1,294 tokens/query — 4.97% of full-context footprint; ~67% prompt-size reduction vs Zep, >20× context-cost reduction vs full-context. Outperforms Zep, LangMem, Mem0 (per [paper](https://arxiv.org/abs/2603.19935)).
- **Surfaces:** Dashboard at `app.memorilabs.ai` (Memories, Analytics, Playground, API Keys).
- **Differentiators:**
  1. Mandatory `entity_id` + `process_id` attribution model — explicit, structured, multi-tenant by default.
  2. Drop-in `register(client)` over OpenAI/etc. SDK — auto-persists and auto-recalls in background.
  3. Quantified token-savings benchmark (4.97% of full-context cost on LoCoMo).

### 2.12 agentmemory — rohitg00/agentmemory
- **Source:** [README](https://github.com/rohitg00/agentmemory), benchmarks `benchmark/LONGMEMEVAL.md`, `benchmark/QUALITY.md`, `benchmark/SCALE.md`, `benchmark/COMPARISON.md`.
- **Pitch:** "Persistent memory for AI coding agents based on real-world benchmarks" — Karpathy's LLM Wiki pattern + confidence scoring + lifecycle + KG + hybrid search.
- **Auto-capture:** **12 hooks** (zero manual effort).
- **Search:** **BM25 + Vector + Graph** with **Reciprocal Rank Fusion (RRF)**.
- **Storage:** built-in (no external DBs).
- **Embedder:** all-MiniLM-L6-v2 (local, free).
- **Surface area:** **51 MCP tools**, **104 REST endpoints**, AgentSDKProvider, real-time viewer, **iii Console** (trace-level engine inspection), filesystem connector `@agentmemory/fs-watcher`.
- **Multi-agent coordination:** MCP + REST + leases + signals.
- **Compatible agents:** Claude Code (12 hooks + MCP + skills), OpenClaw, Hermes, Cursor, Gemini CLI, OpenCode, Codex CLI, Cline, Goose, Kilo Code, Aider, Claude Desktop, Windsurf, Roo Code, Claude SDK.
- **Audit:** "audit policy codified across every delete path."
- **Benchmarks:** **LongMemEval-S R@5 95.2%, R@10 98.6%, MRR 88.2%**; comparison vs mem0 (53k★, 68.5% R@5 LoCoMo), Letta/MemGPT (22k★, 83.2% R@5 LoCoMo), CLAUDE.md baselines. Token savings: ~170K tokens/year vs ~650K (LLM-summarized) vs 19.5M+ (full paste).
- **Differentiators:**
  1. RRF fusion of BM25 + Vector + Graph in one retrieval pipeline (most rivals do at most two of these).
  2. 12 deterministic hooks for auto-capture — no `add()` calls needed.
  3. iii Console for trace-level engine inspection — explicit observability story.

### 2.13 Engram — Gentleman-Programming/engram
- **Source:** [README](https://github.com/Gentleman-Programming/engram), [DOCS.md](https://github.com/Gentleman-Programming/engram/blob/main/DOCS.md), [architecture](https://github.com/Gentleman-Programming/engram/blob/main/docs/ARCHITECTURE.md).
- **Pitch:** "Persistent memory for AI coding agents — Agent-agnostic. Single binary. Zero dependencies." Go binary; SQLite + FTS5.
- **Surfaces:** CLI, HTTP API (port 7437), MCP server (stdio), interactive **TUI** (Catppuccin Mocha, vim keys), opt-in cloud.
- **Storage:** single SQLite file at `~/.engram/engram.db`; FTS5 full-text search.
- **MCP tools (17):** save (`mem_save`, `mem_update`, `mem_delete`, `mem_suggest_topic_key`), search (`mem_search`, `mem_context`, `mem_timeline`, `mem_get_observation`), session lifecycle (`mem_session_start`, `_end`, `_summary`), conflict surfacing (`mem_judge`), utilities (`mem_save_prompt`, `mem_stats`, `mem_capture_passive`, `mem_merge_projects`, `mem_current_project`).
- **Memory schema:** title, type, **What/Why/Where/Learned**.
- **Cloud (opt-in):** project-scoped only (`--project` required); upgrade flow with `doctor` / `repair` / `bootstrap` / `status`.
- **Sync:** git-based (compressed chunks, no merge conflicts) — local SQLite stays authoritative.
- **Agent integrations:** Claude Code (plugin marketplace), OpenCode, Gemini CLI, Codex, VS Code (Copilot), Antigravity, Cursor, Windsurf, "any MCP client".
- **Beta:** Obsidian export (memories as Obsidian KG).
- **Differentiators:**
  1. Single Go binary, **zero dependencies** (no Node/Python/Docker required).
  2. Native TUI — none of the other apps in this catalog ship one.
  3. Explicit conflict surfacing tool (`mem_judge`) plus chronological timeline (`mem_timeline`).

### 2.14 claude-mem (bonus — viral 2026) — thedotmack/claude-mem
- **Source:** [README](https://github.com/thedotmack/claude-mem), [docs.claude-mem.ai](https://docs.claude-mem.ai), [claude-mem.ai](https://claude-mem.ai/).
- **Pitch:** Claude Code plugin that captures everything Claude does, compresses with Claude's agent-sdk, injects context into next session.
- **Hooks:** 5 lifecycle hooks — SessionStart → UserPromptSubmit → PostToolUse → Summary → SessionEnd.
- **Architecture:** TypeScript ESM hooks + Worker Service (Express :37777, Bun-managed) for async AI processing.
- **Storage:** SQLite at `~/.claude-mem/claude-mem.db` + ChromaDB vector store.
- **Install:** `/plugin marketplace add thedotmack/claude-mem` then `/plugin install claude-mem`.
- **Stars (notable):** 69,130 — most-starred memory tool in this catalog by a wide margin.
- **Scope:** Claude Code only (not agent-agnostic).
- **Differentiators:**
  1. Uses Claude's agent-sdk *itself* to compress past sessions.
  2. Worker service architecture (Express on `:37777`) for asynchronous compression vs blocking the chat.
  3. The viral "set-and-forget" UX (one-line install + zero config) drove its breakout star count.

---

## 3. Novel features in this cohort (vs mainstream Mem0/Zep/MemGPT/etc.)

These are capabilities that are **distinctive to Part-B apps** and largely absent from mainstream Part-A memory systems:

1. **OS-wide passive multi-modal capture** (Pieces LTM-2 / 2.5 / 2.7) — captures every app, screen, and audio source, not just LLM I/O. No mainstream peer ships this.
2. **On-device nano-models** for memory subtasks (Pieces LTM-2.5) — temporal understanding offloaded to distilled small models, not just embeddings.
3. **Five-sector cognitive memory taxonomy** (OpenMemory) — episodic, semantic, procedural, emotional, reflective. Mainstream tools usually expose only "long-term" + "short-term".
4. **Waypoint graph + explainable recall traces** (OpenMemory) — UI-level "why was this recalled?" provenance.
5. **Memory-as-files** (ReMe ReMeLight) — Markdown + JSONL on disk, human-readable, copyable, directly editable; an explicit anti-DB stance.
6. **ReActAgent-driven compaction / summarization** (ReMe) — the memory layer itself runs an LLM agent loop with `read`/`write`/`edit` file tools.
7. **Personalized PageRank over a KG for retrieval** (HippoRAG 2) — neurobiologically-motivated, not dense-only.
8. **Three-axis evaluation** (HippoRAG 2): factual memory + sense-making + associativity. Most peers only report fact recall.
9. **Reciprocal Rank Fusion of BM25 + Vector + Graph** (agentmemory) — true tri-modal retrieval pipeline; mainstream peers usually fuse at most two.
10. **51 MCP tools / 104 REST endpoints + iii Console** (agentmemory) — observability and surface-area depth far beyond peer memory tools.
11. **Single-binary Go deployment with native TUI** (Engram) — zero runtime dependencies; vim-keys terminal UI; not seen elsewhere in this space.
12. **Mandatory entity/process attribution** (Memori) — every memory must be tied to an `entity_id` + `process_id`, baking multi-tenant and multi-agent into the schema.
13. **Native browser Remote MCP via Cloudflare Tunnel for claude.ai** (mcp-memory-service) — most MCP memory servers are desktop-only.
14. **Tags-as-inter-agent-bus pattern** (mcp-memory-service) — the storage layer doubles as an inter-agent messaging substrate via sentinel tags + SSE notifications, with no extra protocol.
15. **Typed knowledge-graph edges** (`causes` / `fixes` / `contradicts`) — mcp-memory-service. Most KG-memory systems treat edges as untyped or generic.
16. **Heat-driven hierarchical promotion** (MemoryOS) — short → mid → long-term moves on accumulated "heat" score, OS-paging analogy.
17. **Distinct user-side and assistant-side long-term knowledge bases** (MemoryOS) — separate stores, not just one user store.
18. **Built-in cross-vendor migration tool** (OpenMemory) — imports Mem0, Zep, Supermemory dumps. None of the mainstream tools advertise this.
19. **MemoryBench eval framework as an installable Agent skill** (Supermemory) — `npx skills add supermemoryai/memorybench` + `/benchmark-context` runs it for you.
20. **Tool-result TTL bucket** (ReMe) — long tool outputs siphoned to `tool_result/<uuid>.txt` with N-day TTL, keeping main history clean.
21. **`mem_judge` conflict-surfacing MCP tool** (Engram) — explicit dedicated tool for surfacing contradictions, vs peers that resolve silently.
22. **Git-based memory sync via compressed chunks** (Engram) — versioned, mergeable memory across machines via plain git, with no merge conflicts and no big files.
23. **Zettelkasten-inspired agentic note evolution** (A-MEM) — adding a note can rewrite tags/context of *historical* notes via agent decisions.
24. **Pluggable memory strategies** (Redis Agent Memory Server) — `discrete`, `summary`, `preferences`, `custom` extraction policies as a configurable primitive.
25. **Hybrid 5ms-local + background-cloud-sync backend** (mcp-memory-service Hybrid) — local SQLite as authority + async Cloudflare replication as a single backend mode.

---

## 4. Caveats

- Star counts and timestamps captured 2026-04-28 from the GitHub REST API; numbers move daily.
- For Pieces LTM-2 (closed source) all claims come from vendor blog/docs and are not independently verified.
- Some apps (notably OpenMemory) note an ongoing rewrite — feature surface may diverge from the README in the near term.
- A-MEM has two parallel repos (`agiresearch/A-mem` library, `WujiangXu/A-mem` paper-reproduction); features above are from the library README.
- Where a feature is `?`, it means it was not declared in the public README/docs we fetched, **not** that it is absent.
