# Memory Apps Feature Catalogue — Part A (Mainstream / Top 10 GitHub)

**Goal.** Catalogue every declared feature of the 10 most prominent open-source AI memory projects so we can compare them side-by-side against `memoirs`.

**Method.** All claims below are pulled from the project's own `README`, official docs, or release notes. Every cell in the matrix and every bullet has a source URL at the end of the relevant section. No claims are inferred from training data — only from material fetched on 2026-04-28 via `gh api` and `WebSearch`.

**Conventions.**
- ✓ = explicitly declared / supported in docs.
- ✗ = explicitly absent, deprecated, or not mentioned in any README/doc consulted.
- partial = supported but with caveats (e.g. only via integration, only in cloud tier, only one provider, etc.).
- "?" = could not be confirmed in the time-boxed research window.

**Important repo notes.**
- `cpacker/MemGPT` now redirects (HTTP) to `letta-ai/letta`. The legacy MemGPT codebase lives inside the Letta repo; the *research artifact* (paper + original architecture) is treated below as a separate row called "MemGPT (legacy/research)" because its memory model is conceptually distinct and is still cited by every other project.
- `getzep/zep` is currently described in the README as a wrapper of examples + integrations; the *Community Edition has been deprecated* and Zep Cloud is closed. The OSS engine that powers it (Graphiti) is the real comparable artifact, so we list both.
- `GibsonAI/memori` redirects to `MemoriLabs/Memori` — same project, renamed.
- `getmetal/motorhead` is explicitly DEPRECATED in its README banner — kept here because it's still the most cited Redis-based memory server.

---

## 1. Comparative Feature Matrix

> Columns: M0 = Mem0 · Zep = Zep Cloud (the OSS wrapper repo) · GR = Graphiti (Zep's OSS engine) · Lt = Letta · Cg = Cognee · MG = MemGPT legacy/research · LM = LangMem · LI = LlamaIndex Memory · Mi = Memori · Mp = Memoripy · Mh = Motorhead

### 1.1 Identity

| Feature                     | M0      | Zep     | GR      | Lt      | Cg      | MG (legacy) | LM     | LI      | Mi      | Mp     | Mh    |
|-----------------------------|---------|---------|---------|---------|---------|-------------|--------|---------|---------|--------|-------|
| OSS license (Apache/MIT)    | Apache  | Apache  | Apache  | Apache  | Apache  | Apache      | MIT    | MIT     | Apache  | Apache | Apache|
| Actively maintained (2026)  | ✓       | partial | ✓       | ✓       | ✓       | ✗ (→Letta)  | ✓      | ✓       | ✓       | partial| ✗     |
| Stars (approx, 2026-04-28)  | 54k     | 4.5k    | 25.5k   | 22.4k   | 16.9k   | (=Letta)    | 1.4k   | 49k     | 14k     | 690    | 0.9k  |
| Has paid hosted tier        | ✓       | ✓       | (Zep)   | ✓       | ✓       | n/a         | (LangSmith) | ✗ | ✓       | ✗      | ✗     |
| Cloud SaaS                  | ✓       | ✓       | ✗       | ✓       | ✓       | n/a         | partial| ✗       | ✓       | ✗      | ✗     |
| Dedicated SDK               | py+ts   | py+ts+go| py      | py+ts   | py      | py          | py     | py+ts   | py+ts   | py     | rest  |

### 1.2 Memory taxonomy declared

| Memory type            | M0    | Zep   | GR  | Lt   | Cg   | MG   | LM   | LI   | Mi   | Mp   | Mh   |
|------------------------|-------|-------|-----|------|------|------|------|------|------|------|------|
| Semantic (facts)       | ✓     | ✓     | ✓   | ✓    | ✓    | ✓    | ✓    | ✓    | ✓    | ✓    | ✗    |
| Episodic               | ✓     | ✓     | ✓ (episodes) | partial | ✓ | partial | ✓    | partial | ✓ | ✓    | ✗    |
| Procedural             | ✗     | ✗     | ✗   | ✗    | partial | ✗ | ✓    | ✗    | partial (rules/skills) | ✗ | ✗ |
| Working / short-term   | ✓ (session) | ✓ | ✗ | ✓ (core mem) | ✓ (session) | ✓ (main ctx) | ✓ | ✓ (FIFO) | ✓ (session) | ✓ | ✓ |
| Sensory                | ✗     | ✗     | ✗   | ✗    | ✗    | ✗    | ✗    | ✗    | ✗    | ✗    | ✗    |
| Preference / profile   | ✓     | ✓     | ✓   | ✓ (persona) | ✓ | ✓ (persona block) | ✓ (profile) | ✓ (Static) | ✓ | ✓ | ✗ |
| Fact extraction        | ✓     | ✓     | ✓   | partial | ✓ | partial | ✓    | ✓ (FactExtractionMemoryBlock) | ✓ | ✓ | ✗ |
| Multi-level / tiered   | ✓ (user/session/agent/org) | ✓ | ✗ | ✓ (core/recall/archival) | ✓ | ✓ (main/external) | ✓ | ✓ (short/long) | ✓ (entity/process/session) | ✓ (short/long) | ✓ (window+long) |
| Rules / skills         | ✗     | ✗     | ✗   | ✗    | ✗    | ✗    | ✓ (procedural) | ✗ | ✓ (rules,skills) | ✗ | ✗ |
| Tool-call memory       | ✓     | ✗     | ✗   | ✓    | ✓ (Claude Code plugin) | ✗ | ✗ | ✗ | partial | ✗ | ✗ |

### 1.3 Storage backends

| Backend                | M0    | Zep | GR  | Lt   | Cg   | MG   | LM   | LI   | Mi   | Mp   | Mh   |
|------------------------|-------|-----|-----|------|------|------|------|------|------|------|------|
| SQLite                 | partial | ✗ | ✗   | ✓    | ✓ (default) | ✓ | partial | partial | ✓ | ✗ | ✗ |
| Postgres               | ✓     | (cloud) | ✗ | ✓ | ✓ | ✓ | ✓ (AsyncPostgresStore) | ✓ (via VS) | ✓ | ✗ | ✗ |
| MySQL                  | ✗     | ✗   | ✗   | ✗    | ✗    | ✗    | ✗    | ✗    | ✓    | ✗    | ✗    |
| MongoDB                | ✓ (Atlas) | ✗ | ✗ | ✗ | partial | ✗ | ✗ | ✓    | ✓    | ✗    | ✗    |
| Redis (cache/vec)      | ✓     | ✗   | ✗   | ✗    | ✓    | ✓    | ✗    | ✓    | ✗    | ✗    | ✓ (required) |
| JSON file              | ✗     | ✗   | ✗   | ✗    | ✗    | ✓    | ✗    | ✓    | ✗    | ✓    | ✗    |
| Qdrant                 | ✓ (default) | ✗ | (via Falkor/Kuzu/Neo4j) | partial | ✓ | ✓ | partial (any vec store) | ✓ | partial | ✗ | ✗ |
| Pinecone               | ✓     | ✗   | ✗   | ✗    | partial | ✓ | ✓    | ✓    | ✗    | ✗    | ✗    |
| Chroma                 | ✓     | ✗   | ✗   | ✗    | ✓    | ✓    | ✓    | ✓    | ✗    | ✗    | ✗    |
| FAISS                  | ✓     | ✗   | ✗   | ✗    | ✗    | ✓    | partial | ✓ | ✗    | ✓    | ✗    |
| pgvector               | ✓     | ✗   | ✗   | ✓ (default) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| LanceDB                | ✗     | ✗   | ✗   | ✗    | ✓ (default) | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Weaviate               | ✓     | ✗   | ✗   | ✗    | ✗    | ✓    | ✗    | ✓    | ✗    | ✗    | ✗    |
| Milvus                 | ✓     | ✗   | ✗   | ✗    | ✗    | ✓    | ✗    | ✓    | ✗    | ✗    | ✗    |
| Elasticsearch          | ✓     | ✗   | ✗   | ✗    | ✗    | ✗    | ✗    | ✓    | ✗    | ✗    | ✗    |
| OpenSearch             | ✓     | ✗   | ✓ (Neptune fts backend) | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Azure AI Search        | ✓     | ✗   | ✗   | ✗    | ✗    | ✗    | ✗    | ✓    | ✗    | ✗    | ✗    |
| Vertex AI Vec Search   | ✓     | ✗   | ✗   | ✗    | ✗    | ✗    | ✗    | ✓    | ✗    | ✗    | ✗    |
| Supabase               | ✓     | ✗   | ✗   | ✗    | ✗    | ✗    | ✗    | ✓    | ✓ (pg-compat) | ✗ | ✗ |
| Upstash Vector         | ✓     | ✗   | ✗   | ✗    | ✗    | ✗    | ✗    | ✓    | ✗    | ✗    | ✗    |
| Neo4j (graph)          | ✗ (removed v2) | (via GR) | ✓ | ✗ | ✓ | ✗ | ✗ | partial (via integ) | ✗ | ✗ | ✗ |
| Memgraph (graph)       | ✗ (removed v2) | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| FalkorDB (graph)       | ✗     | ✗   | ✓   | ✗    | ✓    | ✗    | ✗    | ✗    | ✗    | ✗    | ✗    |
| Kuzu (graph)           | ✗ (removed v2) | ✗ | ✓ | ✗ | ✓ (default) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Amazon Neptune         | ✗     | ✗   | ✓   | ✗    | ✓    | ✗    | ✗    | ✗    | ✗    | ✗    | ✗    |
| Apache AGE             | ✗ (removed v2) | ✗ | ✗ | ✗ | partial | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

### 1.4 LLM providers

| Provider               | M0  | Zep | GR  | Lt  | Cg  | MG  | LM  | LI  | Mi  | Mp  | Mh  |
|------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| OpenAI                 | ✓   | ✓   | ✓ (default) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Anthropic              | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓ (default in README) | ✓ | ✓ | ✗ | ✗ |
| Google Gemini          | ✓   | ✓   | ✓   | ✓ (google_ai/google_vertex) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Azure OpenAI           | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✗   | ✓   | ✓ |
| Groq                   | ✓   | ✗   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✗   | ✗   | ✗ |
| DeepSeek               | ✓   | ✗   | ✗   | ✓   | ✓   | ✗   | ✓   | ✓   | ✓   | ✗   | ✗ |
| Bedrock                | ✓   | ✗   | ✗   | partial | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Ollama (local)         | ✓   | ✗   | ✓ (via OpenAIGenericClient) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| OpenRouter             | ✓   | ✗   | ✗   | ✗   | ✓ (custom) | ✗ | partial | ✓ | ✗ | ✓ | ✗ |
| Grok / xAI             | ✓   | ✗   | ✗   | ✗   | ✗   | ✗   | partial | ✓ | ✓ | ✗ | ✗ |
| vLLM / custom OpenAI-compat | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ |

### 1.5 Embedders

| Embedder               | M0  | Zep | GR  | Lt  | Cg  | MG  | LM  | LI  | Mi  | Mp  | Mh  |
|------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| OpenAI text-embedding-3 | ✓ (default) | ✓ | ✓ (default) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Voyage                 | ✓   | ✗   | ✓   | ✗   | ✓   | ✗   | ✗   | ✓   | ✗   | ✗   | ✗ |
| Cohere                 | ✓   | ✗   | ✗   | ✓   | ✓   | ✗   | ✗   | ✓   | ✗   | ✗   | ✗ |
| HuggingFace            | ✓   | ✗   | ✗   | ✓   | ✓   | ✓   | ✗   | ✓   | ✗   | ✗   | ✗ |
| Ollama embeddings      | ✓ (Qwen/nomic) | ✗ | ✓ (nomic-embed-text) | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | ✗ |
| Gemini embeddings      | ✓   | ✓   | ✓   | ✓   | ✓   | ✗   | ✗   | ✓   | ✗   | ✗   | ✗ |

### 1.6 Retrieval modes

| Retrieval mode             | M0  | Zep | GR  | Lt  | Cg  | MG  | LM  | LI  | Mi  | Mp  | Mh  |
|----------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Semantic (vector)          | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓ |
| BM25 / keyword             | ✓ (multi-signal v3) | ✓ | ✓ | ✗ | partial | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Hybrid (vec + keyword)     | ✓   | ✓   | ✓   | ✗   | ✓   | ✗   | ✗   | ✓   | ✓ (combined mode) | ✗ | ✗ |
| Graph traversal            | partial (entity-link v3) | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | partial (knowledge graph index) | ✓ (entity rel) | ✓ (concept graph) | ✗ |
| Multi-hop                  | ✓ (LoCoMo +23.1) | ✓ | ✓ | partial | ✓ | partial | ✗ | partial | ✓ | partial | ✗ |
| Time-aware / temporal      | ✓ (LoCoMo +29.6) | ✓ | ✓ (bi-temporal valid_at/invalid_at) | ✗ | ✓ (temporal_cognify) | ✗ | ✗ | ✗ | partial | partial (decay) | ✗ |
| Bi-temporal queries        | ✗   | ✓   | ✓   | ✗   | partial | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Reranker / cross-encoder   | ✗   | ✓   | ✓ (OpenAI/Gemini reranker) | ✗ | partial | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| Spreading activation       | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✓   | ✗ |
| Auto-routing recall        | ✗   | ✗   | ✗   | ✗   | ✓ (`recall` auto-picks) | ✗ | ✗ | ✗ | ✓ (Auto Mode) | ✗ | ✗ |
| Streaming retrieval        | ✗   | ?   | ✗   | ✗   | ✗   | ✗   | ✗   | partial | ✓ (stream chat) | ✗ | ✗ |

### 1.7 Lifecycle features

| Feature                     | M0  | Zep | GR  | Lt  | Cg  | MG  | LM  | LI  | Mi  | Mp  | Mh  |
|-----------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Auto summarization          | ✗   | ✓   | partial (entity summaries) | ✓ (recursive) | ✓ | ✓ (when ctx full) | ✓ | ✓ (ChatSummaryMemoryBuffer) | ✓ | ✗ | ✓ (incremental) |
| Decay (time-based)          | ✗   | partial | ✓ (validity windows) | ✗ | partial | ✗ | ✗ | ✗ | partial (retention_policy) | ✓ | ✗ |
| Reinforcement (usage)       | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✓   | ✗ |
| Consolidation               | partial | ✓ | ✓ (community detection / entity merge) | ✓ (sleep-time) | ✓ (improve) | partial | ✓ (background mgr) | ✓ (flush) | ✓ (background augmentation) | ✓ (clustering) | ✗ |
| Deduplication               | ✓ (entity linking) | ✓ | ✓ (autonomous) | partial | ✓ (add_data_points) | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ |
| Versioning                  | ✓ (audit logs)  | ✓ (valid_at/invalid_at) | ✓ | partial | partial | ✗ | ✗ | ✗ | partial | ✗ | ✗ |
| Expire / archive            | ✗   | ✓   | ✓ (invalidation, never deleted) | ✓ (archival tier) | ✓ (`forget`) | ✓ (archival) | ✗ | partial (priority levels) | ✓ (retention_policy) | partial | ✓ (DELETE) |
| Conflict resolution         | ✓ (ADD-only v3) | ✓ (auto-invalidate) | ✓ (auto-invalidate) | partial (agent-driven) | ✓ (knowledge updates) | ✗ | ✓ | ✗ | partial | ✗ | ✗ |
| Background memory manager   | ✗ | ✓ | ✓ | ✓ (sleep-time agents) | partial (`improve`) | ✗ | ✓ | partial (flush) | ✓ (advanced augment) | ✗ | ✓ |

### 1.8 Knowledge graph

| Feature                     | M0  | Zep | GR  | Lt  | Cg  | MG  | LM  | LI  | Mi  | Mp  | Mh  |
|-----------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Has KG                      | partial (entity linking, removed graph DB) | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | partial (KG index in core LI) | partial (entity rel) | partial (concept graph) | ✗ |
| Custom entity types         | ✗   | ✓ (Pydantic) | ✓ (Pydantic) | ✗ | ✓ (ontology RDF/OWL) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Learned ontology            | ✗   | ✓   | ✓   | ✗   | ✓   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗ |
| Provenance (source episodes)| ✗   | ✓   | ✓ (episodes ground every fact) | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Community / cluster nodes   | ✗   | ✓   | ✓   | ✗   | partial | ✗ | ✗ | ✗ | ✗ | ✓ (hier. clustering) | ✗ |
| Graph traversal in retrieval| partial (entity boost) | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | partial | ✓ | ✓ | ✗ |

### 1.9 Multi-tenancy / scoping / security

| Feature                     | M0  | Zep | GR  | Lt  | Cg  | MG  | LM  | LI  | Mi  | Mp  | Mh  |
|-----------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Per-user scoping            | ✓ (user_id) | ✓ | ✓ (group_id) | ✓ (per agent) | ✓ (tenant) | ✓ (per agent) | ✓ (namespace) | ✓ | ✓ (entity_id) | partial | ✓ (session_id) |
| Multi-agent / process scope | ✓ (agent_id) | ✓ | ✓ | ✓ (agent_id, shared blocks) | ✓ | partial | ✓ | partial | ✓ (process_id) | ✗ | ✗ |
| Org / team scope            | ✓ (organizational mem) | ✓ | partial | ✓ (orgs) | ✓ | ✗ | ✗ | ✗ | partial | ✗ | ✗ |
| ACL / permissions           | ✓ (admin API key, pause/revoke per app) | ✓ (cloud) | ✗ | ✓ (org) | partial (tenant isolation) | ✗ | ✗ | ✗ | partial (API key) | ✗ | ✗ |
| At-rest encryption          | partial (cloud) | ✓ (SOC2/HIPAA) | ✗ | ✓ (cloud) | partial (tenant) | ✗ | ✗ | ✗ | ✓ (cloud) | ✗ | ✗ |
| In-transit encryption (HTTPS)| ✓ | ✓ | ✓ (uses driver TLS) | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ |
| SOC2 / HIPAA compliance     | ✗ (cloud only?) | ✓ (Type 2 + HIPAA) | ✗ | partial (cloud) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

### 1.10 Transports / integrations

| Feature                     | M0  | Zep | GR  | Lt  | Cg  | MG  | LM  | LI  | Mi  | Mp  | Mh  |
|-----------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Python lib                  | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✗ |
| TypeScript / JS SDK         | ✓   | ✓   | ✗   | ✓   | partial | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ |
| Go SDK                      | ✗   | ✓   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗ |
| REST / HTTP server          | ✓ (self-host) | (cloud) | ✓ (FastAPI) | ✓ | ✓ (serve) | ✓ | partial (LangGraph platform) | ✗ | ✓ (cloud HTTP MCP) | ✗ | ✓ (only mode) |
| MCP server                  | ✓ (OpenMemory) | ✗ | ✓ | ✓ (stdio + HTTP) | ✓ (Claude Code plugin) | ✗ | ✗ | ✗ | ✓ (HTTP MCP) | ✗ | ✗ |
| CLI                         | ✓ (`mem0`) | ✗ | ✗ | ✓ (letta-code) | ✓ (cognee-cli) | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ |
| LangChain / LangGraph integ | ✓   | ✓   | ✓ (LangGraph)  | partial | ✓ | partial | ✓ (native) | ✓ | ✓ | ✓ | ✗ |
| LlamaIndex integ            | ✓   | ✓   | ✓   | ✗   | ✓ (cognee llamaindex pkg) | ✗ | ✗ | (is) | ✗ | ✗ | ✗ |
| CrewAI integ                | ✓   | ✗   | ✗   | ✗   | partial | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| AutoGen integ               | ✗   | ✓   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗ |
| Agno / Pydantic AI integ    | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✓   | ✗   | ✗ |

### 1.11 IDE / agent-host integrations

| Host                        | M0  | Zep | GR  | Lt  | Cg  | MG  | LM  | LI  | Mi  | Mp  | Mh  |
|-----------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Claude Desktop / Claude Code| ✓ (MCP) | ✗ | ✓ (MCP) | ✓ (MCP) | ✓ (Claude Code plugin) | ✗ | ✗ | ✗ | ✓ (MCP) | ✗ | ✗ |
| Cursor                      | ✓   | ✗   | ✓   | ✓   | ✓   | ✗   | ✗   | ✗   | ✓   | ✗   | ✗ |
| ChatGPT / web extension     | ✓ (Chrome ext) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Codex / Warp / Antigravity  | partial | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Hermes Agent                | ✗   | ✗   | ✗   | ✗   | ✓   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗ |
| OpenClaw gateway            | ✗   | ✗   | ✗   | ✗   | ✓   | ✗   | ✗   | ✗   | ✓   | ✗   | ✗ |

### 1.12 Misc capabilities

| Feature                     | M0  | Zep | GR  | Lt  | Cg  | MG  | LM  | LI  | Mi  | Mp  | Mh  |
|-----------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Multimodal (images)         | ✓   | partial | ✗ | partial | ✓ | ✗ | ✗ | ✓ (ContentBlock) | ✗ | ✗ | ✗ |
| Audit logs                  | ✓   | ✓   | partial (provenance) | partial | ✓ (audit traits) | ✗ | ✗ | ✗ | partial (dashboard) | ✗ | ✗ |
| Explainability ("why this?")| ✗   | ✓ (provenance edges) | ✓ (episodes) | ✗ | ✓ (provenance) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| OpenTelemetry export        | ✗   | ✗   | ✗   | ✗   | ✓ (OTEL collector) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Telemetry / opt-out         | ✗   | ✗   | ✓   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗   | ✗ |
| Dashboard / UI              | ✓ (self-host wizard + cloud) | ✓ (cloud) | partial (graph viz cloud only) | ✓ (ADE) | ✓ (cognee-cli -ui) | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Eval harness shipped        | ✓ (memory-benchmarks repo) | partial | partial (paper repro) | partial (leaderboard) | ✗ | ✓ (DMR/MSC tasks in paper) | ✗ | ✗ | partial | ✗ | ✗ |
| Published benchmark numbers | ✓ (LoCoMo 91.6, LongMemEval 93.4, BEAM) | ✓ (LongMemEval, MSC) | ✓ (paper) | partial (leaderboard.letta.com) | partial (paper) | ✓ (DMR, MSC) | ✗ | ✗ | ✓ (LoCoMo 81.95%) | ✗ | ✗ |
| Streaming chat support      | ✗   | ✓   | ✗   | ✓   | ✗   | ✗   | ✗   | ✓   | ✓   | ✗   | ✗ |
| Self-improving prompts      | ✗   | ✗   | ✗   | ✓ (ADE/skills) | ✗ | ✗ | ✓ (procedural mem) | ✗ | ✗ | ✗ | ✗ |

---

## 2. Per-app deep-dives

### 2.1 Mem0 — `mem0ai/mem0`

**Identity.** `https://github.com/mem0ai/mem0` · ~54.3k★ · Apache-2.0 · last release `v2.0.1` (2026-04-25) · last push 2026-04-28. Y Combinator S24. Paper: arXiv:2504.19413.

**Memory taxonomy.** Multi-Level Memory: User, Session, Agent, Organizational. Short-term (conversation/working/attention), long-term (factual + episodic). Preferences, account details, domain facts.

**Storage backends.** Vector DBs declared: Qdrant (default), Chroma, Pinecone, PGVector, MongoDB Atlas, Milvus, Weaviate, FAISS, Redis, Elasticsearch, OpenSearch, Azure AI Search, Vertex AI Vector Search, Upstash Vector, Supabase, Baidu. Self-hosted stack: Docker + Postgres + Qdrant. **As of v2.0 (April 2026) graph DBs were removed** — Neo4j, Memgraph, Kuzu, Apache AGE drivers (~4k LoC) deleted; entity linking replaces graph backends.

**LLM providers.** OpenAI (default `gpt-5-mini`), Anthropic, Gemini, Azure OpenAI, Groq, DeepSeek, Bedrock, Ollama, OpenRouter, Grok, custom OpenAI-compatible.

**Embedders.** OpenAI `text-embedding-3-small` (default), Voyage, Cohere, HuggingFace, Ollama (Qwen 600M / nomic recommended for hybrid), Gemini.

**Retrieval.** v3 algorithm (April 2026): single-pass ADD-only extraction · entity linking (extracted, embedded, linked across memories) · multi-signal hybrid (semantic + BM25 + entity matching, scored in parallel and fused). Optional NLP extra (`mem0ai[nlp]` + spaCy `en_core_web_sm`) for keyword + entity boost. All 15 vector stores expose `keyword_search()` + `search_batch()` (Qdrant uses sparse vectors / BM25; others use native FTS).

**Lifecycle.** ADD-only (memories accumulate, never overwritten). Entity dedup. Versioning + audit logs. No automatic decay/reinforcement declared.

**Knowledge graph.** Removed in v2 — replaced by entity linking. No custom entity types, no learned ontology, no episode/provenance model.

**Temporal queries.** Strong on temporal reasoning (LongMemEval +42.1, LoCoMo temporal +29.6) but no explicit bi-temporal `valid_at/invalid_at`.

**Conflict resolution.** ADD-only design — agent-generated facts are first-class and stored with equal weight; no UPDATE/DELETE.

**Multi-tenancy.** `user_id`, `agent_id`, `run_id`, organizational memory.

**ACL / privacy.** Self-hosted auth on by default with `ADMIN_API_KEY`; admin wizard; `AUTH_DISABLED=true` for local dev. OpenMemory MCP supports per-app pause/revoke + audit logs. SOC2/HIPAA not declared on OSS.

**Transports.** `pip install mem0ai`, `npm install mem0ai`, self-hosted Docker server (port 3000), cloud platform, **OpenMemory MCP** (Docker + Postgres + Qdrant local-first), CLI (`mem0 init/add/search`).

**IDE integrations.** Chrome extension (ChatGPT/Perplexity/Claude), MCP for Cursor/Claude Desktop/Codex.

**Audit / explainability.** Audit logs for every memory operation, dashboard for read/write history, version tracking.

**Streaming retrieval.** Not declared.

**Summarization.** Not auto. Memories accumulate.

**Tool-call memory.** "Agent-generated facts are first-class — when an agent confirms an action, that information is now stored with equal weight."

**Multimodal.** ✓ Images: extracts textual info + relevant details from visual content.

**Eval harness.** Yes — open-sourced `mem0ai/memory-benchmarks` repo.

**Benchmarks (April 2026).** LoCoMo **91.6** (+20 vs old), LongMemEval **93.4** (+26, assistant memory recall +53.6), BEAM 1M **64.1**, BEAM 10M **48.6**. Single-pass retrieval; ~7K tokens; p50 latency 0.88-1.09s.

**Differentiators (README highlights).**
1. New v3 algorithm (April 2026) — single-pass ADD-only, no UPDATE/DELETE; agent-generated facts first-class.
2. Multi-signal hybrid retrieval (semantic + BM25 + entity).
3. Largest declared vector-store coverage (15+ backends) and dedicated multimodal support.

Sources: [README](https://github.com/mem0ai/mem0/blob/main/README.md), [memory types docs](https://docs.mem0.ai/core-concepts/memory-types), [vector DB list](https://docs.mem0.ai/components/vectordbs/overview), [OpenMemory MCP](https://mem0.ai/blog/introducing-openmemory-mcp), [v2.0 release notes](https://newreleases.io/project/github/mem0ai/mem0/release/v2.0.0), [research/benchmarks](https://mem0.ai/research), [arXiv:2504.19413](https://arxiv.org/abs/2504.19413).

---

### 2.2 Zep — `getzep/zep` (+ Graphiti)

**Identity.** `https://github.com/getzep/zep` · ~4.5k★ · Apache-2.0 · last push 2026-04-09. README explicitly says "**Zep Community Edition is no longer supported and has been deprecated**" — code moved to `legacy/`. The current repo hosts examples + integrations only.

**Memory taxonomy.** End-to-end **Context Engineering Platform**. Manages threads, messages, business data, documents, and app events. Relationship-aware context blocks pre-formatted for LLMs.

**Storage backends.** Cloud-only (managed). OSS engine (Graphiti, see §2.3) supports Neo4j, FalkorDB, Kuzu, Amazon Neptune.

**LLM providers.** Cloud-managed; supports any LLM via context blocks.

**Embedders.** Managed.

**Retrieval.** Sub-200ms latency claim. Graph RAG: extracts relationships, maintains temporal knowledge graph, returns "relationship-aware context blocks".

**Lifecycle.** Built-in users, threads, message storage. Auto fact invalidation with temporal history preserved (via Graphiti).

**Knowledge graph.** Yes (delegated to Graphiti).

**Temporal queries.** Yes — `valid_at` / `invalid_at` on each fact. Bi-temporal.

**Conflict resolution.** Old facts invalidated, not deleted; full historical state preserved.

**Multi-tenancy.** Cloud-managed users, threads, organizations.

**ACL / privacy.** SOC2 Type 2 + HIPAA compliance (cloud).

**Transports.** Python `zep-cloud`, TypeScript `@getzep/zep-cloud`, Go `zep-go/v2`. AutoGen integration (`integration/autogen/`).

**IDE integrations.** None directly (cloud SDK).

**Audit / explainability.** Dashboard with graph visualization, debug logs, API logs.

**Streaming.** Not declared in OSS repo.

**Summarization.** Yes (relationship-aware context blocks pre-formatted).

**Multimodal.** Not declared.

**Eval harness.** Paper + benchmarks (LongMemEval, MSC). Public claim: "State of the Art in Agent Memory" — but contested (see [zep-papers issue #5](https://github.com/getzep/zep-papers/issues/5)).

**Differentiators.**
1. Pre-formatted relationship-aware context blocks delivered with sub-200ms latency.
2. SOC2 Type 2 / HIPAA compliance baked into cloud.
3. Built-in user/thread/message storage on top of the temporal KG.

Sources: [README](https://github.com/getzep/zep/blob/main/README.md), [Zep paper arXiv:2501.13956](https://arxiv.org/abs/2501.13956), [help.getzep.com](https://help.getzep.com).

---

### 2.3 Graphiti — `getzep/graphiti`

**Identity.** `https://github.com/getzep/graphiti` · ~25.5k★ · Apache-2.0 · `v0.29.0` (2026-04-27) · paper arXiv:2501.13956.

**Memory taxonomy.** Context graph composed of Entities (nodes), Facts/Relationships (edges with validity windows), Episodes (provenance / raw data), Custom Types (Pydantic ontology). Not framed as semantic/episodic/procedural — uses the graph + temporal-fact taxonomy.

**Storage backends.** Neo4j 5.26 / FalkorDB 1.1.2 / Kuzu 0.11.2 / Amazon Neptune (Database or Analytics) + OpenSearch Serverless (FTS backend for Neptune). Pluggable via `graph_driver=`.

**LLM providers.** OpenAI (default), Anthropic (`graphiti-core[anthropic]`), Groq (`[groq]`), Google Gemini (`[google-genai]`), Azure OpenAI, Ollama (via `OpenAIGenericClient`), any OpenAI-compatible. Note: "works best with LLM services that support Structured Output (such as OpenAI and Gemini)".

**Embedders.** OpenAI (default `text-embedding-3-small`), Voyage, Gemini, Azure OpenAI, Ollama (`nomic-embed-text` 768-dim), any OpenAI-compatible.

**Retrieval.** Hybrid: semantic embeddings + BM25 keyword + graph traversal. Reranker/cross-encoder: OpenAI reranker, Gemini reranker (`gemini-2.5-flash-lite` default, uses log-probs for boolean classification). Graph-distance reranking. Predefined "search recipes" for nodes.

**Lifecycle.** Incremental graph construction (no batch recomputation). Autonomous fact invalidation — old facts not deleted, just invalidated. Entity summaries evolve over time. Community detection.

**Knowledge graph.** Core feature. Custom entity + edge types via Pydantic models. Both **prescribed** ontology (you define) and **learned** ontology (emerges from data).

**Temporal queries.** Bi-temporal (`valid_at`, `invalid_at`). Query "what's true now" or "what was true at any point in time". Sub-second latency.

**Conflict resolution.** Automatic fact invalidation with temporal history preserved (no LLM summarization judgments like GraphRAG).

**Multi-tenancy.** `group_id` based grouping for organizing related data.

**ACL / privacy.** Inherits from graph driver; no built-in ACL.

**Transports.** `pip install graphiti-core` · FastAPI REST service (`server/`) · MCP server (`mcp_server/`).

**IDE integrations.** MCP server supports Claude, Cursor, and other MCP clients.

**Audit / explainability.** Episodes provide full provenance: every entity and relationship traces back to the raw data that produced it.

**Streaming retrieval.** Not declared.

**Summarization.** Entity summaries evolve over time as new episodes arrive.

**Tool-call memory.** Not declared explicitly.

**Multimodal.** Not declared.

**Telemetry.** PostHog anonymous, opt-out via `GRAPHITI_TELEMETRY_ENABLED=false`. Auto-disabled during pytest.

**Concurrency.** `SEMAPHORE_LIMIT` env var (default 10) for ingestion.

**Eval harness.** Paper benchmarks. Public arXiv paper.

**Benchmarks.** Cited in Zep paper (LongMemEval, MSC).

**Differentiators.**
1. **Bi-temporal** validity windows on every fact (`valid_at` + `invalid_at`) — closest thing to a true temporal KG in the OSS landscape.
2. Pluggable graph backends (Neo4j / FalkorDB / Kuzu / Neptune) via `graph_driver` constructor.
3. Both **prescribed** Pydantic ontology AND **learned** ontology that emerges from data.

Sources: [README](https://github.com/getzep/graphiti/blob/main/README.md), [paper](https://arxiv.org/abs/2501.13956), [help.getzep.com/graphiti](https://help.getzep.com/graphiti), [custom entity types](https://help.getzep.com/graphiti/core-concepts/custom-entity-and-edge-types).

---

### 2.4 Letta — `letta-ai/letta` (formerly MemGPT)

**Identity.** `https://github.com/letta-ai/letta` · ~22.4k★ · Apache-2.0 · `v0.16.7` (2026-03-31) · last push 2026-04-12. `cpacker/MemGPT` redirects here.

**Memory taxonomy.** Three-tier OS-inspired model:
- **Core memory** — always-in-context structured memory blocks (RAM-like). Persona, human, custom labels.
- **Recall memory** — searchable conversation history.
- **Archival memory** — out-of-context vector store; on-demand semantic search via tool calls; "passages".

**Storage backends.** Postgres (Docker default) or SQLite (pip default) for agent state + recall + archival passages. pgvector for archival embedding storage.

**LLM providers.** Provider prefixes: `openai/`, `anthropic/`, `google_ai/`, `google_vertex/`, `groq/`, `deepseek/`, `ollama/`. Recommended: GPT-5.2 / Claude Opus 4.5.

**Embedders.** OpenAI, Cohere, HuggingFace, Ollama, Gemini.

**Retrieval.** Semantic search over archival passages via tool calls (`archival_memory_search`). List/get/create/update/delete passages exposed as tools.

**Lifecycle.** **Sleep-time agents** — special background agents that share memory blocks with primary agents and modify them asynchronously (compaction, summarization). When the sleep-time agent edits a shared block, primary sees update immediately. Memory tier movement (data flows main↔external) is agent-driven via tool calls.

**Knowledge graph.** Not part of core. `letta-ai/ai-memory-sdk` is an experimental pluggable SDK.

**Temporal queries.** Not declared.

**Conflict resolution.** Agent-driven — agent decides what to keep/edit in core blocks; no automatic invalidation.

**Multi-tenancy.** Per-agent state, organizations, **shared memory blocks** (attach same block ID to multiple agents → real-time read/write coordination without messaging).

**ACL / privacy.** Org scoping in cloud. Self-host = your infra.

**Transports.** Python `letta-client` / `letta`; TypeScript `@letta-ai/letta-client`. **Letta Code** CLI (`@letta-ai/letta-code`). MCP server with **dual transport** — stdio (Claude Desktop, Cursor) + HTTP (production). MCP 2025-11-25 compliance. Letta is also an MCP **client** (can call other MCP servers from inside agents).

**IDE integrations.** Claude Desktop, Cursor (via MCP). Letta Code CLI for terminal-based coding agent.

**Audit / explainability.** ADE (Agent Development Environment) with debug logs, tool manager, leaderboard.

**Streaming retrieval.** Streaming chat supported.

**Summarization.** Recursive summarization when context fills up (MemGPT-inspired).

**Tool-call memory.** Yes — agents use explicit tool calls (`core_memory_replace`, `archival_memory_insert`, `archival_memory_search`, etc.) to manage memory.

**Multimodal.** Partial (depends on underlying LLM).

**Skills / subagents.** Letta Code supports skills + subagents bundled with pre-built memory + continual-learning skills.

**Self-improving.** Sleep-time agents enable continual learning; ADE supports prompt/skill iteration.

**Eval harness.** Public model leaderboard (`leaderboard.letta.com`).

**Benchmarks.** Letta-ai leaderboard for memory-using agents per model.

**Differentiators.**
1. **Three-tier OS-inspired architecture** (core / recall / archival) controlled by agent tool calls — direct from the MemGPT paper.
2. **Sleep-time agents** that share memory blocks and run asynchronously to maintain primary agent's memory.
3. Shared memory blocks — multiple agents attach the same block for real-time coordination without explicit messaging.

Sources: [README](https://github.com/letta-ai/letta/blob/main/README.md), [Memory blocks docs](https://docs.letta.com/guides/agents/memory-blocks/), [Sleep-time agents](https://docs.letta.com/guides/agents/architectures/sleeptime), [Shared memory](https://docs.letta.com/guides/core-concepts/memory/shared-memory/), [Archival memory](https://docs.letta.com/guides/agents/archival-memory/), [Agent Memory blog](https://www.letta.com/blog/agent-memory).

---

### 2.5 Cognee — `topoteretes/cognee`

**Identity.** `https://github.com/topoteretes/cognee` · ~16.9k★ · Apache-2.0 · `v1.0.4.dev0` (2026-04-25) · last push 2026-04-28. Paper: arXiv:2505.24478.

**Memory taxonomy.** Episodic (events / past interactions) + Semantic (factual knowledge). Procedural memory mentioned conceptually. Session memory (fast cache, syncs to graph in background) + permanent knowledge graph. Memory types: explicit (declarative: episodic + semantic) and implicit (procedural).

**Storage backends.**
- **Vector**: LanceDB (default), PGVector, Qdrant, Redis, ChromaDB, FalkorDB, Neptune Analytics.
- **Graph**: Kuzu (default), Kuzu-remote, Neo4j, Neptune, Neptune Analytics, Memgraph, FalkorDB.
- **Relational/metadata**: SQLite (default), Postgres.

Defaults: SQLite + LanceDB + Kuzu (zero infrastructure).

**LLM providers.** OpenAI (default), Azure, Gemini, Anthropic, Ollama, custom (vLLM).

**Embedders.** OpenAI, Voyage, Cohere, HuggingFace, Ollama, Gemini.

**Retrieval.** Vector + graph traversal hybrid. **Auto-routing**: `cognee.recall(query)` automatically picks best search strategy. Session memory queried first, then falls through to graph.

**Lifecycle.** Pipelines: **Extract → Cognify → Load (ECL)**. `add_data_points` extracts nodes/edges, deduplicates, integrates simultaneously into vectors + metadata + graph. `improve` operation (continuous learning). Session syncs to graph in background.

**Knowledge graph.** Core feature. **Ontology integration**: optional RDF/OWL file as reference vocabulary (formal schema linking entity types/individuals to canonical concepts). **Node Sets** = labels you pin on documents → automatic `belongs_to_set()` edges.

**Temporal queries.** `temporal_cognify=True` adds time-aware facts during cognify.

**Conflict resolution.** Knowledge updates handled by continuous re-ingestion and dedup in `add_data_points`.

**Multi-tenancy.** Tenant isolation, traceability, OTEL collector, audit traits.

**ACL / privacy.** Agentic user/tenant isolation. Runs locally for privacy.

**Transports.** `pip install cognee`, `cognee-cli` (also `-ui` opens local UI), `cognee.serve(url=, api_key=)` for self-hosted server, Cognee Cloud, MCP-compatible.

**IDE integrations.** **Claude Code plugin** (`cognee-integrations/integrations/claude-code`) hooks into Claude lifecycle (`SessionStart`, `PostToolUse`, `UserPromptSubmit`, `PreCompact`, `SessionEnd`). **Hermes Agent** (`memory: cognee` config). **OpenClaw** plugin (`@cognee/cognee-openclaw`).

**Audit / explainability.** OTEL collector, audit traits, traceability.

**Streaming retrieval.** Not declared.

**Summarization.** Not auto-summarization in core; `improve` continually refines.

**Tool-call memory.** Yes — Claude Code plugin captures tool calls into session memory automatically.

**Multimodal.** ✓ "multimodal" is listed as a knowledge-infrastructure feature.

**Eval harness.** Paper-driven; no shipped harness in README.

**Differentiators.**
1. **ECL pipeline** (Extract → Cognify → Load) writes simultaneously to vectors, metadata, and graph DB.
2. **Optional RDF/OWL ontology** as reference vocabulary + **Node Sets** for arbitrary labels with auto-edges.
3. Defaults are pure file-based (SQLite + LanceDB + Kuzu) — zero infrastructure to start; swap in Neo4j/Neptune/Qdrant when scaling.

Sources: [README](https://github.com/topoteretes/cognee/blob/main/README.md), [docs.cognee.ai/setup-configuration](https://docs.cognee.ai/setup-configuration/overview), [ontology blog](https://www.cognee.ai/blog/deep-dives/ontology-ai-memory), [paper arXiv:2505.24478](https://arxiv.org/abs/2505.24478).

---

### 2.6 MemGPT (legacy / research) — `cpacker/MemGPT` (now redirects to Letta)

**Identity.** Paper: arXiv:2310.08560 ("MemGPT: Towards LLMs as Operating Systems", Charles Packer et al., UC Berkeley, October 2023). The repo redirects to Letta. Treated here as the **research artifact** because every other system cites it.

**Memory taxonomy (paper).**
- **Main context (fast memory)** — the LLM's immediate working space, constrained by the underlying model's token limits. Includes a system instruction, a working context block, and a FIFO queue of recent messages.
- **External context (slow memory)** — massive searchable archive of past interactions. Subdivided into:
  - **Recall memory** — searchable database that lets the agent reconstruct specific past memories via semantic search.
  - **Archival memory** — long-term storage for important info, can be moved back into core or recall as needed.
- **Core memory** — always-accessible compressed representation of essential facts and personal information (persona + human blocks).

**Storage backends.** SQLite default. Postgres + pgvector (current Letta successor). FAISS, Chroma, Qdrant in early versions.

**LLM providers.** OpenAI primary; later versions added Anthropic, local models. Best results with GPT-4 / Claude (paper note: requires structured-output / tool-calling).

**Embedders.** OpenAI, HuggingFace, Ollama (in later versions).

**Retrieval.** Semantic search via tool calls (`archival_memory_search`, `conversation_search`). LLM moves data between main and external context using interrupts (OS-style).

**Lifecycle.** **Recursive summarization** when main context fills (queue is summarized into working context). Agent decides when to insert into archival.

**Knowledge graph.** None.

**Temporal queries.** Not declared.

**Conflict resolution.** Agent-driven — LLM rewrites core memory blocks via tool calls.

**Multi-tenancy.** Single-agent in original paper; multi-agent added in later versions and now in Letta.

**ACL.** None in paper.

**Transports.** Python lib + REST server + CLI in late MemGPT releases (now subsumed by Letta).

**IDE integrations.** None in original paper.

**Audit.** None.

**Streaming.** Not declared in original paper.

**Summarization.** Yes — recursive summarization is a core paper contribution.

**Tool-call memory.** Yes — interrupts and function-calls drive all memory ops.

**Multimodal.** No.

**Eval harness.** Paper benchmarks: **DMR** (Deep Memory Retrieval) and **MSC** (Multi-Session Chat) — both introduced/reused in the paper.

**Differentiators.**
1. **OS-inspired virtual context management** — first system to frame LLM context as an OS memory hierarchy, with the LLM itself as the kernel managing data movement between fast and slow memory.
2. **Self-directed recursive summarization** — the LLM compresses its own context when it fills.
3. **Heartbeat / interrupt model** for proactive memory operations between user turns.

Sources: [paper arXiv:2310.08560](https://arxiv.org/abs/2310.08560), [research.memgpt.ai](https://research.memgpt.ai/), legacy archive at [archive.org/details/github.com-cpacker-MemGPT_-_2023-10-20_17-34-33](https://archive.org/details/github.com-cpacker-MemGPT_-_2023-10-20_17-34-33).

---

### 2.7 LangMem — `langchain-ai/langmem`

**Identity.** `https://github.com/langchain-ai/langmem` · ~1.4k★ · MIT · last push 2026-04-27.

**Memory taxonomy (declared).** Three explicit types — **semantic** (facts + relationships, two representations: collections and profiles), **episodic** (full context of an interaction → few-shot examples), **procedural** (internalized how-to-perform; saved as updated **prompt instructions**, distinct from model weights/code).

**Storage backends.** Any LangGraph `BaseStore`. `InMemoryStore` (dev only, lost on restart), `AsyncPostgresStore` for production. Vector index config: `{dims, embed}`.

**LLM providers.** Any LangChain-supported provider. README defaults to Anthropic (`claude-3-5-sonnet-latest`). Embed: e.g. `openai:text-embedding-3-small`.

**Embedders.** Any LangChain embed string.

**Retrieval.** `create_search_memory_tool(namespace=...)` for agent-controlled search. Memory tools are namespace-scoped.

**Lifecycle.** Two mechanisms:
- **Hot path** — agent uses memory tools (`create_manage_memory_tool`, `create_search_memory_tool`) during conversations.
- **Background memory manager** — auto-extracts, consolidates, and updates knowledge after conversations ("subconscious" memory formation).

Memory Manager extracts new memories, **updates or removes outdated memories**, and consolidates/generalizes from existing memories.

**Prompt optimization (procedural memory).** Multiple algorithms:
- `metaprompt` — reflection + thinking time + meta-prompt to propose updates.
- `gradient` — splits into critique + prompt-proposal steps.
- `prompt_memory` — simple algorithm.

**Knowledge graph.** Not declared.

**Temporal queries.** Not declared.

**Conflict resolution.** Manager updates/removes outdated memories.

**Multi-tenancy.** Namespaces (e.g. `("memories",)`).

**ACL.** None declared.

**Transports.** Python lib only; native integration with LangGraph storage layer; available by default in all LangGraph Platform deployments.

**IDE integrations.** None directly.

**Audit / explainability.** Use LangSmith.

**Streaming.** Not declared.

**Summarization.** Yes (background manager).

**Tool-call memory.** Yes (the manage_memory_tool itself is a tool call).

**Multimodal.** Not declared.

**Eval harness.** Not declared.

**Benchmarks.** None in README.

**Differentiators.**
1. Only project that **explicitly names all three psychological memory types** (semantic / episodic / procedural) with concrete representations for each.
2. **Procedural memory = prompt updates** — saves learned procedures as updated agent instructions via `metaprompt` / `gradient` / `prompt_memory` algorithms.
3. **Two-mechanism design**: hot-path tools (in-conversation) + background manager ("subconscious" formation).

Sources: [README](https://github.com/langchain-ai/langmem/blob/main/README.md), [conceptual guide](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/), [DeepLearning.AI course](https://www.deeplearning.ai/courses/long-term-agentic-memory-with-langgraph/), [LangMem launch blog](https://blog.langchain.com/langmem-sdk-launch/).

---

### 2.8 LlamaIndex Memory — `run-llama/llama_index`

**Identity.** `https://github.com/run-llama/llama_index` · ~49k★ · MIT · last push 2026-04-28. Memory module is a sub-component of LlamaIndex framework.

**Memory taxonomy.**
- **Short-term** — FIFO queue of `ChatMessage` objects (the `Memory` class).
- **Long-term** — Memory Block objects that receive flushed messages.

Block types:
- `ChatMemoryBuffer` — last X messages within token limit (deprecated; replaced by `Memory`).
- `ChatSummaryMemoryBuffer` — buffer + LLM-summarized history.
- `VectorMemory` / `VectorMemoryBlock` — vector-similarity retrieval over past messages.
- `SimpleComposableMemory` — primary buffer + secondary sources.
- `StaticMemoryBlock` — static content (string or `ContentBlock` list: `TextBlock`, `ImageBlock`, etc.) always inserted into memory.
- `FactExtractionMemoryBlock` — LLM extracts list of facts from chat history.

Each block has a **priority**: priority 0 = always kept; priority 1 = temporarily disabled when token-limit pressure hits; etc.

**Storage backends.** Any LlamaIndex storage backend: SQLite, Postgres, MongoDB, Redis, Qdrant, Pinecone, Chroma, FAISS, pgvector, LanceDB, Weaviate, Milvus, Elasticsearch, OpenSearch, Azure AI Search, Vertex AI Vector Search, Supabase, Upstash, etc. (One of the broadest backend lists in the ecosystem.)

**LLM providers.** All LlamaIndex LLMs (OpenAI, Anthropic, Gemini, Bedrock, Azure, Groq, DeepSeek, Ollama, OpenRouter, Grok, vLLM, etc.).

**Embedders.** All LlamaIndex embeddings.

**Retrieval.** Vector similarity in `VectorMemoryBlock`. Full LlamaIndex retriever stack reusable. Hybrid retrieval via underlying vector store.

**Lifecycle.** Auto-flush from short-term to long-term blocks when queue exceeds size. Priority-based eviction under token-limit pressure. Fact extraction + static blocks compose into context.

**Knowledge graph.** **KnowledgeGraphIndex** in core LlamaIndex (separate module). Cognee integration available (`cognee` LlamaIndex pkg).

**Temporal queries.** Not declared in memory module.

**Conflict resolution.** Not declared.

**Multi-tenancy.** Per-session memory; namespaces via storage layer.

**ACL.** None.

**Transports.** Python and TypeScript SDKs. No standalone server.

**IDE integrations.** None directly.

**Audit / explainability.** None.

**Streaming.** Yes — chat streaming.

**Summarization.** `ChatSummaryMemoryBuffer` (LLM-summarized).

**Tool-call memory.** Not declared.

**Multimodal.** ✓ — `StaticMemoryBlock` and `ContentBlock`s support `ImageBlock`, etc.

**Eval harness.** Separate (`llama-index-eval`).

**Differentiators.**
1. **Memory Blocks with priorities** (0 = always kept, 1+ = drop under pressure) for fine-grained context control.
2. **Composable memory** — `SimpleComposableMemory` combines a primary buffer with multiple secondary memory sources.
3. **Multimodal blocks** — `StaticMemoryBlock` accepts `ContentBlock`s including `ImageBlock`, etc.

Sources: [Memory module guide](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/), [Memory class API](https://developers.llamaindex.ai/python/framework-api-reference/memory/memory/), [Improved short/long-term memory blog](https://www.llamaindex.ai/blog/improved-long-and-short-term-memory-for-llamaindex-agents), [chat_summary_memory_buffer.py](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/memory/chat_summary_memory_buffer.py).

---

### 2.9 Memori — `MemoriLabs/Memori` (formerly `GibsonAI/memori`)

**Identity.** `https://github.com/MemoriLabs/Memori` · ~14k★ · Apache-2.0 · `v3.3.1` (2026-04-27) · last push 2026-04-28. (`GibsonAI/memori` redirects here.)

**Memory taxonomy.** Three levels:
- **Entity** — person, place, or thing (like a user). `entity_id`.
- **Process** — agent, LLM interaction, or program. `process_id`.
- **Session** — current interactions between entity, process, and LLM.

**Memori Advanced Augmentation** enriches each level with: attributes, events, facts, people, preferences, relationships, **rules**, **skills**.

Configuration: `max_short_term_memories`, `max_long_term_memories`, `retention_policy` (`7_days`/`30_days`/`90_days`/`permanent`), `importance_threshold`.

**Storage backends.** Native: PostgreSQL, MySQL, SQLite, MongoDB. Postgres-compatible: Supabase, Neon, AWS RDS. **BYODB** ("Bring your own DB") with adapter architecture.

**LLM providers.** Anthropic, Bedrock, DeepSeek, Gemini, Grok (xAI), OpenAI (Chat Completions + Responses API). Streamed/unstreamed, sync/async.

**Embedders.** Cloud-managed (the SDK `register`s an OpenAI client).

**Retrieval.**
- **Auto Mode** — dynamic intelligent search across entire database every time.
- **Combined Mode** — layered approach balancing quick recall with deep retrieval.

**Lifecycle.** Background augmentation (no latency impact). Retention policies. Sessions auto-managed.

**Knowledge graph.** Partial — entity relationships explicit; not a full KG product.

**Temporal queries.** Not declared explicitly; retention_policy provides time bucketing.

**Conflict resolution.** Not declared.

**Multi-tenancy.** Strong — `entity_id`, `process_id`, `session_id` are first-class. Attribution (`mem.attribution(entity_id, process_id)`) is required to make memories.

**ACL / privacy.** API-key based (`MEMORI_API_KEY`). Per-key quotas.

**Transports.** Python (`pip install memori`), TypeScript (`npm install @memorilabs/memori`), HTTP MCP server (`https://api.memorilabs.ai/mcp/`). Python CLI (`python -m memori`) for account/key/quota management. **No SDK integration needed for MCP** — drop-in for Claude Code, Cursor, Codex, Warp, Antigravity.

**Frameworks.** Agno, LangChain, Pydantic AI. Platforms: DeepSeek, Nebius AI Studio.

**IDE integrations.** Claude Code, Cursor, Codex, Warp, Antigravity, OpenClaw (drop-in plugin `@memorilabs/openclaw-memori`).

**Audit / explainability.** Dashboard (Memories, Analytics, Playground, API Keys).

**Streaming retrieval.** Streamed/unstreamed both supported.

**Summarization.** Background.

**Tool-call memory.** "Memory from what agents do, not just what they say" (README tagline).

**Multimodal.** Not declared.

**Eval harness.** Cloud benchmark.

**Benchmarks.** **LoCoMo: 81.95% overall accuracy** with avg **1,294 tokens/query** (4.97% of full-context). Claims to outperform Zep, LangMem, Mem0 while reducing prompt size ~67% vs Zep, ~20× vs full-context.

**Differentiators.**
1. **Entity / process / session** triple as the scoping primitive — required for all memories ("if you do not provide attribution, Memori cannot make memories for you").
2. **No-code MCP drop-in** for Claude Code / Cursor / Codex / Warp / Antigravity (single `claude mcp add` command, no SDK code).
3. Augmentation includes **rules and skills** as first-class memory dimensions in addition to facts/preferences.

Sources: [README](https://github.com/MemoriLabs/Memori/blob/main/README.md), [memorilabs.ai/docs/open-source/databases/overview](https://memorilabs.ai/docs/open-source/databases/overview/), [LoCoMo benchmark](https://memorilabs.ai/benchmark), [MCP setup docs](https://memorilabs.ai/docs/memori-cloud/mcp/client-setup).

---

### 2.10 Memoripy — `caspianmoon/memoripy`

**Identity.** `https://github.com/caspianmoon/memoripy` · ~690★ · Apache-2.0 · last push 2026-03-18. Small project, no GitHub releases page populated.

**Memory taxonomy.** Short-term and long-term (managed by usage and relevance).

**Storage backends.** `JSONStorage` (file), `InMemoryStorage`. `BaseStorage` abstract base for custom backends.

**LLM providers.** OpenAI, Azure OpenAI, OpenRouter, Ollama.

**Embedders.** OpenAI, Ollama (e.g. `mxbai-embed-large`).

**Retrieval.** Cosine similarity over embeddings + decay factors + **spreading activation** over a concept graph.

**Lifecycle.**
- **Decay** — older/unused memories decay over time.
- **Reinforcement** — frequently accessed memories are reinforced.
- **Hierarchical clustering** — clusters similar memories into semantic groups.

**Knowledge graph.** Concept graph (built with NetworkX) + spreading activation for retrieval.

**Temporal queries.** Implicit via decay.

**Conflict resolution.** Not declared.

**Multi-tenancy.** Single-store; not a multi-tenant design.

**ACL.** None.

**Transports.** Python lib only.

**IDE integrations.** None.

**Audit.** None.

**Streaming.** None.

**Summarization.** None auto.

**Tool-call memory.** None.

**Multimodal.** None.

**Concept extraction.** Yes — uses OpenAI or Ollama models for concept extraction + embeddings.

**Dependencies.** `openai`, `faiss-cpu`, `numpy`, `networkx`, `scikit-learn`, `langchain`, `ollama`.

**Eval harness.** None.

**Benchmarks.** None.

**Differentiators.**
1. **Spreading activation over a concept graph** for retrieval — explicitly cognitive-science-inspired. (Unique among the 10.)
2. **Decay + reinforcement** as a first-class lifecycle pair.
3. **Hierarchical clustering** of memories into semantic groups.

Sources: [README](https://github.com/caspianmoon/memoripy/blob/main/README.md), [Medium walkthrough](https://medium.com/@theivision/memoripy-the-ultimate-python-library-for-context-aware-memory-management-fc895f0e6e08).

---

### 2.11 Motorhead — `getmetal/motorhead` (DEPRECATED)

**Identity.** `https://github.com/getmetal/motorhead` · ~0.9k★ · Apache-2.0 · last release `v3.0.2` (Dec 2023) · last push 2025-07-22. README banner: **"DEPRECATED — Support is no longer maintained for this project."**

**Memory taxonomy.** Conversation messages + incremental summary `context`. Optional long-term memory via Redisearch VSS.

**Storage backends.** Redis (required) — both for messages and Redisearch-based VSS retrieval.

**LLM providers.** OpenAI (default `gpt-3.5-turbo` or `gpt-4`), Azure OpenAI (deployment IDs).

**Embedders.** OpenAI ada (Azure: `AZURE_DEPLOYMENT_ID_ADA`).

**Retrieval.** `POST /sessions/:id/retrieval` — text query → VSS search filtered by `session_id`.

**Lifecycle.** When messages exceed `MOTORHEAD_MAX_WINDOW_SIZE` (default 12), a job halves them and **incrementally summarizes** the half. Subsequent summaries are incremental.

**Knowledge graph.** None.

**Temporal queries.** None.

**Conflict resolution.** None.

**Multi-tenancy.** Per-session (`session_id`), auto-created on first POST.

**ACL.** None.

**Transports.** **REST-only** — no Python/TS SDK. Endpoints:
- `GET /sessions/:id/memory` — recent messages + summary `context` + token count.
- `POST /sessions/:id/memory` — store messages, optional `context`.
- `DELETE /sessions/:id/memory` — clear session.
- `POST /sessions/:id/retrieval` — VSS search.

Docker image: `ghcr.io/getmetal/motorhead:latest`. Deployable to Railway in one click.

**IDE integrations.** None.

**Audit.** None.

**Streaming.** None.

**Summarization.** Yes — **incremental summarization** is the marquee feature.

**Tool-call memory.** None.

**Multimodal.** None.

**Eval harness.** None.

**Benchmarks.** None.

**Differentiators.**
1. **Pure REST server** model — language-agnostic, the only one of the 10 with no SDK at all.
2. **Incremental summarization** when window fills (window/2 messages summarized into rolling `context`) — predates and shaped the design of newer summarizing memories.
3. Redis-only stack: messages + Redisearch VSS in a single store. Trivial deploy.

Sources: [README](https://github.com/getmetal/motorhead/blob/main/README.md).

---

## 3. Distinctive features that appear in only 1-2 apps

These are unique-or-rare capabilities that are likely to matter for differentiation against `memoirs`:

| Feature                                         | Found in              | Notes |
|-------------------------------------------------|-----------------------|-------|
| **Bi-temporal validity windows** (`valid_at` + `invalid_at`) on every fact | Graphiti (+ Zep cloud) | Cognee has time-aware facts via `temporal_cognify` but not declared as bi-temporal. Mem0 v3 has strong temporal accuracy but no explicit windows. |
| **Sleep-time agents** (background agents sharing memory blocks with primary) | Letta only | Async memory maintenance. |
| **Procedural memory = prompt updates** (`metaprompt` / `gradient` / `prompt_memory`) | LangMem only | Saves learned procedures into agent instructions. |
| **Spreading activation** over concept graph     | Memoripy only         | Cognitive-science-inspired retrieval. |
| **Decay + reinforcement** as first-class lifecycle | Memoripy (+ Memori partial via retention_policy) | Frequency-of-access boosts. |
| **Hierarchical clustering** of memories into semantic groups | Memoripy only | NetworkX-based. |
| **Incremental summarization** as a defining product feature | Motorhead (+ LlamaIndex `ChatSummaryMemoryBuffer`) | Window-half compaction. |
| **OS-style virtual context management** (heartbeat/interrupts, recursive summarization) | MemGPT/Letta | The original paper's idea, lives on in Letta. |
| **Custom Pydantic entity + edge types** (prescribed ontology) | Graphiti (+ Zep cloud) | Cognee uses RDF/OWL instead. |
| **Optional RDF/OWL ontology** as reference vocabulary | Cognee only | Plus Node Sets for arbitrary labels. |
| **Self-hosted local-first MCP server** (Docker + Postgres + Qdrant) | Mem0 OpenMemory only | Other MCP integrations are remote/cloud. |
| **Dual-transport MCP** (stdio + HTTP)            | Letta only            | Plus Letta is also an MCP **client**. |
| **Drop-in MCP for Claude/Cursor/Codex/Warp/Antigravity with no SDK** | Memori only | Single `claude mcp add` command. |
| **Three-tier OS-inspired memory (core / recall / archival)** | Letta + MemGPT (legacy) | Direct lineage from MemGPT paper. |
| **Multi-Level Memory matrix** (User / Session / Agent / Organizational) | Mem0 | Explicit organizational scope. |
| **Entity / Process / Session triple** required for attribution | Memori only | "No attribution → no memories." |
| **Pre-formatted relationship-aware context blocks** at <200ms | Zep cloud only | Cloud-only product feature. |
| **ECL pipeline** (Extract → Cognify → Load) writing to vec + meta + graph simultaneously | Cognee only | Single transactional ingestion. |
| **Auto-routing recall** (system picks best search strategy automatically) | Cognee only | `cognee.recall(query)` chooses. |
| **Prescribed + Learned ontology** (declare upfront OR let it emerge) | Graphiti only | Both modes supported. |
| **Multimodal `ImageBlock` / `ContentBlock` in memory** | LlamaIndex (+ Mem0 image extraction) | Block-level multimodal. |
| **Multi-signal hybrid scoring** (dense + BM25 + entity, fused in parallel) | Mem0 v3 only | `mem0ai[nlp]` + spaCy. |
| **Reranker / cross-encoder layer**               | Graphiti (+ LlamaIndex) | Gemini log-prob boolean classification. |
| **Concurrency primitives** (`SEMAPHORE_LIMIT` env) | Graphiti only | Documented LLM-rate-limit-aware ingestion. |
| **Built-in opt-out telemetry**                  | Graphiti only         | PostHog, env-var disable, auto-disabled in pytest. |
| **OTEL collector + audit traits**                | Cognee only           | Production observability story. |
| **Public model leaderboard** specifically for memory | Letta (`leaderboard.letta.com`) | Per-model agent performance. |
| **OSS evaluation framework + reproducible benchmarks** | Mem0 (`memory-benchmarks` repo) | LoCoMo / LongMemEval / BEAM. |
| **Pause / revoke per-app access for memory**     | Mem0 OpenMemory only  | Granular runtime ACL. |
| **SOC2 Type 2 + HIPAA compliance**               | Zep cloud only        | Of the 10, only one declares both. |
| **CLI for memory ops**                           | Mem0, Letta, Cognee, Memori | `mem0`, `letta`, `cognee-cli`, `python -m memori`. |
| **Pure REST-only / no SDK design**               | Motorhead only        | Language-agnostic by construction. |
| **Hosted leaderboard / hosted dashboard**        | Letta, Mem0, Cognee, Memori, Zep | Five of ten. |
| **Plugin into agent host's lifecycle hooks** (`SessionStart`, `PostToolUse`, `PreCompact`, `SessionEnd`) | Cognee Claude-Code plugin only | Full Claude Code lifecycle hookup. |
| **Both `prescribed` + `learned` modes for entity types** | Graphiti only | "Start simple, evolve as patterns appear." |
| **Built-in users / threads / message storage**   | Zep cloud + Letta     | Most others leave this to the host app. |
| **Serverless deploy targets in README** (Modal, Railway, Fly, Render, Daytona) | Cognee only | Explicit 1-click matrix. |
| **Self-host auth wizard + admin API key bootstrap** | Mem0 only | `make bootstrap` issues first key. |

---

## 4. Coverage notes & caveats

- **Zep**: the OSS repo is now an examples/integrations workspace; the comparable engine is Graphiti (§2.3). Treat the "Zep" column as Zep Cloud declared features.
- **MemGPT (legacy)**: `cpacker/MemGPT` 301-redirects to `letta-ai/letta`. The architecture in column "MG" reflects the **2023 paper**, since the modern repo is Letta.
- **Motorhead**: deprecated. Included for historical comparison since it's still cited as the canonical Redis memory server.
- **Memoripy**: small project (~690★). README is the only authoritative source — many features beyond what's listed there are not declared.
- **LangMem**: small repo, but the conceptual guide on `langchain-ai.github.io/langmem` declares more memory-type machinery than any other project in the list.
- **LlamaIndex Memory**: not a standalone project — extracted feature set from the memory submodule (`llama_index/core/memory/`) of the larger LlamaIndex framework.
- All star counts and dates were captured 2026-04-28 via `gh api repos/{owner}/{repo}`.

---

## 5. Source index (key URLs)

- Mem0: https://github.com/mem0ai/mem0 · https://docs.mem0.ai · https://mem0.ai/research · https://mem0.ai/blog/introducing-openmemory-mcp · https://github.com/mem0ai/memory-benchmarks
- Zep: https://github.com/getzep/zep · https://help.getzep.com · https://arxiv.org/abs/2501.13956
- Graphiti: https://github.com/getzep/graphiti · https://help.getzep.com/graphiti
- Letta: https://github.com/letta-ai/letta · https://docs.letta.com · https://leaderboard.letta.com
- Cognee: https://github.com/topoteretes/cognee · https://docs.cognee.ai · https://www.cognee.ai/blog
- MemGPT: https://arxiv.org/abs/2310.08560 · https://research.memgpt.ai
- LangMem: https://github.com/langchain-ai/langmem · https://langchain-ai.github.io/langmem
- LlamaIndex: https://github.com/run-llama/llama_index · https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/
- Memori: https://github.com/MemoriLabs/Memori · https://memorilabs.ai/docs
- Memoripy: https://github.com/caspianmoon/memoripy
- Motorhead: https://github.com/getmetal/motorhead
