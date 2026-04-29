# Configuration

memoirs is fully env-var driven so the same binary works in dev, daemon, CI and production. CLI flags exist for one-off overrides; everything that should be a sticky setting is an env var.

## Storage

| variable | default | meaning |
|---|---|---|
| `MEMOIRS_DB` | `.memoirs/memoirs.sqlite` | SQLite path. Created on first `memoirs init`. |
| `MEMOIRS_ENCRYPT_KEY` | _(unset)_ | Set to enable SQLCipher encryption-at-rest. Requires `pip install -e '.[encryption]'`. |
| `MEMOIRS_SQLITE_MMAP_MB` | `256` | Memory-mapped IO size. Lower for memory-constrained hosts. |
| `MEMOIRS_SQLITE_CACHE_MB` | `64` | SQLite page cache. Affects p95 on cold reads. |

## Curator (LLM)

The curator handles extraction, consolidation, conflict detection, summarization, and project narratives. It auto-detects which GGUF model is present and picks Qwen → Phi → Gemma in that order.

| variable | default | meaning |
|---|---|---|
| `MEMOIRS_CURATOR_BACKEND` | _auto_ | `qwen` / `phi` / `gemma`. `auto` (default) picks the first available. |
| `MEMOIRS_CURATOR_MODEL` | _per-backend_ | Absolute path to a `.gguf` file. Overrides backend defaults. |
| `MEMOIRS_GEMMA_THREADS` | `cpu_count - 4` | Threads passed to llama-cpp. Legacy name; applies to whichever backend is loaded. |
| `MEMOIRS_GEMMA_CTX` | `4096` | Context window. Bigger ⇒ more memory per call. |
| `MEMOIRS_GEMMA_BATCH` | `512` | Prompt batch size. |
| `MEMOIRS_GEMMA_GPU_LAYERS` | `0` | Vulkan offload count. `99` = full GPU offload. |
| `MEMOIRS_GEMMA_CURATOR` | `auto` | `on` / `off` / `auto` — controls whether `decide_memory_action` calls the curator at consolidation time. |

Default model paths under `~/.local/share/memoirs/models/`:

```
qwen2.5-3b-instruct-q4_k_m.gguf       (~2.0 GB)  — recommended, 20/20 JSON
Phi-3.5-mini-instruct-Q4_K_M.gguf     (~2.4 GB)  — fallback, salvage parser
gemma-2-2b-it-Q4_K_M.gguf             (~1.6 GB)  — legacy, JSON-fragile
```

## Retrieval

| variable | default | meaning |
|---|---|---|
| `MEMOIRS_RETRIEVAL_MODE` | `hybrid` | `dense` / `bm25` / `hybrid` / `graph` / `hybrid_graph` / `raptor` / `hybrid_raptor`. |
| `MEMOIRS_RETRIEVAL_GEMMA` | `off` | `on` lets `_resolve_conflicts` consult the curator. Slower but smarter. |
| `MEMOIRS_RETRIEVAL_GEMMA_MAX` | `2` | Cap LLM calls per query when `RETRIEVAL_GEMMA=on`. Bounds tail latency. |
| `MEMOIRS_HYDE` | `off` | `on` runs query expansion via curator before retrieval. |
| `MEMOIRS_RERANKER_BACKEND` | `none` | `none` / `bge`. BGE rerankers cost ~50 ms per top-N=20. |
| `MEMOIRS_RERANK_TOP_N` | `50` | Only rerank the first N candidates (perf knob). |
| `MEMOIRS_MMR` | `on` | Maximal Marginal Relevance diversification. |
| `MEMOIRS_MMR_LAMBDA` | `0.7` | 1.0 = top-K by score, 0.0 = max diversity. |

## Embeddings

| variable | default | meaning |
|---|---|---|
| `MEMOIRS_EMBED_BACKEND` | `sentence_transformers` | `sentence_transformers` (default) or `fastembed` (ONNX, ~2× faster). |

The model itself is set in `memoirs/config.py` (`EMBEDDING_MODEL`, default `all-MiniLM-L6-v2`, 384-dim). LRU cache (`maxsize=1024`) wraps `embed_text` for query repeats.

## Privacy & multi-tenancy

| variable | default | meaning |
|---|---|---|
| `MEMOIRS_REDACT` | `on` | `on` / `off` / `strict`. `strict` raises if a secret is detected at ingest. |
| `MEMOIRS_USER_ID` | `local` | Scope writes & reads to this user. Single-user installs leave this alone. |
| `MEMOIRS_AGENT_ID` | _(unset)_ | Distinguish multiple agents under one user. |
| `MEMOIRS_NAMESPACE` | _(unset)_ | Free-form label (e.g. `work`, `personal`). |

## Lifecycle

| variable | default | meaning |
|---|---|---|
| `MEMOIRS_ZETTELKASTEN` | `on` | Auto-link new memories to top-k semantic neighbors on insert. |

## Observability

| variable | default | meaning |
|---|---|---|
| `MEMOIRS_LOG_FORMAT` | `text` if TTY else `json` | Force with `text` / `json`. |
| `MEMOIRS_LOG_LEVEL` | `INFO` | Standard Python logging levels. |
| `MEMOIRS_LOG_TARGET` | `file` | `file` / `stderr` / `both`. |
| `MEMOIRS_OTEL_ENDPOINT` | _(unset)_ | OTLP gRPC endpoint. Requires `pip install -e '.[otel]'`. |
| `MEMOIRS_OTEL_SERVICE_NAME` | `memoirs` | OTel service.name. |

## Daemon

| variable | default | meaning |
|---|---|---|
| `MEMOIRS_DAEMON_MAX_LOAD` | `0.85` | Skip extract tick when 1-min load > this. |
| `MEMOIRS_DAEMON_MIN_FREE_MEM_MB` | `512` | Skip extract tick when free memory below this. |
| `MEMOIRS_GRAPH_TTL` | `300` | Cache TTL (seconds) for `build_graph()` in PPR retrieval. |

## Optional extras

```bash
pip install -e '.[realtime]'      # watchdog (file watcher)
pip install -e '.[extract]'       # spaCy fallback NER
pip install -e '.[gemma]'         # llama-cpp-python (curator)
pip install -e '.[embeddings]'    # sentence-transformers + sqlite-vec
pip install -e '.[embeddings_fast]' # fastembed (ONNX)
pip install -e '.[reranker]'      # BGE cross-encoder
pip install -e '.[viz]'           # pyvis + networkx (graph rendering)
pip install -e '.[api]'           # FastAPI + uvicorn + jinja2 (HTTP + UI)
pip install -e '.[clustering]'    # scikit-learn (RAPTOR k-means)
pip install -e '.[privacy]'       # Microsoft Presidio
pip install -e '.[encryption]'    # sqlcipher3-binary
pip install -e '.[otel]'          # OpenTelemetry
pip install -e '.[dev]'           # pytest + coverage + httpx
pip install -e '.[all]'           # all of the above
```

Base install (no extras) runs end-to-end on heuristics — useful for CI smoke tests and zero-deps demos.
