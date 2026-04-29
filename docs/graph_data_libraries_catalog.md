# Graph + Data Quality Library Catalog for Memoirs

> Research as of 2026-04-28. Stack assumptions: SQLite + sqlite-vec + FTS5, fastembed (sentence-transformers/all-MiniLM-L6-v2 on CPU), single-file DB, single-user laptop, no GPU, hand-rolled Personalized PageRank in `memoirs/engine/graph_retrieval.py` (≤50k nodes, ≤20 iter, 1e-4 tol). Permissive licenses preferred (MIT/Apache/BSD); GPL/AGPL flagged.

## Summary table

| Library | Category | Version (latest) | License | Fits stack? | Recommendation |
|---|---|---|---|---|---|
| Kuzu | Graph DB (embedded) | 0.11.3 | MIT | Yes (file-based) | **Caution** — upstream abandoned Oct 2025, fork "bighorn"; powerful but risky |
| Memgraph + GQLAlchemy | Graph DB | GQLAlchemy 1.7+ | Apache-2.0 | No (server only) | Skip — needs separate Memgraph server |
| NetworkX | Graph algos | 3.6.1 | BSD-3 | Yes (pure Python) | **Drop-in replacement** for our PPR; SciPy-backed pagerank() with `personalization=` |
| igraph (python-igraph) | Graph algos | 0.11.8 | GPL-2.0 | Yes via wheels | **Skip** for distribution — GPL-2.0 is viral; great perf otherwise |
| graph-tool | Graph algos | 2.x | LGPL-3 | Heavy (Boost C++) | Skip — install pain (no PyPI wheels), LGPL OK but not worth it |
| RustWorkX | Graph algos | 0.17.1 | Apache-2.0 | Yes (Rust wheel) | **Strong drop-in** for PPR; 5–20× faster than NetworkX, mature |
| EasyGraph | Graph algos | 1.6 (Feb 2026) | BSD-3 | Yes | Useful for structural-hole / hypergraph; niche but actively shipped |
| cuGraph (RAPIDS) | Graph algos | 26.04.00 | Apache-2.0 | **No** (CUDA-only) | Skip — GPU required |
| graspologic | Graph algos (Leiden) | 3.x | MIT | Yes | **Adopt for hierarchical Leiden** if we add community-summary RAG |
| FAISS (faiss-cpu) | Vector search | 1.14.1 | MIT | Yes | Sidecar only — overkill vs sqlite-vec at our scale |
| hnswlib | Vector search | 0.8.0 (Dec 2023) | Apache-2.0 | Yes | Skip — stale; sqlite-vec already covers us |
| USearch | Vector search | 2.25.1 (Apr 2026) | Apache-2.0 | Yes | **Sidecar candidate** — single-file index, SQLite extension exists |
| ScaNN | Vector search | 1.4.x | Apache-2.0 | Linux+TF only | Skip — heavy TF dep, Linux-only |
| qdrant-client (local) | Vector DB | 1.17.1 | Apache-2.0 | Embedded mode | Skip — capped at ~20k points; we exceed |
| chromadb | Vector DB | 1.5.8 | Apache-2.0 | Yes | Skip — duplicates our SQLite layer |
| Milvus Lite | Vector DB | pymilvus 2.6.12 | Apache-2.0 | Linux/macOS only | Skip — separate `.db` file, no Windows |
| LanceDB | Vector DB | 0.30.2 | Apache-2.0 | Sidecar | Sidecar candidate for multimodal future |
| sqlite-vec | Vector search | 0.1.9 / 0.1.10-α | Apache-2.0 / MIT | **Already used** | Keep; track 0.1.10 for metadata fixes |
| DuckDB-VSS | Vector search | DuckDB ext | MIT | Sidecar | Skip — different DB engine |
| sqlite-vss | Vector search | deprecated | MIT | — | Skip — superseded by sqlite-vec (same author) |
| BAAI/bge-small-en-v1.5 | Embedding | — | MIT | Yes (CPU) | **Upgrade target** — 33M, 384-dim, beats MiniLM-L6 on MTEB |
| Snowflake/arctic-embed-xs | Embedding | — | Apache-2.0 | Yes (CPU) | **Top upgrade** — 22M params, 384-dim, fastembed-supported |
| Alibaba GTE-small | Embedding | — | MIT | Yes (CPU) | Drop-in alt to MiniLM, 33M params, 384-dim |
| intfloat/e5-small-v2 | Embedding | — | MIT | Yes (CPU) | Solid CPU pick, 33M params |
| jinaai/jina-embeddings-v2-small-en | Embedding | — | Apache-2.0 | Yes (CPU) | Useful when 8k context matters |
| nomic-embed-text-v1.5 | Embedding | — | Apache-2.0 | Heavier (137M) | Skip on CPU laptop unless quality > speed |
| BAAI/bge-reranker-v2-m3 | Reranker | — | MIT | Yes (CPU slow) | **Adopt as optional reranker stage**, gated by env var |
| jina-reranker-v3 | Reranker | 0.6B | CC-BY-NC-4.0 | **Non-commercial** | Flag — license restricts |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranker | — | Apache-2.0 | Yes (fast) | **Drop-in default reranker** if we add cross-encoder rerank |
| spaCy | NER | 3.8.x | MIT | **Already used** | Keep |
| GLiNER | NER (zero-shot) | 0.2.26 (Mar 2026) | Apache-2.0 | Yes (CPU) | **Strong addition** — zero-shot custom labels, ~166MB |
| NuExtract 2.0 | NER / extraction | Mar 2026 SDK | MIT models | Yes (smaller variants) | Sidecar for structured extraction (typed JSON) |
| ReLiK (SapienzaNLP) | EL + RE | — | Apache-2.0 | Yes (CPU) | **Adopt for joint EL+RE** — single forward pass, light |
| REL | EL | — | MIT | Heavy (full Wiki) | Skip — Wikipedia-scale KB, too heavy |
| BLINK | EL | — | MIT | Heavy | Skip — model >GB |
| ReFinED (Amazon) | EL | — | Apache-2.0 | Yes | Sidecar candidate if we adopt Wikidata KB |
| REBEL | RE | — | CC-BY-NC-SA-4.0 | **Non-commercial** | Flag — non-commercial license |
| LLMGraphTransformer (LangChain) | KG construction | exp. v0.x | MIT | Yes | Useful as reference, but pulls LC deps |
| Microsoft GraphRAG | KG + community RAG | 2.x | MIT | Heavyweight | Mine ideas (Leiden communities), not full adoption |
| LightRAG-HKU | KG-RAG | Apr 2026 release | MIT | Yes | Mine architecture ideas; full adoption is heavy |
| HippoRAG (OSU-NLP) | KG-RAG (PPR) | active 2026 | MIT | Yes | **Already aligned with our PPR**; borrow PPR-on-passage-nodes from HippoRAG 2 |
| instructor | LLM JSON | 1.14.5 (Jan 2026) | MIT | Yes | **Adopt for extraction** — Pydantic-typed LLM outputs |
| owlready2 | Ontology | 0.50 | LGPL-2.1 | Yes | Skip unless we add OWL ontology layer |
| rdflib | RDF | 7.x | BSD-3 | Yes | Useful only if we expose SPARQL surface |
| pyshacl | Schema validation | 0.31.0 (Jan 2026) | Apache-2.0 | Yes | Skip unless RDF route taken |
| rapidfuzz | Fuzzy strings | 3.14.5 | MIT | Yes (C++ wheel) | **Adopt for entity dedup** — replace any difflib usage |
| datasketch | MinHash LSH | 1.10.0 | MIT | Yes (pure Py) | **Adopt for near-dup memory detection** at ingest |
| splink | Record linkage | 4.x | MIT | Yes (DuckDB) | Sidecar for batch dedup, not online |
| dedupe.io | Record linkage | 3.x | MIT | Yes | Skip — needs labeled training data |
| recordlinkage | Record linkage | 0.16.x | BSD-3 | Yes | Useful for offline audits |
| pandera | DataFrame schemas | 0.31.1 (Apr 2026) | MIT | Yes | Skip — pandas-centric, we use sqlite |
| pydantic | Schemas | 2.x | MIT | **Already used** | Keep |
| great_expectations | Data quality | 1.17.0 | Apache-2.0 | Yes (heavy) | Skip — too heavy for single-file laptop DB |
| opentelemetry-sdk | Observability | 1.x | Apache-2.0 | Yes | **Adopt** for tracing |
| opentelemetry-instrumentation-sqlite3 | Observability | 0.62b1 (Apr 2026) | Apache-2.0 | Yes | **Adopt** — auto-traces sqlite3 cursors |
| Langfuse | LLM tracing | 3.x | MIT | Sidecar (server) | Optional for LLM-call tracing |
| Arize Phoenix | LLM tracing | latest | Elastic-2.0 | Sidecar | Flag — Elastic license is source-available, not OSI-approved |

---

## 1. Graph databases

### Kuzu — https://github.com/kuzudb/kuzu

- **Version:** 0.11.3 (PyPI). Last upstream release ≈ mid-2025; **Kùzu Inc. shut down October 2025**, project status now "abandoned by sponsor; community fork `bighorn` started by Kineviz".
- **What it provides:** Embedded property graph DB. Cypher, columnar storage, native HNSW vector index, FTS, file-based.
- **License:** MIT.
- **Footprint:** PyPI wheel, compiled C++. ~50–80MB install.
- **Fit:** File-based & in-process — slot-compatible with our single-file ethos but adds a *second* DB file alongside SQLite. Not a SQLite extension; you'd run Kuzu and SQLite side-by-side.
- **Performance:** Substantially faster traversal than NetworkX/igraph for large graphs (>1M edges). At our ≤50k node scale the gap is small.
- **Activity:** **Concerning** — upstream archived; fork is early.
- **Recommendation:** *Skip for now.* Don't pin Memoirs to an abandoned upstream. Revisit if `bighorn` gets traction in 2026.

### Memgraph + GQLAlchemy — https://github.com/memgraph/gqlalchemy

- **Version:** GQLAlchemy 1.7.x; Memgraph itself is a server.
- **What it provides:** OGM over Memgraph or Neo4j. Cypher.
- **License:** Apache-2.0 (GQLAlchemy). Memgraph community edition is BSL.
- **Footprint:** Python client small; **server is a separate process**. No embedded mode.
- **Fit:** Doesn't fit — single-user laptop, single-file DB philosophy excludes a server.
- **Recommendation:** *Skip.* Server-required.

### Apache AGE

- **License:** Apache-2.0. Postgres-only extension.
- **Recommendation:** *Skip — Memoirs is on SQLite.*

---

## 2. Graph algorithms (no graph DB)

### NetworkX — https://github.com/networkx/networkx

- **Version:** 3.6.1 (`pagerank` defaults to SciPy backend in 3.x).
- **License:** BSD-3.
- **Footprint:** Pure Python; needs SciPy for fast pagerank.
- **Fit:** Pure-drop-in for our PPR loop. `networkx.pagerank(G, personalization=…, alpha=0.5, max_iter=20, tol=1e-4)` is a 1-line replacement and yields the *same* algorithm we hand-rolled.
- **Performance:** SciPy backend is ~10× faster than our pure-Python loop for ≥10k nodes; below that, comparable.
- **Activity:** Very active.
- **Recommendation:** *Adopt as the canonical reference implementation* — replace our hand-rolled solver with `networkx.pagerank` behind a thin shim, keep our graph-build code. Removes a class of correctness risk.

### igraph (python-igraph) — https://github.com/igraph/python-igraph

- **Version:** 0.11.8 (PyPI as `igraph`).
- **License:** **GPL-2.0** ← copyleft; flag for distribution.
- **Footprint:** C wheels for all major platforms.
- **Performance:** Best-in-class CPU community detection (Leiden via `leidenalg`).
- **Recommendation:** *Skip if we ever distribute Memoirs as a binary or include in an Apache/MIT product.* GPL-2.0 propagates. If we keep Memoirs source-only and self-hosted, it's fine.

### graph-tool — https://graph-tool.skewed.de

- **License:** LGPL-3.
- **Footprint:** Boost C++; **no PyPI wheel**, must conda or system-package.
- **Recommendation:** *Skip* — install pain dwarfs the perf win at our scale.

### RustWorkX — https://github.com/Qiskit/rustworkx

- **Version:** 0.17.1.
- **License:** Apache-2.0.
- **Footprint:** Rust wheel, ~5MB.
- **Performance:** Typically 5–20× NetworkX on PageRank/BFS; pure-Python API mirrors NetworkX.
- **Activity:** Very active (Qiskit-maintained).
- **Recommendation:** *Adopt as the perf-tier PPR backend* — feature-flag via env, same interface as NetworkX shim.

### EasyGraph — https://github.com/easy-graph/Easy-Graph

- **Version:** 1.6 (Feb 1, 2026). 1.5.3 added Hypergraph Interchange Format.
- **License:** BSD-3.
- **Recommendation:** *Skip* unless we need structural-hole spanners or hypergraph features.

### cuGraph (RAPIDS) — https://github.com/rapidsai/cugraph

- **Version:** 26.04.00.
- **License:** Apache-2.0.
- **Footprint:** **CUDA 12.2+, Volta GPU minimum.**
- **Recommendation:** *Skip — laptop has no GPU.*

### graspologic — https://github.com/graspologic-org/graspologic

- **License:** MIT (Microsoft).
- **What it provides:** `partition.leiden`, `partition.hierarchical_leiden` — the algorithm Microsoft GraphRAG uses for community summaries.
- **Recommendation:** *Adopt if we add hierarchical-community summaries (GraphRAG-style)*; avoids the GPL of `leidenalg` and integrates with NetworkX/numpy graphs.

---

## 3. Vector / similarity search

### sqlite-vec — https://github.com/asg017/sqlite-vec

- **Version:** 0.1.9 stable; 0.1.10-α (Apr 2026).
- **License:** Apache-2.0 / MIT dual.
- **Status:** Already used.
- **Recommendation:** *Keep.* Watch 0.1.10 for the long-metadata-text DELETE fix.

### FAISS (faiss-cpu) — https://github.com/facebookresearch/faiss

- **Version:** 1.14.1.
- **License:** MIT.
- **Footprint:** PyPI wheel ~15MB; community-maintained.
- **Fit:** Excellent perf, but we don't beat sqlite-vec at our scale (≤50k–500k vectors with int8). Adds a second index to keep in sync.
- **Recommendation:** *Skip* unless we exceed ~10M vectors.

### hnswlib — https://github.com/nmslib/hnswlib

- **Version:** 0.8.0 (Dec 2023, no release in 12+ months).
- **License:** Apache-2.0.
- **Recommendation:** *Skip* — stale; `chroma-hnswlib` fork is more active but still niche.

### USearch — https://github.com/unum-cloud/usearch

- **Version:** 2.25.1 (Apr 16, 2026).
- **License:** Apache-2.0.
- **Footprint:** Single-file index, SQLite extension exists.
- **Notable:** User-defined metrics; can store 4B+ vectors via 40-bit refs.
- **Recommendation:** *Strongest sqlite-vec alternative if we ever need richer metrics*; keep sqlite-vec for now, USearch on the radar.

### ScaNN — https://github.com/google-research/google-research/tree/master/scann

- **License:** Apache-2.0.
- **Footprint:** Linux-only; pulls TF dep.
- **Recommendation:** *Skip* — too heavy.

### qdrant-client (local mode) — https://github.com/qdrant/qdrant-client

- **Version:** 1.17.1 (Mar 2026).
- **License:** Apache-2.0.
- **Local mode cap:** ~20k points (officially "dev/test only").
- **Recommendation:** *Skip.*

### chromadb — https://pypi.org/project/chromadb/

- **Version:** 1.5.8 (Apr 2026).
- **License:** Apache-2.0.
- **Backend:** SQLite + hnswlib.
- **Recommendation:** *Skip* — duplicates our stack.

### Milvus Lite (pymilvus) — https://github.com/milvus-io/milvus-lite

- **Version:** pymilvus 2.6.12 (Apr 2026). Lite is Linux/macOS only, no Windows.
- **License:** Apache-2.0.
- **Recommendation:** *Skip* — platform gap.

### LanceDB — https://github.com/lancedb/lancedb

- **Version:** 0.30.2 (Mar 2026).
- **License:** Apache-2.0.
- **Strengths:** Multimodal, billions of vectors, Lance file format.
- **Recommendation:** *Sidecar candidate* if we add image/audio memories; overkill for text-only.

### DuckDB-VSS — https://github.com/duckdb/duckdb-vss

- **License:** MIT.
- **Notes:** HNSW via USearch under the hood; index doesn't fit memory-managed buffer; experimental.
- **Recommendation:** *Skip* — different DB engine.

### sqlite-vss

- **Recommendation:** *Skip* — same author replaced it with sqlite-vec.

---

## 4. Embedding models (CPU, <500MB, faster than MiniLM-L6)

Reference baseline: `sentence-transformers/all-MiniLM-L6-v2` — 22M params, 384-dim, MTEB ≈ 56.

| Model | Params | Dim | License | MTEB | Fastembed-supported | Notes |
|---|---|---|---|---|---|---|
| `Snowflake/snowflake-arctic-embed-xs` | 22M | 384 | Apache-2.0 | ~62 | Yes | Same size as MiniLM, ~6 pts MTEB lift |
| `BAAI/bge-small-en-v1.5` | 33M | 384 | MIT | ~62 | Yes | Slightly larger, very strong CPU pick |
| `thenlper/gte-small` | 33M | 384 | MIT | 61.4 | Yes | Alibaba DAMO |
| `intfloat/e5-small-v2` | 33M | 384 | MIT | 59 | Yes | Reliable, fast |
| `jinaai/jina-embeddings-v2-small-en` | 33M | 512 | Apache-2.0 | 56 | Yes | Strength: 8k context |
| `nomic-ai/nomic-embed-text-v1.5` | 137M | 768 (Matryoshka) | Apache-2.0 | 62.4 | Yes | Heaviest — skip on laptop |

**Recommendation:** *Adopt `Snowflake/snowflake-arctic-embed-xs`* as a drop-in replacement for MiniLM-L6 — same dim (384), same CPU cost, ~6 MTEB points better. `bge-small-en-v1.5` is the tied runner-up. Both are supported by fastembed natively, so the swap is one model-id change.

---

## 5. Reranking

### `cross-encoder/ms-marco-MiniLM-L-6-v2` — https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2

- **License:** Apache-2.0 (model + sentence-transformers).
- **Performance:** 39 MRR@10 on MS-MARCO; ~1800 docs/s on V100, ~30–80 docs/s on CPU.
- **Recommendation:** *Drop-in default reranker.* Smallest, fastest, broadest support.

### `BAAI/bge-reranker-v2-m3` — https://huggingface.co/BAAI/bge-reranker-v2-m3

- **License:** MIT.
- **Size:** 278M params (slow on CPU).
- **Quality:** SOTA-class, multilingual (51.8 nDCG@10 BEIR).
- **Recommendation:** *Optional quality-tier reranker* gated by env var (`MEMOIRS_RERANKER=bge-m3`). Don't make it default — too slow on CPU.

### `jinaai/jina-reranker-v3` — https://huggingface.co/jinaai/jina-reranker-v3

- **License:** **CC-BY-NC-4.0** (non-commercial).
- **Recommendation:** *Flag — non-commercial.* Don't ship in a permissive product.

### Cohere reranker — API only, paid, network — *skip for local-first*.

---

## 6. Entity extraction / NER

### spaCy — https://github.com/explosion/spaCy

- **Version:** 3.8.x (3.8.13/14 patches in 2026).
- **License:** MIT.
- **Status:** Already used (`memoirs/engine/extract_spacy.py`).
- **Recommendation:** *Keep.* `memory_zone()` context manager (3.8) is worth wiring into long-running ingest paths.

### GLiNER — https://github.com/urchade/GLiNER

- **Version:** 0.2.26 (Mar 19, 2026).
- **License:** Apache-2.0.
- **Footprint:** ~166MB model, CPU-friendly.
- **Strength:** Zero-shot — pass any list of entity labels at inference time, no retraining.
- **Recommendation:** *Adopt* as a configurable second-pass NER on top of spaCy when users define custom entity types (project-specific labels like "feature flag", "incident").

### NuExtract 2.0 — https://github.com/numindai/nuextract

- **License:** MIT (model weights), Python SDK released Mar 2026.
- **Strength:** Schema-driven structured extraction (JSON), small variants run on CPU.
- **Recommendation:** *Sidecar candidate* — pair with `instructor` for typed extraction at consolidation time.

### REL — https://github.com/informagi/REL

- **License:** MIT.
- **Footprint:** Wikipedia-scale KB, multi-GB.
- **Recommendation:** *Skip* — KB too heavy for laptop.

### BLINK — https://github.com/facebookresearch/BLINK

- **License:** MIT, but model files >GB.
- **Recommendation:** *Skip* — same size issue.

### ReFinED (Amazon) — https://github.com/amazon-science/ReFinED

- **License:** Apache-2.0.
- **Strength:** Single forward pass: detection + typing + linking to Wikidata. ~200MB model.
- **Recommendation:** *Sidecar candidate* if we ever want Wikidata-grounded entity IDs (cross-conversation entity normalization).

### ReLiK (SapienzaNLP) — https://github.com/SapienzaNLP/relik

- **License:** Apache-2.0 (code), models on HF.
- **Strength:** Joint EL + RE in one pass; "small" variant runs on CPU.
- **Recommendation:** *Adopt* as a stronger replacement for spaCy + REBEL when extracting entity *and* relation triples for the graph.

---

## 7. Knowledge graph construction

### REBEL — https://github.com/Babelscape/rebel

- **License:** **CC-BY-NC-SA-4.0** (model). Code Apache-2.0.
- **Recommendation:** *Flag* — model weights are non-commercial. ReLiK is a permissive substitute.

### LLMGraphTransformer (LangChain experimental) — `langchain_experimental.graph_transformers.llm.LLMGraphTransformer`

- **License:** MIT.
- **Recommendation:** *Mine the prompt design* (good template for triple extraction); pulling LangChain experimental as a runtime dep is heavy. Reimplement the prompt + Pydantic schema with `instructor`.

### Microsoft GraphRAG — https://github.com/microsoft/graphrag

- **License:** MIT.
- **What it provides:** Indexer (entity + relation + claim extraction via LLM), hierarchical Leiden communities, community summaries, "drift search", "global search" over community reports.
- **Recommendation:** *Mine ideas, don't adopt full pipeline.* Specifically: borrow (a) hierarchical-Leiden-via-graspologic, (b) per-community summary memories, (c) the claims-extraction prompt. Avoid full GraphRAG runtime — it expects parquet-based indexes and is heavy.

### LightRAG-HKU — https://github.com/HKUDS/LightRAG

- **License:** MIT (EMNLP 2025).
- **Notable:** Dual-level retrieval (low-level entity-anchored + high-level keyword-anchored) over a graph. April 2026 release adds OpenSearch backend & Docker setup wizard.
- **Recommendation:** *Mine the dual-level retrieval idea*; the runtime is heavy and ships its own storage layer.

### HippoRAG (OSU-NLP) — https://github.com/OSU-NLP-Group/HippoRAG

- **License:** MIT.
- **Status:** HippoRAG 2 (NeurIPS'24) is what our PPR is inspired by.
- **Borrowable upstream improvements:** (a) including passage nodes (not just entities) directly in the PPR graph — what HippoRAG 2 calls "synonym + passage edges"; (b) the OpenIE-with-LLM pre-extraction prompt; (c) seed-weighting by NER salience. Our current implementation already does memory↔entity↔memory; aligning with HippoRAG 2's "synonym edges" between entities (cosine > τ on entity name embeddings) is a small, free quality win.
- **Recommendation:** *Cherry-pick HippoRAG 2 ideas:* synonym edges + passage-node integration. No need to depend on the package.

### `instructor` — https://github.com/567-labs/instructor

- **Version:** 1.14.5 (Jan 29, 2026).
- **License:** MIT.
- **Recommendation:** *Adopt* for any LLM-driven extraction in Memoirs (entities, relations, claims). Pydantic-typed outputs + retries + validation, multi-provider.

---

## 8. Ontology / schema

### owlready2 — https://pypi.org/project/owlready2/

- **Version:** 0.50 (Jan 4, 2026).
- **License:** **LGPL-2.1**.
- **Recommendation:** *Skip* unless we explicitly add an OWL ontology layer — overkill for a memory engine.

### rdflib — https://github.com/RDFLib/rdflib

- **License:** BSD-3.
- **Recommendation:** *Skip* unless we expose a SPARQL surface.

### pyshacl — https://github.com/RDFLib/pySHACL

- **Version:** 0.31.0 (Jan 16, 2026).
- **License:** Apache-2.0.
- **Recommendation:** *Skip* — couples with RDF stack.

---

## 9. Deduplication / fuzzy matching

### rapidfuzz — https://github.com/rapidfuzz/RapidFuzz

- **Version:** 3.14.5.
- **License:** MIT.
- **Recommendation:** *Adopt.* Replace any difflib/fuzzywuzzy use; use `process.cdist` for entity-name dedup at ingest.

### datasketch — https://github.com/ekzhu/datasketch

- **Version:** 1.10.0.
- **License:** MIT.
- **Footprint:** Pure Python + numpy.
- **Recommendation:** *Adopt* a `MinHashLSH` index keyed on memory-text shingles to detect near-duplicate memories at ingest (much cheaper than embedding similarity for the obvious-dup case).

### splink — https://github.com/moj-analytical-services/splink

- **License:** MIT.
- **Backend:** DuckDB / Spark / Athena.
- **Recommendation:** *Sidecar* for offline batch dedup audits — overkill for online ingest.

### dedupe.io — https://github.com/dedupeio/dedupe

- **License:** MIT.
- **Recommendation:** *Skip* — needs labeled training pairs.

### recordlinkage — https://github.com/J535D165/recordlinkage

- **License:** BSD-3.
- **Recommendation:** *Optional* for offline audits, not online.

---

## 10. Conflict detection / temporal reasoning

There is **no maintained Python library** for bi-temporal SQL pattern abstractions that fits our scale. `python-temporal-tables` is unmaintained. Postgres has `temporal_tables`; SQLite has none.

- **Recommendation:** *Keep the hand-rolled approach in `memoirs/engine/conflicts.py`.* Document the bi-temporal columns explicitly (valid_from, valid_to, recorded_from, recorded_to). Optionally add a tiny `conflicts/` module with constants and helpers — no library buys us much.

---

## 11. Data quality / validation

### great_expectations — https://github.com/great-expectations/great_expectations

- **Version:** 1.17.0 (Apr 22, 2026).
- **License:** Apache-2.0.
- **Footprint:** Heavy (~36MB sdist, many transitive deps).
- **Recommendation:** *Skip* — way too heavy for a single-user laptop memory engine.

### pandera — https://pandera.readthedocs.io

- **Version:** 0.31.1 (Apr 15, 2026).
- **License:** MIT.
- **Recommendation:** *Skip* — pandas/polars-centric; we're on raw sqlite rows.

### pydantic — already used. *Keep.* Pydantic v2 + `instructor` covers our LLM-output validation needs.

---

## 12. Observability / tracing

### opentelemetry-sdk — https://github.com/open-telemetry/opentelemetry-python

- **License:** Apache-2.0.
- **Recommendation:** *Adopt* — minimal cost, optional exporter.

### opentelemetry-instrumentation-sqlite3 — https://pypi.org/project/opentelemetry-instrumentation-sqlite3/

- **Version:** 0.62b1 (Apr 24, 2026).
- **License:** Apache-2.0.
- **Recommendation:** *Adopt* — auto-traces every cursor, useful for the latency budgets in `benchmarks.md`.

### Langfuse — https://github.com/langfuse/langfuse

- **License:** MIT (server is Apache-2.0 + commercial).
- **Recommendation:** *Optional sidecar* for LLM-call traces if/when we add Gemma/HyDE in the loop. Self-hostable.

### Arize Phoenix — https://github.com/Arize-ai/phoenix

- **License:** **Elastic-2.0** (source-available, not OSI-approved).
- **Recommendation:** *Flag.* Local-first dev tool, but the license isn't permissive.

---

## Also-check group

Already covered above:

- **Microsoft GraphRAG** — §7. *Mine ideas, don't adopt full pipeline.*
- **LightRAG-HKU** — §7. *Mine dual-level retrieval idea.*
- **HippoRAG (OSU-NLP)** — §7. *Cherry-pick synonym + passage-node edges into our existing PPR graph.*
- **Memgraph + GQLAlchemy** — §1. *Skip — server-required, no embedded mode.*
- **Apache AGE** — §1. *Skip — Postgres-only.*

---

## Recommended adoptions (top 5 by signal-to-effort)

Ranked for our SQLite + Python, no-GPU, single-user laptop stack.

1. **`Snowflake/snowflake-arctic-embed-xs`** (or `BAAI/bge-small-en-v1.5`) — *embedding upgrade.* Same 384-dim, same CPU cost as MiniLM-L6, ~6 MTEB points better. One-line model-id change in fastembed config; embeddings table doesn't even need a re-shape if we keep dim=384. **Effort: < 1 day. Signal: high — every retrieval improves.**

2. **NetworkX (default) + RustWorkX (perf-tier) for PPR** — *replace hand-rolled solver.* `networkx.pagerank(personalization=…)` is the canonical algorithm; RustWorkX is a 5–20× drop-in for it. Removes correctness risk in `graph_retrieval.py` and gives us a benchmarked perf path. **Effort: 1–2 days incl. tests. Signal: medium — same algorithm, fewer bugs, faster.**

3. **HippoRAG 2 synonym + passage-node edges** — *cheap graph quality boost.* Add (a) entity-synonym edges where `cosine(emb(name_i), emb(name_j)) > 0.85`, (b) treat memory passages as PPR nodes (we already do — but verify weights). No new dep; pure SQL + numpy. **Effort: 1–2 days. Signal: medium-high — directly improves multi-hop recall, the reason we have PPR.**

4. **`instructor` + GLiNER for entity/relation extraction** — *quality + flexibility upgrade over pure spaCy.* `instructor` for LLM-driven typed extraction at consolidation time; GLiNER for zero-shot custom-label NER at ingest. Both Apache-2.0/MIT, both CPU-friendly. **Effort: 2–3 days. Signal: high — fewer extraction errors flowing into the graph.**

5. **rapidfuzz + datasketch for dedup** — *data quality.* `rapidfuzz` for entity-name canonicalization, `datasketch.MinHashLSH` for near-duplicate memory detection at ingest. Both pure-ish Python, MIT, ms-scale per call. **Effort: 1–2 days. Signal: medium — keeps the graph from drifting into 17 variants of "Anthropic / anthropic / Anthropic Inc."**

Honorable mentions: `opentelemetry-instrumentation-sqlite3` (free observability), `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker (small, MIT, +precision@k).

---

Sources for the catalog (highlights):

- Kuzu: https://github.com/kuzudb/kuzu, https://www.theregister.com/2025/10/14/kuzudb_abandoned/
- RustWorkX: https://github.com/Qiskit/rustworkx, https://www.rustworkx.org/
- NetworkX: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html
- igraph: https://github.com/igraph/python-igraph, https://igraph.org/news.html
- graph-tool: https://graph-tool.skewed.de/
- EasyGraph: https://github.com/easy-graph/Easy-Graph
- cuGraph: https://github.com/rapidsai/cugraph
- graspologic: https://github.com/graspologic-org/graspologic
- FAISS: https://github.com/facebookresearch/faiss, https://pypi.org/project/faiss-cpu/
- hnswlib: https://github.com/nmslib/hnswlib
- USearch: https://github.com/unum-cloud/usearch, https://pypi.org/project/usearch/
- ScaNN: https://pypi.org/project/scann/
- qdrant-client: https://github.com/qdrant/qdrant-client
- chromadb: https://pypi.org/project/chromadb/
- Milvus Lite: https://github.com/milvus-io/milvus-lite
- LanceDB: https://github.com/lancedb/lancedb
- sqlite-vec: https://github.com/asg017/sqlite-vec
- DuckDB-VSS: https://github.com/duckdb/duckdb-vss
- BGE: https://huggingface.co/BAAI/bge-small-en-v1.5
- Arctic-embed: https://huggingface.co/Snowflake/snowflake-arctic-embed-xs
- GTE / E5 / Jina / Nomic: HF model cards
- BGE reranker v2-m3: https://huggingface.co/BAAI/bge-reranker-v2-m3
- ms-marco MiniLM: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
- Jina reranker v3: https://huggingface.co/jinaai/jina-reranker-v3
- spaCy: https://github.com/explosion/spaCy/releases
- GLiNER: https://github.com/urchade/GLiNER, https://pypi.org/project/gliner/
- NuExtract: https://github.com/numindai/nuextract
- ReLiK: https://github.com/SapienzaNLP/relik
- REL: https://github.com/informagi/REL
- BLINK: https://github.com/facebookresearch/BLINK
- ReFinED: https://github.com/amazon-science/ReFinED
- REBEL: https://github.com/Babelscape/rebel
- LangChain LLMGraphTransformer: https://python.langchain.com/api_reference/experimental/graph_transformers/langchain_experimental.graph_transformers.llm.LLMGraphTransformer.html
- Microsoft GraphRAG: https://github.com/microsoft/graphrag
- LightRAG: https://github.com/HKUDS/LightRAG
- HippoRAG: https://github.com/OSU-NLP-Group/HippoRAG
- instructor: https://github.com/567-labs/instructor
- owlready2: https://pypi.org/project/owlready2/
- rdflib: https://github.com/RDFLib/rdflib
- pyshacl: https://github.com/RDFLib/pySHACL
- rapidfuzz: https://github.com/rapidfuzz/RapidFuzz
- datasketch: https://github.com/ekzhu/datasketch
- splink: https://github.com/moj-analytical-services/splink
- dedupe.io: https://github.com/dedupeio/dedupe
- recordlinkage: https://github.com/J535D165/recordlinkage
- great_expectations: https://github.com/great-expectations/great_expectations
- pandera: https://pandera.readthedocs.io/
- opentelemetry-instrumentation-sqlite3: https://pypi.org/project/opentelemetry-instrumentation-sqlite3/
- Langfuse / Phoenix: https://github.com/langfuse/langfuse, https://github.com/Arize-ai/phoenix

---

## Summary for the user

- **Total libraries catalogued:** 53
- **Path to MD file (to be written upon plan approval):** `/home/misael/Desktop/projects/memoirs/docs/graph_data_libraries_catalog.md`
- **Top 3 recommendations:**
  1. Swap MiniLM-L6 → `Snowflake/snowflake-arctic-embed-xs` (or `bge-small-en-v1.5`): same 384-dim & CPU cost, ~6 MTEB pts better, one-line fastembed change.
  2. Replace hand-rolled PPR with `networkx.pagerank(..., personalization=...)` (default) and `rustworkx.pagerank` (perf tier) behind a thin shim.
  3. Cherry-pick HippoRAG 2 ideas: add entity-synonym edges (cosine > 0.85 on entity names) and confirm passage-node weights — pure-Python, no new dep, direct multi-hop recall lift.
