# Benchmarks

All numbers below come from `scripts/bench_*.py` against a real, daily-used corpus of **4,196 active memories** on a Linux laptop (Ryzen-class CPU, 30 GB RAM, AMD iGPU via Vulkan). Reproduce them with the commands in each section.

## Retrieval latency

```
$ python -c "...assemble_context per mode, 8 queries × 3 reps, warm cache..."
```

| Mode | p50 | p95 | max | avg recall@10 | LLM calls |
|---|---|---|---|---|---|
| `bm25` (FTS5 only) | 2.7 ms | 63 ms | — | 3.4 / 10 | 0 |
| `dense` (sqlite-vec only) | 12.8 ms | 4.2 s¹ | — | 5.0 / 10 | 0 |
| `hybrid` (BM25 + dense via RRF) | 3.9 ms | 10 s¹ | — | 4.4 / 10 | 0 |
| `graph` (entity PPR, no embeddings) | 2.1 ms | 333 ms | — | 1.2 / 10 | 0 |
| **`hybrid_graph`** (RRF of hybrid + graph) | **6.2 ms** | **88 ms** | — | **5.0 / 10** | 0 |
| `raptor` / `hybrid_raptor` | … | … | … | … | 0 |

¹ Cold-start outliers were caused by `_resolve_conflicts` synchronously loading the curator on first call. Fixed by gating it behind `MEMOIRS_RETRIEVAL_GEMMA=on` (default off) and capping at `MEMOIRS_RETRIEVAL_GEMMA_MAX=2`.

**Recommendation:** `hybrid_graph` for best balance of recall and latency on graphs with > 100 entities.

## Curator JSON adherence

`scripts/bench_models.py` runs the production contradiction-detection prompt 20 times against each backend (same temperature 0.1, same max_tokens 200, same chat template per model).

| Backend | Raw JSON valid | After tolerant parser | p50 | p95 | output tokens |
|---|---|---|---|---|---|
| **qwen2.5-3b-Q4_K_M** | **20 / 20** | 20 / 20 | 870 ms | 1158 ms | 23 |
| phi-3.5-mini-Q4_K_M | 8 / 20 | 20 / 20 (12 salvaged) | 1387 ms | 2042 ms | 28 |
| gemma-2-2b-Q4_K_M | 0 / 20 | 0 / 20 | 83 ms | 304 ms | 1 |

Gemma's 0/20 is a chat-template + stop-token interaction, not a model capability problem — but the result is the same: it returns 1 token of output and the JSON parser sees nothing. Qwen 2.5 3B is the auto-detected default when its GGUF is present.

Reproduce:
```
python scripts/bench_models.py --n 20 --max-tokens 200
```

## Embedding cache

LRU cache around `embed_text(query)`:

| | mean | p50 | p95 |
|---|---|---|---|
| sentence-transformers cold | 9.94 ms | 9.73 ms | 11.51 ms |
| sentence-transformers + LRU mix (50/50 hit/miss) | 4.94 ms | 8.20 ms | 11.35 ms |
| **LRU hit only** | **0.005 ms** | **0.0047 ms** | **0.006 ms** |

≈ 2,070× speedup on repeated queries. `clear_embed_cache()` is exposed for tests; production has `maxsize=1024`.

Reproduce:
```
python scripts/bench_embed.py
```

## SQLite tunings

`scripts/bench_pragmas.py` runs 5 hot queries × 15 iterations on the live 2 GB DB.

| metric | default | tuned (`mmap=256MB`, `temp_store=MEMORY`, `cache=64MB`) | Δ |
|---|---|---|---|
| p50 | 3.1 ms | 3.1 ms | ≈ 0 (working set already in OS page cache) |
| p95 | 4.0 ms | 3.4 ms | **−15%** |
| max | 19.9 ms | 4.0 ms | **−80%** |

Tail wins concentrate on cold-page reads and contention with the write daemon. Median was already excellent.

## Test suite

```
$ .venv/bin/pytest tests/ --tb=no -q
693 passed, 2 skipped in 90s
```

Skipped tests gate on Microsoft Presidio (`pip install -e '.[privacy]'`) and SQLCipher (`pip install -e '.[encryption]'`) being available.

Coverage:
```
$ bash scripts/coverage.sh
TOTAL  60.10%   (28 → 60)
```

Critical files: `engine/memory_engine.py 88%`, `engine/lifecycle.py 91%`, `engine/embeddings.py 89%`, `mcp/tools.py 100%`.

## Methodology notes

- All numbers are **warm cache** unless explicitly stated. Cold start is documented separately.
- Bench machine: 1 CPU socket, 20 threads available, 30 GB RAM, no GPU offload by default.
- The DB is a real working corpus from daily use, not synthetic — 4,196 memories spanning Claude Code transcripts, Cursor sessions, ChatGPT exports.
- Numbers will vary on your hardware; the relative ordering between modes / backends should hold.

## What's next

- LongMemEval / LoCoMo head-to-head vs Mem0 / Zep / Letta / Cognee. Eval harness is already built (`memoirs eval --suite ...`); the missing piece is standing up the comparators locally and running each through the same dataset. Tracked in [`GAP.md`](../GAP.md) Phase 5 / Section E.
