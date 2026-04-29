# Curator model bench

Side-by-side scoreboard for every local LLM we evaluated as the memoirs curator. The curator handles `gemma_extract`, `gemma_consolidate`, `gemma_resolve_conflict`, and `gemma_summarize_project`.

All numbers below come from `scripts/bench_models_known_cases.py` and `scripts/bench_thinking_vs_no_think.py` running on the same Linux laptop (Ryzen-class CPU, 30 GB RAM, AMD iGPU via Vulkan), Q4_K_M quantization across the board, identical chat templates per model.

## Test suite

15 hand-crafted "ground-truth" cases:

- 10 contradiction-detection pairs (same vs different topic, with vs without superseding)
- 5 consolidation decisions (ADD / UPDATE / MERGE / IGNORE given a candidate + neighbors)

For each case, the correct action is obvious to a human, so model output is graded automatically. We track:

- **correct/n** — accuracy per task
- **parse fails** — JSON output unparseable even with the salvage parser
- **p50 / p95** — median + tail latency in ms

## Scoreboard (auto-detect order)

| Model | Contradiction | Consolidation | Parse fails | p50 | Notes |
|---|---|---|---|---|---|
| **qwen3-4b-instruct-2507** | 7/10 | **4/5** | 0 | 1062 ms | **Default** — wins on consolidation, balanced latency |
| qwen3.5-4b (no-think) | 7/10 | 2/5 | 0 | 1668 ms | Newer ≠ better for instruct workloads |
| qwen3.5-4b (thinking) | **8/10** | 2/5 | 0 | 1654 ms | Thinking gains 1 contradiction case (DB v3 → v9 supersede) |
| qwen2.5-3b (legacy) | 6/10 | 2/5 | 0 | 798 ms | Previous default; smaller and faster but less accurate |
| phi-3.5-mini | 6/10 | 2/5 | 0 | 1271 ms | No advantage over qwen2.5 |
| gemma-2-2b | 6/10 | 2/5 | 0 | 859 ms | Lightweight fallback (1.6 GB vs 2.5 GB Qwen3) |
| qwen3.5-2b | 2/10 | 1/5 | 8 | 4901 ms | Thinking emits chain-of-thought past max_tokens budget |
| deepseek-r1-distill-qwen-1.5b | 4/10 | 1/5 | 2 | 9280 ms | Always thinks; not viable for hot-path latency |
| smollm3-3b | n/a | n/a | — | — | Skipped — Jinja chat-template incompatibility with current llama-cpp-python |

## Decision

**Default: `qwen3-4b-instruct-2507-Q4_K_M`** (auto-detected when its GGUF is present at `~/.local/share/memoirs/models/qwen3-4b-instruct-q4_k_m.gguf`).

Override via `MEMOIRS_CURATOR_BACKEND ∈ {qwen3, qwen3.5, qwen, phi, gemma}`.

Why Qwen3 over the newer Qwen3.5:

1. **Consolidation wins 4/5 vs 2/5** — the `_chat_user_turn` work in `decide_memory_action` is consolidation-heavy.
2. **60% faster** (1062 ms vs 1668 ms p50).
3. **Thinking only gains 1 contradiction case** in our sample, not enough to offset the consolidation regression.

Qwen3.5 with thinking remains opt-in for niche workloads (sleep-cron `gemma_resolve_conflict` where temporal supersede detection matters) — set `MEMOIRS_CURATOR_BACKEND=qwen3.5` to enable.

## Important caveats

- **Sample size is small (15 cases)**. The +1 / −2 deltas above can be noise. A larger evaluation on LongMemEval or LoCoMo is on the roadmap (Phase 5E in the project tracker).
- **Per-model chat template matters**. We tracked down a Gemma-2 stop-token interaction that initially produced 0/20 valid JSON in synthetic prompts; the same Gemma scores 6/10 with the correct stops. Always verify the chat template before declaring a model "broken".
- **Reasoning models (`<think>...</think>`) need either `/no_think` in the system prompt OR a max_tokens budget large enough to absorb the preamble (~1024 tokens minimum)**. The `_strip_fences` parser in `engine/gemma.py` skips `</think>` blocks defensively.

## Reproduce

```bash
# All 7 chat-template-compatible models, both tasks, 15 cases each
python scripts/bench_models_known_cases.py \
  --models qwen2.5-3b-Q4,qwen3-4b-2507-Q4,qwen3.5-4b-Q4,qwen3.5-2b-Q4,deepseek-r1-distill-qwen-1.5b-Q4,phi-3.5-mini-Q4,gemma-2-2b-Q4

# Thinking-vs-no-think head-to-head on Qwen3.5-4B
python scripts/bench_thinking_vs_no_think.py
```

Both write JSON reports to `.memoirs/bench_*.json` for further analysis.
