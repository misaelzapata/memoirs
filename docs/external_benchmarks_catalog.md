# External Benchmarks Catalog for Memory Engines

> Inventory of publicly available benchmarks usable for evaluating long-term memory
> engines for AI agents (Memoirs, Mem0, Zep, Letta, Cognee, Memori, LangMem,
> LlamaIndex, MemoryOS, Memoripy, Graphiti).
>
> **Status**: research-only. No downloads performed. Each entry includes a fetch
> recipe so a follow-up step can pull what we choose.
>
> **Scope notes**
> - "Engine bench" → measures the memory engine itself (retrieval, MRR, recall, etc.).
> - "LLM-with-memory bench" → measures end-to-end QA accuracy where memory is one of many factors.
> Some benchmarks support both modes; the per-benchmark detail clarifies.

## Summary table

| # | Benchmark | Focus | Sizes | Format | License | ~Size | Type |
|---|-----------|-------|-------|--------|---------|-------|------|
| 1 | LongMemEval | Info-extract, multi-session, temporal, knowledge-update, abstention | 500 Q × 3 variants (oracle/S/M) | HF + GitHub | Apache-2.0 (code) / CC-BY-NC (data per repo) | 15MB–~1GB | Both |
| 2 | LoCoMo | Long-conversation QA, summarization, multi-modal | 10 conversations, ~7k QAs | GitHub JSON | CC-BY-NC-4.0 | ~3MB | Both |
| 3 | LoCoMo-MC10 | Multiple-choice variant of LoCoMo | 1,986 MC items | HF dataset | derivative (CC-BY-NC) | <2MB | Engine |
| 4 | PerLTQA | Personal long-term memory: profiles, social, events, dialogues | 8,593 Q / 30 personas | GitHub | research/non-commercial | ~10MB | Engine |
| 5 | MSC (Multi-Session Chat) | Long-term open-domain dialogue persistence | 5k convs sessions 2-5 (+PersonaChat as session 1) | ParlAI | CC-BY (ParlAI standard) | ~80MB | Both |
| 6 | DMR (Deep Memory Retrieval) | Single-fact recall over MSC | 500 conv subset | MemGPT repo | research | <5MB | Engine |
| 7 | MemBench (ACL'25) | Effectiveness × efficiency × capacity, factual + reflective memory | tasks across multiple scenarios | GitHub | research | TBD on download | Both |
| 8 | MemoryBench (Oct 2025) | Memory + continual learning in LLM systems | multi-task | OpenReview/arXiv | research | TBD | LLM-with-memory |
| 9 | MemoryAgentBench | Accurate retrieval, test-time learning, long-range, conflict | 5 datasets incl. EventQA + FactConsolidation | HF + GitHub | Apache-2.0 (per repo) | ~50MB | Engine |
| 10 | EvolMem | Cognitive-driven multi-session dialogue memory | 1,600 dialogues, ~7 sessions avg | GitHub (planned) | TBD | TBD | Engine |
| 11 | AMA-Bench | Long-horizon agent memory (real + synthetic) | scalable | GitHub | research | TBD | Engine |
| 12 | BEAM | 1M & 10M-token long memory across 10 abilities | 128K/500K/1M/10M tiers | HF | research/CC-BY | ~GB at 10M | Both |
| 13 | RULER | Long-context retrieval, multi-hop tracing, aggregation | synthesizable, 13 tasks | GitHub | Apache-2.0 | generator-based | LLM |
| 14 | NIAH (Needle-in-a-Haystack) | Single-fact retrieval at varying depths/lengths | generator | GitHub | MIT | generator-based | LLM |
| 15 | BABILong | Long-context reasoning over distributed bAbI facts in PG19 | 20 tasks, scales to 11M tokens | HF | Apache-2.0 (per HF) | ~GB scaled | LLM |
| 16 | InfiniteBench (∞Bench) | 12 tasks at >100K tokens | 3.9k examples | HF + GitHub | Apache-2.0 | ~600MB | LLM |
| 17 | LongBench / LongBench v2 | Bilingual long-doc QA, multi-doc, summarization, code, dialogue | v1: 21 tasks; v2: 503 MCQs | HF | MIT (v1) / MIT (v2) | ~250MB | LLM |
| 18 | HotpotQA | 2-hop QA over Wikipedia w/ supporting facts | 113k Q | HF | CC-BY-SA-4.0 | ~600MB (full-wiki) / ~100MB (distractor) | Engine |
| 19 | 2WikiMultiHopQA | Strict 2-hop QA, comparison/composition/inference/bridge | ~192k Q | HF | Apache-2.0 | ~150MB | Engine |
| 20 | MuSiQue | 2-4 hop strict multi-hop QA (low shortcut leakage) | ~25k Q | GitHub / HF | CC-BY-4.0 | ~30MB | Engine |
| 21 | NaturalQuestions / PopQA | Single-hop factoid QA, long-tail entities | NQ ≈ 320k; PopQA 14k | HF | CC-BY-SA-3.0 / MIT | NQ ≈ 40GB; PopQA 5MB | Engine |
| 22 | CoQA | Conversational QA with coreference / reasoning | 127k Q across 8k convs | Stanford / HF | various source-paragraph licenses; CoQA itself research | ~50MB | Engine |
| 23 | QuAC | Conversational QA in context (information-seeking) | ~100k Q, 14k dialogues | website / HF | CC-BY-SA-4.0 | ~70MB | Engine |
| 24 | MTEB / BEIR retrieval subset (Quora, NFCorpus, MSMARCO, FiQA, SciFact) | Pure retrieval (NDCG@10, MRR@10, Recall@K) | varies | HF | per-dataset (mostly CC-BY/Apache) | 100MB–GB | Engine |

> Total catalogued: **24 benchmarks** (including LongMemEval which is already partially downloaded).

---

## Per-benchmark detail

### 1. LongMemEval
- **Paper**: Wu, Wang, Yu, Zhang, Chang, Yu — *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*, ICLR 2025. arXiv:2410.10813.
- **Source**: https://github.com/xiaowu0162/LongMemEval • https://xiaowu0162.github.io/long-mem-eval/
- **Measures**: 5 axes — information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention. Final metric is per-axis accuracy + macro avg.
- **Sizes / files**: 500 high-quality questions, three variants:
  - `longmemeval_oracle.json` — only the gold context (used for an "oracle ceiling" run).
  - `longmemeval_s.json` (and `_s_cleaned`) — ~115k tokens / ~30-40 sessions per Q.
  - `longmemeval_m.json` (and `_m_cleaned`) — ~1.5M tokens / ~500 sessions per Q.
- **Format**: JSON files released via HuggingFace; eval scripts in repo.
- **Download**: `git clone https://github.com/xiaowu0162/LongMemEval` then `huggingface-cli download xiaowu0162/longmemeval --repo-type dataset`. Local copy already at `data/longmemeval` (15MB → likely the oracle + S subset, M is ~1GB).
- **License**: code Apache-2.0; data CC-BY-NC-4.0 per repo header (verify before redistribution).
- **Strengths**: industry standard — Zep, Mem0, MemMachine, Cognee all publish on it. Oracle subset lets us pinpoint engine retrieval vs LLM error. The 5 axes overlap exactly with Memoirs primitives (recency, conflict resolution, temporal indexing).
- **Weaknesses**: synthetic (LLM-generated sessions). M variant requires GPU-class infra to run end-to-end.
- **Type**: works as both engine bench (oracle subset, retrieval-only metric) and LLM-with-memory bench (S / M).

### 2. LoCoMo
- **Paper**: Maharana et al., *Evaluating Very Long-Term Conversational Memory of LLM Agents*, ACL 2024. arXiv:2402.17753.
- **Source**: https://github.com/snap-research/locomo • https://snap-research.github.io/locomo/
- **Measures**: QA (single-hop, multi-hop, temporal, commonsense, adversarial), event summarization, multi-modal generation. Metrics: F1, BLEU-1, LLM-judge (J).
- **Sizes**: 10 conversations × ~600 dialogues × 9k tokens × up to 35 sessions; ~7k QA pairs.
- **Format**: single JSON drop (`./data/locomo10.json`).
- **Download**: `git clone https://github.com/snap-research/locomo` (the repo embeds the JSON). Or the clean MC variant: `huggingface-cli download Percena/locomo-mc10 --repo-type dataset`.
- **License**: CC-BY-NC-4.0.
- **Strengths**: de-facto comparison battlefield in 2024-25 — Mem0 (91.6 F1), Zep (75.14 J), MemMachine all quote this. Tiny (~3MB). Categorizes by reasoning type so we get sub-scores.
- **Weaknesses**: only 10 conversations → high variance, ceiling effect risk; methodology disputes between Mem0 and Zep (see issue #5 in `getzep/zep-papers`).
- **Type**: both. The QA subset is engine-friendly; summarization is LLM-with-memory.

### 3. LoCoMo-MC10
- **Source**: https://huggingface.co/datasets/Percena/locomo-mc10
- **Measures**: 1,986 multiple-choice questions derived from LoCoMo. Pure accuracy.
- **Format**: HF dataset, MC4-style fields.
- **Download**: `huggingface-cli download Percena/locomo-mc10 --repo-type dataset`
- **License**: derivative of LoCoMo (CC-BY-NC-4.0).
- **Strengths**: removes generation noise — pure retrieval/disambiguation signal. Perfect for ablation runs.
- **Weaknesses**: derivative (community), not paper-grade citation. MC = leak-prone.
- **Type**: engine bench.

### 4. PerLTQA
- **Paper**: Du et al., *PerLTQA: A Personal Long-Term Memory Dataset…*, SIGHAN 2024.
- **Source**: https://github.com/Elvin-Yiming-Du/PerLTQA
- **Measures**: classification, retrieval, synthesis. Splits memory into semantic (profile/social) and episodic (events/dialogues).
- **Sizes**: 8,593 questions over 30 personas.
- **Format**: JSON in GitHub repo (Chinese + English bilingual).
- **Download**: `git clone https://github.com/Elvin-Yiming-Du/PerLTQA`
- **License**: research / non-commercial (verify per-file headers).
- **Strengths**: maps cleanly to Memoirs's `type` taxonomy (`fact`/`preference`/`profile`/`event`). Good signal for memory classification.
- **Weaknesses**: heavy Chinese — embedding model must be multilingual. Smaller community adoption than LoCoMo.
- **Type**: engine bench.

### 5. MSC (Multi-Session Chat)
- **Paper**: Xu, Szlam, Weston — *Beyond Goldfish Memory: Long-Term Open-Domain Conversation*, ACL 2022. arXiv:2107.07567.
- **Source**: https://parl.ai/projects/msc/ (Facebook ParlAI)
- **Measures**: persona persistence, dialogue continuity. ParlAI metrics: F1, perplexity.
- **Sizes**: 5k full conversations sessions 2-5 (PersonaChat = session 1). 237k train / 25k valid examples.
- **Format**: ParlAI task. Also mirrored at `huggingface.co/datasets/nayohan/multi_session_chat`.
- **Download**: `pip install parlai && parlai display_data -t msc` or HF download.
- **License**: ParlAI standard (CC-BY-style for the data; check exact tag).
- **Strengths**: human-human (not synthetic). Realistic memory across sessions. The base for DMR (#6).
- **Weaknesses**: not designed as an *evaluation* benchmark — no labeled memory questions. Need to derive QA pairs.
- **Type**: both, but mostly used as the corpus for DMR.

### 6. DMR (Deep Memory Retrieval)
- **Paper**: Packer et al., *MemGPT: Towards LLMs as Operating Systems*, arXiv:2310.08560.
- **Source**: https://github.com/cpacker/MemGPT (now Letta) — `data/memgpt-dmr-eval`.
- **Measures**: single-turn fact recall over MSC subset. Accuracy.
- **Sizes**: 500 conversations from MSC.
- **Format**: JSON.
- **Download**: clone Letta/MemGPT repo, eval lives in `data/memgpt-dmr-eval`.
- **License**: research.
- **Strengths**: lightweight; enabled the head-to-head Zep (94.8%) vs MemGPT (93.4%) comparison.
- **Weaknesses**: known limitations — ambiguous question phrasing ("favorite drink to relax with"), no multi-hop, no temporal. 500 single-turn questions doesn't stretch a memory engine.
- **Type**: engine bench (retrieval).

### 7. MemBench (Tan et al., ACL Findings 2025)
- **Paper**: Tan, Zhang, Ma, Chen, Dai, Dong — *MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents*, arXiv:2506.21605.
- **Source**: https://github.com/import-myself/Membench
- **Measures**: factual memory + reflective memory across participation/observation scenarios; effectiveness, efficiency, capacity.
- **Sizes**: tasks released in repo (TBD exact counts).
- **Format**: GitHub.
- **Download**: `git clone https://github.com/import-myself/Membench`
- **License**: research.
- **Strengths**: only benchmark we found that scores efficiency/capacity not just accuracy — Memoirs's perf work (PRAGMA tuning, embed cache) has a place to show up.
- **Weaknesses**: brand-new (2025) so no rival numbers to compare against yet.
- **Type**: both.

### 8. MemoryBench (Oct 2025)
- **Paper**: arXiv:2510.17281 — *MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems*.
- **Source**: paper / OpenReview (https://openreview.net/forum?id=wU4Tjlzg3h).
- **Measures**: memory retention + continual learning interaction.
- **Format**: TBD — code links from the paper.
- **License**: TBD.
- **Strengths**: ties memory to continual learning explicitly — adjacent to Memoirs's importance/forgetting.
- **Weaknesses**: namespace-collision with the older "MemoryBench" line; verify which is which on download.
- **Type**: LLM-with-memory.

### 9. MemoryAgentBench
- **Paper**: HUST-AI / others, ICLR 2026 — *Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions*, arXiv:2507.05257.
- **Source**: https://github.com/HUST-AI-HYZ/MemoryAgentBench • https://huggingface.co/datasets/ai-hyz/MemoryAgentBench
- **Measures**: 4 axes — accurate retrieval, test-time learning, long-range understanding, conflict resolution. Plus two new datasets — **EventQA** (retrieval) and **FactConsolidation** (conflict resolution).
- **Sizes**: multi-dataset, ~50MB total.
- **Format**: HF dataset + GitHub.
- **Download**: `huggingface-cli download ai-hyz/MemoryAgentBench --repo-type dataset`
- **License**: per repo (likely Apache-2.0).
- **Strengths**: every axis maps onto a Memoirs feature. Conflict resolution especially — Memoirs already runs `mcp_get_context` with `conflicts_resolved` field. Includes Mem0 / MemGPT baseline numbers in the paper.
- **Weaknesses**: brand-new; harness still maturing.
- **Type**: engine bench.

### 10. EvolMem
- **Paper**: arXiv:2601.03543 (2026) — *EvolMem: A Cognitive-Driven Benchmark for Multi-Session Dialogue Memory*.
- **Source**: https://github.com/shenye7436/EvolMem (planned; verify on download date).
- **Measures**: 7 fine-grained abilities across declarative + non-declarative memory.
- **Sizes**: 1,600 dialogues, avg 6.82 sessions, 29.49 turns.
- **Format**: TBD (paper says "data and code will be released").
- **License**: TBD.
- **Strengths**: cognitive-science taxonomy aligns with Memoirs's `preference/fact/decision/style/credential_pointer` types.
- **Weaknesses**: code/data may not be live yet — check repo status before adding to harness.
- **Type**: both.

### 11. AMA-Bench
- **Paper**: arXiv:2602.22769 — *AMA-Bench: Evaluating Long-Horizon Memory for Agentic Applications*. Accepted ICLR 2026 Memory Agent workshop.
- **Source**: https://github.com/AMA-Bench/AMA-Bench • https://ama-bench.github.io/
- **Measures**: long-context retention + long-horizon memory in agentic scenarios. Two subsets — Real-world (machine-generated traces) + Synthetic (scalable horizon).
- **Format**: GitHub.
- **Download**: `git clone https://github.com/AMA-Bench/AMA-Bench`
- **License**: TBD on repo.
- **Strengths**: synthetic-scaling lets us push horizon length; real subset gives external validity.
- **Weaknesses**: very new; tooling parity with Memoirs's adapters not yet there.
- **Type**: engine bench.

### 12. BEAM
- **Paper**: arXiv (ICLR 2026) — *Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs*.
- **Source**: https://github.com/mohammadtavakoli78/BEAM • https://huggingface.co/datasets/Mohammadta/BEAM • https://huggingface.co/datasets/Mohammadta/BEAM-10M
- **Measures**: 10 memory abilities at 128K / 500K / 1M / 10M token tiers. Fine-grained "nugget" decomposition for partial recall scoring.
- **Sizes**: see tiers; 10M tier is the killer differentiator.
- **Format**: HF datasets.
- **Download**: `huggingface-cli download Mohammadta/BEAM --repo-type dataset` and `Mohammadta/BEAM-10M`.
- **License**: research / CC-BY (verify on HF cards).
- **Strengths**: only public benchmark at 10M tokens. Mem0 already publishes on it (64.1/48.6 1M/10M). Great showcase for Memoirs's bi-temporal storage.
- **Weaknesses**: 10M tier requires heavy infra to even ingest. We may only target the 1M tier initially.
- **Type**: both.

### 13. RULER
- **Paper**: Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?*, COLM 2024, arXiv:2404.06654.
- **Source**: https://github.com/NVIDIA/RULER
- **Measures**: 13 synthetic tasks in 4 categories — retrieval, multi-hop tracing, aggregation, QA.
- **Format**: generator script — produces synthetic samples at any length.
- **Download**: `git clone https://github.com/NVIDIA/RULER`
- **License**: Apache-2.0.
- **Strengths**: parameterizable. Synthetic = no leakage. Industry-standard for "effective context length" claims.
- **Weaknesses**: pure long-context — does NOT test memory engines that compress / archive. Engine wrapped around a small-context LLM looks worse than it is.
- **Type**: LLM-with-memory (or pure LLM long-context).

### 14. NIAH (Needle-in-a-Haystack)
- **Source**: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- **Measures**: single-fact retrieval at varying context depth × length. Heatmap output.
- **Format**: Python generator.
- **Download**: `git clone https://github.com/gkamradt/LLMTest_NeedleInAHaystack`
- **License**: MIT.
- **Strengths**: famous, fast, visual — great for marketing plots.
- **Weaknesses**: single-fact, single-needle = not memory-engine relevant; multi-needle extensions exist (Lance Martin).
- **Type**: LLM long-context.

### 15. BABILong
- **Paper**: arXiv:2406.10149 — *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack*, NeurIPS 2024.
- **Source**: https://github.com/booydar/babilong • https://huggingface.co/datasets/RMT-team/babilong
- **Measures**: 20 reasoning tasks (bAbI) hidden in PG19 background — fact chaining, deduction, counting.
- **Sizes**: 5k samples in train mirror; full set scales to 11M tokens.
- **Format**: HF dataset.
- **Download**: `huggingface-cli download RMT-team/babilong --repo-type dataset`
- **License**: Apache-2.0 (per HF card).
- **Strengths**: tests *reasoning* across distance, not just retrieval. Good complement to NIAH.
- **Weaknesses**: bAbI feels artificial. Not conversation-shaped.
- **Type**: LLM long-context.

### 16. InfiniteBench (∞Bench)
- **Paper**: arXiv:2402.13718 — *∞Bench: Extending Long Context Evaluation Beyond 100K Tokens*.
- **Source**: https://github.com/OpenBMB/InfiniteBench • https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench
- **Measures**: 12 tasks at 100K+ tokens — retrieval, math, code, novel QA.
- **Sizes**: ~3.9k examples.
- **Format**: HF.
- **Download**: `huggingface-cli download xinrongzhang2022/InfiniteBench --repo-type dataset`
- **License**: Apache-2.0.
- **Strengths**: diverse task mix; solid 100K-token harness.
- **Weaknesses**: like RULER, more LLM-context than memory-engine.
- **Type**: LLM long-context.

### 17. LongBench / LongBench v2
- **Paper**: ACL 2024 (v1) and 2025 (v2). https://longbench2.github.io/
- **Source**: https://github.com/THUDM/LongBench • https://huggingface.co/datasets/zai-org/LongBench • https://huggingface.co/datasets/THUDM/LongBench-v2
- **Measures**: v1 — 21 tasks across 6 categories (single-doc QA, multi-doc QA, summarization, few-shot, synthetic, code). v2 — 503 MCQs at 8k–2M words including long-dialogue history understanding.
- **Sizes**: v1 ~250MB; v2 ~50MB.
- **Format**: HF.
- **Download**: `huggingface-cli download THUDM/LongBench-v2 --repo-type dataset`
- **License**: MIT (per repo).
- **Strengths**: v2's long-dialogue-history split is directly memory-relevant. MIT.
- **Weaknesses**: MCQ format leaks. Mostly LLM-with-memory, not pure engine.
- **Type**: LLM-with-memory.

### 18. HotpotQA
- **Paper**: Yang et al., EMNLP 2018. arXiv:1809.09600. https://hotpotqa.github.io/
- **Source**: https://huggingface.co/datasets/hotpotqa/hotpot_qa
- **Measures**: 2-hop QA, with sentence-level supporting facts. EM, F1, supporting-fact F1, joint EM.
- **Sizes**: 113k Q. Two settings — `distractor` (10 paragraphs incl. 2 gold) and `fullwiki` (retrieve from full Wikipedia).
- **Format**: HF.
- **Download**: `huggingface-cli download hotpotqa/hotpot_qa --repo-type dataset`
- **License**: CC-BY-SA-4.0.
- **Strengths**: gold standard for **bridging multi-hop** retrieval; mature; HippoRAG, MemoryOS all evaluate on it. The bridging case Memoirs just fixed has a direct A/B run here.
- **Weaknesses**: not conversation-shaped — must wrap each Q as a synthetic single-turn memory.
- **Type**: engine bench (retrieval).

### 19. 2WikiMultiHopQA
- **Paper**: Ho et al., COLING 2020.
- **Source**: https://huggingface.co/datasets/xanhho/2WikiMultihopQA • https://huggingface.co/datasets/framolfese/2WikiMultihopQA (HotpotQA-shaped fields)
- **Measures**: strict 2-hop reasoning — comparison, composition, inference, bridge.
- **Sizes**: ~192k Q.
- **Format**: HF, HotpotQA-compatible schema.
- **Download**: `huggingface-cli download xanhho/2WikiMultihopQA --repo-type dataset`
- **License**: Apache-2.0 (per HF card).
- **Strengths**: cleaner 2-hop reasoning chains than HotpotQA (less shortcut). Used in HippoRAG.
- **Weaknesses**: same as HotpotQA — wiki, not conversation.
- **Type**: engine bench.

### 20. MuSiQue
- **Paper**: Trivedi et al., TACL 2022.
- **Source**: https://github.com/StonyBrookNLP/musique
- **Measures**: 2-4 hop strict reasoning, designed to defeat shortcut retrievers. EM, F1.
- **Sizes**: ~25k Q.
- **Format**: GitHub release JSONs; HF mirrors exist.
- **Download**: `git clone https://github.com/StonyBrookNLP/musique`
- **License**: CC-BY-4.0.
- **Strengths**: hardest of the wiki multi-hop bench triplet (HotpotQA + 2Wiki + MuSiQue is the standard "associativity" combo HippoRAG uses).
- **Weaknesses**: same shape limitation as HotpotQA.
- **Type**: engine bench.

### 21. NaturalQuestions / PopQA (single-hop factoid)
- **Sources**:
  - https://huggingface.co/datasets/google-research-datasets/natural_questions
  - https://huggingface.co/datasets/akariasai/PopQA
- **Measures**: single-hop factual recall; PopQA emphasizes long-tail entities.
- **Sizes**: NQ ≈ 320k Q; PopQA 14k Q with 1.4k rare-entity slice.
- **Format**: HF.
- **Download**: `huggingface-cli download akariasai/PopQA --repo-type dataset` (small) and `google-research-datasets/nq_open` for the lighter open variant.
- **License**: NQ → CC-BY-SA-3.0; PopQA → check (HF card lists MIT for several mirrors).
- **Strengths**: factual-memory baseline used by HippoRAG. PopQA tests "rare-entity memory" — which is exactly where vector-only retrieval fails.
- **Weaknesses**: full NQ is 40GB. Also single-hop — easy ceiling.
- **Type**: engine bench.

### 22. CoQA
- **Paper**: Reddy, Chen, Manning, TACL 2019. https://stanfordnlp.github.io/coqa/
- **Source**: https://github.com/stanfordnlp/coqa • https://huggingface.co/datasets (multiple mirrors)
- **Measures**: conversational QA with coreference & reasoning across the dialogue. F1, EM.
- **Sizes**: 127k Q over 8k convs, 7 domains.
- **Format**: JSON.
- **Download**: direct from Stanford or HF mirror.
- **License**: source-passage licenses vary; CoQA itself research-only.
- **Strengths**: in-context multi-turn — small-scale memory test (questions reference earlier turns).
- **Weaknesses**: passages provided in-context → does not exercise *external* memory store.
- **Type**: engine bench (only if we hide the passage and force retrieval).

### 23. QuAC
- **Paper**: Choi et al., EMNLP 2018. https://quac.ai/
- **Source**: https://huggingface.co/datasets/allenai/quac
- **Measures**: information-seeking conversational QA.
- **Sizes**: ~100k Q across 14k dialogues.
- **Format**: HF.
- **Download**: `huggingface-cli download allenai/quac --repo-type dataset`
- **License**: CC-BY-SA-4.0.
- **Strengths**: more realistic information-seeking than CoQA; turns reference each other.
- **Weaknesses**: same — passage provided.
- **Type**: engine bench (only if reformulated).

### 24. MTEB / BEIR retrieval subset
- **Paper**: Muennighoff et al., *MTEB: Massive Text Embedding Benchmark*, EACL 2023. (BEIR — Thakur et al., NeurIPS 2021).
- **Source**: https://github.com/embeddings-benchmark/mteb • https://huggingface.co/mteb
- **Measures**: pure retrieval — NDCG@10, MRR@10, Recall@K. Relevant subsets:
  - `QuoraRetrieval` — duplicate-question.
  - `NFCorpus` — bio-medical IR.
  - `MSMARCO` — passage retrieval.
  - `FiQA2018` — financial QA retrieval.
  - `SciFact` — claim-verification retrieval.
  - `TRECCOVID`, `BioASQ`, `Touche2020`, `CQADupStack` — domain retrieval.
- **Format**: HF datasets, accessible via `mteb` Python package or direct `load_dataset`.
- **Download**: `pip install mteb` then run `mteb.MTEB(tasks=["NFCorpus","Quora","SciFact"])`.
- **License**: per-dataset (mostly CC-BY / Apache).
- **Strengths**: cleanest possible retrieval signal. If Memoirs's hybrid scoring loses on NFCorpus then we know it's the embeddings, not the engine logic. Reproducible — every embedding library publishes here.
- **Weaknesses**: not conversation-shaped at all. Pure IR. Doesn't exercise temporal / conflict / forgetting.
- **Type**: engine bench (retrieval only).

---

## Benchmarks deferred / not promising

- **HotpotQA-fullwiki vs distractor**: same as #18, just two settings. Distractor is enough.
- **TriviaQA + personalized profile injection**: no off-the-shelf personalized variant of TriviaQA was found. We'd have to fabricate the personalization layer ourselves — falls into "internal synthetic" rather than "academic standard". Skip unless we want a custom variant.
- **PILE / OpenAssistant conversations as memory eval**: Pile is a corpus, not a benchmark. OpenAssistant has dialogues but no memory-recall evaluation set. Skip.
- **AgentBench / ToolBench**: agent-capability benches, very little memory signal — recall the paper says LLM agents fail mostly on long-term reasoning, but the benches aren't *measuring* memory in isolation. Skip for memory-engine eval.
- **Mem0 official `memory-benchmarks` repo (https://github.com/mem0ai/memory-benchmarks)**: not a standalone benchmark — it's a *harness* over LoCoMo + LongMemEval + BEAM. Worth using as a reference harness (see "Reference harnesses" below) but it's not a separate dataset.
- **supermemoryai/memorybench (https://github.com/supermemoryai/memorybench)**: also a harness across LoCoMo and others. Same note.
- **Anthropic evals repo**: focuses on persona/sycophancy/safety, not memory. Skip.
- **MultiHop-RAG (Tang et al., COLM 2024)**: relevant but already represented by HotpotQA + 2Wiki + MuSiQue. Optional add.

---

## Reference harnesses (not benchmarks themselves, but worth shadowing)

| Repo | What it gives us |
|------|------------------|
| https://github.com/mem0ai/memory-benchmarks | Reference adapters for LoCoMo + LongMemEval + BEAM; we can crib their harness shape so our numbers are directly comparable. |
| https://github.com/supermemoryai/memorybench | Multi-dataset harness with engine adapters. |
| https://github.com/HUST-AI-HYZ/MemoryAgentBench | Includes Mem0 + MemGPT baseline runners. |

---

## Recommended subset for Memoirs

Memoirs is **single-user, local-first, with bi-temporal queries**. It already has primitives for:
- weighted scoring (recency + importance + relevance)
- A-Mem Zettelkasten links
- LRU embedding cache + tuned SQLite
- conflict resolution (`mcp_get_context` returns `conflicts_resolved`)

Given that, the top-5 by **signal / effort**:

| Rank | Benchmark | Why it wins | Effort | Expected signal |
|------|-----------|-------------|--------|-----------------|
| 1 | **LongMemEval** (oracle + S) | Already partially have. Hits 5 axes that align 1:1 with Memoirs primitives (temporal, knowledge update, conflict). Industry-standard — every rival publishes here. Oracle subset → pure engine signal. | LOW (data already present, harness needed) | HIGH — direct comparability with Mem0/Zep/Cognee public numbers. |
| 2 | **LoCoMo** (snap-research, JSON drop) | The single most-quoted comparison battlefield in 2024-25. 3MB — trivial. Sub-scores by reasoning type (single/multi/temporal/adversarial). | LOW (clone repo + JSON) | HIGH — crowded field of public numbers to beat. |
| 3 | **MemoryAgentBench** | Newest, explicitly tests retrieval + test-time learning + long-range + conflict. The 4 axes Memoirs already has. Includes Mem0/MemGPT baselines in-paper. EventQA + FactConsolidation test exactly the bi-temporal / merge logic. | MEDIUM (HF download, new harness) | HIGH — fresh, methodologically clean, cites our exact axes. |
| 4 | **HotpotQA + 2WikiMultiHopQA + MuSiQue** (the HippoRAG triplet) | The only way to credibly claim "multi-hop bridging works". Memoirs just fixed the bridging case — this is the validation. ~CC-BY licenses → safe to redistribute results. | MEDIUM (3 datasets, but all HF + HotpotQA-shaped) | HIGH for the bridging story; MEDIUM as a generic memory-engine bench (not conversation-shaped). |
| 5 | **MTEB retrieval subset** (NFCorpus + Quora + SciFact + FiQA) | Pure retrieval signal — isolates embedding-quality from engine logic. Tiny (<200MB). MTEB results are universally citable. | LOW (`mteb` python package handles everything) | MEDIUM — confirms hybrid retrieval, not memory specifically. Useful as a "we don't lose on plain IR" sanity check. |

### What we'd skip / defer

- **BEAM 10M tier** — too heavy to ingest on the dev box; revisit when we have GPU infra.
- **NIAH / RULER / BABILong / InfiniteBench** — all measure long-context LLM behavior more than memory engine. Useful only if we publish a "memoirs + small-context model beats long-context model" story.
- **CoQA / QuAC** — provide passages in-context; they don't exercise external memory.
- **DMR** — too small (500 single-fact Q's). Run only as a sanity smoke test.

### Proposed download plan (next step)

Single shell script under `scripts/fetch_external_benches.sh` that:
1. Verifies/extends `data/longmemeval` to all three variants.
2. `git clone --depth 1 https://github.com/snap-research/locomo data/locomo`.
3. `huggingface-cli download ai-hyz/MemoryAgentBench --repo-type dataset --local-dir data/memoryagentbench`.
4. `huggingface-cli download hotpotqa/hotpot_qa --repo-type dataset --local-dir data/hotpotqa` (distractor split only).
5. `huggingface-cli download xanhho/2WikiMultihopQA --repo-type dataset --local-dir data/2wiki`.
6. `git clone --depth 1 https://github.com/StonyBrookNLP/musique data/musique`.
7. `pip install mteb` and persist a config of which subsets to run.

Estimated total disk ≈ 1.3 GB.
