"""Layer 2 — extraction of memory candidates.

Tries Gemma via `llama-cpp-python` if installed and a model is configured.
Falls back to a stdlib heuristic extractor so the pipeline runs end-to-end with
zero external deps. Both paths produce the same Candidate shape and pass
through the same validators before persistence.

Memory types (from chat-blueprint.md):
    preference | fact | project | task | decision | style | credential_pointer
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config import (
    CURATOR_BACKEND,
    CURATOR_MODEL_PATH,
    GEMMA_DEFAULT_BATCH,
    GEMMA_DEFAULT_CTX_TOKENS,
    GEMMA_DEFAULT_THREADS,
    GEMMA_MAX_CONTEXT_CHARS,
    GEMMA_MAX_OUTPUT_TOKENS,
)
from ..core.ids import stable_id, utc_now
from ..core.normalize import should_skip_extraction
from ..db import MemoirsDB


log = logging.getLogger("memoirs.gemma")

ALLOWED_TYPES = {
    "preference",
    "fact",
    "project",
    "task",
    "decision",
    "style",
    "credential_pointer",
    "procedural",
}

GEMMA_MODEL_PATH = Path(
    os.environ.get(
        "MEMOIRS_GEMMA_MODEL",
        Path.home() / ".local/share/memoirs/models/gemma-2-2b-it-Q4_K_M.gguf",
    )
)


# Active curator model path resolved from config (auto-detect Qwen > Phi > Gemma).
# Override per-process with MEMOIRS_CURATOR_BACKEND or MEMOIRS_CURATOR_MODEL.
ACTIVE_CURATOR_PATH = CURATOR_MODEL_PATH


def _chat_user_turn(prompt: str) -> str:
    """Wrap a user prompt in the chat template for the active curator backend.

    Validated by scripts/bench_models_known_cases.py — Qwen3-4B-Instruct-2507
    is the highest-scoring local curator we tested (7/10 contradiction + 4/5
    consolidation, vs 2/5 for every other model).

    For Qwen3 / Qwen3.5 we append ``/no_think`` to the system prompt to disable
    the reasoning preamble that those models emit by default — keeps JSON in
    the response slot rather than hidden behind a long ``<think>...</think>``
    block.
    """
    backend = CURATOR_BACKEND
    if backend in ("qwen", "qwen3", "qwen3.5"):
        sys_msg = "You output ONLY the requested JSON, no prose."
        if backend in ("qwen3", "qwen3.5"):
            sys_msg += " /no_think"
        return (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if backend == "phi":
        return (
            "<|system|>\nYou output ONLY the requested JSON, no prose.<|end|>\n"
            f"<|user|>\n{prompt}<|end|>\n"
            "<|assistant|>\n"
        )
    # gemma 2 (default)
    return (
        "<start_of_turn>user\n"
        + prompt
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def _chat_stops() -> list[str]:
    """Stop tokens appropriate to the active curator backend."""
    backend = CURATOR_BACKEND
    if backend in ("qwen", "qwen3", "qwen3.5"):
        return ["<|im_end|>", "```\n\n", "\n\n\n"]
    if backend == "phi":
        return ["<|end|>", "```\n\n", "\n\n\n"]
    return ["<end_of_turn>", "}\n\n", "\n\n\n"]


@dataclass
class Candidate:
    type: str
    content: str
    importance: int = 3
    confidence: float = 0.5
    entities: list[str] = field(default_factory=list)
    source_message_ids: list[str] = field(default_factory=list)
    extractor: str = "heuristic"

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------
# Validators (run regardless of which extractor produced the candidate)
# ----------------------------------------------------------------------

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password|passwd|bearer)\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
]


def validate_allowed_memory_type(c: Candidate) -> bool:
    return c.type in ALLOWED_TYPES


def validate_confidence(c: Candidate) -> bool:
    """Reject candidates with out-of-range confidence or importance."""
    try:
        conf = float(c.confidence)
        imp = int(c.importance)
    except (TypeError, ValueError):
        return False
    return 0.0 <= conf <= 1.0 and 1 <= imp <= 5


def validate_no_secrets(c: Candidate) -> bool:
    return not any(p.search(c.content) for p in _SECRET_PATTERNS)


def validate_json_output(text: str) -> Any:
    """Parse JSON, repairing markdown fences and truncated arrays.

    Gemma sometimes wraps output in ```json ... ``` despite the prompt saying
    "no markdown fences". We strip those before parsing. We also handle
    `max_tokens` mid-array truncation by salvaging any complete top-level
    objects from the array.
    """
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (with optional language tag)
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        # Remove closing fence
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3].rstrip()
    text = cleaned

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if not text.lstrip().startswith("["):
        # Look for the array start later in the string
        idx = text.find("[")
        if idx == -1:
            raise ValueError(f"invalid JSON from extractor: no array found: {text[:120]}")
        text = text[idx:]
    # Salvage complete objects: scan brace-by-brace, parsing each top-level dict.
    salvaged: list = []
    decoder = json.JSONDecoder()
    i = text.find("[") + 1
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            salvaged.append(obj)
        i = end
    if not salvaged:
        # Truncated mid-first-object or pure garbage. Treat as "nothing extracted"
        # so the cascade falls through to spaCy without raising — noisy warnings
        # for what's effectively a no-op signal aren't useful.
        log.debug("validate_json_output: no salvageable objects (head=%s)", text[:120])
        return []
    return salvaged


def detect_sensitive_content(c: Candidate) -> bool:
    """True if candidate looks like a secret (rejected before persistence)."""
    return not validate_no_secrets(c)


# ----------------------------------------------------------------------
# Gemma path (optional)
# ----------------------------------------------------------------------

_CURATOR_SYSTEM_INSTRUCTIONS = """You extract durable, user-relevant memory from chat transcripts.

CRITICAL OUTPUT FORMAT:
- Start with `[` and end with `]`. Nothing before, nothing after.
- DO NOT wrap in ```json or ``` markdown fences.
- DO NOT add prose before or after.
- Bad: ```json\\n[...]\\n```
- Good: [...]

Each item:
{"type": one of [preference, fact, project, task, decision, style, credential_pointer],
 "content": "concise statement, ≤140 chars",
 "importance": 1..5, "confidence": 0..1,
 "entities": ["normalized entity names"]}

# Type-specific rules — IMPORTANT, biased AGAINST generic facts:

- **preference**  — user's stated preferences, defaults, opinions ("prefers X over Y", "always uses Z").
                    Importance 3-5. Aim for these AGGRESSIVELY: they're the most valuable signal.

- **decision**    — explicit commitments to a path ("will use X", "decided on Y", "switching from A to B").
                    Importance 3-5.

- **task**        — concrete things the user wants done ("add tests to X", "fix the watcher").
                    Importance 2-4. Skip vague intents ("we should improve quality").

- **project**     — emit AGGRESSIVELY when a github repo, codebase name, product, or working directory is
                    mentioned. Examples: "memoirs", "gocracker", "fastfn". Importance 3-5.
                    If a name appears more than once → it's almost always a project.

- **style**       — communication / coding style preferences ("answers should be concise", "no emojis",
                    "prefers async/await over callbacks"). Importance 2-4.

- **credential_pointer** — if user mentions storing keys/tokens/passwords, emit the LOCATION not the value
                    ("user has GitHub PAT in 1Password", NOT the actual token). Importance 4-5.

- **fact**        — RESERVE FOR EXCEPTIONAL CASES. Use type=fact ONLY for durable, user-specific context
                    that doesn't fit any type above (e.g. "user works on local-first systems", "user is
                    based in Buenos Aires"). NEVER use fact for:
                    × tool output, file paths, code snippets, version numbers
                    × generic statements about the world ("Python is interpreted", "SQLite is a database")
                    × statements about external systems unless tied to a user decision
                    × command outputs, log lines, build errors
                    Most candidates that LOOK like facts are actually ruido — emit nothing.
                    Aim for ≤ 20% of total emitted candidates being fact.

# Confidence calibration (REALISTIC distribution required):

- 0.95-1.00 — user explicitly stated this in unambiguous terms ("I prefer X")
- 0.75-0.94 — strongly implied by direct context
- 0.50-0.74 — reasonable inference from one or two messages
- 0.30-0.49 — speculative, only emit if importance is high
- below 0.30 — DO NOT emit; return nothing rather than guess

DO NOT emit everything with confidence=1.0. If you find yourself doing that, you're over-extracting.

# Importance calibration:

- 5 — core identity / always-applicable preference / mission-critical decision
- 4 — strong durable preference, project-level decision, key task
- 3 — useful but project-scoped, will likely apply for weeks
- 2 — interesting tactical detail
- 1 — trivial — usually means SKIP IT

# Hard exclusions — NEVER extract:

- greetings, clarifications, status messages ("ok", "thanks", "let me check")
- questions ("what does X do?", "how do I...?")
- code dumps, tool output, file contents
- IDE/system injections wrapped in <ide_*>, <system-reminder>, <bash-*>, <command-*>, <local-command-*>
- statements not about THE USER (general world facts)
- if user changes their mind in the same conversation, ONLY emit the LATEST version

# Examples of GOOD extractions:

  {"type":"preference","content":"prefers Python over Go for prototyping","importance":4,"confidence":0.9,"entities":["Python","Go"]}
  {"type":"decision","content":"will use SQLite + sqlite-vec instead of Postgres","importance":4,"confidence":0.85,"entities":["SQLite","sqlite-vec","Postgres"]}
  {"type":"project","content":"working on memoirs (local-first memory engine)","importance":5,"confidence":0.95,"entities":["memoirs"]}
  {"type":"style","content":"prefers terse responses over long explanations","importance":3,"confidence":0.85,"entities":[]}
  {"type":"task","content":"add pytest tests for the spaCy extractor","importance":3,"confidence":0.7,"entities":["pytest","spaCy"]}

# Examples of BAD extractions (do NOT produce these):

  ✗ {"type":"fact","content":"Python is interpreted","importance":3,"confidence":0.9}
    — generic world fact, not user-specific
  ✗ {"type":"fact","content":"/home/misael/Desktop/projects/foo","importance":2,"confidence":0.8}
    — file path, not durable knowledge
  ✗ {"type":"task","content":"run the command","importance":3,"confidence":1.0}
    — too vague, no actionable detail
  ✗ {"type":"fact","content":"the file has 1321 lines","importance":1,"confidence":1.0}
    — tool output / volatile state
  ✗ {"type":"preference","content":"thinks SQLite is good","importance":3,"confidence":0.5}
    — vague and speculative; if not explicit, skip"""


def _have_curator() -> bool:
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False
    return ACTIVE_CURATOR_PATH.exists()


_LLM_SINGLETON = None
_LLM_LOCK = threading.Lock()


def _get_llm():
    """Thread-safe curator model loader (Qwen / Phi / Gemma — auto-detected).

    Path resolved from `CURATOR_MODEL_PATH` at config import time.
    Override with MEMOIRS_CURATOR_BACKEND={qwen,phi,gemma} or MEMOIRS_CURATOR_MODEL=path.
    """
    global _LLM_SINGLETON
    if _LLM_SINGLETON is not None:
        return _LLM_SINGLETON
    with _LLM_LOCK:
        if _LLM_SINGLETON is not None:
            return _LLM_SINGLETON
        from llama_cpp import Llama

        # Env var precedence: new MEMOIRS_CURATOR_* names win, fall back to
        # legacy MEMOIRS_GEMMA_* for backward compat. The defaults still come
        # from GEMMA_DEFAULT_* constants in config.py — those are sized by the
        # Gemma 2 2B model originally used and remain reasonable for Qwen/Phi.
        n_threads = int(os.environ.get(
            "MEMOIRS_CURATOR_THREADS",
            os.environ.get("MEMOIRS_GEMMA_THREADS", str(GEMMA_DEFAULT_THREADS)),
        ))
        n_ctx = int(os.environ.get(
            "MEMOIRS_CURATOR_CTX",
            os.environ.get("MEMOIRS_GEMMA_CTX", str(GEMMA_DEFAULT_CTX_TOKENS)),
        ))
        n_batch = int(os.environ.get(
            "MEMOIRS_CURATOR_BATCH",
            os.environ.get("MEMOIRS_GEMMA_BATCH", str(GEMMA_DEFAULT_BATCH)),
        ))
        n_gpu_layers = int(os.environ.get(
            "MEMOIRS_CURATOR_GPU_LAYERS",
            os.environ.get("MEMOIRS_GEMMA_GPU_LAYERS", "0"),
        ))

        log.info(
            "loading curator model %s from %s (threads=%d ctx=%d batch=%d gpu_layers=%d)",
            CURATOR_BACKEND, ACTIVE_CURATOR_PATH, n_threads, n_ctx, n_batch, n_gpu_layers,
        )
        _LLM_SINGLETON = Llama(
            model_path=str(ACTIVE_CURATOR_PATH),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        return _LLM_SINGLETON


_SYSTEM_INJECTION_PREFIXES = (
    "<ide_opened_file>", "<ide_selection>", "<system-reminder>",
    "<command-name>", "<command-message>", "<command-args>",
    "<bash-input>", "<bash-stdout>", "<bash-stderr>",
    "<local-command-stdout>",
)


def _is_user_meaningful(m: dict) -> bool:
    """Filter out system-injected user messages and tool noise before sending to Gemma."""
    if m.get("role") != "user":
        return False
    content = (m.get("content") or "").lstrip()
    if not content or len(content) < 12:
        return False
    if any(content.startswith(p) for p in _SYSTEM_INJECTION_PREFIXES):
        return False
    return True


def _build_curator_chat_prompt(user_turns: list[str]) -> str:
    """Format using Gemma 2's chat template (<start_of_turn>...<end_of_turn>)."""
    convo = "\n---\n".join(user_turns)
    if len(convo) > GEMMA_MAX_CONTEXT_CHARS:
        convo = convo[:GEMMA_MAX_CONTEXT_CHARS] + "\n...[truncated]"
    return _wrap_prompt(convo)


def _wrap_prompt(convo: str) -> str:
    """Wrap a (possibly chunk-sized) conversation slice in the active backend's chat template."""
    return _chat_user_turn(
        _CURATOR_SYSTEM_INSTRUCTIONS
        + "\n\nUser turns to analyze:\n"
        + convo
        + "\n\nReturn the JSON array now."
    )


# ----------------------------------------------------------------------
# Token budgeting (Bug 1 fix): keep prompt + max_tokens under model n_ctx.
# ----------------------------------------------------------------------

# Reserved budget (tokens). Header = static system instructions + chat template.
# Output = max_tokens we'll request from llm. Whatever remains in n_ctx is the
# room available for the actual conversation slice we ship to the model.
_HEADER_TOKEN_BUDGET = 200
_OUTPUT_TOKEN_BUDGET = 700  # ≥ GEMMA_MAX_OUTPUT_TOKENS (512) with slack
_CHUNK_OVERLAP_TOKENS = 200
# Defensive headroom (tokens) on top of measured overhead — protects against
# tokenizer drift when content tokens differ from header tokens (e.g. JSON
# vs prose splits BPE differently). Empirically a 64-tok margin is enough.
_BUDGET_HEADROOM_TOKENS = 64


def _count_tokens(llm, text: str) -> int:
    """Return token length of `text` using llama-cpp-python's tokenizer.

    NEVER use a chars/4 heuristic — Gemma's BPE diverges sharply on JSON,
    code blocks, and non-English text. We rely on the actual tokenizer so
    chunk math matches what the model will see.
    """
    data = text.encode("utf-8", errors="ignore")
    try:
        # llama_cpp.Llama.tokenize returns list[int]; add_bos=False because we
        # are measuring a slice that will be embedded inside a larger prompt.
        toks = llm.tokenize(data, add_bos=False, special=False)
    except TypeError:
        # Older llama-cpp-python signatures lack `special=`; fall back.
        try:
            toks = llm.tokenize(data, add_bos=False)
        except TypeError:
            toks = llm.tokenize(data)
    return len(toks)


def _model_ctx(llm) -> int:
    """Best-effort read of the model's context window (n_ctx)."""
    for attr in ("n_ctx",):
        v = getattr(llm, attr, None)
        if callable(v):
            try:
                return int(v())
            except Exception:
                continue
        if isinstance(v, int) and v > 0:
            return v
    # Conservative fallback to the configured default.
    return GEMMA_DEFAULT_CTX_TOKENS


def _wrapper_overhead_tokens(llm) -> int:
    """Token cost of wrapping an empty conversation slice in the active backend.

    `_wrap_prompt("")` produces the system instructions + chat template headers
    + assistant tag (everything except the actual user content). Counting that
    once with the real tokenizer gives us the exact prompt-side overhead we
    need to subtract from the per-chunk content budget.

    Why this matters (Bug #2): the previous static `_HEADER_TOKEN_BUDGET = 200`
    underestimated overhead by a factor of 2-4× for the larger system prompt,
    causing the wrapped prompt to exceed n_ctx after a chunk that "fit" the
    naive budget — the symptom was llama-cpp emitting
    `Requested tokens (4699) exceed context window of 4096`.
    """
    if llm is None:
        return _HEADER_TOKEN_BUDGET
    sample = _wrap_prompt("")
    try:
        return _count_tokens(llm, sample)
    except Exception:
        # Defensive: fall back to the static budget if tokenization fails.
        return _HEADER_TOKEN_BUDGET


def _content_token_budget(llm) -> int:
    """How many tokens of conversation we can afford per chunk.

    n_ctx − wrapper_overhead − GEMMA_MAX_OUTPUT_TOKENS − HEADROOM.
    Floors at 512 so degenerate configs don't crash with negative budgets.

    Bug #2 fix: we now measure wrapper overhead with the real tokenizer instead
    of relying on a hard-coded 200-token estimate that under-counted the system
    prompt by ~3-4×.
    """
    ctx = _model_ctx(llm)
    overhead = _wrapper_overhead_tokens(llm)
    budget = ctx - overhead - GEMMA_MAX_OUTPUT_TOKENS - _BUDGET_HEADROOM_TOKENS
    return max(512, budget)


def _chunk_user_turns(llm, user_turns: list[str], budget: int,
                       overlap: int = _CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Split user turns into prompt-sized chunks under `budget` tokens each.

    Strategy:
      1. Tokenize each turn once; cache token counts.
      2. Greedily pack turns into a chunk until the next turn would overflow.
      3. If a single turn alone exceeds the budget, slice IT by tokens
         (decode back to text) so we never emit an over-budget chunk.
      4. Between chunks, keep an overlap of the previous tail so candidates
         spanning the split aren't lost.

    Returns a list of joined-conversation strings (one per chunk), already
    formatted with the same `\\n---\\n` separator the single-shot path uses.
    """
    if not user_turns:
        return []

    sep = "\n---\n"
    sep_tok = _count_tokens(llm, sep)

    # Pre-tokenize each turn (as utf-8 bytes -> ints) so we can splice exactly.
    turn_tokens: list[list[int]] = []
    for t in user_turns:
        data = t.encode("utf-8", errors="ignore")
        try:
            toks = llm.tokenize(data, add_bos=False, special=False)
        except TypeError:
            try:
                toks = llm.tokenize(data, add_bos=False)
            except TypeError:
                toks = llm.tokenize(data)
        turn_tokens.append(list(toks))

    def _detok(tokens: list[int]) -> str:
        try:
            raw = llm.detokenize(tokens)
        except Exception:
            return ""
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)

    # Step 1: split any oversize SINGLE turn into sub-turns at token level.
    flat_turns: list[list[int]] = []
    for toks in turn_tokens:
        if len(toks) <= budget:
            flat_turns.append(toks)
            continue
        i = 0
        while i < len(toks):
            slice_end = min(len(toks), i + budget)
            flat_turns.append(toks[i:slice_end])
            if slice_end == len(toks):
                break
            i = slice_end - overlap if slice_end - overlap > i else slice_end

    # Step 2: pack into chunks, greedy.
    chunks: list[list[list[int]]] = []  # list of (list of turn token lists)
    current: list[list[int]] = []
    current_tokens = 0
    for toks in flat_turns:
        # +sep_tok if not first turn in current chunk.
        cost = len(toks) + (sep_tok if current else 0)
        if current and current_tokens + cost > budget:
            chunks.append(current)
            # Start new chunk with overlap from the tail of previous chunk.
            tail: list[int] = []
            tail_budget = overlap
            # Walk previous chunk turns from the end; keep tokens until we hit overlap.
            for prev in reversed(current):
                if len(prev) >= tail_budget:
                    tail = prev[-tail_budget:]
                    break
                tail = prev + tail
                tail_budget -= len(prev)
            current = [tail] if tail else []
            current_tokens = len(tail) if tail else 0
            cost = len(toks) + (sep_tok if current else 0)
        current.append(toks)
        current_tokens += cost

    if current:
        chunks.append(current)

    # Step 3: detokenize each chunk back to text joined by sep.
    out: list[str] = []
    for ch in chunks:
        parts = [_detok(t) for t in ch]
        out.append(sep.join(p for p in parts if p))
    return out


_CURATOR_SUMMARIZE_PROMPT = """You write a single-paragraph summary of a chat thread for long-term memory.

Output ONLY the summary text — no JSON, no markdown, no preamble.

Rules:
- 200-400 characters.
- Capture: what the user is working on, what was decided, what's blocking.
- Do NOT include: tool output, code dumps, individual messages, timestamps.
- Write in third person about the user.
- Skip if the thread is too short or has no durable substance — return exactly: SKIP

Conversation:"""


def curator_summarize(messages: list[dict]) -> str | None:
    """Run Gemma over a conversation and return a 200-400 char summary, or None.

    Returns None when the thread isn't worth summarizing (Gemma replies "SKIP").
    """
    llm = _get_llm()
    user_turns = [m["content"] for m in messages if _is_user_meaningful(m)]
    if not user_turns:
        return None
    convo = "\n---\n".join(user_turns)
    if len(convo) > GEMMA_MAX_CONTEXT_CHARS:
        convo = convo[:GEMMA_MAX_CONTEXT_CHARS] + "\n...[truncated]"
    prompt = _chat_user_turn(
        _CURATOR_SUMMARIZE_PROMPT
        + "\n\n"
        + convo
        + "\n\nReturn the summary now (or SKIP)."
    )
    out = llm.create_completion(
        prompt=prompt,
        max_tokens=200,
        temperature=0.3,
        stop=_chat_stops(),
    )
    text = out["choices"][0]["text"].strip()
    if not text or text.upper().startswith("SKIP"):
        return None
    # Hard cap at 500 chars to stop runaway models
    return text[:500]


_CURATOR_CONFLICT_PROMPT = """You decide whether two short memory statements about the same user are CONTRADICTORY.

Output ONLY one JSON object:
  {"contradictory": true|false, "winner": "A"|"B"|null, "reason": "short rationale"}

If contradictory, choose the WINNER based on:
- Which statement is more SPECIFIC and CONCRETE (wins).
- Otherwise, the one mentioning the LATER moment in time (wins).
- If you can't decide, set winner=null.

If NOT contradictory (e.g. they're about different aspects), set contradictory=false.

Statement A: "{a_content}"
Statement B: "{b_content}"

JSON:"""


_BARE_STRING_ACTIONS = {
    "contradictory": {"action": "MARK_CONFLICT", "reason": "contradictory"},
    "conflict":      {"action": "MARK_CONFLICT", "reason": "contradictory"},
    "merge":         {"action": "MERGE"},
    "keep":          {"action": "NOOP"},
    "none":          {"action": "NOOP"},
    "noop":          {"action": "NOOP"},
}


def _strip_fences(text: str) -> str:
    """Remove BOM, <think>…</think> reasoning preambles, markdown fences."""
    if not text:
        return ""
    s = text.lstrip("﻿").strip()
    # Reasoning models (Qwen3/3.5/3.6, DeepSeek-R1-Distill) emit
    # `<think>...</think>` before the JSON. /no_think suppresses it but if it
    # leaks through (or the system prompt was overridden) we drop it here.
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    elif s.startswith("<think>"):
        # Open think but never closed within budget; jump to first '{'.
        first_brace = s.find("{")
        if first_brace != -1:
            s = s[first_brace:]
    if s.startswith("```"):
        # Drop opening fence + optional language tag.
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    return s.strip()


def parse_conflict_response(text: str) -> dict | None:
    """Tolerant parser for `curator_resolve_conflict` model output (Bug 2).

    Handles:
      - bare JSON strings like `"contradictory"` → maps to action dicts
      - markdown-fenced JSON (```json … ```), BOM, surrounding whitespace
      - well-formed JSON objects (passes through, returning the object plus
        an inferred `action` if applicable)
      - empty / unparseable input → None (caller decides fallback)

    Returns dict with at least an `action` key, or None.
    """
    s = _strip_fences(text or "")
    if not s:
        return None

    # First attempt: full JSON parse.
    parsed = None
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, str):
        key = parsed.strip().strip('"').lower()
        mapped = _BARE_STRING_ACTIONS.get(key)
        if mapped is not None:
            return dict(mapped)
        log.warning("parse_conflict_response: unknown bare string %r", parsed)
        return None

    if isinstance(parsed, dict):
        out: dict = dict(parsed)
        # Infer canonical `action` field when missing but `contradictory` is set.
        if "action" not in out:
            if out.get("contradictory") is True:
                out["action"] = "MARK_CONFLICT"
            elif out.get("contradictory") is False:
                out["action"] = "NOOP"
        return out

    if isinstance(parsed, list):
        # Accept first scalar/object inside.
        for item in parsed:
            if isinstance(item, (str, dict)):
                return parse_conflict_response(json.dumps(item))
        return None

    # Fall back: try to salvage the first {...} block.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            if isinstance(obj, dict):
                return parse_conflict_response(json.dumps(obj))
        except json.JSONDecodeError:
            pass

    # Salvage truncated JSON (mid-string, missing closing brace).
    # Common pattern: '{"contradictory": true, "winner": "a", "reason": "Memory A talks about'
    if start != -1:
        candidate = s[start:]
        # Count unbalanced quotes; if odd, close the string artificially.
        # Then add missing closing braces.
        if candidate.count('"') % 2 == 1:
            candidate = candidate + '"'
        opens = candidate.count("{")
        closes = candidate.count("}")
        if opens > closes:
            candidate = candidate + ("}" * (opens - closes))
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                obj["_salvaged"] = True
                return parse_conflict_response(json.dumps(obj))
        except json.JSONDecodeError:
            pass

    log.warning("parse_conflict_response: unparseable head=%r", s[:80])
    return None


def curator_resolve_conflict(a_content: str, b_content: str) -> dict:
    """Ask Gemma whether two statements about the user are contradictory.

    Returns a dict with both legacy keys (`contradictory`, `winner`, `reason`)
    AND a canonical `action` ∈ {MARK_CONFLICT, MERGE, NOOP}. On any failure
    (no model, parse error), returns the NOOP fallback.

    Bug 2 fix: tolerates bare-string outputs ("contradictory", "merge", …)
    and ```json``` wrappers via `parse_conflict_response`.
    """
    fallback = {
        "contradictory": False,
        "winner": None,
        "reason": "fallback: gemma unavailable",
        "action": "NOOP",
    }
    if not _have_curator():
        return fallback
    try:
        llm = _get_llm()
        # NOTE: avoid str.format() — the prompt body contains literal `{}` in
        # the JSON example which str.format() would treat as fields and raise
        # KeyError. Substitute placeholders explicitly instead.
        body = (
            _CURATOR_CONFLICT_PROMPT
            .replace("{a_content}", a_content[:300])
            .replace("{b_content}", b_content[:300])
        )
        prompt = _chat_user_turn(body)
        out = llm.create_completion(prompt=prompt, max_tokens=200, temperature=0.1,
                                     stop=_chat_stops())
        text = (out["choices"][0]["text"] or "").strip()
        parsed = parse_conflict_response(text)
        if parsed is None:
            log.warning("curator_resolve_conflict: no parseable JSON (head=%r)", text[:80])
            return {**fallback, "reason": "gemma: no JSON in output"}

        action = str(parsed.get("action") or "").upper()
        winner = parsed.get("winner")
        if winner not in ("A", "B", None):
            winner = None

        if action == "MARK_CONFLICT":
            contradictory = True
        elif action in {"MERGE", "NOOP"}:
            contradictory = False
        else:
            contradictory = bool(parsed.get("contradictory", False))
            action = "MARK_CONFLICT" if contradictory else "NOOP"

        return {
            "contradictory": contradictory,
            "winner": winner,
            "reason": str(parsed.get("reason", ""))[:200],
            "action": action,
        }
    except Exception as e:
        log.warning("curator_resolve_conflict failed: %s", e)
        return {**fallback, "reason": f"gemma error: {e}"}


def _candidates_from_text(text: str) -> list[Candidate]:
    """Parse a single Gemma response text into Candidate objects."""
    if not text:
        return []
    parsed = validate_json_output(text)
    if not isinstance(parsed, list):
        log.debug("Gemma output not a list, returning empty: head=%s", text[:120])
        return []
    out: list[Candidate] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            out.append(
                Candidate(
                    type=str(item.get("type", "")).lower(),
                    content=str(item.get("content", "")).strip(),
                    importance=int(item.get("importance", 3)),
                    confidence=float(item.get("confidence", 0.5)),
                    entities=[str(e) for e in (item.get("entities") or [])],
                    source_message_ids=[str(i) for i in (item.get("source_message_ids") or [])],
                    extractor="gemma-2-2b",
                )
            )
        except (TypeError, ValueError):
            continue
    return out


def _candidate_dedup_key(c: Candidate) -> str:
    """Stable hash for dedup across chunks. Same type+normalized-content collapses."""
    import hashlib
    norm = " ".join((c.content or "").lower().split())
    h = hashlib.sha1(f"{c.type}|{norm}".encode("utf-8")).hexdigest()
    return h


def curator_extract(messages: list[dict]) -> list[Candidate]:
    """Extract memory candidates with token-budgeted chunking.

    Bug 1 fix: when the conversation + system prompt would exceed the model's
    context window, we split the user turns into chunks (with overlap) and
    run Gemma per-chunk, deduplicating candidates by (type, content) hash.
    """
    llm = _get_llm()
    user_turns = [m["content"] for m in messages if _is_user_meaningful(m)]
    if not user_turns:
        return []

    budget = _content_token_budget(llm)
    overhead = _wrapper_overhead_tokens(llm)
    chunks = _chunk_user_turns(llm, user_turns, budget=budget,
                                overlap=_CHUNK_OVERLAP_TOKENS)
    if not chunks:
        return []

    n_ctx = _model_ctx(llm)
    log.info(
        "curator_extract chunks=%d budget=%d overhead=%d n_ctx=%d",
        len(chunks), budget, overhead, n_ctx,
    )

    candidates: list[Candidate] = []
    seen: set[str] = set()
    chunks_processed = 0

    # Post-wrap re-validation: when the wrapped prompt + max_tokens still
    # exceeds n_ctx (because tokenizer drift between chunked content and the
    # full wrapped prompt makes the static overhead estimate slip), bisect the
    # chunk by character count and reprocess each half. We bound the recursion
    # so a pathological single oversized turn cannot livelock us.
    pending: list[tuple[str, int]] = [(c, 0) for c in chunks]  # (text, depth)
    _MAX_BISECT_DEPTH = 6
    chunk_idx = 0
    while pending:
        chunk_text, depth = pending.pop(0)
        chunk_idx += 1
        prompt = _wrap_prompt(chunk_text)
        try:
            wrapped_tokens = _count_tokens(llm, prompt)
        except Exception:
            wrapped_tokens = 0
        if (
            wrapped_tokens
            and wrapped_tokens + GEMMA_MAX_OUTPUT_TOKENS > n_ctx
            and depth < _MAX_BISECT_DEPTH
            and len(chunk_text) > 1
        ):
            mid = len(chunk_text) // 2
            # Bias the split toward the nearest separator so we don't slice
            # mid-token / mid-word; falls back to char midpoint otherwise.
            sep_idx = chunk_text.rfind("\n---\n", 0, mid)
            if sep_idx > 0:
                left, right = chunk_text[:sep_idx], chunk_text[sep_idx + len("\n---\n"):]
            else:
                left, right = chunk_text[:mid], chunk_text[mid:]
            log.warning(
                "curator_extract chunk %d wrapped=%d > n_ctx=%d; bisecting (depth=%d)",
                chunk_idx, wrapped_tokens, n_ctx, depth,
            )
            # Insert the halves at the front so they're processed next.
            pending.insert(0, (right, depth + 1))
            pending.insert(0, (left, depth + 1))
            continue
        try:
            out = llm.create_completion(
                prompt=prompt,
                max_tokens=GEMMA_MAX_OUTPUT_TOKENS,
                temperature=0.2,
                stop=["<end_of_turn>", "\n\n\n"],
            )
        except Exception as e:
            log.warning("gemma chunk %d failed: %s; skipping", chunk_idx, e)
            chunks_processed += 1
            continue
        text = (out["choices"][0]["text"] or "").strip()
        chunks_processed += 1
        for c in _candidates_from_text(text):
            # Fix #1 (GAP audit Fase 5C): drop noise BEFORE persisting. Code
            # snippets, file paths, tool output and stack traces are not
            # durable memorias — they are transient environment state.
            skip, reason = should_skip_extraction(c.content)
            if skip:
                log.warning(
                    "curator_extract: dropping noise candidate type=%s reason=%s head=%r",
                    c.type, reason, (c.content or "")[:80],
                )
                continue
            key = _candidate_dedup_key(c)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(c)

    log.info(
        "curator_extract chunks_processed=%d budget=%d overhead=%d candidates=%d",
        chunks_processed, budget, overhead, len(candidates),
    )
    return candidates


# ----------------------------------------------------------------------
# Public API: extract + persist
# ----------------------------------------------------------------------


def extract_memory_candidates(db: MemoirsDB, conversation_id: str) -> list[Candidate]:
    """Run extraction over a conversation's active messages, validate, persist.

    Cascade: Gemma (if model installed) → spaCy (if installed) → noop.
    """
    from . import extract_spacy

    rows = db.conn.execute(
        """
        SELECT id, role, content, ordinal
        FROM messages
        WHERE conversation_id = ? AND is_active = 1
        ORDER BY ordinal
        """,
        (conversation_id,),
    ).fetchall()
    messages = [dict(r) for r in rows]
    if not messages:
        return []

    extractor_used = "none"
    extractors_tried: list[str] = []
    candidates: list[Candidate] = []
    if _have_curator():
        extractors_tried.append("gemma")
        log.info("extract conv=%s msgs=%d → invoking Gemma…", conversation_id[:24], len(messages))
        import time as _t
        _t0 = _t.time()
        try:
            candidates = curator_extract(messages)
            if candidates:
                extractor_used = "gemma"
            log.info(
                "extract conv=%s Gemma done in %.1fs (%d raw candidates before validation)",
                conversation_id[:24], _t.time() - _t0, len(candidates),
            )
        except Exception as e:
            log.warning("gemma extractor failed in %.1fs (%s); falling through", _t.time() - _t0, e)

    if not candidates and extract_spacy.is_available():
        extractors_tried.append("spacy")
        try:
            candidates = extract_spacy.extract(messages)
            if candidates:
                extractor_used = "spacy"
        except Exception as e:
            log.warning("spacy extractor failed (%s); falling through", e)

    if not candidates:
        if extractors_tried:
            log.info(
                "extract conv=%s msgs=%d candidates=0 (tried=%s — nothing durable to extract)",
                conversation_id, len(messages), "+".join(extractors_tried),
            )
        else:
            log.warning(
                "extract conv=%s msgs=%d candidates=0 — NO extractor available. Install: pip install -e '.[extract]' or '.[gemma]'",
                conversation_id, len(messages),
            )
        return []

    # Validate + persist
    persisted: list[Candidate] = []
    rejected_counts: dict[str, int] = {}
    now = utc_now()
    with db.conn:
        for c in candidates:
            if not c.content:
                rejected_counts["empty"] = rejected_counts.get("empty", 0) + 1
                continue
            if not validate_allowed_memory_type(c):
                rejected_counts["bad_type"] = rejected_counts.get("bad_type", 0) + 1
                continue
            if not validate_confidence(c):
                rejected_counts["bad_confidence"] = rejected_counts.get("bad_confidence", 0) + 1
                continue
            if detect_sensitive_content(c):
                rejected_counts["sensitive"] = rejected_counts.get("sensitive", 0) + 1
                continue
            cid = stable_id("cand", conversation_id, c.type, c.content)
            db.conn.execute(
                """
                INSERT OR IGNORE INTO memory_candidates (
                    id, conversation_id, source_message_ids, type, content,
                    importance, confidence, entities, status, extractor,
                    raw_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                """,
                (
                    cid,
                    conversation_id,
                    json.dumps(c.source_message_ids),
                    c.type,
                    c.content,
                    c.importance,
                    c.confidence,
                    json.dumps(c.entities),
                    c.extractor,
                    json.dumps(c.as_dict(), ensure_ascii=False),
                    now,
                    now,
                ),
            )
            persisted.append(c)
    by_type: dict[str, int] = {}
    for c in persisted:
        by_type[c.type] = by_type.get(c.type, 0) + 1
    log.info(
        "extract conv=%s msgs=%d candidates=%d persisted=%d types=%s rejected=%s extractor=%s",
        conversation_id,
        len(messages),
        len(candidates),
        len(persisted),
        by_type or "{}",
        rejected_counts or "{}",
        extractor_used,
    )
    # Log a sample of what was actually extracted so the operator can sanity-check quality.
    for c in persisted[:5]:
        snippet = c.content[:120].replace("\n", " ")
        log.info("  → [%s] imp=%d conf=%.2f  %s", c.type, c.importance, c.confidence, snippet)
    if len(persisted) > 5:
        log.info("  → … +%d more candidates", len(persisted) - 5)
    return persisted


def summarize_conversation(db: MemoirsDB, conversation_id: str) -> dict:
    """Run Gemma summary for a conversation and persist as a fact memoria.

    Returns dict with `memory_id`, `summary`, `skipped` flag. Skipped is True
    when the thread had nothing worth summarizing (Gemma returned SKIP) or
    when no Gemma model is available.
    """
    if not _have_curator():
        return {"memory_id": None, "summary": None, "skipped": True,
                "reason": "Gemma model not available"}

    rows = db.conn.execute(
        "SELECT id, role, content, ordinal FROM messages "
        "WHERE conversation_id = ? AND is_active = 1 ORDER BY ordinal",
        (conversation_id,),
    ).fetchall()
    if not rows:
        return {"memory_id": None, "summary": None, "skipped": True,
                "reason": "no messages"}

    messages = [dict(r) for r in rows]
    log.info("summarize conv=%s msgs=%d → invoking Gemma…", conversation_id[:24], len(messages))
    summary = curator_summarize(messages)
    if not summary:
        log.info("summarize conv=%s → SKIP (no durable content)", conversation_id[:24])
        return {"memory_id": None, "summary": None, "skipped": True,
                "reason": "model returned SKIP"}

    # Persist as a fact memoria with metadata flagging it as a summary
    from .memory_engine import apply_decision, decide_memory_action
    cand = Candidate(
        type="fact",
        content=summary,
        importance=4,
        confidence=0.85,
        source_message_ids=[r["id"] for r in rows],
        extractor="gemma-summary",
    )
    decision = decide_memory_action(db, cand)
    result = apply_decision(db, cand, decision)
    memory_id = result.get("memory_id")
    if memory_id:
        # Tag the memoria as a summary
        db.conn.execute(
            "UPDATE memories SET metadata_json = json_set(COALESCE(metadata_json,'{}'),"
            "  '$.is_summary', json('true'),"
            "  '$.summarized_conversation', ?) WHERE id = ?",
            (conversation_id, memory_id),
        )
        db.conn.commit()
    log.info("summarize conv=%s → memory=%s len=%d",
             conversation_id[:24], (memory_id or '-')[:16], len(summary))
    return {"memory_id": memory_id, "summary": summary, "skipped": False}


def extract_pending(
    db: MemoirsDB,
    *,
    limit: int = 50,
    min_messages: int = 3,
    reprocess_with_gemma: bool = False,
) -> dict:
    """Run extraction over conversations.

    Default: picks conversations that have NO candidates yet (≥`min_messages` msgs).
    With `reprocess_with_gemma=True`: picks conversations whose candidates were
    extracted by spaCy (no Gemma candidates yet) and reprocesses them.
    """
    from . import extract_spacy

    if reprocess_with_gemma:
        rows = db.conn.execute(
            """
            SELECT c.id
            FROM conversations c
            WHERE c.message_count >= ?
              AND c.id IN (
                SELECT DISTINCT conversation_id FROM memory_candidates
                WHERE conversation_id IS NOT NULL
              )
              AND c.id NOT IN (
                SELECT DISTINCT conversation_id FROM memory_candidates
                WHERE extractor = 'gemma-2-2b' AND conversation_id IS NOT NULL
              )
            ORDER BY c.message_count DESC, c.updated_at DESC
            LIMIT ?
            """,
            (min_messages, limit),
        ).fetchall()
        log.info("extract_pending: reprocess-with-gemma mode, %d conversations selected", len(rows))
    else:
        rows = db.conn.execute(
            """
            SELECT c.id
            FROM conversations c
            LEFT JOIN memory_candidates mc ON mc.conversation_id = c.id
            WHERE mc.id IS NULL
              AND c.message_count >= ?
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (min_messages, limit),
        ).fetchall()
    total = 0
    for row in rows:
        candidates = extract_memory_candidates(db, row["id"])
        total += len(candidates)
    return {
        "conversations": len(rows),
        "candidates": total,
        "gemma_available": _have_curator(),
        "spacy_available": extract_spacy.is_available(),
    }


# ----------------------------------------------------------------------
# P1-11 — Gemma-driven consolidation, contradiction detection, summary
# ----------------------------------------------------------------------
#
# These three helpers replace the heuristic curator in `decide_memory_action`
# (and friends) with prompt-driven inference. Each one:
#
#   - lazy-loads the LLM via `_get_llm()` (reuses the global singleton).
#   - tolerates parse failures via `parse_conflict_response` /
#     `parse_consolidation_response` (returns a marker dict rather than
#     raising, so the caller can fall back to the heuristic).
#   - respects the model's context window by truncating neighbors / memory
#     dumps that would overflow `_content_token_budget(llm)`.
#
# We intentionally do NOT touch `curator_extract`, `curator_resolve_conflict`,
# `_wrap_prompt`, `_count_tokens`, or `_chunk_user_turns` — other agents may
# be editing those concurrently. We reuse them through the public API only.


_VALID_CONSOLIDATE_ACTIONS = {"ADD", "UPDATE", "MERGE", "IGNORE", "EXPIRE"}


_CURATOR_CONSOLIDATE_PROMPT = """You are deciding what to do with a new memory candidate.

NEW CANDIDATE: {ctype} | importance={imp} | "{ccontent}"
EXISTING NEIGHBORS (most similar first):
{neighbors_block}

Decide ONE action and respond as JSON only:
  {"action":"ADD|UPDATE|MERGE|IGNORE|EXPIRE", "target_id":"<id_or_null>", "reason":"<short>"}

ADD = candidate is new and useful.
UPDATE = candidate refines/replaces neighbor[i] (set target_id).
MERGE = candidate combines with neighbor[i] (set target_id).
IGNORE = candidate is redundant or low-value.
EXPIRE = candidate makes neighbor[i] obsolete (set target_id).
"""


_CURATOR_CONTRADICTION_PROMPT = """Decide if two memories about the same user are CONTRADICTORY.

Focus on negations / semantic incompatibilities (e.g. "uses Python" vs "stopped using Python").

Output ONE compact JSON line, no markdown, no prose, reason ≤ 8 words:
{"contradictory": true|false, "winner": "a"|"b"|null, "reason": "<≤8 words>"}

If contradictory, winner = the more specific OR more recent statement, else null.
If different aspects (not contradictory): contradictory=false, winner=null.

Memory A: "{a_content}"
Memory B: "{b_content}"

JSON:"""


_CURATOR_PROJECT_SUMMARY_PROMPT = """Write a 2-3 sentence summary of this project's recent activity. The summary MUST mention specific entities (file names, commands, decisions, tools, libraries). Avoid generic phrases ("this conversation discusses", "as mentioned", "in summary", "as discussed").

Project name: {project}

Memories (most important first):
{memories_block}

Summary (2-3 sentences, must reference at least one specific entity from the memories above):"""


_CURATOR_PROJECT_SUMMARY_RETRY_PROMPT = """Your previous summary was rejected because it was empty, too generic, or missed the key entities. Write a NEW 2-3 sentence summary that:
1. Names at least one SPECIFIC thing from the memories below (a tool name, file, library, decision, or command).
2. Does NOT contain the phrases: "this conversation discusses", "as mentioned", "as discussed", "in summary".
3. Is between 60 and 500 characters.

Project name: {project}

Memories (most important first):
{memories_block}

Summary (must mention a specific entity by name):"""


# Phrases that indicate generic, content-free output. Lower-cased substring
# match — if the summary contains ANY of these we reject it and retry once.
_SUMMARY_BAD_PHRASES = (
    "this conversation discusses",
    "this conversation covers",
    "as mentioned",
    "as discussed",
    "in summary",
    "the conversation discusses",
    "the conversation covers",
    "this discussion",
)


def _validate_summary(
    text: str,
    entities: Iterable[str] | None = None,
    *,
    min_chars: int = 60,
    max_chars: int = 500,
) -> tuple[bool, str]:
    """Validate a project summary against length / vapidity / entity coverage.

    Returns ``(ok, reason)``. ``reason`` is a short tag suitable for logging
    when ``ok=False``.

    Rules:
      - length between ``min_chars`` and ``max_chars`` (inclusive).
      - must NOT contain any of ``_SUMMARY_BAD_PHRASES`` (case-insensitive
        substring match) — those are vapid templates.
      - if ``entities`` is provided and non-empty, the summary MUST contain
        at least one entity as a case-insensitive substring.

    Used by :func:`curator_summarize_project` to reject empty / templated
    output and trigger a retry with a more explicit prompt; if the retry
    also fails the caller falls back to a heuristic summary.
    """
    if not text:
        return False, "empty"
    s = text.strip()
    n = len(s)
    if n < min_chars:
        return False, "too_short"
    if n > max_chars:
        return False, "too_long"
    lower = s.lower()
    for phrase in _SUMMARY_BAD_PHRASES:
        if phrase in lower:
            return False, f"vapid_phrase:{phrase}"
    if entities:
        ents = [e for e in entities if e]
        if ents:
            hit = any(str(e).lower() in lower for e in ents)
            if not hit:
                return False, "no_entity_coverage"
    return True, "ok"


def _collect_summary_entities(memories: list[dict]) -> list[str]:
    """Pull candidate entity strings from a list of memories.

    Looks at ``entities`` first (already a normalized list), falls back to
    plucking words from ``content`` if no entities are populated. Used by
    :func:`curator_summarize_project` to gate the validator on real names that
    appear in the input — avoids false positives on memories that genuinely
    have no named entities.
    """
    out: list[str] = []
    seen: set[str] = set()
    for m in memories or []:
        if not isinstance(m, dict):
            continue
        for e in (m.get("entities") or []):
            es = str(e).strip()
            if not es:
                continue
            key = es.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(es)
    return out


_BARE_STRING_CONSOLIDATE_ACTIONS = {
    "add":     {"action": "ADD"},
    "update":  {"action": "UPDATE"},
    "merge":   {"action": "MERGE"},
    "ignore":  {"action": "IGNORE"},
    "expire":  {"action": "EXPIRE"},
    "noop":    {"action": "IGNORE"},
    "keep":    {"action": "IGNORE"},
    "skip":    {"action": "IGNORE"},
    "none":    {"action": "IGNORE"},
}


def parse_consolidation_response(text: str) -> dict | None:
    """Tolerant parser for `curator_consolidate` model output.

    Mirrors :func:`parse_conflict_response` but maps to consolidation actions
    {ADD, UPDATE, MERGE, IGNORE, EXPIRE}. Handles:

      - bare JSON strings like ``"merge"`` -> {"action": "MERGE"}
      - markdown-fenced JSON, BOM, surrounding whitespace
      - well-formed JSON objects with action/target_id/reason
      - empty / unparseable input -> None (caller decides fallback)

    Returns dict with at least an ``action`` key, or None.
    """
    s = _strip_fences(text or "")
    if not s:
        return None

    parsed = None
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, str):
        key = parsed.strip().strip('"').lower()
        mapped = _BARE_STRING_CONSOLIDATE_ACTIONS.get(key)
        if mapped is not None:
            return dict(mapped)
        log.warning("parse_consolidation_response: unknown bare string %r", parsed)
        return None

    if isinstance(parsed, dict):
        out: dict = dict(parsed)
        action = str(out.get("action") or "").strip().upper()
        if action and action in _VALID_CONSOLIDATE_ACTIONS:
            out["action"] = action
            return out
        # Action missing or invalid: try to recover from "decision" / "verb" keys.
        for alt_key in ("decision", "verb", "result"):
            alt = out.get(alt_key)
            if isinstance(alt, str):
                cand = alt.strip().upper()
                if cand in _VALID_CONSOLIDATE_ACTIONS:
                    out["action"] = cand
                    return out
        log.warning("parse_consolidation_response: action missing/invalid in %r",
                    {k: out[k] for k in list(out)[:4]})
        return None

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, (str, dict)):
                return parse_consolidation_response(json.dumps(item))
        return None

    # Salvage first {...} block.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            if isinstance(obj, (dict, str)):
                return parse_consolidation_response(json.dumps(obj))
        except json.JSONDecodeError:
            pass

    log.warning("parse_consolidation_response: unparseable head=%r", s[:80])
    return None


def _short_id(s: str | None) -> str:
    """First 12 chars of an id-like string, with `?` fallback."""
    if not s:
        return "?"
    return str(s)[:12]


def _build_neighbors_block(llm, neighbors: list[dict], header_budget_tokens: int) -> str:
    """Format neighbors as a numbered block, dropping the tail when it would
    push the prompt past the model's content budget.

    `header_budget_tokens` is the rough token count of the rest of the prompt
    (system + candidate). We compare against `_content_token_budget(llm)` and
    truncate if needed so the call to llama_cpp doesn't exceed n_ctx.
    """
    if not neighbors:
        return "  (none)"

    budget = _content_token_budget(llm) if llm is not None else 2048
    available = max(256, budget - header_budget_tokens)

    lines: list[str] = []
    used = 0
    for idx, n in enumerate(neighbors, start=1):
        nid = _short_id(n.get("id"))
        ntype = str(n.get("type") or "?")
        ncontent = str(n.get("content") or "").replace("\n", " ").strip()
        # Trim individual content to keep a single neighbor from blowing the budget.
        ncontent = ncontent[:240]
        sim = n.get("similarity")
        sim_str = ""
        if isinstance(sim, (int, float)):
            sim_str = f" | similarity={sim:.3f}"
        line = f'  {idx}. [{nid}] {ntype} | "{ncontent}"{sim_str}'

        if llm is not None:
            cost = _count_tokens(llm, line + "\n")
        else:
            # Char-based heuristic for the no-LLM path.
            cost = max(1, len(line) // 4)

        if used + cost > available and lines:
            # Stop here — caller knows neighbors were truncated by the index.
            lines.append(f"  … (+{len(neighbors) - idx + 1} more truncated)")
            break
        lines.append(line)
        used += cost

    return "\n".join(lines)


def curator_consolidate(
    candidate: dict,
    neighbors: list[dict],
    *,
    llm=None,
) -> dict:
    """Ask Gemma whether to ADD / UPDATE / MERGE / IGNORE / EXPIRE a candidate.

    Parameters
    ----------
    candidate:
        Dict with at least ``type``, ``content``, optionally ``importance``.
    neighbors:
        List of similar memory dicts (id, type, content, similarity). Most
        similar first. Will be truncated if it overflows the model's context.
    llm:
        Optional pre-loaded LLM. If None, lazy-load via ``_get_llm()``.

    Returns
    -------
    dict
        One of:

        - ``{"action": "ADD"|"UPDATE"|"MERGE"|"IGNORE"|"EXPIRE",
              "target_id": str|None, "reason": str, "source": "gemma"}``
          on success.
        - ``{"action": None, "source": "gemma_unavailable", ...}``
          when the model is missing or load fails.
        - ``{"action": None, "source": "gemma_parse_error", ...}``
          when output couldn't be parsed.

    The caller (typically `memory_engine.decide_memory_action`) treats any
    ``action: None`` response as "fall back to the heuristic curator".
    """
    base_fail = {"action": None, "target_id": None, "reason": ""}

    if llm is None:
        if not _have_curator():
            return {**base_fail, "source": "gemma_unavailable",
                    "reason": "Gemma model not installed"}
        try:
            llm = _get_llm()
        except Exception as e:  # pragma: no cover — defensive
            log.warning("curator_consolidate: _get_llm failed: %s", e)
            return {**base_fail, "source": "gemma_unavailable",
                    "reason": f"gemma load failed: {e}"}

    if llm is None:
        return {**base_fail, "source": "gemma_unavailable",
                "reason": "no llm handle"}

    ctype = str(candidate.get("type") or "?")
    ccontent = str(candidate.get("content") or "").replace("\n", " ").strip()[:300]
    imp = candidate.get("importance", 3)

    # Estimate the header cost (system + candidate frame) so we can size the
    # neighbors block to fit. We don't need to be exact — leave slack.
    header_preview = (
        _CURATOR_CONSOLIDATE_PROMPT
        .replace("{ctype}", ctype)
        .replace("{imp}", str(imp))
        .replace("{ccontent}", ccontent)
        .replace("{neighbors_block}", "")
    )
    try:
        header_tokens = _count_tokens(llm, header_preview)
    except Exception:
        header_tokens = max(1, len(header_preview) // 4)

    neighbors_block = _build_neighbors_block(llm, neighbors, header_tokens)

    body = (
        _CURATOR_CONSOLIDATE_PROMPT
        .replace("{ctype}", ctype)
        .replace("{imp}", str(imp))
        .replace("{ccontent}", ccontent)
        .replace("{neighbors_block}", neighbors_block)
    )
    prompt = _chat_user_turn(body)

    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=120,
            temperature=0.1,
            stop=_chat_stops(),
        )
    except Exception as e:
        log.warning("curator_consolidate: create_completion failed: %s", e)
        return {**base_fail, "source": "gemma_unavailable",
                "reason": f"gemma error: {e}"}

    text = (out.get("choices", [{}])[0].get("text") or "").strip()
    # The stop-sequence may strip the closing brace; restore it for the parser.
    if text and text.lstrip().startswith("{") and not text.rstrip().endswith("}"):
        text = text + "}"

    parsed = parse_consolidation_response(text)
    if parsed is None:
        log.warning("curator_consolidate: unparseable output (head=%r)", text[:120])
        return {**base_fail, "source": "gemma_parse_error",
                "reason": "could not parse gemma output",
                "raw": text[:200]}

    action = str(parsed.get("action") or "").upper()
    if action not in _VALID_CONSOLIDATE_ACTIONS:
        log.warning("curator_consolidate: invalid action %r", action)
        return {**base_fail, "source": "gemma_parse_error",
                "reason": f"invalid action {action!r}",
                "raw": text[:200]}

    target_id = parsed.get("target_id")
    if target_id in (None, "", "null", "None"):
        target_id = None
    else:
        target_id = str(target_id)

    # If the model picked a non-ADD/IGNORE action without a target_id, but we
    # have neighbors, default to the closest one. Otherwise downgrade to ADD.
    if action in {"UPDATE", "MERGE", "EXPIRE"} and target_id is None:
        if neighbors:
            target_id = str(neighbors[0].get("id") or "") or None
        else:
            log.info("curator_consolidate: %s without target & no neighbors → ADD", action)
            action = "ADD"

    reason = str(parsed.get("reason", "") or "")[:200]

    return {
        "action": action,
        "target_id": target_id,
        "reason": reason or f"gemma decided {action}",
        "source": "gemma",
    }


def curator_detect_contradiction(
    memory_a: dict,
    memory_b: dict,
    *,
    llm=None,
) -> dict:
    """Ask Gemma whether two memories contradict each other.

    Returns
    -------
    dict
        ``{"contradictory": bool, "winner": "a"|"b"|None, "reason": str,
           "source": "gemma" | "gemma_unavailable" | "gemma_parse_error"}``.

    On any failure the contradictory flag is False (safe default — never
    auto-archive a memoria because the model timed out).
    """
    fallback = {
        "contradictory": False,
        "winner": None,
        "reason": "",
        "source": "gemma_unavailable",
    }

    if llm is None:
        if not _have_curator():
            return {**fallback, "reason": "Gemma not installed"}
        try:
            llm = _get_llm()
        except Exception as e:  # pragma: no cover — defensive
            return {**fallback, "reason": f"gemma load failed: {e}"}

    if llm is None:
        return fallback

    def _content_of(m):
        if isinstance(m, str):
            return m
        if isinstance(m, dict):
            return str(m.get("content") or "")
        return str(m or "")

    a_content = _content_of(memory_a).replace("\n", " ").strip()[:300]
    b_content = _content_of(memory_b).replace("\n", " ").strip()[:300]
    if not a_content or not b_content:
        return {**fallback, "source": "gemma",
                "reason": "empty content; nothing to compare"}

    body = (
        _CURATOR_CONTRADICTION_PROMPT
        .replace("{a_content}", a_content)
        .replace("{b_content}", b_content)
    )
    prompt = _chat_user_turn(body)

    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,
            stop=_chat_stops(),
        )
    except Exception as e:
        log.warning("curator_detect_contradiction: create_completion failed: %s", e)
        return {**fallback, "reason": f"gemma error: {e}"}

    text = (out.get("choices", [{}])[0].get("text") or "").strip()
    if text and text.lstrip().startswith("{") and not text.rstrip().endswith("}"):
        text = text + "}"

    parsed = parse_conflict_response(text)
    if parsed is None:
        log.warning("curator_detect_contradiction: unparseable head=%r", text[:120])
        return {**fallback, "source": "gemma_parse_error",
                "reason": "could not parse gemma output"}

    contradictory = parsed.get("contradictory")
    if not isinstance(contradictory, bool):
        action = str(parsed.get("action") or "").upper()
        contradictory = (action == "MARK_CONFLICT")

    winner_raw = parsed.get("winner")
    if isinstance(winner_raw, str):
        w = winner_raw.strip().lower().strip('"')
        winner = w if w in ("a", "b") else None
    else:
        winner = None

    return {
        "contradictory": bool(contradictory),
        "winner": winner,
        "reason": str(parsed.get("reason", "") or "")[:200],
        "source": "gemma",
    }


def curator_summarize_project(
    project_name: str,
    memories: list[dict],
    *,
    llm=None,
    max_chars: int = 500,
) -> str | None:
    """Ask Gemma for a short summary of a project's curated memories.

    Returns plain text (no JSON, no fences). Output is hard-trimmed to
    ``max_chars`` so a runaway model can't bloat the caller's payload.

    Bug #3 fix: validates the LLM output via :func:`_validate_summary` —
    rejects vapid templates like "this conversation discusses…" and
    summaries that don't reference any of the input entities. On the first
    rejection we retry once with a stricter retry prompt; if THAT also
    fails we return ``None`` so the caller can fall back to a heuristic.

    Returns
    -------
    str
        The validated summary text on success.
    "" (empty string)
        When ``memories`` is empty (nothing to summarize) — preserves the
        existing API contract used by callers / tests.
    None
        When the model is unavailable, the call fails, OR the output
        cannot pass validation after one retry. Callers should treat this
        as "no LLM-generated summary available" and fall back.
    """
    if not memories:
        return ""

    if llm is None:
        if not _have_curator():
            log.info("curator_summarize_project: gemma unavailable; returning None")
            return None
        try:
            llm = _get_llm()
        except Exception as e:  # pragma: no cover — defensive
            log.warning("curator_summarize_project: _get_llm failed: %s", e)
            return None

    if llm is None:
        return None

    entities = _collect_summary_entities(memories)

    def _build(prompt_template: str) -> str:
        header_preview = (
            prompt_template
            .replace("{project}", project_name or "?")
            .replace("{memories_block}", "")
        )
        try:
            header_tokens = _count_tokens(llm, header_preview)
        except Exception:
            header_tokens = max(1, len(header_preview) // 4)
        block = _build_neighbors_block(llm, memories, header_tokens)
        body = (
            prompt_template
            .replace("{project}", project_name or "?")
            .replace("{memories_block}", block)
        )
        return _chat_user_turn(body)

    def _trim(text: str) -> str:
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        sp = cut.rfind(" ")
        if sp > max_chars // 2:
            cut = cut[:sp]
        return cut.rstrip() + "…"

    # Sentinel returned by `_attempt` when the model explicitly asked to skip
    # (vs. just emitting empty / failed output) — propagated to the caller as
    # the empty-string contract for "nothing durable to summarize".
    _SKIP = object()

    def _attempt(prompt_template: str, *, temperature: float):
        prompt = _build(prompt_template)
        try:
            out = llm.create_completion(
                prompt=prompt,
                max_tokens=256,
                temperature=temperature,
                stop=_chat_stops(),
            )
        except Exception as e:
            log.warning("curator_summarize_project: create_completion failed: %s", e)
            return None
        text = (out.get("choices", [{}])[0].get("text") or "").strip()
        if not text:
            return None
        if text.upper().startswith("SKIP"):
            return _SKIP
        return text

    # Allow a small slack on top of `max_chars` so the ellipsis we append
    # during trimming doesn't get rejected by the length validator.
    _validator_max = max_chars + 10

    # First attempt with the standard prompt.
    text = _attempt(_CURATOR_PROJECT_SUMMARY_PROMPT, temperature=0.3)
    if text is _SKIP:
        # Model explicitly opted out: nothing to summarize. Honour the
        # original API contract and return empty string (NOT None) so the
        # caller doesn't treat this as a fall-back-required failure.
        return ""
    if isinstance(text, str):
        trimmed = _trim(text)
        ok, reason = _validate_summary(trimmed, entities, max_chars=_validator_max)
        if ok:
            return trimmed
        log.info(
            "curator_summarize_project: first attempt rejected (%s); retrying with strict prompt",
            reason,
        )
    else:
        log.info("curator_summarize_project: first attempt empty; retrying with strict prompt")

    # Retry once with the explicit retry prompt and a slightly cooler temperature.
    text = _attempt(_CURATOR_PROJECT_SUMMARY_RETRY_PROMPT, temperature=0.2)
    if text is _SKIP:
        return ""
    if not isinstance(text, str):
        log.warning("curator_summarize_project: retry produced no output; returning None")
        return None
    trimmed = _trim(text)
    ok, reason = _validate_summary(trimmed, entities, max_chars=_validator_max)
    if not ok:
        log.warning(
            "curator_summarize_project: retry also rejected (%s); returning None for fallback",
            reason,
        )
        return None
    return trimmed


# ----------------------------------------------------------------------
# Layer 3 — entity & relationship extraction (LLM-based replacement
# for the regex/CamelCase heuristics historically lived in graph.py).
# ----------------------------------------------------------------------


_ENTITY_TYPES = {"person", "tool", "project", "concept", "org", "location"}


_CURATOR_ENTITIES_PROMPT = """Extract the named entities from the text below.

Output ONLY a JSON array (no markdown fences, no prose). Each item:
  {"name": "<entity surface form, as written>", "type": "<one of: person|tool|project|concept|org|location>"}

Rules:
- Include proper nouns, product/library/tool names, project names, organizations, locations, and named technical concepts.
- Skip generic words (TODO, NOTE, REMINDER), pronouns, articles, code keywords, file paths, command output.
- De-duplicate: emit each unique name at most once.
- Cap output at 20 entities. Return [] if no entities are present.
- Preserve the original casing of the name (e.g. "FastAPI", "memoirs", "Postgres").
- type must be exactly one of: person, tool, project, concept, org, location.

Text:
{TEXT}

JSON array:"""


_CURATOR_RELATIONSHIPS_PROMPT = """Given the memory text and the list of entities already extracted from it, emit explicit subject-predicate-object triples that the text states or directly implies.

Output ONLY a JSON array (no markdown fences). Each item:
  {"subject": "<entity name from the list>", "predicate": "<short verb-phrase, snake_case>", "object": "<entity name from the list>", "confidence": 0.0-1.0}

Rules:
- subject and object MUST be exact strings from the entities list (case-insensitive match against the list is OK; use the list's casing on output).
- predicate is a short snake_case verb phrase such as: works_at, uses, depends_on, prefers, switched_from, located_in, replaces, integrates_with, decided_to_use, contributes_to, owns. Pick the predicate that best fits the text — do not make one up if the relationship is vague.
- Only emit relationships explicitly stated or directly implied by the text. Do NOT emit "co_occurs_with" or generic "related_to" — if there's no clear relation, omit the pair.
- confidence: 0.9+ for explicit statements, 0.6-0.85 for implied, below 0.6 → skip.
- Cap output at 12 triples. Return [] if no clear relationships exist.

Entities (you MUST use these exact names): {ENTITIES}

Memory text:
{TEXT}

JSON array:"""


def parse_entities_response(text: str) -> list[dict] | None:
    """Tolerant parser for ``curator_extract_entities`` model output.

    Returns a list of {"name", "type"} dicts, or ``None`` if nothing
    salvageable was produced (caller decides fallback).
    """
    s = _strip_fences(text or "")
    if not s:
        return None
    try:
        parsed = validate_json_output(s)
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None
    out: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        etype = str(item.get("type") or "").strip().lower()
        if not name:
            continue
        if etype not in _ENTITY_TYPES:
            etype = "concept"  # default bucket for unknown / missing
        out.append({"name": name, "type": etype})
    return out


def parse_relationships_response(text: str) -> list[dict] | None:
    """Tolerant parser for ``curator_extract_relationships`` model output.

    Returns a list of {"subject", "predicate", "object", "confidence"} dicts,
    or ``None`` if nothing salvageable was produced.
    """
    s = _strip_fences(text or "")
    if not s:
        return None
    try:
        parsed = validate_json_output(s)
    except ValueError:
        return None
    if not isinstance(parsed, list):
        return None
    out: list[dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        subj = str(item.get("subject") or "").strip()
        pred = str(item.get("predicate") or "").strip()
        obj = str(item.get("object") or "").strip()
        try:
            conf = float(item.get("confidence", 0.7))
        except (TypeError, ValueError):
            conf = 0.7
        if not (subj and pred and obj):
            continue
        if subj.lower() == obj.lower():
            continue
        # Normalize predicate: lowercase, replace whitespace with underscores.
        pred_norm = "_".join(pred.lower().split())
        if not pred_norm:
            continue
        out.append({
            "subject": subj,
            "predicate": pred_norm,
            "object": obj,
            "confidence": max(0.0, min(1.0, conf)),
        })
    return out


def curator_extract_entities(text: str, *, llm=None) -> list[dict] | None:
    """Ask Qwen/Phi/Gemma to extract named entities from a piece of text.

    Returns a list of ``{"name", "type"}`` dicts on success, or ``None`` on
    any failure (model unavailable, parse error, exception). Callers should
    treat ``None`` as "fall back to heuristic".

    Output is capped at 20 entities. The text is truncated to ~3500 chars
    before being sent to the model so the prompt fits comfortably in any
    of our 4k-context curators.
    """
    if not text or len(text.strip()) < 8:
        return []

    if llm is None:
        if not _have_curator():
            return None
        try:
            llm = _get_llm()
        except Exception as e:  # pragma: no cover — defensive
            log.warning("curator_extract_entities: _get_llm failed: %s", e)
            return None
    if llm is None:
        return None

    snippet = text[:3500]
    body = _CURATOR_ENTITIES_PROMPT.replace("{TEXT}", snippet)
    prompt = _chat_user_turn(body)
    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=300,
            temperature=0.1,
            stop=_chat_stops(),
        )
    except Exception as e:
        log.warning("curator_extract_entities: create_completion failed: %s", e)
        return None
    raw = (out.get("choices", [{}])[0].get("text") or "").strip()
    if raw and raw.lstrip().startswith("[") and not raw.rstrip().endswith("]"):
        raw = raw + "]"
    parsed = parse_entities_response(raw)
    if parsed is None:
        log.warning("curator_extract_entities: unparseable head=%r", raw[:120])
        return None
    # Dedup by lowercased name, keep first casing.
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in parsed:
        key = item["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= 20:
            break
    return deduped


def curator_extract_relationships(
    text: str,
    entity_names: list[str],
    *,
    llm=None,
) -> list[dict] | None:
    """Ask the curator to emit explicit triples between the given entities.

    Parameters
    ----------
    text:
        The memory content. Truncated to ~3500 chars before sending.
    entity_names:
        The exact entity names already extracted from the text. The model
        is told to use these names verbatim. We additionally validate
        subject/object membership against this set on the parsed output;
        triples whose endpoints aren't in the set are dropped.
    llm:
        Optional pre-loaded LLM handle.

    Returns ``None`` on any failure so callers fall back gracefully.
    """
    entity_names = [str(n).strip() for n in (entity_names or []) if str(n).strip()]
    if len(entity_names) < 2:
        return []
    if not text or len(text.strip()) < 8:
        return []

    if llm is None:
        if not _have_curator():
            return None
        try:
            llm = _get_llm()
        except Exception as e:  # pragma: no cover — defensive
            log.warning("curator_extract_relationships: _get_llm failed: %s", e)
            return None
    if llm is None:
        return None

    snippet = text[:3500]
    ents_str = ", ".join(f'"{n}"' for n in entity_names[:30])
    body = (
        _CURATOR_RELATIONSHIPS_PROMPT
        .replace("{ENTITIES}", ents_str)
        .replace("{TEXT}", snippet)
    )
    prompt = _chat_user_turn(body)
    try:
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=400,
            temperature=0.1,
            stop=_chat_stops(),
        )
    except Exception as e:
        log.warning("curator_extract_relationships: create_completion failed: %s", e)
        return None
    raw = (out.get("choices", [{}])[0].get("text") or "").strip()
    if raw and raw.lstrip().startswith("[") and not raw.rstrip().endswith("]"):
        raw = raw + "]"
    parsed = parse_relationships_response(raw)
    if parsed is None:
        log.warning("curator_extract_relationships: unparseable head=%r", raw[:120])
        return None
    # Validate subject/object are in the entities set (case-insensitive).
    name_by_norm = {n.lower(): n for n in entity_names}
    out_triples: list[dict] = []
    for t in parsed:
        s_norm = t["subject"].lower()
        o_norm = t["object"].lower()
        if s_norm not in name_by_norm or o_norm not in name_by_norm:
            log.debug(
                "curator_extract_relationships: dropping triple with unknown endpoint subj=%r obj=%r",
                t["subject"], t["object"],
            )
            continue
        out_triples.append({
            "subject": name_by_norm[s_norm],
            "predicate": t["predicate"],
            "object": name_by_norm[o_norm],
            "confidence": t["confidence"],
        })
        if len(out_triples) >= 12:
            break
    return out_triples
