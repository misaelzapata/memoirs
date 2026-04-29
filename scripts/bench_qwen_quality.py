"""
Qwen 2.5 3B curator quality benchmark on REAL product data.

Unlike `scripts/bench_models.py` (synthetic JSON-adherence over 10 fixed cases),
this bench measures the curator's actual output quality on conversations and
memories sampled from `.memoirs/memoirs.sqlite`. It exercises every Qwen entry
point that ships in production:

  * `gemma_extract`            — sample N real conversations -> candidates
  * `gemma_consolidate`        — sample N pending candidates vs heuristic action
  * `gemma_detect_contradiction` / `gemma_resolve_conflict` — induced pairs
  * `gemma_summarize_project`  / `summarize_conversation`   — long-thread summaries

The DB is never mutated: a SHM-safe copy is made into `tmp_path/qwen_quality.sqlite`
(via VACUUM INTO, falling back to `cp`) and all reads + extraction calls run
against the copy.

Usage
-----
    python scripts/bench_qwen_quality.py --n-extract 30 --n-consolidate 50 \
        --n-conflict 30 --no-conflict 20 --n-summary 10 \
        --out .memoirs/qwen_quality_report.json

Quick smoke (no LLM):
    python scripts/bench_qwen_quality.py --dry-run

Mock LLM (deterministic, no model load):
    python scripts/bench_qwen_quality.py --mock --n-extract 3 ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Make `memoirs.*` importable when running the script directly from the repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


log = logging.getLogger("bench_qwen_quality")


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def _utc_now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _percentile(values: list[float], pct: float) -> float:
    """Simple percentile without numpy (handles empty lists)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return float(s[k])


def _has_proper_noun(text: str) -> bool:
    """Heuristic: a capitalized token (>=3 chars) that's not at sentence start."""
    if not text:
        return False
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
    if len(tokens) < 2:
        return False
    # Skip the first token (sentence-start ambiguity).
    return any(t[0].isupper() for t in tokens[1:])


_FLUFF_PATTERNS = [
    re.compile(r"\bthe user is\b", re.I),
    re.compile(r"\bthis conversation (discusses|is about)\b", re.I),
    re.compile(r"\bin summary\b", re.I),
    re.compile(r"\bto summarize\b", re.I),
    re.compile(r"\bthe (assistant|chat|thread)\b.*\b(discusses|covers|summary|about)\b", re.I),
    re.compile(r"\bbased on the (conversation|messages|context)\b", re.I),
]


def _looks_fluffy(text: str) -> bool:
    if not text:
        return True
    for p in _FLUFF_PATTERNS:
        if p.search(text):
            return True
    return False


def _is_useful_content(content: str, *, min_len: int = 30, max_len: int = 300) -> bool:
    """Heuristic: 30-300 chars, has at least one proper noun, not pure fluff."""
    if not content:
        return False
    n = len(content)
    if n < min_len or n > max_len:
        return False
    if _looks_fluffy(content):
        return False
    return _has_proper_noun(content)


# ----------------------------------------------------------------------
# DB copy helpers
# ----------------------------------------------------------------------


def copy_db_readonly(src: Path, dst: Path) -> Path:
    """Copy the live DB to `dst` so the bench can't accidentally mutate prod.

    Prefer VACUUM INTO (consistent snapshot, ignores WAL/SHM hot files); fall
    back to a plain file copy if that's not supported on this build.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        conn = sqlite3.connect(str(src))
        try:
            conn.execute(f"VACUUM INTO '{dst.as_posix()}'")
            conn.commit()
        finally:
            conn.close()
        return dst
    except Exception:
        shutil.copyfile(src, dst)
        return dst


# ----------------------------------------------------------------------
# Sampling helpers
# ----------------------------------------------------------------------


def sample_long_conversations(db_path: Path, n: int, *, min_messages: int = 10,
                              rng: random.Random | None = None) -> list[str]:
    rng = rng or random.Random(0)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT c.id
              FROM conversations c
             WHERE c.message_count >= ?
            """,
            (min_messages,),
        ).fetchall()
    finally:
        conn.close()
    ids = [r["id"] for r in rows]
    rng.shuffle(ids)
    return ids[:n]


def load_conversation_messages(db_path: Path, conversation_id: str) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, role, content, ordinal FROM messages
             WHERE conversation_id = ? AND is_active = 1
             ORDER BY ordinal
            """,
            (conversation_id,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def sample_pending_candidates(db_path: Path, n: int) -> list[dict]:
    """Pull `n` rows from memory_candidates WHERE status='pending'."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, conversation_id, type, content, importance, confidence
              FROM memory_candidates
             WHERE status = 'pending'
             ORDER BY created_at DESC
             LIMIT ?
            """,
            (n,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def synthesize_candidates_from_memories(db_path: Path, n: int,
                                        rng: random.Random | None = None) -> list[dict]:
    """When the DB has no pending candidates, derive synthetic ones from active
    memories with light textual variation (suffix tweaks, importance jitter)."""
    rng = rng or random.Random(1)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, type, content, importance, confidence FROM memories
             WHERE archived_at IS NULL
             ORDER BY score DESC, importance DESC
             LIMIT ?
            """,
            (max(n * 4, 50),),
        ).fetchall()
    finally:
        conn.close()
    pool = [dict(r) for r in rows]
    rng.shuffle(pool)
    out: list[dict] = []
    suffixes = ["", " (per latest discussion)", " — recent update", " as of 2026"]
    for r in pool[:n]:
        suf = rng.choice(suffixes)
        out.append({
            "id": f"synthcand_{r['id'][:12]}",
            "conversation_id": None,
            "type": r["type"],
            "content": (r["content"] + suf).strip(),
            "importance": int(r["importance"]),
            "confidence": float(r["confidence"]),
            "_source_memory_id": r["id"],
            "_source_content": r["content"],
            "_synthetic": True,
        })
    return out


# ----------------------------------------------------------------------
# Contradiction pair generator
# ----------------------------------------------------------------------


# Hand-crafted contradiction pairs (induced from the corpus's domain).
INDUCED_CONTRADICTORY_PAIRS: list[tuple[str, str]] = [
    ("user prefers Python over Go for prototyping",
     "user switched to Go after the 2025 perf rewrite"),
    ("memoirs uses sqlite-vec for embeddings",
     "memoirs migrated off sqlite-vec onto lancedb"),
    ("the curator is Gemma 2 2B",
     "the curator is now Qwen 2.5 3B (Gemma deprecated)"),
    ("retrieval mode defaults to hybrid",
     "retrieval defaults to graph-PPR after the rewrite"),
    ("user lives in Buenos Aires",
     "user moved to Madrid in 2025"),
    ("user prefers async/await",
     "user dropped async/await in favor of threading"),
    ("dark mode is the default",
     "light mode is now the default"),
    ("MCP server runs over stdio only",
     "MCP server moved to HTTP transport"),
    ("user uses pytest for tests",
     "user switched to unittest after pytest issues"),
    ("Gemma is the curator",
     "Qwen replaces Gemma as the curator"),
    ("project uses GitHub Actions",
     "project migrated CI to GitLab runners"),
    ("user works on memoirs full-time",
     "user paused memoirs to focus on gocracker"),
    ("user's editor is VSCode",
     "user moved to Neovim after burnout with VSCode"),
    ("default DB path is .memoirs/memoirs.sqlite",
     "default DB path is now ~/.local/share/memoirs/db.sqlite"),
    ("user wants verbose responses",
     "user demands terse, no-fluff responses"),
    ("project uses spaCy for NER",
     "project replaced spaCy with stanza"),
    ("memories are stored as JSON",
     "memories are stored as Protobuf"),
    ("user uses Docker for dev",
     "user dropped Docker, runs everything natively"),
    ("user prefers Postgres over SQLite",
     "user committed to SQLite for local-first"),
    ("user's main project is fastfn",
     "user archived fastfn to focus on memoirs"),
    ("emoji is permitted in commit messages",
     "no emoji policy strictly enforced"),
    ("api runs on port 8283",
     "api was moved to port 9090"),
    ("scoring weights sum to 1.0",
     "scoring weights are now unnormalized raw values"),
    ("Qwen 2.5 3B is quantized to Q4",
     "Qwen 2.5 3B uses Q8 quantization now"),
    ("conversations stored as flat messages",
     "conversations stored as nested tree blocks"),
    ("user uses bash as the default shell",
     "user switched default shell to fish"),
    ("user's primary OS is Ubuntu",
     "user moved to Arch Linux for the rolling release"),
    ("memoirs requires Python 3.10+",
     "memoirs dropped Python 3.10 support, requires 3.12"),
    ("the watcher polls every 2 seconds",
     "watcher uses inotify, no polling"),
    ("Gemma summary memos are tagged is_summary=true",
     "summary memos are no longer tagged in metadata"),
]


# Hand-crafted non-contradictory pairs.
INDUCED_NONCONTRADICTORY_PAIRS: list[tuple[str, str]] = [
    ("user works on memoirs (local-first memory engine)",
     "user prefers concise responses"),
    ("memoirs uses sqlite-vec for embeddings",
     "Qwen 2.5 3B is the curator backend"),
    ("user is based in Buenos Aires",
     "user prefers Python for prototyping"),
    ("the project name is memoirs",
     "the curator outputs JSON only"),
    ("scoring weights are configurable",
     "default DB path is .memoirs/memoirs.sqlite"),
    ("user uses pytest",
     "user prefers async/await over callbacks"),
    ("the watcher debounces 1.5 seconds",
     "MCP server transports stdio messages"),
    ("user has a github PAT in 1Password",
     "user maintains gocracker as a side project"),
    ("recency half life is 30 days",
     "low-value archive threshold is 0.15"),
    ("memoirs is local-first",
     "memoirs stores data in SQLite"),
    ("user prefers terse responses",
     "user has multiple github repos"),
    ("project uses sqlite-vec",
     "project also uses BM25 for hybrid retrieval"),
    ("user owns the memoirs repo",
     "user has a fastfn project"),
    ("Gemma was the previous curator",
     "Qwen is the current curator"),  # historical fact + current — not contradictory
    ("the system supports multi-user",
     "user_id defaults to 'local' for single-user"),
    ("memoirs has 700+ tests",
     "memoirs uses pytest"),
    ("the engine performs hybrid retrieval",
     "the engine uses MMR for diversification"),
    ("user prefers no emoji",
     "user prefers terse responses"),
    ("memoirs ships a CLI",
     "memoirs ships an MCP server"),
    ("default embedding model is MiniLM",
     "embeddings have 384 dimensions"),
]


def build_contradiction_pairs(n_contra: int, n_nonconf: int,
                              rng: random.Random | None = None) -> list[dict]:
    """Return list of {a, b, expected_contradictory: bool} dicts."""
    rng = rng or random.Random(2)
    contra_pool = list(INDUCED_CONTRADICTORY_PAIRS)
    non_pool = list(INDUCED_NONCONTRADICTORY_PAIRS)
    rng.shuffle(contra_pool)
    rng.shuffle(non_pool)

    # Cycle if the requested count exceeds the curated pool.
    def _take(pool: list, k: int) -> list:
        if k <= len(pool):
            return pool[:k]
        out = []
        i = 0
        while len(out) < k:
            out.append(pool[i % len(pool)])
            i += 1
        return out

    cases: list[dict] = []
    for a, b in _take(contra_pool, n_contra):
        cases.append({"a": a, "b": b, "expected_contradictory": True})
    for a, b in _take(non_pool, n_nonconf):
        cases.append({"a": a, "b": b, "expected_contradictory": False})
    rng.shuffle(cases)
    return cases


# ----------------------------------------------------------------------
# Mock LLM (used by tests + --mock CLI flag)
# ----------------------------------------------------------------------


class MockLLM:
    """Minimal llama_cpp.Llama-compatible stub.

    Cycles through `responses` (or returns `default_response` once exhausted).
    Implements just enough of the surface (`tokenize`, `detokenize`,
    `create_completion`, `n_ctx`) for the curator helpers to run.
    """

    def __init__(self, responses: list[str] | None = None,
                 default_response: str = '{"action":"ADD","target_id":null,"reason":"mock"}',
                 n_ctx: int = 4096):
        self._responses = list(responses or [])
        self._default = default_response
        self.n_ctx = n_ctx
        self._token_map: dict[str, int] = {}
        self._inv_map: dict[int, str] = {}
        self._next_id = 1
        self.calls: list[dict] = []

    def tokenize(self, data, add_bos: bool = False, special: bool = False):
        if isinstance(data, (bytes, bytearray)):
            text = bytes(data).decode("utf-8", errors="ignore")
        else:
            text = str(data)
        toks: list[int] = []
        for word in text.split(" "):
            if not word:
                continue
            tid = self._token_map.get(word)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self._token_map[word] = tid
                self._inv_map[tid] = word
            toks.append(tid)
        return toks

    def detokenize(self, tokens):
        words = [self._inv_map.get(int(t), "") for t in tokens]
        return (" ".join(w for w in words if w)).encode("utf-8")

    def create_completion(self, prompt: str, **kwargs):
        self.calls.append({"prompt_len": len(prompt), **kwargs})
        if self._responses:
            text = self._responses.pop(0)
        else:
            text = self._default
        return {"choices": [{"text": text}]}


# ----------------------------------------------------------------------
# Bench result dataclasses
# ----------------------------------------------------------------------


@dataclass
class FunctionStats:
    n: int = 0
    json_valid_raw: int = 0
    json_valid_salvaged: int = 0
    json_valid_total: int = 0
    useful_count: int = 0
    fluff_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "n": self.n,
            "json_valid_raw": self.json_valid_raw,
            "json_valid_salvaged": self.json_valid_salvaged,
            "json_valid_total": self.json_valid_total,
            "useful_rate": (self.useful_count / self.n) if self.n else 0.0,
            "fluff_rate": (self.fluff_count / self.n) if self.n else 0.0,
            "latency_p50_ms": _percentile(self.latencies_ms, 50),
            "latency_p95_ms": _percentile(self.latencies_ms, 95),
            "samples": self.samples[:10],
        }
        d.update(self.extras)
        return d


# ----------------------------------------------------------------------
# Stage 1: extract
# ----------------------------------------------------------------------


def run_extract(db_path: Path, n: int, *, llm_override=None,
                rng: random.Random | None = None) -> FunctionStats:
    """Sample `n` real conversations and run gemma_extract; tally quality."""
    from memoirs.engine.gemma import gemma_extract, _is_user_meaningful  # noqa: F401

    stats = FunctionStats()
    conv_ids = sample_long_conversations(db_path, n, min_messages=10, rng=rng)
    stats.n = len(conv_ids)
    if not conv_ids:
        stats.extras["note"] = "no eligible conversations (>=10 active messages)"
        return stats

    no_extraction = 0
    candidates_total = 0
    candidates_useful = 0
    candidates_fluff = 0
    types_seen: dict[str, int] = {}

    for cid in conv_ids:
        msgs = load_conversation_messages(db_path, cid)
        t0 = time.perf_counter()
        try:
            if llm_override is not None:
                # Inject the override into gemma_extract via the singleton hook.
                from memoirs.engine import gemma as _g
                _g._LLM_SINGLETON = llm_override  # type: ignore[attr-defined]
                cands = gemma_extract(msgs)
            else:
                cands = gemma_extract(msgs)
        except Exception as e:
            log.warning("extract conv=%s failed: %s", cid[:12], e)
            cands = []
        dt_ms = (time.perf_counter() - t0) * 1000
        stats.latencies_ms.append(dt_ms)

        if not cands:
            no_extraction += 1
        # JSON-valid here means: candidates were produced (gemma_extract already
        # parses JSON internally; if it returns >0 candidates the response
        # parsed successfully).
        if cands:
            stats.json_valid_total += 1
            stats.json_valid_raw += 1  # we treat returned candidates as raw-OK
        for c in cands:
            candidates_total += 1
            content = c.content or ""
            if _is_useful_content(content):
                candidates_useful += 1
            if _looks_fluffy(content):
                candidates_fluff += 1
            types_seen[c.type] = types_seen.get(c.type, 0) + 1
        # First few samples (preview).
        if len(stats.samples) < 5:
            stats.samples.append({
                "conversation_id": cid,
                "n_messages": len(msgs),
                "n_candidates": len(cands),
                "candidates_preview": [
                    {"type": c.type, "content": (c.content or "")[:140],
                     "importance": c.importance, "confidence": round(c.confidence, 2)}
                    for c in cands[:4]
                ],
            })

    # Per-extract usefulness rate is computed at the conversation level: a
    # conversation is "useful" if it produced >=1 useful candidate.
    useful_convs = sum(1 for s in stats.samples if any(
        _is_useful_content(c["content"]) for c in s.get("candidates_preview", [])
    ))
    if candidates_total:
        stats.useful_count = candidates_useful  # absolute count
        stats.fluff_count = candidates_fluff
    stats.extras.update({
        "no_extraction_count": no_extraction,
        "candidates_total": candidates_total,
        "candidates_useful_count": candidates_useful,
        "candidates_fluff_count": candidates_fluff,
        "useful_candidate_rate": (candidates_useful / candidates_total) if candidates_total else 0.0,
        "fluff_candidate_rate": (candidates_fluff / candidates_total) if candidates_total else 0.0,
        "types_distribution": types_seen,
        "useful_conversations_in_sample": useful_convs,
    })
    return stats


# ----------------------------------------------------------------------
# Stage 2: consolidate
# ----------------------------------------------------------------------


def heuristic_consolidate_action(db_conn: sqlite3.Connection, candidate: dict) -> str:
    """Lightweight stand-in for the heuristic curator.

    We do not import `decide_memory_action` (would require full DB wiring +
    embeddings). Instead we approximate: if the candidate's content matches
    an existing memory exactly -> UPDATE, else ADD. That's enough to score
    "agreement" — Qwen should usually MERGE/UPDATE on synthesized
    near-duplicates.
    """
    content = (candidate.get("content") or "").strip()
    if not content:
        return "IGNORE"
    row = db_conn.execute(
        "SELECT id FROM memories WHERE content = ? AND archived_at IS NULL LIMIT 1",
        (content,),
    ).fetchone()
    if row:
        return "UPDATE"
    # If the candidate was synthesized from a real memory, the close cousin
    # is exact-match-able after stripping the suffix; treat near-dups as MERGE.
    if candidate.get("_synthetic"):
        return "MERGE"
    return "ADD"


def run_consolidate(db_path: Path, n: int, *, llm_override=None,
                    rng: random.Random | None = None) -> FunctionStats:
    from memoirs.engine.gemma import gemma_consolidate

    stats = FunctionStats()

    pending = sample_pending_candidates(db_path, n)
    if len(pending) < n:
        synth = synthesize_candidates_from_memories(db_path, n - len(pending), rng=rng)
        candidates = pending + synth
    else:
        candidates = pending
    stats.n = len(candidates)
    if not candidates:
        stats.extras["note"] = "no candidates available (pending=0, no memories to synthesize)"
        return stats

    db_conn = sqlite3.connect(str(db_path))
    db_conn.row_factory = sqlite3.Row

    agree = 0
    near_dup_merge_or_update = 0
    near_dup_total = 0
    actions_seen: dict[str, int] = {}
    parse_errors = 0

    try:
        for cand in candidates:
            heur_action = heuristic_consolidate_action(db_conn, cand)
            # Pull a few neighbors to give Qwen context. For synthetic candidates
            # we already know the source memory; surface it as the top neighbor.
            neighbors: list[dict] = []
            if cand.get("_source_memory_id"):
                neighbors.append({
                    "id": cand["_source_memory_id"],
                    "type": cand["type"],
                    "content": cand.get("_source_content") or "",
                    "similarity": 0.97,
                })
            # Add a couple of additional same-type neighbors.
            extra = db_conn.execute(
                "SELECT id, type, content FROM memories "
                "WHERE archived_at IS NULL AND type = ? LIMIT 3",
                (cand["type"],),
            ).fetchall()
            for r in extra:
                if r["id"] != cand.get("_source_memory_id"):
                    neighbors.append({
                        "id": r["id"], "type": r["type"], "content": r["content"],
                        "similarity": 0.85,
                    })

            t0 = time.perf_counter()
            try:
                result = gemma_consolidate(cand, neighbors, llm=llm_override)
            except Exception as e:
                log.warning("consolidate cand failed: %s", e)
                result = {"action": None, "source": "error", "reason": str(e)}
            dt_ms = (time.perf_counter() - t0) * 1000
            stats.latencies_ms.append(dt_ms)

            qwen_action = result.get("action")
            source = result.get("source", "")
            if qwen_action is None or "parse_error" in source:
                parse_errors += 1
            else:
                stats.json_valid_total += 1
                stats.json_valid_raw += 1
            if qwen_action:
                actions_seen[qwen_action] = actions_seen.get(qwen_action, 0) + 1

            if qwen_action == heur_action:
                agree += 1

            # On synthesized near-duplicates we EXPECT MERGE or UPDATE.
            if cand.get("_synthetic"):
                near_dup_total += 1
                if qwen_action in {"MERGE", "UPDATE"}:
                    near_dup_merge_or_update += 1

            if len(stats.samples) < 8:
                stats.samples.append({
                    "candidate_content": (cand.get("content") or "")[:140],
                    "type": cand["type"],
                    "synthetic": bool(cand.get("_synthetic")),
                    "heuristic_action": heur_action,
                    "qwen_action": qwen_action,
                    "qwen_reason": (result.get("reason") or "")[:120],
                    "qwen_target_id": result.get("target_id"),
                })
    finally:
        db_conn.close()

    stats.extras.update({
        "agreement_with_heuristic": (agree / stats.n) if stats.n else 0.0,
        "near_dup_total": near_dup_total,
        "near_dup_correctly_merged_or_updated": near_dup_merge_or_update,
        "near_dup_correctness_rate": (near_dup_merge_or_update / near_dup_total)
                                     if near_dup_total else 0.0,
        "actions_distribution": actions_seen,
        "parse_errors": parse_errors,
    })
    return stats


# ----------------------------------------------------------------------
# Stage 3: contradictions
# ----------------------------------------------------------------------


def run_contradictions(n_contra: int, n_nonconf: int, *,
                       llm_override=None, rng: random.Random | None = None) -> FunctionStats:
    from memoirs.engine.gemma import gemma_detect_contradiction, gemma_resolve_conflict

    stats = FunctionStats()
    cases = build_contradiction_pairs(n_contra, n_nonconf, rng=rng)
    stats.n = len(cases)
    if not cases:
        return stats

    tp = fp = tn = fn = 0
    winner_correct = 0
    winner_total = 0
    parse_errors = 0
    detect_results: list[dict] = []
    resolve_results: list[dict] = []

    for case in cases:
        a, b, expected = case["a"], case["b"], case["expected_contradictory"]

        t0 = time.perf_counter()
        det = gemma_detect_contradiction({"content": a}, {"content": b}, llm=llm_override)
        dt_ms = (time.perf_counter() - t0) * 1000
        stats.latencies_ms.append(dt_ms)

        source = det.get("source", "")
        if "parse_error" in source:
            parse_errors += 1
        else:
            stats.json_valid_total += 1
            stats.json_valid_raw += 1

        got = bool(det.get("contradictory"))
        if expected and got:
            tp += 1
        elif expected and not got:
            fn += 1
        elif (not expected) and got:
            fp += 1
        else:
            tn += 1

        # Winner heuristic: contradictory pairs in our pool are written so
        # that the "newer / more specific" claim is in B (the second slot).
        if expected and got:
            winner_total += 1
            if str(det.get("winner") or "").lower() == "b":
                winner_correct += 1

        # Cross-check with gemma_resolve_conflict for half the contra cases.
        if expected and len(resolve_results) < min(10, n_contra // 2 or 1):
            # NOTE: resolve_conflict needs the real Llama (no `llm` injection).
            # If we have a mock, skip — its output is already covered by detect.
            if llm_override is None:
                rc = gemma_resolve_conflict(a, b)
                resolve_results.append({
                    "a": a[:80], "b": b[:80],
                    "action": rc.get("action"), "winner": rc.get("winner"),
                    "contradictory": rc.get("contradictory"),
                })

        if len(detect_results) < 8:
            detect_results.append({
                "a": a[:80], "b": b[:80], "expected": expected,
                "got_contradictory": got,
                "winner": det.get("winner"),
                "reason": (det.get("reason") or "")[:80],
                "source": source,
            })

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    stats.samples = detect_results
    stats.extras.update({
        "n_contradictory": n_contra,
        "n_non_contradictory": n_nonconf,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "winner_picked_total": winner_total,
        "winner_correct": winner_correct,
        "winner_accuracy": (winner_correct / winner_total) if winner_total else 0.0,
        "parse_errors": parse_errors,
        "resolve_conflict_samples": resolve_results,
    })
    # `useful` for contradictions = correctly-classified (TP+TN), `fluff` = FP+FN.
    stats.useful_count = tp + tn
    stats.fluff_count = fp + fn
    return stats


# ----------------------------------------------------------------------
# Stage 4: summaries
# ----------------------------------------------------------------------


def _entity_overlap(text: str, original_corpus: str, *, min_token_len: int = 4) -> int:
    """Count how many proper-noun-like tokens from `text` appear in `original_corpus`."""
    if not text or not original_corpus:
        return 0
    src_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{%d,}" % (min_token_len - 1), original_corpus))
    summ_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{%d,}" % (min_token_len - 1), text)
    overlap = 0
    seen = set()
    for tok in summ_tokens:
        if tok in seen:
            continue
        seen.add(tok)
        if tok in src_tokens and tok[0].isupper():
            overlap += 1
    return overlap


def run_summaries(db_path: Path, n: int, *, llm_override=None,
                  rng: random.Random | None = None) -> FunctionStats:
    from memoirs.engine.gemma import gemma_summarize, gemma_summarize_project

    stats = FunctionStats()
    conv_ids = sample_long_conversations(db_path, n, min_messages=20, rng=rng)
    stats.n = len(conv_ids)
    if not conv_ids:
        stats.extras["note"] = "no eligible long conversations (>=20 active messages)"
        return stats

    reasonable_len = 0
    refs_3plus = 0
    non_generic = 0
    skips = 0

    project_summaries: list[dict] = []

    for cid in conv_ids:
        msgs = load_conversation_messages(db_path, cid)
        # Build a corpus string for entity-overlap measurement.
        corpus = " ".join((m.get("content") or "") for m in msgs)

        t0 = time.perf_counter()
        try:
            if llm_override is not None:
                from memoirs.engine import gemma as _g
                _g._LLM_SINGLETON = llm_override  # type: ignore[attr-defined]
            summary = gemma_summarize(msgs)
        except Exception as e:
            log.warning("summarize conv=%s failed: %s", cid[:12], e)
            summary = None
        dt_ms = (time.perf_counter() - t0) * 1000
        stats.latencies_ms.append(dt_ms)

        if not summary:
            skips += 1
            if len(stats.samples) < 6:
                stats.samples.append({"conversation_id": cid, "summary": None,
                                       "reason": "skipped"})
            continue

        stats.json_valid_total += 1
        stats.json_valid_raw += 1

        n_chars = len(summary)
        is_reasonable_len = 80 <= n_chars <= 1200
        ents = _entity_overlap(summary, corpus)
        is_specific = not _looks_fluffy(summary)

        if is_reasonable_len:
            reasonable_len += 1
        if ents >= 3:
            refs_3plus += 1
        if is_specific:
            non_generic += 1
        if is_reasonable_len and ents >= 3 and is_specific:
            stats.useful_count += 1
        if not is_specific:
            stats.fluff_count += 1

        if len(stats.samples) < 6:
            stats.samples.append({
                "conversation_id": cid,
                "n_messages": len(msgs),
                "summary_chars": n_chars,
                "entity_overlap": ents,
                "fluffy": _looks_fluffy(summary),
                "summary_preview": summary[:400],
            })

    # Bonus: gemma_summarize_project — pull a small set of memories tagged
    # to known projects and summarize each. This stresses a different prompt.
    project_names = ["memoirs", "gocracker", "fastfn"]
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        for proj in project_names:
            mem_rows = conn.execute(
                """
                SELECT id, type, content FROM memories
                 WHERE archived_at IS NULL AND content LIKE ?
                 ORDER BY importance DESC, score DESC
                 LIMIT 6
                """,
                (f"%{proj}%",),
            ).fetchall()
            if not mem_rows:
                continue
            mems = [{"id": r["id"], "type": r["type"],
                     "content": r["content"], "similarity": 0.9} for r in mem_rows]
            t0 = time.perf_counter()
            try:
                ps = gemma_summarize_project(proj, mems, llm=llm_override)
            except Exception as e:
                log.warning("summarize_project %s failed: %s", proj, e)
                ps = ""
            dt_ms = (time.perf_counter() - t0) * 1000
            stats.latencies_ms.append(dt_ms)
            project_summaries.append({
                "project": proj,
                "n_memories": len(mems),
                "summary_chars": len(ps),
                "fluffy": _looks_fluffy(ps),
                "summary_preview": ps[:400],
                "latency_ms": dt_ms,
            })
    finally:
        conn.close()

    stats.extras.update({
        "skipped": skips,
        "reasonable_length_count": reasonable_len,
        "entity_refs_3plus_count": refs_3plus,
        "non_generic_count": non_generic,
        "reasonable_length_rate": (reasonable_len / stats.n) if stats.n else 0.0,
        "entity_refs_3plus_rate": (refs_3plus / stats.n) if stats.n else 0.0,
        "non_generic_rate": (non_generic / stats.n) if stats.n else 0.0,
        "project_summaries": project_summaries,
    })
    return stats


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def render_table(report: dict) -> str:
    lines = []
    lines.append("\n=== Qwen Quality Bench (real corpus) ===")
    lines.append(f"backend     : {report['backend']}")
    lines.append(f"model       : {report['model_path']}")
    lines.append(f"timestamp   : {report['timestamp']}")
    lines.append(f"db          : {report.get('db_snapshot', '?')}")
    lines.append("")
    header = f"{'stage':<14s}  {'n':>4s}  {'json_ok':>8s}  {'useful':>7s}  {'fluff':>7s}  {'p50ms':>7s}  {'p95ms':>7s}  notes"
    lines.append(header)
    lines.append("-" * len(header))
    for stage in ("extract", "consolidate", "contradictions", "summaries"):
        s = report.get(stage) or {}
        if not s:
            continue
        notes = []
        if stage == "extract":
            notes.append(f"cands={s.get('candidates_total', 0)}")
            notes.append(f"useful_cands={s.get('useful_candidate_rate', 0):.2f}")
            notes.append(f"types={len(s.get('types_distribution') or {})}")
        elif stage == "consolidate":
            notes.append(f"agree={s.get('agreement_with_heuristic', 0):.2f}")
            notes.append(f"near_dup_ok={s.get('near_dup_correctness_rate', 0):.2f}")
        elif stage == "contradictions":
            notes.append(f"P={s.get('precision', 0):.2f}")
            notes.append(f"R={s.get('recall', 0):.2f}")
            notes.append(f"F1={s.get('f1', 0):.2f}")
            notes.append(f"winner={s.get('winner_accuracy', 0):.2f}")
        elif stage == "summaries":
            notes.append(f"len_ok={s.get('reasonable_length_rate', 0):.2f}")
            notes.append(f"ents>=3={s.get('entity_refs_3plus_rate', 0):.2f}")
        lines.append(
            f"{stage:<14s}  "
            f"{s.get('n', 0):>4d}  "
            f"{s.get('json_valid_total', 0):>8d}  "
            f"{s.get('useful_rate', 0):>7.2f}  "
            f"{s.get('fluff_rate', 0):>7.2f}  "
            f"{s.get('latency_p50_ms', 0):>7.0f}  "
            f"{s.get('latency_p95_ms', 0):>7.0f}  "
            + ", ".join(notes)
        )

    # Compare vs synthetic bench (scripts/bench_models.py output).
    cmp_path = Path(".memoirs/bench_models.json")
    if cmp_path.exists():
        try:
            cmp = json.loads(cmp_path.read_text())
            lines.append("")
            lines.append("--- vs scripts/bench_models.py (synthetic JSON adherence) ---")
            for entry in cmp:
                if entry.get("name", "").startswith("qwen"):
                    lines.append(
                        f"  synthetic qwen: raw_ok={entry.get('raw_ok')}/{entry.get('n')}"
                        f"  salv_ok={entry.get('salvage_ok')}/{entry.get('n')}"
                        f"  p50={entry.get('latency_p50_ms', 0):.0f}ms"
                        f"  p95={entry.get('latency_p95_ms', 0):.0f}ms"
                    )
        except Exception:
            pass
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Qwen 2.5 3B quality bench on real Memoirs corpus")
    ap.add_argument("--db", default=os.environ.get("MEMOIRS_DB", ".memoirs/memoirs.sqlite"),
                    help="path to live memoirs DB (will be COPIED, never mutated)")
    ap.add_argument("--n-extract", type=int, default=30)
    ap.add_argument("--n-consolidate", type=int, default=50)
    ap.add_argument("--n-conflict", type=int, default=30,
                    help="number of induced contradictory pairs")
    ap.add_argument("--no-conflict", type=int, default=20,
                    help="number of induced non-contradictory pairs")
    ap.add_argument("--n-summary", type=int, default=10)
    ap.add_argument("--out", default=".memoirs/qwen_quality_report.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="do not call any LLM; just verify CLI + sampling work")
    ap.add_argument("--mock", action="store_true",
                    help="use deterministic MockLLM in place of real curator")
    ap.add_argument("--skip-stages", default="",
                    help="comma list of stages to skip: extract,consolidate,contradictions,summaries")
    return ap


def run_bench(args: argparse.Namespace) -> dict:
    """Programmatic entrypoint (used by tests too)."""
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rng = random.Random(args.seed)
    skip = {s.strip() for s in (args.skip_stages or "").split(",") if s.strip()}

    # Read backend info before we touch anything.
    from memoirs.config import CURATOR_BACKEND, CURATOR_MODEL_PATH

    src_db = Path(args.db)
    if not src_db.exists():
        raise FileNotFoundError(f"DB not found: {src_db}")

    tmpdir = Path(tempfile.mkdtemp(prefix="qwen_bench_"))
    snapshot = copy_db_readonly(src_db, tmpdir / "qwen_quality.sqlite")
    log.info("snapshot: %s -> %s", src_db, snapshot)

    report: dict = {
        "backend": CURATOR_BACKEND,
        "model_path": str(CURATOR_MODEL_PATH),
        "timestamp": _utc_now_iso(),
        "db_snapshot": str(snapshot),
        "args": vars(args),
    }

    # Pick LLM override mode.
    override = None
    if args.dry_run:
        report["mode"] = "dry-run"
        report["extract"] = {"n": 0, "skipped": "dry-run"}
        report["consolidate"] = {"n": 0, "skipped": "dry-run"}
        report["contradictions"] = {"n": 0, "skipped": "dry-run"}
        report["summaries"] = {"n": 0, "skipped": "dry-run"}
        return report

    if args.mock:
        report["mode"] = "mock"
        # Defaults that exercise each parser path well.
        override = MockLLM(default_response='{"action":"ADD","target_id":null,"reason":"mock"}')
    else:
        report["mode"] = "real"

    # ---- Stage 1: extract
    if "extract" not in skip and args.n_extract > 0:
        log.info("=== stage: extract (n=%d) ===", args.n_extract)
        report["extract"] = run_extract(snapshot, args.n_extract,
                                        llm_override=override, rng=rng).to_dict()
    else:
        report["extract"] = {"n": 0, "skipped": True}

    # ---- Stage 2: consolidate
    if "consolidate" not in skip and args.n_consolidate > 0:
        log.info("=== stage: consolidate (n=%d) ===", args.n_consolidate)
        # For consolidate, MockLLM needs to emit consolidation JSON.
        co_override = MockLLM(default_response='{"action":"MERGE","target_id":null,"reason":"mock dup"}') if args.mock else override
        report["consolidate"] = run_consolidate(snapshot, args.n_consolidate,
                                                llm_override=co_override, rng=rng).to_dict()
    else:
        report["consolidate"] = {"n": 0, "skipped": True}

    # ---- Stage 3: contradictions
    if "contradictions" not in skip and (args.n_conflict + args.no_conflict) > 0:
        log.info("=== stage: contradictions (contra=%d non=%d) ===",
                 args.n_conflict, args.no_conflict)
        ct_override = MockLLM(default_response='{"contradictory":true,"winner":"b","reason":"mock"}') if args.mock else override
        report["contradictions"] = run_contradictions(args.n_conflict, args.no_conflict,
                                                      llm_override=ct_override,
                                                      rng=rng).to_dict()
    else:
        report["contradictions"] = {"n": 0, "skipped": True}

    # ---- Stage 4: summaries
    if "summaries" not in skip and args.n_summary > 0:
        log.info("=== stage: summaries (n=%d) ===", args.n_summary)
        sm_override = MockLLM(default_response="memoirs is a local-first memory engine with sqlite-vec, Qwen curator, and pytest tests passing.") if args.mock else override
        report["summaries"] = run_summaries(snapshot, args.n_summary,
                                            llm_override=sm_override, rng=rng).to_dict()
    else:
        report["summaries"] = {"n": 0, "skipped": True}

    return report


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = run_bench(args)

    # Persist JSON report.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    # Stdout summary.
    print(render_table(report))
    print(f"\nfull report -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
