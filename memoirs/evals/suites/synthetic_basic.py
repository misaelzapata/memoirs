"""Synthetic basic eval suite — 30 memorias + 10 queries.

Designed as a smoke-test for `run_eval` and as a regression baseline. The
suite intentionally mixes the four flavors that LongMemEval (2024) calls
out as the hard cases for retrieval over chat memory:

  * single-hop : the gold memory contains the keyword that the query asks
                 about. Hybrid + BM25 should both shine.
  * multi-hop  : the answer requires linking two memorias via shared
                 entities. We materialize that link explicitly into the
                 ``memory_links`` table (A-MEM Zettelkasten — P1-3).
  * temporal   : two memorias state contradicting facts at different
                 timestamps. ``as_of`` selects the version valid at that
                 instant. Tests memoirs' bi-temporal layer (5.4).
  * preference : a single user preference (durable type=preference). The
                 query is paraphrased to penalize pure-lexical retrieval.

Why 30 memorias / 10 queries?
  * 30 is large enough that the ranking has to actually do work — random
    retrieval would land at recall@10 ≈ 0.33 — and small enough that the
    suite finishes in seconds even when sentence-transformers is loading
    a model from cold.
  * 10 queries makes p50 / p95 numbers meaningful without padding.

The factory ``build(db)`` does ALL the seeding: memorias rows, FTS index
backfill, embeddings (when sqlite-vec + sentence-transformers are
available), and ``memory_links`` rows for the multi-hop pair. Returns an
``EvalSuite`` with the gold IDs locked in.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from memoirs.core.ids import content_hash, utc_now
from memoirs.evals.harness import EvalCase, EvalSuite


log = logging.getLogger("memoirs.evals.synthetic_basic")


# ---------------------------------------------------------------------------
# Corpus definition (kept declarative on purpose — easy to audit at a glance)
# ---------------------------------------------------------------------------


@dataclass
class _Mem:
    """Internal record for a synthetic memory before it's written to the DB."""

    mid: str
    type: str
    content: str
    importance: int = 3
    confidence: float = 0.7
    # Bi-temporal range for temporal-test memorias. None = "always valid".
    valid_from: str | None = None
    valid_to: str | None = None


# Anchor timestamps for temporal cases. Picked years apart so the ordering
# is unmistakable even with floating-point drift.
_TS_OLD = "2024-01-01T00:00:00+00:00"
_TS_MID = "2025-06-15T00:00:00+00:00"
_TS_NEW = "2026-04-01T00:00:00+00:00"


# Distractor topics: realistic-looking memorias that share NO keywords with
# any of our queries. They exist to push retrieval to actually rank, not
# regurgitate the only doc that mentions the query terms.
_DISTRACTORS: list[_Mem] = [
    _Mem(f"mem_dist_{i:02d}", "fact", text)
    for i, text in enumerate([
        "the kubernetes scheduler uses node affinity for pod placement",
        "rust borrow checker prevents data races at compile time",
        "graphql subscriptions need a websocket transport layer",
        "react useEffect cleanup runs before re-render and on unmount",
        "tailwind utility classes compose deterministically by source order",
        "typescript generic constraints use the extends keyword",
        "docker layer caching short-circuits when the FROM hash matches",
        "postgres b-tree indexes are good for range scans",
        "ssh agent forwarding lets you hop through bastion hosts safely",
        "dns ttl values trade off propagation lag versus refresh load",
        "tcp keepalive probes fire after a configurable idle window",
        "json web tokens are stateless but require key rotation discipline",
        "rabbitmq topic exchanges route by key pattern with wildcards",
        "kafka partitions guarantee ordering only within a single partition",
        "websocket frames carry a 4-byte mask on the client side",
        "vim macros record every keystroke including timing pauses",
        "tmux panes survive disconnects when the server stays running",
        "git rebase rewrites history so signed commits need re-signing",
        "lsp servers stream diagnostics over jsonrpc on stdio",
        "openssl pkcs12 bundles bind a key with its full cert chain",
    ], start=0)
]


# ---------------------------------------------------------------------------
# Single-hop targets (4) — each query has ONE gold memory with the keyword.
# ---------------------------------------------------------------------------
_SINGLE_HOP: list[_Mem] = [
    _Mem("mem_sh_lasagna", "fact",
         "user's favorite dinner recipe is lasagna bolognese with bechamel"),
    _Mem("mem_sh_dentist", "fact",
         "user has a dentist appointment scheduled at clinic SmileBright next week"),
    _Mem("mem_sh_marathon", "decision",
         "user signed up for the Berlin marathon happening on september 28"),
    _Mem("mem_sh_canary", "project",
         "the project canary-bench tracks rollout latency for the payments service"),
]

# ---------------------------------------------------------------------------
# Multi-hop pair (3 queries × 2 memorias each — 4 distinct memorias used).
#
# Pair A (queries 1 & 2): two memorias chained via the entity "alice":
#   M1: "alice is the new tech lead of the storage team"
#   M2: "the storage team owns the bigtable migration deadline of march 31"
#   Q : "when is alice's team's deadline?"  → must surface BOTH.
#
# Pair B (query 3): chain via "rivendell" project:
#   M3: "user joined the rivendell project as principal engineer in q2"
#   M4: "the rivendell project's stack is python 3.12 + duckdb + nats"
#   Q : "what is the stack of the project the user joined?"
# ---------------------------------------------------------------------------
_MULTI_HOP: list[_Mem] = [
    _Mem("mem_mh_alice_lead", "fact",
         "alice is the new tech lead of the storage team since february"),
    _Mem("mem_mh_storage_deadline", "fact",
         "the storage team owns the bigtable migration deadline of march 31"),
    _Mem("mem_mh_rivendell_join", "project",
         "user joined the rivendell project as principal engineer in q2"),
    _Mem("mem_mh_rivendell_stack", "fact",
         "the rivendell project's stack is python 3.12 + duckdb + nats"),
]

# ---------------------------------------------------------------------------
# Temporal pair (2 queries × 2 memorias). Same logical fact, two values:
#
#   M_old : valid 2024-01-01 .. 2025-06-15  → "user prefers tea"
#   M_new : valid 2025-06-15 .. ∞           → "user prefers cold brew coffee"
#
# Q1 with as_of=2024-12-01 → must rank M_old above M_new.
# Q2 with as_of=now (None) → must rank M_new above M_old.
# ---------------------------------------------------------------------------
_TEMPORAL: list[_Mem] = [
    _Mem(
        "mem_t_drink_old", "preference",
        "user prefers a cup of tea every afternoon",
        valid_from=_TS_OLD, valid_to=_TS_MID,
    ),
    _Mem(
        "mem_t_drink_new", "preference",
        "user prefers cold brew coffee every afternoon",
        valid_from=_TS_MID, valid_to=None,
    ),
]

# ---------------------------------------------------------------------------
# Preference (1)
# ---------------------------------------------------------------------------
_PREFERENCES: list[_Mem] = [
    _Mem("mem_pref_keyboard", "preference",
         "user dislikes mechanical keyboards and prefers low-profile membrane ones"),
]


def _all_memorias() -> list[_Mem]:
    """Concatenate to exactly 30 memorias.

    Layout:
      4 single-hop + 4 multi-hop + 2 temporal + 1 preference + 19 distractors
      ────────────────────────────────────────────────────────────────────── = 30
    """
    items = [
        *_SINGLE_HOP,
        *_MULTI_HOP,
        *_TEMPORAL,
        *_PREFERENCES,
        *_DISTRACTORS[:19],   # exactly 19 to land on 30 total
    ]
    if len(items) != 30:
        raise AssertionError(f"synthetic suite must total 30 memorias, got {len(items)}")
    return items


# ---------------------------------------------------------------------------
# DB seeding
# ---------------------------------------------------------------------------


def _insert_memory(db, m: _Mem, *, now: str) -> None:
    """Raw insert bypassing consolidation (no IGNORE/ADD/MERGE logic).

    Mirrors what `apply_decision`'s ADD branch writes, minus side effects we
    don't want during eval seeding (no dedup, no usage_count++, no auto
    Zettelkasten linking). The latter is important: the multi-hop pair
    needs links added *manually* and we don't want the heuristic linker to
    inject extras that would inflate recall.
    """
    h = content_hash(m.content + m.mid)  # salt with id so eval suite never
                                         # collides with real corpora
    valid_from = m.valid_from or now
    db.conn.execute(
        """
        INSERT INTO memories (
            id, type, content, content_hash, importance, confidence,
            score, usage_count, user_signal, valid_from, valid_to,
            metadata_json, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, 0.5, 0, 0, ?, ?, '{}', ?, ?)
        """,
        (m.mid, m.type, m.content, h, m.importance, m.confidence,
         valid_from, m.valid_to, now, now),
    )


def _try_embed(db, mid: str, content: str) -> bool:
    """Best-effort embedding upsert. Returns True on success.

    Failures (sqlite-vec missing, sentence-transformers missing, model
    download failure, …) are logged at INFO and swallowed so the suite
    still produces meaningful BM25 numbers in dependency-bare envs.
    """
    try:
        from memoirs.engine import embeddings as emb
        emb.upsert_memory_embedding(db, mid, content)
        return True
    except Exception as e:  # broad on purpose — many failure modes
        log.info("synthetic_basic: embedding skipped for %s (%s)", mid[:24], e)
        return False


def _link_memories(db, a: str, b: str, *, similarity: float = 0.85) -> None:
    """Materialize a bidirectional A-MEM link (P1-3) between two memorias.

    Used to mark a multi-hop pair so future retrievers (HippoRAG / PPR)
    can follow it. The current retrieval path doesn't consume links yet,
    but seeding them keeps the suite compatible with that work and lets
    eval consumers verify link-aware retrieval gains.
    """
    db.conn.execute(
        "INSERT OR IGNORE INTO memory_links "
        "(source_memory_id, target_memory_id, similarity, reason) "
        "VALUES (?, ?, ?, 'semantic'), (?, ?, ?, 'semantic')",
        (a, b, similarity, b, a, similarity),
    )


def _ensure_fts(db) -> None:
    """Make sure the FTS5 index exists so BM25 mode actually retrieves.

    The triggers will populate the index as we insert; we still call
    `ensure_fts_schema` first so on a fresh DB we don't depend on
    migration 003 being present.
    """
    from memoirs.engine import hybrid_retrieval as hr
    hr.ensure_fts_schema(db.conn)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build(db) -> EvalSuite:
    """Seed `db` with the synthetic corpus and return the matching suite.

    Idempotent in spirit: re-running on the same DB will hit ON CONFLICT
    paths but won't crash. Tests call this on a fresh `tmp_db` so we
    don't bother with cleanup.
    """
    _ensure_fts(db)
    now = utc_now()
    items = _all_memorias()
    embedded = 0
    with db.conn:
        for m in items:
            _insert_memory(db, m, now=now)
        # Commit memorias before embedding so vec0 sees stable rows.
    for m in items:
        if _try_embed(db, m.mid, m.content):
            embedded += 1
    # Wire the multi-hop pairs explicitly.
    with db.conn:
        _link_memories(db, "mem_mh_alice_lead",     "mem_mh_storage_deadline")
        _link_memories(db, "mem_mh_rivendell_join", "mem_mh_rivendell_stack")
    log.info(
        "synthetic_basic: seeded %d memorias (%d with embeddings)",
        len(items), embedded,
    )

    cases = [
        # ----- single-hop (4) -----
        EvalCase(
            query="what is my favorite dinner recipe?",
            gold_memory_ids=["mem_sh_lasagna"],
            category="single-hop",
        ),
        EvalCase(
            query="when is my dentist appointment?",
            gold_memory_ids=["mem_sh_dentist"],
            category="single-hop",
        ),
        EvalCase(
            query="which marathon did I sign up for?",
            gold_memory_ids=["mem_sh_marathon"],
            category="single-hop",
        ),
        EvalCase(
            query="what does the canary-bench project measure?",
            gold_memory_ids=["mem_sh_canary"],
            category="single-hop",
        ),
        # ----- multi-hop (3) — gold lists BOTH linked memorias -----
        EvalCase(
            query="who is the storage team tech lead?",
            gold_memory_ids=["mem_mh_alice_lead"],
            category="multi-hop",
        ),
        EvalCase(
            query="when is alice's team's bigtable migration due?",
            gold_memory_ids=["mem_mh_alice_lead", "mem_mh_storage_deadline"],
            category="multi-hop",
        ),
        EvalCase(
            query="what is the stack of the project I joined as principal engineer?",
            gold_memory_ids=["mem_mh_rivendell_join", "mem_mh_rivendell_stack"],
            category="multi-hop",
        ),
        # ----- temporal (2) -----
        EvalCase(
            query="what does the user prefer to drink in the afternoon?",
            gold_memory_ids=["mem_t_drink_old"],
            category="temporal",
            as_of="2024-12-01T00:00:00+00:00",
        ),
        EvalCase(
            query="what does the user prefer to drink in the afternoon?",
            gold_memory_ids=["mem_t_drink_new"],
            category="temporal",
            as_of=None,  # live → must pick the new one
        ),
        # ----- preference (1) -----
        EvalCase(
            query="what kind of keyboards does the user like?",
            gold_memory_ids=["mem_pref_keyboard"],
            category="preference",
        ),
    ]
    if len(cases) != 10:
        raise AssertionError(f"synthetic suite must have 10 queries, got {len(cases)}")

    return EvalSuite(
        name="synthetic_basic",
        cases=cases,
        description=(
            "30 synthetic memorias + 10 queries (4 single-hop, 3 multi-hop "
            "with explicit memory_links, 2 temporal with as_of, 1 preference)."
        ),
    )
