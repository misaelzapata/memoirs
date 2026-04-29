"""Layer 3 — entity graph.

Extracts entities and relationships from memory content. The PRIMARY path is
LLM-based (`curator_extract_entities` / `curator_extract_relationships`) using
the Qwen3-Instruct curator we already load for consolidation. The legacy regex
/ CamelCase / sufijo / co-occurrence heuristics are kept ONLY as a fallback
for environments without a curator (and as the test contract for older unit
tests).

Toggle via ``MEMOIRS_GRAPH_LLM`` (default ``on``):
    on  → use LLM. Falls through to heuristic when ``_have_curator()`` is False
          OR when the LLM call/parse fails for a given input.
    off → always use the heuristic path. Used by tests that pin the legacy
          contract.

For relationships, repeated calls per memory are cached in
``_graph_relationships_cache`` (auto-created in the same SQLite DB) keyed by
``(memory_id, content_hash, entity_set_hash)`` with a 30-day TTL — so re-running
``index_memory_entities`` doesn't pay the LLM cost twice for unchanged inputs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter
from typing import Iterable

from ..db import MemoirsDB, stable_id, utc_now


log = logging.getLogger("memoirs.graph")


# ---------------------------------------------------------------------------
# Toggles
# ---------------------------------------------------------------------------


def _llm_enabled() -> bool:
    """True when the LLM-based path should be tried.

    Default ON. Set ``MEMOIRS_GRAPH_LLM=off`` to force the heuristic. Tests
    that lock the heuristic contract use ``monkeypatch.setenv``.
    """
    return os.environ.get("MEMOIRS_GRAPH_LLM", "on").lower() not in ("off", "0", "false", "no")


def _heuristic_fallback_enabled() -> bool:
    """True when LLM failures should fall back to the heuristic extractor.

    Default ON. Disable with ``MEMOIRS_GRAPH_HEURISTIC_FALLBACK=off`` so that
    LLM parse errors surface as empty entities (useful for evaluating the LLM
    in isolation).
    """
    return os.environ.get("MEMOIRS_GRAPH_HEURISTIC_FALLBACK", "on").lower() not in ("off", "0", "false", "no")


# ---------------------------------------------------------------------------
# Heuristic path (legacy — kept as fallback / test contract)
# ---------------------------------------------------------------------------

# Heuristic vocabulary — extend as the corpus grows
_TOOL_HINTS = {
    "sqlite", "postgres", "mysql", "redis", "duckdb", "chroma", "sqlite-vec",
    "ollama", "llama.cpp", "llama-cpp-python", "gemma", "openai", "anthropic", "claude",
    "mcp", "fastmcp", "watchdog", "sentence-transformers", "transformers",
    "python", "rust", "go", "typescript", "javascript", "node",
    "vscode", "cursor", "claude code", "continue",
}
_PROJECT_HINTS = {"memoirs", "mentedb", "memengine", "powermem", "memmachine", "letta", "zep", "mem0"}


def _normalize(name: str) -> str:
    return name.strip().lower()


def _is_camel_case(name: str) -> bool:
    """True if name is real CamelCase (e.g. ReactNative, FastAPI) — at least
    one lowercase→uppercase transition, no spaces, ≥ 4 chars. Pure-stdlib check
    instead of regex.
    """
    if " " in name or len(name) < 4:
        return False
    for i in range(len(name) - 1):
        if name[i].islower() and name[i + 1].isupper():
            return True
    return False


def _classify_entity_heuristic(name: str) -> str:
    """Heuristic classifier — vocabulary + CamelCase. Used as fallback."""
    n = _normalize(name)
    if n in _TOOL_HINTS:
        return "tool"
    if n in _PROJECT_HINTS:
        return "project"
    if _is_camel_case(name):
        return "concept"
    return "other"


# Backwards-compatible alias used by other engine modules + tests that pin
# heuristic behaviour. We keep the heuristic semantics here unconditionally
# (the LLM gives type at extraction time, so we don't need a runtime classifier).
_classify_entity = _classify_entity_heuristic


# Generic words that the classifier should NEVER promote to entity even if
# they trigger a CamelCase or hint match.
_ENTITY_STOPLIST = {
    "reminder", "todo", "fixme", "note", "warning", "info", "debug",
    "deprecate", "deprecated", "the", "a", "an", "and", "or", "but",
    "this", "that", "these", "those", "see", "use", "using", "used",
    "type", "kind", "name", "value", "id", "key", "true", "false", "null",
    "none", "yes", "no", "ok",
}


def _extract_entities_heuristic(text: str) -> list[tuple[str, str]]:
    """Legacy heuristic entity extractor (spaCy → vocab → backticks → CamelCase).

    Kept as a fallback for environments without a curator and as the test
    contract for older unit tests. The primary public extractor is
    :func:`extract_entities`, which prefers the LLM path when available.
    """
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    # 1. spaCy NER — high-precision proper noun detection (NOT our heuristic;
    # this is spaCy's model). Kept on the heuristic path because it doesn't
    # require the curator.
    try:
        from . import extract_spacy
        if extract_spacy.is_available() and len(text) > 20:
            nlp_en, nlp_es = extract_spacy._load_models()
            nlp = nlp_en or nlp_es
            if nlp:
                doc = nlp(text[:4000])
                for ent in doc.ents:
                    name = ent.text.strip()
                    if not name or len(name) < 2:
                        continue
                    norm = name.lower()
                    if norm in _ENTITY_STOPLIST or norm in seen:
                        continue
                    seen.add(norm)
                    if ent.label_ in ("PERSON",):
                        ent_type = "person"
                    elif ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
                        ent_type = "project"
                    elif ent.label_ in ("GPE", "LOC", "FAC", "EVENT", "LAW"):
                        ent_type = "concept"
                    else:
                        ent_type = "other"
                    if norm in _TOOL_HINTS:
                        ent_type = "tool"
                    elif norm in _PROJECT_HINTS:
                        ent_type = "project"
                    candidates.append((name, ent_type))
    except Exception as e:
        log.debug("spaCy NER fallback: %s", e)

    # 2. Vocabulary hints
    lower = text.lower()
    for hint in _TOOL_HINTS | _PROJECT_HINTS:
        if hint in lower and hint not in seen:
            seen.add(hint)
            candidates.append((hint, "tool" if hint in _TOOL_HINTS else "project"))

    # 3. Backtick-quoted spans
    parts = text.split("`")
    for i in range(1, len(parts), 2):
        m = parts[i].strip()
        norm = m.lower()
        if not (2 <= len(m) <= 40):
            continue
        if norm in _ENTITY_STOPLIST or norm in seen:
            continue
        seen.add(norm)
        candidates.append((m, _classify_entity_heuristic(m)))

    # 4. CamelCase tokens
    for token in text.split():
        clean = token.strip(".,;:!?\"'()[]{}")
        norm = clean.lower()
        if not (4 <= len(clean) <= 40):
            continue
        if norm in _ENTITY_STOPLIST or norm in seen:
            continue
        if _is_camel_case(clean):
            seen.add(norm)
            candidates.append((clean, "concept"))

    return candidates[:20]


# ---------------------------------------------------------------------------
# Public extractor — LLM-first with heuristic fallback
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Return ``[(name, type), ...]`` for the given text.

    Strategy:
      - If ``MEMOIRS_GRAPH_LLM`` is on AND the curator is available, ask
        :func:`memoirs.engine.curator.curator_extract_entities` to emit a
        ``[{"name","type"}]`` list and convert it to tuples.
      - On any failure (LLM unavailable, parse error) AND when
        ``MEMOIRS_GRAPH_HEURISTIC_FALLBACK`` is on, fall back to
        :func:`_extract_entities_heuristic`.
      - If LLM is OFF, run the heuristic directly.

    Stop-list filter (``REMINDER``, ``TODO``, …) is applied uniformly on top
    of either path so noise is excluded regardless of source.
    """
    if not text:
        return []

    use_llm = _llm_enabled()
    if use_llm:
        try:
            from . import curator as _curator  # local import to avoid cycle/cost
        except Exception as e:  # pragma: no cover — defensive
            log.warning("graph.extract_entities: curator import failed: %s", e)
            _curator = None
        if _curator is not None and _curator._have_curator():
            try:
                out = _curator.curator_extract_entities(text)
            except Exception as e:
                log.warning("graph.extract_entities: curator raised %s", e)
                out = None
            if out is not None:
                tuples: list[tuple[str, str]] = []
                seen: set[str] = set()
                for item in out:
                    name = (item.get("name") or "").strip()
                    etype = (item.get("type") or "concept").strip().lower()
                    norm = name.lower()
                    if not name or norm in seen:
                        continue
                    if norm in _ENTITY_STOPLIST:
                        continue
                    seen.add(norm)
                    tuples.append((name, etype))
                return tuples[:20]
            # LLM produced no parseable output → optionally fall through.
            if not _heuristic_fallback_enabled():
                log.warning("graph.extract_entities: LLM failed and fallback disabled")
                return []
            log.info("graph.extract_entities: LLM failed; falling back to heuristic")

    return _extract_entities_heuristic(text)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def upsert_entity(db: MemoirsDB, name: str, etype: str | None = None) -> str:
    norm = _normalize(name)
    etype = etype or _classify_entity_heuristic(name)
    eid = stable_id("ent", norm, etype)
    now = utc_now()
    db.conn.execute(
        """
        INSERT INTO entities (id, name, normalized_name, type, metadata_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, '{}', ?, ?)
        ON CONFLICT(normalized_name, type) DO UPDATE SET
            name = excluded.name,
            updated_at = excluded.updated_at
        """,
        (eid, name, norm, etype, now, now),
    )
    db.conn.commit()
    return eid


def link_memory_to_entities(db: MemoirsDB, memory_id: str, entity_ids: Iterable[str]) -> int:
    n = 0
    for eid in entity_ids:
        cur = db.conn.execute(
            "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
            (memory_id, eid),
        )
        n += cur.rowcount or 0
    db.conn.commit()
    return n


def create_relationship(db: MemoirsDB, source: str, relation: str, target: str, *, confidence: float = 0.6) -> str:
    sid = upsert_entity(db, source)
    tid = upsert_entity(db, target)
    rid = stable_id("rel", sid, relation, tid)
    db.conn.execute(
        """
        INSERT INTO relationships (id, source_entity_id, target_entity_id, relation, confidence, metadata_json, created_at)
        VALUES (?, ?, ?, ?, ?, '{}', ?)
        ON CONFLICT(source_entity_id, relation, target_entity_id) DO UPDATE SET
            confidence = MAX(relationships.confidence, excluded.confidence)
        """,
        (rid, sid, tid, relation, confidence, utc_now()),
    )
    db.conn.commit()
    return rid


# ---------------------------------------------------------------------------
# Per-memory relationships cache (auto-migrated, in the same SQLite DB)
# ---------------------------------------------------------------------------


def _ensure_relationships_cache(db: MemoirsDB) -> None:
    """Create the LLM relationships cache table if it doesn't exist.

    Schema is dead simple: one row per (memory_id, key) where key is a hash
    of the entity-set + content. Stored payload is the JSON triples list.
    """
    db.conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _graph_relationships_cache (
            memory_id   TEXT NOT NULL,
            key         TEXT NOT NULL,
            triples_json TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            PRIMARY KEY (memory_id, key)
        )
        """
    )
    db.conn.commit()


def _cache_key(content: str, entity_names: list[str]) -> str:
    """Stable hash of (content, sorted entity set) for cache lookup."""
    h = hashlib.sha1()
    h.update((content or "").encode("utf-8", errors="ignore"))
    h.update(b"\x00")
    for n in sorted(set(name.lower() for name in entity_names)):
        h.update(n.encode("utf-8", errors="ignore"))
        h.update(b"\x01")
    return h.hexdigest()


_CACHE_TTL_SECONDS = 30 * 24 * 3600  # 30 days


def _cache_get(db: MemoirsDB, memory_id: str, key: str) -> list[dict] | None:
    row = db.conn.execute(
        "SELECT triples_json, created_at FROM _graph_relationships_cache WHERE memory_id = ? AND key = ?",
        (memory_id, key),
    ).fetchone()
    if not row:
        return None
    # TTL check — best-effort; ISO timestamps compare lexicographically only
    # for same offset, so use epoch math via datetime.
    try:
        from datetime import datetime
        created = datetime.fromisoformat(row["created_at"])
        if (datetime.now(created.tzinfo) - created).total_seconds() > _CACHE_TTL_SECONDS:
            return None
    except Exception:
        pass
    try:
        data = json.loads(row["triples_json"])
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _cache_put(db: MemoirsDB, memory_id: str, key: str, triples: list[dict]) -> None:
    db.conn.execute(
        """
        INSERT OR REPLACE INTO _graph_relationships_cache
          (memory_id, key, triples_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (memory_id, key, json.dumps(triples, ensure_ascii=False), utc_now()),
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# Index entities for memories (LLM or heuristic, depending on toggle)
# ---------------------------------------------------------------------------


def index_memory_entities(db: MemoirsDB, *, limit: int = 500) -> dict:
    """Extract entities for memories without entity links and rebuild relationships.

    Picks up to ``limit`` non-archived memories that have no entries in
    ``memory_entities`` yet, runs :func:`extract_entities` on each, persists
    the entities + links, and finally calls :func:`build_relationships` to
    refresh the relationships graph.
    """
    rows = db.conn.execute(
        """
        SELECT m.id, m.content
        FROM memories m
        LEFT JOIN memory_entities me ON me.memory_id = m.id
        WHERE m.archived_at IS NULL AND me.memory_id IS NULL
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    known_projects = {
        r["normalized_name"]
        for r in db.conn.execute("SELECT normalized_name FROM entities WHERE type = 'project'").fetchall()
    }
    new_projects: set[str] = set()
    entities_total = 0
    for r in rows:
        ents = extract_entities(r["content"])
        for name, etype in ents:
            if etype == "project":
                norm = name.strip().lower()
                if norm not in known_projects and norm not in new_projects:
                    new_projects.add(norm)
                    log.info("graph: new project detected → %s", name)
        ids = [upsert_entity(db, name, etype) for name, etype in ents]
        entities_total += link_memory_to_entities(db, r["id"], ids)
    rel_stats = build_relationships(db, min_co_occurrence=2)
    return {
        "memories_processed": len(rows),
        "links_created": entities_total,
        "relationships": rel_stats,
    }


# ---------------------------------------------------------------------------
# Build relationships — LLM-first per memoria, heuristic fallback
# ---------------------------------------------------------------------------


def _build_relationships_heuristic(
    db: MemoirsDB,
    *,
    min_co_occurrence: int = 2,
    limit: int | None = None,
) -> dict:
    """Legacy heuristic relationship builder: co-occurrence + decision_link + uses.

    Kept as a fallback for environments without a curator (and as the test
    contract for older unit tests). Idempotent thanks to the unique constraint
    on (source, relation, target).
    """
    now = utc_now()
    inserted_co = 0
    inserted_decision = 0
    inserted_uses = 0

    # 1) Co-occurrence
    rows = db.conn.execute(
        """
        SELECT a.entity_id AS a_id, b.entity_id AS b_id, COUNT(*) AS n
        FROM memory_entities a
        JOIN memory_entities b ON a.memory_id = b.memory_id AND a.entity_id < b.entity_id
        GROUP BY a.entity_id, b.entity_id
        HAVING n >= ?
        ORDER BY n DESC
        """,
        (min_co_occurrence,),
    ).fetchall()
    if limit:
        rows = rows[:limit]
    for r in rows:
        conf = min(0.95, 0.4 + 0.05 * r["n"])
        rid = stable_id("rel", r["a_id"], "co_occurs_in", r["b_id"])
        cur = db.conn.execute(
            """
            INSERT OR IGNORE INTO relationships
              (id, source_entity_id, target_entity_id, relation, confidence,
               metadata_json, created_at)
            VALUES (?, ?, ?, 'co_occurs_in', ?, ?, ?)
            """,
            (rid, r["a_id"], r["b_id"], conf, json.dumps({"count": r["n"]}), now),
        )
        if cur.rowcount:
            inserted_co += 1

    # 2) Decision link
    decision_rows = db.conn.execute(
        """
        SELECT m.id AS memory_id, GROUP_CONCAT(me.entity_id) AS entity_ids
        FROM memories m
        JOIN memory_entities me ON me.memory_id = m.id
        WHERE m.archived_at IS NULL AND m.type = 'decision'
        GROUP BY m.id
        HAVING COUNT(me.entity_id) >= 2
        """
    ).fetchall()
    for r in decision_rows:
        ents = (r["entity_ids"] or "").split(",")
        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                if a >= b:
                    a, b = b, a
                rid = stable_id("rel", a, "decision_link", b)
                cur = db.conn.execute(
                    """
                    INSERT OR IGNORE INTO relationships
                      (id, source_entity_id, target_entity_id, relation, confidence,
                       metadata_json, created_at)
                    VALUES (?, ?, ?, 'decision_link', 0.7, ?, ?)
                    """,
                    (rid, a, b, json.dumps({"memory_id": r["memory_id"]}), now),
                )
                if cur.rowcount:
                    inserted_decision += 1

    # 3) "uses" via spaCy dep parser
    _USE_LEMMAS = {"use", "usar", "require", "powered", "with", "via", "depend",
                   "rely", "import", "leverage", "integrate"}
    ent_by_name: dict[str, str] = {}
    for er in db.conn.execute("SELECT id, normalized_name FROM entities"):
        ent_by_name[er["normalized_name"]] = er["id"]

    nlp = None
    if ent_by_name:
        try:
            from . import extract_spacy
            if extract_spacy.is_available():
                nlp_en, nlp_es = extract_spacy._load_models()
                nlp = nlp_en or nlp_es
        except Exception as e:
            log.debug("uses-extraction: spaCy unavailable (%s) — skipping", e)

    if nlp:
        for mr in db.conn.execute(
            "SELECT id, content FROM memories WHERE archived_at IS NULL "
            "AND content IS NOT NULL LIMIT 1000"
        ):
            content = mr["content"] or ""
            if len(content) < 12 or len(content) > 1000:
                continue
            try:
                doc = nlp(content)
            except Exception:
                continue
            for token in doc:
                lemma = (token.lemma_ or token.text).lower()
                if lemma not in _USE_LEMMAS:
                    continue
                subj = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
                obj = next((c for c in token.children if c.dep_ in ("dobj", "pobj", "obj")), None)
                if obj is None:
                    for c in token.children:
                        if c.dep_ == "prep":
                            obj = next((g for g in c.children if g.dep_ == "pobj"), None)
                            if obj:
                                break
                if not (subj and obj):
                    continue
                a_norm = subj.text.strip().lower()
                b_norm = obj.text.strip().lower()
                a_id = ent_by_name.get(a_norm)
                b_id = ent_by_name.get(b_norm)
                if not (a_id and b_id) or a_id == b_id:
                    continue
                rid = stable_id("rel", a_id, "uses", b_id)
                cur = db.conn.execute(
                    """
                    INSERT OR IGNORE INTO relationships
                      (id, source_entity_id, target_entity_id, relation, confidence,
                       metadata_json, created_at)
                    VALUES (?, ?, ?, 'uses', 0.6, ?, ?)
                    """,
                    (rid, a_id, b_id, json.dumps({"memory_id": mr["id"]}), now),
                )
                if cur.rowcount:
                    inserted_uses += 1

    db.conn.commit()
    log.info(
        "graph: built %d co_occurs_in + %d decision_link + %d uses relationships (heuristic)",
        inserted_co, inserted_decision, inserted_uses,
    )
    return {
        "co_occurs_in": inserted_co,
        "decision_link": inserted_decision,
        "uses": inserted_uses,
        "total": inserted_co + inserted_decision + inserted_uses,
    }


def _build_relationships_llm(
    db: MemoirsDB,
    *,
    batch_size: int = 50,
    limit: int | None = None,
) -> dict:
    """Build relationships by asking the curator for explicit triples per memoria.

    Walks non-archived memories that have ≥ 2 linked entities, asks the LLM
    for ``(subject, predicate, object)`` triples grounded in the linked
    entity set, and persists them via the standard ``relationships`` table
    (idempotent on the (source, relation, target) UNIQUE).

    Caches per-memory results in ``_graph_relationships_cache`` so re-runs
    don't re-invoke the LLM for unchanged inputs.
    """
    from . import curator as _curator

    if not _curator._have_curator():
        log.warning("graph: LLM relationship builder requested but curator unavailable; "
                    "falling back to heuristic")
        return _build_relationships_heuristic(db)

    _ensure_relationships_cache(db)

    # Pull memories with ≥ 2 linked entities, newest first.
    sql = """
        SELECT m.id, m.content,
               GROUP_CONCAT(e.id, '|||') AS entity_ids,
               GROUP_CONCAT(e.name, '|||') AS entity_names
        FROM memories m
        JOIN memory_entities me ON me.memory_id = m.id
        JOIN entities e ON e.id = me.entity_id
        WHERE m.archived_at IS NULL
        GROUP BY m.id
        HAVING COUNT(me.entity_id) >= 2
        ORDER BY m.updated_at DESC
        LIMIT ?
    """
    rows = db.conn.execute(sql, (batch_size if limit is None else min(batch_size, limit),)).fetchall()

    inserted = 0
    skipped_lt2 = 0
    cache_hits = 0
    llm_calls = 0
    parse_failures = 0
    by_relation: Counter[str] = Counter()
    now = utc_now()
    t0 = time.time()

    for r in rows:
        memory_id = r["id"]
        content = r["content"] or ""
        entity_ids = (r["entity_ids"] or "").split("|||")
        entity_names = (r["entity_names"] or "").split("|||")
        if len(entity_names) < 2:
            skipped_lt2 += 1
            continue

        name_to_id = {n: eid for n, eid in zip(entity_names, entity_ids) if n}
        if len(name_to_id) < 2:
            skipped_lt2 += 1
            continue

        key = _cache_key(content, list(name_to_id.keys()))
        cached = _cache_get(db, memory_id, key)
        if cached is not None:
            cache_hits += 1
            triples = cached
        else:
            try:
                triples = _curator.curator_extract_relationships(content, list(name_to_id.keys()))
            except Exception as e:
                log.warning("graph._llm: curator raised on memory=%s: %s", memory_id[:12], e)
                triples = None
            llm_calls += 1
            if triples is None:
                parse_failures += 1
                # cache the empty result to avoid retrying every batch — TTL
                # will expire it; meanwhile we don't repeatedly slam the LLM.
                _cache_put(db, memory_id, key, [])
                continue
            _cache_put(db, memory_id, key, triples)

        for t in triples:
            subj_name = t.get("subject")
            obj_name = t.get("object")
            pred = t.get("predicate")
            conf = float(t.get("confidence", 0.7))
            sid = name_to_id.get(subj_name)
            oid = name_to_id.get(obj_name)
            if not sid or not oid or sid == oid or not pred:
                continue
            rid = stable_id("rel", sid, pred, oid)
            cur = db.conn.execute(
                """
                INSERT INTO relationships
                  (id, source_entity_id, target_entity_id, relation, confidence,
                   metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_entity_id, relation, target_entity_id) DO UPDATE SET
                    confidence = MAX(relationships.confidence, excluded.confidence)
                """,
                (rid, sid, oid, pred, conf,
                 json.dumps({"memory_id": memory_id, "extractor": "llm"}), now),
            )
            if cur.rowcount:
                inserted += 1
            by_relation[pred] += 1

    db.conn.commit()
    elapsed = time.time() - t0
    log.info(
        "graph: LLM relationships memories=%d inserted=%d cache_hits=%d "
        "llm_calls=%d parse_failures=%d skipped_lt2=%d elapsed=%.2fs",
        len(rows), inserted, cache_hits, llm_calls, parse_failures, skipped_lt2, elapsed,
    )
    return {
        "memories_processed": len(rows),
        "inserted": inserted,
        "cache_hits": cache_hits,
        "llm_calls": llm_calls,
        "parse_failures": parse_failures,
        "skipped_lt2_entities": skipped_lt2,
        "by_relation": dict(by_relation),
        "total": inserted,
        # Legacy keys preserved so callers reading {"co_occurs_in", "decision_link", "uses"}
        # don't KeyError. They're zero on the LLM path because we no longer emit those.
        "co_occurs_in": 0,
        "decision_link": 0,
        "uses": 0,
        "elapsed_seconds": round(elapsed, 3),
    }


def build_relationships(
    db: MemoirsDB,
    *,
    min_co_occurrence: int = 2,
    limit: int | None = None,
    batch_size: int = 50,
) -> dict:
    """Generate edges in the ``relationships`` table.

    Default (``MEMOIRS_GRAPH_LLM=on``): asks the curator for explicit triples
    per memoria, grounded in its linked entity set. Cap ``batch_size`` memories
    per call. Cached in ``_graph_relationships_cache``.

    Heuristic mode (``MEMOIRS_GRAPH_LLM=off``): co-occurrence (≥
    ``min_co_occurrence`` shared memorias) + decision_link + spaCy "uses".
    Used by tests that pin the legacy contract.

    Idempotent thanks to the (source, relation, target) UNIQUE constraint.
    """
    if _llm_enabled():
        # When the LLM toggle is on but the curator isn't actually installed
        # (e.g. CI), fall back to heuristic. _build_relationships_llm itself
        # also handles this — but checking here keeps the legacy keys clean.
        try:
            from . import curator as _curator
            have = _curator._have_curator()
        except Exception:
            have = False
        if have:
            return _build_relationships_llm(db, batch_size=batch_size, limit=limit)
        log.info("graph: MEMOIRS_GRAPH_LLM=on but curator unavailable; using heuristic")
    return _build_relationships_heuristic(
        db, min_co_occurrence=min_co_occurrence, limit=limit,
    )


# ---------------------------------------------------------------------------
# Project inference (untouched — purely metadata-driven, no NLP heuristic)
# ---------------------------------------------------------------------------


def refresh_projects_from_conversations(db: MemoirsDB) -> dict:
    """Derive project entities from the conversations corpus.

    Sources scanned (in order of authority):
      1. ``conversations.metadata.cwd`` (Claude Code transcripts carry the real
         working directory — ``/home/misael/Desktop/projects/<name>`` → project).
      2. ``conversations.metadata.project_dir`` (encoded form for Claude Code).
      3. ``memories WHERE type='project'`` (Gemma-extracted).

    Each unique project gets an entity with type='project', and every memory
    whose source conversation belongs to that project gets linked via
    ``memory_entities``. Idempotent.
    """
    now = utc_now()
    project_to_convs: dict[str, set[str]] = {}

    rows = db.conn.execute(
        "SELECT id, metadata_json, source_id FROM conversations"
    ).fetchall()
    for r in rows:
        try:
            md = json.loads(r["metadata_json"] or "{}")
        except json.JSONDecodeError:
            continue

        cwd = md.get("cwd")
        project_name = None
        if cwd:
            parts = [p for p in cwd.split("/") if p]
            for marker in ("projects", "Desktop", "src", "code"):
                if marker in parts:
                    idx = parts.index(marker)
                    if idx + 1 < len(parts):
                        project_name = parts[idx + 1]
                        break
            if not project_name and parts:
                project_name = parts[-1]
        elif md.get("project_dir"):
            pd = str(md["project_dir"])
            segs = [s for s in pd.split("-") if s]
            for marker in ("projects", "Desktop", "src", "code"):
                if marker in segs:
                    idx = segs.index(marker)
                    if idx + 1 < len(segs):
                        project_name = "-".join(segs[idx + 1:])
                        break
            if not project_name and segs:
                project_name = segs[-1]

        if project_name and len(project_name) > 1 and not project_name.startswith("."):
            project_to_convs.setdefault(project_name.lower(), set()).add(r["id"])

    proj_mem_rows = db.conn.execute(
        "SELECT id, content FROM memories "
        "WHERE archived_at IS NULL AND type = 'project'"
    ).fetchall()
    for m in proj_mem_rows:
        content = (m["content"] or "").strip()
        if not content:
            continue
        first_word = content.split()[0].strip(".,;:'\"") if content.split() else ""
        if first_word and len(first_word) > 2:
            project_to_convs.setdefault(first_word.lower(), set())

    created = 0
    linked = 0
    for project_name, conv_ids in project_to_convs.items():
        eid = upsert_entity(db, project_name, "project")
        db.conn.execute(
            "UPDATE entities SET type = 'project', updated_at = ? WHERE id = ?",
            (now, eid),
        )
        created += 1
        if conv_ids:
            ph = ",".join("?" * len(conv_ids))
            mem_rows = db.conn.execute(
                f"SELECT DISTINCT mc.promoted_memory_id FROM memory_candidates mc "
                f"WHERE mc.conversation_id IN ({ph}) "
                f"AND mc.promoted_memory_id IS NOT NULL",
                list(conv_ids),
            ).fetchall()
            for mr in mem_rows:
                cur = db.conn.execute(
                    "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) "
                    "VALUES (?, ?)",
                    (mr["promoted_memory_id"], eid),
                )
                linked += cur.rowcount or 0
    db.conn.commit()
    log.info("projects refresh: %d entities, %d memory→project links", created, linked)
    return {"projects": created, "memory_links": linked,
            "names": sorted(project_to_convs.keys())}


def get_project_context(db: MemoirsDB, project_name: str, *, limit: int = 20) -> dict:
    """Return memories + related entities for a given project name."""
    norm = _normalize(project_name)
    proj = db.conn.execute(
        "SELECT id, name FROM entities WHERE normalized_name = ? AND type = 'project' LIMIT 1",
        (norm,),
    ).fetchone()
    if not proj:
        return {"project": project_name, "memories": [], "related_entities": []}

    memories = [
        dict(r)
        for r in db.conn.execute(
            """
            SELECT m.id, m.type, m.content, m.score
            FROM memories m
            JOIN memory_entities me ON me.memory_id = m.id
            WHERE me.entity_id = ? AND m.archived_at IS NULL
            ORDER BY m.score DESC
            LIMIT ?
            """,
            (proj["id"], limit),
        ).fetchall()
    ]
    related = [
        dict(r)
        for r in db.conn.execute(
            """
            SELECT DISTINCT e.id, e.name, e.type, r.relation
            FROM relationships r
            JOIN entities e ON e.id = r.target_entity_id
            WHERE r.source_entity_id = ?
            UNION
            SELECT DISTINCT e.id, e.name, e.type, r.relation
            FROM relationships r
            JOIN entities e ON e.id = r.source_entity_id
            WHERE r.target_entity_id = ?
            """,
            (proj["id"], proj["id"]),
        ).fetchall()
    ]
    return {"project": proj["name"], "memories": memories, "related_entities": related}
