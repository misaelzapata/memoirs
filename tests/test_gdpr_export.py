"""Tests for memoirs.export — GDPR portable export + import (P3-6)."""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import time
import zipfile
from pathlib import Path

import pytest

from memoirs.core.ids import content_hash, stable_id, utc_now
from memoirs.db import MemoirsDB
from memoirs.export import (
    SCHEMA_VERSION,
    anonymize_memory,
    export_user_data,
    import_user_data,
    verify_bundle,
)


# ---------------------------------------------------------------------------
# Synthetic DB seeding helpers
# ---------------------------------------------------------------------------


def _seed_corpus(db: MemoirsDB, *, n_memories: int = 20, with_embeddings: bool = True,
                 with_email: bool = False) -> dict:
    """Seed a small synthetic corpus the export module can dump.

    Returns a dict of the inserted ids so tests can assert against them.
    """
    now = utc_now()
    conn = db.conn
    # ---- source + conversation ------------------------------------------
    src_uri = "test://gdpr"
    cur = conn.execute(
        "INSERT INTO sources (uri, kind, name, content_hash, mtime_ns, "
        "size_bytes, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (src_uri, "test", "gdpr-fixture", "abc", 0, 0, now, now),
    )
    src_id = int(cur.lastrowid)
    conv_id = stable_id("conv", src_uri, "gdpr-conv-1")
    conn.execute(
        "INSERT INTO conversations (id, source_id, external_id, title, "
        "created_at, updated_at, message_count, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (conv_id, src_id, "gdpr-conv-1", "GDPR Test", now, now, 2, "{}"),
    )

    # Two messages — optionally embedding an email so the redact tests can
    # assert the placeholder substitution.
    msg_text = "hello world"
    if with_email:
        msg_text = "ping me at alice@example.com to discuss the deal"
    for ord_ in (0, 1):
        mid = stable_id("msg", conv_id, ord_)
        conn.execute(
            "INSERT INTO messages (id, conversation_id, external_id, role, "
            "content, ordinal, created_at, content_hash, raw_json, "
            "metadata_json, is_active, first_seen_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
            (mid, conv_id, f"m{ord_}", "user", msg_text, ord_, now,
             content_hash(msg_text), "{}", "{}", now, now),
        )

    # ---- memories + embeddings ------------------------------------------
    memory_ids: list[str] = []
    for i in range(n_memories):
        mid = stable_id("mem", "gdpr", i)
        text = f"memory item #{i}"
        if i == 0 and with_email:
            text = "user prefers contact via alice@example.com"
        conn.execute(
            "INSERT INTO memories (id, type, content, content_hash, "
            "importance, confidence, score, usage_count, valid_from, "
            "metadata_json, created_at, updated_at) "
            "VALUES (?, 'fact', ?, ?, 3, 0.9, 0.5, 0, ?, '{}', ?, ?)",
            (mid, text, content_hash(text + mid), now, now, now),
        )
        memory_ids.append(mid)

        if with_embeddings:
            dim = 4
            vec = [float(i + 1), float(i + 2), 0.5, -0.5]
            blob = struct.pack(f"{dim}f", *vec)
            conn.execute(
                "INSERT INTO memory_embeddings (memory_id, dim, embedding, "
                "model, created_at) VALUES (?, ?, ?, ?, ?)",
                (mid, dim, blob, "test-model", now),
            )

    # ---- entity + memory_entities + relationships ----------------------
    ent_id = stable_id("ent", "alice")
    conn.execute(
        "INSERT INTO entities (id, name, normalized_name, type, "
        "metadata_json, created_at, updated_at) "
        "VALUES (?, 'alice', 'alice', 'person', '{}', ?, ?)",
        (ent_id, now, now),
    )
    conn.execute(
        "INSERT INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
        (memory_ids[0], ent_id),
    )
    ent_id2 = stable_id("ent", "bob")
    conn.execute(
        "INSERT INTO entities (id, name, normalized_name, type, "
        "metadata_json, created_at, updated_at) "
        "VALUES (?, 'bob', 'bob', 'person', '{}', ?, ?)",
        (ent_id2, now, now),
    )
    rel_id = stable_id("rel", ent_id, ent_id2)
    conn.execute(
        "INSERT INTO relationships (id, source_entity_id, target_entity_id, "
        "relation, confidence, metadata_json, created_at) "
        "VALUES (?, ?, ?, 'knows', 0.8, '{}', ?)",
        (rel_id, ent_id, ent_id2, now),
    )

    # ---- memory_links ---------------------------------------------------
    if len(memory_ids) >= 2:
        conn.execute(
            "INSERT INTO memory_links (source_memory_id, target_memory_id, "
            "similarity, reason, created_at) VALUES (?, ?, ?, 'semantic', ?)",
            (memory_ids[0], memory_ids[1], 0.92, now),
        )

    # ---- candidate row pointing to a promoted memory (provenance) -----
    cand_id = stable_id("cand", memory_ids[0])
    msg_id_ref = stable_id("msg", conv_id, 0)
    conn.execute(
        "INSERT INTO memory_candidates (id, conversation_id, "
        "source_message_ids, type, content, importance, confidence, "
        "entities, status, raw_json, created_at, updated_at, "
        "promoted_memory_id) VALUES (?, ?, ?, 'fact', 'p', 3, 0.5, '[]', "
        "'accepted', '{}', ?, ?, ?)",
        (cand_id, conv_id, json.dumps([msg_id_ref]), now, now, memory_ids[0]),
    )

    conn.commit()
    return {
        "memory_ids": memory_ids,
        "conv_id": conv_id,
        "source_uri": src_uri,
        "entity_ids": [ent_id, ent_id2],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_export_creates_valid_zip(tmp_db, tmp_path):
    _seed_corpus(tmp_db, n_memories=20)
    out = tmp_path / "bundle.zip"
    manifest = export_user_data(tmp_db, out_path=out)

    assert out.exists()
    assert zipfile.is_zipfile(out)
    assert manifest.schema_version == SCHEMA_VERSION
    # Manifest is also embedded inside the zip.
    with zipfile.ZipFile(out, "r") as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "memories.jsonl" in names
        assert "embeddings.npz" in names


def test_manifest_counts_match_db(tmp_db, tmp_path):
    seed = _seed_corpus(tmp_db, n_memories=20)
    out = tmp_path / "bundle.zip"
    manifest = export_user_data(tmp_db, out_path=out)

    assert manifest.counts["memories"] == 20
    assert manifest.counts["embeddings"] == 20
    assert manifest.counts["conversations"] == 1
    assert manifest.counts["messages"] == 2
    assert manifest.counts["entities"] == 2
    assert manifest.counts["relationships"] == 1
    assert manifest.counts["memory_links"] == 1
    assert manifest.counts["provenance"] >= 1


def test_manifest_hashes_match_payload(tmp_db, tmp_path):
    _seed_corpus(tmp_db, n_memories=10)
    out = tmp_path / "bundle.zip"
    manifest = export_user_data(tmp_db, out_path=out)

    with zipfile.ZipFile(out, "r") as zf:
        for name, meta in manifest.files.items():
            data = zf.read(name)
            assert hashlib.sha256(data).hexdigest() == meta["sha256"], name
            assert len(data) == meta["size"], name


def test_import_merge_round_trip(tmp_db, tmp_path):
    seed = _seed_corpus(tmp_db, n_memories=15)
    out = tmp_path / "bundle.zip"
    export_user_data(tmp_db, out_path=out)

    # Re-hydrate into a brand new DB.
    target = MemoirsDB(tmp_path / "target.sqlite")
    target.init()
    try:
        report = import_user_data(target, in_path=out, mode="merge")
        assert report.inserted["memories"] == 15
        assert report.inserted["embeddings"] == 15
        assert report.inserted["conversations"] == 1
        assert report.inserted["messages"] == 2
        # Verify a couple of memories actually landed.
        n = target.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert n == 15
        # The first memory's content should be preserved verbatim.
        row = target.conn.execute(
            "SELECT content FROM memories WHERE id = ?", (seed["memory_ids"][0],)
        ).fetchone()
        assert row is not None
    finally:
        target.close()


def test_import_replace_wipes_previous(tmp_db, tmp_path):
    _seed_corpus(tmp_db, n_memories=10)
    out = tmp_path / "bundle.zip"
    export_user_data(tmp_db, out_path=out)

    # Build target DB with DIFFERENT pre-existing memorias.
    target = MemoirsDB(tmp_path / "target.sqlite")
    target.init()
    try:
        now = utc_now()
        for i in range(5):
            target.conn.execute(
                "INSERT INTO memories (id, type, content, content_hash, "
                "valid_from, metadata_json, created_at, updated_at) "
                "VALUES (?, 'fact', ?, ?, ?, '{}', ?, ?)",
                (f"pre-{i}", f"old memory {i}", f"hash-{i}", now, now, now),
            )
        target.conn.commit()
        assert target.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 5

        report = import_user_data(target, in_path=out, mode="replace")
        # Pre-existing rows should be gone, replaced by the bundle's 10.
        n = target.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert n == 10
        assert report.deleted.get("memories", 0) >= 5
    finally:
        target.close()


def test_import_new_user_keeps_ids(tmp_db, tmp_path):
    """new_user mode should preserve original IDs (no user_id column → falls
    back to merge semantics)."""
    seed = _seed_corpus(tmp_db, n_memories=5)
    out = tmp_path / "bundle.zip"
    export_user_data(tmp_db, out_path=out)

    target = MemoirsDB(tmp_path / "target.sqlite")
    target.init()
    try:
        report = import_user_data(
            target, in_path=out, mode="new_user", new_user_id="alice",
        )
        assert report.user_id_target == "alice"
        # IDs preserved
        for mid in seed["memory_ids"]:
            row = target.conn.execute(
                "SELECT id FROM memories WHERE id = ?", (mid,)
            ).fetchone()
            assert row is not None, mid
    finally:
        target.close()


def test_redact_pii_replaces_email(tmp_db, tmp_path):
    _seed_corpus(tmp_db, n_memories=3, with_email=True)
    out = tmp_path / "bundle.zip"
    export_user_data(tmp_db, out_path=out, redact_pii=True)

    with zipfile.ZipFile(out, "r") as zf:
        mem_text = zf.read("memories.jsonl").decode("utf-8")
        assert "alice@example.com" not in mem_text
        assert "[EMAIL_" in mem_text
        # Conversation messages also redacted.
        conv_files = [n for n in zf.namelist() if n.startswith("conversations/")]
        assert conv_files
        body = zf.read(conv_files[0]).decode("utf-8")
        assert "alice@example.com" not in body


def test_round_trip_preserves_content(tmp_db, tmp_path):
    seed = _seed_corpus(tmp_db, n_memories=10)
    out = tmp_path / "bundle.zip"
    export_user_data(tmp_db, out_path=out)

    target = MemoirsDB(tmp_path / "target.sqlite")
    target.init()
    try:
        import_user_data(target, in_path=out, mode="merge")
        for mid in seed["memory_ids"]:
            src = tmp_db.conn.execute(
                "SELECT content, content_hash, importance FROM memories WHERE id = ?",
                (mid,),
            ).fetchone()
            dst = target.conn.execute(
                "SELECT content, content_hash, importance FROM memories WHERE id = ?",
                (mid,),
            ).fetchone()
            assert dst is not None
            assert dst["content"] == src["content"]
            assert dst["content_hash"] == src["content_hash"]
            assert dst["importance"] == src["importance"]
    finally:
        target.close()


def test_verify_rejects_tampered_bundle(tmp_db, tmp_path):
    _seed_corpus(tmp_db, n_memories=5)
    out = tmp_path / "bundle.zip"
    export_user_data(tmp_db, out_path=out)

    # Tamper: rebuild the zip with a flipped memories.jsonl byte.
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(out, "r") as src, zipfile.ZipFile(tampered, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            if name == "memories.jsonl" and data:
                data = data + b"\n# tampered\n"
            dst.writestr(name, data)

    verdict = verify_bundle(tampered)
    assert verdict["ok"] is False
    assert "memories.jsonl" in verdict["mismatches"] or any(
        m.startswith("memories.jsonl") for m in verdict["mismatches"]
    )


def test_verify_passes_clean_bundle(tmp_db, tmp_path):
    _seed_corpus(tmp_db, n_memories=4)
    out = tmp_path / "bundle.zip"
    export_user_data(tmp_db, out_path=out)
    verdict = verify_bundle(out)
    assert verdict["ok"] is True
    assert verdict["missing"] == []
    assert verdict["mismatches"] == []


def test_export_performance_1000_memories(tmp_db, tmp_path):
    """1000 memorias should export in < 5 s on any reasonable machine."""
    _seed_corpus(tmp_db, n_memories=1000)
    out = tmp_path / "bundle.zip"
    t0 = time.perf_counter()
    manifest = export_user_data(tmp_db, out_path=out, include_embeddings=True)
    elapsed = time.perf_counter() - t0
    assert manifest.counts["memories"] == 1000
    assert elapsed < 5.0, f"export took {elapsed:.2f}s (>5s)"


def test_anonymize_memory_redacts_strings():
    raw = {
        "id": "mem-1",
        "content": "ping alice@example.com please",
        "importance": 3,
    }
    out = anonymize_memory(raw)
    assert "alice@example.com" not in out["content"]
    assert out["importance"] == 3
    assert out["id"] == "mem-1"


def test_no_embeddings_flag(tmp_db, tmp_path):
    _seed_corpus(tmp_db, n_memories=5)
    out = tmp_path / "bundle.zip"
    manifest = export_user_data(tmp_db, out_path=out, include_embeddings=False)
    assert manifest.counts.get("embeddings", 0) == 0
    with zipfile.ZipFile(out, "r") as zf:
        # embeddings.npz should not be in the bundle when omitted.
        assert "embeddings.npz" not in zf.namelist()


def test_import_skips_duplicates_in_merge(tmp_db, tmp_path):
    _seed_corpus(tmp_db, n_memories=8)
    out = tmp_path / "bundle.zip"
    export_user_data(tmp_db, out_path=out)

    # Re-hydrate twice. Second run should report skipped > 0 since the rows
    # already exist (INSERT OR IGNORE).
    target = MemoirsDB(tmp_path / "target.sqlite")
    target.init()
    try:
        first = import_user_data(target, in_path=out, mode="merge")
        second = import_user_data(target, in_path=out, mode="merge")
        assert first.inserted["memories"] == 8
        assert second.inserted["memories"] == 0
        assert second.skipped["memories"] == 8
    finally:
        target.close()


def test_mcp_tool_round_trip(tmp_db, tmp_path):
    """mcp_export_user_data + mcp_import_user_data over the b64 surface."""
    from memoirs.mcp.tools import call_tool

    _seed_corpus(tmp_db, n_memories=6)
    payload = call_tool(tmp_db, "mcp_export_user_data", {})
    assert payload["ok"] is True
    assert payload["size_bytes"] > 0
    blob_b64 = payload["bundle_b64"]
    decoded = base64.b64decode(blob_b64)
    assert zipfile.is_zipfile(__import__("io").BytesIO(decoded))

    # Import into a fresh DB via the MCP surface.
    target = MemoirsDB(tmp_path / "mcp-target.sqlite")
    target.init()
    try:
        report = call_tool(
            target, "mcp_import_user_data",
            {"bundle_b64": blob_b64, "mode": "merge"},
        )
        assert report["ok"] is True
        assert report["inserted"]["memories"] == 6
    finally:
        target.close()
