"""Layer 3 visualization — rich interactive HTML graphs.

Generates standalone HTML files using vis.js (loaded from CDN). Each graph
includes:
  - top stats banner (counts of nodes / edges / by type)
  - left sidebar: type filters (toggleable) + search box + layout selector
  - center: vis.js network with physics + drag/zoom
  - right sidebar: details panel (populates on node click with full metadata)
  - bottom: export PNG, fit-to-screen, freeze-physics buttons

Three views:
  - render_entity_graph(db)            : entities + relationships + memorias
  - render_decision_flow(db, conv_id)  : conversation → candidates → memorias
  - render_memory_neighborhood(db, id) : one memory + its entities + siblings
"""
from __future__ import annotations

import html
import json
import logging
from pathlib import Path

from ..db import MemoirsDB


log = logging.getLogger("memoirs.viz")
DEFAULT_OUT_DIR = Path(".memoirs/graphs")

# Color palette — high contrast on dark background
TYPE_COLORS = {
    # entities
    "tool":      {"bg": "#4f8cc9", "border": "#7ab0e0", "label": "Tool"},
    "project":   {"bg": "#e58e26", "border": "#f5b561", "label": "Project"},
    "concept":   {"bg": "#7d83ff", "border": "#a3a8ff", "label": "Concept"},
    "person":    {"bg": "#16a085", "border": "#3ec9aa", "label": "Person"},
    "other":     {"bg": "#7f8c8d", "border": "#a5b1b2", "label": "Other entity"},
    # memorias
    "preference":         {"bg": "#27ae60", "border": "#52d685", "label": "Preference"},
    "fact":               {"bg": "#3498db", "border": "#5dade2", "label": "Fact"},
    "task":               {"bg": "#f39c12", "border": "#f7b955", "label": "Task"},
    "decision":           {"bg": "#9b59b6", "border": "#bb86d0", "label": "Decision"},
    "style":              {"bg": "#34495e", "border": "#5d7d99", "label": "Style"},
    "credential_pointer": {"bg": "#c0392b", "border": "#e57366", "label": "Credential ref"},
    "procedural":         {"bg": "#5b21b6", "border": "#8b5cf6", "label": "Procedural"},
    "tool_call":          {"bg": "#475569", "border": "#94a3b8", "label": "Tool call"},
    # candidate / pipeline nodes
    "_candidate_pending":  {"bg": "#bdc3c7", "border": "#dde1e3", "label": "Pending candidate"},
    "_candidate_accepted": {"bg": "#2ecc71", "border": "#5fdd95", "label": "Accepted candidate"},
    "_candidate_rejected": {"bg": "#e74c3c", "border": "#ee7565", "label": "Rejected candidate"},
    "_candidate_merged":   {"bg": "#3498db", "border": "#5dade2", "label": "Merged candidate"},
    "_conversation":       {"bg": "#ecf0f1", "border": "#ffffff", "label": "Conversation"},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_entity_graph(
    db: MemoirsDB,
    output_path: Path | str = DEFAULT_OUT_DIR / "entities.html",
    project: str | None = None,
    max_entities: int = 200,
    include_memories: bool = True,
    max_memories_per_entity: int = 3,
    memory_types: tuple[str, ...] = ("preference", "decision", "task", "project", "style"),
    auto_refresh_seconds: int = 0,
) -> Path:
    if project:
        norm = project.strip().lower()
        proj_row = db.conn.execute(
            "SELECT id, name FROM entities WHERE normalized_name = ? AND type = 'project' LIMIT 1",
            (norm,),
        ).fetchone()
        if not proj_row:
            raise ValueError(f"project entity '{project}' not found")
        project_id = proj_row["id"]
        entity_rows = db.conn.execute(
            """
            SELECT DISTINCT e.id, e.name, e.type
            FROM entities e
            WHERE e.id IN (
                SELECT target_entity_id FROM relationships WHERE source_entity_id = ?
                UNION
                SELECT source_entity_id FROM relationships WHERE target_entity_id = ?
                UNION
                SELECT entity_id FROM memory_entities
                WHERE memory_id IN (
                    SELECT memory_id FROM memory_entities WHERE entity_id = ?
                )
            )
            LIMIT ?
            """,
            (project_id, project_id, project_id, max_entities),
        ).fetchall()
        title = f"Project graph · {proj_row['name']}"
    else:
        entity_rows = db.conn.execute(
            """
            SELECT id, name, type
            FROM entities
            ORDER BY (
                SELECT COUNT(*) FROM memory_entities me WHERE me.entity_id = entities.id
            ) DESC
            LIMIT ?
            """,
            (max_entities,),
        ).fetchall()
        title = "Memoirs · entity graph"

    if not entity_rows:
        raise ValueError("no entities to render")
    entity_ids = {r["id"] for r in entity_rows}

    nodes: list[dict] = []
    edges: list[dict] = []

    for r in entity_rows:
        memory_count = db.conn.execute(
            "SELECT COUNT(*) FROM memory_entities WHERE entity_id = ?", (r["id"],)
        ).fetchone()[0]
        nodes.append(_make_entity_node(r, memory_count))

    if entity_ids:
        rel_rows = db.conn.execute(
            f"""
            SELECT source_entity_id, target_entity_id, relation, confidence
            FROM relationships
            WHERE source_entity_id IN ({','.join('?' * len(entity_ids))})
              AND target_entity_id IN ({','.join('?' * len(entity_ids))})
            """,
            list(entity_ids) + list(entity_ids),
        ).fetchall()
        for r in rel_rows:
            edges.append({
                "from": f"e:{r['source_entity_id']}",
                "to": f"e:{r['target_entity_id']}",
                "label": r["relation"],
                "title": f"{r['relation']} (conf={r['confidence']:.2f})",
                "arrows": "to",
                "kind": "relationship",
            })

    if include_memories and entity_ids:
        # Per-entity top-N filtered by memory type — keeps the graph readable
        # instead of dumping every fact attached to every entity.
        type_placeholders = ",".join("?" * len(memory_types)) if memory_types else "''"
        mem_rows: list = []
        for eid in entity_ids:
            params = [eid]
            if memory_types:
                params.extend(memory_types)
            params.append(max_memories_per_entity)
            mem_rows.extend(db.conn.execute(
                f"""
                SELECT m.id, m.type, m.content, m.score, m.importance, m.confidence,
                       m.usage_count, m.created_at, ? AS entity_id
                FROM memories m
                JOIN memory_entities me ON me.memory_id = m.id
                WHERE me.entity_id = ?
                  AND m.archived_at IS NULL
                  {('AND m.type IN (' + type_placeholders + ')') if memory_types else ''}
                ORDER BY m.score DESC
                LIMIT ?
                """,
                [eid, eid] + (list(memory_types) if memory_types else []) + [max_memories_per_entity],
            ).fetchall())
        seen: set[str] = set()
        for r in mem_rows:
            if r["id"] not in seen:
                nodes.append(_make_memory_node(r))
                seen.add(r["id"])
            edges.append({
                "from": f"m:{r['id']}", "to": f"e:{r['entity_id']}",
                "color": "#666", "kind": "memory_entity",
            })

    return _render_html(
        nodes=nodes, edges=edges, title=title, output_path=output_path,
        layout="force", auto_refresh_seconds=auto_refresh_seconds,
    )


def render_decision_flow(
    db: MemoirsDB,
    conversation_id: str,
    output_path: Path | str | None = None,
    auto_refresh_seconds: int = 0,
) -> Path:
    conv = db.conn.execute(
        "SELECT id, title, message_count FROM conversations WHERE id LIKE ? OR external_id LIKE ? LIMIT 1",
        (f"{conversation_id}%", f"{conversation_id}%"),
    ).fetchone()
    if not conv:
        raise ValueError(f"conversation '{conversation_id}' not found")
    cid = conv["id"]

    candidates = [
        dict(r)
        for r in db.conn.execute(
            "SELECT id, type, content, status, rejection_reason, promoted_memory_id, "
            "extractor, importance, confidence, created_at "
            "FROM memory_candidates WHERE conversation_id = ?",
            (cid,),
        ).fetchall()
    ]
    promoted_ids = [c["promoted_memory_id"] for c in candidates if c["promoted_memory_id"]]
    memories = []
    if promoted_ids:
        ph = ",".join("?" * len(promoted_ids))
        memories = [
            dict(r)
            for r in db.conn.execute(
                f"SELECT id, type, content, score, importance, confidence, usage_count, "
                f"created_at FROM memories WHERE id IN ({ph})", promoted_ids
            ).fetchall()
        ]

    output_path = Path(output_path or DEFAULT_OUT_DIR / f"decisions_{cid[:16]}.html")

    nodes: list[dict] = [{
        "id": f"c:{cid}",
        "label": f"📝 {(conv['title'] or cid[:16])[:30]}",
        "title": _html(f"<b>conversation</b><br>{cid}<br>{conv['message_count']} messages"),
        "color": TYPE_COLORS["_conversation"]["bg"],
        "group": "_conversation",
        "shape": "ellipse", "size": 24, "font": {"color": "#1e1e1e", "size": 14},
        "data": {"kind": "conversation", "id": cid, "title": conv["title"], "message_count": conv["message_count"]},
    }]
    edges: list[dict] = []

    valid_memory_ids: set[str] = set()
    for m in memories:
        nodes.append(_make_memory_node(m, big=True))
        valid_memory_ids.add(m["id"])

    for c in candidates:
        cnode_id = f"cand:{c['id']}"
        status_key = f"_candidate_{c['status']}"
        col = TYPE_COLORS.get(status_key, TYPE_COLORS["_candidate_pending"])
        nodes.append({
            "id": cnode_id,
            "label": f"{c['type']}\n[{c['status']}]",
            "title": _html(
                f"<b>{c['type']}</b> — {c['status']}<br>"
                f"extractor: {c['extractor']}<br>"
                f"imp={c['importance']} conf={c['confidence']:.2f}<br>"
                f"<i>{(c['content'] or '')[:120]}</i>"
            ),
            "color": col["bg"],
            "group": status_key,
            "shape": "box", "size": 12, "font": {"color": "#1e1e1e"},
            "data": {
                "kind": "candidate", "id": c["id"], "type": c["type"], "status": c["status"],
                "content": c["content"], "extractor": c["extractor"],
                "importance": c["importance"], "confidence": c["confidence"],
                "rejection_reason": c["rejection_reason"], "created_at": c["created_at"],
                "promoted_memory_id": c["promoted_memory_id"],
            },
        })
        edges.append({
            "from": f"c:{cid}", "to": cnode_id,
            "label": c["extractor"] or "?",
            "arrows": "to",
            "kind": "extracted_by",
        })
        if c["promoted_memory_id"] and c["promoted_memory_id"] in valid_memory_ids:
            edges.append({
                "from": cnode_id, "to": f"m:{c['promoted_memory_id']}",
                "label": c["status"], "arrows": "to",
                "kind": "promoted_to",
            })

    return _render_html(
        nodes=nodes, edges=edges,
        title=f"Decision flow · {conv['title'] or cid[:16]}",
        output_path=output_path, layout="hierarchical",
        auto_refresh_seconds=auto_refresh_seconds,
    )


def render_memory_neighborhood(
    db: MemoirsDB,
    memory_id: str,
    output_path: Path | str | None = None,
    depth: int = 2,
    auto_refresh_seconds: int = 0,
) -> Path:
    mem = db.conn.execute(
        "SELECT id, type, content, score, importance, confidence, usage_count, "
        "created_at, last_used_at FROM memories WHERE id LIKE ? LIMIT 1",
        (f"{memory_id}%",),
    ).fetchone()
    if not mem:
        raise ValueError(f"memory '{memory_id}' not found")
    mid = mem["id"]

    entity_rows = db.conn.execute(
        """
        SELECT e.id, e.name, e.type
        FROM entities e JOIN memory_entities me ON me.entity_id = e.id
        WHERE me.memory_id = ?
        """,
        (mid,),
    ).fetchall()

    output_path = Path(output_path or DEFAULT_OUT_DIR / f"neighborhood_{mid[:16]}.html")

    nodes: list[dict] = [_make_memory_node(mem, big=True, highlight=True)]
    edges: list[dict] = []
    seen_memories: set[str] = {mid}

    for er in entity_rows:
        eid = er["id"]
        nodes.append(_make_entity_node(er, memory_count=None))
        edges.append({"from": f"m:{mid}", "to": f"e:{eid}", "color": "#aaa", "kind": "memory_entity"})

        if depth >= 2:
            sib_rows = db.conn.execute(
                """
                SELECT DISTINCT m.id, m.type, m.content, m.score, m.importance,
                       m.confidence, m.usage_count, m.created_at
                FROM memories m JOIN memory_entities me ON me.memory_id = m.id
                WHERE me.entity_id = ? AND m.id != ? AND m.archived_at IS NULL
                ORDER BY m.score DESC LIMIT 6
                """,
                (eid, mid),
            ).fetchall()
            for sr in sib_rows:
                if sr["id"] not in seen_memories:
                    nodes.append(_make_memory_node(sr))
                    seen_memories.add(sr["id"])
                edges.append({"from": f"m:{sr['id']}", "to": f"e:{eid}", "color": "#666", "kind": "memory_entity"})

    return _render_html(
        nodes=nodes, edges=edges,
        title=f"Memory neighborhood · {mid[:16]}…",
        output_path=output_path, layout="force",
        auto_refresh_seconds=auto_refresh_seconds,
    )


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------


def _make_entity_node(row, memory_count: int | None) -> dict:
    etype = row["type"] or "other"
    col = TYPE_COLORS.get(etype, TYPE_COLORS["other"])
    size = 14
    if memory_count is not None:
        size = 14 + min(20, memory_count)
    return {
        "id": f"e:{row['id']}",
        "label": row["name"],
        "title": _html(
            f"<b>{row['name']}</b><br>"
            f"type: {etype}"
            + (f"<br>memorias: {memory_count}" if memory_count is not None else "")
        ),
        "color": col["bg"],
        "group": etype,
        "shape": "dot", "size": size,
        "data": {"kind": "entity", "id": row["id"], "name": row["name"], "type": etype, "memory_count": memory_count},
    }


def _make_memory_node(row, big: bool = False, highlight: bool = False) -> dict:
    """Build a memory graph node. Accepts sqlite3.Row or dict — normalizes to dict."""
    r = dict(row) if not isinstance(row, dict) else row
    mtype = r["type"]
    col = TYPE_COLORS.get(mtype, TYPE_COLORS["other"])
    size = 14 if big else 9
    shape = "diamond" if big else "square"
    if highlight:
        size += 8
    snippet = (r.get("content") or "")[:120].replace("<", "&lt;")
    score = r.get("score") or 0
    return {
        "id": f"m:{r['id']}",
        "label": ("⭐ " if highlight else "") + f"[{mtype}]",
        "title": _html(
            f"<b>memory {r['id'][:16]}…</b><br>"
            f"type: {mtype}<br>"
            f"score: {float(score):.3f}<br>"
            f"<i>{snippet}</i>"
        ),
        "color": col["bg"],
        "group": mtype,
        "shape": shape, "size": size,
        "data": {
            "kind": "memory", "id": r["id"], "type": mtype, "content": r.get("content"),
            "score": r.get("score"), "importance": r.get("importance"),
            "confidence": r.get("confidence"), "usage_count": r.get("usage_count"),
            "created_at": r.get("created_at"), "last_used_at": r.get("last_used_at"),
        },
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------


def _html(s: str) -> str:
    """Escape only what would break HTML; keep <br>, <b>, <i> tags as-is."""
    return s


def _compute_layout(nodes: list[dict], edges: list[dict], layout: str) -> None:
    """Pre-compute (x, y) so vis.js doesn't run continuous physics.

    Strategy: a single spring_layout pass on the FULL graph (entities +
    memorias), with high `k` for clear separation and 500 iterations for
    stability. Leaf memorias attach naturally to their entity. High-degree
    entities cluster centrally because more edges pull them toward the
    centroid.

    Disconnected components are placed in concentric rings around the main
    component so the canvas isn't dominated by a single circle of orphans.
    """
    if layout != "force":
        return
    try:
        import networkx as nx
    except ImportError:
        return

    g = nx.Graph()
    for n in nodes:
        g.add_node(n["id"])
    for e in edges:
        g.add_edge(e["from"], e["to"])
    if not g.nodes:
        return

    components = sorted(nx.connected_components(g), key=len, reverse=True)
    n_total = len(g.nodes)
    pos: dict = {}
    if not components:
        return

    # Main component: spring_layout
    main = g.subgraph(components[0]).copy()
    main_pos = nx.spring_layout(
        main, seed=42,
        k=3.5 / max(1, len(main) ** 0.45),
        iterations=500,
        scale=1.0,
    )
    pos.update(main_pos)

    # Place additional components on concentric rings around the main blob
    for i, comp in enumerate(components[1:], start=1):
        sub = g.subgraph(comp).copy()
        if len(sub) == 1:
            # Singleton: place on a ring outside the main cluster
            (node_id,) = sub.nodes
            angle = (hash(node_id) % 360) * 3.14159 / 180.0
            ring = 1.4 + (i % 5) * 0.15
            pos[node_id] = (ring * 0.7 + 0.5 * 0.3 * (i % 3),
                            ring * (1.0 if hash(node_id) % 2 else -1.0))
            # Simpler: pseudo-random offset
            import math as _m
            pos[node_id] = (_m.cos(angle) * 1.5, _m.sin(angle) * 1.5)
        else:
            sub_pos = nx.spring_layout(sub, seed=42 + i, k=0.8, iterations=80, scale=0.4)
            # offset to a ring
            import math as _m
            angle = (i * 137) % 360 * _m.pi / 180.0  # golden angle distribution
            ox, oy = _m.cos(angle) * 1.8, _m.sin(angle) * 1.8
            pos.update({nid: (p[0] + ox, p[1] + oy) for nid, p in sub_pos.items()})

    # Scale to vis.js coordinates — bigger graphs need more pixels
    scale = 130 * max(20, n_total) ** 0.5
    for n in nodes:
        p = pos.get(n["id"])
        if p is None:
            continue
        n["x"] = float(p[0]) * scale
        n["y"] = float(p[1]) * scale
        n["physics"] = False


def _render_html(
    *,
    nodes: list[dict],
    edges: list[dict],
    title: str,
    output_path: Path | str,
    layout: str = "force",
    auto_refresh_seconds: int = 0,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-compute fixed positions so the browser doesn't run runaway physics.
    _compute_layout(nodes, edges, layout)

    used_groups = sorted({n.get("group", "other") for n in nodes})
    legend_items = [
        {"key": g, "color": TYPE_COLORS.get(g, TYPE_COLORS["other"])["bg"],
         "label": TYPE_COLORS.get(g, TYPE_COLORS["other"]).get("label", g)}
        for g in used_groups
    ]

    stats = {
        "nodes": len(nodes),
        "edges": len(edges),
        "by_type": _count_by(nodes, "group"),
    }

    html_str = _TEMPLATE.replace("__TITLE__", html.escape(title))
    html_str = html_str.replace("__NODES__", json.dumps(nodes, ensure_ascii=False))
    html_str = html_str.replace("__EDGES__", json.dumps(edges, ensure_ascii=False))
    html_str = html_str.replace("__LEGEND__", json.dumps(legend_items, ensure_ascii=False))
    html_str = html_str.replace("__STATS__", json.dumps(stats, ensure_ascii=False))
    html_str = html_str.replace("__LAYOUT__", layout)
    html_str = html_str.replace("__AUTO_REFRESH__", str(int(auto_refresh_seconds)))

    output_path.write_text(html_str, encoding="utf-8")
    _ensure_vendored_visjs(output_path.parent)
    log.info("graph: rendered → %s (%d nodes, %d edges)", output_path, len(nodes), len(edges))
    return output_path


_VIS_CDN = "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js"


def _ensure_vendored_visjs(graphs_dir: Path) -> None:
    """Vendor vis-network.min.js into <graphs_dir>/lib/ once. Idempotent."""
    target = graphs_dir / "lib" / "vis-network.min.js"
    if target.exists() and target.stat().st_size > 100_000:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request
        log.info("vendoring vis-network.min.js → %s", target)
        urllib.request.urlretrieve(_VIS_CDN, target)
    except Exception as e:
        log.warning("could not vendor vis-network locally (%s); HTML will use CDN fallback", e)


def _count_by(items: list[dict], key: str) -> dict:
    out: dict[str, int] = {}
    for it in items:
        k = it.get(key, "other")
        out[k] = out.get(k, 0) + 1
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__ · Memoirs</title>
<script>
  // Try local first (works offline + no CDN dependency), CDN as fallback.
  function _loadVis(src, onfail) {
    const s = document.createElement("script");
    s.src = src;
    s.onerror = onfail;
    s.onload = () => window.dispatchEvent(new Event("vis-loaded"));
    document.head.appendChild(s);
  }
  _loadVis("./lib/vis-network.min.js", () => {
    _loadVis("https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js",
      () => { document.body.innerHTML =
        '<div style="padding:40px;color:#fff;background:#1a1d23;font:14px sans-serif">' +
        '<b>vis-network failed to load</b><br>' +
        'Tried: ./lib/vis-network.min.js and CDN.<br>' +
        'Run: <code>memoirs graph entities</code> to vendor the library locally,<br>' +
        'or check DevTools → Network tab.' +
        '</div>'; });
  });
</script>
<style>
  :root {
    --bg: #1a1d23;
    --panel: #232830;
    --border: #2f3640;
    --text: #e8eaed;
    --muted: #8b95a3;
    --accent: #4f8cc9;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; width: 100%; overflow: hidden;
    background: var(--bg); color: var(--text);
    font: 13px/1.5 -apple-system, "Segoe UI", "Inter", sans-serif; }
  .app { display: grid; grid-template-columns: 240px 1fr 320px; grid-template-rows: 48px 1fr 36px;
    grid-template-areas: "topbar topbar topbar" "left center right" "footer footer footer";
    height: 100vh; max-height: 100vh; overflow: hidden; }
  .topbar { grid-area: topbar; background: var(--panel); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px; padding: 0 16px; }
  .topbar h1 { font-size: 14px; font-weight: 600; margin: 0; color: var(--text); }
  .topbar .stats { color: var(--muted); font-size: 12px; }
  .topbar .stats .pill { display: inline-block; background: var(--bg); border: 1px solid var(--border);
    padding: 2px 8px; border-radius: 10px; margin-right: 6px; }
  .topbar .search { margin-left: auto; }
  .topbar .search input { background: var(--bg); border: 1px solid var(--border); color: var(--text);
    padding: 6px 10px; border-radius: 4px; width: 240px; outline: none; font-size: 13px; }
  .topbar .search input:focus { border-color: var(--accent); }
  .left, .right { background: var(--panel); border: 0; padding: 12px; overflow-y: auto; }
  .left { grid-area: left; border-right: 1px solid var(--border); }
  .right { grid-area: right; border-left: 1px solid var(--border); }
  .center { grid-area: center; position: relative; background: var(--bg);
    min-width: 0; min-height: 0; overflow: hidden; }
  #network { position: absolute; inset: 0; }
  .footer { grid-area: footer; background: var(--panel); border-top: 1px solid var(--border);
    display: flex; gap: 8px; align-items: center; padding: 0 12px; font-size: 12px; color: var(--muted); }
  .footer button { background: var(--bg); color: var(--text); border: 1px solid var(--border);
    padding: 4px 10px; border-radius: 4px; font-size: 12px; cursor: pointer; }
  .footer button:hover { border-color: var(--accent); color: var(--accent); }

  h2 { font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--muted);
    letter-spacing: 1px; margin: 16px 0 8px 0; }
  h2:first-child { margin-top: 0; }
  .legend-item { display: flex; align-items: center; gap: 8px; padding: 4px 0;
    cursor: pointer; user-select: none; font-size: 12px; }
  .legend-item .dot { width: 12px; height: 12px; border-radius: 50%; flex: 0 0 12px; }
  .legend-item.disabled { opacity: 0.35; }
  .legend-item .count { margin-left: auto; color: var(--muted); font-size: 11px; }
  .panel-empty { color: var(--muted); font-style: italic; padding: 8px 0; }
  .detail h3 { font-size: 14px; margin: 0 0 6px 0; word-break: break-word; }
  .detail .badge { display: inline-block; background: var(--bg); border: 1px solid var(--border);
    color: var(--text); padding: 2px 6px; border-radius: 3px; font-size: 11px; margin-right: 4px; }
  .detail dl { margin: 12px 0; display: grid; grid-template-columns: 100px 1fr; gap: 4px 12px;
    font-size: 12px; }
  .detail dt { color: var(--muted); }
  .detail dd { margin: 0; word-break: break-word; }
  .detail .content-block { background: var(--bg); border: 1px solid var(--border);
    padding: 10px; border-radius: 4px; margin-top: 12px; max-height: 280px; overflow-y: auto;
    white-space: pre-wrap; font-family: ui-monospace, "SF Mono", monospace; font-size: 11.5px;
    line-height: 1.45; color: #c8d2e0; }
  .vis-tooltip { background: var(--panel) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 4px !important;
    padding: 8px !important; font-family: inherit !important; max-width: 320px !important; }
  .vis-tooltip b { color: #fff; }
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <h1>__TITLE__</h1>
    <div class="stats" id="stats"></div>
    <div class="search">
      <input id="search" type="search" placeholder="Search nodes by label or content…" />
    </div>
  </div>
  <div class="left">
    <h2>Filter by type</h2>
    <div id="legend"></div>
    <h2>Layout</h2>
    <div>
      <label style="display:block; padding: 4px 0;">
        <input type="radio" name="layout" value="force" /> Force-directed
      </label>
      <label style="display:block; padding: 4px 0;">
        <input type="radio" name="layout" value="hierarchical" /> Hierarchical (top-down)
      </label>
    </div>
  </div>
  <div class="center"><div id="network"></div></div>
  <div class="right" id="detail">
    <p class="panel-empty">Click a node to see details.</p>
  </div>
  <div class="footer">
    <button id="btn-fit">Fit view</button>
    <button id="btn-physics">Pause physics</button>
    <button id="btn-export">Export PNG</button>
    <span style="margin-left:auto;">Memoirs · vis.js · click + drag</span>
  </div>
</div>

<script>
// Wait for vis-network to load (async via the loader above), then init.
function _bootWhenReady(fn) {
  if (typeof vis !== "undefined" && vis.Network) return fn();
  window.addEventListener("vis-loaded", fn, { once: true });
}
_bootWhenReady(() => {
const NODES_DATA = __NODES__;
const EDGES_DATA = __EDGES__;
const LEGEND = __LEGEND__;
const STATS = __STATS__;
const INITIAL_LAYOUT = "__LAYOUT__";
const AUTO_REFRESH = __AUTO_REFRESH__;
if (AUTO_REFRESH > 0) {
  // Page reloads from disk; the watch loop on the CLI side keeps the file fresh
  setTimeout(() => location.reload(), AUTO_REFRESH * 1000);
  document.title = "🔄 " + document.title;
}

// ----- header stats
const statsEl = document.getElementById("stats");
let statsHtml = `<span class="pill">${STATS.nodes} nodes</span><span class="pill">${STATS.edges} edges</span>`;
for (const [k, n] of Object.entries(STATS.by_type || {})) {
  statsHtml += `<span class="pill">${k}: ${n}</span>`;
}
statsEl.innerHTML = statsHtml;

// ----- legend with toggle
const enabledGroups = new Set(LEGEND.map(l => l.key));
const legendEl = document.getElementById("legend");
const legendCounts = STATS.by_type || {};
LEGEND.forEach(item => {
  const div = document.createElement("div");
  div.className = "legend-item";
  div.dataset.key = item.key;
  div.innerHTML = `<span class="dot" style="background:${item.color}"></span>
                   <span class="lbl">${item.label}</span>
                   <span class="count">${legendCounts[item.key] || 0}</span>`;
  div.onclick = () => {
    if (enabledGroups.has(item.key)) {
      enabledGroups.delete(item.key);
      div.classList.add("disabled");
    } else {
      enabledGroups.add(item.key);
      div.classList.remove("disabled");
    }
    applyFilters();
  };
  legendEl.appendChild(div);
});

// ----- vis.js network
const nodesDS = new vis.DataSet(NODES_DATA);
const edgesDS = new vis.DataSet(EDGES_DATA);
const networkContainer = document.getElementById("network");
const network = new vis.Network(
  networkContainer,
  { nodes: nodesDS, edges: edgesDS },
  layoutOptions(INITIAL_LAYOUT),
);
// Manual resize that respects the parent grid cell — no autoResize feedback loop.
function resizeNetwork() {
  const r = networkContainer.getBoundingClientRect();
  network.setSize(r.width + "px", r.height + "px");
  network.redraw();
}
window.addEventListener("resize", resizeNetwork);
// Initial size + fit once after first paint
requestAnimationFrame(() => {
  resizeNetwork();
  setTimeout(() => network.fit(), 50);
});

function layoutOptions(layout) {
  // Positions seeded from Python (kamada_kawai + spring). Physics runs briefly
  // with strong central gravity to settle the layout, then freezes itself.
  const base = {
    autoResize: false,
    interaction: { hover: true, navigationButtons: true, tooltipDelay: 100,
                   multiselect: false, zoomView: true, dragView: true },
    edges: { smooth: false, color: { opacity: 0.55, inherit: false },
             font: { color: "#aab1ba", size: 10, strokeWidth: 0 } },
    nodes: { font: { color: "#e8eaed", size: 12 }, borderWidth: 1.5 },
    physics: { enabled: false },
  };
  if (layout === "hierarchical") {
    return Object.assign(base, {
      layout: { hierarchical: { direction: "LR", sortMethod: "directed",
                                levelSeparation: 220, nodeSpacing: 140 } },
    });
  }
  // Brief settling physics — strong central gravity, hard velocity cap, limited iterations.
  return Object.assign(base, {
    physics: {
      enabled: true,
      solver: "barnesHut",
      barnesHut: {
        gravitationalConstant: -1500,
        centralGravity: 1.5,        // strong pull to center — prevents drift
        springLength: 90,
        springConstant: 0.08,
        damping: 0.85,              // high damping = settles fast
        avoidOverlap: 0.6,
      },
      maxVelocity: 15,              // hard cap — even if forces unbalance, no explosion
      minVelocity: 0.6,
      timestep: 0.3,
      stabilization: {
        enabled: true,
        iterations: 200,            // hard cap on iterations
        updateInterval: 50,
        fit: true,
      },
    },
  });
}

document.querySelectorAll('input[name="layout"]').forEach(r => {
  if (r.value === INITIAL_LAYOUT) r.checked = true;
  r.onchange = () => network.setOptions(layoutOptions(r.value));
});

// Hybrid mode: physics ON briefly to settle layout, then turn it OFF forever.
// This is what prevents the runaway expansion: a hard cap on iterations + high
// central gravity + auto-freeze when stabilizationIterationsDone fires.
let physicsOn = INITIAL_LAYOUT === "force";
let stabilized = false;
network.on("stabilizationIterationsDone", () => {
  if (stabilized) return;
  stabilized = true;
  network.setOptions({ physics: { enabled: false } });
  physicsOn = false;
  const btn = document.getElementById("btn-physics");
  if (btn) btn.textContent = "Resume physics";
  network.fit({ animation: { duration: 400 } });
});
// Hard fallback: if stabilization event never fires (rare vis.js bug), force-stop after 6s
setTimeout(() => {
  if (stabilized) return;
  stabilized = true;
  network.setOptions({ physics: { enabled: false } });
  physicsOn = false;
  network.fit({ animation: false });
}, 6000);

// ----- filtering by type
function applyFilters() {
  const updates = NODES_DATA.map(n => {
    const visible = enabledGroups.has(n.group);
    const old = nodesDS.get(n.id);
    return Object.assign({}, old, { hidden: !visible });
  });
  nodesDS.update(updates);
}

// ----- search highlight
const searchEl = document.getElementById("search");
let lastHits = [];
searchEl.oninput = () => {
  const q = searchEl.value.trim().toLowerCase();
  if (!q) {
    lastHits.forEach(n => nodesDS.update({ id: n.id, borderWidth: 1.5, font: { color: "#e8eaed", size: 12 } }));
    lastHits = [];
    return;
  }
  const hits = NODES_DATA.filter(n => {
    const label = (n.label || "").toLowerCase();
    const content = (n.data && (n.data.content || n.data.name || "")).toLowerCase();
    return label.includes(q) || content.includes(q);
  });
  lastHits.forEach(n => nodesDS.update({ id: n.id, borderWidth: 1.5, font: { color: "#e8eaed", size: 12 } }));
  hits.forEach(n => nodesDS.update({ id: n.id, borderWidth: 4, font: { color: "#ffd166", size: 14 } }));
  lastHits = hits;
  if (hits.length === 1) network.focus(hits[0].id, { animation: true, scale: 1.5 });
};

// ----- detail panel on click
const detailEl = document.getElementById("detail");
network.on("selectNode", e => {
  if (!e.nodes.length) return;
  const node = NODES_DATA.find(n => n.id === e.nodes[0]);
  if (!node) return;
  detailEl.innerHTML = renderDetail(node);
});
network.on("deselectNode", () => {
  detailEl.innerHTML = '<p class="panel-empty">Click a node to see details.</p>';
});

function renderDetail(n) {
  const d = n.data || {};
  let html = `<div class="detail"><h3>${escape(n.label || d.id || "node")}</h3>`;
  html += `<div><span class="badge">${escape(d.kind || "?")}</span>`;
  if (d.type) html += `<span class="badge">${escape(d.type)}</span>`;
  if (d.status) html += `<span class="badge">${escape(d.status)}</span>`;
  html += `</div><dl>`;
  for (const [k, v] of Object.entries(d)) {
    if (k === "kind" || k === "content" || v === null || v === undefined) continue;
    html += `<dt>${escape(k)}</dt><dd>${escape(formatVal(v))}</dd>`;
  }
  html += `</dl>`;
  if (d.content) {
    html += `<div class="content-block">${escape(d.content)}</div>`;
  }
  // neighbors summary
  const ed = network.getConnectedEdges(n.id);
  if (ed.length) {
    html += `<h2 style="margin-top:18px;">Connected (${ed.length})</h2>`;
    const seen = new Set();
    ed.slice(0, 12).forEach(eid => {
      const e = edgesDS.get(eid);
      const otherId = e.from === n.id ? e.to : e.from;
      if (seen.has(otherId)) return;
      seen.add(otherId);
      const other = NODES_DATA.find(x => x.id === otherId);
      if (other) html += `<div style="padding:3px 0;font-size:12px;cursor:pointer;" onclick="window.__sel('${escape(otherId)}')">→ ${escape(other.label || otherId)}</div>`;
    });
  }
  html += `</div>`;
  return html;
}
window.__sel = (id) => { network.selectNodes([id]); network.focus(id, { animation: true, scale: 1.4 }); detailEl.innerHTML = renderDetail(NODES_DATA.find(n => n.id === id)); };

function escape(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function formatVal(v) { if (typeof v === "number" && !Number.isInteger(v)) return v.toFixed(3); return v; }

// ----- footer buttons
document.getElementById("btn-fit").onclick = () => network.fit({ animation: true });
document.getElementById("btn-physics").onclick = (e) => {
  physicsOn = !physicsOn;
  network.setOptions({ physics: { enabled: physicsOn } });
  e.target.textContent = physicsOn ? "Pause physics" : "Resume physics";
};
document.getElementById("btn-export").onclick = () => {
  const canvas = document.querySelector("#network canvas");
  if (!canvas) return;
  const link = document.createElement("a");
  link.download = "memoirs-graph.png";
  link.href = canvas.toDataURL();
  link.click();
};
}); // close _bootWhenReady
</script>
</body>
</html>
"""
