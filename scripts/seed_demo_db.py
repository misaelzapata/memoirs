"""Seed a self-contained demo DB that exercises every feature memoirs ships.

The output is `data/demo/memoirs_demo.sqlite` — committed to the repo so a
contributor can do:

    memoirs --db data/demo/memoirs_demo.sqlite serve

…and immediately see a populated dashboard, populated provenance,
non-trivial conflicts, a memory graph, recent tool_call captures, and a
couple of point-in-time snapshots.

The seed simulates a fictional engineer ("Lara, backend dev at Acme")
working across two projects: ``payments-api`` and ``mobile-android``.
Every memoria lands through the heuristic curator path, which means the
``provenance_json`` column is filled in honestly (actor=heuristic /
process=extract). A small subset is hand-edited afterwards to simulate
``actor=curator`` decisions for visual variety.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force heuristic-only seeding so the script is fast (no LLM warmup).
os.environ["MEMOIRS_CURATOR_ENABLED"] = "off"
os.environ["MEMOIRS_GEMMA_CURATOR"] = "off"
os.environ["MEMOIRS_GRAPH_LLM"] = "off"

from memoirs.core.ids import content_hash, utc_now
from memoirs.db import MemoirsDB
from memoirs.engine import embeddings as emb
from memoirs.engine import hybrid_retrieval as hr
from memoirs.engine import memory_engine as me
from memoirs.engine import snapshots as snap
from memoirs.engine.curator import Candidate
from memoirs.models import DEFAULT_SCOPE


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

# A persona: Lara, senior backend engineer at Acme, 8 years XP. Two active
# repos: ``payments-api`` (Go) and ``mobile-android`` (Kotlin). Recently
# rolled a feature flag for the new Stripe integration. Some prefs about
# tooling. Style guide. A few tool_call captures to make the "commands"
# inspector demo-able. Conflicts seeded by repeating the same fact at two
# different time points (``valid_from``).

SAMPLES: list[tuple[str, str, int]] = [
    # ---- preferences ----
    ("preference", "Lara prefers Go for backend services and avoids dynamic typing in production", 5),
    ("preference", "Lara prefers Kotlin coroutines over RxJava on the Android side", 4),
    ("preference", "Lara likes terse PR descriptions and a single bullet list of changes", 3),
    ("preference", "Lara prefers vim mode in any editor; remaps Caps Lock to Esc on first install", 4),
    ("preference", "Lara prefers structured logs in JSON over plain text everywhere", 4),
    ("preference", "Lara dislikes auto-formatters that rewrite imports — prefers `goimports -local acme`", 3),

    # ---- facts (durable knowledge about user / org / projects) ----
    ("fact", "Lara works at Acme Inc. as a senior backend engineer, on the payments platform team", 5),
    ("fact", "Acme's payments-api is a Go monolith on AWS ECS Fargate, deployed via GitHub Actions", 4),
    ("fact", "Stripe is the primary payment processor; Adyen is a fallback for EU SEPA payments", 4),
    ("fact", "Lara's GitHub handle is @lara-acme; SSH key fingerprint cached in 1Password", 3),
    ("fact", "The mobile-android app uses Kotlin 2.0 + Compose 1.7 + Hilt for DI", 4),
    ("fact", "payments-api runs Postgres 16 on RDS, single-writer + 2 read replicas", 4),
    ("fact", "Acme uses DataDog for metrics, Sentry for errors, Slack for paging", 3),

    # ---- projects ----
    ("project", "payments-api: Go monolith owning checkout flow, refunds, webhook intake, ledger", 5),
    ("project", "mobile-android: Kotlin + Compose app shipping fortnightly via Play internal track", 5),
    ("project", "Migration to Stripe Issuing for virtual corporate cards — Q3 2026 target", 4),
    ("project", "Webhook retry redesign — current exponential backoff misfires on 24h+ partner outages", 4),
    ("project", "Internal CLI `acmectl` to wrap kubectl + AWS SSO + DataDog dashboards", 3),

    # ---- decisions ----
    ("decision", "Decided to ship Stripe Issuing integration behind a `flag.virtual_cards` GrowthBook flag for staged rollout", 5),
    ("decision", "Adopted bge-reranker for retrieval quality after measuring +108% MRR vs hybrid baseline", 4),
    ("decision", "Standardized on `httpx` instead of `requests` in payments-api after async refactor", 4),
    ("decision", "Chose Postgres `CITEXT` for email columns to avoid LOWER() on every WHERE", 4),
    ("decision", "Reject SQLAlchemy ORM in payments-api; stick with `pgx` + hand-written queries", 5),
    ("decision", "Defer GraphRAG-style entity-graph retrieval — bench showed no lift over PRF + reranker", 3),

    # ---- style / coding rules ----
    ("style", "Code style: no docstrings on private helpers; only WHY comments when non-obvious", 4),
    ("style", "Tests must hit a real Postgres in CI, not mocks (per integration-test policy)", 5),
    ("style", "Go errors: always wrap with `%w` and a context phrase; never `errors.New` at the boundary", 4),
    ("style", "Compose state: hoist all UI state to ViewModels; no `remember { mutableStateOf(...) }` in screen-level composables", 4),
    ("style", "SQL migrations: one file per change; never edit a merged migration", 5),

    # ---- tasks (in-flight work) ----
    ("task", "Add idempotency-key support to /webhook/stripe — TICKET-3041 — due Q3 W2", 4),
    ("task", "Backfill `users.country_code` on the prod Postgres replica off-hours", 3),
    ("task", "Audit IAM roles for the `payments-api-deployer` ECS task — minimal permissions check", 4),
    ("task", "Write runbook for the Adyen fallback when Stripe webhook intake breaks", 3),
    ("task", "Update mobile-android targetSdk from 34 to 35 before Play Store deadline", 4),

    # ---- credential pointers (locations, never values) ----
    ("credential_pointer", "OpenAI API key location: 1Password vault 'Acme/Engineering' item 'openai-prod'", 5),
    ("credential_pointer", "Stripe live secret key: AWS Secrets Manager `payments-api/stripe/live`", 5),
    ("credential_pointer", "DataDog API key location: ~/.config/datadog/api.key (chmod 600); rotate quarterly", 4),
    ("credential_pointer", "Adyen webhook HMAC secret: AWS Secrets Manager `payments-api/adyen/webhook-hmac`", 4),

    # ---- procedural (always-on agent policies) ----
    ("procedural", "When the user asks for code, prefer minimal diffs and zero new abstractions unless asked", 5),
    ("procedural", "When summarizing benchmark results, lead with concrete metrics, not narrative prose", 5),
    ("procedural", "When introducing a new feature, write the test first and only then the implementation", 4),
    ("procedural", "Never commit code that disables a CI check; if a check is broken, fix the check", 5),
    ("procedural", "When an agent runs a destructive shell command, always preview the action and ask for confirmation", 5),
    ("procedural", "Keep PR descriptions to under 100 words; include a `## Test plan` checklist", 3),
]


def _seed_memorias(db: MemoirsDB) -> list[str]:
    """Run each candidate through the curator path so provenance gets stamped."""
    out: list[str] = []
    for typ, content, imp in SAMPLES:
        cand = Candidate(type=typ, content=content, importance=imp, confidence=0.92)
        decision = me.decide_memory_action(db, cand)
        res = me.apply_decision(db, cand, decision, scope=DEFAULT_SCOPE)
        out.append(res["memory_id"])
    return out


def _seed_conflicts(db: MemoirsDB) -> None:
    """Insert two pairs of memorias that disagree on the same fact at
    different times so the conflicts table has visible entries."""
    now = utc_now()
    earlier = (datetime.now(timezone.utc) - timedelta(days=180)).strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )

    with db.conn:
        # Conflict 1: targetSdk old vs new.
        old = db.conn.execute(
            "INSERT INTO memories (id, type, content, content_hash, importance, "
            "confidence, score, usage_count, user_signal, valid_from, valid_to, "
            "metadata_json, provenance_json, created_at, updated_at) "
            "VALUES ('mem_demo_targetSdk_old', 'fact', "
            "'mobile-android targetSdk is 34 (current production)', ?, 4, 0.9, "
            "0.5, 0, 0, ?, ?, '{}', "
            "'{\"actor\":\"curator\",\"process\":\"consolidate\",\"decision\":\"ADD\","
            "\"reason\":\"durable platform fact\"}', ?, ?)",
            (content_hash("targetSdk-old"), earlier, now, earlier, earlier),
        )
        new = db.conn.execute(
            "INSERT INTO memories (id, type, content, content_hash, importance, "
            "confidence, score, usage_count, user_signal, valid_from, "
            "metadata_json, provenance_json, created_at, updated_at) "
            "VALUES ('mem_demo_targetSdk_new', 'fact', "
            "'mobile-android targetSdk is 35 (per Q3 Play Store deadline)', ?, 5, 0.95, "
            "0.6, 0, 0, ?, '{}', "
            "'{\"actor\":\"curator\",\"process\":\"consolidate\",\"decision\":\"ADD\","
            "\"reason\":\"newer durable fact supersedes older\"}', ?, ?)",
            (content_hash("targetSdk-new"), now, now, now),
        )
    # Add a row to the `conflicts` table linking the two memorias.
    try:
        db.conn.execute(
            "INSERT INTO conflicts (memory_a_id, memory_b_id, kind, "
            "similarity, status, created_at) VALUES (?, ?, ?, ?, 'open', ?)",
            (
                "mem_demo_targetSdk_old",
                "mem_demo_targetSdk_new",
                "temporal-versioned",
                0.92,
                now,
            ),
        )
        db.conn.commit()
    except Exception as e:
        print(f"  conflicts insert skipped: {e}")


def _seed_tool_calls(db: MemoirsDB) -> None:
    """Insert a handful of `tool_call` memorias with realistic cwd/project
    metadata so the `commands list` page and resume_thread surface them."""
    now = utc_now()
    samples = [
        ("Bash", "go test ./internal/ledger/... -run TestRefund", "success", "payments-api"),
        ("Read", "internal/webhook/stripe.go", "success", "payments-api"),
        ("Edit", "ui/screens/Settings.kt", "success", "mobile-android"),
        ("Bash", "kubectl rollout status deploy/payments-api -n prod", "success", "payments-api"),
        ("Bash", "datadog-cli timeboard get 'payments-api SLO'", "success", "payments-api"),
    ]
    cwd_map = {
        "payments-api": "/home/lara/code/payments-api",
        "mobile-android": "/home/lara/code/mobile-android",
    }
    for i, (tool, args, status, project) in enumerate(samples):
        mid = f"tc_demo_{i:02d}"
        prov = {
            "actor": "tool_recorder",
            "process": "record_tool_call",
            "decision": "ADD",
            "reason": f"captured {tool} call from {project}",
        }
        meta = {
            "cwd": cwd_map[project],
            "project_name": project,
            "conversation_id": f"conv_demo_{project}",
            "message_ordinal": i + 1,
            "timestamp": now,
            "tool_use_id": f"toolu_demo_{i:02d}",
        }
        with db.conn:
            db.conn.execute(
                "INSERT INTO memories (id, type, content, content_hash, importance, "
                "confidence, score, usage_count, user_signal, valid_from, "
                "metadata_json, provenance_json, tool_name, tool_args_json, "
                "tool_status, created_at, updated_at) VALUES "
                "(?, 'tool_call', ?, ?, 2, 0.9, 0.4, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    mid, f"{tool}: {args}", content_hash(f"{tool}{args}"), now,
                    json.dumps(meta), json.dumps(prov), tool,
                    json.dumps({"command": args} if tool == "Bash" else {"path": args}),
                    status, now, now,
                ),
            )


def _seed_links(db: MemoirsDB, memory_ids: list[str]) -> None:
    """Add a few zettelkasten links so the graph view has edges."""
    if len(memory_ids) < 4:
        return
    pairs = [
        (memory_ids[0], memory_ids[1], 0.78),  # two prefs
        (memory_ids[6], memory_ids[7], 0.83),  # facts about Acme
        (memory_ids[13], memory_ids[15], 0.71),  # projects
        (memory_ids[18], memory_ids[19], 0.67),  # decisions
    ]
    now = utc_now()
    for a, b, sim in pairs:
        try:
            db.conn.execute(
                "INSERT INTO memory_links (source_id, target_id, similarity, "
                "kind, created_at) VALUES (?, ?, ?, 'semantic', ?) "
                "ON CONFLICT DO NOTHING",
                (a, b, sim, now),
            )
        except Exception:
            pass
    db.conn.commit()


def _seed_snapshots(db_path: Path) -> None:
    """Take one named snapshot so the snapshots page is non-empty."""
    snap.create(db_path, name="seed-baseline")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out",
        default=str(ROOT / "data" / "demo" / "memoirs_demo.sqlite"),
        help="output DB path (will be wiped + recreated)",
    )
    args = p.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    # Clear any sibling snapshots from a prior run.
    snap_dir = out.parent / "snapshots"
    if snap_dir.exists():
        for f in snap_dir.glob("*.sqlite"):
            f.unlink()

    print(f"seeding → {out}")
    db = MemoirsDB(out)
    db.init()
    hr.ensure_fts_schema(db.conn)

    print("  · memorias …")
    ids = _seed_memorias(db)
    print(f"    {len(ids)} memorias persisted")

    print("  · conflicts …")
    _seed_conflicts(db)

    print("  · tool_call captures …")
    _seed_tool_calls(db)

    print("  · zettelkasten links …")
    _seed_links(db, ids)

    n_active = db.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE archived_at IS NULL"
    ).fetchone()[0]
    n_with_prov = db.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE provenance_json != '{}' "
        "AND archived_at IS NULL"
    ).fetchone()[0]
    db.close()

    print("  · point-in-time snapshot …")
    _seed_snapshots(out)

    print()
    print(f"DONE — {n_active} active memorias, {n_with_prov} with provenance, "
          f"DB at {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
