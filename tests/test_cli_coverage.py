"""Targeted coverage tests for memoirs/cli.py.

Focuses on the argparse construction and the lightweight `main(argv=...)`
dispatch paths that don't need network / Gemma / spaCy. Heavy commands
(daemon, watch, ui, doctor, setup, models pull, eval) are not exercised here
— covering them requires more setup than tests/test_<module>_coverage.py is
meant to carry.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memoirs import cli


# ---------------------------------------------------------------------------
# Parser construction — every subparser is created
# ---------------------------------------------------------------------------


def test_build_parser_has_core_commands():
    parser = cli.build_parser()
    # argparse exposes choices via the dest='command' subparsers action.
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions
    choices = set(actions[0].choices)
    for cmd in (
        "init", "ingest", "watch", "status", "doctor", "setup", "daemon",
        "logs", "conversations", "messages", "mcp", "serve", "ui",
        "extract", "consolidate", "maintenance", "cleanup", "audit", "ask",
        "index-entities", "projects-refresh", "trace", "review",
        "graph", "models", "links", "eval", "db",
    ):
        assert cmd in choices


def test_build_parser_parses_ingest_args():
    parser = cli.build_parser()
    args = parser.parse_args(["ingest", "/tmp/foo.md", "--kind", "auto"])
    assert args.command == "ingest"
    assert args.path == "/tmp/foo.md"
    assert args.kind == "auto"


def test_build_parser_parses_links_rebuild_args():
    parser = cli.build_parser()
    args = parser.parse_args([
        "links", "rebuild", "--top-k", "7", "--threshold", "0.6",
        "--mode", "absolute", "--batch-size", "50",
    ])
    assert args.command == "links"
    assert args.links_cmd == "rebuild"
    assert args.top_k == 7
    assert args.threshold == 0.6
    assert args.mode == "absolute"


def test_build_parser_requires_subcommand():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # subcommand required


# ---------------------------------------------------------------------------
# Helpers (logging path)
# ---------------------------------------------------------------------------


def test_log_path_for_resolves_next_to_db(tmp_path):
    p = tmp_path / "subdir" / "memoirs.sqlite"
    p.parent.mkdir(parents=True)
    p.touch()
    assert cli._log_path_for(p) == p.parent / "memoirs.log"


def test_configure_logging_writes_log_file(tmp_path):
    db_path = tmp_path / "memoirs.sqlite"
    log_file = cli._configure_logging(db_path, verbose=True)
    assert log_file.parent == db_path.parent
    # Re-running should be idempotent (clean handlers, set new ones).
    cli._configure_logging(db_path, verbose=False)


def test_make_reporter_prints_and_logs(capsys):
    report = cli._make_reporter()
    report("hello world")
    captured = capsys.readouterr()
    assert "hello world" in captured.out


# ---------------------------------------------------------------------------
# main() dispatch — light commands that don't need external services
# ---------------------------------------------------------------------------


def test_main_init_creates_db(tmp_path, capsys):
    db = tmp_path / "memoirs.sqlite"
    rc = cli.main(["--db", str(db), "init"])
    assert rc == 0
    assert db.exists()
    out = capsys.readouterr().out
    assert "initialized" in out


def test_main_status_prints_json(tmp_path, capsys):
    db = tmp_path / "memoirs.sqlite"
    cli.main(["--db", str(db), "init"])
    capsys.readouterr()  # drain init output
    rc = cli.main(["--db", str(db), "status"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert "sources" in parsed


def test_main_conversations_empty_table(tmp_path, capsys):
    db = tmp_path / "memoirs.sqlite"
    cli.main(["--db", str(db), "init"])
    capsys.readouterr()
    rc = cli.main(["--db", str(db), "conversations"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out == ""


def test_main_conversations_json_flag(tmp_path, capsys):
    db = tmp_path / "memoirs.sqlite"
    cli.main(["--db", str(db), "init"])
    capsys.readouterr()
    rc = cli.main(["--db", str(db), "conversations", "--json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == []


def test_main_messages_empty_table(tmp_path, capsys):
    db = tmp_path / "memoirs.sqlite"
    cli.main(["--db", str(db), "init"])
    capsys.readouterr()
    rc = cli.main(["--db", str(db), "messages", "--json"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == []


def test_main_consolidate_empty_corpus(tmp_path, capsys, monkeypatch):
    db = tmp_path / "memoirs.sqlite"
    cli.main(["--db", str(db), "init"])
    capsys.readouterr()
    rc = cli.main(["--db", str(db), "consolidate", "--limit", "10"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["processed"] == 0


def test_main_index_entities_empty_corpus(tmp_path, capsys):
    db = tmp_path / "memoirs.sqlite"
    cli.main(["--db", str(db), "init"])
    capsys.readouterr()
    rc = cli.main(["--db", str(db), "index-entities"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["memories_processed"] == 0


def test_main_projects_refresh_empty(tmp_path, capsys):
    db = tmp_path / "memoirs.sqlite"
    cli.main(["--db", str(db), "init"])
    capsys.readouterr()
    rc = cli.main(["--db", str(db), "projects-refresh"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["projects"] == 0


def test_main_maintenance_empty_corpus(tmp_path, capsys, monkeypatch):
    """`maintenance` runs run_daily_maintenance which calls embeddings.

    On an empty DB it should still complete; we monkeypatch the lifecycle
    auto_merge to skip embeddings model load.
    """
    from memoirs.engine import lifecycle as lc
    monkeypatch.setattr(lc, "auto_merge_near_duplicates",
                        lambda db, **k: {"merged": 0, "contradictions": 0,
                                         "scanned": 0, "dry_run": False})
    db = tmp_path / "memoirs.sqlite"
    cli.main(["--db", str(db), "init"])
    capsys.readouterr()
    rc = cli.main(["--db", str(db), "maintenance"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["scores_updated"] == 0


def test_main_cleanup_dry_run(tmp_path, capsys, monkeypatch):
    from memoirs.engine import lifecycle as lc
    monkeypatch.setattr(lc, "auto_merge_near_duplicates",
                        lambda db, **k: {"merged": 0, "contradictions": 0,
                                         "scanned": 0, "dry_run": k.get("dry_run", False)})
    db = tmp_path / "memoirs.sqlite"
    cli.main(["--db", str(db), "init"])
    capsys.readouterr()
    rc = cli.main(["--db", str(db), "cleanup", "--dry-run"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["dry_run"] is True


def test_main_ask_with_query(tmp_path, capsys, monkeypatch):
    from memoirs.engine import memory_engine as me
    monkeypatch.setattr(me, "assemble_context",
                        lambda db, query, **k: {"query": query, "context": [], "memories": []})
    db = tmp_path / "memoirs.sqlite"
    cli.main(["--db", str(db), "init"])
    capsys.readouterr()
    rc = cli.main(["--db", str(db), "ask", "what do I prefer?"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["query"] == "what do I prefer?"


def test_main_extract_returns_zero(tmp_path, capsys, monkeypatch):
    """`extract` (non-daemon mode) calls into gemma.extract_pending."""
    from memoirs.engine import curator as _curator
    monkeypatch.setattr(_curator, "extract_pending",
                        lambda db, *, limit, reprocess_with_gemma: {"processed": 0})
    db = tmp_path / "memoirs.sqlite"
    cli.main(["--db", str(db), "init"])
    capsys.readouterr()
    rc = cli.main(["--db", str(db), "extract", "--limit", "3"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["processed"] == 0
