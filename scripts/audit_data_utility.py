"""
Phase 5C — Data utility audit (read-only).

Audits the persisted memoirs corpus to determine whether the items are
useful (specific, actionable, durable) signal or generic noise. Reports:

    * Useful-rate over a random sample (heuristic + optional Qwen-as-judge).
    * False-negative rate among rejected candidates (when the column has data).
    * Number of duplicate pairs that survived dedup (cosine similarity >= 0.85).
    * Stale rate (usage_count == 0 AND age > 30d).
    * Distributions over importance, type, age, usage, last_accessed_at.
    * Top / bottom 20 memories by score for sanity checking.

The script ONLY reads from the database (opened via the SQLite read-only URI),
so it cannot accidentally mutate the production store.

Usage:

    python scripts/audit_data_utility.py \\
        --db .memoirs/memoirs.sqlite \\
        --sample 50 \\
        --out .memoirs/data_utility_report.json

Set MEMOIRS_AUDIT_USE_QWEN=on to additionally ask the curator LLM to
classify the same sample (USEFUL / NOISE) and report agreement with the
heuristic. Capped at 50 calls per run for safety.
"""
from __future__ import annotations

import argparse
import array
import json
import math
import os
import random
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

_GENERIC_PATTERNS = [
    re.compile(r"\bthe user\b", re.IGNORECASE),
    re.compile(r"\bthis conversation\b", re.IGNORECASE),
    re.compile(r"\bin summary\b", re.IGNORECASE),
    re.compile(r"\bas discussed\b", re.IGNORECASE),
    re.compile(r"\bas mentioned\b", re.IGNORECASE),
]
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b")
_STRONG_TYPES = {"decision", "preference", "project"}


@dataclass
class HeuristicVerdict:
    label: str  # "useful" | "noise" | "borderline"
    reasons: list[str]
    shape_ok: bool
    has_proper_noun: bool
    no_filler: bool
    type_importance_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


def heuristic_classify(content: str, mem_type: str, importance: int) -> HeuristicVerdict:
    """Classify a memory as useful / noise / borderline using cheap heuristics.

    The heuristic checks four orthogonal signals:

    * length in [30, 300] characters (well-formed shape),
    * at least one proper-noun-like token (specific subject),
    * absence of filler / referential phrases ("the user", "this conversation"),
    * for "strong" types (decision, preference, project) importance must be >= 3.

    A score >= 3 of those four signals counts as USEFUL, exactly 2 as
    BORDERLINE, otherwise NOISE.
    """
    reasons: list[str] = []
    text = (content or "").strip()

    shape_ok = 30 <= len(text) <= 300
    if not shape_ok:
        reasons.append(f"len={len(text)} out of [30,300]")

    has_proper_noun = bool(_PROPER_NOUN_RE.search(text))
    if not has_proper_noun:
        reasons.append("no proper-noun-like token")

    no_filler = not any(p.search(text) for p in _GENERIC_PATTERNS)
    if not no_filler:
        reasons.append("contains filler phrase")

    if mem_type in _STRONG_TYPES:
        type_importance_ok = importance >= 3
        if not type_importance_ok:
            reasons.append(f"{mem_type} with importance={importance} (<3)")
    else:
        # weak types are not penalised on importance, just neutrally ok
        type_importance_ok = True

    score = sum([shape_ok, has_proper_noun, no_filler, type_importance_ok])
    if score >= 3:
        label = "useful"
    elif score == 2:
        label = "borderline"
    else:
        label = "noise"

    return HeuristicVerdict(
        label=label,
        reasons=reasons,
        shape_ok=shape_ok,
        has_proper_noun=has_proper_noun,
        no_filler=no_filler,
        type_importance_ok=type_importance_ok,
    )


# ---------------------------------------------------------------------------
# Embedding / similarity helpers
# ---------------------------------------------------------------------------


def _unpack_embedding(blob: bytes, dim: int) -> list[float]:
    if len(blob) != dim * 4:
        raise ValueError(f"blob size {len(blob)} != dim*4 {dim*4}")
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)


def cosine(a: list[float], b: list[float]) -> float:
    """Plain numpy-free cosine similarity. Returns 0.0 for zero vectors."""
    if len(a) != len(b):
        raise ValueError("dim mismatch")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def find_duplicate_pairs(
    rows: list[dict],
    threshold: float = 0.85,
    *,
    max_pairs: int = 1000,
) -> list[dict]:
    """Brute-force O(N^2) duplicate detection over already-decoded vectors.

    `rows` items must carry: id, content, type, vec.

    Returns up to `max_pairs` pairs sorted by descending similarity.
    """
    pairs: list[dict] = []
    n = len(rows)
    for i in range(n):
        ai = rows[i]
        avec = ai["vec"]
        for j in range(i + 1, n):
            bj = rows[j]
            sim = cosine(avec, bj["vec"])
            if sim >= threshold:
                pairs.append(
                    {
                        "a_id": ai["id"],
                        "b_id": bj["id"],
                        "a_type": ai["type"],
                        "b_type": bj["type"],
                        "sim": round(sim, 4),
                        "a_content": ai["content"][:200],
                        "b_content": bj["content"][:200],
                    }
                )
                if len(pairs) >= max_pairs:
                    break
        if len(pairs) >= max_pairs:
            break
    pairs.sort(key=lambda p: p["sim"], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# DB helpers (read-only)
# ---------------------------------------------------------------------------


def open_ro(path: Path) -> sqlite3.Connection:
    """Open the SQLite DB in read-only URI mode. Refuses to mutate."""
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _table_has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == col for r in rows)


# ---------------------------------------------------------------------------
# Qwen-as-judge (optional)
# ---------------------------------------------------------------------------


_QWEN_PROMPT = (
    "Rate this memory as USEFUL (specific, actionable, durable) or "
    "NOISE (generic, redundant, ephemeral). Memory: {content}. "
    "Answer one word: USEFUL or NOISE."
)


def qwen_classify_batch(samples: list[dict], cap: int = 50) -> list[str | None]:
    """Optional LLM judge. Returns parallel list of "USEFUL" | "NOISE" | None.

    Reuses ``memoirs.engine.gemma._get_llm`` for the curator singleton so we
    do not load a second copy of the model.
    """
    try:
        from memoirs.engine.gemma import _get_llm  # type: ignore
    except Exception as e:  # pragma: no cover - LLM optional
        print(f"[qwen-judge] cannot import _get_llm: {e}", file=sys.stderr)
        return [None] * len(samples)

    try:
        llm = _get_llm()
    except Exception as e:  # pragma: no cover - LLM optional
        print(f"[qwen-judge] cannot load LLM: {e}", file=sys.stderr)
        return [None] * len(samples)

    out: list[str | None] = []
    for idx, s in enumerate(samples):
        if idx >= cap:
            out.append(None)
            continue
        prompt = _QWEN_PROMPT.format(content=(s["content"] or "")[:300])
        try:
            resp = llm(prompt, max_tokens=4, temperature=0.0, stop=["\n"])
            text = (resp.get("choices") or [{}])[0].get("text", "").strip().upper()
        except Exception as e:  # pragma: no cover
            print(f"[qwen-judge] LLM call failed at idx={idx}: {e}", file=sys.stderr)
            out.append(None)
            continue
        if "USEFUL" in text:
            out.append("USEFUL")
        elif "NOISE" in text:
            out.append("NOISE")
        else:
            out.append(None)
    return out


# ---------------------------------------------------------------------------
# Audit pipeline
# ---------------------------------------------------------------------------


def _age_days(created_at: str, now: datetime) -> float:
    try:
        # SQLite stores ISO 8601 with a "+00:00" suffix
        ts = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max(0.0, (now - ts).total_seconds() / 86400.0)


def _bucket_age(days: float) -> str:
    if days <= 7:
        return "0-7"
    if days <= 30:
        return "7-30"
    if days <= 90:
        return "30-90"
    return "90+"


def _bucket_usage(uc: int) -> str:
    if uc <= 0:
        return "0"
    if uc == 1:
        return "1"
    if uc <= 5:
        return "2-5"
    return "6+"


def compute_distributions(rows: list[dict], now: datetime) -> dict:
    importance_hist = Counter()
    type_hist = Counter()
    age_hist = Counter()
    usage_hist = Counter()
    last_accessed_null = 0
    last_accessed_set = 0
    for r in rows:
        importance_hist[int(r["importance"])] += 1
        type_hist[r["type"]] += 1
        age_hist[_bucket_age(_age_days(r["created_at"], now))] += 1
        usage_hist[_bucket_usage(int(r["usage_count"] or 0))] += 1
        if r["last_accessed_at"]:
            last_accessed_set += 1
        else:
            last_accessed_null += 1
    return {
        "importance": {str(k): importance_hist[k] for k in sorted(importance_hist)},
        "type": dict(sorted(type_hist.items())),
        "age_days": {k: age_hist[k] for k in ("0-7", "7-30", "30-90", "90+") if k in age_hist},
        "usage_count": {k: usage_hist[k] for k in ("0", "1", "2-5", "6+") if k in usage_hist},
        "last_accessed_at": {"null": last_accessed_null, "set": last_accessed_set},
    }


def compute_stale(conn: sqlite3.Connection) -> dict:
    total = conn.execute(
        "SELECT COUNT(*) AS c FROM memories "
        "WHERE archived_at IS NULL "
        "AND usage_count = 0 "
        "AND julianday('now') - julianday(created_at) > 30"
    ).fetchone()["c"]
    active = conn.execute(
        "SELECT COUNT(*) AS c FROM memories WHERE archived_at IS NULL"
    ).fetchone()["c"]
    by_type_rows = conn.execute(
        "SELECT type, COUNT(*) AS c FROM memories "
        "WHERE archived_at IS NULL "
        "AND usage_count = 0 "
        "AND julianday('now') - julianday(created_at) > 30 "
        "GROUP BY type"
    ).fetchall()
    return {
        "total": int(total),
        "rate": round(total / active, 4) if active else 0.0,
        "by_type": {r["type"]: int(r["c"]) for r in by_type_rows},
        "active_total": int(active),
    }


def evaluate_sample(rows: list[dict], use_qwen: bool) -> dict:
    """Run heuristic classifier (and optional LLM judge) over `rows`."""
    verdicts: list[dict] = []
    label_counter: Counter[str] = Counter()
    for r in rows:
        verdict = heuristic_classify(r["content"] or "", r["type"], int(r["importance"]))
        label_counter[verdict.label] += 1
        verdicts.append(
            {
                "id": r["id"],
                "type": r["type"],
                "importance": int(r["importance"]),
                "score": float(r["score"] or 0.0),
                "content": (r["content"] or "")[:200],
                "verdict": verdict.to_dict(),
            }
        )
    qwen_block = {"enabled": False, "agreement_with_heuristic": None, "labels": []}
    if use_qwen:
        labels = qwen_classify_batch(rows, cap=50)
        agree = 0
        considered = 0
        for v, lab in zip(verdicts, labels):
            if lab is None:
                continue
            considered += 1
            heur_useful = v["verdict"]["label"] == "useful"
            qwen_useful = lab == "USEFUL"
            if heur_useful == qwen_useful:
                agree += 1
        qwen_block = {
            "enabled": True,
            "n_judged": considered,
            "agreement_with_heuristic": (
                round(agree / considered, 4) if considered else None
            ),
            "labels": labels,
        }

    n = len(rows) or 1
    return {
        "n_evaluated": len(rows),
        "useful": label_counter["useful"],
        "noise": label_counter["noise"],
        "borderline": label_counter["borderline"],
        "useful_rate": round(label_counter["useful"] / n, 4),
        "noise_rate": round(label_counter["noise"] / n, 4),
        "verdicts": verdicts,
        "qwen_judge": qwen_block,
    }


def evaluate_false_negatives(conn: sqlite3.Connection, sample_n: int) -> dict:
    """Re-classify rejected candidates with the heuristic. Useful=true => false negative."""
    if not _table_has_column(conn, "memory_candidates", "status"):
        return {"available": False, "reason": "no status column"}
    rows = conn.execute(
        "SELECT id, type, content, importance, status, rejection_reason, created_at "
        "FROM memory_candidates WHERE status = 'rejected' "
        "ORDER BY RANDOM() LIMIT ?",
        (sample_n,),
    ).fetchall()
    if not rows:
        return {
            "available": True,
            "n_evaluated": 0,
            "n_useful_rejected": 0,
            "false_neg_rate": 0.0,
            "samples": [],
            "note": "no rejected candidates in DB",
        }
    samples = []
    n_useful_rejected = 0
    for r in rows:
        verdict = heuristic_classify(r["content"] or "", r["type"], int(r["importance"]))
        if verdict.label == "useful":
            n_useful_rejected += 1
        samples.append(
            {
                "id": r["id"],
                "type": r["type"],
                "importance": int(r["importance"]),
                "rejection_reason": r["rejection_reason"],
                "content": (r["content"] or "")[:200],
                "verdict": verdict.to_dict(),
            }
        )
    n = len(rows)
    return {
        "available": True,
        "n_evaluated": n,
        "n_useful_rejected": n_useful_rejected,
        "false_neg_rate": round(n_useful_rejected / n, 4) if n else 0.0,
        "samples": samples,
    }


def collect_embedding_rows(
    conn: sqlite3.Connection,
    *,
    limit: int | None = None,
) -> list[dict]:
    """Read active memories alongside their embedding vectors."""
    sql = (
        "SELECT m.id AS id, m.type AS type, m.content AS content, "
        "       e.dim AS dim, e.embedding AS embedding "
        "FROM memories m JOIN memory_embeddings e ON e.memory_id = m.id "
        "WHERE m.archived_at IS NULL"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows: list[dict] = []
    for r in conn.execute(sql):
        try:
            vec = _unpack_embedding(r["embedding"], int(r["dim"]))
        except Exception:
            continue
        rows.append({"id": r["id"], "type": r["type"], "content": r["content"] or "", "vec": vec})
    return rows


def run_audit(
    db_path: Path,
    sample_n: int,
    rejected_sample_n: int,
    dup_threshold: float,
    dup_subset: int,
    seed: int,
    use_qwen: bool,
) -> dict:
    rng = random.Random(seed)
    conn = open_ro(db_path)
    try:
        active_count = conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE archived_at IS NULL"
        ).fetchone()["c"]

        active_rows = conn.execute(
            "SELECT id, type, content, importance, confidence, score, "
            "       usage_count, last_accessed_at, created_at "
            "FROM memories WHERE archived_at IS NULL"
        ).fetchall()
        active_dicts = [dict(r) for r in active_rows]

        # Sample for heuristic + qwen
        sample = rng.sample(active_dicts, min(sample_n, len(active_dicts)))
        sample_block = evaluate_sample(sample, use_qwen=use_qwen)

        # Distributions over the full active set
        now = datetime.now(timezone.utc)
        distributions = compute_distributions(active_dicts, now)

        # Stale (DB-side aggregates)
        stale = compute_stale(conn)

        # False negatives (rejected candidates)
        false_neg = evaluate_false_negatives(conn, rejected_sample_n)

        # Duplicates: cap on a subset for runtime control
        emb_rows = collect_embedding_rows(conn, limit=None)
        if dup_subset and len(emb_rows) > dup_subset:
            emb_rows = rng.sample(emb_rows, dup_subset)
        dup_pairs = find_duplicate_pairs(emb_rows, threshold=dup_threshold)
        duplicates = {
            "subset_size": len(emb_rows),
            "threshold": dup_threshold,
            "n_pairs_above_threshold": len(dup_pairs),
            "samples": dup_pairs[:5],
        }

        # Top / bottom by score
        top_rows = conn.execute(
            "SELECT id, type, content, importance, score, usage_count, created_at "
            "FROM memories WHERE archived_at IS NULL "
            "ORDER BY score DESC, importance DESC LIMIT 20"
        ).fetchall()
        bottom_rows = conn.execute(
            "SELECT id, type, content, importance, score, usage_count, created_at "
            "FROM memories WHERE archived_at IS NULL "
            "ORDER BY score ASC, importance ASC LIMIT 20"
        ).fetchall()

        def _row_dict(r: sqlite3.Row) -> dict:
            d = dict(r)
            d["content"] = (d.get("content") or "")[:200]
            return d

        report = {
            "timestamp": now.isoformat(),
            "db_path": str(db_path),
            "corpus_size": active_count,
            "sample": {
                k: v
                for k, v in sample_block.items()
                if k != "verdicts" and k != "qwen_judge"
            },
            "sample_verdicts": sample_block["verdicts"],
            "qwen_judge": sample_block["qwen_judge"],
            "false_negatives": false_neg,
            "duplicates": duplicates,
            "stale": stale,
            "distributions": distributions,
            "top_memories": [_row_dict(r) for r in top_rows],
            "bottom_memories": [_row_dict(r) for r in bottom_rows],
        }
        return report
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_summary(report: dict) -> None:
    s = report["sample"]
    dup = report["duplicates"]
    stale = report["stale"]
    fn = report["false_negatives"]
    print()
    print("=" * 72)
    print("Memoirs data utility audit — executive summary")
    print("=" * 72)
    print(f"Corpus active: {report['corpus_size']}")
    print(
        f"Sample {s['n_evaluated']}: "
        f"useful={s['useful']} ({s['useful_rate']*100:.1f}%) "
        f"borderline={s['borderline']} "
        f"noise={s['noise']} ({s['noise_rate']*100:.1f}%)"
    )
    if report["qwen_judge"].get("enabled"):
        agr = report["qwen_judge"].get("agreement_with_heuristic")
        print(
            f"Qwen judge: n_judged={report['qwen_judge'].get('n_judged')} "
            f"agreement={agr}"
        )
    print(
        f"Duplicates >= {dup['threshold']}: {dup['n_pairs_above_threshold']} pairs "
        f"(over subset {dup['subset_size']})"
    )
    print(
        f"Stale (uc=0, age>30d): {stale['total']} / {stale['active_total']} "
        f"({stale['rate']*100:.1f}%)"
    )
    if fn.get("available"):
        if fn.get("n_evaluated"):
            print(
                f"False negatives in rejected candidates: "
                f"{fn['n_useful_rejected']} / {fn['n_evaluated']} "
                f"({fn['false_neg_rate']*100:.1f}%)"
            )
        else:
            print("False negatives: no rejected candidates in DB")
    else:
        print(f"False negatives: skipped ({fn.get('reason')})")
    print()
    print("Top noise samples (heuristic):")
    noises = [v for v in report["sample_verdicts"] if v["verdict"]["label"] == "noise"]
    for v in noises[:3]:
        print(f"  - [{v['type']}] {v['content']}")
        print(f"      reasons: {', '.join(v['verdict']['reasons'])}")
    if not noises:
        print("  (none)")
    print()
    print("Top duplicate pairs:")
    for p in dup["samples"][:3]:
        print(f"  - sim={p['sim']:.3f} {p['a_id']} <-> {p['b_id']}")
        print(f"      A: {p['a_content']}")
        print(f"      B: {p['b_content']}")
    if not dup["samples"]:
        print("  (none above threshold)")
    print()
    print("Distributions:")
    for k, v in report["distributions"].items():
        print(f"  {k}: {v}")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 5C — read-only audit of memoirs data utility.",
    )
    parser.add_argument("--db", type=Path, required=True, help="Path to memoirs.sqlite (read-only).")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the JSON report.")
    parser.add_argument("--sample", type=int, default=50, help="N random active memories to score.")
    parser.add_argument(
        "--rejected-sample", type=int, default=30,
        help="N rejected candidates to score for false negatives."
    )
    parser.add_argument(
        "--dup-threshold", type=float, default=0.85, help="Cosine similarity threshold."
    )
    parser.add_argument(
        "--dup-subset", type=int, default=600,
        help="Cap on items considered for the O(N^2) duplicate scan (0 = all)."
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 2

    use_qwen = os.environ.get("MEMOIRS_AUDIT_USE_QWEN", "").lower() in {"1", "on", "true", "yes"}
    report = run_audit(
        db_path=args.db,
        sample_n=args.sample,
        rejected_sample_n=args.rejected_sample,
        dup_threshold=args.dup_threshold,
        dup_subset=args.dup_subset,
        seed=args.seed,
        use_qwen=use_qwen,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if not args.quiet:
        _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
