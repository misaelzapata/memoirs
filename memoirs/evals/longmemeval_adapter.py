"""LongMemEval (2024) → memoirs EvalSuite adapter.

Paper: Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term
Interactive Memory" (2024) — https://arxiv.org/abs/2410.10813
Repo : https://github.com/xiaowu0162/LongMemEval

The official dataset ships as JSONL (one record per query) and weighs
several gigabytes — too large to commit. This adapter consumes a
locally-available dump and produces an `EvalSuite` shaped for our harness.

Expected LongMemEval record (subset we use):

    {
        "question_id":   "Q-0001",
        "question":      "When did Alice mention her birthday?",
        "question_type": "single-session-user|multi-session|temporal-reasoning|...",
        "answer":        "2023-08-12",
        "haystack_sessions": [
            {"session_id": "...", "turns": [{"role": "...", "content": "..."}, ...]},
            ...
        ],
        # Either a list of (session_id, turn_index) tuples or, in some
        # variants, a precomputed list of evidence chunk IDs.
        "answer_session_ids":   [...],   # canonical
        "answer_session_id":    "...",   # some flavors single-string
        # Time-travel queries carry an optional reference timestamp.
        "as_of":                "2024-01-15T12:00:00Z",
    }

Why only a subset? The harness needs:
  * a string query
  * a list of gold memory IDs
  * (optionally) an `as_of` timestamp
  * a category for breakdowns

We MAP `question_type` to our four-bucket taxonomy (single-hop, multi-hop,
temporal, preference) so reports stay comparable across suites:

    single-session-user        → single-hop
    single-session-assistant   → single-hop
    multi-session              → multi-hop
    knowledge-update           → temporal
    temporal-reasoning         → temporal
    preference                 → preference  (LongMemEval calls these
                                              "user-pref" in some splits)

The adapter does NOT seed memoirs with the haystack — that's a separate
"ingest the dataset" step (out of scope for the harness, since each
adapter would have its own ingest path). For now we only emit cases; the
caller decides whether to run them against memoirs (after ingesting the
haystack) or skip the eval. If the file is missing we return a clear
"skip" reason instead of raising — convenient for CI matrices where the
dataset is optional.

Download instructions:
  1. clone https://github.com/xiaowu0162/LongMemEval
  2. follow the dataset README (it points to a HuggingFace mirror)
  3. point this adapter at the unpacked JSONL:

       from memoirs.evals.longmemeval_adapter import load_longmemeval
       suite, info = load_longmemeval("/path/to/longmemeval_oracle.jsonl")
       if suite is None:
           print("skip:", info["reason"])
       else:
           run_eval(db, suite, ...)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .harness import EvalCase, EvalSuite


log = logging.getLogger("memoirs.evals.longmemeval")


# ---------------------------------------------------------------------------
# question_type → harness category
# ---------------------------------------------------------------------------


_CATEGORY_MAP: dict[str, str] = {
    "single-session-user":      "single-hop",
    "single-session-assistant": "single-hop",
    "single-session":           "single-hop",
    "multi-session":            "multi-hop",
    "multihop":                 "multi-hop",
    "knowledge-update":         "temporal",
    "temporal-reasoning":       "temporal",
    "temporal":                 "temporal",
    "user-pref":                "preference",
    "preference":               "preference",
}


def _map_category(qtype: str | None) -> str:
    if not qtype:
        return "single-hop"
    return _CATEGORY_MAP.get(qtype.lower(), "single-hop")


# ---------------------------------------------------------------------------
# Gold ID extraction
#
# LongMemEval has gone through schema iterations. We try the obvious keys
# in priority order and fall back to building IDs from (session_id, turn).
# ---------------------------------------------------------------------------


def _extract_gold_ids(rec: dict[str, Any]) -> list[str]:
    # Direct list of evidence chunk IDs
    if isinstance(rec.get("answer_evidence_ids"), list):
        return [str(x) for x in rec["answer_evidence_ids"]]
    if isinstance(rec.get("evidence_ids"), list):
        return [str(x) for x in rec["evidence_ids"]]

    # Oracle split format: ``haystack_session_ids`` parallel to
    # ``haystack_sessions`` (a list of lists of turn dicts). Gold = the
    # turn IDs ``"<session_id>:<turn_idx>"`` where ``has_answer=true``.
    sids = rec.get("haystack_session_ids")
    sessions = rec.get("haystack_sessions")
    if isinstance(sids, list) and isinstance(sessions, list) and sids and sessions:
        gold: list[str] = []
        for sid, turns in zip(sids, sessions):
            if not isinstance(turns, list):
                continue
            for tidx, turn in enumerate(turns):
                if isinstance(turn, dict) and turn.get("has_answer"):
                    gold.append(f"{sid}:{tidx}")
        if gold:
            return gold

    # Older split: list of session IDs that contain the evidence.
    sids2 = rec.get("answer_session_ids") or rec.get("answer_session_id")
    if isinstance(sids2, str):
        sids2 = [sids2]
    if isinstance(sids2, list) and sids2:
        return [str(x) for x in sids2]

    # Pairs of (session_id, turn_index) — older oracle splits.
    pairs = rec.get("answer_evidence") or rec.get("evidence")
    if isinstance(pairs, list) and pairs:
        out: list[str] = []
        for p in pairs:
            if isinstance(p, dict):
                sid = p.get("session_id") or p.get("sid")
                turn = p.get("turn_index") or p.get("turn")
                if sid is not None:
                    out.append(f"{sid}:{turn}" if turn is not None else str(sid))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append(f"{p[0]}:{p[1]}")
            elif isinstance(p, str):
                out.append(p)
        if out:
            return out

    return []


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def load_longmemeval(
    jsonl_path: str | Path,
    *,
    limit: int | None = None,
    suite_name: str | None = None,
) -> tuple[EvalSuite | None, dict[str, Any]]:
    """Load LongMemEval JSONL and produce an EvalSuite.

    Returns ``(suite, info)``.

    * suite is None when the file does not exist; ``info["reason"]`` is
      a human-readable explanation. Callers SHOULD treat this as a soft
      skip (the dataset is large, not bundled with memoirs).
    * On success, ``info`` carries provenance for the report header
      (path, record count, category histogram).

    `limit` truncates after parsing so quick smoke tests don't iterate
    the full 500-question oracle split.
    """
    p = Path(jsonl_path).expanduser()
    if not p.exists():
        return None, {
            "reason": f"dataset not installed: {p}",
            "hint": (
                "download from https://github.com/xiaowu0162/LongMemEval, "
                "follow the dataset README, point this loader at the "
                "unpacked JSONL file."
            ),
            "path": str(p),
        }

    cases: list[EvalCase] = []
    skipped = 0
    by_cat: dict[str, int] = {}
    # Format auto-detect: official LongMemEval ships a JSON array
    # (longmemeval_oracle.json / longmemeval_s.json), but earlier mirrors
    # used JSONL. Sniff the first non-blank char to pick the parser.
    try:
        with p.open("r", encoding="utf-8") as fh:
            head = fh.read(4096)
            fh.seek(0)
            stripped_head = head.lstrip()
            if stripped_head.startswith("["):
                # JSON array — read whole file once.
                try:
                    records = json.load(fh)
                except json.JSONDecodeError as e:
                    return None, {
                        "reason": f"could not parse {p} as JSON array: {e}",
                        "path": str(p),
                    }
                if not isinstance(records, list):
                    return None, {
                        "reason": f"{p}: expected JSON array, got {type(records).__name__}",
                        "path": str(p),
                    }
                iterable = enumerate(records, start=1)
            else:
                iterable = enumerate(
                    (line.strip() for line in fh), start=1
                )
            for lineno, raw in iterable:
                if isinstance(raw, str):
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError as e:
                        log.warning("longmemeval[%d]: skipping malformed line (%s)", lineno, e)
                        skipped += 1
                        continue
                else:
                    rec = raw
                query = rec.get("question") or rec.get("query")
                if not query:
                    skipped += 1
                    continue
                gold = _extract_gold_ids(rec)
                if not gold:
                    # Without gold IDs precision/recall are undefined — skip.
                    skipped += 1
                    continue
                category = _map_category(rec.get("question_type"))
                as_of = rec.get("as_of") or rec.get("reference_time")
                cases.append(EvalCase(
                    query=str(query),
                    gold_memory_ids=list(gold),
                    category=category,
                    as_of=str(as_of) if as_of else None,
                    notes=str(rec.get("question_id") or ""),
                ))
                by_cat[category] = by_cat.get(category, 0) + 1
                if limit is not None and len(cases) >= limit:
                    break
    except OSError as e:
        return None, {"reason": f"could not read {p}: {e}", "path": str(p)}

    if not cases:
        return None, {
            "reason": f"no usable records in {p} (skipped {skipped})",
            "path": str(p),
        }

    suite = EvalSuite(
        name=suite_name or f"longmemeval:{p.stem}",
        cases=cases,
        description=(
            f"LongMemEval adapter from {p.name}: "
            f"{len(cases)} cases, {skipped} skipped, "
            f"distribution = {by_cat}."
        ),
    )
    info = {
        "path": str(p),
        "n_cases": len(cases),
        "skipped": skipped,
        "by_category": by_cat,
    }
    return suite, info


def is_available(jsonl_path: str | Path) -> bool:
    """Cheap pre-check: does the dataset exist on disk?"""
    return Path(jsonl_path).expanduser().exists()
