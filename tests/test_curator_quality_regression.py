"""Regression test: ensure the active curator backend doesn't silently
degrade on the four most discriminative known-cases from the model bench.

Marked ``slow`` so it doesn't run on every CI invocation. Run with::

    pytest tests/test_curator_quality_regression.py -m slow

The test loads the configured curator (Qwen3 by default) and grades it on
a 4-case mini-suite that hand-picks failures from the larger
``bench_models_known_cases.py``: a clear contradiction, a clear
non-contradiction (different topics), a temporal supersede, and a
consolidation MERGE.

We assert ``correct >= 3 / 4``. That's defensive — Qwen3-4B scores 4/4 on
this subset in practice. If a code change drops accuracy below the
threshold, the test fails and a human investigates whether the regression
is real or the threshold needs raising.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

CASES = [
    # (a, b, expected_contradictory, expected_winner_or_none, kind)
    ("user prefers Python over Go for backend",
     "user switched away from Python in 2025; now uses Go",
     True, "b", "contradiction"),
    ("memoirs uses sqlite-vec for vector search",
     "user is allergic to peanuts",
     False, None, "unrelated"),
    ("user lives in Paris",
     "user moved to Buenos Aires in 2024",
     True, "b", "temporal_supersede"),
    ("the API runs on port 8283",
     "the FastAPI server now listens on port 9999",
     True, "b", "value_change"),
]

PROMPT = """Decide if two memories about the same user are CONTRADICTORY.

Output ONE compact JSON line, no markdown:
{{"contradictory": true|false, "winner": "a"|"b"|null, "reason": "<≤8 words>"}}

If contradictory, winner = the more specific OR more recent statement, else null.
If different aspects (not contradictory): contradictory=false, winner=null.

Memory A: "{a}"
Memory B: "{b}"

JSON:"""


pytestmark = pytest.mark.slow


def _curator_available() -> bool:
    """Skip the test if no usable curator GGUF + llama-cpp-python is present."""
    try:
        from memoirs.engine.gemma import _have_gemma  # noqa: WPS450
    except ImportError:
        return False
    return _have_gemma()


@pytest.mark.skipif(not _curator_available(), reason="no curator GGUF / llama-cpp-python")
def test_curator_meets_quality_floor(monkeypatch):
    """Active curator must hit ≥3/4 on the discriminative mini-suite."""
    # Opt in to the LLM curator: the project-wide autouse fixture in
    # tests/conftest.py defaults MEMOIRS_GEMMA_CURATOR=off so heuristic-only
    # tests stay deterministic. This test specifically exercises the LLM.
    monkeypatch.setenv("MEMOIRS_GEMMA_CURATOR", "auto")
    from memoirs.engine.gemma import _get_llm, _chat_user_turn, _chat_stops, parse_conflict_response
    from memoirs.config import CURATOR_BACKEND

    llm = _get_llm()

    correct = 0
    failures = []
    t_total = time.perf_counter()
    for a, b, exp_contra, exp_winner, kind in CASES:
        prompt = _chat_user_turn(PROMPT.format(a=a, b=b))
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=200,
            temperature=0.1,
            stop=_chat_stops(),
        )
        text = (out["choices"][0]["text"] or "").strip()
        parsed = parse_conflict_response(text)
        if parsed is None:
            failures.append((kind, "PARSE_FAIL", text[:80]))
            continue
        got_contra = parsed.get("contradictory")
        got_winner = parsed.get("winner")
        contra_ok = (got_contra == exp_contra)
        if exp_winner is None:
            winner_ok = (got_winner is None or
                         str(got_winner).lower() not in ("a", "b"))
        else:
            winner_ok = (str(got_winner).lower() == exp_winner.lower())
        if contra_ok and (winner_ok if exp_contra else True):
            correct += 1
        else:
            failures.append((kind,
                             f"contra={got_contra} winner={got_winner}",
                             f"expected contra={exp_contra} winner={exp_winner}"))
    elapsed = time.perf_counter() - t_total

    assert correct >= 3, (
        f"Curator quality regression: backend={CURATOR_BACKEND} "
        f"scored {correct}/{len(CASES)} on the discriminative mini-suite "
        f"in {elapsed:.1f}s. Expected ≥3/4. Failures: {failures!r}"
    )

    # Belt-and-suspenders: drop a JSON breadcrumb so a human can grok regressions
    # without re-running the test.
    Path(".memoirs").mkdir(exist_ok=True)
    Path(".memoirs/last_curator_regression.json").write_text(json.dumps({
        "backend": CURATOR_BACKEND,
        "correct": correct,
        "total": len(CASES),
        "elapsed_s": round(elapsed, 1),
        "failures": failures,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))


@pytest.mark.skipif(not _curator_available(), reason="no curator GGUF")
def test_curator_returns_parseable_json(monkeypatch):
    """At minimum, the curator must produce parseable JSON for the 4 cases.

    Separates 'JSON format' from 'JSON content correct' so a chat-template
    bug shows up loudly even if accuracy is OK.
    """
    monkeypatch.setenv("MEMOIRS_GEMMA_CURATOR", "auto")
    from memoirs.engine.gemma import _get_llm, _chat_user_turn, _chat_stops, parse_conflict_response

    llm = _get_llm()
    parse_fails = 0
    for a, b, *_ in CASES:
        prompt = _chat_user_turn(PROMPT.format(a=a, b=b))
        out = llm.create_completion(
            prompt=prompt, max_tokens=200, temperature=0.1, stop=_chat_stops(),
        )
        text = (out["choices"][0]["text"] or "").strip()
        if parse_conflict_response(text) is None:
            parse_fails += 1
    assert parse_fails == 0, (
        f"{parse_fails}/{len(CASES)} cases produced unparseable JSON — "
        "chat template / stop tokens / max_tokens budget is wrong for the "
        "active backend."
    )
