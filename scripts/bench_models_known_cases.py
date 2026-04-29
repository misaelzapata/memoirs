"""Side-by-side bench: Qwen vs Phi vs Gemma on hand-crafted "ground-truth" cases.

For each case the correct answer is obvious to a human, so model output can be
graded automatically. We measure: correctness, JSON adherence, latency.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODELS_DIR = Path.home() / ".local/share/memoirs/models"

def _qwen_wrap(p):
    return (
        "<|im_start|>system\nYou output ONLY the requested JSON, no prose.<|im_end|>\n"
        f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
    )


def _qwen3_no_think_wrap(p):
    """Qwen3 / Qwen3.5 / Qwen3.6 default to thinking mode. /no_think disables it."""
    return (
        "<|im_start|>system\nYou output ONLY the requested JSON, no prose. /no_think<|im_end|>\n"
        f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
    )


def _qwen3_think_wrap(p):
    """Allow thinking; rely on parser to skip <think>...</think>."""
    return (
        "<|im_start|>system\nYou output ONLY the requested JSON, no prose.<|im_end|>\n"
        f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
    )


QWEN_STOPS = ["<|im_end|>", "```\n\n", "\n\n\n"]

MODELS = {
    "qwen2.5-3b-Q4": {
        "path": MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf",
        "wrap": _qwen_wrap,
        "stop": QWEN_STOPS,
    },
    "qwen3-4b-2507-Q4": {
        "path": MODELS_DIR / "qwen3-4b-instruct-q4_k_m.gguf",
        "wrap": _qwen3_no_think_wrap,
        "stop": QWEN_STOPS,
        "max_tokens": 200,
    },
    "qwen3.5-4b-Q4": {
        "path": MODELS_DIR / "qwen3.5-4b-q4_k_m.gguf",
        "wrap": _qwen3_no_think_wrap,
        "stop": QWEN_STOPS,
        "max_tokens": 200,
    },
    "qwen3.5-2b-Q4": {
        "path": MODELS_DIR / "qwen3.5-2b-q4_k_m.gguf",
        "wrap": _qwen3_no_think_wrap,
        "stop": QWEN_STOPS,
        "max_tokens": 200,
    },
    "deepseek-r1-distill-qwen-1.5b-Q4": {
        # R1-distill ALWAYS thinks — needs big budget + parser that skips
        # everything before the final JSON block.
        "path": MODELS_DIR / "deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf",
        "wrap": _qwen_wrap,
        "stop": ["<|im_end|>", "\n\n\n"],
        "max_tokens": 1024,
    },
    "phi-3.5-mini-Q4": {
        "path": MODELS_DIR / "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "wrap": lambda p: (
            "<|system|>\nYou output ONLY the requested JSON, no prose.<|end|>\n"
            f"<|user|>\n{p}<|end|>\n<|assistant|>\n"
        ),
        "stop": ["<|end|>", "```\n\n", "\n\n\n"],
    },
    "gemma-2-2b-Q4": {
        "path": MODELS_DIR / "gemma-2-2b-it-Q4_K_M.gguf",
        "wrap": lambda p: (
            "<start_of_turn>user\n" + p + "<end_of_turn>\n<start_of_turn>model\n"
        ),
        "stop": ["<end_of_turn>", "}\n\n", "\n\n\n"],
    },
    "smollm3-3b-Q4": {
        "path": MODELS_DIR / "smollm3-3b-q4_k_m.gguf",
        "wrap": lambda p: (  # SmolLM3 uses standard ChatML-ish, no system role
            f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
        ),
        "stop": ["<|im_end|>", "```\n\n", "\n\n\n"],
    },
}


# ── Task 1: contradiction detection ─────────────────────────────────────
CONTRADICTION_PROMPT = """Decide if two memories about the same user are CONTRADICTORY.

Output ONE compact JSON line, no markdown, no prose, reason ≤ 8 words:
{{"contradictory": true|false, "winner": "a"|"b"|null, "reason": "<≤8 words>"}}

If contradictory, winner = the more specific OR more recent statement, else null.
If different aspects (not contradictory): contradictory=false, winner=null.

Memory A: "{a}"
Memory B: "{b}"

JSON:"""

CONTRADICTION_CASES = [
    # (a, b, expected_contradictory, expected_winner_or_None)
    ("user prefers Python over Go for backend",
     "user switched away from Python in 2025; now uses Go",
     True, "b"),
    ("user lives in Buenos Aires, Argentina",
     "user is currently based in Madrid, Spain since 2024",
     True, "b"),
    ("user prefers TDD with pytest",
     "user values comprehensive integration tests",
     False, None),
    ("memoirs uses SQLite for storage",
     "memoirs uses sqlite-vec for vector search",
     False, None),
    ("the API runs on port 8283",
     "the FastAPI server now listens on port 9999",
     True, "b"),
    ("user is allergic to peanuts",
     "user used to be allergic to peanuts but outgrew it",
     True, "b"),
    ("project deadline is Q4 2025",
     "team uses Linear for issue tracking",
     False, None),
    ("user prefers TypeScript",
     "user dislikes TypeScript, never uses it",
     True, None),  # both could be "winner" — accept None or A or B
    ("memoirs DB version is v3",
     "memoirs DB schema now at v9 after sprint",
     True, "b"),
    ("user owns a 2018 Mazda",
     "user works as software engineer",
     False, None),
]


# ── Task 2: consolidation action ────────────────────────────────────────
CONSOLIDATION_PROMPT = """Decide what to do with a NEW candidate memory given EXISTING similar memories.

Output ONE compact JSON line:
{{"action": "ADD"|"UPDATE"|"MERGE"|"IGNORE", "target_id": "<id_or_null>", "reason": "<≤10 words>"}}

ADD = candidate is new and useful.
UPDATE = candidate refines/replaces a specific neighbor (set target_id).
MERGE = candidate combines well with a neighbor (set target_id).
IGNORE = redundant or low-value (no target_id needed).

NEW CANDIDATE: type={ctype} content="{ccontent}"
NEIGHBORS:
{neighbors}

JSON:"""

CONSOLIDATION_CASES = [
    {
        "ctype": "fact",
        "ccontent": "memoirs uses SQLite for primary storage",
        "neighbors": '1. [m1] fact "memoirs stores data in SQLite" sim=0.91',
        "expected_action": {"MERGE", "IGNORE"},  # near-duplicate → MERGE or IGNORE
        "expected_target": "m1",
    },
    {
        "ctype": "preference",
        "ccontent": "user prefers tabs over spaces",
        "neighbors": '1. [m1] preference "user prefers spaces, 4-wide" sim=0.82',
        "expected_action": {"UPDATE"},  # contradictory preference → UPDATE
        "expected_target": "m1",
    },
    {
        "ctype": "fact",
        "ccontent": "alpha v2 release shipped 2026-04-01",
        "neighbors": "(no similar neighbors)",
        "expected_action": {"ADD"},
        "expected_target": None,
    },
    {
        "ctype": "task",
        "ccontent": "fix the typo in README",
        "neighbors": '1. [m1] task "fix typo in README.md line 42" sim=0.93',
        "expected_action": {"MERGE", "IGNORE"},
        "expected_target": "m1",
    },
    {
        "ctype": "decision",
        "ccontent": "we use PostgreSQL for production",
        "neighbors": '1. [m1] decision "we use SQLite for everything" sim=0.65',
        "expected_action": {"ADD", "UPDATE"},  # different scope; either reasonable
        "expected_target": None,
    },
]


def parse_json_loose(text: str) -> dict | None:
    """Tolerant JSON parser; also skips <think>...</think> blocks emitted by
    reasoning models (Qwen3, DeepSeek-R1-Distill, etc.).
    """
    s = (text or "").strip()
    # Reasoning models: drop everything up to and including </think>
    if "</think>" in s:
        s = s.split("</think>", 1)[1].strip()
    # If <think> is open but never closed within budget, jump to first '{'
    elif s.startswith("<think>"):
        first_brace = s.find("{")
        if first_brace != -1:
            s = s[first_brace:]
    # strip markdown fences
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    s = s.lstrip("﻿").strip()
    if not s:
        return None
    # strict
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # salvage truncated
    start = s.find("{")
    if start != -1:
        cand = s[start:]
        if cand.count('"') % 2 == 1:
            cand += '"'
        opens = cand.count("{"); closes = cand.count("}")
        if opens > closes:
            cand += "}" * (opens - closes)
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                obj["_salvaged"] = True
                return obj
        except Exception:
            pass
    return None


def run_contradiction(llm, conf, cases):
    rows = []
    max_tok = conf.get("max_tokens", 160)
    for i, (a, b, exp_contra, exp_winner) in enumerate(cases):
        prompt = conf["wrap"](CONTRADICTION_PROMPT.format(a=a, b=b))
        t0 = time.perf_counter()
        out = llm.create_completion(
            prompt=prompt, max_tokens=max_tok, temperature=0.1, stop=conf["stop"]
        )
        dt = (time.perf_counter() - t0) * 1000
        text = (out["choices"][0]["text"] or "").strip()
        parsed = parse_json_loose(text)
        if parsed is None:
            verdict = ("?", "?", "PARSE_FAIL")
        else:
            got_contra = parsed.get("contradictory")
            got_winner = parsed.get("winner")
            contra_match = (got_contra == exp_contra)
            if exp_winner is None:
                winner_match = (got_winner is None or got_winner not in ("a", "b", "A", "B"))
            else:
                winner_match = (str(got_winner).lower() == exp_winner.lower())
            ok = contra_match and (winner_match if exp_contra else True)
            verdict = (
                "✓" if ok else "✗",
                "contra=%s/winner=%s" % (got_contra, got_winner),
                "" if ok else f"expected contra={exp_contra} winner={exp_winner}"
            )
        rows.append({
            "case": i,
            "ms": int(dt),
            "ok": verdict[0],
            "got": verdict[1],
            "note": verdict[2],
            "raw": text[:120],
        })
    return rows


def run_consolidation(llm, conf, cases):
    rows = []
    max_tok = conf.get("max_tokens", 160)
    for i, c in enumerate(cases):
        prompt = conf["wrap"](CONSOLIDATION_PROMPT.format(
            ctype=c["ctype"], ccontent=c["ccontent"], neighbors=c["neighbors"]
        ))
        t0 = time.perf_counter()
        out = llm.create_completion(
            prompt=prompt, max_tokens=max_tok, temperature=0.1, stop=conf["stop"]
        )
        dt = (time.perf_counter() - t0) * 1000
        text = (out["choices"][0]["text"] or "").strip()
        parsed = parse_json_loose(text)
        if parsed is None:
            verdict = ("?", "?", "PARSE_FAIL")
        else:
            got_action = (parsed.get("action") or "").upper()
            got_target = parsed.get("target_id")
            action_ok = got_action in c["expected_action"]
            if c["expected_target"] is None:
                target_ok = (got_target in (None, "null", ""))
            else:
                target_ok = (got_target == c["expected_target"])
            ok = action_ok and target_ok
            verdict = (
                "✓" if ok else "✗",
                f"{got_action}/{got_target}",
                "" if ok else f"expected one of {c['expected_action']} target={c['expected_target']}"
            )
        rows.append({
            "case": i,
            "ms": int(dt),
            "ok": verdict[0],
            "got": verdict[1],
            "note": verdict[2],
            "raw": text[:120],
        })
    return rows


def grade(rows):
    n = len(rows)
    correct = sum(1 for r in rows if r["ok"] == "✓")
    parse_fail = sum(1 for r in rows if r["ok"] == "?")
    p50 = sorted(r["ms"] for r in rows)[n // 2] if n else 0
    return {"n": n, "correct": correct, "parse_fail": parse_fail, "p50_ms": p50}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--out", default=".memoirs/bench_known_cases.json")
    args = ap.parse_args()

    pick = list(MODELS.keys()) if args.models == "all" else args.models.split(",")
    results = {}
    for name in pick:
        if name not in MODELS:
            print(f"unknown {name!r}"); continue
        conf = MODELS[name]
        if not conf["path"].exists():
            print(f"== {name}: SKIP (model not found)")
            results[name] = {"skipped": "model file missing"}
            continue
        from llama_cpp import Llama
        print(f"\n== loading {name} ...")
        t0 = time.perf_counter()
        llm = Llama(
            model_path=str(conf["path"]), n_ctx=4096, n_gpu_layers=99, verbose=False,
        )
        print(f"   loaded in {time.perf_counter()-t0:.1f}s")
        contra_rows = run_contradiction(llm, conf, CONTRADICTION_CASES)
        consol_rows = run_consolidation(llm, conf, CONSOLIDATION_CASES)
        results[name] = {
            "contradiction": {
                "grade": grade(contra_rows),
                "rows": contra_rows,
            },
            "consolidation": {
                "grade": grade(consol_rows),
                "rows": consol_rows,
            },
        }

    # ─── render ─────────────────────────────────────────────────────
    print("\n=== KNOWN-CASES SCOREBOARD ===\n")
    print(f"{'model':<22s}  {'contradiction':>20s}  {'consolidation':>20s}")
    print(f"{'':22s}  {'correct/n  parse  p50':>20s}  {'correct/n  parse  p50':>20s}")
    for name, r in results.items():
        if "skipped" in r:
            print(f"{name:<22s}  SKIPPED")
            continue
        g1 = r["contradiction"]["grade"]
        g2 = r["consolidation"]["grade"]
        print(f"{name:<22s}  "
              f"{g1['correct']}/{g1['n']}  fail={g1['parse_fail']}  {g1['p50_ms']:>4d}ms  "
              f"{g2['correct']}/{g2['n']}  fail={g2['parse_fail']}  {g2['p50_ms']:>4d}ms")

    # detail per model
    for name, r in results.items():
        if "skipped" in r:
            continue
        print(f"\n--- {name} contradiction details ---")
        for row in r["contradiction"]["rows"]:
            note = f" — {row['note']}" if row["note"] else ""
            print(f"  [{row['case']:>2}] {row['ok']}  {row['ms']:>4}ms  {row['got']}{note}")
        print(f"\n--- {name} consolidation details ---")
        for row in r["consolidation"]["rows"]:
            note = f" — {row['note']}" if row["note"] else ""
            print(f"  [{row['case']:>2}] {row['ok']}  {row['ms']:>4}ms  {row['got']}{note}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nfull report -> {out_path}")


if __name__ == "__main__":
    main()
