"""Try to fix Gemma's false-positive bias via prompt + decoding tweaks.

Hypothesis: Gemma 2 marks unrelated pairs as 'contradictory' because
(a) Gemma 2 has no native system-role and the instruction gets diluted,
(b) temperature 0.1 still allows 'true' to win on ambiguous logits,
(c) lacking few-shot anchors, the model defaults to 'something must be wrong'.

We try 5 variants and re-grade against the 10 known-truth cases.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

GEMMA = Path.home() / ".local/share/memoirs/models/gemma-2-2b-it-Q4_K_M.gguf"

CASES = [
    ("user prefers Python over Go for backend",
     "user switched away from Python in 2025; now uses Go", True, "b"),
    ("user lives in Buenos Aires, Argentina",
     "user is currently based in Madrid, Spain since 2024", True, "b"),
    ("user prefers TDD with pytest",
     "user values comprehensive integration tests", False, None),
    ("memoirs uses SQLite for storage",
     "memoirs uses sqlite-vec for vector search", False, None),
    ("the API runs on port 8283",
     "the FastAPI server now listens on port 9999", True, "b"),
    ("user is allergic to peanuts",
     "user used to be allergic to peanuts but outgrew it", True, "b"),
    ("project deadline is Q4 2025",
     "team uses Linear for issue tracking", False, None),
    ("user prefers TypeScript",
     "user dislikes TypeScript, never uses it", True, None),
    ("memoirs DB version is v3",
     "memoirs DB schema now at v9 after sprint", True, "b"),
    ("user owns a 2018 Mazda",
     "user works as software engineer", False, None),
]

# ── 5 prompt variants ───────────────────────────────────────────────

V1_BASELINE = """Decide if two memories about the same user are CONTRADICTORY.

Output ONE compact JSON line, no markdown, no prose, reason ≤ 8 words:
{{"contradictory": true|false, "winner": "a"|"b"|null, "reason": "<≤8 words>"}}

If contradictory, winner = the more specific OR more recent statement, else null.
If different aspects (not contradictory): contradictory=false, winner=null.

Memory A: "{a}"
Memory B: "{b}"

JSON:"""

V2_EXPLICIT_UNRELATED = """Decide if two memories about the same user are CONTRADICTORY.

CRITICAL: Two memories on UNRELATED topics are NOT contradictory. Only mark contradictory=true when the SAME attribute is stated differently (e.g. "lives in X" vs "lives in Y", "uses tool A" vs "stopped using tool A").

Output ONE compact JSON line, no markdown:
{{"contradictory": true|false, "winner": "a"|"b"|null, "reason": "<≤8 words>"}}

Rules:
- Same topic + incompatible values → contradictory=true. Winner = more specific/recent.
- Different topics or different facets of same topic → contradictory=false, winner=null.
- One statement entails the other → not contradictory.

Memory A: "{a}"
Memory B: "{b}"

JSON:"""

V3_FEW_SHOT = """Decide if two memories about the same user are CONTRADICTORY.

Examples:
A: "user lives in Paris" / B: "user moved to London in 2024" → {{"contradictory":true,"winner":"b","reason":"location changed"}}
A: "user uses Python" / B: "user uses Go" → {{"contradictory":false,"winner":null,"reason":"different languages, both fine"}}
A: "API on port 80" / B: "API on port 443" → {{"contradictory":true,"winner":null,"reason":"can't run both"}}
A: "user owns a Mazda" / B: "user is a doctor" → {{"contradictory":false,"winner":null,"reason":"unrelated topics"}}
A: "DB version 1" / B: "DB version 2" → {{"contradictory":true,"winner":"b","reason":"version superseded"}}

Now grade the new pair. Output ONE JSON line, no markdown:
{{"contradictory": true|false, "winner": "a"|"b"|null, "reason": "<≤8 words>"}}

Memory A: "{a}"
Memory B: "{b}"

JSON:"""

V4_TEMP0_BASELINE = V1_BASELINE  # same prompt, temperature=0.0
V5_FEWSHOT_TEMP0 = V3_FEW_SHOT  # few-shot + temperature=0.0


def gemma_wrap(prompt: str) -> str:
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


STOP = ["<end_of_turn>", "}\n\n", "\n\n\n"]


def parse_loose(text: str) -> dict | None:
    s = (text or "").strip().lstrip("`").strip()
    if s.startswith("json"):
        s = s[4:]
    s = s.lstrip().rstrip("`").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = s.find("{")
    if start == -1:
        return None
    cand = s[start:]
    if cand.count('"') % 2 == 1:
        cand += '"'
    o, c = cand.count("{"), cand.count("}")
    if o > c:
        cand += "}" * (o - c)
    try:
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def grade_pair(parsed, exp_contra, exp_winner):
    if parsed is None:
        return False, "PARSE_FAIL"
    got_c = parsed.get("contradictory")
    got_w = parsed.get("winner")
    contra_ok = (got_c == exp_contra)
    if exp_winner is None:
        win_ok = (got_w in (None, "null", ""))
    else:
        win_ok = (str(got_w).lower() == exp_winner.lower())
    return contra_ok and (win_ok if exp_contra else True), f"c={got_c}/w={got_w}"


def run_variant(llm, name, prompt_tmpl, temperature):
    rows = []
    for i, (a, b, ec, ew) in enumerate(CASES):
        prompt = gemma_wrap(prompt_tmpl.format(a=a, b=b))
        t0 = time.perf_counter()
        out = llm.create_completion(
            prompt=prompt, max_tokens=160, temperature=temperature, stop=STOP,
        )
        dt = (time.perf_counter() - t0) * 1000
        text = (out["choices"][0]["text"] or "").strip()
        parsed = parse_loose(text)
        ok, info = grade_pair(parsed, ec, ew)
        rows.append({
            "case": i, "ms": int(dt), "ok": ok, "info": info,
            "raw": text[:90], "expected": (ec, ew),
        })
    correct = sum(1 for r in rows if r["ok"])
    p50 = sorted(r["ms"] for r in rows)[len(rows) // 2]
    return {"name": name, "correct": correct, "n": len(rows),
            "p50_ms": p50, "rows": rows}


def main():
    if not GEMMA.exists():
        print("Gemma model not found — install via `memoirs models pull gemma-2b`")
        return
    from llama_cpp import Llama
    print("== loading Gemma 2 2B Q4 ...")
    t0 = time.perf_counter()
    llm = Llama(model_path=str(GEMMA), n_ctx=4096, n_gpu_layers=99, verbose=False)
    print(f"   loaded in {time.perf_counter()-t0:.1f}s\n")

    variants = [
        ("V1 baseline (T=0.1)", V1_BASELINE, 0.1),
        ("V2 explicit-unrelated (T=0.1)", V2_EXPLICIT_UNRELATED, 0.1),
        ("V3 few-shot (T=0.1)", V3_FEW_SHOT, 0.1),
        ("V4 baseline T=0.0", V4_TEMP0_BASELINE, 0.0),
        ("V5 few-shot T=0.0", V5_FEWSHOT_TEMP0, 0.0),
    ]

    results = []
    for name, prompt, temp in variants:
        print(f">>> {name} ...")
        r = run_variant(llm, name, prompt, temp)
        results.append(r)
        print(f"    {r['correct']}/{r['n']}  p50={r['p50_ms']}ms")

    print("\n=== SCOREBOARD ===")
    print(f"{'variant':<32s}  {'correct/n':>10s}  {'p50':>7s}")
    for r in results:
        print(f"{r['name']:<32s}  {r['correct']:>2}/{r['n']:<2}        {r['p50_ms']:>4}ms")

    print("\n=== per-case detail (where variants disagree) ===")
    n = len(CASES)
    for i in range(n):
        case_results = [(r["name"], r["rows"][i]) for r in results]
        # only print if at least 2 variants disagree
        oks = {row["ok"] for _, row in case_results}
        if len(oks) > 1:
            a, b, ec, ew = CASES[i]
            print(f"\nCase {i}: A={a!r:.50} B={b!r:.50}")
            print(f"  ground truth: contra={ec} winner={ew}")
            for name, row in case_results:
                mark = "✓" if row["ok"] else "✗"
                print(f"   {mark} {name:<32s}  {row['info']}")

    out = Path(".memoirs/gemma_tune_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nfull report -> {out}")


if __name__ == "__main__":
    main()
