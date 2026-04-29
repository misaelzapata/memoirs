"""Head-to-head: Qwen3.5-4B in thinking mode vs no-think mode.

Same 15 known-cases as bench_models_known_cases.py. The only variable is
the system prompt (`/no_think` toggle) and the max_tokens budget.

Hypothesis: thinking buys accuracy at the cost of latency. We measure both
and decide whether to expose it as a per-path config (hot vs sleep cron).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

# Reuse the cases + grader + parser from the other script
from bench_models_known_cases import (
    CONTRADICTION_CASES as CASES,
    CONSOLIDATION_CASES,
    run_contradiction, run_consolidation,
)

GGUF = Path.home() / ".local/share/memoirs/models/qwen3.5-4b-q4_k_m.gguf"


def _wrap_no_think(p):
    return (
        "<|im_start|>system\nYou output ONLY the requested JSON, no prose. /no_think<|im_end|>\n"
        f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
    )


def _wrap_thinking(p):
    """Default Qwen3.5: thinking mode is ON unless /no_think is present.
    System prompt explicitly enables for clarity.
    """
    return (
        "<|im_start|>system\nThink step by step, then output ONLY a JSON line. /think<|im_end|>\n"
        f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
    )


CONFIGS = {
    "no_think": {"wrap": _wrap_no_think, "max_tokens": 200, "stop": ["<|im_end|>", "\n\n\n"]},
    "thinking": {"wrap": _wrap_thinking, "max_tokens": 1500, "stop": ["<|im_end|>", "\n\n\n\n"]},
}


def grade(rows):
    n = len(rows)
    correct = sum(1 for r in rows if r["ok"] == "✓")
    parse_fail = sum(1 for r in rows if r["ok"] == "?")
    p50 = sorted(r["ms"] for r in rows)[n // 2] if n else 0
    p95 = sorted(r["ms"] for r in rows)[int(n * 0.95)] if n else 0
    total_tokens = sum(r["ms"] for r in rows)
    return {"n": n, "correct": correct, "parse_fail": parse_fail,
            "p50_ms": p50, "p95_ms": p95}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".memoirs/bench_thinking.json")
    args = ap.parse_args()

    if not GGUF.exists():
        print(f"GGUF not found at {GGUF}"); return
    from llama_cpp import Llama
    print(f"== loading {GGUF.name} ...")
    t0 = time.perf_counter()
    llm = Llama(model_path=str(GGUF), n_ctx=8192, n_gpu_layers=99, verbose=False)
    print(f"   loaded in {time.perf_counter()-t0:.1f}s\n")

    results = {}
    for cfg_name, conf in CONFIGS.items():
        print(f"\n>>> mode={cfg_name}  (max_tokens={conf['max_tokens']})")
        contra_rows = run_contradiction(llm, conf, CASES)
        consol_rows = run_consolidation(llm, conf, CONSOLIDATION_CASES)
        results[cfg_name] = {
            "contradiction": {"grade": grade(contra_rows), "rows": contra_rows},
            "consolidation": {"grade": grade(consol_rows), "rows": consol_rows},
        }
        gC = results[cfg_name]["contradiction"]["grade"]
        gS = results[cfg_name]["consolidation"]["grade"]
        print(f"    contra {gC['correct']}/{gC['n']}  parse_fail={gC['parse_fail']}  p50={gC['p50_ms']}ms p95={gC['p95_ms']}ms")
        print(f"    consol {gS['correct']}/{gS['n']}  parse_fail={gS['parse_fail']}  p50={gS['p50_ms']}ms p95={gS['p95_ms']}ms")

    # ─── compare ───
    print("\n=== HEAD-TO-HEAD (Qwen3.5-4B) ===")
    print(f"{'mode':<12s}  {'contra':>8s}  {'consol':>8s}  {'p50 c':>7s}  {'p50 s':>7s}  {'p95 c':>7s}  {'p95 s':>7s}")
    for name, r in results.items():
        gC = r["contradiction"]["grade"]; gS = r["consolidation"]["grade"]
        print(f"{name:<12s}  {gC['correct']}/{gC['n']:<2}     {gS['correct']}/{gS['n']:<2}    "
              f"{gC['p50_ms']:>4d}ms  {gS['p50_ms']:>4d}ms  {gC['p95_ms']:>4d}ms  {gS['p95_ms']:>4d}ms")

    # disagreement detail
    print("\n=== Cases where the two modes disagree on contradiction ===")
    for i in range(len(CASES)):
        a = results["no_think"]["contradiction"]["rows"][i]
        b = results["thinking"]["contradiction"]["rows"][i]
        if a["ok"] != b["ok"]:
            tA = CASES[i][0][:55]; tB = CASES[i][1][:55]
            print(f"\nCase {i}: A={tA!r} B={tB!r}")
            print(f"  ground truth: contra={CASES[i][2]} winner={CASES[i][3]}")
            print(f"  no_think  {a['ok']}  {a['ms']:>5}ms  {a['got']}")
            print(f"  thinking  {b['ok']}  {b['ms']:>5}ms  {b['got']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nfull report -> {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
