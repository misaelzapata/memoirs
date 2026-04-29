"""
Model JSON-adherence benchmark.

Compares local GGUF models on the contradiction-detection task that has been
giving Gemma trouble (truncated JSON in production logs).

Metrics:
- raw JSON parse rate (no salvage)
- salvage parse rate (using parse_conflict_response)
- avg latency per call
- avg output tokens
- truncation rate (response cut by max_tokens or stop)

Usage:
    python scripts/bench_models.py [--n 20] [--max-tokens 200]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

MODELS_DIR = Path.home() / ".local/share/memoirs/models"

# Test cases: (memory_a, memory_b) tuples covering different patterns.
CASES = [
    ("user prefers Python over Go for backend work",
     "user switched to Go after the 2025 refactor for performance"),
    ("user lives in Buenos Aires, Argentina",
     "user is based in Madrid, Spain"),
    ("memoirs uses sqlite-vec for embeddings",
     "memoirs migrated to lancedb for vector search"),
    ("dark mode is enabled by default",
     "light mode is the default in version 2"),
    ("the API runs on port 8283",
     "the FastAPI server listens on port 8284"),
    ("user prefers TDD with pytest",
     "user values comprehensive integration tests"),
    ("project uses GitHub Actions for CI",
     "project uses GitLab CI runners on self-hosted infra"),
    ("Gemma 2 2B is the default curator model",
     "Qwen2.5 3B replaces Gemma due to JSON adherence"),
    ("user_id defaults to 'local' for single-user",
     "user_id is required for multi-tenant deployments"),
    ("retrieval mode is hybrid by default",
     "default retrieval is now graph-based PPR"),
]

PROMPT_TEMPLATE = """Decide if two memories about the same user are CONTRADICTORY.

Focus on negations / semantic incompatibilities (e.g. "uses Python" vs "stopped using Python").

Output ONE compact JSON line, no markdown, no prose, reason ≤ 8 words:
{{"contradictory": true|false, "winner": "a"|"b"|null, "reason": "<≤8 words>"}}

If contradictory, winner = the more specific OR more recent statement, else null.
If different aspects (not contradictory): contradictory=false, winner=null.

Memory A: "{a}"
Memory B: "{b}"

JSON:"""


def gemma_chat_template(prompt: str) -> str:
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def qwen_chat_template(prompt: str) -> str:
    return (
        "<|im_start|>system\nYou output ONLY the requested JSON, no prose.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def phi_chat_template(prompt: str) -> str:
    return (
        f"<|system|>\nYou output ONLY the requested JSON, no prose.<|end|>\n"
        f"<|user|>\n{prompt}<|end|>\n"
        f"<|assistant|>\n"
    )


MODELS = {
    "gemma-2-2b-Q4": {
        "path": MODELS_DIR / "gemma-2-2b-it-Q4_K_M.gguf",
        "template": gemma_chat_template,
        "stop": ["}\n", "\n\n", "<end_of_turn>", "```"],
    },
    "qwen2.5-3b-Q4": {
        "path": MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf",
        "template": qwen_chat_template,
        "stop": ["}\n", "\n\n", "<|im_end|>", "```"],
    },
    "phi-3.5-mini-Q4": {
        "path": MODELS_DIR / "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "template": phi_chat_template,
        "stop": ["}\n", "\n\n", "<|end|>", "```"],
    },
}


def run_model(name: str, conf: dict, *, n: int, max_tokens: int) -> dict:
    if not conf["path"].exists():
        return {"name": name, "skipped": f"not found: {conf['path'].name}"}

    from llama_cpp import Llama
    from memoirs.engine.gemma import parse_conflict_response

    print(f"\n=== loading {name} ({conf['path'].name}) ===", flush=True)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=str(conf["path"]),
        n_ctx=4096,
        n_gpu_layers=99,
        verbose=False,
    )
    load_s = time.perf_counter() - t0
    print(f"  loaded in {load_s:.1f}s", flush=True)

    raw_ok = 0
    salvage_ok = 0
    salvaged_count = 0
    truncated = 0
    latencies: list[float] = []
    out_tokens: list[int] = []
    samples: list[dict] = []

    cases = CASES * (n // len(CASES) + 1)
    cases = cases[:n]
    for i, (a, b) in enumerate(cases):
        prompt = conf["template"](PROMPT_TEMPLATE.format(a=a, b=b))
        t0 = time.perf_counter()
        out = llm.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.1,
            stop=conf["stop"],
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)

        text = (out.get("choices", [{}])[0].get("text") or "").strip()
        finish = out.get("choices", [{}])[0].get("finish_reason", "?")
        out_tokens.append(out.get("usage", {}).get("completion_tokens", 0))
        if finish == "length":
            truncated += 1

        # Raw parse
        raw_dict = None
        try:
            stripped = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                raw_dict = obj
                raw_ok += 1
        except Exception:
            pass

        # Salvage parse (using our existing parser)
        salv = parse_conflict_response(text)
        if isinstance(salv, dict):
            salvage_ok += 1
            if salv.get("_salvaged"):
                salvaged_count += 1

        if i < 3:  # keep first 3 samples for show
            samples.append({"text": text[:200], "raw_ok": raw_dict is not None,
                            "salvage": salv})

    return {
        "name": name,
        "load_s": load_s,
        "n": n,
        "raw_ok": raw_ok,
        "salvage_ok": salvage_ok,
        "salvaged_count": salvaged_count,
        "truncated": truncated,
        "latency_p50_ms": statistics.median(latencies),
        "latency_p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "avg_out_tokens": statistics.mean(out_tokens),
        "samples": samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="cases per model")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--models", default="all", help="comma list or 'all'")
    args = ap.parse_args()

    pick = list(MODELS.keys()) if args.models == "all" else args.models.split(",")
    results = []
    for name in pick:
        if name not in MODELS:
            print(f"unknown model {name!r}; available: {list(MODELS)}")
            continue
        r = run_model(name, MODELS[name], n=args.n, max_tokens=args.max_tokens)
        results.append(r)

    print("\n=== RESULTS ===")
    print(f"{'model':<20s}  {'raw_ok':>8s}  {'salv_ok':>8s}  {'salvaged':>8s}  {'trunc':>5s}  {'p50ms':>6s}  {'p95ms':>6s}  {'tok':>5s}")
    for r in results:
        if "skipped" in r:
            print(f"{r['name']:<20s}  SKIPPED  ({r['skipped']})")
            continue
        print(
            f"{r['name']:<20s}  "
            f"{r['raw_ok']:>3d}/{r['n']:<3d}  "
            f"{r['salvage_ok']:>3d}/{r['n']:<3d}  "
            f"{r['salvaged_count']:>8d}  "
            f"{r['truncated']:>5d}  "
            f"{r['latency_p50_ms']:>6.0f}  "
            f"{r['latency_p95_ms']:>6.0f}  "
            f"{r['avg_out_tokens']:>5.0f}"
        )

    # save raw results for inspection
    import json as _j
    out_path = Path(".memoirs/bench_models.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_j.dumps(results, indent=2))
    print(f"\nresults saved → {out_path}")

    print("\n=== SAMPLES (first 3 per model) ===")
    for r in results:
        if "skipped" in r:
            continue
        print(f"\n--- {r['name']} ---")
        for i, s in enumerate(r["samples"]):
            ok = "✓" if s["raw_ok"] else ("salv" if isinstance(s.get("salvage"), dict) else "✗")
            print(f"  [{i}] {ok}  {s['text']!r}")


if __name__ == "__main__":
    main()
