"""LoCoMo retrieval-only eval for memoirs.

LoCoMo (Snap Research): 10 long conversations × ~32 sessions × tens of
turns each, with 1986 QA pairs across 5 categories (single-hop,
multi-hop, open-domain, temporal, adversarial). Each QA's evidence is
the dia_id(s) of the turn(s) containing the answer.

Usage:
    .venv/bin/python scripts/eval_locomo.py [--top-k 10] [--limit 500]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.adapters.memoirs_adapter import MemoirsAdapter
from scripts.bench_vs_others_dataset import BenchMemory, BenchQuery


_CAT_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "open-domain",
    4: "temporal",
    5: "adversarial",
}


def load_locomo(path: Path) -> list[dict]:
    with path.open() as f:
        return json.load(f)


def conv_memories(conv: dict) -> list[BenchMemory]:
    sample_id = conv["sample_id"]
    bms: list[BenchMemory] = []
    convo = conv["conversation"]
    for k in convo:
        if not k.startswith("session_") or k.endswith("_date_time"):
            continue
        turns = convo[k]
        date = convo.get(f"{k}_date_time", "")
        for turn in turns:
            dia_id = turn.get("dia_id")
            if not dia_id:
                continue
            content = f"{turn.get('speaker','?')}: {turn.get('text','').strip()}"
            if date:
                content = f"[{date}] {content}"
            bms.append(BenchMemory(
                id=f"{sample_id}_{dia_id}",
                type="fact",
                content=content,
            ))
    return bms


def conv_queries(conv: dict) -> list[BenchQuery]:
    sample_id = conv["sample_id"]
    out: list[BenchQuery] = []
    for q in conv["qa"]:
        evidence = q.get("evidence")
        if not evidence:
            continue
        gold = [f"{sample_id}_{e}" for e in evidence]
        cat = _CAT_NAMES.get(q.get("category", 0), "unknown")
        out.append(BenchQuery(
            query=q["question"],
            gold_memory_ids=gold,
            category=cat,
        ))
    return out


def eval_one(adapter, queries, *, top_k):
    metrics = defaultdict(lambda: {"mrr": 0.0, "h1": 0.0, "h5": 0.0, "r10": 0.0, "n": 0, "lat": 0.0})
    for q in queries:
        gold = set(q.gold_memory_ids)
        if not gold:
            continue
        t0 = time.perf_counter()
        ids = adapter.query(q, top_k=top_k)
        dt = (time.perf_counter() - t0) * 1000.0
        rr = 0.0
        for rank, mid in enumerate(ids, start=1):
            if mid in gold:
                rr = 1.0 / rank
                break
        c = q.category
        m = metrics[c]
        m["mrr"] += rr
        m["h1"] += 1.0 if (ids[:1] and ids[0] in gold) else 0.0
        m["h5"] += 1.0 if (gold & set(ids[:5])) else 0.0
        m["r10"] += len(gold & set(ids[:top_k])) / len(gold)
        m["n"] += 1
        m["lat"] += dt
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default="/home/misael/datasets/locomo10.json")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of conversations (default: all 10)")
    ap.add_argument("--out", default=str(ROOT / ".memoirs/locomo_v1.json"))
    args = ap.parse_args()

    convs = load_locomo(Path(args.locomo))
    if args.limit:
        convs = convs[:args.limit]

    overall = defaultdict(lambda: {"mrr": 0.0, "h1": 0.0, "h5": 0.0, "r10": 0.0, "n": 0, "lat": 0.0})
    per_conv = []
    t_start = time.perf_counter()
    for ci, conv in enumerate(convs, start=1):
        sid = conv["sample_id"]
        memories = conv_memories(conv)
        queries = conv_queries(conv)
        if not queries:
            print(f"[{ci}/{len(convs)}] {sid}: skipped (no queries)")
            continue
        ad = MemoirsAdapter()
        ad.add_memories(memories)
        m = eval_one(ad, queries, top_k=args.top_k)
        ad.shutdown()
        for c, vals in m.items():
            for k in ("mrr", "h1", "h5", "r10", "n", "lat"):
                overall[c][k] += vals[k]
        per_conv.append({"sample_id": sid, "n_memories": len(memories),
                         "n_queries": sum(v["n"] for v in m.values()),
                         "by_category": {c: {k: v[k] for k in v} for c, v in m.items()}})
        line = " ".join(
            f"{c}:n={vals['n']} mrr={vals['mrr']/max(1,vals['n']):.2f}"
            for c, vals in sorted(m.items())
        )
        print(f"[{ci}/{len(convs)}] {sid}  mems={len(memories)} qs={sum(v['n'] for v in m.values())}  {line}")
    total_dur = time.perf_counter() - t_start

    print("\n=== OVERALL ===")
    print(f"{'category':<13} {'n':>4} {'MRR':>6} {'H@1':>6} {'H@5':>6} {'R@10':>6} {'p50_ms':>8}")
    total_n = 0
    sums = {"mrr": 0.0, "h1": 0.0, "h5": 0.0, "r10": 0.0}
    for c, v in sorted(overall.items()):
        n = max(1, v["n"])
        total_n += v["n"]
        for k in sums:
            sums[k] += v[k]
        avg_lat = v["lat"] / n
        print(f"{c:<13} {v['n']:>4} {v['mrr']/n:>6.3f} {v['h1']/n:>6.3f} {v['h5']/n:>6.3f} {v['r10']/n:>6.3f} {avg_lat:>8.1f}")
    if total_n > 0:
        print(f"{'TOTAL':<13} {total_n:>4} {sums['mrr']/total_n:>6.3f} {sums['h1']/total_n:>6.3f} {sums['h5']/total_n:>6.3f} {sums['r10']/total_n:>6.3f}")
    print(f"\nelapsed: {total_dur:.1f}s")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "overall": {str(c): {k: v[k] for k in v} for c, v in overall.items()},
        "per_conv": per_conv,
        "config": {"prf": (__import__("os").environ.get("MEMOIRS_PRF", "off"))},
    }, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
