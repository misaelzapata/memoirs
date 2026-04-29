"""Probe the ``CogneeAdapter`` end-to-end against a live cognee install.

This is a debugging tool that drives a tiny corpus through the
adapter and reports HIT / MISS per query. Use it whenever the bench
reports MRR=0.0 for cognee — if this probe is HIT-ing every gold ID,
the bug is in the bench plumbing or the dataset, not the adapter.

What it does:
  1. (optional) Resets cognee state with ``cognee.prune``.
  2. Constructs ``CogneeAdapter`` (which mirrors ``OPENAI_API_KEY``
     into the LiteLLM env vars cognee needs).
  3. Calls ``adapter.add_memories(...)`` with 5 memorias (subset of
     the bench corpus) and ``adapter.query(...)`` for 3 queries.
  4. Prints HIT / MISS per query and returns non-zero exit if any
     gold ID was missed — handy in CI for quick smoke runs.

Why no raw ``cognee.search`` dump? Cognify is the dominant cost
(~30s per dataset), so we only run it once via the adapter. If the
shape ever changes, switch ``SearchType.CHUNKS`` -> ``CHUNKS_LEXICAL``
in the adapter or extend ``_flatten_search_result``.

Usage:
    export OPENAI_API_KEY=...
    .venv/bin/python scripts/probe_cognee.py [--reset]

NOT part of the test suite — it requires a live OpenAI key and ~60s
of LLM calls.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make `scripts.*` importable regardless of CWD.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Quiet cognee + litellm so the HIT/MISS lines aren't lost in the noise.
logging.getLogger().setLevel(logging.WARNING)
for noisy in ("cognee", "litellm", "httpcore", "httpx", "openai"):
    logging.getLogger(noisy).setLevel(logging.ERROR)


_PROBE_MEMORIES: list[tuple[str, str]] = [
    ("mem_sh_lasagna",
     "user's favorite dinner recipe is lasagna bolognese with bechamel"),
    ("mem_sh_marathon",
     "user signed up for the Berlin marathon happening on september 28"),
    ("mem_sh_passport",
     "user's passport number is XR882134 and expires in 2031"),
    ("mem_sh_allergy",
     "user is severely allergic to shellfish, especially shrimp and lobster"),
    ("mem_pref_keyboard",
     "user dislikes mechanical keyboards and prefers low-profile membrane ones"),
]


_PROBE_QUERIES: list[tuple[str, str]] = [
    ("what is my favorite dinner recipe?", "mem_sh_lasagna"),
    ("what is my passport number?", "mem_sh_passport"),
    ("what kind of keyboards does the user like?", "mem_pref_keyboard"),
]


def _amain_sync(reset: bool) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set; the probe needs LLM access.")
        return 2

    # IMPORTANT: import + construct the adapter FIRST so its
    # ``__init__`` mirrors ``OPENAI_API_KEY`` into the LiteLLM env vars
    # BEFORE any cognee module is touched. Cognee snapshots config at
    # import time; doing it the other way around (prune first, then
    # construct adapter) caused the probe to hang in ``cognify``.
    print("=" * 72)
    print("Adapter end-to-end (CogneeAdapter.add_memories + .query)")
    print("=" * 72)
    from scripts.adapters.cognee_adapter import CogneeAdapter
    from scripts.bench_vs_others_dataset import BenchMemory, BenchQuery

    adapter = CogneeAdapter()
    if not adapter.status.ok:
        print(f"adapter not ready: {adapter.status.reason}")
        return 1

    if reset:
        # Use the same async bridge the adapter uses so we share its
        # event-loop policy and cognee's import is finalized.
        import cognee

        async def _reset():
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)

        try:
            adapter._run(_reset())
            print("[probe] reset cognee state.", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[probe] reset skipped: {e!r}", flush=True)

    mems = [BenchMemory(id=mid, type="fact", content=content)
            for mid, content in _PROBE_MEMORIES]
    adapter.add_memories(mems)
    if not adapter.status.ok:
        print(f"add_memories degraded adapter: {adapter.status.reason}")
        return 1

    hits = 0
    for query, gold in _PROBE_QUERIES:
        ids = adapter.query(BenchQuery(query=query, gold_memory_ids=[gold]))
        ok = "HIT " if gold in ids else "MISS"
        if gold in ids:
            hits += 1
        print(f"  {ok} q={query!r} -> {ids} (gold={gold})")
    adapter.shutdown()
    print(f"\n[probe] adapter HIT {hits}/{len(_PROBE_QUERIES)} gold IDs")
    return 0 if hits == len(_PROBE_QUERIES) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true",
                        help="prune cognee state before ingesting")
    args = parser.parse_args(argv)
    return _amain_sync(reset=args.reset)


if __name__ == "__main__":
    raise SystemExit(main())
