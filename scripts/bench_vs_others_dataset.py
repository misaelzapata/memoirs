"""Reproducible synthetic dataset for the head-to-head other engine bench.

Why hand-crafted instead of LongMemEval?
  * LongMemEval ships ~5GB of conversational data and requires a GitHub
    sign-in to download; embedding it for every engine is expensive and
    its license is restrictive for redistribution.
  * For a head-to-head comparison we only need a *common* corpus that
    every engine can ingest in <30 s and that exercises the four stress
    modes the literature cares about: single-hop, multi-hop, temporal,
    preference. 80 memorias + 20 queries is the sweet spot — large
    enough to make ranking matter, small enough to bench in seconds.

Optional escape hatch: if `~/datasets/longmemeval.jsonl` exists the
runner can use the existing `memoirs.evals.longmemeval_adapter`. This
module owns the synthetic fallback.

Layout (totals = 80 memorias / 20 queries):
  * 8 single-hop  : 8 memorias with a unique keyword + 8 queries.
  * 6 multi-hop   : 12 memorias chained in pairs (entity bridge) + 6
                    queries whose gold lists BOTH memorias of the pair.
  * 4 temporal    : 8 memorias (4 pairs), each pair is the same logical
                    fact at two different timestamps; the latest should
                    win when `as_of` is None.
  * 2 preference  : 2 memorias, durable user preferences.
  * 50 distractors: realistic-looking facts that share NO keywords with
                    any query — they force the ranker to actually rank.

Every memory has a stable string ID (``mem_<bucket>_<slug>``) so test
assertions and gold lists can name them deterministically across runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Memory + Query data classes (engine-agnostic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchMemory:
    """A single memory record shipped to every engine.

    `valid_from` / `valid_to` (ISO-8601 strings) are interpreted by
    bi-temporal engines (memoirs, Zep). Engines without bi-temporal
    semantics (mem0, cognee) ignore them; the latest-wins behaviour is
    approximated by inserting them in chronological order so the most
    recent fact lands last.
    """

    id: str
    type: str
    content: str
    importance: int = 3
    confidence: float = 0.7
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None


@dataclass(frozen=True)
class BenchQuery:
    """A single query case with the gold IDs that satisfy it.

    `category` slices metrics by query type; `as_of` triggers a
    time-travel query when set.
    """

    query: str
    gold_memory_ids: list[str]
    category: str = "single-hop"
    as_of: Optional[str] = None
    notes: str = ""


@dataclass
class BenchDataset:
    memories: list[BenchMemory] = field(default_factory=list)
    queries: list[BenchQuery] = field(default_factory=list)


# ---------------------------------------------------------------------------
# End-to-end conversation dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchConversation:
    """A multi-turn conversation that an engine must process end-to-end.

    Unlike `BenchMemory` (which is a pre-curated atomic fact), this is
    the **raw** input every engine sees in production: a sequence of
    chat messages from which durable memories must be extracted.

    Attributes:
      id: stable identifier (e.g. ``conv_pref_python``).
      messages: chronological list of ``{"role", "content"}`` dicts.
      expected_memories: gold memory contents in natural language —
        adapters that perform extraction should surface these.
    """

    id: str
    messages: list[dict] = field(default_factory=list)
    expected_memories: list[str] = field(default_factory=list)


@dataclass
class BenchSuite:
    """End-to-end suite: raw conversations + queries that target the
    durable memories embedded in those conversations.

    The bench's pipeline cost is measured *here*, not in `BenchDataset`:
    each engine ingests the conversations through ITS OWN extraction +
    consolidation pipeline before ranking. A conv-id can map to many
    expected memories; queries look those up by content.
    """

    conversations: list[BenchConversation] = field(default_factory=list)
    queries: list[BenchQuery] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Anchor timestamps for the temporal cases
# ---------------------------------------------------------------------------

_TS_OLD = "2024-01-01T00:00:00+00:00"
_TS_MID = "2025-06-15T00:00:00+00:00"
_TS_NEW = "2026-04-01T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Single-hop (8) — keyword unique to the gold memory
# ---------------------------------------------------------------------------

_SINGLE_HOP_MEMS: list[BenchMemory] = [
    BenchMemory("mem_sh_lasagna", "fact",
                "user's favorite dinner recipe is lasagna bolognese with bechamel"),
    BenchMemory("mem_sh_dentist", "fact",
                "user has a dentist appointment scheduled at clinic SmileBright next week"),
    BenchMemory("mem_sh_marathon", "decision",
                "user signed up for the Berlin marathon happening on september 28"),
    BenchMemory("mem_sh_canary", "project",
                "the project canary-bench tracks rollout latency for the payments service"),
    BenchMemory("mem_sh_passport", "fact",
                "user's passport number is XR882134 and expires in 2031"),
    BenchMemory("mem_sh_birthday", "fact",
                "user's mother's birthday falls on november 12 every year"),
    BenchMemory("mem_sh_mortgage", "fact",
                "user's monthly mortgage payment is 1842 euros to bankia branch 045"),
    BenchMemory("mem_sh_allergy", "fact",
                "user is severely allergic to shellfish, especially shrimp and lobster"),
]

_SINGLE_HOP_QUERIES: list[BenchQuery] = [
    BenchQuery("what is my favorite dinner recipe?",
               ["mem_sh_lasagna"], "single-hop"),
    BenchQuery("when is my dentist appointment at SmileBright?",
               ["mem_sh_dentist"], "single-hop"),
    BenchQuery("which marathon did I sign up for?",
               ["mem_sh_marathon"], "single-hop"),
    BenchQuery("what does the canary-bench project measure?",
               ["mem_sh_canary"], "single-hop"),
    BenchQuery("what is my passport number?",
               ["mem_sh_passport"], "single-hop"),
    BenchQuery("when is my mother's birthday?",
               ["mem_sh_birthday"], "single-hop"),
    BenchQuery("how much is my monthly mortgage payment?",
               ["mem_sh_mortgage"], "single-hop"),
    BenchQuery("what foods am I allergic to?",
               ["mem_sh_allergy"], "single-hop"),
]


# ---------------------------------------------------------------------------
# Multi-hop (6 queries × 2 memorias = 12 memorias)
#
# Each query bridges two memorias via a shared entity. The gold list
# names BOTH memorias so retrievers that surface only one get penalised.
# ---------------------------------------------------------------------------

_MULTI_HOP_MEMS: list[BenchMemory] = [
    # Pair 1: alice -> storage team -> bigtable deadline
    BenchMemory("mem_mh_alice_lead", "fact",
                "alice is the new tech lead of the storage team since february"),
    BenchMemory("mem_mh_storage_deadline", "fact",
                "the storage team owns the bigtable migration deadline of march 31"),
    # Pair 2: rivendell project -> stack
    BenchMemory("mem_mh_rivendell_join", "project",
                "user joined the rivendell project as principal engineer in q2"),
    BenchMemory("mem_mh_rivendell_stack", "fact",
                "the rivendell project's stack is python 3.12 + duckdb + nats"),
    # Pair 3: bob -> growth team -> tokyo office
    BenchMemory("mem_mh_bob_growth", "fact",
                "bob runs the growth team after the reorg announced in october"),
    BenchMemory("mem_mh_growth_tokyo", "fact",
                "the growth team is fully based out of the tokyo shibuya office"),
    # Pair 4: oncology study -> phase III trial
    BenchMemory("mem_mh_study_drug", "fact",
                "the oncology study is led by dr. mendes and tests molecule MK-9912"),
    BenchMemory("mem_mh_study_phase", "fact",
                "molecule MK-9912 entered a phase III trial in q1 across 14 sites"),
    # Pair 5: carla -> design team -> figma migration
    BenchMemory("mem_mh_carla_design", "fact",
                "carla heads the design team since the merger closed last year"),
    BenchMemory("mem_mh_design_figma", "decision",
                "the design team migrated all source files from sketch to figma in march"),
    # Pair 6: project apollo -> on-call rotation
    BenchMemory("mem_mh_apollo_owner", "project",
                "project apollo is owned by the platform reliability squad"),
    BenchMemory("mem_mh_apollo_oncall", "fact",
                "the platform reliability squad runs a weekly on-call rotation through pagerduty"),
]

_MULTI_HOP_QUERIES: list[BenchQuery] = [
    BenchQuery("when is alice's team's bigtable migration due?",
               ["mem_mh_alice_lead", "mem_mh_storage_deadline"], "multi-hop"),
    BenchQuery("what is the stack of the project I joined as principal engineer?",
               ["mem_mh_rivendell_join", "mem_mh_rivendell_stack"], "multi-hop"),
    BenchQuery("where is the team bob runs based?",
               ["mem_mh_bob_growth", "mem_mh_growth_tokyo"], "multi-hop"),
    BenchQuery("what trial phase is dr. mendes' molecule in?",
               ["mem_mh_study_drug", "mem_mh_study_phase"], "multi-hop"),
    BenchQuery("which design tool did carla's team migrate to?",
               ["mem_mh_carla_design", "mem_mh_design_figma"], "multi-hop"),
    BenchQuery("how does the team that owns project apollo handle on-call?",
               ["mem_mh_apollo_owner", "mem_mh_apollo_oncall"], "multi-hop"),
]


# ---------------------------------------------------------------------------
# Temporal (4 pairs, each pair = same fact at 2 timestamps)
#
# Two queries fire `as_of=<old_ts>` (gold = old version), two fire live
# (gold = new version). For non-temporal engines, we still expect the
# more-recent insert to win because it's added LAST and many engines
# implicitly reward recency.
# ---------------------------------------------------------------------------

_TEMPORAL_MEMS: list[BenchMemory] = [
    # Drink preference: tea -> coffee
    BenchMemory("mem_t_drink_old", "preference",
                "user prefers a cup of tea every afternoon",
                valid_from=_TS_OLD, valid_to=_TS_MID),
    BenchMemory("mem_t_drink_new", "preference",
                "user prefers cold brew coffee every afternoon",
                valid_from=_TS_MID, valid_to=None),
    # Job title: senior eng -> staff eng
    BenchMemory("mem_t_title_old", "fact",
                "user's job title is senior software engineer at acme corp",
                valid_from=_TS_OLD, valid_to=_TS_MID),
    BenchMemory("mem_t_title_new", "fact",
                "user's job title is staff software engineer at acme corp",
                valid_from=_TS_MID, valid_to=None),
    # Home city: barcelona -> madrid
    BenchMemory("mem_t_city_old", "fact",
                "user lives in the gracia district of barcelona",
                valid_from=_TS_OLD, valid_to=_TS_MID),
    BenchMemory("mem_t_city_new", "fact",
                "user lives in the malasana district of madrid",
                valid_from=_TS_MID, valid_to=None),
    # Project lead: dani -> elena (only newest is queried; old is distractor)
    BenchMemory("mem_t_lead_old", "fact",
                "dani is the project lead for the cobalt initiative",
                valid_from=_TS_OLD, valid_to=_TS_MID),
    BenchMemory("mem_t_lead_new", "fact",
                "elena is the project lead for the cobalt initiative",
                valid_from=_TS_MID, valid_to=None),
]

_TEMPORAL_QUERIES: list[BenchQuery] = [
    # 2 queries with as_of (old should win)
    BenchQuery("what does the user prefer to drink in the afternoon?",
               ["mem_t_drink_old"], "temporal", as_of="2024-12-01T00:00:00+00:00"),
    BenchQuery("what is the user's job title?",
               ["mem_t_title_old"], "temporal", as_of="2024-12-01T00:00:00+00:00"),
    # 2 live queries (new should win)
    BenchQuery("which city does the user live in?",
               ["mem_t_city_new"], "temporal", as_of=None),
    BenchQuery("who is the current project lead for the cobalt initiative?",
               ["mem_t_lead_new"], "temporal", as_of=None),
]


# ---------------------------------------------------------------------------
# Preference (2)
# ---------------------------------------------------------------------------

_PREFERENCE_MEMS: list[BenchMemory] = [
    BenchMemory("mem_pref_keyboard", "preference",
                "user dislikes mechanical keyboards and prefers low-profile membrane ones"),
    BenchMemory("mem_pref_meeting", "preference",
                "user refuses meetings before 10am local time and on fridays"),
]

_PREFERENCE_QUERIES: list[BenchQuery] = [
    BenchQuery("what kind of keyboards does the user like?",
               ["mem_pref_keyboard"], "preference"),
    BenchQuery("when does the user not want to be scheduled for meetings?",
               ["mem_pref_meeting"], "preference"),
]


# ---------------------------------------------------------------------------
# Distractors (50) — realistic facts with NO overlap with any query keyword.
#
# Generated as `mem_dist_<idx>` so they sort/dedupe deterministically. Mix
# of tech, ops, and lifestyle topics so embedding-based engines actually
# have to discriminate.
# ---------------------------------------------------------------------------

_DISTRACTOR_TEXTS: list[str] = [
    "the kubernetes scheduler uses node affinity for pod placement",
    "rust borrow checker prevents data races at compile time",
    "graphql subscriptions need a websocket transport layer",
    "react useEffect cleanup runs before re-render and on unmount",
    "tailwind utility classes compose deterministically by source order",
    "typescript generic constraints use the extends keyword",
    "docker layer caching short-circuits when the FROM hash matches",
    "postgres b-tree indexes are good for range scans",
    "ssh agent forwarding lets you hop through bastion hosts safely",
    "dns ttl values trade off propagation lag versus refresh load",
    "tcp keepalive probes fire after a configurable idle window",
    "json web tokens are stateless but require key rotation discipline",
    "rabbitmq topic exchanges route by key pattern with wildcards",
    "kafka partitions guarantee ordering only within a single partition",
    "websocket frames carry a 4-byte mask on the client side",
    "vim macros record every keystroke including timing pauses",
    "tmux panes survive disconnects when the server stays running",
    "git rebase rewrites history so signed commits need re-signing",
    "lsp servers stream diagnostics over jsonrpc on stdio",
    "openssl pkcs12 bundles bind a key with its full cert chain",
    "the espresso machine in the kitchen uses 9 bar pressure",
    "the office printer accepts only legal-size paper after the upgrade",
    "the building elevator goes out of service every wednesday morning",
    "the cafeteria menu rotates on a 14-day cycle",
    "the gym on the third floor closes at 10pm sharp",
    "the parking garage gate uses a magnetic stripe card",
    "weekly all-hands meetings happen on tuesday mornings via zoom",
    "the procurement team requires three quotes for purchases over 5k",
    "expense reports must be filed within 30 days of travel",
    "the company laptop refresh cycle is every 36 months",
    "the wifi password rotates quarterly per security policy",
    "video calls default to mute upon join",
    "calendar invites without an agenda get auto-declined by some folks",
    "lunch is catered every other thursday by the local thai place",
    "the rooftop terrace closes during high winds for safety",
    "city-bike subscriptions are reimbursed by the wellness budget",
    "the metro red line runs every 4 minutes during rush hour",
    "the bookstore on main street holds author signings on saturdays",
    "the local farmers market opens at 7am on sundays",
    "weekend brunch reservations are required at popular spots",
    "the museum on park avenue waives entry fees on the first sunday",
    "the swimming pool closes for maintenance every august",
    "the running track loop measures exactly 400 meters",
    "the bakery sells fresh croissants only between 7 and 9am",
    "the bike repair shop on elm street offers free tune-ups in spring",
    "the cinema downtown shows classic films every wednesday night",
    "the public library waives late fees during national reading week",
    "the train station has lockers available for short-term storage",
    "the waterfront promenade hosts a craft fair every saturday in june",
    "the planetarium offers free shows for kids on the second tuesday",
]


_DISTRACTOR_MEMS: list[BenchMemory] = [
    BenchMemory(f"mem_dist_{i:02d}", "fact", text)
    for i, text in enumerate(_DISTRACTOR_TEXTS)
]


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_dataset() -> BenchDataset:
    """Assemble the full 80-memory / 20-query dataset.

    Memories are returned in a stable order: single-hop, multi-hop,
    temporal (chronological), preferences, distractors. Inserting in
    this order means non-temporal engines see the "newer" temporal
    record last, which approximates latest-wins for them.
    """
    mems: list[BenchMemory] = []
    mems.extend(_SINGLE_HOP_MEMS)
    mems.extend(_MULTI_HOP_MEMS)
    # Sort temporal pairs so old < new — engines without `as_of` will
    # still tend to favor the more recent insert.
    mems.extend(sorted(_TEMPORAL_MEMS, key=lambda m: m.valid_from or ""))
    mems.extend(_PREFERENCE_MEMS)
    mems.extend(_DISTRACTOR_MEMS)
    if len(mems) != 80:
        raise AssertionError(
            f"bench dataset must contain exactly 80 memorias, got {len(mems)}"
        )

    queries: list[BenchQuery] = []
    queries.extend(_SINGLE_HOP_QUERIES)
    queries.extend(_MULTI_HOP_QUERIES)
    queries.extend(_TEMPORAL_QUERIES)
    queries.extend(_PREFERENCE_QUERIES)
    if len(queries) != 20:
        raise AssertionError(
            f"bench dataset must contain exactly 20 queries, got {len(queries)}"
        )

    # Sanity: every gold ID must exist in the corpus.
    known_ids = {m.id for m in mems}
    for q in queries:
        for gid in q.gold_memory_ids:
            if gid not in known_ids:
                raise AssertionError(
                    f"gold id {gid!r} for query {q.query!r} not in corpus"
                )

    return BenchDataset(memories=mems, queries=queries)


# ---------------------------------------------------------------------------
# End-to-end suite builder
# ---------------------------------------------------------------------------
#
# The conversations below embed 2-3 durable facts per thread inside
# realistic multi-turn dialogue. Each memory the engine SHOULD extract
# is recorded under `expected_memories` for both diffing and as a gold
# anchor for query routing (queries match by content/keyword overlap).


_E2E_CONVERSATIONS: list[BenchConversation] = [
    BenchConversation(
        id="conv_pref_python",
        messages=[
            {"role": "user", "content": "hey, working on a new microservice today"},
            {"role": "assistant", "content": "nice — anything in particular blocking you?"},
            {"role": "user", "content": "I just really prefer Python over Go for this kind of work, the typing story in Go feels clunky to me"},
            {"role": "assistant", "content": "fair, Python's ergonomics for prototyping are hard to beat"},
            {"role": "user", "content": "yeah, also I'm based out of the Madrid office now, moved last month"},
            {"role": "assistant", "content": "got it, noted"},
            {"role": "user", "content": "ok, let's wrap up — talk later"},
        ],
        expected_memories=[
            "user prefers Python over Go for backend work",
            "user is based out of the Madrid office",
        ],
    ),
    BenchConversation(
        id="conv_project_atlas",
        messages=[
            {"role": "user", "content": "we just kicked off project Atlas this week"},
            {"role": "assistant", "content": "cool, what's the scope?"},
            {"role": "user", "content": "Atlas is a real-time analytics pipeline for the billing team"},
            {"role": "assistant", "content": "got it"},
            {"role": "user", "content": "I'm the tech lead, reporting to Sandra in finance"},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "deadline is end of Q3"},
        ],
        expected_memories=[
            "project Atlas is a real-time analytics pipeline for the billing team",
            "user is the tech lead of project Atlas",
            "project Atlas deadline is end of Q3",
        ],
    ),
    BenchConversation(
        id="conv_health",
        messages=[
            {"role": "user", "content": "had a checkup this morning"},
            {"role": "assistant", "content": "all good?"},
            {"role": "user", "content": "doctor said I'm severely allergic to peanuts, need to carry an epipen"},
            {"role": "assistant", "content": "important to know, glad you got it confirmed"},
            {"role": "user", "content": "also got cleared to run again, training for the Valencia half marathon in October"},
        ],
        expected_memories=[
            "user is severely allergic to peanuts and carries an epipen",
            "user is training for the Valencia half marathon in October",
        ],
    ),
    BenchConversation(
        id="conv_tooling_decision",
        messages=[
            {"role": "user", "content": "we had a long architecture review yesterday"},
            {"role": "assistant", "content": "what did you decide?"},
            {"role": "user", "content": "team agreed to migrate the data warehouse from Snowflake to DuckDB on S3"},
            {"role": "assistant", "content": "interesting move"},
            {"role": "user", "content": "yeah, cost is the main driver, expected savings around 60%"},
            {"role": "assistant", "content": "makes sense"},
        ],
        expected_memories=[
            "team decided to migrate the data warehouse from Snowflake to DuckDB on S3",
            "expected cost savings from the warehouse migration are around 60 percent",
        ],
    ),
    BenchConversation(
        id="conv_family",
        messages=[
            {"role": "user", "content": "my sister Lucia just had her second kid"},
            {"role": "assistant", "content": "congrats! boy or girl?"},
            {"role": "user", "content": "boy, named Mateo, born March 14"},
            {"role": "assistant", "content": "lovely name"},
            {"role": "user", "content": "we're flying to Lisbon next weekend to meet him"},
        ],
        expected_memories=[
            "user's sister Lucia had a second child, a boy named Mateo born March 14",
            "user is flying to Lisbon next weekend to meet his nephew Mateo",
        ],
    ),
    BenchConversation(
        id="conv_work_schedule",
        messages=[
            {"role": "user", "content": "could we move the daily standup?"},
            {"role": "assistant", "content": "to when?"},
            {"role": "user", "content": "I refuse meetings before 10am, my kid's school dropoff runs till 9:45"},
            {"role": "assistant", "content": "totally reasonable"},
            {"role": "user", "content": "also no meetings on Fridays please, that's my deep-work day"},
        ],
        expected_memories=[
            "user refuses meetings before 10am due to school dropoff",
            "user keeps Fridays meeting-free for deep work",
        ],
    ),
    BenchConversation(
        id="conv_credentials",
        messages=[
            {"role": "user", "content": "remind me of my office wifi setup"},
            {"role": "assistant", "content": "you mentioned it earlier, what specifically?"},
            {"role": "user", "content": "the wifi network is BCN-OFFICE-5G and I'm on desk D-114"},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "the printer code I use is 8821"},
        ],
        expected_memories=[
            "user's office wifi network is BCN-OFFICE-5G",
            "user's office desk is D-114",
            "user's printer code is 8821",
        ],
    ),
    BenchConversation(
        id="conv_books",
        messages=[
            {"role": "user", "content": "finished reading Project Hail Mary last night"},
            {"role": "assistant", "content": "what did you think?"},
            {"role": "user", "content": "loved it, Andy Weir's best work imo"},
            {"role": "assistant", "content": "same"},
            {"role": "user", "content": "next on my list is Three Body Problem by Liu Cixin"},
            {"role": "assistant", "content": "great pick"},
            {"role": "user", "content": "I tend to read sci-fi exclusively these days, fantasy bores me"},
        ],
        expected_memories=[
            "user just finished Project Hail Mary by Andy Weir and loved it",
            "user's next read is Three Body Problem by Liu Cixin",
            "user prefers sci-fi over fantasy",
        ],
    ),
]


_E2E_QUERIES: list[BenchQuery] = [
    # conv_pref_python (2 memories)
    BenchQuery("which programming language does the user prefer for backend work?",
               ["conv_pref_python"], "preference"),
    BenchQuery("which office is the user based out of?",
               ["conv_pref_python"], "single-hop"),
    # conv_project_atlas (3 memories)
    BenchQuery("what is project Atlas?",
               ["conv_project_atlas"], "single-hop"),
    BenchQuery("who is the tech lead of project Atlas?",
               ["conv_project_atlas"], "single-hop"),
    BenchQuery("when is the project Atlas deadline?",
               ["conv_project_atlas"], "single-hop"),
    # conv_health (2 memories)
    BenchQuery("what is the user severely allergic to?",
               ["conv_health"], "single-hop"),
    BenchQuery("which marathon is the user training for?",
               ["conv_health"], "single-hop"),
    # conv_tooling_decision (2 memories)
    BenchQuery("what data warehouse migration did the team decide on?",
               ["conv_tooling_decision"], "single-hop"),
    BenchQuery("what cost savings does the warehouse migration target?",
               ["conv_tooling_decision"], "single-hop"),
    # conv_family (2 memories)
    BenchQuery("what is the name of the user's nephew?",
               ["conv_family"], "single-hop"),
    BenchQuery("when is the user flying to Lisbon?",
               ["conv_family"], "single-hop"),
    # conv_work_schedule (2 memories)
    BenchQuery("when does the user refuse meetings?",
               ["conv_work_schedule"], "preference"),
    BenchQuery("which day of the week is meeting-free?",
               ["conv_work_schedule"], "preference"),
    # conv_credentials (3 memories)
    BenchQuery("what is the user's office wifi network?",
               ["conv_credentials"], "single-hop"),
    BenchQuery("which desk does the user sit at?",
               ["conv_credentials"], "single-hop"),
    BenchQuery("what is the user's printer code?",
               ["conv_credentials"], "single-hop"),
    # conv_books (3 memories)
    BenchQuery("which book did the user just finish?",
               ["conv_books"], "single-hop"),
    BenchQuery("what is next on the user's reading list?",
               ["conv_books"], "single-hop"),
    BenchQuery("what genre of books does the user prefer?",
               ["conv_books"], "preference"),
    # cross-thread sanity
    BenchQuery("what office and language preferences has the user mentioned?",
               ["conv_pref_python"], "multi-hop"),
]


def build_end_to_end_suite() -> BenchSuite:
    """Assemble the end-to-end suite: 8 conversations + 20 queries.

    Gold IDs are CONVERSATION IDs (not memory IDs) because each engine
    extracts its own memory representation. A retrieval is considered a
    hit when at least one memory traces back to the gold conversation —
    adapters return synthetic ``conv_id::idx`` IDs to make the join
    explicit.
    """
    if len(_E2E_CONVERSATIONS) < 8:
        raise AssertionError(
            f"end-to-end suite must have at least 8 conversations, got "
            f"{len(_E2E_CONVERSATIONS)}"
        )
    if len(_E2E_QUERIES) < 20:
        raise AssertionError(
            f"end-to-end suite must have at least 20 queries, got "
            f"{len(_E2E_QUERIES)}"
        )
    known = {c.id for c in _E2E_CONVERSATIONS}
    for q in _E2E_QUERIES:
        for gid in q.gold_memory_ids:
            if gid not in known:
                raise AssertionError(
                    f"end-to-end query {q.query!r} references unknown "
                    f"conversation id {gid!r}"
                )
    return BenchSuite(
        conversations=list(_E2E_CONVERSATIONS),
        queries=list(_E2E_QUERIES),
    )


__all__ = [
    "BenchConversation",
    "BenchDataset",
    "BenchMemory",
    "BenchQuery",
    "BenchSuite",
    "build_dataset",
    "build_end_to_end_suite",
]
