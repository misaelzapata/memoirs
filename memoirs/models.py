from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class RawMessage:
    role: str
    content: str
    ordinal: int
    created_at: str | None = None
    external_id: str | None = None
    metadata: JsonDict = field(default_factory=dict)
    raw: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class RawConversation:
    external_id: str
    title: str
    source_kind: str
    source_uri: str
    messages: list[RawMessage]
    created_at: str | None = None
    metadata: JsonDict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Multi-tenant scoping (P0-5 / P3-3 in GAP.md)
# ---------------------------------------------------------------------------
#
# ``Scope`` describes WHO is creating / reading a memory. Local-first defaults
# (``user_id="local"``, ``visibility="private"``) keep the single-user shape
# behaviorally identical to pre-scoping memoirs: every legacy row was written
# by ``local`` and is private. Multi-tenant deployments override these per
# request.
#
# ``ScopeFilter`` is the WHERE-clause companion: it describes which memories
# a caller is allowed (or wants) to see. ``None`` means "no constraint on
# this dimension". An empty set means "match nothing" (callers should treat
# that as 'no rows' rather than 'all rows').


@dataclass(frozen=True)
class Scope:
    """Identity + visibility for a memory operation.

    Attributes
    ----------
    user_id:
        Owner of the operation. Defaults to ``"local"`` so all legacy
        callers and single-user installs continue to write rows that are
        readable by the same single-user retrieval path.
    agent_id:
        Optional agent identifier (e.g. ``"claude-code"``, ``"web"``).
        Lets a single user partition memories by which agent produced
        them — useful for auditing and per-agent retrieval.
    run_id:
        Optional run / session id. Useful for grouping ephemeral memories
        from a single chat session.
    namespace:
        Optional logical bucket (``"work"``, ``"personal"``, project name,
        …). Cheap secondary axis on top of ``user_id``.
    visibility:
        One of ``private`` (default), ``shared``, ``org``, ``public``.
        Drives the ACL check in :mod:`memoirs.engine.acl`.
    """

    user_id: str = "local"
    agent_id: str | None = None
    run_id: str | None = None
    namespace: str | None = None
    visibility: str = "private"


# Sentinel used as the default in engine APIs that take an optional scope.
DEFAULT_SCOPE = Scope()


@dataclass(frozen=True)
class ScopeFilter:
    """WHERE-clause description for retrieval.

    Each field is a *set* (or ``None``) so callers can ask for "any of these
    user_ids", "any of these namespaces", etc. The semantics across fields
    is AND (a row must satisfy every non-None field). ``None`` on a field
    means "no constraint on this dimension" — behaviorally identical to the
    pre-scoping path.
    """

    user_ids: set[str] | None = None
    agent_ids: set[str] | None = None
    namespaces: set[str] | None = None
    visibilities: set[str] | None = None

    def is_empty(self) -> bool:
        """True when every field is ``None`` — i.e. equivalent to no filter."""
        return (
            self.user_ids is None
            and self.agent_ids is None
            and self.namespaces is None
            and self.visibilities is None
        )

    def matches(self, row: dict) -> bool:
        """Return True if ``row`` satisfies every active dimension.

        Pure-Python helper used post-fetch to avoid having to thread the
        filter through every retrieval backend's SQL. Rows missing a column
        (e.g. an older fixture without ``user_id``) fall back to the
        default ``"local"`` so the filter never accidentally drops them.
        """
        if self.user_ids is not None:
            if (row.get("user_id") or "local") not in self.user_ids:
                return False
        if self.agent_ids is not None:
            if row.get("agent_id") not in self.agent_ids:
                return False
        if self.namespaces is not None:
            if row.get("namespace") not in self.namespaces:
                return False
        if self.visibilities is not None:
            if (row.get("visibility") or "private") not in self.visibilities:
                return False
        return True
