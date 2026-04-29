"""Stable id generation + content hashing.

A previous version used \\x1f as separator which can collide if a `part`
contains that byte literally. We now use a length-prefix scheme that cannot
collide regardless of the bytes inside any `part`.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_id(prefix: str, *parts: object) -> str:
    """Deterministic id derived from (prefix, parts...).

    Serialization is collision-safe: each part is encoded as `len:bytes`, so
    `("foo", "bar")` and `("foo\\x1fbar",)` produce different inputs.
    """
    pieces: list[bytes] = []
    for part in parts:
        s = "" if part is None else str(part)
        b = s.encode("utf-8")
        pieces.append(f"{len(b)}:".encode("ascii") + b)
    payload = b"\x00".join(pieces)
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return f"{prefix}_{digest}"


def content_hash(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()
