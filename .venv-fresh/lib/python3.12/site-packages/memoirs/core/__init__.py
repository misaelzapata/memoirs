"""Core primitives: ids, normalization. Every other layer depends on this."""
from .ids import content_hash, stable_id, utc_now
from .normalize import flatten_content, normalize_role, normalize_timestamp

__all__ = [
    "content_hash",
    "stable_id",
    "utc_now",
    "flatten_content",
    "normalize_role",
    "normalize_timestamp",
]
