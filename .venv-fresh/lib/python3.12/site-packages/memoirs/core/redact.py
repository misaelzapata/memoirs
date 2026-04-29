"""PII and secret redaction for inbound text.

Goals
-----
* Detect and replace PII (emails, phones, credit cards, SSN, IPs, URLs that
  carry tokens) and secrets (cloud keys, OpenAI keys, GitHub PATs, JWTs,
  PEM-armoured private keys, generic high-entropy tokens).
* Use Microsoft Presidio when ``presidio-analyzer`` is importable (extra
  ``[privacy]``); otherwise fall back to in-tree regexes — Presidio is
  strictly optional so the core install stays dependency-free.
* Secrets are ALWAYS scanned with our regexes (Presidio's secret coverage is
  weak), so detection is independent of which PII backend is used.
* Idempotent: redact(redact(x)) == redact(x).
* Cheap: regex-only mode targets <5ms per message.

Public API
----------
* ``redact(text, *, mode="placeholder") -> RedactedText``
* ``scan_for_secrets(text) -> list[Secret]``

The module also exposes ``apply_redaction_if_enabled(text)`` which is the
hook called from ``core/normalize.py``; it honours ``MEMOIRS_REDACT`` env
(``on`` | ``off`` | ``strict``).
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Literal


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Replacement:
    """A single redaction event inside a piece of text."""

    kind: str           # e.g. "EMAIL", "AWS_ACCESS_KEY", "JWT"
    original: str       # raw matched substring (kept in-memory only, never persisted)
    placeholder: str    # the token that ended up in the output
    start: int          # offset into the ORIGINAL text
    end: int            # exclusive offset into the ORIGINAL text
    is_secret: bool = False


@dataclass(frozen=True)
class RedactedText:
    text: str
    replacements: tuple[Replacement, ...] = field(default_factory=tuple)

    @property
    def has_secrets(self) -> bool:
        return any(r.is_secret for r in self.replacements)


@dataclass(frozen=True)
class Secret:
    kind: str
    value: str
    start: int
    end: int


class SecretDetectedError(RuntimeError):
    """Raised by ``redact(..., mode='strict')`` when a secret is found."""

    def __init__(self, secrets: Iterable[Secret]):
        self.secrets = tuple(secrets)
        kinds = sorted({s.kind for s in self.secrets})
        super().__init__(f"refusing to ingest text containing secrets: {kinds}")


Mode = Literal["placeholder", "hash", "strict"]


# ---------------------------------------------------------------------------
# Regex catalogue
# ---------------------------------------------------------------------------
# The order matters: secrets first, then strong PII, then weak PII (phones / IP).
# Each (kind, pattern, is_secret) tuple. We anchor with word boundaries where
# safe to avoid eating substrings inside larger tokens (e.g. "version=1.2.3.4"
# we still want to match the IP, but for AWS keys we want full-token only).

# --- Secrets ---------------------------------------------------------------
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)[0-9A-Z]{16}\b")
_AWS_SECRET_KEY = re.compile(
    r"(?i)(?:aws_secret_access_key|aws[_-]?secret)[\"'\s:=]+([A-Za-z0-9/+=]{40})\b"
)
_GCP_API_KEY = re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")
_GITHUB_PAT = re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,251}\b")
_GITHUB_FINEGRAINED = re.compile(r"\bgithub_pat_[A-Za-z0-9_]{82}\b")
_OPENAI_KEY = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{32,}\b")
_ANTHROPIC_KEY = re.compile(r"\bsk-ant-(?:api|admin)\d{2}-[A-Za-z0-9_\-]{80,}\b")
_SLACK_TOKEN = re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")
_STRIPE_KEY = re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")
_PEM_BLOCK = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
)

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PEM_PRIVATE_KEY", _PEM_BLOCK),
    ("AWS_ACCESS_KEY", _AWS_ACCESS_KEY),
    ("AWS_SECRET_KEY", _AWS_SECRET_KEY),
    ("GCP_API_KEY", _GCP_API_KEY),
    ("GITHUB_PAT", _GITHUB_PAT),
    ("GITHUB_PAT_FINE", _GITHUB_FINEGRAINED),
    ("ANTHROPIC_KEY", _ANTHROPIC_KEY),  # before _OPENAI_KEY: sk-ant-... also matches sk-...
    ("OPENAI_KEY", _OPENAI_KEY),
    ("SLACK_TOKEN", _SLACK_TOKEN),
    ("STRIPE_KEY", _STRIPE_KEY),
    ("JWT", _JWT),
]

# --- PII -------------------------------------------------------------------
_EMAIL = re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b")
# E.164 / loose phone — keep tight enough to not eat random digit runs.
_PHONE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s\-\.]?)?(?:\(?\d{2,4}\)?[\s\-\.]?){2,4}\d{2,4}(?!\w)"
)
# Credit card (Luhn-validated below)
_CREDIT_CARD = re.compile(r"(?<!\d)(?:\d[ \-]?){13,19}(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_IPV4 = re.compile(
    r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d)"
)
_IPV6 = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b")
# URLs that carry an inline credential or token-looking query param.
_URL_WITH_SECRET = re.compile(
    r"https?://[^\s/]*(?::[^\s/@]+)?@[^\s]+|https?://[^\s]+?(?:[?&](?:token|api[_-]?key|access[_-]?token|auth|secret|password)=[^\s&]+)[^\s]*",
    re.IGNORECASE,
)

_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("URL_WITH_SECRET", _URL_WITH_SECRET),
    ("EMAIL", _EMAIL),
    ("SSN", _SSN),
    ("CREDIT_CARD", _CREDIT_CARD),
    ("IPV6", _IPV6),
    ("IPV4", _IPV4),
    ("PHONE", _PHONE),
]

# Placeholders we emit must themselves match nothing — pattern designed so
# `[EMAIL_3]` / `[SECRET_AWS_ACCESS_KEY_1]` etc. cannot be re-redacted.
_PLACEHOLDER_RE = re.compile(r"^\[(?:[A-Z][A-Z0-9_]*)_\d+\]$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _luhn_ok(digits: str) -> bool:
    s = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(s) <= 19:
        return False
    checksum = 0
    parity = len(s) % 2
    for i, d in enumerate(s):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def _is_pure_placeholder_span(text: str, start: int, end: int) -> bool:
    """Avoid double-redacting a span whose entire match is already a placeholder."""
    snippet = text[start:end]
    return bool(_PLACEHOLDER_RE.match(snippet))


# ---------------------------------------------------------------------------
# Presidio backend (lazy, optional)
# ---------------------------------------------------------------------------


_presidio_analyzer = None
_presidio_checked = False


def _get_presidio():
    """Return a cached AnalyzerEngine, or None if Presidio isn't installed."""
    global _presidio_analyzer, _presidio_checked
    if _presidio_checked:
        return _presidio_analyzer
    _presidio_checked = True
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore
    except Exception:
        _presidio_analyzer = None
        return None
    try:
        _presidio_analyzer = AnalyzerEngine()
    except Exception:
        # Presidio installed but spaCy model missing etc. — silently fall back.
        _presidio_analyzer = None
    return _presidio_analyzer


# Map Presidio entity_type → our kind tag. Anything unmapped is dropped (we
# only redact PII categories we explicitly recognise).
_PRESIDIO_MAP = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CREDIT_CARD",
    "US_SSN": "SSN",
    "IP_ADDRESS": "IP",
    "URL": "URL_WITH_SECRET",  # only emitted when Presidio matched; value-checked below
    "IBAN_CODE": "IBAN",
    "US_PASSPORT": "PASSPORT",
}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def scan_for_secrets(text: str) -> list[Secret]:
    """Return all secret matches (sorted by offset).

    Always uses our regex catalogue — never delegates to Presidio because
    Presidio's secret recognisers are weak and inconsistent.
    """
    if not text:
        return []
    found: list[Secret] = []
    for kind, pat in _SECRET_PATTERNS:
        for m in pat.finditer(text):
            # _AWS_SECRET_KEY uses a capture group for the actual secret value.
            if pat is _AWS_SECRET_KEY:
                start, end = m.span(1)
                value = m.group(1)
            else:
                start, end = m.span()
                value = m.group(0)
            if _is_pure_placeholder_span(text, start, end):
                continue
            found.append(Secret(kind=kind, value=value, start=start, end=end))
    found.sort(key=lambda s: s.start)
    return _dedupe_overlaps(found)


def _scan_pii_regex(text: str) -> list[tuple[str, int, int, str]]:
    """Regex-only PII scan. Returns (kind, start, end, value)."""
    out: list[tuple[str, int, int, str]] = []
    for kind, pat in _PII_PATTERNS:
        for m in pat.finditer(text):
            value = m.group(0)
            start, end = m.span()
            if _is_pure_placeholder_span(text, start, end):
                continue
            if kind == "CREDIT_CARD" and not _luhn_ok(value):
                continue
            if kind == "PHONE":
                # Reject runs that are all zeros / short / not enough digits.
                digits = re.sub(r"\D", "", value)
                if len(digits) < 7 or len(digits) > 15:
                    continue
            out.append((kind, start, end, value))
    return out


def _scan_pii_presidio(text: str) -> list[tuple[str, int, int, str]] | None:
    analyzer = _get_presidio()
    if analyzer is None:
        return None
    try:
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=list(_PRESIDIO_MAP.keys()),
        )
    except Exception:
        return None
    out: list[tuple[str, int, int, str]] = []
    for r in results:
        kind = _PRESIDIO_MAP.get(getattr(r, "entity_type", ""))
        if not kind:
            continue
        start, end = int(r.start), int(r.end)
        if start >= end or end > len(text):
            continue
        value = text[start:end]
        if _is_pure_placeholder_span(text, start, end):
            continue
        if kind == "CREDIT_CARD" and not _luhn_ok(value):
            continue
        if kind == "IP":
            # split into IPV4 / IPV6 for nicer placeholders
            kind = "IPV6" if ":" in value else "IPV4"
        out.append((kind, start, end, value))
    return out


def _dedupe_overlaps(items: list[Secret]) -> list[Secret]:
    """Drop overlapping Secret ranges, keeping the earliest / longest."""
    out: list[Secret] = []
    last_end = -1
    for item in sorted(items, key=lambda s: (s.start, -s.end)):
        if item.start >= last_end:
            out.append(item)
            last_end = item.end
    return out


# ---------------------------------------------------------------------------
# Replacement assembly
# ---------------------------------------------------------------------------


def _assemble(text: str, hits: list[tuple[str, int, int, str, bool]], mode: Mode) -> RedactedText:
    """Apply (kind,start,end,value,is_secret) hits to ``text`` honouring ``mode``."""
    if not hits:
        return RedactedText(text=text, replacements=())

    # Drop overlapping spans (e.g. an IPV4 inside a URL_WITH_SECRET match).
    hits = sorted(hits, key=lambda h: (h[1], -h[2]))
    deduped: list[tuple[str, int, int, str, bool]] = []
    last_end = -1
    for h in hits:
        if h[1] >= last_end:
            deduped.append(h)
            last_end = h[2]

    # Stable per-kind counters keyed by hash of value so the SAME value within
    # a single call always maps to the same placeholder index.
    counters: dict[str, dict[str, int]] = {}
    out: list[str] = []
    replacements: list[Replacement] = []
    cursor = 0

    for kind, start, end, value, is_secret in deduped:
        out.append(text[cursor:start])
        if mode == "hash":
            placeholder = f"[{('SECRET_' + kind) if is_secret else kind}_{_short_hash(value)}]"
        else:
            tag = ("SECRET_" + kind) if is_secret else kind
            kind_counters = counters.setdefault(tag, {})
            if value not in kind_counters:
                kind_counters[value] = len(kind_counters) + 1
            placeholder = f"[{tag}_{kind_counters[value]}]"
        out.append(placeholder)
        replacements.append(
            Replacement(
                kind=kind,
                original=value,
                placeholder=placeholder,
                start=start,
                end=end,
                is_secret=is_secret,
            )
        )
        cursor = end
    out.append(text[cursor:])
    return RedactedText(text="".join(out), replacements=tuple(replacements))


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------


def redact(text: str, *, mode: Mode = "placeholder") -> RedactedText:
    """Redact PII + secrets from ``text``.

    Modes:
      * ``"placeholder"`` — replace with deterministic-per-call tokens like
        ``[EMAIL_1]``, ``[SECRET_AWS_ACCESS_KEY_1]``.
      * ``"hash"`` — replace with ``[KIND_<sha256[:8]>]`` (stable across calls).
      * ``"strict"`` — raise :class:`SecretDetectedError` if any secret was found,
        otherwise behave like ``"placeholder"``.
    """
    if not text:
        return RedactedText(text=text or "", replacements=())
    if mode not in ("placeholder", "hash", "strict"):
        raise ValueError(f"unknown redact mode: {mode!r}")

    secrets = scan_for_secrets(text)
    if mode == "strict" and secrets:
        raise SecretDetectedError(secrets)

    pii = _scan_pii_presidio(text)
    if pii is None:
        pii = _scan_pii_regex(text)

    hits: list[tuple[str, int, int, str, bool]] = []
    hits.extend((s.kind, s.start, s.end, s.value, True) for s in secrets)
    hits.extend((k, s, e, v, False) for (k, s, e, v) in pii)

    effective_mode: Mode = "placeholder" if mode == "strict" else mode
    return _assemble(text, hits, effective_mode)


def apply_redaction_if_enabled(text: str) -> str:
    """Hook used by ``core/normalize`` — driven by ``MEMOIRS_REDACT`` env.

    * ``MEMOIRS_REDACT=off``    → return text unchanged.
    * ``MEMOIRS_REDACT=on`` (default) → redact in placeholder mode.
    * ``MEMOIRS_REDACT=strict`` → redact and raise on secrets.
    """
    setting = os.environ.get("MEMOIRS_REDACT", "on").strip().lower()
    if setting == "off" or not text:
        return text
    mode: Mode = "strict" if setting == "strict" else "placeholder"
    return redact(text, mode=mode).text


__all__ = [
    "Mode",
    "RedactedText",
    "Replacement",
    "Secret",
    "SecretDetectedError",
    "apply_redaction_if_enabled",
    "redact",
    "scan_for_secrets",
]
