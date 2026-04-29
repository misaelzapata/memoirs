"""Tests for memoirs.core.redact.

These tests must pass without Presidio installed (we exercise the regex
fallback by default). A small subset is parametrised to also run against
Presidio when ``presidio_analyzer`` is importable, but it is silently
skipped otherwise — see ``test_with_presidio_if_available``.
"""
from __future__ import annotations

import importlib
import time

import pytest

from memoirs.core import redact as redact_mod
from memoirs.core.redact import (
    RedactedText,
    Replacement,
    Secret,
    SecretDetectedError,
    redact,
    scan_for_secrets,
)


PRESIDIO_AVAILABLE = importlib.util.find_spec("presidio_analyzer") is not None


# ---------------------------------------------------------------------------
# Email / phone / IP / SSN / credit card — PII regex fallback
# ---------------------------------------------------------------------------


def test_email_redacted():
    out = redact("contact me at alice@example.com please")
    assert "alice@example.com" not in out.text
    assert any(r.kind == "EMAIL" for r in out.replacements)
    assert "[EMAIL_1]" in out.text


def test_phone_redacted():
    out = redact("call +1 415-555-0199 anytime")
    assert "415-555-0199" not in out.text
    assert any(r.kind == "PHONE" for r in out.replacements)


def test_ipv4_redacted():
    out = redact("server is up at 192.168.10.42 right now")
    assert "192.168.10.42" not in out.text
    kinds = {r.kind for r in out.replacements}
    assert "IPV4" in kinds


def test_ssn_redacted():
    out = redact("SSN 123-45-6789 on file")
    assert "123-45-6789" not in out.text
    assert any(r.kind == "SSN" for r in out.replacements)


def test_credit_card_luhn_match():
    # 4111 1111 1111 1111 is the canonical Visa test number (passes Luhn).
    out = redact("card: 4111 1111 1111 1111 thanks")
    assert "4111 1111 1111 1111" not in out.text
    assert any(r.kind == "CREDIT_CARD" for r in out.replacements)


def test_credit_card_invalid_luhn_ignored():
    # Random 16-digit run that does NOT pass Luhn must not be flagged.
    out = redact("order id 1234567890123456 today")
    assert all(r.kind != "CREDIT_CARD" for r in out.replacements)


def test_url_with_token_redacted():
    out = redact("see https://api.example.com/v1?token=abcdef12345 thanks")
    assert "token=abcdef12345" not in out.text
    assert any(r.kind == "URL_WITH_SECRET" for r in out.replacements)


# ---------------------------------------------------------------------------
# Secrets — always regex
# ---------------------------------------------------------------------------


def test_aws_access_key_is_secret():
    out = redact("AWS=AKIAIOSFODNN7EXAMPLE rest")
    assert "AKIAIOSFODNN7EXAMPLE" not in out.text
    assert any(r.is_secret and r.kind == "AWS_ACCESS_KEY" for r in out.replacements)


def test_openai_key_is_secret():
    key = "sk-" + "A" * 40
    out = redact(f"key={key}")
    assert key not in out.text
    assert any(r.is_secret and r.kind == "OPENAI_KEY" for r in out.replacements)


def test_github_pat_is_secret():
    pat = "ghp_" + "a" * 36
    out = redact(f"clone with {pat}")
    assert pat not in out.text
    assert any(r.is_secret and r.kind == "GITHUB_PAT" for r in out.replacements)


def test_jwt_redacted():
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    out = redact(f"Authorization: Bearer {jwt}")
    assert jwt not in out.text
    assert any(r.is_secret and r.kind == "JWT" for r in out.replacements)


def test_pem_private_key_redacted():
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ\n"
        "-----END PRIVATE KEY-----"
    )
    out = redact(f"key:\n{pem}\nend")
    assert "BEGIN PRIVATE KEY" not in out.text
    assert any(r.is_secret and r.kind == "PEM_PRIVATE_KEY" for r in out.replacements)


def test_scan_for_secrets_returns_classified_list():
    text = "AKIAIOSFODNN7EXAMPLE and ghp_" + "x" * 36
    secrets = scan_for_secrets(text)
    assert {s.kind for s in secrets} == {"AWS_ACCESS_KEY", "GITHUB_PAT"}
    assert all(isinstance(s, Secret) for s in secrets)


# ---------------------------------------------------------------------------
# Modes & idempotency
# ---------------------------------------------------------------------------


def test_strict_mode_raises_on_secret():
    with pytest.raises(SecretDetectedError) as ei:
        redact("token AKIAIOSFODNN7EXAMPLE", mode="strict")
    assert any(s.kind == "AWS_ACCESS_KEY" for s in ei.value.secrets)


def test_strict_mode_passes_when_no_secret():
    out = redact("hello alice@example.com", mode="strict")
    assert "alice@example.com" not in out.text


def test_hash_mode_is_stable_across_calls():
    a = redact("write to bob@example.com", mode="hash")
    b = redact("nothing here, but bob@example.com", mode="hash")
    # Same email → same hash placeholder regardless of surrounding text.
    a_email = next(r.placeholder for r in a.replacements if r.kind == "EMAIL")
    b_email = next(r.placeholder for r in b.replacements if r.kind == "EMAIL")
    assert a_email == b_email


def test_idempotent_redaction():
    text = "alice@example.com and AKIAIOSFODNN7EXAMPLE and 192.168.1.1"
    once = redact(text)
    twice = redact(once.text)
    assert twice.text == once.text
    assert twice.replacements == ()


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        redact("x", mode="bogus")  # type: ignore[arg-type]


def test_empty_text_is_safe():
    out = redact("")
    assert out.text == ""
    assert out.replacements == ()


def test_same_value_gets_same_placeholder_within_call():
    out = redact("a alice@example.com b alice@example.com c")
    emails = [r.placeholder for r in out.replacements if r.kind == "EMAIL"]
    assert len(emails) == 2
    assert emails[0] == emails[1] == "[EMAIL_1]"


def test_distinct_values_get_distinct_placeholders():
    out = redact("a alice@example.com b bob@example.com c")
    emails = sorted({r.placeholder for r in out.replacements if r.kind == "EMAIL"})
    assert emails == ["[EMAIL_1]", "[EMAIL_2]"]


# ---------------------------------------------------------------------------
# Hook + env behaviour
# ---------------------------------------------------------------------------


def test_apply_redaction_if_enabled_off(monkeypatch):
    monkeypatch.setenv("MEMOIRS_REDACT", "off")
    raw = "alice@example.com"
    assert redact_mod.apply_redaction_if_enabled(raw) == raw


def test_apply_redaction_if_enabled_on(monkeypatch):
    monkeypatch.setenv("MEMOIRS_REDACT", "on")
    out = redact_mod.apply_redaction_if_enabled("alice@example.com")
    assert "alice@example.com" not in out
    assert "[EMAIL_1]" in out


def test_apply_redaction_if_enabled_strict(monkeypatch):
    monkeypatch.setenv("MEMOIRS_REDACT", "strict")
    with pytest.raises(SecretDetectedError):
        redact_mod.apply_redaction_if_enabled("token AKIAIOSFODNN7EXAMPLE")


def test_normalize_flatten_redacts_email(monkeypatch):
    """Hook integration: flatten_content should pass through the redactor."""
    monkeypatch.setenv("MEMOIRS_REDACT", "on")
    from memoirs.core.normalize import flatten_content

    out = flatten_content([{"type": "text", "text": "ping alice@example.com"}])
    assert "alice@example.com" not in out
    assert "[EMAIL_" in out


# ---------------------------------------------------------------------------
# Performance: regex-only mode under 5ms / message average
# ---------------------------------------------------------------------------


def test_regex_only_performance(monkeypatch):
    # Force the Presidio path off even if it ever becomes available.
    monkeypatch.setattr(redact_mod, "_get_presidio", lambda: None)
    sample = (
        "User alice@example.com from 10.0.0.5 says: my key is sk-"
        + "A" * 40
        + " and old AKIAIOSFODNN7EXAMPLE plus card 4111 1111 1111 1111"
    )
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        redact(sample)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n
    # generous bound; on a developer laptop this is typically <1ms
    assert elapsed_ms < 5.0, f"redact too slow: {elapsed_ms:.2f}ms/msg"


# ---------------------------------------------------------------------------
# Optional: exercise Presidio path if installed
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PRESIDIO_AVAILABLE, reason="presidio-analyzer not installed")
def test_with_presidio_if_available():
    out = redact("contact alice@example.com or +1 415 555 0199")
    kinds = {r.kind for r in out.replacements}
    assert "EMAIL" in kinds
