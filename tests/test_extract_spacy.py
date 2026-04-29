"""Tests for spaCy extractor — must reject low-quality content."""
import pytest

# Skip if spacy not installed
spacy = pytest.importorskip("spacy")

from memoirs.engine.extract_spacy import (
    _is_question,
    _is_system_injection,
    _looks_like_code,
    _looks_like_dump,
    extract,
)


def _doc(text: str):
    """Build a minimal pseudo-doc for testing helpers that need a doc."""
    nlp = spacy.blank("en")
    return nlp(text)


def test_is_system_injection():
    assert _is_system_injection("<ide_opened_file>blah blah")
    assert _is_system_injection("<system-reminder>todo")
    assert _is_system_injection("<bash-stdout>output")
    assert not _is_system_injection("user prefers Python")


def test_looks_like_code():
    assert _looks_like_code("foo := bar.baz()")
    assert _looks_like_code("if x { y }")
    assert _looks_like_code("/etc/passwd")
    assert not _looks_like_code("user prefers Python over Go")
    # URLs are NOT code
    assert not _looks_like_code("see https://example.com for details")


def test_looks_like_dump_diff():
    diff_text = "+added line\n-removed line\n+more added\n-more removed\n+another\n"
    assert _looks_like_dump(diff_text)


def test_looks_like_dump_line_numbers():
    text = "1   first\n2   second\n3   third\n4   fourth\n"
    assert _looks_like_dump(text)


def test_extract_skips_questions():
    msgs = [
        {"id": "1", "role": "user", "content": "What does this function do?"},
        {"id": "2", "role": "user", "content": "How do I install X?"},
    ]
    out = extract(msgs)
    # Questions should not produce candidates
    assert all("question" not in c.content.lower() for c in out)


def test_extract_skips_system_injections():
    msgs = [
        {"id": "1", "role": "user", "content": "<ide_opened_file>foo.py was opened</ide_opened_file>"},
        {"id": "2", "role": "user", "content": "<system-reminder>todo list</system-reminder>"},
    ]
    out = extract(msgs)
    assert out == []


def test_extract_skips_short_messages():
    msgs = [{"id": "1", "role": "user", "content": "ok"}]
    assert extract(msgs) == []


def test_extract_only_user_role():
    msgs = [
        {"id": "1", "role": "assistant", "content": "I prefer Python over Go for prototyping"},
        {"id": "2", "role": "system", "content": "This is a long system message with content."},
    ]
    # Assistant + system should be ignored
    out = extract(msgs)
    assert out == []
