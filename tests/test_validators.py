"""Tests for Gemma candidate validators."""
from memoirs.engine.gemma import (
    Candidate,
    detect_sensitive_content,
    validate_allowed_memory_type,
    validate_confidence,
    validate_json_output,
    validate_no_secrets,
)


def test_allowed_types():
    for t in ["preference", "fact", "project", "task", "decision", "style", "credential_pointer"]:
        assert validate_allowed_memory_type(Candidate(type=t, content="x"))


def test_disallowed_type():
    assert not validate_allowed_memory_type(Candidate(type="random_thing", content="x"))


def test_validate_confidence_valid():
    assert validate_confidence(Candidate(type="fact", content="x", importance=3, confidence=0.5))
    assert validate_confidence(Candidate(type="fact", content="x", importance=1, confidence=0.0))
    assert validate_confidence(Candidate(type="fact", content="x", importance=5, confidence=1.0))


def test_validate_confidence_out_of_range():
    assert not validate_confidence(Candidate(type="fact", content="x", importance=3, confidence=1.5))
    assert not validate_confidence(Candidate(type="fact", content="x", importance=3, confidence=-0.1))
    assert not validate_confidence(Candidate(type="fact", content="x", importance=10, confidence=0.5))
    assert not validate_confidence(Candidate(type="fact", content="x", importance=0, confidence=0.5))


def test_no_secrets_clean():
    assert validate_no_secrets(Candidate(type="fact", content="user prefers Python"))


def test_no_secrets_catches_api_key():
    assert not validate_no_secrets(Candidate(type="fact", content="api_key=abc123xyzabc123xyzabc"))


def test_no_secrets_catches_openai_key():
    assert not validate_no_secrets(Candidate(type="fact", content="my key is sk-abcdefghijklmnopqrstuv"))


def test_no_secrets_catches_github_pat():
    assert not validate_no_secrets(Candidate(type="fact", content="ghp_abcdefghijklmnopqrstu"))


def test_no_secrets_catches_private_key():
    content = "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."
    assert not validate_no_secrets(Candidate(type="fact", content=content))


def test_detect_sensitive_negation_of_no_secrets():
    assert detect_sensitive_content(Candidate(type="fact", content="api_key=abc123xyzabc"))
    assert not detect_sensitive_content(Candidate(type="fact", content="benign content"))


def test_validate_json_output_strict():
    assert validate_json_output('[{"type":"fact","content":"x"}]') == [{"type": "fact", "content": "x"}]


def test_validate_json_output_truncated_array_salvages():
    """Gemma can hit max_tokens mid-array; the salvage parser must recover complete objects."""
    truncated = '[{"type":"fact","content":"a"},{"type":"task","content":"b"},{"type":"task","content":'
    result = validate_json_output(truncated)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["content"] == "a"
    assert result[1]["content"] == "b"


def test_validate_json_output_completely_invalid_raises():
    import pytest
    with pytest.raises(ValueError):
        validate_json_output("this is not json at all")
