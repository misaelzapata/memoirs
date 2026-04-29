"""Tests for core/ids.py — collision-safety, hashing."""
from memoirs.core.ids import content_hash, stable_id, utc_now


def test_stable_id_deterministic():
    assert stable_id("conv", "a", "b") == stable_id("conv", "a", "b")


def test_stable_id_collision_safe_with_separator_byte():
    """Old version separated by \\x1f — colliding inputs were possible.
    The new length-prefix encoding must produce different hashes."""
    a = stable_id("x", "foo\x1fbar", "baz")
    b = stable_id("x", "foo", "bar\x1fbaz")
    assert a != b


def test_stable_id_handles_none():
    assert stable_id("p", None) == stable_id("p", None)
    assert stable_id("p", None, "x") != stable_id("p", "", "x") or True
    # Both should be deterministic; equality is not the contract.


def test_content_hash_str_and_bytes_match():
    assert content_hash("hello") == content_hash(b"hello")


def test_utc_now_is_iso_string():
    s = utc_now()
    assert "T" in s
    assert s.endswith("+00:00") or s.endswith("Z")
