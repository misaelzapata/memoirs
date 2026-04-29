"""Tests for memoirs.observability — structured logging + trace context."""
from __future__ import annotations

import io
import json
import logging
import time

import pytest

from memoirs import observability as obs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logger state between tests to keep them hermetic."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)
    # Reset OTel state too.
    obs.set_otel_tracer(None)
    obs._otel_initialized = False


def _capture_root() -> tuple[io.StringIO, logging.Handler]:
    """Install a JsonFormatter-backed handler on the root logger and return it."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(obs.JsonFormatter())
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    return buf, handler


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------

def test_json_formatter_produces_parseable_json():
    buf, _ = _capture_root()
    log = logging.getLogger("memoirs.test")
    log.info("hello world")
    line = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["msg"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "memoirs.test"
    assert "ts" in payload
    # Trace context unset → null.
    assert payload["trace_id"] is None
    assert payload["span_id"] is None


def test_json_formatter_includes_extra_fields():
    buf, _ = _capture_root()
    log = logging.getLogger("memoirs.test")
    log.info("evt", extra={"foo": "bar", "n": 42})
    line = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["extra"]["foo"] == "bar"
    assert payload["extra"]["n"] == 42


def test_json_formatter_serializes_exception():
    buf, _ = _capture_root()
    log = logging.getLogger("memoirs.test")
    try:
        raise ValueError("kaboom")
    except ValueError:
        log.exception("oops")
    line = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["exc_type"] == "ValueError"
    assert payload["exc_msg"] == "kaboom"
    assert "ValueError: kaboom" in payload["traceback"]


def test_json_formatter_handles_non_serializable_extra():
    """Non-JSON values must not crash the formatter."""
    buf, _ = _capture_root()
    log = logging.getLogger("memoirs.test")

    class Weird:
        def __repr__(self):
            return "<Weird>"

    log.info("evt", extra={"obj": Weird()})
    line = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    # Coerced via json default=str → must serialize to *something*.
    assert "obj" in payload["extra"]


# ---------------------------------------------------------------------------
# Trace context
# ---------------------------------------------------------------------------

def test_with_trace_context_sets_contextvars_and_logs_carry_them():
    buf, _ = _capture_root()
    log = logging.getLogger("memoirs.test")
    assert obs.get_trace_id() is None
    with obs.with_trace_context() as (tid, sid):
        assert obs.get_trace_id() == tid
        assert obs.get_span_id() == sid
        assert isinstance(tid, str) and len(tid) >= 16
        log.info("inside")
    assert obs.get_trace_id() is None  # reset after exit
    line = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["trace_id"] == tid
    assert payload["span_id"] == sid


def test_with_trace_context_honors_provided_ids():
    with obs.with_trace_context(trace_id="abc123", span_id="def456") as (tid, sid):
        assert tid == "abc123"
        assert sid == "def456"
        assert obs.get_trace_id() == "abc123"


# ---------------------------------------------------------------------------
# traced decorator
# ---------------------------------------------------------------------------

def test_traced_logs_start_end_and_duration():
    buf, _ = _capture_root()

    @obs.traced("test.op")
    def sleeper(x: int) -> int:
        time.sleep(0.01)
        return x * 2

    result = sleeper(3)
    assert result == 6
    lines = [
        json.loads(ln)
        for ln in buf.getvalue().strip().splitlines()
        if ln.strip()
    ]
    starts = [ln for ln in lines if ln["msg"] == "traced.start"]
    ends = [ln for ln in lines if ln["msg"] == "traced.end"]
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0]["extra"]["op"] == "test.op"
    assert ends[0]["extra"]["op"] == "test.op"
    assert ends[0]["extra"]["duration_ms"] >= 5  # we slept ~10ms


def test_traced_propagates_exception_and_logs_error():
    buf, _ = _capture_root()

    @obs.traced("test.fail")
    def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        boom()
    lines = [json.loads(ln) for ln in buf.getvalue().strip().splitlines() if ln.strip()]
    errors = [ln for ln in lines if ln["msg"] == "traced.error"]
    assert errors, "expected a traced.error log line"
    assert errors[0]["extra"]["error"] == "RuntimeError"


def test_traced_inherits_outer_trace_context():
    buf, _ = _capture_root()

    @obs.traced("test.inner")
    def inner():
        return obs.get_trace_id()

    with obs.with_trace_context(trace_id="outer-tid") as (outer, _):
        seen = inner()
    assert seen == "outer-tid"
    # Logs from inside the decorator should also carry outer-tid.
    lines = [json.loads(ln) for ln in buf.getvalue().strip().splitlines() if ln.strip()]
    inside = [ln for ln in lines if ln["msg"] in ("traced.start", "traced.end")]
    assert all(ln["trace_id"] == "outer-tid" for ln in inside)


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

def test_setup_logging_is_idempotent_and_replaces_managed_handlers():
    obs.setup_logging(level="DEBUG", format="json", target="stderr")
    root = logging.getLogger()
    managed_first = [h for h in root.handlers if getattr(h, "_memoirs_managed", False)]
    assert len(managed_first) == 1

    # User adds their own handler — must NOT be removed by re-init.
    user_handler = logging.NullHandler()
    root.addHandler(user_handler)

    obs.setup_logging(level="INFO", format="text", target="stderr")
    managed_second = [h for h in root.handlers if getattr(h, "_memoirs_managed", False)]
    assert len(managed_second) == 1  # still exactly one
    assert user_handler in root.handlers  # ours preserved


def test_setup_logging_writes_json_file(tmp_path):
    log_path = tmp_path / "out.log"
    obs.setup_logging(level="INFO", format="json", target="file", json_path=log_path)
    logging.getLogger("memoirs.test").info("hi", extra={"k": "v"})
    # FileHandler flushes on emit.
    contents = log_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(contents[-1])
    assert payload["msg"] == "hi"
    assert payload["extra"]["k"] == "v"


# ---------------------------------------------------------------------------
# OpenTelemetry — optional
# ---------------------------------------------------------------------------

def test_init_otel_no_endpoint_is_noop():
    # No env var, no endpoint → silent no-op.
    obs._otel_initialized = False
    obs._otel_tracer = None
    activated = obs.init_otel(endpoint=None)
    assert activated is False
    # Subsequent calls also stable.
    assert obs.init_otel(endpoint=None) is False


def test_traced_uses_injected_tracer():
    """When a tracer is set (e.g. an in-memory one), traced() opens a span."""
    spans_started: list[str] = []

    class FakeSpanCM:
        def __init__(self, name: str):
            self.name = name

        def __enter__(self):
            spans_started.append(self.name)
            return self

        def __exit__(self, *a):
            return False

    class FakeTracer:
        def start_as_current_span(self, name: str):
            return FakeSpanCM(name)

    obs.set_otel_tracer(FakeTracer())

    @obs.traced("op.x")
    def op():
        return 7

    assert op() == 7
    assert spans_started == ["op.x"]


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def test_json_formatter_performance_under_threshold():
    """Rendering a single log record must take < 0.1ms on average."""
    formatter = obs.JsonFormatter()
    record = logging.LogRecord(
        name="memoirs.perf",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    # Warm up.
    for _ in range(50):
        formatter.format(record)
    iterations = 2000
    t0 = time.perf_counter()
    for _ in range(iterations):
        formatter.format(record)
    elapsed = time.perf_counter() - t0
    avg_ms = (elapsed / iterations) * 1000.0
    assert avg_ms < 0.1, f"JsonFormatter too slow: {avg_ms:.4f}ms/record"
