"""Structured logging + optional OpenTelemetry for memoirs.

Goals:
- One-line JSON logs (parseable by any log shipper).
- Trace context propagation via contextvars (FastAPI middleware + MCP wrapper).
- Optional OTel — a no-op when `opentelemetry-api` is not installed or
  ``MEMOIRS_OTEL_ENDPOINT`` is unset.

This module is import-safe: no side effects beyond reading environment
variables. ``setup_logging()`` is called once at package import; tests that
need pristine handlers re-invoke it (it is idempotent).
"""
from __future__ import annotations

import contextvars
import functools
import json
import logging
import os
import sys
import time
import traceback as _traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


# ----------------------------------------------------------------------------
# Trace context (contextvars)
# ----------------------------------------------------------------------------

_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "memoirs_trace_id", default=None
)
_span_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "memoirs_span_id", default=None
)


def get_trace_id() -> Optional[str]:
    """Return the current trace id, or None if no active trace context."""
    return _trace_id_var.get()


def get_span_id() -> Optional[str]:
    """Return the current span id, or None if no active span context."""
    return _span_id_var.get()


def new_trace_id() -> str:
    """Generate a new trace id (16-byte hex, OTel-compatible)."""
    return uuid.uuid4().hex


def new_span_id() -> str:
    """Generate a new span id (8-byte hex, OTel-compatible)."""
    return uuid.uuid4().hex[:16]


@contextmanager
def with_trace_context(
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
) -> Iterator[tuple[str, str]]:
    """Bind trace_id / span_id to contextvars for the duration of the block.

    Generates fresh ids when either argument is None. Yields the (trace_id,
    span_id) tuple actually applied, so callers can echo it back to clients.
    """
    tid = trace_id or new_trace_id()
    sid = span_id or new_span_id()
    t_token = _trace_id_var.set(tid)
    s_token = _span_id_var.set(sid)
    try:
        yield tid, sid
    finally:
        _trace_id_var.reset(t_token)
        _span_id_var.reset(s_token)


# ----------------------------------------------------------------------------
# JSON formatter
# ----------------------------------------------------------------------------

# These are the LogRecord attributes set by the standard library; anything
# else passed via `logger.info("...", extra={...})` shows up as an attribute
# and we emit it under the "extra" key.
_RESERVED_LOGRECORD_KEYS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
})


class JsonFormatter(logging.Formatter):
    """One-line JSON formatter.

    Schema::

        {
          "ts":       "<iso8601 utc>",
          "level":    "<INFO|...>",
          "logger":   "<name>",
          "msg":      "<rendered message>",
          "trace_id": "<hex|null>",
          "span_id":  "<hex|null>",
          "extra":    { ... }                   # only when present
          "exc_type": "<ExceptionClass>",       # only on errors
          "exc_msg":  "<str(exc)>",
          "traceback":"<formatted multi-line traceback>"
        }
    """

    # ISO8601 with millisecond precision and "Z" suffix.
    @staticmethod
    def _format_ts(record: logging.LogRecord) -> str:
        # `record.created` is a UNIX timestamp; render to UTC.
        ts_struct = time.gmtime(record.created)
        ms = int(record.msecs)
        return time.strftime("%Y-%m-%dT%H:%M:%S", ts_struct) + f".{ms:03d}Z"

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self._format_ts(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": _trace_id_var.get(),
            "span_id": _span_id_var.get(),
        }

        extras: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_KEYS or key.startswith("_"):
                continue
            # Best-effort coercion to JSON-friendly types.
            try:
                json.dumps(value, default=str)
                extras[key] = value
            except (TypeError, ValueError):
                extras[key] = repr(value)
        if extras:
            payload["extra"] = extras

        if record.exc_info:
            etype, evalue, etb = record.exc_info
            payload["exc_type"] = etype.__name__ if etype else None
            payload["exc_msg"] = str(evalue) if evalue else None
            payload["traceback"] = "".join(
                _traceback.format_exception(etype, evalue, etb)
            )

        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # Last-ditch fallback; never raise from a formatter.
            return json.dumps({
                "ts": payload["ts"],
                "level": payload["level"],
                "logger": payload["logger"],
                "msg": str(payload.get("msg")),
                "trace_id": payload.get("trace_id"),
                "span_id": payload.get("span_id"),
            })


# ----------------------------------------------------------------------------
# setup_logging
# ----------------------------------------------------------------------------

# Marker attribute attached to handlers we install so we can find/remove them
# on subsequent calls without disturbing handlers an embedding application
# may have added.
_HANDLER_MARK = "_memoirs_managed"


def _detect_default_format() -> str:
    """Default to JSON when stdin/stderr is NOT a TTY, else text.

    Override via env: ``MEMOIRS_LOG_FORMAT=json|text``.
    """
    env = os.environ.get("MEMOIRS_LOG_FORMAT", "").strip().lower()
    if env in ("json", "text"):
        return env
    try:
        return "text" if sys.stderr.isatty() else "json"
    except Exception:
        return "text"


def _detect_default_level() -> str:
    return os.environ.get("MEMOIRS_LOG_LEVEL", "INFO").strip().upper() or "INFO"


def _make_formatter(fmt: str) -> logging.Formatter:
    if fmt == "json":
        return JsonFormatter()
    return logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")


def setup_logging(
    *,
    level: Optional[str] = None,
    format: Optional[str] = None,
    target: str = "stderr",
    json_path: Optional[str | Path] = None,
) -> logging.Logger:
    """Configure the root logger.

    Idempotent: replaces any handlers previously installed by this function;
    leaves user-installed handlers intact.

    Parameters
    ----------
    level : "DEBUG" | "INFO" | ...
        Defaults to env ``MEMOIRS_LOG_LEVEL`` or "INFO".
    format : "json" | "text"
        Defaults to env ``MEMOIRS_LOG_FORMAT``, or "json" when stderr is not a
        TTY, otherwise "text".
    target : "stderr" | "file" | "both"
        Where to send log records. ``"file"`` and ``"both"`` require
        ``json_path`` to be a writable path.
    json_path : path to file
        Used when ``target`` includes a file.
    """
    fmt_name = (format or _detect_default_format()).lower()
    if fmt_name not in ("json", "text"):
        fmt_name = "json"
    lvl_name = (level or _detect_default_level()).upper()
    lvl = logging.getLevelName(lvl_name)
    if not isinstance(lvl, int):
        lvl = logging.INFO

    root = logging.getLogger()
    root.setLevel(lvl)

    # Drop only handlers we previously installed.
    for h in list(root.handlers):
        if getattr(h, _HANDLER_MARK, False):
            root.removeHandler(h)

    formatter = _make_formatter(fmt_name)

    if target in ("stderr", "both"):
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(formatter)
        sh.setLevel(lvl)
        setattr(sh, _HANDLER_MARK, True)
        root.addHandler(sh)

    if target in ("file", "both"):
        if json_path is None:
            raise ValueError("json_path is required when target includes 'file'")
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(lvl)
        setattr(fh, _HANDLER_MARK, True)
        root.addHandler(fh)

    return logging.getLogger("memoirs")


# ----------------------------------------------------------------------------
# traced — entry/exit + duration logging, optional OTel span
# ----------------------------------------------------------------------------

def traced(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: log entry/exit + duration; optionally open an OTel span.

    Usage::

        @traced("memoirs.search")
        def search(...): ...

    The wrapped function runs inside a fresh ``with_trace_context(...)`` only
    when no trace is currently active; otherwise it inherits the caller's.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        log = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = _get_otel_tracer()
            existing = _trace_id_var.get()
            ctx = (
                with_trace_context()
                if existing is None
                else _noop_context()
            )
            span_cm = tracer.start_as_current_span(name) if tracer else _noop_context()
            t0 = time.perf_counter()
            with ctx, span_cm:
                log.info("traced.start", extra={"op": name})
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    log.exception(
                        "traced.error",
                        extra={"op": name, "duration_ms": round(elapsed_ms, 3),
                               "error": type(exc).__name__},
                    )
                    raise
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                log.info(
                    "traced.end",
                    extra={"op": name, "duration_ms": round(elapsed_ms, 3)},
                )
                return result

        return wrapper

    return decorator


@contextmanager
def _noop_context() -> Iterator[None]:
    yield None


# ----------------------------------------------------------------------------
# OpenTelemetry — best effort, optional
# ----------------------------------------------------------------------------

_otel_initialized = False
_otel_tracer: Any = None


def _get_otel_tracer() -> Any:
    """Return an active OTel tracer or None.

    Not lazy-initializing here on purpose: ``init_otel`` is the explicit
    entrypoint. When tests want an in-memory tracer they install one directly
    via :func:`set_otel_tracer`.
    """
    return _otel_tracer


def set_otel_tracer(tracer: Any) -> None:
    """Test hook: inject a tracer (e.g., InMemorySpanExporter-backed)."""
    global _otel_tracer, _otel_initialized
    _otel_tracer = tracer
    _otel_initialized = tracer is not None


def init_otel(
    *,
    service_name: str = "memoirs",
    endpoint: Optional[str] = None,
) -> bool:
    """Initialize OpenTelemetry tracing.

    Returns True if tracing was activated, False otherwise (silent no-op).

    Conditions for activation:
      1. ``opentelemetry`` is importable.
      2. ``endpoint`` is provided OR ``MEMOIRS_OTEL_ENDPOINT`` env is set.
    """
    global _otel_initialized, _otel_tracer
    if _otel_initialized:
        return _otel_tracer is not None

    endpoint = endpoint or os.environ.get("MEMOIRS_OTEL_ENDPOINT")
    if not endpoint:
        _otel_initialized = True
        _otel_tracer = None
        return False

    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
            OTLPSpanExporter,
        )
    except Exception:
        _otel_initialized = True
        _otel_tracer = None
        return False

    try:
        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        trace.set_tracer_provider(provider)
        _otel_tracer = trace.get_tracer("memoirs")
        _otel_initialized = True
        return True
    except Exception:
        _otel_initialized = True
        _otel_tracer = None
        return False


__all__ = [
    "JsonFormatter",
    "setup_logging",
    "with_trace_context",
    "get_trace_id",
    "get_span_id",
    "new_trace_id",
    "new_span_id",
    "traced",
    "init_otel",
    "set_otel_tracer",
]
