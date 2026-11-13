"""Context-local provider events for callers that persist progress before failure."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from extract_bench.inference.providers.base import ProviderError, ProviderPermanentError

ProviderEventSink = Callable[[str, Mapping[str, Any]], None]
_sink: ContextVar[ProviderEventSink | None] = ContextVar("provider_event_sink", default=None)


class ProviderEventError(ProviderPermanentError):
    """Event persistence failed; repeating paid extraction cannot repair the journal."""


@dataclass
class ProviderEventCapture:
    failure: ProviderEventError | None = None


@contextmanager
def bind_provider_event_sink(sink: ProviderEventSink) -> Iterator[ProviderEventCapture]:
    """Defer sink failures until vendor polling and cancellation have finished.

    Preserve an existing provider error, attaching the first failed event as
    diagnostics. Otherwise fail permanently at the caller boundary so a storage
    error cannot trigger another paid extraction.
    """
    capture = ProviderEventCapture()

    def record(event: str, payload: Mapping[str, Any]) -> None:
        try:
            sink(event, payload)
        except Exception as error:
            if capture.failure is None:
                capture.failure = ProviderEventError(
                    f"Provider event {event} could not be persisted: {error}",
                    debug_payload={"provider_event_failure": {"event": event, "payload": dict(payload)}},
                )
                capture.failure.__cause__ = error

    token = _sink.set(record)
    try:
        yield capture
    except BaseException as error:
        if capture.failure is not None:
            error.add_note(str(capture.failure))
            if isinstance(error, ProviderError):
                error.debug_payload = {**(error.debug_payload or {}), **(capture.failure.debug_payload or {})}
        raise
    else:
        if capture.failure is not None:
            raise capture.failure
    finally:
        _sink.reset(token)


def emit_provider_event(event: str, **payload: Any) -> None:
    sink = _sink.get()
    if sink is not None:
        sink(event, payload)
