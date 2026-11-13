"""Context-local provider events for callers that persist progress before failure."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

ProviderEventSink = Callable[[str, Mapping[str, Any]], None]
_sink: ContextVar[ProviderEventSink | None] = ContextVar("provider_event_sink", default=None)


@contextmanager
def bind_provider_event_sink(sink: ProviderEventSink) -> Iterator[None]:
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def emit_provider_event(event: str, **payload: Any) -> None:
    sink = _sink.get()
    if sink is not None:
        sink(event, payload)
