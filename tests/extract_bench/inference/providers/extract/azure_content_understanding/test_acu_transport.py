from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import pytest
import requests

from extract_bench.inference.providers.azure_content_understanding_common import (
    DEFAULT_RATE_LIMIT_RETRIES,
    MAX_RETRY_AFTER_S,
    content_understanding_request,
    poll_deadline,
    retry_after_seconds,
    sleep_for_poll_interval,
)
from extract_bench.inference.providers.base import (
    ProviderPermanentError,
    ProviderRateLimitError,
    ProviderTransientError,
)


def _response(status: int, *, headers: dict[str, str] | None = None, text: str = "") -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = text.encode()
    if headers is not None:
        response.headers.update(headers)
    return response


class _FakeSession:
    def __init__(self, responses: list[requests.Response]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append((method, url, kwargs))
        return self._responses.pop(0)


def test_retry_after_seconds_parses_integer_and_caps() -> None:
    assert retry_after_seconds(_response(429, headers={"Retry-After": "12"}), default=5.0) == 12.0
    assert retry_after_seconds(_response(429, headers={"Retry-After": "120"}), default=5.0) == MAX_RETRY_AFTER_S
    assert retry_after_seconds(_response(429), default=5.0) == 5.0
    assert retry_after_seconds(_response(429, headers={"Retry-After": "soon"}), default=5.0) == 5.0


def test_retry_after_seconds_parses_http_date() -> None:
    when = datetime.now(UTC) + timedelta(seconds=30)
    wait = retry_after_seconds(
        _response(429, headers={"Retry-After": format_datetime(when, usegmt=True)}),
        default=5.0,
    )
    assert wait == pytest.approx(30.0, abs=2.0)


def test_content_understanding_request_sleeps_on_429_then_succeeds() -> None:
    slept: list[float] = []
    session = _FakeSession(
        [
            _response(429, headers={"Retry-After": "7"}, text="slow down"),
            _response(200, text="ok"),
        ]
    )

    response = content_understanding_request(
        session,  # type: ignore[arg-type]
        "GET",
        "https://example.test/operation",
        context="poll",
        timeout=30.0,
        sleep=slept.append,
    )

    assert response.status_code == 200
    assert slept == [7.0]
    assert len(session.calls) == 2


def test_content_understanding_request_raises_after_exhausted_429s() -> None:
    slept: list[float] = []
    session = _FakeSession(
        [_response(429, headers={"Retry-After": "3"}, text="still limited")] * (DEFAULT_RATE_LIMIT_RETRIES + 1)
    )

    with pytest.raises(ProviderRateLimitError, match="rate limited during poll"):
        content_understanding_request(
            session,  # type: ignore[arg-type]
            "GET",
            "https://example.test/operation",
            context="poll",
            timeout=30.0,
            sleep=slept.append,
        )

    assert slept == [3.0] * DEFAULT_RATE_LIMIT_RETRIES
    assert len(session.calls) == DEFAULT_RATE_LIMIT_RETRIES + 1


def test_content_understanding_request_classifies_other_errors() -> None:
    with pytest.raises(ProviderTransientError, match="503"):
        content_understanding_request(
            _FakeSession([_response(503, text="unavailable")]),  # type: ignore[arg-type]
            "GET",
            "https://example.test/operation",
            context="poll",
            timeout=30.0,
            sleep=lambda _: None,
        )
    with pytest.raises(ProviderPermanentError, match="400"):
        content_understanding_request(
            _FakeSession([_response(400, text="bad request")]),  # type: ignore[arg-type]
            "POST",
            "https://example.test/analyze",
            context="analyze",
            timeout=30.0,
            sleep=lambda _: None,
        )


def test_sleep_for_poll_interval_uses_remaining_time_not_ping_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    slept: list[float] = []
    monkeypatch.setattr(
        "extract_bench.inference.providers.azure_content_understanding_common.time.monotonic",
        lambda: clock["now"],
    )

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(
        "extract_bench.inference.providers.azure_content_understanding_common.time.sleep",
        fake_sleep,
    )

    deadline = poll_deadline(25.0)
    assert sleep_for_poll_interval(deadline, 10.0)
    assert sleep_for_poll_interval(deadline, 10.0)
    assert sleep_for_poll_interval(deadline, 10.0) is False
    assert slept == [10.0, 10.0, 5.0]


@pytest.mark.parametrize("status", [404, 409])
def test_explicit_lifecycle_http_statuses_are_returned_without_loosening_other_errors(status: int) -> None:
    expected = _response(status)
    session = _FakeSession([expected])
    actual = content_understanding_request(
        session,
        "GET",
        "https://example.test",
        context="lifecycle",
        timeout=1,
        accepted_status_codes=(status,),
    )
    assert actual is expected
    with pytest.raises(ProviderPermanentError):
        content_understanding_request(
            _FakeSession([_response(403)]),
            "GET",
            "https://example.test",
            context="lifecycle",
            timeout=1,
            accepted_status_codes=(status,),
        )
