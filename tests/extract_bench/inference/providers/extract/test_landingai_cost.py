"""Retain paid parse usage when extraction fails or omits its billing."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("landingai_ade")

from extract_bench.inference.events import bind_provider_event_sink
from extract_bench.inference.providers.base import ProviderTransientError
from extract_bench.inference.providers.extract.landingai import LandingAIExtractProvider


@pytest.mark.parametrize("extract_credits", [None, 0, 5, "failure"])
def test_parse_cost_survives_missing_or_failed_extraction(extract_credits: float | str | None) -> None:
    provider = LandingAIExtractProvider.__new__(LandingAIExtractProvider)
    provider._base_config = {}
    provider._model = None
    provider._client = Mock()
    provider._client.parse.return_value = SimpleNamespace(
        markdown="text",
        metadata=SimpleNamespace(credit_usage=12, page_count=2),
    )
    provider._client.extract.return_value = SimpleNamespace(
        extraction={},
        metadata=SimpleNamespace(credit_usage=extract_credits),
    )
    events = []
    if extract_credits == "failure":
        provider._client.extract.side_effect = RuntimeError("503 unavailable")
    with bind_provider_event_sink(lambda event, payload: events.append((event, payload))):
        if extract_credits == "failure":
            with pytest.raises(ProviderTransientError) as caught:
                provider._extract_from_document(Path("unused.pdf"), {})
            assert caught.value.debug_payload["cost_accounting"] == {"known_cost_usd": 0.12, "complete": False}
        else:
            result = provider._extract_from_document(Path("unused.pdf"), {})
            expected = (12 + (0 if extract_credits is None else extract_credits)) * 0.01
            assert result["cost_usd"] == pytest.approx(expected)
            assert result["_cost_complete"] is (extract_credits is not None)
    charges = [payload["cost_usd"] for event, payload in events if event == "cost.component"]
    assert charges[0] == 0.12
    assert len(charges) == (2 if extract_credits in (0, 5) else 1)


def test_multiple_invocations_do_not_replace_each_others_charges() -> None:
    provider = LandingAIExtractProvider.__new__(LandingAIExtractProvider)
    provider._base_config = {}
    provider._model = None
    provider._client = Mock()
    provider._client.parse.return_value = SimpleNamespace(markdown="text", metadata=SimpleNamespace(credit_usage=12))
    provider._client.extract.return_value = SimpleNamespace(extraction={}, metadata=SimpleNamespace(credit_usage=5))
    charges = {}

    def record(event: str, payload: dict) -> None:
        if event == "cost.component":
            charges[payload["component_id"]] = payload["cost_usd"]

    with bind_provider_event_sink(record):
        provider._extract_from_document(Path("unused.pdf"), {})
        provider._extract_from_document(Path("unused.pdf"), {})
    assert sum(charges.values()) == pytest.approx(0.34)
