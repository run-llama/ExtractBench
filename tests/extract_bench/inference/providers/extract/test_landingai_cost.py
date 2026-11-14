"""Retain paid parse usage when extraction fails or omits its billing."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("landingai_ade")

from extract_bench.inference.cost_contract import (
    COMPLETE,
    COMPONENT_ID,
    COST_ACCOUNTING,
    COST_COMPONENT_EVENT,
    COST_USD,
    KNOWN_COST_USD,
    RAW_COST_COMPLETE,
)
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
        metadata=SimpleNamespace(credit_usage=12, page_count=2, job_id="parse-1"),
    )
    provider._client.extract.return_value = SimpleNamespace(
        extraction={},
        metadata=SimpleNamespace(credit_usage=extract_credits, job_id="extract-1"),
    )
    events = []
    if extract_credits == "failure":
        provider._client.extract.side_effect = RuntimeError("503 unavailable")
    with bind_provider_event_sink(lambda event, payload: events.append((event, payload))):
        if extract_credits == "failure":
            with pytest.raises(ProviderTransientError) as caught:
                provider._extract_from_document(Path("unused.pdf"), {})
            assert caught.value.debug_payload[COST_ACCOUNTING] == {KNOWN_COST_USD: 0.12, COMPLETE: False}
        else:
            result = provider._extract_from_document(Path("unused.pdf"), {})
            expected = (12 + (0 if extract_credits is None else extract_credits)) * 0.01
            assert result[COST_USD] == pytest.approx(expected)
            assert result[RAW_COST_COMPLETE] is (extract_credits is not None)
    charges = [payload[COST_USD] for event, payload in events if event == COST_COMPONENT_EVENT]
    assert charges[0] == 0.12
    assert len(charges) == (2 if extract_credits in (0, 5) else 1)


def test_multiple_invocations_do_not_replace_each_others_charges() -> None:
    provider = LandingAIExtractProvider.__new__(LandingAIExtractProvider)
    provider._base_config = {}
    provider._model = None
    provider._client = Mock()
    provider._client.parse.side_effect = [
        SimpleNamespace(markdown="text", metadata=SimpleNamespace(credit_usage=12, job_id=f"parse-{index}"))
        for index in range(2)
    ]
    provider._client.extract.side_effect = [
        SimpleNamespace(extraction={}, metadata=SimpleNamespace(credit_usage=5, job_id=f"extract-{index}"))
        for index in range(2)
    ]
    charges = {}

    def record(event: str, payload: dict) -> None:
        if event == COST_COMPONENT_EVENT:
            charges[payload[COMPONENT_ID]] = payload[COST_USD]

    with bind_provider_event_sink(record):
        provider._extract_from_document(Path("unused.pdf"), {})
        provider._extract_from_document(Path("unused.pdf"), {})
    assert sum(charges.values()) == pytest.approx(0.34)
