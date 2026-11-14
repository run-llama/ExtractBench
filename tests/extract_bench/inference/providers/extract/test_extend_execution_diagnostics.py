"""Preserve remote execution evidence across success and terminal failure."""

from typing import Any

import pytest

pytest.importorskip("extend_ai", reason="Extend SDK is an optional provider dependency")

import httpx
from extend_ai import Extend

from extract_bench.inference.cost_contract import (
    COMPLETE,
    COST_ACCOUNTING,
    COST_PER_PAGE_USD,
    COST_USD,
    KNOWN_COST_USD,
)
from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.inference.providers.extract.extend import ExtendProvider


@pytest.mark.parametrize("terminal_status", ["PROCESSED", "FAILED"])
def test_async_run_retains_receipt_and_terminal_payload(monkeypatch: pytest.MonkeyPatch, terminal_status: str) -> None:
    response: dict[str, Any] = {
        "object": "document_processor_run",
        "id": "dpr_test",
        "processorId": "dp_test",
        "processorVersionId": "dpv_test",
        "processorName": "test",
        "status": "PROCESSING",
        "output": {"value": {}, "metadata": {}},
        "reviewed": False,
        "edited": False,
        "edits": {},
        "type": "EXTRACT",
        "config": {"type": "EXTRACT", "schema": {"type": "object"}},
        "files": [],
        "mergedProcessors": [],
        "url": "https://dashboard.extend.ai/runs/dpr_test",
    }

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"success": True, "processorRun": response})
        assert request.url.path == "/processor_runs/dpr_test"
        return httpx.Response(
            200,
            json={
                "success": True,
                "processorRun": {
                    **response,
                    "status": terminal_status,
                    "failureReason": "PARSING_ERROR" if terminal_status == "FAILED" else None,
                    "failureMessage": "diagnostic detail" if terminal_status == "FAILED" else None,
                    "usage": {"credits": 7, "totalCredits": 9},
                    "futureDiagnostic": {"requestId": "request-test"},
                },
            },
        )

    monkeypatch.setenv("EXTEND_API_KEY", "test")
    provider = ExtendProvider("extend", {"async_poll_interval": 0})
    provider._client = Extend(token="test", httpx_client=httpx.Client(transport=httpx.MockTransport(handle)))
    if terminal_status == "FAILED":
        with pytest.raises(ProviderPermanentError) as caught:
            provider._run_processor_async("dp_test", "file_test")
        assert caught.value.job_id == "dpr_test"
        payload = caught.value.debug_payload
        assert payload is not None
        assert payload["processor_run"]["failure_message"] == "diagnostic detail"
        assert payload[COST_ACCOUNTING][KNOWN_COST_USD] == pytest.approx(0.1125)
        assert payload[COST_ACCOUNTING][COMPLETE] is True
    else:
        payload = provider._run_processor_async("dp_test", "file_test")
    assert payload["processor_run"]["usage"]["credits"] == 7
    assert payload["processor_run"]["futureDiagnostic"]["requestId"] == "request-test"
    execution = payload["_extend_execution"]
    assert execution["create_response"]["processor_run"]["status"] == "PROCESSING"
    assert [entry["status"] for entry in execution["poll_history"]] == [terminal_status]


def test_extend_cost_uses_configured_credit_rate_without_rebilling_parse() -> None:
    """A rate override must apply to the API roll-up, with parsing counted once."""
    from extract_bench.inference.providers.extract.extend import _attach_cost_stats

    raw = {
        "processor_run": {
            "files": [{"metadata": {"page_count": 2}}],
            "usage": {"credits": 12, "totalCredits": 16},
        }
    }
    _attach_cost_stats(raw, {"usd_per_credit": 0.02})
    assert raw[COST_USD] == pytest.approx(0.32)
    assert raw[COST_PER_PAGE_USD] == pytest.approx(0.16)
    assert raw["parse_credits"] == 4
    assert raw["usd_per_credit"] == 0.02
