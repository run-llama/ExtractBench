"""Paid work remains inspectable when output validation or vendor execution fails."""

import asyncio
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from extract_bench.inference.events import bind_provider_event_sink
from extract_bench.inference.providers.base import ProviderPermanentError, ProviderTransientError
from extract_bench.inference.providers.extract import azure_content_understanding as azure
from extract_bench.inference.providers.extract import kynex, llamaextract_v2_api, parseur, table_codegen_extract
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest


def test_failed_llamaextract_retains_partial_billed_usage() -> None:
    provider = llamaextract_v2_api.LlamaExtractV2Provider("llamaextract_v2", {"api_key": "offline"})
    response = {"status": "FAILED", "usage": {"extract_credits": 12}}
    events = []
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response)), base_url="https://test.invalid"
    ) as client:
        with bind_provider_event_sink(lambda name, payload: events.append((name, payload))):
            with pytest.raises(ProviderPermanentError) as caught:
                provider._poll_job(client, "job-1", {})
    assert caught.value.job_id == "job-1"
    assert caught.value.debug_payload["usage"] == response["usage"]
    assert caught.value.debug_payload["cost_accounting"] == {
        "known_cost_usd": 0.015,
        "complete": False,
        "usage": response["usage"],
    }
    assert events[-1][1]["cost_usd"] == 0.015


def test_cancelled_parseur_retains_latest_document(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARSEUR_API_KEY", "offline")
    provider = parseur.ParseurProvider("parseur", {})
    stop = threading.Event()
    document = {"status": "PROGRESS", "credits_used": 3}

    def poll(*args: object, **kwargs: object) -> dict:
        stop.set()
        return document

    monkeypatch.setattr(provider, "_request", poll)
    provider._poll_interval = 0
    with pytest.raises(ProviderTransientError) as caught:
        provider._wait_for_document("doc-1", float("inf"), stop)
    assert caught.value.debug_payload == document


def test_rejected_codegen_retains_parse_cost_and_agent_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = table_codegen_extract.TableCodegenExtractProvider("table_codegen_extract", {"api_key": "offline"})
    document = SimpleNamespace(tables=lambda: [1], to_pages=lambda: [])
    monkeypatch.setattr(provider, "_parse_agentic", lambda request: (document, 1.25, 1, "parse-1"))
    usage = {"input": 1000, "output": 100, "reported_calls": 1, "unreported_calls": 1}
    monkeypatch.setattr(
        table_codegen_extract, "generate_extractor", lambda *args, **kwargs: {"status": "max_turns", "usage": usage}
    )
    source = tmp_path / "document.pdf"
    source.write_bytes(b"not parsed")
    request = InferenceRequest.model_validate(
        {
            "example_id": "one",
            "source_file_path": str(source),
            "product_type": "extract",
            "schema_override": {"type": "object"},
        }
    )
    pipeline = PipelineSpec.model_validate(
        {"pipeline_name": "table_codegen_extract", "provider_name": "table_codegen_extract", "product_type": "extract"}
    )
    with pytest.raises(ProviderTransientError) as caught:
        provider.run_inference(pipeline, request)
    payload = caught.value.debug_payload
    assert payload["parse_cost_usd"] == 1.25
    assert payload["usage"] == usage
    assert payload["cost_accounting"]["known_cost_usd"] > 1.25
    assert payload["cost_accounting"]["complete"] is False


@pytest.mark.parametrize("usage", [{"documentPagesStandard": 2}, {}, {"tokens": {}}, {"documentPagesStandard": None}])
def test_failed_azure_operation_retains_usage(monkeypatch: pytest.MonkeyPatch, usage: dict) -> None:
    provider = azure.AzureContentUnderstandingProvider(
        "azure_content_understanding", {"endpoint": "https://test.invalid", "api_key": "offline"}
    )
    body = {"status": "Failed", "usage": usage}
    monkeypatch.setattr(provider, "_request", lambda *args, **kwargs: SimpleNamespace(json=lambda: body, content=b"{}"))
    with pytest.raises(ProviderPermanentError) as caught:
        provider._poll_operation("https://test.invalid/op/1", context="analyze")
    assert caught.value.job_id == "https://test.invalid/op/1"
    assert caught.value.debug_payload["usage"] == body["usage"]
    if usage.get("documentPagesStandard") is None:
        assert caught.value.debug_payload.get("cost_usd") is None


@pytest.mark.parametrize("timeout", [False, True])
def test_failed_kynex_retains_full_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout: bool) -> None:
    stdout = "known usage " + "x" * 500
    failure = subprocess.TimeoutExpired("harness", 1, output=stdout.encode(), stderr=b"failure")
    run = (
        Mock(side_effect=failure)
        if timeout
        else Mock(return_value=SimpleNamespace(returncode=1, stdout=stdout, stderr="failure"))
    )
    monkeypatch.setattr(kynex.subprocess, "run", run)
    events = []
    with bind_provider_event_sink(lambda name, payload: events.append((name, payload))):
        with pytest.raises(ProviderPermanentError) as caught:
            asyncio.run(kynex._extract_async(tmp_path / "one.pdf", {}, 1, tmp_path, Path("python")))
    assert caught.value.debug_payload["stdout"] == stdout
    assert events[-1][1]["stdout"] == stdout


def test_codegen_records_usage_before_malformed_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    from extract_bench.inference.providers.extract.table_codegen.agent_loop import run_agent

    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=17, completion_tokens=4), choices=[])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    recorded = []
    with pytest.raises(IndexError):
        run_agent("openai", "fixture", "extract", [], on_usage=recorded.append)
    assert recorded[0]["input"] == 17
    assert recorded[0]["output"] == 4


def test_codegen_retains_parse_charge_when_parse_normalization_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = table_codegen_extract.TableCodegenExtractProvider("table_codegen_extract", {"api_key": "offline"})
    parse_provider = SimpleNamespace(
        run_inference=lambda *args: SimpleNamespace(raw_output={"cost_usd": 0.75}),
        normalize=Mock(side_effect=ProviderPermanentError("malformed parse")),
    )
    monkeypatch.setattr(table_codegen_extract, "LlamaParseProvider", lambda *args, **kwargs: parse_provider)
    request = InferenceRequest.model_validate(
        {"example_id": "one", "source_file_path": "one.pdf", "product_type": "extract"}
    )
    events = []
    with bind_provider_event_sink(lambda name, payload: events.append((name, payload))):
        with pytest.raises(ProviderPermanentError, match="malformed parse"):
            provider._parse_agentic(request)
    assert events[0][1]["cost_usd"] == 0.75


def test_codegen_missing_usage_does_not_erase_prior_observed_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    from extract_bench.inference.providers.extract.table_codegen.agent_loop import run_agent

    tool = SimpleNamespace(id="call", function=SimpleNamespace(name="missing_tool", arguments="{}"))
    responses = iter(
        [
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=17, completion_tokens=4),
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tool]), finish_reason="tool_calls")
                ],
            ),
            SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]), finish_reason="stop")],
            ),
        ]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: next(responses))))
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    result = run_agent("openai", "fixture", "extract", [])
    assert result["usage"]["input"] == 17
    assert result["usage"]["reported_calls"] == 1
    assert result["usage"]["unreported_calls"] == 1
