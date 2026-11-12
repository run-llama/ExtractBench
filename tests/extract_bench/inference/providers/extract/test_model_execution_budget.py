"""Stage and recovery requests must share the original document deadline."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("pypdf", reason="dev and runners extras required")
pytest.importorskip("google.genai", reason="dev and runners extras required")
pytest.importorskip("openai", reason="dev and runners extras required")

from extract_bench.inference.limits import ExecutionBudget
from extract_bench.inference.providers.base import ProviderPermanentError
from extract_bench.inference.providers.extract import gemini_direct, nuextract3_extract, openai_responses, vllm_extract


@pytest.mark.parametrize("cancel_upload", [False, True])
def test_openai_upload_consumes_budget_and_cannot_submit_after_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cancel_upload: bool,
) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"document")
    budget = ExecutionBudget(100)
    client = Mock()

    def upload(**kwargs: object) -> SimpleNamespace:
        budget.deadline -= 30
        if cancel_upload:
            budget.cancel()
        return SimpleNamespace(id="file")

    client.files.create.side_effect = upload
    client.responses.create.return_value = SimpleNamespace(output_text="{}", usage=None)
    monkeypatch.setattr(openai_responses, "OpenAI", lambda **kwargs: client)
    provider = openai_responses.OpenAIResponsesExtractProvider("openai_extract", {"api_key": "test"})
    with budget.activate():
        if cancel_upload:
            with pytest.raises(TimeoutError):
                provider._call_api({}, source_path=pdf)
            client.responses.create.assert_not_called()
        else:
            provider._call_api({}, source_path=pdf)
            assert 0 < client.responses.create.call_args.kwargs["timeout"] < 70
            assert client.files.create.call_args.kwargs["timeout"] > 90
    client.files.delete.assert_called_once_with("file", timeout=10)
    assert client.files.create.call_args.kwargs["file"].closed


def test_gemini_uses_remaining_milliseconds_and_blocks_cancelled_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from google import genai

    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(text="{}", candidates=[], usage_metadata=None)
    monkeypatch.setattr(genai, "Client", lambda **kwargs: client)
    provider = gemini_direct.GeminiDirectExtractProvider("gemini_extract", {"api_key": "test"})
    budget = ExecutionBudget(2)
    with budget.activate():
        provider._call_api({}, document_text="text")
        options = client.models.generate_content.call_args.kwargs["config"].http_options
        assert 0 < options.timeout <= 2000
        budget.cancel()
        with pytest.raises(TimeoutError):
            provider._call_api({}, document_text="text")
    assert client.models.generate_content.call_count == 1


@pytest.mark.parametrize("cancel_generation", [False, True])
def test_vllm_recovery_cannot_restart_budget(monkeypatch: pytest.MonkeyPatch, cancel_generation: bool) -> None:
    client = Mock()
    budget = ExecutionBudget(2)

    def generate(**kwargs: object) -> SimpleNamespace:
        if cancel_generation:
            budget.cancel()
            raise RuntimeError("maximum context length is 8192 tokens, prompt contains at least 1024 input tokens")
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="{}"))],
            usage=None,
        )

    client.chat.completions.create.side_effect = generate
    monkeypatch.setattr(vllm_extract, "OpenAI", lambda **kwargs: client)
    provider = vllm_extract.VLLMExtractProvider(
        "vllm_extract",
        {"server_url": "http://unused", "model": "test"},
    )
    with budget.activate():
        if cancel_generation:
            with pytest.raises(TimeoutError):
                provider._call_api_fitting_context({}, [])
        else:
            provider._call_api_fitting_context({}, [])
            assert 0 < client.chat.completions.create.call_args.kwargs["timeout"] <= 2
    assert client.chat.completions.create.call_count == 1


def test_nuextract_image_cap_recovery_stops_after_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = ExecutionBudget(2)
    provider = nuextract3_extract.NuExtract3ExtractProvider("nuextract3_extract", {"server_url": "http://unused"})
    calls = []

    async def rejected(pages: list[str], template: dict, instructions: str) -> dict:
        calls.append(pages)
        budget.cancel()
        raise ProviderPermanentError("at most 1 image")

    monkeypatch.setattr(provider, "_extract_async", rejected)
    with budget.activate(), pytest.raises(TimeoutError):
        provider._extract_with_recovery(["one", "two"], {}, "")
    assert len(calls) == 1


def test_nuextract_http_timeout_uses_remaining_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = nuextract3_extract.NuExtract3ExtractProvider("nuextract3_extract", {"server_url": "http://unused"})
    response = SimpleNamespace(status=200, json=AsyncMock(return_value={"choices": [{"message": {"content": "{}"}}]}))
    session = Mock()
    session.post.return_value = AsyncMock()
    session.post.return_value.__aenter__.return_value = response
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(nuextract3_extract.aiohttp, "ClientSession", lambda: context)
    budget = ExecutionBudget(2)
    with budget.activate():
        asyncio.run(provider._extract_async([], {}, ""))
    assert 0 < session.post.call_args.kwargs["timeout"].total <= 2
