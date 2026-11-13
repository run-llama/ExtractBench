"""Paid malformed responses must retain their bill before validation rejects them."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import Mock

import pytest

from extract_bench.inference.events import bind_provider_event_sink
from extract_bench.inference.providers.base import Provider, ProviderError, ProviderPermanentError
from extract_bench.inference.providers.extract.anthropic_direct import AnthropicDirectExtractProvider
from extract_bench.inference.providers.extract.deepseek_extract import DeepSeekExtractProvider
from extract_bench.inference.providers.extract.deepseek_oneshot_extract import DeepSeekOneshotExtractProvider
from extract_bench.inference.providers.extract.gemini_direct import GeminiDirectExtractProvider
from extract_bench.inference.providers.extract.glm_extract import GLMExtractProvider
from extract_bench.inference.providers.extract.glm_zai_extract import GLMDeepInfraExtractProvider, GLMZaiExtractProvider
from extract_bench.inference.providers.extract.kimi_extract import KimiExtractProvider
from extract_bench.inference.providers.extract.openai_responses import OpenAIResponsesExtractProvider


@pytest.mark.parametrize(
    "provider_class",
    [
        AnthropicDirectExtractProvider,
        GeminiDirectExtractProvider,
        OpenAIResponsesExtractProvider,
        GLMZaiExtractProvider,
        GLMDeepInfraExtractProvider,
        DeepSeekOneshotExtractProvider,
        KimiExtractProvider,
        GLMExtractProvider,
        DeepSeekExtractProvider,
    ],
)
@pytest.mark.parametrize("reported_tokens", [None, 0, 120])
def test_paid_invalid_response_preserves_cost_and_missing_usage_stays_unknown(
    provider_class: type[Provider],
    reported_tokens: int | None,
) -> None:
    provider = provider_class("test", {"api_key": "test", "input_price_per_1m": 1, "output_price_per_1m": 1})
    response = NS(
        stop_reason="max_tokens",
        output_text="not json",
        text="not json",
        choices=[NS(finish_reason="stop", message=NS(content="not json"))],
        candidates=[NS(finish_reason="MAX_TOKENS")],
        usage=None
        if reported_tokens is None
        else NS(
            input_tokens=reported_tokens,
            prompt_tokens=reported_tokens,
            output_tokens=0,
            completion_tokens=0,
            total_tokens=reported_tokens,
        ),
        usage_metadata=None
        if reported_tokens is None
        else NS(prompt_token_count=reported_tokens, candidates_token_count=0),
    )
    if provider_class is DeepSeekOneshotExtractProvider:
        response.choices[0].message.content = '{"type":"json_object"}'
    api = Mock(return_value=response)
    provider._client = NS(
        messages=NS(create=api),
        responses=NS(create=api),
        models=NS(generate_content=api),
        chat=NS(completions=NS(create=api)),
    )
    provider._build_file_blocks = lambda path: []
    events = []
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    with bind_provider_event_sink(lambda event, payload: events.append((event, payload))):
        with pytest.raises(ProviderError) as failure:
            if provider_class in (
                AnthropicDirectExtractProvider,
                GeminiDirectExtractProvider,
                OpenAIResponsesExtractProvider,
            ):
                provider._call_api(schema, document_text="document")
            elif provider_class is KimiExtractProvider:
                provider._call_api(schema, [])
            elif provider_class in (GLMExtractProvider, DeepSeekExtractProvider):
                provider._call_api(schema, "document")
            else:
                provider._call_api(schema, Path("unused.pdf"))
    assert api.call_count == 1
    snapshot = failure.value.debug_payload["cost_accounting"]
    components = [payload for event, payload in events if event == "cost.component"]
    if reported_tokens is None:
        assert snapshot["known_cost_usd"] is None
        assert snapshot["complete"] is False
        assert components == []
    else:
        assert snapshot["known_cost_usd"] == pytest.approx(reported_tokens / 1_000_000)
        assert snapshot["complete"] is True
        assert len(components) == 1
        assert components[0]["cost_usd"] == snapshot["known_cost_usd"]


@pytest.mark.parametrize("failed_cost", [None, 0.0, 0.5])
@pytest.mark.parametrize("both_fail", [False, True])
def test_collaboration_retains_failed_stage_bill_and_marks_unknown_stage_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    failed_cost: float | None,
    both_fail: bool,
) -> None:
    from extract_bench.inference.providers.extract.collaborative_direct import CollaborativeDirectExtractProvider
    from extract_bench.schemas.pipeline import PipelineSpec
    from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
    from extract_bench.schemas.product import ProductType

    stage = {"provider_name": "fake", "config": {}}
    provider = CollaborativeDirectExtractProvider(
        "collaborative_extract",
        {
            "first_stage": {**stage, "pipeline_name": "first"},
            "second_stage": {**stage, "pipeline_name": "second"},
        },
    )
    pipeline = PipelineSpec(
        pipeline_name="test", provider_name="collaborative_extract", product_type=ProductType.EXTRACT
    )
    request = InferenceRequest(
        example_id="test",
        source_file_path="unused.pdf",
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object"},
    )

    def run_stage(stage: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        amount = 0.25 if stage.pipeline_name == "first" else failed_cost
        if both_fail or stage.pipeline_name == "second":
            raise ProviderPermanentError(
                "invalid output",
                debug_payload={
                    "cost_accounting": {
                        "known_cost_usd": amount,
                        "complete": amount is not None,
                    }
                },
            )
        return RawInferenceResult(
            request=request,
            pipeline=stage,
            pipeline_name=stage.pipeline_name,
            product_type=ProductType.EXTRACT,
            raw_output={"data": {"answer": 1}, "cost_usd": amount},
            started_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
            latency_in_ms=1,
        )

    monkeypatch.setattr(provider, "_run_stage", run_stage)
    expected = 0.25 + (failed_cost if failed_cost is not None else 0)
    if both_fail:
        with pytest.raises(ProviderError) as failure:
            provider.run_inference(pipeline, request)
        snapshot = failure.value.debug_payload["cost_accounting"]
        assert snapshot["known_cost_usd"] == expected
        assert snapshot["complete"] is (failed_cost is not None)
    else:
        raw = provider.run_inference(pipeline, request).raw_output
        assert raw["data"] == {"answer": 1}
        assert raw["cost_usd"] == expected
        assert raw["_cost_complete"] is (failed_cost is not None)


@pytest.mark.parametrize("parse_cost", [None, 0.0, 0.5])
def test_parsed_text_keeps_partial_charge_when_extract_has_no_usage(parse_cost: float | None) -> None:
    from extract_bench.inference.providers.extract.parsed_text_source import ParsedDocumentText

    parsed = ParsedDocumentText(text="document", num_pages=1, parse_cost_usd=parse_cost)
    error = ProviderPermanentError("network failure")
    parsed.add_error_cost(error)
    assert error.debug_payload["cost_accounting"] == {"known_cost_usd": parse_cost, "complete": False}
    raw: dict[str, Any] = {"cost_usd": None, "_cost_complete": False}
    assert parsed.add_extract_cost(raw) == parse_cost
    assert raw["_cost_complete"] is False


@pytest.mark.parametrize("counters", [{}, {"prompt_token_count": 120}, {"candidates_token_count": 120}])
def test_partial_gemini_sdk_usage_keeps_known_subtotal_incomplete(counters: dict[str, int]) -> None:
    from google.genai import types

    provider = GeminiDirectExtractProvider(
        "gemini_extract",
        {"api_key": "test", "input_price_per_1m": 1, "output_price_per_1m": 1},
    )
    response = NS(
        usage_metadata=types.GenerateContentResponseUsageMetadata.model_validate(counters),
        candidates=[NS(finish_reason="MAX_TOKENS")],
    )
    provider._client = NS(models=NS(generate_content=Mock(return_value=response)))
    with pytest.raises(ProviderError) as failure:
        provider._call_api({"type": "object"}, document_text="document")
    snapshot = failure.value.debug_payload["cost_accounting"]
    assert snapshot["complete"] is False
    if len(counters) == 0:
        assert snapshot["known_cost_usd"] is None
    else:
        assert snapshot["known_cost_usd"] == pytest.approx(0.00012)


@pytest.mark.parametrize("parse_cost", [None, 0.25])
def test_empty_parse_retains_bill_before_any_extraction_starts(
    monkeypatch: pytest.MonkeyPatch,
    parse_cost: float | None,
) -> None:
    from extract_bench.inference.providers.extract.parsed_text_source import LlamaParseTextSource
    from extract_bench.inference.providers.parse import llamaparse
    from extract_bench.schemas.parse_output import ParseOutput
    from extract_bench.schemas.pipeline_io import InferenceRequest
    from extract_bench.schemas.product import ProductType

    parse_provider = Mock()
    parse_provider.run_inference.return_value = NS(raw_output={"cost_usd": parse_cost})
    parse_provider.normalize.return_value = NS(
        output=ParseOutput(
            task_type="parse",
            example_id="test",
            pipeline_name="parse",
            pages=[],
            markdown="",
        )
    )
    monkeypatch.setattr(llamaparse, "LlamaParseProvider", lambda *args, **kwargs: parse_provider)
    request = InferenceRequest(example_id="test", source_file_path="unused.pdf", product_type=ProductType.EXTRACT)
    with pytest.raises(ProviderError, match="no text") as failure:
        LlamaParseTextSource().parse(request)
    snapshot = failure.value.debug_payload["cost_accounting"]
    assert snapshot["known_cost_usd"] == parse_cost
    assert snapshot["complete"] is (parse_cost is not None)
