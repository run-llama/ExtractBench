from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("openai", reason="dev and runners extras required; run: uv sync --extra dev --extra runners")

from extract_bench.evaluation.stats import build_operational_stats
from extract_bench.inference.providers.extract import openai_responses
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult
from extract_bench.schemas.product import ProductType


def test_openai_responses_extract_provider_model_pricing() -> None:
    pricing = openai_responses.OpenAIResponsesExtractProvider._pricing_for_model

    assert pricing("gpt-4.1") == (2.00, 8.00)
    assert pricing("gpt-4.1-mini-2025-04-14") == (0.40, 1.60)
    assert pricing("gpt-4.1-nano-2025-04-14") == (0.10, 0.40)
    assert pricing("gpt-5-nano-2025-08-07") == (0.05, 0.40)
    assert pricing("gpt-5.4") == (2.50, 15.00)
    assert pricing("gpt-5.4-mini") == (0.75, 4.50)
    assert pricing("gpt-5.4-nano-2026-03-17") == (0.20, 1.25)
    assert pricing("gpt-5.5") == (5.00, 30.00)
    assert pricing("gpt-5.6-sol") == (4.00, 20.00)
    assert pricing("gpt-5.6-terra") == (2.00, 12.00)
    assert pricing("gpt-5.6-luna") == (0.20, 1.20)
    assert pricing("unknown-model") == (0.0, 0.0)


def test_every_registered_openai_extract_pipeline_has_a_pricing_row() -> None:
    """An unpriced model reports $0.00 instead of failing, so guard the registry.

    ``_pricing_for_model`` returns zeros on a miss, which is indistinguishable
    from a genuinely free run in every downstream cost report. The gpt-5.6
    sol/luna/terra one-shot pipelines shipped that way until this test existed.
    """
    from extract_bench.inference.pipelines import get_pipeline, list_pipelines

    provider_cls = openai_responses.OpenAIResponsesExtractProvider
    unpriced = []
    for name in list_pipelines():
        spec = get_pipeline(name)
        if spec.provider_name != "openai_extract":
            continue
        config = spec.config or {}
        if "input_price_per_1m" in config and "output_price_per_1m" in config:
            continue
        model = config.get("model", provider_cls.DEFAULT_MODEL)
        if provider_cls._pricing_for_model(model) == (0.0, 0.0):
            unpriced.append(f"{name} (model={model})")

    assert not unpriced, "openai_extract pipelines whose model has no pricing row (would report $0.00): " + ", ".join(
        unpriced
    )


class _FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text='{"invoice_number": "INV-001"}',
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200, total_tokens=1200),
        )


class _FakeFiles:
    def __init__(self) -> None:
        self.created: list[Any] = []

    def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return SimpleNamespace(id="file-123")

    def delete(self, file_id: str) -> None:
        pass


class _FakeParseSource:
    def __init__(self, parsed: Any):
        self.parsed = parsed
        self.requests: list[InferenceRequest] = []

    def parse(self, request: InferenceRequest) -> Any:
        self.requests.append(request)
        return self.parsed


def test_openai_responses_parsed_text_mode_sends_text_and_totals_parse_cost(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from extract_bench.inference.providers.extract.parsed_text_source import ParsedDocumentText

    source = tmp_path / "invoice.pdf"
    source.write_bytes(b"not-a-real-pdf")
    fake_responses = _FakeResponses()
    fake_files = _FakeFiles()
    monkeypatch.setattr(
        openai_responses,
        "OpenAI",
        lambda api_key: SimpleNamespace(responses=fake_responses, files=fake_files),
    )
    fake_parse = _FakeParseSource(
        ParsedDocumentText(
            text="<!-- Page 1 -->\n\n# Invoice INV-001",
            num_pages=3,
            parse_cost_usd=0.0375,
            metadata={"type": "llamaparse", "tier": "agentic"},
        )
    )
    monkeypatch.setattr(openai_responses, "create_parse_text_source", lambda cfg: fake_parse)

    provider = openai_responses.OpenAIResponsesExtractProvider(
        "openai_extract",
        {
            "api_key": "test-key",
            "model": "gpt-5.4",
            "input_mode": "parsed_text",
            "parse_source": {"tier": "agentic"},
        },
    )
    pipeline = PipelineSpec(
        pipeline_name="openai_gpt_5_4_extract_twostage_parse_agentic_structured_output_text",
        provider_name="openai_extract",
        product_type=ProductType.EXTRACT,
        config={},
    )
    request = InferenceRequest(
        example_id="example-1",
        source_file_path=str(source),
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object", "properties": {"invoice_number": {"type": "string"}}},
    )
    raw = provider.run_inference(pipeline, request)

    # Text input — no Files-API upload even though the source is a PDF.
    assert fake_files.created == []
    user_content = fake_responses.calls[0]["input"][1]["content"]
    assert user_content[0] == {"type": "input_text", "text": "<!-- Page 1 -->\n\n# Invoice INV-001"}

    extract_cost = (1000 * 2.50 + 200 * 15.00) / 1_000_000
    assert raw.raw_output["extract_cost_usd"] == pytest.approx(extract_cost)
    assert raw.raw_output["parse_cost_usd"] == pytest.approx(0.0375)
    assert raw.raw_output["cost_usd"] == pytest.approx(extract_cost + 0.0375)
    assert raw.raw_output["num_pages"] == 3
    assert raw.raw_output["parsed_text"] == "<!-- Page 1 -->\n\n# Invoice INV-001"
    assert raw.raw_output["_config"]["input_mode"] == "parsed_text"


def test_recompute_cost_reprices_a_saved_run_from_its_recorded_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale run must be fixable by re-normalizing, not only by re-running inference.

    The GPT-5.6 one-shot pipelines ran while their model had no pricing row and
    recorded cost_usd 0.00. Cost is derived from usage and the pricing table, so
    correcting the table and re-running `extract-bench inference renormalize --force`
    re-prices those runs; evaluation reads the cost off the normalized result.
    """
    monkeypatch.setattr(openai_responses, "OpenAI", lambda api_key: SimpleNamespace())
    provider = openai_responses.OpenAIResponsesExtractProvider(
        "openai_extract", {"api_key": "test-key", "model": "gpt-5.6-terra"}
    )
    pipeline = PipelineSpec(
        pipeline_name="openai_gpt_5_6_terra_extract_oneshot_structured_output_file",
        provider_name="openai_extract",
        product_type=ProductType.EXTRACT,
        config={"model": "gpt-5.6-terra"},
    )
    request = InferenceRequest(
        example_id="example-1",
        source_file_path="/nonexistent/invoice.pdf",
        product_type=ProductType.EXTRACT,
    )
    # As written to .raw.json by the unpriced run.
    raw_result = RawInferenceResult(
        request=request,
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.EXTRACT,
        raw_output={
            "data": {"invoice_number": "INV-001"},
            "usage": {"input_tokens": 20_000, "output_tokens": 1_000, "total_tokens": 21_000},
            "num_pages": 4,
            "cost_usd": 0.0,
            "cost_per_page_usd": 0.0,
            "_config": {"input_price_per_1m": 0.0, "output_price_per_1m": 0.0},
        },
        started_at=datetime(2026, 8, 19, 12, 0, 0),
        completed_at=datetime(2026, 8, 19, 12, 0, 30),
        latency_in_ms=30_000,
    )

    # recompute_cost is the generic seam the renormalize path calls for every
    # provider; it re-prices in place from the recorded usage.
    provider.recompute_cost(raw_result.raw_output)
    result = provider.normalize(raw_result)

    expected = (20_000 * 2.00 + 1_000 * 12.00) / 1_000_000
    assert result.raw_output["cost_usd"] == pytest.approx(expected)
    assert result.raw_output["cost_per_page_usd"] == pytest.approx(expected / 4)
    # The recorded rates move with the cost they explain.
    assert result.raw_output["_config"]["input_price_per_1m"] == 2.00
    assert result.raw_output["_config"]["output_price_per_1m"] == 12.00
    # Cost lands where evaluation/stats.py reads it.
    stats = {stat.name: stat.value for stat in build_operational_stats(result)}
    assert stats["cost_usd"] == pytest.approx(expected)


def test_recompute_cost_leaves_cost_alone_when_a_raw_artifact_has_no_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-pricing an artifact that predates token accounting would zero a real number."""
    monkeypatch.setattr(openai_responses, "OpenAI", lambda api_key: SimpleNamespace())
    provider = openai_responses.OpenAIResponsesExtractProvider(
        "openai_extract", {"api_key": "test-key", "model": "gpt-5.6-terra"}
    )
    pipeline = PipelineSpec(
        pipeline_name="openai_gpt_5_6_terra_extract_oneshot_structured_output_file",
        provider_name="openai_extract",
        product_type=ProductType.EXTRACT,
        config={"model": "gpt-5.6-terra"},
    )
    raw_result = RawInferenceResult(
        request=InferenceRequest(
            example_id="example-1",
            source_file_path="/nonexistent/invoice.pdf",
            product_type=ProductType.EXTRACT,
        ),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.EXTRACT,
        raw_output={"data": {}, "cost_usd": 1.23},
        started_at=datetime(2026, 8, 19, 12, 0, 0),
        completed_at=datetime(2026, 8, 19, 12, 0, 30),
        latency_in_ms=30_000,
    )

    provider.recompute_cost(raw_result.raw_output)
    assert provider.normalize(raw_result).raw_output["cost_usd"] == 1.23
