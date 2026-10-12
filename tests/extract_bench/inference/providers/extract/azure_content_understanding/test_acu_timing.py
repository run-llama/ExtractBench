from datetime import datetime

import pytest
import requests

from extract_bench.evaluation.stats import build_operational_stats
from extract_bench.inference.providers.extract.azure_content_understanding import AzureContentUnderstandingProvider
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType


def get_pipeline(name: str) -> PipelineSpec:
    return PipelineSpec(
        pipeline_name=name,
        provider_name="azure_content_understanding",
        product_type=ProductType.EXTRACT,
        config={"field_method": "generate" if name.endswith("gen") else "extract"},
    )


@pytest.mark.parametrize("pipeline_name", ["acu-ga-ext", "acu-ga-gen"])
@pytest.mark.parametrize("setup_seconds", [1.0, 10.0])
def test_document_latency_excludes_setup_and_survives_normalization(
    tmp_path, monkeypatch: pytest.MonkeyPatch, pipeline_name: str, setup_seconds: float
) -> None:
    module = "extract_bench.inference.providers.extract.azure_content_understanding"
    ticks = iter([0.0, setup_seconds, setup_seconds + 2.0, setup_seconds + 3.0, 20.0, 20.25])
    monkeypatch.setattr(f"{module}.perf_counter", lambda: next(ticks))
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"pdf bytes")
    provider = AzureContentUnderstandingProvider.__new__(AzureContentUnderstandingProvider)
    monkeypatch.setattr(provider, "_ensure_analyzer", lambda pipeline, schema: ("analyzer", {}, {}, "created"))
    monkeypatch.setattr(
        provider,
        "_analyze_binary",
        lambda analyzer_id, path: {"result": {"contents": [{"kind": "document", "fields": {}}]}},
    )
    pipeline = get_pipeline(pipeline_name)
    request = InferenceRequest(
        example_id="timing",
        source_file_path=str(file_path),
        product_type=ProductType.EXTRACT,
        schema_override={"type": "object", "properties": {}},
    )
    raw = provider.run_inference(pipeline, request)
    assert raw.processing_duration_ms == 2000
    assert raw.raw_output["_client_timings"] == {
        "analyzer_setup_and_readiness_ms": int(setup_seconds * 1000),
        "document_processing_ms": 2000,
        "postprocessing_ms": 1000,
        "inference_total_ms": int((setup_seconds + 3.0) * 1000),
    }
    raw = RawInferenceResult.model_validate_json(raw.model_dump_json())
    legacy_latency = raw.latency_in_ms
    normalized = provider.normalize(raw)
    assert normalized.latency_in_ms == legacy_latency
    assert normalized.processing_duration_ms == 2000
    assert normalized.source == "acu_client_document_processing"
    assert normalized.raw_output["_client_timings"]["normalization_ms"] == 250
    normalized = InferenceResult.model_validate_json(normalized.model_dump_json())
    stats = {stat.name: stat.value for stat in build_operational_stats(normalized)}
    assert stats["latency_ms"] == 2000


@pytest.mark.parametrize("poll", [False, True])
def test_client_document_breakdown_measures_submission_and_polling(
    tmp_path, monkeypatch: pytest.MonkeyPatch, poll: bool
) -> None:
    ticks = iter([0.0, 0.1, 0.6, 2.6])
    monkeypatch.setattr(
        "extract_bench.inference.providers.extract.azure_content_understanding.perf_counter",
        lambda: next(ticks),
    )
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"pdf bytes")
    provider = AzureContentUnderstandingProvider.__new__(AzureContentUnderstandingProvider)
    provider._url = lambda path: path
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"result":{"contents":[{"kind":"document","fields":{}}]}}'
    if poll:
        response.headers["Operation-Location"] = "https://example/operation"
    monkeypatch.setattr(provider, "_request", lambda *args, **kwargs: response)
    polling_calls = []

    def poll_operation(url, *, context):
        polling_calls.append(url)
        return response.json()

    monkeypatch.setattr(provider, "_poll_operation", poll_operation)
    result = provider._analyze_binary("analyzer", file_path)
    assert result["_client_timings"] == {
        "document_preparation_ms": 100,
        "document_submission_ms": 500,
        "document_polling_ms": 2000 if poll else 0,
    }
    assert polling_calls == (["https://example/operation"] if poll else [])


@pytest.mark.parametrize("processing_duration_ms", [None, 2000])
def test_retrieval_timing_survives_normalization_and_reporting(processing_duration_ms):
    pipeline = get_pipeline("acu-ga-ext")
    raw = RawInferenceResult(
        request=InferenceRequest(example_id="saved", source_file_path="saved.pdf", product_type=ProductType.EXTRACT),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type=ProductType.EXTRACT,
        raw_output={"num_pages": 2, "result": {"contents": [{"kind": "document", "fields": {}}]}},
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1, 0, 0, 10),
        latency_in_ms=10000,
        processing_duration_ms=processing_duration_ms,
        retrieval_duration_ms=250,
    )
    provider = AzureContentUnderstandingProvider.__new__(AzureContentUnderstandingProvider)
    restored_raw = RawInferenceResult.model_validate_json(raw.model_dump_json())
    normalized = provider.normalize(restored_raw)
    restored = InferenceResult.model_validate_json(normalized.model_dump_json())
    assert restored.retrieval_duration_ms == 250
    assert restored.latency_in_ms == 10000
    assert {stat.name: stat.value for stat in build_operational_stats(restored)} == {
        "retrieval_latency_ms": 250,
        "retrieval_latency_ms_per_page": 125,
    }
