"""Keep processing, provider-total, and artifact retrieval timings distinct."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from extract_bench.evaluation.stats import build_operational_stats
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType


def _result(**timings) -> InferenceResult:
    return InferenceResult(
        request=InferenceRequest(example_id="doc", source_file_path="doc.pdf", product_type=ProductType.EXTRACT),
        pipeline_name="test",
        product_type=ProductType.EXTRACT,
        raw_output={"num_pages": 2, "cost_usd": 0.25},
        output=ExtractOutput(task_type="extract", example_id="doc", pipeline_name="test", extracted_data={}),
        started_at=datetime(2026, 1, 1),
        completed_at=datetime(2026, 1, 1, 0, 0, 10),
        latency_in_ms=10000,
        **timings,
    )


@pytest.mark.parametrize(
    ("processing", "retrieval", "name", "expected"),
    [
        (None, None, "latency_ms", 10000),
        (2000, None, "latency_ms", 2000),
        (0, None, "latency_ms", 0),
        (None, 250, "retrieval_latency_ms", 250),
        (2000, 0, "retrieval_latency_ms", 0),
    ],
)
def test_timing_selection_survives_serialization(processing, retrieval, name, expected):
    result = _result(processing_duration_ms=processing, retrieval_duration_ms=retrieval)
    restored = InferenceResult.model_validate_json(result.model_dump_json())
    stats = {stat.name: stat.value for stat in build_operational_stats(restored)}
    assert stats == {name: expected, name + "_per_page": expected / 2, "cost_usd": 0.25}
    assert restored.latency_in_ms == 10000


@pytest.mark.parametrize("model", [InferenceResult, RawInferenceResult])
def test_legacy_results_keep_total_latency(model):
    payload = _result().model_dump(
        exclude={"processing_duration_ms", "retrieval_duration_ms", "source", "availability"}
    )
    if model is RawInferenceResult:
        payload["pipeline"] = PipelineSpec(
            pipeline_name="test", provider_name="test", product_type=ProductType.EXTRACT, config={}
        )
    result = model.model_validate(payload)
    assert result.latency_in_ms == 10000
    assert result.processing_duration_ms is None
    assert result.retrieval_duration_ms is None
    assert result.source is None
    assert result.availability == {}


@pytest.mark.parametrize("field", ["processing_duration_ms", "retrieval_duration_ms"])
def test_negative_durations_rejected(field):
    with pytest.raises(ValidationError):
        _result(**{field: -1})


@pytest.mark.parametrize("pages", [None, 0, -1])
def test_missing_or_nonpositive_pages_keep_document_latency(pages):
    result = _result(processing_duration_ms=2000)
    result.raw_output["num_pages"] = pages
    stats = {stat.name: stat.value for stat in build_operational_stats(result)}
    assert stats == {"latency_ms": 2000, "cost_usd": 0.25}
