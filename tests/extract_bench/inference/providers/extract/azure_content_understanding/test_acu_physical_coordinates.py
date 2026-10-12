"""ACU document source polygons use page units even near the page origin."""

from datetime import datetime

import pytest

from extract_bench.inference.providers.extract import azure_content_understanding

_SMALL_SOURCE = "D(1,0.2,0.2,0.8,0.2,0.8,0.8,0.2,0.8)"


@pytest.fixture
def acu():
    return azure_content_understanding


def _normalized_citation(acu, source, page):
    pipeline = acu.PipelineSpec(
        pipeline_name="acu-coordinates", provider_name="azure_content_understanding", product_type="extract", config={}
    )
    content = {
        "kind": "document",
        "pages": [{"pageNumber": 1, **page}],
        "fields": {"amount": {"type": "number", "valueNumber": 42, "source": source}},
    }
    raw = acu.RawInferenceResult(
        request=acu.InferenceRequest(example_id="coordinates", source_file_path="example.pdf", product_type="extract"),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type="extract",
        raw_output={"result": {"contents": [content]}},
        started_at=datetime.now(),
        completed_at=datetime.now(),
        latency_in_ms=0,
    )
    provider = acu.AzureContentUnderstandingProvider.__new__(acu.AzureContentUnderstandingProvider)
    result = provider.normalize(raw)
    assert result.output.extracted_data == {"amount": 42}
    assert len(result.output.field_citations) == 1
    citation = result.output.field_citations[0]
    assert citation.field_path == "amount"
    assert citation.page == 1
    return citation


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (_SMALL_SOURCE, [0.2 / 8.5, 0.2 / 11, 0.6 / 8.5, 0.6 / 11]),
        ("D(1,1,2,3,2,3,4,1,4)", [1 / 8.5, 2 / 11, 2 / 8.5, 2 / 11]),
    ],
    ids=["below-one", "above-one"],
)
def test_source_coordinates_use_page_dimensions(acu, source, expected):
    citation = _normalized_citation(acu, source, {"width": 8.5, "height": 11.0, "unit": "inch"})
    assert citation.bbox == pytest.approx(expected)


@pytest.mark.parametrize(
    "page",
    [
        {},
        {"width": 0, "height": 11},
        {"width": 8.5, "height": 0},
        {"width": -1, "height": 11},
        {"width": 8.5, "height": -1},
        {"width": "invalid", "height": 11},
        {"width": float("inf"), "height": 11},
        {"width": 8.5, "height": float("nan")},
    ],
)
def test_missing_or_invalid_dimensions_do_not_scale_source_coordinates(acu, page):
    citation = _normalized_citation(acu, _SMALL_SOURCE, page)
    assert citation.bbox == pytest.approx([0.2, 0.2, 0.6, 0.6])
