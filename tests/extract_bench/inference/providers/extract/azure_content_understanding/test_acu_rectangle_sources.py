"""ACU four-number source boxes contain left, top, width, and height."""

from datetime import datetime

import pytest

from extract_bench.inference.providers.extract import azure_content_understanding


@pytest.fixture
def acu():
    return azure_content_understanding


@pytest.mark.parametrize(
    ("source", "expected_boxes"),
    [
        ("D(1,2,3,1,2)", [(1, [0.2, 0.3, 0.1, 0.2])]),
        ("D(1,2,3,3,4)", [(1, [0.2, 0.3, 0.3, 0.4])]),
        ("D(1,0.2,0.3,0.1,0.2)", [(1, [0.02, 0.03, 0.01, 0.02])]),
        ("D(1,0,0,1,2)", [(1, [0, 0, 0.1, 0.2])]),
        ("D(1,2,3,3,3,3,5,2,5)", [(1, [0.2, 0.3, 0.1, 0.2])]),
        ("D(1,8,9,4,3)", [(1, [0.8, 0.9, 0.2, 0.1])]),
        ("D(1,2,3,1,2);D(1,4,3,2,2)", [(1, [0.2, 0.3, 0.4, 0.2])]),
        ("D(1,2,3,1,2);D(2,4,6,2,4)", [(1, [0.2, 0.3, 0.1, 0.2]), (2, [0.2, 0.3, 0.1, 0.2])]),
    ],
    ids=[
        "smaller-extents",
        "larger-extents",
        "small-decimals",
        "origin-control",
        "polygon-control",
        "clipped",
        "union",
        "pages",
    ],
)
def test_normalize_rectangle_sources(acu, source, expected_boxes):
    pipeline = acu.PipelineSpec(
        pipeline_name="acu-rectangles", provider_name="azure_content_understanding", product_type="extract", config={}
    )
    content = {
        "kind": "document",
        "pages": [
            {"pageNumber": 1, "width": 10, "height": 10},
            {"pageNumber": 2, "width": 20, "height": 20},
        ],
        "fields": {"amount": {"type": "number", "valueNumber": 42, "source": source}},
    }
    raw = acu.RawInferenceResult(
        request=acu.InferenceRequest(example_id="rectangles", source_file_path="example.pdf", product_type="extract"),
        pipeline=pipeline,
        pipeline_name=pipeline.pipeline_name,
        product_type="extract",
        raw_output={"status": "succeeded", "result": {"contents": [content]}},
        started_at=datetime.now(),
        completed_at=datetime.now(),
        latency_in_ms=0,
    )
    provider = acu.AzureContentUnderstandingProvider.__new__(acu.AzureContentUnderstandingProvider)
    result = provider.normalize(raw)
    assert result.output.extracted_data == {"amount": 42}
    assert len(result.output.field_citations) == len(expected_boxes)
    for citation, (page, bbox) in zip(result.output.field_citations, expected_boxes, strict=True):
        assert citation.field_path == "amount"
        assert citation.page == page
        assert citation.bbox == pytest.approx(bbox)
