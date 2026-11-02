from pathlib import Path

import pytest

from extract_bench.inference.providers.extract.citations import (
    extract_llamaextract_field_citations,
    extract_reducto_field_citations,
)


@pytest.mark.parametrize("provider", ["llamaextract", "reducto"])
@pytest.mark.parametrize("field_name", ["properties", "metadata", "fields", "document_metadata", "path"])
@pytest.mark.parametrize("nested", [False, True])
def test_schema_field_names_and_array_indexes_survive_normalization(
    provider: str, field_name: str, nested: bool
) -> None:
    tree = {
        field_name: [
            {"label": {"value": "A", "citations": [{"page": 1, "bbox": [0.1, 0.2, 0.1, 0.1]}]}},
            {"label": {"value": "B", "citations": [{"page": 2, "bbox": [0.3, 0.4, 0.1, 0.1]}]}},
        ]
    }
    if nested:
        tree = {"portfolio": tree}
    if provider == "llamaextract":
        citations = extract_llamaextract_field_citations(
            {"field_metadata": {"document_metadata": tree}}, source=provider
        )
    else:
        citations = extract_reducto_field_citations({"result": tree})
    prefix = "portfolio." if nested else ""
    assert [(c.field_path, c.page) for c in citations] == [
        (f"{prefix}{field_name}[0].label", 1),
        (f"{prefix}{field_name}[1].label", 2),
    ]


@pytest.mark.parametrize("envelope", ["field_metadata", "document_metadata", "fields"])
def test_llamaextract_legacy_envelopes_preserve_root_fields(envelope: str) -> None:
    metadata = {envelope: {"metadata": {"citation": [{"page": 2, "bbox": [0.1, 0.2, 0.1, 0.1]}]}}}
    citations = extract_llamaextract_field_citations(metadata, source="llamaextract")
    assert [(c.field_path, c.page) for c in citations] == [("metadata", 2)]


@pytest.mark.parametrize("section", ["page_metadata", "row_metadata"])
@pytest.mark.parametrize("field_key", ["field_metadata", "document_metadata", "fields"])
def test_llamaextract_nested_fields_inherit_page_and_dimensions(section: str, field_key: str) -> None:
    metadata = {
        "field_metadata": {
            section: [
                {
                    "page_number": 3,
                    "page_dimensions": {"width": 1000, "height": 2000},
                    field_key: {"properties": [{"amount": {"citation": [{"bounding_boxes": [[100, 200, 50, 40]]}]}}]},
                }
            ]
        }
    }
    citations = extract_llamaextract_field_citations(metadata, source="llamaextract")
    assert [(c.field_path, c.page, c.bbox) for c in citations] == [("properties[0].amount", 3, [0.1, 0.1, 0.05, 0.02])]


def test_reducto_top_level_properties_scalar_keeps_its_box() -> None:
    citations = extract_reducto_field_citations(
        {"result": [{"properties": {"value": "A", "citations": [{"page": 2, "bbox": [0.1, 0.2, 0.1, 0.1]}]}}]}
    )
    assert [(c.field_path, c.page) for c in citations] == [("properties", 2)]


def test_reducto_citations_follow_wrapped_object_values() -> None:
    citations = extract_reducto_field_citations(
        {
            "result": {
                "portfolio": {
                    "value": {"name": {"value": "A", "citations": [{"page": 2, "bbox": [0.1, 0.2, 0.1, 0.1]}]}},
                    "citations": [],
                }
            }
        }
    )
    assert [(c.field_path, c.page) for c in citations] == [("portfolio.name", 2)]


def test_confidence_without_location_cannot_earn_page_or_box_credit() -> None:
    metadata = {
        "field_metadata": {
            "document_metadata": {
                "properties": [
                    {
                        "uncited": {"confidence": 0.8},
                        "page_only": {"confidence": 0.9, "citation": [{"page": 3}]},
                        "boxed": {"confidence": 0.95, "citation": [{"page": 2, "bbox": [0.1, 0.2, 0.1, 0.1]}]},
                    }
                ],
            }
        }
    }
    citations = extract_llamaextract_field_citations(metadata, source="llamaextract")
    by_path = {citation.field_path: citation for citation in citations}
    assert by_path["properties[0].uncited"].confidence == 0.8
    assert by_path["properties[0].uncited"].page is None
    assert by_path["properties[0].uncited"].bbox is None
    from extract_bench.evaluation.metrics.extract.unified_evidence_metric import index_citations

    boxes, pages = index_citations(list(citations))
    assert "properties[0].uncited" not in pages
    assert "properties[0].uncited" not in boxes
    assert pages["properties[0].page_only"] == {3}
    assert "properties[0].page_only" not in boxes
    assert by_path["properties[0].page_only"].page == 3
    assert by_path["properties[0].page_only"].bbox is None
    assert by_path["properties[0].boxed"].bbox == [0.1, 0.2, 0.1, 0.1]


def test_extend_nested_page_object_normalizes_pixel_polygon() -> None:
    from extract_bench.inference.providers.extract.citations import extract_extend_field_citations

    citations = extract_extend_field_citations(
        {
            "processor_run": {
                "output": {
                    "metadata": {
                        "amount": {
                            "citations": [
                                {
                                    "page": {"number": 2, "width": 1000, "height": 2000},
                                    "polygon": [
                                        {"x": 100, "y": 200},
                                        {"x": 300, "y": 200},
                                        {"x": 300, "y": 260},
                                        {"x": 100, "y": 260},
                                    ],
                                }
                            ]
                        },
                    }
                }
            }
        }
    )
    assert [(c.field_path, c.page, c.bbox) for c in citations] == [("amount", 2, [0.1, 0.1, 0.2, 0.03])]


def test_citation_box_without_page_is_preserved_but_not_grounded() -> None:
    from extract_bench.evaluation.metrics.extract.unified_evidence_metric import index_citations

    citations = extract_reducto_field_citations(
        {
            "result": {
                "amount": {"value": 10, "citations": [{"bbox": [0.1, 0.2, 0.3, 0.1]}]},
            }
        }
    )
    assert len(citations) == 1
    assert citations[0].page is None
    assert citations[0].bbox == [0.1, 0.2, 0.3, 0.1]
    assert index_citations(list(citations)) == ({}, {})


def test_chunk_refinement_does_not_search_page_one_when_page_is_unknown(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")

    from extract_bench.inference.providers.extract.citations import _refine_chunk_citations_via_text_search
    from extract_bench.schemas.extract_output import FieldCitation

    pdf_path = tmp_path / "document.pdf"
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text((50, 50), "Invoice")
        document.save(pdf_path)
    citation = FieldCitation(
        field_path="name",
        page=None,
        bbox=[0.0, 0.0, 1.0, 1.0],
        metadata={"type": "chunkText"},
    )
    refined = _refine_chunk_citations_via_text_search(
        [citation],
        pdf_path=pdf_path,
        extracted_data={"name": "Invoice"},
    )
    assert refined[0].bbox == citation.bbox
    assert refined[0].page is None


@pytest.mark.parametrize("page_index", [None, -1, 0])
def test_landingai_only_converts_explicit_nonnegative_page_indexes(page_index: int | None) -> None:
    from extract_bench.inference.providers.extract.citations import extract_landingai_field_citations

    citations = extract_landingai_field_citations(
        {
            "extraction_metadata": {"amount": {"references": ["block"]}},
            "parse_response": {
                "grounding": {
                    "block": {
                        "page": page_index,
                        "box": {"left": 0.1, "top": 0.2, "right": 0.4, "bottom": 0.3},
                    }
                }
            },
        }
    )
    assert len(citations) == 1
    assert citations[0].page == (1 if page_index == 0 else None)


def test_azure_unresolved_source_preserves_confidence_without_claiming_page_one() -> None:
    from extract_bench.inference.providers.extract.azure_content_understanding import _citation_from_field

    citation = _citation_from_field("amount", {"source": "D(unknown)", "confidence": 0.8}, {})
    assert citation is not None
    assert citation.page is None
    assert citation.bbox is None
    assert citation.confidence == 0.8


@pytest.mark.parametrize(
    ("nested_number", "alternate_key", "alternate_value", "expected_page"),
    [
        (None, "page_number", 2, 2),
        (0, "pageNumber", 3, 3),
        ("unknown", "page_index", 0, 1),
        (4, "page_number", 2, 4),
    ],
)
def test_extend_invalid_nested_page_does_not_hide_explicit_page(
    nested_number: int | str | None,
    alternate_key: str,
    alternate_value: int,
    expected_page: int,
) -> None:
    from extract_bench.inference.providers.extract.citations import extract_extend_field_citations

    citations = extract_extend_field_citations(
        {
            "processor_run": {
                "output": {
                    "metadata": {
                        "amount": {"citations": [{"page": {"number": nested_number}, alternate_key: alternate_value}]},
                    }
                }
            }
        }
    )
    assert len(citations) == 1
    assert citations[0].page == expected_page
