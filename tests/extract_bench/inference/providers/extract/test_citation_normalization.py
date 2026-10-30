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


@pytest.mark.parametrize("include_confidence_only", [False, True])
def test_confidence_only_option_keeps_genuine_page_citations(include_confidence_only: bool) -> None:
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
    citations = extract_llamaextract_field_citations(
        metadata, source="llamaextract", include_confidence_only=include_confidence_only
    )
    by_path = {citation.field_path: citation for citation in citations}
    assert ("properties[0].uncited" in by_path) == include_confidence_only
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
