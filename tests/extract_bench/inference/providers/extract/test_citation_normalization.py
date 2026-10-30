import pytest

from extract_bench.inference.providers.extract.citations import (
    extract_llamaextract_field_citations,
    extract_reducto_field_citations,
)


@pytest.mark.parametrize("field_name", ["properties", "metadata", "fields"])
def test_llamaextract_preserves_wrapper_named_fields_inside_objects(field_name: str) -> None:
    metadata = {
        "field_metadata": {
            "portfolio": {
                field_name: [
                    {"label": {"citation": [{"page": 1, "bbox": [0.1, 0.2, 0.1, 0.1]}]}},
                    {"label": {"citation": [{"page": 2, "bbox": [0.3, 0.4, 0.1, 0.1]}]}},
                ]
            }
        }
    }
    citations = extract_llamaextract_field_citations(metadata, source="llamaextract")
    assert [(c.field_path, c.page) for c in citations] == [
        (f"portfolio.{field_name}[0].label", 1),
        (f"portfolio.{field_name}[1].label", 2),
    ]


def test_llamaextract_preserves_top_level_properties_array() -> None:
    metadata = {
        "field_metadata": {"properties": [{"label": {"citation": [{"page": 1, "bbox": [0.1, 0.2, 0.1, 0.1]}]}}]}
    }
    citations = extract_llamaextract_field_citations(metadata, source="llamaextract")
    assert [c.field_path for c in citations] == ["properties[0].label"]


def test_llamaextract_envelope_wrappers_do_not_become_field_paths() -> None:
    metadata = {
        "field_metadata": {"document_metadata": {"label": {"citation": [{"page": 2, "bbox": [0.1, 0.2, 0.1, 0.1]}]}}}
    }
    citations = extract_llamaextract_field_citations(metadata, source="llamaextract")
    assert [(c.field_path, c.page) for c in citations] == [("label", 2)]


def test_reducto_nested_structural_wrappers_do_not_become_field_paths() -> None:
    raw_output = {
        "result": {"portfolio": {"fields": {"label": {"citation": [{"page": 2, "bbox": [0.1, 0.2, 0.1, 0.1]}]}}}}
    }
    citations = extract_reducto_field_citations(raw_output)
    assert [(c.field_path, c.page) for c in citations] == [("portfolio.label", 2)]
