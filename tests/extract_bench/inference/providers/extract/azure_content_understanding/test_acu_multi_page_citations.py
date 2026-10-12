"""ACU citations preserve pages after span selection in both shipped providers."""

import pytest

from extract_bench.inference.providers.extract import azure_content_understanding


@pytest.fixture
def provider_module():
    return azure_content_understanding


@pytest.mark.parametrize("selected", [False, True])
def test_retains_each_selected_source_page(provider_module, selected):
    field = {
        "type": "string",
        "valueString": "total",
        "confidence": 0.9,
        "source": "D(1,1,1,2,1,2,2,1,2);D(2,2,2,3,2,3,3,2,3);D(2,3,2,4,2,4,3,3,3)",
    }
    if selected:
        field["spans"] = [{"offset": 0, "length": 5}, {"offset": 6, "length": 5}, {"offset": 12, "length": 5}]
    citations = provider_module._field_citations(
        {"amount": field},
        {1: (10, 10), 2: (20, 20)},
        document_text="label total total" if selected else None,
    )
    assert [c.page for c in citations] == ([2] if selected else [1, 2])
    assert citations[-1].bbox == pytest.approx([0.1, 0.1, 0.1, 0.05])
    assert all(c.field_path == "amount" and c.confidence == 0.9 for c in citations)
    assert all(c.reference_text == ("total total" if selected else "total") for c in citations)


def test_single_page_citation_control(provider_module):
    citations = provider_module._field_citations(
        {"amount": {"type": "string", "valueString": "total", "source": "D(1,1,1,2,1,2,2,1,2)"}},
        {1: (10, 10)},
    )
    assert len(citations) == 1
    assert citations[0].page == 1
    assert citations[0].bbox == pytest.approx([0.1, 0.1, 0.1, 0.1])


def test_reference_text_stays_with_its_source_page(provider_module):
    citations = provider_module._field_citations(
        {
            "amount": {
                "type": "string",
                "valueString": "Alpha Beta",
                "source": "D(1,1,1,2,1,2,2,1,2);D(2,1,1,2,1,2,2,1,2)",
                "spans": [{"offset": 0, "length": 5}, {"offset": 6, "length": 4}],
            }
        },
        {1: (10, 10), 2: (10, 10)},
        document_text="Alpha Beta",
    )
    assert [(citation.page, citation.reference_text) for citation in citations] == [(1, "Alpha"), (2, "Beta")]
