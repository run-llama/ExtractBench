import json

from extract_bench.inference.providers.extract.jev.variant_geometric import extract, geometric_lines, span_candidates


def word(text, x, y):
    return {"text": text, "x": x, "y": y, "height": 10, "width": len(text) * 5}


def test_reconstruct_baselines_instead_of_parser_line_order():
    doc = {
        "pages": [
            {
                "page_index": 2,
                "text": "158.50\nSubtotal 17.04\nTax",
                "words": [
                    word("Subtotal", 0, 10),
                    word("158.50", 100, 11),
                    word("Tax", 0, 30),
                    word("17.04", 100, 29),
                ],
            }
        ]
    }
    rows = geometric_lines(doc)
    assert rows[0]["text"] == "Subtotal 158.50"
    assert rows[1]["text"] == "Tax 17.04"
    assert rows[0]["page"] == 2


def test_complete_multicell_name_candidates():
    rows = geometric_lines({"pages": [{"words": [word("Name:", 0, 0), word("Alex", 80, 0), word("Morgan", 125, 0)]}]})
    values = [option["value"] for option in span_candidates(rows, "name", {"type": "string"})]
    assert "Alex Morgan" in values


def test_extract_uses_local_state_and_nullable_boolean():
    class Client:
        def decide(self, state, questions):
            assert len(state) < 200
            result = {}
            for key, question in questions.items():
                assert "Schema:" in question["instructions"]
                choice = next(
                    (
                        k
                        for k, v in question["criteria"].items()
                        if k != "none" and json.loads(v.split(" | evidence:")[0]) == "Alex Morgan"
                    ),
                    "none",
                )
                result[key] = {"choice": choice, "confidence": 0.9}
            return result

    result = extract(
        {"text": "Name: Alex Morgan\nApproved?"},
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "approved": {"type": ["boolean", "null"]},
            },
        },
        Client(),
    )
    assert result["data"] == {"name": "Alex Morgan", "approved": None}
