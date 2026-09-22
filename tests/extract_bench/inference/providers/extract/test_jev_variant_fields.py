import json

from extract_bench.inference.providers.extract.jev.variant_fields import _lines, candidates, extract


class EvidenceClient:
    def decide(self, state, questions):
        results = {}
        for key, question in questions.items():
            assert question["type"] == "choice"
            criteria = question["criteria"]
            if key.startswith("row"):
                choice = (
                    "yes" if "Widget" in question["instructions"].split("Row: ")[1].split(". Context:")[0] else "no"
                )
            else:
                wanted = 1250.5 if "total" in question["instructions"] else "Widget"
                choice = next(
                    (
                        k
                        for k, v in criteria.items()
                        if k != "none" and json.loads(v.split(" | evidence: ")[0]) == wanted
                    ),
                    "none",
                )
            results[key] = {"choice": choice, "confidence": 1.0}
        return results


def test_numeric_and_nested_fields():
    document = {"text": "Grand total: $1,250.50\n"}
    schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "object",
                "properties": {
                    "total": {"anyOf": [{"type": "number"}, {"type": "null"}], "description": "Grand total"},
                    "missing": {"type": "string"},
                },
            }
        },
    }
    result = extract(document, schema, EvidenceClient())
    assert result["data"] == {"summary": {"total": 1250.5, "missing": None}}
    assert result["evidence"][0]["page_index"] == 0


def test_candidate_dates_and_signed_amounts():
    lines = _lines({"text": "Date: 2026-09-22\nAdjustment: (1,250.50)"})
    assert "2026-09-22" in [c["value"] for c in candidates(lines, "date", {"type": "string"})]
    assert -1250.5 in [c["value"] for c in candidates(lines, "adjustment", {"type": "number"})]


def test_row_objects_and_refs():
    document = {"text": "Description  Amount\nWidget  1,250.50\n"}
    schema = {
        "type": "object",
        "$defs": {"Row": {"type": "object", "properties": {"name": {"type": "string"}, "total": {"type": "number"}}}},
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Row"}}},
    }
    assert extract(document, schema, EvidenceClient())["data"] == {"items": [{"name": "Widget", "total": 1250.5}]}
