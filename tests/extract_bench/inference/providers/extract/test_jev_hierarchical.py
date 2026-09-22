import importlib.util
from pathlib import Path

MODULE = (
    Path(__file__).resolve().parents[5] / "src/extract_bench/inference/providers/extract/jev/variant_hierarchical.py"
)
spec = importlib.util.spec_from_file_location("hierarchical", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_grounded_scalar_and_nested_ref():
    class Client:
        def decide(self, state, questions):
            result = {}
            for key, question in questions.items():
                assert question["type"] == "choice"
                if key.startswith("line_"):
                    choice = "L0"
                else:
                    # Numeric candidate is normalized from the source currency amount.
                    assert "1234.5" in question["instructions"]
                    choice = "V0"
                result[key] = {"choice": choice}
            return result

    schema = {
        "type": "object",
        "properties": {"nested": {"$ref": "#/$defs/amount"}},
        "$defs": {
            "amount": {"type": "object", "properties": {"total": {"anyOf": [{"type": "number"}, {"type": "null"}]}}}
        },
    }
    result = module.extract({"text": "Total: $1,234.50"}, schema, Client())
    assert result["data"] == {"nested": {"total": 1234.5}}
    assert result["evidence"][0]["text"] == "Total: $1,234.50"


def test_absent_does_not_fabricate():
    class Client:
        def decide(self, state, questions):
            return {key: {"choice": "NONE"} for key in questions}

    result = module.extract(
        {"text": "Unrelated content"}, {"type": "object", "properties": {"name": {"type": "string"}}}, Client()
    )
    assert result["data"] == {"name": None}
    assert result["evidence"] == []


def test_span_generation_and_integer_integrity():
    assert "Alice Smith" in module._spans(["Employee: Alice Smith"], {"type": "string"})
    assert module._normalize("4.5", {"type": "integer"}) is None
    assert module._normalize("(2,000)", {"type": "number"}) == -2000
