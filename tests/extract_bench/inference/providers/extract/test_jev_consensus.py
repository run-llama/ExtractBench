from extract_bench.inference.providers.extract.jev import variant_consensus as variant


def test_union_has_cell_and_minimal_source_phrase():
    doc = {"text": "Employee: Alice Smith     Salary: $2,000\nAccount reference: AB123"}
    choices, context = variant.candidates(doc, "name", {"type": "string", "description": "Employee name"})
    values = [choice["value"] for choice in choices]
    assert "Alice Smith" in values
    assert "Alice" in values
    assert len(values) == len(set(values))
    assert len(values) <= 200
    assert "Alice Smith" in context


def test_one_model_pass_and_no_per_option_context():
    class Client:
        calls = 0

        def decide(self, state, questions):
            self.calls += 1
            result = {}
            for key, question in questions.items():
                assert all("evidence:" not in value for value in question["criteria"].values())
                result[key] = {
                    "choice": next(k for k, value in question["criteria"].items() if value == '"Alice Smith"')
                }
            return result

    client = Client()
    result = variant.extract(
        {"text": "Employee: Alice Smith"}, {"type": "object", "properties": {"name": {"type": "string"}}}, client
    )
    assert result["data"] == {"name": "Alice Smith"}
    assert client.calls == 1


def test_typed_numeric_enum_boolean_candidates():
    doc = {"text": "Amount: $2,000.50\n[ ] Approved"}
    choices, _ = variant.candidates(doc, "amount", {"type": "number"})
    assert 2000.5 in [choice["value"] for choice in choices]
    assert all(isinstance(choice["value"], (int, float)) for choice in choices)
    choices, _ = variant.candidates(doc, "approved", {"type": "boolean"})
    assert [choice["value"] for choice in choices] == [True, False]
    choices, _ = variant.candidates(doc, "state", {"enum": ["pending", "done", None]})
    assert [choice["value"] for choice in choices] == ["pending", "done"]
