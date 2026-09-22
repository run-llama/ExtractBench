from extract_bench.inference.providers.extract.jev import variant_localized as variant


def test_retrieval_reaches_final_page():
    document = {"pages": [{"page_index": i, "text": "Unrelated boilerplate\n" * 20} for i in range(80)]}
    document["pages"][-1]["text"] += "Certificate serial number: Z999\n"
    regions = variant._regions(document)
    chosen = variant._retrieve(regions, "certificate_serial_number", {"type": "string"})
    assert chosen[0]["page_index"] == 79
    assert "Z999" in chosen[0]["text"]


def test_spacing_preserved_until_normalization():
    values = variant._values("Employee: Alice  Smith           Salary: $2,000", {"type": "string"})
    assert "Alice  Smith" in values
    assert variant._normalize("Alice  Smith", {"type": "string"}) == "Alice Smith"
    assert variant._normalize("(2,000)", {"type": "number"}) == -2000
    assert variant._normalize("4.5", {"type": "integer"}) is None


def test_boolean_blank_rule_and_selected_value():
    class Client:
        def decide(self, state, questions):
            results = {}
            for key, question in questions.items():
                criteria = question["criteria"]
                if "Find the source region" in question["instructions"]:
                    choice = "v0"
                elif "checked" in question["instructions"] and "is_checked" in question["instructions"]:
                    choice = "none"
                else:
                    choice = next(k for k, v in criteria.items() if v == '"Alice  Smith"')
                results[key] = {"choice": choice}
            return results

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "is_checked": {
                "type": ["boolean", "null"],
                "description": "True if checked, false otherwise. Never return null.",
            },
        },
    }
    result = variant.extract({"text": "Employee: Alice  Smith\n[ ] Checked"}, schema, Client())
    assert result["data"] == {"name": "Alice Smith", "is_checked": False}
