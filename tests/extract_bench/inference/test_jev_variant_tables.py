from extract_bench.inference.providers.extract.jev.variant_tables import discover_tables, extract


def test_preserves_empty_pipe_cells():
    tables = discover_tables("Name | Amount | Currency\nA | | USD\nB | 12 | GBP")
    assert tables[0].rows[1] == ["A", "", "USD"]


def test_table_column_mapping_and_header_exclusion():
    class Client:
        def decide(self, state, questions):
            if "t0" in questions:
                return {"t0": {"choice": "c0"}, "s0": {"choice": "c0"}}
            return {
                "c0": {"choice": "c0"},
                "c1": {"choice": "c1"},
                "r0": {"choice": "c1"},
                "r1": {"choice": "c0"},
                "r2": {"choice": "c0"},
            }

    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": "string"}, "amount": {"type": "number"}}},
            }
        },
    }
    result = extract({"text": "Name  Amount\nWidget  $1,234.50\nService  (10.00)"}, schema, Client())
    assert result["data"] == {"rows": [{"name": "Widget", "amount": 1234.5}, {"name": "Service", "amount": -10.0}]}
    assert result["evidence"][0]["line_index"] == 1
