from extract_bench.inference.providers.extract.jev import variant_router as router


def test_counts_source_rows_without_separators_empty_or_escaped_pipes():
    document = {"markdown": "| Name | Value |\n|:---|---:|\n| Alice | 3 |\n| | |\nplain text\\|literal\nBob | 4"}
    assert router.markdown_row_count(document) == 3
    assert router.markdown_row_count({"pages": [{"markdown": "| x |"}, {"markdown": "| y |"}]}) == 2
    assert router.markdown_row_count({"markdown": "| x |", "pages": [{"markdown": "| x |"}]}) == 1


def test_object_array_detection_resolves_refs_unions_and_cycles():
    schema = {
        "type": "object",
        "properties": {"nested": {"$ref": "#/$defs/Nested"}},
        "$defs": {
            "Nested": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {"records": {"type": "array", "items": {"$ref": "#/$defs/Record"}}},
                    },
                ]
            },
            "Record": {"type": "object", "properties": {"value": {"type": "string"}}},
        },
    }
    assert router.has_object_array(schema)
    assert not router.has_object_array({"type": "array", "items": {"type": "string"}})
    assert not router.has_object_array(
        {
            "$ref": "#/$defs/cycle",
            "$defs": {"cycle": {"type": "object", "properties": {"self": {"$ref": "#/$defs/cycle"}}}},
        }
    )
    assert not router.has_object_array(
        {"type": "object", "$defs": {"unused": {"type": "array", "items": {"type": "object"}}}}
    )


def test_router_calls_exactly_one_engine_and_preserves_payload(monkeypatch):
    calls = []
    data, evidence = {"rows": [{"id": 1}]}, [{"path": "rows[0].id"}]

    def engine(name):
        def extract(document, schema, client):
            calls.append(name)
            return {"data": data, "evidence": evidence, "anchor_diagnostics": ["unchanged"]}

        return extract

    monkeypatch.setattr(router.variant_anchors, "extract", engine("anchors"))
    monkeypatch.setattr(router.variant_compact, "extract", engine("compact"))
    array_schema = {"type": "object", "properties": {"rows": {"type": "array", "items": {"type": "object"}}}}
    for count, schema, expected in [
        (99, array_schema, "compact"),
        (100, array_schema, "anchors"),
        (101, {"type": "string"}, "compact"),
    ]:
        result = router.extract({"markdown": "\n".join("| record | 1 |" for _ in range(count))}, schema, object())
        assert calls[-1] == expected
        assert result["route_diagnostics"]["engine"] == expected
        assert result["data"] is data and result["evidence"] is evidence
        assert result["anchor_diagnostics"] == ["unchanged"]
    assert calls == ["compact", "anchors", "compact"]
