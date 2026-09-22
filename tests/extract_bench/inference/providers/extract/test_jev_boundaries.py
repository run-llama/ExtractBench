from extract_bench.inference.providers.extract.jev import variant_boundaries as variant


def test_complete_long_name_assembled_from_boundaries():
    class Client:
        def decide(self, state, questions):
            answers = {}
            for key, question in questions.items():
                instructions = question["instructions"]
                if "Which source region" in instructions:
                    choice = "v0"
                elif "FIRST word" in instructions:
                    choice = "w1"
                elif "LAST word" in instructions:
                    choice = "w10"
                answers[key] = {"choice": choice}
            return answers

    document = {
        "text": "Name: The Very Long Global International Engineering Research And Development Company    ID: 900"
    }
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    result = variant.extract(document, schema, Client())
    assert result["data"]["name"] == "The Very Long Global International Engineering Research And Development Company"
    assert result["evidence"][0]["text"] in document["text"]


def test_prune_array_schema_keeps_original_paths_and_description():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "nested": {
                "type": "object",
                "properties": {"rows": {"type": "array", "items": {"type": "string"}}, "other": {"type": "number"}},
            },
        },
    }
    arrays = variant._array_schema(schema, schema)
    assert set(arrays["properties"]) == {"nested"}
    assert set(arrays["properties"]["nested"]["properties"]) == {"rows"}
    dest = {"name": "Alice", "nested": {"other": 2, "rows": []}}
    variant._merge(dest, {"nested": {"rows": ["x"]}})
    assert dest == {"name": "Alice", "nested": {"other": 2, "rows": ["x"]}}


def test_word_boundaries_stay_under_api_option_limit():
    document = {"text": " ".join("word" + str(i) for i in range(2000))}
    regions = variant._bounded_regions(document)
    assert all(len(r["text"].split()) <= 220 for r in regions)
    assert any("word1999" in r["text"] for r in regions)
